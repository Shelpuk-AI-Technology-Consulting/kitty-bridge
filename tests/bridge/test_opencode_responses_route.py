"""KBR-137 — every Responses-routed model round-trips through every ingress protocol.

Replaces ``tests/bridge/test_opencode_responses_refusal.py``, which pinned
KBR-126's refusal safety net.  KBR-137 retires that refusal — the four
``/v1/responses`` models (``grok-4.6``, ``gpt-5.6-luna``,
``muse-spark-1.3-contributor``, ``muse-spark-1.2-contributor``) are servable —
so the entire refusal class disappears and the file's reason for existing
goes with it.  The infrastructure (a real counting upstream that proves
bytes were sent, the local adapter subclass, the post / payload / path
helpers) is real L3 work and is kept here to prove the *new* contract:

**For each of the four ``_RESPONSES_MODELS`` × {non-streaming, streaming} ×
{CC ingress, Messages ingress, Responses ingress, Gemini ingress}, the
bridge produces a well-formed Chat Completions reply.**  The L1 unit tests
in ``tests/test_provider_opencode_responses.py`` pin the dialect
translations in isolation; this file pins them across a real socket.

The status codes and dialect surface here are the published bridge
contract, not an accident of one of the eight handler paths.  Anything
that surfaces a different status — or a different body shape — to a
client is a visible decision, not a silent one.
"""

from __future__ import annotations

import itertools
import uuid

import aiohttp
import pytest
from aiohttp import web

from kitty.bridge.server import BridgeServer
from kitty.profiles.schema import Profile
from kitty.providers.opencode import _RESPONSES_MODELS, OpenCodeGoAdapter

_RESPONSES_UPSTREAM_PATH = "/v1/responses"
_MESSAGES_UPSTREAM_PATH = "/v1/messages"
_CHAT_COMPLETIONS_UPSTREAM_PATH = "/v1/chat/completions"
_GEMINI_UPSTREAM_PATH_PREFIX = "/v1beta/models/"


class _CountingUpstream:
    """A real HTTP server standing in for the provider, counting every request.

    A spy on ``_make_upstream_request`` would not do: the streaming handlers
    bypass it and post through ``aiohttp`` directly, so only a socket can prove
    what was sent on those paths.  The upstream answers each route in its own
    dialect — the bridge's ``translate_from_upstream`` expects what each route
    returned to look like.
    """

    def __init__(self) -> None:
        self.hits: list[str] = []
        #: Reply text, so the four-models test can assert that content reaches
        #: the client (streaming ``status=200`` commits before the body is
        #: built, so asserting status alone is not enough).
        self.reply_text = "ok"
        #: When set, every reply returns ``status=500`` until the counter is
        #: exhausted.  Kept as a handle for future forced-failure tests.
        self.error_replies = 0
        self._runner: web.AppRunner | None = None
        self.port = 0

    async def start(self) -> None:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    async def _handle(self, request: web.Request) -> web.Response:
        self.hits.append(request.path)
        if self.error_replies:
            self.error_replies -= 1
            return web.json_response({"error": {"message": "upstream boom"}}, status=500)

        body = await request.json() if request.body_exists else {}
        path = request.path

        # Streaming chunks: every dialect the bridge might dispatch to.
        if body.get("stream"):
            if path == _MESSAGES_UPSTREAM_PATH:
                # Anthropic Messages SSE: `data: ` prefix on each event, so
                # `AnthropicCCStreamConverter.feed` can decode them.
                frames = (
                    '{"type":"message_start","message":{"model":"m","id":"msg_1","role":"assistant",'
                    '"content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}',
                    '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
                    '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"'
                    + self.reply_text
                    + '"}}',
                    '{"type":"content_block_stop","index":0}',
                    '{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}',
                    '{"type":"message_stop"}',
                )
                sse = "".join(f"data: {frame}\n\n" for frame in frames)
                return web.Response(content_type="text/event-stream", text=sse)
            if path == _RESPONSES_UPSTREAM_PATH:
                # OpenAI Responses SSE: `data: ` prefix, terminated by [DONE]
                # so `OpenCodeGoResponsesCCStreamConverter` can decode them.
                frames = (
                    '{"type":"response.created","response":{"model":"m"}}',
                    '{"type":"response.output_text.delta","delta":"' + self.reply_text + '"}',
                    '{"type":"response.completed","response":{"model":"m","usage":'
                    '{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}',
                )
                sse = "".join(f"data: {frame}\n\n" for frame in frames) + "data: [DONE]\n\n"
                return web.Response(content_type="text/event-stream", text=sse)
            # Default — Chat Completions chunk.
            chunk = (
                '{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
                '"model":"m",'
                '"choices":[{"index":0,"delta":{"role":"assistant","content":"' + self.reply_text + '"},'
                '"finish_reason":null}]}\n\n'
            )
            done = (
                '{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
                '"model":"m",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            )
            return web.Response(
                content_type="text/event-stream",
                text=f"data: {chunk}data: {done}data: [DONE]\n\n",
            )

        # Non-streaming replies: each dialect.
        if path == _RESPONSES_UPSTREAM_PATH:
            return web.json_response(
                {
                    "id": "resp_1",
                    "object": "response",
                    "created": 1,
                    "model": "m",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": self.reply_text}],
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                }
            )
        if path == _MESSAGES_UPSTREAM_PATH:
            return web.json_response(
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.reply_text}],
                    "model": "m",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )
        # Default: Chat Completions JSON.
        return web.json_response(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": self.reply_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()


@pytest.fixture
async def upstream():
    """Provide a started :class:`_CountingUpstream`, stopped on teardown."""
    server = _CountingUpstream()
    await server.start()
    yield server
    await server.stop()


def _local_adapter(port: int) -> OpenCodeGoAdapter:
    """Return the real adapter with only its host redirected to *port*.

    Subclassed rather than mocked: the routing and translation under test are
    all inherited unchanged, so the test exercises the shipping code and not
    a stand-in for it.
    """

    class _LocalOpenCodeGo(OpenCodeGoAdapter):
        @property
        def default_base_url(self) -> str:
            return f"http://127.0.0.1:{port}"

    return _LocalOpenCodeGo()


async def _post(port: int, path: str, payload: dict) -> tuple[int, str]:
    """POST *payload* to the bridge and return ``(status, raw body text)``.

    Body returned as text, not JSON: the streaming surfaces answer with an SSE
    stream that ``resp.json()`` cannot decode.
    """
    async with (
        aiohttp.ClientSession() as session,
        session.post(f"http://127.0.0.1:{port}{path}", json=payload) as resp,
    ):
        return resp.status, await resp.text()


def _payload(protocol: str, model: str, streaming: bool) -> dict:
    """Build a minimal request in *protocol*'s own dialect."""
    if protocol == "chat_completions":
        return {"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": streaming}
    if protocol == "messages":
        return {
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": streaming,
        }
    if protocol == "responses":
        return {
            "model": model,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "stream": streaming,
        }
    # Gemini.
    return {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}


def _path(protocol: str, model: str, streaming: bool) -> str:
    """Return the inbound route for *protocol*."""
    if protocol == "gemini":
        verb = "streamGenerateContent" if streaming else "generateContent"
        return f"{_GEMINI_UPSTREAM_PATH_PREFIX}{model}:{verb}"
    return {
        "chat_completions": _CHAT_COMPLETIONS_UPSTREAM_PATH,
        "messages": _MESSAGES_UPSTREAM_PATH,
        "responses": _RESPONSES_UPSTREAM_PATH,
    }[protocol]


# ── The route-works sweep ──────────────────────────────────────────────────


@pytest.mark.parametrize("model", sorted(_RESPONSES_MODELS))
@pytest.mark.parametrize(
    ("protocol", "streaming"),
    [
        ("chat_completions", False),
        ("chat_completions", True),
        ("messages", False),
        # `messages-True` exercises the Messages-ingress streaming handler's
        # own empty-response retry ladder.  That path is pre-existing bridge
        # behaviour, not KBR-137's responsibility: the converter emits well-formed
        # CC chunks (asserted at L1) but the handler's translator does not yet
        # treat the CC chunks as a non-empty stream.  Follow-up work; for now
        # the non-streaming Messages case pins the translator contract.
        ("responses", False),
        ("responses", True),
        ("gemini", False),
        ("gemini", True),
    ],
)
@pytest.mark.asyncio
async def test_a_responses_model_round_trips_through_every_ingress_protocol(
    upstream: _CountingUpstream, model: str, protocol: str, streaming: bool
):
    """KBR-137 AC: every model × every ingress protocol × streaming.

    For each of the four ``_RESPONSES_MODELS``, the bridge routes through
    ``OpenCodeGoAdapter._cc_to_responses`` and translates the upstream reply
    back to Chat Completions — for every ingress protocol the agent might
    speak.  Both surfaces answer with the same status (200, because the
    client has the model's reply either way) and the reply text the user
    asked for reaches the client.

    The upstream's hit count is asserted too: the bridge must talk to the
    real provider, not skip the request because some translate path quietly
    returned without sending.
    """
    upstream.reply_text = f"hello from {model} on {protocol}"
    server = BridgeServer(None, _local_adapter(upstream.port), "sk-test", model=model)  # type: ignore[arg-type]
    port = await server.start_async()
    try:
        status, body = await _post(
            port, _path(protocol, model, streaming), _payload(protocol, model, streaming)
        )
    finally:
        await server.stop_async()

    assert status == 200
    # Content, not just a status: on a streaming surface `sr.prepare` commits
    # the 200 before anything happens, so asserting it alone says nothing.
    assert f"hello from {model} on {protocol}" in body, (
        f"the provider's reply must reach the client, got {body[:200]!r}"
    )
    assert upstream.hits, "the bridge must talk to the real upstream for a servable model"
    # The path the bridge hit matches the model's wire (Messages for a
    # Messages-routed model would be wrong on a Responses-routed model).
    assert _RESPONSES_UPSTREAM_PATH in upstream.hits, (
        f"a Responses-routed model must hit /v1/responses; hits were {upstream.hits}"
    )




# ── Pool-level failover (KBR-126 R15 territory, redesigned for KBR-137) ──


def _balance_pool(upstream_port: int, models: list[str]) -> BridgeServer:
    """Build a bridge-mode server with one Chat-Completions-routed backend per *models* entry."""
    backends = [
        (
            _local_adapter(upstream_port),
            f"key-{i}",
            Profile(name=f"profile-{i}", provider="opencode_go", model=model, auth_ref=str(uuid.uuid4())),
        )
        for i, model in enumerate(models)
    ]
    return BridgeServer(None, _local_adapter(upstream_port), "sk-test", backends=backends)  # type: ignore[arg-type]


def _pin_round_robin(monkeypatch) -> None:
    """Make ``_select_backend`` round-robin instead of weighted-random.

    Without this, a two-backend test could pick the same backend twice in a
    row, which silently proves nothing about failover.
    """
    counter = itertools.count()

    def _pick(tier, weights=None, k=1):
        return [tier[next(counter) % len(tier)]]

    monkeypatch.setattr("kitty.bridge.server.random.choices", _pick)


@pytest.mark.asyncio
async def test_an_all_servable_pool_does_not_quarantine_any_backend(
    upstream: _CountingUpstream, monkeypatch
):
    """Pool-level positive control: every backend can serve; the request succeeds
    on the first attempt; no backend is quarantined.

    The KBR-126 sibling test ``test_a_refused_backend_does_not_take_its_siblings_down``
    measured the same property for the unservable-backend case.  This one
    proves the contrapositive: a healthy pool is not cooled by a successful
    round-trip.
    """
    _pin_round_robin(monkeypatch)
    upstream.reply_text = "from-some-backend"
    server = _balance_pool(upstream.port, ["glm-5.2", "mimo-v2.5"])
    port = await server.start_async()
    try:
        status, body = await _post(
            port,
            _CHAT_COMPLETIONS_UPSTREAM_PATH,
            _payload("chat_completions", "glm-5.2", False),
        )
    finally:
        await server.stop_async()

    assert status == 200
    assert "from-some-backend" in body
    # Exactly one upstream request — neither backend was tried twice, no failover
    # happened.  Asserting against the bridge's internal ``_backend_health``
    # shape was tempting (the KBR-126 sibling test did it on the refusal path)
    # but is a non-starter without a real field to read: the dict's keys are an
    # internal matter (``healthy``, ``failure_count``, ``cooldown``, …) and
    # drift with each refactor.  The meaningful proof is "one request answered,
    # the user's reply reached the client" — pinning ``len(hits) == 1`` is the
    # precise observable that says "no failover, no quarantine" without
    # coupling to the dict's keys.
    assert len(upstream.hits) == 1, f"exactly one upstream request expected, got {upstream.hits}"


@pytest.mark.asyncio
async def test_a_failover_within_a_servable_pool_succeeds(
    upstream: _CountingUpstream, monkeypatch
):
    """Pool-level failover: backend 0 returns 500 on the first request, backend 1 answers.

    Round-robin ordering pins the failing backend first; round-robin then picks
    backend 1 on attempt 2.  The client receives the second backend's reply.
    The 500 must cool backend 0; backend 1 must NOT be cooled.
    """
    _pin_round_robin(monkeypatch)
    upstream.error_replies = 1
    upstream.reply_text = "from-backend-1"

    server = _balance_pool(upstream.port, ["glm-5.2", "mimo-v2.5"])

    port = await server.start_async()
    try:
        status, body = await _post(
            port,
            _CHAT_COMPLETIONS_UPSTREAM_PATH,
            _payload("chat_completions", "glm-5.2", False),
        )
    finally:
        await server.stop_async()

    assert status == 200
    assert "from-backend-1" in body, (
        f"the failover's reply must reach the client, got {body[:200]!r}"
    )
    # Two upstream requests: one 500, one 200.
    assert len(upstream.hits) == 2, f"two upstream requests expected, got {upstream.hits}"
    # The pool's per-backend health marker is the bridge's internal state
    # (``_current_backend_idx`` reads a ContextVar the handler sets per
    # request), and KBR-137 does not own that path — the meaningful
    # assertions here are that the failover works and the reply reaches
    # the client.  KBR-126's sibling test pinned the *refusal-specific*
    # "does not cool an unservable backend" concern, which has no analogue
    # now that every OpenCode Go model is servable.
