"""KBR-126 — a model kitty cannot serialize fails visibly, and sends nothing upstream.

``OpenCodeGoAdapter`` refuses the four models OpenCode Go serves on
``/v1/responses`` (:class:`~kitty.providers.base.UnsupportedModelError`) instead
of posting them a Chat Completions body, which the provider answers with a
``401`` that the bridge then reports to the user as a bad API key.

These tests drive the real handlers over real sockets, against a real local
upstream that counts requests, so "nothing was sent" is an observation rather
than an inference.

**The status codes below were measured, not chosen.**  The refusal is raised
inside ``translate_to_upstream``, and what a client sees depends on how far its
handler had got when that happened:

===================  =========  ======  =============================
Inbound protocol     Streaming  Status  Carrier
===================  =========  ======  =============================
Chat Completions     no         500     JSON body
Chat Completions     yes        200     in-stream ``data:`` frame
Messages             no         500     JSON body
Messages             yes        502     JSON body
Responses            no         500     JSON body
Responses            yes        200     in-stream ``event: error`` frame
Gemini               no         500     JSON body
Gemini               yes        200     in-stream ``data:`` frame
===================  =========  ======  =============================

The three streaming ``200``\\ s are not a choice this change made: those handlers
call ``await sr.prepare(request)`` on a ``status=200`` ``StreamResponse`` before
they build the body, so the response is already committed.  ``_stream_messages``
alone prepares lazily and can still answer ``502``.  Making all eight a clean
``400`` needs a request-admission check in four handlers plus an adapter hook —
plumbing whose only purpose is to refuse, and which **KBR-137 deletes** when the
Responses route starts working.  So the surfaces are pinned here rather than
changed, and the property that actually matters is asserted on every one of
them: **the message reaches the user, and no upstream request is made.**

**Not covered here.**  The four streaming handlers call ``translate_to_upstream``
again after a mid-stream failover, so a pool that fails over *to* a
Responses-routed backend ends the stream at that point.  No pool damage results —
those handlers' outer ``except Exception`` does not mark a backend unhealthy —
so it is uncovered rather than defective, and said out loud so a green run here
is not read as covering it.
"""

from __future__ import annotations

import asyncio
import itertools
import uuid

import aiohttp
import pytest
from aiohttp import web

from kitty.bridge.server import BridgeServer
from kitty.profiles.schema import Profile
from kitty.providers.opencode import OpenCodeGoAdapter

#: Served on ``/v1/responses`` — refused until KBR-137.
REFUSED_MODEL = "grok-4.6"

#: Served on ``/v1/chat/completions`` — the positive control.
SERVED_MODEL = "glm-5.2"


class _CountingUpstream:
    """A real HTTP server standing in for the provider, counting every request.

    A spy on ``_make_upstream_request`` would not do: the streaming handlers
    bypass it and post through ``aiohttp`` directly, so only a socket can prove
    that nothing was sent on those paths.
    """

    def __init__(self) -> None:
        self.hits: list[str] = []
        #: Number of leading non-streaming replies to answer with empty content,
        #: so a test can drive the bridge's empty-response retry ladder.
        self.empty_replies = 0
        self._runner: web.AppRunner | None = None
        self.port = 0

    async def start(self) -> None:
        """Bind to an ephemeral port and begin recording."""
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    async def _handle(self, request: web.Request) -> web.Response:
        """Record the path and answer in the shape the request asked for.

        The streaming branch is load-bearing, not politeness.  Answering a
        ``stream: true`` request with a plain JSON body makes the bridge read an
        empty stream and walk its entire empty-response retry ladder: six
        upstream requests and 76 seconds of `sleep`, ending in a 200 with an
        empty body.  The positive control below would still pass — `status` was
        committed by `sr.prepare` and `hits` is non-empty *because* of the
        retries — so it would be proving nothing, slowly.
        """
        self.hits.append(request.path)
        # Answers Chat Completions chunks whatever dialect the request was
        # written in. Correct today because every positive control here uses
        # SERVED_MODEL, which is Chat-Completions-routed — but a Messages-routed
        # control added later would be answered in the wrong dialect and would
        # need this keyed off `request.path`.
        body = await request.json()
        if body.get("stream"):
            chunk = (
                '{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
                f'"model":"{SERVED_MODEL}",'
                '"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}'
            )
            done = (
                '{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
                f'"model":"{SERVED_MODEL}",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            )
            return web.Response(
                content_type="text/event-stream",
                text=f"data: {chunk}\n\ndata: {done}\n\ndata: [DONE]\n\n",
            )
        content = "ok"
        if self.empty_replies:
            self.empty_replies -= 1
            content = ""
        return web.json_response(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": SERVED_MODEL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    async def stop(self) -> None:
        """Release the port."""
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

    Subclassed rather than mocked: the routing, the refusal and the translation
    under test are all inherited unchanged, so the test exercises the shipping
    code and not a stand-in for it.
    """

    class _LocalOpenCodeGo(OpenCodeGoAdapter):
        @property
        def default_base_url(self) -> str:
            return f"http://127.0.0.1:{port}"

    return _LocalOpenCodeGo()


async def _post(port: int, path: str, payload: dict) -> tuple[int, str]:
    """POST *payload* to the bridge and return ``(status, raw body text)``.

    The body is returned as text, not JSON: three of the eight surfaces answer
    with an SSE stream, which ``resp.json()`` cannot decode.
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
    return {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}


def _path(protocol: str, model: str, streaming: bool) -> str:
    """Return the inbound route for *protocol*."""
    if protocol == "gemini":
        verb = "streamGenerateContent" if streaming else "generateContent"
        return f"/v1beta/models/{model}:{verb}"
    return {
        "chat_completions": "/v1/chat/completions",
        "messages": "/v1/messages",
        "responses": "/v1/responses",
    }[protocol]


# (protocol, streaming, measured status)
SURFACES = [
    ("chat_completions", False, 500),
    ("chat_completions", True, 200),
    ("messages", False, 500),
    ("messages", True, 502),
    ("responses", False, 500),
    ("responses", True, 200),
    ("gemini", False, 500),
    ("gemini", True, 200),
]


@pytest.mark.parametrize(("protocol", "streaming", "expected_status"), SURFACES)
@pytest.mark.asyncio
async def test_a_responses_model_is_refused_and_nothing_is_sent(
    upstream: _CountingUpstream, protocol: str, streaming: bool, expected_status: int
):
    """R14 — every inbound protocol, streaming and not.

    Two assertions carry the requirement, and the status is pinned alongside
    them so a change to any surface is a visible decision rather than a
    surprise: the user is told what is wrong, and the provider never sees a
    request it would answer with a misleading 401.
    """
    server = BridgeServer(None, _local_adapter(upstream.port), "sk-test", model=REFUSED_MODEL)  # type: ignore[arg-type]
    port = await server.start_async()
    try:
        status, body = await _post(
            port, _path(protocol, REFUSED_MODEL, streaming), _payload(protocol, REFUSED_MODEL, streaming)
        )
    finally:
        await server.stop_async()

    assert status == expected_status
    assert REFUSED_MODEL in body
    assert "/v1/responses" in body
    # What the user can act on, rather than an internal ticket id they cannot
    # look up: the alternative routes are named in the message itself. Pinned
    # here as well as at L1 because this is the only place that observes the
    # body a user actually receives, on all eight surfaces.
    assert "/v1/chat/completions" in body
    assert "KBR-" not in body, "no internal tracker id may reach an end user's terminal"
    assert upstream.hits == [], "the provider must never see a request for a model kitty cannot serialize"


@pytest.mark.parametrize(("protocol", "streaming", "_status"), SURFACES)
@pytest.mark.asyncio
async def test_a_servable_model_still_reaches_the_provider(
    upstream: _CountingUpstream, protocol: str, streaming: bool, _status: int
):
    """R14's positive control — without it, "zero requests" is vacuous.

    Every assertion in the test above is satisfied by a bridge that has stopped
    working altogether.  This is the half that proves the refusal is scoped to
    the four models rather than being a blanket failure.
    """
    server = BridgeServer(None, _local_adapter(upstream.port), "sk-test", model=SERVED_MODEL)  # type: ignore[arg-type]
    port = await server.start_async()
    try:
        status, body = await _post(
            port, _path(protocol, SERVED_MODEL, streaming), _payload(protocol, SERVED_MODEL, streaming)
        )
    finally:
        await server.stop_async()

    assert status == 200
    # Content, not just a status: on a streaming surface `sr.prepare` commits
    # the 200 before anything happens, so asserting it alone says nothing. The
    # model's reply has to come back out the other side.
    assert "ok" in body, f"the provider's reply must reach the client, got {body[:200]!r}"
    assert len(upstream.hits) == 1, f"exactly one upstream request, got {upstream.hits}"


# ── The balancing pool ─────────────────────────────────────────────────────


def _balancing_server(upstream: _CountingUpstream, models: list[str]) -> BridgeServer:
    """Build a bridge-mode server with one backend per entry in *models*."""
    backends = [
        (
            _local_adapter(upstream.port),
            f"key-{i}",
            Profile(name=f"profile-{i}", provider="opencode_go", model=model, auth_ref=str(uuid.uuid4())),
        )
        for i, model in enumerate(models)
    ]
    return BridgeServer(None, _local_adapter(upstream.port), "sk-test", backends=backends)  # type: ignore[arg-type]


def _pin_backend_order(monkeypatch) -> None:
    """Make backend selection round-robin instead of weighted-random.

    ``_select_backend`` picks with ``random.choices``, so without this seam a
    two-backend test may never try the backend it is about to refuse — and the
    test would pass with the refusal handler deleted, which is how this was
    found.  Injecting the seam is the determinism rule applied to an ambient
    random source, not a change of behaviour.

    Worth knowing while reading the test below: because a refusal deliberately
    leaves ``failure_count`` untouched, real weighted selection can offer the
    same unservable backend twice within one request's attempt budget.  That is
    the same trade-off the ``CompactionFailedError`` branch already accepts, and
    the reason the all-unservable case is tested separately.
    """
    calls = itertools.count()

    def _round_robin(tier, weights=None, k=1):
        # A true cycle, not "first unseen then stick on tier[0]". The sticky
        # form offers the same backend forever once the pool has been walked
        # once, which silently starves the empty-response ladder of every
        # backend but one — a property of the seam, not of the code under test.
        return [tier[next(calls) % len(tier)]]

    # `kitty.bridge.server.random` is the stdlib module object, so this patches
    # `random.choices` process-wide for the duration. `monkeypatch` reverts it,
    # which is what keeps that contained.
    monkeypatch.setattr("kitty.bridge.server.random.choices", _round_robin)


@pytest.mark.asyncio
async def test_a_refused_backend_does_not_take_its_siblings_down(upstream: _CountingUpstream, monkeypatch, caplog):
    """R15 — one unservable model must not quarantine the pool.

    Without the ``UnsupportedModelError`` handler in
    ``_request_with_retry_balancing``, the refusal falls into ``except
    Exception``, is classified ``"hard"``, and every backend is cooled for
    ``backend_cooldown`` — because one profile names a model kitty cannot
    serve.  The same reasoning the ``CompactionFailedError`` branch already
    records: no upstream request was made, so there is no evidence against the
    backend.
    """
    _pin_backend_order(monkeypatch)
    server = _balancing_server(upstream, [REFUSED_MODEL, SERVED_MODEL])
    cooled: list[int] = []
    monkeypatch.setattr(server, "_mark_backend_unhealthy", lambda idx, **kw: cooled.append(idx))
    port = await server.start_async()
    try:
        status, _body = await _post(port, "/v1/chat/completions", _payload("chat_completions", REFUSED_MODEL, False))
    finally:
        await server.stop_async()

    assert status == 200, "the servable sibling must answer"
    assert cooled == [], "a serialization refusal is not evidence against any backend"
    assert upstream.hits, "the sibling's request must actually have been sent"
    # Without this the test degrades into "a healthy backend answers": it passes
    # in either backend order, and would pass with the refusal handler deleted
    # if selection happened to try the servable backend first.
    assert any("cannot serve this model" in record.message for record in caplog.records), (
        "the refused backend was never tried — this test proved nothing"
    )


@pytest.mark.asyncio
async def test_an_entirely_unservable_pool_fails_fast_without_sleeping(upstream: _CountingUpstream, monkeypatch):
    """R15 — and it must not sleep its way through the final retry ladder.

    Suppressing the mark in the failover loop alone is not enough.  The final
    empty-response retry loop that follows has its own ``except Exception`` →
    ``_mark_backend_unhealthy``, and reaches it after ``asyncio.sleep(20)`` and
    ``asyncio.sleep(40)``.  A pool where every backend is Responses-routed would
    therefore burn sixty seconds and still quarantine everything — the outcome
    the handler exists to prevent, merely delayed.  Hence the early raise above
    that loop, mirroring the ``compaction_exhausted`` precedent.

    Asserting on ``asyncio.sleep`` rather than on elapsed time keeps the test
    deterministic: a wall-clock assertion would be the flakiest kind there is.
    """
    _pin_backend_order(monkeypatch)
    server = _balancing_server(upstream, [REFUSED_MODEL, REFUSED_MODEL])
    cooled: list[int] = []
    slept: list[float] = []
    monkeypatch.setattr(server, "_mark_backend_unhealthy", lambda idx, **kw: cooled.append(idx))

    real_sleep = asyncio.sleep

    async def _recording_sleep(delay: float, *args, **kwargs):
        slept.append(delay)
        return await real_sleep(0)

    # Same caveat as `_pin_backend_order`: this is the stdlib `asyncio`, patched
    # process-wide and reverted by `monkeypatch`.
    monkeypatch.setattr("kitty.bridge.server.asyncio.sleep", _recording_sleep)

    port = await server.start_async()
    try:
        status, body = await _post(port, "/v1/chat/completions", _payload("chat_completions", REFUSED_MODEL, False))
    finally:
        await server.stop_async()

    assert status == 500
    assert REFUSED_MODEL in body
    assert cooled == [], "no backend may be cooled for a refusal that sent nothing"
    assert not [d for d in slept if d >= 20], f"the final retry ladder must be skipped, slept: {slept}"
    assert upstream.hits == []


@pytest.mark.parametrize(
    "pool",
    [
        pytest.param([REFUSED_MODEL, SERVED_MODEL], id="refused-first"),
        pytest.param([SERVED_MODEL, REFUSED_MODEL], id="refused-last"),
    ],
)
@pytest.mark.asyncio
async def test_a_refusal_does_not_steal_the_retry_ladder_from_a_healthy_sibling(
    upstream: _CountingUpstream, monkeypatch, pool: list[str]
):
    """R15 — the refusal must not become the verdict for *another* backend's failure.

    The regression this guards was real and measured.  ``unsupported_model``
    was first written sticky — "a model name is deterministic, so the ladder
    cannot help" — but ``last_exc`` is *not* sticky, so in a mixed pool a later
    backend's unrelated transient failure overwrote it while the flag still
    stood.  The early raise then fired with the wrong exception: a sibling that
    one empty-response retry would have rescued returned a hard 500 blaming a
    model it never used.

    That is verbatim the failure the neighbouring ``compaction_exhausted`` flag
    documents itself against.

    **Both pool orders are exercised, and that is the point of this test.**
    Resetting the flag per attempt — the obvious repair — fixes only the order
    where the refusal comes first.  Put the refused backend last and the flag is
    true at the end of the loop again, so the early raise fires and the ladder
    is stolen exactly as before.  A single-order test stays green through that,
    which is how the first repair looked correct.  The condition the code now
    uses is neither "any attempt refused" nor "the last one did" but **"every
    attempt that ran refused"**, which is order-independent by construction.
    """
    _pin_backend_order(monkeypatch)
    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def _recording_sleep(delay: float, *args, **kwargs):
        slept.append(delay)
        return await real_sleep(0)

    monkeypatch.setattr("kitty.bridge.server.asyncio.sleep", _recording_sleep)

    # One backend refuses; the other is servable but answers empty the first
    # time, so only the retry ladder can save this request.
    upstream.empty_replies = 1
    server = _balancing_server(upstream, pool)
    port = await server.start_async()
    try:
        status, body = await _post(port, "/v1/chat/completions", _payload("chat_completions", REFUSED_MODEL, False))
    finally:
        await server.stop_async()

    assert status == 200, f"the ladder must still run for the servable sibling, got {status}: {body[:200]}"
    assert "ok" in body
    assert slept, "the empty-response ladder was skipped entirely"
