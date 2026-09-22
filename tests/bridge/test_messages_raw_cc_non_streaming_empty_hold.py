"""A raw-CC upstream's empty non-streaming completion ends in the D4 terminal on ``/v1/messages`` (KBR-300).

KBR-298 closed the ``use_custom_transport`` cell on ``/v1/messages``'
non-streaming branch — the ``elif`` at ``src/kitty/bridge/server.py:5353`` that
gates only when ``self._active_provider.use_custom_transport`` is true. The
raw-CC cell was deliberately scoped out and filed as this ticket: a
``use_custom_transport = False`` (the default) provider's empty completion
falls through the elif to ``else: result = translator.translate_response(...)``,
``MessagesTranslator.translate_response`` dresses the empty parsed body in
``_EMPTY_ASSISTANT_FALLBACK_TEXT``, ``_log_usage`` bills the fabricated turn,
and ``_mark_backend_healthy`` keeps the broken upstream in rotation.

KBR-300 widens the existing elif by replacing the transport conjunct with the
ticket's single predicate (``not self._active_provider.use_native_messages``)
so the same gate covers both the KBR-298 cell (still closed) and the raw-CC
cell. Content-bearing translates and responds unchanged; judged-empty ends in
the route's D4 terminal (``502`` + ``reason: "empty_response"``), with no usage
billed and no healthy-mark. The raw-CC ladder is ``_request_with_retry``'s
existing walk — single-backend:
``len(_EMPTY_RETRY_DELAYS) + len(_EMPTY_FINAL_DELAYS) + 1`` attempts;
balancing: ``n_backends``, empties never marking a backend unhealthy — and is
unchanged.

The KBR-277 ``reasoning_content`` predicate carries over: on a raw-CC upstream
the wire shape is CC, so a ``reasoning_content``-only reply is judged
non-empty and ``MessagesTranslator`` maps it to a thinking block (one upstream
call, no ladder).

Every test drives a real in-process :class:`~kitty.bridge.server.BridgeServer`
against an ``aioresponses`` upstream, the harness KBR-276 proved for the
plain-POST raw-CC class. The provider is the plain ``OpenAIAdapter`` (the
adapter's ``translate_from_upstream`` is the base passthrough, so the canned
CC JSON arrives at the handler unchanged). Content oracles are parsed from
the JSON response body, never raw substrings (KBR-249 vacuous-oracle rule).
"""

from __future__ import annotations

import json
import uuid

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge import server as server_module
from kitty.bridge.messages.translator import _EMPTY_ASSISTANT_FALLBACK_TEXT
from kitty.bridge.server import _NATIVE_EMPTY_REPLY_MESSAGE, BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.openai import OpenAIAdapter
from kitty.types import BridgeProtocol


class _MessagesLauncher(LauncherAdapter):
    """Minimal launcher that selects the Messages bridge protocol."""

    @property
    def name(self) -> str:
        """Return the launcher name.

        Returns:
            A fixed placeholder.
        """
        return "stub"

    @property
    def binary_name(self) -> str:
        """Return the launcher binary name.

        Returns:
            A fixed placeholder.
        """
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        """Return the protocol the bridge serves.

        Returns:
            The Messages protocol — every test here is a Claude Code client.
        """
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(self, profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        """Return an empty spawn configuration.

        Args:
            profile: Unused.
            bridge_port: Unused.
            resolved_key: Unused.

        Returns:
            A spawn configuration that changes nothing.
        """
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


# ── Canned upstream shapes ────────────────────────────────────────────────
#
# OpenAI Chat Completions JSON bodies; the OpenAI adapter passes them through
# ``translate_from_upstream`` unchanged, so what the handler judges is the
# exact JSON the route receives.


def _cc_hello() -> dict:
    """Return a content-bearing CC response.

    Returns:
        A CC body whose ``choices[0].message.content`` carries ``"hello"``.
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


def _cc_empty() -> dict:
    """Return a content-less CC response.

    Returns:
        A CC body whose ``choices[0].message.content`` is empty and which carries
        no tool calls — the upstream shape the KBR-285 lockstep judge counts
        as empty.
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    }


def _cc_tool_calls() -> dict:
    """Return a tool-call-only CC response.

    Returns:
        A CC body whose ``choices[0].message.tool_calls`` carries two complete
        calls and no text content.
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Read", "arguments": '{"path": "a"}'},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "Write", "arguments": '{"path": "b"}'},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


def _cc_reasoning_only() -> dict:
    """Return a reasoning-content-only CC response.

    Returns:
        A CC body whose ``choices[0].message`` carries ``reasoning_content``
        and an empty ``content`` — the KBR-277 predicate counts
        ``reasoning_content`` non-empty as content, so the gate must NOT fire
        (the reply is released on the first attempt, ``MessagesTranslator``
        maps ``reasoning_content`` to a thinking block).
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "only reasoning",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    }


#: The plain raw-CC upstream URL — the ``OpenAIAdapter`` default endpoint, the
#: same URL the KBR-298 mixed-pool test registers for the plain peer.
_UPSTREAM_URL = "https://api.openai.com/v1/chat/completions"


# ── Harness ───────────────────────────────────────────────────────────────


def _client_request() -> dict:
    """Return a minimal non-streaming Messages API request.

    Returns:
        The request body a Claude Code client sends with ``stream: false``.
    """
    return {
        "model": "test-model",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }


def _parse_response(body_text: str) -> dict:
    """Parse the non-streaming JSON response body.

    Args:
        body_text: The body the client received.

    Returns:
        The decoded JSON document. The route answers JSON in every outcome
        under test (a translated message or the D4 error), so a decode
        failure is itself a defect this helper surfaces loudly.
    """
    return json.loads(body_text)


def _message_text_blocks(body_text: str) -> list[str]:
    """Collect every ``text`` block's text from a translated Messages body.

    Args:
        body_text: The JSON body the client received.

    Returns:
        The text block texts, in content order — the route-protocol content
        oracle (never a raw-substring check).
    """
    parsed = _parse_response(body_text)
    return [
        block.get("text", "")
        for block in parsed.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]


def _message_tool_blocks(body_text: str) -> list[dict]:
    """Collect every ``tool_use`` block from a translated Messages body.

    Args:
        body_text: The JSON body the client received.

    Returns:
        The tool_use blocks, in content order.
    """
    parsed = _parse_response(body_text)
    return [
        block for block in parsed.get("content", []) if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def _message_thinking_blocks(body_text: str) -> list[dict]:
    """Collect every ``thinking`` block from a translated Messages body.

    Args:
        body_text: The JSON body the client received.

    Returns:
        The thinking blocks, in content order. The KBR-277 / KBR-285
        ``reasoning_content`` interaction test (AC-FR-1.4) pins the
        MessagesTranslator mapping ``reasoning_content`` to a thinking block.
    """
    parsed = _parse_response(body_text)
    return [
        block for block in parsed.get("content", []) if isinstance(block, dict) and block.get("type") == "thinking"
    ]


def _respond(body: dict) -> CallbackResult:
    """Build a CallbackResult for the canned CC JSON body.

    Args:
        body: The CC response dict to return.

    Returns:
        The aioresponses CallbackResult carrying the body as JSON.
    """
    return CallbackResult(
        status=200,
        body=json.dumps(body),
        content_type="application/json",
    )


async def _post(
    provider,
    upstream_bodies: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    backends=None,
    on_build=None,
    draws: list[list[int]] | None = None,
) -> tuple[BridgeServer, int, str, int]:
    """POST a non-streaming request through a real bridge against scripted raw-CC responses.

    The provider is a plain ``OpenAIAdapter``; the bridge is constructed
    against an ``aioresponses`` upstream so the canned CC JSON is what the
    handler judges. The retry backoff and the empty-ladder final delays are
    collapsed (lengths untouched — the F30 coupling derives bounds from
    ``len()``) so ladder tests stay fast.

    Args:
        provider: A plain ``OpenAIAdapter`` instance.
        upstream_bodies: The canned CC bodies each ``_make_upstream_request``
            call yields, in order.
        monkeypatch: Pytest fixture, used for the delay collapse and the
            weighted-draw pinning.
        backends: Optional ``[(provider, key, profile), ...]`` tuple — when
            supplied the bridge is constructed in balancing mode; otherwise
            single-backend mode.
        on_build: Optional callable invoked with the constructed server before
            it starts — the seam for tests that patch instance methods (e.g.
            the usage and health recorders).
        draws: Optional list of weighted-draw pin lists (e.g.
            ``[[0], [1]]``) overriding ``random.choices`` to make backend
            selection deterministic; only meaningful with ``backends``.

    Returns:
        The server, the HTTP status, the client's body text, and how many
        times the upstream was entered.
    """
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.01)
    monkeypatch.setattr(server_module, "_EMPTY_RETRY_DELAYS", [0.01, 0.01])
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])
    if backends is not None:
        server = BridgeServer(
            _MessagesLauncher(),
            backends[0][0],
            backends[0][1],
            host="127.0.0.1",
            port=0,
            backends=backends,
        )
    else:
        server = BridgeServer(_MessagesLauncher(), provider, "sk-test", host="127.0.0.1", port=0)
    calls = {"n": 0}

    def _callback(url, **kwargs):
        """Serve the next scripted CC body and record the hit.

        Args:
            url: The request URL, unused.
            **kwargs: The request parameters, unused.

        Returns:
            The CallbackResult carrying the next scripted CC body. The last
            body is repeated — a defect that keeps the ladder walking fails on
            its call-count assertion instead of starving the script.
        """
        calls["n"] += 1
        body = upstream_bodies[min(calls["n"] - 1, len(upstream_bodies) - 1)]
        return _respond(body)

    with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
        for _registration in range(len(server_module._EMPTY_RETRY_DELAYS) + len(server_module._EMPTY_FINAL_DELAYS) + 4):
            mocked.post(_UPSTREAM_URL, callback=_callback)
        if draws is not None:
            iter_draws = iter(draws)
            monkeypatch.setattr(server_module.random, "choices", lambda tier, weights=None, k=None: next(iter_draws))
        if on_build is not None:
            on_build(server)
        await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(f"http://127.0.0.1:{server.port}/v1/messages", json=_client_request()) as resp,
            ):
                return server, resp.status, await resp.text(), calls["n"]
        finally:
            await server.stop_async()


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_content_bearing_raw_cc_non_streaming_completion_reaches_the_client(monkeypatch):
    """KBR-300 AC-FR-1.1 — a content-bearing raw-CC completion translates and responds unchanged.

    The widened KBR-298 gate sits after the ladder; the ladder ends on the
    first non-retryable reply (a content-bearing CC body). The client receives
    a Messages body whose parsed ``content`` carries the upstream text.
    Exactly one upstream call; usage is logged exactly once.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff and records
            ``_log_usage`` / ``_mark_backend_healthy`` calls.
    """
    usage_log: list[dict | None] = []
    healthy_log: list[int] = []

    def _record(server):
        """Patch the usage and healthy-mark recorders onto the server.

        Args:
            server: The bridge instance, attached just before ``start_async``.
        """
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))
        monkeypatch.setattr(server, "_mark_backend_healthy", lambda idx: healthy_log.append(idx))

    _server, status, client_body, calls = await _post(
        OpenAIAdapter(), [_cc_hello()], monkeypatch, on_build=_record
    )

    assert status == 200
    assert calls == 1
    assert "hello" in _message_text_blocks(client_body)
    assert len(usage_log) == 1


@pytest.mark.asyncio
async def test_an_empty_raw_cc_non_streaming_completion_ends_in_the_d4_terminal(monkeypatch):
    """KBR-300 AC-FR-1.2 — a content-free raw-CC completion ends in the D4 terminal.

    ``_request_with_retry``'s walk exhausts on a single-backend pool at
    ``len(_EMPTY_RETRY_DELAYS) + len(_EMPTY_FINAL_DELAYS) + 1`` attempts — the
    route's existing bound, unchanged by this fix — and the widened gate then
    ends the request in the D4 terminal: a JSON ``502`` with
    ``_NATIVE_EMPTY_REPLY_MESSAGE`` and ``reason: "empty_response"``. Before
    the fix the handler shipped the translator's fabricated fallback text as
    a ``200``, billed it, and marked the backend healthy; after it no
    fallback text reaches the client, no usage is billed, and no healthy-mark
    fires.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff and records
            ``_log_usage`` / ``_mark_backend_healthy`` calls.
    """
    usage_log: list[dict | None] = []
    healthy_log: list[int] = []

    def _record(server):
        """Patch the usage and healthy-mark recorders onto the server.

        Args:
            server: The bridge instance, attached just before ``start_async``.
        """
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))
        monkeypatch.setattr(server, "_mark_backend_healthy", lambda idx: healthy_log.append(idx))

    _server, status, client_body, calls = await _post(
        OpenAIAdapter(), [_cc_empty()], monkeypatch, on_build=_record
    )

    assert status == 502
    assert calls == len(server_module._EMPTY_RETRY_DELAYS) + len(server_module._EMPTY_FINAL_DELAYS) + 1
    error_body = _parse_response(client_body)
    assert error_body["error"]["reason"] == "empty_response"
    assert error_body["error"]["type"] == "api_error"
    assert _NATIVE_EMPTY_REPLY_MESSAGE in error_body["error"]["message"]
    # The exhausted completion was judged, not dressed: no fabricated
    # fallback text, no billed usage, no healthy-mark.
    assert _EMPTY_ASSISTANT_FALLBACK_TEXT not in client_body
    assert usage_log == []
    assert healthy_log == []


@pytest.mark.asyncio
async def test_a_tool_call_only_raw_cc_non_streaming_completion_releases_the_verdict(monkeypatch):
    """KBR-300 AC-FR-1.3 — a tool-call-only raw-CC completion is not treated as empty.

    ``_is_empty_cc_response`` counts a non-empty ``tool_calls`` list as
    content (KBR-285 lockstep), so the verdict releases the reply on the
    first attempt: the client receives a Messages body whose ``content``
    carries the two tool-use blocks with their arguments whole.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls = await _post(OpenAIAdapter(), [_cc_tool_calls()], monkeypatch)

    assert status == 200
    assert calls == 1
    tool_blocks = _message_tool_blocks(client_body)
    assert [block["name"] for block in tool_blocks] == ["Read", "Write"]
    assert tool_blocks[0]["input"] == {"path": "a"}
    assert tool_blocks[1]["input"] == {"path": "b"}


@pytest.mark.asyncio
async def test_a_reasoning_only_raw_cc_non_streaming_completion_releases_the_verdict(monkeypatch):
    """KBR-300 AC-FR-1.4 — a reasoning-only raw-CC completion is not treated as empty.

    ``_is_empty_cc_response`'s CC arm counts ``reasoning_content`` non-empty
    as content (KBR-277 lockstep); on a raw-CC upstream the wire shape is CC,
    so the reply is judged non-empty and released on the first attempt.
    ``MessagesTranslator`` maps ``reasoning_content`` to a Messages thinking
    block — the reasoning carriage reaches the client whole. **Pre-existing
    translator rider (out of KBR-300 scope):** ``MessagesTranslator`'s
    "never emit thinking-only or empty assistant output" defensive fallback
    appends the empty-assistant text to a thinking-only release; this is the
    translator's design, predates KBR-300, and is unchanged by the gate. The
    gate's job here is to confirm the ladder releases the reply (no
    fabricated gate firing) — not to retire the translator's defensive
    fallback text.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls = await _post(OpenAIAdapter(), [_cc_reasoning_only()], monkeypatch)

    assert status == 200
    assert calls == 1
    thinking_blocks = _message_thinking_blocks(client_body)
    assert len(thinking_blocks) == 1
    # The thinking block carries the upstream reasoning carriage whole.
    assert "only reasoning" in (thinking_blocks[0].get("thinking") or "")


@pytest.mark.asyncio
async def test_an_empty_raw_cc_non_streaming_attempt_crosses_to_a_healthy_plain_peer(monkeypatch):
    """KBR-300 AC-FR-1.5 — on a raw-CC pool the empty walk crosses to the healthy plain peer.

    ``_request_with_retry_balancing``'s empty arm selects the next backend
    class-agnostically. On a plain-peer pool the crossed attempt lands on a
    content-bearing reply; the gate must not fire on a content-bearing
    crossed reply, and the peer content reaches the client as a Messages
    body. Backend 0 was entered exactly once.

    Args:
        monkeypatch: Pytest fixture, pins the weighted draws and collapses
            the retry backoff.
    """
    plain_a = OpenAIAdapter()
    plain_b = OpenAIAdapter()
    profile_a = Profile(
        name="p-a",
        provider="openai",
        model="test-model",
        auth_ref=str(uuid.uuid4()),
    )
    profile_b = Profile(
        name="p-b",
        provider="openai",
        model="test-model",
        auth_ref=str(uuid.uuid4()),
    )
    backends = [
        (plain_a, "key-a", profile_a),
        (plain_b, "key-b", profile_b),
    ]
    _server, status, client_body, calls = await _post(
        plain_a,
        [_cc_empty(), _cc_hello()],
        monkeypatch,
        backends=backends,
        draws=[[0], [1]],
    )

    assert status == 200
    assert calls == 2
    assert "hello" in _message_text_blocks(client_body)
    assert _EMPTY_ASSISTANT_FALLBACK_TEXT not in client_body
