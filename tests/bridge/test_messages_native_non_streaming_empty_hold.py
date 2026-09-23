"""A native-Messages-shaped empty non-streaming completion ends in the D4 terminal on ``/v1/messages`` (KBR-306).

The KBR-298/300/304 sweep closed every non-streaming silent-skeleton cell
**except** the native-Messages arm of ``/v1/messages`` — the
``if cc_response.get("type") == "message"`` branch at
``src/kitty/bridge/server.py:5426``. KBR-300 widened the elif (the KBR-298
``use_custom_transport`` gate) by replacing the transport conjunct with
``not use_native_messages and _is_empty_cc_response`` so the same gate
covers the raw-CC cell (``use_custom_transport = False``, the default);
KBR-304 dropped the ``use_native_messages`` conjunct so native providers —
including the KBR-237 tool-use-format fallback cell on this route (a native
provider whose reply is CC-shaped) — hit the same D4 terminal. Both tickets
left the ``if`` branch structurally unreachable
from the elif (it dispatches on the reply's shape, and Messages-shaped
replies never fall through to the elif).

KBR-306 closes the residual: when the ladder
(``_request_with_retry_single`` / ``_request_with_retry_balancing``)
exhausts a native Messages-shaped empty reply, the ``if`` branch currently
``result = cc_response`` and ships it as a billed, healthy-marked ``200``.
After KBR-306 the branch judges the parsed native reply through the
**Messages-shaped arm** of ``BridgeServer._is_empty_cc_response`` (already
present and pinned by the streaming twin, ``PreambleHold._block_start_releases``
Q14 D1) and ends the request in the route's D4 terminal — bare-JSON
``502`` + ``_NATIVE_EMPTY_REPLY_MESSAGE`` + ``reason: "empty_response"``,
byte-identical to the elif's D4 (``server.py:5492``) and the streaming S11
terminal (``server.py:7077``). One client branch ``(502, reason=empty_response)``
covers the route in both stream modes.

D3 truncation 400 is unchanged — ``_messages_truncation_before_content``
catches ``max_tokens`` / ``model_context_window_exceeded`` on a content-less
reply and fires before the new empty gate. Content-bearing, tool-use-only,
and thinking-only carries over: the Messages arm judges a thinking block as
empty (FR-2, mirroring the streaming hold) and a tool_use block as content.
The reasoning-only trade-off (KBR-287/293/297/298/300) and the
whitespace-only-text (arm-as-is) trade-off carry over unchanged (see
REQUIREMENTS.md).

Every test drives a real in-process :class:`~kitty.bridge.server.BridgeServer`
against an ``aioresponses`` upstream — the KBR-300/304 harness shape. The
provider is the test-only ``_NativeOpenAIAdapter`` stub whose only override
is ``use_native_messages -> True`` (the stub deliberately leaves
``upstream_wire_shape`` at the inherited ``WireShape.CHAT_COMPLETIONS``;
the test exercises the gate's dispatch dimension, not the wire shape). The
upstream mock returns Messages-shaped bodies; ``_make_upstream_request``'s
``_native_messages_request + type == "message"`` guard returns them as-is,
so the handler sees a Messages-shaped ``cc_response`` and dispatches into
the ``if`` branch — the cell the KBR-300/304 gate never reached.

Content oracles are parsed from the JSON response body, never raw substrings
(KBR-249 vacuous-oracle rule).
"""

from __future__ import annotations

import json
import uuid

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge import server as server_module
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
        """Return the launcher binary binary name.

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


class _NativeOpenAIAdapter(OpenAIAdapter):
    """An OpenAI-compatible adapter whose ``use_native_messages`` is True.

    Mirrors the KBR-304 sibling stub (``tests/bridge/test_messages_raw_cc_non_streaming_empty_hold.py``).
    Drives the native-Messages arm of ``/v1/messages`` non-streaming: the
    handler sets ``_native_messages_request = True`` and the mock returns
    Messages-shaped bodies; ``_make_upstream_request``'s native guard
    returns the body as-is, and the handler dispatches on
    ``cc_response.get("type") == "message"`` → True → the ``if`` branch.

    The stub deliberately leaves :attr:`upstream_wire_shape` at the inherited
    :attr:`WireShape.CHAT_COMPLETIONS` — the base-class invariant
    ``use_native_messages ⇒ WireShape.MESSAGES`` is a production-adapter rule
    (see :class:`~kitty.providers.base.ProviderAdapter.use_native_messages`),
    and this test exercises only the gate's dispatch dimension, not the wire
    shape.
    """

    @property
    def use_native_messages(self) -> bool:
        """Return True — the dimension under test.

        Returns:
            Always ``True`` to drive the native-Messages arm.
        """
        return True


# ── Canned upstream shapes ────────────────────────────────────────────────
#
# Native Anthropic Messages JSON bodies. The native guard in
# ``_make_upstream_request`` (the ``_native_messages_request`` flag and the
# ``type == "message"`` shape) returns the body as-is — no
# ``translate_from_upstream`` — so what the handler judges is the exact JSON
# the route receives.


def _native_hello() -> dict:
    """Return a content-bearing native Messages response.

    Returns:
        A Messages body whose ``content`` carries a single text block —
        the R4 fixture from ``test_native_provider_reply_shape.py``, kept
        here for the regression pin.
    """
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }


def _native_empty() -> dict:
    """Return a content-less native Messages response.

    Returns:
        A Messages body whose ``content`` is ``[]`` and ``stop_reason`` is
        ``end_turn`` — the shape the KBR-306 gate judges empty via the
        Messages-shaped arm.
    """
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": [],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }


def _native_thinking_only() -> dict:
    """Return a thinking-only native Messages response.

    Returns:
        A Messages body whose ``content`` is a single ``thinking`` block —
        the shape the Messages-shaped arm judges empty (mirrors
        ``PreambleHold._block_start_releases``: a thinking block does not
        count as content, consistent with the streaming hold).
    """
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": [{"type": "thinking", "thinking": "only reasoning", "signature": "sig"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }


def _native_tool_use_only() -> dict:
    """Return a tool-use-only native Messages response.

    Returns:
        A Messages body whose ``content`` carries two ``tool_use`` blocks
        and no text — the shape the Messages-shaped arm judges non-empty
        (tool_use is not in {text, thinking, redacted_thinking}, so any
        non-text/non-thinking/non-redacted_thinking block counts as content
        per D1).
    """
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a"}},
            {"type": "tool_use", "id": "toolu_2", "name": "Write", "input": {"path": "b"}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }


def _native_truncated() -> dict:
    """Return a truncated native Messages response.

    Returns:
        A Messages body whose ``content`` is ``[]`` and ``stop_reason`` is
        ``max_tokens`` — the KBR-235 D3 shape. ``_messages_truncation_before_content``
        catches this before the new empty gate, so the route answers the
        D3 ``400`` instead of the D4 ``502``.
    """
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": [],
        "stop_reason": "max_tokens",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }


#: The plain raw-CC upstream URL — the ``OpenAIAdapter`` default endpoint,
#: the same URL the KBR-304 harness registers. Native-Messages bodies flow
#: through the same URL because ``_NativeOpenAIAdapter`` inherits
#: ``OpenAIAdapter.get_upstream_path``; the handler's ``_native_messages_request``
#: flag is stripped from the upstream body via ``_INTERNAL_KEYS`` before the
#: POST goes out.
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


def _respond(body: dict) -> CallbackResult:
    """Build a CallbackResult for the canned Messages JSON body.

    Args:
        body: The native Messages response dict to return.

    Returns:
        The aioresponses CallbackResult carrying the body as JSON.
    """
    return CallbackResult(
        status=200,
        body=json.dumps(body),
        content_type="application/json",
    )


async def _post(
    upstream_bodies: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    backends=None,
    on_build=None,
    draws: list[list[int]] | None = None,
) -> tuple[BridgeServer, int, str, int]:
    """POST a non-streaming request through a real bridge against scripted native-Messages responses.

    The provider is the test-only ``_NativeOpenAIAdapter``; the bridge is
    constructed against an ``aioresponses`` upstream so the canned
    Messages JSON is what the handler judges (the native guard in
    ``_make_upstream_request`` returns the body as-is). The retry backoff
    and the empty-ladder final delays are collapsed (lengths untouched —
    the F30 coupling derives bounds from ``len()``) so ladder tests stay
    fast.

    Args:
        upstream_bodies: The canned Messages bodies each ``_make_upstream_request``
            call yields, in order.
        monkeypatch: Pytest fixture, used for the delay collapse and the
            weighted-draw pinning.
        backends: Optional ``[(provider, key, profile), ...]`` tuple — when
            supplied the bridge is constructed in balancing mode; otherwise
            single-backend mode (the provider is the ``_NativeOpenAIAdapter``).
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
        server = BridgeServer(
            _MessagesLauncher(),
            _NativeOpenAIAdapter(),
            "sk-test",
            host="127.0.0.1",
            port=0,
        )
    calls = {"n": 0}

    def _callback(url, **kwargs):
        """Serve the next scripted Messages body and record the hit.

        Args:
            url: The request URL, unused.
            **kwargs: The request parameters, unused.

        Returns:
            The CallbackResult carrying the next scripted Messages body.
            The last body is repeated — a defect that keeps the ladder
            walking fails on its call-count assertion instead of starving
            the script.
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
async def test_an_empty_native_messages_non_streaming_completion_ends_in_the_d4_terminal(monkeypatch):
    """KBR-306 FR-1 — a content-free native-Messages completion ends in the D4 terminal.

    ``_request_with_retry_single`` walks the empty ladder via
    ``_is_non_retryable_reply`` → the Messages-shaped arm of
    ``_is_empty_cc_response`` for
    ``len(_EMPTY_RETRY_DELAYS) + len(_EMPTY_FINAL_DELAYS) + 1`` attempts
    (the route's existing bound, unchanged by this fix), then the new
    empty gate ends the request in the D4 terminal: HTTP ``502`` with
    ``_NATIVE_EMPTY_REPLY_MESSAGE`` and ``reason: "empty_response"``.
    Pre-fix the ``if`` branch ships the empty ``content: []`` body as a
    ``200``, bills it, and marks the backend healthy (in balancing mode);
    post-fix no fabricated fallback text reaches the client, no usage is
    billed, and no healthy-mark fires.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff and
            records ``_log_usage`` / ``_mark_backend_healthy`` calls.
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

    _server, status, client_body, calls = await _post([_native_empty()], monkeypatch, on_build=_record)

    assert status == 502
    assert calls == len(server_module._EMPTY_RETRY_DELAYS) + len(server_module._EMPTY_FINAL_DELAYS) + 1
    error_body = _parse_response(client_body)
    assert error_body["error"]["reason"] == "empty_response"
    assert error_body["error"]["type"] == "api_error"
    assert _NATIVE_EMPTY_REPLY_MESSAGE in error_body["error"]["message"]
    # The exhausted completion was judged, not dressed: no fabricated
    # fallback text, no billed usage, no healthy-mark. The healthy_log
    # assertion is vacuous in single-backend mode (the mark site guards on
    # `if self._backends and self._current_backend_idx >= 0`, always False
    # here — the recorded KBR-300 asymmetry);
    # test_an_empty_native_pool_exhausts_into_the_d4_terminal_without_a_healthy_mark
    # carries the meaningful balancing-mode pin. Both are kept so a refactor
    # does not silently lose either half.
    assert usage_log == []
    assert healthy_log == []


@pytest.mark.asyncio
async def test_a_thinking_only_native_messages_non_streaming_completion_ends_in_the_d4_terminal(monkeypatch):
    """KBR-306 FR-2 — a thinking-only native-Messages completion ends in the D4 terminal.

    The Messages-shaped arm judges a thinking block as empty (mirrors
    ``PreambleHold._block_start_releases``: a thinking block does not
    count as content, consistent with the streaming hold). The ladder
    exhausts the attempts; the new empty gate fires the D4 terminal. The
    carry-over is deliberate — a thinking-only native reply takes the
    ladder — and is documented in REQUIREMENTS.md.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff and
            records the usage / healthy-mark recorders.
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

    _server, status, client_body, calls = await _post([_native_thinking_only()], monkeypatch, on_build=_record)

    assert status == 502
    assert calls == len(server_module._EMPTY_RETRY_DELAYS) + len(server_module._EMPTY_FINAL_DELAYS) + 1
    error_body = _parse_response(client_body)
    assert error_body["error"]["reason"] == "empty_response"
    assert error_body["error"]["type"] == "api_error"
    assert _NATIVE_EMPTY_REPLY_MESSAGE in error_body["error"]["message"]
    # The exhausted completion was judged, not dressed: no fabricated
    # fallback text, no billed usage, no healthy-mark.
    assert usage_log == []
    assert healthy_log == []


@pytest.mark.asyncio
async def test_a_content_bearing_native_messages_non_streaming_completion_reaches_the_client(monkeypatch):
    """KBR-306 FR-3 — a content-bearing native-Messages completion reaches the client.

    The Messages-shaped arm counts a non-empty ``text`` block as content
    (whitespace included — the documented arm-as-is carry-over), so the
    verdict releases the reply on the first attempt: the client receives
    a Messages body whose parsed ``content`` carries the upstream text.
    Exactly one upstream call; usage is logged exactly once. The
    existing R4 test in ``test_native_provider_reply_shape.py`` pins the
    same contract on a real ``_NATIVE_ADAPTERS`` pool; this test pins it
    on the ``_NativeOpenAIAdapter`` stub so the gate's pass-through is
    regression-protected in this file too.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff and
            records ``_log_usage`` calls.
    """
    usage_log: list[dict | None] = []

    def _record(server):
        """Patch the usage recorder onto the server.

        Args:
            server: The bridge instance, attached just before ``start_async``.
        """
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))

    _server, status, client_body, calls = await _post([_native_hello()], monkeypatch, on_build=_record)

    assert status == 200
    assert calls == 1
    assert "hello" in _message_text_blocks(client_body)
    assert len(usage_log) == 1


@pytest.mark.asyncio
async def test_a_tool_use_only_native_messages_non_streaming_completion_releases_the_verdict(monkeypatch):
    """KBR-306 FR-4 — a tool-use-only native-Messages completion is not treated as empty.

    The Messages-shaped arm counts a non-text/non-thinking/non-redacted_thinking
    block as content (D1: a block whose type is not text/thinking/redacted_thinking
    is content), so a ``tool_use`` block — never mind with no text — is
    judged non-empty and the reply releases on the first attempt: the
    client receives a Messages body whose ``content`` carries the two
    tool-use blocks with their arguments whole.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls = await _post([_native_tool_use_only()], monkeypatch)

    assert status == 200
    assert calls == 1
    tool_blocks = _message_tool_blocks(client_body)
    assert [block["name"] for block in tool_blocks] == ["Read", "Write"]
    assert tool_blocks[0]["input"] == {"path": "a"}
    assert tool_blocks[1]["input"] == {"path": "b"}


@pytest.mark.asyncio
async def test_a_truncated_native_messages_non_streaming_completion_ends_in_the_d3_terminal(monkeypatch):
    """KBR-306 FR-5 — a truncated native-Messages completion ends in the D3 terminal.

    ``_messages_truncation_before_content`` catches a Messages-shaped
    reply with ``stop_reason`` in ``_NATIVE_TRUNCATING_STOP_REASONS`` and
    no content, and the handler answers the KBR-235 D3 ``400`` —
    ``_d3_truncation_error_body`` with ``reason: "{stop_reason}_before_content"``.
    The D3 check fires before the new empty gate, so truncation keeps its
    shape (the agent sees a request-shaped failure it will not retry).
    The truncated reply is non-retryable per ``_is_non_retryable_reply``,
    so the ladder ends on attempt 1; no usage is billed (the D3 return
    skips ``_log_usage``); no healthy-mark fires (single-backend mode).

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff and
            records the usage recorder.
    """
    usage_log: list[dict | None] = []

    def _record(server):
        """Patch the usage recorder onto the server.

        Args:
            server: The bridge instance, attached just before ``start_async``.
        """
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))

    _server, status, client_body, calls = await _post([_native_truncated()], monkeypatch, on_build=_record)

    assert status == 400
    assert calls == 1
    error_body = _parse_response(client_body)
    assert error_body["error"]["type"] == "invalid_request_error"
    assert error_body["error"]["reason"] == "max_tokens_before_content"
    assert usage_log == []


@pytest.mark.asyncio
async def test_an_empty_native_messages_non_streaming_attempt_crosses_to_a_healthy_plain_peer(monkeypatch):
    """KBR-306 FR-6 — on a ``[native, plain]`` pool the empty native walk crosses to the plain peer.

    ``_request_with_retry_balancing``'s empty arm selects the next
    backend class-agnostically. The native backend's empty Messages walk
    crosses to the plain peer; the plain peer's content-bearing CC reply
    passes through ``translate_from_upstream`` (passthrough for the
    ``OpenAIAdapter``) and reaches the handler as a CC-shaped body; the
    handler dispatches on shape — ``type != "message"`` → elif → non-empty
    → translate → 200 with the upstream content as a Messages body. The
    new empty gate sits in the ``if`` branch, so it does not fire on the
    content-bearing crossed peer reply (this is the regression pin the
    test enforces — a future regression that fires the gate on a
    content-bearing crossed reply would break it).

    Args:
        monkeypatch: Pytest fixture, pins the weighted draws and collapses
            the retry backoff.
    """
    native = _NativeOpenAIAdapter()
    plain = OpenAIAdapter()
    native_profile = Profile(
        name="p-native",
        provider="openai",
        model="test-model",
        auth_ref=str(uuid.uuid4()),
    )
    plain_profile = Profile(
        name="p-plain",
        provider="openai",
        model="test-model",
        auth_ref=str(uuid.uuid4()),
    )
    backends = [
        (native, "key-native", native_profile),
        (plain, "key-plain", plain_profile),
    ]
    plain_hello_cc = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    _server, status, client_body, calls = await _post(
        [_native_empty(), plain_hello_cc],
        monkeypatch,
        backends=backends,
        draws=[[0], [1]],
    )

    assert status == 200
    assert calls == 2
    assert "hello" in _message_text_blocks(client_body)


@pytest.mark.asyncio
async def test_an_empty_native_pool_exhausts_into_the_d4_terminal_without_a_healthy_mark(monkeypatch):
    """KBR-306 FR-1 in balancing mode — full native-pool exhaustion ends in the D4 terminal with no healthy-mark.

    The single-backend test's ``healthy_log == []`` assertion is vacuous
    in single-backend mode (the route's ``_mark_backend_healthy`` site
    guards on ``if self._backends and self._current_backend_idx >= 0``,
    which is always False in single mode). Balancing-mode full
    exhaustion is the meaningful pin: a ``[native, native]`` pool whose
    both backends return empty Messages-shaped replies exhausts the
    failover loop (``n_backends`` attempts) plus the
    ``_EMPTY_FINAL_DELAYS`` final retries, and the new empty gate fires
    the D4 terminal. No ``_log_usage``, no ``_mark_backend_healthy`` —
    the meaningful balancing-mode "no healthy-mark" pin.

    Args:
        monkeypatch: Pytest fixture, pins the weighted draws and collapses
            the retry backoff. Records ``_log_usage`` /
            ``_mark_backend_healthy`` calls.
    """
    native_a = _NativeOpenAIAdapter()
    native_b = _NativeOpenAIAdapter()
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
        (native_a, "key-a", profile_a),
        (native_b, "key-b", profile_b),
    ]
    usage_log: list[dict | None] = []
    healthy_log: list[int] = []

    def _record(server):
        """Patch the usage and healthy-mark recorders onto the server.

        Args:
            server: The bridge instance, attached just before ``start_async``.
        """
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))
        monkeypatch.setattr(server, "_mark_backend_healthy", lambda idx: healthy_log.append(idx))

    n_backends = len(backends)
    expected_calls = n_backends + len(server_module._EMPTY_FINAL_DELAYS)
    # Four draws: initial, failover, final retry 1, final retry 2.
    _server, status, client_body, calls = await _post(
        [_native_empty()],
        monkeypatch,
        backends=backends,
        draws=[[0], [1], [0], [0]],
        on_build=_record,
    )

    assert status == 502
    assert calls == expected_calls
    error_body = _parse_response(client_body)
    assert error_body["error"]["reason"] == "empty_response"
    assert error_body["error"]["type"] == "api_error"
    assert _NATIVE_EMPTY_REPLY_MESSAGE in error_body["error"]["message"]
    # The exhausted completion was judged, not dressed: no billed usage,
    # no healthy-mark. This is the meaningful balancing-mode pin for the
    # route's "no healthy-mark on the empty arm" guarantee.
    assert usage_log == []
    assert healthy_log == []
