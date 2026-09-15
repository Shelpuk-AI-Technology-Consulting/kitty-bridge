"""L2 — the native-passthrough route forwards the agent's cache breakpoints.

Jira: `KBR-200 <https://shelpuk.atlassian.net/browse/KBR-200>`_ (CB-3, epic
`KBR-197 <https://shelpuk.atlassian.net/browse/KBR-197>`_). Design:
``.system_design/TEST_SUITE.md`` §2.2 (allocation rule), §3.2.1 row M16, and
§6.2.3 (L2 contract drives observe at the ``_upstream_body_for``
serialization boundary).

Native passthrough (``use_native_messages == True``) is the only route on
which a Claude Code user's prompt cache survives today. Three adapters
qualify — ``zai_anthropic`` (always native), ``custom_anthropic`` (always),
and ``minimax_token`` (opt-in); the L1 sweep in
``tests/providers/test_native_passthrough_cache_breaks.py`` covers all three.
This L2 file drives the **bridge branch end to end** through a real
``BridgeServer`` and observes the **wire-bound body** the adapter ships, at
the ``_upstream_body_for`` boundary §6.2.3 names. The L1 sweep guards the
adapter in isolation; this one guards the bridge's branch around it.

**Two hazards the guard covers.** The native branch's first statement is
``cc_request = dict(body)`` ``server.py`` ~4111 — a shallow copy. Two later
steps mutate ``cc_request`` before it ships:

1. ``BridgeServer._normalize_model`` — writes ``cc_request["model"]``.
2. ``provider.normalize_request`` — no-op today for all three native
   adapters, but the hook is per-adapter and free to mutate.

A test that observes only ``dict(body)`` proves nothing about what ships; the
serialization-boundary observation proves what *actually* reaches the wire.
A second test pins the shallow copy's safety for the inbound ``body`` at the
branch's top level.

**Why every site is asserted (the ``tool_result_nested`` correction).** KBR-200
comment 4 notes that *whether* Anthropic honours a breakpoint nested inside a
``tool_result`` content block is not established. What is established by
KBR-198/KBR-199 is that the bridge forwards it on the translated route; the
native branch's passthrough does the same on its route. CB-3 covers
**forwarding**, not Anthropic's honouring of the forwarded field — that is
CB-4/CB-5's wire concern, not this one's. So every site in
``harness.cache_breakpoints.SITES`` is asserted here.

**The M9 fallback's exact breakpoint behaviour (pinned, not fixed).** The
``tool_use`` format-error fallback — ``_convert_native_to_cc_format``
``server.py`` ~702 — is reached only from the four streaming handlers
(``_stream_responses`` ~3580, ``_stream_messages`` ~4669, ``_stream_gemini``
~5964, ``_stream_chat_completions`` ~7137). Its rebuild loses most
breakpoints; two survive by carriage and restore, on the adapters whose
flags admit the restore:

- **``system``** survives on ``zai_anthropic`` and ``custom_anthropic`` (both
  set ``forwards_thinking_signature = True``). The converter carries
  ``body["system"]`` verbatim under ``_anthropic_system`` (~845-847, KBR-228
  part B); the rebuild restores it verbatim.
- **``document``** survives on all three native adapters. The shared user-
  content builder (``bridge.messages.translator.build_user_content_message``)
  appends the whole block verbatim into ``_documents``; the rebuild
  re-attaches those blocks verbatim.

The other eight sites (``tool``, ``image``, ``user_text``, ``assistant_text``,
``tool_use``, ``tool_result``, ``tool_result_nested``, ``top_level``) do not
survive — pinned as today's behaviour.

**Why L2 drives the serialization boundary, not the socket.** Per §6.2.3,
the wire-bound body lives at ``BridgeServer._upstream_body_for``. A transport-
level capture adds two layers of indirection (the HTTP client and the
recording upstream) that obscure which byte came from which adapter hook.
The serialization boundary makes the guard readable: the file's asserts name
the layer they observe.

**Bedrock caveat.** A top-level ``cache_control`` — automatic caching — is
400s on legacy Bedrock (Opus 4.6 and earlier). If a native Bedrock-shaped
provider is ever added, "preserve faithfully" and "do not break the request"
pull in opposite directions on that endpoint. Recorded here as the next
reader's trade-off, not as a behaviour this suite asserts.

**This proves the field survives, not that the cache hits.** Anthropic
invalidates the cache when the thinking configuration or
``output_config.effort`` changes; that defect is
`CB-6 (KBR-203) <https://shelpuk.atlassian.net/browse/KBR-203>`_'s scope,
not this one's. A reader of green tests here must not conclude the native
route's caching is fully proven.
"""

from __future__ import annotations

import contextlib
import copy
import json
from collections.abc import Callable, Iterable
from typing import Any

import pytest
from aiohttp.test_utils import make_mocked_request
from harness import cache_breakpoints as cb

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter
from kitty.providers.base import ProviderAdapter
from kitty.providers.custom_anthropic import CustomAnthropicAdapter
from kitty.providers.minimax_token import MiniMaxTokenAnthropicAdapter
from kitty.providers.zai_anthropic import ZaiAnthropicAdapter
from kitty.types import BridgeProtocol

pytestmark = pytest.mark.l2


# ── Test doubles ───────────────────────────────────────────────────────────


class _FakeLauncher(LauncherAdapter):
    """A launcher adapter stub the bridge accepts without spawning anything."""

    @property
    def name(self) -> str:
        return "fake"

    @property
    def binary_name(self) -> str:
        return "fake"

    @property
    def agent_name(self) -> str:
        return "fake"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(
        self, profile: Any, bridge_port: int, resolved_key: str, *, context_tokens: int | None = None
    ) -> dict:
        return {}

    def prepare_launch(self, spawn_config: dict) -> None:
        return None

    def cleanup_launch(self, spawn_config: dict) -> None:
        return None


class _StopBeforeWire(Exception):
    """Raised by the capture spy once the wire-bound body has been recorded."""


def _breakpoint_paths(node: Any, path: tuple[Any, ...] = ()) -> Iterable[tuple[Any, ...]]:
    """Yield the path of every ``cache_control`` key anywhere in *node*.

    The fixture's breakpoint detector matches by value and by key name; this
    yields the *position* of each key. A matched key is not recursed into —
    the breakpoint dict has no nested ``cache_control``, and the detector
    does not search the value either.

    Args:
        node: A JSON-shaped structure of dicts, lists and scalars.
        path: The path leading to *node*; callers omit it, recursion
            accumulates it.

    Yields:
        One path per ``cache_control`` key found, outermost key first.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and "cache_control" in key:
                yield path + (key,)
            else:
                yield from _breakpoint_paths(value, path + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _breakpoint_paths(value, path + (index,))


def _make_request(body: Any) -> Any:
    """Build a fake aiohttp ``web.Request`` that parses to *body* on ``.json()``.

    Uses ``aiohttp.test_utils.make_mocked_request`` so the handler sees a real
    request surface, then instance-patches ``json`` so the parsed body is
    *body* by identity — load-bearing for the shallow-copy test, where the
    branch must write into a *copy* of the dict we hold.

    Args:
        body: The object ``await request.json()`` returns — the same object
            the caller holds, not a copy.

    Returns:
        A request usable with ``BridgeServer._handle_messages``.
    """
    request = make_mocked_request("POST", "/v1/messages")
    # ``transport.is_closing()`` defaults to a truthy Mock; the stream handler
    # polls the transport on the recovery-hold path. Pin it to ``False``.
    request.transport.is_closing.return_value = False

    async def _json() -> Any:
        return body

    request.json = _json
    return request


# ── Drive the bridge branch to the serialization boundary ────────────────


async def _capture_wire_body(
    monkeypatch: pytest.MonkeyPatch, server: BridgeServer, body: Any
) -> Any:
    """Return the body ``_upstream_body_for`` would ship — without firing the network.

    Class-patches ``BridgeServer._upstream_body_for`` so the real serialization
    runs and records its result, then raises ``_StopBeforeWire`` so the
    handler returns a 500 instead of opening a real TCP socket. The 500 is
    swallowed at the call site.

    Args:
        monkeypatch: The calling test's patch context; the spy is undone
            with it.
        server: The bridge to drive.
        body: The inbound request body; handed to ``_make_request``.

    Returns:
        The single wire-bound body ``_upstream_body_for`` produced.

    Raises:
        AssertionError: When the drive produced zero or several wire bodies.
    """
    captures: list[Any] = []
    original = BridgeServer._upstream_body_for

    def spy(self: BridgeServer, cc_request: dict) -> dict:
        sent = original(self, cc_request)  # type: ignore[arg-type]
        captures.append(sent)
        raise _StopBeforeWire

    monkeypatch.setattr(BridgeServer, "_upstream_body_for", spy)
    request = _make_request(body)
    # The drive's exit path is irrelevant; only the capture matters.
    with contextlib.suppress(Exception):
        await server._handle_messages(request)
    assert len(captures) == 1
    return captures[0]


@pytest.fixture
def native_server() -> BridgeServer:
    """A ``BridgeServer`` wired with ``ZaiAnthropicAdapter`` and no real launcher."""
    return BridgeServer(_FakeLauncher(), ZaiAnthropicAdapter(), "sk-zai-test123", host="127.0.0.1", port=0)


# ── AC1 / R1 — every breakpoint reaches the wire at full value ────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("site", cb.SITES)
async def test_breakpoint_reaches_the_wire_at_full_value(
    monkeypatch: pytest.MonkeyPatch, native_server: BridgeServer, site: str
) -> None:
    """The breakpoint at *site* survives the native branch at its full value.

    Drives a real ``BridgeServer`` with the fixture body for *site*, observes
    the wire-bound body at ``_upstream_body_for``, and asserts:

    * exactly one breakpoint at the same position the fixture placed it;
    * equal to ``cache_breakpoints.BREAKPOINT`` (``{"type": "ephemeral",
      "ttl": "1h"}``) by value — a presence check would silently pass a
      carry-through that downgraded the TTL.

    The site set includes ``tool_result_nested`` (see module docstring):
    whether Anthropic honours a breakpoint at that depth is not established,
    but the bridge forwarding it is.
    """
    body = cb.build_request(site)
    sent = await _capture_wire_body(monkeypatch, native_server, body)

    assert cb.find_breakpoints(sent) == [dict(cb.BREAKPOINT)]
    assert list(_breakpoint_paths(sent)) == list(_breakpoint_paths(body))
    # The internal carriage never reaches the wire.
    assert "_native_messages_request" not in sent
    # The fixture's top-level keys all reach the wire (the branch is a true
    # pass-through for the agent's request body).
    assert sent["model"] == body["model"]
    assert sent["system"] == body["system"]


# ── R4 (L2) — the bridge drive fails when the adapter drops the breakpoint ─


class _BreakpointDropper(ZaiAnthropicAdapter):
    """Stand-in adapter whose ``normalize_request`` removes every breakpoint."""

    def normalize_request(self, cc_request: dict) -> None:
        def strip(node: object) -> None:
            if isinstance(node, dict):
                node.pop("cache_control", None)
                for value in node.values():
                    strip(value)
            elif isinstance(node, list):
                for value in node:
                    strip(value)

        strip(cc_request)


@pytest.mark.asyncio
@pytest.mark.parametrize("site", cb.SITES)
async def test_falsification_bridge_drive_detects_a_dropper_adapter(
    monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    """Wire the dropper in and the bridge drive sees no breakpoint at *site*.

    The sibling of the guard above: same drive, dropper adapter. The
    captured wire body carries zero breakpoints, so the R1 assertion
    ``find_breakpoints(sent) == [BREAKPOINT]`` would go red. This test
    characterises the broken side — proving the guard is not vacuous.
    """
    server = BridgeServer(_FakeLauncher(), _BreakpointDropper(), "sk-zai-test123", host="127.0.0.1", port=0)
    body = cb.build_request(site)
    sent = await _capture_wire_body(monkeypatch, server, body)
    assert cb.find_breakpoints(sent) == []


# ── R5 / AC5 — the shallow copy does not let the branch write back ───────


@pytest.mark.asyncio
async def test_shallow_copy_does_not_write_back_into_the_caller_body(
    monkeypatch: pytest.MonkeyPatch, native_server: BridgeServer
) -> None:
    """The native branch's writes stay on the copy.

    Patches ``_truncate_oversized_tool_results`` (called immediately after the
    branch's last statement) to raise ``_BranchObserved`` once the branch has
    run. The drive ends without ever opening a network connection. The test
    then asserts three properties:

    * the recorded ``cc_request`` is **not** the parsed body — the branch
      wrote a shallow copy, so this is true today and false the day the
      branch stops copying (``cc_request = body``);
    * the recorded ``cc_request`` carries the native flag (the branch's
      write landed on the copy);
    * the parsed body is byte-identical to its pre-drive snapshot — none of
      the branch's writes aliased back.

    The copy protects top-level writes. Block-level writes (i.e., mutating an
    entry inside ``body["messages"]``) are not protected by ``dict(body)``;
    no branch in this code path performs one today, and the test documents
    that fact rather than asserting a guarantee the shallow copy does not
    offer.
    """
    parsed = cb.build_request("top_level")
    snapshot = copy.deepcopy(parsed)

    observed: dict = {}

    class _BranchObserved(Exception):
        """The branch has run; stop before the next step."""

    def stop_after_branch(self: BridgeServer, cc_request: dict) -> None:
        observed["cc"] = cc_request
        raise _BranchObserved

    monkeypatch.setattr(BridgeServer, "_truncate_oversized_tool_results", stop_after_branch)

    request = _make_request(parsed)
    with contextlib.suppress(_BranchObserved):
        await native_server._handle_messages(request)

    cc = observed["cc"]

    # The branch produced a different object — the shallow copy held.
    assert cc is not parsed
    # The branch's writes landed on the copy.
    assert cc["_native_messages_request"] is True
    assert cb.find_breakpoints(cc) == [dict(cb.BREAKPOINT)]
    # No write-back into the caller's body — full deep equality on every
    # cache_control-carrying key.
    assert parsed == snapshot
    assert "_native_messages_request" not in parsed
    # The breakpoint survives the round trip on the body side, at full value.
    assert cb.find_breakpoints(parsed) == [dict(cb.BREAKPOINT)]


# ── R6 / AC6 — M9 fallback breakpoint behaviour pinned ────────────────────


_TOOL_USE_ERROR_TEXT = (
    "unknown variant 'tool_use', expected one of 'text', 'image', 'document'"
)
"""A 400 body that matches ``_is_tool_use_format_error`` but not the
thinking-signature or thinking-roundtrip detectors, forcing the M9 fallback."""

_NON_TOOL_USE_ERROR_TEXT = "still no"
"""A 400 body that matches no retryable classifier — used to break the loop
after the fallback fires once."""


class _StubStreamUpstream:
    """An aiohttp-like stream response carrying a fixed ``status`` and body.

    The streaming handler does ``async with session.post(...) as upstream:``
    then reads ``upstream.status`` and ``upstream.text()``.
    """

    def __init__(self, status: int, text: str) -> None:
        """Store the status and body text the handler will read.

        Args:
            status: HTTP status code returned to the handler.
            text: Plain-text body returned by ``await upstream.text()``.
        """
        self.status = status
        self._text = text

    async def text(self) -> str:
        """Return the configured body text."""
        return self._text


class _StubSession:
    """An aiohttp session stub whose ``post`` defers to an attempt factory.

    ``_stream_messages`` opens its upstream with
    ``async with session.post(url, json=..., headers=..., timeout=...)``;
    each ``post`` consumes one response from the factory, indexed by attempt.
    """

    def __init__(
        self, factory: Callable[[int], _StubStreamUpstream], attempt_index: dict[str, int]
    ) -> None:
        """Stash the factory and the shared attempt counter.

        Args:
            factory: Callable mapping attempt index to a stub upstream.
            attempt_index: Mutable counter shared with the drive; ``post``
                increments and reads it to pick the factory's input.
        """
        self._factory = factory
        self._attempt_index = attempt_index

    def post(self, url: str, *, json: Any = None, headers: Any = None, timeout: Any = None) -> _AsyncCtx:
        """Return a context manager yielding the next stub upstream.

        Args:
            url: Matched by the streaming loop but irrelevant under the stub.
            json: Outgoing body; ignored by the stub.
            headers: Outgoing headers; ignored by the stub.
            timeout: Request timeout; ignored by the stub.

        Returns:
            A context manager whose ``__aenter__`` yields the next upstream.
        """
        idx = self._attempt_index["n"]
        self._attempt_index["n"] += 1
        return _AsyncCtx(self._factory(idx))


class _AsyncCtx:
    """An async context manager yielding a fixed value, like ``session.post(...)``."""

    def __init__(self, value: Any) -> None:
        """Stash the value the context manager yields.

        Args:
            value: The object returned by ``__aenter__``.
        """
        self._value = value

    async def __aenter__(self) -> Any:
        """Return the stashed value."""
        return self._value

    async def __aexit__(self, *exc: object) -> bool:
        """Never suppress; return ``False``."""
        return False


async def _drive_streaming_fallback(
    monkeypatch: pytest.MonkeyPatch,
    server: BridgeServer,
    body: Any,
    upstream_factory: Callable[[int], _StubStreamUpstream],
) -> list[dict]:
    """Run ``_stream_messages`` against a stubbed upstream, returning every wire body.

    Class-patches ``_session_for`` and ``_upstream_body_for``:

    * ``_upstream_body_for`` records its result (real serialization) and lets
      the drive continue — capturing both the pre-fallback native passthrough
      and the post-fallback Anthropic rebuild bodies.
    * ``_session_for`` returns a :class:`_StubSession` whose posts defer to
      ``upstream_factory(attempt_index)``, so the test hands back a
      400-with-tool-use-wording on attempt 0 and a non-retryable 400 on
      attempt 1, terminating the loop. No socket is ever opened.

    Args:
        monkeypatch: The calling test's patch context.
        server: The bridge to drive; must have ``use_native_messages`` true.
        body: The inbound request body; handed to ``_make_request``.
        upstream_factory: Maps attempt index to the stub upstream's status
            and text.

    Returns:
        Every wire-bound body ``_upstream_body_for`` produced, in order.
    """
    captures: list[dict] = []
    original_upstream_body_for = BridgeServer._upstream_body_for

    def capture_upstream_body(self: BridgeServer, cc_request: dict) -> dict:
        sent = original_upstream_body_for(self, cc_request)  # type: ignore[arg-type]
        captures.append(sent)
        return sent

    attempt_index = {"n": 0}

    async def session_for(self: BridgeServer, url: str) -> _StubSession:
        return _StubSession(upstream_factory, attempt_index)

    monkeypatch.setattr(BridgeServer, "_upstream_body_for", capture_upstream_body)
    monkeypatch.setattr(BridgeServer, "_session_for", session_for)

    request = _make_request(body)
    # The handler catches generic exceptions and returns an error response;
    # some paths still propagate. We only care about the captures.
    with contextlib.suppress(Exception):
        await server._handle_messages(request)

    return captures


def _fallback_factory() -> Callable[[int], _StubStreamUpstream]:
    """Build the upstream factory that forces the M9 fallback once.

    The first attempt's body matches ``_is_tool_use_format_error`` (forcing
    the converter to run); every later attempt returns a 400 with a message
    that no retryable classifier matches, breaking the streaming loop.

    Returns:
        A factory suitable for ``_drive_streaming_fallback``.
    """

    def factory(attempt: int) -> _StubStreamUpstream:
        if attempt == 0:
            return _StubStreamUpstream(400, json.dumps({
                "type": "error",
                "error": {"type": "invalid_request_error", "message": _TOOL_USE_ERROR_TEXT},
            }))
        return _StubStreamUpstream(400, json.dumps({
            "type": "error",
            "error": {"type": "invalid_request_error", "message": _NON_TOOL_USE_ERROR_TEXT},
        }))

    return factory


#: The sites that survive the M9 fallback per native adapter.
#:
#: * ``zai_anthropic`` and ``custom_anthropic`` set
#:   ``forwards_thinking_signature = True``, so the rebuild restores the
#:   ``_anthropic_system`` carriage verbatim — ``system`` survives alongside
#:   the always-restored ``_documents`` carriage.
#: * ``minimax_token`` sets the flag False: its rebuild re-joins the system
#:   messages to one string, so ``system`` is lost. ``document`` still
#:   survives (the ``_documents`` restore is not gated on the flag).
_SURVIVES_ON_ADAPTER: dict[str, frozenset[str]] = {
    "zai_anthropic": frozenset({"system", "document"}),
    "custom_anthropic": frozenset({"system", "document"}),
    "minimax_token": frozenset({"document"}),
}

_FALLBACK_ADAPTERS: tuple[tuple[str, Callable[[], ProviderAdapter]], ...] = (
    ("zai_anthropic", ZaiAnthropicAdapter),
    ("custom_anthropic", CustomAnthropicAdapter),
    (
        "minimax_token",
        lambda: MiniMaxTokenAnthropicAdapter(native_messages=True),
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("site", cb.SITES)
@pytest.mark.parametrize("adapter_name, adapter_factory", _FALLBACK_ADAPTERS, ids=[a[0] for a in _FALLBACK_ADAPTERS])
async def test_m9_fallback_preserves_carriage_breakpoints_and_loses_the_rest(
    monkeypatch: pytest.MonkeyPatch,
    site: str,
    adapter_name: str,
    adapter_factory: Callable[[], ProviderAdapter],
) -> None:
    """The M9 fallback's breakpoint behaviour is pinned per site × per adapter.

    Drives the streaming native route to force ``_convert_native_to_cc_format``
    (the tool_use format-error fallback): the upstream first returns a 400
    whose body matches ``_is_tool_use_format_error``; the bridge converts and
    retries; the test captures both wire bodies. Two axes are pinned:

    * **Per site** (10 values): what the converter preserves vs. drops.
    * **Per native adapter** (3 values): which carriers the rebuild restores.
      The split is gated by the adapter's flags:

      - ``system`` survives on the adapters that set
        ``forwards_thinking_signature = True`` (``zai_anthropic``,
        ``custom_anthropic``); ``minimax_token``'s flag is False and its
        rebuild joins system to a string.
      - ``document`` survives on all three — the ``_documents`` restore is
        not flag-gated.

    The per-site + per-adapter cross-product proves the gate, not just one
    half of it: a regression that breaks the ``_documents`` restore on
    ``minimax_token``'s native-opt-in path is caught here, and a regression
    that drops the ``_anthropic_system`` carriage on either signature-
    binding adapter is caught here too. For ``tool_result_nested`` the
    rebuild flattens ``tool_result.content`` to a string on every adapter —
    the mechanism pin catches a regression that un-flattens it.
    """
    server = BridgeServer(_FakeLauncher(), adapter_factory(), "sk-test-key", host="127.0.0.1", port=0)
    body = cb.build_request(site)
    body["stream"] = True
    captures = await _drive_streaming_fallback(monkeypatch, server, body, _fallback_factory())

    # Pre-fallback capture: native passthrough carries the breakpoint intact
    # (the same claim R1 makes on the streaming path).
    assert cb.find_breakpoints(captures[0]) == [dict(cb.BREAKPOINT)]

    # Post-fallback capture: the rebuild.
    retry = captures[1]
    survives = _SURVIVES_ON_ADAPTER[adapter_name]
    if site in survives:
        assert cb.find_breakpoints(retry) == [dict(cb.BREAKPOINT)], (
            f"site {site!r} survives the M9 fallback on {adapter_name!r} via "
            "the _anthropic_system / _documents carriage"
        )
    else:
        assert cb.find_breakpoints(retry) == [], (
            f"site {site!r} should be lost on the M9 fallback for {adapter_name!r}"
        )

    # Mechanism pin for tool_result_nested: the rebuild flattens content on
    # every adapter (verified at the serialization boundary rather than from
    # the converter output, so the pin catches a regression at either layer).
    # The retry body is Anthropic-shaped (rebuilt from the CC intermediate by
    # ``AnthropicAdapter.translate_to_upstream``), so the relevant block type
    # is ``tool_result``, not the CC shape's ``tool``.
    if site == "tool_result_nested":
        for message in retry.get("messages", []):
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    inner = block.get("content")
                    assert isinstance(inner, str), (
                        "M9 fallback flattens tool_result.content to a string — "
                        "pin the mechanism, not Anthropic's nested-depth behaviour"
                    )


@pytest.mark.asyncio
async def test_no_fallback_when_body_has_no_tool_use_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: a body without tool_use blocks never triggers the fallback.

    The fallback's condition checks ``_has_tool_use_blocks(body)``; this
    drive strips the fixture's tool_use block and asserts exactly **one**
    wire-body capture — proving the two-capture drive in the test above is
    the fallback, not a coincidence.
    """
    server = BridgeServer(_FakeLauncher(), ZaiAnthropicAdapter(), "sk-zai-test123", host="127.0.0.1", port=0)
    body = cb.build_request("user_text")
    # Drop the assistant turn's tool_use so ``_has_tool_use_blocks(body)`` is False.
    body["messages"] = [msg for msg in body["messages"] if msg["role"] != "assistant"]
    body["stream"] = True
    captures = await _drive_streaming_fallback(monkeypatch, server, body, _fallback_factory())
    assert len(captures) == 1
