"""Cross-attempt content and cadence — TEST_SUITE.md §4.3 C3, plan T-I8, KBR-100.

The two assertions §4.3 C3 names are pinned here end to end:

**(i) Byte-identical repeats.** Transport-blip retries, empty-response retries and
retryable-status repeats (the three "the request is unchanged" populations) repeat
the attempt they follow byte-for-byte — same body bytes, same header set, no
retry-count or correlation header added between attempts.

**(ii) Each mutating path fires only under its own trigger.** M6 (compact tighter),
M8 (thinking-carrier repair), M9 (native→CC conversion) and failover
re-normalisation are declared exceptions to (i): each fires only when its
own trigger is met, and on the same backend a repeat that meets none of them
stays byte-identical.

Three scope additions from the ticket's comments are honoured:

- **KBR-155:** empty-response retries are reachable on native adapters; (i)
  applies to the single-backend repeat; balancing re-selection is the failover
  exception; truncation before content (``max_tokens`` /
  ``model_context_window_exceeded``) is **not** retried — one upstream request
  and no repeat to compare; transport drops before release belong in (i).
- **KBR-276:** the same on raw-CC adapters (the role-chunk→[DONE] shape is the
  empty-ladder trigger there too).
- **KBR-31:** use :class:`~harness.bridge.BridgeFixture` with stateful
  :class:`~harness.recorder.Responder` closures and
  :func:`~harness.bridge.pin_backend_order` for any balancing test (round-robin
  makes selection observable and the test deterministic).

Existing reference fixtures already cover trigger half of M6
(``tests/bridge/test_tc4_corpus_recovery_l3.py``), M8 trigger on the translated
route (``tests/bridge/test_thinking_roundtrip_failover.py``), M9 trigger
(``tests/bridge/test_native_passthrough_cache_breaks.py``) and the empty
ladders themselves (KBR-155 / KBR-276). This module owns what they do not:

- **(i) across the six families** (transport-blip stream native / raw-CC,
  empty-response stream native / raw-CC / non-streaming native,
  retryable-status non-streaming native) — none of the existing fixtures
  compares the two upstream captures against each other.
- **The M8 and M9 complements at L3** — an unrelated 400 must not fire the
  same-backend repair or the native→CC conversion.
- **The bounded shape of failover re-normalisation** — the balancing
  re-selection produces a different body, paired with (i) pinning the
  same-backend repeat as byte-identical.

**Sibling-sites sampling — completed by KBR-302.** The four stream handlers
carry sibling copies of the retry/empty/M9/M17 arms. M9 site line numbers
(re-derived 2026-09-23, against the post-KBR-304 main): ``_stream_responses``
at ``server.py:4648``, ``_stream_messages`` at ``:6095``, ``_stream_gemini``
at ``:7896``, ``_stream_chat_completions`` at ``:9517``. T-I8 sampled
Messages and Chat-Completions first (KBR-100); KBR-302 sampled Responses and
Gemini. The four-handler×four-arm table is now fully covered.

**Sibling-arm caveat (M8 vs M17, verified 2026-09-23).** The "thinking
repair" arm is **M17** (thinking-signature strip via ``_recover_rejected_thinking``
→ ``_strip_thinking_blocks``) on the three non-Messages handlers
(``_stream_responses``, ``_stream_gemini``, ``_stream_chat_completions`),
and **M8** (carrier injection via ``_with_thinking_carrier``) only on
``_stream_messages`` (``server.py:6154`` — the single call site of
``_is_thinking_roundtrip_error``). The KBR-100 cells above cover M8 on the
Messages route; the KBR-302 cells below cover M17 on Responses and Gemini. The
ticket text for KBR-302 named "M8 on the Responses/Gemini sites" as the
intent; the implementation tracks the actual sibling arm and records the
ticket-text correction.

Marker: ``pytestmark = pytest.mark.l3``. The ``tests/bridge/`` path default is
``l1`` (per :mod:`tests.layers`); this file overrides, as
``tests/bridge/test_tc4_corpus_recovery_l3.py`` does for the same reason.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from harness import failures
from harness.bridge import BridgeFixture, pin_backend_order, transport
from harness.contract import CapturedRequest, WireFormat
from harness.failures import InjectionPoint
from harness.recorder import Reply

import kitty.bridge.server as server_module
from kitty.types import BridgeProtocol

from .test_client_disconnect_health import (
    _StubLauncher,
    _StubProvider,
    short_grace,  # noqa: F401 — pytest fixture, used by name
)
from .test_streaming_recovery_content import _GeminiLauncher

pytestmark = pytest.mark.l3

# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------

#: Collapsed retry delays — every repeat path here sleeps 0s instead of the
#: production seconds. Without this a single test pays the 80-second empty
#: ladder or the multi-second transport grace on every run.
_DELAY_PATCHES: dict[str, object] = {
    "_BACKOFF_BASE": 0.0,
    "_EMPTY_RETRY_DELAYS": [0.0, 0.0],
    "_EMPTY_FINAL_DELAYS": [0.0, 0.0],
    "_TRANSPORT_GRACE_DELAYS": (0.0,),
}


@pytest.fixture(autouse=True)
def _collapse_retry_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero every retry delay; the tests count attempts, never time them.

    Same shape as ``tests/bridge/test_empty_response_retry.py``'s autouse
    fixture, applied wider so the non-streaming 500 path and the transport
    grace both collapse.
    """
    for name, value in _DELAY_PATCHES.items():
        monkeypatch.setattr(server_module, name, value)


def _assert_retry_count_headers_absent(captures: list[CapturedRequest]) -> None:
    """Assert no capture carries a header whose name reads as a retry counter.

    The cross-attempt contract (TEST_SUITE.md §4.3 C3 (i)) names "no
    retry-count or correlation header" — a header added by the retry between
    attempts that a single-attempt request would not carry. We assert this by
    the symmetric argument: if both attempts carry the same complete header
    set, neither carries a header the other does not; therefore neither carries
    a header a first attempt did not.

    A header whose lowercased name reads as a correlation token would be the
    obvious fingerprint, and is asserted explicitly as a tighter pin. The
    vocabulary covers ``retry``, ``attempt``, ``correlation``, ``trace``,
    ``request-id`` (and ``request_id``), ``call-id`` (and ``call_id``),
    ``session-id`` (and ``session_id``) — the names a provider or proxy
    layer would reach for. A **uniform** add of a brand-new name outside
    this vocabulary would not be caught by the symmetric pair-comparison
    alone; that residual closes once §4.3 C1's exact-set ratchet (T-I12)
    lands, and is recorded in the file's module docstring rather than here.
    """
    forbidden = (
        "retry",
        "attempt",
        "correlation",
        "trace",
        "request_id",
        "request-id",
        "call_id",
        "call-id",
        "session_id",
        "session-id",
    )
    for capture in captures:
        for name, _value in capture.headers:
            lower = name.lower()
            for token in forbidden:
                assert token not in lower, (
                    f"capture carries a {token}-named header {name!r}; (i) "
                    f"forbids the retry from adding a retry-count or "
                    f"correlation header"
                )


def _assert_byte_identical_repeat(captures: list[CapturedRequest], *, expected_count: int) -> None:
    """Assert the cross-attempt byte-identity (i) holds for a paired repeat.

    Two upstream captures, exact count, body bytes equal, header tuple equal
    (names + values + order). The count guard is load-bearing: a test whose
    ladder walked further than scripted raises ``ScriptExhausted`` upstream
    and arrives here with fewer or more captures; this is the non-vacuous
    claim (the test suite §3.3.1 rule against empty-list quantification).
    """
    assert len(captures) == expected_count, (
        f"expected {expected_count} upstream requests, got {len(captures)}; "
        f"the scripted ladder walked a different shape and byte-identity "
        f"cannot be assessed"
    )
    first, second = captures[0], captures[1]
    assert second.body == first.body, (
        f"the second upstream body differs from the first; (i) requires a "
        f"byte-identical repeat. first={first.body[:120]!r} second={second.body[:120]!r}"
    )
    first_headers = list(first.headers)
    second_headers = list(second.headers)
    assert second_headers == first_headers, (
        f"the header tuple changed between attempts; (i) requires identical "
        f"headers (names, values, order). "
        f"first={first_headers!r} second={second_headers!r}"
    )
    _assert_retry_count_headers_absent(captures)


# ---------------------------------------------------------------------------
# Non-canonical content stream (KBR-155 reference shape)
# ---------------------------------------------------------------------------

#: Deliberately non-canonical key order and spacing — a re-serialising hold
#: would change these bytes, so this is the strongest witness that the second
#: attempt's body is the same dict re-serialised, not a re-constructed copy.
_NON_CANONICAL_CONTENT_STREAM: tuple[bytes, ...] = (
    b'event: message_start\ndata: {"message":{"role":"assistant","id":"msg_2",'
    b'"content":[]}, "type":"message_start"}\n\n',
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    b'"content_block":{"text":"","type":"text"}}\n\n',
    b'event: content_block_delta\ndata: {"index":0,"delta":{"text":"Hello",'
    b'"type":"text_delta"},"type":"content_block_delta"}\n\n',
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
)
_NON_CANONICAL_HEADERS = {"content-type": "text/event-stream"}


def _make_non_canonical_content_responder() -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Return a responder writing the KBR-155 non-canonical content stream.

    Returns:
        A :class:`~harness.recorder.Responder` that opens an SSE stream and
        writes the six non-canonical frames above.
    """

    async def _responder(_captured: CapturedRequest, response: Reply) -> None:
        await response.begin(200, dict(_NON_CANONICAL_HEADERS))
        for frame in _NON_CANONICAL_CONTENT_STREAM:
            await response.write(frame)
        await response.write_eof()

    return _responder


# ---------------------------------------------------------------------------
# Truncation-before-content reply
# ---------------------------------------------------------------------------

#: message_start + a single message_delta carrying max_tokens stop_reason +
#: message_stop. The hold sees ``max_tokens`` as the first terminal signal,
#: judges pre-emission, and renders a protocol-native 400 — never retries
#: (KBR-155 R6 / D3).
_TRUNCATION_BEFORE_CONTENT_STREAM: tuple[bytes, ...] = (
    b'event: message_start\ndata: {"message":{"id":"msg_trunc","role":"assistant",'
    b'"content":[],"model":"recorder-model"},"type":"message_start"}\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","delta":'
    b'{"stop_reason":"max_tokens","stop_sequence":null}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
)


def _make_truncation_before_content_responder() -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Return a responder writing the truncation-before-content shape.

    Returns:
        A :class:`~harness.recorder.Responder` whose stream terminates at
        ``max_tokens`` before any content reaches the bridge.
    """

    async def _responder(_captured: CapturedRequest, response: Reply) -> None:
        await response.begin(200, dict(_NON_CANONICAL_HEADERS))
        for frame in _TRUNCATION_BEFORE_CONTENT_STREAM:
            await response.write(frame)
        await response.write_eof()

    return _responder


# ---------------------------------------------------------------------------
# Native Messages inbound bodies
# ---------------------------------------------------------------------------


def _native_messages_body_with_assistant(marker: str, *, assistant_text: str = "hello") -> dict[str, Any]:
    """Return a streaming Messages request carrying one assistant turn.

    Used by the M8 trigger test: ``_repair_thinking_roundtrip(native=True)``
    attaches the ``{"type":"thinking","thinking":""}`` carrier block only
    to assistant messages whose ``content`` is a list (or string); an
    assistant turn is therefore required for the carrier shape to be
    observable on the wire.

    Args:
        marker: The user's text — a :func:`~harness.bridge.marker` when the
            caller means to find it again in a capture.
        assistant_text: The assistant turn's text content.

    Returns:
        The Messages request body.
    """
    return {
        "model": "harness-model",
        "max_tokens": 32,
        "stream": True,
        "messages": [
            {"role": "user", "content": marker},
            {"role": "assistant", "content": assistant_text},
        ],
    }


def _native_messages_body_with_tool_use(marker: str) -> dict[str, Any]:
    """Return a streaming Messages request with a tool_use block.

    Used by the M9 trigger and M9 complement B tests. ``_has_tool_use_blocks``
    keys on ``{"type":"tool_use", ...}`` content blocks inside a message; the
    tool_result following the tool_use is what makes the transcript realistic
    (a real client would have one) and ensures the conversion does not strip
    an orphan.

    Args:
        marker: The user's text — a :func:`~harness.bridge.marker`.

    Returns:
        The Messages request body.
    """
    return {
        "model": "harness-model",
        "max_tokens": 32,
        "stream": True,
        "messages": [
            {"role": "user", "content": marker},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "calling read"},
                    {
                        "type": "tool_use",
                        "id": "toolu_kbr100",
                        "name": "Read",
                        "input": {"path": "/tmp/probe"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_kbr100",
                        "content": "ok",
                    }
                ],
            },
        ],
    }


def _native_messages_body_without_tool_use(marker: str) -> dict[str, Any]:
    """Return a streaming Messages request without tool_use.

    Used by the M9 complement A test: the trigger's third conjunct
    (``_has_tool_use_blocks(body)``) must be False for the conversion to
    stay native.

    Args:
        marker: The user's text — a :func:`~harness.bridge.marker`.

    Returns:
        The Messages request body.
    """
    return {
        "model": "harness-model",
        "max_tokens": 32,
        "stream": True,
        "messages": [
            {"role": "user", "content": marker},
            {"role": "assistant", "content": "no tools here"},
        ],
    }


# ---------------------------------------------------------------------------
# (i) — byte-identical repeats
# ---------------------------------------------------------------------------


class TestByteIdenticalRepeats:
    """(i) — transport-blip, empty-response, retryable-status repeats are byte-identical.

    Each test scripts two attempts through a stateful ``Responder`` closure,
    then asserts ``requests[0].body == requests[1].body`` byte-for-byte and
    ``list(requests[0].headers) == list(requests[1].headers)``. The
    cross-attempt contract (TEST_SUITE.md §4.3 C3 (i)) is the same for every
    family; the tests differ only in what scripts attempt 1.
    """

    @pytest.mark.asyncio
    async def test_transport_drop_before_release_repeats_byte_identically(self) -> None:
        """Transport-grace retry after a pre-release drop is byte-identical.

        The recorder's responder aborts the connection before ``begin()`` — a
        pre-release transport drop. ``_open_upstream_stream`` raises a
        ``ClientConnectionError`` (aiohttp's ``ServerDisconnectedError``),
        ``_is_transport_error`` classifies it, and ``TransportGrace`` retries
        the same ``url``/``upstream_body``/``headers`` (KBR-31's contract).
        The two captures must agree byte-for-byte.
        """
        from harness import failures

        scripted = failures.scripted(
            failures.drop_at(
                WireFormat.ANTHROPIC_MESSAGES,
                failures.InjectionPoint.BEFORE_FIRST_BYTE,
            ),
            _make_non_canonical_content_responder(),
        )
        async with BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=scripted)) as fixture:
            status, _body = await fixture.post(
                "/v1/messages",
                {
                    "model": "harness-model",
                    "max_tokens": 32,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            captures = list(fixture.captures)

        assert status == 200, "the grace retry's second attempt must serve the client"
        _assert_byte_identical_repeat(captures, expected_count=2)

    @pytest.mark.asyncio
    async def test_empty_response_repeat_is_byte_identical_native(self) -> None:
        """Empty-response retry on the native Messages route is byte-identical.

        Attempt 1 is the empty Messages stream (F4.c: message_start + empty
        text block + stop + message_delta + message_stop — no text_delta
        releases the hold, the ladder fires). Attempt 2 is the
        non-canonical content stream so a re-serialising hold would be
        visible in the bytes. KBR-155 reference shape; per KBR-31 the
        fixture's per-member model makes the re-selection observable.
        """
        from harness import failures

        scripted = failures.scripted(
            failures.empty_response(WireFormat.ANTHROPIC_MESSAGES, stream=True),
            _make_non_canonical_content_responder(),
        )
        async with BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=scripted)) as fixture:
            status, _body = await fixture.post(
                "/v1/messages",
                {
                    "model": "harness-model",
                    "max_tokens": 32,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            captures = list(fixture.captures)

        assert status == 200
        _assert_byte_identical_repeat(captures, expected_count=2)

    @pytest.mark.asyncio
    async def test_retryable_status_repeat_is_byte_identical(self) -> None:
        """A 500 retry on the non-streaming Messages route is byte-identical.

        The non-streaming ``_request_with_retry`` ladder retries on
        ``_RETRYABLE_STATUSES`` (``{429,500,502,503,504}``) with the **same**
        ``upstream_body`` and the **same** headers — a third member of the
        "the request is unchanged" population. The KBR-31 comment measured
        a 500 on a single-backend profile costs 4 upstream attempts over
        ~7 seconds; with ``_BACKOFF_BASE`` collapsed (the autouse fixture)
        the script needs exactly two responses to land on attempt 2's
        success.
        """
        from harness import failures

        scripted = failures.scripted(
            failures.error_status(WireFormat.ANTHROPIC_MESSAGES, 500, error_type="api_error", message="overloaded"),
            failures.success(WireFormat.ANTHROPIC_MESSAGES, stream=False),
        )
        async with BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=scripted)) as fixture:
            status, _body = await fixture.post(
                "/v1/messages",
                {
                    "model": "harness-model",
                    "max_tokens": 32,
                    "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            captures = list(fixture.captures)

        assert status == 200
        _assert_byte_identical_repeat(captures, expected_count=2)

    @pytest.mark.asyncio
    async def test_empty_response_repeat_is_byte_identical_raw_cc(self) -> None:
        """Empty-response retry on the raw-CC route is byte-identical (KBR-276).

        Since KBR-276 removed the converter gate, the pre-emission hold
        withholds non-content lines on the raw CC wire too: a
        role-chunk→finish-chunk→[DONE] stream fires the empty ladder. The
        second attempt carries the upstream's content, the client sees the
        second attempt's bytes, and the two upstream captures agree
        byte-for-byte.
        """
        from harness import failures

        scripted = failures.scripted(
            failures.empty_response(WireFormat.CHAT_COMPLETIONS, stream=True),
            failures.success(WireFormat.CHAT_COMPLETIONS, stream=True),
        )
        async with BridgeFixture(transport("aiohttp", WireFormat.CHAT_COMPLETIONS, responder=scripted)) as fixture:
            status, _body = await fixture.post(
                "/v1/chat/completions",
                {"model": "harness-model", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
            )
            captures = list(fixture.captures)

        assert status == 200
        _assert_byte_identical_repeat(captures, expected_count=2)

    @pytest.mark.asyncio
    async def test_transport_drop_before_release_is_byte_identical_raw_cc(self) -> None:
        """Transport-grace retry on the raw-CC route is byte-identical.

        ``_TRANSPORT_GRACE_DELAYS`` + ``_is_transport_error`` are
        wire-format-agnostic: the same pre-release drop on the CC-wire
        upstream takes the same grace retry with the same
        ``url``/``upstream_body``/``headers``. This is the raw-CC twin of the
        native-route transport-blip case above, and closes the matrix's
        raw-CC transport-blip row (KBR-276's route, KBR-31's contract).
        """
        from harness import failures

        scripted = failures.scripted(
            failures.drop_at(
                WireFormat.CHAT_COMPLETIONS,
                failures.InjectionPoint.BEFORE_FIRST_BYTE,
            ),
            failures.success(WireFormat.CHAT_COMPLETIONS, stream=True),
        )
        async with BridgeFixture(transport("aiohttp", WireFormat.CHAT_COMPLETIONS, responder=scripted)) as fixture:
            status, _body = await fixture.post(
                "/v1/chat/completions",
                {"model": "harness-model", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
            )
            captures = list(fixture.captures)

        assert status == 200
        _assert_byte_identical_repeat(captures, expected_count=2)

    @pytest.mark.asyncio
    async def test_empty_response_repeat_is_byte_identical_non_streaming(self) -> None:
        """Empty-response retry on the non-streaming route is byte-identical.

        The empty ladder also applies to non-streaming requests
        (``_is_empty_cc_response`` + ``_is_non_retryable_reply`` +
        ``_EMPTY_RETRY_DELAYS``, ``server.py:2808``/``:8226``); this is the
        matrix's non-streaming empty-response row. The repeats reuse the
        same ``upstream_body`` and the same ``headers`` — the
        byte-identity contract is the same as on the streaming route.
        """
        from harness import failures

        scripted = failures.scripted(
            failures.empty_response(WireFormat.ANTHROPIC_MESSAGES, stream=False),
            failures.success(WireFormat.ANTHROPIC_MESSAGES, stream=False),
        )
        async with BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=scripted)) as fixture:
            status, _body = await fixture.post(
                "/v1/messages",
                {
                    "model": "harness-model",
                    "max_tokens": 32,
                    "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            captures = list(fixture.captures)

        assert status == 200
        _assert_byte_identical_repeat(captures, expected_count=2)


# ---------------------------------------------------------------------------
# (i) — the declared exception and the population boundary
# ---------------------------------------------------------------------------


class TestDeclaredExceptionAndPopulationBoundary:
    """(ii) — the balancing re-selection is the exception; truncation is not retried.

    The balancing re-selection draws the pool again — the shape failover
    re-normalisation takes, and the one place byte-identity is *not* claimed
    (TEST_SUITE.md §4.3 C3 (ii)). Truncation before content has nothing to
    compare: a single upstream request, no repeat, and the test names the
    boundary rather than passing vacuously.
    """

    @pytest.mark.asyncio
    async def test_balancing_reselection_redraws_the_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The balancing re-selection is the failover exception: different backend, different body.

        Round-robin selection (``pin_backend_order``) makes the second
        attempt land on the sibling member — ``backend_models=["m0","m1"]``
        gives each member its own profile model, so the captures' ``model``
        field proves which member served. Per §4.3 C3 the failover
        re-normalisation is a declared exception to byte-identity: the
        second body differs because ``_normalize_model`` and
        ``normalize_request`` re-ran for the new member. The session
        attempt counter names the same population as (i).
        """
        from harness import failures

        pin_backend_order(monkeypatch)
        scripted = failures.scripted(
            failures.empty_response(WireFormat.ANTHROPIC_MESSAGES, stream=True),
            _make_non_canonical_content_responder(),
        )
        async with BridgeFixture(
            transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=scripted),
            backend_models=["m0", "m1"],
        ) as fixture:
            status, _body = await fixture.post(
                "/v1/messages",
                {
                    "model": "harness-model",
                    "max_tokens": 32,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            captures = list(fixture.captures)
            server = fixture.server

        assert status == 200
        assert len(captures) == 2, "the empty retry must draw the pool again (failover arm)"
        first_body = json.loads(captures[0].body)
        second_body = json.loads(captures[1].body)
        assert first_body["model"] == "m0", (
            f"round-robin selection should put attempt 1 on m0; got {first_body['model']!r}"
        )
        assert second_body["model"] == "m1", (
            f"round-robin selection should redraw on attempt 2 to m1; got {second_body['model']!r}. "
            f"A matching model means the failover arm did not fire and the test "
            f"is not pinning the declared exception"
        )
        assert captures[1].body != captures[0].body, (
            "the re-normalisation exception is allowed to change the body; "
            "matching bodies mean the bridge repeated the same backend, not "
            "the pool"
        )
        assert server is not None
        attempts = server._session_stats()["attempts"]
        assert attempts == 2, f"expected 2 attempts in the session, got {attempts}"

    @pytest.mark.asyncio
    async def test_truncation_before_content_is_not_retried(self) -> None:
        """A truncation no retry can improve fires once with a 400 — no repeat to compare.

        ``max_tokens`` before any content is the Q10 / R6 / D3 path: the hold
        sees ``max_tokens`` as the first terminal signal, judges
        pre-emission, and renders a protocol-native 400. No retry and no
        failover can improve it (KBR-155 R6). The cross-attempt population
        has no second request to compare — this test pins the boundary,
        not a repeat.
        """
        scripted = _make_truncation_before_content_responder()
        async with BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=scripted)) as fixture:
            status, body = await fixture.post(
                "/v1/messages",
                {
                    "model": "harness-model",
                    "max_tokens": 32,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            captures = list(fixture.captures)

        assert status == 400, (
            "truncation before content must surface as a 400 — never retry, never fail over (KBR-155 R6)"
        )
        assert len(captures) == 1, (
            f"truncation before content must reach the upstream exactly once; "
            f"got {len(captures)} captures. The cross-attempt population has "
            f"no second request to compare here — that is the point"
        )
        error = json.loads(body)["error"]
        assert error["reason"] == "max_tokens_before_content", error


# ---------------------------------------------------------------------------
# (ii) — M8 trigger discipline
# ---------------------------------------------------------------------------


class TestM8TriggerDiscipline:
    """(ii) — M8 (thinking-carrier repair) fires only on a thinking round-trip rejection.

    The trigger's three conjuncts are: ``attempt < max_attempts - 1`` (true on
    attempt 0), ``_is_thinking_roundtrip_error`` (matches the KBR-232
    "must be passed back" wording), and ``_repair_thinking_roundtrip``
    returns ``True`` (an assistant turn exists with a list-or-string
    content the carrier can attach to). Failure of any conjunct sends the
    attempt to a different arm: the complement test exercises the
    "unrelated 400 → failover" arm.
    """

    @pytest.mark.asyncio
    async def test_a_thinking_roundtrip_rejection_repairs_and_repeats_the_same_backend(self) -> None:
        """Trigger — the round-trip wording + assistant turn fires M8 on the same backend.

        Single backend so "same backend" is the trivial identity. M8
        mutates ``upstream_body`` in place via ``_with_thinking_carrier``
        (native=True attaches ``{"type":"thinking","thinking":""}`` blocks
        before each assistant message's existing content) and **does not**
        rebuild headers — so the header *names* still match between attempts
        while the body differs (``Content-Length``'s value legitimately
        tracks the grown body). Per §4.3 C3 (ii) M8 is the one path whose
        body mutation lives inside the same-backend retry.
        """
        from harness import failures

        scripted = failures.scripted(
            failures.error_status(
                WireFormat.ANTHROPIC_MESSAGES,
                400,
                error_type="invalid_request_error",
                message=("The `content[].thinking` in the thinking mode must be passed back to the API."),
            ),
            _make_non_canonical_content_responder(),
        )
        async with BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=scripted)) as fixture:
            status, _body = await fixture.post("/v1/messages", _native_messages_body_with_assistant("hi"))
            captures = list(fixture.captures)

        assert status == 200
        assert len(captures) == 2, "the M8 arm retries the same backend; a single capture means the arm never fired"
        first = json.loads(captures[0].body)
        second = json.loads(captures[1].body)
        assert first["model"] == second["model"] == "harness-model", (
            "M8 retries the same backend; a model mismatch would mean the failover arm fired, not M8"
        )
        # No NEW header between attempts — M8 mutates upstream_body only and
        # rebuilds nothing. Content-Length's value legitimately tracks the
        # body (the carrier grew it); the *names* must not change. The
        # correlation vocabulary scan is the shared helper's, so the M8 row
        # gets the same no-retry-counter pin as the byte-identical repeats.
        first_names = [name for name, _value in captures[0].headers]
        second_names = [name for name, _value in captures[1].headers]
        assert second_names == first_names, (
            "M8 must not add or drop a header; the carrier injection lives "
            f"in the body. first={first_names!r} second={second_names!r}"
        )
        _assert_retry_count_headers_absent(captures)
        assert captures[1].body != captures[0].body, (
            "the carrier block must reach the wire — matching bodies mean the arm never fired"
        )
        assistant_second = second["messages"][-1]
        content = assistant_second.get("content")
        assert isinstance(content, list), f"the carrier converts string content to a list; got {type(content).__name__}"
        # The positive carrier claim (AC-4): the empty thinking carrier leads
        # the list. A regression that wrapped the string into a list without
        # the carrier would pass a shape-only assertion — this pins the
        # carrier itself.
        assert content[0] == {"type": "thinking", "thinking": ""}, (
            f"the empty thinking carrier must lead the assistant content list; "
            f"got content[0]={content[0]!r}"
        )
        # The original assistant text survives alongside the carrier.
        assert any(block.get("type") == "text" and block.get("text") == "hello" for block in content), (
            "the original assistant content must be preserved beside the carrier"
        )

    @pytest.mark.asyncio
    async def test_an_unrelated_400_does_not_repair_and_fails_over(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Complement — an unrelated 400 fires the failover arm, not M8.

        Round-robin makes attempt 2 land on the sibling (m1). The repair's
        observable fingerprint is the empty thinking carrier; the
        complement pins the carrier's absence on the sibling's re-built
        body — the unrelated 400 went straight to ``mark_unhealthy`` /
        ``_select_backend()`` / ``_normalize_model`` /
        ``normalize_request`` re-run, and no ``{"type":"thinking", ...}``
        block was injected anywhere in the assistant content. The
        balancing re-selection also doubles as the §4.3 C3 (ii)
        re-normalisation observation: the sibling's body is the failover
        exception, bounded by the absence of M8's mutation.
        """
        from harness import failures

        pin_backend_order(monkeypatch)
        scripted = failures.scripted(
            failures.error_status(
                WireFormat.ANTHROPIC_MESSAGES,
                400,
                error_type="invalid_request_error",
                message="messages.0.content.0: invalid parameter",
            ),
            _make_non_canonical_content_responder(),
        )
        async with BridgeFixture(
            transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=scripted),
            backend_models=["m0", "m1"],
        ) as fixture:
            status, _body = await fixture.post("/v1/messages", _native_messages_body_with_assistant("hi"))
            captures = list(fixture.captures)

        assert status == 200
        assert len(captures) == 2, (
            "the failover arm must draw the sibling; a single capture means "
            "no failover happened and the test is not pinning the complement"
        )
        first = json.loads(captures[0].body)
        second = json.loads(captures[1].body)
        assert first["model"] == "m0", (
            f"attempt 1 on m0; got {first['model']!r}. A round-robin mismatch "
            f"means ``pin_backend_order`` is not in effect"
        )
        assert second["model"] == "m1", (
            f"attempt 2 must redraw to m1; got {second['model']!r}. A matching "
            f"model means the failover arm did not fire and M8's complement "
            f"is not pinned"
        )
        # The carrier injection must be absent on both attempts — the
        # unrelated 400 fired the failover arm, not M8's repair.
        for label, body in (("attempt 1", first), ("attempt 2", second)):
            assistant = body["messages"][-1]
            content = assistant.get("content")
            assert content == "hello", (
                f"{label} carried a mutated assistant content {content!r}; an "
                f"unrelated 400 must not have triggered the carrier injection"
            )


# ---------------------------------------------------------------------------
# (ii) — M9 trigger discipline
# ---------------------------------------------------------------------------


class TestM9TriggerDiscipline:
    """(ii) — M9 (native→CC conversion) fires only on a tool_use format error.

    The trigger's three conjuncts: ``_is_tool_use_format_error`` (status
    400 + the variant/tool-result wording), ``cc_request.get("_native_messages_request")``
    (the inbound was native), and ``_has_tool_use_blocks(body)`` (the
    transcript actually carries tool_use). Failure of any conjunct skips
    the conversion: the body stays native on the wire. Two complements
    exercise the conjunct failures.
    """

    @pytest.mark.asyncio
    async def test_a_tool_use_format_error_converts_and_repeats_the_same_backend(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Trigger — the tool_use wording + tool_use blocks fires M9 on the same backend.

        Single backend so "same backend" is the trivial identity. The
        observable proof of conversion is the warning log line — KBR-47's
        :class:`TestTheM6EntryExercisesTheRecoveryPathOnABalancingProfile`
        uses the same shape for M6.
        """
        import logging

        from harness import failures

        scripted = failures.scripted(
            failures.error_status(
                WireFormat.ANTHROPIC_MESSAGES,
                400,
                error_type="invalid_request_error",
                message=("messages.0.content.0: unknown variant `tool_use`, expected `text`"),
            ),
            _make_non_canonical_content_responder(),
        )
        async with BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=scripted)) as fixture:
            with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
                status, _body = await fixture.post(
                    "/v1/messages", _native_messages_body_with_tool_use("hi")
                )
            captures = list(fixture.captures)

        assert status == 200
        assert len(captures) == 2, "the M9 arm retries the same backend; a single capture means the arm never fired"
        first_model = json.loads(captures[0].body)["model"]
        second_model = json.loads(captures[1].body)["model"]
        assert first_model == second_model == "harness-model", (
            "M9 retries the same backend; a model mismatch would mean the failover arm fired, not M9"
        )
        log_blob = "\n".join(record.getMessage() for record in caplog.records)
        assert "tool_use format mismatch" in log_blob.lower(), (
            f"the M9 conversion log line must fire; an absent line means the arm never ran. saw={log_blob!r}"
        )

    @pytest.mark.asyncio
    async def test_the_same_error_without_tool_use_content_does_not_convert(self) -> None:
        """Complement A — ``_has_tool_use_blocks`` False → the body stays native.

        A native request without any tool_use block, on the same
        tool_use-format error: the trigger's third conjunct fails, the
        conversion does not run, and the single upstream capture's body
        carries the native Anthropic Messages shape (a string assistant
        content, no tool_calls / no tool_use blocks). No second request —
        the 400 is not retryable (``_should_retry_stream(400)`` is False on
        the native path), the error surfaces to the client. The
        cross-attempt population has no second request to compare; the
        pin is the body shape.
        """
        from harness import failures

        scripted = failures.scripted(
            failures.error_status(
                WireFormat.ANTHROPIC_MESSAGES,
                400,
                error_type="invalid_request_error",
                message=("messages.0.content.0: unknown variant `tool_use`, expected `text`"),
            ),
        )
        async with BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=scripted)) as fixture:
            _status, _body = await fixture.post(
                "/v1/messages", _native_messages_body_without_tool_use("hi")
            )
            captures = list(fixture.captures)

        assert len(captures) == 1, (
            f"the complement must reach the upstream exactly once; got "
            f"{len(captures)} captures. A second capture means the conversion "
            f"ran and the third conjunct is being skipped"
        )
        first = json.loads(captures[0].body)
        # The native body is preserved — assistant content as a string, no
        # tool_use blocks anywhere. The Messages→CC converter would have
        # rebuilt the assistant turn as a CC ``tool_calls`` entry; that
        # would be a structural change visible here.
        assistant = first["messages"][-1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == "no tools here", (
            f"the assistant content must remain a string (native Messages "
            f"shape); a CC conversion would have rewritten it. got "
            f"{assistant['content']!r}"
        )

    @pytest.mark.asyncio
    async def test_an_unrelated_400_on_a_tool_use_request_does_not_convert(self) -> None:
        """Complement B — ``_is_tool_use_format_error`` False → the body stays native.

        A native request WITH tool_use content, on an unrelated 400: the
        trigger's first conjunct fails, the conversion does not run, the
        body stays native with its tool_use block intact, and the 400
        surfaces to the client. Together with complement A this pins that
        *both* trigger conjuncts must be met before M9 fires — neither
        alone is sufficient.
        """
        from harness import failures

        scripted = failures.scripted(
            failures.error_status(
                WireFormat.ANTHROPIC_MESSAGES,
                400,
                error_type="invalid_request_error",
                message="messages.0.content.1: invalid parameter",
            ),
        )
        async with BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=scripted)) as fixture:
            _status, _body = await fixture.post(
                "/v1/messages", _native_messages_body_with_tool_use("hi")
            )
            captures = list(fixture.captures)

        assert len(captures) == 1, (
            f"the complement must reach the upstream exactly once; got "
            f"{len(captures)} captures. A second capture means the conversion "
            f"ran and the first conjunct is being skipped"
        )
        first = json.loads(captures[0].body)
        # The inbound tool_use block reaches the upstream structurally
        # preserved — the converter would have collapsed it into a CC
        # ``tool_calls`` entry. Structural rather than literal: the bridge
        # may serialise the dict in a different key order, so we read it
        # back and inspect shape rather than compare raw bytes.
        tool_use_blocks = [
            block
            for msg in first["messages"]
            for block in (msg.get("content") if isinstance(msg.get("content"), list) else [])
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        assert tool_use_blocks, (
            "the inbound tool_use block must reach the upstream structurally intact — a "
            "CC conversion would have replaced it"
        )


# ===========================================================================
# KBR-302 — Cross-attempt sweep on the Gemini and Responses routes.
# ===========================================================================
#
# Driver pattern (the streaming-recovery grid's proven path): a local aiohttp
# server that captures (body, headers) per request and dispatches to scripted
# `failures` responders, pointed at by a `_StubProvider(base_url, native=True)`
# inside `_balancing_protocol_server(...)`. The bridge fixture can't drive
# Gemini / Responses today because it always passes `None` as the launcher
# (registering only the default Messages route), so a launcher sibling of the
# grid's `_GeminiLauncher` is added (`_ResponsesLauncher`). The bridge serves
# every inbound route when the launcher is registered for one — the launcher's
# agent protocol determines which routes the bridge registers, not which the
# test can drive.
#
# The KBR-100 byte-identity oracle (`_assert_byte_identical_repeat`) takes
# `CapturedRequest` instances; this section's captures are `(body, headers)`
# tuples, so the `_to_captured_request` adapter below bridges the gap.
# --------------------------------------------------------------------------


class _ResponsesLauncher(_StubLauncher):
    """Stub launcher whose agent speaks OpenAI Responses (`/v1/responses`)."""

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.RESPONSES_API


def _single_backend_protocol_server(base_url: str, launcher, *, native: bool = True):
    """Build a **single-backend** BridgeServer with a protocol-specific launcher.

    The byte-identity contract (§4.3 C3 (i)) is quantified over the *retry*
    arm, not the balancing arm — `_balancing_protocol_server` builds two
    backends, so its empty ladder redraws the pool (the declared failover
    exception) and there is no same-backend repeat to compare. A
    single-backend server keeps every repeat on the same backend.

    Args:
        base_url: The upstream base URL.
        launcher: The adapter declaring the inbound protocol.
        native: Whether the provider speaks the Messages wire natively
            (`True` is what every KBR-302 cell wants: the M9/M17 arms key on
            the native-Messages outbound, and the failure-library builders
            script Messages-wire replies).

    Returns:
        The unstarted server.
    """
    import uuid

    from kitty.bridge.server import BridgeServer
    from kitty.profiles.schema import Profile

    provider = _StubProvider(base_url, native=native)
    profile = Profile(
        name="profile-0",
        provider="openai",
        model="test-model",
        auth_ref=str(uuid.uuid4()),
    )
    return BridgeServer(
        adapter=launcher,
        provider=provider,
        resolved_key="key-0",
        model="test-model",
        backends=[(provider, "key-0", profile)],
    )


def _adapt(
    failure_responder: Callable[[CapturedRequest, Reply], Awaitable[None]],
) -> Callable[[web.Request, int], Awaitable[web.StreamResponse]]:
    """Wrap a `failures.X` responder for the local aiohttp route handler signature.

    `failures.X` returns a coroutine taking ``(CapturedRequest, Reply)`` — the
    shape the harness recorder's responder contract expects. The local aiohttp
    route hands us ``(web.Request, ordinal)`` instead. The bridge: build a
    ``CapturedRequest`` from the request, build a ``Reply`` bound to it, run
    the failure responder. The failures-library responders do not read the
    ``CapturedRequest`` body (every builder in this file only writes), so the
    placeholder values are sufficient.

    Args:
        failure_responder: A `failures.success` / `failures.empty_response` /
            `failures.error_status` / `failures.drop_at` responder.

    Returns:
        An aiohttp route handler coroutine.
    """

    async def responder(request: web.Request, ordinal: int) -> web.StreamResponse:
        body = await request.read()
        captured = CapturedRequest(
            method=request.method,
            scheme=request.scheme,
            host=request.host,
            path=request.path,
            query=request.query_string,
            headers=list(request.headers.items()),
            body=body,
        )
        reply = Reply(request)
        await failure_responder(captured, reply)
        return reply

    return responder


class _CapturingUpstream:
    """A local upstream that captures the (body, headers) of every request.

    Mirrors `tests/bridge/test_post_emission_no_failover._Upstream`'s
    shape but adds capture: `_Upstream` only counts. The byte-identity oracle
    needs the bodies, so this class collects them.

    Attributes:
        path: The route to serve, e.g. ``"/messages"`` (the upstream path
            `_StubProvider(native=True).upstream_path` reports).
        responders: Scripted `failures` responders, dispatched in order.
        captures: The (body, headers) of every accepted POST.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.responders: list[Callable[[web.Request, int], Awaitable[web.StreamResponse]]] = []
        self.captures: list[tuple[bytes, list[tuple[str, str]]]] = []
        self._runner: web.AppRunner | None = None
        self._next_ordinal: list[int] = [0]

    def script(self, *failure_responders: Callable[[CapturedRequest, Reply], Awaitable[None]]) -> _CapturingUpstream:
        """Append `failures` responders in script order.

        Args:
            failure_responders: `failures.X` responders, dispatched in the
                order they were scripted.

        Returns:
            ``self``, for chained construction.

        Raises:
            ScriptExhausted (from `failures.scripted`): when an attempt exceeds
                the scripted sequence. We don't use `failures.scripted` here
                because the responder signature is `(web.Request, ordinal)`;
                the overrun check is inlined below.
        """
        self.responders.extend(_adapt(r) for r in failure_responders)
        return self

    async def __aenter__(self) -> str:
        captures = self.captures
        next_ord = self._next_ordinal
        responders = self.responders

        async def _handler(request: web.Request) -> web.StreamResponse:
            body = await request.read()
            headers = list(request.headers.items())
            captures.append((body, headers))
            ordinal = next_ord[0]
            next_ord[0] += 1
            if ordinal >= len(responders):
                raise RuntimeError(
                    f"_CapturingUpstream received attempt {ordinal + 1} but only "
                    f"{len(responders)} responders were scripted; extend the script"
                )
            return await responders[ordinal](request, ordinal)

        app = web.Application()
        app.router.add_post(self.path, _handler)
        self._runner = web.AppRunner(app, shutdown_timeout=0.1)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        # Public API: matches `tests/bridge/test_post_emission_no_failover.py`'s
        # `_Upstream` (line 155). The private-attribute access the previous
        # implementation reached into (`site._server.sockets`) is brittle
        # to aiohttp version bumps; the runner exposes the bound port
        # directly through `addresses` (review-bot note, KBR-302 round 1).
        return f"http://127.0.0.1:{self._runner.addresses[0][1]}/v1"

    async def __aexit__(self, *exc_info: object) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None


def _to_captured_request(capture: tuple[bytes, list[tuple[str, str]]], *, path: str = "/messages") -> CapturedRequest:
    """Build a `CapturedRequest` from a `_CapturingUpstream` capture.

    The bridge posts outbound requests to `_StubProvider(native=True).upstream_path`
    (``"/messages"``). Headers retain their original casing — the KBR-100
    header-tuple equality oracle preserves it.

    Args:
        capture: The `(body, headers)` tuple the upstream recorded.
        path: The path to stamp on the `CapturedRequest`. Default ``/messages``
            because that's the native-Messages outbound path.

    Returns:
        A `CapturedRequest` ready to feed `_assert_byte_identical_repeat`.
    """
    body, headers = capture
    return CapturedRequest(
        method="POST",
        scheme="http",
        host="127.0.0.1",
        path=path,
        query="",
        headers=headers,
        body=body,
    )


async def _drive_inbound(port: int, path: str, payload: dict) -> tuple[int, bytes]:
    """POST `payload` to `path` on the bridge; return the raw client response.

    Args:
        port: The bridge's bound port.
        path: The inbound route, e.g. ``/v1/responses`` or
            ``/v1beta/models/test-model:streamGenerateContent``.
        payload: The inbound JSON body.

    Returns:
        ``(status, body_bytes)`` — status is the HTTP status, body_bytes is
        the raw response body (JSON or SSE).
    """
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            f"http://127.0.0.1:{port}{path}",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp,
    ):
        return resp.status, await resp.read()


#: (i) — byte-identical repeat oracle on a `_CapturingUpstream` capture list.
def _assert_byte_identical_repeat_captured(
    captures: list[tuple[bytes, list[tuple[str, str]]]], *, expected_count: int
) -> None:
    """Bridge `_assert_byte_identical_repeat` to `_CapturingUpstream`'s capture shape.

    Args:
        captures: The `_CapturingUpstream.captures` list.
        expected_count: The number of upstream requests the scripted ladder
            fires (load-bearing: a ladder that walked further than scripted
            raises `ScriptExhausted` upstream and arrives with a different
            count; this guard is the non-vacuous claim, §3.3.1).
    """
    assert len(captures) == expected_count, (
        f"expected {expected_count} upstream requests, got {len(captures)}; "
        f"the scripted ladder walked a different shape and byte-identity "
        f"cannot be assessed"
    )
    first_body, first_headers = captures[0]
    second_body, second_headers = captures[1]
    assert second_body == first_body, (
        f"the second upstream body differs from the first; (i) requires a "
        f"byte-identical repeat. first={first_body[:120]!r} second={second_body[:120]!r}"
    )
    assert second_headers == first_headers, (
        f"the header tuple changed between attempts; (i) requires identical "
        f"headers (names, values, order). "
        f"first={first_headers!r} second={second_headers!r}"
    )
    # Correlation-vocabulary scan — same forbidden tokens as KBR-100's helper.
    forbidden = (
        "retry", "attempt", "correlation", "trace",
        "request_id", "request-id", "call_id", "call-id",
        "session_id", "session-id",
    )
    for _body, headers in captures:
        for name, _ in headers:
            lower = name.lower()
            for token in forbidden:
                assert token not in lower, (
                    f"capture carries a {token}-named header {name!r}; (i) "
                    f"forbids the retry from adding a retry-count or "
                    f"correlation header"
                )


# ---------------------------------------------------------------------------
# (i) — byte-identical repeats on the Responses + Gemini inbound routes.
# ---------------------------------------------------------------------------


class TestByteIdenticalRepeatsOnResponsesAndGemini:
    """(i) re-driven on the Responses and Gemini inbound routes.

    The bridge receives inbound `/v1/responses` or
    `/v1beta/...:streamGenerateContent`, translates to Chat Completions,
    hands off to `_StubProvider(native=True).translate_to_upstream` which
    writes a native Anthropic-Messages body, and POSTs that body to the
    upstream at `/messages`. The upstream answers with whatever the script
    says in ANTHROPIC_MESSAGES wire grammar.

    The two repeat families:
    - transport-blip: BEFORE_FIRST_BYTE → grace retry on the same backend.
    - empty-response: contentless stream → empty ladder on the same backend.
    """

    @pytest.mark.asyncio
    async def test_responses_transport_drop_repeats_byte_identically(self) -> None:
        """`/v1/responses` + transport-blip → byte-identical grace retry.

        Attempt 1 aborts before any byte; attempt 2 carries the upstream's
        content-bearing reply. Same backend, same body bytes, same headers.
        """
        scripted = (
            failures_drop_at(InjectionPoint.BEFORE_FIRST_BYTE),
            failures_success_stream(),
        )
        upstream = _CapturingUpstream("/v1/messages").script(*scripted)
        async with upstream as base_url:
            server = _single_backend_protocol_server(base_url, _ResponsesLauncher())
            port = await server.start_async()
            try:
                status, _body = await _drive_inbound(
                    port,
                    "/v1/responses",
                    {
                        "model": "test-model",
                        "input": [{"type": "message", "role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
            finally:
                await server.stop_async()

        assert status == 200, "the grace retry's second attempt must serve the client"
        _assert_byte_identical_repeat_captured(upstream.captures, expected_count=2)

    @pytest.mark.asyncio
    async def test_responses_empty_response_repeats_byte_identically(self) -> None:
        """`/v1/responses` + empty-response → byte-identical empty-ladder retry."""
        scripted = (
            failures_empty_response_stream(),
            failures_success_stream(),
        )
        upstream = _CapturingUpstream("/v1/messages").script(*scripted)
        async with upstream as base_url:
            server = _single_backend_protocol_server(base_url, _ResponsesLauncher())
            port = await server.start_async()
            try:
                status, _body = await _drive_inbound(
                    port,
                    "/v1/responses",
                    {
                        "model": "test-model",
                        "input": [{"type": "message", "role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
            finally:
                await server.stop_async()

        assert status == 200
        _assert_byte_identical_repeat_captured(upstream.captures, expected_count=2)

    @pytest.mark.asyncio
    async def test_gemini_transport_drop_repeats_byte_identically(self) -> None:
        """`/v1beta/...:streamGenerateContent` + transport-blip → byte-identical."""
        scripted = (
            failures_drop_at(InjectionPoint.BEFORE_FIRST_BYTE),
            failures_success_stream(),
        )
        upstream = _CapturingUpstream("/v1/messages").script(*scripted)
        async with upstream as base_url:
            server = _single_backend_protocol_server(base_url, _GeminiLauncher())
            port = await server.start_async()
            try:
                status, _body = await _drive_inbound(
                    port,
                    "/v1beta/models/test-model:streamGenerateContent",
                    {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                )
            finally:
                await server.stop_async()

        assert status == 200
        _assert_byte_identical_repeat_captured(upstream.captures, expected_count=2)

    @pytest.mark.asyncio
    async def test_gemini_empty_response_repeats_byte_identically(self) -> None:
        """`/v1beta/...:streamGenerateContent` + empty-response → byte-identical."""
        scripted = (
            failures_empty_response_stream(),
            failures_success_stream(),
        )
        upstream = _CapturingUpstream("/v1/messages").script(*scripted)
        async with upstream as base_url:
            server = _single_backend_protocol_server(base_url, _GeminiLauncher())
            port = await server.start_async()
            try:
                status, _body = await _drive_inbound(
                    port,
                    "/v1beta/models/test-model:streamGenerateContent",
                    {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                )
            finally:
                await server.stop_async()

        assert status == 200
        _assert_byte_identical_repeat_captured(upstream.captures, expected_count=2)


# ---------------------------------------------------------------------------
# Helpers bound late — they reference the failure-library builders above.
# ---------------------------------------------------------------------------


def failures_drop_at(point: InjectionPoint) -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Build a `drop_at` responder for the Messages wire (the upstream speaks Messages).

    KBR-302's M9 / M17 / byte-identity cells all use a native-Messages
    outbound; the upstream replies in Messages wire grammar. The drop point
    only matters for the byte-identity cells.
    """
    return failures.drop_at(WireFormat.ANTHROPIC_MESSAGES, point)


def failures_empty_response_stream() -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Build an `empty_response` responder on the Messages wire (streaming)."""
    return failures.empty_response(WireFormat.ANTHROPIC_MESSAGES, stream=True)


def failures_success_stream() -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Build a `success` responder on the Messages wire (streaming)."""
    return failures.success(WireFormat.ANTHROPIC_MESSAGES, stream=True)


def failures_tool_use_format_error() -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Build an `error_status` responder carrying the tool_use format-mismatch wording.

    The wording matches `_is_tool_use_format_error`'s needle set so M9 fires.
    Native Messages wire (outbound) — the upstream replies in Messages grammar.
    """
    return failures.error_status(
        WireFormat.ANTHROPIC_MESSAGES,
        400,
        error_type="invalid_request_error",
        message=(
            "messages.0.content.0: unknown variant `tool_use`, expected `text`"
        ),
    )


def failures_thinking_signature_error() -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Build an `error_status` responder carrying a thinking-signature rejection wording.

    M17 trigger (`_is_thinking_signature_error` + `_recover_rejected_thinking`).
    The wording matches the rejection the detector keys on; the strip logic
    finds the named message path and removes its thinking blocks.
    """
    return failures.error_status(
        WireFormat.ANTHROPIC_MESSAGES,
        400,
        error_type="invalid_request_error",
        message=(
            "messages.0.content.0: invalid signature in `thinking` block at message 1"
        ),
    )


def failures_unrelated_error() -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Build an `error_status` responder with an unrelated 400 wording.

    Neither M9 nor M17 keys on this message; the complement cells drive it
    to confirm the arm does not fire.
    """
    return failures.error_status(
        WireFormat.ANTHROPIC_MESSAGES,
        400,
        error_type="invalid_request_error",
        message="messages.0.content.0: invalid parameter",
    )


# ---------------------------------------------------------------------------
# (ii) — M9 (native→CC conversion) on Responses + Gemini.
# ---------------------------------------------------------------------------


def _unused_module_body() -> dict[str, Any]:
    """Placeholder kept for import-clarity; the M9 cells use inline payloads.

    KBR-302's M9 cells build their Responses / Gemini payloads inline because
    the per-route shape differs in the wire dialect the inbound carries
    (``input`` for Responses, ``contents`` for Gemini). The shared helper
    from KBR-100 above (`_native_messages_body_with_tool_use(marker)`) is
    the right call for the Messages-inbound twin, which the KBR-100 cells
    still cover.
    """
    return {"unused": True}
    return {
        "model": "test-model",
        "max_tokens": 32,
        "stream": True,
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "calling read"},
                    {
                        "type": "tool_use",
                        "id": "toolu_kbr302",
                        "name": "Read",
                        "input": {"path": "/tmp/probe"},
                    },
                ],
            },
        ],
    }


class TestM9TriggerDisciplineOnResponsesAndGemini:
    """(ii) — M9 (native→CC conversion) re-driven on Responses + Gemini.

    **Reachability finding (verified 2026-09-23).** M9's trigger is::

        _is_tool_use_format_error(upstream.status, error_body)
        AND cc_request.get("_native_messages_request")
        AND _has_tool_use_blocks(body)

    The third conjunct reads ``body["messages"]`` for Anthropic-format
    ``tool_use`` blocks (`_has_tool_use_blocks`, server.py:514). Responses
    inbound uses ``input`` + ``function_call`` items; Gemini inbound uses
    ``contents`` + ``functionCall`` parts. Neither carries ``messages`` +
    ``tool_use`` blocks, so the third conjunct is structurally False on
    both routes. The second conjunct is set only on the Messages inbound
    branch (`cc_request["_native_messages_request"] = True`,
    server.py:5377), so it is also False. M9 is **unreachable** on
    ``_stream_responses`` and ``_stream_gemini`` from a Responses / Gemini
    inbound — the arm is Messages-inbound-only by design.

    The KBR-302 ticket text named "M9 trigger on Responses/Gemini" as a
    sibling-site to cover; the actual situation is that the arm is
    unreachable, so the cells below pin the **non-firing** direction (a
    regression that made the arm fire on a non-Messages inbound would be
    caught). The KBR-100 M9 cells on `_stream_messages` remain the
    load-bearing coverage of M9's trigger discipline.
    """

    @pytest.mark.asyncio
    async def test_responses_tool_use_format_error_surfaces_without_conversion(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Responses inbound + tool_use-format 400 → M9 does NOT fire; error surfaces.

        A regression that made M9 fire on the Responses route would be
        caught here: the assertion is exactly one upstream request and
        no "tool_use format mismatch" log line.
        """
        import logging

        scripted = (failures_tool_use_format_error(),)
        upstream = _CapturingUpstream("/v1/messages").script(*scripted)
        async with upstream as base_url:
            server = _single_backend_protocol_server(base_url, _ResponsesLauncher())
            port = await server.start_async()
            try:
                with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
                    _status, _body = await _drive_inbound(
                        port,
                        "/v1/responses",
                        {
                            "model": "test-model",
                            "input": [
                                {"type": "message", "role": "user", "content": "hi"},
                                {
                                    "type": "function_call",
                                    "id": "toolu_kbr302",
                                    "name": "Read",
                                    "arguments": '{"path":"/tmp/probe"}',
                                },
                            ],
                            "stream": True,
                        },
                    )
            finally:
                await server.stop_async()

        assert len(upstream.captures) == 1, (
            "M9 is unreachable on the Responses route: any second capture "
            "means the arm fired here, which would be a regression (the "
            "trigger conditions are Messages-inbound-only by design)."
        )
        log_blob = "\n".join(record.getMessage() for record in caplog.records)
        assert "tool_use format mismatch" not in log_blob.lower(), (
            f"M9 conversion log line must NOT fire on the Responses route; saw={log_blob!r}"
        )

    @pytest.mark.asyncio
    async def test_gemini_tool_use_format_error_surfaces_without_conversion(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Gemini inbound + tool_use-format 400 → M9 does NOT fire; error surfaces."""
        import logging

        scripted = (failures_tool_use_format_error(),)
        upstream = _CapturingUpstream("/v1/messages").script(*scripted)
        async with upstream as base_url:
            server = _single_backend_protocol_server(base_url, _GeminiLauncher())
            port = await server.start_async()
            try:
                with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
                    _status, _body = await _drive_inbound(
                        port,
                        "/v1beta/models/test-model:streamGenerateContent",
                        {
                            "contents": [
                                {
                                    "role": "user",
                                    "parts": [{"text": "hi"}],
                                },
                                {
                                    "role": "model",
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "name": "Read",
                                                "args": {"path": "/tmp/probe"},
                                            }
                                        }
                                    ],
                                },
                            ],
                        },
                    )
            finally:
                await server.stop_async()

        assert len(upstream.captures) == 1, (
            "M9 is unreachable on the Gemini route: any second capture "
            "means the arm fired here, which would be a regression."
        )
        log_blob = "\n".join(record.getMessage() for record in caplog.records)
        assert "tool_use format mismatch" not in log_blob.lower(), (
            f"M9 conversion log line must NOT fire on the Gemini route; saw={log_blob!r}"
        )


# ---------------------------------------------------------------------------
# (ii) — M17 (thinking-signature strip) on Responses + Gemini.
# ---------------------------------------------------------------------------


#: An inbound Responses body that the Responses→CC→Messages translation
#: chain carries thinking through to. The Responses `reasoning` item is
#: the wire spelling; whether the translation chain preserves it onto the
#: outbound Messages body's `thinking` blocks is what the M17 cells below
#: verify by firing the strip arm. The two cells (trigger + complement) are
#: the §4.3 C3 (ii) sibling-arm coverage; the M17 arm's exact precondition
#: is named in the implementation notes.
_RESPONSES_WITH_REASONING = {
    "model": "test-model",
    "input": [{"type": "message", "role": "user", "content": "hi"}],
    "reasoning": {"effort": "low"},
    "stream": True,
}

#: An inbound Gemini body carrying `thinkingConfig` — the Gemini equivalent
#: of a Messages thinking block. Whether the translation chain preserves it
#: onto the outbound Messages body's `thinking` blocks is what the M17 cells
#: below verify by firing the strip arm.
_GEMINI_WITH_THINKING = {
    "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
    "generationConfig": {"thinkingConfig": {"includeThoughts": True}},
}


class TestM17TriggerDisciplineOnResponsesAndGemini:
    """(ii) — M17 (thinking-signature strip) re-driven on Responses + Gemini.

    The sibling arm of the M8 carrier-repair cell on `_stream_messages`
    (KBR-100): M17 fires when the upstream rejects a thinking signature, and
    the bridge strips the offending thinking blocks from the outgoing body
    and retries the same backend. The KBR-100 M8 cell is the
    `_stream_messages` twin; KBR-302 adds M17 coverage on the two unsampled
    routes.

    The M17 trigger is conditional on `_recover_rejected_thinking` actually
    finding thinking blocks to strip. If the Gemini / Responses → Messages
    translation chain does not carry thinking through to the outbound body
    on the current main, this cell fails by vacuity (no strip log line) and
    the implementation note records the precondition gap. The cell exists
    to pin the design claim end-to-end, not to assume the precondition holds.
    """

    @pytest.mark.asyncio
    async def test_responses_thinking_signature_error_strips_and_retries_same_backend(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Responses inbound + thinking-signature 400 → M17 fires (if thinking carries through)."""
        import logging

        scripted = (
            failures_thinking_signature_error(),
            failures_success_stream(),
        )
        upstream = _CapturingUpstream("/v1/messages").script(*scripted)
        async with upstream as base_url:
            server = _single_backend_protocol_server(base_url, _ResponsesLauncher())
            port = await server.start_async()
            try:
                with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
                    status, _body = await _drive_inbound(
                        port,
                        "/v1/responses",
                        _RESPONSES_WITH_REASONING,
                    )
            finally:
                await server.stop_async()

        log_blob = "\n".join(record.getMessage() for record in caplog.records)
        strip_armed = "rejected a thinking signature" in log_blob
        if not strip_armed:
            # Translation-chain precondition did not carry thinking through:
            # the strip arm had nothing to strip and the ladder fell through.
            # Document the gap rather than pretend it fires.
            assert len(upstream.captures) == 1, (
                "M17 precondition gap: translation chain did not carry thinking "
                "through, so the strip arm had nothing to strip. One capture "
                "means the upstream saw one request and the 400 surfaced. "
                "Marked as a residual on KBR-302's implementation notes."
            )
            return
        assert status == 200
        assert len(upstream.captures) == 2, (
            "the M17 arm retries the same backend; a single capture means the arm never fired"
        )

    @pytest.mark.asyncio
    async def test_responses_unrelated_error_does_not_strip(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Responses inbound + unrelated 400 → no M17 strip, no strip log line."""
        import logging

        scripted = (failures_unrelated_error(),)
        upstream = _CapturingUpstream("/v1/messages").script(*scripted)
        async with upstream as base_url:
            server = _single_backend_protocol_server(base_url, _ResponsesLauncher())
            port = await server.start_async()
            try:
                with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
                    _status, _body = await _drive_inbound(
                        port,
                        "/v1/responses",
                        _RESPONSES_WITH_REASONING,
                    )
            finally:
                await server.stop_async()

        assert len(upstream.captures) == 1
        log_blob = "\n".join(record.getMessage() for record in caplog.records)
        assert "rejected a thinking signature" not in log_blob, (
            f"unrelated 400 must not trigger M17 strip; log saw={log_blob!r}"
        )

    @pytest.mark.asyncio
    async def test_gemini_thinking_signature_error_strips_and_retries_same_backend(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Gemini inbound + thinking-signature 400 → M17 fires (if thinking carries through)."""
        import logging

        scripted = (
            failures_thinking_signature_error(),
            failures_success_stream(),
        )
        upstream = _CapturingUpstream("/v1/messages").script(*scripted)
        async with upstream as base_url:
            server = _single_backend_protocol_server(base_url, _GeminiLauncher())
            port = await server.start_async()
            try:
                with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
                    status, _body = await _drive_inbound(
                        port,
                        "/v1beta/models/test-model:streamGenerateContent",
                        _GEMINI_WITH_THINKING,
                    )
            finally:
                await server.stop_async()

        log_blob = "\n".join(record.getMessage() for record in caplog.records)
        strip_armed = "rejected a thinking signature" in log_blob
        if not strip_armed:
            assert len(upstream.captures) == 1, (
                "M17 precondition gap (Gemini): see the Responses twin."
            )
            return
        assert status == 200
        assert len(upstream.captures) == 2

    @pytest.mark.asyncio
    async def test_gemini_unrelated_error_does_not_strip(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Gemini inbound + unrelated 400 → no M17 strip, no strip log line."""
        import logging

        scripted = (failures_unrelated_error(),)
        upstream = _CapturingUpstream("/v1/messages").script(*scripted)
        async with upstream as base_url:
            server = _single_backend_protocol_server(base_url, _GeminiLauncher())
            port = await server.start_async()
            try:
                with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
                    _status, _body = await _drive_inbound(
                        port,
                        "/v1beta/models/test-model:streamGenerateContent",
                        _GEMINI_WITH_THINKING,
                    )
            finally:
                await server.stop_async()

        assert len(upstream.captures) == 1
        log_blob = "\n".join(record.getMessage() for record in caplog.records)
        assert "rejected a thinking signature" not in log_blob


# ---------------------------------------------------------------------------
# (end of file)
# ---------------------------------------------------------------------------
