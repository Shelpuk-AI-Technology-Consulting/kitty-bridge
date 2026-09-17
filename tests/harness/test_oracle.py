"""Tests for the transparency oracle.

`.system_design/TEST_SUITE.md` §3.3, §3.3.2, §4.3 C2, §7.4, §10 · plan task
**T-D1** (KBR-51).

Tests run at ``l1`` (path-default per ``tests/layers.py`` and the
vertical-slice precedent). ``l3`` activation is T-K6's job.

**Two test populations, deliberately separated** (the T-W9 attribution rule
— a failing test must name the layer that failed):

* **Unit tests** drive the oracle's internals —
  :func:`oracle._structural_diff`, :func:`oracle._run_assertions`,
  :func:`oracle._conditional_violations`, :func:`oracle._native_passthrough_check`
  — against synthesised :class:`~harness.contract.Request` objects. No body,
  no reader, no bridge; a failure here is the oracle's own.
* **Integration tests** drive the §7.4-shaped public
  :func:`oracle.assert_no_unclaimed_mutation` with real captures through the
  landed readers. Driven end-to-end tests (bridge fixture, corpus loader)
  live beside the slice, not here.
"""

from __future__ import annotations

import pytest

from harness import contract as c
from harness import oracle
from harness import register as r
from harness.contract import (
    CapturedRequest,
    Conversation,
    Envelope,
    Request,
    Text,
    ToolDecl,
    ToolResult,
    ToolUse,
    Turn,
    WireFormat,
)
from harness.oracle import (
    NativePassthroughKeyOrderError,
    OracleReport,
    UnclaimedMutationError,
    assert_no_unclaimed_mutation,
)

# --------------------------------------------------------------------------
# Test fixtures
# --------------------------------------------------------------------------


def _capture(
    body: bytes = b"", method: str = "POST", path: str = "/v1/messages"
) -> CapturedRequest:
    """Return a minimal :class:`CapturedRequest` for body-only tests.

    Args:
        body: The raw request body bytes.
        method: HTTP method.
        path: URL path (unused by the diff but present for realism).

    Returns:
        A :class:`CapturedRequest` with the body and minimal metadata.
    """
    return CapturedRequest(
        method=method, scheme="http", host="example.invalid", path=path, query="", body=body
    )


def _empty_request(envelope: Envelope | None = None) -> Request:
    """Return a minimal :class:`Request` projection with one empty user turn.

    Args:
        envelope: An optional :class:`Envelope`. The default is empty.

    Returns:
        A :class:`Request` carrying the supplied envelope and a single
        ``user`` turn with one :class:`Text` part of ``""``.
    """
    return Request(
        envelope=envelope or Envelope(),
        conversation=Conversation(turns=(Turn(role="user", parts=(Text(text=""),)),)),
    )


def _envelope_with_model(model: str | None) -> Envelope:
    """Return an :class:`Envelope` with the model set.

    Args:
        model: The model name; ``None`` for absent.

    Returns:
        An :class:`Envelope` whose ``model`` field is ``model``.
    """
    return Envelope(model=model, stream=False, store=False)


# --------------------------------------------------------------------------
# §7.4 signature and reader registry
# --------------------------------------------------------------------------


class TestSignature:
    """The oracle's signature matches §7.4 and the registry is complete."""

    def test_format_registry_completeness(self) -> None:
        """Every :class:`WireFormat` member has a registered reader."""
        # The oracle module raises at import time if a reader is missing,
        # so this test mostly exercises the assertion that import succeeded.
        assert set(oracle._REQUEST_PROJECTIONS.keys()) == set(WireFormat)

    def test_reader_for_returns_registered_reader(self) -> None:
        """``_reader_for`` returns the registered reader for each format."""
        for fmt in WireFormat:
            assert oracle._reader_for(fmt).wire_format == fmt

    def test_reader_for_unknown_format_raises_value_error(self) -> None:
        """A missing reader raises :class:`ValueError` at call time."""
        # Use a wire format not in the closed enum by mocking the dict.
        saved = oracle._REQUEST_PROJECTIONS.copy()
        try:
            oracle._REQUEST_PROJECTIONS.pop(WireFormat.ANTHROPIC_MESSAGES, None)
            with pytest.raises(ValueError):
                oracle._reader_for(WireFormat.ANTHROPIC_MESSAGES)
        finally:
            oracle._REQUEST_PROJECTIONS.update(saved)


# --------------------------------------------------------------------------
# Totality gate (§3.3.1, §7.4)
# --------------------------------------------------------------------------


class TestTotalityGate:
    """A non-empty residual fails the run before the structural diff."""

    def test_verify_total_rejects_non_empty_residual(self) -> None:
        """A projection with a non-empty ``residual`` is rejected by
        :func:`~harness.contract.verify_total`, the oracle's first gate."""
        projected = Request(
            envelope=Envelope(),
            conversation=Conversation(),
            residual={"unexpected": "value"},
        )
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_verify_total_rejects_dropped_field(self) -> None:
        """A projection whose ``consumed`` misses a source key raises."""
        projected = Request(
            envelope=Envelope(),
            conversation=Conversation(),
            source={"model": "x", "surprise": "y"},
            consumed=frozenset({"model"}),
        )
        with pytest.raises(c.DroppedFieldsError):
            c.verify_total(projected)


# --------------------------------------------------------------------------
# §3.3.2 assertion 1 — no unclaimed delta
# --------------------------------------------------------------------------


class TestStructuralDiff:
    """The diff emits concrete delta paths in the §3.3.1a vocabulary."""

    def test_no_diff_on_identical_projections_passes(self) -> None:
        """Two byte-identical projections produce zero deltas."""
        inbound = _empty_request(_envelope_with_model("agent-model"))
        captured = _empty_request(_envelope_with_model("agent-model"))

        deltas = oracle._structural_diff(inbound, captured)
        assert deltas == ()

    def test_changed_model_yields_one_concrete_delta(self) -> None:
        """One differing envelope field yields one concrete delta path."""
        inbound = _empty_request(_envelope_with_model("agent-model"))
        captured = _empty_request(_envelope_with_model("profile-model"))

        deltas = oracle._structural_diff(inbound, captured)
        assert deltas == (c.ENVELOPE_MODEL,)

    def test_changed_extra_yields_one_extra_path(self) -> None:
        """An ``envelope.extra`` key difference names the wire key."""
        inbound = _empty_request(Envelope(extra={"thinking": {"type": "enabled"}}))
        captured = _empty_request(Envelope())

        deltas = oracle._structural_diff(inbound, captured)
        assert deltas == (c.extra_path("thinking"),)

    def test_missing_turn_yields_turn_path(self) -> None:
        """A turn present on one side only yields its turn path."""
        inbound = _empty_request()
        captured = Request(envelope=Envelope(), conversation=Conversation())

        deltas = oracle._structural_diff(inbound, captured)
        assert deltas == (c.turn_path(0),)

    def test_missing_part_yields_part_path(self) -> None:
        """A part present on one side only yields its part path."""
        inbound = _empty_request()
        captured = Request(
            envelope=Envelope(),
            conversation=Conversation(turns=(Turn(role="user", parts=(Text(text=""), Text(text="x"))),)),
        )

        deltas = oracle._structural_diff(inbound, captured)
        assert deltas == (c.part_path(0, 1),)

    def test_changed_text_yields_text_path(self) -> None:
        """A differing Text part yields its ``text`` leaf path."""
        inbound = _empty_request(
            Envelope(model=None, stream=False, store=False)
        )
        captured = _empty_request(
            Envelope(model=None, stream=False, store=False)
        )
        # Rewrite both turns' parts to differ in the text field.
        inbound = Request(
            envelope=Envelope(),
            conversation=Conversation(turns=(Turn(role="user", parts=(Text(text="hello"),)),)),
        )
        captured = Request(
            envelope=Envelope(),
            conversation=Conversation(turns=(Turn(role="user", parts=(Text(text="goodbye"),)),)),
        )

        deltas = oracle._structural_diff(inbound, captured)
        assert deltas == (c.part_path(0, 0, "text"),)

    def test_changed_tool_field_yields_by_name_path(self) -> None:
        """A differing tool field addresses the tool by name (§3.3.1a)."""
        inbound = Request(
            envelope=Envelope(),
            conversation=Conversation(
                tools=(ToolDecl(name="get_weather", description="old"),),
            ),
        )
        captured = Request(
            envelope=Envelope(),
            conversation=Conversation(
                tools=(ToolDecl(name="get_weather", description="new"),),
            ),
        )

        deltas = oracle._structural_diff(inbound, captured)
        assert deltas == (c.tool_path("get_weather", "description"),)

    def test_missing_tool_yields_bare_tool_path(self) -> None:
        """A tool present on one side only yields its bare tool path."""
        inbound = Request(
            envelope=Envelope(),
            conversation=Conversation(
                tools=(ToolDecl(name="get_weather"),),
            ),
        )
        captured = Request(envelope=Envelope(), conversation=Conversation())

        deltas = oracle._structural_diff(inbound, captured)
        assert deltas == (c.tool_path("get_weather"),)

    def test_changed_tool_result_id_yields_tool_use_id_path(self) -> None:
        """A differing ToolResult ``tool_use_id`` names the leaf."""
        inbound = Request(
            envelope=Envelope(),
            conversation=Conversation(
                turns=(Turn(role="user", parts=(ToolResult(content=(), tool_use_id="a"),)),),
            ),
        )
        captured = Request(
            envelope=Envelope(),
            conversation=Conversation(
                turns=(Turn(role="user", parts=(ToolResult(content=(), tool_use_id="b"),)),),
            ),
        )

        deltas = oracle._structural_diff(inbound, captured)
        assert deltas == (c.part_path(0, 0, "tool_use_id"),)


class TestAssertion1:
    """Every concrete delta must be claimed by a triggered register row."""

    def test_run_assertions_passes_on_zero_deltas(self) -> None:
        """Identical projections produce zero deltas → no raise."""
        inbound = _empty_request(_envelope_with_model("agent-model"))
        captured = _empty_request(_envelope_with_model("agent-model"))

        deltas = oracle._run_assertions(
            inbound, captured, register=r.REGISTER, triggers_met=frozenset()
        )
        assert deltas == ()

    def test_run_assertions_reports_unclaimed_extra(self) -> None:
        """An unclaimed ``envelope.extra`` delta raises with the path named."""
        inbound = _empty_request()
        captured = _empty_request(Envelope(extra={"temperature": 0.5}))

        with pytest.raises(UnclaimedMutationError) as info:
            oracle._run_assertions(inbound, captured, register=r.REGISTER, triggers_met=frozenset())
        assert "envelope.extra[temperature]" in info.value.paths

    def test_changed_model_claimed_when_trigger_present(self) -> None:
        """M1 claims a mutated model when ``PROFILE_SETS_MODEL`` is met."""
        inbound = _empty_request(_envelope_with_model("agent-model"))
        captured = _empty_request(_envelope_with_model("profile-model"))

        deltas = oracle._run_assertions(
            inbound,
            captured,
            register=r.REGISTER,
            triggers_met=frozenset({r.Trigger.PROFILE_SETS_MODEL}),
        )
        assert deltas == (c.ENVELOPE_MODEL,)

    def test_changed_model_fails_oracle_when_trigger_omitted(self) -> None:
        """Falsification case (plan §1.4): the diff sees the model field.

        With ``PROFILE_SETS_MODEL`` deliberately omitted, M1 does not claim
        the mutated ``envelope.model``. The delta is unclaimed, the oracle
        fails, and the failure names ``envelope.model``. This proves the
        diff sees the model — §10's "projection that could not see the
        model name" trap.
        """
        inbound = _empty_request(_envelope_with_model("agent-model"))
        captured = _empty_request(_envelope_with_model("profile-model"))

        with pytest.raises(UnclaimedMutationError) as info:
            oracle._run_assertions(
                inbound, captured, register=r.REGISTER, triggers_met=frozenset()
            )
        assert "envelope.model" in info.value.paths

    def test_not_projectable_row_claims_nothing(self) -> None:
        """A row taking the NOT_PROJECTABLE escape claims no delta."""
        row = r.MutationRow(
            id="Z-NP",
            site=("tests/harness/test_oracle.py:Z-NP",),
            trigger=r.Trigger.ALWAYS,
            paths=(c.NOT_PROJECTABLE,),
            conditional=False,
            design_ref="test",
            not_projectable_reason="test escape",
        )
        inbound = _empty_request(_envelope_with_model("a"))
        captured = _empty_request(_envelope_with_model("b"))

        # The escape row is triggered but claims nothing — the model delta
        # stays unclaimed (M1's trigger is absent).
        with pytest.raises(UnclaimedMutationError) as info:
            oracle._run_assertions(
                inbound, captured, register=(row,), triggers_met=frozenset({r.Trigger.ALWAYS})
            )
        assert "envelope.model" in info.value.paths


# --------------------------------------------------------------------------
# Path matching — §3.3.1a prefix rule
# --------------------------------------------------------------------------


class TestPathMatching:
    """A register pattern is a prefix over its descendants."""

    def test_path_matches_prefix_under_register_anchor(self) -> None:
        """A delta beneath a registered pattern's anchor is claimed."""
        # M16 anchors on ``conversation.turns[*].parts[*].cache_control``;
        # a delta at the leaf is claimed when M16's trigger is met.
        inbound = Request(
            envelope=Envelope(),
            conversation=Conversation(
                turns=(
                    Turn(
                        role="user",
                        parts=(Text(text="hello", cache_control={"type": "ephemeral"}),),
                    ),
                ),
            ),
        )
        captured = Request(
            envelope=Envelope(),
            conversation=Conversation(turns=(Turn(role="user", parts=(Text(text="hello"),)),)),
        )

        deltas = oracle._run_assertions(
            inbound,
            captured,
            register=r.REGISTER,
            triggers_met=frozenset({r.Trigger.NON_NATIVE_UPSTREAM_WIRE}),
        )
        # M16's trigger is met → the cache_control delta is claimed.
        assert any("cache_control" in p for p in deltas)


# --------------------------------------------------------------------------
# §3.3.2 assertion 2 — no conditional row firing without trigger
# --------------------------------------------------------------------------


def _conditional_row() -> r.MutationRow:
    """Return a conditional register row anchored on the part ``id`` field.

    Returns:
        A :class:`MutationRow` in the shape of M18, but with a
        test-local id so a failure names the test's row.
    """
    return r.MutationRow(
        id="Z-TEST",
        site=("tests/harness/test_oracle.py:Z-TEST",),
        trigger=r.Trigger.GEMINI_INBOUND_ID_ABSENT,
        paths=(c.part_path(c.WILDCARD, c.WILDCARD, "id"),),
        conditional=True,
        design_ref="test",
    )


class TestAssertion2:
    """A conditional row whose trigger is not met must not produce a delta."""

    def test_conditional_row_fires_without_trigger_raises_unclaimed(self) -> None:
        """A conditional row whose anchor has an unclaimed delta surfaces as
        an assertion-1 violation (the delta is unclaimed by any triggered row).

        With no triggered rows in the register, a delta at the conditional
        row's anchor is unclaimed, and §3.3.2 assertion 1 fires first. The
        conditional row's presence is what makes the delta *named* in the
        register's vocabulary — the failure message still references the
        path the row would have claimed.
        """
        row = _conditional_row()

        inbound = Request(
            envelope=Envelope(),
            conversation=Conversation(
                turns=(Turn(role="assistant", parts=(ToolUse(name="f", arguments={}, id="a"),)),),
            ),
        )
        captured = Request(
            envelope=Envelope(),
            conversation=Conversation(
                turns=(Turn(role="assistant", parts=(ToolUse(name="f", arguments={}, id="b"),)),),
            ),
        )

        # Assertion 1 fires first because the part-id delta is unclaimed
        # (no triggered row exists). The conditional row's anchor is what
        # *names* the delta; assertion 2 does not get a chance to run.
        with pytest.raises(UnclaimedMutationError) as info:
            oracle._run_assertions(
                inbound, captured, register=(row,), triggers_met=frozenset()
            )
        assert any("id" in p for p in info.value.paths)

    def test_conditional_row_complement_passes(self) -> None:
        """A conditional row's trigger absent and no delta at its anchor."""
        row = _conditional_row()

        inbound = _empty_request()
        captured = _empty_request()

        # Identical projections → no deltas → no violation.
        deltas = oracle._run_assertions(
            inbound, captured, register=(row,), triggers_met=frozenset()
        )
        assert deltas == ()

    def test_anchored_paths_overlap_no_false_violation(self) -> None:
        """A coarser-anchor triggered row claims deltas beneath an untriggered
        conditional row's anchor without false-failing assertion 2."""
        # Conditional row with the broader anchor ``conversation.turns``.
        conditional = r.MutationRow(
            id="Z-COND",
            site=("tests/harness/test_oracle.py:Z-COND",),
            trigger=r.Trigger.GEMINI_INBOUND_ID_ABSENT,
            paths=(c.CONVERSATION_TURNS,),
            conditional=True,
            design_ref="test",
        )
        # Triggered row anchored at the narrower
        # ``conversation.turns[*].parts[*].id``.
        triggered = r.MutationRow(
            id="Z-TRIG",
            site=("tests/harness/test_oracle.py:Z-TRIG",),
            trigger=r.Trigger.ALWAYS,
            paths=(c.part_path(c.WILDCARD, c.WILDCARD, "id"),),
            conditional=False,
            design_ref="test",
        )

        inbound = Request(
            envelope=Envelope(),
            conversation=Conversation(
                turns=(Turn(role="assistant", parts=(ToolUse(name="f", arguments={}, id="a"),)),),
            ),
        )
        captured = Request(
            envelope=Envelope(),
            conversation=Conversation(
                turns=(Turn(role="assistant", parts=(ToolUse(name="f", arguments={}, id="b"),)),),
            ),
        )

        # The triggered row (Z-TRIG) is active and claims the part-level
        # ``id`` delta. The conditional row (Z-COND) has its trigger absent
        # but its delta is claimed by Z-TRIG → no assertion-2 violation.
        deltas = oracle._run_assertions(
            inbound,
            captured,
            register=(conditional, triggered),
            triggers_met=frozenset({r.Trigger.ALWAYS}),
        )
        assert deltas == (c.part_path(0, 0, "id"),)

    def test_unconditional_row_never_violates(self) -> None:
        """An unconditional row is exempt from assertion 2 by definition.

        ``conditional=False`` means the row's mutation is *expected* on every
        input, so §3.3.2 assertion 2 does not apply. With the row's trigger
        met, the row fires and claims the delta — assertion 1 passes.
        """
        unconditional = r.MutationRow(
            id="Z-UNCOND",
            site=("tests/harness/test_oracle.py:Z-UNCOND",),
            trigger=r.Trigger.GEMINI_INBOUND_ID_ABSENT,
            paths=(c.part_path(c.WILDCARD, c.WILDCARD, "id"),),
            conditional=False,
            design_ref="test",
        )

        inbound = Request(
            envelope=Envelope(),
            conversation=Conversation(
                turns=(Turn(role="assistant", parts=(ToolUse(name="f", arguments={}, id="a"),)),),
            ),
        )
        captured = Request(
            envelope=Envelope(),
            conversation=Conversation(
                turns=(Turn(role="assistant", parts=(ToolUse(name="f", arguments={}, id="b"),)),),
            ),
        )

        # Trigger is met → the row fires → the delta is claimed.
        deltas = oracle._run_assertions(
            inbound,
            captured,
            register=(unconditional,),
            triggers_met=frozenset({r.Trigger.GEMINI_INBOUND_ID_ABSENT}),
        )
        assert deltas == (c.part_path(0, 0, "id"),)


# --------------------------------------------------------------------------
# §4.3 C2 — native passthrough key-order assertion
# --------------------------------------------------------------------------


class TestNativePassthrough:
    """The captured body must equal the inbound body byte-for-byte on a
    native passthrough route."""

    def test_native_passthrough_key_order_match_passes(self) -> None:
        """Identical bodies on a native route → no raise."""
        body = b'{"messages":[{"role":"user","content":"hi"}]}'
        oracle._native_passthrough_check(_capture(body=body), _capture(body=body))

    def test_native_passthrough_key_order_byte_level(self) -> None:
        """A reordered body on a native route fails the byte-level check."""
        inbound_body = b'{"a":1,"b":2}'
        captured_body = b'{"b":2,"a":1}'

        with pytest.raises(NativePassthroughKeyOrderError):
            oracle._native_passthrough_check(
                _capture(body=inbound_body), _capture(body=captured_body)
            )

    def test_gated_by_trigger_vocabulary_not_byte_equality(self) -> None:
        """The public oracle skips the byte-level check on a translation route.

        Two bodies with the same keys in a different order project
        identically — JSON key order is serialisation noise in the projected
        diff (§3.3.4) — and with ``NON_NATIVE_UPSTREAM_WIRE`` in
        ``triggers_met`` the byte-level check is skipped, so the run passes
        despite the byte-level difference.
        """
        body_in = b'{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
        body_out = b'{"max_tokens":8,"messages":[{"role":"user","content":"hi"}],"model":"m"}'

        report = assert_no_unclaimed_mutation(
            inbound=_capture(body=body_in),
            inbound_format=WireFormat.ANTHROPIC_MESSAGES,
            captured=_capture(body=body_out),
            captured_format=WireFormat.ANTHROPIC_MESSAGES,
            register=r.REGISTER,
            triggers_met=frozenset({r.Trigger.NON_NATIVE_UPSTREAM_WIRE}),
        )
        assert report.deltas == ()


# --------------------------------------------------------------------------
# `expected_route` plumbing
# --------------------------------------------------------------------------


def _valid_messages_body() -> bytes:
    """Return the smallest body the Anthropic Messages reader accepts.

    Returns:
        JSON bytes carrying one user turn with one text part.
    """
    return b'{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'


class TestExpectedRoute:
    """The ``expected_route`` parameter is accepted and recorded."""

    def test_expected_route_none_does_not_raise(self) -> None:
        """``expected_route=None`` is the default; no raise on the routing side."""
        body = _valid_messages_body()
        report = assert_no_unclaimed_mutation(
            inbound=_capture(body=body),
            inbound_format=WireFormat.ANTHROPIC_MESSAGES,
            captured=_capture(body=body),
            captured_format=WireFormat.ANTHROPIC_MESSAGES,
            register=r.REGISTER,
            triggers_met=frozenset(),
        )
        assert report.expected_route is None

    def test_expected_route_object_recorded_without_assertion(self) -> None:
        """An arbitrary ``expected_route`` is recorded without assertion."""
        body = _valid_messages_body()
        marker = object()
        report = assert_no_unclaimed_mutation(
            inbound=_capture(body=body),
            inbound_format=WireFormat.ANTHROPIC_MESSAGES,
            captured=_capture(body=body),
            captured_format=WireFormat.ANTHROPIC_MESSAGES,
            register=r.REGISTER,
            triggers_met=frozenset(),
            expected_route=marker,
        )
        assert report.expected_route is marker
        assert isinstance(report, OracleReport)
