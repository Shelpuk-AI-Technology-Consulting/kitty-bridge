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

import json

import pytest

from harness import contract as c
from harness import oracle
from harness import register as r
from harness.contract import (
    CapturedRequest,
    Conversation,
    Envelope,
    Image,
    Request,
    Text,
    ToolDecl,
    ToolResult,
    ToolUse,
    Turn,
    WireFormat,
)
from harness.oracle import (
    ConditionalRowFiredWithoutTriggerError,
    ExpectedRoute,
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

    def test_reader_for_unknown_format_raises_runtime_error(self) -> None:
        """A missing reader raises :class:`RuntimeError` at call time.

        Same exception type as the import-time ``_REGISTRY_GUARD`` so a
        downstream caller that catches one catches both.
        """
        # Use a wire format not in the closed enum by mocking the dict.
        saved = oracle._REQUEST_PROJECTIONS.copy()
        try:
            oracle._REQUEST_PROJECTIONS.pop(WireFormat.ANTHROPIC_MESSAGES, None)
            with pytest.raises(RuntimeError):
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

    def test_injected_unrecognised_field_fails_closed(self) -> None:
        """§3.3.1 falsification: an injected ``x-kitty-trace`` fails closed.

        The captured body carries an unrecognised top-level field — the
        shape a bridge-added metadata field takes on the wire. The reader
        residuals it under its bare key and ``verify_total`` rejects the
        projection before the structural diff runs: a non-empty residual is
        neither reported as a diff nor ignored (§3.3.1). The
        ``NON_NATIVE_UPSTREAM_WIRE`` trigger skips the native-passthrough
        byte check, which is not the subject here — the two bodies differ
        in content, and the gate must fire before that check is reached.
        """
        injected = json.loads(_valid_messages_body())
        injected["x-kitty-trace"] = "injected"
        with pytest.raises(c.ResidualFieldsError) as info:
            assert_no_unclaimed_mutation(
                inbound=_capture(body=_valid_messages_body()),
                inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                captured=_capture(body=json.dumps(injected).encode("utf-8")),
                captured_format=WireFormat.ANTHROPIC_MESSAGES,
                register=r.REGISTER,
                triggers_met=frozenset({r.Trigger.NON_NATIVE_UPSTREAM_WIRE}),
            )
        assert "x-kitty-trace" in str(info.value)


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

    def test_changed_system_role_yields_the_m20_path(self) -> None:
        """A dropped ``system_role`` is a positive delta at M20's anchor."""
        inbound = Request(
            envelope=Envelope(),
            conversation=Conversation(system_role="user"),
        )
        captured = Request(envelope=Envelope(), conversation=Conversation())

        deltas = oracle._structural_diff(inbound, captured)
        assert deltas == (c.SYSTEM_ROLE_PATH,)

    def test_changed_image_display_name_yields_the_m25_path(self) -> None:
        """A differing Image ``display_name`` names the M25 anchor field."""
        inbound = Request(
            envelope=Envelope(),
            conversation=Conversation(
                turns=(Turn(role="user", parts=(Image(digest="d", display_name="a"),)),),
            ),
        )
        captured = Request(
            envelope=Envelope(),
            conversation=Conversation(
                turns=(Turn(role="user", parts=(Image(digest="d", display_name="b"),)),),
            ),
        )

        deltas = oracle._structural_diff(inbound, captured)
        assert deltas == (c.part_path(0, 0, "display_name"),)

    def test_changed_image_video_metadata_yields_the_m24_path(self) -> None:
        """A differing Image ``video_metadata`` names the M24 anchor field."""
        inbound = Request(
            envelope=Envelope(),
            conversation=Conversation(
                turns=(
                    Turn(role="user", parts=(Image(digest="d", video_metadata={"fps": 24}),)),
                ),
            ),
        )
        captured = Request(
            envelope=Envelope(),
            conversation=Conversation(turns=(Turn(role="user", parts=(Image(digest="d"),)),)),
        )

        deltas = oracle._structural_diff(inbound, captured)
        assert deltas == (c.part_path(0, 0, "video_metadata"),)

    def test_claim_matching_records_every_claimer(self) -> None:
        """A delta claimed by two triggered rows records both ids."""
        broad = r.MutationRow(
            id="Z-BROAD",
            site=("tests/harness/test_oracle.py:Z-BROAD",),
            trigger=r.Trigger.ALWAYS,
            paths=(c.CONVERSATION_TURNS,),
            conditional=False,
            design_ref="test",
            scope=(r.ALL_PROVIDERS,),
        )
        narrow = r.MutationRow(
            id="Z-NARROW",
            site=("tests/harness/test_oracle.py:Z-NARROW",),
            trigger=r.Trigger.ALWAYS,
            paths=(c.part_path(c.WILDCARD, c.WILDCARD, "id"),),
            conditional=False,
            design_ref="test",
            scope=(r.ALL_PROVIDERS,),
        )

        claimers = oracle._claim_matching(
            (c.part_path(0, 0, "id"),),
            (broad, narrow),
            frozenset({r.Trigger.ALWAYS}),
        )
        assert claimers[c.part_path(0, 0, "id")] == ("Z-BROAD", "Z-NARROW")


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
            scope=(r.ALL_PROVIDERS,),
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
# §3.3.1 oracle falsification suite — T-D3 (KBR-53)
# --------------------------------------------------------------------------


def _request_with_tools(tools: tuple[ToolDecl, ...]) -> Request:
    """Return a minimal :class:`Request` whose conversation carries ``tools``.

    Args:
        tools: The tool declarations both sides of a diff start from.

    Returns:
        A :class:`Request` with an empty envelope and a conversation whose
        only content is the supplied tools.
    """
    return Request(envelope=Envelope(), conversation=Conversation(tools=tools))


class TestFalsificationSuite:
    """§3.3.1's remaining body-falsification cases and §3.3.1a's tripwire.

    Each case arranges one defect — a mutation the bridge is not registered
    to make — and asserts the oracle raises with the exact delta path. The
    cases run in the suite (plan §1.4), not demonstrated once by hand: the
    §3.3.1 totality argument is only as good as the suite's evidence that
    the projection sees each control field.

    All cases run at projection level through :func:`oracle._run_assertions`
    (the T-D1 precedent), so a failure here names the oracle's own layer.
    The trigger vocabulary is wholesale under-declared (``frozenset()``)
    except where a case needs a row live: no register row claims anything,
    so every delta the differ emits is unclaimed and assertion 1 must fire.
    """

    def test_flipped_stream_unclaimed(self) -> None:
        """A flipped ``stream`` is an unclaimed ``envelope.stream`` delta.

        P17 claims ``envelope.stream``, but only under ``ALWAYS`` — not in
        the under-declared vocabulary — and M11, P18 and P19 anchor the
        same path on other routes. None is active, so the flip must fail
        assertion 1 naming ``envelope.stream``.
        """
        inbound = _empty_request(Envelope(stream=True))
        captured = _empty_request(Envelope(stream=False))

        with pytest.raises(UnclaimedMutationError) as info:
            oracle._run_assertions(
                inbound, captured, register=r.REGISTER, triggers_met=frozenset()
            )
        assert "envelope.stream" in info.value.paths

    def test_deleted_tool_description_unclaimed(self) -> None:
        """A deleted tool ``description`` is an unclaimed delta.

        No register row anchors a tool's ``description`` — M21 anchors
        ``.behavior`` and P15 ``.strict`` precisely so this deletion stays
        claimable by nothing (§3.3.1a). The differ must see the leaf and
        assertion 1 must fail naming it.
        """
        inbound = _request_with_tools(
            (ToolDecl(name="get_weather", description="Get the weather", schema={}),)
        )
        captured = _request_with_tools((ToolDecl(name="get_weather", schema={}),))

        with pytest.raises(UnclaimedMutationError) as info:
            oracle._run_assertions(
                inbound, captured, register=r.REGISTER, triggers_met=frozenset()
            )
        assert "conversation.tools[get_weather].description" in info.value.paths

    def test_stripped_tool_strict_unclaimed(self) -> None:
        """A stripped ``strict`` where no register row applies is unclaimed.

        P15 strips ``strict`` only on the Responses-origin path; its
        trigger is not met here, so the strip is a defect the oracle must
        catch. ``strict=None`` (absent) and ``strict=False`` are distinct
        values in the projection, so the delta is visible.
        """
        inbound = _request_with_tools(
            (ToolDecl(name="get_weather", description="d", schema={}, strict=True),)
        )
        captured = _request_with_tools(
            (ToolDecl(name="get_weather", description="d", schema={}),)
        )

        with pytest.raises(UnclaimedMutationError) as info:
            oracle._run_assertions(
                inbound, captured, register=r.REGISTER, triggers_met=frozenset()
            )
        assert "conversation.tools[get_weather].strict" in info.value.paths

    def test_mutation_beneath_registered_anchor_is_unclaimed(self) -> None:
        """§3.3.1a tripwire: P15's live anchor does not swallow the delta.

        P15 (``conversation.tools[*].strict``, ``RESPONSES_ORIGIN_PATH``)
        is active — its trigger is met. The deleted ``description`` sits
        beneath the ``conversation.tools`` node, yet P15's anchor is the
        ``.strict`` leaf, so the delta must stay unclaimed. This is the
        tripwire: if P15 were re-anchored at the coarser
        ``conversation.tools[*]`` (the §3.3.1a hazard), the coarse pattern
        would claim the deep delta, this raise would silently stop
        happening, and this test would go red.
        """
        inbound = _request_with_tools(
            (
                ToolDecl(
                    name="get_weather", description="Get the weather", schema={}, strict=True
                ),
            )
        )
        captured = _request_with_tools((ToolDecl(name="get_weather", schema={}, strict=True),))

        with pytest.raises(UnclaimedMutationError) as info:
            oracle._run_assertions(
                inbound,
                captured,
                register=r.REGISTER,
                triggers_met=frozenset({r.Trigger.RESPONSES_ORIGIN_PATH}),
            )
        assert "conversation.tools[get_weather].description" in info.value.paths

    def test_coarse_anchor_claims_a_descendant_delta(self) -> None:
        """Mechanism control: a coarse anchor claims everything beneath it.

        The tripwire test's same input, judged against a fixture register
        whose single row is anchored at the coarse ``conversation.tools[*]``:
        the deep ``.description`` delta is claimed via the prefix rule and
        the oracle passes. This is what a re-anchored P15 would do — the
        reason the tripwire above must keep failing (§3.3.1a: a pattern is
        a prefix, and the matcher cannot detect over-claiming, by
        construction).
        """
        coarse_row = r.MutationRow(
            id="Z-COARSE",
            site=("tests/harness/test_oracle.py:Z-COARSE",),
            trigger=r.Trigger.RESPONSES_ORIGIN_PATH,
            paths=(c.tool_path(c.WILDCARD),),
            conditional=False,
            design_ref="test",
            scope=(r.ALL_PROVIDERS,),
        )
        inbound = _request_with_tools(
            (
                ToolDecl(
                    name="get_weather", description="Get the weather", schema={}, strict=True
                ),
            )
        )
        captured = _request_with_tools((ToolDecl(name="get_weather", schema={}, strict=True),))

        deltas = oracle._run_assertions(
            inbound,
            captured,
            register=(coarse_row,),
            triggers_met=frozenset({r.Trigger.RESPONSES_ORIGIN_PATH}),
        )
        assert deltas == ("conversation.tools[get_weather].description",)


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

    def test_pattern_prefix_honours_the_legacy_empty_bracket_spelling(self) -> None:
        """``pattern_is_proper_prefix_of`` applies ``_segment_matches``' rules,
        including the legacy ``[]`` wildcard spelling §3.3.1a keeps legal.

        A ``[]``-anchored triggered row must count as finer than a
        bare-collection untriggered row — the two predicates
        §3.3.1a says must agree cannot drift on the legacy spelling.
        """
        assert c.pattern_is_proper_prefix_of(
            "conversation.turns", "conversation.turns[].parts[].id"
        )
        # And the identity case: `[]` and `[*]` spell the same anchor, so
        # neither is a proper prefix of the other.
        assert not c.pattern_is_proper_prefix_of(
            "conversation.turns[]", "conversation.turns[*]"
        )
        assert not c.pattern_is_proper_prefix_of(
            "conversation.turns[*]", "conversation.turns[]"
        )


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
        scope=(r.ALL_PROVIDERS,),
    )


class TestAssertion2:
    """A conditional row whose trigger is not met must not produce a delta."""

    def test_untriggered_conditional_anchor_surfaces_as_unclaimed_first(self) -> None:
        """A delta at an untriggered conditional row's anchor with no
        triggered rows surfaces as §3.3.2 **assertion 1** (unclaimed), not
        assertion 2.

        With no triggered rows in the register, the delta is unclaimed and
        assertion 1 fires before assertion 2 gets a chance to run. The
        assertion-2 sentence — a conditional row firing without its trigger
        while other rows explain the deltas — is
        :meth:`test_conditional_row_raises_when_only_a_broader_row_claims_its_anchor`
        below.
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

    def test_conditional_row_raises_when_only_a_broader_row_claims_its_anchor(self) -> None:
        """The load-bearing assertion-2 case: a delta at an untriggered
        conditional row's anchor claimed *only* by a broader triggered row.

        The broader row's anchor is a proper prefix of the conditional
        row's, so it does not *specifically* claim the delta — a
        collection-level rewrite legitimately produces part-level deltas,
        and the conditional row could equally have produced this one.
        Assertion 1 passes (the broad row claims); assertion 2 fires.
        This is the "quietly becoming unconditional" scenario §3.3.2
        assertion 2 exists for, and it is invisible to assertion 1
        because the delta is claimed.
        """
        # The conditional row under test — narrow anchor.
        conditional = _conditional_row()
        # A triggered row with the BROADER anchor: conversation.turns is a
        # proper prefix of conversation.turns[*].parts[*].id.
        broader = r.MutationRow(
            id="Z-BROAD",
            site=("tests/harness/test_oracle.py:Z-BROAD",),
            trigger=r.Trigger.ALWAYS,
            paths=(c.CONVERSATION_TURNS,),
            conditional=False,
            design_ref="test",
            scope=(r.ALL_PROVIDERS,),
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

        with pytest.raises(ConditionalRowFiredWithoutTriggerError) as info:
            oracle._run_assertions(
                inbound,
                captured,
                register=(conditional, broader),
                triggers_met=frozenset({r.Trigger.ALWAYS}),
            )
        assert info.value.row_id == "Z-TEST"
        assert info.value.paths == (c.part_path(0, 0, "id"),)

    def test_equal_anchor_triggered_row_exempts_the_conditional_row(self) -> None:
        """Equal anchors co-claim: a triggered row sharing the conditional
        row's exact anchor exempts it (the M8/M3 shape)."""
        conditional = _conditional_row()
        equal = r.MutationRow(
            id="Z-EQUAL",
            site=("tests/harness/test_oracle.py:Z-EQUAL",),
            trigger=r.Trigger.ALWAYS,
            paths=(c.part_path(c.WILDCARD, c.WILDCARD, "id"),),
            conditional=False,
            design_ref="test",
            scope=(r.ALL_PROVIDERS,),
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

        # Z-EQUAL's anchor equals Z-TEST's — not a proper prefix — so it
        # specifically claims the delta and Z-TEST is exempt.
        deltas = oracle._run_assertions(
            inbound,
            captured,
            register=(conditional, equal),
            triggers_met=frozenset({r.Trigger.ALWAYS}),
        )
        assert deltas == (c.part_path(0, 0, "id"),)

    def test_narrower_triggered_row_exempts_the_untriggered_broad_anchor(self) -> None:
        """A narrower triggered row specifically claims the delta, exempting
        the untriggered broad-anchor conditional row (the M16/M3 shape)."""
        # Conditional row with the broader anchor ``conversation.turns``.
        conditional = r.MutationRow(
            id="Z-COND",
            site=("tests/harness/test_oracle.py:Z-COND",),
            trigger=r.Trigger.GEMINI_INBOUND_ID_ABSENT,
            paths=(c.CONVERSATION_TURNS,),
            conditional=True,
            design_ref="test",
            scope=(r.ALL_PROVIDERS,),
        )
        # Triggered row anchored at the narrower
        # ``conversation.turns[*].parts[*].id`` — its anchor is deeper
        # than Z-COND's, so it specifically claims the delta.
        triggered = r.MutationRow(
            id="Z-TRIG",
            site=("tests/harness/test_oracle.py:Z-TRIG",),
            trigger=r.Trigger.ALWAYS,
            paths=(c.part_path(c.WILDCARD, c.WILDCARD, "id"),),
            conditional=False,
            design_ref="test",
            scope=(r.ALL_PROVIDERS,),
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
            scope=(r.ALL_PROVIDERS,),
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
# Scope enforcement (KBR-307) — rows outside provider_key do not claim
# --------------------------------------------------------------------------


def _p17_shaped_row(scope: tuple[str, ...]) -> r.MutationRow:
    """Return a P17-shaped row with a caller-chosen scope.

    P17 is the canonical ALWAYS-triggered, narrowly-scoped row
    (``scope=("openai_subscription",)``, ``paths=(envelope.stream,
    envelope.store)``), so the fixtures that exercise the scope filter
    build one with the caller's scope rather than pointing at P17 itself —
    a doctored production row would be a mutable fact under test.

    Args:
        scope: The scope tuple the fixture row carries.

    Returns:
        An unconditional ``ALWAYS`` row anchored at ``envelope.stream``.
    """
    return r.MutationRow(
        id="Z-SCOPE-P17",
        site=("tests/harness/test_oracle.py:Z-SCOPE-P17",),
        trigger=r.Trigger.ALWAYS,
        paths=(c.ENVELOPE_STREAM,),
        conditional=False,
        design_ref="test",
        scope=scope,
    )


def _stream_flip() -> tuple[Request, Request]:
    """Return an inbound and a captured projection differing only at stream.

    Returns:
        A pair of :class:`Request` projections whose only structural delta
        is ``envelope.stream`` (``True`` inbound, ``False`` captured).
    """
    return (
        _empty_request(Envelope(stream=True)),
        _empty_request(Envelope(stream=False)),
    )


class TestScopeFilter:
    """A row whose scope excludes ``provider_key`` does not claim (KBR-307).

    KBR-139 delivered the scope data and ``row_is_in_scope``; KBR-307
    threads it into the runtime oracle surface. Every case here exercises
    the filter through the same three seams — ``_claim_matching``,
    ``_conditional_violations``, ``_run_assertions`` — so a helper-only
    test cannot pass while the runtime wiring is missing.
    """

    def test_claim_matching_filters_rows_outside_provider_scope(self) -> None:
        """A scope-narrow row contributes no claimer on a foreign adapter."""
        p17 = _p17_shaped_row(scope=("openai_subscription",))
        broad = r.MutationRow(
            id="Z-SCOPE-BROAD",
            site=("tests/harness/test_oracle.py:Z-SCOPE-BROAD",),
            trigger=r.Trigger.ALWAYS,
            paths=(c.ENVELOPE_STREAM,),
            conditional=False,
            design_ref="test",
            scope=(r.ALL_PROVIDERS,),
        )

        # On anthropic, the P17-shaped row is scope-gated out; the
        # ALL_PROVIDERS row is the only claimer.
        claimers = oracle._claim_matching(
            (c.ENVELOPE_STREAM,),
            (p17, broad),
            frozenset({r.Trigger.ALWAYS}),
            provider_key="anthropic",
        )
        assert claimers[c.ENVELOPE_STREAM] == ("Z-SCOPE-BROAD",)

        # On openai_subscription, both rows are live and both claim.
        claimers = oracle._claim_matching(
            (c.ENVELOPE_STREAM,),
            (p17, broad),
            frozenset({r.Trigger.ALWAYS}),
            provider_key="openai_subscription",
        )
        assert claimers[c.ENVELOPE_STREAM] == ("Z-SCOPE-P17", "Z-SCOPE-BROAD")

    def test_default_provider_key_is_all_providers_permissive(self) -> None:
        """Omitting ``provider_key`` leaves both rows live (the sentinel short-circuit).

        The contract is on the call-site short-circuit, not on
        ``row_is_in_scope``: the helper returns ``False`` when handed the
        sentinel as a key (the sentinel is a value of ``row.scope``, not of
        ``provider_key``), so the filter must pass through untouched under
        the default.
        """
        p17 = _p17_shaped_row(scope=("openai_subscription",))
        broad = r.MutationRow(
            id="Z-SCOPE-BROAD",
            site=("tests/harness/test_oracle.py:Z-SCOPE-BROAD",),
            trigger=r.Trigger.ALWAYS,
            paths=(c.ENVELOPE_STREAM,),
            conditional=False,
            design_ref="test",
            scope=(r.ALL_PROVIDERS,),
        )

        claimers = oracle._claim_matching(
            (c.ENVELOPE_STREAM,), (p17, broad), frozenset({r.Trigger.ALWAYS})
        )
        assert claimers[c.ENVELOPE_STREAM] == ("Z-SCOPE-P17", "Z-SCOPE-BROAD")

    def test_conditional_violations_filters_iteration(self) -> None:
        """A conditional row scoped away is not iterated on a foreign adapter.

        Two-row fixture so the discriminator reaches assertion 2 (a
        one-row fixture would raise assertion 1 before
        ``_conditional_violations`` runs): a broader triggered row B
        claims the delta via the prefix rule (assertion 1 passes), and
        the narrow conditional row R is the violation under specificity
        attribution. On ``openai_subscription``, R is iterated and its
        violation surfaces; on ``anthropic``, R is filtered out of the
        iteration and the run is clean — the discriminator.
        """
        conditional = r.MutationRow(
            id="Z-SCOPE-COND",
            site=("tests/harness/test_oracle.py:Z-SCOPE-COND",),
            trigger=r.Trigger.RESPONSES_ORIGIN_PATH,
            paths=(c.part_path(c.WILDCARD, c.WILDCARD, "id"),),
            conditional=True,
            design_ref="test",
            scope=("openai_subscription",),
        )
        broader = r.MutationRow(
            id="Z-SCOPE-BROAD2",
            site=("tests/harness/test_oracle.py:Z-SCOPE-BROAD2",),
            trigger=r.Trigger.ALWAYS,
            paths=(c.CONVERSATION_TURNS,),
            conditional=False,
            design_ref="test",
            scope=(r.ALL_PROVIDERS,),
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

        # In scope: the conditional row fires without its trigger (assertion 2).
        with pytest.raises(ConditionalRowFiredWithoutTriggerError) as info:
            oracle._run_assertions(
                inbound,
                captured,
                register=(conditional, broader),
                triggers_met=frozenset({r.Trigger.ALWAYS}),
                provider_key="openai_subscription",
            )
        assert info.value.row_id == "Z-SCOPE-COND"

        # Out of scope: the conditional row is not iterated; B claims via
        # the prefix rule and the run is clean.
        deltas = oracle._run_assertions(
            inbound,
            captured,
            register=(conditional, broader),
            triggers_met=frozenset({r.Trigger.ALWAYS}),
            provider_key="anthropic",
        )
        assert deltas == (c.part_path(0, 0, "id"),)

    def test_run_assertions_forwards_provider_key_to_conditional_violations(self) -> None:
        """The ``_run_assertions`` → ``_conditional_violations`` forward is live.

        Same two-row fixture as
        :meth:`test_conditional_violations_filters_iteration`, driven
        through ``_run_assertions`` rather than the helper directly. A
        dropped ``provider_key`` argument inside ``_run_assertions`` would
        let R through the filter under the permissive default and raise
        ``ConditionalRowFiredWithoutTriggerError`` where the real forward
        keeps the run clean.
        """
        conditional = r.MutationRow(
            id="Z-SCOPE-COND",
            site=("tests/harness/test_oracle.py:Z-SCOPE-COND",),
            trigger=r.Trigger.RESPONSES_ORIGIN_PATH,
            paths=(c.part_path(c.WILDCARD, c.WILDCARD, "id"),),
            conditional=True,
            design_ref="test",
            scope=("openai_subscription",),
        )
        broader = r.MutationRow(
            id="Z-SCOPE-BROAD2",
            site=("tests/harness/test_oracle.py:Z-SCOPE-BROAD2",),
            trigger=r.Trigger.ALWAYS,
            paths=(c.CONVERSATION_TURNS,),
            conditional=False,
            design_ref="test",
            scope=(r.ALL_PROVIDERS,),
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

        deltas = oracle._run_assertions(
            inbound,
            captured,
            register=(conditional, broader),
            triggers_met=frozenset({r.Trigger.ALWAYS}),
            provider_key="anthropic",
        )
        assert deltas == (c.part_path(0, 0, "id"),)

    def test_run_assertions_raises_unclaimed_when_scope_filter_removes_the_only_claimer(
        self,
    ) -> None:
        """A P17-shaped row on a foreign adapter leaves the delta unclaimed.

        Single-row register, the trigger met, the delta claimed nowhere:
        on ``openai_subscription`` the row claims and the run is clean; on
        ``anthropic`` the row is scope-gated out and assertion 1 fires
        naming ``envelope.stream``. This is the AC5 half of the pair.
        """
        p17 = _p17_shaped_row(scope=("openai_subscription",))
        inbound, captured = _stream_flip()

        deltas = oracle._run_assertions(
            inbound,
            captured,
            register=(p17,),
            triggers_met=frozenset({r.Trigger.ALWAYS}),
            provider_key="openai_subscription",
        )
        assert deltas == (c.ENVELOPE_STREAM,)

        with pytest.raises(UnclaimedMutationError) as info:
            oracle._run_assertions(
                inbound,
                captured,
                register=(p17,),
                triggers_met=frozenset({r.Trigger.ALWAYS}),
                provider_key="anthropic",
            )
        assert "envelope.stream" in info.value.paths

    def test_run_assertions_p17_claims_on_openai_subscription(self) -> None:
        """The production register's P17 claims on its own adapter (AC5's pin).

        The single-row fixtures above are surgical; this case runs the
        same stream flip against :data:`harness.register.REGISTER` — where
        P17 is live — and flips only the ``provider_key``. On
        ``openai_subscription`` P17 claims the delta and the run passes;
        on ``anthropic`` the flip is unclaimed. The gating is per-adapter,
        not global.
        """
        inbound, captured = _stream_flip()

        deltas = oracle._run_assertions(
            inbound,
            captured,
            register=r.REGISTER,
            triggers_met=frozenset({r.Trigger.ALWAYS}),
            provider_key="openai_subscription",
        )
        assert deltas == (c.ENVELOPE_STREAM,)

        with pytest.raises(UnclaimedMutationError) as info:
            oracle._run_assertions(
                inbound,
                captured,
                register=r.REGISTER,
                triggers_met=frozenset({r.Trigger.ALWAYS}),
                provider_key="anthropic",
            )
        assert "envelope.stream" in info.value.paths

    def test_scope_filter_removal_breaks_claim_matching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mechanism control: a permissive helper re-opens the over-claim.

        The same shape the R5b control uses (KBR-53): the helper is
        weakened to "always True", the filter goes permissive, and the
        P17-shaped row claims on ``anthropic`` — the exact delta the real
        filter keeps unclaimed. If this case stops failing-by-design, the
        filter has stopped being the load-bearing behaviour.
        """
        p17 = _p17_shaped_row(scope=("openai_subscription",))
        inbound, captured = _stream_flip()

        monkeypatch.setattr(r, "row_is_in_scope", lambda row, key: True)

        # With the helper permissive, P17 claims on anthropic and the run
        # is clean — the over-claim this ticket's filter exists to prevent.
        deltas = oracle._run_assertions(
            inbound,
            captured,
            register=(p17,),
            triggers_met=frozenset({r.Trigger.ALWAYS}),
            provider_key="anthropic",
        )
        assert deltas == (c.ENVELOPE_STREAM,)


# --------------------------------------------------------------------------
# §4.3 C2 — native passthrough key-order assertion
# --------------------------------------------------------------------------


class TestNativePassthrough:
    """The captured body's JSON key order must equal the inbound body's on
    a native passthrough route — key order, not bytes: the native branch
    rewrites ``model`` through M1, a registered mutation a byte check
    would fail."""

    def test_native_passthrough_key_order_match_passes(self) -> None:
        """Identical bodies on a native route → no raise."""
        body = b'{"messages":[{"role":"user","content":"hi"}]}'
        oracle._native_passthrough_check(_capture(body=body), _capture(body=body))

    def test_native_passthrough_reordered_keys_fail(self) -> None:
        """A reordered body on a native route fails the key-order check."""
        inbound_body = b'{"a":1,"b":2}'
        captured_body = b'{"b":2,"a":1}'

        with pytest.raises(NativePassthroughKeyOrderError):
            oracle._native_passthrough_check(
                _capture(body=inbound_body), _capture(body=captured_body)
            )

    def test_native_passthrough_same_keys_different_values_pass(self) -> None:
        """M1's model rewrite preserves key order → the check passes.

        This is the case the earlier byte-equality implementation got
        wrong: the native branch rewrites ``model`` through
        ``_normalize_model``, so a run whose profile model differs from
        the agent's model differs in bytes while preserving key order —
        a legitimate, registered mutation C2 must not fail.
        """
        inbound_body = b'{"model":"agent-model","messages":[]}'
        captured_body = b'{"model":"profile-model","messages":[]}'

        oracle._native_passthrough_check(
            _capture(body=inbound_body), _capture(body=captured_body)
        )

    def test_native_passthrough_nested_reorder_fails(self) -> None:
        """A reorder at a nested object level fails the check."""
        inbound_body = b'{"outer":{"x":1,"y":2}}'
        captured_body = b'{"outer":{"y":2,"x":1}}'

        with pytest.raises(NativePassthroughKeyOrderError):
            oracle._native_passthrough_check(
                _capture(body=inbound_body), _capture(body=captured_body)
            )

    def test_native_passthrough_structural_divergence_does_not_trip_c2(self) -> None:
        """A content difference (a key one side lacks) is not an ordering
        fingerprint — the walk stops and C2 does not fire.

        Assertion 1 owns content differences, through the projections;
        C2 owns only the ordering a serialiser fingerprint reads.
        """
        inbound_body = b'{"a":1,"b":2}'
        captured_body = b'{"a":1}'

        oracle._native_passthrough_check(
            _capture(body=inbound_body), _capture(body=captured_body)
        )

    def test_native_passthrough_unparseable_differing_bytes_fall_back_strict(self) -> None:
        """Bodies that are not JSON cannot be walked for key order; byte
        equality is the only ordering claim left, applied strictly."""
        inbound_body = b"not json at all"
        captured_body = b"not the same bytes"

        with pytest.raises(NativePassthroughKeyOrderError):
            oracle._native_passthrough_check(
                _capture(body=inbound_body), _capture(body=captured_body)
            )

    def test_gated_by_trigger_vocabulary_not_byte_equality(self) -> None:
        """The public oracle skips the key-order check on a translation route.

        Two bodies with the same keys in a different order project
        identically — JSON key order is serialisation noise in the projected
        diff (§3.3.4) — and with ``NON_NATIVE_UPSTREAM_WIRE`` in
        ``triggers_met`` the key-order check is skipped, so the run passes
        despite the ordering difference.
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
    """The ``expected_route`` parameter is accepted, asserted, and recorded."""

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

    def test_expected_route_matching_the_capture_is_accepted_and_reported(self) -> None:
        """A matching ``ExpectedRoute`` is asserted and recorded.

        T-D2 (KBR-52) turned the parameter from recorded-without-assertion
        into asserted: an expectation that agrees with the captured route on
        every component reaches the report, and one that disagrees raises
        (the routing suite in ``test_oracle_routing.py`` owns the failure
        cases). The oracle-level claim here is the plumbing — accepted,
        returned unchanged.
        """
        body = _valid_messages_body()
        captured = _capture(body=body)
        expected = ExpectedRoute(
            method=captured.method,
            scheme=captured.scheme,
            host=captured.host,
            path=captured.path,
            query=captured.query,
        )
        report = assert_no_unclaimed_mutation(
            inbound=_capture(body=body),
            inbound_format=WireFormat.ANTHROPIC_MESSAGES,
            captured=captured,
            captured_format=WireFormat.ANTHROPIC_MESSAGES,
            register=r.REGISTER,
            triggers_met=frozenset(),
            expected_route=expected,
        )
        assert report.expected_route is expected
        assert isinstance(report, OracleReport)
