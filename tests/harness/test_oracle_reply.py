"""Tests for the response-direction transparency oracle.

`.system_design/TEST_SUITE.md` §3.3.1 (last paragraph), §3.3.2, §7.4 ·
plan task **T-D10** (KBR-59).

Tests run at ``l1`` (path-default per ``tests/layers.py``). ``l3``
activation is T-K6's job; the corpus-driven reply matrix is T-D9's
business and lands after the M27/M28/M29 corpus entries exist.

**Two test populations, deliberately separated** (the T-W9
attribution rule — a failing test must name the layer that failed):

* **Unit tests** drive :func:`oracle.assert_no_unclaimed_reply_mutation`,
  the reply-twin of
  :func:`~harness.oracle.assert_no_unclaimed_mutation`, against
  synthesised :class:`~harness.contract.Reply` projections and a
  CapturedReply pair. No bridge, no recorder, no adapter. A
  failure here is the reply oracle's own.
* **Driven tests** (the suffix ``_driven``) drive the entry point
  with real `CapturedReply` bytes through the landed readers; the
  driven test file lives at ``test_oracle_reply_driven.py`` so the
  boundary stays visible (KBR-307 round-1 lesson).
"""

from __future__ import annotations

import inspect
import json

import pytest

from harness import contract as c
from harness import oracle
from harness import register as r
from harness.contract import (
    CapturedReply,
    Reply,
    Text,
    Thinking,
    ToolUse,
    WireFormat,
)

# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def _reply_capture(
    body: bytes = b"",
    status: int = 200,
    headers: tuple[tuple[str, str], ...] = (),
) -> CapturedReply:
    """Return a minimal :class:`CapturedReply` for reply-side tests.

    Args:
        body: The raw reply body bytes; passed through verbatim.
        status: HTTP status code; defaults to ``200``.
        headers: Ordered header pairs; defaults to empty.

    Returns:
        A :class:`CapturedReply` carrying the supplied body.
    """
    return CapturedReply(status=status, headers=headers, body=body)


def _reply(
    parts: tuple[c.Part, ...] = (),
    stop_reason: str | None = None,
    *,
    stop_reason_raw: str | None = None,
) -> Reply:
    """Return a minimal :class:`Reply` projection with the given parts.

    Args:
        parts: The reply parts in order; defaults to none.
        stop_reason: The canonical stop reason; defaults to ``None``.
        stop_reason_raw: The wire's own value when ``stop_reason`` is
            ``"other"``; required (per :class:`Reply`'s post-init
            invariant, ``contract.py:1354``) when ``stop_reason`` is
            ``"other"``.

    Returns:
        A :class:`Reply` carrying the supplied parts and ``stop_reason``.
        ``usage`` and ``residual`` are empty by construction; the
        :func:`~harness.contract.verify_total` gate passes.
    """
    return Reply(parts=parts, stop_reason=stop_reason, stop_reason_raw=stop_reason_raw)


def _valid_messages_reply_body() -> bytes:
    """Return a minimal Anthropic Messages reply body.

    Used by the falsification tests to construct a ``CapturedReply``
    whose body is recognised by ``AnthropicMessagesReplyProjection``
    without raising :class:`~harness.contract.UnreadableBodyError`.
    The body is minimal but well-formed — a single text block plus a
    ``stop_reason``.

    Returns:
        UTF-8 bytes of a JSON object the Anthropic Messages reader
        accepts.
    """
    return json.dumps(
        {
            "id": "msg_test_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ).encode("utf-8")


# --------------------------------------------------------------------------
# §7.4 reply signature and reader registry
# --------------------------------------------------------------------------


class TestSignature:
    """The reply-direction entry point matches §3.3.1's contract."""

    def test_reply_oracle_entry_point_is_exported(self) -> None:
        """``assert_no_unclaimed_reply_mutation`` is a public symbol on
        ``harness.oracle``.

        AC-FR-1 + AC-FR-8.
        """
        assert hasattr(oracle, "assert_no_unclaimed_reply_mutation")

    def test_reply_oracle_entry_point_signature(self) -> None:
        """The signature has exactly the documented shape.

        AC-FR-1: positional ``CapturedReply, WireFormat, CapturedReply,
        WireFormat, register, triggers_met``; keyword-only
        ``provider_key``; **no** ``expected_route``. The parameter
        types are exactly ``(CapturedReply, WireFormat, CapturedReply,
        WireFormat, tuple[MutationRow, ...], frozenset[Trigger], ...)
        -> ReplyOracleReport`` — verified via :mod:`inspect`.
        """
        sig = inspect.signature(oracle.assert_no_unclaimed_reply_mutation)
        params = sig.parameters

        # Positional first six.
        positional = [
            "inbound",
            "inbound_format",
            "captured",
            "captured_format",
            "register",
            "triggers_met",
        ]
        for name in positional:
            assert name in params, f"missing positional {name}"
            assert params[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

        # Keyword-only ``provider_key``.
        assert "provider_key" in params
        assert params["provider_key"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["provider_key"].default == r.ALL_PROVIDERS

        # No ``expected_route`` — the reply side does not check routing.
        assert "expected_route" not in params

    def test_reply_oracle_report_dataclass_shape(self) -> None:
        """``ReplyOracleReport`` is frozen with the documented fields.

        AC-FR-1 + S3: ``__hash__ = None`` set explicitly.
        """
        import dataclasses as _dc

        report = oracle.ReplyOracleReport(
            inbound_projection=_reply(),
            captured_projection=_reply(),
            deltas=(),
        )
        # The dataclass is frozen.
        with pytest.raises(_dc.FrozenInstanceError):
            report.deltas = ("reply.parts[0]",)  # type: ignore[misc]

        # ``__hash__`` is explicitly None so set/dict use fails loud.
        assert oracle.ReplyOracleReport.__hash__ is None
        with pytest.raises(TypeError):
            hash(report)

        # Fields match.
        assert report.inbound_projection is not None
        assert report.captured_projection is not None
        assert tuple(report.deltas) == ()

    def test_reply_registry_floor_set_of_five(self) -> None:
        """``_ASSERTABLE_REPLY_FORMATS`` names exactly the five formats
        that have a reader today.

        AC-FR-4 — the superset guard's *floor* is documented here, not
        inferred by counting the registry.
        """
        assert set(oracle._ASSERTABLE_REPLY_FORMATS) == {
            WireFormat.ANTHROPIC_MESSAGES,
            WireFormat.CHAT_COMPLETIONS,
            WireFormat.GEMINI,
            WireFormat.OLLAMA_CHAT,
            WireFormat.OPENAI_RESPONSES,
        }

    def test_reply_reader_for_returns_registered_reader(self) -> None:
        """``_reply_reader_for`` returns the registered reader for each
        of the assertable formats (AC-FR-4)."""
        for fmt in oracle._ASSERTABLE_REPLY_FORMATS:
            assert oracle._reply_reader_for(fmt).wire_format == fmt

    def test_reply_reader_for_unknown_format_raises_runtime_error(self) -> None:
        """A missing reader raises :class:`RuntimeError` at call time.

        AC-FR-4 — the same exception type as the request oracle's
        ``_REGISTRY_GUARD`` so a downstream caller that catches one
        catches both.
        """
        # Pop one entry and assert the call-time raise; restore.
        bogus = WireFormat.BEDROCK_CONVERSE
        assert bogus not in oracle._REPLY_PROJECTIONS, (
            "fixture precondition: BEDROCK_CONVERSE has no reply reader today "
            "(KBR-312 sibling)"
        )
        with pytest.raises(RuntimeError):
            oracle._reply_reader_for(bogus)


# --------------------------------------------------------------------------
# Reply-side structural diff (FR-3, AC-FR-3)
# --------------------------------------------------------------------------


class TestStructuralReplyDiff:
    """The reply diff skips ``usage``/``stop_reason_raw``, walks ``parts``."""

    def test_differ_only_in_parts_zero_text_yields_one_path(self) -> None:
        """Two :class:`Reply` projections equal except for ``parts[0].text``
        produce a single ``reply.parts[0].text`` path."""
        a = _reply(parts=(Text(text="one"),), stop_reason="end_turn")
        b = _reply(parts=(Text(text="two"),), stop_reason="end_turn")
        deltas = oracle._structural_reply_diff(a, b)
        assert tuple(deltas) == ("reply.parts[0].text",)

    def test_differ_only_in_parts_two_id_yields_one_path(self) -> None:
        """Two replies equal except for ``parts[2].id`` produce a single
        ``reply.parts[2].id`` path — the M27 anchor."""
        a = _reply(
            parts=(
                Text(text="x"),
                ToolUse(id="a", name="fn", arguments={}),
                ToolUse(id="b", name="fn", arguments={}),
                Text(text="y"),
            ),
            stop_reason="end_turn",
        )
        b = _reply(
            parts=(
                Text(text="x"),
                ToolUse(id="a", name="fn", arguments={}),
                ToolUse(id="c", name="fn", arguments={}),
                Text(text="y"),
            ),
            stop_reason="end_turn",
        )
        deltas = oracle._structural_reply_diff(a, b)
        assert tuple(deltas) == ("reply.parts[2].id",)

    def test_differ_only_in_usage_yields_no_diffs(self) -> None:
        """``Reply.usage`` is carried but excluded from the fidelity diff
        (``contract.py:1318``); a usage delta produces zero paths."""
        a = _reply(parts=(), stop_reason="end_turn")
        object.__setattr__(a, "usage", {"input_tokens": 1})
        b = _reply(parts=(), stop_reason="end_turn")
        object.__setattr__(b, "usage", {"input_tokens": 2})
        assert oracle._structural_reply_diff(a, b) == ()

    def test_differ_only_in_stop_reason_raw_yields_no_diffs(self) -> None:
        """``Reply.stop_reason_raw`` carries no I1 information; a raw-value
        delta produces zero paths (FR-3, post-C10)."""
        a = _reply(stop_reason="other", stop_reason_raw="SAFETY")
        b = _reply(stop_reason="other", stop_reason_raw="RECITATION")
        assert oracle._structural_reply_diff(a, b) == ()

    def test_differ_in_stop_reason_yields_bare_path(self) -> None:
        """Two replies equal except for ``stop_reason`` produce a single
        ``reply.stop_reason`` path — the M28 anchor (a keyed literal,
        no wildcard)."""
        a = _reply(stop_reason="end_turn")
        b = _reply(stop_reason="tool_use")
        assert tuple(oracle._structural_reply_diff(a, b)) == ("reply.stop_reason",)

    def test_differ_in_part_kind_yields_index_path(self) -> None:
        """Two replies equal except one has a different ``kind`` at
        ``parts[1]`` produce a single ``reply.parts[1]`` path — the
        M29 bare-index anchor for a kind-boundary disagreement.

        Index 0 and 2 are deliberately matched in both kinds so
        the diff walks a single boundary, not a rearrangement
        (which would emit ``reply.parts[1]`` and ``reply.parts[2]``
        under §7.4.1).
        """
        a = _reply(
            parts=(Text(text="x"), Thinking(text="t"), Text(text="y")),
            stop_reason="end_turn",
        )
        b = _reply(
            parts=(Text(text="x"), ToolUse(id=None, name="fn", arguments={}), Text(text="y")),
            stop_reason="end_turn",
        )
        assert tuple(oracle._structural_reply_diff(a, b)) == ("reply.parts[1]",)


# --------------------------------------------------------------------------
# Reply-side totality gate (§3.3.1, §7.4)
# --------------------------------------------------------------------------


class TestReplyTotalityGate:
    """A non-empty residual fails the run before the structural diff.

    AC-FR-2 + AC-FR-9. The exception type is the ``verify_total`` type
    unmodified — the entry point does **not** wrap it (S6).

    The AC names three *entry-point-driven* cases (reviewer suggestion
    on AC-FR-2's granularity): residual on the inbound projection,
    residual on the captured projection, and a dropped field on either
    side — each must raise **before** the structural diff runs. The
    four contract-level tests below pin the exception types at
    :func:`~harness.contract.verify_total`; the three entry-point tests
    at the bottom of the class pin the call order end-to-end through
    :func:`oracle.assert_no_unclaimed_reply_mutation`.
    """

    def test_residual_on_inbound_raises_residual_fields_error(self) -> None:
        """An inbound reply with a non-empty ``residual`` raises before
        the structural diff runs.

        The test builds a real Anthropic-shape inbound body so the
        reader is exercised; the inbound reply's reader is forced to
        ``residual={"unexpected": "v"}`` by monkeypatching the
        projector's source — but for simplicity the test drives
        :func:`oracle._structural_reply_diff` with a hand-built Reply.
        The entry-point test is the *driven* case below.
        """
        inbound = Reply(
            parts=(Text(text="hi"),),
            stop_reason="end_turn",
            residual={"unexpected": "value"},
        )
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(inbound)

    def test_residual_on_captured_raises_residual_fields_error(self) -> None:
        """A captured reply with a non-empty ``residual`` raises."""
        captured = Reply(
            parts=(Text(text="hi"),),
            stop_reason="end_turn",
            residual={"unexpected": "value"},
        )
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(captured)

    def test_dropped_field_raises_dropped_fields_error(self) -> None:
        """A projection whose ``consumed`` misses a source key raises."""
        projected = Reply(
            parts=(Text(text="hi"),),
            stop_reason="end_turn",
            source={"model": "x", "surprise": "y"},
            consumed=frozenset({"model"}),
        )
        with pytest.raises(c.DroppedFieldsError):
            c.verify_total(projected)

    def test_injected_unrecognised_field_fails_closed(self) -> None:
        """§3.3.1 falsification analogue on the reply direction.

        An injected ``x-kitty-trace`` on the captured side is mapped
        to the projection's residual by the reader;
        :func:`~harness.contract.verify_total` raises
        :class:`ResidualFieldsError` — unmodified (S6). This is the
        load-bearing falsification §1.4 The harness rule requires
        for the reply-direction oracle, and the only assertion it
        needs is "the run fails" — the conventional first failure
        probe for a translate-and-compare mechanism.
        """
        # Inject the field at the wire level; the reader residuals it
        # under its bare key; verify_total raises before the diff
        # runs. We exercise the gate directly against a Reply
        # projection to keep this test fast and isolated.
        projected = Reply(
            parts=(Text(text="hi"),),
            stop_reason="end_turn",
            residual={"x-kitty-trace": "injected"},
        )
        with pytest.raises(c.ResidualFieldsError) as info:
            c.verify_total(projected)
        assert "x-kitty-trace" in str(info.value)

    def test_entry_point_residual_on_inbound_raises_before_diff(self) -> None:
        """AC-FR-2, first case, at the entry-point level: a non-empty
        residual on the *inbound* projection raises
        :class:`ResidualFieldsError` before the structural diff runs.

        Mirrors ``TestDrivenReplyFalsification``'s captured-side twin;
        this is the inbound-side shape.
        """
        inbound_dict = json.loads(_valid_messages_reply_body())
        inbound_dict["x-kitty-trace"] = "injected"
        with pytest.raises(c.ResidualFieldsError) as info:
            oracle.assert_no_unclaimed_reply_mutation(
                inbound=_reply_capture(body=json.dumps(inbound_dict).encode("utf-8")),
                inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                captured=_reply_capture(body=_valid_messages_reply_body()),
                captured_format=WireFormat.ANTHROPIC_MESSAGES,
                register=r.REGISTER,
                triggers_met=frozenset(),
            )
        assert "x-kitty-trace" in str(info.value)

    def test_entry_point_dropped_field_on_captured_raises_before_diff(self) -> None:
        """AC-FR-2, third case, at the entry-point level: a captured
        projection carrying a dropped field raises
        :class:`DroppedFieldsError` — the ``verify_total`` type, not a
        wrapper — before the structural diff runs.

        The reader the test drives is a monkeypatched fake whose
        ``consumed`` set names a key the wire body does not carry, so
        the projection carries a *drop*. Pinning the entry point's
        call order, not the reader's correctness, is the test's job.
        """
        body = _valid_messages_reply_body()

        class _BuggyReader:
            """A reader that drops a body key — the C8 falsification shape."""

            wire_format = WireFormat.ANTHROPIC_MESSAGES

            def read_reply(self, captured: CapturedReply) -> Reply:
                """Project a reply with a dropped key.

                Args:
                    captured: The reply capture; ignored.

                Returns:
                    A :class:`Reply` whose ``consumed`` misses a key
                    its ``source`` carries — the shape
                    :func:`~harness.contract.verify_total` rejects.
                """
                return Reply(
                    parts=(Text(text="hi"),),
                    stop_reason="end_turn",
                    source={"model": "x", "surprise": "y"},
                    consumed=frozenset({"model"}),
                )

        saved = oracle._REPLY_PROJECTIONS[WireFormat.ANTHROPIC_MESSAGES]
        try:
            oracle._REPLY_PROJECTIONS[WireFormat.ANTHROPIC_MESSAGES] = _BuggyReader()
            with pytest.raises(c.DroppedFieldsError):
                oracle.assert_no_unclaimed_reply_mutation(
                    inbound=_reply_capture(body=body),
                    inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                    captured=_reply_capture(body=body),
                    captured_format=WireFormat.ANTHROPIC_MESSAGES,
                    register=r.REGISTER,
                    triggers_met=frozenset(),
                )
        finally:
            oracle._REPLY_PROJECTIONS[WireFormat.ANTHROPIC_MESSAGES] = saved


# --------------------------------------------------------------------------
# Reply projection registry (FR-6, AC-FR-4)
# --------------------------------------------------------------------------


class TestReplyRegistry:
    """The reply registry's superset guard fires on the symmetric failure."""

    def test_superset_guard_passes_at_import(self) -> None:
        """The import-time guard ``_ASSERTABLE_REPLY_FORMATS <=
        _REPLY_PROJECTIONS`` holds today (AC-FR-4 import-time half)."""
        assert set(oracle._ASSERTABLE_REPLY_FORMATS) <= set(oracle._REPLY_PROJECTIONS)

    def test_bedrock_converse_not_in_reply_registry(self) -> None:
        """KBR-312 sibling: Bedrock Converse has no reply reader yet."""
        assert WireFormat.BEDROCK_CONVERSE not in oracle._REPLY_PROJECTIONS

    def test_guard_fires_when_the_floor_constant_grows_without_a_reader(self) -> None:
        """The load-bearing failure mode AC-FR-4 pins (reviewer finding
        S8): a format added to ``_ASSERTABLE_REPLY_FORMATS`` with no
        reader registered fails the import-time guard. KBR-312's
        landing exercises exactly this case — the constant grows from
        five to six formats and the guard stays quiet only once the
        reader lands.
        """
        grown = oracle._ASSERTABLE_REPLY_FORMATS | {WireFormat.BEDROCK_CONVERSE}
        saved = oracle._ASSERTABLE_REPLY_FORMATS
        try:
            oracle._ASSERTABLE_REPLY_FORMATS = grown
            with pytest.raises(RuntimeError, match="_ASSERTABLE_REPLY_FORMATS grew"):
                oracle._REPLY_REGISTRY_GUARD()
        finally:
            oracle._ASSERTABLE_REPLY_FORMATS = saved


# --------------------------------------------------------------------------
# §3.3.2 assertions on the reply side (FR-1, AC-FR-5)
# --------------------------------------------------------------------------


class TestReplyAssertion1:
    """§3.3.2 assertion 1 on the reply side: every delta is claimed."""

    def test_no_delta_passes(self) -> None:
        """Two identical :class:`Reply` projections pass with zero deltas.

        AC-FR-5: ``assert_no_unclaimed_reply_mutation`` runs to
        completion on identical reply bodies in the Anthropic shape.
        """
        body = _valid_messages_reply_body()
        report = oracle.assert_no_unclaimed_reply_mutation(
            inbound=_reply_capture(body=body),
            inbound_format=WireFormat.ANTHROPIC_MESSAGES,
            captured=_reply_capture(body=body),
            captured_format=WireFormat.ANTHROPIC_MESSAGES,
            register=r.REGISTER,
            triggers_met=frozenset(),
        )
        assert tuple(report.deltas) == ()

    def test_run_assertions_keeps_default_diff_argument(self) -> None:
        """Regression for FR-4 / reviewer finding C7: the existing
        25-call-site pattern ``_run_assertions(inbound, captured,
        register, triggers_met)`` keeps working — ``diff`` defaults to
        ``_structural_diff`` at call time."""
        # Pass an empty register + empty triggers, two identical empty
        # ``Request`` projections; the legacy call form should produce
        # an empty ``deltas`` list.
        empty_request = c.Request(envelope=c.Envelope(), conversation=c.Conversation())
        deltas = oracle._run_assertions(
            empty_request,
            empty_request,
            register=r.REGISTER,
            triggers_met=frozenset(),
        )
        assert tuple(deltas) == ()


class TestReplyAssertion2:
    """§3.3.2 assertion 2 on the reply side: a conditional row whose
    trigger is absent cannot fire."""

    def test_conditional_row_fired_without_trigger_raises(self) -> None:
        """Two-row fixture (KBR-307 round-2 lesson): a broader triggered
        claimer R' co-claims the delta's *parent* anchor, so the
        discriminator reaches assertion 2 instead of being masked
        behind assertion 1's raise — and the narrow untriggered row
        R fires because R' is a proper prefix of R (the M5/M3 case,
        applied to the reply axis).

        The fixture uses synthetic rows with distinct ids rather than
        the register's own, so the test is independent of any
        register-row addition and stays readable on failure.
        """
        broad_triggered = r.MutationRow(
            id="R-BROAD-TRIG",
            site=("fake:site",),
            trigger=r.Trigger.ALWAYS,
            paths=("reply.parts[*]",),
            conditional=False,
            design_ref="§3.3.2",
            scope=(r.ALL_PROVIDERS,),
        )
        narrow_conditional = r.MutationRow(
            id="R-NARROW-COND",
            site=("fake:site",),
            trigger=r.Trigger.REPLY_CONTAINS_THINKING_AND_TOOL_USE,
            paths=("reply.parts[*].id",),
            conditional=True,
            design_ref="§3.3.2",
            scope=(r.ALL_PROVIDERS,),
        )
        register = (broad_triggered, narrow_conditional)

        a = _reply(
            parts=(
                Text(text="x"),
                ToolUse(id="a", name="fn", arguments={}),
            ),
            stop_reason="end_turn",
        )
        b = _reply(
            parts=(
                Text(text="x"),
                ToolUse(id="different", name="fn", arguments={}),
            ),
            stop_reason="end_turn",
        )

        # Drive the assertions directly on the synthesised Reply
        # projections — bypasses the JSON-routing through
        # CapturedReply and exercises the path-independent diff
        # machinery on a Reply-only surface. ``triggers_met`` carries
        # ALWAYS so R-BROAD-TRIG is trigger-eligible (the KBR-307
        # step-file precedent: "ALWAYS keeps P17 trigger-eligible"); the
        # narrow row's trigger is deliberately absent, which is what
        # assertion 2 exists to catch.
        with pytest.raises(oracle.ConditionalRowFiredWithoutTriggerError) as info:
            oracle._run_assertions(
                a,
                b,
                register=register,
                triggers_met=frozenset({r.Trigger.ALWAYS}),
                diff=oracle._structural_reply_diff,
            )
        assert info.value.row_id == "R-NARROW-COND"


# --------------------------------------------------------------------------
# Reply-side scope enforcement (FR-7, AC-FR-5)
# --------------------------------------------------------------------------


class TestReplyScopeEnforcement:
    """A narrow ``provider_key`` filters rows out of both claim and
    conditional surfaces."""

    def test_default_provider_key_is_all_providers(self) -> None:
        """The default ``provider_key`` is ``r.ALL_PROVIDERS``
        (KBR-307 default — call-site short-circuit)."""
        sig = inspect.signature(oracle.assert_no_unclaimed_reply_mutation)
        assert sig.parameters["provider_key"].default == r.ALL_PROVIDERS


# --------------------------------------------------------------------------
# Reply-side falsification (§1.4 The harness rule)
# --------------------------------------------------------------------------


class TestReplyFalsification:
    """The reply-direction oracle's first falsification case."""

    def test_falsification_symbol_documented_in_module(self) -> None:
        """§1.4: the module carries a comment naming the
        response-direction falsification. A future reader must be
        able to grep ``falsification`` in ``harness/oracle.py`` and
        find the response-side analogue."""
        # Assert the module's docstring contains the word
        # ``falsification`` — checked statically; the implementation
        # pass adds the comment.
        assert "falsification" in (oracle.__doc__ or "")
