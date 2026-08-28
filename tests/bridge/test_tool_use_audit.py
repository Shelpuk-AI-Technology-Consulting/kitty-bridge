"""Unit tests for response-side ``tool_use`` auditing.

Covers kitty-bridge#33: an upstream returns a ``tool_use`` whose ``input`` is
the wrong shape, the bridge forwards it, and nothing in kitty's log records
what was forwarded.  The detector here is what makes that visible.

The detector's whole value is precision.  A warning that fires on ordinary
traffic trains operators to ignore it, so the negative cases below carry as
much weight as the positive one.

Covers ``.requirements/20260828T210855Z_tool_use_response_audit`` FR-1 and
FR-5.
"""

from __future__ import annotations

import pytest

from kitty.bridge.tool_audit import describe_tool_input_anomaly

# The schema Claude Code compiled into its ajv validator, reduced to the parts
# the detector reads.  From the incident report.
STRUCTURED_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {"type": "array"},
        "conversation_notes": {"type": "string"},
    },
    "required": ["findings", "conversation_notes"],
    "additionalProperties": False,
}


class TestReportedFingerprint:
    """AC-1.1 — the shape the incident actually produced."""

    def test_envelope_wrapped_payload_is_reported(self):
        """The ajv triple from the issue, reproduced as a shape mismatch."""
        tool_input = {
            "result": {
                "findings": [{"file": "a.py"}],
                "conversation_notes": "looks fine",
            }
        }
        description = describe_tool_input_anomaly("StructuredOutput", tool_input, STRUCTURED_OUTPUT_SCHEMA)

        assert description is not None
        assert "result" in description
        assert "findings" in description
        assert "conversation_notes" in description

    @pytest.mark.parametrize("wrapper", ["result", "arguments", "StructuredOutput"])
    def test_every_documented_wrapper_key_is_reported(self, wrapper):
        """The wrapper key is documented as rotating between three values.

        The detector keys off the schema, not off a list of wrapper names, so
        all three are caught by the same rule — and so would a fourth.
        """
        tool_input = {wrapper: {"findings": [], "conversation_notes": ""}}
        assert describe_tool_input_anomaly("StructuredOutput", tool_input, STRUCTURED_OUTPUT_SCHEMA) is not None

    def test_envelope_hint_is_offered_for_a_single_object_key(self):
        """AC-1.6 — a single root key wrapping an object is called out as a hint."""
        tool_input = {"result": {"findings": [], "conversation_notes": ""}}
        description = describe_tool_input_anomaly("StructuredOutput", tool_input, STRUCTURED_OUTPUT_SCHEMA)

        assert description is not None
        assert "envelope" in description.lower()

    def test_no_envelope_hint_for_a_scalar_key(self):
        """AC-1.6 — a single unexpected scalar is a mismatch but not an envelope.

        Calling it an envelope would send a reader looking for a nesting bug
        that isn't there.
        """
        description = describe_tool_input_anomaly("StructuredOutput", {"oops": "text"}, STRUCTURED_OUTPUT_SCHEMA)

        assert description is not None
        assert "envelope" not in description.lower()


class TestDetectorStaysQuiet:
    """The negative cases — each one is traffic that must never warn."""

    def test_wellformed_input_is_silent(self):
        """AC-1.2 — the ordinary case."""
        tool_input = {"findings": [], "conversation_notes": "ok"}
        assert describe_tool_input_anomaly("StructuredOutput", tool_input, STRUCTURED_OUTPUT_SCHEMA) is None

    def test_missing_required_alone_is_silent(self):
        """AC-1.3 — an incomplete payload is not the reported fingerprint.

        The model simply omitting a field is a different, commoner failure and
        is the client validator's business, not this detector's.
        """
        assert describe_tool_input_anomaly("StructuredOutput", {"findings": []}, STRUCTURED_OUTPUT_SCHEMA) is None

    def test_extra_key_alone_is_silent(self):
        """AC-1.4 — a harmless extra key is not the reported fingerprint."""
        tool_input = {"findings": [], "conversation_notes": "ok", "extra": 1}
        assert describe_tool_input_anomaly("StructuredOutput", tool_input, STRUCTURED_OUTPUT_SCHEMA) is None

    def test_unknown_tool_is_silent(self):
        """AC-1.5 — a tool the client never declared cannot be judged."""
        assert describe_tool_input_anomaly("SomeServerTool", {"anything": 1}, None) is None

    def test_schema_without_properties_or_required_is_silent(self):
        """AC-1.5 — nothing to check against.

        Claude Code declares several tools with a bare object schema; those
        must never produce a warning.
        """
        assert describe_tool_input_anomaly("Bare", {"whatever": 1}, {"type": "object"}) is None

    def test_empty_schema_is_silent(self):
        """AC-1.5 — an empty schema is not evidence of anything."""
        assert describe_tool_input_anomaly("Bare", {"whatever": 1}, {}) is None

    def test_non_dict_input_is_silent(self):
        """AC-1.5 — a non-object input is a different defect, not this one."""
        assert describe_tool_input_anomaly("StructuredOutput", ["findings"], STRUCTURED_OUTPUT_SCHEMA) is None
        assert describe_tool_input_anomaly("StructuredOutput", "text", STRUCTURED_OUTPUT_SCHEMA) is None
        assert describe_tool_input_anomaly("StructuredOutput", None, STRUCTURED_OUTPUT_SCHEMA) is None

    def test_empty_input_against_required_schema_is_silent(self):
        """An empty object has no unexpected key, so it is not the fingerprint."""
        assert describe_tool_input_anomaly("StructuredOutput", {}, STRUCTURED_OUTPUT_SCHEMA) is None

    def test_schema_with_only_required_and_no_properties_still_works(self):
        """`required` alone is enough to detect a missing field, but not an extra one.

        With no `properties` there is no way to call any key unexpected, so the
        both-conditions rule keeps this silent.
        """
        schema = {"type": "object", "required": ["findings"]}
        assert describe_tool_input_anomaly("T", {"result": {}}, schema) is None


class TestRulePrecisionIsPinned:
    """The two rules each have a case only they can catch — pin both.

    Without these, either rule could be deleted and the suite would stay green,
    which would make the spec's "must never be relaxed into an or" unenforced.
    """

    def test_fires_under_additional_properties_true(self):
        """AC-1.7 — a missing root `required` is a violation regardless.

        `additionalProperties: true` permits the extra key, but `required` at
        the root is unconditional, so the input is provably invalid. This AC
        exists to stop a future change from suppressing the warning here.
        """
        schema = {
            "type": "object",
            "properties": {"findings": {"type": "array"}, "conversation_notes": {"type": "string"}},
            "required": ["findings", "conversation_notes"],
            "additionalProperties": True,
        }
        tool_input = {"result": {"findings": [], "conversation_notes": "x"}}
        assert describe_tool_input_anomaly("StructuredOutput", tool_input, schema) is not None

    def test_nesting_rule_fires_when_the_wrapper_collides_with_a_declared_property(self):
        """The case that justifies the nesting rule's existence.

        The documented wrapper key rotates between `result`, `arguments` and
        the tool's own name. If a tool happens to declare a property called
        `result`, nothing is "unexpected" and the first rule cannot fire — the
        wrapped payload would go unreported without the nesting fingerprint.
        """
        schema = {
            "type": "object",
            "properties": {
                "result": {"type": "object"},
                "findings": {"type": "array"},
                "conversation_notes": {"type": "string"},
            },
            "required": ["findings", "conversation_notes"],
        }
        tool_input = {"result": {"findings": [], "conversation_notes": "x"}}

        description = describe_tool_input_anomaly("StructuredOutput", tool_input, schema)
        assert description is not None, "the nesting rule is the only thing that can catch this"
        assert "envelope" in description.lower()
        # No key is unexpected here, so the message must not claim otherwise.
        assert "[]" not in description

    def test_malformed_required_entry_does_not_raise(self):
        """A schema comes from the client, so a bad entry must not raise.

        Containment would disable auditing for the whole response, and the same
        schema arrives on every turn — so one malformed tool declaration would
        silently switch the audit off for the session.
        """
        schema = {"type": "object", "properties": {"p": {}}, "required": [{"bad": 1}, "p"]}

        # The malformed entry is skipped rather than poisoning the whole check:
        # the valid `p` is still judged, so one bad declaration cannot blind the
        # audit to the rest of the schema.
        description = describe_tool_input_anomaly("T", {"other": 1}, schema)
        assert description is not None
        assert "'p'" in description
        assert "bad" not in description


class TestComposedSchemas:
    """Schemas whose real property set is not in the top-level `properties`.

    A composed schema hides its properties inside `oneOf`/`anyOf`/`allOf` or
    behind `$ref`.  Treating the top-level `properties` as complete would make
    every valid key look unexpected, so the detector must stand down instead.
    """

    @pytest.mark.parametrize("keyword", ["oneOf", "anyOf", "allOf"])
    def test_composed_schema_is_silent(self, keyword):
        schema = {
            "type": "object",
            "properties": {"kind": {"type": "string"}},
            "required": ["kind"],
            keyword: [{"properties": {"findings": {"type": "array"}}}],
        }
        assert describe_tool_input_anomaly("T", {"findings": []}, schema) is None

    def test_ref_schema_is_silent(self):
        schema = {
            "type": "object",
            "properties": {"kind": {"type": "string"}},
            "required": ["kind"],
            "$ref": "#/definitions/Other",
        }
        assert describe_tool_input_anomaly("T", {"findings": []}, schema) is None
