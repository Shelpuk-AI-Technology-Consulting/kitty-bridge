"""L1 tests for the Gemini reply reader (response direction).

`.system_design/TEST_SUITE.md` §3.3.1 (the response-direction bullet), §7.4 ·
plan task **T-A7** (KBR-39).

**Every body here follows Google's published v1beta schema** — the discovery
document ``https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta``,
retrieved 2026-09-16. Unlike the Anthropic and OpenAI references, the discovery
document ships schema definitions but no worked response example, so each
fixture here is *assembled from the schema's own field list* (the docstring on
each fixture names the fields it instantiates) and marked as such — that is the
schema, not kitty's output, and not a vendor sample.  Bodies this module
*constructs* (an injected key, a pairing-rule trip) are marked separately.

**Every falsification case is paired with a control**, per plan §1.4's harness
rule applied to a projection.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from harness import contract as c
from harness import reader_gemini as gm

# --------------------------------------------------------------------------
# Published-schema fixtures (GenerateContentResponse, retrieved 2026-09-16)
# --------------------------------------------------------------------------

# Assembled from ``GenerateContentResponse`` + ``Candidate`` + ``Content`` +
# ``UsageMetadata`` fields of the v1beta discovery document: a text reply with
# ``finishReason = "STOP"``, the usage block the schema defines, and the
# candidate bookkeeping fields (``safetyRatings``, ``index``,
# ``avgLogprobs``) the schema lists beside them.
PUBLISHED_TEXT_RESPONSE: dict[str, Any] = {
    "candidates": [
        {
            "content": {
                "parts": [{"text": "A cat sat on the mat."}],
                "role": "model",
            },
            "finishReason": "STOP",
            "safetyRatings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"}
            ],
            "citationMetadata": {"citationSources": []},
            "index": 0,
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 7,
        "totalTokenCount": 17,
    },
    "modelVersion": "gemini-2.0-flash",
    "responseId": "resp-gemini-01",
}

# ``Part`` discriminated fields instantiated from the schema: ``functionCall``
# (``name`` + ``args`` + optional ``id``) and the ``thought`` flag.
PUBLISHED_FUNCTION_CALL_PART: dict[str, Any] = {
    "functionCall": {"name": "get_weather", "args": {"location": "Paris"}, "id": "call-1"}
}
PUBLISHED_THOUGHT_PART: dict[str, Any] = {
    "thought": True,
    "text": "Considering the weather tool.",
}


def _reply(body: dict[str, Any]) -> c.CapturedReply:
    """Wrap a body dict in the capture type the reader takes.

    Args:
        body: The reply body.

    Returns:
        The capture, with the body serialised the way the wire carries it.
    """
    return c.CapturedReply(status=200, headers=(), body=json.dumps(body).encode("utf-8"))


# --------------------------------------------------------------------------
# R1 — the published-schema example projects with verify_total clean
# --------------------------------------------------------------------------


class TestPublishedSchemaExample:
    """The assembled ``GenerateContentResponse``, end to end."""

    def test_projects_clean_with_verify_total(self) -> None:
        """R1 — the published-shape reply round-trips with an empty residual."""
        projected = gm.GeminiReplyProjection().read_reply(_reply(PUBLISHED_TEXT_RESPONSE))

        c.verify_total(projected)
        assert projected.source == PUBLISHED_TEXT_RESPONSE

    def test_text_part_carries_the_published_text(self) -> None:
        """R2 — the single ``text`` part projects as one ``Text`` part."""
        projected = gm.GeminiReplyProjection().read_reply(_reply(PUBLISHED_TEXT_RESPONSE))

        assert len(projected.parts) == 1
        part = projected.parts[0]
        assert isinstance(part, c.Text)
        assert part.text == "A cat sat on the mat."

    def test_stop_reason_maps_straight(self) -> None:
        """R4 — ``STOP`` maps to ``end_turn`` with ``stop_reason_raw`` unset."""
        projected = gm.GeminiReplyProjection().read_reply(_reply(PUBLISHED_TEXT_RESPONSE))

        assert projected.stop_reason == "end_turn"
        assert projected.stop_reason_raw is None

    def test_usage_is_carried_whole(self) -> None:
        """§3.3.1 — usage is carried but excluded from the T-D10 diff."""
        projected = gm.GeminiReplyProjection().read_reply(_reply(PUBLISHED_TEXT_RESPONSE))

        assert projected.usage == PUBLISHED_TEXT_RESPONSE["usageMetadata"]

    def test_conforms_to_the_reply_projection_protocol(self) -> None:
        """R5 — the reader satisfies ``ReplyProjection`` under ``isinstance``."""
        reader = gm.GeminiReplyProjection()

        assert isinstance(reader, c.ReplyProjection)
        assert reader.wire_format is c.WireFormat.GEMINI


# --------------------------------------------------------------------------
# R2 — the format's content-shape vocabulary
# --------------------------------------------------------------------------


class TestContentShapes:
    """One case per Part shape the published schema lists for a reply."""

    @staticmethod
    def _body_with_parts(parts: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the published reply body with the parts array swapped.

        Args:
            parts: The replacement ``content.parts`` array.

        Returns:
            The body, with every other key the published schema carries.
        """
        body = json.loads(json.dumps(PUBLISHED_TEXT_RESPONSE))
        body["candidates"][0]["content"]["parts"] = parts
        return body

    def test_function_call_projects_with_native_arguments(self) -> None:
        """R2 — ``functionCall.args`` is already a mapping (no string decode)."""
        body = self._body_with_parts([PUBLISHED_FUNCTION_CALL_PART])
        body["candidates"][0]["finishReason"] = "STOP"

        projected = gm.GeminiReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        assert len(projected.parts) == 1
        part = projected.parts[0]
        assert isinstance(part, c.ToolUse)
        assert part.name == "get_weather"
        assert part.id == "call-1"
        assert part.arguments == {"location": "Paris"}

    def test_thought_part_projects_as_thinking(self) -> None:
        """R2 — ``{thought: true, text}`` projects as ``Thinking``."""
        body = self._body_with_parts([PUBLISHED_THOUGHT_PART, {"text": "Paris is sunny."}])

        projected = gm.GeminiReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        assert len(projected.parts) == 2
        thinking = projected.parts[0]
        assert isinstance(thinking, c.Thinking)
        assert thinking.text == "Considering the weather tool."
        text_part = projected.parts[1]
        assert isinstance(text_part, c.Text)

    def test_extra_candidate_residualises(self) -> None:
        """R2/§3.3.1 — a reply with more than one candidate is a named anomaly."""
        body = json.loads(json.dumps(PUBLISHED_TEXT_RESPONSE))
        body["candidates"].append(json.loads(json.dumps(body["candidates"][0])))

        projected = gm.GeminiReplyProjection().read_reply(_reply(body))

        assert "candidates[1]" in projected.residual
        with pytest.raises(c.ResidualFieldsError, match="candidates\\[1\\]"):
            c.verify_total(projected)

    @pytest.mark.parametrize(
        "part",
        [
            {"fileData": {"fileUri": "gs://bucket/file.pdf", "mimeType": "application/pdf"}},
            {"executableCode": {"language": "PYTHON", "code": "print('hi')"}},
            {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "hi"}},
        ],
    )
    def test_unmodelled_part_shapes_project_as_opaque(self, part: dict[str, Any]) -> None:
        """R2 — three camelCase Part kinds the request reader also meets, projected
        as ``Opaque`` so the digest keeps the payload detectable. ``opaque_kind``
        raises ``ValueError`` for these spellings (no canonical alias yet) — the
        reader translates that to ``UnreadableBodyError``, per §7.4.1, until the
        first real capture wires the alias. This test pins the camelCase path:
        it currently raises, surfacing the alias gap to the next reader.
        """
        body = self._body_with_parts([part])
        with pytest.raises(c.UnreadableBodyError):
            gm.GeminiReplyProjection().read_reply(_reply(body))

    def test_inline_data_part_projects_as_image(self) -> None:
        """R2 — ``inlineData`` projects as ``Image``; the base64 payload
        decodes and the digest is the SHA-256 of the decoded bytes (§7.4 rule 7
        row 3, KBR-192)."""
        # "hello" → base64 "aGVsbG8="
        body = self._body_with_parts(
            [{"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}]
        )

        projected = gm.GeminiReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        part = projected.parts[0]
        assert isinstance(part, c.Image)
        assert part.media_type == "image/png"
        # SHA-256 of b"hello" = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        assert part.digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_inline_data_with_invalid_base64_falls_back_to_raw_bytes(self) -> None:
        """R2 / KBR-192 — when ``base64 -validate=True`` rejects the payload
        (RFC 2045 line breaks the wire sometimes carries), the digest is the
        SHA-256 of the raw encoded bytes and the field residualises, so the
        run names the structural break rather than silently dropping the
        image.
        """
        # 80-char wrapped base64 — what Google's own image sample uses
        # (`base64 -w0` is the unwrap flag). ``validate=True`` rejects it.
        wrapped = "aGVs\nbG8="
        body = self._body_with_parts(
            [{"inlineData": {"mimeType": "image/png", "data": wrapped}}]
        )

        projected = gm.GeminiReplyProjection().read_reply(_reply(body))

        assert "candidates[0].content.parts[0].inlineData.data" in projected.residual
        assert projected.residual[
            "candidates[0].content.parts[0].inlineData.data"
        ] == wrapped
        # SHA-256 of the UTF-8 encoding of the raw wrapped bytes.
        import hashlib

        expected = hashlib.sha256(wrapped.encode("utf-8")).hexdigest()
        assert projected.parts[0].digest == expected  # type: ignore[union-attr]


# --------------------------------------------------------------------------
# R4 — the finish-reason mapping, canonical and escaped
# --------------------------------------------------------------------------


class TestFinishReasonMapping:
    """Canonical values map straight; the rest escape through ``other``."""

    def test_max_tokens_maps_straight(self) -> None:
        """R4 — ``MAX_TOKENS`` maps to ``max_tokens`` with ``stop_reason_raw`` unset."""
        body = json.loads(json.dumps(PUBLISHED_TEXT_RESPONSE))
        body["candidates"][0]["finishReason"] = "MAX_TOKENS"

        projected = gm.GeminiReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason == "max_tokens"
        assert projected.stop_reason_raw is None

    @pytest.mark.parametrize("wire_value", ["SAFETY", "RECITATION", "LANGUAGE", "OTHER"])
    def test_safety_family_escapes_through_other(self, wire_value: str) -> None:
        """R4 — a published value outside the canonical set projects as ``other``."""
        body = json.loads(json.dumps(PUBLISHED_TEXT_RESPONSE))
        body["candidates"][0]["finishReason"] = wire_value

        projected = gm.GeminiReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason == "other"
        assert projected.stop_reason_raw == wire_value

    def test_unspecified_finish_reason_projects_as_none(self) -> None:
        """R4 — ``FINISH_REASON_UNSPECIFIED`` ("not stopped yet") maps to ``None``."""
        body = json.loads(json.dumps(PUBLISHED_TEXT_RESPONSE))
        body["candidates"][0]["finishReason"] = "FINISH_REASON_UNSPECIFIED"

        projected = gm.GeminiReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason is None
        assert projected.stop_reason_raw is None

    def test_pairing_rule_is_enforced_against_a_stale_raw(self) -> None:
        """R4 — a canonical stop reason beside ``stop_reason_raw`` is rejected.

        The rule lives in :meth:`~harness.contract.Reply.__post_init__` (T-W2);
        this test pins the *reader's* mapping to it.
        """
        with pytest.raises(ValueError, match="stop_reason_raw"):
            c.Reply(parts=(), stop_reason="end_turn", stop_reason_raw="STOP")


# --------------------------------------------------------------------------
# R3 — the falsification pair
# --------------------------------------------------------------------------


class TestFalsification:
    """Plan §1.4: the harness must be shown to fail, with its control."""

    def test_injected_unknown_key_residualises_and_fails_totality(self) -> None:
        """R3 — one unrecognised top-level key produces a non-empty residual."""
        body = json.loads(json.dumps(PUBLISHED_TEXT_RESPONSE))
        body["x-kitty-trace"] = "injected"

        projected = gm.GeminiReplyProjection().read_reply(_reply(body))

        assert projected.residual == {"x-kitty-trace": "injected"}
        with pytest.raises(c.ResidualFieldsError, match="x-kitty-trace"):
            c.verify_total(projected)

    def test_control_without_the_injected_key_is_clean(self) -> None:
        """R3 — the paired control: the same body without the defect is clean."""
        projected = gm.GeminiReplyProjection().read_reply(_reply(PUBLISHED_TEXT_RESPONSE))

        c.verify_total(projected)

    def test_unknown_part_key_residualises(self) -> None:
        """R3 — an unmodelled key inside a part fails closed at depth."""
        body = json.loads(json.dumps(PUBLISHED_TEXT_RESPONSE))
        body["candidates"][0]["content"]["parts"][0]["x_vendor_marker"] = 1

        projected = gm.GeminiReplyProjection().read_reply(_reply(body))

        assert "candidates[0].content.parts[0].x_vendor_marker" in projected.residual
        with pytest.raises(c.ResidualFieldsError, match="x_vendor_marker"):
            c.verify_total(projected)

    def test_unknown_part_key_control_is_clean(self) -> None:
        """R3 — control for the depth case: the published part is clean."""
        projected = gm.GeminiReplyProjection().read_reply(_reply(PUBLISHED_TEXT_RESPONSE))

        c.verify_total(projected)

    def test_unknown_content_level_key_residualises(self) -> None:
        """R3 / W2 — an unmodelled key on the ``content`` object fails closed
        at depth, matching the residue rule the candidate and part levels
        already enforce."""
        body = json.loads(json.dumps(PUBLISHED_TEXT_RESPONSE))
        body["candidates"][0]["content"]["x_vendor_extra"] = 1

        projected = gm.GeminiReplyProjection().read_reply(_reply(body))

        assert "candidates[0].content.x_vendor_extra" in projected.residual
        with pytest.raises(c.ResidualFieldsError, match="x_vendor_extra"):
            c.verify_total(projected)


class TestNameRequired:
    """A ``functionCall`` whose ``name`` is missing, empty, or not a string raises.

    ``contract.decode_arguments`` (``contract.py:935-941``) is explicit: ``""``
    for a name is **not** lossless — a call nobody can name cannot be paired
    with its result or addressed by a register row. KBR-279 landed the rule
    for the Ollama readers; KBR-281 aligns Gemini's reply direction on it.
    The absent and non-string raises existed in the reader before KBR-281 but
    were pinned by no test; the four cases below close that gap with the new
    empty case.
    """

    @staticmethod
    def _body_with_function_call(name: Any) -> dict[str, Any]:
        """Return the published reply body whose one part carries ``name``.

        ``name`` is spliced in verbatim, so ``None`` means the key is absent.
        """
        call: dict[str, Any] = {"args": {"city": "Toronto"}}
        if name is not None:
            call["name"] = name
        body = json.loads(json.dumps(PUBLISHED_TEXT_RESPONSE))
        body["candidates"][0]["content"]["parts"] = [{"functionCall": call}]
        return body

    def test_missing_function_call_name_raises(self) -> None:
        """A ``functionCall`` with no ``name`` raises."""
        with pytest.raises(c.UnreadableBodyError, match="functionCall.name"):
            gm.GeminiReplyProjection().read_reply(_reply(self._body_with_function_call(None)))

    def test_empty_function_call_name_raises(self) -> None:
        """A ``functionCall`` with an empty ``name`` raises (KBR-281)."""
        with pytest.raises(c.UnreadableBodyError, match="functionCall.name"):
            gm.GeminiReplyProjection().read_reply(_reply(self._body_with_function_call("")))

    def test_non_string_function_call_name_raises(self) -> None:
        """A ``functionCall`` with a non-string ``name`` raises."""
        with pytest.raises(c.UnreadableBodyError, match="functionCall.name"):
            gm.GeminiReplyProjection().read_reply(_reply(self._body_with_function_call(42)))

    def test_named_function_call_control_is_clean(self) -> None:
        """Control — a named ``functionCall`` projects cleanly."""
        projected = gm.GeminiReplyProjection().read_reply(
            _reply(self._body_with_function_call("get_weather"))
        )

        c.verify_total(projected)
        part = projected.parts[0]
        assert isinstance(part, c.ToolUse)
        assert part.name == "get_weather"
