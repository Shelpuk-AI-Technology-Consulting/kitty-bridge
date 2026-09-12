"""L1 tests for the Gemini projection.

`.system_design/TEST_SUITE.md` §3.3.1, §3.3.1a, §3.3.1b, §3.3.5, §7.4.1 · plan
task **T-A4** (KBR-36).

**Every fixture is a published example or is assembled from the published
schema.**  :class:`TestPublishedExamples` carries the ten request bodies the
Gemini API reference publishes as ``curl`` samples at
``https://ai.google.dev/api/generate-content`` (retrieved 2026-09-12), with two
stated deviations and no others: two of them carry a trailing comma and are
therefore not JSON, and file URIs are replaced with ``files/example`` so the
suite names no host it does not control.  Everything else is assembled from the
discovery document at revision
:data:`~harness.reader_gemini.SCHEMA_VERSION`.  None is copied from kitty's
output — §3.3.1's independent-oracle rule makes a reader validated against
kitty's output circular.

**This reader consumes the URL**, which no other does, so the route cases are
first-class here: :class:`TestRoute` owns §3.3.5's claim that two requests with
byte-identical bodies are told apart by their paths.

**The falsification set is required, not decorative.**  Plan §1.4:
:class:`TestFalsification` holds five deliberate defects the reader must detect,
running in the suite.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from harness import contract as c
from harness import reader_gemini as r

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

#: The published endpoint, as every example spells it.
_HOST = "generativelanguage.googleapis.com"

#: A model name the examples use, so a route assertion names something real.
_MODEL = "gemini-2.0-flash"


def captured(
    body: Any,
    *,
    model: str = _MODEL,
    operation: str = "generateContent",
    query: str = "",
    path: str | None = None,
) -> c.CapturedRequest:
    """Wrap a body as a capture aimed at the published Gemini endpoint.

    Args:
        body: A mapping to encode as JSON, or raw bytes to pass through
            undecoded for the failure-shape tests.
        model: The model segment of the path.
        operation: The operation suffix, ``generateContent`` or
            ``streamGenerateContent``.
        query: The raw query string.
        path: A complete path, overriding ``model`` and ``operation`` for the
            tests that hand the reader a path it must reject.

    Returns:
        A capture the reader can be handed directly.
    """
    raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")

    return c.CapturedRequest(
        method="POST",
        scheme="https",
        host=_HOST,
        path=path if path is not None else f"/v1beta/models/{model}:{operation}",
        query=query,
        headers=(("content-type", "application/json"),),
        body=raw,
    )


def project(body: Any, **route: Any) -> c.Request:
    """Project a body and assert it accounted for itself.

    Running :func:`harness.contract.verify_total` here rather than in each test
    means no test can pass while quietly leaving a key unaccounted for.

    Args:
        body: The request body.
        **route: Route overrides, forwarded to :func:`captured`.

    Returns:
        The projection.
    """
    projected = r.GeminiProjection().read_request(captured(body, **route))
    c.verify_total(projected)

    return projected


def project_untotalled(body: Any, **route: Any) -> c.Request:
    """Project a body without asserting totality.

    Args:
        body: The request body.
        **route: Route overrides, forwarded to :func:`captured`.

    Returns:
        The projection, residual and all — which is what the residual tests
        assert on and what :func:`project` would refuse to return.
    """
    return r.GeminiProjection().read_request(captured(body, **route))


# --------------------------------------------------------------------------
# R0 — protocol conformance
# --------------------------------------------------------------------------


class TestProtocolConformance:
    """The reader is a `Projection` for the Gemini format."""

    def test_it_declares_the_gemini_wire_format(self) -> None:
        """A `str` would let six authors spell one format three ways (§7.4)."""
        assert r.GeminiProjection.wire_format is c.WireFormat.GEMINI

    def test_it_satisfies_the_projection_protocol(self) -> None:
        """`isinstance`, never `issubclass` — the protocol carries a data member."""
        assert isinstance(r.GeminiProjection(), c.Projection)

    def test_it_actually_projects_rather_than_merely_having_the_members(self) -> None:
        """The paired assertion `isinstance` cannot make.

        `contract.Projection` checks member *presence* only, never signatures,
        so the test above passes against any object with two attributes.
        """
        projected = project({"contents": [{"parts": [{"text": "hello"}]}]})

        assert projected.envelope.model == _MODEL
        assert projected.conversation.turns == (c.Turn("user", [c.Text("hello")]),)

    def test_it_imports_nothing_from_kitty(self) -> None:
        """§3.3.1: a reader written in terms of the code under test proves nothing.

        Asserted structurally rather than by convention, because the import that
        would break it is one line and reads as harmless.
        """
        source = Path(r.__file__).read_text(encoding="utf-8")
        offenders = re.findall(r"^\s*(?:from|import)\s+kitty\b.*$", source, flags=re.MULTILINE)

        assert offenders == [], f"the reader must not import from src/kitty: {offenders}"


# --------------------------------------------------------------------------
# R1 — the route is an input
# --------------------------------------------------------------------------


class TestRoute:
    """§3.3.5: Gemini carries the model and the operation in the URL.

    This is the claim no body-only oracle can make, and the reason
    `Projection.read_request` takes a whole capture rather than a body.
    """

    def test_the_model_comes_from_the_path(self) -> None:
        """R1.1 — the inbound path is where the model actually is."""
        projected = project({"contents": []}, model="gemini-2.5-pro")

        assert projected.envelope.model == "gemini-2.5-pro"

    def test_a_models_prefix_on_the_path_segment_is_stripped(self) -> None:
        """R1.1 — the published path form is `/{version}/models/{model}:{method}`.

        A profile naming `models/gemini-2.5-pro` and one naming
        `gemini-2.5-pro` reach the same destination, so they must project the
        same model or M1's register row reports a delta on the spelling.
        """
        projected = project({"contents": []}, path="/v1beta/models/models/x:generateContent")

        assert projected.envelope.model == "x"

    @pytest.mark.parametrize(
        ("operation", "expected"),
        [("generateContent", False), ("streamGenerateContent", True)],
    )
    def test_stream_comes_from_the_operation_suffix(self, operation: str, expected: bool) -> None:
        """R1.2 — M11 forces `stream` false, so it must be projectable."""
        projected = project({"contents": []}, operation=operation)

        assert projected.envelope.stream is expected

    @pytest.mark.parametrize(
        ("operation", "query", "expected"),
        [
            ("streamGenerateContent", "", True),
            ("streamGenerateContent", "alt=sse&key=SECRET", True),
            ("generateContent", "alt=sse", False),
        ],
    )
    def test_stream_is_the_operation_and_never_alt_sse(self, operation: str, query: str, expected: bool) -> None:
        """R1.3 — `alt` picks the framing of a method that streams either way.

        Reading `stream` off `alt=sse` would make a streaming request with the
        JSON-array framing project as non-streaming, and P17's register row
        anchored at `envelope.stream` would then claim a delta nobody caused.
        """
        projected = project({"contents": []}, operation=operation, query=query)

        assert projected.envelope.stream is expected

    @pytest.mark.parametrize(
        "path",
        [
            "/v1beta/models/x:countTokens",
            "/v1beta/models/x:embedContent",
            "/v1beta/models/x",
            "/v1beta/chat",
            "/v1beta/tunedModels/x:generateContent",
            "",
        ],
    )
    def test_a_path_that_is_not_a_published_generate_route_is_unreadable(self, path: str) -> None:
        """R1.4 — structural: there is no partial projection without a model.

        `countTokens` and `embedContent` are published methods on the same
        resource, so "any method" would silently project a token count as a
        conversation.
        """
        with pytest.raises(c.UnreadableBodyError):
            project_untotalled({"contents": []}, path=path)

    def test_the_query_is_left_entirely_to_the_route_assertion(self) -> None:
        """R1.5 — a boundary, stated so a green run is not over-read.

        `verify_total` compares `consumed | residual` against the **body**, so a
        query key in either account is reported as a claim on a key the body
        does not have. Asserting the route is T-D2's (KBR-52).
        """
        projected = project_untotalled({"contents": []}, operation="streamGenerateContent", query="alt=sse&key=SECRET")

        assert "alt" not in projected.consumed and "alt" not in projected.residual
        assert "key" not in projected.consumed and "key" not in projected.residual


# --------------------------------------------------------------------------
# R7 — falsification (plan §1.4)
# --------------------------------------------------------------------------


class TestFalsification:
    """Five deliberate defects the reader must detect, running in the suite.

    Plan §1.4: "The first working version of every harness ships with at least
    one falsification case — a deliberate defect it must detect, running in the
    suite." Four review rounds on the design produced four harnesses that would
    have passed while proving nothing, and one of them was "a projection that
    could not see the model name, in a product whose purpose is changing the
    model name" — which is exactly what R7.4 pins here.
    """

    def test_an_unrecognised_top_level_key_produces_a_residual(self) -> None:
        """R7.1 — §3.3.1's fifth mandatory oracle falsification case."""
        projected = project_untotalled({"contents": [], "x-kitty-trace": "abc"})

        assert projected.residual == {"x-kitty-trace": "abc"}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_an_unrecognised_key_inside_a_part_produces_a_residual_at_its_path(self) -> None:
        """R7.2 — §7.4.1: failing closed is not a top-level rule.

        `verify_total` cannot see past the top level, so this is the rule that
        closes it, and this test is what holds the rule.
        """
        projected = project_untotalled({"contents": [{"parts": [{"text": "hi", "x-kitty-trace": "abc"}]}]})

        assert projected.residual == {"contents[0].parts[0].x-kitty-trace": "abc"}

    def test_consumed_claims_only_what_the_reader_actually_mapped(self) -> None:
        """A reader that over-claims defeats the check `consumed` exists to be.

        `verify_total` still fails such a body — the residual is non-empty — so
        the falsification case above passes either way and cannot see this.
        But `consumed` would be *wrong data*: it is what makes a **dropped** key
        detectable, and T-D8 diffs these key sets across all six readers.
        Mutation testing found this one; no behavioural test could.
        """
        projected = project_untotalled({"contents": [], "x-kitty-trace": "abc"})

        assert set(projected.consumed) == {"contents"}
        assert "x-kitty-trace" not in projected.consumed

    def test_a_dropping_reader_is_caught_by_totality_not_by_the_residual(self) -> None:
        """R7.3 — the case a "residual must be empty" rule cannot see.

        A reader that *drops* an unknown key leaves the residual empty and
        sails through. `Request.consumed` exists for this, and this is the
        deliberate defect that proves it does its job.
        """
        body = {"contents": [], "cachedContent": "cachedContents/abc"}
        honest = project(body)
        assert honest.envelope.extra["cachedContent"] == "cachedContents/abc"

        # The defect: the same projection with the key claimed by neither account.
        dropping = c.Request(
            envelope=honest.envelope,
            conversation=honest.conversation,
            residual={},
            consumed=frozenset({"contents"}),
            source=honest.source,
        )

        with pytest.raises(c.DroppedFieldsError):
            c.verify_total(dropping)

    def test_two_identical_bodies_to_two_models_are_told_apart(self) -> None:
        """R7.4 — §3.3.5's headline claim, for the format that needs it most.

        The bodies are asserted byte-identical first: without that the test
        would pass against a reader that read the model out of the body and
        never looked at the path at all.
        """
        body = {"contents": [{"parts": [{"text": "hi"}]}]}
        left = captured(body, model="gemini-2.0-flash")
        right = captured(body, model="gemini-2.5-pro")
        assert left.body == right.body

        reader = r.GeminiProjection()
        assert reader.read_request(left).envelope.model == "gemini-2.0-flash"
        assert reader.read_request(right).envelope.model == "gemini-2.5-pro"

    def test_two_identical_bodies_on_two_operations_are_told_apart(self) -> None:
        """R7.5 — M11 forces `stream` false, and the body cannot show it."""
        body = {"contents": [{"parts": [{"text": "hi"}]}]}
        left = captured(body, operation="generateContent")
        right = captured(body, operation="streamGenerateContent")
        assert left.body == right.body

        reader = r.GeminiProjection()
        assert reader.read_request(left).envelope.stream is False
        assert reader.read_request(right).envelope.stream is True


# --------------------------------------------------------------------------
# R3 — ProtoJSON's two spellings
# --------------------------------------------------------------------------


class TestBothSpellings:
    """Gemini's JSON is ProtoJSON, and every field has two legal names.

    "Parsers accept both the lowerCamelCase name … and the original proto field
    name" (``protobuf.dev/programming-guides/json/``), and Google's own
    published examples mix the two — ``system_instruction`` and
    ``function_declarations`` beside ``maxOutputTokens`` and ``stopSequences``.
    A reader that knew one spelling would residualise the other and fail the run
    on Google's own sample.
    """

    @pytest.mark.parametrize(
        ("body", "check"),
        [
            (
                {"system_instruction": {"parts": [{"text": "be brief"}]}},
                lambda p: p.conversation.system == (c.Text("be brief"),),
            ),
            (
                {"generation_config": {"max_output_tokens": 8}},
                lambda p: p.conversation.sampling == {"max_tokens": 8},
            ),
            (
                {"cached_content": "cachedContents/a"},
                lambda p: p.envelope.extra["cachedContent"] == "cachedContents/a",
            ),
            (
                {"contents": [{"parts": [{"file_data": {"file_uri": "files/example"}}]}]},
                lambda p: p.conversation.turns[0].parts[0] == c.Image(ref="files/example"),
            ),
            (
                {"tools": [{"function_declarations": [{"name": "f"}]}]},
                lambda p: p.conversation.tools == (c.ToolDecl("f"),),
            ),
            (
                {"tool_config": {"function_calling_config": {"mode": "ANY"}}},
                lambda p: p.envelope.extra["tool_choice"] == "any",
            ),
        ],
    )
    def test_the_snake_case_original_reads_as_the_published_name(self, body: Any, check: Any) -> None:
        """R3.1 — at the top level and at every depth beneath it."""
        assert check(project(body))

    @pytest.mark.parametrize("published_first", [True, False])
    def test_two_spellings_of_one_field_resolve_the_same_way_either_order(self, published_first: bool) -> None:
        """R3.2 — a body cannot mean two things, and order must not decide which.

        The published spelling wins wherever it appears. Resolving by *position*
        instead would make the same semantic body project two ways depending on
        which alias a serialiser emitted first — and §7.4.1 designs key order
        out of the projection elsewhere for exactly this reason ("canonical JSON
        rather than the raw wire slice, because a translator that reorders keys
        must not change the digest").

        The loser residualises rather than vanishing: a silent drop is the shape
        `consumed` exists to catch.
        """
        camel = {"parts": [{"text": "published"}]}
        snake = {"parts": [{"text": "alias"}]}
        body = (
            {"systemInstruction": camel, "system_instruction": snake}
            if published_first
            else {"system_instruction": snake, "systemInstruction": camel}
        )

        projected = project_untotalled(body)

        assert projected.conversation.system == (c.Text("published"),)
        assert projected.residual == {"system_instruction": snake}

    def test_a_published_key_that_already_begins_with_an_underscore_resolves_to_itself(self) -> None:
        """R3.3 — `_responseJsonSchema` is a published key, not a snake_case original.

        Converting it would produce `ResponseJsonSchema`, which matches nothing,
        so the field would residualise and fail the run on a legitimate request.
        """
        projected = project({"generationConfig": {"_responseJsonSchema": {"type": "object"}}})

        assert projected.envelope.extra["_responseJsonSchema"] == {"type": "object"}

    def test_an_unrecognised_key_is_not_rescued_by_conversion(self) -> None:
        """A snake_case key naming nothing published still residualises."""
        projected = project_untotalled({"kitty_trace": "abc"})

        assert projected.residual == {"kitty_trace": "abc"}


# --------------------------------------------------------------------------
# R4 — the envelope
# --------------------------------------------------------------------------


class TestEnvelope:
    """Control fields land on the envelope, keyed by the published wire key."""

    def test_store_is_a_named_field(self) -> None:
        """R4.1 — P17 injects `store`, so it needs its own anchor."""
        assert project({"store": False}).envelope.store is False

    def test_an_absent_control_field_is_absent_rather_than_none(self) -> None:
        """P17 injects `store`, so its absence must be observable (§3.3.2 assertion 2)."""
        projected = project({"contents": []})

        assert projected.envelope.store is None
        assert projected.envelope.extra == {}

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("serviceTier", "SERVICE_TIER_PRIORITY"),
            ("cachedContent", "cachedContents/abc"),
            ("safetySettings", [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"}]),
        ],
    )
    def test_a_top_level_control_field_keeps_its_published_key(self, key: str, value: Any) -> None:
        """R4.2 — a register row addresses `envelope.extra[<key>]` by that name."""
        assert project({key: value}).envelope.extra[key] == value

    def test_a_nested_control_field_is_addressed_by_its_leaf_key(self) -> None:
        """R4.3/R4.4 — `extra_path()` raises on a dotted key.

        §3.3.1a: "`envelope.extra` is diffed one wire key at a time … the
        dotted-bracket form exists for `residual` alone". Gemini is the first
        format whose control fields nest, so the leaf is the address.
        """
        projected = project(
            {
                "generationConfig": {"responseSchema": {"type": "ARRAY"}},
                "toolConfig": {"retrievalConfig": {"languageCode": "en"}},
            }
        )

        assert projected.envelope.extra["responseSchema"] == {"type": "ARRAY"}
        assert projected.envelope.extra["retrievalConfig"] == {"languageCode": "en"}

    def test_every_extra_key_this_reader_can_emit_is_addressable(self) -> None:
        """R4.7 — flattening two nested objects into one namespace must not collide.

        Asserted over the tables rather than over one body, because a collision
        would be introduced by a *schema revision*, not by a request: the loser
        would silently overwrite the winner with no residual and no delta.
        """
        emitted = [
            r._TOP_LEVEL_EXTRA_KEYS,
            r._GENERATION_EXTRA_KEYS,
            r._TOOL_CONFIG_EXTRA_KEYS,
            r._BUILT_IN_TOOL_KEYS,
            frozenset({c.TOOL_CHOICE_KEY}),
        ]
        flattened = [key for group in emitted for key in group]

        assert len(flattened) == len(set(flattened)), "two extra sources name one key"

        # And every one of them is a legal `envelope.extra` address, which is
        # what makes the flattening safe rather than merely collision-free.
        for key in flattened:
            assert c.extra_path(key) == f"envelope.extra[{key}]"

    @pytest.mark.parametrize("value", [f"models/{_MODEL}", "models/other"])
    def test_a_body_level_model_residualises_whether_or_not_it_agrees(self, value: str) -> None:
        """R4.6 — the REST request body has no `model` field.

        The endpoint is `/v1beta/{model=models/*}:generateContent` and the
        reference's own "Request body" table lists nine fields without it; the
        discovery document carries one only because that is the proto message.
        So a body `model` is an unrecognised key.

        **Both cases behave the same, deliberately.** An earlier draft
        residualised only on disagreement, which put an *assertion* inside a
        reader: §3.3.1a makes `residual` an illegal register anchor, so that
        comparison could only ever kill the run, and finding disagreements is
        §3.3.2's job over a route T-D2 owns.
        """
        projected = project_untotalled({"model": value}, model=_MODEL)

        assert projected.envelope.model == _MODEL
        assert projected.residual == {"model": value}


class TestSampling:
    """`generationConfig` splits into the closed canonical set and the rest."""

    def test_every_member_with_a_canonical_spelling(self) -> None:
        """R5.1 — §3.3.1b's set is closed and `Conversation` enforces it."""
        projected = project(
            {
                "generationConfig": {
                    "temperature": 1.0,
                    "topP": 0.8,
                    "topK": 10,
                    "maxOutputTokens": 800,
                    "presencePenalty": 0.1,
                    "frequencyPenalty": 0.2,
                    "seed": 7,
                    "candidateCount": 2,
                    "stopSequences": ["Title"],
                    "responseLogprobs": True,
                    "logprobs": 5,
                }
            }
        )

        assert projected.conversation.sampling == {
            "temperature": 1.0,
            "top_p": 0.8,
            "top_k": 10,
            "max_tokens": 800,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "seed": 7,
            "n": 2,
            "stop": ["Title"],
            "logprobs": True,
            "top_logprobs": 5,
        }

    def test_the_logprobs_name_collision_is_not_carried_across(self) -> None:
        """R5.2 — the trap this format sets, and the one a careful reader walks into.

        Gemini's `logprobs` is the *count* Chat Completions calls
        `top_logprobs`; its `responseLogprobs` is the *boolean* Chat Completions
        calls `logprobs`. Carrying `logprobs` across unchanged types-checks
        clean and shows a delta on every cross-format comparison that asks for
        them.
        """
        sampling = project({"generationConfig": {"logprobs": 5}}).conversation.sampling

        assert sampling == {"top_logprobs": 5}
        assert "logprobs" not in sampling

    def test_the_response_format_family_is_control_not_sampling(self) -> None:
        """R5.3 — none of the five folds onto the Chat Completions union.

        `responseFormat` is per-modality output configuration and the other four
        are schema constraints; folding any onto `response_format` would put a
        claim into the projection that the wire does not make.
        """
        projected = project(
            {
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": {"type": "ARRAY"},
                    "responseJsonSchema": {"type": "array"},
                    "_responseJsonSchema": {"type": "array"},
                    "responseFormat": {"text": {}},
                }
            }
        )

        assert "response_format" not in projected.conversation.sampling
        assert set(projected.envelope.extra) == {
            "responseMimeType",
            "responseSchema",
            "responseJsonSchema",
            "_responseJsonSchema",
            "responseFormat",
        }

    def test_a_true_is_not_read_as_an_integer(self) -> None:
        """`bool` is a subclass of `int`, so an unguarded isinstance carries it.

        `topK: true` would arrive as the integer 1 — a value the agent never
        sent, and the coercion §7.4.1 forbids.
        """
        projected = project_untotalled({"generationConfig": {"topK": True}})

        assert projected.conversation.sampling == {}
        assert projected.residual == {"generationConfig.topK": True}

    def test_an_integer_is_accepted_where_the_schema_publishes_a_float(self) -> None:
        """`"temperature": 1` is a legal JSON number for a float field."""
        assert project({"generationConfig": {"temperature": 1}}).conversation.sampling == {"temperature": 1}

    def test_an_unrecognised_generation_config_member_residualises_under_its_path(self) -> None:
        """§7.4.1: failing closed is not a top-level rule."""
        projected = project_untotalled({"generationConfig": {"kittyKnob": 1}})

        assert projected.residual == {"generationConfig.kittyKnob": 1}


class TestToolChoice:
    """`functionCallingConfig.mode` normalises onto the canonical vocabulary."""

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [("AUTO", "auto"), ("ANY", "any"), ("NONE", "none"), ("auto", "auto"), ("any", "any")],
    )
    def test_a_published_mode_with_a_canonical_value(self, mode: str, expected: str) -> None:
        """R4.5 — matched case-insensitively, because Google's own example is lower case."""
        projected = project({"toolConfig": {"functionCallingConfig": {"mode": mode}}})

        assert projected.envelope.extra["tool_choice"] == expected

    @pytest.mark.parametrize("mode", ["MODE_UNSPECIFIED", "VALIDATED"])
    def test_a_mode_with_no_canonical_value_residualises(self, mode: str) -> None:
        """R4.5 — `Envelope` enforces the closed set, so inventing a value raises.

        A `ValueError` out of a reader means the reader mis-routed a field, and
        T-D1 reads it as a reader bug rather than as an unreadable body — so a
        mode the vocabulary cannot carry must residualise instead.
        """
        projected = project_untotalled({"toolConfig": {"functionCallingConfig": {"mode": mode}}})

        assert "tool_choice" not in projected.envelope.extra
        assert projected.residual == {"toolConfig.functionCallingConfig.mode": mode}

    def test_any_restricted_to_one_function_is_that_tool(self) -> None:
        """R4.5 — "must call a function, and only this one" is `tool:<name>`."""
        projected = project({"toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["f"]}}})

        assert projected.envelope.extra["tool_choice"] == "tool:f"

    def test_any_restricted_to_several_keeps_the_mode_and_names_what_was_lost(self) -> None:
        """R4.5 — the restriction has no canonical form; the mode still projects."""
        projected = project_untotalled(
            {"toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["f", "g"]}}}
        )

        assert projected.envelope.extra["tool_choice"] == "any"
        assert projected.residual == {"toolConfig.functionCallingConfig.allowedFunctionNames": ["f", "g"]}


class TestTools:
    """Function declarations are content; built-in capabilities are control."""

    def test_a_function_declaration_projects_by_name(self) -> None:
        """R6.11 — §3.3.1a addresses tools by name, never by index."""
        projected = project(
            {
                "tools": [
                    {
                        "functionDeclarations": [
                            {
                                "name": "get_weather",
                                "description": "Look up the weather.",
                                "parameters": {"type": "object"},
                            }
                        ]
                    }
                ]
            }
        )

        assert projected.conversation.tools == (
            c.ToolDecl(
                name="get_weather",
                description="Look up the weather.",
                schema={"type": "object"},
                strict=None,
            ),
        )

    def test_strict_is_absent_rather_than_false(self) -> None:
        """P15's presence and absence must stay distinguishable."""
        projected = project({"tools": [{"functionDeclarations": [{"name": "f"}]}]})

        assert projected.conversation.tools[0].strict is None

    def test_parameters_json_schema_fills_the_same_slot(self) -> None:
        """R6.11 — the published mutually exclusive alternative to `parameters`."""
        projected = project(
            {"tools": [{"functionDeclarations": [{"name": "f", "parametersJsonSchema": {"type": "object"}}]}]}
        )

        assert projected.conversation.tools[0].schema == {"type": "object"}

    def test_a_declaration_carrying_both_schemas_keeps_parameters(self) -> None:
        """R6.11 — carrying both is outside the schema, so the loser residualises."""
        projected = project_untotalled(
            {
                "tools": [
                    {
                        "functionDeclarations": [
                            {
                                "name": "f",
                                "parameters": {"type": "object"},
                                "parametersJsonSchema": {"type": "array"},
                            }
                        ]
                    }
                ]
            }
        )

        assert projected.conversation.tools[0].schema == {"type": "object"}
        assert projected.residual == {"tools[0].functionDeclarations[0].parametersJsonSchema": {"type": "array"}}

    @pytest.mark.parametrize("key", ["behavior", "response", "responseJsonSchema"])
    def test_a_declaration_member_the_grammar_cannot_carry_residualises(self, key: str) -> None:
        """R6.10 — `ToolDecl` has four slots and the format publishes seven keys."""
        projected = project_untotalled({"tools": [{"functionDeclarations": [{"name": "f", key: "x"}]}]})

        assert projected.residual == {f"tools[0].functionDeclarations[0].{key}": "x"}

    def test_a_declaration_with_no_usable_name_residualises_rather_than_raising(self) -> None:
        """§3.3.1b settles this, and not the way the leaf rule usually goes.

        "An absent `name` *does* residualise … a call nobody can name cannot be
        paired or addressed." Absent and `null` residualise as well as a wrong
        type — but the declaration is still **projected**, because raising would
        blind the oracle to the rest of a request it could otherwise diff. T-A3
        takes the same branch; an earlier draft of this reader raised, which was
        a third answer to a question §3.3.1b had already settled.
        """
        projected = project_untotalled({"tools": [{"functionDeclarations": [{"description": "x"}]}]})

        assert projected.conversation.tools == (c.ToolDecl(name="", description="x"),)
        assert projected.residual == {"tools[0].functionDeclarations[0].name": None}

    @pytest.mark.parametrize(
        "key",
        [
            "googleSearch",
            "codeExecution",
            "urlContext",
            "fileSearch",
            "computerUse",
            "googleMaps",
            "googleSearchRetrieval",
            "mcpServers",
        ],
    )
    def test_a_built_in_capability_is_control_not_a_tool_declaration(self, key: str) -> None:
        """R6.12 — `ToolDecl` needs a name, and inventing one is a vendor spelling.

        Residualising instead would fail the run on every request that enables
        Google Search, which is not a defect signal but a legitimate control
        field the format publishes.
        """
        projected = project({"tools": [{key: {}}]})

        assert projected.conversation.tools == ()
        assert projected.envelope.extra[key] == {}

    def test_a_second_entry_redeclaring_a_capability_residualises(self) -> None:
        """R6.12 — `envelope.extra[<key>]` is one address, so the second has none."""
        projected = project_untotalled({"tools": [{"googleSearch": {}}, {"googleSearch": {"x": 1}}]})

        assert projected.envelope.extra["googleSearch"] == {}
        assert projected.residual == {"tools[1].googleSearch": {"x": 1}}


# --------------------------------------------------------------------------
# R6 — the conversation
# --------------------------------------------------------------------------

#: One pixel of PNG, so an image fixture carries real bytes rather than a
#: placeholder whose digest would be meaningless.
_PIXEL = base64.b64encode(bytes.fromhex("89504e470d0a1a0a")).decode("ascii")


class TestSystemInstruction:
    """`systemInstruction` lifts into `Conversation.system`, never into a turn."""

    def test_each_text_part_becomes_one_system_entry(self) -> None:
        """R6.1 — §3.3.1b: the system prompt is a top-level field in this format."""
        projected = project({"systemInstruction": {"parts": [{"text": "be brief"}, {"text": "be kind"}]}})

        assert projected.conversation.system == (c.Text("be brief"), c.Text("be kind"))
        assert projected.conversation.turns == ()

    def test_a_non_text_part_in_a_system_instruction_residualises(self) -> None:
        """The schema says system instructions are "text only"."""
        projected = project_untotalled({"systemInstruction": {"parts": [{"fileData": {"fileUri": "files/example"}}]}})

        assert projected.conversation.system == ()
        assert projected.residual == {"systemInstruction.parts[0].fileData": {"fileUri": "files/example"}}

    def test_a_role_on_a_system_instruction_residualises(self) -> None:
        """`Conversation.system` has no role slot, and the grammar cannot carry one."""
        projected = project_untotalled({"systemInstruction": {"role": "user", "parts": [{"text": "be brief"}]}})

        assert projected.conversation.system == (c.Text("be brief"),)
        assert projected.residual == {"systemInstruction.role": "user"}


class TestTurns:
    """`contents` becomes turns, with Gemini's `model` mapping to `assistant`."""

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ({"role": "user", "parts": [{"text": "hi"}]}, "user"),
            ({"role": "model", "parts": [{"text": "hi"}]}, "assistant"),
            ({"parts": [{"text": "hi"}]}, "user"),
            ({"role": "", "parts": [{"text": "hi"}]}, "user"),
        ],
    )
    def test_the_published_producers_map_onto_the_closed_role_set(self, content: Any, expected: str) -> None:
        """R6.2 — §3.3.1b: "Gemini's `model` maps to `assistant`".

        An absent or blank role is `user`: the schema says the field "can be
        left blank or unset", and request content with no stated producer is the
        client's. Reading it as anything else would split one turn in two and,
        because paths are index-based, report a delta on every turn after it.
        """
        projected = project({"contents": [content]})

        assert projected.conversation.turns[0].role == expected

    @pytest.mark.parametrize("role", ["system", "assistant", "developer", 7])
    def test_a_role_outside_the_published_pair_is_unreadable(self, role: Any) -> None:
        """R2.5 — structural: the turn cannot be built at all.

        `assistant` is in the list deliberately: it is the *projection's* role,
        not a wire value, and accepting it would let a reader validated against
        another reader's output pass.
        """
        with pytest.raises(c.UnreadableBodyError):
            project_untotalled({"contents": [{"role": role, "parts": []}]})

    def test_a_singleton_content_and_part_read_as_one_of_each(self) -> None:
        """R6.3 — Google's published `system_instruction.sh` sends both that way.

        The schema declares repeated fields; the published example does not send
        lists. Accepting only a list fails the run on Google's own sample.
        """
        projected = project(
            {
                "system_instruction": {"parts": {"text": "You are a cat. Your name is Neko."}},
                "contents": {"parts": {"text": "Hello there"}},
            }
        )

        assert projected.conversation.system == (c.Text("You are a cat. Your name is Neko."),)
        assert projected.conversation.turns == (c.Turn("user", [c.Text("Hello there")]),)


class TestParts:
    """Every published `Part` member, into the grammar or into the residual."""

    def test_text(self) -> None:
        """An empty part is a part with an empty string, never nothing (§3.3.1)."""
        projected = project({"contents": [{"parts": [{"text": ""}]}]})

        assert projected.conversation.turns[0].parts == (c.Text(""),)

    def test_a_thought_carries_its_signature(self) -> None:
        """R6.4 — `Thinking.signature` is what M8's carrier repair manipulates."""
        projected = project(
            {
                "contents": [
                    {
                        "role": "model",
                        "parts": [{"text": "hmm", "thought": True, "thoughtSignature": "sig"}],
                    }
                ]
            }
        )

        assert projected.conversation.turns[0].parts == (c.Thinking("hmm", signature="sig"),)

    def test_a_signature_on_a_part_that_is_not_a_thought_residualises(self) -> None:
        """R6.10 — `ToolUse` has no signature slot, so there is nowhere to put it.

        Gemini 2.5 and later attach a `thoughtSignature` to a `functionCall`
        part and clients echo it back, so this **will** fail the oracle run on
        real traffic — the same shape as KBR-167's block-level `cache_control`.
        That is the correct signal: the grammar cannot carry it, and a reader
        that dropped it would hide a field a translator could silently lose.
        """
        projected = project_untotalled(
            {
                "contents": [
                    {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "f"}, "thoughtSignature": "sig"}],
                    }
                ]
            }
        )

        assert projected.conversation.turns[0].parts == (c.ToolUse("f"),)
        assert projected.residual == {"contents[0].parts[0].thoughtSignature": "sig"}

    def test_inline_data_is_identified_by_the_digest_of_its_decoded_bytes(self) -> None:
        """R6.4 — §3.3.1: the media type is carried separately, not digested.

        `inlineData` is the format's only bytes carrier whatever the media type,
        so audio and PDF payloads project the same way and `media_type` records
        which it was.
        """
        projected = project({"contents": [{"parts": [{"inlineData": {"mimeType": "image/png", "data": _PIXEL}}]}]})

        assert projected.conversation.turns[0].parts == (
            c.Image(
                digest=c.image_digest(base64.b64decode(_PIXEL)),
                media_type="image/png",
            ),
        )

    def test_file_data_carries_a_reference_and_no_digest(self) -> None:
        """R6.4 — §3.3.1 names this case: "`fileData.fileUri` has no bytes"."""
        projected = project(
            {"contents": [{"parts": [{"fileData": {"mimeType": "audio/mpeg", "fileUri": "files/example"}}]}]}
        )

        assert projected.conversation.turns[0].parts == (
            c.Image(digest=None, media_type="audio/mpeg", ref="files/example"),
        )

    def test_a_function_call_carries_the_published_optional_id(self) -> None:
        """R6.4 — Gemini `v1beta` publishes `FunctionCall.id`, contrary to §3.3.1's note.

        The *decision* that note justifies — an optional id — is unchanged and
        still right, because Gemini's id is optional too. Only the stated reason
        was wrong, and KBR-36's comment asked for it to be confirmed rather than
        assumed.
        """
        projected = project(
            {
                "contents": [
                    {
                        "role": "model",
                        "parts": [{"functionCall": {"id": "fc-1", "name": "f", "args": {"q": "x"}}}],
                    }
                ]
            }
        )

        assert projected.conversation.turns[0].parts == (c.ToolUse(name="f", arguments={"q": "x"}, id="fc-1"),)

    def test_a_function_call_without_an_id_still_projects(self) -> None:
        """§3.3.1: pairing then falls back to name and position."""
        projected = project({"contents": [{"role": "model", "parts": [{"functionCall": {"name": "f"}}]}]})

        assert projected.conversation.turns[0].parts == (c.ToolUse(name="f", arguments={}),)

    def test_a_function_response_carries_its_struct_as_json(self) -> None:
        """R6.4 — §3.3.1: "Gemini's `functionResponse.response` is a bare struct"."""
        projected = project(
            {
                "contents": [
                    {
                        "parts": [
                            {
                                "functionResponse": {
                                    "id": "fc-1",
                                    "name": "f",
                                    "response": {"price": 259.75},
                                }
                            }
                        ]
                    }
                ]
            }
        )

        assert projected.conversation.turns[0].parts == (
            c.ToolResult(content=[c.Json({"price": 259.75})], tool_use_id="fc-1", is_error=False),
        )

    def test_a_function_response_media_part_follows_its_struct(self) -> None:
        """R6.6 — `FunctionResponsePart` carries inline media beside the struct."""
        projected = project(
            {
                "contents": [
                    {
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "f",
                                    "response": {"ok": True},
                                    "parts": [{"inlineData": {"mimeType": "image/png", "data": _PIXEL}}],
                                }
                            }
                        ]
                    }
                ]
            }
        )

        result = projected.conversation.turns[0].parts[0]
        assert isinstance(result, c.ToolResult)
        assert result.content == (
            c.Json({"ok": True}),
            c.Image(digest=c.image_digest(base64.b64decode(_PIXEL)), media_type="image/png"),
        )

    @pytest.mark.parametrize(
        ("key", "kind"),
        [
            ("executableCode", "executable_code"),
            ("codeExecutionResult", "code_execution_result"),
            ("toolCall", "tool_call"),
            ("toolResponse", "tool_response"),
        ],
    )
    def test_an_unmodelled_part_projects_as_opaque_with_a_digest(self, key: str, kind: str) -> None:
        """R6.4/R6.8 — all four name a concept no other format has.

        KBR-35's stated exception therefore applies and the wire name in
        snake_case is the canonical `kind`. **T-A5 is not covered by this**: it
        meets `document` and `searchResult`, which do have a second spelling.
        """
        projected = project({"contents": [{"parts": [{key: {"a": 1}}]}]})

        part = projected.conversation.turns[0].parts[0]
        assert isinstance(part, c.Opaque)
        assert part.kind == kind
        assert part.digest is not None

    def test_two_different_payloads_do_not_project_identically(self) -> None:
        """§3.3.1 put `digest` on `Opaque` to keep unmodelled content detectable.

        A bare `Opaque("executable_code")` would make a swapped program produce
        no delta at all.
        """
        left = project({"contents": [{"parts": [{"executableCode": {"code": "print(1)"}}]}]})
        right = project({"contents": [{"parts": [{"executableCode": {"code": "print(2)"}}]}]})

        assert left.conversation.turns[0].parts[0] != right.conversation.turns[0].parts[0]

    def test_the_opaque_digest_recipe_is_pinned_to_an_external_literal(self) -> None:
        """R6.9 — the only assertion that can see a changed recipe.

        Every other digest assertion compares two projections against each
        other, which cannot detect a recipe change that stays internally
        consistent: in T-A1 `ensure_ascii=False`, default `separators` and
        `sort_keys=False` all survived mutation testing until one digest was
        pinned this way. The payload is deliberately **non-ASCII and
        multi-key**, because that is the only shape on which all three wrong
        spellings differ from the right one.
        """
        payload = {"language": "PYTHON", "code": "print('héllo — wörld')"}
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()

        projected = project({"contents": [{"parts": [{"executableCode": payload}]}]})
        part = projected.conversation.turns[0].parts[0]

        assert isinstance(part, c.Opaque)
        assert part.digest == expected
        # Spelled out, so a reader of this test can see what is being pinned
        # rather than reading the recipe back out of the code under test.
        assert part.digest == "972e7573e3ff448274e80598557d57cf4b3fd9bdf77b862b02be2c73bb599cbd"

    @pytest.mark.parametrize(
        "key",
        [
            "partMetadata",
            "videoMetadata",
            "mediaResolution",
            "mediaProcessing",
            "audioTranscription",
        ],
    )
    def test_a_part_modifier_with_no_slot_residualises(self, key: str) -> None:
        """R6.10 — §7.4.1: a key the mapping does not consume residualises."""
        projected = project_untotalled({"contents": [{"parts": [{"text": "hi", key: {"a": 1}}]}]})

        assert projected.residual == {f"contents[0].parts[0].{key}": {"a": 1}}

    @pytest.mark.parametrize(
        ("part", "case"),
        [
            ({}, "an empty part"),
            ({"thought": False}, "only the thought flag, which is not a data member"),
            ({"thoughtSignature": "s"}, "only a signature, which modifies a member it lacks"),
            ({"videoMetadata": {"fps": 1.0}}, "only a modifier"),
        ],
    )
    def test_a_part_carrying_no_published_member_is_unreadable(self, part: Any, case: str) -> None:
        """R2.5 — §7.4.1's "a content block with no type": nothing to salvage.

        `thought` is the one that surprises. It is a part-level *flag*, so
        `{"thought": true}` alone is a thought part and projects `Thinking("")`
        — Gemini returns signature-only thoughts — while `{"thought": false}`
        alone says only that the part the wire forgot to send was not a thought.
        Dispatching on "is `thought` present" rather than "is it true" would
        project that as `Text("")`, inventing a part out of a flag.
        """
        with pytest.raises(c.UnreadableBodyError):
            project_untotalled({"contents": [{"parts": [part]}]})


class TestMergePipeline:
    """§3.3.1b's four clauses, in order, and never a re-sort."""

    def test_results_come_first_within_one_content(self) -> None:
        """R6.13 — clause 3, scoped per content and to `user`.

        Omitting it would project `[Text, ToolResult]` where the Chat
        Completions and Responses readers both produce `[ToolResult, Text]` for
        the same content — a delta no mutation caused.
        """
        projected = project(
            {
                "contents": [
                    {
                        "parts": [
                            {"text": "here you go"},
                            {"functionResponse": {"name": "f", "response": {}}},
                        ]
                    }
                ]
            }
        )

        parts = projected.conversation.turns[0].parts
        assert [type(part) for part in parts] == [c.ToolResult, c.Text]

    def test_clause_three_never_becomes_a_re_sort_of_the_merged_turn(self) -> None:
        """R6.13 — the defect §7.4.1 records, spelled as a test.

        `functionResponse -> user(text) -> functionResponse` must project as
        `[ToolResult, Text, ToolResult]`. A re-sort after the same-role merge
        gives `[ToolResult, ToolResult, Text]`, hoisting a result ahead of text
        the agent sent **before** it — moving history the bridge did not move.
        Because paths are index-based, that invented delta lands on every part
        of the turn and every turn after it.
        """
        projected = project(
            {
                "contents": [
                    {"parts": [{"functionResponse": {"name": "f", "response": {"n": 1}}}]},
                    {"parts": [{"text": "and also"}]},
                    {"parts": [{"functionResponse": {"name": "g", "response": {"n": 2}}}]},
                ]
            }
        )

        parts = projected.conversation.turns[0].parts
        assert [type(part) for part in parts] == [c.ToolResult, c.Text, c.ToolResult]

    def test_an_assistant_content_is_left_in_wire_order(self) -> None:
        """Clause 3 is about a `user` turn; a model turn is not reordered.

        **The fixture has to carry a `ToolResult` after text**, or this test
        passes without testing its own name: with a `ToolUse` the partition is a
        no-op whatever the role guard does, so deleting the guard leaves the
        suite green. Mutation testing found exactly that here, and T-A1 records
        the identical defect — "a test can pass without testing its name".

        A `functionResponse` inside a `model` content is unusual but
        structurally legal, and it is the only shape that makes the guard
        observable.
        """
        projected = project(
            {
                "contents": [
                    {
                        "role": "model",
                        "parts": [
                            {"text": "here it is"},
                            {"functionResponse": {"name": "f", "response": {}}},
                        ],
                    }
                ]
            }
        )

        parts = projected.conversation.turns[0].parts
        assert [type(part) for part in parts] == [c.Text, c.ToolResult]

    def test_consecutive_same_role_contents_merge(self) -> None:
        """Clause 4 — the format's own behaviour, and what keeps turn boundaries
        agreeing with the Chat Completions reader on the standard exchange."""
        projected = project(
            {
                "contents": [
                    {"role": "user", "parts": [{"text": "one"}]},
                    {"role": "user", "parts": [{"text": "two"}]},
                    {"role": "model", "parts": [{"text": "three"}]},
                ]
            }
        )

        assert projected.conversation.turns == (
            c.Turn("user", [c.Text("one"), c.Text("two")]),
            c.Turn("assistant", [c.Text("three")]),
        )


# --------------------------------------------------------------------------
# The published examples
# --------------------------------------------------------------------------

#: The ``curl`` request bodies the Gemini API reference publishes at
#: ``https://ai.google.dev/api/generate-content`` (retrieved 2026-09-12), keyed
#: by the sample file each comes from, with the route each is sent to.
#:
#: **Two deviations, and no others.**  ``chat.sh`` and ``controlled_generation.sh``
#: each carry a **trailing comma** as published, which is not JSON and which
#: ``json.loads`` rejects; it is removed.  The three File-API samples interpolate
#: a shell variable into ``file_uri``, so ``files/example`` stands in — the
#: suite names no host it does not control, which is the same substitution T-A3
#: made for the Responses examples.
#:
#: Note how freely the samples mix ProtoJSON's two spellings: ``file_data``,
#: ``system_instruction``, ``function_declarations``, ``tool_config`` and
#: ``response_mime_type`` in snake_case beside ``generationConfig``,
#: ``stopSequences``, ``maxOutputTokens`` and ``topP`` in camelCase. A reader
#: that knew only the schema's spelling would fail on five of these ten.
PUBLISHED_EXAMPLES: dict[str, tuple[str, Any]] = {
    "text_generation.sh": (
        "generateContent",
        {"contents": [{"parts": [{"text": "Write a story about a magic backpack."}]}]},
    ),
    "audio.sh": (
        "generateContent",
        {
            "contents": [
                {
                    "parts": [
                        {"text": "Please describe this file."},
                        {"file_data": {"mime_type": "audio/mpeg", "file_uri": "files/example"}},
                    ]
                }
            ]
        },
    ),
    "video.sh": (
        "generateContent",
        {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Transcribe the audio from this video, giving timestamps for "
                                "salient events in the video. Also provide visual descriptions."
                            )
                        },
                        {"file_data": {"mime_type": "video/mp4", "file_uri": "files/example"}},
                    ]
                }
            ]
        },
    ),
    "pdf.sh": (
        "generateContent",
        {
            "contents": [
                {
                    "parts": [
                        {"text": "Can you add a few more lines to this poem?"},
                        {
                            "file_data": {
                                "mime_type": "application/pdf",
                                "file_uri": "files/example",
                            }
                        },
                    ]
                }
            ]
        },
    ),
    "chat.sh": (
        "generateContent",
        {
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
                {
                    "role": "model",
                    "parts": [{"text": "Great to meet you. What would you like to know?"}],
                },
                {
                    "role": "user",
                    "parts": [{"text": "I have two dogs in my house. How many paws are in my house?"}],
                },
            ]
        },
    ),
    "controlled_generation.sh": (
        "generateContent",
        {
            "contents": [{"parts": [{"text": "List 5 popular cookie recipes"}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {"recipe_name": {"type": "STRING"}},
                    },
                },
            },
        },
    ),
    "configure_model_parameters.sh": (
        "generateContent",
        {
            "contents": [{"parts": [{"text": "Explain how AI works"}]}],
            "generationConfig": {
                "stopSequences": ["Title"],
                "temperature": 1.0,
                "maxOutputTokens": 800,
                "topP": 0.8,
                "topK": 10,
            },
        },
    ),
    "system_instruction.sh": (
        "generateContent",
        {
            "system_instruction": {"parts": {"text": "You are a cat. Your name is Neko."}},
            "contents": {"parts": {"text": "Hello there"}},
        },
    ),
    "text_generation_streaming.sh": (
        "streamGenerateContent",
        {"contents": [{"parts": [{"text": "Write a story about a magic backpack."}]}]},
    ),
    "function_calling.sh": (
        "generateContent",
        {
            "system_instruction": {
                "parts": {
                    "text": (
                        "You are a helpful lighting system bot. You can turn lights on and off, "
                        "and you can set the color. Do not perform any other tasks."
                    )
                }
            },
            "tools": [
                {
                    "function_declarations": [
                        {"name": "enable_lights", "description": "Turn on the lighting system."},
                        {
                            "name": "set_light_color",
                            "description": ("Set the light color. Lights must be enabled for this to work."),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "rgb_hex": {
                                        "type": "string",
                                        "description": (
                                            "The light color as a 6-digit hex string, e.g. ff0000 for red."
                                        ),
                                    }
                                },
                                "required": ["rgb_hex"],
                            },
                        },
                        {"name": "stop_lights", "description": "Turn off the lighting system."},
                    ]
                }
            ],
            "tool_config": {"function_calling_config": {"mode": "auto"}},
            "contents": {"role": "user", "parts": {"text": "Turn on the lights please."}},
        },
    ),
}


class TestPublishedExamples:
    """Every published example round-trips with an empty residual.

    This is T-A4's stated acceptance, and §3.3.1's independent-oracle rule is
    why it is *published* examples rather than captures of kitty: "a reader
    validated against kitty's output inherits kitty's bugs and the oracle
    becomes circular".
    """

    @pytest.mark.parametrize("sample", sorted(PUBLISHED_EXAMPLES))
    def test_it_round_trips_with_an_empty_residual(self, sample: str) -> None:
        """The acceptance criterion, one sample at a time so a failure names it."""
        operation, body = PUBLISHED_EXAMPLES[sample]
        projected = project(body, operation=operation)

        assert projected.residual == {}
        assert set(projected.consumed) == set(body)

    def test_the_streaming_sample_is_the_non_streaming_one_on_another_route(self) -> None:
        """§3.3.5, on Google's own samples: the bodies are identical.

        `text_generation.sh` publishes the same body for both operations, so
        these two are the published-example form of R7.5 — and evidence that
        nothing in a Gemini body says whether it streams.
        """
        assert PUBLISHED_EXAMPLES["text_generation.sh"][1] == PUBLISHED_EXAMPLES["text_generation_streaming.sh"][1]

        assert project(PUBLISHED_EXAMPLES["text_generation.sh"][1]).envelope.stream is False
        assert (
            project(
                PUBLISHED_EXAMPLES["text_generation_streaming.sh"][1],
                operation="streamGenerateContent",
            ).envelope.stream
            is True
        )

    def test_the_function_calling_sample_projects_its_three_tools_and_its_choice(self) -> None:
        """The richest published sample, asserted for content rather than emptiness.

        An empty-residual assertion alone would pass against a reader that read
        nothing at all, provided it claimed the keys.
        """
        operation, body = PUBLISHED_EXAMPLES["function_calling.sh"]
        projected = project(body, operation=operation)

        assert [tool.name for tool in projected.conversation.tools] == [
            "enable_lights",
            "set_light_color",
            "stop_lights",
        ]
        assert projected.envelope.extra["tool_choice"] == "auto"
        assert projected.conversation.turns == (c.Turn("user", [c.Text("Turn on the lights please.")]),)
        assert len(projected.conversation.system) == 1


# --------------------------------------------------------------------------
# Schema agreement
# --------------------------------------------------------------------------


class TestSchemaAgreement:
    """The reader's tables still agree with the published schema.

    Each expected set below is transcribed from the Gemini API discovery
    document at revision :data:`~harness.reader_gemini.SCHEMA_VERSION`, and
    exists so that a schema revision adding a key fails **here**, with the key
    named, rather than silently widening the residual in T-D5 — where it would
    read as an I1 breach.
    """

    def test_the_schema_version_is_recorded(self) -> None:
        """A table with no stated provenance cannot be re-derived."""
        assert r.SCHEMA_VERSION == "20260910"

    def test_the_request_body_has_nine_keys_and_model_is_not_one(self) -> None:
        """`GenerateContentRequest`, as the **REST** surface publishes it.

        The discovery document lists ten because it carries the proto message;
        the reference's "Request body" table lists these nine and puts `model`
        under "Path parameters". Pinning the REST nine is what makes a body
        `model` an unrecognised key rather than a tenth classification.
        """
        assert set(r.PUBLISHED_TOP_LEVEL_KEYS) == {
            "cachedContent",
            "contents",
            "generationConfig",
            "safetySettings",
            "serviceTier",
            "store",
            "systemInstruction",
            "toolConfig",
            "tools",
        }
        assert "model" not in r.PUBLISHED_TOP_LEVEL_KEYS

    def test_generation_config_has_twenty_five_keys(self) -> None:
        """`GenerationConfig`, split between the canonical set and the envelope."""
        assert set(r.PUBLISHED_GENERATION_CONFIG_KEYS) == {
            "_responseJsonSchema",
            "audioTranscriptionConfig",
            "candidateCount",
            "enableAffectiveDialog",
            "enableEnhancedCivicAnswers",
            "frequencyPenalty",
            "imageConfig",
            "logprobs",
            "maxOutputTokens",
            "mediaResolution",
            "presencePenalty",
            "responseFormat",
            "responseJsonSchema",
            "responseLogprobs",
            "responseMimeType",
            "responseModalities",
            "responseSchema",
            "seed",
            "speechConfig",
            "stopSequences",
            "temperature",
            "thinkingConfig",
            "topK",
            "topP",
            "translationConfig",
        }

    def test_part_has_sixteen_keys(self) -> None:
        """`Part` — a union discriminated by field name, not by a `type` tag."""
        assert set(r.PUBLISHED_PART_KEYS) == {
            "audioTranscription",
            "codeExecutionResult",
            "executableCode",
            "fileData",
            "functionCall",
            "functionResponse",
            "inlineData",
            "mediaProcessing",
            "mediaResolution",
            "partMetadata",
            "text",
            "thought",
            "thoughtSignature",
            "toolCall",
            "toolResponse",
            "videoMetadata",
        }

    def test_the_remaining_object_tables(self) -> None:
        """`Tool`, `ToolConfig`, `FunctionCallingConfig`, `FunctionDeclaration`,
        `FunctionCall`, `FunctionResponse`, `Content`, `Blob` and `FileData`."""
        assert set(r.PUBLISHED_TOOL_KEYS) == {
            "codeExecution",
            "computerUse",
            "fileSearch",
            "functionDeclarations",
            "googleMaps",
            "googleSearch",
            "googleSearchRetrieval",
            "mcpServers",
            "urlContext",
        }
        assert set(r.PUBLISHED_TOOL_CONFIG_KEYS) == {
            "functionCallingConfig",
            "includeServerSideToolInvocations",
            "retrievalConfig",
        }
        assert set(r.PUBLISHED_FUNCTION_CALLING_CONFIG_KEYS) == {"mode", "allowedFunctionNames"}
        assert set(r.PUBLISHED_FUNCTION_DECLARATION_KEYS) == {
            "behavior",
            "description",
            "name",
            "parameters",
            "parametersJsonSchema",
            "response",
            "responseJsonSchema",
        }
        assert set(r.PUBLISHED_FUNCTION_CALL_KEYS) == {"id", "name", "args"}
        assert set(r.PUBLISHED_FUNCTION_RESPONSE_KEYS) == {
            "id",
            "name",
            "parts",
            "response",
            "scheduling",
            "willContinue",
        }
        assert set(r.PUBLISHED_CONTENT_KEYS) == {"parts", "role"}
        assert set(r.PUBLISHED_BLOB_KEYS) == {"data", "displayName", "mimeType"}
        assert set(r.PUBLISHED_FILE_DATA_KEYS) == {"displayName", "fileUri", "mimeType"}

    def test_every_published_tool_choice_mode_is_accounted_for(self) -> None:
        """Five published modes: three carry a canonical value, two residualise.

        Stated as a partition so that a sixth mode cannot be silently dropped
        into the residualising half by omission.
        """
        published = {"MODE_UNSPECIFIED", "AUTO", "ANY", "NONE", "VALIDATED"}

        assert set(r._TOOL_CHOICE_MODES) < published
        assert published - set(r._TOOL_CHOICE_MODES) == {"MODE_UNSPECIFIED", "VALIDATED"}
        assert set(r._TOOL_CHOICE_MODES.values()) == c.TOOL_CHOICE_VALUES

    def test_the_canonical_sampling_targets_are_all_canonical(self) -> None:
        """`Conversation` rejects a non-canonical key, so a typo here would raise."""
        targets = {canonical for canonical, _ in r._SAMPLING_KEYS.values()}

        assert targets <= c.SAMPLING_KEYS
        assert len(targets) == len(r._SAMPLING_KEYS), "two Gemini keys share one canonical name"


# --------------------------------------------------------------------------
# The decisions the design review surfaced, each pinned by its own test
# --------------------------------------------------------------------------


class TestEvidencedAndUnpublishedShapes:
    """Shapes the published schema does not list, and what each one costs."""

    def test_a_function_role_projects_as_a_user_turn(self) -> None:
        """§7.4.1 evidence rule 1 — the client demonstrably sends it.

        The published `Content.role` is `user` or `model`, but
        `src/kitty/bridge/gemini/translator.py`'s `_ROLE_MAP` reads
        `role: "function"` straight off the inbound body — the same shape as
        that section's worked example, Anthropic's `effort`. `user` is where
        §3.3.1b already puts a tool result, so nothing is invented.

        Raising instead would be strictly worse than residualising, which is
        strictly worse than this: an `UnreadableBodyError` aborts the whole
        projection, so T-D1 would read a legitimate legacy request as a
        malformed corpus entry and never check it at all — kitty could mutate
        that request freely and the oracle would never see it.
        """
        projected = project(
            {
                "contents": [
                    {
                        "role": "function",
                        "parts": [{"functionResponse": {"name": "f", "response": {"n": 1}}}],
                    }
                ]
            }
        )

        assert projected.conversation.turns == (c.Turn("user", [c.ToolResult(content=[c.Json({"n": 1})])]),)

    def test_a_thought_part_need_not_carry_text(self) -> None:
        """Gemini returns signature-only thought parts, and they must project.

        §3.3.1 settles the shape: "An empty block is a part with an empty
        string, never nothing." Raising on one would abort the projection of a
        request Gemini 3 requires clients to echo back verbatim.
        """
        projected = project({"contents": [{"role": "model", "parts": [{"thought": True, "thoughtSignature": "sig"}]}]})

        assert projected.conversation.turns[0].parts == (c.Thinking("", signature="sig"),)

    def test_a_thought_flag_beside_a_data_member_marks_that_member_and_residualises(self) -> None:
        """`thought` is a part-level flag, not a data member.

        A `thought` beside a `functionCall` marks that call; only a part with no
        data member at all is itself a thought. The flag has no slot on
        `ToolUse`, so it residualises rather than being dropped on a branch that
        did not read it — the partial-consume trap T-A1 hit.
        """
        projected = project_untotalled(
            {
                "contents": [
                    {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "f"}, "thought": False}],
                    }
                ]
            }
        )

        assert projected.conversation.turns[0].parts == (c.ToolUse("f"),)
        assert projected.residual == {"contents[0].parts[0].thought": False}

    def test_a_function_response_name_is_accounted_for_and_then_lost(self) -> None:
        """A stated blind spot, not an oversight.

        §3.3.1 makes the name load-bearing — "pairing there is by tool name and
        the k-th unanswered call of that name" — but `ToolResult` has three
        fields and none of them is a name, so the grammar cannot hold what that
        sentence promises. Residualising would fail the run on every Gemini tool
        turn, since `name` is **required** on `FunctionResponse`.

        So it is consumed and the loss is recorded here rather than left
        implicit: a mutation that rewrote `functionResponse.name` while leaving
        the payload alone would produce no delta.
        """
        projected = project({"contents": [{"parts": [{"functionResponse": {"name": "get_weather", "response": {}}}]}]})

        assert projected.residual == {}
        assert projected.conversation.turns[0].parts == (c.ToolResult(content=[c.Json({})]),)

    @pytest.mark.parametrize(
        ("member", "payload"),
        [
            ("inlineData", {"mimeType": "image/png", "data": "", "displayName": "my.png"}),
            ("fileData", {"fileUri": "files/example", "displayName": "my.pdf"}),
        ],
    )
    def test_a_display_name_on_a_media_part_residualises(self, member: str, payload: Any) -> None:
        """`Image` has three fields and none of them names the blob to the model."""
        projected = project_untotalled({"contents": [{"parts": [{member: payload}]}]})

        assert projected.residual == {f"contents[0].parts[0].{member}.displayName": payload["displayName"]}


class TestResidualsExpectedOnRealTraffic:
    """The residualisations that will fail the first oracle run, gathered.

    §7.4.1 says of the identical `cache_control` case: "That is the correct
    signal and it is also a deadline." Each of these is a field the format
    publishes, real clients send, and the grammar cannot carry — so each fails
    the run until T-W2 grows a slot or a declared-ignored mechanism.

    All **six** are listed and asserted together, so the ticket that closes them
    has one inventory rather than six scattered findings — and so that a later
    reader cannot mistake any of them for an accident.

    A `VALIDATED` tool-calling mode is deliberately **not** here. It also
    residualises, but it is a gap in `TOOL_CHOICE_VALUES` — a closed vocabulary
    with no raw companion — rather than a missing slot in the grammar, so it has
    a different fix and belongs to a different ticket.
    :meth:`TestToolChoice.test_a_mode_with_no_canonical_value_residualises` owns
    it.
    """

    CASES: dict[str, tuple[Any, str]] = {
        # Gemini 3 *requires* clients to echo a functionCall's thoughtSignature
        # back verbatim, so this fires on every tool turn.
        "thoughtSignature on a functionCall": (
            {"contents": [{"role": "model", "parts": [{"functionCall": {"name": "f"}, "thoughtSignature": "s"}]}]},
            "contents[0].parts[0].thoughtSignature",
        ),
        # Google's own SDKs set a role on the system instruction.
        "role on a systemInstruction": (
            {"systemInstruction": {"role": "user", "parts": [{"text": "hi"}]}},
            "systemInstruction.role",
        ),
        # NON_BLOCKING function calling, a live feature.
        "behavior on a functionDeclaration": (
            {"tools": [{"functionDeclarations": [{"name": "f", "behavior": "NON_BLOCKING"}]}]},
            "tools[0].functionDeclarations[0].behavior",
        ),
        # The scheduling half of the same feature.
        "scheduling on a functionResponse": (
            {"contents": [{"parts": [{"functionResponse": {"name": "f", "response": {}, "scheduling": "SILENT"}}]}]},
            "contents[0].parts[0].functionResponse.scheduling",
        ),
        # Video understanding, also live.
        "videoMetadata on a part": (
            {"contents": [{"parts": [{"text": "x", "videoMetadata": {"fps": 1.0}}]}]},
            "contents[0].parts[0].videoMetadata",
        ),
        # Naming a blob or file to the model, which `Image` has no slot for.
        "displayName on a media part": (
            {"contents": [{"parts": [{"fileData": {"fileUri": "files/x", "displayName": "my.pdf"}}]}]},
            "contents[0].parts[0].fileData.displayName",
        ),
    }

    def test_the_inventory_is_the_count_the_design_and_the_ticket_quote(self) -> None:
        """Six, in `TEST_SUITE.md` §7.4.2, in R6.10a, and here.

        Asserted because three documents quote this number and a silent
        disagreement between them gives whoever picks up the ticket an ambiguous
        scope.
        """
        assert len(self.CASES) == 6

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_it_residualises_at_the_path_the_ticket_names(self, case: str) -> None:
        """Each fails closed, at a path a maintainer can act on."""
        body, path = self.CASES[case]
        projected = project_untotalled(body)

        assert path in projected.residual
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_a_validated_mode_is_not_quietly_mapped_onto_a_near_neighbour(self) -> None:
        """The alternative to residualising, and why it is worse.

        `VALIDATED` means "constrained to predict either function calls **or**
        natural language, and ensures function schema adherence" — so it is
        `auto` with validation, not `any`. Mapping it to either would make an
        `AUTO` to `VALIDATED` mutation invisible, and `TOOL_CHOICE_VALUES` is
        closed with no raw companion (unlike `stop_reason`, which §3.3.1b gave
        `stop_reason_raw` for exactly this reason), so there is no lossless home
        for it. A loud failure is recoverable; a silent equivalence is not.
        """
        projected = project_untotalled({"toolConfig": {"functionCallingConfig": {"mode": "VALIDATED"}}})

        assert "tool_choice" not in projected.envelope.extra


# --------------------------------------------------------------------------
# The two techniques T-A1 measured as worth their cost
# --------------------------------------------------------------------------


class TestEveryOptionalLeafFailsClosed:
    """§7.4.1's wrongly-typed-leaf rule, over **every** optional leaf.

    "It binds *every* optional leaf, not the ones a bug happened to be found
    in." T-A1 measured eight of nine leaves failing open when the rule was
    stated generally and applied field by field, which is why this is a table
    over the whole surface rather than a sample: the contract validates only
    roles, sampling keys and `tool_choice`, so an unguarded leaf declared
    `str | None` carries a dict silently and the residual stays empty.

    Each case asserts **both** halves — the field residualises at its own path,
    *and* the projection carries the grammar's absent value in its place. Half
    the rule would be satisfied by a reader that residualised and then coerced.
    """

    #: ``(case, body, residual path, a check that the field is at its absent value)``.
    CASES: dict[str, tuple[Any, str, Any]] = {
        "store": ({"store": "yes"}, "store", lambda p: p.envelope.store is None),
        "generationConfig.temperature": (
            {"generationConfig": {"temperature": "hot"}},
            "generationConfig.temperature",
            lambda p: "temperature" not in p.conversation.sampling,
        ),
        "generationConfig.stopSequences": (
            {"generationConfig": {"stopSequences": "Title"}},
            "generationConfig.stopSequences",
            lambda p: "stop" not in p.conversation.sampling,
        ),
        "generationConfig.responseLogprobs": (
            {"generationConfig": {"responseLogprobs": 1}},
            "generationConfig.responseLogprobs",
            lambda p: "logprobs" not in p.conversation.sampling,
        ),
        "generationConfig.topP": (
            {"generationConfig": {"topP": "wide"}},
            "generationConfig.topP",
            lambda p: "top_p" not in p.conversation.sampling,
        ),
        "generationConfig.topK": (
            {"generationConfig": {"topK": 1.5}},
            "generationConfig.topK",
            lambda p: "top_k" not in p.conversation.sampling,
        ),
        "generationConfig.maxOutputTokens": (
            {"generationConfig": {"maxOutputTokens": "800"}},
            "generationConfig.maxOutputTokens",
            lambda p: "max_tokens" not in p.conversation.sampling,
        ),
        "generationConfig.presencePenalty": (
            {"generationConfig": {"presencePenalty": True}},
            "generationConfig.presencePenalty",
            lambda p: "presence_penalty" not in p.conversation.sampling,
        ),
        "generationConfig.frequencyPenalty": (
            {"generationConfig": {"frequencyPenalty": []}},
            "generationConfig.frequencyPenalty",
            lambda p: "frequency_penalty" not in p.conversation.sampling,
        ),
        "generationConfig.seed": (
            {"generationConfig": {"seed": "7"}},
            "generationConfig.seed",
            lambda p: "seed" not in p.conversation.sampling,
        ),
        "generationConfig.candidateCount": (
            {"generationConfig": {"candidateCount": 2.5}},
            "generationConfig.candidateCount",
            lambda p: "n" not in p.conversation.sampling,
        ),
        "generationConfig.logprobs": (
            {"generationConfig": {"logprobs": True}},
            "generationConfig.logprobs",
            lambda p: "top_logprobs" not in p.conversation.sampling,
        ),
        "blob.data": (
            {"contents": [{"parts": [{"inlineData": {"mimeType": "image/png", "data": 7}}]}]},
            "contents[0].parts[0].inlineData.data",
            lambda p: p.conversation.turns[0].parts[0].digest == c.image_digest(b""),
        ),
        "functionCallingConfig.mode": (
            {"toolConfig": {"functionCallingConfig": {"mode": {"a": 1}}}},
            "toolConfig.functionCallingConfig.mode",
            lambda p: "tool_choice" not in p.envelope.extra,
        ),
        "functionCallingConfig.allowedFunctionNames": (
            {"toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": "f"}}},
            "toolConfig.functionCallingConfig.allowedFunctionNames",
            lambda p: p.envelope.extra["tool_choice"] == "any",
        ),
        "functionDeclaration.description": (
            {"tools": [{"functionDeclarations": [{"name": "f", "description": {"a": 1}}]}]},
            "tools[0].functionDeclarations[0].description",
            lambda p: p.conversation.tools[0].description is None,
        ),
        "functionDeclaration.parametersJsonSchema": (
            {"tools": [{"functionDeclarations": [{"name": "f", "parametersJsonSchema": ["ab", "cd"]}]}]},
            "tools[0].functionDeclarations[0].parametersJsonSchema",
            lambda p: p.conversation.tools[0].schema is None,
        ),
        "functionDeclaration.parameters": (
            {"tools": [{"functionDeclarations": [{"name": "f", "parameters": ["ab", "cd"]}]}]},
            "tools[0].functionDeclarations[0].parameters",
            lambda p: p.conversation.tools[0].schema is None,
        ),
        "part.thoughtSignature-as-name-placeholder": (
            {"contents": [{"parts": [{"functionCall": {"name": 7}}]}]},
            "contents[0].parts[0].functionCall.name",
            lambda p: p.conversation.turns[0].parts[0] == c.ToolUse(name=""),
        ),
        "blob.data undecodable": (
            {"contents": [{"parts": [{"inlineData": {"mimeType": "image/png", "data": "aGk=\n"}}]}]},
            "contents[0].parts[0].inlineData.data",
            lambda p: p.conversation.turns[0].parts[0] == c.Image(media_type="image/png"),
        ),
        "part.thought": (
            {"contents": [{"parts": [{"text": "x", "thought": "yes"}]}]},
            "contents[0].parts[0].thought",
            lambda p: p.conversation.turns[0].parts[0] == c.Text("x"),
        ),
        "part.thoughtSignature": (
            {"contents": [{"parts": [{"text": "x", "thought": True, "thoughtSignature": 7}]}]},
            "contents[0].parts[0].thoughtSignature",
            lambda p: p.conversation.turns[0].parts[0] == c.Thinking("x"),
        ),
        "blob.mimeType": (
            {"contents": [{"parts": [{"inlineData": {"data": "", "mimeType": 7}}]}]},
            "contents[0].parts[0].inlineData.mimeType",
            lambda p: p.conversation.turns[0].parts[0].media_type is None,
        ),
        "fileData.fileUri": (
            {"contents": [{"parts": [{"fileData": {"fileUri": ["a"]}}]}]},
            "contents[0].parts[0].fileData.fileUri",
            lambda p: p.conversation.turns[0].parts[0].ref is None,
        ),
        "fileData.mimeType": (
            {"contents": [{"parts": [{"fileData": {"fileUri": "files/x", "mimeType": 7}}]}]},
            "contents[0].parts[0].fileData.mimeType",
            lambda p: p.conversation.turns[0].parts[0].media_type is None,
        ),
        "functionCall.args": (
            {"contents": [{"parts": [{"functionCall": {"name": "f", "args": ["ab", "cd"]}}]}]},
            "contents[0].parts[0].functionCall.args",
            lambda p: p.conversation.turns[0].parts[0].arguments == {},
        ),
        "functionCall.id": (
            {"contents": [{"parts": [{"functionCall": {"name": "f", "id": 7}}]}]},
            "contents[0].parts[0].functionCall.id",
            lambda p: p.conversation.turns[0].parts[0].id is None,
        ),
        "functionResponse.response": (
            {"contents": [{"parts": [{"functionResponse": {"name": "f", "response": "ok"}}]}]},
            "contents[0].parts[0].functionResponse.response",
            lambda p: p.conversation.turns[0].parts[0].content == (),
        ),
        "functionResponse.id": (
            {"contents": [{"parts": [{"functionResponse": {"name": "f", "response": {}, "id": 7}}]}]},
            "contents[0].parts[0].functionResponse.id",
            lambda p: p.conversation.turns[0].parts[0].tool_use_id is None,
        ),
        "systemInstruction part text": (
            {"systemInstruction": {"parts": [{"text": 7}]}},
            "systemInstruction.parts[0].text",
            lambda p: p.conversation.system == (),
        ),
    }

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_it_residualises_and_leaves_the_field_absent(self, case: str) -> None:
        """Neither coerced nor raised on: `str(7)` and `dict(["ab","cd"])` invent."""
        body, path, absent = self.CASES[case]
        projected = project_untotalled(body)

        assert path in projected.residual, f"{case} failed open: residual={dict(projected.residual)}"
        assert absent(projected), f"{case} residualised but did not fall back to the absent value"

        # The residual's *value* is asserted too, not only its key. A reader can
        # record the right key and then overwrite it with the default it fell
        # back to — which tells a maintainer the client sent `null` when it sent
        # an object. That defect was live in `_read_tool_choice` and 28 of these
        # cases could not see it, because every one asserted only the key.
        assert projected.residual[path] == _body_value_at(body, path), (
            f"{case} residualised the wrong value: the residual must carry what the "
            f"client actually sent, or the failure report misdiagnoses it"
        )

    #: The one leaf `_typed_leaf` guards that this table deliberately omits,
    #: because it **raises** instead of residualising: §7.4.1's exception for a
    #: value that *is* its part. `Text` and `Thinking` have no absent value to
    #: fall back to, and `Text("")` would fabricate an empty part — which is
    #: meaningful in this grammar, since P5e and P8 both inject one.
    #: `TestUnionMemberValues` pins it directly.
    STRUCTURAL_LEAVES = frozenset({"text"})

    #: Leaves whose name reaches `_typed_leaf` in a variable rather than as a
    #: literal, so the source scan below cannot find them. Two today, both in
    #: `_read_declaration_schema`'s loop over the mutually exclusive parameter
    #: spellings; listed rather than inferred, because a silently missed call
    #: site is exactly what this guard exists to prevent.
    VARIABLE_DRIVEN_LEAVES = frozenset({"parameters", "parametersJsonSchema"})

    def test_the_table_covers_every_leaf_the_reader_guards(self) -> None:
        """A leaf added later must be added here, or this goes red.

        Derived from the source rather than trusted to an author's memory,
        because trusting the author is exactly what left eight of nine leaves
        unguarded in T-A1. `_typed_leaf` is the one helper every optional leaf
        goes through, so its call sites — plus the sampling table it is driven
        from in a loop — are the population this table must match.
        """
        source = Path(r.__file__).read_text(encoding="utf-8")
        literal = re.findall(r'_typed_leaf\(\w+, "([^"]+)"', source)
        guarded = set(literal) | set(r._SAMPLING_KEYS) | self.VARIABLE_DRIVEN_LEAVES

        # The regex above cannot see a call site that passes the leaf name in a
        # variable, so "derived from the source" would quietly become "derived
        # from the source, except where it is not". Counting the call sites it
        # *did* match against the ones it could not keeps that claim honest.
        assert len(literal) + 2 == source.count("_typed_leaf(") - 1, (
            "a `_typed_leaf` call site is neither a string literal nor listed in VARIABLE_DRIVEN_LEAVES"
        )

        guarded -= self.STRUCTURAL_LEAVES

        covered = {path.rpartition(".")[2] or path for _, path, _ in self.CASES.values()}
        missing = sorted(guarded - covered)

        assert missing == [], f"optional leaves with no wrongly-typed case: {missing}"


def _body_value_at(body: Any, path: str) -> Any:
    """Return the value a body carries at a residual path, or ``None`` if absent.

    The residual's keys are "the body's own path … with array positions as
    indices" (§7.4.1), so they can be walked straight back against the body.
    Doing that, rather than hand-maintaining an expected value per case, is what
    keeps the table's value assertion honest as cases are added.

    Args:
        body: The request body the case projected.
        path: A residual key, such as ``contents[0].parts[0].inlineData.data``.

    Returns:
        The value at that position, or ``None`` when the body does not carry one
        — which is the value a reader must residualise for an absent required
        field.
    """
    node: Any = body
    for segment in path.split("."):
        name, _, indices = segment.partition("[")
        if name not in node:
            return None
        node = node[name]
        for index in re.findall(r"(\d+)\]", indices):
            node = node[int(index)]

    return node


class TestInjectionProbe:
    """Replace every position of a maximal body with junk, and with nothing.

    T-A1 ran 938 of these in seconds and found an `Opaque.kind` hole that 103
    hand-written tests had missed. Two claims, and neither can be made by a
    hand-written suite of any realistic size:

    1. **Nothing but `UnreadableBodyError` escapes.** The contract names three
       failure shapes, and a `KeyError` out of a reader would be an undefined
       fourth on the one path T-D1 uses to tell an unreadable body from an I1
       breach. A `ValueError` is a *reader bug* by the contract's own
       definition, so it must not escape either.
    2. **No projected field violates its declared type.** A residual proves the
       reader noticed; this proves it did not also carry the junk through.
    """

    #: A body reaching every branch this reader has, so the walk below covers
    #: the whole surface rather than the shapes a test author thought of.
    MAXIMAL: dict[str, Any] = {
        "contents": [
            {"role": "user", "parts": [{"text": "hi"}, {"inlineData": {"mimeType": "image/png", "data": ""}}]},
            {
                "role": "model",
                "parts": [
                    {"text": "thinking", "thought": True, "thoughtSignature": "sig"},
                    {"functionCall": {"id": "fc-1", "name": "f", "args": {"q": 1}}},
                    {"executableCode": {"language": "PYTHON", "code": "print(1)"}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"id": "fc-1", "name": "f", "response": {"n": 1}}},
                    {"fileData": {"mimeType": "application/pdf", "fileUri": "files/example"}},
                ],
            },
        ],
        "systemInstruction": {"parts": [{"text": "be brief"}]},
        "tools": [
            {
                "functionDeclarations": [{"name": "f", "description": "d", "parameters": {"type": "object"}}],
                "googleSearch": {},
            }
        ],
        "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["f"]}},
        "generationConfig": {"temperature": 1.0, "topK": 4, "responseSchema": {"type": "ARRAY"}},
        "safetySettings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"}],
        "cachedContent": "cachedContents/abc",
        "serviceTier": "SERVICE_TIER_PRIORITY",
        "store": True,
    }

    #: Twelve junk values plus deletion, chosen to cross every type boundary the
    #: reader branches on — and `{}`/`[]` because an *empty* container of the
    #: right shape is the value a guard written as a truth test waves through.
    JUNK: tuple[Any, ...] = (
        None,
        True,
        0,
        -1,
        1.5,
        "",
        "x",
        [],
        {},
        [None],
        {"a": None},
        ["ab", "cd"],
    )

    @staticmethod
    def _positions(node: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        """Return every addressable position in a nested structure.

        Args:
            node: The structure to walk.
            path: The path accumulated so far.

        Returns:
            One tuple of keys and indices per position, the root excluded.
        """
        found: list[tuple[Any, ...]] = []
        if isinstance(node, dict):
            for key, value in node.items():
                found.append((*path, key))
                found.extend(TestInjectionProbe._positions(value, (*path, key)))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                found.append((*path, index))
                found.extend(TestInjectionProbe._positions(value, (*path, index)))
        return found

    @staticmethod
    def _replaced(body: Any, path: tuple[Any, ...], value: Any) -> Any:
        """Return a deep copy of ``body`` with one position replaced or removed.

        Args:
            body: The structure to copy.
            path: The position to change.
            value: The replacement, or :data:`TestInjectionProbe.DELETE` to
                remove the position entirely.

        Returns:
            The mutated copy.
        """
        mutated = json.loads(json.dumps(body))
        node = mutated
        for step in path[:-1]:
            node = node[step]

        if value is TestInjectionProbe.DELETE:
            del node[path[-1]]
        else:
            node[path[-1]] = value

        return mutated

    #: The "replacement" that removes a position instead of changing it.
    DELETE = object()

    def test_nothing_but_an_unreadable_body_error_escapes(self) -> None:
        """Claim 1, over every position crossed with every junk value."""
        positions = self._positions(self.MAXIMAL)
        # 70 positions x 13 replacements is ~900 injections, the order T-A1
        # needed before the probe found what its 103 hand-written tests had not.
        # Asserted so that shrinking the maximal body silently weakens the probe.
        assert len(positions) >= 70, f"the maximal body stopped being maximal: {len(positions)}"

        escapes: list[str] = []
        for path in positions:
            for value in (*self.JUNK, self.DELETE):
                mutated = self._replaced(self.MAXIMAL, path, value)
                try:
                    projected = r.GeminiProjection().read_request(captured(mutated))
                except c.UnreadableBodyError:
                    continue
                except Exception as exc:  # noqa: BLE001 - the point of the probe
                    escapes.append(f"{'.'.join(map(str, path))} := {value!r} raised {exc!r}")
                    continue

                violation = _type_violation(projected)
                if violation is not None:
                    escapes.append(f"{'.'.join(map(str, path))} := {value!r} projected {violation}")

        assert escapes == [], "\n".join(escapes[:20])


def _type_violation(projected: c.Request) -> str | None:
    """Return the first projected field whose value is outside its declared type.

    Claim 2 of the injection probe. A residual proves the reader *noticed* a
    junk value; this proves it did not also carry one through into a field the
    grammar declares narrowly, which no amount of residual checking can show.

    Args:
        projected: A projection to inspect.

    Returns:
        A description of the first violation, or ``None`` when there is none.
    """
    envelope = projected.envelope
    for name, value, expected in (
        ("envelope.model", envelope.model, str),
        ("envelope.stream", envelope.stream, bool),
        ("envelope.store", envelope.store, bool),
    ):
        if value is not None and not isinstance(value, expected):
            return f"{name}={value!r}"

    for tool in projected.conversation.tools:
        if not isinstance(tool.name, str):
            return f"ToolDecl.name={tool.name!r}"
        if tool.description is not None and not isinstance(tool.description, str):
            return f"ToolDecl.description={tool.description!r}"
        if tool.schema is not None and not isinstance(tool.schema, Mapping):
            return f"ToolDecl.schema={tool.schema!r}"

    for turn in projected.conversation.turns:
        for part in turn.parts:
            violation = _part_violation(part)
            if violation is not None:
                return violation

    return None


def _part_violation(part: c.Part) -> str | None:
    """Return a description of a part carrying a value outside its declared type.

    Args:
        part: A projected part.

    Returns:
        A description, or ``None`` when the part is well typed.
    """
    if isinstance(part, c.Text) and not isinstance(part.text, str):
        return f"Text.text={part.text!r}"
    if isinstance(part, c.Thinking):
        if not isinstance(part.text, str):
            return f"Thinking.text={part.text!r}"
        if part.signature is not None and not isinstance(part.signature, str):
            return f"Thinking.signature={part.signature!r}"
    if isinstance(part, c.ToolUse):
        if not isinstance(part.name, str):
            return f"ToolUse.name={part.name!r}"
        if part.id is not None and not isinstance(part.id, str):
            return f"ToolUse.id={part.id!r}"
    if isinstance(part, c.Image):
        for name, value in (("digest", part.digest), ("media_type", part.media_type), ("ref", part.ref)):
            if value is not None and not isinstance(value, str):
                return f"Image.{name}={value!r}"
    if isinstance(part, c.Opaque) and not isinstance(part.kind, str):
        return f"Opaque.kind={part.kind!r}"
    if isinstance(part, c.ToolResult):
        if part.tool_use_id is not None and not isinstance(part.tool_use_id, str):
            return f"ToolResult.tool_use_id={part.tool_use_id!r}"
        for member in part.content:
            violation = _part_violation(member)
            if violation is not None:
                return violation
    return None


class TestUnionMemberValues:
    """Where a union member's own value is wrong, and where a field of one is.

    §7.4.2 rule 7. The three shipped readers reached three answers here, so this
    class pins T-A4's and says which rule each follows:

    - **The value that *is* the part** — `Part.text` — raises. `Text` and
      `Thinking` have no absent value in the grammar, and `Text("")` would
      fabricate an empty part the agent never sent, which is meaningful here
      because P5e and P8 both inject one. T-A1 agrees.
    - **A required *field* of a part** — a `name` — residualises and the part is
      still projected. §3.3.1b settles it in those words. T-A3 agrees.
    - **An undecodable payload** — base64 that does not decode — residualises
      the leaf and the part keeps its place, because `Image.digest` is
      `str | None` and §7.4.1 says "raising is the other wrong answer".

    The common thread, and the reason all three land here rather than in three
    different classes: **no branch ever drops a part.** Paths are index-based,
    so a dropped part shifts every later index and invents a delta on content
    nobody touched.
    """

    def test_a_wrongly_typed_text_raises_rather_than_projecting_an_empty_part(self) -> None:
        """`Text` *is* its value, so there is no partial part to salvage."""
        with pytest.raises(c.UnreadableBodyError):
            project_untotalled({"contents": [{"parts": [{"text": {"a": 1}}]}]})

    def test_a_function_call_with_no_name_still_projects(self) -> None:
        """The call keeps its place; the residual names what was missing."""
        projected = project_untotalled({"contents": [{"parts": [{"text": "a"}, {"functionCall": {"args": {"q": 1}}}]}]})

        assert projected.conversation.turns[0].parts == (
            c.Text("a"),
            c.ToolUse(name="", arguments={"q": 1}),
        )
        assert projected.residual == {"contents[0].parts[1].functionCall.name": None}

    def test_line_wrapped_base64_residualises_instead_of_killing_the_request(self) -> None:
        """A single newline in one blob must not abort the whole projection.

        `validate=True` rejects every RFC 2045 line break, and Google's own
        image-understanding sample passes `-w0` to `base64(1)` precisely because
        its default output is wrapped — so this is real traffic, not a
        hypothetical. If it raised, T-D1 would read a legitimate corpus entry as
        malformed and never diff it, and kitty could mutate that request freely.
        """
        wrapped = base64.b64encode(b"hello world" * 8).decode("ascii")
        wrapped = wrapped[:20] + "\n" + wrapped[20:]

        projected = project_untotalled(
            {"contents": [{"parts": [{"inlineData": {"mimeType": "image/png", "data": wrapped}}, {"text": "and"}]}]}
        )

        # The part keeps its place and the later part keeps its index.
        assert projected.conversation.turns[0].parts == (
            c.Image(digest=None, media_type="image/png"),
            c.Text("and"),
        )
        assert projected.residual == {"contents[0].parts[0].inlineData.data": wrapped}

    def test_no_failure_branch_drops_a_part_and_shifts_the_later_indices(self) -> None:
        """The claim all three branches share, asserted as one fact.

        A reader that returned `None` for a part it could not read would shift
        every later part's index — §7.4.1: "that invented delta lands on every
        part of the turn and on every turn after it". T-A3 does exactly that for
        two of these three cases, which is why rule 7 exists.
        """
        projected = project_untotalled(
            {
                "contents": [
                    {
                        "parts": [
                            {"functionCall": {"args": {}}},
                            {"inlineData": {"data": "aGk=\n"}},
                            {"text": "last"},
                        ]
                    }
                ]
            }
        )

        parts = projected.conversation.turns[0].parts
        assert len(parts) == 3
        assert parts[2] == c.Text("last")

    def test_a_part_carrying_two_data_members_dispatches_in_the_fixed_order(self) -> None:
        """R6.4 — a `oneof` carrying two members is ill-formed, and must still be
        read the same way by every reader of a field-name-discriminated union.

        The loser residualises either way, so the order decides only which field
        the failure *names* — but two readers naming different fields is a
        diagnosis that drifts, which is what §7.4.2 rule 6 exists to stop. Data
        members win over `text`, so the part that *carries content* is the one
        projected.
        """
        projected = project_untotalled(
            {"contents": [{"parts": [{"text": "shadowed", "fileData": {"fileUri": "files/x"}}]}]}
        )

        assert projected.conversation.turns[0].parts == (c.Image(ref="files/x"),)
        assert projected.residual == {"contents[0].parts[0].text": "shadowed"}


class TestFailureShapes:
    """The structural failures R2.5 names, each held by a test.

    Correct behaviour that nothing asserts is behaviour a refactor can delete
    silently — and two of these guards (`_parse_body`'s `UnicodeDecodeError`
    arm and its not-an-object check) are single lines that read as redundant.
    T-D1 tells "this body was unreadable" from "this is an I1 breach" on exactly
    this exception, so the arms matter more than their size suggests.
    """

    @pytest.mark.parametrize(
        ("raw", "case"),
        [
            (b"{not json", "malformed JSON"),
            (b"", "an empty body"),
            (b"\xff\xfe not utf-8", "bytes that are not UTF-8"),
            (b"[1, 2]", "a JSON array, which is valid JSON and an invalid request"),
            (b'"a string"', "a JSON string"),
            (b"null", "a JSON null"),
        ],
    )
    def test_a_body_that_is_not_a_json_object_is_unreadable(self, raw: bytes, case: str) -> None:
        """R2.5 — the body has to be an object before anything else is true."""
        with pytest.raises(c.UnreadableBodyError):
            project_untotalled(raw)

    @pytest.mark.parametrize("member", ["x", 7, None, ["nested"]])
    def test_a_contents_member_that_is_not_an_object_is_unreadable(self, member: Any) -> None:
        """R2.5 — structural: a turn cannot be built from a scalar."""
        with pytest.raises(c.UnreadableBodyError):
            project_untotalled({"contents": [member]})


class TestTotalityOverTheWholeSurface:
    """The acceptance criterion for R2.1–R2.3, on a body that uses every key.

    The published examples each carry a handful of keys, and the injection probe
    only ever projects the maximal body *mutated*. Without this, "every one of
    the nine published top-level keys is classified" is asserted key by key and
    never all at once — and a reader can satisfy nine separate tests while
    mis-handling a body that carries all nine together.
    """

    def test_the_maximal_body_accounts_for_every_key_it_carries(self) -> None:
        """Nine keys in, nine consumed, nothing residual."""
        body = TestInjectionProbe.MAXIMAL
        assert set(body) == set(r.PUBLISHED_TOP_LEVEL_KEYS), "the maximal body stopped being total"

        projected = project(body)

        assert set(projected.consumed) == set(body)
        assert projected.residual == {}

    @pytest.mark.parametrize(
        ("key", "value"),
        [("store", None), ("cachedContent", None), ("generationConfig", {"temperature": None})],
    )
    def test_an_explicit_null_reads_as_the_field_being_absent(self, key: str, value: Any) -> None:
        """ProtoJSON: "null is an accepted value for all field types and treated
        as the default value of the corresponding field type".

        Residualising an explicit null would fail an oracle run on a body the
        API accepts, so the branch that reads it is load-bearing — and it is one
        line, which is how it would come to be deleted.
        """
        projected = project({key: value})

        assert projected.residual == {}
        assert projected.envelope.store is None
        assert projected.conversation.sampling == {}


class TestSingletonRepeatedFields:
    """R6.3 applies to every repeated field, not only the two with samples.

    Google's published examples send `contents` and `parts` as bare objects; the
    other three repeated fields have no published example either way. The rule
    is applied to all five because the asymmetry costs more than the leniency:
    accepting a form the API rejects reads a body no client sends, while
    rejecting one it accepts aborts the projection of a legitimate request.
    """

    def test_a_singleton_tools_entry(self) -> None:
        """`tools` — no published sample, same rule."""
        projected = project({"tools": {"functionDeclarations": {"name": "f"}}})

        assert projected.conversation.tools == (c.ToolDecl("f"),)

    def test_a_singleton_function_response_part(self) -> None:
        """`functionResponse.parts` — likewise."""
        projected = project(
            {
                "contents": [
                    {
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "f",
                                    "response": {},
                                    "parts": {"inlineData": {"data": "", "mimeType": "image/png"}},
                                }
                            }
                        ]
                    }
                ]
            }
        )

        result = projected.conversation.turns[0].parts[0]
        assert isinstance(result, c.ToolResult)
        assert result.content == (c.Json({}), c.Image(digest=c.image_digest(b""), media_type="image/png"))

    @pytest.mark.parametrize("value", [7, "x", None])
    def test_a_repeated_field_that_is_neither_a_list_nor_an_object_is_unreadable(self, value: Any) -> None:
        """R2.5's container rule, on the field whose message says both forms."""
        with pytest.raises(c.UnreadableBodyError):
            project_untotalled({"tools": value})
