"""L1 tests for the OpenAI Responses projection.

`.system_design/TEST_SUITE.md` §3.3.1, §3.3.1a, §3.3.1b · plan task **T-A3** (KBR-35).

**Every fixture is a published example.**  The eight bodies in
:class:`TestPublishedExamples` are the ``x-oaiMeta`` request examples carried by
``openai/openai-openapi``'s own ``openapi.yaml`` at ``info.version``
:data:`~harness.reader_responses.SCHEMA_VERSION`, **adapted only by substituting
``example.test`` for the third-party image and file URLs** so the suite names no
host it does not control; the rest are assembled from
``CreateResponse`` and its referenced schemas in that same file.  None is copied
from kitty's output — §3.3.1's independent-oracle rule makes a reader validated
against kitty's output circular, because it would inherit kitty's bugs and the
oracle would then prove only that kitty agrees with itself.

**The falsification set is required, not decorative.**  Plan §1.4: the first
working version of every harness ships with at least one deliberate defect it
must detect, running in the suite.  :class:`TestFalsification` holds three — an
unrecognised top-level key, an unrecognised item type, and a *dropping* reader
that residualises nothing.  The third is the one a "residual must be empty" rule
cannot catch, which is why :attr:`harness.contract.Request.consumed` exists.

**Two guards this module owns on behalf of others.**  Register row **P16** is
``NOT_PROJECTABLE``, and ``contract.py`` names "the Responses reader's own L1
test" as one of its two guards — :class:`TestP16StaysInvisible` is it.  And the
31 published top-level keys and 31 item types are asserted against the reader's
own tables, so a schema revision that adds one fails here rather than silently
widening the residual in T-D5.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from harness import contract as c
from harness import reader_responses as r


def captured(body: Any) -> c.CapturedRequest:
    """Wrap a body as a capture aimed at the published Responses endpoint.

    Args:
        body: A mapping to encode as JSON, or raw bytes to pass through
            undecoded for the failure-shape tests.

    Returns:
        A capture the reader can be handed directly.
    """
    raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")

    return c.CapturedRequest(
        method="POST",
        scheme="https",
        host="api.openai.com",
        path="/v1/responses",
        query="",
        headers=(("content-type", "application/json"),),
        body=raw,
    )


def project(body: Any) -> c.Request:
    """Project a body and assert it accounted for itself.

    Every caller wants both, and running :func:`harness.contract.verify_total`
    here rather than in each test means no test can pass while quietly leaving a
    key unaccounted for.

    Args:
        body: The request body.

    Returns:
        The projection.
    """
    projected = r.ResponsesProjection().read_request(captured(body))
    c.verify_total(projected)

    return projected


# --------------------------------------------------------------------------
# R1 — protocol conformance
# --------------------------------------------------------------------------


class TestProtocolConformance:
    """The reader is a `Projection` for the Responses format."""

    def test_it_declares_the_responses_wire_format(self) -> None:
        """A `str` would let six authors spell one format three ways (§7.4)."""
        assert r.ResponsesProjection.wire_format is c.WireFormat.OPENAI_RESPONSES

    def test_it_satisfies_the_projection_protocol(self) -> None:
        """`isinstance`, never `issubclass` — the protocol carries a data member."""
        assert isinstance(r.ResponsesProjection(), c.Projection)

    def test_it_actually_projects_rather_than_merely_having_the_members(self) -> None:
        """The paired assertion `isinstance` cannot make.

        `contract.Projection` checks member *presence* only, never signatures,
        so the test above passes against any object with two attributes. Without
        this, "conforms to the protocol" would be satisfied by a stub.
        """
        projected = project({"model": "gpt-6-astra", "input": "hello"})

        assert projected.envelope.model == "gpt-6-astra"
        assert projected.conversation.turns == (c.Turn("user", [c.Text("hello")]),)


# --------------------------------------------------------------------------
# R2 — envelope and tool_choice
# --------------------------------------------------------------------------


class TestEnvelope:
    """Control fields land on the envelope, keyed by the wire key."""

    def test_the_three_named_control_fields(self) -> None:
        """M1 replaces `model`; P17 injects `stream` and `store`."""
        projected = project({"model": "gpt-6-astra", "input": "hi", "stream": True, "store": False})

        assert projected.envelope.model == "gpt-6-astra"
        assert projected.envelope.stream is True
        assert projected.envelope.store is False

    def test_a_format_specific_control_field_keeps_its_wire_key(self) -> None:
        """P3/P4's register rows address `envelope.extra[reasoning]` by that name."""
        projected = project({"input": "hi", "reasoning": {"effort": "high"}})

        assert projected.envelope.extra["reasoning"] == {"effort": "high"}

    def test_extra_is_keyed_and_never_nested(self) -> None:
        """§3.3.1b: the oracle compares an extra entry whole.

        Six readers cannot quietly disagree about whether `reasoning.effort` has
        an address of its own, so no reader may emit one.
        """
        projected = project(
            {
                "input": "hi",
                "reasoning": {"effort": "high", "summary": "concise"},
                "text": {"format": {"type": "json_object"}},
            }
        )

        # Asserted non-empty first: "no key contains a dot" is vacuously true of
        # an empty mapping, so the flatness check alone passes against a reader
        # that populated nothing.
        assert set(projected.envelope.extra) == {"reasoning", "text"}

        nested = [key for key in projected.envelope.extra if "." in key]
        assert nested == [], f"extra must be flat, found nested keys: {nested}"

    def test_an_absent_control_field_is_absent_rather_than_none(self) -> None:
        """P17 injects `store`, so its absence must be observable (§3.3.2 assertion 2)."""
        projected = project({"input": "hi"})

        assert projected.envelope.store is None
        assert "reasoning" not in projected.envelope.extra


class TestToolChoiceNormalisation:
    """All nine published `ToolChoiceParam` forms, onto the canonical vocabulary.

    Nine *forms* — the members of `ToolChoiceParam.oneOf` — but twenty *values*,
    because three of them carry closed enumerations of their own:
    `ToolChoiceOptions` has three strings, `ToolChoiceAllowed` two modes, and
    `ToolChoiceTypes` eight built-in tool types. Every value is exercised, not
    one representative per form, because the normalisation differs *within*
    those enumerations — `required` becomes `any` while `auto` does not.
    """

    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            # ToolChoiceOptions — the three bare strings.
            ("none", "none"),
            ("auto", "auto"),
            ("required", "any"),
            # ToolChoiceAllowed — by its own mode, not flattened to `any`.
            ({"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function"}]}, "auto"),
            ({"type": "allowed_tools", "mode": "required", "tools": []}, "any"),
            # ToolChoiceFunction and ToolChoiceCustom — by name.
            ({"type": "function", "name": "get_weather"}, "tool:get_weather"),
            ({"type": "custom", "name": "my_tool"}, "tool:my_tool"),
            # ToolChoiceMCP — by server, optionally narrowed to one tool. The
            # spelling embeds the declaration's own name (`mcp:<label>`) so the
            # selection corresponds to something `conversation.tools` holds.
            ({"type": "mcp", "server_label": "docs"}, "tool:mcp:docs"),
            ({"type": "mcp", "server_label": "docs", "name": "search"}, "tool:mcp:docs:search"),
            # ToolChoiceTypes — the eight built-ins.
            ({"type": "file_search"}, "tool:file_search"),
            ({"type": "web_search_preview"}, "tool:web_search_preview"),
            ({"type": "computer"}, "tool:computer"),
            ({"type": "computer_use_preview"}, "tool:computer_use_preview"),
            ({"type": "computer_use"}, "tool:computer_use"),
            ({"type": "web_search_preview_2025_03_11"}, "tool:web_search_preview_2025_03_11"),
            ({"type": "image_generation"}, "tool:image_generation"),
            ({"type": "code_interpreter"}, "tool:code_interpreter"),
            # The three Specific* singletons.
            ({"type": "programmatic_tool_calling"}, "tool:programmatic_tool_calling"),
            ({"type": "apply_patch"}, "tool:apply_patch"),
            ({"type": "shell"}, "tool:shell"),
        ],
    )
    def test_each_published_form_normalises(self, choice: Any, expected: str) -> None:
        """§3.3.1b R8.6 pins the value to auto / any / none / tool:<name>."""
        projected = project({"input": "hi", "tool_choice": choice})

        assert projected.envelope.extra[c.TOOL_CHOICE_KEY] == expected

    def test_allowed_tools_distinguishes_its_two_modes(self) -> None:
        """The mutation a flat `any` would hide.

        `allowed_tools` is the one object form carrying a mode of its own.
        Collapsing both to `any` would make a `mode: auto -> required` rewrite
        invisible — in the field whose job is constraining what the model may
        call.
        """
        permissive = project({"input": "hi", "tool_choice": {"type": "allowed_tools", "mode": "auto", "tools": []}})
        forced = project({"input": "hi", "tool_choice": {"type": "allowed_tools", "mode": "required", "tools": []}})

        assert permissive.envelope.extra[c.TOOL_CHOICE_KEY] != forced.envelope.extra[c.TOOL_CHOICE_KEY]

    def test_two_mcp_servers_do_not_collide(self) -> None:
        """A bare `tool:mcp` would equate two different servers."""
        first = project({"input": "hi", "tool_choice": {"type": "mcp", "server_label": "docs"}})
        second = project({"input": "hi", "tool_choice": {"type": "mcp", "server_label": "wiki"}})

        assert first.envelope.extra[c.TOOL_CHOICE_KEY] != second.envelope.extra[c.TOOL_CHOICE_KEY]

    def test_an_unrecognised_selector_residualises_rather_than_being_guessed(self) -> None:
        """Guessing would equate a shape nobody has looked at with a real selection."""
        projected = r.ResponsesProjection().read_request(
            captured({"input": "hi", "tool_choice": {"type": "telepathy"}})
        )

        assert projected.residual == {"tool_choice": {"type": "telepathy"}}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)


# --------------------------------------------------------------------------
# R3 — sampling
# --------------------------------------------------------------------------


class TestSampling:
    """Sampling normalises to the Chat Completions spelling (§3.3.1b)."""

    def test_max_output_tokens_becomes_max_tokens(self) -> None:
        """Register row P14 is written against exactly this mapping."""
        projected = project({"input": "hi", "max_output_tokens": 2048})

        assert projected.conversation.sampling == {"max_tokens": 2048}

    def test_the_parameters_already_spelled_canonically(self) -> None:
        """All four are in `SAMPLING_KEYS`, so they map by their own names."""
        projected = project(
            {
                "input": "hi",
                "temperature": 0.2,
                "top_p": 0.9,
                "top_logprobs": 5,
                "stream_options": {"include_obfuscation": False},
            }
        )

        assert projected.conversation.sampling == {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_logprobs": 5,
            "stream_options": {"include_obfuscation": False},
        }

    def test_text_is_not_folded_onto_response_format(self) -> None:
        """§3.3.1b mandates only the `max_output_tokens` rename.

        `text` is Responses' own spelling of a related but not identical
        concept; folding it would manufacture a false equality with a Chat
        Completions `response_format` and hide a real translation step.
        """
        projected = project({"input": "hi", "text": {"format": {"type": "json_object"}}})

        assert projected.envelope.extra["text"] == {"format": {"type": "json_object"}}
        assert "response_format" not in projected.conversation.sampling


# --------------------------------------------------------------------------
# R4 — system lifting
# --------------------------------------------------------------------------


class TestSystemLifting:
    """All three carriers lift into `conversation.system`, never into a turn."""

    def test_instructions_and_both_system_roles_lift_in_order(self) -> None:
        """§3.3.1b R8.2 names four carriers; Responses uses three of them."""
        projected = project(
            {
                "instructions": "You are terse.",
                "input": [
                    {"role": "developer", "content": "Prefer SI units."},
                    {"role": "system", "content": "Never apologise."},
                    {"role": "user", "content": "hi"},
                ],
            }
        )

        assert projected.conversation.system == (
            c.Text("You are terse."),
            c.Text("Prefer SI units."),
            c.Text("Never apologise."),
        )
        assert [turn.role for turn in projected.conversation.turns] == ["user"]

    def test_null_instructions_contribute_nothing(self) -> None:
        """`CreateResponse.instructions` is `anyOf[string, null]`.

        `Text.text` is unvalidated, so a naive reader would put `Text(None)` into
        `conversation.system` — a system entry carrying a `None` where a `str` is
        declared, present on one side of a diff and absent on the other.
        """
        projected = project({"instructions": None, "input": "hi"})

        assert projected.conversation.system == ()

    def test_empty_instructions_do_contribute_an_empty_part(self) -> None:
        """The deliberate asymmetry with `null`, and the reason for it.

        An empty string is a value the agent sent, so it stays observable — the
        same posture the contract takes on empty thinking blocks, where "an empty
        block is a part with an empty string, never nothing". A bridge dropping
        an empty `instructions` is then a visible delta.
        """
        projected = project({"instructions": "", "input": "hi"})

        assert projected.conversation.system == (c.Text(""),)

    def test_a_non_text_part_in_a_lifted_message_residualises(self) -> None:
        """`Conversation.system` is enforced Text-only, and a developer message may carry an image.

        Residualising fails closed. Letting the contract's `TypeError` out would
        add a **fourth** failure shape beside the three §3.3.1 defines on
        purpose, and T-D1 could not classify it.
        """
        projected = r.ResponsesProjection().read_request(
            captured(
                {
                    "input": [
                        {
                            "role": "developer",
                            "content": [
                                {"type": "input_text", "text": "Use this diagram."},
                                {"type": "input_image", "image_url": "https://example.test/d.png"},
                            ],
                        }
                    ]
                }
            )
        )

        assert projected.conversation.system == (c.Text("Use this diagram."),)
        assert set(projected.residual) == {"input[0].content[1]"}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_the_lifted_residual_is_keyed_by_the_wire_index_not_the_projected_offset(self) -> None:
        """Two unclassifiable parts must produce two residual entries, correctly named.

        The projected list is shorter than the wire list as soon as anything
        residualises, so indexing the residual by the projected offset both
        names the wrong element *and* collides with the key the content reader
        already wrote — silently destroying one of the two values. The offending
        entry is deliberately first, so the two indices cannot coincide by luck.
        """
        projected = r.ResponsesProjection().read_request(
            captured(
                {
                    "input": [
                        {
                            "role": "developer",
                            "content": [
                                {"type": "unheard_of", "x": 1},
                                {"type": "input_image", "image_url": "https://example.test/a.png"},
                            ],
                        }
                    ]
                }
            )
        )

        assert set(projected.residual) == {"input[0].content[0]", "input[0].content[1]"}

        # The wire value is stored, not the projected object: every other
        # residual site stores what was on the wire, and T-D8 compares them.
        assert projected.residual["input[0].content[0]"] == {"type": "unheard_of", "x": 1}
        assert projected.residual["input[0].content[1]"] == {
            "type": "input_image",
            "image_url": "https://example.test/a.png",
        }


# --------------------------------------------------------------------------
# R5 — the two published `input` shapes
# --------------------------------------------------------------------------


class TestInputShapes:
    """`InputParam` is `oneOf[string, array]`, and both are real traffic."""

    def test_a_bare_string_is_one_user_turn(self) -> None:
        """The schema calls it "equivalent to a text input with the user role"."""
        projected = project({"input": "Tell me a story."})

        assert projected.conversation.turns == (c.Turn("user", [c.Text("Tell me a story.")]),)

    def test_the_string_and_array_forms_agree(self) -> None:
        """The same conversation, spelled two ways the schema both permits.

        KBR-144 is a live 500 on the string form, so this is not hypothetical.
        The array fixture also carries no `type`, which is the `EasyInputMessage`
        dispatch case.
        """
        from_string = project({"input": "hello"})
        from_array = project({"input": [{"role": "user", "content": "hello"}]})

        # Pinned to a concrete expectation for the same reason as P16's guard:
        # two empty conversations are also equal.
        expected = c.Conversation(turns=[c.Turn("user", [c.Text("hello")])])
        assert from_string.conversation == expected
        assert from_array.conversation == expected


# --------------------------------------------------------------------------
# R6 / R7 — items and content parts
# --------------------------------------------------------------------------


class TestP16StaysInvisible:
    """The guard `contract.py` delegates to this module by name."""

    def test_input_text_and_output_text_project_identically(self) -> None:
        """Register row P16 is `NOT_PROJECTABLE`, and this is what makes it so.

        P16 rewrites `input_text` to `output_text` unconditionally on the
        Responses-origin path. The tag is redundant with the turn's role, so
        carrying it would put one vendor's spelling into a form whose purpose is
        wire independence — and would make a deliberate, registered mutation
        show as an unclaimed delta on every request.
        """
        inbound = project({"input": [{"role": "assistant", "content": [{"type": "input_text", "text": "ok"}]}]})
        upstream = project({"input": [{"role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]})

        # Asserted against a concrete expectation, not only against each other:
        # a reader returning an empty conversation for both would satisfy the
        # equality while projecting nothing. `contract.py` names this test by
        # name as one of P16's two guards, so a vacuous version of it is the
        # worst place for that shape.
        expected = c.Conversation(turns=[c.Turn("assistant", [c.Text("ok")])])
        assert inbound.conversation == expected
        assert upstream.conversation == expected


class TestContentParts:
    """Each published content type projects to its grammar counterpart."""

    def test_a_refusal_is_not_collapsed_into_text(self) -> None:
        """Unlike P16's tag, refusal-ness is not recoverable from the role.

        An assistant refusal and an assistant answer would project identically,
        so a bridge that turned one into the other would be invisible.
        """
        refusal = project({"input": [{"role": "assistant", "content": [{"type": "refusal", "refusal": "no"}]}]})
        answer = project({"input": [{"role": "assistant", "content": [{"type": "output_text", "text": "no"}]}]})

        assert refusal.conversation != answer.conversation
        assert refusal.conversation.turns[0].parts == (c.Opaque("refusal", digest=c.text_digest("no")),)

    def test_a_rewritten_refusal_is_visible(self) -> None:
        """`kind` alone would make a rewritten refusal invisible.

        `Opaque` carries a digest for exactly this: naming the *kind* restores
        refusal-ness, but without the digest a bridge that changed the refusal's
        wording would still project identically.
        """
        original = project({"input": [{"role": "assistant", "content": [{"type": "refusal", "refusal": "no"}]}]})
        rewritten = project(
            {
                "input": [
                    {
                        "role": "assistant",
                        "content": [{"type": "refusal", "refusal": "absolutely not"}],
                    }
                ]
            }
        )

        assert original.conversation != rewritten.conversation

    def test_an_input_file_is_opaque_rather_than_dropped(self) -> None:
        """`Opaque` keeps content the grammar cannot express *detectable*."""
        projected = project(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_file", "file_id": "file-123"}],
                    }
                ]
            }
        )

        assert projected.conversation.turns[0].parts == (c.Opaque("document"),)

    def test_a_data_url_image_is_digested(self) -> None:
        """`image_digest` is pinned so six readers agree on one image.

        A reader that left the data URL in `ref` while T-A2 decoded it would
        reproduce exactly the disagreement the pinning exists to prevent.
        """
        raw = b"\x89PNG\r\n\x1a\nfake"
        url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")

        projected = project({"input": [{"role": "user", "content": [{"type": "input_image", "image_url": url}]}]})

        assert projected.conversation.turns[0].parts == (
            c.Image(digest=hashlib.sha256(raw).hexdigest(), media_type="image/png"),
        )

    def test_the_media_type_is_excluded_from_the_digest(self) -> None:
        """So a changed media type is its own delta, not an unexplained digest change."""
        raw = b"bytes"
        payload = base64.b64encode(raw).decode("ascii")

        as_png = project(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": f"data:image/png;base64,{payload}"}],
                    }
                ]
            }
        )
        as_jpeg = project(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": f"data:image/jpeg;base64,{payload}"}],
                    }
                ]
            }
        )

        first = as_png.conversation.turns[0].parts[0]
        second = as_jpeg.conversation.turns[0].parts[0]
        assert isinstance(first, c.Image) and isinstance(second, c.Image)
        assert first.digest == second.digest
        assert first.media_type != second.media_type

    def test_a_remote_image_carries_its_uri_instead_of_a_digest(self) -> None:
        """There are no bytes to digest — the shape §3.3.1 gives Gemini's `fileUri`."""
        projected = project(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": "https://example.test/cat.png"}],
                    }
                ]
            }
        )

        assert projected.conversation.turns[0].parts == (c.Image(ref="https://example.test/cat.png"),)

    def test_a_data_url_that_is_not_base64_residualises_rather_than_becoming_a_ref(self) -> None:
        """Putting a data URL in `ref` is the shape the pinned digest exists to prevent.

        Another reader decoding the same bytes would produce a digest, and the
        two projections would then differ on an unchanged image — a permanent
        unclaimed delta that no register row could ever explain.
        """
        projected = r.ResponsesProjection().read_request(
            captured(
                {
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_image", "image_url": "data:image/png,abc"}],
                        }
                    ]
                }
            )
        )

        assert set(projected.residual) == {"input[0].content[0]"}

    def test_an_image_given_only_by_file_id_carries_the_id_as_its_reference(self) -> None:
        """`InputImageContent` permits `file_id` instead of `image_url`."""
        projected = project(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_image", "file_id": "file-9", "detail": "auto"}],
                    }
                ]
            }
        )

        assert projected.conversation.turns[0].parts == (c.Image(ref="file-9"),)


class TestFunctionCall:
    """`function_call` items and their JSON-string arguments."""

    def test_a_function_call_projects_with_its_call_id(self) -> None:
        """`ToolUse.id` pairs the call with its result."""
        projected = project(
            {
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"location": "Boston"}',
                    }
                ]
            }
        )

        assert projected.conversation.turns == (
            c.Turn(
                "assistant",
                [c.ToolUse(name="get_weather", arguments={"location": "Boston"}, id="call_1")],
            ),
        )

    def test_an_empty_arguments_string_is_no_arguments(self) -> None:
        """Real corpus traffic, not a hypothetical.

        kitty's own Responses builder writes `arguments` as `""` whenever a Chat
        Completions tool call carried none, so `json.loads("")` would raise on a
        body the bridge itself produces.
        """
        projected = project({"input": [{"type": "function_call", "call_id": "c", "name": "ping", "arguments": ""}]})

        call = projected.conversation.turns[0].parts[0]
        assert isinstance(call, c.ToolUse)
        assert call.arguments == {}

    @pytest.mark.parametrize("arguments", ["[1, 2]", '"a string"', "{oops"])
    def test_arguments_that_are_not_a_json_object_residualise_without_raising(self, arguments: str) -> None:
        """Three shapes the contract would turn into a misdiagnosis.

        `_freeze_mapping` does `dict(value or {})`, so a list raises `ValueError`
        and a string `TypeError` — and `contract` defines a reader-raised
        `ValueError` as "the reader mis-routed a field", i.e. a reader bug rather
        than a bad body. Failing closed into the residual keeps the diagnosis
        honest.
        """
        projected = r.ResponsesProjection().read_request(
            captured(
                {
                    "input": [
                        {
                            "type": "function_call",
                            "call_id": "c",
                            "name": "ping",
                            "arguments": arguments,
                        }
                    ]
                }
            )
        )

        call = projected.conversation.turns[0].parts[0]
        assert isinstance(call, c.ToolUse)
        assert call.arguments == {}
        assert projected.residual == {"input[0].arguments": arguments}

    def test_arguments_sent_as_an_object_rather_than_a_string_residualise(self) -> None:
        """`FunctionToolCall.arguments` is a *string* in the published schema.

        Accepting the decoded object would make a bridge that emitted the object
        form instead of the string invisible — a wire-format breach this oracle
        exists to see. It is also the shape that hides a string↔object rewrite:
        both would project identically.
        """
        projected = r.ResponsesProjection().read_request(
            captured(
                {
                    "input": [
                        {
                            "type": "function_call",
                            "call_id": "c",
                            "name": "ping",
                            "arguments": {"a": 1},
                        }
                    ]
                }
            )
        )

        assert projected.residual == {"input[0].arguments": {"a": 1}}

    def test_a_wrongly_typed_call_id_or_name_residualises_rather_than_being_coerced(self) -> None:
        """`str(7)` and `str(None)` invent a value the agent never sent.

        `verify_total` cannot see a nested coercion, because `consumed` covers
        top-level keys only — so a silent `"None"` would be a projection the
        oracle trusts and nobody can falsify.
        """
        projected = r.ResponsesProjection().read_request(
            captured({"input": [{"type": "function_call", "call_id": 7, "name": None, "arguments": "{}"}]})
        )

        assert set(projected.residual) == {"input[0].call_id", "input[0].name"}


class TestFunctionCallOutput:
    """`function_call_output` in both published `output` shapes."""

    def test_a_json_string_output_keeps_its_structure(self) -> None:
        """`Json` exists because a structured result is the common case."""
        projected = project({"input": [{"type": "function_call_output", "call_id": "c1", "output": '{"temp": 12}'}]})

        assert projected.conversation.turns == (
            c.Turn("user", [c.ToolResult(content=[c.Json({"temp": 12})], tool_use_id="c1")]),
        )

    def test_a_prose_output_stays_text(self) -> None:
        """The schema calls `output` a JSON string, but tools return prose."""
        projected = project({"input": [{"type": "function_call_output", "call_id": "c1", "output": "it is cold"}]})

        result = projected.conversation.turns[0].parts[0]
        assert isinstance(result, c.ToolResult)
        assert result.content == (c.Text("it is cold"),)

    def test_the_array_output_branch_projects_all_three_published_types_in_order(self) -> None:
        """`output` is `oneOf[string, array]`, and the array carries three types.

        The image leg matters on its own: R6a's digest rules must apply here too,
        or a tool returning a screenshot would project differently depending on
        where in the body it appeared.
        """
        raw = b"screenshot-bytes"
        payload = base64.b64encode(raw).decode("ascii")

        projected = project(
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "c1",
                        "output": [
                            {"type": "input_text", "text": "see attached"},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{payload}",
                            },
                            {"type": "input_file", "file_id": "file-1"},
                        ],
                    }
                ]
            }
        )

        result = projected.conversation.turns[0].parts[0]
        assert isinstance(result, c.ToolResult)
        assert result.content == (
            c.Text("see attached"),
            c.Image(digest=hashlib.sha256(raw).hexdigest(), media_type="image/png"),
            c.Opaque("document"),
        )

    def test_a_content_type_the_output_branch_does_not_publish_residualises(self) -> None:
        """The array's union is narrower than a message's.

        `FunctionCallOutputItemParam.output` publishes only `input_text`,
        `input_image` and `input_file` — no `refusal`, no `output_text`.
        Accepting a message's full set here would let a shape the schema forbids
        pass unremarked.
        """
        projected = r.ResponsesProjection().read_request(
            captured(
                {
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "c1",
                            "output": [{"type": "refusal", "refusal": "no"}],
                        }
                    ]
                }
            )
        )

        assert set(projected.residual) == {"input[0].output[0]"}

    def test_is_error_is_always_false_because_the_format_carries_no_flag(self) -> None:
        """Written down as a decision rather than left as silence."""
        projected = project({"input": [{"type": "function_call_output", "call_id": "c1", "output": "boom"}]})

        result = projected.conversation.turns[0].parts[0]
        assert isinstance(result, c.ToolResult)
        assert result.is_error is False

    def test_an_output_without_a_call_id_still_projects(self) -> None:
        """`call_id` is not required by the schema, and KBR-159 saw this shape live."""
        projected = project({"input": [{"type": "function_call_output", "output": "ok"}]})

        result = projected.conversation.turns[0].parts[0]
        assert isinstance(result, c.ToolResult)
        assert result.tool_use_id is None


class TestReasoning:
    """One `Thinking` part per text entry, so a dropped entry is a visible delta."""

    def test_summary_then_content_in_wire_order_with_the_signature_on_the_first(self) -> None:
        """The fan-out the alternative would blur.

        Collapsing the item into one joined part would show a dropped summary
        entry as *changed text* rather than a missing part, and would require
        inventing a separator that is not on the wire.
        """
        projected = project(
            {
                "input": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [
                            {"type": "summary_text", "text": "Weighing A"},
                            {"type": "summary_text", "text": "Then B"},
                        ],
                        "content": [{"type": "reasoning_text", "text": "full trace"}],
                        "encrypted_content": "gAAAA",
                    }
                ]
            }
        )

        assert projected.conversation.turns == (
            c.Turn(
                "assistant",
                [
                    c.Thinking("Weighing A", signature="gAAAA"),
                    c.Thinking("Then B"),
                    c.Thinking("full trace"),
                ],
            ),
        )

    def test_dropping_one_summary_entry_is_visible(self) -> None:
        """The property the fan-out exists to give the oracle."""
        both = project(
            {
                "input": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [{"text": "one"}, {"text": "two"}],
                    }
                ]
            }
        )
        one = project({"input": [{"type": "reasoning", "id": "rs_1", "summary": [{"text": "one"}]}]})

        assert both.conversation != one.conversation

    @pytest.mark.parametrize(
        ("summary", "expected_texts", "expected_residual"),
        [
            pytest.param(
                [{"text": "real"}, {"text": 5}],
                ["real"],
                {"input[0].summary[1]"},
                id="one-good-one-ill-typed",
            ),
            pytest.param([{"text": 5}], [], {"input[0].summary[0]"}, id="only-ill-typed"),
            pytest.param(["plain string"], [], {"input[0].summary[0]"}, id="entry-not-a-dict"),
            pytest.param("oops", [], {"input[0].summary"}, id="summary-not-a-list"),
        ],
    )
    def test_an_unreadable_reasoning_entry_residualises_instead_of_vanishing(
        self, summary: Any, expected_texts: list[str], expected_residual: set[str]
    ) -> None:
        """Skipping a bad entry performs the exact loss the fan-out exists to expose.

        The first case is the sharp one: the reader would project a single
        `Thinking("real")` and say nothing about the second entry — which is
        precisely what `test_dropping_one_summary_entry_is_visible` is there to
        catch a *bridge* doing. `consumed` is top-level only, so `verify_total`
        could never see it.

        It is also the `{}`-is-a-lie shape R6d forbids for a modelled type: a
        lone `Thinking("")` asserts "an empty reasoning block" about an item that
        carried content.
        """
        projected = r.ResponsesProjection().read_request(
            captured({"input": [{"type": "reasoning", "id": "r1", "summary": summary}]})
        )

        thinking = [part.text for part in projected.conversation.turns[0].parts]
        assert thinking == (expected_texts or [""])
        assert set(projected.residual) == expected_residual
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_an_item_with_no_text_still_projects_one_empty_part(self) -> None:
        """Keep an empty reasoning block observable as a part.

        A reasoning item carrying only an encrypted blob would otherwise vanish,
        and its presence would stop being observable.
        """
        projected = project({"input": [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "x"}]})

        assert projected.conversation.turns == (c.Turn("assistant", [c.Thinking("", signature="x")]),)


class TestOpaqueItems:
    """The 27 published item types the grammar does not model semantically."""

    @pytest.mark.parametrize("kind", sorted(r._OPAQUE_USER_ITEMS))
    def test_a_client_supplied_item_lands_in_a_user_turn(self, kind: str) -> None:
        """Because paths are index-based, one wrong role corrupts every later path."""
        projected = project({"input": [{"type": kind}]})

        assert projected.conversation.turns == (c.Turn("user", [c.Opaque(kind)]),)

    @pytest.mark.parametrize("kind", sorted(r._OPAQUE_ASSISTANT_ITEMS))
    def test_a_model_produced_item_lands_in_an_assistant_turn(self, kind: str) -> None:
        """The other half of the role table, asserted per type rather than in prose."""
        projected = project({"input": [{"type": kind}]})

        assert projected.conversation.turns == (c.Turn("assistant", [c.Opaque(kind)]),)

    def test_a_null_type_counts_as_absent_rather_than_as_an_unknown_type(self) -> None:
        """`ItemReferenceParam.type` is `anyOf[enum, null]`, so this is legal traffic.

        A reader dispatching on `"type" in item` rather than on the value being a
        string would residualise this and fail the run on a valid body. The
        `type`-absent test cannot catch that regression; this one can.
        """
        projected = project({"input": [{"id": "msg_1", "type": None}]})

        assert projected.conversation.turns == (c.Turn("user", [c.Opaque("item_reference")]),)

    def test_additional_tools_lands_in_a_user_turn_despite_declaring_developer(self) -> None:
        """The named exception whose reasoning is hardest to guess from the rule.

        `AdditionalToolsItemParam` is the only opaque item carrying a `role` of
        its own, and its sole published value is `developer` — which everywhere
        else in this reader lifts into `conversation.system`. It cannot here:
        `Conversation.system` is enforced `Text`-only and an `Opaque` has no home
        in it. So the item lands in a `user` turn, on the ground that
        `developer` is client-supplied.

        Worth its own test rather than only the parametrised sweep, because a
        future reader seeing `role: "developer"` would reasonably try to lift it
        and would then be changing turn indices for every later turn.
        """
        projected = project({"input": [{"type": "additional_tools", "role": "developer"}]})

        assert projected.conversation.system == ()
        assert projected.conversation.turns == (c.Turn("user", [c.Opaque("additional_tools")]),)

    def test_an_item_reference_is_recognised_without_a_type(self) -> None:
        """`ItemReferenceParam.type` is nullable, so `{"id": ...}` alone is legal."""
        projected = project({"input": [{"id": "msg_1"}]})

        assert projected.conversation.turns == (c.Turn("user", [c.Opaque("item_reference")]),)

    def test_the_closed_sets_are_internally_consistent_and_sized_as_recorded(self) -> None:
        """A guard on the module's own tables — **not** on OpenAI's schema.

        Worth being exact about, because the obvious reading is wrong and an
        earlier version of this docstring made the claim: the suite has no
        network and vendors no copy of `openapi.yaml`, so nothing here can
        notice OpenAI *adding* an item type. What this catches is an edit to
        this module — a type moved between the role tables, or dropped — which
        is the direction a maintainer can cause.

        **Spec drift is caught elsewhere, and loudly.** A type outside these
        sets residualises, and a non-empty residual fails the run (§3.3.1) — so
        the first request carrying a new item type fails at the point it
        matters, rather than being quietly absorbed. That is the designed
        behaviour, not a gap: §3.3.1 wants a new wire field to *force a
        deliberate decision*. `SCHEMA_VERSION` records which revision these
        tables were derived from so the decision has a starting point.
        """
        assert len(r.PUBLISHED_ITEM_TYPES) == 31
        assert len(r._OPAQUE_USER_ITEMS) == 12
        assert len(r._OPAQUE_ASSISTANT_ITEMS) == 15
        assert r._OPAQUE_USER_ITEMS.isdisjoint(r._OPAQUE_ASSISTANT_ITEMS)

        # The modelled four and the opaque twenty-seven must partition the set,
        # or a type could be both modelled and placeheld and the dispatch order
        # would silently decide which wins.
        assert r._MODELLED_ITEMS.isdisjoint(r._OPAQUE_USER_ITEMS | r._OPAQUE_ASSISTANT_ITEMS)
        assert len(r._MODELLED_ITEMS) + len(r._OPAQUE_USER_ITEMS) + len(r._OPAQUE_ASSISTANT_ITEMS) == 31

    def test_the_mechanical_role_rule_reproduces_the_table(self) -> None:
        """The rule in the docstring and the table must not drift apart.

        The rule is "a type ending `_output`, plus five named exceptions"; if
        somebody adds a type to one and not the other, the prose stops
        describing the code.
        """
        exceptions = {
            "mcp_approval_response",
            "compaction_trigger",
            "additional_tools",
            "item_reference",
            "configuration_update",
        }
        by_rule = {k for k in r.PUBLISHED_ITEM_TYPES if k.endswith("_output")} | exceptions

        assert by_rule - r._MODELLED_ITEMS == r._OPAQUE_USER_ITEMS


# --------------------------------------------------------------------------
# R8 — tool declarations
# --------------------------------------------------------------------------


class TestToolDeclarations:
    """`tools[]`, addressed by name because translators reorder them (§3.3.1a)."""

    def test_a_function_tool_projects_every_field(self) -> None:
        """One oracle falsification case deletes the description (§3.3.1)."""
        schema = {"type": "object", "properties": {"location": {"type": "string"}}}
        projected = project(
            {
                "input": "hi",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": schema,
                        "strict": True,
                    }
                ],
            }
        )

        assert projected.conversation.tools == (
            c.ToolDecl(name="get_weather", description="Get the weather", schema=schema, strict=True),
        )

    @pytest.mark.parametrize(
        ("declared", "expected"),
        [({"strict": True}, True), ({"strict": False}, False), ({"strict": None}, None), ({}, None)],
    )
    def test_strict_is_a_four_case_tri_state(self, declared: dict[str, Any], expected: bool | None) -> None:
        """`FunctionTool.required` includes `strict`, typed `anyOf[boolean, null]`.

        So against the published schema "no strict" is spelled `null`, not
        omission — and register row P15 strips this field, so `None` must stay
        distinct from `False` or that row's presence and absence become
        indistinguishable.
        """
        projected = project({"input": "hi", "tools": [{"type": "function", "name": "f", **declared}]})

        assert projected.conversation.tools[0].strict is expected

    def test_a_wrongly_typed_description_or_schema_residualises(self) -> None:
        """Both fields carry register weight, so a silent `None` is dangerous.

        P15 strips `strict` and §3.3.1's oracle falsification set deletes a tool
        description — so a reader that quietly turned a malformed description
        into `None` would produce exactly the shape those checks exist to catch,
        with nothing to distinguish it from the real mutation.
        """
        projected = r.ResponsesProjection().read_request(
            captured(
                {
                    "input": "hi",
                    "tools": [
                        {
                            "type": "function",
                            "name": "f",
                            "description": 123,
                            "parameters": "not-an-object",
                        }
                    ],
                }
            )
        )

        assert set(projected.residual) == {"tools[0].description", "tools[0].parameters"}

    def test_a_function_tool_without_a_name_residualises(self) -> None:
        """The same rule as a `function_call`, for a stronger reason.

        `FunctionTool.required` includes `name`, and §3.3.1a addresses tools by
        name with no index to fall back on. Two unnamed declarations would both
        sit at `conversation.tools[]` — and `path_matches` accepts `[]` as the
        **legacy wildcard spelling**, so a register row anchored at
        `conversation.tools[*].strict` would match them by accident rather than
        by name. That is the collision the MCP naming rule exists to prevent,
        reached by a different route.
        """
        projected = r.ResponsesProjection().read_request(
            captured({"input": "hi", "tools": [{"type": "function", "parameters": {}}]})
        )

        assert set(projected.residual) == {"tools[0].name"}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_a_custom_tool_is_named_by_its_own_name(self) -> None:
        """`CustomToolParam` makes `name` required; using `type` would discard it."""
        projected = project({"input": "hi", "tools": [{"type": "custom", "name": "run_python"}]})

        assert projected.conversation.tools[0].name == "run_python"

    def test_two_mcp_servers_get_distinct_names(self) -> None:
        """Naming both `mcp` would put two declarations at one path.

        §3.3.1a addresses tools by name *because* positions are unstable, so a
        name collision is not recoverable by falling back to an index.
        """
        projected = project(
            {
                "input": "hi",
                "tools": [
                    {"type": "mcp", "server_label": "docs"},
                    {"type": "mcp", "server_label": "wiki"},
                ],
            }
        )

        names = [tool.name for tool in projected.conversation.tools]
        assert names == ["mcp:docs", "mcp:wiki"]
        assert len(set(names)) == 2

    def test_a_builtin_tool_keeps_its_whole_declaration_as_the_schema(self) -> None:
        """So a changed `vector_store_ids` is still a visible delta."""
        declaration = {"type": "file_search", "vector_store_ids": ["vs_1"], "max_num_results": 20}
        projected = project({"input": "hi", "tools": [declaration]})

        assert projected.conversation.tools[0].name == "file_search"
        assert projected.conversation.tools[0].schema == declaration


# --------------------------------------------------------------------------
# R9 — turn merging
# --------------------------------------------------------------------------


class TestTurnMerging:
    """§3.3.1b's merge rule, and the reordering it must not do."""

    def test_a_run_of_tool_results_forms_one_user_turn(self) -> None:
        """The standard exchange ends `assistant(calls) -> output -> output`.

        A "lift into the following user turn" rule does not work, because there
        is no following user message at all.
        """
        projected = project(
            {
                "input": [
                    {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"},
                    {"type": "function_call", "call_id": "b", "name": "g", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "a", "output": "1"},
                    {"type": "function_call_output", "call_id": "b", "output": "2"},
                ]
            }
        )

        roles = [turn.role for turn in projected.conversation.turns]
        assert roles == ["assistant", "user"]
        assert len(projected.conversation.turns[0].parts) == 2
        assert len(projected.conversation.turns[1].parts) == 2

    def test_a_following_user_message_merges_into_the_run(self) -> None:
        """Merge an immediately following non-tool user message into the run."""
        projected = project(
            {
                "input": [
                    {"type": "function_call_output", "call_id": "a", "output": "1"},
                    {"role": "user", "content": "and now?"},
                ]
            }
        )

        assert len(projected.conversation.turns) == 1
        parts = projected.conversation.turns[0].parts
        assert isinstance(parts[0], c.ToolResult)
        assert parts[1] == c.Text("and now?")

    def test_a_preceding_user_message_is_not_reordered_behind_the_result(self) -> None:
        """The reordering §3.3.1b's rule must not be read into.

        Hoisting results ahead of text the agent sent *first* would invent a
        delta on every such turn — and because paths are index-based, a
        disagreement about turn boundaries reports a delta on every subsequent
        turn as well.
        """
        projected = project(
            {
                "input": [
                    {"role": "user", "content": "here is context"},
                    {"type": "function_call_output", "call_id": "a", "output": "1"},
                ]
            }
        )

        assert len(projected.conversation.turns) == 1
        parts = projected.conversation.turns[0].parts
        assert parts[0] == c.Text("here is context")
        assert isinstance(parts[1], c.ToolResult)

    def test_an_interleaved_result_text_result_sequence_keeps_wire_order(self) -> None:
        """The case where the two readings of §3.3.1b's merge rule diverge.

        All three items are `user`, so they merge into one turn. Read literally,
        "`ToolResult` parts come first" would hoist both results ahead of the
        text and produce `[TR, TR, Text]`; this reader preserves wire order and
        produces `[TR, Text, TR]`.

        The clause is pinned here because it is a *shared* rule — T-A2 reading
        §3.3.1b literally would disagree, and because paths are index-based a
        turn-boundary or ordering disagreement reports a delta on every
        subsequent turn. Raised for the design owner as open item 4a.
        """
        projected = project(
            {
                "input": [
                    {"type": "function_call_output", "call_id": "a", "output": "1"},
                    {"role": "user", "content": "and now?"},
                    {"type": "function_call_output", "call_id": "b", "output": "2"},
                ]
            }
        )

        assert len(projected.conversation.turns) == 1
        parts = projected.conversation.turns[0].parts
        assert [type(part).__name__ for part in parts] == ["ToolResult", "Text", "ToolResult"]

    def test_an_orphan_tool_result_projects_rather_than_raising(self) -> None:
        """Register row M7 exists to *drop* orphans.

        A reader that raised on one would fail the run instead of producing the
        delta that names the mutation.
        """
        projected = project({"input": [{"type": "function_call_output", "call_id": "nobody", "output": "1"}]})

        # What it projects *to* is asserted, not merely that nothing raised: a
        # reader that swallowed the item would also "not raise".
        assert projected.conversation.turns == (
            c.Turn("user", [c.ToolResult(content=[c.Json(1)], tool_use_id="nobody")]),
        )

    def test_consecutive_same_role_messages_merge(self) -> None:
        """Otherwise every later turn index shifts against another reader's."""
        projected = project(
            {
                "input": [
                    {"role": "user", "content": "one"},
                    {"role": "user", "content": "two"},
                    {"role": "assistant", "content": [{"type": "output_text", "text": "three"}]},
                ]
            }
        )

        assert projected.conversation.turns == (
            c.Turn("user", [c.Text("one"), c.Text("two")]),
            c.Turn("assistant", [c.Text("three")]),
        )


# --------------------------------------------------------------------------
# R10 — totality
# --------------------------------------------------------------------------

#: Every published top-level key, with the bucket and path it must land in.
#: Table-driven because the bare totality check is satisfied by a reader that
#: dumps everything into `extra` and claims `consumed=frozenset(body)` — it
#: checks bookkeeping, not classification.
_TOP_LEVEL_EXPECTATIONS: tuple[tuple[str, Any, str], ...] = (
    ("model", "gpt-6-astra", "envelope.model"),
    ("stream", True, "envelope.stream"),
    ("store", False, "envelope.store"),
    ("max_output_tokens", 512, "conversation.sampling[max_tokens]"),
    ("temperature", 0.5, "conversation.sampling[temperature]"),
    ("top_p", 0.8, "conversation.sampling[top_p]"),
    ("top_logprobs", 3, "conversation.sampling[top_logprobs]"),
    ("stream_options", {"include_obfuscation": True}, "conversation.sampling[stream_options]"),
    ("input", "hi", "conversation.turns"),
    ("instructions", "be terse", "conversation.system"),
    ("tools", [], "conversation.tools"),
    ("tool_choice", "auto", "envelope.extra[tool_choice]"),
    ("background", False, "envelope.extra[background]"),
    ("context_management", [], "envelope.extra[context_management]"),
    ("conversation", "conv_1", "envelope.extra[conversation]"),
    ("include", [], "envelope.extra[include]"),
    ("max_tool_calls", 4, "envelope.extra[max_tool_calls]"),
    ("metadata", {"k": "v"}, "envelope.extra[metadata]"),
    ("moderation", {}, "envelope.extra[moderation]"),
    ("parallel_tool_calls", True, "envelope.extra[parallel_tool_calls]"),
    ("previous_response_id", "resp_1", "envelope.extra[previous_response_id]"),
    ("prompt", {"id": "p_1"}, "envelope.extra[prompt]"),
    ("prompt_cache_key", "ck", "envelope.extra[prompt_cache_key]"),
    ("prompt_cache_options", {}, "envelope.extra[prompt_cache_options]"),
    ("prompt_cache_retention", "24h", "envelope.extra[prompt_cache_retention]"),
    ("reasoning", {"effort": "high"}, "envelope.extra[reasoning]"),
    ("safety_identifier", "sid", "envelope.extra[safety_identifier]"),
    ("service_tier", "auto", "envelope.extra[service_tier]"),
    ("text", {"format": {"type": "text"}}, "envelope.extra[text]"),
    ("truncation", "auto", "envelope.extra[truncation]"),
    ("user", "u_1", "envelope.extra[user]"),
)


def _resolve(projected: c.Request, path: str) -> Any:
    """Read the value a projection carries at one of the paths above.

    Args:
        projected: The projection.
        path: One of the path strings in :data:`_TOP_LEVEL_EXPECTATIONS`.

    Returns:
        The value found there.
    """
    found = re.fullmatch(r"(\w+)\.(\w+)(?:\[(\w+)\])?", path)
    assert found, f"unparsable expectation path {path!r}"

    section, field_name, key = found.groups()
    holder = getattr(projected.envelope if section == "envelope" else projected.conversation, field_name)

    return holder[key] if key else holder


class TestTotality:
    """Every published key is classified, and into the right bucket."""

    def test_the_table_covers_the_published_schema_exactly(self) -> None:
        """A guard on the guard: a key missing from the table is untested."""
        assert {key for key, _, _ in _TOP_LEVEL_EXPECTATIONS} == set(r.PUBLISHED_TOP_LEVEL_KEYS)
        assert len(r.PUBLISHED_TOP_LEVEL_KEYS) == 31

    def test_a_body_using_every_published_key_is_total(self) -> None:
        """`verify_total` passing is necessary but nowhere near sufficient."""
        body = {key: value for key, value, _ in _TOP_LEVEL_EXPECTATIONS}
        projected = project(body)

        assert projected.residual == {}
        assert projected.consumed == set(body)

    @pytest.mark.parametrize(
        ("key", "value", "path"),
        [(key, value, path) for key, value, path in _TOP_LEVEL_EXPECTATIONS],
    )
    def test_each_published_key_lands_at_its_stated_path(self, key: str, value: Any, path: str) -> None:
        """The companion the bare totality check needs.

        Without this, a reader that swept all 31 keys into `envelope.extra` and
        claimed `consumed=frozenset(body)` would pass every totality assertion
        while classifying nothing — and M1, the model override that is the
        product's whole purpose, would be unreadable.
        """
        body = {key: value for key, value, _ in _TOP_LEVEL_EXPECTATIONS}
        projected = project(body)

        found = _resolve(projected, path)

        # The three collection paths are asserted by their own tests; here it is
        # enough that the key reached a populated collection rather than `extra`.
        if path in {"conversation.turns", "conversation.system", "conversation.tools"}:
            assert key not in projected.envelope.extra
        else:
            assert found == value

    def test_a_key_stays_consumed_when_a_value_beneath_it_residualises(self) -> None:
        """The rule six readers and T-D8 must agree on, pinned by a test.

        `verify_total` computes `dropped = source - (consumed | residual)`, and
        `"input[0].arguments"` is not the key `"input"`. So a reader that removed
        `input` from `consumed` because something beneath it residualised would
        be reported as having *dropped* `input` — the wrong diagnosis for a
        reader that read the body fine and could not classify one field.

        For a *top-level* residual either choice works, because a key claimed in
        both accounts counts as a residual. For a nested one only this does.
        """
        projected = r.ResponsesProjection().read_request(
            captured({"input": [{"type": "function_call", "call_id": "c", "name": "f", "arguments": "{oops"}]})
        )

        assert "input" in projected.consumed
        assert set(projected.residual) == {"input[0].arguments"}

        # The diagnosis must be "could not classify", never "dropped".
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_source_carries_the_body_the_reader_parsed(self) -> None:
        """So `verify_total` needs no second parse that could disagree about duplicates."""
        body = {"model": "m", "input": "hi"}
        projected = project(body)

        assert dict(projected.source) == body


# --------------------------------------------------------------------------
# R11 — failure shapes
# --------------------------------------------------------------------------


class TestFailureShapes:
    """One exception type across six readers, so T-D1 can classify a failure."""

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param(b"not json", id="not-json"),
            pytest.param(b"[]", id="json-but-not-an-object"),
            pytest.param(b'{"input": 7}', id="input-neither-string-nor-array"),
            pytest.param(b'{"input": [{"role": "tool", "content": "x"}]}', id="role-no-format-defines"),
        ],
    )
    def test_an_unreadable_body_raises_the_contract_s_own_error(self, body: bytes) -> None:
        """§3.3.1 keeps three failure shapes distinct on purpose.

        `UnreadableBodyError` means *the body* is wrong; a `ValueError` escaping
        a reader means the reader mis-routed a field, which is a different
        diagnosis. Handing a bad role straight to `Turn(role=...)` would raise
        the wrong one.
        """
        with pytest.raises(c.UnreadableBodyError):
            r.ResponsesProjection().read_request(captured(body))

    @pytest.mark.parametrize(
        ("body", "expected_key"),
        [
            pytest.param(
                {"tool_choice": {"type": "allowed_tools", "mode": ["a"]}},
                "tool_choice",
                id="tool-choice-mode",
            ),
            pytest.param(
                {"tool_choice": {"type": ["function"], "name": "f"}},
                "tool_choice",
                id="tool-choice-type",
            ),
            pytest.param(
                {"input": [{"role": "user", "content": [{"type": {"a": 1}}]}]},
                "input[0].content[0]",
                id="content-type",
            ),
            pytest.param({"input": [{"type": ["reasoning"]}]}, "input[0]", id="item-type"),
            pytest.param({"tools": [{"type": ["function"], "name": "f"}]}, "tools[0]", id="tool-type"),
        ],
    )
    def test_an_ill_typed_nested_value_never_raises_a_bare_type_error(self, body: Any, expected_key: str) -> None:
        """Every set-membership test needs a hashability guard before it.

        A JSON body may legally carry a list or an object where the schema
        declares a string, and `["function"] in frozenset(...)` raises
        `TypeError`. `contract` defines a reader-raised `TypeError` as "the
        reader mis-routed a field — a reader bug", so letting one out would make
        T-D1 diagnose a malformed body (T-C6 contributes one to the corpus) as a
        defect in the harness itself.

        These bodies are unclassifiable, not unreadable, so the required outcome
        is a residual — never an exception of any kind.
        """
        projected = r.ResponsesProjection().read_request(captured(body))

        # The exact key, not merely "something residualised": a guard that
        # residualised the *wrong* thing would otherwise pass.
        assert set(projected.residual) == {expected_key}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_a_message_role_that_is_not_even_a_string_is_an_unreadable_body(self) -> None:
        """R11's named case, in the shape that breaks a naive membership test.

        A role of `["user"]` *is* outside the published set, so the requirement
        applies — but testing membership of an unhashable value raises the one
        exception type R11 forbids.
        """
        with pytest.raises(c.UnreadableBodyError):
            r.ResponsesProjection().read_request(captured({"input": [{"role": ["user"], "content": "x"}]}))


# --------------------------------------------------------------------------
# §1.4 — the falsification set
# --------------------------------------------------------------------------


class TestFalsification:
    """Three deliberate defects the reader must detect, running in the suite.

    Plan §1.4: "a fidelity oracle never shown to fail is indistinguishable from
    one that cannot fail."
    """

    def test_an_unrecognised_top_level_key_fails_closed(self) -> None:
        """The fifth of §3.3.1's five oracle falsification cases.

        Bridge-added metadata the projection discarded could never have been
        scanned for a vendor token either, so this also keeps §3.3.3 honest.
        """
        projected = r.ResponsesProjection().read_request(
            captured({"model": "m", "input": "hi", "x_kitty_trace": "abc"})
        )

        assert projected.residual == {"x_kitty_trace": "abc"}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_an_unrecognised_item_type_fails_closed_at_its_exact_path(self) -> None:
        """Asserted by key, not merely by non-emptiness.

        Six readers and T-D8 must agree on the exact residual spelling, so a
        test that accepted any non-empty residual would let two readers drift.
        """
        projected = r.ResponsesProjection().read_request(
            captured({"input": [{"role": "user", "content": "hi"}, {"type": "brand_new_2027"}]})
        )

        assert set(projected.residual) == {"input[1]"}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_a_dropping_reader_is_caught_even_though_its_residual_is_empty(self) -> None:
        """The case `consumed` exists for, and the one a residual-only rule misses.

        The defect is deliberate: this stub ignores a key instead of
        residualising it, which leaves the residual empty. "The residual must be
        empty" would pass it.
        """
        body = {"model": "m", "input": "hi", "x_kitty_trace": "abc"}
        dropping = c.Request(
            envelope=c.Envelope(model="m"),
            conversation=c.Conversation(turns=[c.Turn("user", [c.Text("hi")])]),
            residual={},
            consumed=frozenset({"model", "input"}),
            source=body,
        )

        assert dropping.residual == {}
        with pytest.raises(c.DroppedFieldsError):
            c.verify_total(dropping)


# --------------------------------------------------------------------------
# The published examples
# --------------------------------------------------------------------------

#: The eight `x-oaiMeta` request examples carried by `openapi.yaml` itself, at
#: `info.version` matching `reader_responses.SCHEMA_VERSION`. Vendored as
#: literals so the suite needs no network and pins what it was validated
#: against — §3.3.1 requires validation against the format's published examples,
#: "never against kitty's output".
_PUBLISHED_EXAMPLES: dict[str, Any] = {
    "Text input": {
        "model": "gpt-6-astra",
        "input": "Tell me a three sentence bedtime story about a unicorn.",
    },
    "Image input": {
        "model": "gpt-6-astra",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is in this image?"},
                    {"type": "input_image", "image_url": "https://example.test/boardwalk.jpg"},
                ],
            }
        ],
    },
    "File input": {
        "model": "gpt-6-astra",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is in this file?"},
                    {
                        "type": "input_file",
                        "file_url": "https://example.test/2024ltr.pdf",
                        "detail": "auto",
                    },
                ],
            }
        ],
    },
    "Web search": {
        "model": "gpt-6-astra",
        "tools": [{"type": "web_search_preview"}],
        "input": "What was a positive news story from today?",
    },
    "File search": {
        "model": "gpt-6-astra",
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_1234567890"], "max_num_results": 20}],
        "input": "What are the attributes of an ancient brown dragon?",
    },
    "Streaming": {
        "model": "gpt-6-astra",
        "instructions": "You are a helpful assistant.",
        "input": "Hello!",
        "stream": True,
    },
    "Functions": {
        "model": "gpt-6-astra",
        "input": "What is the weather like in Boston today?",
        "tools": [
            {
                "type": "function",
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                },
            }
        ],
        "tool_choice": "auto",
    },
    "Reasoning": {
        "model": "gpt-6-astra",
        "input": "How much wood would a woodchuck chuck?",
        "reasoning": {"effort": "high"},
    },
}


class TestPublishedExamples:
    """The acceptance criterion the task's own ticket names."""

    @pytest.mark.parametrize("title", sorted(_PUBLISHED_EXAMPLES))
    def test_each_published_example_round_trips_with_an_empty_residual(self, title: str) -> None:
        """Round-trip the format's published examples with an empty residual.

        Validated against the published schema, never against kitty's output: a
        reader validated against kitty's output inherits kitty's bugs and the
        oracle becomes circular.
        """
        projected = project(_PUBLISHED_EXAMPLES[title])

        assert projected.residual == {}
        assert projected.envelope.model == "gpt-6-astra"
        assert projected.conversation.turns, "every published example carries input"

    def test_the_schema_version_this_reader_agrees_with_is_recorded(self) -> None:
        """A change detector, deliberately — and no more than that.

        It does not verify the examples against the live schema; the suite has
        no network. What it gives is an audit trail: when the reader is updated
        for a newer revision, this fails and forces the version in the module
        docstring to move with it, so "which schema was this validated against"
        always has an answer.
        """
        assert r.SCHEMA_VERSION == "2.3.0"


# --------------------------------------------------------------------------
# R12 — independence
# --------------------------------------------------------------------------

#: Anchored at line start, so prose mentioning an import cannot trip it; the two
#: dynamic forms stay unanchored because they appear mid-expression. Mirrors
#: `test_contract.py`'s guard deliberately — one spelling for one rule.
_KITTY_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(?:src\.)?kitty\b"
    r"|import_module\(\s*[\"'](?:src\.)?kitty"
    r"|__import__\(\s*[\"'](?:src\.)?kitty"
)


def test_the_reader_imports_nothing_from_kitty() -> None:
    """§3.3.1's independent-oracle rule, enforced structurally rather than by review.

    This is the single structural guarantee the whole I1 claim rests on. A
    projection that asked kitty how to read a body would inherit kitty's bugs,
    and the oracle would prove self-consistency rather than fidelity.

    The stricter of the two readings in the design is taken deliberately:
    §3.3.1b and §7.4 say `src/kitty/bridge`, `contract.py` says `src/kitty`. A
    reader importing `src/kitty/providers` would be just as circular.
    """
    source = Path(r.__file__).read_text(encoding="utf-8")

    # Assert the subject was actually read: a guard that passes on an empty
    # string is indistinguishable from one that cannot fail.
    assert len(source) > 1000, "read no meaningful source; the guard would pass vacuously"

    offending = [line.strip() for line in source.splitlines() if _KITTY_IMPORT.search(line)]

    assert offending == [], f"reader_responses.py must not import kitty: {offending}"


def test_the_import_guard_fires_on_every_form_it_claims_to_catch() -> None:
    """The positive control.

    Without it, a pattern that had stopped matching anything would read as a
    clean bill of health forever.
    """
    forms = [
        "from kitty.bridge import server",
        "import kitty",
        "from src.kitty import server",
        "import  kitty.providers",
        'importlib.import_module("kitty.bridge.server")',
        '__import__("kitty")',
        'importlib.import_module("src.kitty.providers.openai_subscription")',
    ]

    undetected = [form for form in forms if not _KITTY_IMPORT.search(form)]

    assert undetected == [], f"the guard would miss these: {undetected}"


def test_the_import_guard_does_not_fire_on_innocent_text() -> None:
    """The negative control: a pattern matching everything would also pass above."""
    innocent = [
        "# kitty-bridge is the product under test",
        "from harness import contract as c",
        "# never write `import kitty` in this module",
    ]

    assert [line for line in innocent if _KITTY_IMPORT.search(line)] == []
