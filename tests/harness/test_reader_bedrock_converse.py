"""L1 tests for the Bedrock Converse projection.

`.system_design/TEST_SUITE.md` §3.3.1, §3.3.1a, §3.3.1b, §7.4.1 · plan task
**T-A5** (KBR-37).

**Every fixture is a published example or is assembled from the published
schema.** :class:`TestPublishedExamples` carries bodies derived from the
botocore ``bedrock-runtime`` service model at revision
:data:`~harness.reader_bedrock_converse.SCHEMA_VERSION`. None is copied from
kitty's output — §3.3.1's independent-oracle rule makes a reader validated
against kitty's output circular.

**This reader consumes the URL** (model and operation in the path), which
no other reader but Gemini does; :class:`TestRoute` owns §3.3.5's claim that
two requests with byte-identical bodies are told apart by their paths.

**The falsification set is required, not decorative.** Plan §1.4:
:class:`TestFalsification` carries five deliberate defects the reader must
detect, running in the suite. The set is the harness rule's deliverable.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any

import botocore
import pytest
from botocore.session import Session

from harness import contract as c
from harness import reader_bedrock_converse as r

# Re-export so the test class bodies can reference them without the module prefix.
PUBLISHED_TOP_LEVEL_KEYS = r.PUBLISHED_TOP_LEVEL_KEYS

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

#: The published Bedrock Runtime endpoint pattern.  The reader does not
#: consume the host, but :class:`CapturedRequest` requires one — a synthetic
#: host keeps the capture well-formed without naming a host the suite does
#: not control.
_HOST = "bedrock-runtime.us-east-1.amazonaws.com"

#: A model name the AWS-published Converse path uses, so a route assertion
#: names something real.
_MODEL = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"


def captured(
    body: Any,
    *,
    model: str = _MODEL,
    operation: str = "converse",
    path: str | None = None,
) -> c.CapturedRequest:
    """Wrap a body as a capture aimed at the published Bedrock endpoint.

    Args:
        body: A mapping to encode as JSON, or raw bytes to pass through
            undecoded for the failure-shape tests.
        model: The ``modelId`` URI parameter — placed in the path segment.
        operation: The operation suffix — ``converse`` or ``converse-stream``.
        path: A complete path, overriding ``model`` and ``operation`` for
            the tests that hand the reader a path it must reject.

    Returns:
        A capture the reader can be handed directly.
    """
    raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")

    return c.CapturedRequest(
        method="POST",
        scheme="https",
        host=_HOST,
        path=path if path is not None else f"/model/{model}/{operation}",
        query="",
        headers=(("content-type", "application/json"),),
        body=raw,
    )


def project(body: Any, **route: Any) -> c.Request:
    """Project a body and assert it accounted for itself.

    Running :func:`harness.contract.verify_total` here rather than in each
    test means no test can pass while quietly leaving a key unaccounted for.

    Args:
        body: The request body.
        **route: Route overrides, forwarded to :func:`captured`.

    Returns:
        The projection.
    """
    projected = r.BedrockConverseProjection().read_request(captured(body, **route))
    c.verify_total(projected)

    return projected


def project_untotalled(body: Any, **route: Any) -> c.Request:
    """Project a body without asserting totality.

    Args:
        body: The request body.
        **route: Route overrides, forwarded to :func:`captured`.

    Returns:
        The projection, residual and all — which is what the residual
        tests assert on and what :func:`project` would refuse to return.
    """
    return r.BedrockConverseProjection().read_request(captured(body, **route))


def _b64(data: bytes) -> str:
    """Encode ``bytes`` for use in a JSON-serialisable test fixture.

    The AWS JSON protocol serialises blob members as base64-encoded
    strings; the reader decodes them on the way to :func:`harness.contract.image_digest`.
    """
    return base64.b64encode(data).decode("ascii")


# --------------------------------------------------------------------------
# R0 — protocol conformance
# --------------------------------------------------------------------------


class TestProtocolConformance:
    """The reader is a `Projection` for the Bedrock Converse format."""

    def test_it_declares_the_converse_wire_format(self) -> None:
        """The wire format is the enum value, not a stringly-typed stand-in."""
        assert r.BedrockConverseProjection().wire_format is c.WireFormat.BEDROCK_CONVERSE

    def test_it_satisfies_the_projection_protocol(self) -> None:
        """``isinstance`` against the runtime-checkable protocol passes."""
        assert isinstance(r.BedrockConverseProjection(), c.Projection)

    def test_the_schema_version_is_recorded(self) -> None:
        """A reader with no stated provenance cannot be re-derived."""
        assert f"botocore-{botocore.__version__}" == r.SCHEMA_VERSION


# --------------------------------------------------------------------------
# R7 — the route cases (R7.5 analogue: stream from the URL)
# --------------------------------------------------------------------------


class TestRoute:
    """The reader consumes the URL: model + stream from the path.

    R7.5's analogue, since :attr:`~harness.contract.Envelope.stream` is
    populated from the operation, not the body. Two byte-identical bodies
    on ``converse`` and ``converse-stream`` must project to different
    :attr:`~harness.contract.Envelope.stream` values.
    """

    def test_two_identical_bodies_to_two_models_are_told_apart(self) -> None:
        """The bodies are byte-identical; the difference is the path's ``modelId``."""
        body = {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}
        left = captured(body, model="anthropic.claude-3-haiku-20240307-v1:0")
        right = captured(body, model="anthropic.claude-3-5-sonnet-20241022-v2:0")
        assert left.body == right.body

        reader = r.BedrockConverseProjection()
        assert reader.read_request(left).envelope.model == "anthropic.claude-3-haiku-20240307-v1:0"
        assert reader.read_request(right).envelope.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def test_two_identical_bodies_on_two_operations_are_told_apart(self) -> None:
        """``stream`` is the operation, not a body field — bodies are identical."""
        body = {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}
        left = captured(body, operation="converse")
        right = captured(body, operation="converse-stream")
        assert left.body == right.body

        reader = r.BedrockConverseProjection()
        assert reader.read_request(left).envelope.stream is False
        assert reader.read_request(right).envelope.stream is True

    def test_an_unrecognised_path_segment_raises_unreadable_body(self) -> None:
        """``/model/{id}/invoke`` is not a Converse operation — UnreadableBodyError."""
        cap = captured({"messages": []}, path=f"/model/{_MODEL}/invoke")

        with pytest.raises(c.UnreadableBodyError):
            r.BedrockConverseProjection().read_request(cap)

    def test_a_non_converse_path_segment_raises_unreadable_body(self) -> None:
        """Anything outside ``converse`` / ``converse-stream`` is unreadable."""
        cap = captured({"messages": []}, path=f"/model/{_MODEL}/converse-and-stream")

        with pytest.raises(c.UnreadableBodyError):
            r.BedrockConverseProjection().read_request(cap)


# --------------------------------------------------------------------------
# Falsification — plan §1.4's deliverable
# --------------------------------------------------------------------------


class TestFalsification:
    """Five deliberate defects the reader must detect, running in the suite.

    Plan §1.4: "The first working version of every harness ships with at least
    one falsification case — a deliberate defect it must detect, running in
    the suite." Four review rounds on the design produced four harnesses
    that would have passed while proving nothing, and one of them was "a
    projection that could not see the model name, in a product whose purpose
    is changing the model name" — which is exactly what R7.4 pins here.
    """

    def test_an_unrecognised_top_level_key_produces_a_residual(self) -> None:
        """R7.1 — §3.3.1's fifth mandatory oracle falsification case."""
        projected = project_untotalled(
            {"messages": [], "x-kitty-trace": "abc"}
        )

        assert projected.residual == {"x-kitty-trace": "abc"}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_an_unrecognised_key_inside_a_part_produces_a_residual_at_its_path(self) -> None:
        """R7.2 — §7.4.1: failing closed is not a top-level rule."""
        projected = project_untotalled(
            {
                "messages": [
                    {"role": "user", "content": [{"text": "hi", "x-kitty-trace": "abc"}]}
                ]
            }
        )

        assert projected.residual == {
            "messages[0].content[0].x-kitty-trace": "abc"
        }

    def test_consumed_claims_only_what_the_reader_actually_mapped(self) -> None:
        """A reader that over-claims defeats the check ``consumed`` exists to be.

        ``verify_total`` still fails such a body — the residual is non-empty
        — so the falsification case above passes either way and cannot see
        this.  But ``consumed`` would be wrong data: it is what makes a
        dropped key detectable, and T-D8 diffs these key sets across all
        six readers.
        """
        projected = project_untotalled({"messages": [], "x-kitty-trace": "abc"})

        assert set(projected.consumed) == {"messages"}
        assert "x-kitty-trace" not in projected.consumed

    def test_a_dropping_reader_is_caught_by_totality_not_by_the_residual(self) -> None:
        """R7.3 — the case a "residual must be empty" rule cannot see.

        A reader that *drops* an unknown key leaves the residual empty and
        sails through. ``Request.consumed`` exists for this, and this is
        the deliberate defect that proves it does its job. The defect
        pattern (after :mod:`harness.reader_gemini`): a constructed
        ``Request`` whose ``consumed`` matches the honest reader's *minus*
        the unrecognised key. ``verify_total`` then sees the key in
        ``source`` but in neither ``consumed`` nor ``residual`` and raises
        :class:`~harness.contract.DroppedFieldsError`.
        """
        body = {"messages": [], "x-kitty-trace": "abc"}
        honest = project_untotalled(body)
        assert "x-kitty-trace" in honest.residual

        # The dropping reader's ``consumed`` is the honest one with
        # ``x-kitty-trace`` removed — it claims the keys it recognised
        # but does not claim the unknown one.
        dropping = c.Request(
            envelope=honest.envelope,
            conversation=honest.conversation,
            residual={},
            consumed=honest.consumed,
            source=honest.source,
        )

        with pytest.raises(c.DroppedFieldsError):
            c.verify_total(dropping)

    def test_a_body_model_id_is_rejected_as_a_residual(self) -> None:
        """AC-6: modelId lives on the URL — a body modelId is an unrecognised key."""
        projected = project_untotalled(
            {"modelId": "anthropic.claude-3-haiku-20240307-v1:0", "messages": []}
        )

        assert "modelId" in projected.residual
        assert projected.envelope.model == _MODEL  # the URL, not the body


# --------------------------------------------------------------------------
# R8.4 — Envelope
# --------------------------------------------------------------------------


class TestEnvelope:
    """Every declared control field the format publishes carries through."""

    def test_the_eight_declared_control_fields_land_in_extra(self) -> None:
        """Each of the eight declared control fields round-trips whole.

        The value is compared whole (§3.3.1a), so the reader writes it
        verbatim. ``inferenceConfig`` and ``toolChoice`` have their own
        addresses — they are covered by ``TestSampling`` and ``TestToolChoice``.
        """
        body = {
            "additionalModelRequestFields": {"trace": "abc"},
            "additionalModelResponseFieldPaths": ["$.usage"],
            "guardrailConfig": {
                "guardrailIdentifier": "gr-1",
                "guardrailVersion": "1",
            },
            "outputConfig": {"textFormat": {"type": "json"}, "effort": "low"},
            "performanceConfig": {"latency": "optimized"},
            "promptVariables": {"name": "value"},
            "requestMetadata": {"requestId": "req-1"},
            "serviceTier": {"type": "default"},
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        }
        projected = project(body)

        assert projected.envelope.extra["additionalModelRequestFields"] == {"trace": "abc"}
        assert projected.envelope.extra["additionalModelResponseFieldPaths"] == ["$.usage"]
        assert projected.envelope.extra["guardrailConfig"]["guardrailIdentifier"] == "gr-1"
        assert projected.envelope.extra["outputConfig"]["textFormat"] == {"type": "json"}
        assert projected.envelope.extra["performanceConfig"]["latency"] == "optimized"
        assert projected.envelope.extra["promptVariables"] == {"name": "value"}
        assert projected.envelope.extra["requestMetadata"] == {"requestId": "req-1"}
        assert projected.envelope.extra["serviceTier"] == {"type": "default"}


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


class TestSampling:
    """``inferenceConfig`` normalises onto the §3.3.1b closed set."""

    def test_the_four_inference_config_keys_normalise(self) -> None:
        """Each wire spelling carries to its canonical mapping."""
        projected = project(
            {
                "messages": [],
                "inferenceConfig": {
                    "maxTokens": 256,
                    "temperature": 0.5,
                    "topP": 0.9,
                    "stopSequences": ["STOP"],
                },
            }
        )

        assert projected.conversation.sampling == {
            "max_tokens": 256,
            "temperature": 0.5,
            "top_p": 0.9,
            "stop": ["STOP"],
        }

    def test_an_unknown_inference_config_child_residualises_at_its_path(self) -> None:
        """§7.4.1's depth rule: a leaf the schema does not name fails closed."""
        projected = project_untotalled(
            {
                "messages": [],
                "inferenceConfig": {
                    "maxTokens": 256,
                    "mystery": "value",
                },
            }
        )

        assert projected.residual == {"inferenceConfig.mystery": "value"}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)


# --------------------------------------------------------------------------
# Tool choice
# --------------------------------------------------------------------------


class TestToolChoice:
    """``toolConfig.toolChoice`` unifies onto ``envelope.extra["tool_choice"]``."""

    def test_auto_round_trips(self) -> None:
        """``auto`` is canonical."""
        projected = project(
            {"messages": [], "toolConfig": {"toolChoice": {"auto": {}}}}
        )

        assert projected.envelope.extra["tool_choice"] == "auto"

    def test_any_round_trips(self) -> None:
        """``any`` is canonical."""
        projected = project(
            {"messages": [], "toolConfig": {"toolChoice": {"any": {}}}}
        )

        assert projected.envelope.extra["tool_choice"] == "any"

    def test_tool_with_name_round_trips(self) -> None:
        """``tool:<name>`` is the canonical value for a targeted choice."""
        projected = project(
            {
                "messages": [],
                "toolConfig": {"toolChoice": {"tool": {"name": "get_weather"}}},
            }
        )

        assert projected.envelope.extra["tool_choice"] == "tool:get_weather"

    def test_none_residualises_at_its_own_path(self) -> None:
        """``none`` is not in Converse's published ``ToolChoice`` union.

        The envelope constructor accepts ``"none"`` (it is canonical for
        the other wire formats), so the failure mode lives at the depth,
        not the constructor.
        """
        projected = project_untotalled(
            {"messages": [], "toolConfig": {"toolChoice": {"none": {}}}}
        )

        assert "toolConfig.toolChoice.none" in projected.residual


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


class TestTools:
    """``toolConfig.tools`` unifies onto ``Conversation.tools``.

    ``toolSpec`` carries ``name``/``description``/``schema``/``strict``.
    ``Tool.cachePoint`` and ``Tool.systemTool`` are server-side toggles
    that map to ``envelope.extra`` (§7.4.2 rule 5), not to ``Opaque``
    parts — the tools list is for declared client-side tool specs.
    """

    def test_a_tool_spec_with_description_schema_and_strict_round_trips(self) -> None:
        """All four fields survive."""
        projected = project(
            {
                "messages": [],
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "get_weather",
                                "description": "Look up the weather.",
                                "inputSchema": {
                                    "json": {
                                        "type": "object",
                                        "properties": {"city": {"type": "string"}},
                                    }
                                },
                                "strict": True,
                            }
                        }
                    ]
                },
            }
        )

        assert len(projected.conversation.tools) == 1
        tool = projected.conversation.tools[0]
        assert tool.name == "get_weather"
        assert tool.description == "Look up the weather."
        assert tool.schema == {
            "json": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }
        }
        assert tool.strict is True

    def test_a_tool_without_strict_carries_none(self) -> None:
        """Absent ``strict`` is ``None``, not ``False``.

        ``ToolDecl.strict`` docstring (contract.py): "None means absent,
        which must stay distinct from False or that row's presence and
        absence would be indistinguishable." The reader does not coerce.
        """
        projected = project(
            {
                "messages": [],
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "get_weather",
                                "inputSchema": {"type": "object"},
                            }
                        }
                    ]
                },
            }
        )

        assert projected.conversation.tools[0].strict is None

    def test_a_tool_without_description_carries_none(self) -> None:
        """Absent ``description`` is ``None``."""
        projected = project(
            {
                "messages": [],
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "get_weather",
                                "inputSchema": {"type": "object"},
                            }
                        }
                    ]
                },
            }
        )

        assert projected.conversation.tools[0].description is None

    def test_a_tool_without_name_residualises(self) -> None:
        """A tool without a name cannot be paired or addressed."""
        projected = project_untotalled(
            {
                "messages": [],
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "inputSchema": {"type": "object"},
                            }
                        }
                    ]
                },
            }
        )

        assert "toolConfig.tools[0].toolSpec.name" in projected.residual
        assert projected.conversation.tools == ()

    def test_tools_are_addressed_by_name(self) -> None:
        """Tools appear in declaration order, not index-paired."""
        projected = project(
            {
                "messages": [],
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "z",
                                "inputSchema": {"type": "object"},
                            }
                        },
                        {
                            "toolSpec": {
                                "name": "a",
                                "inputSchema": {"type": "object"},
                            }
                        },
                    ]
                },
            }
        )

        assert [t.name for t in projected.conversation.tools] == ["z", "a"]


class TestServerSideToolToggles:
    """``Tool.cachePoint`` and ``Tool.systemTool`` map to ``envelope.extra``."""

    def test_a_cache_point_tool_entry_lands_in_extra(self) -> None:
        """The tools list is empty; the entry is an envelope extra."""
        projected = project(
            {
                "messages": [],
                "toolConfig": {
                    "tools": [
                        {"cachePoint": {"type": "default"}},
                    ]
                },
            }
        )

        assert projected.conversation.tools == ()
        assert projected.envelope.extra["cachePoint"] == {"type": "default"}

    def test_a_system_tool_entry_lands_in_extra(self) -> None:
        """``systemTool`` carries only the tool's name."""
        projected = project(
            {
                "messages": [],
                "toolConfig": {
                    "tools": [
                        {"systemTool": {"name": "built_in_search"}},
                    ]
                },
            }
        )

        assert projected.conversation.tools == ()
        assert projected.envelope.extra["systemTool"] == {"name": "built_in_search"}


# --------------------------------------------------------------------------
# System instruction
# --------------------------------------------------------------------------


class TestSystemInstruction:
    """``system`` array lifts onto ``Conversation.system`` (Text only).

    Converse's ``SystemContentBlock`` union carries ``text``, ``cachePoint``,
    ``guardContent``. The reader carries the text parts into ``system``;
    the other two residualise at the entry path so the totality check sees
    them. ``Conversation.system`` is ``Sequence[Text]`` — non-text blocks
    cannot live there without violating the contract.
    """

    def test_a_single_text_system_block_lifts(self) -> None:
        """``text`` → ``Text(content)``."""
        projected = project({"messages": [], "system": [{"text": "You are helpful."}]})

        assert projected.conversation.system == (c.Text("You are helpful."),)

    def test_multiple_text_system_blocks_lift_in_order(self) -> None:
        """Multiple blocks preserve declaration order."""
        projected = project(
            {
                "messages": [],
                "system": [{"text": "First."}, {"text": "Second."}],
            }
        )

        assert projected.conversation.system == (
            c.Text("First."),
            c.Text("Second."),
        )

    def test_a_cache_point_system_block_residualises(self) -> None:
        """A non-text block does not enter ``system``; it residualises."""
        projected = project_untotalled(
            {"messages": [], "system": [{"cachePoint": {"type": "default"}}]}
        )

        assert "system[0]" in projected.residual
        assert projected.conversation.system == ()

    def test_a_guard_content_system_block_residualises(self) -> None:
        """``guardContent`` is also not text."""
        projected = project_untotalled(
            {
                "messages": [],
                "system": [{"guardContent": {"text": {"text": "guardrail output"}}}],
            }
        )

        assert "system[0]" in projected.residual
        assert projected.conversation.system == ()


# --------------------------------------------------------------------------
# Turn merge pipeline
# --------------------------------------------------------------------------


class TestTurns:
    """§3.3.1b's four merge clauses, applied to Converse's ``messages``."""

    def test_a_maximal_run_of_tool_results_merges_into_one_user_turn(self) -> None:
        """Clause 1: toolResult runs become one user turn."""
        projected = project(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"toolResult": {"toolUseId": "1", "content": [{"text": "a"}]}},
                            {"toolResult": {"toolUseId": "2", "content": [{"text": "b"}]}},
                        ],
                    }
                ]
            }
        )

        assert len(projected.conversation.turns) == 1
        assert projected.conversation.turns[0].role == "user"
        assert len(projected.conversation.turns[0].parts) == 2

    def test_a_tool_result_run_with_a_following_user_message_merges(self) -> None:
        """Clause 2: a user message following the run merges into the tool turn."""
        projected = project(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"toolResult": {"toolUseId": "1", "content": [{"text": "a"}]}},
                            {"text": "now do X."},
                        ],
                    }
                ]
            }
        )

        assert len(projected.conversation.turns) == 1
        assert projected.conversation.turns[0].role == "user"

    def test_tool_result_parts_come_first_within_a_turn(self) -> None:
        """Clause 1 again: within the run, toolResult parts lead."""
        projected = project(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "before"},
                            {"toolResult": {"toolUseId": "1", "content": [{"text": "x"}]}},
                        ],
                    }
                ]
            }
        )

        turn = projected.conversation.turns[0]
        assert turn.parts
        assert isinstance(turn.parts[0], c.ToolResult)
        assert isinstance(turn.parts[1], c.Text)

    def test_consecutive_same_role_turns_merge(self) -> None:
        """Clause 4: same-role turns merge."""
        projected = project(
            {
                "messages": [
                    {"role": "user", "content": [{"text": "first"}]},
                    {"role": "user", "content": [{"text": "second"}]},
                ]
            }
        )

        assert len(projected.conversation.turns) == 1
        assert len(projected.conversation.turns[0].parts) == 2

    def test_user_role_then_assistant_role_does_not_merge(self) -> None:
        """Different roles remain distinct turns."""
        projected = project(
            {
                "messages": [
                    {"role": "user", "content": [{"text": "hi"}]},
                    {"role": "assistant", "content": [{"text": "hello"}]},
                ]
            }
        )

        assert len(projected.conversation.turns) == 2


# --------------------------------------------------------------------------
# Content block dispatch
# --------------------------------------------------------------------------


class TestParts:
    """Every ``ContentBlock`` union member dispatches to its canonical form."""

    def test_text(self) -> None:
        """``text`` → ``Text``."""
        projected = project({"messages": [{"role": "user", "content": [{"text": "hi"}]}]})

        assert projected.conversation.turns[0].parts == (c.Text("hi"),)

    def test_image_bytes_carries_a_digest(self) -> None:
        """``image.source.bytes`` → ``Image(digest, media_type)``.

        AWS JSON protocol serialises blob members as base64-encoded strings;
        the reader decodes them.
        """
        raw = b"PNGDATA"
        projected = project(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "png",
                                    "source": {"bytes": _b64(raw)},
                                }
                            }
                        ],
                    }
                ]
            }
        )

        image = projected.conversation.turns[0].parts[0]
        assert isinstance(image, c.Image)
        assert image.digest == c.image_digest(raw)
        assert image.media_type == "image/png"
        assert image.ref is None

    def test_image_s3_location_carries_a_ref_without_a_digest(self) -> None:
        """``image.source.s3Location`` → ``Image(ref=uri)``, no digest (§3.3.1)."""
        projected = project(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": {
                                        "s3Location": {"uri": "s3://bucket/key"}
                                    },
                                }
                            }
                        ],
                    }
                ]
            }
        )

        image = projected.conversation.turns[0].parts[0]
        assert isinstance(image, c.Image)
        assert image.digest is None
        assert image.ref == "s3://bucket/key"

    def test_tool_use(self) -> None:
        """``toolUse`` → ``ToolUse(name, arguments, id)``."""
        projected = project(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "call_1",
                                    "name": "get_weather",
                                    "input": {"city": "Berlin"},
                                }
                            }
                        ],
                    }
                ]
            }
        )

        part = projected.conversation.turns[0].parts[0]
        assert isinstance(part, c.ToolUse)
        assert part.name == "get_weather"
        assert part.id == "call_1"
        assert part.arguments == {"city": "Berlin"}

    def test_tool_use_input_must_be_an_object_to_round_trip(self) -> None:
        """A non-mapping ``input`` residualises; the part is not produced."""
        projected = project_untotalled(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "call_1",
                                    "name": "get_weather",
                                    "input": [1, 2, 3],
                                }
                            }
                        ],
                    }
                ]
            }
        )

        assert "messages[0].content[0].toolUse.input" in projected.residual

    def test_tool_result_text(self) -> None:
        """``toolResult.content[text]`` → ``ToolResult(content=[Text])``."""
        projected = project(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "call_1",
                                    "content": [{"text": "result"}],
                                }
                            }
                        ],
                    }
                ]
            }
        )

        part = projected.conversation.turns[0].parts[0]
        assert isinstance(part, c.ToolResult)
        assert part.tool_use_id == "call_1"
        assert part.is_error is False
        assert part.content == (c.Text("result"),)

    def test_tool_result_error_status_sets_is_error(self) -> None:
        """``status: "error"`` → ``is_error = True``."""
        projected = project(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "call_1",
                                    "content": [{"text": "boom"}],
                                    "status": "error",
                                }
                            }
                        ],
                    }
                ]
            }
        )

        part = projected.conversation.turns[0].parts[0]
        assert isinstance(part, c.ToolResult)
        assert part.is_error is True

    def test_tool_result_json(self) -> None:
        """``toolResult.content[json]`` → ``ToolResult(content=[Json])``."""
        projected = project(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "call_1",
                                    "content": [{"json": {"k": "v"}}],
                                }
                            }
                        ],
                    }
                ]
            }
        )

        part = projected.conversation.turns[0].parts[0]
        assert isinstance(part, c.ToolResult)
        assert part.content == (c.Json({"k": "v"}),)

    def test_opaque_blocks_round_trip_with_canonical_kind(self) -> None:
        """Eight Opaque-only ``ContentBlock`` discriminators project to ``Opaque``.

        Parametrised to keep each case one assertion. The canonical kind
        names come from :func:`harness.contract.opaque_kind` —
        :data:`harness.contract.OPAQUE_ALIASES`.
        """
        cases: list[tuple[str, dict[str, Any]]] = [
            ("cachePoint", {"type": "default"}),
            ("guardContent", {"text": {"text": "guard output"}}),
            ("document", {"name": "doc", "source": {"bytes": _b64(b"DOC")}}),
            ("video", {"format": "mp4", "source": {"bytes": _b64(b"VID")}}),
            ("audio", {"format": "wav", "source": {"bytes": _b64(b"AUD")}}),
            ("searchResult", {"content": [], "source": "src", "title": "t"}),
            ("citationsContent", {"citations": [], "content": []}),
            ("toolAddition", {"tool": {"name": "x"}}),
            ("toolRemoval", {"tool": {"name": "x"}}),
        ]
        for discriminator, payload in cases:
            block = {discriminator: payload}
            projected = project({"messages": [{"role": "user", "content": [block]}]})
            part = projected.conversation.turns[0].parts[0]
            assert isinstance(part, c.Opaque), discriminator
            assert part.kind == c.opaque_kind(discriminator), discriminator


class TestImageSources:
    """``ImageSource`` dispatch: bytes vs s3Location.

    §3.3.1's "Unpinned, the Messages reader and the Chat Completions
    reader would produce different digests for one image" — the s3Location
    branch produces no digest because there are no bytes to digest.
    """

    def test_bytes_digest_matches_image_digest_helper(self) -> None:
        """A specific bytes value produces a specific digest."""
        raw = b"\x89PNG\r\n\x1a\n"
        projected = project(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "png",
                                    "source": {"bytes": _b64(raw)},
                                }
                            }
                        ],
                    }
                ]
            }
        )

        image = projected.conversation.turns[0].parts[0]
        assert isinstance(image, c.Image)
        assert image.digest == c.image_digest(raw)

    def test_s3location_never_produces_a_digest(self) -> None:
        """``s3Location.uri`` carries only as ``Image.ref``."""
        projected = project(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "png",
                                    "source": {"s3Location": {"uri": "s3://bucket/key"}},
                                }
                            }
                        ],
                    }
                ]
            }
        )

        image = projected.conversation.turns[0].parts[0]
        assert isinstance(image, c.Image)
        assert image.digest is None
        assert image.ref == "s3://bucket/key"


class TestToolResultStatus:
    """``ToolResultStatus`` enum dispatch."""

    def test_success_means_is_error_false(self) -> None:
        """``status: "success"`` is the documented non-error."""
        projected = project(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "x",
                                    "content": [{"text": "ok"}],
                                    "status": "success",
                                }
                            }
                        ],
                    }
                ]
            }
        )

        part = projected.conversation.turns[0].parts[0]
        assert isinstance(part, c.ToolResult)
        assert part.is_error is False

    def test_absent_status_means_is_error_false(self) -> None:
        """No ``status`` key → ``is_error = False``."""
        projected = project(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "x",
                                    "content": [{"text": "ok"}],
                                }
                            }
                        ],
                    }
                ]
            }
        )

        part = projected.conversation.turns[0].parts[0]
        assert isinstance(part, c.ToolResult)
        assert part.is_error is False


class TestReasoningContent:
    """``ReasoningContentBlock`` dispatch."""

    def test_reasoning_text_projects_to_thinking(self) -> None:
        """``reasoningContent.reasoningText`` → ``Thinking(text, signature)``."""
        projected = project(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "reasoningContent": {
                                    "reasoningText": {
                                        "text": "step-by-step",
                                        "signature": "sig-1",
                                    }
                                }
                            }
                        ],
                    }
                ]
            }
        )

        part = projected.conversation.turns[0].parts[0]
        assert isinstance(part, c.Thinking)
        assert part.text == "step-by-step"
        assert part.signature == "sig-1"

    def test_redacted_content_projects_to_opaque_redacted_thinking(self) -> None:
        """``reasoningContent.redactedContent`` → ``Opaque(kind="redacted_thinking")``."""
        projected = project(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "reasoningContent": {
                                    "redactedContent": _b64(b"redacted")
                                }
                            }
                        ],
                    }
                ]
            }
        )

        part = projected.conversation.turns[0].parts[0]
        assert isinstance(part, c.Opaque)
        assert part.kind == "redacted_thinking"


# --------------------------------------------------------------------------
# Falsification: every optional leaf fails closed
# --------------------------------------------------------------------------


class TestEveryOptionalLeafFailsClosed:
    """§7.4.1's typed-leaf rule binds every optional leaf.

    T-A1's reviewer experience found ~9/10 optional leaves failing open
    without this test (KBR-1 family). Each parametrised case asserts that
    a wrong-typed value residualises at the leaf, not the parent's path.
    """

    @pytest.mark.parametrize(
        "wire_key,wrong_value",
        [
            ("maxTokens", "not-an-int"),
            ("temperature", "warm"),
            ("topP", "very-high"),
            ("stopSequences", "STOP"),
            ("stopSequences", [1, 2, 3]),
        ],
    )
    def test_inference_config_leaves_fail_closed(self, wire_key: str, wrong_value: Any) -> None:
        """A typed-mismatch on ``inferenceConfig`` residualises at the leaf path."""
        projected = project_untotalled(
            {"messages": [], "inferenceConfig": {wire_key: wrong_value}}
        )

        assert projected.residual == {f"inferenceConfig.{wire_key}": wrong_value}

    def test_a_non_string_tool_description_residualises(self) -> None:
        """``ToolSpecification.description`` must be a string."""
        projected = project_untotalled(
            {
                "messages": [],
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "x",
                                "inputSchema": {"type": "object"},
                                "description": ["not", "a", "string"],
                            }
                        }
                    ]
                },
            }
        )

        assert "toolConfig.tools[0].toolSpec.description" in projected.residual

    def test_a_non_boolean_tool_strict_residualises(self) -> None:
        """``ToolSpecification.strict`` must be a boolean."""
        projected = project_untotalled(
            {
                "messages": [],
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "x",
                                "inputSchema": {"type": "object"},
                                "strict": "true",
                            }
                        }
                    ]
                },
            }
        )

        assert "toolConfig.tools[0].toolSpec.strict" in projected.residual

    def test_a_non_bytes_image_source_residualises(self) -> None:
        """``image.source.bytes`` must be a base64 string — a non-string is a breach."""
        projected = project_untotalled(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "png",
                                    "source": {"bytes": [1, 2, 3]},
                                }
                            }
                        ],
                    }
                ]
            }
        )

        assert "messages[0].content[0].image.source.bytes" in projected.residual

    def test_a_non_string_image_format_residualises(self) -> None:
        """``image.format`` must be a string."""
        projected = project_untotalled(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": {"format": 7, "source": {"bytes": _b64(b"x")}}}
                        ],
                    }
                ]
            }
        )

        assert "messages[0].content[0].image.format" in projected.residual

    def test_a_non_string_tool_use_name_residualises(self) -> None:
        """``toolUse.name`` must be a string."""
        projected = project_untotalled(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"toolUse": {"toolUseId": "x", "name": 42, "input": {}}}
                        ],
                    }
                ]
            }
        )

        assert "messages[0].content[0].toolUse.name" in projected.residual

    def test_an_unknown_tool_result_status_residualises(self) -> None:
        """``status`` outside the enum residualises; ``is_error`` stays False."""
        projected = project_untotalled(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "x",
                                    "content": [{"text": "ok"}],
                                    "status": "partial",
                                }
                            }
                        ],
                    }
                ]
            }
        )

        assert "messages[0].content[0].toolResult.status" in projected.residual


# --------------------------------------------------------------------------
# Injection probe
# --------------------------------------------------------------------------


class TestInjectionProbe:
    """Maximal body × junk probe.

    A single maximal body that populates every published key, then a junk
    payload probe on a representative ``ContentBlock``. Catches ``Opaque.kind``
    holes and dispatch defects that hand-written tests miss — Gemini's
    reviewer experience found one such hole that 103 hand-written tests had
    not.
    """

    def test_a_maximal_body_round_trips(self) -> None:
        """Every top-level key plus one entry per ContentBlock union member."""
        body = _build_maximal_body()
        projected = project(body)

        assert projected.residual == {}
        assert set(projected.consumed) == PUBLISHED_TOP_LEVEL_KEYS & set(body)

    @pytest.mark.parametrize(
        "junk",
        [
            {"$ref": "../elsewhere", "x-kitty-trace": "abc"},
            {"__proto__": {"polluted": True}, "x": 1},
            {"junk": 1, "another": "field"},
        ],
    )
    def test_a_content_block_with_junk_payload_residualises(self, junk: Mapping[str, Any]) -> None:
        """An injected junk payload on a text part residualises at the leaf."""
        projected = project_untotalled(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": "ok", **junk}],
                    }
                ]
            }
        )

        # The text part is still produced; the injected keys residualise at
        # the content-block path so the reader is honest about what it has
        # not classified.
        for key in junk:
            path = f"messages[0].content[0].{key}"
            assert path in projected.residual, path


# --------------------------------------------------------------------------
# Union-member value tests
# --------------------------------------------------------------------------


class TestUnionMemberValues:
    """§7.4.2 rule 7's three branches: value-that-is-the-part, required-field-of-part, undecodable-payload."""

    def test_tool_use_with_a_non_string_tool_use_id_residualises(self) -> None:
        """``toolUseId`` must be a string."""
        projected = project_untotalled(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"toolUse": {"toolUseId": 42, "name": "x", "input": {}}}
                        ],
                    }
                ]
            }
        )

        assert "messages[0].content[0].toolUse.toolUseId" in projected.residual

    def test_tool_result_with_missing_tool_use_id_residualises(self) -> None:
        """``toolUseId`` is required."""
        projected = project_untotalled(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"toolResult": {"content": [{"text": "x"}]}}
                        ],
                    }
                ]
            }
        )

        assert "messages[0].content[0].toolResult.toolUseId" in projected.residual

    def test_an_unrecognised_tool_entry_residualises(self) -> None:
        """A ``Tool`` entry with an unrecognised discriminator fails closed."""
        projected = project_untotalled(
            {
                "messages": [],
                "toolConfig": {
                    "tools": [
                        {"unknownTool": {"name": "x"}},
                    ]
                },
            }
        )

        assert "toolConfig.tools[0]" in projected.residual

    def test_a_content_block_without_a_discriminator_residualises(self) -> None:
        """No discriminator key → residualise at the entry path."""
        projected = project_untotalled(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"unknownKey": "x"}],
                    }
                ]
            }
        )

        assert "messages[0].content[0]" in projected.residual


# --------------------------------------------------------------------------
# Failure shapes
# --------------------------------------------------------------------------


class TestFailureShapes:
    """``UnreadableBodyError`` arms."""

    def test_malformed_json_raises(self) -> None:
        """Not JSON → ``UnreadableBodyError``."""
        cap = captured(b"not json at all")
        with pytest.raises(c.UnreadableBodyError):
            r.BedrockConverseProjection().read_request(cap)

    def test_a_non_object_body_raises(self) -> None:
        """A JSON array is valid JSON and an invalid Converse body."""
        cap = captured(b"[1, 2, 3]")
        with pytest.raises(c.UnreadableBodyError):
            r.BedrockConverseProjection().read_request(cap)

    def test_a_non_list_messages_raises(self) -> None:
        """``messages`` must be a list."""
        cap = captured({"messages": "not-a-list"})
        with pytest.raises(c.UnreadableBodyError):
            r.BedrockConverseProjection().read_request(cap)

    def test_a_non_list_content_raises(self) -> None:
        """``messages[*].content`` must be a list."""
        cap = captured(
            {"messages": [{"role": "user", "content": "not-a-list"}]}
        )
        with pytest.raises(c.UnreadableBodyError):
            r.BedrockConverseProjection().read_request(cap)

    def test_an_unrecognised_role_raises(self) -> None:
        """A role outside ``user``/``assistant`` is unprojectable."""
        cap = captured(
            {"messages": [{"role": "system", "content": [{"text": "x"}]}]}
        )
        with pytest.raises(c.UnreadableBodyError):
            r.BedrockConverseProjection().read_request(cap)


# --------------------------------------------------------------------------
# Totality over the whole surface
# --------------------------------------------------------------------------


class TestTotalityOverTheWholeSurface:
    """A maximal body with every published top-level key and one entry per ContentBlock."""

    def test_consumed_equals_the_twelve_published_wire_body_keys(self) -> None:
        """The ``consumed`` set is the addressable surface — no more, no less."""
        projected = project(_build_maximal_body())

        assert set(projected.consumed) == set(PUBLISHED_TOP_LEVEL_KEYS)
        assert projected.residual == {}


# --------------------------------------------------------------------------
# Residuals expected on real traffic
# --------------------------------------------------------------------------


class TestResidualsExpectedOnRealTraffic:
    """Blocks the design expects on real traffic but the reader has no first-class part for.

    Each block type has a canonical projection shape so a future reader
    does not "fix" the projection into a different ``Opaque.kind`` or move
    the toggle.
    """

    def test_cache_point_in_content(self) -> None:
        """``ContentBlock.cachePoint`` → ``Opaque(kind="cache_point")``."""
        projected = project(
            {"messages": [{"role": "user", "content": [{"cachePoint": {"type": "default"}}]}]}
        )

        part = projected.conversation.turns[0].parts[0]
        assert isinstance(part, c.Opaque)
        assert part.kind == "cache_point"

    def test_guard_content_in_content(self) -> None:
        """``ContentBlock.guardContent`` → ``Opaque(kind="guard_content")``."""
        projected = project(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"guardContent": {"text": {"text": "guard output"}}}],
                    }
                ]
            }
        )

        part = projected.conversation.turns[0].parts[0]
        assert isinstance(part, c.Opaque)
        assert part.kind == "guard_content"


# --------------------------------------------------------------------------
# Depth-falsification
# --------------------------------------------------------------------------


class TestDepthFalsification:
    """Failing closed at every depth the totality check cannot see."""

    @pytest.mark.parametrize(
        "body,expected_path",
        [
            # InferenceConfig child — depth one inside the envelope.
            (
                {"messages": [], "inferenceConfig": {"mystery": "x"}},
                "inferenceConfig.mystery",
            ),
            # ToolChoice discriminator — depth two inside toolConfig.
            (
                {"messages": [], "toolConfig": {"toolChoice": {"none": {}}}},
                "toolConfig.toolChoice.none",
            ),
            # Content-block leaf — depth three.
            (
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": "x", "mystery": "y"}],
                        }
                    ]
                },
                "messages[0].content[0].mystery",
            ),
            # System-block leaf — depth one inside system.
            (
                {"messages": [], "system": [{"text": "x", "mystery": "y"}]},
                "system[0].mystery",
            ),
        ],
    )
    def test_unknown_field_residualises_at_its_path(
        self, body: Mapping[str, Any], expected_path: str
    ) -> None:
        """Each depth produces a residual at the unknown field's path.

        ``envelope.extra[<key>]`` is compared whole (§3.3.1a) — a difference
        inside a control field is reported at ``envelope.extra[<key>]``,
        not at a dotted sub-path. These cases test the depths where
        ``verify_total`` cannot see past the top level: ``inferenceConfig``
        children, ``toolChoice`` discriminator keys, content-block
        extras, and system-block extras.
        """
        projected = project_untotalled(body)

        assert expected_path in projected.residual, expected_path


# --------------------------------------------------------------------------
# Published examples
# --------------------------------------------------------------------------


class TestPublishedExamples:
    """Bodies derived from the published schema round-trip with an empty residual."""

    def test_minimal_text_only_round_trips(self) -> None:
        """A minimal user message — the smallest valid Converse body."""
        projected = project(
            {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}
        )

        assert projected.residual == {}
        assert projected.conversation.turns == (
            c.Turn("user", (c.Text("Hello"),)),
        )

    def test_mixed_conversation_with_tools_and_reasoning_round_trips(self) -> None:
        """The richest body — every declared control field populated together."""
        body = _build_maximal_body()
        projected = project(body)

        assert projected.residual == {}
        # Smoke-assertions on the projection, since "empty residual" alone
        # would pass against a reader that read nothing but claimed the
        # keys.
        assert projected.envelope.model == _MODEL
        assert projected.envelope.stream is False
        assert projected.conversation.system == (c.Text("System."),)
        assert [t.name for t in projected.conversation.tools] == ["get_weather"]
        assert projected.envelope.extra["tool_choice"] == "auto"
        assert projected.conversation.sampling == {
            "max_tokens": 256,
            "temperature": 0.5,
            "top_p": 0.9,
            "stop": ["STOP"],
        }


# --------------------------------------------------------------------------
# Schema agreement — the static set must match the live botocore model
# --------------------------------------------------------------------------


class TestSchemaAgreement:
    """The reader's static frozensets still match the published schema."""

    @pytest.fixture(scope="module")
    def live_schema(self) -> dict[str, Any]:
        """Pull the live service model — pinned by ``uv.lock``."""
        sm = Session().get_service_model("bedrock-runtime")
        op = sm.operation_model("Converse")
        return {"input_shape": op.input_shape}

    def test_top_level_keys_match_the_live_service_model(self, live_schema: Mapping[str, Any]) -> None:
        """Every wire-body key in the published schema is in the static set.

        ``ConverseRequest`` shape's input members include ``modelId``, which
        is a URI parameter — not a wire-body key — and is excluded from the
        static set by design (req 4).
        """
        body_keys = {
            name
            for name in live_schema["input_shape"].members
            if name != "modelId"
        }

        assert set(r.PUBLISHED_TOP_LEVEL_KEYS) == body_keys

    def test_content_block_types_match_the_live_service_model(
        self, live_schema: Mapping[str, Any]
    ) -> None:
        """Every ``ContentBlock`` union member is in the static set."""
        sm = Session().get_service_model("bedrock-runtime")
        # ``ContentBlocks`` is the list-of-ContentBlock type; the union
        # members live on the member shape.
        content_blocks = sm.shape_for("ContentBlocks")
        member_names = set(content_blocks.member.members)

        assert set(r.PUBLISHED_CONTENT_BLOCK_TYPES) == member_names

    def test_system_content_block_types_match_the_live_service_model(
        self, live_schema: Mapping[str, Any]
    ) -> None:
        """Every ``SystemContentBlock`` union member is in the static set."""
        sm = Session().get_service_model("bedrock-runtime")
        sys_block = sm.shape_for("SystemContentBlock")
        member_names = set(sys_block.members)

        assert set(r.PUBLISHED_SYSTEM_CONTENT_BLOCK_TYPES) == member_names

    def test_tool_block_types_match_the_live_service_model(
        self, live_schema: Mapping[str, Any]
    ) -> None:
        """Every ``Tool`` union member is in the static set."""
        sm = Session().get_service_model("bedrock-runtime")
        tool = sm.shape_for("Tool")
        member_names = set(tool.members)

        assert set(r.PUBLISHED_TOOL_BLOCK_TYPES) == member_names

    def test_tool_result_content_block_types_match_the_live_service_model(
        self, live_schema: Mapping[str, Any]
    ) -> None:
        """Every ``ToolResultContentBlock`` union member is in the static set."""
        sm = Session().get_service_model("bedrock-runtime")
        result_blocks = sm.shape_for("ToolResultContentBlocks")
        member_names = set(result_blocks.member.members)

        assert set(r.PUBLISHED_TOOL_RESULT_CONTENT_BLOCK_TYPES) == member_names

    def test_inference_config_keys_match_the_live_service_model(
        self, live_schema: Mapping[str, Any]
    ) -> None:
        """``InferenceConfiguration`` (botocore's published name) carries four keys."""
        sm = Session().get_service_model("bedrock-runtime")
        inference = sm.shape_for("InferenceConfiguration")
        member_names = set(inference.members)

        assert set(r.PUBLISHED_INFERENCE_CONFIG_KEYS) == member_names

    def test_tool_choice_members_match_the_live_service_model(
        self, live_schema: Mapping[str, Any]
    ) -> None:
        """``ToolChoice`` union members (no ``none`` in Converse)."""
        sm = Session().get_service_model("bedrock-runtime")
        choice = sm.shape_for("ToolChoice")
        member_names = set(choice.members)

        assert set(r.PUBLISHED_TOOL_CHOICE_MEMBERS) == member_names

    def test_tool_specification_keys_match_the_live_service_model(
        self, live_schema: Mapping[str, Any]
    ) -> None:
        """``ToolSpecification`` keys."""
        sm = Session().get_service_model("bedrock-runtime")
        spec = sm.shape_for("ToolSpecification")
        member_names = set(spec.members)

        assert set(r.PUBLISHED_TOOL_SPECIFICATION_KEYS) == member_names

    def test_image_source_members_match_the_live_service_model(
        self, live_schema: Mapping[str, Any]
    ) -> None:
        """``ImageSource`` union members — exactly two."""
        sm = Session().get_service_model("bedrock-runtime")
        source = sm.shape_for("ImageSource")
        member_names = set(source.members)

        assert set(r.PUBLISHED_IMAGE_SOURCE_MEMBERS) == member_names

    def test_tool_result_status_values_match_the_live_service_model(
        self, live_schema: Mapping[str, Any]
    ) -> None:
        """``ToolResultStatus`` enum values."""
        sm = Session().get_service_model("bedrock-runtime")
        status = sm.shape_for("ToolResultStatus")
        enum_values = set(status.enum)

        assert set(r.PUBLISHED_TOOL_RESULT_STATUS_VALUES) == enum_values

    def test_reasoning_content_block_members_match_the_live_service_model(
        self, live_schema: Mapping[str, Any]
    ) -> None:
        """``ReasoningContentBlock`` union members."""
        sm = Session().get_service_model("bedrock-runtime")
        block = sm.shape_for("ReasoningContentBlock")
        member_names = set(block.members)

        assert set(r.PUBLISHED_REASONING_CONTENT_BLOCK_MEMBERS) == member_names


# --------------------------------------------------------------------------
# Shared body builder
# --------------------------------------------------------------------------


def _build_maximal_body() -> dict[str, Any]:
    """Build a maximal Converse body exercising every published union member.

    Returns:
        A JSON-compatible mapping.
    """
    return {
        "system": [{"text": "System."}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "Hello."},
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": _b64(b"\x89PNG\r\n\x1a\n")},
                        }
                    },
                    {
                        "document": {"name": "doc", "source": {"bytes": _b64(b"DOC")}},
                    },
                    {
                        "video": {"format": "mp4", "source": {"bytes": _b64(b"VID")}},
                    },
                    {
                        "audio": {"format": "wav", "source": {"bytes": _b64(b"AUD")}},
                    },
                    {"guardContent": {"text": {"text": "guard output"}}},
                    {"cachePoint": {"type": "default"}},
                    {
                        "searchResult": {
                            "content": [],
                            "source": "src",
                            "title": "t",
                        },
                    },
                    {"citationsContent": {"citations": [], "content": []}},
                    {"toolAddition": {"tool": {"name": "x"}}},
                    {"toolRemoval": {"tool": {"name": "x"}}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": "thinking", "signature": "sig"}
                        }
                    },
                    {
                        "toolUse": {
                            "toolUseId": "call_1",
                            "name": "get_weather",
                            "input": {"city": "Berlin"},
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_1",
                            "content": [{"text": "result"}],
                        }
                    }
                ],
            },
        ],
        "inferenceConfig": {
            "maxTokens": 256,
            "temperature": 0.5,
            "topP": 0.9,
            "stopSequences": ["STOP"],
        },
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": "get_weather",
                        "description": "Look up the weather.",
                        "inputSchema": {
                            "json": {"type": "object"},
                        },
                        "strict": True,
                    }
                }
            ],
            "toolChoice": {"auto": {}},
        },
        "guardrailConfig": {"guardrailIdentifier": "gr-1", "guardrailVersion": "1"},
        "additionalModelRequestFields": {"trace": "abc"},
        "additionalModelResponseFieldPaths": ["$.usage"],
        "outputConfig": {"textFormat": {"type": "json"}},
        "performanceConfig": {"latency": "optimized"},
        "promptVariables": {"name": "value"},
        "requestMetadata": {"requestId": "req-1"},
        "serviceTier": {"type": "default"},
    }
