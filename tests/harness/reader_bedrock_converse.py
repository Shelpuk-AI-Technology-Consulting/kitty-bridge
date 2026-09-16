"""L1 projection of a Bedrock Converse request into the wire-independent form.

`.system_design/TEST_SUITE.md` §3.3.1, §3.3.1a, §3.3.1b · plan task **T-A5**
(KBR-37).

The reader consumes the **URL** (the model and the streaming flag live in
the path, exactly the way the Gemini reader does for its model) and the body
in that order. It imports nothing from ``src/kitty`` — the independent-oracle
rule §3.3.1 names (a reader validated against kitty's output inherits
kitty's bugs and the oracle becomes circular).

The schema this reader is written against is the botocore ``bedrock-runtime``
service model — the machine-readable shape AWS publishes — at the revision
:data:`SCHEMA_VERSION` (a pinned literal, not a runtime-derived value, so
:class:`TestSchemaAgreement` compares a known revision against the live
model rather than against itself). CI resolves botocore freely, so a live
model revision that adds or renames a wire key fails
:class:`TestSchemaAgreement` with the key named.

**Converse has no ``cache_control`` concept.** Anthropic carries a
``cache_control`` breakpoint on most blocks; Converse carries a separate
``cachePoint`` *block* in the content list, not a field on one (§3.3.1).
This reader projects ``cachePoint`` to :class:`~harness.contract.Opaque`
with the canonical kind ``cache_point`` — not to ``Text.cache_control`` or
``Opaque.cache_control``. §3.3.1 names this as the deliberate exception:
the field is present on one side only, which is what **M16** needs. The
design's §11 entry ``Q-cache-control-converse`` records this decision;
§7.4.3 of the same document names it T-A5's own answer to T-A2's Q16.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from harness import contract as c

# --------------------------------------------------------------------------
# Schema agreement
# --------------------------------------------------------------------------

#: The botocore release this reader was written against. Pinned as a
#: **literal** so the schema-agreement test compares a known revision
#: against the live model, not itself (the Gemini precedent's shape —
#: ``reader_gemini.SCHEMA_VERSION == "20260910"``). A vendor schema
#: revision that bumps the installed botocore fails
#: :class:`TestSchemaAgreement` with the key named; the reader author
#: then re-derives the frozensets and bumps this literal in the same
#: change. CI resolves botocore freely (``pip install -e ".[dev]"`` with
#: an unbounded ``boto3>=1.34`` floor, and ``uv.lock`` is gitignored), so
#: a *live* model revision can also surface here without any local
#: change — that is the intended drift detector, not a defect.
SCHEMA_VERSION = "botocore-1.43.93"

#: The 12 wire-body keys the ``bedrock-runtime:Converse`` operation publishes.
#: ``modelId`` is a URI parameter on the real wire (``POST /model/{modelId}/...``)
#: and is **not** in this set — the reader reads it from
#: :attr:`~harness.contract.CapturedRequest.path`.  Static, per the Gemini
#: precedent (:mod:`harness.reader_gemini`); a schema revision that adds a
#: wire-body key fails :class:`TestSchemaAgreement` here rather than as a
#: silent residual in T-D5.
PUBLISHED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "additionalModelRequestFields",
        "additionalModelResponseFieldPaths",
        "guardrailConfig",
        "inferenceConfig",
        "messages",
        "outputConfig",
        "performanceConfig",
        "promptVariables",
        "requestMetadata",
        "serviceTier",
        "system",
        "toolConfig",
    }
)

#: Every member of the Converse ``ContentBlock`` union, as the wire spells it.
#: The reader dispatches each into a first-class part — an ``Image``,
#: ``ToolUse``, ``ToolResult``, ``Thinking`` or ``Opaque`` — via
#: :func:`_read_content_blocks`.  See req 8.
PUBLISHED_CONTENT_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        "audio",
        "cachePoint",
        "citationsContent",
        "document",
        "guardContent",
        "image",
        "reasoningContent",
        "searchResult",
        "text",
        "toolAddition",
        "toolRemoval",
        "toolResult",
        "toolUse",
        "video",
    }
)

#: Members of the ``SystemContentBlock`` union.
PUBLISHED_SYSTEM_CONTENT_BLOCK_TYPES: frozenset[str] = frozenset(
    {"cachePoint", "guardContent", "text"}
)

#: Members of the ``Tool`` union. ``cachePoint`` and ``systemTool`` here do
#: **not** become Opaque parts — they are envelope extras (req 7), because
#: :attr:`~harness.contract.Conversation.tools` is a sequence of
#: :class:`~harness.contract.ToolDecl` and a server-side capability toggle
#: has no ``name`` to bind.
PUBLISHED_TOOL_BLOCK_TYPES: frozenset[str] = frozenset({"cachePoint", "systemTool", "toolSpec"})

#: Members of the ``ToolResultContentBlock`` union.
PUBLISHED_TOOL_RESULT_CONTENT_BLOCK_TYPES: frozenset[str] = frozenset(
    {"document", "image", "json", "searchResult", "text", "video"}
)

#: Converse's sampling keys (botocore ``InferenceConfiguration``).  Each maps
#: onto the closed §3.3.1b set (``temperature``, ``top_p``, ``max_tokens``,
#: ``stop``).
PUBLISHED_INFERENCE_CONFIG_KEYS: frozenset[str] = frozenset(
    {"maxTokens", "stopSequences", "temperature", "topP"}
)

#: ``ToolChoice`` union — Converse has no ``none``.  A ``toolChoice.none`` on
#: the wire is an unrecognised member and residualises at its path
#: (``toolConfig.toolChoice.none``).
PUBLISHED_TOOL_CHOICE_MEMBERS: frozenset[str] = frozenset({"any", "auto", "tool"})

#: ``ToolSpecification`` keys — ``name`` and ``inputSchema`` are required,
#: ``description`` and ``strict`` optional.
PUBLISHED_TOOL_SPECIFICATION_KEYS: frozenset[str] = frozenset(
    {"description", "inputSchema", "name", "strict"}
)

#: ``ImageSource`` union — exactly two members.
PUBLISHED_IMAGE_SOURCE_MEMBERS: frozenset[str] = frozenset({"bytes", "s3Location"})

#: ``ToolResultStatus`` enum values.
PUBLISHED_TOOL_RESULT_STATUS_VALUES: frozenset[str] = frozenset({"error", "success"})

#: ``ReasoningContentBlock`` members — ``reasoningText`` → Thinking,
#: ``redactedContent`` → Opaque(kind="redacted_thinking").
PUBLISHED_REASONING_CONTENT_BLOCK_MEMBERS: frozenset[str] = frozenset(
    {"reasoningText", "redactedContent"}
)

# --------------------------------------------------------------------------
# Wire-shape constants
# --------------------------------------------------------------------------

#: The published Bedrock Runtime Converse path: ``POST /model/{modelId}/converse[-stream]``.
#: The model and operation are both URI parameters — they are not in the body.
_ROUTE = re.compile(r"^/model/(?P<model>[^/]+)/(?P<operation>converse(?:-stream)?)$")

#: The two operations botocore publishes on the Converse resource.  Any other
#: value is not a recognised Converse operation — :class:`UnreadableBodyError`,
#: not a residual, because there is no partial projection to salvage when the
#: model cannot be read at all.
_OPERATIONS = frozenset({"converse", "converse-stream"})

#: Converse sampling key map (wire spelling → canonical §3.3.1b spelling).
_SAMPLING_KEY_MAP: Mapping[str, str] = {
    "maxTokens": "max_tokens",
    "temperature": "temperature",
    "topP": "top_p",
    "stopSequences": "stop",
}

#: Envelope extras — the eight declared control fields Converse publishes.
#: Each lands at ``envelope.extra[<wire key>]``.
_TOP_LEVEL_EXTRA_KEYS: tuple[str, ...] = (
    "additionalModelRequestFields",
    "additionalModelResponseFieldPaths",
    "guardrailConfig",
    "outputConfig",
    "performanceConfig",
    "promptVariables",
    "requestMetadata",
    "serviceTier",
)

#: ContentBlock discriminators that project to a first-class part rather than
#: :class:`~harness.contract.Opaque`.  Any other wire spelling on a block
#: falls through to :func:`_opaque_for`.
_FIRST_CLASS_DISCRIMINATORS: frozenset[str] = frozenset(
    {"text", "image", "toolUse", "toolResult", "reasoningContent"}
)

#: ToolResultContentBlock discriminators that project to a first-class part
#: rather than :class:`~harness.contract.Opaque`.
_FIRST_CLASS_RESULT_DISCRIMINATORS: frozenset[str] = frozenset(
    {"text", "json", "image"}
)


# --------------------------------------------------------------------------
# Public entry
# --------------------------------------------------------------------------


class BedrockConverseProjection:
    """Reads a Bedrock Converse request into :class:`~harness.contract.Request`.

    Implements :class:`~harness.contract.Projection` for
    :attr:`~harness.contract.WireFormat.BEDROCK_CONVERSE`.

    The reader consumes the URL first (the model is in the path, the operation
    in the path segment), then the body — exactly the shape Gemini follows.
    No other reader but Gemini consumes the URL; this is the second of the
    six to do so.

    Attributes:
        wire_format: Always :attr:`~harness.contract.WireFormat.BEDROCK_CONVERSE`.
    """

    wire_format = c.WireFormat.BEDROCK_CONVERSE

    def read_request(self, captured: c.CapturedRequest) -> c.Request:
        """Project a captured Converse request.

        Args:
            captured: The request as observed on the wire. The path carries
                the ``modelId`` URI parameter and the ``converse`` /
                ``converse-stream`` operation; the body carries the rest.

        Returns:
            The wire-independent projection, total over the body.

        Raises:
            UnreadableBodyError: When the body is not a readable Converse
                request — malformed JSON, a non-object body, an unrecognised
                role, a content block with no discriminator, and the rest of
                the failure-shape list. ``KeyError``/``TypeError``/``AttributeError``
                are caught at the boundary: from inside a reader they mean
                the reader mis-routed a field, which is a reader bug and
                must surface — but as ``UnreadableBodyError`` rather than the
                raw ``KeyError``, which would be a fourth failure shape
                T-D1 does not expect.
        """
        model, stream = _read_route(captured.path)
        body = _parse_body(captured.body)

        try:
            return _project(body, model, stream)
        except (KeyError, TypeError, AttributeError) as exc:
            raise c.UnreadableBodyError(
                f"unreadable Bedrock Converse body: {exc!r}"
            ) from exc


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _read_route(path: str) -> tuple[str, bool]:
    """Read the model and the streaming flag out of the request path.

    Bedrock Converse's published path is ``/model/{modelId}/converse`` or
    ``/model/{modelId}/converse-stream``.  The model and the operation are
    both URI parameters — they are not in the body.  The Gemini reader is
    the existing pattern (§3.3.5's "two requests with byte-identical bodies
    are told apart by their paths" claim); this is the second reader to
    consume the URL.

    Args:
        path: :attr:`~harness.contract.CapturedRequest.path`, including the
            ``converse`` or ``converse-stream`` segment.

    Returns:
        The model name and whether the operation streams.

    Raises:
        UnreadableBodyError: When the path is not one of the two published
            Converse routes. Structural: there is no partial projection to
            salvage when the model cannot be read at all.
    """
    matched = _ROUTE.match(path)
    if matched is None:
        raise c.UnreadableBodyError(
            f"path {path!r} is not a published Bedrock Converse route "
            "(/model/{{modelId}}/converse[-stream])"
        )

    operation = matched.group("operation")
    if operation not in _OPERATIONS:
        raise c.UnreadableBodyError(
            f"operation {operation!r} is not one of {sorted(_OPERATIONS)}"
        )

    return matched.group("model"), operation == "converse-stream"


def _parse_body(raw: bytes) -> Mapping[str, Any]:
    """Decode the request body into a JSON object.

    Args:
        raw: The raw body bytes.

    Returns:
        The parsed body.

    Raises:
        UnreadableBodyError: When the bytes are not JSON, or are JSON that is
            not an object. A JSON array is valid JSON and an invalid Converse
            request — it cannot carry a ``messages`` field.
    """
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise c.UnreadableBodyError(f"body is not valid JSON: {exc}") from exc

    if not isinstance(body, dict):
        raise c.UnreadableBodyError(
            f"body must be a JSON object, got {type(body).__name__}"
        )

    return body


def _project(body: Mapping[str, Any], model: str, stream: bool) -> c.Request:
    """Classify every top-level key into the envelope, the conversation or the residual.

    Args:
        body: The parsed request body.
        model: The model, read from the path.
        stream: Whether the path's operation streams.

    Returns:
        The projection.

    Raises:
        UnreadableBodyError: Propagated from the per-field readers.
    """
    residual: dict[str, Any] = {}
    extra: dict[str, Any] = {}

    for name in _TOP_LEVEL_EXTRA_KEYS:
        if name in body:
            extra[name] = body[name]

    extra.update(_read_tool_choice(body, residual))

    system = _read_system(body, residual)
    sampling = _read_inference_config(body, residual)
    tools, tool_extras = _read_tools(body, residual)
    extra.update(tool_extras)

    conversation = c.Conversation(
        system=system,
        turns=_read_messages(body, residual),
        tools=tools,
        sampling=sampling,
    )

    envelope = c.Envelope(model=model, stream=stream, extra=extra)

    _residualise_top_level(body, residual)

    consumed = frozenset(name for name in body if name in PUBLISHED_TOP_LEVEL_KEYS)
    return c.Request(
        envelope=envelope,
        conversation=conversation,
        residual=residual,
        consumed=consumed,
        source=body,
    )


def _residualise_top_level(body: Mapping[str, Any], residual: dict[str, Any]) -> None:
    """Residualise every top-level body key the reader did not consume.

    A reader that drops an unknown key leaves the residual empty and sails
    through :func:`harness.contract.verify_total`.  The ``consumed`` set on
    the projection is the second-line defence: a key in the body but not in
    ``consumed`` is the dropped-key detector — verify_total raises
    :class:`~harness.contract.DroppedFieldsError` for it.  See
    :class:`TestFalsification`'s R7.3 case.

    Args:
        body: The parsed request body.
        residual: The reader's accumulator of unclassifiable values, mutated here.
    """
    for name in body:
        if name in PUBLISHED_TOP_LEVEL_KEYS:
            continue
        if name in residual:
            continue
        residual[name] = body[name]


# --------------------------------------------------------------------------
# Per-field readers
# --------------------------------------------------------------------------


def _read_system(
    body: Mapping[str, Any], residual: dict[str, Any]
) -> tuple[c.Text, ...]:
    """Lift system blocks into :attr:`~harness.contract.Conversation.system`.

    Converse publishes a ``system`` array of :class:`SystemContentBlock`.
    Three union members: ``text``, ``cachePoint``, ``guardContent``. The
    text block lifts to :class:`~harness.contract.Text`; the others are
    unmodelled carriers for the same control surface Converse carries on a
    content block, and become :class:`~harness.contract.Opaque` with
    canonical kinds (``cache_point`` / ``guard_content``). Unknown extras
    on each block residualise at the block's path (§7.4.1).

    Args:
        body: The parsed request body.
        residual: The reader's accumulator, mutated here.

    Returns:
        The system text parts in order.

    Raises:
        UnreadableBodyError: When ``system`` is not a list, or any entry
            is not an object.
    """
    if "system" not in body:
        return ()

    value = body["system"]
    if not isinstance(value, list):
        raise c.UnreadableBodyError(
            f"system must be a list, got {type(value).__name__}"
        )

    parts: list[c.Text] = []
    for index, block in enumerate(value):
        path = f"system[{index}]"
        if not isinstance(block, Mapping):
            raise c.UnreadableBodyError(
                f"{path} must be an object, got {type(block).__name__}"
            )

        if "text" in block:
            text = block["text"]
            if isinstance(text, str):
                parts.append(c.Text(text))
            else:
                residual[f"{path}.text"] = text
            _residualise_block_extras(block, path, {"text"}, residual)
            continue

        # ``cachePoint`` and ``guardContent`` are Opaque kinds; a System
        # block carrying them is unmodelled control content. The system
        # text list does not retain them — they are not text — but the
        # totality check needs a place for them. The reader documents this
        # contract: a SystemContentBlock other than ``text`` residualises at
        # the entry path; the projection's ``system`` sequence is text only.
        for discriminator in ("cachePoint", "guardContent"):
            if discriminator in block:
                residual[path] = _opaque_for(block, discriminator)
                _residualise_block_extras(block, path, {discriminator}, residual)
                break
        else:
            residual[path] = block

    return tuple(parts)


def _read_inference_config(
    body: Mapping[str, Any], residual: dict[str, Any]
) -> Mapping[str, Any]:
    """Read :class:`InferenceConfiguration`, normalising onto the §3.3.1b closed set.

    The botocore schema publishes four keys (``maxTokens``, ``temperature``,
    ``topP``, ``stopSequences``); §3.3.1b maps each onto a Chat Completions
    spelling (``max_tokens``, ``temperature``, ``top_p``, ``stop``).
    An unrecognised child key residualises at ``inferenceConfig.<key>``.

    Args:
        body: The parsed request body.
        residual: The reader's accumulator, mutated here.

    Returns:
        The frozen sampling mapping — keys are canonical spellings.
    """
    if "inferenceConfig" not in body:
        return MappingProxyType({})

    value = body["inferenceConfig"]
    if not isinstance(value, Mapping):
        residual["inferenceConfig"] = value
        return MappingProxyType({})

    sampling: dict[str, Any] = {}
    for wire_key, canonical in _SAMPLING_KEY_MAP.items():
        if wire_key not in value:
            continue
        child = value[wire_key]

        # §3.3.1b's typed-leaf rule: a wrong-typed optional leaf residualises
        # at the leaf path, not the parent's. T-A1's reviewer experience
        # found ~9/10 optional leaves failing open without this test (the
        # ``TestEveryOptionalLeafFailsClosed`` class). The four InferenceConfig
        # keys each have a distinct primitive type: ``maxTokens`` an integer,
        # ``temperature`` and ``topP`` floats, ``stopSequences`` a string
        # array. ``bool`` is a subclass of ``int`` in Python, so a
        # wrong-typed boolean would pass the int check if not excluded.
        if wire_key == "maxTokens" and (not isinstance(child, int) or isinstance(child, bool)):
            residual[f"inferenceConfig.{wire_key}"] = child
            continue
        if wire_key in ("temperature", "topP") and (
            not isinstance(child, (int, float)) or isinstance(child, bool)
        ):
            residual[f"inferenceConfig.{wire_key}"] = child
            continue
        if wire_key == "stopSequences" and (
            not isinstance(child, list) or any(not isinstance(s, str) for s in child)
        ):
            residual[f"inferenceConfig.{wire_key}"] = child
            continue

        sampling[canonical] = child

    # Unknown children residualise at their leaf path. This is the depth the
    # totality check at the top level cannot see (§3.3.1a's "every field" rule).
    for wire_key in value:
        if wire_key in PUBLISHED_INFERENCE_CONFIG_KEYS:
            continue
        residual[f"inferenceConfig.{wire_key}"] = value[wire_key]

    return MappingProxyType(sampling)


def _read_tool_choice(
    body: Mapping[str, Any], residual: dict[str, Any]
) -> dict[str, Any]:
    """Read ``toolConfig.toolChoice`` and surface it as ``tool_choice`` in extra.

    The four wire spellings that name ``tool_choice`` (§3.3.1b) — Converse
    carries exactly one of them: ``toolConfig.toolChoice``. Three union
    members (``any``, ``auto``, ``tool``); a ``tool`` choice carries the
    targeted tool name. Converse has **no** ``none`` choice — a
    ``toolChoice.none`` on the wire is an unrecognised union member and
    residualises at ``toolConfig.toolChoice.none`` rather than being passed
    through. The ``Envelope`` constructor accepts ``"none"`` because it is
    canonical for the other wire formats; the failure mode is the depth,
    not the constructor.

    Args:
        body: The parsed request body.
        residual: The reader's accumulator, mutated here.

    Returns:
        The ``tool_choice`` extra entry, or empty when the body carries no
        ``toolConfig`` or no ``toolChoice``.
    """
    if "toolConfig" not in body:
        return {}

    tool_config = body["toolConfig"]
    if not isinstance(tool_config, Mapping):
        residual["toolConfig"] = tool_config
        return {}

    choice = tool_config.get("toolChoice")
    if choice is None:
        return {}
    if not isinstance(choice, Mapping):
        residual["toolConfig.toolChoice"] = choice
        return {}

    if "auto" in choice:
        _residualise_block_extras(choice, "toolConfig.toolChoice", {"auto"}, residual)
        return {"tool_choice": "auto"}
    if "any" in choice:
        _residualise_block_extras(choice, "toolConfig.toolChoice", {"any"}, residual)
        return {"tool_choice": "any"}
    if "tool" in choice:
        target = choice["tool"]
        if isinstance(target, Mapping) and isinstance(target.get("name"), str):
            _residualise_block_extras(choice, "toolConfig.toolChoice", {"tool"}, residual)
            return {"tool_choice": f"tool:{target['name']}"}
        residual["toolConfig.toolChoice.tool"] = target
        _residualise_block_extras(choice, "toolConfig.toolChoice", {"tool"}, residual)
        return {}

    # Unrecognised discriminator — every key on the choice residualises at
    # its own depth. ``none`` is the headline case: an unknown Converse
    # choice subkey must not silently propagate to the canonical tool_choice.
    for discriminator in choice:
        residual[f"toolConfig.toolChoice.{discriminator}"] = choice[discriminator]
    return {}


def _read_tools(
    body: Mapping[str, Any], residual: dict[str, Any]
) -> tuple[tuple[c.ToolDecl, ...], dict[str, Any]]:
    """Read tool declarations out of ``toolConfig.tools``.

    Converse publishes three ``Tool`` union members. ``cachePoint`` and
    ``systemTool`` are server-side capability toggles with no ``name`` to
    bind — they map to ``envelope.extra[<wire key>]`` per §7.4.2 rule 5,
    not to :class:`~harness.contract.ToolDecl`. ``toolSpec`` carries the
    ``name``, ``description``, ``inputSchema``, ``strict`` fields and is
    the only one that becomes a :class:`~harness.contract.ToolDecl`.

    Args:
        body: The parsed request body.
        residual: The reader's accumulator, mutated here.

    Returns:
        The declared tool specifications and the toggles extra dict.
    """
    if "toolConfig" not in body:
        return (), {}

    tool_config = body["toolConfig"]
    if not isinstance(tool_config, Mapping):
        return (), {}

    tools_value = tool_config.get("tools")
    if tools_value is None:
        return (), {}
    if not isinstance(tools_value, list):
        raise c.UnreadableBodyError(
            f"toolConfig.tools must be a list, got {type(tools_value).__name__}"
        )

    extras: dict[str, Any] = {}
    declarations: list[c.ToolDecl] = []

    for index, entry in enumerate(tools_value):
        path = f"toolConfig.tools[{index}]"
        if not isinstance(entry, Mapping):
            raise c.UnreadableBodyError(
                f"{path} must be an object, got {type(entry).__name__}"
            )

        if "cachePoint" in entry:
            _toggle_or_residualise(entry, path, "cachePoint", "cachePoint", extras, residual)
            continue
        if "systemTool" in entry:
            _toggle_or_residualise(entry, path, "systemTool", "systemTool", extras, residual)
            continue
        if "toolSpec" in entry:
            tool_spec = entry["toolSpec"]
            if not isinstance(tool_spec, Mapping):
                residual[f"{path}.toolSpec"] = tool_spec
                continue
            tool = _read_tool_specification(tool_spec, path, residual)
            if tool is not None:
                declarations.append(tool)
            _residualise_block_extras(entry, path, {"toolSpec"}, residual)
            continue

        # Unrecognised discriminator — every other entry key on this tool
        # residualises at the entry path.
        residual[path] = entry

    return tuple(declarations), extras


def _toggle_or_residualise(
    entry: Mapping[str, Any],
    entry_path: str,
    discriminator: str,
    extras_key: str,
    extras: dict[str, Any],
    residual: dict[str, Any],
) -> None:
    """Accumulate a tool-config toggle, or residualise a duplicate.

    Converse allows up to four ``cachePoint`` blocks per request (system,
    messages content, tools) — the schema does not constrain count. Two
    ``{"cachePoint": {...}}`` entries inside ``toolConfig.tools`` therefore
    schema-legal; the reader takes the first value and residualises the
    rest at their entry path so a future row anchored at
    ``envelope.extra[cachePoint]`` does not silently lose the count.
    """
    value = entry[discriminator]
    if discriminator in extras:
        # Loser overwrites winner with no residual would be a §7.4.2 rule 2
        # hazard; we surface the duplicate at the entry's path.
        residual[f"{entry_path}.{discriminator}"] = value
    else:
        extras[extras_key] = value
    _residualise_block_extras(entry, entry_path, {discriminator}, residual)


def _read_tool_specification(
    tool_spec: Mapping[str, Any], prefix: str, residual: dict[str, Any]
) -> c.ToolDecl | None:
    """Read a single ``ToolSpecification`` block.

    ``name`` and ``inputSchema`` are required (the schema marks them
    required); ``description`` and ``strict`` are optional. A missing
    required field residualises at its own path and the tool declaration
    is **not** produced — a tool without a name cannot be paired with its
    result or addressed by a register row, the same ground on which
    :data:`~harness.contract.STOP_REASONS` gives ``other`` its escape
    instead of the residual.

    Args:
        tool_spec: The ``toolSpec`` value, already known to be a Mapping.
        prefix: The dotted prefix of the tool entry, e.g.
            ``"toolConfig.tools[2]"``.
        residual: The reader's accumulator, mutated here.

    Returns:
        The :class:`~harness.contract.ToolDecl`, or ``None`` when a required
        field is missing or wrongly typed.
    """
    name = tool_spec.get("name")
    if not isinstance(name, str):
        residual[f"{prefix}.toolSpec.name"] = name
        return None

    input_schema = tool_spec.get("inputSchema")
    if not isinstance(input_schema, Mapping):
        residual[f"{prefix}.toolSpec.inputSchema"] = input_schema
        return None

    description = tool_spec.get("description")
    if description is None:
        description_value: str | None = None
    elif isinstance(description, str):
        description_value = description
    else:
        residual[f"{prefix}.toolSpec.description"] = description
        description_value = None

    strict = tool_spec.get("strict")
    if strict is None:
        strict_value: bool | None = None
    elif isinstance(strict, bool):
        strict_value = strict
    else:
        residual[f"{prefix}.toolSpec.strict"] = strict
        strict_value = None

    for wire_key in tool_spec:
        if wire_key in PUBLISHED_TOOL_SPECIFICATION_KEYS:
            continue
        residual[f"{prefix}.toolSpec.{wire_key}"] = tool_spec[wire_key]

    return c.ToolDecl(
        name=name,
        description=description_value,
        schema=MappingProxyType(input_schema) if input_schema else None,
        strict=strict_value,
    )


def _read_messages(
    body: Mapping[str, Any], residual: dict[str, Any]
) -> tuple[c.Turn, ...]:
    """Read the ``messages`` array into the canonical turn list.

    Converse carries ``messages`` as an array of :class:`Message` —
    ``role`` (``user`` or ``assistant``) and ``content`` (an array of
    :class:`ContentBlock`). The reader applies §3.3.1b's four merge clauses
    to the resulting parts: toolResult parts come first within the run a
    single turn builds; consecutive same-role turns merge.

    Structural failures (non-list ``messages``, non-object message, missing
    or unrecognised role, non-list content) are
    :class:`~harness.contract.UnreadableBodyError`, not residuals — §7.4.1
    names "a role outside ``user``/``assistant``" as the example of a
    failure with no partial projection to salvage. Per-leaf failures
    (a content block with a missing required field) are residualised at
    the leaf path.

    Args:
        body: The parsed request body.
        residual: The reader's accumulator, mutated here.

    Returns:
        The ordered, normalised turns.

    Raises:
        UnreadableBodyError: When the ``messages`` array cannot be walked
            — a non-list, a non-object entry, a role outside the published
            set, a non-list content. The reader has no partial projection
            to salvage at that point.
    """
    if "messages" not in body:
        return ()

    messages = body["messages"]
    if not isinstance(messages, list):
        raise c.UnreadableBodyError(
            f"messages must be a list, got {type(messages).__name__}"
        )

    turns: list[c.Turn] = []
    for index, message in enumerate(messages):
        path = f"messages[{index}]"
        if not isinstance(message, Mapping):
            raise c.UnreadableBodyError(f"{path} must be an object, got {type(message).__name__}")

        role = message.get("role")
        if role not in ("user", "assistant"):
            raise c.UnreadableBodyError(
                f"{path}.role must be 'user' or 'assistant', got {role!r}; "
                "Converse has no system role — a system prompt is the top-level system"
            )

        content = message.get("content")
        if not isinstance(content, list):
            raise c.UnreadableBodyError(
                f"{path}.content must be a list, got {type(content).__name__}"
            )

        parts = _read_content_blocks(content, f"{path}.content", residual)
        if not parts:
            continue

        turn = c.Turn(role, parts)

        # §3.3.1b merge clause 4: consecutive same-role turns merge.
        if turns and turns[-1].role == turn.role:
            turns[-1] = c.Turn(turn.role, _tool_results_first(turns[-1].parts + turn.parts))
        else:
            turns.append(turn)

    return tuple(turns)


def _read_content_blocks(
    content: Sequence[Mapping[str, Any]],
    prefix: str,
    residual: dict[str, Any],
) -> tuple[c.Part, ...]:
    """Read one ``Message.content`` array, dispatching each block.

    The 14 ContentBlock union members are dispatched in the order their
    discriminators appear: ``text``/``image``/``toolUse``/``toolResult``/
    ``reasoningContent`` → first-class parts; the eight Opaque-only
    discriminators (``cachePoint``, ``guardContent``, ``document``,
    ``video``, ``audio``, ``searchResult``, ``citationsContent``,
    ``toolAddition``, ``toolRemoval``) fall through to :func:`_opaque_for`
    via the alias table.

    Per-block extras (keys the dispatch did not consume) residualise at
    the block's prefix — §7.4.1's depth rule. Structural failures
    (non-object block) raise :class:`~harness.contract.UnreadableBodyError`.

    Args:
        content: The ``content`` list, already known to be a list.
        prefix: The dotted prefix of the parent, e.g. ``"messages[2].content"``.
        residual: The reader's accumulator, mutated here.

    Returns:
        The ordered parts, with tool-result runs placed first within each
        turn (§3.3.1b clause 1).

    Raises:
        UnreadableBodyError: When a block is not an object.
    """
    parts: list[c.Part] = []
    for index, block in enumerate(content):
        path = f"{prefix}[{index}]"
        if not isinstance(block, Mapping):
            raise c.UnreadableBodyError(
                f"{path} must be an object, got {type(block).__name__}"
            )

        if "text" in block:
            text = block["text"]
            if isinstance(text, str):
                parts.append(c.Text(text))
            else:
                residual[f"{path}.text"] = text
            _residualise_block_extras(block, path, {"text"}, residual)
        elif "image" in block:
            image = block["image"]
            if isinstance(image, Mapping):
                part = _read_image(image, f"{path}.image", residual)
                if part is not None:
                    parts.append(part)
            else:
                residual[f"{path}.image"] = image
            _residualise_block_extras(block, path, {"image"}, residual)
        elif "toolUse" in block:
            part = _read_tool_use(block["toolUse"], f"{path}.toolUse", residual)
            if part is not None:
                parts.append(part)
            _residualise_block_extras(block, path, {"toolUse"}, residual)
        elif "toolResult" in block:
            part = _read_tool_result(block["toolResult"], f"{path}.toolResult", residual)
            if part is not None:
                parts.append(part)
            _residualise_block_extras(block, path, {"toolResult"}, residual)
        elif "reasoningContent" in block:
            part = _read_reasoning_content(
                block["reasoningContent"], f"{path}.reasoningContent", residual
            )
            if part is not None:
                parts.append(part)
            _residualise_block_extras(block, path, {"reasoningContent"}, residual)
        else:
            # Eight Opaque-only ContentBlock discriminators — dispatch on the
            # first matching discriminator key.
            for discriminator in PUBLISHED_CONTENT_BLOCK_TYPES:
                if discriminator not in _FIRST_CLASS_DISCRIMINATORS and discriminator in block:
                    parts.append(_opaque_for(block, discriminator))
                    _residualise_block_extras(block, path, {discriminator}, residual)
                    break
            else:
                # No discriminator recognised — block-level residualisation.
                residual[path] = block

    return tuple(_tool_results_first(parts))


def _residualise_block_extras(
    block: Mapping[str, Any],
    prefix: str,
    consumed_keys: set[str],
    residual: dict[str, Any],
) -> None:
    """Residualise any keys on a content/system block the dispatch did not consume.

    §7.4.1's depth rule: a value the projection did not classify fails
    closed at the path it sits on, even though ``verify_total`` cannot see
    past the top level. The ``consumed_keys`` set names the keys the
    dispatch above took — anything else on the block is unclassified.

    Args:
        block: The content or system block, already known to be a Mapping.
        prefix: The dotted prefix of the block (e.g. ``"messages[0].content[2]"``).
        consumed_keys: The keys the per-block dispatch consumed.
        residual: The reader's accumulator, mutated here.
    """
    for wire_key in block:
        if wire_key in consumed_keys:
            continue
        residual[f"{prefix}.{wire_key}"] = block[wire_key]


def _read_image(
    image: Mapping[str, Any], prefix: str, residual: dict[str, Any]
) -> c.Image:
    """Project an :class:`ImageBlock` into :class:`~harness.contract.Image`.

    The schema publishes ``format`` (required) and ``source`` (required).
    ``source`` is one of ``bytes`` (raw bytes — digest via
    :func:`harness.contract.image_digest`) or ``s3Location`` (URI only —
    ``Image.ref``, no digest; §3.3.1's unpinned rule).

    §7.4 rule 7 row 3 ("no branch returns *no part*"): when the wire's
    payload is undecodable or a leaf is wrongly typed, the part is still
    produced — residualised at the leaf, the position occupied — so the
    drop does not shift the indices of every later part. ``Image.digest``
    is computed by :func:`harness.contract.image_digest` from the wire's
    raw bytes for the undecodable case (KBR-192's second recipe) and
    from the canonical-JSON of the malformed leaf otherwise. The KBR-251
    conformance note in the design names this exact shape.

    Args:
        image: The ``image`` value, already known to be a Mapping.
        prefix: The dotted prefix of the parent block.
        residual: The reader's accumulator of unclassifiable values, mutated here.

    Returns:
        The :class:`~harness.contract.Image`. A wrongly-typed leaf does
        not yield ``None`` — the part is always produced, identity from
        the wire's own bytes (KBR-192).
    """
    fmt = image.get("format")
    source = image.get("source")
    image_extras = set(image) - {"format", "source", "error"}

    media_type: str | None
    if isinstance(fmt, str):
        media_type = _normalise_media_type(fmt)
    else:
        residual[f"{prefix}.format"] = fmt
        # The wire carried a non-string ``format``. Carry whatever it said
        # as the media_type — the reader is honest about what the wire did,
        # and the residual surfaces the wire-format breach at its own path.
        media_type = None if fmt is None else str(fmt)

    digest: str | None = None
    ref: str | None = None

    if "bytes" in source:
        raw = source["bytes"]
        # AWS JSON protocol serialises blob members as base64-encoded
        # strings; the wire never carries raw bytes for a blob. A non-string
        # value at this depth is a wire-format breach.
        if isinstance(raw, str):
            try:
                decoded = base64.b64decode(raw, validate=True)
            except (ValueError, TypeError):
                # §7.4 rule 7 / KBR-251: residualise the leaf AND keep the
                # part, identifying it by the wire's own raw bytes (KBR-192's
                # second ``image_digest`` recipe).
                residual[f"{prefix}.source.bytes"] = raw
                digest = c.image_digest(raw.encode("utf-8"))
            else:
                digest = c.image_digest(decoded)
        else:
            # Non-string at the leaf — residualise the leaf AND project
            # the part using the wire's own bytes (KBR-251 conformed).
            residual[f"{prefix}.source.bytes"] = raw
            digest = _wire_identity_digest(raw)
        # Source-level siblings that the ``bytes`` union member did not
        # consume (the schema is exactly one of ``bytes`` / ``s3Location``).
        for wire_key in source:
            if wire_key != "bytes":
                residual[f"{prefix}.source.{wire_key}"] = source[wire_key]
    elif "s3Location" in source:
        s3 = source["s3Location"]
        if isinstance(s3, Mapping) and isinstance(s3.get("uri"), str):
            ref = s3["uri"]
        else:
            # Malformed ``s3Location`` — residualise the leaf AND project
            # the part with a canonical-JSON identity digest. ``ref`` is
            # left ``None`` because no string was carried.
            residual[f"{prefix}.source.s3Location"] = s3
            digest = _wire_identity_digest(s3)
        # Source-level siblings that the ``s3Location`` union member did
        # not consume.
        for wire_key in source:
            if wire_key != "s3Location":
                residual[f"{prefix}.source.{wire_key}"] = source[wire_key]
    else:
        # Source carried neither ``bytes`` nor ``s3Location`` — a wire-format
        # breach. §7.4 rule 7: residualise the leaf AND project the part.
        residual[f"{prefix}.source"] = source
        digest = _wire_identity_digest(source)
        for wire_key in source:
            residual[f"{prefix}.source.{wire_key}"] = source[wire_key]

    # Image-object-level siblings (e.g. an ``error`` block, a future field).
    for wire_key in image_extras:
        residual[f"{prefix}.{wire_key}"] = image[wire_key]

    return c.Image(digest=digest, ref=ref, media_type=media_type)


def _wire_identity_digest(value: Any) -> str:
    """A distinguishing digest of a wire value the reader cannot canonicalise.

    Uses the second of :func:`harness.contract.image_digest`'s recipes
    (KBR-192, "of the raw encoded bytes the wire carried") — the
    canonical-JSON serialisation of the value. The result identifies the
    position from what the wire actually said, so a future schema
    revision can re-derive the row at this position without ambiguity.
    """
    try:
        return c.image_digest(
            json.dumps(value, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
        )
    except TypeError:
        # ``default=str`` already handles the unhashable case; this is the
        # belt-and-braces guard for a custom object that resists repr.
        return c.image_digest(repr(value).encode("utf-8"))


def _read_tool_use(
    raw: Any, prefix: str, residual: dict[str, Any]
) -> c.ToolUse:
    """Project a :class:`ToolUseBlock` into :class:`~harness.contract.ToolUse`.

    The schema publishes ``toolUseId``, ``name``, ``input`` (``Document``)
    as required. The reader carries ``id`` through; ``name`` and ``input``
    are required — a missing or wrong-typed value residualises at its
    path AND the part is produced (KBR-251 / §7.4 rule 7 row 3), so the
    position is occupied and the indices of every later part do not shift.

    A non-Mapping ``raw`` — the member itself is wrong, the schema says
    an object and the wire sent a scalar — raises
    :class:`~harness.contract.UnreadableBodyError` per rule 7's
    member-wrong row.

    Args:
        raw: The ``toolUse`` value.
        prefix: The dotted prefix of the parent block.
        residual: The reader's accumulator, mutated here.

    Returns:
        The :class:`~harness.contract.ToolUse`. A wrongly-typed leaf does
        not yield ``None`` — the part is always produced, identity from
        a sentinel ``name`` (``""``) and ``id`` (``None``) when those
        leaves are wrong, and the wire's own bytes for the rest.
    """
    if not isinstance(raw, Mapping):
        # The member itself is wrong (§7.4 rule 7 row 2) — the schema
        # declares an object and the wire sent something else. No value
        # to put in the position; the request is unreadable.
        raise c.UnreadableBodyError(
            f"{prefix} must be an object, got {type(raw).__name__}"
        )

    name = raw.get("name")
    if isinstance(name, str):
        name_value = name
    else:
        residual[f"{prefix}.name"] = name
        # §7.4 rule 7: occupy the position. An empty-string name is a
        # wire-format breach, but the call still needs an address — it
        # cannot be paired with its result otherwise (§3.3.1b's
        # test-the-projection-can-represent-the-absence-losslessly rule
        # names the same trade-off).
        name_value = ""

    tool_use_id = raw.get("toolUseId")
    if tool_use_id is None or isinstance(tool_use_id, str):
        id_value: str | None = tool_use_id
    else:
        residual[f"{prefix}.toolUseId"] = tool_use_id
        id_value = None

    arguments = raw.get("input")
    # Converse carries ``input`` as a Document (JSON value). The
    # lossless form is the value whole — for an object, pass through;
    # for an array/scalar, residualise at the leaf AND project the part.
    # ``decode_arguments`` is the string-carrying-formats tool only;
    # applying it would reject the very form Converse's wire requires.
    if isinstance(arguments, Mapping):
        arguments_value: Mapping[str, Any] = MappingProxyType(arguments)
    elif arguments is None:
        arguments_value = MappingProxyType({})
    else:
        # §7.4 rule 7: residualise the leaf AND project the part.
        # The argument object is non-empty (`{"x": 1}`) on a real wire;
        # a wire that carried `"input": 1` is a breach but the part still
        # needs an address. The default is the empty object — the same
        # decoder rule §3.3.1b uses for absent arguments.
        residual[f"{prefix}.input"] = arguments
        arguments_value = MappingProxyType({})

    for wire_key in raw:
        if wire_key in ("input", "name", "toolUseId", "type"):
            continue
        residual[f"{prefix}.{wire_key}"] = raw[wire_key]

    return c.ToolUse(name=name_value, arguments=arguments_value, id=id_value)


def _read_tool_result(
    raw: Any, prefix: str, residual: dict[str, Any]
) -> c.ToolResult:
    """Project a :class:`ToolResultBlock` into :class:`~harness.contract.ToolResult`.

    The schema publishes ``toolUseId`` and ``content`` as required, plus an
    optional ``status`` (``success`` / ``error``). The six
    ``ToolResultContentBlock`` union members dispatch to ``Text``, ``Image``,
    ``Json``, or one of three Opaque kinds.

    §7.4 rule 7 ("no branch returns *no part*"): when the wire's payload
    is undecodable or a leaf is wrongly typed, the part is still produced —
    residualised at the leaf, the position occupied — so the drop does
    not shift later indices. The KBR-251 conformance note in the design
    names this exact shape.

    A non-Mapping ``raw`` raises :class:`~harness.contract.UnreadableBodyError`
    per rule 7 row 2 (the member itself is wrong).

    Args:
        raw: The ``toolResult`` value.
        prefix: The dotted prefix of the parent block.
        residual: The reader's accumulator, mutated here.

    Returns:
        The :class:`~harness.contract.ToolResult`. A wrongly-typed leaf does
        not yield ``None`` — the part is always produced.
    """
    if not isinstance(raw, Mapping):
        # The member itself is wrong (§7.4 rule 7 row 2) — the schema
        # declares an object and the wire sent something else.
        raise c.UnreadableBodyError(
            f"{prefix} must be an object, got {type(raw).__name__}"
        )

    raw_tool_use_id = raw.get("toolUseId")
    if isinstance(raw_tool_use_id, str):
        tool_use_id: str | None = raw_tool_use_id
    else:
        # §7.4 rule 7: residualise the leaf AND project the part. ``None``
        # is the documented absent-paired-id value (§3.3.1's ToolResult
        # class docstring); the position is occupied; the residual
        # surfaces the wire-format breach at its own path.
        residual[f"{prefix}.toolUseId"] = raw_tool_use_id
        tool_use_id = None

    content_value = raw.get("content")
    if isinstance(content_value, list):
        content_list: list[Any] = content_value
    else:
        # §7.4 rule 7: residualise the leaf AND project the part. The
        # default is an empty content list — the same absent-value table
        # §7.4 rule 7 row 2 names.
        residual[f"{prefix}.content"] = content_value
        content_list = []

    status = raw.get("status")
    is_error = False
    if status is not None:
        if status not in PUBLISHED_TOOL_RESULT_STATUS_VALUES:
            residual[f"{prefix}.status"] = status
        else:
            is_error = status == "error"

    parts: list[c.Text | c.Image | c.Json | c.Opaque] = []
    for index, block in enumerate(content_list):
        path = f"{prefix}.content[{index}]"
        if not isinstance(block, Mapping):
            residual[path] = block
            continue

        if "text" in block:
            text = block["text"]
            if isinstance(text, str):
                parts.append(c.Text(text))
            else:
                residual[f"{path}.text"] = text
            _residualise_block_extras(block, path, {"text"}, residual)
            continue
        if "json" in block:
            json_value = block["json"]
            parts.append(c.Json(json_value))
            _residualise_block_extras(block, path, {"json"}, residual)
            continue
        if "image" in block:
            image = block["image"]
            if isinstance(image, Mapping):
                part = _read_image(image, f"{path}.image", residual)
                parts.append(part)  # KBR-251: always occupy the position
            else:
                residual[f"{path}.image"] = image
            _residualise_block_extras(block, path, {"image"}, residual)
            continue
        for discriminator in PUBLISHED_TOOL_RESULT_CONTENT_BLOCK_TYPES:
            if discriminator not in _FIRST_CLASS_RESULT_DISCRIMINATORS and discriminator in block:
                parts.append(_opaque_for(block, discriminator))
                _residualise_block_extras(block, path, {discriminator}, residual)
                break
        else:
            residual[path] = block

    for wire_key in raw:
        if wire_key in ("content", "toolUseId", "status", "type"):
            continue
        residual[f"{prefix}.{wire_key}"] = raw[wire_key]

    return c.ToolResult(
        content=tuple(parts),
        tool_use_id=tool_use_id,
        is_error=is_error,
    )


def _read_reasoning_content(
    raw: Any, prefix: str, residual: dict[str, Any]
) -> c.Thinking | c.Opaque:
    """Project a :class:`ReasoningContentBlock` into Thinking or Opaque.

    The block has two union members. ``reasoningText`` carries ``text`` and
    ``signature``; ``redactedContent`` is a blob of redacted reasoning.
    The former projects to :class:`~harness.contract.Thinking` (no
    ``cache_control`` slot — Converse has no such concept). The latter
    projects to :class:`~harness.contract.Opaque` with the canonical kind
    ``redacted_thinking`` reached via :func:`harness.contract.opaque_kind`.

    §7.4 rule 7 ("no branch returns *no part*"): when a leaf is wrong,
    the part is still produced — residualised at the leaf, the position
    occupied. The KBR-251 conformance note in the design names this shape.

    A non-Mapping ``raw`` raises :class:`~harness.contract.UnreadableBodyError`
    per rule 7 row 2 (the member itself is wrong).

    Args:
        raw: The ``reasoningContent`` value.
        prefix: The dotted prefix of the parent block.
        residual: The reader's accumulator, mutated here.

    Returns:
        The :class:`~harness.contract.Thinking` (for ``reasoningText``)
        or :class:`~harness.contract.Opaque` (for ``redactedContent``).
        A wrongly-typed leaf does not yield ``None``.
    """
    if not isinstance(raw, Mapping):
        raise c.UnreadableBodyError(
            f"{prefix} must be an object, got {type(raw).__name__}"
        )

    if "reasoningText" in raw:
        text_block = raw["reasoningText"]
        if not isinstance(text_block, Mapping):
            # §7.4 rule 7: residualise the leaf AND project the part.
            # The default is an empty-text Thinking — the part occupies
            # its position, the residual surfaces the breach.
            residual[f"{prefix}.reasoningText"] = text_block
            _residualise_block_extras(raw, prefix, {"reasoningText"}, residual)
            return c.Thinking(text="", signature=None)
        text = text_block.get("text")
        if isinstance(text, str):
            text_value: str = text
        else:
            # §7.4 rule 7: residualise the leaf AND project the part.
            residual[f"{prefix}.reasoningText.text"] = text
            text_value = ""
        signature = text_block.get("signature")
        if signature is None or isinstance(signature, str):
            signature_value: str | None = signature
        else:
            residual[f"{prefix}.reasoningText.signature"] = signature
            signature_value = None
        # ReasoningText-block siblings — beyond ``text`` and ``signature``.
        _residualise_block_extras(
            text_block, f"{prefix}.reasoningText", {"text", "signature"}, residual
        )
        # reasoningContent-block siblings — beyond ``reasoningText``.
        _residualise_block_extras(raw, prefix, {"reasoningText"}, residual)
        return c.Thinking(text=text_value, signature=signature_value)

    if "redactedContent" in raw:
        # The wire spelling reaches the alias table; the reader passes it
        # through opaque_kind rather than restating the reconciliation.
        _residualise_block_extras(raw, prefix, {"redactedContent"}, residual)
        return _opaque_for(raw, "redactedContent")

    # §7.4 rule 7: residualise the leaf AND project the part. The
    # default is an Opaque identity — the part occupies its position,
    # the residual surfaces the unrecognised union member at its path.
    # The digest is of the wire's own bytes, so two unrecognised
    # reasoning blocks with different wire bytes produce different
    # identities.
    residual[prefix] = raw
    return _opaque_for(raw, "unknown_reasoning")


def _opaque_for(block: Mapping[str, Any], discriminator: str) -> c.Opaque:
    """Build an :class:`~harness.contract.Opaque` for a single-discriminator block.

    Args:
        block: The Converse block, which carries exactly the named discriminator.
        discriminator: The wire spelling of the discriminator (e.g.
            ``"cachePoint"``).

    Returns:
        The Opaque with the canonical kind (via
        :func:`harness.contract.opaque_kind`) and the digest computed by
        :func:`harness.contract.opaque_digest`.
    """
    return c.Opaque(
        kind=c.opaque_kind(discriminator),
        digest=c.opaque_digest(block),
    )


# --------------------------------------------------------------------------
# Helpers shared with the Gemini reader's shape
# --------------------------------------------------------------------------


def _tool_results_first(parts: Sequence[c.Part]) -> tuple[c.Part, ...]:
    """Apply §3.3.1b's first-clause ordering: tool results come first within the run.

    The rule is **per-turn** — a single Message's parts list can carry
    interleaved toolResult and non-toolResult blocks, and the run the merge
    builds puts the toolResult blocks first. Outside that run, ordering is
    preserved verbatim. This matches the Anthropic Messages reader's
    clause-3 implementation.

    Args:
        parts: The ordered parts of one turn.

    Returns:
        The parts with tool-result blocks grouped first; non-tool blocks
        follow in their original relative order.
    """
    tool_results: list[c.Part] = []
    others: list[c.Part] = []
    for part in parts:
        if isinstance(part, c.ToolResult):
            tool_results.append(part)
        else:
            others.append(part)
    return tuple(tool_results + others)


def _normalise_media_type(format_str: str) -> str:
    """Map Converse's bare image format to the conventional ``image/<fmt>`` media type.

    Converse's ``ImageFormat`` enum names ``png``, ``jpeg``, ``gif``,
    ``webp`` (and a few vendor variants) without the ``image/`` prefix.
    Anthropic's Messages format carries the prefixed form. The reader
    normalises here so an image's media_type is comparable across the
    six readers T-D8 diffs.

    Args:
        format_str: The wire spelling — already known to be a str.

    Returns:
        The conventional ``image/<fmt>`` form, or the original string
        when it does not match a known bare format (a vendor-specific
        extension is carried through; the
        :class:`~harness.tests.harness.test_reader_bedrock_converse.TestImageSources`
        class catches a regression here).
    """
    bare = format_str.lower()
    if bare in {"png", "jpeg", "gif", "webp"}:
        return f"image/{bare}"
    return format_str
