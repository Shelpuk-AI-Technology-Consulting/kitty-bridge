"""The Anthropic Messages reader — `POST /v1/messages` into the common form.

`.system_design/TEST_SUITE.md` §3.3.1, §3.3.1a, §3.3.1b, §7.4.1 · plan task
**T-A1** (KBR-33).

This module **imports nothing from** ``src/kitty``, and must not.  §3.3.1's
independent-oracle rule: a reader validated against kitty's output inherits
kitty's bugs and the oracle becomes circular.  It is written against Anthropic's
published request schema, retrieved 2026-09-11.

**It reads both ends of the comparison.**  The oracle calls it on the inbound
body Claude Code sent the bridge *and* on the captured upstream body wherever
the adapter speaks Anthropic Messages — whether that is the passthrough shape
(``custom_anthropic``, ``zai_anthropic``) or the translated shape
``AnthropicAdapter.translate_to_upstream`` rebuilds.  One reader, two dialects,
which is why ``effort`` is recognised here: Claude Code sends it, so
residualising it would fail the run on every real request (§7.4.1's evidence
rule).

**Totality is the load-bearing property.**  Every key of the body, at every
depth, is either mapped and named in :attr:`~harness.contract.Request.consumed`,
or placed in :attr:`~harness.contract.Request.residual` under its path from the
body root.  Nothing is dropped silently — an unaccounted field is precisely
where an unregistered mutation hides.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from harness import contract as c

# --------------------------------------------------------------------------
# The recognised key sets
# --------------------------------------------------------------------------

#: Published top-level control fields that map to ``envelope.extra`` under their
#: own wire key.  Sourced from the Messages API reference's "Body parameters"
#: (retrieved 2026-09-11), except the last two, which the reference table omits
#: because it omits beta fields: ``context_management`` is documented on the
#: context-editing page (beta header ``context-management-2025-06-27``) and
#: ``mcp_servers`` on the MCP connector page.
_PUBLISHED_EXTRA_KEYS = frozenset(
    {
        "thinking",
        "metadata",
        "service_tier",
        "container",
        "cache_control",
        "inference_geo",
        "output_config",
        "context_management",
        "mcp_servers",
    }
)

#: Control fields the **client** sends that the published reference does not
#: list.  §7.4.1's first evidence rule, and ``effort`` is its worked example:
#: ``MessagesTranslator`` reads it off the inbound body, so Claude Code sends
#: it, so residualising it would fail the run on every real request.  This set
#: grows only on that kind of evidence — a key kitty alone emits, that no
#: register row names, is an unregistered mutation and must residualise.
_CLIENT_SENT_EXTRA_KEYS = frozenset({"effort"})

# §3.3.1a forbids a dotted key inside `extra`: the value under a wire key is
# compared whole, so `envelope.extra[thinking.budget_tokens]` is not a path this
# vocabulary defines. Pinned once here rather than checked per request, because
# both sets above are literals.
assert all("." not in key for key in _PUBLISHED_EXTRA_KEYS | _CLIENT_SENT_EXTRA_KEYS)

#: Sampling parameters, mapped onto the canonical Chat Completions spelling
#: (§3.3.1b).  ``stop_sequences`` is the one rename: Chat Completions calls it
#: ``stop``, and :data:`~harness.contract.SAMPLING_KEYS` is closed around that
#: spelling, so carrying the Anthropic name would put a vendor spelling into a
#: wire-independent form — and ``Conversation`` would reject it outright.
_SAMPLING_KEYS = {
    "max_tokens": "max_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "stop_sequences": "stop",
}

#: Tool-declaration keys the grammar carries.  Anything else on a tool entry
#: residualises — ``type`` on a server tool and ``cache_control`` are the two
#: that occur (KBR-167).
_TOOL_KEYS = frozenset({"name", "description", "input_schema"})

#: ``tool_choice.type`` values that map straight onto the canonical value.
#: ``tool`` is handled separately because it carries a name.
_SIMPLE_TOOL_CHOICES = frozenset({"auto", "any", "none"})


class AnthropicMessagesProjection:
    """Reads an Anthropic Messages request into :class:`~harness.contract.Request`.

    Implements :class:`~harness.contract.Projection` for
    :attr:`~harness.contract.WireFormat.ANTHROPIC_MESSAGES`.

    Attributes:
        wire_format: Always :attr:`~harness.contract.WireFormat.ANTHROPIC_MESSAGES`.
    """

    wire_format = c.WireFormat.ANTHROPIC_MESSAGES

    def read_request(self, captured: c.CapturedRequest) -> c.Request:
        """Project a captured Messages request.

        Args:
            captured: The request as observed on the wire. Only the body is
                read: unlike Gemini, the Messages format carries neither the
                model nor the operation in the URL.

        Returns:
            The wire-independent projection, total over the body.

        Raises:
            UnreadableBodyError: When the body is not a readable Messages
                request — malformed JSON, a role no format defines, a content
                block with no type, and the rest of R7's list. The catch-all
                below is the actual guarantee: the contract names three failure
                shapes, and an escaping ``KeyError`` would be an undefined
                fourth on the one path T-D1 uses to tell an unreadable body from
                an I1 breach. ``IndexError`` is deliberately **not** in the
                tuple: the only positional index in this module is
                ``merged[-1]``, guarded by ``if merged``, so catching it would
                be handling an impossible case.
        """
        body = _parse_body(captured.body)

        # Structural access is guarded case by case where the diagnosis is worth
        # naming, and blanket-guarded here for the shapes nobody anticipated.
        # `ValueError` is deliberately NOT caught: from inside a reader it means
        # the reader mis-routed a field, which is a reader bug and must surface.
        try:
            return _project(body)
        except (KeyError, TypeError, AttributeError) as exc:
            raise c.UnreadableBodyError(f"unreadable Anthropic Messages body: {exc!r}") from exc


def _parse_body(raw: bytes) -> Mapping[str, Any]:
    """Decode the request body into a JSON object.

    Args:
        raw: The raw body bytes.

    Returns:
        The parsed body.

    Raises:
        UnreadableBodyError: When the bytes are not JSON, or are JSON that is
            not an object. A JSON array is valid JSON and an invalid request.
    """
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise c.UnreadableBodyError(f"body is not valid JSON: {exc}") from exc

    if not isinstance(body, dict):
        raise c.UnreadableBodyError(f"body must be a JSON object, got {type(body).__name__}")

    return body


def _project(body: Mapping[str, Any]) -> c.Request:
    """Classify every top-level key into the envelope, the conversation or the residual.

    Args:
        body: The parsed request body.

    Returns:
        The projection.

    Raises:
        UnreadableBodyError: Propagated from the per-field readers.
    """
    residual: dict[str, Any] = {}
    consumed: set[str] = set()
    extra: dict[str, Any] = {}
    sampling: dict[str, Any] = {}

    # The envelope's two named fields, then every other control field the format
    # defines, keyed by the wire key so a register row can name it (§3.3.1a).
    for key, value in body.items():
        if key in ("model", "stream", "messages", "system", "tools"):
            consumed.add(key)
        elif key in _SAMPLING_KEYS:
            sampling[_SAMPLING_KEYS[key]] = value
            consumed.add(key)
        elif key == "tool_choice":
            extra[c.TOOL_CHOICE_KEY] = _read_tool_choice(value, residual)
            consumed.add(key)
        elif key in _PUBLISHED_EXTRA_KEYS or key in _CLIENT_SENT_EXTRA_KEYS:
            # Keyed by the wire key, never nested (§3.3.1a). The two key sets are
            # dot-free literals, so the guard belongs where they are declared
            # rather than on every request.
            extra[key] = value
            consumed.add(key)
        else:
            residual[key] = value

    conversation = c.Conversation(
        system=_read_system(body.get("system"), residual),
        turns=_read_turns(body.get("messages"), residual),
        tools=_read_tools(body.get("tools"), residual),
        sampling=sampling,
    )

    envelope = c.Envelope(
        model=_typed_leaf(body, "model", str, "", residual),
        stream=_typed_leaf(body, "stream", bool, "", residual),
        store=None,  # The Messages format defines no `store`; P17 is Responses-only.
        extra=extra,
    )

    return c.Request(
        envelope=envelope,
        conversation=conversation,
        residual=residual,
        consumed=frozenset(consumed),
        source=body,
    )


def _read_tool_choice(value: Any, residual: dict[str, Any]) -> str:
    """Normalise a ``tool_choice`` onto its canonical value.

    Four wire keys across the formats name one concept, so §3.3.1b makes the
    *value* canonical too — ``auto``, ``any``, ``none`` or ``tool:<name>``.

    Args:
        value: The wire value.
        residual: The residual mapping, extended with any key the canonical
            value does not carry.

    Returns:
        The canonical value.

    Raises:
        UnreadableBodyError: When the shape is not one the format publishes.
            Classifying here rather than letting :class:`~harness.contract.Envelope`
            judge is deliberate: it accepts any string starting ``tool:``, so a
            missing name would become the canonical-looking ``"tool:None"``
            that no register row can interpret, and an unrecognised type would
            escape as a bare ``ValueError`` — which the contract defines as a
            reader bug, a different diagnosis from "this body is unreadable".
    """
    if not isinstance(value, dict):
        raise c.UnreadableBodyError(f"tool_choice must be an object, got {type(value).__name__}")

    kind = value.get("type")
    if kind in _SIMPLE_TOOL_CHOICES:
        canonical = str(kind)
    elif kind == "tool":
        name = value.get("name")
        if not isinstance(name, str):
            raise c.UnreadableBodyError("tool_choice type 'tool' requires a name")
        canonical = f"tool:{name}"
    else:
        raise c.UnreadableBodyError(f"unrecognised tool_choice type {kind!r}")

    # `name` is accounted for only on the branch that read it: a stale `name`
    # beside `type: "auto"` is exactly the mutation the oracle should report, and
    # excluding it unconditionally would drop it silently. `disable_parallel_tool_use`
    # is a separate knob, not part of the concept four formats share, so folding
    # it into the value would make that value match nothing.
    _residualise(value, {"type", "name"} if kind == "tool" else {"type"}, "tool_choice", residual)

    return canonical


def _read_system(value: Any, residual: dict[str, Any]) -> tuple[c.Text, ...]:
    """Lift system instructions into ordered text parts.

    Args:
        value: The ``system`` field, absent, a string, or a list of text blocks.
        residual: The residual mapping, extended with any block key the grammar
            cannot carry.

    Returns:
        The system text, in order.

    Raises:
        UnreadableBodyError: When ``system`` is neither a string nor a list, a
            member is not an object, or a block's type is not ``text`` —
            :attr:`~harness.contract.Conversation.system` admits only
            :class:`~harness.contract.Text`, so there is nowhere else to put it.
    """
    if value is None:
        return ()

    if isinstance(value, str):
        return (c.Text(value),)

    if not isinstance(value, list):
        raise c.UnreadableBodyError(f"system must be a string or a list, got {type(value).__name__}")

    parts: list[c.Text] = []
    for index, block in enumerate(value):
        if not isinstance(block, dict):
            raise c.UnreadableBodyError(f"system[{index}] must be an object, got {type(block).__name__}")
        if block.get("type") != "text":
            raise c.UnreadableBodyError(f"system[{index}] must be a text block, got {block.get('type')!r}")

        text = block["text"]
        if not isinstance(text, str):
            raise c.UnreadableBodyError(f"system[{index}] text must be a string, got {type(text).__name__}")

        parts.append(c.Text(text))
        _residualise(block, {"type", "text"}, f"system[{index}]", residual)

    return tuple(parts)


def _read_tools(value: Any, residual: dict[str, Any]) -> tuple[c.ToolDecl, ...]:
    """Read the declared tools.

    Args:
        value: The ``tools`` field, absent or a list of declarations.
        residual: The residual mapping, extended with any entry key the grammar
            cannot carry — ``type`` on a server tool and ``cache_control``.

    Returns:
        The declarations, in order.

    Raises:
        UnreadableBodyError: When ``tools`` is not a list, or an entry is not an
            object or carries no name.
    """
    if value is None:
        return ()

    if not isinstance(value, list):
        raise c.UnreadableBodyError(f"tools must be a list, got {type(value).__name__}")

    declared: list[c.ToolDecl] = []
    for index, tool in enumerate(value):
        if not isinstance(tool, dict):
            raise c.UnreadableBodyError(f"tools[{index}] must be an object, got {type(tool).__name__}")
        if not isinstance(tool.get("name"), str):
            raise c.UnreadableBodyError(f"tools[{index}] must carry a name")

        # A wrongly-typed leaf residualises rather than coercing or raising.
        # `_freeze_mapping` calls `dict()` on whatever it is handed, which turns
        # a list of two-character strings into a *fabricated* schema and a plain
        # string into a bare `ValueError`. Residualising instead keeps the rest
        # of the request diffable while still failing the run, which raising
        # would not: the oracle could not report anything else about the body.
        schema = tool.get("input_schema")
        if schema is not None and not isinstance(schema, dict):
            residual[f"tools[{index}].input_schema"] = schema
            schema = None

        declared.append(
            c.ToolDecl(
                name=tool["name"],
                description=_typed_leaf(tool, "description", str, f"tools[{index}]", residual),
                schema=schema,
                # Absent, not False: the Messages format defines no `strict`,
                # and P15's presence and absence must stay distinguishable.
                strict=None,
            )
        )
        # Indexed, not by name (§7.4.1): a residual key is the body's own path,
        # and §3.3.1a's by-name tool addressing is a delta-path convention whose
        # reason — translators reorder declarations — is about comparison.
        _residualise(tool, _TOOL_KEYS, f"tools[{index}]", residual)

    return tuple(declared)


def _read_turns(value: Any, residual: dict[str, Any]) -> tuple[c.Turn, ...]:
    """Read the conversation turns.

    Args:
        value: The ``messages`` field, absent or a list of messages.
        residual: The residual mapping, extended with anything the grammar
            cannot carry.

    Returns:
        The turns, normalised per §3.3.1b.

    Raises:
        UnreadableBodyError: When ``messages`` is not a list, a message is not
            an object, lacks ``role`` or ``content``, or carries a role outside
            ``user``/``assistant``.
    """
    if value is None:
        # Absent is not an error: the oracle catches a vanished conversation as
        # a `conversation.turns` delta, which is a better diagnosis than an
        # unreadable-body error — and a reader that rejected an incomplete body
        # could not project the very mutation M5, M6 and M7 produce.
        return ()

    if not isinstance(value, list):
        raise c.UnreadableBodyError(f"messages must be a list, got {type(value).__name__}")

    turns: list[c.Turn] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise c.UnreadableBodyError(f"messages[{index}] must be an object, got {type(message).__name__}")
        if "role" not in message:
            raise c.UnreadableBodyError(f"messages[{index}] lacks a role")
        if "content" not in message:
            raise c.UnreadableBodyError(f"messages[{index}] lacks content")

        role = message["role"]
        if role not in c.ROLES:
            raise c.UnreadableBodyError(
                f"messages[{index}] role must be one of {sorted(c.ROLES)}, got {role!r}; "
                "the Messages format has no system role — a system prompt is the top-level field"
            )

        turns.append(c.Turn(role=role, parts=_read_content(message["content"], index, residual)))
        _residualise(message, {"role", "content"}, f"messages[{index}]", residual)

    return _normalise_turns(turns)


def _read_content(value: Any, message_index: int, residual: dict[str, Any]) -> tuple[c.Part, ...]:
    """Read one message's content into ordered parts.

    Args:
        value: The ``content`` field: a string, or a list of blocks.
        message_index: The message's position, for residual keys.
        residual: The residual mapping.

    Returns:
        The parts, in order.

    Raises:
        UnreadableBodyError: When content is neither a string nor a list, or a
            block is not an object or carries no type.
    """
    if isinstance(value, str):
        # A string is the format's own shorthand for one text block.
        return (c.Text(value),)

    if not isinstance(value, list):
        raise c.UnreadableBodyError(
            f"messages[{message_index}].content must be a string or a list, got {type(value).__name__}"
        )

    parts: list[c.Part] = []
    for index, block in enumerate(value):
        path = f"messages[{message_index}].content[{index}]"
        parts.append(_read_block(block, path, residual))

    return tuple(parts)


def _read_block(block: Any, path: str, residual: dict[str, Any]) -> c.Part:
    """Read one content block into a part.

    Args:
        block: The block.
        path: The block's path from the body root, for residual keys.
        residual: The residual mapping.

    Returns:
        The part.

    Raises:
        UnreadableBodyError: When the block is not an object or carries no type.
    """
    if not isinstance(block, dict):
        raise c.UnreadableBodyError(f"{path} must be an object, got {type(block).__name__}")
    if "type" not in block:
        raise c.UnreadableBodyError(f"{path} carries no type")

    kind = block["type"]
    if not isinstance(kind, str):
        # `Opaque.kind` is declared `str` and §7.4.1 makes it the wire type in
        # snake_case; an object here would become a non-string kind that no
        # canonical name could ever match.
        raise c.UnreadableBodyError(f"{path} type must be a string, got {type(kind).__name__}")

    if kind == "text":
        text = block["text"]
        if not isinstance(text, str):
            raise c.UnreadableBodyError(f"{path} text must be a string, got {type(text).__name__}")
        _residualise(block, {"type", "text"}, path, residual)
        return c.Text(text)

    if kind == "thinking":
        # `signature` is mapped, never residualised: M8's carrier repair
        # manipulates exactly this field, and its register row claims
        # `conversation.turns[*].parts[*]`. Residualising it would fail the run
        # on every real thinking block and leave M8 unclaimable.
        thinking = block["thinking"]
        if not isinstance(thinking, str):
            raise c.UnreadableBodyError(f"{path} thinking must be a string, got {type(thinking).__name__}")

        _residualise(block, {"type", "thinking", "signature"}, path, residual)
        return c.Thinking(text=thinking, signature=_typed_leaf(block, "signature", str, path, residual))

    if kind == "image":
        return _read_image(block, path, residual)

    if kind == "tool_use":
        if not isinstance(block.get("name"), str):
            raise c.UnreadableBodyError(f"{path} tool_use carries no name")

        # As for `input_schema`: Chat Completions encodes arguments as a JSON
        # *string*, so an upstream body that failed to parse one back lands here
        # and `dict()` would fabricate arguments from it. Residualised, not
        # raised, for the reason given there.
        arguments = block.get("input")
        if arguments is not None and not isinstance(arguments, dict):
            residual[f"{path}.input"] = arguments
            arguments = None

        _residualise(block, {"type", "name", "input", "id"}, path, residual)
        return c.ToolUse(
            name=block["name"],
            arguments=arguments or {},
            id=_typed_leaf(block, "id", str, path, residual),
        )

    if kind == "tool_result":
        # A wrongly-typed `is_error` residualises rather than being coerced, for
        # the reason §7.4.1 gives: `bool("false")` is `True`, so the coercion
        # invents the opposite of what the body said. `False` is the absent
        # value the grammar already carries, so there is one to fall back to.
        is_error = block.get("is_error", False)
        if not isinstance(is_error, bool):
            residual[f"{path}.is_error"] = is_error
            is_error = False

        _residualise(block, {"type", "tool_use_id", "content", "is_error"}, path, residual)
        return c.ToolResult(
            content=_read_result_content(block.get("content"), path, residual),
            # Absent where a format carries none; pairing is then by name and
            # position. Anthropic always carries one, but an orphan result — a
            # `tool_use_id` matching no call — still projects: M7 exists to drop
            # orphans, so a reader that raised on one would fail instead of
            # producing the delta that names it.
            tool_use_id=_typed_leaf(block, "tool_use_id", str, path, residual),
            is_error=is_error,
        )

    return _read_opaque(block, kind, path, residual)


def _read_image(block: Mapping[str, Any], path: str, residual: dict[str, Any]) -> c.Image:
    """Read an image block, identifying it by digest rather than carrying bytes.

    Args:
        block: The image block.
        path: The block's path from the body root.
        residual: The residual mapping.

    Returns:
        The image part.

    Raises:
        UnreadableBodyError: When the block carries no source, the source is not
            an object, its type is none of ``base64``/``url``/``file``, or its
            base64 payload does not decode.
    """
    source = block.get("source")
    if not isinstance(source, dict):
        raise c.UnreadableBodyError(f"{path} image carries no source object")

    kind = source.get("type")
    _residualise(block, {"type", "source"}, path, residual)

    if kind == "base64":
        try:
            decoded = base64.b64decode(source["data"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise c.UnreadableBodyError(f"{path} image data is not valid base64: {exc}") from exc

        _residualise(source, {"type", "data", "media_type"}, f"{path}.source", residual)
        # The media type is excluded from the digest and carried separately, so
        # a changed media type is its own delta rather than an unexplained
        # digest change.
        return c.Image(
            digest=c.image_digest(decoded),
            media_type=_typed_leaf(source, "media_type", str, f"{path}.source", residual),
        )

    if kind == "url":
        _residualise(source, {"type", "url"}, f"{path}.source", residual)
        return c.Image(ref=_typed_leaf(source, "url", str, f"{path}.source", residual))

    if kind == "file":
        _residualise(source, {"type", "file_id"}, f"{path}.source", residual)
        return c.Image(ref=_typed_leaf(source, "file_id", str, f"{path}.source", residual))

    raise c.UnreadableBodyError(f"{path} image source type {kind!r} is not one the format defines")


def _read_result_content(
    value: Any, path: str, residual: dict[str, Any]
) -> tuple[c.Text | c.Image | c.Json | c.Opaque, ...]:
    """Read a tool result's content.

    Args:
        value: The ``content`` field: absent, a string, or a list of blocks.
        path: The tool-result block's path from the body root.
        residual: The residual mapping.

    Returns:
        The result content, in order.

    Raises:
        UnreadableBodyError: When content is neither a string nor a list, or a
            member is not a readable block.
    """
    if value is None:
        return ()

    if isinstance(value, str):
        # A tool-result string is always `Text`, never `Json`, even when it
        # parses (§7.4.1). `Json` is for a format carrying a structured value
        # natively; without the rule fixed once, a JSON-shaped tool result shows
        # an unclaimed delta on every Messages-to-Chat-Completions comparison.
        return (c.Text(value),)

    if not isinstance(value, list):
        raise c.UnreadableBodyError(
            f"{path} tool_result content must be a string or a list, got {type(value).__name__}"
        )

    parts: list[c.Text | c.Image | c.Json | c.Opaque] = []
    for index, block in enumerate(value):
        member_path = f"{path}.content[{index}]"
        if not isinstance(block, dict):
            raise c.UnreadableBodyError(f"{member_path} must be an object, got {type(block).__name__}")
        if "type" not in block:
            raise c.UnreadableBodyError(f"{member_path} carries no type")

        # Repeated here, not delegated: deciding on the wire type below creates a
        # second `_read_opaque` call site that does not pass through
        # `_read_block`, so without this a non-string type reaches `Opaque.kind`,
        # which is declared `str` and which §7.4.1 makes the wire type in
        # snake_case.
        kind = block["type"]
        if not isinstance(kind, str):
            raise c.UnreadableBodyError(f"{member_path} type must be a string, got {type(kind).__name__}")

        # Decided on the wire type *before* reading, not by reading and then
        # re-reading: nothing nests a tool call inside a tool result, so a block
        # outside the narrower union is content the grammar cannot place — and
        # reading it first would residualise its keys and then digest them too,
        # accounting for one key twice.
        if kind in ("text", "image"):
            part = _read_block(block, member_path, residual)
            assert isinstance(part, c.RESULT_PART_TYPES)  # noqa: S101 - narrowing for mypy
            parts.append(part)
        else:
            parts.append(_read_opaque(block, kind, member_path, residual))

    return tuple(parts)


def _read_opaque(block: Mapping[str, Any], kind: str, path: str, residual: dict[str, Any]) -> c.Opaque:
    """Read a block the grammar does not model, keeping it detectable.

    Covers ``document``, ``search_result``, ``redacted_thinking``,
    ``server_tool_use`` and the server-tool result blocks — Anthropic's own
    spellings are already the canonical snake_case, so no alias table is needed
    for this format.

    Args:
        block: The block.
        kind: The block's wire type, which becomes :attr:`~harness.contract.Opaque.kind`.
        path: The block's path from the body root.
        residual: The residual mapping, extended with the block's
            ``cache_control`` only.

    Returns:
        The opaque part, carrying a digest of its payload.
    """
    # `cache_control` residualises exactly as it does on a modelled block, so
    # one field does not behave two ways — inside the digest it would produce a
    # delta with no named cause (KBR-167).
    _residualise(block, set(block) - {"cache_control"}, path, residual)
    return c.Opaque(kind=kind, digest=_payload_digest(block))


def _payload_digest(block: Mapping[str, Any]) -> str:
    """Return the digest of an unmodelled block's payload.

    Over **canonical** JSON rather than the raw wire slice, so a translator that
    reorders keys does not change the digest. ``ensure_ascii`` is pinned
    alongside ``sort_keys`` and ``separators`` because its default is ``True``
    while the surrounding prose says UTF-8: an author who passed ``False`` would
    get a different digest for the same block, visible only on non-ASCII
    content, which is the cross-reader disagreement §7.4.1 exists to prevent.

    Args:
        block: The block, whose ``type`` and ``cache_control`` are excluded —
            ``type`` because it is already :attr:`~harness.contract.Opaque.kind`,
            ``cache_control`` because it residualises instead.

    Returns:
        Lowercase hex SHA-256 of the canonical payload.
    """
    payload = {key: value for key, value in block.items() if key not in ("type", "cache_control")}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _typed_leaf(
    source: Mapping[str, Any],
    key: str,
    expected: type | tuple[type, ...],
    path: str,
    residual: dict[str, Any],
    default: Any = None,
) -> Any:
    """Return an optional leaf, residualising it when the wire carried the wrong type.

    §7.4.1's wrongly-typed-leaf rule, applied wherever the grammar has an absent
    value to fall back to. Without it these fields *fail open*: the contract
    validates only roles, sampling keys and ``tool_choice``, so a dict in a field
    declared ``str | None`` is carried silently and the residual stays empty.

    Args:
        source: The object being read.
        key: The leaf's key.
        expected: The type the format publishes for it.
        path: The object's path from the body root, or ``""`` for a top-level
            key, whose residual key is its bare name.
        residual: The residual mapping, extended in place.
        default: The grammar's absent value for this field.

    Returns:
        The leaf, or ``default`` when the wire value was the wrong type.
    """
    value = source.get(key, default)
    if value is not None and not isinstance(value, expected):
        residual[f"{path}.{key}" if path else key] = value
        return default
    return value


def _residualise(
    source: Mapping[str, Any], mapped: set[str], prefix: str, residual: dict[str, Any]
) -> None:
    """Record every key of ``source`` the reader did not map.

    §3.3.1's "unknown fields fail closed", applied at depth: ``verify_total``
    sees top-level keys only, so this is what closes the gap beneath them.

    Args:
        source: The object being read.
        mapped: The keys the caller accounted for.
        prefix: The object's path from the body root, to which each unmapped
            key is appended.
        residual: The residual mapping, extended in place.
    """
    for key, value in source.items():
        if key not in mapped:
            residual[f"{prefix}.{key}"] = value


def _normalise_turns(turns: Sequence[c.Turn]) -> tuple[c.Turn, ...]:
    """Merge consecutive same-role turns, preserving part order.

    §3.3.1b's merge rule is an **ordered pipeline**, not a set of independent
    clauses: a maximal run of tool results forms one turn, an immediately
    following non-tool user message merges into it, results come first *within
    the turn those clauses build*, and only then do consecutive same-role turns
    merge. In the Messages format a tool result already arrives inside a user
    turn, so the first three clauses are satisfied by the wire order and this
    function is the fourth — which is why it moves nothing.

    **A re-sort after the merge would be a defect, not a simplification.** It
    projects ``tool_result -> user(text) -> tool_result`` as
    ``[ToolResult, ToolResult, Text]``, hoisting a result ahead of text the
    agent sent *before* it — moving history the bridge did not move. Because
    paths are index-based, that invented delta lands on every part of the turn
    and every turn after it. An earlier draft of this reader did exactly that.

    **What the merge still hides**, recorded because §3.3.1 requires a
    projection's blind spots to be on the record: a mutation whose only effect
    is to split or join two consecutive same-role turns produces no delta, which
    weakens §3.3.2 assertion 2 for M5, M6 and M7. Accepted, because without the
    merge this reader and the Chat Completions reader disagree about turn
    boundaries on the standard ``assistant(tool_calls) -> tool -> tool``
    exchange, and that disagreement reports a false delta on every subsequent
    turn. It is also the format's own stated behaviour: "Consecutive ``user`` or
    ``assistant`` turns in your request will be combined into a single turn."

    Args:
        turns: The turns, one per message, in wire order.

    Returns:
        The normalised turns.
    """
    merged: list[c.Turn] = []
    for turn in turns:
        if merged and merged[-1].role == turn.role:
            previous = merged.pop()
            merged.append(c.Turn(role=turn.role, parts=tuple(previous.parts) + tuple(turn.parts)))
        else:
            merged.append(turn)

    return tuple(merged)
