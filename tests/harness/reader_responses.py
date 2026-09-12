"""The OpenAI Responses wire projection — one of the six independent readers.

`.system_design/TEST_SUITE.md` §3.3.1, §3.3.1a, §3.3.1b · plan task **T-A3** (KBR-35).

Projects an OpenAI Responses request body into :class:`harness.contract.Request`,
so the fidelity oracle can compare what an agent sent against what kitty actually
put on the wire.  This is the format the Codex path speaks, both as an inbound
shape and as the upstream shape of the ChatGPT-subscription provider.

**Written against the published schema, never against kitty's output.**  Every
key, item type and union member below was read from ``openai/openai-openapi``,
``openapi.yaml``, ``info.version`` :data:`SCHEMA_VERSION` — not from a kitty
request and not from memory.  §3.3.1's independent-oracle rule: a reader
validated against kitty's output inherits kitty's bugs, and the whole of I1 then
proves only that kitty agrees with itself.

**It imports nothing from ``src/kitty``, and must not.**
``test_reader_responses.py`` asserts the absence structurally.

**What totality means here, and where it stops.**  Every one of the 31 published
top-level keys is classified into the envelope, the conversation, or the
residual, and :func:`harness.contract.verify_total` fails the run on anything
left over.  But ``consumed`` covers *top-level* keys only (§3.3.1's stated
boundary), so a value dropped from *inside* a key this module claims is not
caught here.  Two such blind spots are deliberate and are named in the task's
requirements rather than left implied: the internals of the 27 opaque item types
(a mutated shell command inside a ``local_shell_call`` is invisible), and
``previous_response_id`` / ``conversation``, which move history server-side so a
formally total projection can still be missing turns.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from harness import contract as c

#: The published schema version every table in this module was derived from.
#: Recorded so a future divergence is traceable rather than mysterious: when
#: OpenAI adds a top-level key or an item type, the falsification tests go red
#: and this string says which revision the reader last agreed with.
SCHEMA_VERSION = "2.3.0"

# --------------------------------------------------------------------------
# Top-level key classification — `CreateResponse`, 31 keys
# --------------------------------------------------------------------------

#: Control fields the projection names in their own right.
_ENVELOPE_NAMED = frozenset({"model", "stream", "store"})

#: Sampling parameters whose Responses spelling is already canonical.
_SAMPLING_DIRECT = frozenset({"temperature", "top_p", "top_logprobs", "stream_options"})

#: The one sampling rename §3.3.1b mandates.  ``max_completion_tokens`` is
#: deliberately absent: P13 drops it in its own right, so collapsing both onto
#: one canonical key would make two register rows indistinguishable.
_SAMPLING_RENAMED: Mapping[str, str] = {"max_output_tokens": "max_tokens"}

#: Keys carrying semantic content rather than control.
_CONVERSATION_KEYS = frozenset({"input", "instructions", "tools"})

#: Every remaining published control field.  These project to
#: ``envelope.extra[<wire key>]`` per §3.3.1b, which is what makes them
#: addressable by a register row — sixteen of them are dropped by the Codex
#: allowlist and are claimed by **P23** (KBR-171), whose guard rebuilds its
#: sixteen paths out of this table, so an edit here must reach that row.
_EXTRA_KEYS = frozenset(
    {
        "background",
        "context_management",
        "conversation",
        "include",
        "max_tool_calls",
        "metadata",
        "moderation",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "text",
        "tool_choice",
        "truncation",
        "user",
    }
)

#: All 31, for the totality check and for the test that asserts this module's
#: tables still agree with the published schema.
PUBLISHED_TOP_LEVEL_KEYS = (
    _ENVELOPE_NAMED | _SAMPLING_DIRECT | frozenset(_SAMPLING_RENAMED) | _CONVERSATION_KEYS | _EXTRA_KEYS
)

# --------------------------------------------------------------------------
# Item types — `InputItem`, walked transitively, 31 discriminator values
# --------------------------------------------------------------------------

#: The four types this reader models semantically.
_MODELLED_ITEMS = frozenset({"message", "function_call", "function_call_output", "reasoning"})

#: Opaque item types that are client-supplied, and therefore project into a
#: ``user`` turn.  The mechanical rule is "a type ending ``_output``"; the five
#: after it are named exceptions, each for a stated reason rather than by
#: falling through a default:
#:
#: - ``mcp_approval_response`` answers the model's request;
#: - ``compaction_trigger`` is client-initiated;
#: - ``additional_tools`` carries its own ``role``, whose only published value is
#:   ``developer`` — client-supplied, but :class:`harness.contract.Opaque` cannot
#:   live in ``conversation.system``, so it lands in a ``user`` turn;
#: - ``item_reference`` is client-supplied history addressing;
#: - ``configuration_update`` carries ``reasoning`` — the same control field
#:   ``CreateResponse`` carries at top level — so it is client-supplied
#:   configuration, not model output.
_OPAQUE_USER_ITEMS = frozenset(
    {
        "apply_patch_call_output",
        "computer_call_output",
        "custom_tool_call_output",
        "local_shell_call_output",
        "program_output",
        "shell_call_output",
        "tool_search_output",
        "mcp_approval_response",
        "compaction_trigger",
        "additional_tools",
        "item_reference",
        "configuration_update",
    }
)

#: Opaque item types the model produced, which project into an ``assistant`` turn.
_OPAQUE_ASSISTANT_ITEMS = frozenset(
    {
        "apply_patch_call",
        "code_interpreter_call",
        "compaction",
        "computer_call",
        "custom_tool_call",
        "file_search_call",
        "image_generation_call",
        "local_shell_call",
        "mcp_approval_request",
        "mcp_call",
        "mcp_list_tools",
        "program",
        "shell_call",
        "tool_search_call",
        "web_search_call",
    }
)

#: Every published ``type`` an ``input`` array member may carry.
#:
#: **Derived from ``InputItem``, not from ``Item``.**  ``Item.oneOf`` has 28
#: members but omits ``compaction_trigger``, ``program``, ``program_output`` and
#: ``item_reference``, which reach the wire only through ``InputItem``.  A reader
#: whose closed set came from ``Item`` would residualise all four and go red on
#: legal traffic — the outcome the "placeholder the rest" decision exists to
#: avoid.
PUBLISHED_ITEM_TYPES = _MODELLED_ITEMS | _OPAQUE_USER_ITEMS | _OPAQUE_ASSISTANT_ITEMS

#: The content types a ``function_call_output``'s array branch may carry.
#: Deliberately narrower than a message's set — the published union is
#: ``InputTextContentParam | InputImageContentParamAutoParam |
#: InputFileContentParam``, with no ``refusal`` and no ``output_text``.
_TOOL_OUTPUT_CONTENT_TYPES = frozenset({"input_text", "input_image", "input_file"})

#: Message roles that lift into ``conversation.system`` rather than becoming a
#: turn (§3.3.1b R8.2).
_SYSTEM_ROLES = frozenset({"system", "developer"})

#: Every role a published ``message`` item may carry.  Anything else is a body
#: this reader cannot read — :class:`harness.contract.UnreadableBodyError`'s own
#: documented case, "a role no format defines".
_MESSAGE_ROLES = _SYSTEM_ROLES | {"user", "assistant"}

# --------------------------------------------------------------------------
# Tool choice — `ToolChoiceParam`, nine published forms
# --------------------------------------------------------------------------

#: ``ToolChoiceOptions`` string values, mapped onto the canonical vocabulary.
#: ``required`` becomes ``any``; the other two already agree.
_TOOL_CHOICE_STRINGS: Mapping[str, str] = {"none": "none", "auto": "auto", "required": "any"}

#: Object forms whose ``name`` field selects one tool.
_TOOL_CHOICE_BY_NAME = frozenset({"function", "custom"})

#: Object forms that select a tool by their ``type`` alone — the eight
#: ``ToolChoiceTypes`` built-ins plus the three ``Specific*`` singletons.
_TOOL_CHOICE_BY_TYPE = frozenset(
    {
        "file_search",
        "web_search_preview",
        "computer",
        "computer_use_preview",
        "computer_use",
        "web_search_preview_2025_03_11",
        "image_generation",
        "code_interpreter",
        "programmatic_tool_calling",
        "apply_patch",
        "shell",
    }
)

#: A ``data:`` URL carrying base64 image bytes, with its media type.
_DATA_URL = re.compile(r"^data:([^;,]+);base64,(.*)$", re.DOTALL)

#: Any ``data:`` URL at all.  Matched separately so that a *non*-base64 data URL
#: — ``data:image/png,abc`` — residualises instead of falling through to
#: :attr:`harness.contract.Image.ref`.  Leaving a data URL in ``ref`` is exactly
#: the shape pinning :func:`harness.contract.image_digest` exists to prevent:
#: another reader decoding the same bytes would produce a digest, and the two
#: projections would differ on an unchanged image.
_ANY_DATA_URL = re.compile(r"^data:", re.IGNORECASE)


def _mcp_tool_name(server_label: str) -> str:
    """Return the projected name of an MCP tool declaration.

    Spelled once, because two callers must agree exactly: the ``tools`` reader
    names the declaration, and ``tool_choice`` names a selection of it. A
    selection naming a tool the declaration list holds under a different
    spelling would correspond to nothing, and §3.3.1a's by-name addressing has
    no index to fall back on.

    Args:
        server_label: The MCP server's label, which its schema requires.

    Returns:
        The projected tool name.
    """
    return f"mcp:{server_label}"


class ResponsesProjection:
    """Reads an OpenAI Responses request into the wire-independent form.

    Implements :class:`harness.contract.Projection` for
    :attr:`harness.contract.WireFormat.OPENAI_RESPONSES`.

    The reader is stateless; every method takes what it needs and returns what it
    produced, so one instance is safe to share across a whole corpus run.
    """

    wire_format = c.WireFormat.OPENAI_RESPONSES

    def read_request(self, captured: c.CapturedRequest) -> c.Request:
        """Project a captured Responses request.

        Args:
            captured: The request as observed on the wire. Only the body is
                consulted — unlike Gemini, Responses carries neither the model
                nor the operation in the URL.

        Returns:
            The wire-independent projection, with ``consumed`` naming every
            top-level key accounted for and ``residual`` everything else.

            **A top-level key stays in ``consumed`` even when a value beneath it
            residualises.**  ``consumed`` records that the reader *read* the key;
            the nested residual records what inside it could not be classified.
            Only a key the reader did not read at all is omitted.  The asymmetry
            matters and is easy to get wrong: for a *top-level* residual either
            choice works, because :func:`harness.contract.verify_total` counts a
            key claimed in both accounts as a residual — but for a *nested* one,
            dropping the parent from ``consumed`` makes ``verify_total`` report
            ``DroppedFieldsError: reader dropped ['input']``, the wrong
            diagnosis for a reader that read the body fine.

        Raises:
            UnreadableBodyError: When the body is not a JSON object, when
                ``input`` is neither a string nor an array, or when a ``message``
                item carries a role no published schema defines.
        """
        body = self._parse(captured.body)

        residual: dict[str, Any] = {}
        consumed: set[str] = set()

        envelope = self._read_envelope(body, consumed, residual)
        conversation = self._read_conversation(body, consumed, residual)

        # Anything the tables above did not recognise is unaccounted for, and a
        # non-empty residual fails the run. This is the fail-closed half of
        # §3.3.1: an unmapped field is exactly where an unregistered mutation
        # hides.
        for key, value in body.items():
            if key not in consumed and key not in residual:
                residual[key] = value

        return c.Request(
            envelope=envelope,
            conversation=conversation,
            residual=residual,
            consumed=frozenset(consumed),
            source=body,
        )

    # ----------------------------------------------------------------
    # Body
    # ----------------------------------------------------------------

    @staticmethod
    def _parse(raw: bytes) -> Mapping[str, Any]:
        """Decode the request body into a JSON object.

        Args:
            raw: The undecoded body bytes.

        Returns:
            The parsed body.

        Raises:
            UnreadableBodyError: When the bytes are not UTF-8, not JSON, or not
                a JSON *object*. Raised rather than returned because §3.3.1
                separates "this body was unreadable" from "this is an I1
                breach", and T-D1 cannot tell them apart from a bare exception.
        """
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise c.UnreadableBodyError(f"body is not JSON: {exc}") from exc

        if not isinstance(decoded, dict):
            raise c.UnreadableBodyError(f"body must be a JSON object, got {type(decoded).__name__}")

        return decoded

    # ----------------------------------------------------------------
    # Envelope
    # ----------------------------------------------------------------

    def _read_envelope(self, body: Mapping[str, Any], consumed: set[str], residual: dict[str, Any]) -> c.Envelope:
        """Map the control fields onto the envelope.

        Args:
            body: The parsed request body.
            consumed: Accumulator of top-level keys accounted for, mutated here.
            residual: Accumulator of unclassifiable values, mutated here.

        Returns:
            The populated envelope.
        """
        extra: dict[str, Any] = {}

        # `extra` is keyed by the wire key and compared whole (§3.3.1b), so no
        # entry is ever nested: a register row addresses `envelope.extra[text]`,
        # never `envelope.extra[text.format]`.
        for key in sorted(_EXTRA_KEYS):
            if key not in body:
                continue
            consumed.add(key)
            if key == "tool_choice":
                self._read_tool_choice(body[key], extra, residual)
            else:
                extra[key] = body[key]

        for key in sorted(_ENVELOPE_NAMED):
            if key in body:
                consumed.add(key)

        return c.Envelope(
            model=body.get("model"),
            stream=body.get("stream"),
            store=body.get("store"),
            extra=extra,
        )

    def _read_tool_choice(self, choice: Any, extra: dict[str, Any], residual: dict[str, Any]) -> None:
        """Normalise ``tool_choice`` onto the canonical vocabulary.

        Four wire keys across four formats name one concept, so §3.3.1b makes
        this the single deliberate exception to keying ``extra`` by the wire key
        and pins the *value* to ``auto`` · ``any`` · ``none`` · ``tool:<name>``.

        Args:
            choice: The raw ``tool_choice`` value.
            extra: The envelope's extra mapping, mutated here.
            residual: Accumulator of unclassifiable values, mutated here.
        """
        normalised = self._normalise_tool_choice(choice)

        # An unrecognised selector is residualised rather than guessed into
        # `any`: guessing would silently equate a shape nobody has looked at
        # with a real selection, which is the blindness the residual exists to
        # prevent.
        if normalised is None:
            residual["tool_choice"] = choice
            return

        extra[c.TOOL_CHOICE_KEY] = normalised

    @staticmethod
    def _normalise_tool_choice(choice: Any) -> str | None:
        """Return the canonical form of one published ``tool_choice`` value.

        Args:
            choice: The raw value, in any of the nine published shapes.

        Returns:
            The canonical string, or ``None`` when the shape is unrecognised.
        """
        if isinstance(choice, str):
            return _TOOL_CHOICE_STRINGS.get(choice)

        if not isinstance(choice, dict):
            return None

        # Every branch below tests `kind` for set membership, so it must be
        # hashable first: a JSON body may legally carry a list here, and
        # `["function"] in frozenset(...)` raises TypeError — an exception shape
        # R11 forbids the reader from letting out.
        kind = choice.get("type")
        if not isinstance(kind, str):
            return None

        # `allowed_tools` maps by its own `mode`, not to a flat `any`: collapsing
        # it would make a `mode: auto -> required` mutation invisible, in the very
        # field whose job is constraining what the model may call. Its `tools`
        # list has no home in the canonical vocabulary and is not modelled.
        if kind == "allowed_tools":
            mode = choice.get("mode")
            return _TOOL_CHOICE_STRINGS.get(mode) if isinstance(mode, str) else None

        # MCP is selected by server, optionally narrowed to one tool. Two servers
        # would otherwise both normalise to a bare `tool:mcp`.
        #
        # The spelling deliberately embeds `_mcp_tool_name`'s output, so that the
        # selected value contains the *declaration's* name as a prefix. §3.3.1a
        # addresses tools by name, and a `tool_choice` naming a tool that
        # `conversation.tools` holds under a different spelling would correspond
        # to nothing.
        if kind == "mcp":
            label = choice.get("server_label")
            if not isinstance(label, str):
                return None
            name = choice.get("name")
            declared = _mcp_tool_name(label)
            return f"tool:{declared}:{name}" if isinstance(name, str) else f"tool:{declared}"

        if kind in _TOOL_CHOICE_BY_NAME:
            name = choice.get("name")
            return f"tool:{name}" if isinstance(name, str) else None

        if kind in _TOOL_CHOICE_BY_TYPE:
            return f"tool:{kind}"

        return None

    # ----------------------------------------------------------------
    # Conversation
    # ----------------------------------------------------------------

    def _read_conversation(
        self, body: Mapping[str, Any], consumed: set[str], residual: dict[str, Any]
    ) -> c.Conversation:
        """Map the semantic content onto the conversation.

        Args:
            body: The parsed request body.
            consumed: Accumulator of top-level keys accounted for, mutated here.
            residual: Accumulator of unclassifiable values, mutated here.

        Returns:
            The populated conversation.

        Raises:
            UnreadableBodyError: When ``input`` is neither a string nor an array.
        """
        system: list[c.Text] = []
        turns: list[c.Turn] = []

        # `instructions` is one of the four carriers §3.3.1b lifts into
        # `conversation.system`, and it comes before the input array.
        if "instructions" in body:
            consumed.add("instructions")
            instructions = body["instructions"]
            if isinstance(instructions, str):
                system.append(c.Text(instructions))
            elif instructions is not None:
                residual["instructions"] = instructions

        if "input" in body:
            consumed.add("input")
            self._read_input(body["input"], system, turns, residual)

        tools = self._read_tools(body, consumed, residual)

        return c.Conversation(
            system=system,
            turns=self._merge(turns),
            tools=tools,
            sampling=self._read_sampling(body, consumed),
        )

    @staticmethod
    def _read_sampling(body: Mapping[str, Any], consumed: set[str]) -> dict[str, Any]:
        """Collect the sampling parameters under their canonical names.

        Args:
            body: The parsed request body.
            consumed: Accumulator of top-level keys accounted for, mutated here.

        Returns:
            The sampling mapping, keyed by :data:`harness.contract.SAMPLING_KEYS`.
        """
        sampling: dict[str, Any] = {}

        for key in sorted(_SAMPLING_DIRECT):
            if key in body:
                consumed.add(key)
                sampling[key] = body[key]

        # `max_output_tokens` -> `max_tokens` is the one rename §3.3.1b mandates,
        # and register row P14 is written against exactly this mapping.
        for wire_key, canonical in _SAMPLING_RENAMED.items():
            if wire_key in body:
                consumed.add(wire_key)
                sampling[canonical] = body[wire_key]

        return sampling

    def _read_input(
        self,
        raw: Any,
        system: list[c.Text],
        turns: list[c.Turn],
        residual: dict[str, Any],
    ) -> None:
        """Read the ``input`` field in either of its two published shapes.

        Args:
            raw: The ``input`` value — a string or an array of items.
            system: Accumulator of lifted system text, mutated here.
            turns: Accumulator of turns, mutated here.
            residual: Accumulator of unclassifiable values, mutated here.

        Raises:
            UnreadableBodyError: When ``input`` is neither a string nor a list.
        """
        # `InputParam` is `oneOf[string, array]`; the bare string is "equivalent
        # to a text input with the user role" in the schema's own words.
        if isinstance(raw, str):
            turns.append(c.Turn("user", [c.Text(raw)]))
            return

        if not isinstance(raw, list):
            raise c.UnreadableBodyError(f"input must be a string or an array of items, got {type(raw).__name__}")

        for index, item in enumerate(raw):
            self._read_item(item, index, system, turns, residual)

    def _read_item(
        self,
        item: Any,
        index: int,
        system: list[c.Text],
        turns: list[c.Turn],
        residual: dict[str, Any],
    ) -> None:
        """Read one member of the ``input`` array.

        Args:
            item: The raw item.
            index: Its position, for residual paths.
            system: Accumulator of lifted system text, mutated here.
            turns: Accumulator of turns, mutated here.
            residual: Accumulator of unclassifiable values, mutated here.

        Raises:
            UnreadableBodyError: When a ``message`` item carries an undefined role.
        """
        path = f"input[{index}]"

        if not isinstance(item, dict):
            residual[path] = item
            return

        kind = self._item_type(item)

        if kind == "message":
            self._read_message(item, path, system, turns, residual)
        elif kind == "function_call":
            turns.append(c.Turn("assistant", [self._read_function_call(item, path, residual)]))
        elif kind == "function_call_output":
            turns.append(c.Turn("user", [self._read_function_output(item, path, residual)]))
        elif kind == "reasoning":
            turns.append(c.Turn("assistant", self._read_reasoning(item, path, residual)))
        elif kind in _OPAQUE_USER_ITEMS:
            turns.append(c.Turn("user", [c.Opaque(kind)]))
        elif kind in _OPAQUE_ASSISTANT_ITEMS:
            turns.append(c.Turn("assistant", [c.Opaque(kind)]))
        else:
            # An item type outside all 31 published values. §3.3.1: adding a
            # shape to a wire format must force a deliberate decision, so this
            # fails the run rather than being silently placeheld.
            residual[path] = item

    @staticmethod
    def _item_type(item: Mapping[str, Any]) -> str | None:
        """Return an item's discriminator, inferring it when the wire omits it.

        ``EasyInputMessage``, ``FunctionCallOutputItemParam`` and
        ``ItemReferenceParam`` all make ``type`` optional or nullable, so
        dispatch cannot key on it alone.

        Args:
            item: The raw item.

        Returns:
            The item's type, or ``None`` when it cannot be inferred.
        """
        declared = item.get("type")
        if isinstance(declared, str):
            return declared

        # A `role` makes it a message — the published `EasyInputMessage`, which
        # is the shape every hand-written example uses. An `id` alone is an item
        # reference, whose `type` is explicitly nullable.
        if "role" in item:
            return "message"
        if "id" in item:
            return "item_reference"

        return None

    # ----------------------------------------------------------------
    # Items
    # ----------------------------------------------------------------

    def _read_message(
        self,
        item: Mapping[str, Any],
        path: str,
        system: list[c.Text],
        turns: list[c.Turn],
        residual: dict[str, Any],
    ) -> None:
        """Read a ``message`` item into the system text or a turn.

        Args:
            item: The raw message item.
            path: Its residual path prefix.
            system: Accumulator of lifted system text, mutated here.
            turns: Accumulator of turns, mutated here.
            residual: Accumulator of unclassifiable values, mutated here.

        Raises:
            UnreadableBodyError: When the role is outside the published set.
        """
        role = item.get("role")

        # The isinstance check is not redundant with the membership test: a role
        # of `["user"]` is outside the set, but testing membership of an
        # unhashable value raises TypeError, which `contract` defines as a reader
        # bug rather than a bad body. R11 names this exact case.
        if not isinstance(role, str) or role not in _MESSAGE_ROLES:
            raise c.UnreadableBodyError(f"{path}: message role must be one of {sorted(_MESSAGE_ROLES)}, got {role!r}")

        raw_content = item.get("content")
        parts = self._read_content(raw_content, f"{path}.content", residual)

        # System and developer instructions lift into `conversation.system`
        # rather than becoming a turn (§3.3.1b R8.2).
        if role in _SYSTEM_ROLES:
            entries = raw_content if isinstance(raw_content, list) else []
            for wire_index, part in parts:
                # `Conversation.system` is enforced Text-only, so there is
                # nowhere to put an image a developer message may legally carry.
                # Residualising fails closed; letting the contract's TypeError
                # out would add a fourth failure shape beside the three §3.3.1
                # defines on purpose.
                if isinstance(part, c.Text):
                    system.append(part)
                else:
                    # Keyed by the WIRE index and holding the WIRE value. Using
                    # the projected offset would name the wrong element whenever
                    # an earlier part residualised, and would collide with the
                    # key `_read_content` already wrote — silently destroying one
                    # of two unclassified values.
                    residual[f"{path}.content[{wire_index}]"] = entries[wire_index]
            return

        turns.append(c.Turn(role, [part for _, part in parts]))

    def _read_content(
        self, raw: Any, path: str, residual: dict[str, Any], allowed: frozenset[str] | None = None
    ) -> list[tuple[int, c.Part]]:
        """Read a message's ``content`` in either published shape.

        Returns each part beside its **wire index**, not merely in order: the
        caller that lifts system text needs to residualise at the index the body
        actually used, and a projected offset drifts from it as soon as one
        entry residualises.

        Args:
            raw: A string, or a list of content parts.
            path: The residual path prefix for this content list.
            residual: Accumulator of unclassifiable values, mutated here.
            allowed: The content types permitted here, or ``None`` for a
                message's full set.

        Returns:
            ``(wire index, part)`` pairs, in wire order. A string ``content``
            yields one pair at index 0.
        """
        if isinstance(raw, str):
            return [(0, c.Text(raw))]

        if not isinstance(raw, list):
            residual[path] = raw
            return []

        parts: list[tuple[int, c.Part]] = []
        for offset, entry in enumerate(raw):
            part = self._read_content_part(entry, f"{path}[{offset}]", residual, allowed)
            if part is not None:
                parts.append((offset, part))

        return parts

    def _read_content_part(
        self, entry: Any, path: str, residual: dict[str, Any], allowed: frozenset[str] | None = None
    ) -> c.Part | None:
        """Read one content part of a message.

        Args:
            entry: The raw content part.
            path: Its residual path.
            residual: Accumulator of unclassifiable values, mutated here.
            allowed: The content types permitted in this position, or ``None``
                for a message's full set. A tool output's array branch publishes
                only three of them, and accepting the others there would let a
                shape the schema forbids pass unremarked.

        Returns:
            The projected part, or ``None`` when it was residualised.
        """
        if not isinstance(entry, dict):
            residual[path] = entry
            return None

        # Hashable before any set membership test: a JSON body may legally carry
        # a dict here, and `{"a": 1} in {"input_text", ...}` raises TypeError.
        kind = entry.get("type")
        if not isinstance(kind, str) or (allowed is not None and kind not in allowed):
            residual[path] = entry
            return None

        # `input_text` and `output_text` both project to a bare `Text`. That is
        # what makes register row P16 NOT_PROJECTABLE: the tag is redundant with
        # the turn's role, and carrying it would put one vendor's spelling into a
        # form whose whole purpose is wire independence.
        if kind in {"input_text", "output_text"}:
            text = entry.get("text")
            if isinstance(text, str):
                return c.Text(text)
            residual[path] = entry
            return None

        # `refusal` is NOT collapsed into Text. Unlike the input/output tag, it
        # is not recoverable from the role: an assistant refusal and an assistant
        # answer would project identically, so a bridge that turned one into the
        # other would be invisible.
        #
        # The digest carries the text's identity without the grammar having a
        # refusal type: `kind` alone would make a *rewritten* refusal invisible,
        # which is the same blindness one step down.
        if kind == "refusal":
            text = entry.get("refusal")
            if not isinstance(text, str):
                residual[path] = entry
                return None
            return c.Opaque("refusal", digest=c.image_digest(text.encode("utf-8")))

        if kind == "input_image":
            return self._read_image(entry, path, residual)

        if kind == "input_file":
            return c.Opaque("file")

        residual[path] = entry
        return None

    @staticmethod
    def _read_image(entry: Mapping[str, Any], path: str, residual: dict[str, Any]) -> c.Part | None:
        """Project an ``input_image`` content part.

        ``contract.image_digest`` is pinned so six independently written readers
        agree on one digest for one image; a reader that left a data URL in
        ``ref`` while another decoded it would reproduce exactly the disagreement
        that pinning exists to prevent.

        Args:
            entry: The raw ``input_image`` part.
            path: Its residual path.
            residual: Accumulator of unclassifiable values, mutated here.

        Returns:
            The projected image, or ``None`` when it was residualised.
        """
        url = entry.get("image_url")

        if isinstance(url, str):
            found = _DATA_URL.match(url)
            if found:
                media_type, payload = found.group(1), found.group(2)
                try:
                    raw = base64.b64decode(payload, validate=True)
                except (binascii.Error, ValueError):
                    # Undecodable bytes are not an image this reader can digest,
                    # and inventing a digest would make two unequal images
                    # compare equal.
                    residual[path] = entry
                    return None
                return c.Image(digest=c.image_digest(raw), media_type=media_type)

            # A data URL that is not base64 carries bytes this reader cannot
            # canonicalise; putting it in `ref` would make it compare unequal to
            # another reader's digest of the same image.
            if _ANY_DATA_URL.match(url):
                residual[path] = entry
                return None

            # A remote image has no bytes to digest; the URI is the identity, the
            # same shape §3.3.1 gives Gemini's `fileData.fileUri`.
            return c.Image(ref=url)

        file_id = entry.get("file_id")
        if isinstance(file_id, str):
            return c.Image(ref=file_id)

        residual[path] = entry
        return None

    def _read_function_call(self, item: Mapping[str, Any], path: str, residual: dict[str, Any]) -> c.ToolUse:
        """Project a ``function_call`` item.

        Args:
            item: The raw item.
            path: Its residual path prefix.
            residual: Accumulator of unclassifiable values, mutated here.

        Returns:
            The projected tool call.
        """
        # A wrongly-typed id or name is residualised rather than coerced: `str(7)`
        # and `str(None)` invent a value the agent never sent, and `verify_total`
        # cannot see a nested coercion because `consumed` is top-level only.
        call_id = item.get("call_id")
        if call_id is not None and not isinstance(call_id, str):
            residual[f"{path}.call_id"] = call_id
            call_id = None

        # `name` is required by `FunctionToolCall` and, unlike the id, there is
        # no format that omits it — a call nobody can name cannot be paired with
        # its result or addressed by a register row. So `null` and absent are
        # residualised too, not just a wrong type.
        name = item.get("name")
        if not isinstance(name, str):
            residual[f"{path}.name"] = name
            name = ""

        return c.ToolUse(
            name=name,
            arguments=self._read_arguments(item.get("arguments"), f"{path}.arguments", residual),
            id=call_id,
        )

    @staticmethod
    def _read_arguments(raw: Any, path: str, residual: dict[str, Any]) -> Mapping[str, Any]:
        """Decode a tool call's JSON-string arguments.

        Chat Completions and Responses both encode arguments as a *string*, while
        Messages sends an object; normalising here stops a spurious delta on
        every cross-format comparison.

        Only an absent or blank value is silently empty — an absent ``arguments``
        honestly means *no arguments*, which is why it does **not** residualise
        the way an absent tool ``name`` does even though the schema requires
        both: a name is unrecoverable, an empty argument set is not. And kitty
        already emits the blank form: its Responses builder writes ``arguments`` as ``""`` whenever a
        Chat Completions tool call carried none, so the empty string is real
        corpus traffic rather than a hypothetical. Every other shape the schema
        forbids residualises, because mapping it to ``{}`` would claim the agent
        sent no arguments when it sent something — and would hide the breach
        where kitty *drops* them.

        Args:
            raw: The wire value.
            path: The residual path for this field.
            residual: Accumulator of unclassifiable values, mutated here.

        Returns:
            The decoded arguments, empty when the wire value carried none.
        """
        if raw is None:
            return {}

        # `FunctionToolCall.arguments` is a string in the published schema. A
        # decoded object is NOT accepted: it would make a bridge that emitted the
        # object form instead of the string invisible to the oracle, which is the
        # wire-format breach this reader exists to see.
        if not isinstance(raw, str):
            residual[path] = raw
            return {}

        if not raw.strip():
            return {}

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            # Never an exception: `contract` defines a reader-raised ValueError
            # as "the reader mis-routed a field", so failing closed into the
            # residual keeps the diagnosis honest.
            residual[path] = raw
            return {}

        if not isinstance(decoded, dict):
            residual[path] = raw
            return {}

        return decoded

    def _read_function_output(self, item: Mapping[str, Any], path: str, residual: dict[str, Any]) -> c.ToolResult:
        """Project a ``function_call_output`` item.

        Args:
            item: The raw item.
            path: Its residual path prefix.
            residual: Accumulator of unclassifiable values, mutated here.

        Returns:
            The projected tool result.
        """
        call_id = item.get("call_id")
        raw = item.get("output")

        content: list[c.Text | c.Image | c.Json | c.Opaque] = []

        if isinstance(raw, str):
            content.append(self._read_output_string(raw))
        elif isinstance(raw, list):
            # The array branch publishes only these three content types, so the
            # message set is narrowed here: accepting `refusal` or `output_text`
            # in a tool result would let a shape the schema forbids pass
            # unremarked. `_read_content_part` returns only members of
            # `RESULT_PART_TYPES`, so no further narrowing is needed after it.
            for _offset, part in self._read_content(
                raw, f"{path}.output", residual, allowed=_TOOL_OUTPUT_CONTENT_TYPES
            ):
                content.append(part)  # type: ignore[arg-type]
        elif raw is not None:
            residual[f"{path}.output"] = raw

        # Responses carries no error flag on a function output, so `is_error` is
        # always False here. Written down rather than left as silence: the field
        # exists on the contract because Messages has one.
        return c.ToolResult(
            content=content,
            tool_use_id=call_id if isinstance(call_id, str) else None,
            is_error=False,
        )

    @staticmethod
    def _read_output_string(raw: str) -> c.Text | c.Json:
        """Project a string tool output, preferring its structured form.

        Args:
            raw: The wire value — "a JSON string of the output" per the schema,
                though tools routinely return prose.

        Returns:
            A :class:`harness.contract.Json` when the string parses as JSON,
            otherwise a :class:`harness.contract.Text`.
        """
        try:
            return c.Json(json.loads(raw))
        except json.JSONDecodeError:
            return c.Text(raw)

    @staticmethod
    def _read_reasoning(item: Mapping[str, Any], path: str, residual: dict[str, Any]) -> list[c.Part]:
        """Project a ``reasoning`` item into thinking parts.

        One part per ``summary`` entry, then one per ``content`` entry, in wire
        order, so dropping one entry of three is a visible delta rather than
        changed text. ``encrypted_content`` rides on the first part as its
        signature — it is what M8's carrier repair manipulates.

        Args:
            item: The raw reasoning item.
            path: Its residual path prefix.
            residual: Accumulator of unclassifiable values, mutated here.

        Returns:
            The projected parts, never empty.
        """
        # `Thinking.text` is declared `str` but unvalidated, so a non-string
        # `text` cannot simply flow into the projection — and it cannot simply be
        # skipped either. Skipping performs exactly the loss this fan-out exists
        # to make visible: an item carrying two summary entries would project one
        # part with nothing to say the other vanished, and `consumed` is
        # top-level only so `verify_total` could never see it.
        texts: list[str] = []
        for field_name in ("summary", "content"):
            entries = item.get(field_name)
            if entries is None:
                continue

            if not isinstance(entries, list):
                residual[f"{path}.{field_name}"] = entries
                continue

            for offset, entry in enumerate(entries):
                text = entry.get("text") if isinstance(entry, dict) else None
                if isinstance(text, str):
                    texts.append(text)
                else:
                    residual[f"{path}.{field_name}[{offset}]"] = entry

        signature = item.get("encrypted_content")
        if not isinstance(signature, str):
            signature = None

        # An empty block is a part with an empty string, never nothing: without
        # one, a reasoning item carrying only an encrypted blob would vanish and
        # its presence would stop being observable.
        if not texts:
            return [c.Thinking("", signature=signature)]

        return [c.Thinking(text, signature=signature if offset == 0 else None) for offset, text in enumerate(texts)]

    # ----------------------------------------------------------------
    # Tools
    # ----------------------------------------------------------------

    def _read_tools(self, body: Mapping[str, Any], consumed: set[str], residual: dict[str, Any]) -> list[c.ToolDecl]:
        """Project the ``tools`` array.

        Args:
            body: The parsed request body.
            consumed: Accumulator of top-level keys accounted for, mutated here.
            residual: Accumulator of unclassifiable values, mutated here.

        Returns:
            The projected declarations, in wire order.
        """
        if "tools" not in body:
            return []

        consumed.add("tools")
        raw = body["tools"]

        if not isinstance(raw, list):
            residual["tools"] = raw
            return []

        tools: list[c.ToolDecl] = []
        for index, entry in enumerate(raw):
            # A declaration whose `type` is not a string is not a published tool
            # shape at all, so it residualises whole rather than being read as a
            # built-in named `str(kind)`.
            if not isinstance(entry, dict) or not isinstance(entry.get("type"), str):
                residual[f"tools[{index}]"] = entry
                continue
            tools.append(self._read_tool(entry, f"tools[{index}]", residual))

        return tools

    @staticmethod
    def _read_tool(entry: Mapping[str, Any], path: str, residual: dict[str, Any]) -> c.ToolDecl:
        """Project one tool declaration.

        Args:
            entry: The raw declaration.
            path: Its residual path prefix.
            residual: Accumulator of unclassifiable values, mutated here.

        Returns:
            The projected declaration.
        """
        kind = entry.get("type")

        if kind == "function":
            # A wrongly-typed description or schema residualises rather than
            # becoming `None`. Both fields carry register weight — P15 strips
            # `strict`, and §3.3.1's oracle falsification set deletes a
            # description — so a silent `None` here would look exactly like the
            # mutation those checks exist to catch.
            description = entry.get("description")
            if description is not None and not isinstance(description, str):
                residual[f"{path}.description"] = description
                description = None

            parameters = entry.get("parameters")
            if parameters is not None and not isinstance(parameters, dict):
                residual[f"{path}.parameters"] = parameters
                parameters = None

            # Same rule as `_read_function_call`, and for a stronger reason:
            # `FunctionTool.required` includes `name`, and §3.3.1a addresses
            # tools by name with no index to fall back on. Two unnamed
            # declarations would both sit at `conversation.tools[]` — which
            # `path_matches` accepts as the legacy wildcard spelling, so a
            # register row would match them by accident rather than by name.
            name = entry.get("name")
            if not isinstance(name, str):
                residual[f"{path}.name"] = name
                name = ""

            strict = entry.get("strict")
            return c.ToolDecl(
                name=name,
                description=description,
                schema=parameters,
                # `strict` is a tri-state, and the published schema spells "no
                # strict" as null rather than as omission. `None` must stay
                # distinct from `False` or register row P15's presence and
                # absence become indistinguishable.
                strict=strict if isinstance(strict, bool) else None,
            )

        # A built-in tool is still addressable by name, so its whole declaration
        # becomes the schema and nothing is lost. MCP is named by its server: two
        # servers named by bare `type` would both occupy `conversation.tools[mcp]`,
        # a collision §3.3.1a's by-name addressing cannot recover from.
        name = entry.get("name")
        if kind == "mcp":
            label = entry.get("server_label")
            name = _mcp_tool_name(label) if isinstance(label, str) else "mcp"
        elif not isinstance(name, str):
            name = str(kind)

        return c.ToolDecl(name=name, schema=dict(entry))

    # ----------------------------------------------------------------
    # Turns
    # ----------------------------------------------------------------

    @staticmethod
    def _merge(turns: Sequence[c.Turn]) -> list[c.Turn]:
        """Merge consecutive same-role turns, preserving wire order.

        §3.3.1b's merge rule has four clauses, and on this format three of them
        reduce to this one. A maximal run of consecutive tool results forms one
        ``user`` turn, and an immediately following user message merges into it —
        both are consecutive ``user`` emissions. "``ToolResult`` parts come
        first" is then satisfied **vacuously**, because every member of a run
        *is* a tool result: Responses delivers results contiguously in the input
        array, so there is nothing to reorder.

        Reordering was rejected deliberately. Hoisting results ahead of a
        *preceding* user message would move text the agent sent first to sit
        after the result, inventing a delta on every such turn — and because
        paths are index-based, a disagreement about turn boundaries reports a
        delta on every subsequent turn too.

        Args:
            turns: The turns in wire order, one per input item.

        Returns:
            The merged turns.
        """
        merged: list[c.Turn] = []

        for turn in turns:
            if merged and merged[-1].role == turn.role:
                previous = merged[-1]
                merged[-1] = c.Turn(previous.role, [*previous.parts, *turn.parts])
            else:
                merged.append(turn)

        return merged
