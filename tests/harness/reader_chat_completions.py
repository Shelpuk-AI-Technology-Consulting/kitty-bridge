"""The Chat Completions reader — `POST /v1/chat/completions` into the common form.

`.system_design/TEST_SUITE.md` §3.3.1, §3.3.1a, §3.3.1b, §7.4.1 · plan task
**T-A2** (KBR-34).

This module **imports nothing from** ``src/kitty``, and must not. §3.3.1's
independent-oracle rule: a reader validated against kitty's output inherits
kitty's bugs and the oracle becomes circular. It is written against OpenAI's
published schema — `openai/openai-openapi` master, retrieved 2026-09-14, the same
source the design doc cites for G37's cache-breakpoint facts.

**It reads both ends of the comparison.** The oracle calls it on the inbound body
Claude Code sent the bridge *and* on the captured upstream body wherever an
adapter speaks Chat Completions — the translated shape `AnthropicAdapter`,
`OllamaCloudAdapter` and friends rebuild. One reader, both sides, which is why
`system` and `developer` roles and the deprecated `function` role are all
accounted for: real Claude Code and real adapters can both send them.

**Totality is the load-bearing property.** Every key of the body, at every depth,
is either mapped and named in :attr:`~harness.contract.Request.consumed`, or
placed in :attr:`~harness.contract.Request.residual` under its path from the body
root. Nothing is dropped silently — an unaccounted field is precisely where an
unregistered mutation hides.

**§3.3.1b's merge rule, on this format, is satisfied vacuously for clause 3.**
Chat Completions delivers tool results contiguously in their own ``tool``
messages, so the run and following-message clauses do all the work; the
results-first clause has nothing to reorder. §7.4.1 records the distinction.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from harness import contract as c

# --------------------------------------------------------------------------
# The recognised key sets
# --------------------------------------------------------------------------

#: Published top-level control fields that map to ``envelope.extra`` under their
#: own wire key. Sourced from `openai/openai-openapi` master's
#: ``CreateChatCompletionRequest`` (retrieved 2026-09-14), which folds the
#: request properties of ``CreateModelResponseProperties`` into the CC surface.
#:
#: ``parallel_tool_calls`` is deliberately **not** in this set: §3.3.1b (KBR-205,
#: closing G36) fixes a canonical address for the parallel-tool-use knob,
#: and this reader is the first to *read* it directly rather than write it.
#:
#: ``audio`` carries the audio-output config (modalities=audio), ``moderation``
#: is the moderation config (o-series, safety tooling), ``prompt_cache_options``
#: is the request-wide cache TTL (`30m` only) — all §3.3.1b's "declared control
#: field of the format" rule says goes to ``extra[<wire key>]`` rather than
#: the residual. ``functions`` and ``function_call`` are the deprecated top-level
#: spellings the older ``tools``/``tool_calls`` replaced; they map onto
#: ``extra[<wire key>]`` the same way (a residual entry would force the
#: schema validator to accept the spelling, which the bridge does not do —
#: the row plan for the deprecated spellings is owed to M16's twin and
#: arrives with the next register pass).
_PUBLISHED_EXTRA_KEYS = frozenset(
    {
        "store",
        "metadata",
        "service_tier",
        "reasoning_effort",
        "verbosity",
        "modalities",
        "prediction",
        "user",
        "web_search_options",
        "prompt_cache_options",
        "audio",
        "moderation",
        "functions",
        "function_call",
    }
)

#: Sampling parameters, mapped onto the canonical Chat Completions spelling
#: (§3.3.1b). CC **is** the spelling the closed set was derived from for
#: thirteen of the fifteen keys, and the table is therefore the identity for
#: those thirteen. Two keys renames: ``stop_sequences`` (Anthropic) → ``stop``
#: is the spelling-only case; ``top_k`` is the one Chat Completions does
#: *not* carry — it is in SAMPLING_KEYS as the closed set, but excluded from
#: this reader because no CC body can carry it. P13 drops the closed set's
#: other fourteen; collapsing ``max_tokens`` and ``max_completion_tokens``
#: would hide a P13 delta.
_SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "top_logprobs",
        "response_format",
        "stop",
        "n",
        "stream_options",
        "seed",
        "logit_bias",
    }
)

#: Roles that lift into ``conversation.system`` rather than becoming a turn
#: (§3.3.1b). ``system`` is the older spelling; ``developer`` replaced it, and
#: both name one concept on the wire.
_SYSTEM_ROLES = frozenset({"system", "developer"})

#: Every role a published Chat Completions message may carry. Anything else is a
#: body this reader cannot read — :class:`~harness.contract.UnreadableBodyError`'s
#: own documented case. ``function`` (deprecated, replaced by ``tool``) is
#: **deliberately not** in the set: the bridge does not translate the legacy
#: spelling, so a body carrying it is one the product cannot faithfully forward,
#: and residualising it at its own path names the shape rather than inventing a
#: projection.
_MESSAGE_ROLES = _SYSTEM_ROLES | {"user", "assistant", "tool"}

#: Content-part types a ``user`` message may carry. ``input_audio`` and ``file``
#: project as :class:`~harness.contract.Opaque` per the alias rule — neither the
#: grammar nor any reader models their payload.
_USER_PART_TYPES = frozenset({"text", "image_url", "input_audio", "file"})

#: Content-part types an ``assistant`` message may carry. The schema allows
#: exactly ``text`` and ``refusal`` — a refusal is a bare string on the message
#: or one ``refusal`` part in the array, and both name one concept.
_ASSISTANT_PART_TYPES = frozenset({"text", "refusal"})

#: Content-part types a ``tool`` message may carry. The schema allows only
#: ``text`` — a tool result's richer shapes (image, file) belong to the
#: destination wire, not the CC request body.
_TOOL_PART_TYPES = frozenset({"text"})

#: The cache-breakpoint keys a content part or a tool declaration may carry.
#: Both spellings fill :attr:`~harness.contract.Text.cache_control` **verbatim**
#: (§11 Q16, closing G37): the slot is ``Mapping[str, Any] | None``, and §3.3.1's
#: "carried whole, not reduced" rule applies to a spelling with no TTL
#: (``prompt_cache_breakpoint``, request-wide TTL in ``prompt_cache_options``)
#: the same way it applies to one with. An **ordered** ``tuple``, not a
#: ``frozenset`` — when both spellings are present on one part the first
#: wins, the second residualises (R6.4); the rule is "first non-null wins",
#: and a hash-randomised iteration order would make that rule
#: hash-seed-dependent. The CC spelling (``cache_control``) is checked
#: first — OpenRouter's CC dialect carries Anthropic's own field.
_CACHE_KEYS: tuple[str, ...] = ("cache_control", "prompt_cache_breakpoint")

#: Tool-choice strings whose canonical values already agree — CC's ``required``
#: maps to ``any`` because both name "the model must call one or more tools",
#: which is the mapping KBR-214 fixed and T-A3 shipped.
_TOOL_CHOICE_STRINGS: Mapping[str, str] = {
    "none": "none",
    "auto": "auto",
    "required": "any",
}

#: Tool-choice object ``type`` spellings whose ``function``/``custom`` member
#: carries a ``name`` — the named form §3.3.1b maps to ``tool:<name>``.
_TOOL_CHOICE_BY_NAME = frozenset({"function", "custom"})

#: The ``allowed_tools`` tool-choice shape — ``{"type": "allowed_tools",
#: "allowed_tools": {"mode": "auto|required", "tools": [...]}}`` —
#: selects one of a restricted set of declared tools. ``mode`` follows the
#: same string-to-canonical mapping as the top-level ``tool_choice``: a
#: missing ``mode`` defaults to ``auto``, ``required`` maps to ``any``.
_TOOL_CHOICE_ALLOWED_MODES: Mapping[str, str] = {
    "auto": "auto",
    "required": "any",
}

#: A ``data:`` URL carrying base64 image bytes, with its media type.
_DATA_URL = re.compile(r"^data:([^;,]+);base64,(.*)$", re.DOTALL)

#: Any ``data:`` URL at all — a non-base64 one residualises instead of falling
#: through to :attr:`~harness.contract.Image.ref`, so a projection cannot agree
#: with the Responses reader on an unchanged image one way and disagree on
#: another. The two readers' :attr:`~harness.contract.Image` shapes must match
#: for the §3.3.1b comparison to be meaningful.
_ANY_DATA_URL = re.compile(r"^data:", re.IGNORECASE)


class ChatCompletionsProjection:
    """Reads an OpenAI Chat Completions request into the wire-independent form.

    Implements :class:`~harness.contract.Projection` for
    :attr:`~harness.contract.WireFormat.CHAT_COMPLETIONS`.

    The reader is stateless; every method takes what it needs and returns what
    it produced, so one instance is safe to share across a whole corpus run.
    """

    wire_format = c.WireFormat.CHAT_COMPLETIONS

    def read_request(self, captured: c.CapturedRequest) -> c.Request:
        """Project a captured Chat Completions request.

        Args:
            captured: The request as observed on the wire. Only the body is
                consulted — unlike Gemini, Chat Completions carries neither the
                model nor the operation in the URL.

        Returns:
            The wire-independent projection, total over the body.

        Raises:
            UnreadableBodyError: When the body cannot be read — malformed JSON,
                a role no Chat Completions message defines, a message with no
                ``role``, or a content block with no ``type``. ``ValueError`` is
                deliberately **not** caught: from inside a reader it means the
                reader mis-routed a field, which is a reader bug and must
                surface.
        """
        body = _parse_body(captured.body)

        try:
            return _project(body)
        except (KeyError, TypeError, AttributeError) as exc:
            raise c.UnreadableBodyError(f"unreadable Chat Completions body: {exc!r}") from exc


class ChatCompletionsReplyProjection:
    """Reads a Chat Completions reply into :class:`~harness.contract.Reply`.

    Implements :class:`~harness.contract.ReplyProjection` for
    :attr:`~harness.contract.WireFormat.CHAT_COMPLETIONS`. Schema retrieved
    2026-09-16 from
    ``https://raw.githubusercontent.com/openai/openai-openapi/main/openapi.yaml``
    (``CreateChatCompletionResponse`` at line ~36934,
    ``CreateChatCompletionResponse.choices[].finish_reason`` enum at line ~37155:
    ``stop``, ``length``, ``tool_calls``, ``content_filter``, ``function_call``).

    ``finish_reason`` maps onto :data:`~harness.contract.STOP_REASONS`:
    ``stop`` → ``end_turn``; ``length`` → ``max_tokens``; ``tool_calls`` and the
    deprecated ``function_call`` → ``tool_use``; ``content_filter`` → ``error``;
    ``null`` (the streaming-chunk case) → ``stop_reason = None``. The wire carries
    only these five values, so the canonical set covers the mapping and there is
    no ``other`` escape for this format.

    ``choices[0].message.tool_calls[]`` arguments arrive as a JSON string; the
    one shared decode rule (§7.4.1, KBR-174) is :func:`decode_arguments`.
    Reasoning content (``reasoning_content``) is a CC extension; P8's complement
    projects it as :class:`~harness.contract.Thinking`.

    Attributes:
        wire_format: Always :attr:`~harness.contract.WireFormat.CHAT_COMPLETIONS`.
    """

    wire_format = c.WireFormat.CHAT_COMPLETIONS

    #: ``finish_reason`` values that map straight onto a canonical member of
    #: :data:`~harness.contract.STOP_REASONS`.
    _FINISH_REASON_MAP: Mapping[str, str] = MappingProxyType(
        {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
            "content_filter": "error",
        }
    )

    _PROJECTION_KEYS = frozenset(
        {
            "id",
            "object",
            "created",
            "model",
            "metadata",
            "moderation",
            "service_tier",
            "system_fingerprint",
            "usage",
            "choices",
        }
    )

    #: Top-level keys a reply's first choice carries. ``index`` and ``logprobs``
    #: are declared by :class:`CreateChatCompletionResponse` and are consumed
    #: rather than projected (``logprobs`` is provider-reported, ``index`` is
    #: the choice's positional identifier).
    _CHOICE_KEYS = frozenset({"index", "message", "finish_reason", "logprobs"})

    #: Keys a choice's ``message`` object carries. ``audio`` is the audio-output
    #: variant; ``annotations`` and ``name`` are CC extensions consumed at
    #: depth so the fail-closed rule does not fire on real replies.
    _MESSAGE_KEYS = frozenset(
        {
            "role",
            "content",
            "refusal",
            "tool_calls",
            "function_call",
            "reasoning_content",
            "annotations",
            "audio",
            "name",
        }
    )

    def read_reply(self, captured: c.CapturedReply) -> c.Reply:
        """Project a captured Chat Completions reply.

        Args:
            captured: The reply as observed on the wire. SSE reassembly is the
                caller's responsibility (§7.4 boundary).

        Returns:
            The wire-independent projection, total over the body.

        Raises:
            UnreadableBodyError: When the body is not a readable Chat
                Completions reply (malformed JSON, ``choices`` not an array).
        """
        body = _parse_body(captured.body)

        residual: dict[str, Any] = {}
        consumed: set[str] = set()
        for key, value in body.items():
            if key in self._PROJECTION_KEYS:
                consumed.add(key)
            else:
                residual[c.residual_key(key)] = value

        parts, stop_reason, stop_reason_raw = self._read_choices(body.get("choices"), residual)
        usage_raw = body.get("usage")
        if isinstance(usage_raw, dict):
            usage: Mapping[str, Any] = dict(usage_raw)
        elif usage_raw is None:
            usage = {}
        else:
            residual[c.residual_key("usage")] = usage_raw
            usage = {}

        return c.Reply(
            parts=parts,
            stop_reason=stop_reason,
            stop_reason_raw=stop_reason_raw,
            usage=usage,
            residual=residual,
            consumed=frozenset(consumed),
            source=body,
        )

    @classmethod
    def _read_choices(
        cls, value: Any, residual: dict[str, Any]
    ) -> tuple[tuple[c.Part, ...], str | None, str | None]:
        """Read the reply's ``choices`` array.

        A reply is expected to carry exactly one choice — the bridge serves
        ``n = 1`` and a reply with more is a real fidelity anomaly, so extra
        choices residualise at their indexed paths (§3.3.1's *non-empty residual
        fails the run*). An empty or absent ``choices`` array returns
        ``((), None, None)``.

        Args:
            value: The wire value (array of choices, or ``None`` when absent).
            residual: The residual mapping, extended in place.

        Returns:
            ``(parts, stop_reason, stop_reason_raw)`` from the first choice.

        Raises:
            UnreadableBodyError: When the first choice is not an object.
        """
        if not isinstance(value, list):
            residual[c.residual_key("choices")] = value
            return (), None, None
        if not value:
            return (), None, None
        first = value[0]
        if not isinstance(first, dict):
            raise c.UnreadableBodyError(f"choices[0] must be an object, got {type(first).__name__}")
        for index in range(1, len(value)):
            # A bridge reply with ``n > 1`` is the anomaly: residualise the
            # extras at their indexed paths so the run names it.
            residual[c.residual_key("choices", index=index)] = value[index]
        _residualise(first, cls._CHOICE_KEYS, "choices[0]", residual)
        parts = cls._read_message(first.get("message"), residual)
        stop_reason, stop_reason_raw = cls._map_finish_reason(first.get("finish_reason"), residual)
        return parts, stop_reason, stop_reason_raw

    @classmethod
    def _read_message(cls, value: Any, residual: dict[str, Any]) -> tuple[c.Part, ...]:
        """Read one ``message`` object into a parts tuple.

        Args:
            value: The wire value (the choice's ``message`` object, or ``None``).
            residual: The residual mapping, extended in place.

        Returns:
            The projected parts, in wire order: ``Text`` (content), ``Text``
            (refusal — §7.4.1 records the refusal case as *content whose
            identity is a run of text*), then ``ToolUse`` per ``tool_call``,
            then ``Thinking`` for ``reasoning_content``. The legacy
            ``function_call`` key projects as an additional ``ToolUse`` when
            present (the deprecated single-call form). Unknown message keys
            residualise at ``choices[0].message.<key>`` so the fail-closed
            rule fires on unmodelled payload.

        Raises:
            UnreadableBodyError: When ``message`` is not an object or a
                ``tool_calls`` entry lacks the required ``function.name``.
        """
        if not isinstance(value, dict):
            residual[c.residual_key("choices[0].message")] = value
            return ()

        _residualise(value, cls._MESSAGE_KEYS, "choices[0].message", residual)

        parts: list[c.Part] = []

        # ``content`` — a bare string, or ``null`` for tool-only replies.
        content = value.get("content")
        if isinstance(content, str):
            parts.append(c.Text(content))
        elif content is not None:
            residual[c.residual_key("choices[0].message", "content")] = content

        # ``refusal`` — a bare string; §7.4.1 records it as text-identity. An
        # empty string carries no semantic and is treated as absent so a wire
        # that always emits ``refusal: ""`` does not invent a refusal text.
        refusal = value.get("refusal")
        if isinstance(refusal, str) and refusal:
            parts.append(c.Text(refusal))
        elif refusal is not None and not isinstance(refusal, str):
            residual[c.residual_key("choices[0].message", "refusal")] = refusal

        # ``tool_calls`` — each is a function-call payload with arguments as a
        # JSON string; the one shared decode rule (§7.4.1, KBR-174) handles it.
        # A wrongly-typed value residualises at its own path; a malformed
        # *entry* (not a dict, or no string ``function.name``) raises
        # ``UnreadableBodyError`` because there is no partial projection to
        # salvage — §7.4.1.
        tool_calls = value.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                residual[c.residual_key("choices[0].message", "tool_calls")] = tool_calls
            else:
                for index, call in enumerate(tool_calls):
                    if not isinstance(call, dict):
                        residual[
                            c.residual_key("choices[0].message.tool_calls", index=index)
                        ] = call
                        continue
                    if not isinstance(call.get("function"), dict):
                        raise c.UnreadableBodyError(
                            f"choices[0].message.tool_calls[{index}].function must be an object"
                        )
                    if not isinstance(call["function"].get("name"), str):
                        raise c.UnreadableBodyError(
                            f"choices[0].message.tool_calls[{index}].function.name must be a string"
                        )
                    parts.append(
                        c.ToolUse(
                            name=call["function"]["name"],
                            arguments=c.decode_arguments(
                                call["function"].get("arguments"),
                                f"choices[0].message.tool_calls[{index}].function.arguments",
                                residual,
                            ),
                            id=call.get("id") if isinstance(call.get("id"), str) else None,
                        )
                    )

        # ``reasoning_content`` — CC extension; P8's complement. An empty string
        # still projects as a Thinking part so its absence is observable (§3.3.1).
        reasoning = value.get("reasoning_content")
        if isinstance(reasoning, str):
            parts.append(c.Thinking(text=reasoning))
        elif reasoning is not None:
            residual[c.residual_key("choices[0].message", "reasoning_content")] = reasoning

        # The legacy ``function_call`` key, when present alongside ``tool_calls``,
        # projects as an additional ``ToolUse`` (deprecated form). A dict
        # without a string ``name`` is malformed — there is no partial call
        # to salvage — and raises.
        function_call = value.get("function_call")
        if function_call is not None:
            if not isinstance(function_call, dict):
                residual[c.residual_key("choices[0].message", "function_call")] = function_call
            elif not isinstance(function_call.get("name"), str):
                raise c.UnreadableBodyError(
                    "choices[0].message.function_call.name must be a string"
                )
            else:
                parts.append(
                    c.ToolUse(
                        name=function_call["name"],
                        arguments=c.decode_arguments(
                            function_call.get("arguments"),
                            "choices[0].message.function_call.arguments",
                            residual,
                        ),
                    )
                )

        # ``audio`` — the audio-output variant of a message. Carries as
        # ``Opaque`` so the payload stays detectable by digest; ``opaque_kind``
        # raises for non-snake_case wire spellings, which is the wire's fault,
        # not the reader's, so the ``ValueError`` becomes ``UnreadableBodyError``
        # per §7.4.1.
        audio = value.get("audio")
        if audio is not None:
            try:
                canonical = c.opaque_kind("audio")
            except ValueError as exc:
                raise c.UnreadableBodyError(
                    f"choices[0].message.audio: {exc}"
                ) from exc
            parts.append(c.Opaque(kind=canonical, digest=c.opaque_digest(audio)))

        return tuple(parts)

    @classmethod
    def _map_finish_reason(
        cls, value: Any, residual: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        """Map the wire ``finish_reason`` onto :data:`~harness.contract.STOP_REASONS`.

        Args:
            value: The wire value (``str``, ``None``, or a non-string — the
                latter residualises at ``choices[0].finish_reason``).
            residual: The residual mapping, extended in place.

        Returns:
            ``(stop_reason, stop_reason_raw)``. ``None`` for an absent or
            non-string value. A string outside the published five-value enum
            escapes through ``other`` with the wire string in
            ``stop_reason_raw`` — the same posture §3.3.1 puts on Anthropic
            and Gemini, so a future vendor addition fails the run only when
            T-D10's register match cannot name it, not here.
        """
        if value is None:
            return None, None
        if not isinstance(value, str):
            residual[c.residual_key("choices[0]", "finish_reason")] = value
            return None, None
        mapped = cls._FINISH_REASON_MAP.get(value)
        if mapped is not None:
            return mapped, None
        return "other", value


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
    # `parallel_tool_calls` is routed through `PARALLEL_TOOL_CALLS_KEY` rather
    # than the wire key so the two spellings cannot drift if either ever
    # changes — the wire key is already the canonical spelling today, and the
    # alias keeps it that way by construction.
    for key, value in body.items():
        if key in ("model", "stream", "messages", "tools"):
            consumed.add(key)
        elif key in _SAMPLING_KEYS:
            sampling[key] = value
            consumed.add(key)
        elif key == "tool_choice":
            extra[c.TOOL_CHOICE_KEY] = _read_tool_choice(value, residual)
            consumed.add(key)
        elif key == "parallel_tool_calls":
            # Conditional on the value type, mirroring the Anthropic
            # reader's analogous site for `disable_parallel_tool_use`: a
            # wrongly-typed value is *not* consumed by this branch — the
            # fall-through ``else: residual[key] = value`` puts it in the
            # residual at its bare name, the run fails closed, and the
            # cross-reader comparison sees the same answer either side.
            #
            # ``None`` is treated as absent (the cache_control precedent):
            # the wire key carries no instruction, and the bridge does
            # not invent one. ``True``/``False`` are read; ``True`` is the
            # CC default so a body carrying ``true`` is *not* written to
            # ``extra[parallel_tool_calls]`` — §3.3.1b: "the reader writes
            # the entry only when the wire carries a non-default value,
            # mirroring KBR-214's forwarding rule; an absent entry and
            # an explicit default are one request on both wires; writing
            # both would invent a second field some providers reject and
            # every comparison would carry". A body carrying ``false`` is
            # a non-default delta the oracle names.
            raw = body.get("parallel_tool_calls")
            if raw is None:
                # Absent — the key is present in the body but the value is
                # null. Treat as no-op (the cache_control precedent) but
                # *consume* the key for totality: a body that explicitly
                # sent ``null`` has named its intent to omit the field.
                consumed.add(key)
            elif raw is False:
                extra[c.PARALLEL_TOOL_CALLS_KEY] = False
                consumed.add(key)
            elif raw is True:
                # Default — not written. The key is consumed for totality
                # so a body that explicitly sent ``true`` is accounted
                # for; the absence in ``extra[parallel_tool_calls]`` is
                # the canonical form.
                consumed.add(key)
            else:
                residual["parallel_tool_calls"] = raw
        elif key in _PUBLISHED_EXTRA_KEYS:
            # Keyed by the wire key, never nested (§3.3.1a). `store` joins
            # this set for the same reason: the canonical address is the wire
            # key, and `Envelope.store`'s `bool | None` would coerce a
            # `"true"` string the schema forbids — a wrongly-typed leaf
            # residualises, so reading it here and letting `Envelope` carry
            # `None` for a body that never sent it is the honest shape.
            extra[key] = value
            consumed.add(key)
        else:
            residual[key] = value

    messages = body.get("messages")
    system = _read_system_messages(messages, residual)
    turns = _read_messages(messages, residual)

    conversation = c.Conversation(
        system=system,
        turns=turns,
        tools=_read_tools(body.get("tools"), residual),
        sampling=sampling,
    )

    envelope = c.Envelope(
        model=_typed_leaf(body, "model", str, "", residual),
        stream=_typed_leaf(body, "stream", bool, "", residual),
        # `store` rides `extra[store]` like every other CC control field; the
        # grammar's named `Envelope.store` is Responses-only (§3.3.1a), and
        # giving CC's `store` a second address would let a comparison see one
        # side carrying it and the other not.
        store=None,
        extra=extra,
    )

    return c.Request(
        envelope=envelope,
        conversation=conversation,
        residual=residual,
        consumed=frozenset(consumed),
        source=body,
    )


def _read_system_messages(
    value: Any, residual: dict[str, Any]
) -> tuple[c.Text, ...]:
    """Lift ``system`` and ``developer`` messages into ordered text parts.

    Runs as a second pass over ``messages`` because :class:`~harness.contract.
    Conversation` takes ``system`` and ``turns`` in one constructor call, and a
    role that is system-shaped in the turns pass is a system here too — lifting
    it in two places would disagree the first time a body carried both.

    Args:
        value: The ``messages`` field, absent or a list of messages.
        residual: The residual mapping, extended with any content-part key the
            grammar cannot carry.

    Returns:
        The system text, in order.

    Raises:
        UnreadableBodyError: Propagated from :func:`_system_parts`.
    """
    if value is None:
        return ()

    parts: list[c.Text] = []
    for index, message in enumerate(value):
        if isinstance(message, dict) and message.get("role") in _SYSTEM_ROLES:
            parts.extend(_system_parts(message, index, residual))
    return tuple(parts)


def _system_parts(
    message: Mapping[str, Any], index: int, residual: dict[str, Any]
) -> tuple[c.Text, ...]:
    """Read one system or developer message's content into text parts.

    Args:
        message: The message object.
        index: The message's position, for residual keys.
        residual: The residual mapping, extended with any content-part key
            the grammar cannot carry.

    Returns:
        The message's text parts, in order.

    Raises:
        UnreadableBodyError: When the content is neither a string nor a list,
            or a part is not an object carrying ``type: "text"``.
    """
    content = message.get("content")
    if isinstance(content, str):
        return (c.Text(content),)
    if not isinstance(content, list):
        raise c.UnreadableBodyError(
            f"messages[{index}] system content must be a string or a list, got {type(content).__name__}"
        )

    parts: list[c.Text] = []
    for part_index, part in enumerate(content):
        if not isinstance(part, dict):
            raise c.UnreadableBodyError(
                f"messages[{index}].content[{part_index}] must be an object"
            )
        if part.get("type") != "text":
            raise c.UnreadableBodyError(
                f"messages[{index}].content[{part_index}] must be a text part"
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise c.UnreadableBodyError(
                f"messages[{index}].content[{part_index}] text must be a string"
            )
        # Read cache_control onto the part — a system or developer message
        # is rare to carry one, but the schema permits it on content parts,
        # and a silent drop would be the M16-shaped loss the residual rule
        # exists to prevent. Residualise every key the grammar does not
        # model on this part, so an unknown field is named rather than
        # swallowed.
        part_path = f"messages[{index}].content[{part_index}]"
        cache_control = _read_cache_control(part, part_path, residual)
        _residualise(part, {"type", "text", *_CACHE_KEYS}, part_path, residual)
        parts.append(c.Text(text, cache_control=cache_control))
    return tuple(parts)


def _read_messages(value: Any, residual: dict[str, Any]) -> tuple[c.Turn, ...]:
    """Read the conversation turns per §3.3.1b's merge rule.

    Args:
        value: The ``messages`` field, absent or a list of messages.
        residual: The residual mapping, extended with anything the grammar
            cannot carry.

    Returns:
        The turns, normalised.

    Raises:
        UnreadableBodyError: When ``messages`` is not a list, a message is not
            an object or lacks ``role``/``content``, or a role is not one
            :data:`_MESSAGE_ROLES` names.
    """
    if value is None:
        # Absent is not an error, the T-A1 precedent: the oracle catches a
        # vanished conversation as a `conversation.turns` delta, which is a
        # better diagnosis than an unreadable-body error.
        return ()

    if not isinstance(value, list):
        raise c.UnreadableBodyError(f"messages must be a list, got {type(value).__name__}")

    turns: list[c.Turn] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise c.UnreadableBodyError(f"messages[{index}] must be an object")
        if "role" not in message:
            raise c.UnreadableBodyError(f"messages[{index}] lacks a role")

        role = message["role"]
        if role in _SYSTEM_ROLES:
            # Lifted by `_read_system_messages`; not a turn. Residualising the
            # message's keys here would put `content` in the residual twice —
            # once here, once in the system pass — so this pass just claims
            # the keys it consumed.
            consumed_system = {"role", "content"} | ({"name"} if "name" in message else set())
            _residualise(message, consumed_system, c.residual_key("messages", index=index), residual)
            continue
        if role not in _MESSAGE_ROLES:
            raise c.UnreadableBodyError(
                f"messages[{index}] role must be one of {sorted(_MESSAGE_ROLES)}, got {role!r}"
            )

        turns.append(_read_one_message(message, index, residual))

    return _normalise_turns(turns)


def _read_one_message(
    message: Mapping[str, Any], index: int, residual: dict[str, Any]
) -> c.Turn:
    """Read one ``user``, ``assistant`` or ``tool`` message into a turn.

    Args:
        message: The message object.
        index: The message's position, for residual keys.
        residual: The residual mapping.

    Returns:
        The turn, with its parts in the wire's order.

    Raises:
        UnreadableBodyError: When a content part or a tool call is malformed.
    """
    role = message["role"]
    path = c.residual_key("messages", index=index)

    if role == "tool":
        # A `tool` message is a `ToolResult` part inside a `user` turn. Its
        # `tool_call_id` populates `ToolResult.tool_use_id`; its content is
        # always text per the schema.
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            raise c.UnreadableBodyError(f"{path} tool message must carry a string tool_call_id")
        content = _read_result_content(message.get("content"), path, residual)
        _residualise(message, {"role", "content", "tool_call_id"}, path, residual)
        return c.Turn(role="user", parts=(c.ToolResult(content=content, tool_use_id=tool_call_id),))

    if role == "assistant":
        parts: list[c.Part] = []
        # The message-level `refusal` field (§7.4.1's `text_digest` rationale
        # — "a CC refusal is a bare string on the message, so there is no
        # block for T-A2 to hash"). Projects as the same `Opaque("refusal",
        # digest=text_digest)` shape the content-part form uses, so the two
        # representations of a refusal in the same format don't drift. An
        # absent `refusal` (the common case) is the absent value — the turn's
        # parts come from `content` and `tool_calls`.
        refusal = message.get("refusal")
        if refusal is not None:
            if not isinstance(refusal, str):
                raise c.UnreadableBodyError(
                    f"{path} assistant.refusal must be a string when present"
                )
            parts.append(c.Opaque(kind="refusal", digest=c.text_digest(refusal)))

        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise c.UnreadableBodyError(f"{path} tool_calls must be a list")
            parts.extend(_read_tool_calls(tool_calls, path, residual))

        content_parts = _read_content(message.get("content"), path, residual, _ASSISTANT_PART_TYPES)
        parts.extend(content_parts)

        _residualise(
            message,
            # `audio` and `function_call` are NOT in the mapped set: they
            # are recognised CC fields (assistant-level `audio` for audio
            # responses, deprecated `function_call` for the older tool-call
            # spelling), but neither is carried by the projection today.
            # Residualising them at the message's path names the field,
            # the run fails closed, and a future reader that grows an
            # `Audio` part or a `function_call` mapping has the residual
            # entry to consult. A silent drop would be exactly the totality
            # violation the residual rule exists to prevent.
            {"role", "content", "tool_calls", "name", "refusal"},
            path,
            residual,
        )
        return c.Turn(role="assistant", parts=tuple(parts))

    # role == "user"
    parts = list(_read_content(message.get("content"), path, residual, _USER_PART_TYPES))
    _residualise(message, {"role", "content", "name"}, path, residual)
    return c.Turn(role="user", parts=tuple(parts))


def _read_tool_calls(
    value: Sequence[Any], path: str, residual: dict[str, Any]
) -> list[c.Part]:
    """Read one assistant message's ``tool_calls`` into ``ToolUse`` parts.

    Args:
        value: The message's ``tool_calls`` list.
        path: The message's path from the body root, for residual keys.
        residual: The residual mapping.

    Returns:
        The ``ToolUse`` parts, in the wire's order.

    Raises:
        UnreadableBodyError: When a call carries no ``id``, no ``function``,
            or a ``function`` with no ``name``.
    """
    parts: list[c.Part] = []
    for index, call in enumerate(value):
        call_path = f"{path}.tool_calls[{index}]"
        if not isinstance(call, dict):
            raise c.UnreadableBodyError(f"{call_path} must be an object")

        kind = call.get("type")
        function = call.get("function")
        if kind == "custom" and function is None:
            # OpenAI's `custom` tool-call shape carries `custom.name` and
            # `custom.input`, not `function`. `ToolUse` models the `function`
            # shape; a `custom` one residuals at its own path — it is rare,
            # and forcing it into `ToolUse` would invent a mapping the design
            # has not decided.
            _residualise(call, set(call), call_path, residual)
            continue

        if not isinstance(function, dict):
            raise c.UnreadableBodyError(f"{call_path} must carry a function object")
        if not isinstance(call.get("id"), str):
            raise c.UnreadableBodyError(f"{call_path} must carry a string id")
        if not isinstance(function.get("name"), str):
            raise c.UnreadableBodyError(f"{call_path}.function must carry a name")

        _residualise(
            call, {"type", "id", "function", "index"}, call_path, residual
        )
        _residualise(
            function,
            {"name", "arguments"},
            f"{call_path}.function",
            residual,
        )
        parts.append(
            c.ToolUse(
                name=function["name"],
                arguments=c.decode_arguments(
                    function.get("arguments"),
                    c.residual_key(f"{call_path}.function", "arguments"),
                    residual,
                ),
                id=call["id"],
            )
        )
    return parts


def _read_content(
    value: Any,
    path: str,
    residual: dict[str, Any],
    part_types: frozenset[str],
) -> tuple[c.Part, ...]:
    """Read a message's ``content`` into ordered parts.

    Args:
        value: The ``content`` field: a string, a list of parts, or ``None``
            (which an assistant message may carry when only ``tool_calls`` are
            present).
        path: The message's path from the body root, for residual keys.
        residual: The residual mapping.
        part_types: The part types this message's role admits.

    Returns:
        The parts, in order.

    Raises:
        UnreadableBodyError: When content is neither a string, a list, nor
            ``None``, or a part is not an object carrying a type in
            ``part_types``.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (c.Text(value),)
    if not isinstance(value, list):
        raise c.UnreadableBodyError(
            f"{path} content must be a string, a list, or null, got {type(value).__name__}"
        )

    parts: list[c.Part] = []
    for index, part in enumerate(value):
        part_path = f"{path}.content[{index}]"
        parts.append(_read_content_part(part, part_path, residual, part_types))
    return tuple(parts)


def _read_content_part(
    part: Any, path: str, residual: dict[str, Any], part_types: frozenset[str]
) -> c.Part:
    """Read one content part into a part.

    Args:
        part: The content-part object.
        path: The part's path from the body root, for residual keys.
        residual: The residual mapping.
        part_types: The part types the part's carrying role admits.

    Returns:
        The part.

    Raises:
        UnreadableBodyError: When the part is not an object, carries no
            ``type``, or its type is not one this role admits.
    """
    if not isinstance(part, dict):
        raise c.UnreadableBodyError(f"{path} must be an object, got {type(part).__name__}")
    kind = part.get("type")
    if kind not in part_types:
        raise c.UnreadableBodyError(
            f"{path} type must be one of {sorted(part_types)}, got {kind!r}"
        )

    if kind == "text":
        text = part.get("text")
        if not isinstance(text, str):
            raise c.UnreadableBodyError(f"{path} text must be a string")
        cache_control = _read_cache_control(part, path, residual)
        _residualise(part, {"type", "text", *_CACHE_KEYS}, path, residual)
        return c.Text(text, cache_control=cache_control)

    if kind == "refusal":
        # A refusal is NOT collapsed into Text. T-A3 ships the same decision
        # for Responses: unlike a plain text part, a refusal is not recoverable
        # from the turn's role — an assistant refusal and an assistant answer
        # would otherwise project identically, and a bridge that turned one
        # into the other would be invisible to the oracle. `text_digest`, not
        # `opaque_digest`: a refusal is identified by its text, and §7.4.1
        # pins the two recipes separately because Chat Completions carries a
        # refusal as a bare string with no block to hash — that is the same
        # recipe applied one wrapper up, here to the content-part form the
        # schema also publishes.
        refusal = part.get("refusal")
        if not isinstance(refusal, str):
            raise c.UnreadableBodyError(f"{path} refusal must be a string")
        _residualise(part, {"type", "refusal", *_CACHE_KEYS}, path, residual)
        return c.Opaque(
            kind="refusal",
            digest=c.text_digest(refusal),
            cache_control=_read_cache_control(part, path, residual),
        )

    if kind == "image_url":
        image = _read_image_part(part, path, residual)
        if image is None:
            # The image could not be projected (undecodable base64, non-base64
            # data URL, etc.); the helper residualised the offending entry.
            # Returning a placeholder Part would still count as an image in
            # ``verify_total``, but the reader has nothing to put there —
            # contract's XOR invariant rejects both-None, and an ``Opaque``
            # would invent a `kind` the schema does not name.
            return c.Text("")
        return image

    # `input_audio` and `file` — unmodelled shapes, `Opaque` per the alias rule.
    return _read_opaque_part(part, str(kind), path, residual)


def _read_image_part(
    part: Mapping[str, Any], path: str, residual: dict[str, Any]
) -> c.Image | None:
    """Read an ``image_url`` content part into an :class:`~harness.contract.Image`.

    Args:
        part: The content-part object.
        path: The part's path from the body root, for residual keys.
        residual: The residual mapping.

    Returns:
        The image part, with a digest when the URL carries decodable bytes and
        a ``ref`` otherwise; ``None`` when the part carries an undecodable
        or non-base64 data URL — the helper residualises the offending entry
        and the caller substitutes an empty Text to keep ``verify_total``
        honest.

    Raises:
        UnreadableBodyError: When the part carries no ``image_url`` object.
    """
    image_url = part.get("image_url")
    if not isinstance(image_url, dict):
        raise c.UnreadableBodyError(f"{path} must carry an image_url object")

    url = image_url.get("url")
    if not isinstance(url, str):
        raise c.UnreadableBodyError(f"{path}.image_url must carry a string url")

    cache_control = _read_cache_control(part, path, residual)
    _residualise(part, {"type", "image_url", *_CACHE_KEYS}, path, residual)
    _residualise(image_url, {"url", "detail"}, f"{path}.image_url", residual)

    # A `data:` URL carrying base64 bytes projects as a digest — the same
    # recipe the Responses reader applies, so the two projections agree on an
    # unchanged image. Any other URL keeps its `ref`.
    data_match = _DATA_URL.match(url)
    if data_match:
        media_type = data_match.group(1)
        encoded = data_match.group(2)
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            residual[f"{path}.image_url.url"] = url
            # A `cache_control` on a part whose bytes failed to decode
            # cannot survive — the part has no Image to carry it on, and
            # an `Opaque` or `Text` substitute would invent a carrier the
            # schema does not name. Residualising at the field's own path
            # names it, the run fails closed, and a body carrying a
            # undecodable image with a cache breakpoint is one the bridge
            # cannot faithfully forward.
            if cache_control is not None:
                residual[f"{path}.cache_control"] = cache_control
            return None

        return c.Image(
            digest=c.image_digest(decoded),
            media_type=media_type,
            cache_control=cache_control,
        )
    if _ANY_DATA_URL.match(url):
        # A non-base64 data URL — `data:image/png,abc` — cannot be digested
        # without inventing a decoding, and `ref` would make the two readers
        # disagree. Residualised at its own path, the run fails with the
        # field named. The `cache_control` residualises here too, for the
        # same reason the base64-decode-failure branch does: the part has
        # no `Image` to carry it on, and a silent drop would be the
        # M16-shaped loss the residual rule exists to prevent.
        residual[f"{path}.image_url.url"] = url
        if cache_control is not None:
            residual[f"{path}.cache_control"] = cache_control
        return None

    return c.Image(ref=url, cache_control=cache_control)


def _read_opaque_part(
    part: Mapping[str, Any], kind: str, path: str, residual: dict[str, Any]
) -> c.Opaque:
    """Read an ``input_audio`` or ``file`` content part into an :class:`~harness.contract.Opaque`.

    Args:
        part: The content-part object.
        kind: The part's wire type, which becomes :attr:`~harness.contract.Opaque.kind`.
        path: The part's path from the body root, for residual keys.
        residual: The residual mapping.

    Returns:
        The opaque part, carrying a digest of its payload.

    Raises:
        UnreadableBodyError: When the wire type has no canonical name.
    """
    _residualise(part, set(part), path, residual)

    try:
        canonical = c.opaque_kind(kind)
    except ValueError as exc:
        raise c.UnreadableBodyError(f"{path}: {exc}") from exc

    return c.Opaque(
        kind=canonical,
        digest=c.opaque_digest(part),
        cache_control=_read_cache_control(part, path, residual),
    )


def _read_result_content(
    value: Any, path: str, residual: dict[str, Any]
) -> tuple[c.Text, ...]:
    """Read a ``tool`` message's ``content`` into text parts.

    Args:
        value: The ``content`` field: a string or a list of text parts.
        path: The message's path from the body root.
        residual: The residual mapping.

    Returns:
        The result content, in order.

    Raises:
        UnreadableBodyError: When content is neither a string nor a list, or a
            part is not an object carrying ``type: "text"``.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (c.Text(value),)
    if not isinstance(value, list):
        raise c.UnreadableBodyError(
            f"{path} tool content must be a string or a list, got {type(value).__name__}"
        )

    parts: list[c.Text] = []
    for index, part in enumerate(value):
        part_path = f"{path}.content[{index}]"
        if not isinstance(part, dict):
            raise c.UnreadableBodyError(f"{part_path} must be an object")
        if part.get("type") != "text":
            raise c.UnreadableBodyError(f"{part_path} must be a text part")
        text = part.get("text")
        if not isinstance(text, str):
            raise c.UnreadableBodyError(f"{part_path} text must be a string")
        _residualise(part, {"type", "text", *_CACHE_KEYS}, part_path, residual)
        parts.append(c.Text(text, cache_control=_read_cache_control(part, part_path, residual)))
    return tuple(parts)


def _read_tools(value: Any, residual: dict[str, Any]) -> tuple[c.ToolDecl, ...]:
    """Read the declared tools.

    Args:
        value: The ``tools`` field, absent or a list of declarations.
        residual: The residual mapping, extended with any entry key the grammar
            cannot carry.

    Returns:
        The declarations, in order.

    Raises:
        UnreadableBodyError: When ``tools`` is not a list, an entry is not an
            object, or its ``function`` carries no name.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise c.UnreadableBodyError(f"tools must be a list, got {type(value).__name__}")

    declared: list[c.ToolDecl] = []
    for index, tool in enumerate(value):
        if not isinstance(tool, dict):
            raise c.UnreadableBodyError(f"tools[{index}] must be an object")
        path = c.residual_key("tools", index=index)

        kind = tool.get("type")
        function = tool.get("function")
        if kind != "function" or not isinstance(function, dict):
            # A non-function tool declaration (OpenAI's built-ins) residualises
            # at its own path — the Chat Completions surface kitty translates
            # to declares only `function` tools, so any other kind is one the
            # product cannot faithfully forward, and naming it is the
            # honest answer.
            residual[path] = tool
            continue

        if not isinstance(function.get("name"), str):
            raise c.UnreadableBodyError(f"{path}.function must carry a name")

        schema = function.get("parameters")
        if schema is not None and not isinstance(schema, dict):
            residual[c.residual_key("tools", "parameters", index=index)] = schema
            schema = None

        _residualise(tool, {"type", "function"}, path, residual)
        _residualise(
            function,
            {"name", "description", "parameters", "strict", *_CACHE_KEYS},
            f"{path}.function",
            residual,
        )
        declared.append(
            c.ToolDecl(
                name=function["name"],
                description=_typed_leaf(
                    function, "description", str, f"{path}.function", residual
                ),
                schema=schema,
                strict=_typed_leaf(function, "strict", bool, f"{path}.function", residual),
                # The Chat Completions format does not carry a tool `type`
                # leaf the way Anthropic's does — the discriminator is the
                # outer object's `type`, which is `"function"`. Carried as
                # such, so the two readers' `ToolDecl.type` agree.
                type="function",
                cache_control=_read_cache_control(function, f"{path}.function", residual),
            )
        )
    return tuple(declared)


def _read_tool_choice(value: Any, residual: dict[str, Any]) -> str:
    """Normalise a ``tool_choice`` onto its canonical value.

    Four wire keys across the formats name one concept, so §3.3.1b makes the
    *value* canonical too. Chat Completions carries three strings —
    ``none``, ``auto``, ``required`` — where ``required`` maps to ``any``
    (both name "the model must call one or more tools"; the mapping is
    KBR-214's and T-A3 shipped it), and one named form whose
    ``function``/``custom`` member carries a ``name``.

    Args:
        value: The wire value.

    Returns:
        The canonical value.

    Raises:
        UnreadableBodyError: When the shape is not one the format publishes.
            Classified here rather than letting
            :class:`~harness.contract.Envelope` judge, for the reason T-A1
            gives: ``Envelope`` accepts any string starting ``tool:``, so a
            missing name would become the canonical-looking ``"tool:None"``
            that no register row can interpret.
    """
    if isinstance(value, str):
        if value in _TOOL_CHOICE_STRINGS:
            return _TOOL_CHOICE_STRINGS[value]
        raise c.UnreadableBodyError(f"unrecognised tool_choice string {value!r}")

    if not isinstance(value, dict):
        raise c.UnreadableBodyError(
            f"tool_choice must be a string or an object, got {type(value).__name__}"
        )

    kind = value.get("type")
    if kind in _TOOL_CHOICE_BY_NAME:
        # The named form. `function` and `custom` carry their member's `name`
        # — the shape §3.3.1b maps to `tool:<name>`. Residualise every other
        # key the reader does not model, on both the `tool_choice` object and
        # its member, so an unknown field is named rather than swallowed.
        member_key = "function" if kind == "function" else "custom"
        member = value.get(member_key)
        if not isinstance(member, dict):
            raise c.UnreadableBodyError(
                f"tool_choice type {kind!r} requires a {member_key} object"
            )
        name = member.get("name")
        if not isinstance(name, str):
            raise c.UnreadableBodyError(
                f"tool_choice type {kind!r} requires a {member_key}.name string"
            )
        for key, item in value.items():
            if key != "type" and key != member_key:
                residual[f"tool_choice.{key}"] = item
        for key, item in member.items():
            if key != "name":
                residual[f"tool_choice.{member_key}.{key}"] = item
        return f"tool:{name}"

    if kind == "allowed_tools":
        # A restricted set of tools the model may call. `mode` follows the
        # top-level strings — CC's `required` maps to `any`, the same mapping
        # KBR-214 fixed and T-A3 ships. The member's `tools` list is not
        # projected (the canonical form names the *mode*, not the set), so
        # the whole entry's presence is what the projection records. The
        # list itself residuals at its own path: a body that names a
        # specific toolset is one the bridge does not silently swallow,
        # and the residual entry names what was sent.
        allowed = value.get("allowed_tools")
        if not isinstance(allowed, dict):
            raise c.UnreadableBodyError(
                "tool_choice type 'allowed_tools' requires an allowed_tools object"
            )
        mode = allowed.get("mode", "auto")
        if mode not in _TOOL_CHOICE_ALLOWED_MODES:
            raise c.UnreadableBodyError(
                f"tool_choice.allowed_tools.mode must be one of {sorted(_TOOL_CHOICE_ALLOWED_MODES)}, got {mode!r}"
            )
        for key, item in value.items():
            if key != "type" and key != "allowed_tools":
                residual[f"tool_choice.{key}"] = item
        for key, item in allowed.items():
            if key != "mode":
                residual[f"tool_choice.allowed_tools.{key}"] = item
        return _TOOL_CHOICE_ALLOWED_MODES[mode]

    # A `type` this reader does not recognise residuals whole: the caller
    # wraps the value in `extra[tool_choice]`, and a body carrying an
    # unknown `type` is one the bridge cannot faithfully forward. The
    # reader-side failure mode is UnreadableBodyError, which is the
    # T-A1 precedent.
    raise c.UnreadableBodyError(f"unrecognised tool_choice type {kind!r}")


def _typed_leaf(
    source: Mapping[str, Any],
    key: str,
    expected: type | tuple[type, ...],
    path: str,
    residual: dict[str, Any],
    default: Any = None,
) -> Any:
    """Return an optional leaf, residualising it when the wire carried the wrong type.

    §7.4.1's wrongly-typed-leaf rule, the same shape T-A1 ships.

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


def _read_cache_control(
    part: Mapping[str, Any], path: str, residual: dict[str, Any]
) -> Mapping[str, Any] | None:
    """Read a content part's or tool declaration's cache breakpoint into the slot.

    Both spellings fill the same slot **verbatim** — the G37 decision, §11 Q16.
    ``prompt_cache_breakpoint`` carries ``{"mode": "explicit"}`` with no TTL;
    ``cache_control`` (OpenRouter's CC dialect, Anthropic's own spelling)
    carries ``{"type": "ephemeral"}`` with an optional ``ttl``. The slot is
    ``Mapping[str, Any] | None``, and §3.3.1's "carried whole, not reduced"
    rule applies to a spelling with no TTL the same way it applies to one
    with: a flattened form would make a vendor's future TTL invisible, and a
    normalised form would put a vendor spelling into a wire-independent value.

    When **both** spellings are present on one part (the schema forbids it,
    so a body carrying both is a mutation the reader is obliged to name), the
    first one in :data:`_CACHE_KEYS` order fills the slot and the second
    residualises at its own path. A silent drop of the second is exactly the
    shape M16 and G37 exist to prevent — a strip the bridge did not
    register.

    Args:
        part: The content part or tool declaration being read.
        path: The object's path from the body root, for the residual key.
        residual: The residual mapping, extended in place when the value is
            not an object, or when both spellings are present.

    Returns:
        The first spelling's value as the wire mapping, or ``None`` when
        absent or unusable.
    """
    filled: Mapping[str, Any] | None = None
    for key in _CACHE_KEYS:
        value = part.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            residual[f"{path}.{key}"] = value
            continue
        if filled is None:
            filled = value
        else:
            residual[f"{path}.{key}"] = value
    return filled


def _residualise(
    source: Mapping[str, Any],
    mapped: Collection[str],
    prefix: str,
    residual: dict[str, Any],
) -> None:
    """Record every key of ``source`` the reader did not map.

    §3.3.1's "unknown fields fail closed", applied at depth.

    Args:
        source: The object being read.
        mapped: The keys the caller accounted for. Accepts a ``set`` or a
            ``frozenset`` — the per-block call sites build a fresh ``set``
            from a literal, while :data:`_TOOL_KEYS` is a ``frozenset`` so a
            reader cannot quietly grow it mid-request. Mirrors the Anthropic
            reader's widened signature.
        prefix: The object's path from the body root, to which each unmapped
            key is appended.
        residual: The residual mapping, extended in place.
    """
    for key, value in source.items():
        if key not in mapped:
            residual[c.residual_key(prefix, key)] = value


def _normalise_turns(turns: Sequence[c.Turn]) -> tuple[c.Turn, ...]:
    """Merge consecutive same-role turns (§3.3.1b's fourth clause).

    On Chat Completions, clause 1 and clause 2 of the merge rule are satisfied
    by the wire itself — a run of ``tool`` messages forms the run, and an
    immediately following non-tool ``user`` message is a separate message the
    reader already sees contiguously. Clause 3 is vacuous: every ``tool``
    message is already a ``ToolResult`` turn. Clause 4 — consecutive same-role
    turns merge — is the only work left.

    Args:
        turns: The turns as the wire delivered them.

    Returns:
        The merged turns, same-role runs collapsed into one turn.
    """
    merged: list[c.Turn] = []
    for turn in turns:
        if merged and merged[-1].role == turn.role:
            merged[-1] = c.Turn(role=turn.role, parts=(*merged[-1].parts, *turn.parts))
        else:
            merged.append(turn)
    return tuple(merged)
