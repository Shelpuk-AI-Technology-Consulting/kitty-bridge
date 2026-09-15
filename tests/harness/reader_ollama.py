"""The Ollama ``/api/chat`` reader — projects an Ollama chat body into the canonical form.

``.system_design/TEST_SUITE.md`` §3.3.1, §3.3.1a, §3.3.1b, §7.4, §7.4.1, §7.4.2 ·
plan task **T-A6** ([KBR-38](https://shelpuk.atlassian.net/browse/KBR-38)).

This module **imports nothing from** ``src/kitty``, and must not — §3.3.1's
independent-oracle rule: a reader validated against kitty's output inherits
kitty's bugs and the oracle becomes circular. It is written against Ollama's
published ``docs/api.md``, retrieved 2026-09-15 from
``raw.githubusercontent.com/ollama/ollama/main/docs/api.md`` (the same source
the design doc cites). The companion L1 test module
``tests/harness/test_reader_ollama.py`` asserts every published ``/api/chat``
example round-trips with an empty residual.

**The wire format.** ``POST /api/chat`` carries a JSON object with these
top-level keys (from the parameters block at ``docs/api.md`` line 492-513):

* ``model`` (required) — the model name
* ``messages`` — ordered chat messages
* ``tools`` — tool declarations
* ``think`` — boolean or ``"low"|"medium"|"high"|"max"`` (thinking toggle)
* ``format`` — ``"json"`` or a JSON schema (structured outputs)
* ``options`` — model-parameter dict (see ``_CANONICAL_OPTIONS`` /
  ``_NON_CANONICAL_OPTIONS`` below)
* ``stream`` — boolean
* ``keep_alive`` — duration string

Messages carry ``role`` (``system|user|assistant|tool``), ``content``,
``thinking`` (response-side thinking text), ``images`` (multimodal
base64 bytes), ``tool_calls`` (assistant-side), and ``tool_name``
(tool-result pairing key — no id on the wire).

**Tool calls carry no id.** Unlike Gemini (which KBR-36 proved *does*
carry an id, growing ``ToolResult.tool_use_id``), Ollama publishes
``tool_calls[*].function = {name, arguments}`` with no ``id`` on either
side — verified against ``docs/api.md``'s streaming-with-tools example
(line 624-643). ``ToolUse.id`` stays ``None``; ``role: "tool"``
``tool_name`` is consumed (not residualised) so the residual stays empty
and the oracle's name-and-position pairing rule reads the name from
``Request.source`` — the same convention ``reader_gemini.py:1473``
establishes for ``functionResponse.name``.

**Tool-call arguments are objects, not strings.** Ollama publishes
``tool_calls[*].function.arguments`` as a JSON object (e.g.
``{"city": "Tokyo"}``); Chat Completions and Responses use the JSON
*string* form, which ``contract.decode_arguments`` (KBR-174) decodes.
Ollama is not a caller of ``decode_arguments`` — the object form has
no parse step, and ``decode_arguments``'s docstring
(``contract.py:900-980``) names Gemini explicitly excluded for the same
reason (``reader_gemini.py:1386-1388``).

**Totality is the load-bearing property.** Every key of the body, at
every depth, is either mapped and named in
:attr:`~harness.contract.Request.consumed`, or placed in
:attr:`~harness.contract.Request.residual` under its path from the body
root (per ``contract.residual_key()``, KBR-193). Nothing is dropped
silently — an unaccounted field is precisely where an unregistered
mutation hides.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from typing import Any

from harness import contract as c

# --------------------------------------------------------------------------
# The recognised key sets
# --------------------------------------------------------------------------

#: Top-level declared control fields that map to ``envelope.extra`` under
#: their own wire key per §3.3.1b. Sourced from ``docs/api.md`` parameters
#: block (line 494-513):
#:
#: * ``format`` — structured-outputs config (json or JSON schema).
#: * ``think`` — request-level thinking toggle; a fourth spelling of the
#:   same concept Anthropic carries as ``thinking``, Gemini as
#:   ``thinkingConfig`` and Responses as ``reasoning``.
#: * ``keep_alive`` — server-side model lifetime.
#:
#: ``raw`` is **deliberately not** here: it is published only for
#: ``/api/generate`` (the "Raw Mode" example at ``docs/api.md`` line 329),
#: not for ``/api/chat``. A ``/api/chat`` body carrying ``raw`` is an
#: unrecognised wire key and residualises.
_PUBLISHED_EXTRA_KEYS: frozenset[str] = frozenset({"format", "think", "keep_alive"})

#: ``options.*`` keys that map onto the canonical Chat Completions sampling
#: spelling (§3.3.1b, the closed fifteen). Eight keys here — seven
#: identities (``temperature``, ``top_p``, ``top_k``, ``seed``, ``stop``,
#: ``frequency_penalty``, ``presence_penalty``) and one rename
#: (``num_predict`` → ``max_tokens``). ``stop`` is a list on the Ollama
#: wire (e.g. ``["\\n", "user:"]``) but the canonical ``Sampling.stop`` is
#: a list too, so the identity holds whole.
#:
#: ``num_predict → max_tokens`` is the natural semantic identity: both bound
#: generated tokens. Anthropic's ``max_tokens`` is the same concept; Gemini
#: has ``maxOutputTokens``; Responses has ``max_output_tokens`` /
#: ``max_completion_tokens``. ``max_completion_tokens`` is a separate
#: canonical key (§3.3.1b second bullet) and not a synonym for
#: ``max_tokens`` — Ollama's ``num_predict`` is the upper-bound semantic,
#: not the completion-token semantic, so it lands on ``max_tokens``.
_CANONICAL_OPTIONS: Mapping[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "seed": "seed",
    "stop": "stop",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "num_predict": "max_tokens",
}

#: ``options.*`` keys that are recognised as declared control fields of
#: the Ollama wire format (§3.3.1b's "recognised as a declared control
#: field" rule) but are not on the closed sampling set. They land at
#: ``envelope.extra[<bare key>]`` under §7.4.2 rule 2 — the
#: ``options`` container is a wire detail, not a projection concept.
#:
#: Sourced from the ``docs/api.md`` "Generate request (With options)"
#: example (line 388-410), which the parameters block calls "every
#: available option" as of 2026-09-15. ``temperature``, ``top_p``,
#: ``top_k``, ``seed``, ``stop``, ``frequency_penalty``,
#: ``presence_penalty`` are removed from this set because
#: :data:`_CANONICAL_OPTIONS` already names them.
_NON_CANONICAL_OPTIONS: frozenset[str] = frozenset(
    {
        "num_ctx",
        "repeat_penalty",
        "repeat_last_n",
        "min_p",
        "num_keep",
        "draft_num_predict",
        "penalize_newline",
        "numa",
        "num_batch",
        "num_gpu",
        "main_gpu",
        "use_mmap",
        "num_thread",
    }
)

#: Roles that lift into ``conversation.system`` rather than becoming a
#: turn (§3.3.1b second bullet). Ollama's published role enum is
#: ``system | user | assistant | tool`` (``docs/api.md`` line 501).
_SYSTEM_ROLES: frozenset[str] = frozenset({"system"})

#: Roles that may carry ``tool_calls`` (§3.3.1b's merge rule, KBR-25 R7.1:
#: the canonical ``assistant(tool_calls) → tool → tool`` exchange). Only
#: ``assistant`` on Ollama — the wire spec places ``tool_calls`` on
#: assistant messages.
_ASSISTANT_ROLES: frozenset[str] = frozenset({"assistant"})

#: Roles that may carry ``tool_name`` (§3.3.1b: a tool result is a
#: ``ToolResult`` part inside a user turn, identified by tool name).
_TOOL_RESULT_ROLES: frozenset[str] = frozenset({"tool"})

# --------------------------------------------------------------------------
# The projection
# --------------------------------------------------------------------------

#: Roles recognised by this reader. Anything else raises
#: :class:`~harness.contract.UnreadableBodyError`. ``function`` (the
#: deprecated Chat Completions spelling) is deliberately **not** in this
#: set: Ollama never published it, and a body carrying it would silently
#: fall into a different reader's hands downstream.
_RECOGNISED_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})


class OllamaChatProjection:
    """Reads an Ollama ``/api/chat`` request into the wire-independent form.

    Implements :class:`~harness.contract.Projection` for
    :attr:`~harness.contract.WireFormat.OLLAMA_CHAT`. The reader is stateless;
    every method takes what it needs and returns what it produced, so one
    instance is safe to share across a whole corpus run.
    """

    wire_format = c.WireFormat.OLLAMA_CHAT

    def read_request(self, captured: c.CapturedRequest) -> c.Request:
        """Project a captured Ollama chat request.

        Args:
            captured: The request as observed on the wire. Only the body is
                consulted — Ollama carries the model in the body
                (``docs/api.md`` line 494), not in the URL path.

        Returns:
            The wire-independent projection, total over the body.

        Raises:
            UnreadableBodyError: When the body is not JSON, not an object,
                carries a role no Ollama message defines, carries a message
                with no ``role``, or carries a content block of a shape no
                published example exercises. ``ValueError`` is deliberately
                **not** caught: from inside a reader it means the reader
                mis-routed a field, which is a reader bug and must surface.
        """
        body = _parse_body(captured.body)

        try:
            return _project(body)
        except (KeyError, TypeError, AttributeError) as exc:
            raise c.UnreadableBodyError(f"unreadable Ollama chat body: {exc!r}") from exc


# --------------------------------------------------------------------------
# Body parsing
# --------------------------------------------------------------------------


def _parse_body(raw: bytes) -> Mapping[str, Any]:
    """Decode the request body into a JSON object.

    Args:
        raw: The raw body bytes.

    Returns:
        The parsed body.

    Raises:
        UnreadableBodyError: When the bytes are not JSON, or are JSON that
            is not an object. A JSON array is valid JSON and an invalid
            request.
    """
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise c.UnreadableBodyError(f"body is not valid JSON: {exc}") from exc

    if not isinstance(body, dict):
        raise c.UnreadableBodyError(f"body must be a JSON object, got {type(body).__name__}")

    return body


# --------------------------------------------------------------------------
# Top-level projection
# --------------------------------------------------------------------------


def _project(body: Mapping[str, Any]) -> c.Request:
    """Classify every top-level key into the envelope, the conversation, or the residual.

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

    for key, value in body.items():
        if key in ("model", "stream", "messages", "tools"):
            consumed.add(key)
        elif key in _PUBLISHED_EXTRA_KEYS:
            extra[key] = value
            consumed.add(key)
        elif key == "options":
            consumed.add(key)
            _project_options(value, sampling, extra, residual)
        else:
            residual[key] = value

    messages = body.get("messages")
    system = _read_system_messages(messages, residual)
    turns = _read_messages(messages, residual)
    tools = _read_tools(body.get("tools"), residual)

    conversation = c.Conversation(
        system=system,
        turns=turns,
        tools=tools,
        sampling=sampling,
    )

    envelope = c.Envelope(
        model=_typed_leaf(body, "model", (str,), "", residual),
        stream=_typed_leaf(body, "stream", (bool,), "", residual),
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


# --------------------------------------------------------------------------
# Options flattening
# --------------------------------------------------------------------------


def _project_options(
    value: Any,
    sampling: dict[str, Any],
    extra: dict[str, Any],
    residual: dict[str, Any],
) -> None:
    """Flatten an ``options`` dict onto the canonical sampling set or ``envelope.extra``.

    Per §7.4.2 rule 2 the ``options`` container is a wire detail — every
    extra key is the bare leaf published name. An unknown ``options.*``
    key residualises at ``options.<key>`` (the container-prefixed wire
    path per ``contract.residual_key()``, KBR-193) — fail-closed on a
    vendor revision.

    Args:
        value: The body-level ``options`` field.
        sampling: The canonical-sampling dict, extended in place.
        extra: The ``envelope.extra`` mapping, extended in place.
        residual: The residual mapping, extended in place.

    Raises:
        UnreadableBodyError: When ``options`` is not a JSON object.
            ``options`` is an envelope-level container (§7.4.2 rule 7
            table, last row): a wrong-typed envelope container
            residualises whole rather than raising, because every
            envelope field has an absent value and the rest of the
            request still projects.
    """
    if value is None:
        return
    if not isinstance(value, Mapping):
        # §7.4.2 rule 7 last row: envelope-level non-object container
        # residualises whole (the rest of the request still projects).
        residual["options"] = value
        return

    for key, sub_value in value.items():
        if key in _CANONICAL_OPTIONS:
            target = _CANONICAL_OPTIONS[key]
            # Preserve type: the canonical sampling field is the wire
            # value whole (a list for ``stop``, a number for the rest,
            # etc.). The reader does not validate the inner type — that
            # is the corpus and the oracle's job, not this reader's.
            sampling[target] = sub_value
        elif key in _NON_CANONICAL_OPTIONS:
            extra[key] = sub_value
        else:
            # Unknown ``options.*`` — fail-closed on a vendor revision.
            # Container-prefixed wire path: ``options.foo`` is the
            # indexed nested-key form (§7.4.1 rule 2 second bullet).
            residual[c.residual_key("options", key)] = sub_value


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------


def _read_system_messages(value: Any, residual: dict[str, Any]) -> tuple[c.Text, ...]:
    """Lift ``role: "system"`` messages into ordered text parts.

    Runs as a second pass over ``messages`` so a role that is
    system-shaped in the turns pass is a system here too — lifting
    on the same pass would require carrying state across the message
    loop, and the merge rule §3.3.1b applies to *non*-system messages
    only.

    Args:
        value: The body-level ``messages`` field (its raw form — the
            full reader is independent of the ``messages`` shape).
        residual: The residual mapping.

    Returns:
        The system parts, in wire order.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        # ``messages`` itself residualises whole in the turns pass; this
        # function sees the validated list. Defensive only.
        return ()

    parts: list[c.Text] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            continue
        if message.get("role") != "system":
            continue
        path = f"messages[{index}]"
        mapped: set[str] = {"role"}
        content = message.get("content")
        mapped.add("content")
        parts.extend(_read_text_content(content, f"{path}.content", residual))
        # Fail closed at depth (§7.4.1): an unknown key on a system
        # message residualises at its indexed path — the same rule the
        # other roles follow in ``_read_message_parts``.
        for key in message:
            if key in mapped:
                continue
            residual[c.residual_key(path, key)] = message[key]
    return tuple(parts)


def _read_messages(value: Any, residual: dict[str, Any]) -> tuple[c.Turn, ...]:
    """Project ``messages`` into ``Turn``s, applying §3.3.1b's merge rule.

    The four clauses in order (§3.3.1b merge rule):

    1. A ``ToolResult`` part comes first within the turn it lands in.
    2. A maximal run of consecutive ``ToolResult`` parts (plus any
       immediately-following non-tool user message) forms one turn.
    3. Consecutive same-role turns merge.
    4. A ``role: "system"`` message is lifted into
       ``conversation.system`` (handled in
       :func:`_read_system_messages`).

    Args:
        value: The body-level ``messages`` field.
        residual: The residual mapping.

    Returns:
        The conversation turns.

    Raises:
        UnreadableBodyError: When a message carries an unrecognised
            role, no ``role`` at all, or a wrong-typed content block.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        residual["messages"] = value
        return ()

    # First, project every message to a (role, parts) pair. System
    # messages are dropped here — they are lifted in
    # ``_read_system_messages``. Tool-result messages become ToolResult
    # parts but stay attached to their raw-message index for the merge
    # step.
    projected: list[tuple[str, tuple[c.Part, ...], str | None]] = []
    # tuple = (role, parts, raw_message_path_for_residual)
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            residual[c.residual_key("messages", str(index))] = message
            continue
        path = f"messages[{index}]"
        role = message.get("role")
        if role is None:
            raise c.UnreadableBodyError(f"{path} missing required field 'role'")
        if role not in _RECOGNISED_ROLES:
            raise c.UnreadableBodyError(f"{path} role {role!r} is not a published Ollama role")
        if role == "system":
            # Lifted elsewhere; skip.
            continue
        parts = _read_message_parts(message, role, path, residual)
        projected.append((role, parts, path))

    return _merge_turns(projected, residual)


def _read_message_parts(
    message: Mapping[str, Any],
    role: str,
    path: str,
    residual: dict[str, Any],
) -> tuple[c.Part, ...]:
    """Project one message into its ordered parts.

    §7.4.1 fail-closed at every depth: every message key is either
    mapped into a part, consumed (e.g. ``tool_name`` on a tool result),
    or residualised. Unknown message keys land in the residual at
    their indexed path.

    Args:
        message: The message object.
        role: The message's role.
        path: The message's path from the body root.
        residual: The residual mapping.

    Returns:
        The parts.

    Raises:
        UnreadableBodyError: When the message's content is a shape no
            published Ollama example exercises.
    """
    if role == "tool":
        return _read_tool_result_parts(message, path, residual)

    # Known message-level keys, per role. Everything else residualises.
    mapped: set[str] = set()

    content = message.get("content")
    mapped.add("content")

    text_parts: tuple[c.Text, ...]
    if role == "assistant":
        # Convergence on the canonical exchange: ``content == ""`` with
        # ``tool_calls`` present is consumed without a Text part — the
        # published "no text alongside tool_calls" marker, mirroring
        # CC's ``content: null → ()`` at ``reader_chat_completions.py:
        # 665-666``. ``content == ""`` without ``tool_calls`` (or on
        # any user message) projects ``Text("")`` per §3.3.1's
        # "empty block is a part" rule. ``content: null`` projects no
        # part regardless.
        if content is None:
            text_parts = ()
        elif isinstance(content, str):
            text_parts = (c.Text(content),) if content != "" or "tool_calls" not in message else ()
        elif isinstance(content, list):
            # Ollama published examples carry only string content;
            # list content is unmodelled and raises per §7.4 rule 7
            # row 1 (the value that IS the part).
            raise c.UnreadableBodyError(
                f"{path} content must be a string or null, got list "
                "(no published Ollama chat example carries list content)"
            )
        else:
            raise c.UnreadableBodyError(f"{path} content must be a string or null, got {type(content).__name__}")

        thinking_parts: tuple[c.Thinking, ...] = ()
        if "thinking" in message:
            mapped.add("thinking")
            thinking_value = message["thinking"]
            if isinstance(thinking_value, str):
                # §3.3.1's "empty block is a part": an empty Thinking
                # is still a Thinking, never nothing — P5e injects one
                # and P8's trigger is conditional and inferred.
                thinking_parts = (c.Thinking(text=thinking_value),)
            else:
                residual[c.residual_key(path, "thinking")] = thinking_value

        if "tool_calls" in message:
            mapped.add("tool_calls")
            tool_use_parts = _read_tool_calls(message["tool_calls"], path, residual)
        else:
            tool_use_parts = ()

        if "images" in message:
            mapped.add("images")
            image_parts = _read_images(message["images"], path, residual)
        else:
            image_parts = ()

        # Indexed-path convergence: order is Text → Thinking →
        # ToolUse × N → Image × N. A different ordering reports deltas
        # on every part and every turn after.
        result: tuple[c.Part, ...] = text_parts + thinking_parts + tool_use_parts + image_parts
    else:
        # user message: content + images.
        text_parts = _read_text_content(content, f"{path}.content", residual)
        if "images" in message:
            mapped.add("images")
            image_parts = _read_images(message["images"], path, residual)
        else:
            image_parts = ()
        result = text_parts + image_parts

    # Fail closed: any message-level key the reader did not map is
    # residualised at its indexed path.
    for key in message:
        if key in mapped:
            continue
        if key == "role":
            # The role is part of the projection's turn accounting,
            # not a residual entry.
            continue
        residual[c.residual_key(path, key)] = message[key]

    return result


def _read_images(
    value: Any,
    path: str,
    residual: dict[str, Any],
) -> tuple[c.Image, ...]:
    """Project a message's ``images`` list into ``Image`` parts.

    Each entry is a base64-encoded PNG/JPEG byte string (no media
    type on the wire — ``docs/api.md`` line 504). The canonical case
    decodes the base64 first and digests the decoded bytes; an
    undecodable entry residualises the leaf at its indexed path and
    digests the raw encoded bytes (the second of
    ``image_digest``'s recipes, §7.4 rule 7 row 3, KBR-192).

    Args:
        value: The ``images`` field value.
        path: The message's path from the body root.
        residual: The residual mapping.

    Returns:
        The image parts, in wire order.

    Raises:
        UnreadableBodyError: When ``images`` is not a list, when an
            entry is not a string, or when a non-base64 string entry
            does not surface as ``binascii.Error`` / ``ValueError`` —
            a wrong-typed image leaves a typed error in the residual
            rather than raising (rule 7 table, row 3).
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        # An envelope-adjacent list container: residualise whole (rule 7
        # last row for envelope-adjacent containers).
        residual[c.residual_key(path, "images")] = value
        return ()

    parts: list[c.Image] = []
    for index, entry in enumerate(value):
        entry_path = f"{path}.images[{index}]"
        if not isinstance(entry, str):
            residual[entry_path] = entry
            continue
        try:
            decoded = base64.b64decode(entry, validate=True)
        except (binascii.Error, ValueError):
            # Second recipe — digest the raw encoded bytes so the part
            # keeps a distinguishing identity, residualise the leaf.
            parts.append(c.Image(digest=c.image_digest(entry.encode("utf-8"))))
            residual[entry_path] = entry
            continue
        # Canonical case — digest the decoded bytes.
        parts.append(c.Image(digest=c.image_digest(decoded)))
    return tuple(parts)


def _read_text_content(
    value: Any,
    path: str,
    residual: dict[str, Any],
) -> tuple[c.Text, ...]:
    """Project a message content field that is a string or null.

    Args:
        value: The content field.
        path: The content's path from the body root.
        residual: The residual mapping.

    Returns:
        The text parts (``Text(value)`` for a non-empty string; ``()``
            for an empty string or ``None``).

    Raises:
        UnreadableBodyError: When the content is a shape no published
            example exercises (a list, a number, a bool).
    """
    if value is None:
        return ()
    if isinstance(value, str):
        if value == "":
            # §3.3.1: "empty block is a part" — but the assistant
            # empty-content + tool_calls convergence rule consumes this
            # before reaching here; a user message with ``content == ""``
            # projects ``Text("")``. Caller-side handling for the
            # tool_calls case is in ``_read_message_parts``.
            return (c.Text(""),)
        return (c.Text(value),)
    if isinstance(value, list):
        raise c.UnreadableBodyError(
            f"{path} content must be a string or null, got list (no published Ollama chat example carries list content)"
        )
    raise c.UnreadableBodyError(f"{path} content must be a string or null, got {type(value).__name__}")


def _read_tool_result_parts(
    message: Mapping[str, Any],
    path: str,
    residual: dict[str, Any],
) -> tuple[c.Part, ...]:
    """Project a ``role: "tool"`` message into a ``ToolResult`` part.

    ``tool_name`` is consumed (added to the mapped set, not
    residualised) so the residual stays empty — mirrors the Gemini
    ``functionResponse.name`` convention at
    ``reader_gemini.py:1473``. The oracle's name-and-position pairing
    rule reads the name from ``Request.source`` (the wire body is
    preserved whole on ``Request.source``).

    ``is_error`` is always ``False``: Ollama publishes no error flag
    on a tool result, and inventing one would make an absent flag and
    a reported success indistinguishable.

    Args:
        message: The tool-result message.
        path: The message's path from the body root.
        residual: The residual mapping.

    Returns:
        A single ``ToolResult`` part (followed by any image parts).
    """
    mapped: set[str] = set()

    content = message.get("content")
    mapped.add("content")
    content_parts = _read_text_content(content, f"{path}.content", residual)

    if "tool_name" in message:
        mapped.add("tool_name")

    if "images" in message:
        mapped.add("images")
        image_parts = _read_images(message["images"], path, residual)
    else:
        image_parts = ()

    # Fail closed: any message-level key the reader did not map is
    # residualised at its indexed path. ``role`` is turn accounting,
    # not a residual entry.
    for key in message:
        if key in mapped or key == "role":
            continue
        residual[c.residual_key(path, key)] = message[key]

    return (
        c.ToolResult(
            content=content_parts,
            tool_use_id=None,
            is_error=False,
        ),
        *image_parts,
    )


def _read_tool_calls(
    value: Any,
    path: str,
    residual: dict[str, Any],
) -> tuple[c.ToolUse, ...]:
    """Project a ``tool_calls`` array into ``ToolUse`` parts.

    Each entry's ``arguments`` is decoded with the object-form rule
    (no JSON parse; mirror of ``contract.decode_arguments``'s absent
    branch at ``contract.py:953-954`` and Gemini's
    ``_typed_leaf(..., default={})`` at ``reader_gemini.py:1411``):

    * absent / ``null`` → ``ToolUse(arguments={})``, no residual;
    * non-object (str / number / list / bool) → residualise the raw
      wire value at ``<path>.tool_calls[<j>].function.arguments`` and
      project ``ToolUse(arguments={})``.

    A top-level ``id`` on the wire residualises at the same path —
    Ollama publishes no id, and a future schema revision adding one
    would otherwise change the projection silently.

    Args:
        value: The ``tool_calls`` field value.
        path: The message's path from the body root.
        residual: The residual mapping.

    Returns:
        The tool-use parts, one per ``tool_calls`` entry, in wire order.

    Raises:
        UnreadableBodyError: When ``tool_calls`` is not a list, when an
            entry is not an object, or when ``tool_calls[*].function``
            is missing / not an object. (A non-object ``function``
            container raises per §7.4.2 rule 7 table last row: a member
            that is the schema-declared object cannot be a scalar.)
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise c.UnreadableBodyError(f"{path}.tool_calls must be a list, got {type(value).__name__}")

    parts: list[c.ToolUse] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            residual[f"{path}.tool_calls[{index}]"] = entry
            continue
        function = entry.get("function")
        fn_path = f"{path}.tool_calls[{index}].function"
        if not isinstance(function, Mapping):
            # Absent, null, or a scalar: raise. §7.4.2 rule 7 row 4 —
            # the schema declares an object, the position is
            # index-addressed, and "the position cannot be vacated":
            # a continue here would shift every later part's index and
            # invent deltas on content nobody touched.
            raise c.UnreadableBodyError(f"{fn_path} must be an object, got {type(function).__name__}")
        name = _typed_leaf(function, "name", (str,), fn_path, residual) or ""
        raw_args = function.get("arguments")
        arguments = _decode_arguments_object(raw_args, fn_path, residual)

        # Account for ``id`` if the wire ever sends one. Ollama
        # publishes no id; ``tool_name`` (the tool-result pairing key)
        # is the orthogonal concept on the result side and is
        # consumed in ``_read_tool_result_parts``.
        if "id" in entry:
            raw_id = entry["id"]
            if raw_id is not None:
                residual[c.residual_key(f"{path}.tool_calls[{index}]", "id")] = raw_id

        # Fail closed: every entry key other than ``function`` (mapped)
        # and ``id`` (handled above) residualises at its indexed path.
        for key in entry:
            if key not in ("function", "id"):
                residual[c.residual_key(f"{path}.tool_calls[{index}]", key)] = entry[key]

        parts.append(c.ToolUse(name=name, id=None, arguments=arguments))

    return tuple(parts)


def _decode_arguments_object(
    raw: Any,
    path: str,
    residual: dict[str, Any],
) -> Mapping[str, Any]:
    """Decode the object-form ``arguments`` field with the same fail-closed policy.

    Mirrors :func:`harness.contract.decode_arguments` and Gemini's
    ``_typed_leaf(..., default={})`` pattern. A non-object value
    residualises the **raw wire value** (per ``contract.residual_key()``)
    so a residual-key set diff across the six readers does not store
    two different renderings of one unreadable value
    (``contract.py:912-913``). Absent / ``null`` projects ``{}`` with
    no residual — the absent branch. The residual is a parameter, not
    a return flag, for the reason ``decode_arguments``'s docstring
    records: the fail-closed step cannot be skipped by the caller.

    Args:
        raw: The wire value of ``function.arguments``.
        path: The arguments' path from the body root, used as the
            residual key when the value residualises.
        residual: The residual mapping, extended in place.

    Returns:
        The decoded arguments mapping (``{}`` unless the wire carried
        a JSON object).
    """
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return raw
    # Non-object: residualise the raw value whole.
    residual[c.residual_key(path, "arguments")] = raw
    return {}


def _merge_turns(
    projected: Sequence[tuple[str, tuple[c.Part, ...], str]],
    residual: dict[str, Any],
) -> tuple[c.Turn, ...]:
    """Apply §3.3.1b's merge rule to a list of ``(role, parts, path)`` tuples.

    The clauses in order:

    1. A maximal run of consecutive tool-result parts forms one turn.
       An immediately-following non-tool user message merges into the
       run.
    2. ``ToolResult`` parts come first within the turn the run + the
       following-message clause build.
    3. Consecutive same-role turns merge.

    Args:
        projected: The projected messages (system messages already
            filtered out), in wire order, with their paths.
        residual: The residual mapping.

    Returns:
        The merged turns.
    """
    if not projected:
        return ()

    # Group consecutive same-role messages; for a run of tool-result
    # messages, fold in an immediately-following user message.
    raw_runs: list[list[tuple[str, tuple[c.Part, ...], str]]] = []
    current_role: str | None = None
    current_run: list[tuple[str, tuple[c.Part, ...], str]] = []

    for role, parts, path in projected:
        # All ``role: "tool"`` messages project to a single
        # ``ToolResult`` part each; treat them uniformly with the
        # tool-result rule.
        if role == "tool":
            # ``ToolResult`` runs form their own turn; absorb a
            # following non-tool user message.
            if current_run and current_role == "tool":
                current_run.append((role, parts, path))
                continue
            if current_run:
                raw_runs.append(current_run)
            current_run = [(role, parts, path)]
            current_role = "tool"
            continue

        if role == "user" and current_role == "tool":
            # Following-message merge: a user message that immediately
            # follows a tool-result run merges into it.
            current_run.append((role, parts, path))
            continue

        # Other roles (assistant, or a user message that did not
        # follow a tool run): same-role merge applies.
        if current_run and current_role == role:
            current_run.append((role, parts, path))
            continue

        if current_run:
            raw_runs.append(current_run)
        current_run = [(role, parts, path)]
        current_role = role

    if current_run:
        raw_runs.append(current_run)

    # Build ``Turn``s from each run.
    turns: list[c.Turn] = []
    for run in raw_runs:
        # §3.3.1b: a tool result is a ``ToolResult`` part inside a
        # ``user`` turn, by the merge rule. A run that contains any
        # tool-role message therefore projects as a ``user`` turn —
        # the ``Turn`` grammar rejects ``"tool"``. Tool-result parts
        # come first (already true: the loop above appends in run
        # order, and the run was built tool-first-then-following-user).
        role = "user" if any(r == "tool" for r, _, _ in run) else run[0][0]
        ordered_parts: list[c.Part] = []
        for _, parts, _ in run:
            ordered_parts.extend(parts)
        turns.append(c.Turn(role=role, parts=tuple(ordered_parts)))

    return tuple(turns)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


def _read_tools(
    value: Any,
    residual: dict[str, Any],
) -> tuple[c.ToolDecl, ...]:
    """Project the body-level ``tools`` array into ``ToolDecl``s.

    Each ``tools[*]`` is ``{"type": "function", "function": {name,
    description, parameters}}`` per ``docs/api.md`` line 597-614. The
    nested ``function`` object carries the schema; ``type: "function"``
    is the only published variant today. ``strict`` is not on the
    wire — ``ToolDecl.strict`` stays ``None``.

    A tool whose ``type`` is present and is not ``"function"`` is a
    declaration the product cannot faithfully forward (the Chat
    Completions reader's answer at ``reader_chat_completions.py:
    920-967``): the whole entry residualises at ``tools[<i>]``. The
    same fate meets an entry whose ``function`` is absent, ``null``
    or not an object — no declaration can be built from it, and
    register paths address tools by name, so vacating the position
    shifts no indexed path (unlike ``tool_calls``, whose parts are
    index-addressed and therefore raise instead).

    Fail closed at every depth (§7.4.1): every key on the entry and
    on ``function`` is either mapped or residualised at its indexed
    path — the same discipline ``_read_message_parts`` applies.

    Args:
        value: The body-level ``tools`` field.
        residual: The residual mapping.

    Returns:
        The tool declarations, in wire order.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        residual["tools"] = value
        return ()

    decls: list[c.ToolDecl] = []
    for index, entry in enumerate(value):
        path = f"tools[{index}]"
        if not isinstance(entry, Mapping):
            residual[c.residual_key("tools", str(index))] = entry
            continue
        # A non-function tool is not this reader's to translate: the
        # whole entry residualises, which fails the run and names the
        # entry, rather than silently translating a declaration the
        # wire says is something else.
        tool_type = entry.get("type")
        if tool_type is not None and tool_type != "function":
            residual[path] = entry
            continue
        function = entry.get("function")
        if not isinstance(function, Mapping):
            # Absent, null, or a scalar: no declaration can be built.
            # Residualise whole and vacate the slot — tools are
            # name-addressed (§3.3.1a), so no indexed path shifts.
            residual[path] = entry
            continue
        name = _typed_leaf(function, "name", (str,), f"{path}.function", residual) or ""
        description = _typed_leaf(function, "description", (str,), f"{path}.function", residual)
        schema = function.get("parameters")
        if schema is not None and not isinstance(schema, Mapping):
            residual[c.residual_key(f"{path}.function", "parameters")] = schema
            schema = None
        # Fail closed: every other key on the entry and on `function`
        # residualises at its indexed path.
        for key in entry:
            if key not in ("type", "function"):
                residual[c.residual_key(path, key)] = entry[key]
        for key in function:
            if key not in ("name", "description", "parameters"):
                residual[c.residual_key(f"{path}.function", key)] = function[key]
        decls.append(
            c.ToolDecl(
                name=name,
                description=description,
                schema=schema,
                strict=None,
            )
        )
    return tuple(decls)


# --------------------------------------------------------------------------
# Leaf readers
# --------------------------------------------------------------------------


def _typed_leaf(
    body: Mapping[str, Any],
    key: str,
    types: tuple[type, ...],
    path_prefix: str,
    residual: dict[str, Any],
) -> Any:
    """Read a typed leaf from a body, residualising on a wrong type.

    Args:
        body: The mapping to read from.
        key: The leaf's key in ``body``.
        types: The acceptable Python types for the value.
        path_prefix: The path of ``body`` from the root, used to
            build the residual key when the leaf is wrong-typed.
        residual: The residual mapping.

    Returns:
        The leaf's value, or ``None`` when the leaf is absent or
        wrong-typed (after residualising the wrong-typed case).
    """
    if key not in body:
        return None
    value = body[key]
    if value is None:
        return None
    if not isinstance(value, types):
        residual[c.residual_key(path_prefix, key)] = value
        return None
    return value
