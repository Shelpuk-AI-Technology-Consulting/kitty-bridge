"""GeminiTranslator — translates between Gemini generateContent API and Chat Completions.

The Gemini API uses:
- Request: ``POST /v1beta/models/{model}:generateContent`` with body ``{contents, tools, ...}``
- Response: ``{candidates: [{content: {parts: [...]}}, ...], usageMetadata: {...}}``
- Streaming: SSE ``data: {json}\\n\\n`` (no event-type prefix)

This translator converts to/from Chat Completions format for upstream providers.
"""

from __future__ import annotations

import json
import uuid

from kitty.bridge.engine import ToolCallBuffer, ToolCallBufferError
from kitty.bridge.gemini.events import format_gemini_sse

__all__ = ["GeminiTranslator", "carry_gemini_tool_choice"]

#: Gemini ``functionCallingConfig.mode`` values that map onto a Chat Completions
#: string.  ``VALIDATED`` and ``MODE_UNSPECIFIED`` have no canonical form and
#: are omitted (KBR-221 D5) -- the harness reader residualises them, so neither
#: side writes the entry.
_GEMINI_MODE_TO_CC: dict[str, str] = {"AUTO": "auto", "ANY": "required", "NONE": "none"}


def carry_gemini_tool_choice(gemini_request: dict, cc_request: dict) -> None:
    """Carry a Gemini ``toolConfig.functionCallingConfig`` onto a Chat Completions body.

    Called from :meth:`GeminiTranslator.translate_request`, the way KBR-214's
    :func:`kitty.bridge.messages.translator.carry_tool_choice_and_metadata` is
    called from the Messages converter.  ``functionCallingConfig.mode`` is a
    constraint, not a hint: dropping ``ANY`` lets the model answer in prose
    where the agent demanded a function call, and dropping ``NONE`` lets it
    call a function the agent forbade (KBR-221).

    The published ``mode`` enum is ``AUTO``/``ANY``/``NONE`` and is matched
    case-insensitively (Google's own examples send lower case).  ``ANY`` beside
    exactly one ``allowedFunctionNames`` entry maps onto the Chat Completions
    named-function form.  The omissions mirror the KBR-214 decisions:

    * **No tools, no choice** (D9).  A choice beside no tools is rejected by
      Chat Completions backends ("'tool_choice' is only allowed when 'tools'
      are specified").  The gate reads the *Chat Completions* tool list the
      body will ship, not the inbound Gemini ``tools`` key.
    * **Restrictions with no Chat Completions form ride the mode only**
      (D5).  Multi-name ``ANY``, ``AUTO`` or ``NONE`` beside any names, an
      empty ``allowedFunctionNames``, a non-list value, and a names list with
      non-string members all carry the mode; the name restriction itself has
      no CC home on any destination wire.  Recorded as prose under
      ``TEST_SUITE.md §3.3.2`` (KBR-139); a register row becomes due with the
      first corpus entry that carries one of these shapes.
    * **Modes with no canonical mapping are omitted** (D5).  ``VALIDATED`` and
      ``MODE_UNSPECIFIED`` have no Chat Completions reading, and the harness
      reader residualises them, so neither side writes the entry and no
      delta is manufactured.

    Args:
        gemini_request: The inbound Gemini ``generateContent`` body.  Not
            modified.
        cc_request: The Chat Completions body being built, mutated in place.

    Returns:
        None.  ``cc_request`` gains ``tool_choice`` only where the inbound
        body carries a mode with a canonical mapping.
    """
    # D9: the gate reads the CC list the body will ship.  The Gemini
    # translator writes ``tools`` only when non-empty, so an absent key means
    # no tools at all and the choice must not ride along.
    if not cc_request.get("tools"):
        return

    tool_config = gemini_request.get("toolConfig")
    if not isinstance(tool_config, dict):
        return
    config = tool_config.get("functionCallingConfig")
    if not isinstance(config, dict):
        return

    # The published enumeration is upper case; CaseInSensitiveEnum accepts
    # lower case on the wire (Google's own examples send ``"auto"``).
    mode = config.get("mode")
    choice = _GEMINI_MODE_TO_CC.get(mode.upper()) if isinstance(mode, str) else None
    if choice is None:
        return

    names = config.get("allowedFunctionNames")
    # The CC named-function form requires a single string name.  A multi-name
    # set, AUTO/NONE + names, an empty list, a non-list value, and a list with
    # a non-string member all share the same disposition: the mode still
    # projects; the restriction has no canonical home.  Carry the mode.
    if choice == "required" and isinstance(names, list) and len(names) == 1 and isinstance(names[0], str):
        cc_request["tool_choice"] = {"type": "function", "function": {"name": names[0]}}
    else:
        cc_request["tool_choice"] = choice

# ── Finish-reason mappings ───────────────────────────────────────────────────

_CC_TO_GEMINI_FINISH: dict[str | None, str] = {
    "stop": "STOP",
    "tool_calls": "STOP",
    # KBR-285: the legacy single-function wire finishes with
    # ``function_call``; Gemini has no tool-call finish reason (v1beta ends
    # tool turns on STOP), so the default mapping already lands right.
    "function_call": "STOP",
    "length": "MAX_TOKENS",
    "content_filter": "SAFETY",
    None: "STOP",
}


def _extract_text_parts(content: list) -> str:
    """Extract a joined text string from a multimodal parts list.

    KBR-285: newer multimodal streaming on OpenAI-shaped backends carries
    ``content`` as a list of content parts. Only text parts carry a string
    the Gemini wire can hold; image parts have no output equivalent and are
    dropped (the raw-CC route delivers them verbatim).

    Args:
        content: The parts list from ``delta.content`` / ``message.content``.

    Returns:
        The joined text of the list's text parts (empty string when the
        list carries no text part).
    """
    text_parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text_val = part.get("text")
            if isinstance(text_val, str) and text_val.strip():
                text_parts.append(text_val)
    return "\n".join(text_parts)

# ── Gemini role → Chat Completions role ──────────────────────────────────────

_ROLE_MAP: dict[str, str] = {
    "user": "user",
    "model": "assistant",
    "function": "tool",
}


class GeminiTranslator:
    """Translates between Gemini generateContent API and Chat Completions format."""

    def __init__(self) -> None:
        self._tool_call_buffers: dict[int, ToolCallBuffer] = {}
        self._tool_call_meta: dict[int, dict] = {}  # {index: {id, name}}
        self._last_was_empty: bool = False
        self._saw_content: bool = False

    @property
    def response_was_empty(self) -> bool:
        """True if the last translated response produced no meaningful content."""
        return self._last_was_empty

    def reset(self) -> None:
        """Clear all streaming state between requests."""
        self._tool_call_buffers.clear()
        self._tool_call_meta.clear()
        self._last_was_empty = False
        self._saw_content = False

    # ── Request translation ──────────────────────────────────────────────────

    def translate_request(self, gemini_request: dict) -> dict:
        """Convert a Gemini ``generateContent`` request to Chat Completions format.

        The model name is NOT included — it lives in the URL path and must be
        injected by the route handler.
        """
        messages: list[dict] = []

        # System instruction → system message. Gemini's published schema
        # types ``systemInstruction`` as a Content object, but the wire
        # accepts whatever JSON the client sends: the schemathesis
        # conformance suite (KBR-82) generated ``true`` here and the
        # extraction below crashed with AttributeError, answering 500
        # where the schema documents 200. A malformed shape is the
        # client's mistake — treat it as absent and let the request
        # proceed, rather than taking the server down.
        system_instruction = gemini_request.get("systemInstruction")
        if isinstance(system_instruction, dict):
            text = self._extract_text(system_instruction)
            if text:
                messages.append({"role": "system", "content": text})

        # Translate contents → messages. A non-list ``contents`` contributes
        # no messages: iterating a scalar raises, and iterating a dict walks
        # its keys, neither of which is a Content. Same posture as the
        # generationConfig guard below (the conformance fuzzer generates
        # scalars here too — KBR-82's run).
        contents = gemini_request.get("contents")
        if not isinstance(contents, list):
            contents = []
        for content in contents:
            msg = self._translate_content(content)
            if msg is not None:
                if isinstance(msg, list):
                    messages.extend(msg)
                else:
                    messages.append(msg)

        cc_request: dict = {"messages": messages, "stream": True}

        # generationConfig mapping. The schema names ``type: object``, so a
        # real Gemini client always sends a mapping; the conformance fuzzer
        # nevertheless occasionally generates scalars (``None``, ``int``),
        # and the ``in`` tests below cannot iterate those. Tolerate the
        # wrong-type case by treating anything not a dict as absent — the
        # same shape ``.get(..., {})`` was meant to provide, but failed for
        # present-but-null because ``None`` was kept as the value.
        gen_config = gemini_request.get("generationConfig")
        if not isinstance(gen_config, dict):
            gen_config = {}
        if "temperature" in gen_config:
            cc_request["temperature"] = gen_config["temperature"]
        if "maxOutputTokens" in gen_config:
            cc_request["max_tokens"] = gen_config["maxOutputTokens"]
        if "topP" in gen_config:
            cc_request["top_p"] = gen_config["topP"]

        # KBR-213: Gemini names the stop-sequence list ``stopSequences``; Chat
        # Completions names it ``stop``. Without the rename the user's stop
        # sequences die in the first hop and nothing downstream can restore
        # them — the same shape KBR-178 settled for the Messages ingress.
        # An empty list is omitted (it asks for no stop behaviour, and
        # ``stop: []`` violates the published CC schema's ``StopConfiguration``
        # ``minItems: 1``). Everything else is forwarded verbatim: a bare
        # string, a list with non-string members, an over-limit list. The
        # provider rejects the wrong shape; an explicit error beats an
        # instruction silently thrown away. The empty-string scalar guard in
        # CC-ingress ``_normalize_cc_stop`` does not run here — normalisation
        # is single-site (KBR-178 R11, authority M15).
        stop_sequences = gen_config.get("stopSequences")
        if stop_sequences:
            cc_request["stop"] = stop_sequences

        # KBR-213 / KBR-178: Chat Completions declares no ``top_k`` at all, so
        # a bare key would be a field no CC provider accepts. The value rides
        # the internal key KBR-178 introduced; Anthropic-family adapters
        # restore it and ``_INTERNAL_KEYS`` strips it everywhere else.
        # Booleans are excluded because ``isinstance(True, int)`` is True;
        # the harness reader excludes them from ``(int,)`` typing via
        # ``_typed_leaf`` (the same bool/int subclass trap), so this guard
        # mirrors the reader — a wire ``true`` should not project a value
        # the projection itself rejects.
        top_k = gen_config.get("topK")
        if isinstance(top_k, int) and not isinstance(top_k, bool):
            cc_request["_top_k"] = top_k

        # KBR-301: the four remaining sampling mappings whose Gemini
        # spellings map onto Chat Completions spellings without a name
        # collision. `candidateCount` is Chat Completions' `n` (choices
        # per prompt); `presencePenalty` / `frequencyPenalty` / `seed`
        # are the CC spellings verbatim. Every guard mirrors the harness
        # reader's `_typed_leaf` (`tests/harness/reader_gemini.py:915-960`):
        # presence + type, with booleans excluded because `bool` is an
        # `int` subclass and an unguarded `isinstance` would carry `true`
        # through as the integer 1 — the same trap KBR-213 guarded `topK`
        # against.
        candidate_count = gen_config.get("candidateCount")
        if isinstance(candidate_count, int) and not isinstance(candidate_count, bool):
            cc_request["n"] = candidate_count
        presence_penalty = gen_config.get("presencePenalty")
        if isinstance(presence_penalty, (int, float)) and not isinstance(presence_penalty, bool):
            cc_request["presence_penalty"] = presence_penalty
        frequency_penalty = gen_config.get("frequencyPenalty")
        if isinstance(frequency_penalty, (int, float)) and not isinstance(frequency_penalty, bool):
            cc_request["frequency_penalty"] = frequency_penalty
        seed = gen_config.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            cc_request["seed"] = seed

        # KBR-301: the `logprobs` name collision. Gemini `logprobs` is
        # the **integer** count Chat Completions calls `top_logprobs`;
        # Gemini `responseLogprobs` is the **boolean** flag Chat
        # Completions calls `logprobs`. Each maps onto the *other*
        # spelling — carrying either onto its own spelling would silently
        # break every request that asks for logprobs on a Gemini route
        # (the CC wire would interpret the integer as the flag, or vice
        # versa). `reader_gemini.py:129-137` documents the same
        # collision on the projection side; `tests/test_reader_gemini.py`
        # pins it.
        response_logprobs = gen_config.get("responseLogprobs")
        if isinstance(response_logprobs, bool):
            cc_request["logprobs"] = response_logprobs
        logprobs = gen_config.get("logprobs")
        if isinstance(logprobs, int) and not isinstance(logprobs, bool):
            cc_request["top_logprobs"] = logprobs

        # KBR-301: the five format-specific control fields
        # (`responseMimeType`, `responseSchema`, `thinkingConfig`,
        # `mediaResolution`, `speechConfig`) are deliberately dropped —
        # Chat Completions declares no equivalent, and folding any onto
        # `response_format` would mis-carry a schema constraint as a
        # format request. The omission is registered in
        # `.system_design/TEST_SUITE.md` §9.2 with a gap row that waits
        # on the T-D5 corpus for its trigger case + §3.3.2 complement.

        # Tools mapping
        tools = self._translate_tools(gemini_request.get("tools", []))
        if tools:
            cc_request["tools"] = tools

        # KBR-221: carry the agent's functionCallingConfig.
        carry_gemini_tool_choice(gemini_request, cc_request)

        return cc_request

    def _translate_content(self, content: dict) -> dict | list[dict] | None:
        """Translate a single Gemini Content object to CC message(s).

        Container shapes (``content`` a dict, ``parts`` a list of dicts,
        ``functionResponse`` / ``functionCall`` dicts with a string
        ``name``) are validated at the ingress boundary since KBR-288 —
        ``_normalize_gemini_request`` 400s the malformed ones before this
        method runs. The ``isinstance`` guards below are defence in depth,
        not the primary contract: they keep the skip-never-raise posture
        for any caller that bypasses the boundary (tests, future routes)
        and for the leaf values this method reads directly (``role``,
        ``text``, ``functionCall`` / ``functionResponse`` names).
        """
        if not isinstance(content, dict):
            return None
        role = content.get("role", "user")
        # Boundary guarantees a string here; defence-in-depth defaults to
        # the documented ``"user"`` so a non-string role cannot trigger
        # ``_ROLE_MAP``'s ``unhashable type`` crash on bypass callers.
        if not isinstance(role, str):
            role = "user"
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            parts = []
        parts = [p for p in parts if isinstance(p, dict)]
        cc_role = _ROLE_MAP.get(role, role)

        # Check for functionResponse (tool result)
        if cc_role == "tool":
            results = []
            for part in parts:
                fr = part.get("functionResponse")
                if isinstance(fr, dict) and isinstance(fr.get("name"), str):
                    # Echo the inbound wire id; synthesise only when absent (KBR-195).
                    results.append(
                        {
                            "role": "tool",
                            "tool_call_id": fr.get("id") or self._make_tool_call_id(fr["name"]),
                            "content": json.dumps(fr.get("response", {})),
                        }
                    )
            return results if results else None

        # Check for functionCall in assistant messages
        if cc_role == "assistant":
            tool_calls = []
            text_parts = []
            thought_parts = []
            for part in parts:
                fc = part.get("functionCall")
                if isinstance(fc, dict) and isinstance(fc.get("name"), str):
                    # Echo the inbound wire id; synthesise only when absent (KBR-195).
                    tool_calls.append(
                        {
                            "id": fc.get("id") or self._make_tool_call_id(fc["name"]),
                            "type": "function",
                            "function": {
                                "name": fc["name"],
                                "arguments": json.dumps(fc.get("args", {})),
                            },
                        }
                    )
                elif isinstance(part.get("text"), str) and part.get("thought"):
                    # Defence-in-depth: boundary 400s non-string text; on
                    # bypass, drop the part rather than crash the join.
                    thought_parts.append(part["text"])
                elif isinstance(part.get("text"), str):
                    text_parts.append(part["text"])

            msg: dict = {"role": "assistant"}
            if text_parts:
                msg["content"] = "\n".join(text_parts)
            else:
                msg["content"] = None
            if thought_parts:
                msg["reasoning_content"] = "\n".join(thought_parts)
            if tool_calls:
                msg["tool_calls"] = tool_calls
            return msg

        # Regular user message
        texts = [p["text"] for p in parts if isinstance(p.get("text"), str)]
        if texts:
            return {"role": "user", "content": "\n".join(texts)}
        return None

    def _translate_tools(self, gemini_tools: list[dict]) -> list[dict]:
        """Convert Gemini functionDeclarations to CC tools.

        ``tools`` shape is validated at the ingress boundary since KBR-288
        (``_normalize_gemini_request``): a list ``tools`` whose members are
        dicts with list ``functionDeclarations`` of dicts with string
        ``name``. The ``isinstance`` guards below are defence in depth for
        callers that bypass the boundary, keeping the skip-never-raise
        posture (a function tool without a name is unusable on the CC
        wire; dropping it beats crashing the request).
        """
        cc_tools: list[dict] = []
        if not isinstance(gemini_tools, list):
            return cc_tools
        for tool in gemini_tools:
            if not isinstance(tool, dict):
                continue
            for fd in tool.get("functionDeclarations", []):
                if not isinstance(fd, dict) or not isinstance(fd.get("name"), str):
                    continue
                cc_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": fd["name"],
                            "description": fd.get("description", ""),
                            "parameters": fd.get("parameters", {}),
                        },
                    }
                )
        return cc_tools

    @staticmethod
    def _extract_text(content: dict) -> str:
        """Extract concatenated text from a Gemini Content object.

        Calls ``parts[i].text`` for each entry of ``parts``. Schemathesis
        fuzzing (KBR-82's conformance run) found that a malformed
        ``systemInstruction`` can be any JSON value at all -- an integer, a
        list -- and its ``parts`` can hold entries that are not ``dict``
        instances; the unguarded ``.get`` raised ``AttributeError`` into the
        request handler, which answered with a 500. Per the Gemini schema a
        Content is always ``{parts: [{text: ...}, ...]}``; the fuzzer's job
        is to find bodies that violate the schema, the translator's job is
        to return no text from the ones that do. The 400 the rest of the
        bridge returns for a malformed body is the right outcome, not a 500.
        """
        if not isinstance(content, dict):
            return ""
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            return ""
        # A tolerated envelope gets no boundary check, so this leaf read
        # carries its own guard (KBR-288): a non-string ``text`` would
        # otherwise crash the join with "sequence item 0". Per §12.3 the
        # whole envelope is tolerated-as-absent, so the part contributes
        # nothing rather than raising.
        return "\n".join(
            p["text"]
            for p in parts
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )

    @staticmethod
    def _make_tool_call_id(name: str) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex}"

    # ── Response translation ─────────────────────────────────────────────────

    def translate_response(self, cc_response: dict, *, context: dict | None = None) -> dict:
        """Convert a Chat Completions response to Gemini generateContent format.

        Each upstream ``tool_calls[].id`` is carried onto the emitted
        ``functionCall`` part when present and omitted when absent (KBR-257,
        the response-direction mirror of KBR-195). Gemini
        ``FunctionCall.id`` is optional per ``v1beta`` so synthesis stays on
        the request side. See ``SYSTEM_DESIGN.md`` §4 X4 for the rule and
        its round-trip rationale.

        Args:
            cc_response: The Chat Completions response body, with
                ``choices[0].message.tool_calls`` carrying one entry per
                upstream tool call.
            context: Optional correlation context (unused; kept for the
                server's signature compatibility).

        Returns:
            A ``candidates[0].content.parts[]`` body where each
            ``functionCall`` part carries the upstream ``name`` and parsed
            ``args`` and, when the upstream sent one, the upstream ``id``.
            Empty responses emit a single ``{"text": ""}`` part.
        """
        choice = cc_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")
        usage = cc_response.get("usage", {})

        parts: list[dict] = []

        # Reasoning content -> thought part
        reasoning = message.get("reasoning_content")
        if reasoning:
            parts.append({"text": reasoning, "thought": True})

        # Text content. KBR-285: a raw-CC upstream may deliver content as a
        # list of multimodal parts — coerce to a joined string — and a
        # refusal-only reply carries the model's reply on ``refusal`` with
        # ``content`` null, which becomes text too. A parts list with no
        # text element coerces to "" and emits no text part: the Gemini wire
        # has no image-delta equivalent.
        text = message.get("content")
        if isinstance(text, list):
            text = _extract_text_parts(text)
        if not text:
            refusal = message.get("refusal")
            if isinstance(refusal, str) and refusal:
                text = refusal
        if text:
            parts.append({"text": text})

        # Tool calls → functionCall parts
        for tc in message.get("tool_calls", []):
            args_str = tc["function"]["arguments"]
            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                # TypeError: a malformed non-string ``arguments`` value
                # (dict or number) — the detector widening makes the shape
                # reachable on translated routes, so degrade to {} rather
                # than a 500.
                args = {}
            function_call: dict = {"name": tc["function"]["name"], "args": args}
            # Echo the upstream wire id when present; omit when absent (KBR-257).
            if tc.get("id") is not None:
                function_call["id"] = tc["id"]
            parts.append({"functionCall": function_call})

        # KBR-285: the deprecated single-dict ``function_call`` maps to one
        # functionCall part, the same shape the ``tool_calls`` loop emits.
        # No real upstream carries both; if one did, the loop above already
        # ran and the legacy path appends a second part — the least-bad merge,
        # recorded for the precondition.
        function_call = message.get("function_call")
        has_function_call = isinstance(function_call, dict) and bool(function_call)
        if has_function_call:
            try:
                fc_args = json.loads(function_call.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                fc_args = {}
            fc_part: dict = {"name": function_call.get("name", ""), "args": fc_args}
            if function_call.get("id") is not None:
                fc_part["id"] = function_call["id"]
            parts.append({"functionCall": fc_part})

        if not parts:
            parts.append({"text": ""})

        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": parts},
                    "finishReason": _CC_TO_GEMINI_FINISH.get(finish_reason, "STOP"),
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0),
            },
            "modelVersion": cc_response.get("model", ""),
        }

    # ── Streaming translation ────────────────────────────────────────────────

    def translate_stream_chunk(self, chunk: dict) -> list[str]:
        """Convert one Chat Completions streaming chunk to Gemini SSE events.

        Tool-call arguments are buffered per CC ``tool_calls[].index`` and
        emitted once, at the finish chunk. The ``id`` riding the opening
        (name-bearing) delta is carried into ``_tool_call_meta`` and onto
        the emitted ``functionCall`` part; ids on later deltas and absent
        ids are omitted, not synthesised (KBR-257, §4 X4). The open is
        name-keyed per §4.2's allocate-at-open contract.

        Args:
            chunk: One Chat Completions streaming chunk.

        Returns:
            A list of SSE event strings (``data: {json}\\n\\n``) — buffered
            ``functionCall`` parts first, then the finish event. Empty
            until the finish chunk when nothing else streamed.
        """
        events: list[str] = []
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        usage = chunk.get("usage")

        # Reasoning delta -> thought part
        reasoning = delta.get("reasoning_content")
        if reasoning:
            self._saw_content = True
            events.append(
                format_gemini_sse(
                    {
                        "candidates": [
                            {
                                "content": {"role": "model", "parts": [{"text": reasoning, "thought": True}]},
                                "index": 0,
                            }
                        ],
                    }
                )
            )

        # Text delta. KBR-285: a raw-CC upstream may deliver content as a
        # list of multimodal parts — coerce to a joined string — and a
        # refusal-only delta carries the model's reply on ``refusal`` with
        # ``content`` null, which becomes text too.
        text = delta.get("content")
        if isinstance(text, list):
            text = _extract_text_parts(text)
        if not text:
            refusal = delta.get("refusal")
            if isinstance(refusal, str) and refusal:
                text = refusal
        if text:
            self._saw_content = True
            events.append(
                format_gemini_sse(
                    {
                        "candidates": [
                            {
                                "content": {"role": "model", "parts": [{"text": text}]},
                                "index": 0,
                            }
                        ],
                    }
                )
            )

        # Tool call delta — buffer arguments. KBR-285: the deprecated
        # single-dict ``function_call`` maps onto the same machinery — the
        # opening delta synthesises the index and carries the name, later
        # deltas argument-append. The existing ``tool_calls`` branch handles
        # both shapes unchanged. No real upstream carries both fields; if one
        # did, a ``tool_calls`` list wins and a later legacy delta appends to
        # the index-0 buffer that call opened.
        tool_calls = delta.get("tool_calls") or []
        if not tool_calls:
            legacy_call = delta.get("function_call")
            if isinstance(legacy_call, dict) and legacy_call:
                if 0 in self._tool_call_meta:
                    tool_calls = [
                        {
                            "index": 0,
                            "function": {"arguments": legacy_call.get("arguments", "")},
                        }
                    ]
                else:
                    tool_calls = [
                        {
                            "index": 0,
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": legacy_call.get("name", ""),
                                "arguments": legacy_call.get("arguments", ""),
                            },
                        }
                    ]
        for tc in tool_calls:
            idx = tc.get("index", 0)
            func = tc.get("function", {})

            if "name" in func and func.get("name"):
                # New tool call starts — carry the upstream wire id for the emit (KBR-257).
                self._tool_call_meta[idx] = {"name": func["name"], "id": tc.get("id")}
                self._tool_call_buffers[idx] = ToolCallBuffer()

            if "arguments" in func and idx in self._tool_call_buffers:
                self._tool_call_buffers[idx].append(func["arguments"])

        # Finish — emit any buffered tool calls + finish event
        if finish_reason is not None:
            # Emit buffered tool calls
            for idx in sorted(self._tool_call_buffers):
                try:
                    args_str = self._tool_call_buffers[idx].finalize()
                    args = json.loads(args_str)
                except (ToolCallBufferError, json.JSONDecodeError):
                    args = {}
                meta = self._tool_call_meta.get(idx, {"name": "unknown"})
                function_call: dict = {"name": meta["name"], "args": args}
                # Echo the upstream wire id when present; omit when absent (KBR-257).
                if meta.get("id") is not None:
                    function_call["id"] = meta["id"]
                events.append(
                    format_gemini_sse(
                        {
                            "candidates": [
                                {
                                    "content": {
                                        "role": "model",
                                        "parts": [{"functionCall": function_call}],
                                    },
                                    "index": 0,
                                }
                            ],
                        }
                    )
                )

            # Finish event
            finish_data: dict = {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": []},
                        "finishReason": _CC_TO_GEMINI_FINISH.get(finish_reason, "STOP"),
                        "index": 0,
                    }
                ],
            }
            if usage:
                finish_data["usageMetadata"] = {
                    "promptTokenCount": usage.get("prompt_tokens", 0),
                    "candidatesTokenCount": usage.get("completion_tokens", 0),
                    "totalTokenCount": usage.get("total_tokens", 0),
                }
            events.append(format_gemini_sse(finish_data))
            was_empty = not self._saw_content and not self._tool_call_buffers
            self.reset()
            self._last_was_empty = was_empty

        return events
