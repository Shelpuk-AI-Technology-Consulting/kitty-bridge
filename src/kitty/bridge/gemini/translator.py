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
    "length": "MAX_TOKENS",
    "content_filter": "SAFETY",
    None: "STOP",
}

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

        # System instruction → system message
        system_instruction = gemini_request.get("systemInstruction")
        if system_instruction:
            text = self._extract_text(system_instruction)
            if text:
                messages.append({"role": "system", "content": text})

        # Translate contents → messages
        for content in gemini_request.get("contents", []):
            msg = self._translate_content(content)
            if msg is not None:
                if isinstance(msg, list):
                    messages.extend(msg)
                else:
                    messages.append(msg)

        cc_request: dict = {"messages": messages, "stream": True}

        # generationConfig mapping
        gen_config = gemini_request.get("generationConfig", {})
        if "temperature" in gen_config:
            cc_request["temperature"] = gen_config["temperature"]
        if "maxOutputTokens" in gen_config:
            cc_request["max_tokens"] = gen_config["maxOutputTokens"]
        if "topP" in gen_config:
            cc_request["top_p"] = gen_config["topP"]

        # Tools mapping
        tools = self._translate_tools(gemini_request.get("tools", []))
        if tools:
            cc_request["tools"] = tools

        # KBR-221: carry the agent's functionCallingConfig.
        carry_gemini_tool_choice(gemini_request, cc_request)

        return cc_request

    def _translate_content(self, content: dict) -> dict | list[dict] | None:
        """Translate a single Gemini Content object to CC message(s)."""
        role = content.get("role", "user")
        parts = content.get("parts", [])
        cc_role = _ROLE_MAP.get(role, role)

        # Check for functionResponse (tool result)
        if cc_role == "tool":
            results = []
            for part in parts:
                fr = part.get("functionResponse")
                if fr:
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
                if fc:
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
                elif "text" in part and part.get("thought"):
                    thought_parts.append(part["text"])
                elif "text" in part:
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
        texts = [p["text"] for p in parts if "text" in p]
        if texts:
            return {"role": "user", "content": "\n".join(texts)}
        return None

    def _translate_tools(self, gemini_tools: list[dict]) -> list[dict]:
        """Convert Gemini functionDeclarations to CC tools."""
        cc_tools: list[dict] = []
        for tool in gemini_tools:
            for fd in tool.get("functionDeclarations", []):
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
        """Extract concatenated text from a Gemini Content object."""
        return "\n".join(p.get("text", "") for p in content.get("parts", []) if "text" in p)

    @staticmethod
    def _make_tool_call_id(name: str) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex}"

    # ── Response translation ─────────────────────────────────────────────────

    def translate_response(self, cc_response: dict, *, context: dict | None = None) -> dict:
        """Convert a Chat Completions response to Gemini generateContent format."""
        choice = cc_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")
        usage = cc_response.get("usage", {})

        parts: list[dict] = []

        # Reasoning content -> thought part
        reasoning = message.get("reasoning_content")
        if reasoning:
            parts.append({"text": reasoning, "thought": True})

        # Text content
        text = message.get("content")
        if text:
            parts.append({"text": text})

        # Tool calls → functionCall parts
        for tc in message.get("tool_calls", []):
            args_str = tc["function"]["arguments"]
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
            parts.append({"functionCall": {"name": tc["function"]["name"], "args": args}})

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

        Returns a list of SSE event strings (``data: {json}\\n\\n``).
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

        # Text delta
        text = delta.get("content")
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

        # Tool call delta — buffer arguments
        for tc in delta.get("tool_calls", []):
            idx = tc.get("index", 0)
            func = tc.get("function", {})

            if "name" in func and func.get("name"):
                # New tool call starts
                self._tool_call_meta[idx] = {"name": func["name"]}
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
                events.append(
                    format_gemini_sse(
                        {
                            "candidates": [
                                {
                                    "content": {
                                        "role": "model",
                                        "parts": [{"functionCall": {"name": meta["name"], "args": args}}],
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
