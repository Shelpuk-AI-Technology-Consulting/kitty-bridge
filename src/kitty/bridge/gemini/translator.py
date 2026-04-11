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

__all__ = ["GeminiTranslator"]

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

    def reset(self) -> None:
        """Clear all streaming state between requests."""
        self._tool_call_buffers.clear()
        self._tool_call_meta.clear()

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
                    results.append(
                        {
                            "role": "tool",
                            "tool_call_id": self._make_tool_call_id(fr["name"]),
                            "content": json.dumps(fr.get("response", {})),
                        }
                    )
            return results if results else None

        # Check for functionCall in assistant messages
        if cc_role == "assistant":
            tool_calls = []
            text_parts = []
            for part in parts:
                fc = part.get("functionCall")
                if fc:
                    tool_calls.append(
                        {
                            "id": self._make_tool_call_id(fc["name"]),
                            "type": "function",
                            "function": {
                                "name": fc["name"],
                                "arguments": json.dumps(fc.get("args", {})),
                            },
                        }
                    )
                elif "text" in part:
                    text_parts.append(part["text"])

            msg: dict = {"role": "assistant"}
            if text_parts:
                msg["content"] = "\n".join(text_parts)
            else:
                msg["content"] = None
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

    def translate_response(self, cc_response: dict) -> dict:
        """Convert a Chat Completions response to Gemini generateContent format."""
        choice = cc_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")
        usage = cc_response.get("usage", {})

        parts: list[dict] = []

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

        # Text delta
        text = delta.get("content")
        if text:
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
            self.reset()

        return events
