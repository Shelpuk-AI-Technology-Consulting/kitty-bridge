"""Ollama Cloud API adapter.

Translates between Kitty's internal Chat Completions (CC) format and
Ollama's native ``/api/chat`` endpoint on ``https://ollama.com``.

Why custom transport?
  The bridge server's standard streaming path assumes OpenAI-style SSE
  (``data: {...}\\n\\n``).  Ollama Cloud returns NDJSON — one JSON object
  per line with no ``data: `` prefix and no ``[DONE]`` sentinel.  Because
  the formats are incompatible, this adapter sets ``use_custom_transport``
  and manages its own aiohttp sessions for both non-streaming and streaming
  requests.

Translation layers (bridge → adapter → Ollama Cloud):

  1. **Message content**: CC can send ``content`` as a plain string *or*
     a list of content blocks like ``[{"type": "text", "text": "..."}]``.
     Ollama expects a plain string.  ``_flatten_content`` handles both.

  2. **Tool calls (request)**: CC encodes ``function.arguments`` as a JSON
     *string* (e.g. ``"{"city": "NYC"}"``).  Ollama's native API expects
     ``arguments`` as a parsed *dict*.  ``_cc_tool_calls_to_ollama``
     converts between the two.

  3. **Tool calls (response)**: Ollama returns ``arguments`` as a dict;
     CC expects a JSON string.  ``_translate_tool_calls`` does the reverse.

  4. **Tool result messages**: CC uses ``{"role": "tool", "name": "...",
     "tool_call_id": "...", "content": "..."}``.  Ollama uses
     ``{"role": "tool", "tool_name": "...", "content": "..."}`` — no
     ``tool_call_id``, uses ``tool_name`` instead of ``name``.

  5. **Streaming**: Ollama returns NDJSON lines; the adapter collects
     content deltas and emits CC-format SSE chunks.  For the Messages
     API path (Claude Code), the bridge collects those SSE bytes, then
     calls ``parse_stream_to_cc_response`` to re-parse them into a CC
     response dict before translating to Messages API SSE events.

  6. **Sampling parameters**: CC ``temperature`` / ``top_p`` / ``max_tokens``
     are mapped into Ollama's ``options`` dict (``num_predict`` for
     ``max_tokens``).
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Awaitable, Callable

import aiohttp

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["OllamaCloudAdapter"]

logger = logging.getLogger(__name__)

_DONE_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
}


class OllamaCloudAdapter(ProviderAdapter):
    """Ollama Cloud provider using the native ``/api/chat`` endpoint.

    Uses custom transport because Ollama returns NDJSON streaming
    (not SSE).  The adapter manages its own aiohttp session and
    translates between CC and Ollama formats.
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    @property
    def provider_type(self) -> str:
        return "ollama_cloud"

    @property
    def default_base_url(self) -> str:
        return "https://ollama.com"

    @property
    def upstream_path(self) -> str:
        return "/api/chat"

    @property
    def use_custom_transport(self) -> bool:
        return True

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Build headers with Bearer token auth for Ollama Cloud."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def normalize_model_name(self, model: str) -> str:
        """Strip known provider prefix if present."""
        for prefix in ("ollama_cloud/", "ollama/"):
            if model.startswith(prefix):
                return model[len(prefix) :]
        return model

    # ── CC → Ollama request translation ────────────────────────────────

    def translate_to_upstream(self, cc_request: dict) -> dict:
        """Translate a CC request into Ollama ``/api/chat`` format.

        Handles:
        - System messages (forwarded as-is; Ollama supports system role)
        - Tool result messages (CC ``tool_call_id``/``name`` → Ollama ``tool_name``)
        - Options (CC ``temperature``/``top_p`` → Ollama ``options``)
        - Strips internal metadata keys
        """
        result: dict = {
            "model": cc_request["model"],
            "messages": self._translate_messages(cc_request.get("messages", [])),
        }

        if "stream" in cc_request:
            result["stream"] = cc_request["stream"]

        if cc_request.get("tools"):
            result["tools"] = cc_request["tools"]

        # Map CC sampling params → Ollama options
        options: dict = {}
        for key in ("temperature", "top_p", "top_k"):
            if key in cc_request and cc_request[key] is not None:
                options[key] = cc_request[key]
        if options:
            result["options"] = options

        if "max_tokens" in cc_request and cc_request["max_tokens"] is not None:
            result["options"] = result.get("options", {})
            result["options"]["num_predict"] = cc_request["max_tokens"]

        return result

    def _translate_messages(self, messages: list[dict]) -> list[dict]:
        """Translate CC messages to Ollama message format.

        CC → Ollama differences handled:
        - ``tool`` messages: CC uses ``name``/``tool_call_id``, Ollama uses ``tool_name``
        - ``content``: CC can be a list of content blocks; Ollama expects a string
        - ``tool_calls[].function.arguments``: CC uses a JSON string, Ollama expects a dict
        """
        result = []
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                result.append({
                    "role": "tool",
                    "tool_name": msg.get("name", ""),
                    "content": self._flatten_content(msg.get("content", "")),
                })
            else:
                translated = {"role": role}
                content = msg.get("content")
                if content is not None:
                    translated["content"] = self._flatten_content(content)
                if msg.get("tool_calls"):
                    translated["tool_calls"] = self._cc_tool_calls_to_ollama(msg["tool_calls"])
                if msg.get("thinking"):
                    translated["thinking"] = msg["thinking"]
                result.append(translated)
        return result

    @staticmethod
    def _flatten_content(content: str | list | None) -> str:
        """Flatten CC content (string or list of content blocks) to a plain string for Ollama."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _cc_tool_calls_to_ollama(tool_calls: list[dict]) -> list[dict]:
        """Convert CC tool_calls (arguments as JSON string) to Ollama format (arguments as dict)."""
        result = []
        for tc in tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result.append({
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": args,
                },
            })
        return result

    # ── Ollama → CC response translation ───────────────────────────────

    def translate_from_upstream(self, raw_response: dict) -> dict:
        """Translate an Ollama ``/api/chat`` response into CC format.

        Ollama response fields:
        - ``message``: ``{role, content, tool_calls?}``
        - ``done_reason``: ``"stop"`` | ``"length"``
        - ``prompt_eval_count``, ``eval_count``: token counts
        """
        message = raw_response.get("message", {})
        done_reason = raw_response.get("done_reason", "stop")
        finish_reason = _DONE_REASON_MAP.get(done_reason, "stop")

        cc_message: dict = {"role": message.get("role", "assistant")}
        content = message.get("content") or None
        cc_message["content"] = content

        tool_calls = message.get("tool_calls")
        if tool_calls:
            cc_message["tool_calls"] = self._translate_tool_calls(tool_calls)

        prompt_tokens = raw_response.get("prompt_eval_count") or 0
        completion_tokens = raw_response.get("eval_count") or 0

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": 0,
            "model": raw_response.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": cc_message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    # ── SSE chunk builder ──────────────────────────────────────────────

    @staticmethod
    def _translate_tool_calls(tool_calls: list[dict]) -> list[dict]:
        """Translate Ollama tool calls to CC format (arguments as JSON string)."""
        cc_tool_calls = []
        for tc in tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, dict):
                args = json.dumps(args)
            cc_tool_calls.append({
                "index": tc.get("function", {}).get("index", len(cc_tool_calls)),
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": args,
                },
            })
        return cc_tool_calls

    def _make_sse_chunk(
        self,
        response_id: str,
        model: str,
        delta: dict,
        finish_reason: str | None = None,
    ) -> str:
        """Build a CC streaming SSE chunk string."""
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    # ── Session management ─────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _build_url(self, provider_config: dict) -> str:
        """Build the full upstream URL."""
        base = (provider_config.get("base_url") or self.default_base_url).rstrip("/")
        return f"{base}{self.upstream_path}"

    def parse_stream_to_cc_response(self, raw: bytes) -> dict:
        """Parse collected Chat Completions SSE chunks into a CC response."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        model = ""
        finish_reason = "stop"

        for line in raw.decode("utf-8", errors="replace").split("\n"):
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            model = chunk.get("model", model)
            choice = (chunk.get("choices") or [{}])[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if delta.get("content"):
                text_parts.append(delta["content"])
            if delta.get("tool_calls"):
                tool_calls.extend(delta["tool_calls"])

        message: dict = {"role": "assistant", "content": "".join(text_parts) or None}
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # ── Custom transport: non-streaming ────────────────────────────────

    async def make_request(self, cc_request: dict) -> dict:
        """Perform a non-streaming Ollama Cloud request via aiohttp."""
        api_key = cc_request.get("_resolved_key", "")
        provider_config = cc_request.get("_provider_config", {})

        ollama_body = self.translate_to_upstream(cc_request)
        ollama_body["stream"] = False

        url = self._build_url(provider_config)
        headers = self.build_upstream_headers(api_key)

        session = await self._get_session()
        async with session.post(url, json=ollama_body, headers=headers) as resp:
            if resp.status >= 400:
                try:
                    body = await resp.json()
                except Exception:
                    text = await resp.text()
                    raise ProviderError(f"Ollama Cloud HTTP {resp.status}: {text[:500]}") from None
                raise self.map_error(resp.status, body)
            response_data = await resp.json()
            return self.translate_from_upstream(response_data)

    # ── Custom transport: streaming ────────────────────────────────────

    async def stream_request(
        self,
        cc_request: dict,
        write: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Perform a streaming Ollama Cloud request: NDJSON → CC SSE."""
        api_key = cc_request.get("_resolved_key", "")
        provider_config = cc_request.get("_provider_config", {})

        ollama_body = self.translate_to_upstream(cc_request)
        ollama_body["stream"] = True

        url = self._build_url(provider_config)
        headers = self.build_upstream_headers(api_key)

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model = cc_request.get("model", "")

        session = await self._get_session()
        async with session.post(url, json=ollama_body, headers=headers) as resp:
            if resp.status >= 400:
                try:
                    body = await resp.json()
                except Exception:
                    text = await resp.text()
                    raise ProviderError(f"Ollama Cloud HTTP {resp.status}: {text[:500]}") from None
                raise self.map_error(resp.status, body)

            # Emit initial role chunk
            await write(
                self._make_sse_chunk(response_id, model, {"role": "assistant"}).encode()
            )

            stream_done = False
            line_buffer = ""
            async for chunk_bytes in resp.content:
                raw = chunk_bytes.decode("utf-8", errors="replace")
                line_buffer += raw

                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse Ollama NDJSON line: %s", line[:200])
                        continue

                    # Check for in-stream error
                    if "error" in data and data.get("done") is not True:
                        raise ProviderError(f"Ollama stream error: {data['error']}")

                    if data.get("done"):
                        # Final chunk — emit finish reason
                        done_reason = data.get("done_reason", "stop")
                        finish_reason = _DONE_REASON_MAP.get(done_reason, "stop")
                        await write(
                            self._make_sse_chunk(
                                response_id, model, {}, finish_reason=finish_reason
                            ).encode()
                        )
                        stream_done = True
                        break
                    else:
                        # Content / tool call chunk
                        message = data.get("message", {})
                        content = message.get("content", "")
                        if content:
                            await write(
                                self._make_sse_chunk(
                                    response_id, model, {"content": content}
                                ).encode()
                            )
                        tool_calls = message.get("tool_calls")
                        if tool_calls:
                            cc_tool_calls = self._translate_tool_calls(tool_calls)
                            await write(
                                self._make_sse_chunk(
                                    response_id, model, {"tool_calls": cc_tool_calls}
                                ).encode()
                            )

                if stream_done:
                    break

            # Flush remaining buffer (only if stream wasn't already done)
            if not stream_done and line_buffer.strip():
                try:
                    data = json.loads(line_buffer.strip())
                    if data.get("done"):
                        done_reason = data.get("done_reason", "stop")
                        finish_reason = _DONE_REASON_MAP.get(done_reason, "stop")
                        await write(
                            self._make_sse_chunk(
                                response_id, model, {}, finish_reason=finish_reason
                            ).encode()
                        )
                except json.JSONDecodeError:
                    pass

            await write(b"data: [DONE]\n\n")

    # ── Abstract method implementations ─────────────────────────────────

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        """Build a CC-format request dict (for abstract compliance)."""
        request: dict = {"model": model, "messages": messages}
        if "stream" in kwargs:
            request["stream"] = kwargs["stream"]
        if "tools" in kwargs and kwargs["tools"]:
            request["tools"] = kwargs["tools"]
        return request

    def parse_response(self, response_data: dict) -> dict:
        """Parse a CC-format response (for abstract compliance)."""
        choices = response_data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        message = choice.get("message", {})
        result: dict = {
            "content": message.get("content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": response_data.get("usage", {}),
        }
        if "tool_calls" in message:
            result["tool_calls"] = message["tool_calls"]
        return result

    def map_error(self, status_code: int, body: dict) -> Exception:
        """Map an Ollama HTTP error to ProviderError."""
        if not isinstance(body, dict):
            err = ProviderError(f"Ollama Cloud error {status_code}: {body}")
            err.http_status = status_code
            return err
        error = body.get("error", body)
        msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        err = ProviderError(f"Ollama Cloud error {status_code}: {msg}")
        err.http_status = status_code
        return err
