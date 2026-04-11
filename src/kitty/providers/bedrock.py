"""AWS Bedrock provider adapter — Converse API with boto3 transport."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["BedrockAdapter"]

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_REGION = "us-east-1"

# Bedrock stopReason → CC finish_reason
_STOP_REASON_MAP: dict[str | None, str] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
    None: "stop",
}


class BedrockAdapter(ProviderAdapter):
    """AWS Bedrock Converse API adapter.

    Uses boto3 SDK for SigV4 authentication and EventStream decoding.
    Supports two auth modes via provider_config:
    - AWS credentials stored in Kitty (access_key:secret_key)
    - AWS SSO / named profile (boto3 credential chain)
    """

    @property
    def provider_type(self) -> str:
        return "bedrock"

    @property
    def default_base_url(self) -> str:
        return f"https://bedrock-runtime.{_DEFAULT_REGION}.amazonaws.com"

    @property
    def use_custom_transport(self) -> bool:
        return True

    # ── Credential helpers ───────────────────────────────────────────────

    def parse_aws_credentials(self, raw: str) -> tuple[str, ...]:
        """Parse stored credential string into AWS credential parts.

        Formats:
        - ``"access_key:secret_key"``
        - ``"access_key:secret_key:session_token"``
        """
        parts = raw.split(":", 2)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ProviderError("Invalid AWS credentials format: expected 'access_key:secret_key[:session_token]'")
        return tuple(parts)

    def is_sso_mode(self, resolved_key: str) -> bool:
        """Check if the resolved key indicates SSO/profile auth."""
        return resolved_key.strip().lower() in ("sso", "")

    def get_region(self, provider_config: dict) -> str:
        """Get AWS region from provider_config."""
        return provider_config.get("region", _DEFAULT_REGION)

    def get_profile_name(self, provider_config: dict) -> str | None:
        """Get AWS profile name from provider_config."""
        return provider_config.get("profile_name")

    # ── boto3 client factory ─────────────────────────────────────────────

    def _get_boto3_client(self, resolved_key: str, provider_config: dict):
        """Create a boto3 bedrock-runtime client.

        Uses AWS credentials from resolved_key or SSO profile from
        provider_config.
        """
        try:
            import boto3
        except ImportError as err:
            raise ProviderError(
                "boto3 is required for the Bedrock provider. Install with: pip install boto3"
            ) from err

        region = self.get_region(provider_config)

        if self.is_sso_mode(resolved_key):
            profile_name = self.get_profile_name(provider_config)
            if profile_name:
                session = boto3.Session(profile_name=profile_name, region_name=region)
            else:
                session = boto3.Session(region_name=region)
            return session.client("bedrock-runtime")
        else:
            parts = self.parse_aws_credentials(resolved_key)
            access_key = parts[0]
            secret_key = parts[1]
            session_token = parts[2] if len(parts) > 2 else None
            session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                aws_session_token=session_token,
                region_name=region,
            )
            return session.client("bedrock-runtime")

    # ── CC → Bedrock request translation ─────────────────────────────────

    def translate_to_upstream(self, cc_request: dict) -> dict:
        """Translate a CC request into Bedrock Converse API format."""
        bedrock: dict = {
            "modelId": cc_request["model"],
            "messages": [],
            "inferenceConfig": {
                "maxTokens": cc_request.get("max_tokens", _DEFAULT_MAX_TOKENS),
            },
        }

        # Temperature and topP
        if "temperature" in cc_request and cc_request["temperature"] is not None:
            bedrock["inferenceConfig"]["temperature"] = cc_request["temperature"]
        if "top_p" in cc_request and cc_request["top_p"] is not None:
            bedrock["inferenceConfig"]["topP"] = cc_request["top_p"]

        # Extract system messages
        system_parts: list[str] = []
        for msg in cc_request.get("messages", []):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)

        if system_parts:
            bedrock["system"] = [{"text": s} for s in system_parts]

        # Translate messages
        for msg in cc_request.get("messages", []):
            role = msg.get("role")
            if role == "system":
                continue

            if role == "assistant":
                bedrock["messages"].append(self._translate_assistant_msg(msg))
            elif role == "tool":
                bedrock["messages"].append(self._translate_tool_result_msg(msg))
            else:
                content = msg.get("content", "")
                if isinstance(content, str):
                    content = [{"text": content}]
                bedrock["messages"].append({"role": role, "content": content})

        # Translate tools
        if "tools" in cc_request and cc_request["tools"]:
            bedrock["toolConfig"] = {
                "tools": self._translate_tools(cc_request["tools"]),
                "toolChoice": {"auto": {}},
            }

        return bedrock

    def _translate_assistant_msg(self, msg: dict) -> dict:
        """Translate assistant message with optional tool_calls."""
        content_blocks: list[dict] = []

        text = msg.get("content")
        if text:
            content_blocks.append({"text": text})

        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            content_blocks.append(
                {
                    "toolUse": {
                        "toolUseId": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        "name": func.get("name", ""),
                        "input": json.loads(func.get("arguments") or "{}"),
                    }
                }
            )

        return {"role": "assistant", "content": content_blocks or [{"text": ""}]}

    def _translate_tool_result_msg(self, msg: dict) -> dict:
        """Translate a CC tool result message to Bedrock toolResult."""
        content = msg.get("content", "")
        result_content = [{"text": content}] if isinstance(content, str) else content

        return {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": msg.get("tool_call_id", ""),
                        "content": result_content,
                        "status": "success",
                    }
                }
            ],
        }

    def _translate_tools(self, cc_tools: list[dict]) -> list[dict]:
        """Translate CC tool definitions to Bedrock toolSpec format."""
        bedrock_tools = []
        for tool in cc_tools:
            func = tool.get("function", {})
            bedrock_tools.append(
                {
                    "toolSpec": {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "inputSchema": {
                            "json": func.get("parameters", {"type": "object", "properties": {}}),
                        },
                    }
                }
            )
        return bedrock_tools

    # ── Bedrock → CC response translation ────────────────────────────────

    def translate_from_upstream(self, raw_response: dict) -> dict:
        """Translate a Bedrock Converse response into CC format."""
        output = raw_response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        text_parts: list[str] = []
        tool_uses: list[dict] = []

        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tool_uses.append(block["toolUse"])

        cc_message: dict = {"role": "assistant", "content": "\n".join(text_parts) or None}

        if tool_uses:
            cc_message["tool_calls"] = [
                {
                    "id": tu.get("toolUseId", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": tu.get("name", ""),
                        "arguments": json.dumps(tu.get("input", {})),
                    },
                }
                for tu in tool_uses
            ]

        stop_reason = raw_response.get("stopReason")
        finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop")

        usage = raw_response.get("usage", {})
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": 0,
            "model": "",
            "choices": [
                {
                    "index": 0,
                    "message": cc_message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("inputTokens", 0) + usage.get("outputTokens", 0),
            },
        }

    # ── Custom transport: non-streaming ──────────────────────────────────

    async def make_request(self, cc_request: dict) -> dict:
        """Perform a non-streaming Bedrock Converse call via boto3."""
        bedrock_request = self.translate_to_upstream(cc_request)
        model_id = bedrock_request.pop("modelId")

        # Extract provider_config for credential resolution
        provider_config = cc_request.get("_provider_config", {})
        resolved_key = cc_request.get("_resolved_key", "")

        client = self._get_boto3_client(resolved_key, provider_config)
        bedrock_request.pop("stream", None)

        # Run boto3 call in thread pool (it's synchronous)
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.converse(modelId=model_id, **bedrock_request),
            )
        except Exception as exc:
            raise ProviderError(f"Bedrock request failed: {exc}") from exc

        return self.translate_from_upstream(response)

    # ── Custom transport: streaming ──────────────────────────────────────

    async def stream_request(
        self,
        cc_request: dict,
        write: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Perform a streaming Bedrock ConverseStream call via boto3."""
        bedrock_request = self.translate_to_upstream(cc_request)
        model_id = bedrock_request.pop("modelId")

        provider_config = cc_request.get("_provider_config", {})
        resolved_key = cc_request.get("_resolved_key", "")

        client = self._get_boto3_client(resolved_key, provider_config)
        bedrock_request.pop("stream", None)

        # Run boto3 call in thread pool
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.converse_stream(modelId=model_id, **bedrock_request),
            )
        except Exception as exc:
            raise ProviderError(f"Bedrock streaming request failed: {exc}") from exc

        # Process EventStream events
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        tool_buffer: dict[int, dict] = {}  # index → {id, name, input_chunks}

        for event in response.get("stream", []):
            chunks = self._translate_stream_event(event, response_id, tool_buffer)
            for chunk in chunks:
                await write(chunk.encode() if isinstance(chunk, str) else chunk)

    def _translate_stream_event(self, event: dict, response_id: str, tool_buffer: dict) -> list[str]:
        """Translate a Bedrock EventStream event to CC SSE chunks."""
        chunks: list[str] = []

        if "messageStart" in event:
            chunks.append(self._make_sse_chunk(response_id, {"role": "assistant"}))

        elif "contentBlockStart" in event:
            start_data = event["contentBlockStart"]
            start = start_data.get("start", {})
            if "toolUse" in start:
                idx = start_data.get("contentBlockIndex", 0)
                tool_use = start["toolUse"]
                tool_buffer[idx] = {
                    "id": tool_use.get("toolUseId", f"call_{uuid.uuid4().hex[:24]}"),
                    "name": tool_use.get("name", ""),
                    "input_chunks": [],
                }

        elif "contentBlockDelta" in event:
            delta_data = event["contentBlockDelta"]
            delta = delta_data.get("delta", {})
            idx = delta_data.get("contentBlockIndex", 0)
            if "text" in delta:
                chunks.append(self._make_sse_chunk(response_id, {"content": delta["text"]}))
            elif "toolUse" in delta:
                tool_delta = delta["toolUse"]
                if "input" in tool_delta and idx in tool_buffer:
                    tool_buffer[idx]["input_chunks"].append(tool_delta["input"])

        elif "contentBlockStop" in event:
            idx = event.get("contentBlockStop", {}).get("contentBlockIndex", 0)
            if idx in tool_buffer:
                buf = tool_buffer.pop(idx)
                raw_input = "".join(buf["input_chunks"])
                try:
                    parsed_input = json.loads(raw_input) if raw_input else {}
                except json.JSONDecodeError:
                    parsed_input = {}
                tc = {
                    "id": buf["id"],
                    "type": "function",
                    "function": {
                        "name": buf["name"],
                        "arguments": json.dumps(parsed_input),
                    },
                }
                chunks.append(self._make_sse_chunk(response_id, {"tool_calls": [tc]}))

        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "end_turn")
            finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop")
            chunks.append(self._make_sse_chunk(response_id, {}, finish_reason=finish_reason))
            chunks.append("data: [DONE]\n\n")

        elif "metadata" in event:
            pass  # Usage metadata — captured but not streamed to agent

        return chunks

    def _make_sse_chunk(
        self,
        response_id: str,
        delta: dict,
        finish_reason: str | None = None,
    ) -> str:
        """Build a CC streaming SSE chunk string."""
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    # ── Standard ProviderAdapter methods ─────────────────────────────────

    def normalize_model_name(self, model: str) -> str:
        return model

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        request: dict = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
        }
        for key in ("temperature", "top_p", "max_tokens"):
            if key in kwargs and kwargs[key] is not None:
                request[key] = kwargs[key]
        if "tools" in kwargs and kwargs["tools"]:
            request["tools"] = kwargs["tools"]
        return request

    def parse_response(self, response_data: dict) -> dict:
        choice = response_data.get("choices", [{}])[0]
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
        error_msg = body.get("error", body.get("message", body))
        msg = error_msg.get("message", str(error_msg)) if isinstance(error_msg, dict) else str(error_msg)
        return ProviderError(f"Bedrock error {status_code}: {msg}")
