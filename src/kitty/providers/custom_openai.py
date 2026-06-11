"""Custom OpenAI-compatible provider adapter — connect to any OpenAI-compatible API endpoint."""

from __future__ import annotations

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["CustomOpenAIAdapter"]


class CustomOpenAIAdapter(ProviderAdapter):
    """Generic OpenAI-compatible API adapter.

    Allows connecting to any service that exposes an OpenAI-compatible
    Chat Completions API (``POST /v1/chat/completions`` with Bearer auth
    and SSE streaming).  The base URL is configured via
    ``provider_config["base_url"]`` at profile creation time.

    Typical targets include DeepSeek, Together AI, Groq, Fireworks AI,
    vLLM, and similar services.

    Since the upstream format is already Chat Completions (which is Kitty's
    internal format), this adapter is mostly passthrough.
    """

    @property
    def provider_type(self) -> str:
        return "custom_openai"

    @property
    def default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    @property
    def upstream_path(self) -> str:
        return "/chat/completions"

    @property
    def requires_custom_url(self) -> bool:
        return True

    def build_base_url(self, provider_config: dict | None) -> str:
        if provider_config and "base_url" in provider_config:
            return provider_config["base_url"]
        return self.default_base_url

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def normalize_model_name(self, model: str) -> str:
        return model

    def normalize_request(self, cc_request: dict) -> None:
        pass

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        request: dict = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
        }
        if "tools" in kwargs and kwargs["tools"]:
            request["tools"] = kwargs["tools"]
        for key in ("temperature", "top_p", "max_tokens"):
            if key in kwargs and kwargs[key] is not None:
                request[key] = kwargs[key]
        return request

    def parse_response(self, response_data: dict) -> dict:
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

    def _detect_thinking_from_messages(self, messages: list[dict]) -> bool:
        """Return True if any assistant message already has ``reasoning_content``.

        When the upstream provider defaults thinking to enabled (e.g. DeepSeek)
        but the agent does not send an explicit ``thinking`` signal, the presence
        of ``reasoning_content`` in any assistant message proves thinking is
        active and reasoning echo-back is required.
        """
        return any(msg.get("role") == "assistant" and "reasoning_content" in msg for msg in messages)

    def translate_to_upstream(self, cc_request: dict) -> dict:
        result = {k: v for k, v in cc_request.items() if k not in self._INTERNAL_KEYS}
        effort = cc_request.get("_reasoning_effort")
        thinking = cc_request.get("_thinking_enabled")
        thinking_active = (effort and effort != "none") or thinking
        if not thinking_active:
            messages = result.get("messages")
            if messages and self._detect_thinking_from_messages(messages):
                thinking_active = True
        if thinking_active:
            messages = result.get("messages")
            if messages:
                result["messages"] = self._inject_empty_reasoning_content(messages)
        return result

    def translate_from_upstream(self, raw_response: dict) -> dict:
        return raw_response

    def map_error(self, status_code: int, body: dict) -> Exception:
        if not isinstance(body, dict):
            return ProviderError(f"Custom OpenAI error {status_code}: {body}")
        error_obj = body.get("error", body)
        msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
        return ProviderError(f"Custom OpenAI error {status_code}: {msg}")
