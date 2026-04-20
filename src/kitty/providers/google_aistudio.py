"""Google AI Studio provider adapter."""

from __future__ import annotations

import re

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["GoogleAIStudioAdapter"]

_AISTUDIO_PREFIX = re.compile(r"^google_aistudio/", re.IGNORECASE)


class GoogleAIStudioAdapter(ProviderAdapter):
    """Google AI Studio adapter (OpenAI-compatible).

    Google AI Studio provides an OpenAI-compatible Chat Completions API at
    ``https://generativelanguage.googleapis.com/v1beta/openai/``.
    API keys are obtained from https://aistudio.google.com/apikey
    """

    @property
    def provider_type(self) -> str:
        return "google_aistudio"

    @property
    def default_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta/openai"

    def normalize_model_name(self, model: str) -> str:
        """Strip provider prefix if present (e.g. 'google_aistudio/gemini-2.5-flash')."""
        stripped = _AISTUDIO_PREFIX.sub("", model, count=1)
        return stripped if stripped else model

    @property
    def validation_model(self) -> str:
        """gemini-2.5-flash is a stable model accepted by Google AI Studio API."""
        return "gemini-2.5-flash"

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
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        result: dict = {
            "content": message.get("content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": response_data.get("usage") or {},
        }
        if "tool_calls" in message:
            result["tool_calls"] = message["tool_calls"]
        return result

    def map_error(self, status_code: int, body: dict) -> ProviderError:
        error_dict = body.get("error", {})
        if isinstance(error_dict, dict):
            error_msg = error_dict.get("message", "Unknown error")
        else:
            error_msg = error_dict or "Unknown error"
        return ProviderError(f"Google AI Studio error {status_code}: {error_msg}")
