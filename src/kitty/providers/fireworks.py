"""Fireworks AI provider adapter.

Fireworks provides an OpenAI-compatible Chat Completions API at
``https://api.fireworks.ai/inference/v1``.  Their Fire Pass product
is a flat-rate subscription for personal/development use.
"""

from __future__ import annotations

from kitty.providers.base import ProviderAdapter, ProviderError


class FireworksAdapter(ProviderAdapter):
    """Fireworks AI adapter (OpenAI-compatible).

    Fireworks provides the canonical Chat Completions API format.
    No request normalization or model name stripping is needed.

    Fire Pass plans require the full model path (e.g.
    ``accounts/fireworks/routers/kimi-k2p5-turbo``) — the model name
    must never be altered or truncated.
    """

    @property
    def provider_type(self) -> str:
        return "fireworks"

    @property
    def default_base_url(self) -> str:
        return "https://api.fireworks.ai/inference/v1"

    def normalize_model_name(self, model: str) -> str:
        """Pass through model names unchanged.

        Fireworks uses full model paths that must not be altered
        (e.g. ``accounts/fireworks/routers/kimi-k2p5-turbo``).
        """
        return model

    def normalize_request(self, cc_request: dict) -> None:
        """Cap ``max_tokens`` at 4096 for non-streaming requests.

        Fireworks rejects non-streaming requests with ``max_tokens > 4096``.
        """
        if not cc_request.get("stream"):
            max_tokens = cc_request.get("max_tokens")
            if isinstance(max_tokens, int) and max_tokens > 4096:
                cc_request["max_tokens"] = 4096

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        request: dict = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
        }
        if "tools" in kwargs and kwargs["tools"]:
            request["tools"] = kwargs["tools"]
        if "base_url" in kwargs and kwargs["base_url"]:
            request["base_url"] = kwargs["base_url"]
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
            "usage": response_data.get("usage", {}),
        }
        if "tool_calls" in message:
            result["tool_calls"] = message["tool_calls"]
        return result

    def map_error(self, status_code: int, body: dict) -> Exception:
        error_msg = body.get("error", {}) if isinstance(body.get("error"), dict) else body.get("error", "Unknown error")
        return ProviderError(f"Fireworks error {status_code}: {error_msg}")
