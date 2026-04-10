"""Ollama provider adapter — OpenAI-compatible local LLM deployment."""

from __future__ import annotations

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["OllamaAdapter"]


class OllamaAdapter(ProviderAdapter):
    """Ollama API adapter.

    Ollama provides an OpenAI-compatible Chat Completions API for running
    models locally. The default endpoint is ``http://localhost:11434``,
    but this can be overridden via ``provider_config`` for remote Ollama
    instances.

    Authentication is not required for local deployments (the API key
    is ignored). For cloud/remote Ollama, authentication may be needed
    but is handled transparently through the standard ``Authorization``
    header if configured.

    Supported features:
    - Chat completions
    - Streaming
    - Tools (function calling)
    - JSON mode
    - Vision (for multimodal models)
    """

    @property
    def provider_type(self) -> str:
        return "ollama"

    @property
    def default_base_url(self) -> str:
        return "http://localhost:11434"

    @property
    def upstream_path(self) -> str:
        """Upstream API endpoint path.

        Ollama's OpenAI-compatible endpoint is at ``/v1/chat/completions``.
        """
        return "/v1/chat/completions"

    def build_base_url(self, provider_config: dict) -> str:
        """Build the base URL, allowing override via provider_config.

        Args:
            provider_config: May contain ``base_url`` to override default.

        Returns:
            Base URL string.
        """
        if provider_config and "base_url" in provider_config:
            return provider_config["base_url"]
        return self.default_base_url

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Build headers for Ollama.

        For local Ollama deployments, authentication is not required.
        The Ollama API ignores the Authorization header on localhost.

        Args:
            api_key: Ignored for local deployments.

        Returns:
            HTTP headers dict with just Content-Type.
        """
        # Ollama local API doesn't require or use authentication
        return {"Content-Type": "application/json"}

    def normalize_model_name(self, model: str) -> str:
        """Normalize model name for Ollama.

        Ollama model names are passed through unchanged. Models use
        formats like ``llama3.2``, ``mistral``, ``codellama:7b``.

        Args:
            model: Raw model name.

        Returns:
            The model name unchanged.
        """
        return model

    def normalize_request(self, cc_request: dict) -> None:
        """No-op: Ollama uses OpenAI-compatible request format."""

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        """Build a Chat Completions API request payload.

        Args:
            model: Model identifier string (e.g., ``"llama3.2"``).
            messages: Normalized message list.
            **kwargs: Additional parameters (stream, tools, temperature, etc.).

        Returns:
            Request dict in Chat Completions format.
        """
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
        """Parse an upstream Chat Completions response.

        Args:
            response_data: Parsed JSON response from Ollama.

        Returns:
            Normalized response dict with content, finish_reason, usage.

        Raises:
            ProviderError: If the response is missing 'choices' field.
        """
        choices = response_data.get("choices")
        if not choices:
            raise ProviderError(f"Ollama response missing 'choices': {response_data}")
        choice = choices[0]
        message = choice.get("message", {})
        result: dict = {
            "content": message.get("content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": response_data.get("usage", {}),
        }
        if "tool_calls" in message:
            result["tool_calls"] = message["tool_calls"]
        return result

    def translate_to_upstream(self, cc_request: dict) -> dict:
        """Translate a Chat Completions request for Ollama.

        Ollama uses OpenAI-compatible format, so this is passthrough.
        Strips internal metadata fields.

        Args:
            cc_request: Chat Completions request dict.

        Returns:
            Request dict to send to Ollama.
        """
        return {k: v for k, v in cc_request.items() if k not in (
            "_resolved_key", "_provider_config",
        )}

    def translate_from_upstream(self, raw_response: dict) -> dict:
        """Translate an Ollama response to Chat Completions format.

        Ollama returns OpenAI-compatible responses, so this is passthrough.

        Args:
            raw_response: Parsed JSON response from Ollama.

        Returns:
            Response dict in Chat Completions format.
        """
        return raw_response

    def map_error(self, status_code: int, body: dict) -> Exception:
        """Map an HTTP error to a typed exception.

        Args:
            status_code: HTTP status code.
            body: Error response body.

        Returns:
            ProviderError with details.
        """
        if not isinstance(body, dict):
            return ProviderError(f"Ollama error {status_code}: {body}")
        error_msg = body.get("error", body)
        if isinstance(error_msg, dict):
            msg = error_msg.get("message", str(error_msg))
        else:
            msg = str(error_msg)
        return ProviderError(f"Ollama error {status_code}: {msg}")
