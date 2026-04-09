"""Provider adapter interface — stateless request/response builders for upstream APIs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Awaitable


class ProviderAdapter(ABC):
    """Interface for upstream Chat Completions API providers.

    Implementations are stateless: they build request payloads and parse
    response payloads but do not perform HTTP calls themselves.
    """

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Unique provider type identifier (e.g. ``"zai_regular"``)."""

    @property
    @abstractmethod
    def default_base_url(self) -> str:
        """Default upstream API base URL for this provider."""

    def normalize_model_name(self, model: str) -> str:
        """Normalize a model name for this provider.

        Strips OpenRouter-style prefixes (e.g. ``"minimax/"``) when using
        a direct provider.  OpenRouter itself overrides this to pass names
        through unchanged.

        Args:
            model: Raw model name, possibly with an OpenRouter prefix.

        Returns:
            The provider-native model name.
        """
        return model

    def normalize_request(self, cc_request: dict) -> None:
        """Normalize a Chat Completions request for this provider.

        Mutates ``cc_request`` in place to add or adjust provider-specific
        parameters.  Default implementation does nothing.

        Args:
            cc_request: Chat Completions request dict to normalize.
        """

    @property
    def upstream_path(self) -> str:
        """Upstream API endpoint path appended to ``default_base_url``.

        Default is ``"/chat/completions"``.  Override for providers that use
        a different endpoint (e.g. Anthropic Messages API).
        """
        return "/chat/completions"

    def build_base_url(self, provider_config: dict) -> str:
        """Build the base URL using provider-specific configuration.

        Default returns ``default_base_url``.  Override for providers
        where the URL depends on config parameters (e.g. Vertex AI needs
        project_id and location in the URL).

        Args:
            provider_config: Provider-specific configuration dict.

        Returns:
            Base URL string (without the endpoint path).
        """
        return self.default_base_url

    def get_upstream_path(self, model: str) -> str:
        """Build the upstream path for a specific model.

        Default returns ``upstream_path`` (ignores model).  Override for
        providers where the model is part of the URL path (e.g. Azure OpenAI
        uses ``/openai/deployments/{deployment-id}/chat/completions``).

        Args:
            model: The normalized model identifier.

        Returns:
            URL path to append to ``default_base_url``.
        """
        return self.upstream_path

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Build HTTP headers for the upstream request.

        Default uses ``Authorization: Bearer``.  Override for providers
        with different auth schemes (e.g. ``x-api-key`` for Anthropic).

        Args:
            api_key: Resolved API key for the upstream provider.
        """
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def translate_to_upstream(self, cc_request: dict) -> dict:
        """Translate a normalized CC request into the upstream wire format.

        Default returns the request unchanged (passthrough).  Override for
        providers whose upstream API differs from Chat Completions.

        Args:
            cc_request: Fully normalized Chat Completions request dict.

        Returns:
            Dict to send as JSON body to the upstream endpoint.
        """
        return cc_request

    def translate_from_upstream(self, raw_response: dict) -> dict:
        """Translate an upstream JSON response into Chat Completions format.

        Default returns the response unchanged (passthrough).  Override for
        providers whose response format differs from Chat Completions.

        Args:
            raw_response: Parsed JSON response from the upstream provider.

        Returns:
            Dict in Chat Completions response format.
        """
        return raw_response

    def translate_upstream_stream_event(self, raw_bytes: bytes) -> list[bytes]:
        """Translate a raw upstream SSE chunk into downstream SSE chunks.

        Default wraps the raw bytes in a single-element list (passthrough).
        Override for providers whose SSE event format differs.

        Args:
            raw_bytes: Raw bytes received from the upstream SSE stream.

        Returns:
            List of raw byte chunks to forward to the downstream client.
        """
        return [raw_bytes]

    @abstractmethod
    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        """Build a Chat Completions API request payload.

        Args:
            model: Model identifier string.
            messages: Normalized message list (role, content, tool_calls, etc.).
            **kwargs: Additional request parameters (stream, tools, temperature, etc.).
        """

    @abstractmethod
    def parse_response(self, response_data: dict) -> dict:
        """Parse an upstream Chat Completions response into a normalized dict."""

    @abstractmethod
    def map_error(self, status_code: int, body: dict) -> Exception:
        """Map an HTTP error status and body to a typed exception."""

    # ── Custom transport ─────────────────────────────────────────────────

    @property
    def use_custom_transport(self) -> bool:
        """Whether this provider handles its own HTTP transport.

        When True, the bridge delegates upstream HTTP calls to
        ``make_request`` / ``stream_request`` instead of using its
        own aiohttp client.  Providers that require a specialized
        HTTP client (e.g. AWS SigV4 via boto3) should override
        this to return True.
        """
        return False

    async def make_request(self, cc_request: dict) -> dict:
        """Perform a non-streaming upstream request using custom transport.

        Override when ``use_custom_transport`` is True.  The ``cc_request``
        is a fully normalized Chat Completions request dict — the provider
        must translate it to the upstream format, make the HTTP call, and
        return a Chat Completions response dict.

        Raises:
            NotImplementedError: If not overridden by a custom-transport provider.
        """
        raise NotImplementedError("Custom transport provider must implement make_request()")

    async def stream_request(
        self,
        cc_request: dict,
        write: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Perform a streaming upstream request using custom transport.

        Override when ``use_custom_transport`` is True.  The provider must
        translate the request, open a streaming connection, and call
        ``write`` with CC-format SSE chunks as they arrive.

        Args:
            cc_request: Fully normalized Chat Completions request dict.
            write: Async callback to send bytes to the downstream client.

        Raises:
            NotImplementedError: If not overridden by a custom-transport provider.
        """
        raise NotImplementedError("Custom transport provider must implement stream_request()")


class ProviderError(Exception):
    """Base exception for upstream provider adapter errors."""


__all__ = ["ProviderAdapter", "ProviderError"]
