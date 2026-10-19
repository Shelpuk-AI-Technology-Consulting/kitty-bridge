"""Declare the Claude Code subscription relay provider."""

from __future__ import annotations

from kitty.providers.anthropic import AnthropicAdapter

__all__ = ["AnthropicSubscriptionAdapter"]


class AnthropicSubscriptionAdapter(AnthropicAdapter):
    """Describe the raw Claude Code subscription relay endpoint.

    The bridge handles subscription requests before ordinary Anthropic request
    translation. This adapter therefore declares the Messages endpoint and
    native wire capability without constructing OAuth headers or accessing a
    credential value.
    """

    @property
    def provider_type(self) -> str:
        """Return the stable subscription provider identifier.

        Returns:
            The `anthropic_subscription` provider type.
        """
        return "anthropic_subscription"

    @property
    def use_native_messages(self) -> bool:
        """Declare that the provider accepts Claude Code Messages requests.

        Returns:
            Always true because the raw relay uses the Anthropic Messages wire.
        """
        return True

    def translate_to_upstream(self, cc_request: dict) -> dict:
        """Return a native Messages body unchanged instead of translating it.

        A subscription request must reach Anthropic exactly as Claude Code
        formed it; rebuilding it through the API-key translation would
        abandon the relay contract. Non-native bodies fall through to the
        inherited translation, which no production path sends.

        Args:
            cc_request: The request payload in CC format, carrying the
                ``_native_messages_request`` flag on the Messages path.

        Returns:
            The body with internal keys stripped, untranslated.
        """
        if cc_request.get("_native_messages_request"):
            result = {k: v for k, v in cc_request.items() if k not in self._INTERNAL_KEYS}
            if "messages" in result:
                result["messages"] = self._strip_internal_message_keys(result["messages"])
            return result
        return super().translate_to_upstream(cc_request)
