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
