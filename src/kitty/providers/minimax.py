"""MiniMax provider adapter — global and CN region endpoints."""

from __future__ import annotations

import re

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["MiniMaxAdapter", "minimax_embedded_error_status"]

_GLOBAL_URL = "https://api.minimax.io/v1"
_CN_URL = "https://api.minimaxi.com/v1"

_MINIMAX_PREFIX = re.compile(r"^minimax/", re.IGNORECASE)


# MiniMax reports some failures with HTTP 200 and the real outcome in
# ``base_resp.status_code`` (0 is success), next to ``choices``/``content``
# that is null. Codes from MiniMax's published error-code table; anything else
# non-zero is an upstream fault. KBR-328: such a reply used to be translated as
# a success, iterate the null ``content`` and crash the bridge.
_BASE_RESP_STATUS: dict[int, int] = {
    1002: 429,  # RPM rate limit
    1039: 429,  # TPM rate limit
    1004: 401,  # authentication failed
    1008: 402,  # insufficient balance
    1027: 400,  # output content flagged
    2013: 400,  # invalid parameters
}
_BASE_RESP_DEFAULT_STATUS = 502


def minimax_embedded_error_status(response_data: object) -> int | None:
    """Map a MiniMax ``base_resp`` error inside a 2xx reply to an HTTP status.

    Shared by both MiniMax adapters: the OpenAI-compatible endpoint and the
    Anthropic-compatible Token Plan endpoint return the same envelope.

    Args:
        response_data: The parsed reply body.

    Returns:
        The equivalent HTTP status, or ``None`` when there is no ``base_resp``
        or its ``status_code`` is 0 (success) or not an integer.
    """
    if not isinstance(response_data, dict):
        return None
    base_resp = response_data.get("base_resp")
    if not isinstance(base_resp, dict):
        return None
    code = base_resp.get("status_code")
    if not isinstance(code, int) or isinstance(code, bool) or code == 0:
        return None
    return _BASE_RESP_STATUS.get(code, _BASE_RESP_DEFAULT_STATUS)


class MiniMaxAdapter(ProviderAdapter):
    """MiniMax API adapter with global/CN region support."""

    @property
    def provider_type(self) -> str:
        return "minimax"

    @property
    def default_base_url(self) -> str:
        return _GLOBAL_URL

    def build_base_url(self, provider_config: dict | None = None) -> str:
        """Return MiniMax base URL, honoring CN-region provider_config.

        A configured ``base_url`` is normalised through
        :meth:`ProviderAdapter._strip_endpoint_suffix`, so a user who pastes
        the full documented endpoint (``https://api.minimax.io/v1/chat/completions``)
        composes to that endpoint rather than to a doubled path (KBR-134,
        KBR-157).  The strip runs **after** the ``region == "cn"`` switch and
        the ``http(s)://`` scheme check — moving it in front would let a
        region switch eat a pasted endpoint meant for the global URL, or turn
        a malformed scheme into a ``urlsplit`` error inside the helper.

        Args:
            provider_config: The profile's provider configuration.  ``region
                == "cn"`` wins over a configured ``base_url``; an absent or
                empty ``base_url`` returns :attr:`default_base_url`; a
                non-``http(s)://`` value raises.

        Returns:
            The MiniMax API root — the CN URL when ``region == "cn"`` is
            set, the global URL by default, or the configured ``base_url``
            with a redundant trailing ``/chat/completions`` stripped.

        Raises:
            ValueError: When the configured ``base_url`` is non-empty and
                does not start with ``http://`` or ``https://``.
        """
        if provider_config and provider_config.get("region") == "cn":
            return _CN_URL
        if provider_config and provider_config.get("base_url"):
            url = str(provider_config["base_url"]).strip().rstrip("/")
            if not url.startswith(("http://", "https://")):
                raise ValueError("provider_config.base_url must be a non-empty http:// or https:// URL")
            # KBR-157: strip after the scheme check and after the region switch
            # (ticket decision 1).
            return self._strip_endpoint_suffix(url, self.upstream_path)
        return self.default_base_url

    def embedded_error_status(self, response_data: object) -> int | None:
        return minimax_embedded_error_status(response_data)

    def normalize_model_name(self, model: str) -> str:
        stripped = _MINIMAX_PREFIX.sub("", model, count=1)
        return stripped if stripped else model

    def normalize_request(self, cc_request: dict) -> None:
        """Add MiniMax-specific parameters for better compatibility.

        Enables ``reasoning_split`` so the model returns thinking content
        in a separate ``reasoning_details`` field instead of embedding
        ``<اخل>`` tags in ``content``.
        """
        cc_request["reasoning_split"] = True

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        request: dict = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
        }
        # F15: base_url is NOT included in the CC request body — it is
        # handled at the HTTP layer via build_base_url / provider_config.
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
            "usage": response_data.get("usage", {}),
        }
        if "tool_calls" in message:
            result["tool_calls"] = message["tool_calls"]
        return result

    def map_error(self, status_code: int, body: dict) -> Exception:
        error_msg = body.get("error", {}) if isinstance(body.get("error"), dict) else body.get("error", "Unknown error")
        return ProviderError(f"MiniMax error {status_code}: {error_msg}")
