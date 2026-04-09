"""Pre-flight API key validation for upstream providers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import aiohttp

from kitty.providers.base import ProviderAdapter

logger = logging.getLogger(__name__)

# Timeout for the validation request (seconds).
_VALIDATION_TIMEOUT = 5


@dataclass
class ValidationResult:
    """Result of an API key validation check."""

    valid: bool
    reason: str | None = None
    warning: str | None = None


async def validate_api_key(
    provider: ProviderAdapter,
    api_key: str,
    provider_config: dict | None = None,
) -> ValidationResult:
    """Validate an API key by making a lightweight request to the upstream.

    Sends a minimal chat completions request (max_tokens=1) and checks
    the response status. 401/403 → invalid key, everything else → valid
    (to avoid false negatives from model-not-found, rate limits, etc.).

    For providers with ``use_custom_transport=True``, validation is skipped
    (they use custom auth like boto3 SigV4).

    Note: The validation request uses the Chat Completions format
    (``/chat/completions``). This works for all standard providers
    (ZAI, OpenRouter, MiniMax, Novita, OpenAI). Providers with custom
    transports (Anthropic, Bedrock, Vertex) are skipped automatically.
    If a future provider uses a non-CC endpoint, it should override
    ``use_custom_transport=True`` or this function should be extended.
    """
    if provider.use_custom_transport:
        logger.debug(
            "Skipping validation for custom-transport provider %s",
            provider.provider_type,
        )
        return ValidationResult(valid=True)

    provider_config = provider_config or {}
    base_url = provider.build_base_url(provider_config).rstrip("/")
    model = provider.normalize_model_name("test")
    path = provider.get_upstream_path(model)
    url = f"{base_url}{path}"
    headers = provider.build_upstream_headers(api_key)

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "stream": False,
    }

    timeout = aiohttp.ClientTimeout(total=_VALIDATION_TIMEOUT)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session, session.post(
            url, json=body, headers=headers,
        ) as resp:
            if resp.status in (401, 403):
                return await _handle_auth_failure(provider, resp)
            # Any other status means the key is accepted at the
            # auth layer — the error is elsewhere.
            logger.debug(
                "API key validation passed for %s (status %d)",
                provider.provider_type,
                resp.status,
            )
            return ValidationResult(valid=True)
    except asyncio.TimeoutError:
        warning = (
            f"API key validation timed out for {provider.provider_type}"
            " — proceeding anyway"
        )
        logger.warning(warning)
        return ValidationResult(valid=True, warning=warning)
    except aiohttp.ClientConnectorError as exc:
        warning = (
            f"Cannot reach {provider.provider_type} for key validation:"
            f" {exc} — proceeding anyway"
        )
        logger.warning(warning)
        return ValidationResult(valid=True, warning=warning)
    except (aiohttp.ClientError, OSError) as exc:
        # Client-side HTTP errors (redirect, payload, etc.) — proceed
        warning = (
            f"Key validation network error for {provider.provider_type}:"
            f" {exc} — proceeding anyway"
        )
        logger.warning(warning)
        return ValidationResult(valid=True, warning=warning)


async def _handle_auth_failure(
    provider: ProviderAdapter,
    resp: aiohttp.ClientResponse,
) -> ValidationResult:
    """Handle a 401/403 response from the upstream provider."""
    try:
        error_body = await resp.json()
    except Exception:
        error_body = {}
    error_msg = _extract_error_message(error_body)
    logger.warning(
        "API key validation failed for %s: %d %s",
        provider.provider_type,
        resp.status,
        error_msg,
    )
    return ValidationResult(
        valid=False,
        reason=(
            f"API key rejected by {provider.provider_type}: {error_msg}"
        ),
    )


def _extract_error_message(body: dict | list | str) -> str:
    """Extract a human-readable error message from an upstream error."""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return error.get("message", str(error))
        if isinstance(error, str):
            return error
    return str(body)
