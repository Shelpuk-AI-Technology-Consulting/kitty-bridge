"""Pre-flight API key validation for upstream providers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import urlsplit

import aiohttp

from kitty.egress import EgressConfig, aiohttp_session_kwargs, should_bypass
from kitty.providers.base import ProviderAdapter, ProviderError

logger = logging.getLogger(__name__)

# Timeout for the validation request (seconds).
_VALIDATION_TIMEOUT = 5


@dataclass
class ValidationResult:
    """Result of an API key validation check."""

    valid: bool
    reason: str | None = None
    warning: str | None = None


def _unusable_url_result(
    provider: ProviderAdapter,
    provider_config: dict,
    exc: Exception,
) -> ValidationResult:
    """Build the failure for a profile whose base URL cannot produce a request.

    Before KBR-143 this condition was reported as ``API key for <provider> contains
    invalid characters``, which sent the user to replace a credential that was never
    the problem — the same misdirection KBR-134 was filed about, one layer in.

    Args:
        provider: The adapter whose profile is being validated.
        provider_config: The profile's provider configuration, read only for the
            ``base_url`` the message quotes back.
        exc: What resolving the URL raised.

    Returns:
        An invalid result naming the provider and the configured base URL.  The URL is
        redacted, because this reason is printed at launch and a query is where a
        gateway keeps its credentials.
    """
    # Quote the configured value rather than the composed one: it is what the user
    # would edit, and when `build_base_url` raised there is no composed URL to show.
    configured = provider_config.get("base_url") or provider.default_base_url
    return ValidationResult(
        valid=False,
        reason=(
            f"Cannot build a request URL for {provider.provider_type} from this profile's base URL "
            f"({provider.redact_url_for_display(str(configured))}): {exc}. "
            f"Run `kitty profile` to correct it."
        ),
    )


async def validate_api_key(
    provider: ProviderAdapter,
    api_key: str,
    provider_config: dict | None = None,
    egress: EgressConfig | None = None,
) -> ValidationResult:
    """Validate an API key by making a lightweight request to the upstream.

    Sends a minimal chat completions request (max_tokens=1) and checks
    the response status. 401/403 → invalid key, everything else → valid
    (to avoid false negatives from model-not-found, rate limits, etc.).

    For providers with ``use_custom_transport=True``, validation is skipped
    (they use custom auth like boto3 SigV4).

    When ``egress`` is supplied the request goes through the egress proxy, and
    a connection failure becomes fatal rather than a warning: under egress a
    dead proxy must stop the launch, not let it proceed unproxied.

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

    # Resolve the probe URL under a guard, because everything that consumes it runs
    # before the request block below and each fails misleadingly on a bad URL:
    # `build_base_url` raises for a profile it rejects, `should_bypass` parses the URL
    # again to decide on the proxy, and `aiohttp.InvalidURL` is a `ValueError` —
    # indistinguishable, in the handler below, from a key containing whitespace.
    try:
        base_url = provider.build_base_url(provider_config)
        model = provider.normalize_model_name(provider.validation_model)
        path = provider.get_upstream_path(model)
        url = provider.compose_upstream_url(base_url, path)
        if not urlsplit(url).netloc:
            raise ValueError("the composed URL has no host")
    except (ValueError, ProviderError) as exc:
        return _unusable_url_result(provider, provider_config, exc)

    headers = provider.build_upstream_headers(api_key)

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "stream": False,
    }

    timeout = aiohttp.ClientTimeout(total=_VALIDATION_TIMEOUT)
    # A loopback or LAN provider (local Ollama, a custom endpoint on the LAN)
    # must not be tunnelled: a rented proxy cannot reach the caller's own
    # network. Such traffic never leaves the machine, so it is not a leak.
    effective_egress = None if egress is None or should_bypass(url) else egress
    session_kwargs = aiohttp_session_kwargs(effective_egress) if effective_egress is not None else {}

    try:
        async with (
            aiohttp.ClientSession(timeout=timeout, **session_kwargs) as session,
            session.post(
                url,
                json=body,
                headers=headers,
            ) as resp,
        ):
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
        warning = f"API key validation timed out for {provider.provider_type} — proceeding anyway"
        logger.warning(warning)
        return ValidationResult(valid=True, warning=warning)
    except aiohttp.ClientConnectorError as exc:
        if effective_egress is not None:
            # Fail closed: proceeding here would mean either no connectivity at
            # all, or a broken proxy that a later request might bypass.
            return ValidationResult(
                valid=False,
                reason=(
                    f"Cannot reach {provider.provider_type} through the egress proxy "
                    f"{effective_egress.masked()}: {exc}"
                ),
            )
        warning = f"Cannot reach {provider.provider_type} for key validation: {exc} — proceeding anyway"
        logger.warning(warning)
        return ValidationResult(valid=True, warning=warning)
    except ValueError:
        # aiohttp rejects headers containing newlines/carriage returns.
        # Usually means the stored API key has trailing whitespace.
        return ValidationResult(
            valid=False,
            reason=(
                f"API key for {provider.provider_type} contains invalid characters "
                f"(newlines or whitespace).  Re-run `kitty setup` to re-enter your key."
            ),
        )
    except (aiohttp.ClientError, OSError) as exc:
        if effective_egress is not None:
            return ValidationResult(
                valid=False,
                reason=(
                    f"Network error reaching {provider.provider_type} through the egress proxy "
                    f"{effective_egress.masked()}: {exc}"
                ),
            )
        # Client-side HTTP errors (redirect, payload, etc.) — proceed
        warning = f"Key validation network error for {provider.provider_type}: {exc} — proceeding anyway"
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
        reason=(f"API key rejected by {provider.provider_type}: {error_msg}"),
    )


def _extract_error_message(body: dict | list | str) -> str:
    """Extract a human-readable error message from an upstream error."""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message", str(error)))
        if isinstance(error, str):
            return error
    return str(body)
