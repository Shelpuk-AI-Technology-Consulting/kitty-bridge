"""OpenAI ChatGPT subscription provider — Codex backend via OAuth.

This provider authenticates via OpenAI's Codex OAuth flow, which produces
an access_token JWT that works against the Codex backend at
chatgpt.com/backend-api/codex/responses (Responses API format).

The access_token JWT is NOT a valid Bearer token for api.openai.com — it
only works with the ChatGPT backend.  The token exchange (id_token → API key)
is attempted but treated as best-effort; for org accounts without a Platform
API org mapping, the exchange fails and we fall back to using the access_token
directly against the Codex backend.

Token lifecycle (refresh, re-exchange) is handled by OAuthSession in
kitty.auth.oauth_session.

Transport: curl_cffi (AsyncSession) for Codex backend requests.  curl_cffi
uses libcurl under the hood with HTTP/2 and a Chrome TLS fingerprint
(impersonate="chrome136") that Cloudflare accepts, matching the legitimate-
client behavior of the real Codex CLI (reqwest + rustls).  A long-lived
session automatically persists Cloudflare cookies across requests.  OAuth
token refresh uses a short-lived aiohttp session (auth.openai.com has no
CF issues).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import platform
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import curl_cffi.requests

from kitty.auth.oauth_session import OAuthRefreshFailed, OAuthSession
from kitty.cloudflare import get_cloudflare_signature, is_cloudflare_block
from kitty.providers.base import ProviderError

# Avoid circular import — only need the parent class methods
from kitty.providers.openai import OpenAIAdapter

__all__ = ["OpenAISubscriptionAdapter"]

logger = logging.getLogger(__name__)

# Codex backend endpoint for ChatGPT subscription access
_CODEX_BACKEND_URL = "https://chatgpt.com/backend-api/codex/responses"
# Timeout: (connect_seconds, read_seconds) — matches Codex CLI reqwest defaults
_CODEX_TIMEOUT = (30, 300)
_STREAM_RECV_ERROR_RETRIES = 2
# Codex CLI retry policy — matches model-provider-info/src/lib.rs:248-252 and retry.rs.
_CODEX_RETRY_MAX_ATTEMPTS = 4  # 5 total tries (0..=4)
_CODEX_RETRY_BASE_DELAY_MS = 200


def _codex_backoff(attempt: int) -> float:
    """Exponential backoff with jitter matching Codex CLI's retry.rs.

    Returns delay in seconds.  ``attempt`` is 1-indexed (first retry = 1).
    Formula: ``base_ms * 2^attempt * jitter(0.9..1.1)``.
    """
    exp = 2 ** (attempt - 1)
    millis = _CODEX_RETRY_BASE_DELAY_MS * exp
    jitter = random.uniform(0.9, 1.1)  # noqa: S311
    return (millis * jitter) / 1000.0
# Codex CLI version sent in the version header.
# The checked-in reference workspace has 0.0.0 (dev placeholder), but
# the real released Codex CLI version is used in production builds.
_CODEX_CLI_VERSION = "0.128.0"

# curl_cffi TLS impersonation target — matches the browser-like TLS fingerprint
# that Codex CLI's reqwest+rustls produces.  Without impersonate=, curl_cffi
# uses libcurl's default fingerprint which Cloudflare flags as a bot.
_CODEX_IMPERSONATE = "chrome136"

# Custom CA certificate env vars — matches Codex CLI's custom_ca.rs.
# CODEX_CA_CERTIFICATE takes precedence over SSL_CERT_FILE.
# Empty values are treated as unset.
_CODEX_CA_CERT_ENV = "CODEX_CA_CERTIFICATE"
_SSL_CERT_FILE_ENV = "SSL_CERT_FILE"


def _resolve_ca_cert_path() -> str | None:
    """Resolve custom CA certificate path from environment variables.

    Matches Codex CLI's precedence: ``CODEX_CA_CERTIFICATE`` wins over
    ``SSL_CERT_FILE``.  Empty values are treated as unset.
    """
    path = os.environ.get(_CODEX_CA_CERT_ENV, "").strip()
    if path:
        return path
    path = os.environ.get(_SSL_CERT_FILE_ENV, "").strip()
    if path:
        return path
    return None

# Cloudflare cookie allowlist — matches Codex CLI's chatgpt_cloudflare_cookies.rs.
# Only these cookie names are preserved for ChatGPT hosts; all others are stripped
# to avoid triggering Cloudflare bot detection with stale or unrelated cookies.
_CF_COOKIE_ALLOWLIST = frozenset({
    "__cf_bm", "__cflb", "__cfruid", "__cfseq",
    "__cfwaitingroom", "_cfuvid", "cf_clearance",
    "cf_ob_info", "cf_use_ob",
})
_CF_COOKIE_PREFIX = "cf_chl_"
# ChatGPT hosts for CF cookie filtering — matches Codex CLI's chatgpt_hosts.rs.
# Exact matches + subdomain suffixes for chatgpt.com and chatgpt-staging.com.
_CF_COOKIE_EXACT_HOSTS = frozenset({"chatgpt.com", "chat.openai.com", "chatgpt-staging.com"})
_CF_COOKIE_SUBDOMAIN_SUFFIXES = (".chatgpt.com", ".chatgpt-staging.com")


def _is_chatgpt_host(domain: str) -> bool:
    """Check if a cookie domain is a ChatGPT host (Codex CLI parity).

    Matches Codex CLI's ``is_allowed_chatgpt_host`` in ``chatgpt_hosts.rs``:
    exact match against ``chatgpt.com``, ``chat.openai.com``, ``chatgpt-staging.com``,
    or subdomain suffix match for ``.chatgpt.com`` / ``.chatgpt-staging.com``.
    """
    lower = domain.lower()
    if lower in _CF_COOKIE_EXACT_HOSTS:
        return True
    return any(lower.endswith(suffix) for suffix in _CF_COOKIE_SUBDOMAIN_SUFFIXES)


def _convert_content_types(items: list) -> list:
    """Ensure content types match the role in Responses API format.

    The Codex backend validates content types strictly:
    - User messages must use ``input_text``
    - Assistant messages must use ``output_text``
    """
    converted = []
    for item in items:
        if not isinstance(item, dict):
            converted.append(item)
            continue
        new_item = dict(item)
        role = new_item.get("role", "")
        content = new_item.get("content")
        if isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    if role == "assistant" and part_type == "input_text":
                        new_parts.append({**part, "type": "output_text"})
                    elif role == "user" and part_type == "output_text":
                        new_parts.append({**part, "type": "input_text"})
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            new_item["content"] = new_parts
        converted.append(new_item)
    return converted


class OpenAISubscriptionAdapter(OpenAIAdapter):
    """OpenAI ChatGPT subscription adapter using the Codex backend.

    Uses OAuth (Codex PKCE flow) to obtain an access_token JWT.  Routes
    requests to chatgpt.com/backend-api/codex/responses (Responses API).

    Transport uses curl_cffi AsyncSession to match the real Codex CLI's
    HTTP behavior (reqwest + rustls, HTTP/2, Chrome TLS fingerprint).
    """

    provider_type = "openai_subscription"

    def __init__(self) -> None:
        self._curl_session_instance: curl_cffi.requests.AsyncSession | None = None

    @property
    def requires_oauth(self) -> bool:
        return True

    @property
    def default_base_url(self) -> str:
        # NOTE: This is the *upstream* URL for the Codex backend, not
        # api.openai.com.  The JWT from Codex OAuth only works here.
        return _CODEX_BACKEND_URL

    @property
    def use_custom_transport(self) -> bool:
        return True

    @property
    def _curl_session(self) -> curl_cffi.requests.AsyncSession:
        """Long-lived curl_cffi session for Codex backend requests.

        Lazily created, never explicitly closed during normal operation.
        Automatically persists Cloudflare cookies across requests.
        Not using ``async with`` avoids the curl_cffi segfault (issue #675)
        that occurs when closing a session with an active SSE stream.
        """
        if self._curl_session_instance is None:
            ca_path = _resolve_ca_cert_path()
            kwargs: dict = {"impersonate": _CODEX_IMPERSONATE}
            if ca_path:
                kwargs["verify"] = ca_path
            self._curl_session_instance = curl_cffi.requests.AsyncSession(**kwargs)
            logger.debug(
                "Created curl_cffi session with impersonate=%s, ca_path=%s",
                _CODEX_IMPERSONATE, ca_path,
            )
            # Defensive: strip any non-CF cookies at session init
            self._filter_cloudflare_cookies(self._curl_session_instance.cookies)
        return self._curl_session_instance

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_account_id(id_token: str) -> str | None:
        """Extract chatgpt_account_id from the JWT payload (no sig verify)."""
        try:
            payload_b64 = id_token.split(".")[1]
            # Add padding (modulo to avoid over-padding when already aligned)
            missing = (4 - len(payload_b64) % 4) % 4
            payload_b64 += "=" * missing
            payload = json.loads(__import__("base64").urlsafe_b64decode(payload_b64))
            auth_ns = payload.get("https://api.openai.com/auth", {})
            return auth_ns.get("chatgpt_account_id")
        except Exception:
            return None

    @staticmethod
    def _build_user_agent() -> str:
        """Build a User-Agent string matching the Codex CLI format.

        Codex CLI format: ``codex_cli_rs/<version> (<os_type> <os_version>; <arch>)``

        Matches ``get_codex_user_agent()`` in
        ``codex-rs/login/src/auth/default_client.rs``.
        """
        from kitty import __version__

        os_type = platform.system()
        os_version = platform.release()
        arch = platform.machine()
        return f"codex_cli_rs/{__version__} ({os_type} {os_version}; {arch})"

    def _build_codex_headers(
        self, access_token: str, id_token: str
    ) -> dict[str, str]:
        """Build headers matching the Codex CLI (reqwest + rustls).

        Matches the headers sent by the real Codex CLI binary:
        - User-Agent: codex_cli_rs format (not a browser)
        - Accept: text/event-stream (Codex backend requires streaming)
        - No Origin/Referer (not a browser request)
        - Authorization: Bearer (from OAuth)
        - ChatGPT-Account-ID: from JWT (if present)

        NOTE: Do NOT set ``originator: codex_cli_rs`` — it triggers strict
        tool validation that only allows Codex CLI's built-in tools.
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {access_token}",
            "User-Agent": self._build_user_agent(),
            "version": _CODEX_CLI_VERSION,
        }
        account_id = self._extract_account_id(id_token)
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id
        return headers

    @staticmethod
    def _load_session(cc_request: dict) -> OAuthSession:
        """Load the OAuthSession from the resolved key file path."""
        key = cc_request.get("_resolved_key")
        if not key:
            raise ProviderError("Missing OAuth session file path (_resolved_key)")
        session_file = Path(key)
        if not session_file.exists():
            raise ProviderError(f"OAuth session file not found: {session_file}")
        return OAuthSession.load(session_file)

    @staticmethod
    def _handle_curl_error(exc: Exception) -> ProviderError:
        """Map curl_cffi transport exceptions to ProviderError."""
        exc_msg = str(exc).lower()
        if "timed out" in exc_msg or "timeout" in exc_msg:
            return ProviderError(f"Codex backend request timed out: {exc}")
        if "connection" in exc_msg:
            return ProviderError(f"Codex backend connection failed: {exc}")
        return ProviderError(f"Codex backend request failed: {exc}")

    @staticmethod
    def _is_transient_stream_error(exc: Exception) -> bool:
        """Check if a stream error is transient and worth retrying.

        curl error 56 (RECV_ERROR / "Connection closed abruptly") is the
        most common transient error for SSE streams — the server resets the
        connection mid-stream, often due to idle timeouts or load balancer
        teardown.  The Codex CLI (reqwest) retries these automatically.
        """
        msg = str(exc).lower()
        return "curl: (56)" in msg or "connection closed" in msg or "recv error" in msg

    @staticmethod
    def _filter_cloudflare_cookies(cookies: curl_cffi.requests.Cookies) -> int:
        """Remove non-Cloudflare cookies for ChatGPT hosts from the cookie jar.

        Codex CLI uses a strict allowlist of known Cloudflare cookie names
        (``__cf_bm``, ``cf_clearance``, etc.) and discards everything else
        for ``chatgpt.com``, ``chat.openai.com``, ``chatgpt-staging.com``,
        and their subdomains.  This prevents stale session or auth cookies
        from triggering Cloudflare bot detection.

        Args:
            cookies: The ``curl_cffi`` Cookies object (wraps ``http.cookiejar.CookieJar``).

        Returns:
            Number of cookies removed.
        """
        jar = cookies.jar
        to_remove = []
        for cookie in jar:
            if not _is_chatgpt_host(cookie.domain):
                continue
            if cookie.name in _CF_COOKIE_ALLOWLIST or cookie.name.startswith(_CF_COOKIE_PREFIX):
                continue
            to_remove.append(cookie)

        for cookie in to_remove:
            jar.clear(cookie.domain, cookie.path, cookie.name)

        if to_remove:
            logger.debug(
                "Filtered %d non-CF cookies from chatgpt hosts: %s",
                len(to_remove),
                [c.name for c in to_remove],
            )

        return len(to_remove)

    @staticmethod
    def _log_cf_cookies(cookies: curl_cffi.requests.Cookies) -> None:
        """Log which Cloudflare cookies are present for ChatGPT hosts."""
        jar = cookies.jar
        cf_cookies = [
            f"{c.name}={c.value[:8]}..." if len(c.value) > 8 else f"{c.name}={c.value}"
            for c in jar
            if _is_chatgpt_host(c.domain)
            and (c.name in _CF_COOKIE_ALLOWLIST or c.name.startswith(_CF_COOKIE_PREFIX))
        ]
        if cf_cookies:
            logger.debug("CF cookies present for chatgpt hosts (%d): %s", len(cf_cookies), cf_cookies)
        else:
            logger.debug("No CF cookies present for chatgpt hosts")

    _ALLOWED_RESPONSES_PARAMS = frozenset({
        "model", "stream", "store", "instructions", "input",
        "tools", "tool_choice", "parallel_tool_calls", "include",
        "reasoning",
    })

    @staticmethod
    def _prepare_responses_body(original_body: dict) -> dict:
        """Clean a Responses API request for the Codex backend.

        The Codex backend uses strict parameter validation — it rejects
        any parameter not in the Codex CLI's allowlist (e.g.
        ``max_output_tokens``, ``temperature``, ``strict`` on tools).
        This method builds a clean body from only the allowed fields.
        """
        # Log dropped parameters so users understand why settings don't apply
        dropped = set(original_body) - OpenAISubscriptionAdapter._ALLOWED_RESPONSES_PARAMS
        if dropped:
            logger.debug("Codex backend unsupported parameters (dropped): %s", sorted(dropped))

        # Only pass parameters that the Codex backend accepts.
        # Additional parameters cause 400 "Unsupported parameter: X".
        body: dict = {
            "model": original_body.get("model", "gpt-5.4"),
            "stream": True,
            "store": False,
        }

        if original_body.get("instructions"):
            body["instructions"] = original_body["instructions"]
        if original_body.get("input"):
            # Convert input_text → output_text in content parts
            # (Codex backend only supports output_text and refusal)
            body["input"] = _convert_content_types(original_body["input"])
        if original_body.get("tools"):
            # Strip ``strict`` — the Codex backend rejects it on tools
            tools = []
            for tool in original_body["tools"]:
                t = {k: v for k, v in tool.items() if k != "strict"}
                tools.append(t)
            body["tools"] = tools
        if original_body.get("tool_choice"):
            body["tool_choice"] = original_body["tool_choice"]
        if original_body.get("parallel_tool_calls") is not None:
            body["parallel_tool_calls"] = original_body["parallel_tool_calls"]
        if original_body.get("include"):
            body["include"] = original_body["include"]

        # Reasoning effort: pass through from original body or inject from metadata
        if original_body.get("reasoning"):
            body["reasoning"] = original_body["reasoning"]
        elif original_body.get("_reasoning_effort") and original_body["_reasoning_effort"] != "none":
            body["reasoning"] = {"effort": original_body["_reasoning_effort"]}

        return body

    # ── Custom transport ──────────────────────────────────────────────────

    async def make_request(self, cc_request: dict) -> dict:
        """Handle a non-streaming request via the Codex backend.

        The Codex backend requires streaming — ``stream: false`` is rejected
        with ``"Stream must be set to true"``.  We always send with
        ``stream: true`` and collect all SSE events into a single response.

        401 auth recovery matches Codex CLI's ``UnauthorizedRecovery``:
        1. Reload session from disk and retry
        2. Force-refresh OAuth token and retry
        3. If still 401, raise ``ProviderError``
        """
        original_body = cc_request.get("_original_body")
        if original_body:
            resp_body = self._prepare_responses_body(original_body)
            if "reasoning" not in resp_body:
                effort = cc_request.get("_reasoning_effort")
                if effort and effort != "none":
                    resp_body["reasoning"] = {"effort": effort}
        else:
            resp_body = self._cc_to_responses(cc_request)

        # Codex backend requires streaming
        resp_body["stream"] = True

        # Auth recovery loop (Codex CLI: UnauthorizedRecovery state machine).
        # Step 0: initial attempt, Step 1: reload from disk, Step 2: force-refresh.
        import aiohttp

        resp: object | None = None
        got_401 = False
        async with aiohttp.ClientSession() as oauth_http:
            for _auth_step in range(3):
                resp = None
                session = self._load_session(cc_request)

                force_refresh = _auth_step == 2
                try:
                    access_token = await session.get_valid_api_key(
                        oauth_http, force_refresh=force_refresh,
                    )
                except OAuthRefreshFailed as exc:
                    raise ProviderError(
                        f"Authentication refresh failed. "
                        f"Please re-authenticate with 'kitty auth openai'. Details: {exc}"
                    ) from exc

                headers = self._build_codex_headers(access_token, session.id_token)
                self._filter_cloudflare_cookies(self._curl_session.cookies)
                self._log_cf_cookies(self._curl_session.cookies)

                got_401 = False
                # General retry loop (5xx, CF 403, transport errors)
                for _attempt in range(_CODEX_RETRY_MAX_ATTEMPTS + 1):
                    try:
                        resp = await self._curl_session.post(
                            _CODEX_BACKEND_URL,
                            json=resp_body,
                            headers=headers,
                            timeout=_CODEX_TIMEOUT,
                        )
                    except Exception as exc:
                        if _attempt < _CODEX_RETRY_MAX_ATTEMPTS:
                            logger.debug(
                                "Codex backend transport error (attempt %d/%d): %s",
                                _attempt + 1, _CODEX_RETRY_MAX_ATTEMPTS + 1, exc,
                            )
                            await asyncio.sleep(_codex_backoff(_attempt + 1))
                            continue
                        raise self._handle_curl_error(exc) from exc

                    if resp.status_code >= 400:
                        raw = resp.text

                        # Cloudflare block — retry with specialized handling
                        if self._is_cloudflare_block(resp.status_code, raw):
                            cf_sig = get_cloudflare_signature(raw)
                            logger.debug(
                                "CF block detected in make_request: signature=%s, body_len=%d",
                                cf_sig, len(raw),
                            )
                            if _attempt < _CODEX_RETRY_MAX_ATTEMPTS:
                                logger.warning(
                                    "Codex backend blocked by Cloudflare challenge "
                                    "(signature=%s), retrying",
                                    cf_sig,
                                )
                                await asyncio.sleep(_codex_backoff(_attempt + 1))
                                continue
                            logger.warning(
                                "Codex backend blocked by Cloudflare challenge after retries"
                            )
                            raise self.map_error(resp.status_code, {"error": {"message": raw}})

                        # 5xx server error — retry if attempts remain
                        if resp.status_code >= 500 and _attempt < _CODEX_RETRY_MAX_ATTEMPTS:
                            logger.debug(
                                "Codex backend %d error (attempt %d/%d), retrying",
                                resp.status_code, _attempt + 1, _CODEX_RETRY_MAX_ATTEMPTS + 1,
                            )
                            await asyncio.sleep(_codex_backoff(_attempt + 1))
                            continue

                        # 401 — auth recovery (break retry loop, advance recovery)
                        if resp.status_code == 401:
                            got_401 = True
                            break

                        # Non-retryable error (4xx except CF/401) — raise immediately
                        body = {}
                        with contextlib.suppress(Exception):
                            body = json.loads(raw)
                        if not body:
                            body = {"error": {"message": raw}}
                        raise self.map_error(resp.status_code, body)
                    break  # success — exit retry loop

                if not got_401:
                    break  # success or non-401 error already raised

                # Log recovery step
                if _auth_step == 0:
                    logger.warning("Codex backend 401: reloading session from disk")
                elif _auth_step == 1:
                    logger.warning("Codex backend 401: forcing token refresh")

        # Auth recovery exhausted — still 401
        if got_401:
            logger.warning("Codex backend 401: auth recovery exhausted")
            raw = resp.text
            body = {}
            with contextlib.suppress(Exception):
                body = json.loads(raw)
            if not body:
                body = {"error": {"message": raw}}
            raise self.map_error(401, body)

        # Read the full streamed response.  Do NOT call resp.close() —
        # curl_cffi's internal cleanup callback releases the handle back to
        # the session pool.  Calling close() frees the curl handle, causing a
        # TypeError when the cleanup callback subsequently tries to release it.
        raw = resp.content
        if not raw:
            raise ProviderError("Codex backend returned an empty response body")
        return self._parse_sse_to_response(raw)

    async def stream_request(
        self,
        cc_request: dict,
        write: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Handle a streaming request via the Codex backend.

        If ``_original_body`` is present (Responses API path), forwards the
        Responses API request directly and passes SSE events through.
        Otherwise converts from CC to Responses API format.

        401 auth recovery matches Codex CLI's ``UnauthorizedRecovery``:
        1. Reload session from disk and retry
        2. Force-refresh OAuth token and retry
        3. If still 401, raise ``ProviderError``
        """
        original_body = cc_request.get("_original_body")
        if original_body:
            resp_body = self._prepare_responses_body(original_body)
            if "reasoning" not in resp_body:
                effort = cc_request.get("_reasoning_effort")
                if effort and effort != "none":
                    resp_body["reasoning"] = {"effort": effort}
        else:
            resp_body = self._cc_to_responses(cc_request)

        # Debug: log the full request body for diagnosis
        logger.debug(
            "Codex backend request: %s",
            json.dumps(resp_body, indent=2, ensure_ascii=False)[:3000],
        )

        # Auth recovery loop (Codex CLI: UnauthorizedRecovery state machine).
        # Step 0: initial attempt, Step 1: reload from disk, Step 2: force-refresh.
        import aiohttp

        got_401 = False
        resp: object | None = None
        async with aiohttp.ClientSession() as oauth_http:
            for _auth_step in range(3):
                resp = None
                session = self._load_session(cc_request)

                force_refresh = _auth_step == 2
                try:
                    access_token = await session.get_valid_api_key(
                        oauth_http, force_refresh=force_refresh,
                    )
                except OAuthRefreshFailed as exc:
                    raise ProviderError(
                        f"Authentication refresh failed. "
                        f"Please re-authenticate with 'kitty auth openai'. Details: {exc}"
                    ) from exc

                headers = self._build_codex_headers(access_token, session.id_token)
                self._filter_cloudflare_cookies(self._curl_session.cookies)
                self._log_cf_cookies(self._curl_session.cookies)

                got_401 = False
                last_exc: Exception | None = None
                _stream_attempt = 0
                # General retry loop (5xx, CF 403, transport errors)
                for _attempt in range(_CODEX_RETRY_MAX_ATTEMPTS + 1):
                    try:
                        resp = await self._curl_session.post(
                            _CODEX_BACKEND_URL,
                            json=resp_body,
                            headers=headers,
                            timeout=_CODEX_TIMEOUT,
                            stream=True,
                        )
                    except Exception as exc:
                        if _attempt < _CODEX_RETRY_MAX_ATTEMPTS:
                            logger.debug(
                                "Codex backend transport error (attempt %d/%d): %s",
                                _attempt + 1, _CODEX_RETRY_MAX_ATTEMPTS + 1, exc,
                            )
                            await asyncio.sleep(_codex_backoff(_attempt + 1))
                            continue
                        raise self._handle_curl_error(exc) from exc

                    if resp.status_code >= 400:
                        raw = resp.text

                        # Cloudflare block — retry with specialized handling
                        if self._is_cloudflare_block(resp.status_code, raw):
                            cf_sig = get_cloudflare_signature(raw)
                            logger.debug(
                                "CF block detected in stream_request: signature=%s, body_len=%d",
                                cf_sig, len(raw),
                            )
                            if _attempt < _CODEX_RETRY_MAX_ATTEMPTS:
                                logger.warning(
                                    "Codex backend blocked by Cloudflare challenge "
                                    "(signature=%s), retrying",
                                    cf_sig,
                                )
                                await asyncio.sleep(_codex_backoff(_attempt + 1))
                                continue
                            logger.warning(
                                "Codex backend blocked by Cloudflare challenge after retries"
                            )
                            raise self.map_error(resp.status_code, {"error": {"message": raw}})

                        # 5xx server error — retry if attempts remain
                        if resp.status_code >= 500 and _attempt < _CODEX_RETRY_MAX_ATTEMPTS:
                            logger.debug(
                                "Codex backend %d error (attempt %d/%d), retrying",
                                resp.status_code, _attempt + 1, _CODEX_RETRY_MAX_ATTEMPTS + 1,
                            )
                            await asyncio.sleep(_codex_backoff(_attempt + 1))
                            continue

                        # 401 — auth recovery (break retry loop, advance recovery)
                        if resp.status_code == 401:
                            got_401 = True
                            break

                        # Non-retryable error — raise immediately
                        body = {}
                        with contextlib.suppress(Exception):
                            body = json.loads(raw)
                        logger.debug("Codex backend error %d: %s", resp.status_code, raw[:500])
                        if not body:
                            body = {"error": {"message": raw}}
                        raise self.map_error(resp.status_code, body)

                    # Stream SSE chunks to the downstream client.  Do NOT call
                    # resp.close() — curl_cffi's internal cleanup callback releases
                    # the handle back to the session pool when the stream task completes.
                    try:
                        async for chunk in resp.aiter_content():
                            if chunk:
                                # Strip UTF-8 BOM that some responses include
                                cleaned = chunk.replace(b"\xef\xbb\xbf", b"")
                                if cleaned:
                                    await write(cleaned)
                        # Stream completed successfully
                        return
                    except Exception as exc:
                        if self._is_transient_stream_error(exc) and _stream_attempt < _STREAM_RECV_ERROR_RETRIES:
                            _stream_attempt += 1
                            logger.info(
                                "Codex backend stream reset (attempt %d/%d), retrying: %s",
                                _stream_attempt, _STREAM_RECV_ERROR_RETRIES + 1, exc,
                            )
                            last_exc = exc
                            continue
                        # Non-transient or retries exhausted — surface the error
                        raise self._handle_curl_error(exc) from exc

                if not got_401:
                    break  # success or non-401 error already raised

                # Log recovery step
                if _auth_step == 0:
                    logger.warning("Codex backend 401: reloading session from disk")
                elif _auth_step == 1:
                    logger.warning("Codex backend 401: forcing token refresh")

        # Auth recovery exhausted — still 401
        if got_401:
            logger.warning("Codex backend 401: auth recovery exhausted")
            raw = resp.text
            body = {}
            with contextlib.suppress(Exception):
                body = json.loads(raw)
            if not body:
                body = {"error": {"message": raw}}
            raise self.map_error(401, body)

        # Should not reach here, but guard against it
        if last_exc:
            raise self._handle_curl_error(last_exc) from last_exc

    def _cc_to_responses(self, cc_request: dict) -> dict:
        """Convert a CC (Chat Completions) request to Responses API format.

        This is used when the client sends a Chat Completions request
        (e.g. via /v1/chat/completions) to an openai_subscription profile.
        """
        # CC params that have no Responses API equivalent
        _cc_only_params = frozenset({
            "temperature", "top_p", "max_tokens", "max_completion_tokens",
            "frequency_penalty", "presence_penalty", "logprobs",
            "top_logprobs", "response_format", "stop", "n",
            "stream_options", "seed", "logit_bias",
        })
        dropped = set(cc_request) & _cc_only_params
        if dropped:
            logger.debug("CC parameters with no Codex equivalent (dropped): %s", sorted(dropped))

        messages = cc_request.get("messages", [])
        instructions = ""
        input_items = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                instructions += (content + "\n") if instructions else content

            elif role == "assistant":
                # Assistant message — may contain text and/or tool_calls
                item: dict = {
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                }
                if content:
                    item["content"].append(
                        {"type": "output_text", "text": str(content)}
                    )
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", ""),
                    })
                if item["content"]:
                    input_items.append(item)

            elif role == "user":
                if content:
                    input_items.append({
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": str(content)}],
                    })

            elif role == "tool":
                # Tool result → function_call_output
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": str(content),
                })

        body: dict = {
            "model": cc_request.get("model", "gpt-5.4"),
            "input": input_items,
            "stream": True,
            "store": False,
        }
        if instructions:
            body["instructions"] = instructions.strip()

        # Map tools from CC format to Responses API format
        cc_tools = cc_request.get("tools", [])
        if cc_tools:
            resp_tools = []
            for tool in cc_tools:
                func = tool.get("function", {})
                resp_tools.append({
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
            body["tools"] = resp_tools

        if cc_request.get("tool_choice"):
            body["tool_choice"] = cc_request["tool_choice"]

        # Inject reasoning effort from normalized metadata
        effort = cc_request.get("_reasoning_effort")
        if effort and effort != "none":
            body["reasoning"] = {"effort": effort}

        # NOTE: Do NOT include max_output_tokens or temperature —
        # the Codex backend rejects them with 400 "Unsupported parameter".

        return body

    @staticmethod
    def _parse_sse_to_response(raw: bytes) -> dict:
        """Parse a full SSE stream from the Codex backend into a CC response.

        Extracts text content, tool calls, usage, and model info from the
        Responses API SSE events.
        """
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_args: dict[str, str] = {}  # call_id → accumulated args
        model = ""
        finish_reason = "stop"
        usage: dict = {}

        for line in raw.decode("utf-8", errors="replace").split("\n"):
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    text_parts.append(delta)

            elif event_type == "response.function_call_arguments.delta":
                call_id = event.get("call_id", "")
                delta = event.get("delta", "")
                tool_args[call_id] = tool_args.get(call_id, "") + delta

            elif event_type == "response.function_call_arguments.done":
                # Some backends send the complete arguments in a done event
                call_id = event.get("call_id", "")
                args = event.get("arguments", "")
                if call_id and args and not tool_args.get(call_id):
                    tool_args[call_id] = args

            elif event_type == "response.output_item.added":
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    idx = len(tool_calls)
                    tool_calls.append({
                        "index": idx,
                        "id": item.get("call_id", f"call_{idx}"),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": "",
                        },
                    })

            elif event_type == "response.output_item.done":
                # Some backends include full item data in the done event
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    call_id = item.get("call_id", "")
                    # Check if we already registered this call from .added
                    existing = next(
                        (tc for tc in tool_calls if tc["id"] == call_id), None
                    )
                    if existing is None:
                        idx = len(tool_calls)
                        tool_calls.append({
                            "index": idx,
                            "id": call_id or f"call_{idx}",
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", ""),
                            },
                        })
                        if call_id and item.get("arguments"):
                            tool_args[call_id] = item["arguments"]
                    elif call_id and item.get("arguments"):
                        # Update args if not yet captured via delta events
                        if not tool_args.get(call_id):
                            tool_args[call_id] = item["arguments"]

            elif event_type == "response.completed":
                resp_data = event.get("response", {})
                model = resp_data.get("model", model)
                resp_usage = resp_data.get("usage", {})
                if resp_usage:
                    usage = resp_usage
                status = resp_data.get("status", "completed")
                if status == "incomplete":
                    finish_reason = "length"

                # Check for tool calls embedded in completed output items
                for item in resp_data.get("output", []):
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id", "")
                        existing = next(
                            (tc for tc in tool_calls if tc["id"] == call_id), None
                        )
                        if existing is None:
                            idx = len(tool_calls)
                            tool_calls.append({
                                "index": idx,
                                "id": call_id or f"call_{idx}",
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": item.get("arguments", ""),
                                },
                            })
                            if call_id and item.get("arguments"):
                                tool_args[call_id] = item["arguments"]
                        elif call_id and item.get("arguments"):
                            if not tool_args.get(call_id):
                                tool_args[call_id] = item["arguments"]

            else:
                # Log unknown event types for debugging
                logger.debug("Unknown Codex SSE event type: %s", event_type)

        # Finalize tool call arguments
        for tc in tool_calls:
            call_id = tc["id"]
            tc["function"]["arguments"] = tool_args.get(call_id, "")

        # Debug: warn if tool calls have empty arguments (helps diagnose parsing issues)
        if tool_calls:
            for tc in tool_calls:
                if not tc["function"]["arguments"]:
                    logger.warning(
                        "Tool call '%s' (id=%s) has empty arguments after SSE parsing",
                        tc["function"]["name"], tc["id"],
                    )
                    # Dump raw SSE for diagnosis
                    import tempfile
                    try:
                        dump_path = Path(tempfile.gettempdir()) / "kitty_codex_sse_dump.txt"
                        dump_path.write_bytes(raw)
                        logger.warning("Raw SSE dumped to %s", dump_path)
                    except Exception:
                        pass

        content = "".join(text_parts)
        message: dict = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }

    # ── Cloudflare detection ───────────────────────────────────────────────

    # Delegate to shared utility; keep thin wrapper for backward compat
    _is_cloudflare_block = staticmethod(is_cloudflare_block)

    def map_error(self, status_code: int, body: dict) -> Exception:
        error_obj = body.get("error", body)
        msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
        if status_code == 401:
            return ProviderError(
                f"OpenAI subscription auth failed. "
                f"Please re-authenticate with 'kitty auth openai'. Details: {msg}"
            )
        if status_code == 429:
            return ProviderError(
                f"OpenAI subscription rate limited: {msg}"
            )
        if status_code == 403:
            if is_cloudflare_block(403, msg):
                err = ProviderError(
                    "Cloudflare bot detection blocked the Codex backend request. "
                    "This may be transient — a retry after a short delay may resolve it. "
                    "This is not an API key problem."
                )
                err.is_cloudflare = True
                return err
            return ProviderError(
                f"OpenAI subscription access denied: {msg}"
            )
        if status_code >= 500:
            return ProviderError(
                f"OpenAI subscription server error {status_code}: {msg}"
            )
        return ProviderError(f"OpenAI subscription error {status_code}: {msg}")
