"""OpenAI OAuth flow orchestrator using OpenID Connect / OAuth 2.0 authorization code flow.

This module implements the OpenAI Codex CLI OAuth flow as documented at
https://auth.openai.com/oauth/authorize with PKCE (RFC 7636) support.

The full flow:
  1. Generate PKCE code_verifier + code_challenge (S256)
  2. Start a local callback server on port 1455 (fallback 1456-1459)
  3. Open the authorization URL in the user's browser
  4. Wait for the browser to redirect to localhost with the auth code
  5. Exchange the auth code for tokens (authorization_code grant)
  6. Exchange the id_token for an OpenAI API key (token-exchange grant)
  7. Return an OAuthSession wrapping the tokens and API key
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import html
import json
import logging
import secrets
import sys
import urllib.parse
import webbrowser
from typing import TYPE_CHECKING, cast

import aiohttp
from aiohttp import web

from kitty.auth.headless import is_headless_environment
from kitty.auth.oauth_session import (
    ID_TOKEN_TYPE,
    TOKEN_EXCHANGE_GRANT,
    OAuthSession,
    token_request_headers,
)
from kitty.auth.pkce import generate_code_challenge, generate_code_verifier
from kitty.egress import aiohttp_session_kwargs

if TYPE_CHECKING:
    from asyncio import Future

logger = logging.getLogger(__name__)

# ── OpenAI OAuth endpoints ───────────────────────────────────────────────
OAUTH_AUTH_URL = "https://auth.openai.com/oauth/authorize"
OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"

# OpenAI Codex CLI OAuth client ID
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

# Redirect URI registered for the local callback server
REDIRECT_URI = "http://localhost:1455/auth/callback"

# OAuth scope requested (space-separated, will be URL-encoded in the URL)
SCOPE = "openid profile email offline_access"

# Callback server port range
_CALLBACK_PORT_START = 1455
_CALLBACK_PORT_END = 1459

# OAuth flow timeout (5 minutes)
OAUTH_TIMEOUT_SECONDS = 300


# ── Exceptions ───────────────────────────────────────────────────────────
class OAuthPortConflictError(Exception):
    """Raised when none of the callback ports (1455-1459) are available."""

    pass


class OAuthTimeoutError(Exception):
    """Raised when the OAuth callback is not received within the timeout period."""

    pass


class OAuthTokenExchangeError(Exception):
    """Raised when the id_token → API key exchange fails."""

    def __init__(self, error: str, error_description: str | None = None) -> None:
        self.error = error
        self.error_description = error_description or ""
        super().__init__(f"{error}: {error_description}" if error_description else error)


class OAuthAuthorizationError(Exception):
    """Raised when the authorization server returns an error."""

    def __init__(self, error: str, error_description: str | None = None) -> None:
        self.error = error
        self.error_description = error_description or ""
        super().__init__(f"{error}: {error_description}" if error_description else error)


# ── URL builder ──────────────────────────────────────────────────────────
def _decode_jwt_payload(token: str) -> dict:
    """Decode the payload of a JWT without verifying the signature.

    JWTs are base64url-encoded JSON. We only need the claims, not
    signature verification (the token was received directly from the
    OAuth server over TLS).
    """
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1]
    # Fix padding
    padding = 4 - len(payload) % 4
    if padding != 4:
        payload += "=" * padding
    try:
        decoded = base64.urlsafe_b64decode(payload)
        return cast(dict, json.loads(decoded))
    except Exception:
        return {}


def build_auth_url(code_verifier: str, state: str | None = None) -> str:
    """Build the OpenAI authorization URL with PKCE parameters.

    Args:
        code_verifier: The PKCE code verifier string.
        state: Optional CSRF state token. If not provided, a random one is
               generated.

    Returns:
        The full authorization URL string.
    """
    code_challenge = generate_code_challenge(code_verifier)
    if state is None:
        state = secrets.token_urlsafe(32)

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    return f"{OAUTH_AUTH_URL}?{urllib.parse.urlencode(params)}"


# ── Device-code flow (KBR-317) ───────────────────────────────────────────
# Mirror of codex-rs/login/src/device_code_auth.rs, verified 2026-10-14.
# The interval policy deliberately diverges from codex: upstream serde-
# defaults a missing interval to 0 and then busy-loops the token endpoint
# for 15 minutes. See .system_design/SYSTEM_DESIGN.md §14.4.

OAUTH_DEVICE_USERCODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
OAUTH_DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
OAUTH_DEVICE_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"
OAUTH_DEVICE_VERIFICATION_URL = "https://auth.openai.com/codex/device"

# Poll-loop discipline: the user must switch machines, open a browser, and
# type a one-time code — a fundamentally slower interaction than clicking a
# link, so the device flow gets 15 minutes rather than the browser path's
# 300 seconds (matching codex's cap).
DEVICE_FLOW_MAX_WAIT_SECONDS = 900

# Floor on the poll sleep when the server's interval is positive but tiny,
# and the clamp unit for the final sleep before the deadline. Prevents the
# busy-loop codex's interval=0 default would produce.
DEVICE_FLOW_MIN_INTERVAL_FLOOR = 1.0

# Default poll interval used only when the server response omits the field.
DEVICE_FLOW_DEFAULT_INTERVAL = 5


class OAuthDeviceCodeDisabledError(Exception):
    """Raised when the usercode endpoint reports device-code auth is not enabled.

    A 404 on ``/api/accounts/deviceauth/usercode`` is the server saying this
    account/workspace does not have device-code login available (OpenAI-side
    entitlement, admin toggle for Business/Team/Enterprise workspaces).
    """


class OAuthDeviceCodeResponseError(Exception):
    """Raised when a device endpoint returns a body kitty cannot parse."""


class OAuthDeviceCodeIntervalError(Exception):
    """Raised when the server's poll interval is present but unparseable or not positive.

    Kitty fails fast rather than inheriting codex's interval=0 busy-loop;
    a missing field gets ``DEVICE_FLOW_DEFAULT_INTERVAL`` instead.
    """


class OAuthDeviceCodeCancelledError(Exception):
    """Raised when the user cancels the device flow with Ctrl+C while polling."""


@dataclasses.dataclass(frozen=True)
class DeviceUserCodeResponse:
    """Parsed ``/deviceauth/usercode`` response (codex's ``UserCodeResp``)."""

    device_auth_id: str
    user_code: str
    interval: int


@dataclasses.dataclass(frozen=True)
class DeviceTokenPollResult:
    """Disposition of one ``/deviceauth/token`` poll response.

    ``kind`` is ``"pending"`` (403/404 — keep polling) or ``"success"``
    (200 — the other fields carry the server-issued PKCE pair and the
    authorization code).
    """

    kind: str
    authorization_code: str = ""
    code_challenge: str = ""
    code_verifier: str = ""


def parse_device_user_code_response(body: str) -> DeviceUserCodeResponse:
    """Parse the ``/deviceauth/usercode`` response body.

    Args:
        body: Raw JSON text from the usercode endpoint.

    Returns:
        The parsed device_auth_id, user_code, and poll interval in seconds.

    Raises:
        OAuthDeviceCodeResponseError: When the body is not JSON or is
            missing ``device_auth_id`` / ``user_code``.
        OAuthDeviceCodeIntervalError: When ``interval`` is present but
            unparseable or not positive. A missing ``interval`` gets
            ``DEVICE_FLOW_DEFAULT_INTERVAL``.
    """
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OAuthDeviceCodeResponseError(f"Malformed usercode response: {exc}") from exc
    if not isinstance(data, dict):
        raise OAuthDeviceCodeResponseError("Usercode response is not a JSON object")

    # Read the fields without echoing values into any error message: the
    # user_code and device_auth_id are short-lived secrets (§14.4).
    device_auth_id = data.get("device_auth_id")
    user_code = data.get("user_code", data.get("usercode"))
    if not isinstance(device_auth_id, str) or not device_auth_id:
        raise OAuthDeviceCodeResponseError("Usercode response missing device_auth_id")
    if not isinstance(user_code, str) or not user_code:
        raise OAuthDeviceCodeResponseError("Usercode response missing user_code")

    raw_interval = data.get("interval")
    if raw_interval is None:
        # Kitty's choice: codex serde-defaults a missing interval to 0 and
        # busy-loops; we pick a sane 5 s default instead (§14.4).
        return DeviceUserCodeResponse(device_auth_id, user_code, DEVICE_FLOW_DEFAULT_INTERVAL)
    try:
        interval = int(str(raw_interval).strip())
    except ValueError as exc:
        raise OAuthDeviceCodeIntervalError("Usercode response interval is not an integer") from exc
    if interval <= 0:
        raise OAuthDeviceCodeIntervalError("Usercode response interval is not positive")
    return DeviceUserCodeResponse(device_auth_id, user_code, interval)


def parse_device_token_poll(status: int, body: bytes) -> DeviceTokenPollResult:
    """Classify one ``/deviceauth/token`` poll response.

    Args:
        status: HTTP status code of the poll response.
        body: Raw response body (only read on 200).

    Returns:
        A ``DeviceTokenPollResult``; ``kind == "pending"`` for 403/404
        (keep polling), ``kind == "success"`` with the authorization code
        and server-issued PKCE pair on 200.

    Raises:
        OAuthAuthorizationError: On any other status, or on a 200 whose
            body is missing required fields. Messages carry the status or
            field names only — never body text, which may echo secrets.
    """
    if status in (403, 404):
        return DeviceTokenPollResult(kind="pending")
    if status == 200:
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OAuthAuthorizationError(f"Malformed poll response (HTTP {status})") from exc
        if not isinstance(data, dict):
            raise OAuthAuthorizationError(f"Poll response is not a JSON object (HTTP {status})")
        missing = [
            field
            for field in ("authorization_code", "code_challenge", "code_verifier")
            if not isinstance(data.get(field), str) or not data.get(field)
        ]
        if missing:
            raise OAuthAuthorizationError(
                f"Poll response missing required fields: {', '.join(missing)}"
            )
        return DeviceTokenPollResult(
            kind="success",
            authorization_code=data["authorization_code"],
            code_challenge=data["code_challenge"],
            code_verifier=data["code_verifier"],
        )
    raise OAuthAuthorizationError(f"Device auth poll failed with status {status}")


def format_device_code_prompt(verification_url: str, user_code: str) -> str:
    """Format the one-time-code prompt shown while the device flow polls.

    Kitty-owned prompt modelled on codex's ``device_code_prompt`` (the
    upstream string is a test fixture, not a contract — §14.4). Kitty adds
    the workspace-admin sentence, which answers a question codex's prompt
    does not have to.

    Args:
        verification_url: The URL the user opens in any browser.
        user_code: The one-time code to enter there.

    Returns:
        The full prompt text (ANSI-coloured), ending with a newline.
    """
    ansi_blue = "\x1b[94m"
    ansi_gray = "\x1b[90m"
    ansi_reset = "\x1b[0m"
    return (
        f"\nSign in to ChatGPT for Kitty:\n\n"
        f"1. Open this link in a browser on any machine and sign in to your account\n"
        f"   {ansi_blue}{verification_url}{ansi_reset}\n\n"
        f"2. Enter this one-time code {ansi_gray}(expires in 15 minutes){ansi_reset}\n"
        f"   {ansi_blue}{user_code}{ansi_reset}\n\n"
        f"Personal ChatGPT Plus/Pro accounts are enabled by default. On Business/Team/Enterprise\n"
        f"workspaces an admin must enable device-code authentication first\n"
        f"(chatgpt.com/admin/permissions).\n\n"
        f"{ansi_gray}Continue only if you started this login in Kitty. If a website or another\n"
        f"person gave you this code, cancel.{ansi_reset}\n"
    )


def format_device_code_prompt_plain(verification_url: str, user_code: str) -> str:
    """Format a copy-paste-friendly plain-text copy of the device prompt.

    Emitted right after the ANSI-coloured prompt (FR-4): terminal emulators
    that strip or mangle ANSI, log capture, and copy-paste all work better
    against a line with no escape sequences.

    Args:
        verification_url: The URL the user opens in any browser.
        user_code: The one-time code to enter there.

    Returns:
        The plain prompt text, ending with a newline. Contains no ANSI.
    """
    return (
        f"Sign in at: {verification_url}\n"
        f"One-time code (expires in 15 minutes): {user_code}\n"
    )


def _select_device_flow(device_auth: bool | None) -> bool:
    """Resolve the explicit flag (or auto-detect) into a path decision.

    Single home of the §14.3 selection rule so the flow dispatch and the
    CLI-layer "Opening browser" prediction cannot drift apart.

    Args:
        device_auth: ``True`` forces the device flow, ``False`` forces the
            browser flow, ``None`` defers to ``is_headless_environment``.

    Returns:
        True when the device-code path should run.
    """
    return device_auth if device_auth is not None else is_headless_environment()


async def run_oauth_device_flow(http: aiohttp.ClientSession | None = None) -> OAuthSession:
    """Run the OpenAI device-code OAuth flow (RFC 8628, codex-compatible).

    Asks the OpenAI device server for a one-time code, prints the
    verification URL and code for the user to complete on any machine, and
    polls until the authorization arrives. The resulting tokens flow
    through the same OAuthSession plumbing as the browser flow, so no
    downstream code distinguishes the two paths.

    Args:
        http: Optional aiohttp ClientSession. If not provided, a new one
              is created and closed internally.

    Returns:
        An OAuthSession with access_token, refresh_token, id_token,
        api_key (best-effort), and expiry timestamps.

    Raises:
        OAuthDeviceCodeDisabledError: When the usercode endpoint reports
            device-code auth is not enabled for this account.
        OAuthDeviceCodeResponseError: When a device endpoint returns a
            body kitty cannot parse.
        OAuthDeviceCodeIntervalError: When the server's poll interval is
            unparseable or not positive.
        OAuthDeviceCodeCancelledError: When the user cancels with Ctrl+C.
        OAuthTimeoutError: When the authorization does not arrive within
            ``DEVICE_FLOW_MAX_WAIT_SECONDS``.
        OAuthAuthorizationError: On any other device-endpoint or
            token-exchange failure.
    """
    close_http = http is None
    http = http or aiohttp.ClientSession(**aiohttp_session_kwargs())
    try:
        # Hop 1: request the one-time code. A 404 here is the server's
        # "device-code auth not enabled" signal; any other non-2xx fails fast.
        async with http.post(
            OAUTH_DEVICE_USERCODE_URL,
            data=json.dumps({"client_id": CLIENT_ID}),
            headers={"Content-Type": "application/json", **token_request_headers()},
        ) as resp:
            body_text = await resp.text()
            if resp.status == 404:
                raise OAuthDeviceCodeDisabledError(
                    "Device code login is not enabled for this account. "
                    "Use the browser login or ask a workspace admin to enable "
                    "device code authentication."
                )
            if resp.status >= 400:
                raise OAuthAuthorizationError(
                    f"Device code request failed with status {resp.status}"
                )
        user_code_response = parse_device_user_code_response(body_text)

        # The prompt goes to stderr — the same stream the browser path uses
        # for its "if the browser did not open" hint. The plain-text copy
        # follows the coloured one (FR-4): no-ANSI line for copy-paste,
        # log capture, and terminals that mangle escapes.
        print(
            format_device_code_prompt(OAUTH_DEVICE_VERIFICATION_URL, user_code_response.user_code),
            file=sys.stderr,
        )
        print(
            format_device_code_prompt_plain(OAUTH_DEVICE_VERIFICATION_URL, user_code_response.user_code),
            file=sys.stderr,
        )

        # Hop 2: poll until the user authorizes, the code expires, or a
        # non-retriable error arrives. 403/404 mean "still pending".
        deadline = asyncio.get_running_loop().time() + DEVICE_FLOW_MAX_WAIT_SECONDS
        poll_result: DeviceTokenPollResult | None = None
        while poll_result is None:
            sleep_for = max(user_code_response.interval, DEVICE_FLOW_MIN_INTERVAL_FLOOR)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise OAuthTimeoutError(
                    f"No authorization received within {DEVICE_FLOW_MAX_WAIT_SECONDS} seconds"
                )
            # The final sleep is clamped to the remaining budget so the
            # loop never overshoots the cap (codex's min() discipline).
            # The sleep is inside the try: the loop spends ~99% of its
            # wall clock here, so Ctrl+C lands here in practice and must
            # surface as OAuthDeviceCodeCancelledError, not a raw
            # CancelledError (code-review round 1, High #3).
            try:
                await asyncio.sleep(min(sleep_for, remaining))
                async with http.post(
                    OAUTH_DEVICE_TOKEN_URL,
                    data=json.dumps(
                        {
                            "device_auth_id": user_code_response.device_auth_id,
                            "user_code": user_code_response.user_code,
                        }
                    ),
                    headers={"Content-Type": "application/json", **token_request_headers()},
                ) as resp:
                    poll_body = await resp.read()
                    # Only a success assignment ends the loop; a pending
                    # result must NOT be stored in the sentinel, or the
                    # while-condition would exit and hop 3 would fire with
                    # empty fields (this exact bug motivated the L3
                    # poll-timeout test).
                    disposition = parse_device_token_poll(resp.status, poll_body)
            except asyncio.CancelledError:
                raise OAuthDeviceCodeCancelledError("Device login cancelled") from None
            except OAuthAuthorizationError:
                raise
            except Exception as exc:
                # Transport failures (DNS, connection reset) are not
                # fixable by waiting — fail fast with the error named,
                # matching codex's no-retry posture.
                raise OAuthAuthorizationError(f"Device auth poll failed: {exc}") from exc
            if disposition.kind == "success":
                poll_result = disposition

        # Hop 3: exchange the authorization code for tokens. The device flow's
        # PKCE pair is issued by the server, and the redirect_uri is the
        # server-side device callback — not the localhost one the browser
        # path registers.
        tokens = await _exchange_code_for_tokens(
            poll_result.authorization_code,
            poll_result.code_verifier,
            CLIENT_ID,
            http,
            redirect_uri=OAUTH_DEVICE_REDIRECT_URI,
            error_redactions=(
                poll_result.authorization_code,
                poll_result.code_verifier,
            ),
        )
        # F35-mirror: validate all three required fields before the
        # dataclass constructor sees them, so a partial response from the
        # new endpoint raises a typed error instead of a bare KeyError.
        missing = [
            field
            for field in ("access_token", "id_token", "refresh_token")
            if not tokens.get(field)
        ]
        if missing:
            logger.error(
                "OAuth token exchange returned malformed response (missing fields): %s",
                list(tokens.keys()),
            )
            raise OAuthAuthorizationError(
                f"Token response missing required fields: {', '.join(missing)}"
            )

        # Best-effort id_token → API key exchange, identical to the
        # browser flow's tail; failure leaves the access_token as the
        # ChatGPT-subscription credential.
        try:
            api_key = await _exchange_id_token_for_api_key(
                tokens["id_token"], tokens["access_token"], CLIENT_ID, http
            )
            tokens["api_key"] = api_key
        except OAuthTokenExchangeError as exc:
            logger.warning(
                "API key exchange skipped (%s). Using access_token directly for ChatGPT subscription.",
                exc.error,
            )
    finally:
        if close_http:
            await http.close()

    return OAuthSession.from_token_response(tokens, CLIENT_ID)


# ── Callback server ──────────────────────────────────────────────────────
async def _start_callback_server(
    state: str,
    code_future: Future[str | OAuthAuthorizationError],
    host: str,
    port: int,
) -> web.Application:
    """Create an aiohttp web application that handles the OAuth callback.

    Routes:
        GET /              -- Landing page with a link to the auth URL
        GET /auth/callback -- OAuth redirect endpoint; validates state and
                              resolves *code_future* with either the auth
                              code (success) or an OAuthAuthorizationError
                              (error or state mismatch).

    Args:
        state: The expected state parameter (used for CSRF protection).
        code_future: A Future that will be resolved with the auth code or
                     an OAuthAuthorizationError when the callback is received.
        host: Host to bind to.
        port: Port to bind to.

    Returns:
        A configured web.Application.
    """
    app = web.Application()

    async def handle_index(request: web.Request) -> web.Response:
        return web.Response(
            text="<html><body><p>Waiting for OpenAI authorization...</p><p>You can close this tab.</p></body></html>",
            content_type="text/html",
        )

    async def handle_callback(request: web.Request) -> web.Response:
        received_state = request.query.get("state", "")
        if received_state != state:
            code_future.set_result(
                OAuthAuthorizationError(
                    "state_mismatch",
                    f"State mismatch: expected {state!r}, got {received_state!r}",
                )
            )
            return web.Response(
                status=400,
                text="<html><body><p>State mismatch. Please try again.</p></body></html>",
                content_type="text/html",
            )

        # Check for OAuth error response
        error = request.query.get("error")
        if error:
            error_description = request.query.get("error_description", "")
            oauth_error = OAuthAuthorizationError(error, error_description)
            code_future.set_result(oauth_error)
            # Escaped: both values come straight from the callback query
            # string, so reflecting them raw makes this page an XSS sink for
            # anyone who can get the browser to hit the loopback callback
            # while a login is in progress.
            return web.Response(
                status=400,
                text=(
                    "<html><body>"
                    f"<p>Authorization error: {html.escape(error)}</p>"
                    f"<p>{html.escape(error_description)}</p>"
                    "</body></html>"
                ),
                content_type="text/html",
            )

        code = request.query.get("code", "")
        code_future.set_result(code)
        return web.Response(
            status=200,
            text="<html><body><p>Authorization successful!</p>"
            "<p>You can close this tab and return to the terminal.</p></body></html>",
            content_type="text/html",
        )

    app.router.add_get("/", handle_index)
    app.router.add_get("/auth/callback", handle_callback)

    return app


# ── Token exchange ───────────────────────────────────────────────────────
async def _exchange_code_for_tokens(
    code: str,
    code_verifier: str,
    client_id: str,
    http: aiohttp.ClientSession,
    *,
    redirect_uri: str = REDIRECT_URI,
    error_redactions: tuple[str, ...] = (),
) -> dict:
    """Exchange an authorization code for tokens.

    Args:
        code: The authorization code from the OAuth callback.
        code_verifier: The PKCE code verifier.
        client_id: The OAuth client ID.
        http: an aiohttp ClientSession for making the request.
        redirect_uri: The redirect URI registered for the grant. The
            browser path uses the local-callback URL (the default); the
            device path passes the server-side device callback.
        error_redactions: Secret values to strip from error-body text
            before it lands in an exception message. OAuth error bodies
            sometimes echo request parameters verbatim; the device path
            passes the secrets it holds at this point. Empty (the
            default) preserves the browser path's historical behaviour.

    Returns:
        Parsed JSON response from the token endpoint.
    """
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    async with http.post(OAUTH_TOKEN_URL, data=payload, headers=token_request_headers()) as resp:
        if resp.status >= 400:
            body_text = await resp.text()
            for secret in error_redactions:
                if secret:
                    body_text = body_text.replace(secret, "[redacted]")
            raise OAuthAuthorizationError(
                f"Token exchange failed (HTTP {resp.status}): {body_text}",
            )
        try:
            tokens = await resp.json()
        except Exception as exc:
            raise OAuthAuthorizationError(
                f"Malformed token response: {exc}",
            ) from exc
        # F35: Validate required fields before returning — prevents bare KeyError
        # when upstream returns a partial response (e.g. missing id_token).
        if not tokens.get("access_token") or not tokens.get("id_token"):
            logger.error(
                "OAuth token exchange returned malformed response (missing fields): %s",
                list(tokens.keys()),
            )
            raise OAuthAuthorizationError("Token response missing required fields")
        return cast(dict, tokens)


async def _exchange_id_token_for_api_key(
    id_token: str,
    access_token: str,
    client_id: str,
    http: aiohttp.ClientSession,
) -> str:
    """Exchange an id_token for an OpenAI API key via token-exchange grant.

    Args:
        id_token: The OpenID Connect id_token.
        access_token: The OAuth access_token (used as Bearer auth).
        client_id: The OAuth client ID.
        http: An aiohttp ClientSession for making the request.

    Returns:
        The exchanged OpenAI API key string.
    """
    # Token exchange: request an API key using the id_token as the subject.
    # The access_token is required as Bearer auth (not a form parameter).
    # Note: organization_id is NOT a parameter for this endpoint.
    # The org info is embedded in the id_token JWT itself.
    payload = {
        "grant_type": TOKEN_EXCHANGE_GRANT,
        "requested_token": "openai-api-key",
        "subject_token": id_token,
        "subject_token_type": ID_TOKEN_TYPE,
        "client_id": client_id,
    }
    # The bearer header is how this grant authenticates; the Codex identity is
    # merged on top so this POST claims the same client as every other leg.
    headers = {
        **token_request_headers(),
        "Authorization": f"Bearer {access_token}",
    }
    async with http.post(OAUTH_TOKEN_URL, data=payload, headers=headers) as resp:
        if resp.status >= 400:
            body_text = await resp.text()
            raise OAuthTokenExchangeError(
                "token_exchange_failed",
                f"API key exchange failed (HTTP {resp.status}): {body_text}",
            )
        try:
            result = await resp.json()
        except Exception as exc:
            raise OAuthTokenExchangeError(
                "token_exchange_failed",
                f"Malformed response from token exchange: {exc}",
            ) from exc
        api_key = result.get("openai_api_key")
        if not api_key:
            raise OAuthTokenExchangeError(
                "token_exchange_failed",
                f"Response missing openai_api_key: {result}",
            )
    return cast(str, api_key)


# ── Main orchestrator ────────────────────────────────────────────────────
async def run_oauth_flow(http: aiohttp.ClientSession | None = None, *, device_auth: bool | None = None) -> OAuthSession:
    """Run the full OpenAI OAuth flow.

    This coroutine:
      1. Generates a PKCE code verifier and state
      2. Starts a local callback server on port 1455 (fallback 1456-1459)
      3. Opens the authorization URL in the user's browser
      4. Waits for the browser to redirect to localhost with the auth code
      5. Exchanges the auth code for tokens (authorization_code grant)
      6. Exchanges the id_token for an API key (token-exchange grant)
      7. Returns an OAuthSession wrapping all credentials

    On a headless environment (SSH, CI, no display) the device-code flow
    replaces steps 2-4: the authorization arrives through OpenAI's polling
    endpoint instead of a localhost callback, so no local browser is needed
    (:meth:`run_oauth_device_flow`, KBR-317).

    Args:
        http: Optional aiohttp ClientSession. If not provided, a new one
              is created and closed internally.
        device_auth: Force the device-code path (``True``) or the browser
            path (``False``). ``None`` (the default) auto-detects via
            :func:`kitty.auth.headless.is_headless_environment`
            (SYSTEM_DESIGN §14.3).

    Returns:
        An OAuthSession with access_token, refresh_token, id_token,
        api_key, and expiry timestamps.

    Raises:
        OAuthPortConflictError: If none of the callback ports are available.
        OAuthTimeoutError: If the callback is not received within 5 minutes.
        OAuthAuthorizationError: If the authorization server returns an error.
    """
    # Selection rule (§14.3): explicit flag wins; otherwise the headless
    # predicate decides. One place, so every entry point inherits it.
    if _select_device_flow(device_auth):
        return await run_oauth_device_flow(http=http)

    # Generate PKCE + state
    code_verifier = generate_code_verifier()
    state = secrets.token_urlsafe(32)

    # Build auth URL (pass the state so the URL and callback server match)
    auth_url = build_auth_url(code_verifier, state=state)

    # Prepare the Future that the callback server will resolve
    loop = asyncio.get_running_loop()
    code_future: Future[str | OAuthAuthorizationError] = loop.create_future()

    # Start callback server (try ports 1455-1459)
    runner: web.AppRunner | None = None
    bound_port: int | None = None
    for port in range(_CALLBACK_PORT_START, _CALLBACK_PORT_END + 1):
        try:
            app = await _start_callback_server(state, code_future, "127.0.0.1", port)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", port)
            await site.start()
            bound_port = port
            break
        except OSError:
            # Port in use — clean up runner and try next port
            if runner is not None:
                await runner.cleanup()
                runner = None
            continue

    if runner is None or bound_port is None:
        raise OAuthPortConflictError(
            f"None of the callback ports {_CALLBACK_PORT_START}-{_CALLBACK_PORT_END} are available"
        )

    try:
        # Always print the URL as a fallback for WSL / headless environments
        # where webbrowser.open may appear to succeed but nothing opens.
        import sys

        print(
            f"\n  Opening browser for OpenAI authentication.\n"
            f"  If the browser did not open, visit:\n\n"
            f"    {auth_url}\n",
            file=sys.stderr,
        )
        with contextlib.suppress(Exception):
            webbrowser.open(auth_url)

        # Wait for callback with timeout
        try:
            result = await asyncio.wait_for(code_future, timeout=OAUTH_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            raise OAuthTimeoutError(f"No OAuth callback received within {OAUTH_TIMEOUT_SECONDS} seconds") from exc

        # Handle OAuth errors returned via the callback
        if isinstance(result, OAuthAuthorizationError):
            raise result

        code = result

    finally:
        # Shut down the callback server
        await runner.cleanup()

    # Exchange code for tokens
    close_http = http is None
    http = http or aiohttp.ClientSession(**aiohttp_session_kwargs())
    try:
        tokens = await _exchange_code_for_tokens(code, code_verifier, CLIENT_ID, http)
        # F35: _exchange_code_for_tokens already validates these fields and raises
        # OAuthAuthorizationError if missing.  Use .get() here for defense-in-depth.
        access_token = tokens.get("access_token")
        id_token = tokens.get("id_token")
        assert access_token and id_token  # validated by _exchange_code_for_tokens

        # Exchange id_token for API key (best-effort).
        # Fails for org accounts without Platform API org mapping —
        # in that case we use the access_token directly as Bearer auth.
        try:
            api_key = await _exchange_id_token_for_api_key(id_token, access_token, CLIENT_ID, http)
            tokens["api_key"] = api_key
        except OAuthTokenExchangeError as exc:
            logger.warning(
                "API key exchange skipped (%s). Using access_token directly for ChatGPT subscription.",
                exc.error,
            )
    finally:
        if close_http:
            await http.close()

    # Build and return OAuthSession
    session = OAuthSession.from_token_response(tokens, CLIENT_ID)
    return session
