"""Tests for the OpenAI OAuth flow orchestrator."""

from __future__ import annotations

import asyncio
import urllib.parse
from contextlib import suppress
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import test_utils
from aioresponses import aioresponses

from kitty.auth.oauth_session import (
    ID_TOKEN_TYPE,
    OAUTH_TOKEN_URL,
    TOKEN_EXCHANGE_GRANT,
    OAuthSession,
)
from kitty.auth.openai_oauth import (
    OAuthAuthorizationError,
    OAuthPortConflictError,
    OAuthTimeoutError,
    _exchange_code_for_tokens,
    _exchange_id_token_for_api_key,
    _start_callback_server,
    build_auth_url,
    run_oauth_flow,
)

# ── Constants (mirrored from openai_oauth.py) ──────────────────────────────
OAUTH_AUTH_URL = "https://auth.openai.com/oauth/authorize"
REDIRECT_URI = "http://localhost:1455/auth/callback"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
SCOPE = "openid profile email offline_access"

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def code_verifier() -> str:
    """A valid 43-char PKCE code verifier."""
    return "a" * 43


@pytest.fixture()
def state() -> str:
    """A stable state string for CSRF testing."""
    return "test-state-12345"


# ── build_auth_url ────────────────────────────────────────────────────────


class TestBuildAuthUrl:
    def test_build_auth_url_contains_required_params(self, code_verifier: str) -> None:
        """Authorization URL has all required OAuth + PKCE params."""
        url = build_auth_url(code_verifier)

        assert url.startswith(OAUTH_AUTH_URL + "?")
        # Parse query string and decode URL-encoded values
        query = url[len(OAUTH_AUTH_URL) + 1 :]
        parsed = urllib.parse.parse_qs(query, strict_parsing=True)
        # parse_qs returns dict of lists; flatten to single values
        params = {k: v[0] for k, v in parsed.items()}

        # Core OAuth params
        assert params["response_type"] == "code"
        assert params["client_id"] == CLIENT_ID
        assert params["redirect_uri"] == REDIRECT_URI
        assert params["code_challenge_method"] == "S256"
        assert "code_challenge" in params
        assert "state" in params

        # Scope: decoded scope contains our four scope values
        decoded_scope = urllib.parse.unquote(params["scope"])
        for scope_part in SCOPE.split():
            assert scope_part in decoded_scope

        # Codex CLI specific params
        assert params["id_token_add_organizations"] == "true"
        assert params["codex_cli_simplified_flow"] == "true"


# ── Callback server ──────────────────────────────────────────────────────


class TestCallbackServer:
    """Tests for the local OAuth callback HTTP server."""

    async def _server_client(
        self, state: str, code_future: asyncio.Future
    ) -> tuple[aiohttp.web.Application, test_utils.TestClient]:
        app = await _start_callback_server(state, code_future, "127.0.0.1", 0)
        server = test_utils.TestServer(app)
        client = test_utils.TestClient(server)
        await client.start_server()
        return app, client

    async def test_callback_server_returns_index_html(self, state: str) -> None:
        """GET / returns an HTML landing page."""
        code_future: asyncio.Future = asyncio.get_running_loop().create_future()
        _, client = await self._server_client(state, code_future)
        try:
            async with client.session.get(client.make_url("/")) as resp:
                assert resp.status == 200
                text = await resp.text()
                assert "<html" in text.lower() or "waiting" in text.lower() or "authorization" in text.lower()
        finally:
            await client.close()

    async def test_callback_server_validates_state(self, state: str) -> None:
        """Wrong state → returns 400, resolves future with OAuthAuthorizationError (not a code)."""
        code_future: asyncio.Future = asyncio.get_running_loop().create_future()
        _, client = await self._server_client(state, code_future)
        try:
            async with client.session.get(
                client.make_url("/auth/callback"),
                params={"code": "auth-code-xyz", "state": "wrong-state"},
            ) as resp:
                assert resp.status == 400
            # Future IS resolved, but with an error object (not a code string)
            assert code_future.done()
            result = code_future.result()
            assert isinstance(result, OAuthAuthorizationError)
            assert result.error == "state_mismatch"
        finally:
            await client.close()

    async def test_callback_server_returns_code_on_success(self, state: str) -> None:
        """Valid state + code → resolves future with the auth code."""
        code_future: asyncio.Future = asyncio.get_running_loop().create_future()
        _, client = await self._server_client(state, code_future)
        try:
            async with client.session.get(
                client.make_url("/auth/callback"),
                params={"code": "my-auth-code-42", "state": state},
            ) as resp:
                assert resp.status == 200
            assert code_future.done()
            assert code_future.result() == "my-auth-code-42"
        finally:
            await client.close()

    async def test_callback_server_handles_error_response(self, state: str) -> None:
        """OAuth error params → resolves future with OAuthAuthorizationError."""
        code_future: asyncio.Future = asyncio.get_running_loop().create_future()
        _, client = await self._server_client(state, code_future)
        try:
            async with client.session.get(
                client.make_url("/auth/callback"),
                params={
                    "error": "access_denied",
                    "error_description": "User denied the request",
                    "state": state,
                },
            ) as resp:
                assert resp.status == 400
            assert code_future.done()
            result = code_future.result()
            assert isinstance(result, OAuthAuthorizationError)
            assert result.error == "access_denied"
            assert "User denied" in result.error_description
        finally:
            await client.close()


# ── Port conflict ─────────────────────────────────────────────────────────


class TestPortConflict:
    @pytest.mark.asyncio
    async def test_port_conflict_tries_alternate_ports(self) -> None:
        """Primary port in use → run_oauth_flow tries 1456 next."""
        from kitty.auth import openai_oauth

        attempts: list[int] = []
        original_start = openai_oauth._start_callback_server

        async def tracking_start(
            srv_state: str,
            srv_future: asyncio.Future,
            host: str,
            port: int,
        ) -> aiohttp.web.Application:
            attempts.append(port)
            # Raise OSError so run_oauth_flow thinks the port is in use
            raise OSError(f"Address already in use: {port}")

        openai_oauth._start_callback_server = tracking_start
        try:
            with patch("kitty.auth.openai_oauth.webbrowser.open"), suppress(OAuthPortConflictError):
                await run_oauth_flow()

            assert attempts[0] == 1455, "Should try primary port first"
            assert 1456 in attempts, "Should try first alternate port"
            assert len(attempts) >= 2
        finally:
            openai_oauth._start_callback_server = original_start

    @pytest.mark.asyncio
    async def test_port_conflict_raises_after_5_attempts(self) -> None:
        """All ports in use → raises OAuthPortConflictError."""
        from kitty.auth import openai_oauth

        attempts: list[int] = []
        original_start = openai_oauth._start_callback_server

        async def always_conflict(
            srv_state: str,
            srv_future: asyncio.Future,
            host: str,
            port: int,
        ) -> aiohttp.web.Application:
            attempts.append(port)
            raise OSError(f"Port {port} already in use")

        openai_oauth._start_callback_server = always_conflict
        try:
            with pytest.raises(OAuthPortConflictError):
                await run_oauth_flow()

            assert attempts == [1455, 1456, 1457, 1458, 1459]
        finally:
            openai_oauth._start_callback_server = original_start


# ── Timeout ───────────────────────────────────────────────────────────────


class TestTimeout:
    @pytest.mark.asyncio
    async def test_flow_times_out_if_no_callback(self) -> None:
        """Callback not received within timeout → raises OAuthTimeoutError."""
        from kitty.auth import openai_oauth

        original_start = openai_oauth._start_callback_server

        async def noop_start(
            srv_state: str,
            srv_future: asyncio.Future,
            host: str,
            port: int,
        ) -> aiohttp.web.Application:
            app = aiohttp.web.Application()
            # Server is started but the future is never resolved
            app.router.add_get(
                "/auth/callback",
                lambda req: aiohttp.web.Response(text="ok"),
            )
            return app

        openai_oauth._start_callback_server = noop_start
        try:
            with (
                patch("kitty.auth.openai_oauth.webbrowser.open"),
                patch("kitty.auth.openai_oauth.OAUTH_TIMEOUT_SECONDS", 1),
                pytest.raises(OAuthTimeoutError),
            ):
                await run_oauth_flow()
        finally:
            openai_oauth._start_callback_server = original_start


# ── Token exchange ────────────────────────────────────────────────────────


class TestExchangeCodeForTokens:
    @pytest.mark.asyncio
    async def test_exchange_code_payload_is_correct(self, code_verifier: str) -> None:
        """POST body to token URL has all required authorization_code grant params."""
        captured_body: dict[str, Any] = {}
        code = "exchange-me-code"

        with aioresponses() as m:
            # The callback receives (url, **kwargs); data is in kwargs["data"]
            m.post(
                OAUTH_TOKEN_URL,
                callback=lambda url, **kw: _capture_body(captured_body, kw),
                payload={
                    "access_token": "at_test",
                    "refresh_token": "rt_test",
                    "id_token": "id_test",
                    "api_key": "sk_test",
                    "expires_in": 3600,
                },
            )

            async with aiohttp.ClientSession() as http:
                await _exchange_code_for_tokens(code, code_verifier, CLIENT_ID, http)

        assert captured_body.get("grant_type") == "authorization_code"
        assert captured_body.get("code") == code
        assert captured_body.get("redirect_uri") == REDIRECT_URI
        assert captured_body.get("client_id") == CLIENT_ID
        assert captured_body.get("code_verifier") == code_verifier


class TestExchangeIdTokenForApiKey:
    @pytest.mark.asyncio
    async def test_exchange_id_token_payload_is_correct(self) -> None:
        """Token-exchange POST body has grant_type, requested_token, subject_token, subject_token_type."""
        captured_body: dict[str, Any] = {}
        captured_headers: dict[str, str] = {}
        id_token = "my-id-token-xyz"
        access_token = "my-access-token-abc"

        def capture(url, **kw):
            _capture_body(captured_body, kw)
            if "headers" in kw:
                captured_headers.update(kw["headers"])

        with aioresponses() as m:
            m.post(
                OAUTH_TOKEN_URL,
                callback=capture,
                payload={"openai_api_key": "sk-exchanged-key-123"},
            )

            async with aiohttp.ClientSession() as http:
                result = await _exchange_id_token_for_api_key(id_token, access_token, CLIENT_ID, http)

        assert result == "sk-exchanged-key-123"
        assert captured_body.get("grant_type") == TOKEN_EXCHANGE_GRANT
        assert captured_body.get("requested_token") == "openai-api-key"
        assert captured_body.get("subject_token") == id_token
        assert captured_body.get("subject_token_type") == ID_TOKEN_TYPE
        assert captured_body.get("client_id") == CLIENT_ID
        assert captured_headers.get("Authorization") == f"Bearer {access_token}"


# ── run_oauth_flow integration ─────────────────────────────────────────────


class TestRunOAuthFlow:
    @pytest.mark.asyncio
    async def test_run_oauth_flow_full_success(self) -> None:
        """End-to-end: PKCE → server → browser open → callback → token exchange → OAuthSession."""
        from kitty.auth import openai_oauth

        original_start = openai_oauth._start_callback_server

        async def resolving_start(
            srv_state: str,
            srv_future: asyncio.Future,
            host: str,
            port: int,
        ) -> aiohttp.web.Application:
            app = aiohttp.web.Application()

            async def on_callback(request: aiohttp.web.Request) -> aiohttp.web.Response:
                if not srv_future.done():
                    srv_future.set_result(request.query.get("code", "test-code"))
                return aiohttp.web.Response(text="<html><body>You can close this tab.</body></html>")

            app.router.add_get("/auth/callback", on_callback)

            # Simulate the browser hitting the callback after a short delay
            async def _simulate_callback() -> None:
                await asyncio.sleep(0.1)
                if not srv_future.done():
                    srv_future.set_result("test-code")

            asyncio.get_running_loop().create_task(_simulate_callback())
            return app

        openai_oauth._start_callback_server = resolving_start
        try:
            with patch("kitty.auth.openai_oauth.webbrowser.open"), aioresponses() as m:
                # First POST: authorization_code grant → tokens
                m.post(
                    OAUTH_TOKEN_URL,
                    payload={
                        "access_token": "at_full_flow",
                        "refresh_token": "rt_full_flow",
                        "id_token": "id_full_flow",
                        "api_key": "sk_from_code_exchange",
                        "expires_in": 3600,
                    },
                )
                # Second POST: token-exchange grant → API key
                m.post(
                    OAUTH_TOKEN_URL,
                    payload={"openai_api_key": "sk-exchanged-full"},
                )

                async with aiohttp.ClientSession() as http:
                    result = await run_oauth_flow(http=http)

            assert isinstance(result, OAuthSession)
            assert result.client_id == CLIENT_ID
            assert result.access_token == "at_full_flow"
            assert result.api_key == "sk-exchanged-full"
            assert result.id_token == "id_full_flow"
        finally:
            openai_oauth._start_callback_server = original_start


# ── Helpers ───────────────────────────────────────────────────────────────


def _capture_body(captured: dict[str, Any], kwargs: dict[str, Any]) -> None:
    """Extract the POST form data from aioresponses kwargs into *captured*."""
    data = kwargs.get("data", {})
    if isinstance(data, dict):
        captured.update(data)
    elif isinstance(data, str):
        # application/x-www-form-urlencoded: key=value&key2=value2
        for part in data.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                captured[urllib.parse.unquote(k)] = urllib.parse.unquote(v)
    # If data is bytes, decode first
    elif isinstance(data, bytes):
        _capture_body(captured, {"data": data.decode("utf-8")})



class TestOAuthTokenResponseValidation:
    """F35: missing token fields should raise OAuthAuthorizationError, not KeyError."""

    @pytest.mark.asyncio
    async def test_missing_access_token_raises_oauth_authorization_error(self, monkeypatch):
        """Token response missing access_token → OAuthAuthorizationError."""
        from kitty.auth.openai_oauth import OAuthAuthorizationError, _exchange_code_for_tokens

        with aioresponses() as m:
            m.post(
                OAUTH_TOKEN_URL,
                payload={"id_token": "id-only", "refresh_token": "rt"},
            )
            async with aiohttp.ClientSession() as http:
                with pytest.raises(OAuthAuthorizationError, match="Token response missing required fields"):
                    await _exchange_code_for_tokens("code", "verifier", CLIENT_ID, http)

    @pytest.mark.asyncio
    async def test_missing_id_token_raises_oauth_authorization_error(self, monkeypatch):
        """Token response missing id_token → OAuthAuthorizationError."""
        from kitty.auth.openai_oauth import OAuthAuthorizationError, _exchange_code_for_tokens

        with aioresponses() as m:
            m.post(
                OAUTH_TOKEN_URL,
                payload={"access_token": "at-only", "refresh_token": "rt"},
            )
            async with aiohttp.ClientSession() as http:
                with pytest.raises(OAuthAuthorizationError, match="Token response missing required fields"):
                    await _exchange_code_for_tokens("code", "verifier", CLIENT_ID, http)
