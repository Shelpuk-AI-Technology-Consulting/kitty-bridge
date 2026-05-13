"""Tests for OpenAI subscription provider adapter."""

from __future__ import annotations

import contextlib
import json
import os
import time
import unittest.mock
from collections.abc import AsyncIterator
from http.cookiejar import Cookie
from pathlib import Path

import curl_cffi.requests
import pytest

from kitty.auth.oauth_session import OAuthSession
from kitty.providers.openai_subscription import (
    _CF_COOKIE_ALLOWLIST,
    _CODEX_BACKEND_URL,
    OpenAISubscriptionAdapter,
)

# ── Async mock helpers ────────────────────────────────────────────────────


async def _async_yield(value: object) -> AsyncIterator:
    """Yield a single value as an async context manager body."""
    yield value


def _make_mock_codex_response(
    *,
    status_code: int,
    text: str = "",
    content: bytes | None = None,
) -> unittest.mock.MagicMock:
    """Create a mock curl_cffi response object.

    curl_cffi response properties (status_code, text, content) are
    synchronous — no ``await`` needed.
    """
    mock_resp = unittest.mock.MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = text
    mock_resp.content = content if content is not None else text.encode()
    return mock_resp


def _make_streaming_codex_response(
    *,
    status_code: int,
    chunks: list[bytes],
) -> unittest.mock.MagicMock:
    """Create a mock curl_cffi streaming response with async content iterator."""
    mock_resp = unittest.mock.MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = b"".join(chunks).decode("utf-8", errors="replace")
    mock_resp.content = b"".join(chunks)

    async def _aiter_content():
        for chunk in chunks:
            yield chunk

    mock_resp.aiter_content = _aiter_content
    return mock_resp


def _make_streaming_error_response(
    *,
    status_code: int,
    chunks_before_error: list[bytes],
    error_message: str,
) -> unittest.mock.MagicMock:
    """Create a streaming response that errors mid-iteration."""
    mock_resp = unittest.mock.MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = b"".join(chunks_before_error).decode("utf-8", errors="replace")
    mock_resp.content = b"".join(chunks_before_error)

    async def _aiter_content():
        for chunk in chunks_before_error:
            yield chunk
        raise Exception(error_message)

    mock_resp.aiter_content = _aiter_content
    return mock_resp


@contextlib.contextmanager
def _mock_curl_session(mock_response: object) -> unittest.mock.MagicMock:
    """Patch the provider's ``_curl_session`` property to return a mock.

    The mock session has a ``post()`` async method that returns the given
    mock response.  Also patches ``aiohttp.ClientSession`` so the OAuth
    token refresh path gets a no-op session (tokens are fresh in fixtures).
    """
    mock_session = unittest.mock.AsyncMock()
    mock_session.post = unittest.mock.AsyncMock(return_value=mock_response)
    mock_session.close = unittest.mock.MagicMock()

    with unittest.mock.patch.object(
        OpenAISubscriptionAdapter,
        "_curl_session",
        new_callable=unittest.mock.PropertyMock,
        return_value=mock_session,
    ):
        # aiohttp is imported locally inside the methods, so we must
        # patch it at the module level.  Since test fixtures have
        # non-expired tokens, get_valid_api_key() won't actually call
        # the session — but we need it to be instantiable.
        mock_aiohttp_session = unittest.mock.MagicMock()
        mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
        mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
        with unittest.mock.patch(
            "aiohttp.ClientSession",
            return_value=mock_aiohttp_session,
        ):
            yield mock_session


def _make_mock_oauth_http() -> unittest.mock.MagicMock:
    """Create a mock aiohttp session that supports OAuth refresh calls."""
    mock_aiohttp = unittest.mock.MagicMock()
    mock_aiohttp.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp)
    mock_aiohttp.__aexit__ = unittest.mock.AsyncMock(return_value=False)

    refresh_resp = unittest.mock.MagicMock()
    refresh_resp.status = 200
    refresh_resp.json = unittest.mock.AsyncMock(
        return_value={
            "access_token": "at_refreshed",
            "refresh_token": "rt_refreshed",
            "id_token": _make_id_token("acct-1234"),
            "openai_api_key": "api_key_refreshed",
            "expires_in": 3600,
        },
    )
    refresh_cm = unittest.mock.MagicMock()
    refresh_cm.__aenter__ = unittest.mock.AsyncMock(return_value=refresh_resp)
    refresh_cm.__aexit__ = unittest.mock.AsyncMock(return_value=False)
    mock_aiohttp.post.return_value = refresh_cm
    return mock_aiohttp


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def adapter() -> OpenAISubscriptionAdapter:
    return OpenAISubscriptionAdapter()


def _make_id_token(account_id: str | None = None) -> str:
    """Build a minimal JWT id_token with an optional chatgpt_account_id."""
    header = "eyJhbGciOiJIUzI1NiJ9"
    payload_dict: dict = {}
    if account_id is not None:
        payload_dict["https://api.openai.com/auth"] = {
            "chatgpt_account_id": account_id,
        }
    payload_json = json.dumps(payload_dict)
    import base64

    payload = base64.urlsafe_b64encode(payload_json.encode()).rstrip(b"=").decode()
    signature = "fake_sig"
    return f"{header}.{payload}.{signature}"


@pytest.fixture()
def fresh_session(tmp_path: Path) -> tuple[OAuthSession, Path]:
    """Create a fresh OAuthSession file (tokens not expired)."""
    now = time.time()
    id_token = _make_id_token("acct-1234")
    session = OAuthSession(
        client_id="app_test",
        access_token="at_fresh",
        refresh_token="rt_fresh",
        id_token=id_token,
        api_key=None,
        access_token_expires_at=now + 3600,
        api_key_expires_at=now + 3600,
        _file_path=str(tmp_path / "oauth_session.json"),
    )
    session.save()
    return session, Path(session._file_path)


@pytest.fixture()
def cc_request(fresh_session: tuple[OAuthSession, Path]) -> dict:
    """A typical cc_request with _resolved_key pointing to the session file."""
    _, session_path = fresh_session
    return {
        "model": "gpt-5.4",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        "stream": False,
        "tools": [
            {
                "function": {
                    "name": "bash",
                    "description": "Run a shell command",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                },
                "type": "function",
            }
        ],
        "_resolved_key": str(session_path),
        "_provider_config": {},
    }


@pytest.fixture()
def responses_body() -> dict:
    """A typical Responses API request body."""
    return {
        "model": "gpt-5.4",
        "instructions": "You are helpful.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ],
        "stream": True,
        "tools": [
            {
                "type": "function",
                "name": "bash",
                "description": "Run a shell command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                "strict": True,  # Should be stripped
            }
        ],
    }


# ── Properties ──────────────────────────────────────────────────────────────


class TestProperties:
    def test_provider_type(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.provider_type == "openai_subscription"

    def test_use_custom_transport(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.use_custom_transport is True

    def test_default_base_url_is_codex_backend(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.default_base_url == _CODEX_BACKEND_URL
        assert "chatgpt.com" in adapter.default_base_url

    def test_normalize_model_name_strips_prefix(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.normalize_model_name("openai/gpt-5.4") == "gpt-5.4"

    def test_normalize_model_name_no_prefix(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.normalize_model_name("gpt-5.4") == "gpt-5.4"


# ── Helpers ──────────────────────────────────────────────────────────────


class TestExtractAccountId:
    def test_extracts_account_id_from_jwt(self) -> None:
        token = _make_id_token("my-acct-123")
        assert OpenAISubscriptionAdapter._extract_account_id(token) == "my-acct-123"

    def test_returns_none_if_missing(self) -> None:
        token = _make_id_token(None)
        assert OpenAISubscriptionAdapter._extract_account_id(token) is None

    def test_returns_none_on_garbage(self) -> None:
        assert OpenAISubscriptionAdapter._extract_account_id("not-a-jwt") is None


class TestPrepareResponsesBody:
    def test_strips_strict_from_tools(self) -> None:
        body = {
            "tools": [
                {"type": "function", "name": "bash", "strict": True, "parameters": {}},
            ],
        }
        result = OpenAISubscriptionAdapter._prepare_responses_body(body)
        assert "strict" not in result["tools"][0]

    def test_sets_store_false(self) -> None:
        body = {}
        result = OpenAISubscriptionAdapter._prepare_responses_body(body)
        assert result["store"] is False

    def test_sets_stream_true(self) -> None:
        body = {}
        result = OpenAISubscriptionAdapter._prepare_responses_body(body)
        assert result["stream"] is True


class TestCcToResponses:
    def test_converts_system_message_to_instructions(self, adapter: OpenAISubscriptionAdapter) -> None:
        cc = {
            "model": "gpt-5.4",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = adapter._cc_to_responses(cc)
        assert result["instructions"] == "Be helpful."
        assert len(result["input"]) == 1
        assert result["input"][0]["role"] == "user"

    def test_converts_tools(self, adapter: OpenAISubscriptionAdapter) -> None:
        cc = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
        result = adapter._cc_to_responses(cc)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "bash"
        assert "function" not in result["tools"][0]

    def test_does_not_include_max_output_tokens(self, adapter: OpenAISubscriptionAdapter) -> None:
        """The Codex backend rejects max_output_tokens with 400."""
        cc = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 4096,
        }
        result = adapter._cc_to_responses(cc)
        assert "max_output_tokens" not in result
        assert "max_tokens" not in result


# ── map_error ─────────────────────────────────────────────────────────────


class TestMapError:
    def test_map_error_429_rate_limited(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(429, {"error": {"message": "Rate limited"}})
        assert "rate limited" in str(error).lower()
        assert "Rate limited" in str(error)
        assert error.http_status == 429

    def test_map_error_429_with_retry_after_header(self, adapter: OpenAISubscriptionAdapter) -> None:
        """429 error gets retry_after set when response has Retry-After header."""
        error = adapter.map_error(429, {"error": {"message": "Rate limited"}})
        assert error.http_status == 429
        assert error.retry_after is None

        # Simulate attaching Retry-After from response headers
        mock_resp = unittest.mock.MagicMock()
        mock_resp.headers = {"Retry-After": "17"}
        adapter._attach_retry_after(error, mock_resp)
        assert error.retry_after == 17

    def test_map_error_429_without_retry_after_header(self, adapter: OpenAISubscriptionAdapter) -> None:
        """429 error without Retry-After header leaves retry_after as None."""
        error = adapter.map_error(429, {"error": {"message": "Rate limited"}})
        mock_resp = unittest.mock.MagicMock()
        mock_resp.headers = {}
        adapter._attach_retry_after(error, mock_resp)
        assert error.retry_after is None

    def test_parse_retry_after_accepts_float_seconds(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter._parse_retry_after("1.5") == 1

    def test_parse_retry_after_has_one_second_floor(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter._parse_retry_after("0") == 1
        assert adapter._parse_retry_after("-10") == 1

    def test_map_error_400_includes_status(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(400, {"error": {"message": "Bad request"}})
        assert "400" in str(error)
        assert "Bad request" in str(error)
        assert error.http_status == 400

    def test_map_error_401_mentions_reauth(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(401, {"error": {"message": "Unauthorized", "code": "invalid_api_key"}})
        assert "re-authenticate" in str(error)
        assert error.http_status == 401

    def test_map_error_403_cloudflare_html(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(403, {"error": {"message": "<html><head>cf-mitigated: challenge</head></html>"}})
        msg = str(error).lower()
        assert "cloudflare" in msg
        assert "not an api key problem" in msg
        assert error.http_status == 403

    def test_map_error_403_cloudflare_message_does_not_say_retry_wont_help(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """CF error message should not claim retries won't help."""
        error = adapter.map_error(403, {"error": {"message": "<html><head>cf-mitigated: challenge</head></html>"}})
        msg = str(error).lower()
        assert "will not help" not in msg
        assert "transient" in msg or "retry" in msg

    def test_map_error_403_non_cloudflare(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(403, {"error": {"message": "Forbidden"}})
        msg = str(error)
        assert "access denied" in msg
        assert "cloudflare" not in msg.lower()
        assert error.http_status == 403

    def test_map_error_cf_sets_is_cloudflare(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(403, {"error": {"message": "<html><head>cf-mitigated: challenge</head></html>"}})
        assert error.is_cloudflare is True

    def test_map_error_non_cf_is_cloudflare_false(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(403, {"error": {"message": "Forbidden"}})
        assert error.is_cloudflare is False

    def test_map_error_401_is_cloudflare_false(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(401, {"error": {"message": "Unauthorized"}})
        assert error.is_cloudflare is False

    def test_map_error_429_is_cloudflare_false(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(429, {"error": {"message": "Rate limited"}})
        assert error.is_cloudflare is False

    def test_map_error_500_sets_http_status(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(500, {"error": {"message": "Internal error"}})
        assert error.http_status == 500
        assert "500" in str(error)


# ── Cloudflare HTML detection ──────────────────────────────────────────────

_CLOUDFLARE_HTML = """<html>
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style global>body{font-family:Arial,Helvetica,sans-serif}</style>
    <meta http-equiv="refresh" content="360">
  </head>
  <body>
    <div class="container">
      <div class="logo">
        <svg xmlns="http://www.w3.org/2000/svg"></svg>
      </div>
      <div class="cf-browser-verification cf-im-under-attack">
        Please wait...
      </div>
    </div>
    <script>
    window._cf_chl_opt = {};
    </script>
  </body>
</html>"""

_NON_CF_HTML = "<html><body><h1>Access Denied</h1><p>You do not have permission.</p></body></html>"


class TestIsCloudflareBlock:
    def test_detects_cf_challenge_html(self) -> None:
        assert OpenAISubscriptionAdapter._is_cloudflare_block(403, _CLOUDFLARE_HTML) is True

    def test_detects_cf_mitigated_header(self) -> None:
        body = "cf-mitigated: challenge something something"
        assert OpenAISubscriptionAdapter._is_cloudflare_block(403, body) is True

    def test_detects_cf_chl_opt(self) -> None:
        body = "some text <script>window._cf_chl_opt = {};</script> more text"
        assert OpenAISubscriptionAdapter._is_cloudflare_block(403, body) is True

    def test_non_cf_html_returns_false(self) -> None:
        assert OpenAISubscriptionAdapter._is_cloudflare_block(403, _NON_CF_HTML) is False

    def test_json_error_returns_false(self) -> None:
        body = '{"error": {"message": "Forbidden", "code": "forbidden"}}'
        assert OpenAISubscriptionAdapter._is_cloudflare_block(403, body) is False

    def test_non_403_returns_false(self) -> None:
        assert OpenAISubscriptionAdapter._is_cloudflare_block(200, _CLOUDFLARE_HTML) is False

    def test_429_with_html_returns_false(self) -> None:
        assert OpenAISubscriptionAdapter._is_cloudflare_block(429, _CLOUDFLARE_HTML) is False


# ── Codex CLI headers ──────────────────────────────────────────────────────


class TestCodexHeaders:
    def test_user_agent_codex_format(self, adapter: OpenAISubscriptionAdapter) -> None:
        """User-Agent matches Codex CLI: codex_cli_rs/<version> (<os> <ver>; <arch>)"""
        import re

        headers = adapter._build_codex_headers("tok", _make_id_token())
        ua = headers["User-Agent"]
        # codex_cli_rs/0.28.0 (Linux 6.6.87; x86_64)
        pattern = r"^codex_cli_rs/[\d.]+ \(.+; .+\)$"
        assert re.match(pattern, ua), f"UA does not match Codex format: {ua!r}"

    def test_user_agent_contains_os_info(self, adapter: OpenAISubscriptionAdapter) -> None:
        """User-Agent includes OS type, version, and architecture."""
        headers = adapter._build_codex_headers("tok", _make_id_token())
        ua = headers["User-Agent"]
        # Must contain OS name, version, and arch separated correctly
        import platform

        assert platform.system() in ua
        assert platform.release() in ua
        assert platform.machine() in ua

    def test_no_browser_headers(self, adapter: OpenAISubscriptionAdapter) -> None:
        """Codex CLI does not send browser-like headers."""
        headers = adapter._build_codex_headers("tok", _make_id_token())
        assert "Origin" not in headers
        assert "Referer" not in headers
        assert "OpenAI-Beta" not in headers

    def test_accept_event_stream(self, adapter: OpenAISubscriptionAdapter) -> None:
        """Codex CLI sets Accept: text/event-stream (all Codex requests are streaming)."""
        headers = adapter._build_codex_headers("tok", _make_id_token())
        assert headers["Accept"] == "text/event-stream"

    def test_authorization_bearer(self, adapter: OpenAISubscriptionAdapter) -> None:
        headers = adapter._build_codex_headers("my-token", _make_id_token())
        assert headers["Authorization"] == "Bearer my-token"

    def test_account_id_from_jwt(self, adapter: OpenAISubscriptionAdapter) -> None:
        """Account ID header uses Codex CLI casing: ChatGPT-Account-Id."""
        headers = adapter._build_codex_headers("tok", _make_id_token("acct-42"))
        assert headers["ChatGPT-Account-Id"] == "acct-42"

    def test_no_account_id_without_jwt_claim(self, adapter: OpenAISubscriptionAdapter) -> None:
        headers = adapter._build_codex_headers("tok", _make_id_token(None))
        assert "ChatGPT-Account-Id" not in headers

    def test_no_originator_header(self, adapter: OpenAISubscriptionAdapter) -> None:
        """The originator header triggers strict tool validation on the backend."""
        headers = adapter._build_codex_headers("tok", _make_id_token())
        assert "originator" not in headers

    def test_version_header_matches_codex(self, adapter: OpenAISubscriptionAdapter) -> None:
        """Codex CLI sets a version header from CARGO_PKG_VERSION."""
        headers = adapter._build_codex_headers("tok", _make_id_token())
        assert "version" in headers, "Missing version header"
        # Must match the Codex CLI release version, not Kitty's version
        from kitty.providers.openai_subscription import _CODEX_CLI_VERSION

        assert headers["version"] == _CODEX_CLI_VERSION


# ── curl_cffi error mapping ──────────────────────────────────────────────


class TestHandleCurlError:
    def test_timeout_error(self) -> None:
        exc = Exception("Connection timed out after 30s")
        result = OpenAISubscriptionAdapter._handle_curl_error(exc)
        assert "timed out" in str(result)

    def test_connection_error(self) -> None:
        exc = Exception("Connection refused: chatgpt.com")
        result = OpenAISubscriptionAdapter._handle_curl_error(exc)
        assert "connection failed" in str(result)

    def test_generic_error(self) -> None:
        exc = Exception("Something unexpected")
        result = OpenAISubscriptionAdapter._handle_curl_error(exc)
        assert "request failed" in str(result)


class TestIsTransientStreamError:
    def test_curl_56_is_transient(self) -> None:
        assert OpenAISubscriptionAdapter._is_transient_stream_error(
            Exception("Failed to perform, curl: (56) Connection closed abruptly.")
        )

    def test_recv_error_is_transient(self) -> None:
        assert OpenAISubscriptionAdapter._is_transient_stream_error(Exception("Recv error: connection reset"))

    def test_connection_closed_is_transient(self) -> None:
        assert OpenAISubscriptionAdapter._is_transient_stream_error(Exception("Connection closed by server"))

    def test_timeout_is_not_transient(self) -> None:
        assert not OpenAISubscriptionAdapter._is_transient_stream_error(Exception("Connection timed out after 30s"))

    def test_generic_error_is_not_transient(self) -> None:
        assert not OpenAISubscriptionAdapter._is_transient_stream_error(Exception("Something unexpected"))


# ── Session reuse ────────────────────────────────────────────────────────


class TestCurlSessionReuse:
    def test_same_instance_returned(self, adapter: OpenAISubscriptionAdapter) -> None:
        """The lazy _curl_session property returns the same instance."""
        s1 = adapter._curl_session
        s2 = adapter._curl_session
        assert s1 is s2


# ── TLS fingerprint / impersonation ──────────────────────────────────────


class TestTlsImpersonation:
    """Test that curl_cffi session is created with correct TLS impersonation."""

    def test_session_created_with_impersonate(self, adapter: OpenAISubscriptionAdapter) -> None:
        """AsyncSession must be created with impersonate= for a browser-like TLS fingerprint."""
        from kitty.providers.openai_subscription import _CODEX_IMPERSONATE

        with unittest.mock.patch(
            "kitty.providers.openai_subscription.curl_cffi.requests.AsyncSession",
        ) as mock_session_cls:
            mock_instance = unittest.mock.MagicMock()
            mock_instance.cookies = unittest.mock.MagicMock()
            mock_instance.cookies.jar = []
            mock_session_cls.return_value = mock_instance
            _ = adapter._curl_session
            mock_session_cls.assert_called_once_with(impersonate=_CODEX_IMPERSONATE)

    def test_impersonate_is_valid_browser_type(self) -> None:
        """The _CODEX_IMPERSONATE constant must be a valid curl_cffi browser type."""
        from curl_cffi.requests import BrowserType

        from kitty.providers.openai_subscription import _CODEX_IMPERSONATE

        # BrowserType accepts string values — verify it's recognized
        assert hasattr(BrowserType, _CODEX_IMPERSONATE)


# ── Codex backoff formula ─────────────────────────────────────────────────


class TestCodexBackoff:
    """Test _codex_backoff matches Codex CLI's retry.rs formula."""

    def test_first_retry_base_delay(self) -> None:
        """First retry (attempt=1) returns ~200ms."""
        from kitty.providers.openai_subscription import _codex_backoff

        delay = _codex_backoff(1)
        assert 0.180 <= delay <= 0.220  # 200ms * jitter(0.9..1.1)

    def test_second_retry_doubled(self) -> None:
        """Second retry (attempt=2) returns ~400ms."""
        from kitty.providers.openai_subscription import _codex_backoff

        delay = _codex_backoff(2)
        assert 0.360 <= delay <= 0.440  # 400ms * jitter(0.9..1.1)

    def test_third_retry_quadrupled(self) -> None:
        """Third retry (attempt=3) returns ~800ms."""
        from kitty.providers.openai_subscription import _codex_backoff

        delay = _codex_backoff(3)
        assert 0.720 <= delay <= 0.880  # 800ms * jitter(0.9..1.1)

    def test_fourth_retry_octupled(self) -> None:
        """Fourth retry (attempt=4) returns ~1600ms."""
        from kitty.providers.openai_subscription import _codex_backoff

        delay = _codex_backoff(4)
        assert 1.440 <= delay <= 1.760  # 1600ms * jitter(0.9..1.1)


# ── ChatGPT host matching ────────────────────────────────────────────────


class TestIsChatgptHost:
    """Test _is_chatgpt_host matches Codex CLI's chatgpt_hosts.rs."""

    def test_exact_chatgpt_com(self) -> None:
        from kitty.providers.openai_subscription import _is_chatgpt_host

        assert _is_chatgpt_host("chatgpt.com")

    def test_exact_chat_openai_com(self) -> None:
        from kitty.providers.openai_subscription import _is_chatgpt_host

        assert _is_chatgpt_host("chat.openai.com")

    def test_exact_chatgpt_staging_com(self) -> None:
        from kitty.providers.openai_subscription import _is_chatgpt_host

        assert _is_chatgpt_host("chatgpt-staging.com")

    def test_subdomain_of_chatgpt_com(self) -> None:
        from kitty.providers.openai_subscription import _is_chatgpt_host

        assert _is_chatgpt_host("foo.chatgpt.com")

    def test_subdomain_of_chatgpt_staging_com(self) -> None:
        from kitty.providers.openai_subscription import _is_chatgpt_host

        assert _is_chatgpt_host("bar.chatgpt-staging.com")

    def test_non_matching_host(self) -> None:
        from kitty.providers.openai_subscription import _is_chatgpt_host

        assert not _is_chatgpt_host("api.openai.com")

    def test_evil_chatgpt_rejected(self) -> None:
        from kitty.providers.openai_subscription import _is_chatgpt_host

        assert not _is_chatgpt_host("evilchatgpt.com")

    def test_chat_openai_subdomain_rejected(self) -> None:
        from kitty.providers.openai_subscription import _is_chatgpt_host

        assert not _is_chatgpt_host("foo.chat.openai.com")


# ── Custom CA certificate support ────────────────────────────────────────


class TestCustomCaCertificate:
    """Test CA certificate path resolution matching Codex CLI's custom_ca.rs."""

    def test_resolve_ca_prefers_codex_env(self, tmp_path: Path) -> None:
        """CODEX_CA_CERTIFICATE takes precedence over SSL_CERT_FILE."""
        from kitty.providers.openai_subscription import _resolve_ca_cert_path

        codex_path = str(tmp_path / "codex.pem")
        ssl_path = str(tmp_path / "ssl.pem")
        with unittest.mock.patch.dict(
            os.environ,
            {
                "CODEX_CA_CERTIFICATE": codex_path,
                "SSL_CERT_FILE": ssl_path,
            },
        ):
            result = _resolve_ca_cert_path()
        assert result == codex_path

    def test_resolve_ca_falls_back_to_ssl_cert_file(self, tmp_path: Path) -> None:
        """SSL_CERT_FILE is used when CODEX_CA_CERTIFICATE is not set."""
        from kitty.providers.openai_subscription import _resolve_ca_cert_path

        ssl_path = str(tmp_path / "ssl.pem")
        with unittest.mock.patch.dict(os.environ, {"SSL_CERT_FILE": ssl_path}, clear=False):
            env = os.environ.copy()
            env.pop("CODEX_CA_CERTIFICATE", None)
            with unittest.mock.patch.dict(os.environ, env, clear=True):
                result = _resolve_ca_cert_path()
        assert result == ssl_path

    def test_resolve_ca_returns_none_when_unset(self) -> None:
        """Returns None when neither env var is set."""
        from kitty.providers.openai_subscription import _resolve_ca_cert_path

        env = {k: v for k, v in os.environ.items() if k not in ("CODEX_CA_CERTIFICATE", "SSL_CERT_FILE")}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            result = _resolve_ca_cert_path()
        assert result is None

    def test_resolve_ca_ignores_empty_values(self) -> None:
        """Empty string values are treated as unset."""
        from kitty.providers.openai_subscription import _resolve_ca_cert_path

        with unittest.mock.patch.dict(
            os.environ,
            {
                "CODEX_CA_CERTIFICATE": "",
                "SSL_CERT_FILE": "",
            },
        ):
            result = _resolve_ca_cert_path()
        assert result is None

    def test_resolve_ca_empty_codex_falls_back_to_ssl(self, tmp_path: Path) -> None:
        """Empty CODEX_CA_CERTIFICATE falls back to SSL_CERT_FILE."""
        from kitty.providers.openai_subscription import _resolve_ca_cert_path

        ssl_path = str(tmp_path / "ssl.pem")
        with unittest.mock.patch.dict(
            os.environ,
            {
                "CODEX_CA_CERTIFICATE": "",
                "SSL_CERT_FILE": ssl_path,
            },
        ):
            result = _resolve_ca_cert_path()
        assert result == ssl_path

    def test_session_created_with_verify_when_ca_set(self, adapter: OpenAISubscriptionAdapter, tmp_path: Path) -> None:
        """AsyncSession receives verify= when a CA cert path is configured."""
        from kitty.providers.openai_subscription import _CODEX_IMPERSONATE

        ca_path = str(tmp_path / "ca.pem")
        with (
            unittest.mock.patch.dict(os.environ, {"CODEX_CA_CERTIFICATE": ca_path}),
            unittest.mock.patch(
                "kitty.providers.openai_subscription.curl_cffi.requests.AsyncSession",
            ) as mock_session_cls,
        ):
            mock_instance = unittest.mock.MagicMock()
            mock_instance.cookies = unittest.mock.MagicMock()
            mock_instance.cookies.jar = []
            mock_session_cls.return_value = mock_instance
            _ = adapter._curl_session
            mock_session_cls.assert_called_once_with(
                impersonate=_CODEX_IMPERSONATE,
                verify=ca_path,
            )

    def test_session_created_without_verify_when_no_ca(self, adapter: OpenAISubscriptionAdapter) -> None:
        """AsyncSession does not receive verify= when no CA cert is configured."""
        from kitty.providers.openai_subscription import _CODEX_IMPERSONATE

        env = {k: v for k, v in os.environ.items() if k not in ("CODEX_CA_CERTIFICATE", "SSL_CERT_FILE")}
        with (
            unittest.mock.patch.dict(os.environ, env, clear=True),
            unittest.mock.patch(
                "kitty.providers.openai_subscription.curl_cffi.requests.AsyncSession",
            ) as mock_session_cls,
        ):
            mock_instance = unittest.mock.MagicMock()
            mock_instance.cookies = unittest.mock.MagicMock()
            mock_instance.cookies.jar = []
            mock_session_cls.return_value = mock_instance
            _ = adapter._curl_session
            mock_session_cls.assert_called_once_with(impersonate=_CODEX_IMPERSONATE)


# ── Cloudflare detection in request methods ──────────────────────────────


class TestStreamRequestCloudflare:
    """Test that stream_request detects Cloudflare blocks and raises specific error."""

    @pytest.mark.asyncio()
    async def test_stream_request_raises_cloudflare_error(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """stream_request should raise ProviderError with Cloudflare message on CF block."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        from kitty.providers.base import ProviderError

        mock_resp = _make_mock_codex_response(status_code=403, text=_CLOUDFLARE_HTML)
        with _mock_curl_session(mock_resp):
            with pytest.raises(ProviderError, match="[Cc]loudflare"):
                await adapter.stream_request(cc_request, mock_write)
            assert written == []


class TestMakeRequestCloudflare:
    """Test that make_request detects Cloudflare blocks and raises specific error."""

    @pytest.mark.asyncio()
    async def test_make_request_raises_cloudflare_error(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        from kitty.providers.base import ProviderError

        mock_resp = _make_mock_codex_response(status_code=403, text=_CLOUDFLARE_HTML)
        with _mock_curl_session(mock_resp), pytest.raises(ProviderError, match="[Cc]loudflare"):
            await adapter.make_request(cc_request)


# ── Basic request methods ────────────────────────────────────────────────


class TestStreamRequestBasic:
    """Test successful streaming via curl_cffi."""

    @pytest.mark.asyncio()
    async def test_retries_on_abrupt_connection_close(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        first_resp = _make_streaming_error_response(
            status_code=200,
            chunks_before_error=[b'data: {"type":"response.output_text.delta","delta":"Hel"}\n\n'],
            error_message="Failed to perform, curl: (56) Connection closed abruptly.",
        )
        second_resp = _make_streaming_codex_response(
            status_code=200,
            chunks=[
                b'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n',
                b"data: [DONE]\n\n",
            ],
        )

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(side_effect=[first_resp, second_resp])
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
                await adapter.stream_request(cc_request, mock_write)

        assert mock_session.post.await_count == 2
        assert any(b'"delta":"Hello"' in chunk for chunk in written)

    @pytest.mark.asyncio()
    async def test_raises_after_retry_exhaustion(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        err_resp = _make_streaming_error_response(
            status_code=200,
            chunks_before_error=[],
            error_message="Failed to perform, curl: (56) Connection closed abruptly.",
        )

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(side_effect=[err_resp, err_resp, err_resp])
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
                from kitty.providers.base import ProviderError

                with pytest.raises(ProviderError, match="connection failed|request failed"):
                    await adapter.stream_request(cc_request, mock_write)

        assert mock_session.post.await_count == 3
        assert written == []

    @pytest.mark.asyncio()
    async def test_streams_sse_chunks(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        sse_chunks = [
            b'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n',
            b'data: {"type":"response.output_text.delta","delta":" world"}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock_resp = _make_streaming_codex_response(status_code=200, chunks=sse_chunks)
        with _mock_curl_session(mock_resp):
            await adapter.stream_request(cc_request, mock_write)

        assert len(written) == 3
        assert b'"delta":"Hello"' in written[0]
        assert b'"delta":" world"' in written[1]

    @pytest.mark.asyncio()
    async def test_strips_bom_from_chunks(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        # Chunk with UTF-8 BOM prefix
        sse_chunks = [
            b'\xef\xbb\xbfdata: {"type":"response.output_text.delta","delta":"Hi"}\n\n',
        ]
        mock_resp = _make_streaming_codex_response(status_code=200, chunks=sse_chunks)
        with _mock_curl_session(mock_resp):
            await adapter.stream_request(cc_request, mock_write)

        # BOM should be stripped
        assert written[0] == b'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n'


class TestMakeRequestBasic:
    """Test successful non-streaming request via curl_cffi."""

    @pytest.mark.asyncio()
    async def test_parses_sse_to_cc_response(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        sse_body = (
            b'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
            b'data: {"type":"response.completed","response":{"model":"gpt-5.4",'
            b'"status":"completed","usage":{"input_tokens":10,"output_tokens":5}}}\n\n'
            b"data: [DONE]\n\n"
        )
        mock_resp = _make_mock_codex_response(status_code=200, content=sse_body)
        with _mock_curl_session(mock_resp):
            result = await adapter.make_request(cc_request)

        assert result["object"] == "chat.completion"
        assert result["model"] == "gpt-5.4"
        assert result["choices"][0]["message"]["content"] == "Hello"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["input_tokens"] == 10


# ── Regression guards ────────────────────────────────────────────────────


class TestNoManualResponseClose:
    """Guard against accidentally adding resp.close() calls.

    curl_cffi's internal cleanup callback releases handles back to the
    session pool.  Calling resp.close() frees the curl handle, causing a
    TypeError when the cleanup callback subsequently tries to release it.
    """

    @pytest.mark.asyncio()
    async def test_stream_request_does_not_close_response(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        sse_chunks = [b'data: {"type":"response.output_text.delta","delta":"X"}\n\n']
        mock_resp = _make_streaming_codex_response(status_code=200, chunks=sse_chunks)
        with _mock_curl_session(mock_resp):
            await adapter.stream_request(cc_request, mock_write)

        mock_resp.close.assert_not_called()

    @pytest.mark.asyncio()
    async def test_make_request_does_not_close_response(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        sse_body = (
            b'data: {"type":"response.completed","response":'
            b'{"model":"gpt-5.4","status":"completed"}}\n\n'
            b"data: [DONE]\n\n"
        )
        mock_resp = _make_mock_codex_response(status_code=200, content=sse_body)
        with _mock_curl_session(mock_resp):
            await adapter.make_request(cc_request)

        mock_resp.close.assert_not_called()


class TestEmptyResponseBody:
    """Guard against empty SSE body silently producing a valid response."""

    @pytest.mark.asyncio()
    async def test_make_request_raises_on_empty_body(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        from kitty.providers.base import ProviderError

        mock_resp = _make_mock_codex_response(status_code=200, content=b"")
        with _mock_curl_session(mock_resp), pytest.raises(ProviderError, match="empty response"):
            await adapter.make_request(cc_request)


# ── Cloudflare cookie filtering ───────────────────────────────────────────


def _add_cookie(jar: object, name: str, value: str, domain: str) -> None:
    """Add a cookie to a curl_cffi Cookies jar (wraps http.cookiejar.CookieJar)."""
    cookie = Cookie(
        version=0,
        name=name,
        value=value,
        port=None,
        port_specified=False,
        domain=domain,
        domain_specified=True,
        domain_initial_dot=False,
        path="/",
        path_specified=True,
        secure=True,
        expires=None,
        discard=True,
        comment=None,
        comment_url=None,
        rest={},
        rfc2109=False,
    )
    jar.set_cookie(cookie)


class TestFilterCloudflareCookies:
    """Unit tests for _filter_cloudflare_cookies."""

    def test_removes_non_cf_cookies_on_chatgpt_host(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """Non-CF cookies on chatgpt.com should be removed."""
        jar = adapter._curl_session.cookies.jar
        _add_cookie(jar, "__cf_bm", "good", "chatgpt.com")
        _add_cookie(jar, "session_id", "bad", "chatgpt.com")

        removed = OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        names = {c.name for c in jar}
        assert names == {"__cf_bm"}
        assert removed == 1

    def test_keeps_all_allowlisted_cookies(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """Every cookie in the allowlist should survive filtering."""
        jar = adapter._curl_session.cookies.jar
        for name in _CF_COOKIE_ALLOWLIST:
            _add_cookie(jar, name, "val", "chatgpt.com")

        removed = OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        names = {c.name for c in jar}
        assert names == _CF_COOKIE_ALLOWLIST
        assert removed == 0

    def test_keeps_cf_chl_prefixed_cookies(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """Cookies with the cf_chl_ prefix should survive filtering."""
        jar = adapter._curl_session.cookies.jar
        _add_cookie(jar, "cf_chl_rc_i", "val", "chatgpt.com")
        _add_cookie(jar, "cf_chl_custom", "val", "chatgpt.com")

        removed = OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        names = {c.name for c in jar}
        assert "cf_chl_rc_i" in names
        assert "cf_chl_custom" in names
        assert removed == 0

    def test_only_targets_chatgpt_hosts(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """Non-CF cookies on other hosts should be left alone."""
        jar = adapter._curl_session.cookies.jar
        _add_cookie(jar, "session_id", "keep", "api.openai.com")
        _add_cookie(jar, "session_id", "remove", "chatgpt.com")
        _add_cookie(jar, "session_id", "keep2", "chat.openai.com")

        OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        domains = {c.domain for c in jar}
        assert "api.openai.com" in domains
        # chatgpt.com and chat.openai.com non-CF cookies removed
        assert len([c for c in jar if c.domain == "chatgpt.com"]) == 0
        assert len([c for c in jar if c.domain == "chat.openai.com"]) == 0

    def test_empty_jar_is_noop(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """Filtering an empty jar should return 0 removed."""
        removed = OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        assert removed == 0

    def test_repeated_filter_is_idempotent(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """Calling filter twice on the same jar should be a no-op the second time."""
        jar = adapter._curl_session.cookies.jar
        _add_cookie(jar, "__cf_bm", "good", "chatgpt.com")
        _add_cookie(jar, "session_id", "bad", "chatgpt.com")

        removed1 = OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        removed2 = OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        assert removed1 == 1
        assert removed2 == 0

    def test_removes_cookie_with_deep_path(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """Non-CF cookies with non-root paths should still be removed."""
        jar = adapter._curl_session.cookies.jar
        _add_cookie(jar, "__cf_bm", "good", "chatgpt.com")
        deep_cookie = Cookie(
            version=0,
            name="auth_token",
            value="xyz",
            port=None,
            port_specified=False,
            domain="chatgpt.com",
            domain_specified=True,
            domain_initial_dot=False,
            path="/api/v1",
            path_specified=True,
            secure=True,
            expires=None,
            discard=True,
            comment=None,
            comment_url=None,
            rest={},
            rfc2109=False,
        )
        jar.set_cookie(deep_cookie)

        removed = OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        names = {c.name for c in jar}
        assert "auth_token" not in names
        assert "__cf_bm" in names
        assert removed == 1

    def test_removes_non_cf_on_chatgpt_staging(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """Non-CF cookies on chatgpt-staging.com should be removed."""
        jar = adapter._curl_session.cookies.jar
        _add_cookie(jar, "__cf_bm", "good", "chatgpt-staging.com")
        _add_cookie(jar, "session_id", "bad", "chatgpt-staging.com")

        removed = OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        names = {c.name for c in jar}
        assert names == {"__cf_bm"}
        assert removed == 1

    def test_removes_non_cf_on_chatgpt_subdomain(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """Non-CF cookies on foo.chatgpt.com (subdomain) should be removed."""
        jar = adapter._curl_session.cookies.jar
        _add_cookie(jar, "__cf_bm", "good", "foo.chatgpt.com")
        _add_cookie(jar, "session_id", "bad", "foo.chatgpt.com")

        removed = OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        names = {c.name for c in jar}
        assert names == {"__cf_bm"}
        assert removed == 1

    def test_removes_non_cf_on_staging_subdomain(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """Non-CF cookies on api.chatgpt-staging.com should be removed."""
        jar = adapter._curl_session.cookies.jar
        _add_cookie(jar, "__cf_bm", "good", "api.chatgpt-staging.com")
        _add_cookie(jar, "session_id", "bad", "api.chatgpt-staging.com")

        removed = OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        names = {c.name for c in jar}
        assert names == {"__cf_bm"}
        assert removed == 1

    def test_does_not_match_evil_suffix(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """evilchatgpt.com is NOT a chatgpt host — cookies should be left alone."""
        jar = adapter._curl_session.cookies.jar
        _add_cookie(jar, "session_id", "keep", "evilchatgpt.com")

        removed = OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        names = {c.name for c in jar}
        assert "session_id" in names
        assert removed == 0

    def test_does_not_match_chatgpt_subdomain_of_openai(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """foo.chat.openai.com is NOT a chatgpt host — cookies should be left alone."""
        jar = adapter._curl_session.cookies.jar
        _add_cookie(jar, "session_id", "keep", "foo.chat.openai.com")

        removed = OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        names = {c.name for c in jar}
        assert "session_id" in names
        assert removed == 0

    def test_does_not_match_chatgpt_com_evil(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """chatgpt.com.evil.example is NOT a chatgpt host."""
        jar = adapter._curl_session.cookies.jar
        _add_cookie(jar, "session_id", "keep", "chatgpt.com.evil.example")

        removed = OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        names = {c.name for c in jar}
        assert "session_id" in names
        assert removed == 0

    def test_matches_uppercase_host(
        self,
        adapter: OpenAISubscriptionAdapter,
    ) -> None:
        """Host matching should be case-insensitive."""
        jar = adapter._curl_session.cookies.jar
        _add_cookie(jar, "__cf_bm", "good", "CHATGPT.COM")
        _add_cookie(jar, "session_id", "bad", "CHATGPT.COM")

        removed = OpenAISubscriptionAdapter._filter_cloudflare_cookies(
            adapter._curl_session.cookies,
        )
        names = {c.name for c in jar}
        assert names == {"__cf_bm"}
        assert removed == 1


class TestMakeRequestCookieFilter:
    """Integration: make_request filters non-CF cookies before POST."""

    @pytest.mark.asyncio()
    async def test_make_request_filters_cookies(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        sse_body = (
            b'data: {"type":"response.completed","response":'
            b'{"model":"gpt-5.4","status":"completed"}}\n\n'
            b"data: [DONE]\n\n"
        )
        mock_resp = _make_mock_codex_response(status_code=200, content=sse_body)

        # Use a real AsyncSession so .cookies.jar is a real CookieJar
        real_session = curl_cffi.requests.AsyncSession()
        _add_cookie(real_session.cookies.jar, "__cf_bm", "good", "chatgpt.com")
        _add_cookie(real_session.cookies.jar, "session_id", "bad", "chatgpt.com")
        real_session.post = unittest.mock.AsyncMock(return_value=mock_resp)

        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=real_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
                await adapter.make_request(cc_request)

        # After make_request, non-CF cookies should be gone
        names = {c.name for c in real_session.cookies.jar}
        assert "__cf_bm" in names
        assert "session_id" not in names


class TestStreamRequestCookieFilter:
    """Integration: stream_request filters non-CF cookies before POST."""

    @pytest.mark.asyncio()
    async def test_stream_request_filters_cookies(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        mock_resp = _make_streaming_codex_response(
            status_code=200,
            chunks=[b'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n'],
        )

        # Use a real AsyncSession so .cookies.jar is a real CookieJar
        real_session = curl_cffi.requests.AsyncSession()
        _add_cookie(real_session.cookies.jar, "__cf_bm", "good", "chatgpt.com")
        _add_cookie(real_session.cookies.jar, "session_id", "bad", "chatgpt.com")
        real_session.post = unittest.mock.AsyncMock(return_value=mock_resp)

        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=real_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
                await adapter.stream_request(cc_request, mock_write)

        # After stream_request, non-CF cookies should be gone
        names = {c.name for c in real_session.cookies.jar}
        assert "__cf_bm" in names
        assert "session_id" not in names
        assert any(b'"delta":"Hi"' in chunk for chunk in written)


# ── Cloudflare 403 retry ──────────────────────────────────────────────────


class TestStreamRequestCfRetry:
    """Test that stream_request retries once on CF 403 before giving up."""

    @pytest.mark.asyncio()
    async def test_retries_on_cf_block_then_succeeds(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """First attempt returns CF 403, second attempt succeeds."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        cf_resp = _make_mock_codex_response(status_code=403, text=_CLOUDFLARE_HTML)
        ok_resp = _make_streaming_codex_response(
            status_code=200,
            chunks=[b'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n'],
        )

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(side_effect=[cf_resp, ok_resp])
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
                await adapter.stream_request(cc_request, mock_write)

        assert mock_session.post.await_count == 2
        assert any(b'"delta":"Hi"' in chunk for chunk in written)

    @pytest.mark.asyncio()
    async def test_raises_after_cf_retry_exhausted(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """Both attempts return CF 403 — should raise ProviderError."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        cf_resp = _make_mock_codex_response(status_code=403, text=_CLOUDFLARE_HTML)

        mock_session = unittest.mock.AsyncMock()
        # Codex retry policy: max_attempts=4 → 5 total tries
        mock_session.post = unittest.mock.AsyncMock(side_effect=[cf_resp] * 5)
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
                from kitty.providers.base import ProviderError

                with pytest.raises(ProviderError, match="[Cc]loudflare"):
                    await adapter.stream_request(cc_request, mock_write)

        from kitty.providers.openai_subscription import _CODEX_RETRY_MAX_ATTEMPTS

        assert mock_session.post.await_count == _CODEX_RETRY_MAX_ATTEMPTS + 1
        assert written == []

    @pytest.mark.asyncio()
    async def test_non_cf_error_not_retried(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """Non-CF 403 errors should still raise immediately without retry."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        non_cf_resp = _make_mock_codex_response(status_code=403, text=_NON_CF_HTML)

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(return_value=non_cf_resp)
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
                from kitty.providers.base import ProviderError

                with pytest.raises(ProviderError, match="access denied"):
                    await adapter.stream_request(cc_request, mock_write)

        # Non-CF error: only one attempt, no retry
        assert mock_session.post.await_count == 1


class TestMakeRequestCfRetry:
    """Test that make_request retries once on CF 403 before giving up."""

    @pytest.mark.asyncio()
    async def test_reloads_session_and_resets_curl_after_first_cf_block(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
        tmp_path: Path,
    ) -> None:
        """First CF 403 should trigger a session reload and curl-session reset.

        The first retry should use a fresh session from disk instead of the cached
        one, and it should recreate the curl session to refresh TLS/cookies.
        """
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        cf_resp = _make_mock_codex_response(status_code=403, text=_CLOUDFLARE_HTML)
        sse_body = (
            b'data: {"type":"response.completed","response":'
            b'{"model":"gpt-5.4","status":"completed"}}\n\n'
            b"data: [DONE]\n\n"
        )
        ok_resp = _make_mock_codex_response(status_code=200, content=sse_body)

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(side_effect=[cf_resp, ok_resp])

        fresh_disk_session = OAuthSession(
            client_id="app_test",
            access_token="at_fresh",
            refresh_token="rt_fresh",
            id_token=_make_id_token("acct-1234"),
            api_key="api_key_fresh",
            access_token_expires_at=time.time() + 3600,
            api_key_expires_at=time.time() + 3600,
            _file_path=str(session_path),
        )
        fresh_disk_session.save()

        reload_calls: list[str] = []
        reset_calls = 0
        original_load = adapter._load_session

        def _tracked_load(cc_req: dict) -> OAuthSession:
            reload_calls.append(cc_req["_resolved_key"])
            return original_load(cc_req)

        def _tracked_reset() -> None:
            nonlocal reset_calls
            reset_calls += 1
            adapter._curl_session_instance = None

        with (
            unittest.mock.patch.object(adapter, "_load_session", side_effect=_tracked_load),
            unittest.mock.patch.object(adapter, "_reset_curl_session", side_effect=_tracked_reset),
            unittest.mock.patch.object(
                OpenAISubscriptionAdapter,
                "_curl_session",
                new_callable=unittest.mock.PropertyMock,
                return_value=mock_session,
            ),
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
                result = await adapter.make_request(cc_request)

        assert mock_session.post.await_count == 2
        assert len(reload_calls) >= 2
        assert reset_calls == 1
        assert result["model"] == "gpt-5.4"

    @pytest.mark.asyncio()
    async def test_retries_on_cf_block_then_succeeds(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """First attempt returns CF 403, second attempt succeeds."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        cf_resp = _make_mock_codex_response(status_code=403, text=_CLOUDFLARE_HTML)
        sse_body = (
            b'data: {"type":"response.completed","response":'
            b'{"model":"gpt-5.4","status":"completed"}}\n\n'
            b"data: [DONE]\n\n"
        )
        ok_resp = _make_mock_codex_response(status_code=200, content=sse_body)

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(side_effect=[cf_resp, ok_resp])
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
                result = await adapter.make_request(cc_request)

        assert mock_session.post.await_count == 2
        assert result["model"] == "gpt-5.4"

    @pytest.mark.asyncio()
    async def test_raises_after_cf_retry_exhausted(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """Both attempts return CF 403 — should raise ProviderError."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        cf_resp = _make_mock_codex_response(status_code=403, text=_CLOUDFLARE_HTML)

        mock_session = unittest.mock.AsyncMock()
        # Codex retry policy: max_attempts=4 → 5 total tries
        mock_session.post = unittest.mock.AsyncMock(side_effect=[cf_resp] * 5)
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
                from kitty.providers.base import ProviderError

                with pytest.raises(ProviderError, match="[Cc]loudflare"):
                    await adapter.make_request(cc_request)

        from kitty.providers.openai_subscription import _CODEX_RETRY_MAX_ATTEMPTS

        assert mock_session.post.await_count == _CODEX_RETRY_MAX_ATTEMPTS + 1

    @pytest.mark.asyncio()
    async def test_non_cf_error_not_retried(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """Non-CF 403 errors should still raise immediately without retry."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        non_cf_resp = _make_mock_codex_response(status_code=403, text=_NON_CF_HTML)

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(return_value=non_cf_resp)
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
                from kitty.providers.base import ProviderError

                with pytest.raises(ProviderError, match="access denied"):
                    await adapter.make_request(cc_request)

        # Non-CF error: only one attempt, no retry
        assert mock_session.post.await_count == 1


class TestCodex5xxRetry:
    """Test retry behavior for 5xx responses matches Codex CLI."""

    @pytest.mark.asyncio()
    async def test_make_request_retries_5xx_then_succeeds(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """A 5xx response should be retried and eventually succeed."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        five_xx_resp = _make_mock_codex_response(status_code=500, text="server error")
        sse_body = (
            b'data: {"type":"response.completed","response":'
            b'{"model":"gpt-5.4","status":"completed"}}\n\n'
            b"data: [DONE]\n\n"
        )
        ok_resp = _make_mock_codex_response(status_code=200, content=sse_body)

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(side_effect=[five_xx_resp, ok_resp])
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with (
                unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session),
                unittest.mock.patch("asyncio.sleep", new_callable=unittest.mock.AsyncMock) as mock_sleep,
            ):
                result = await adapter.make_request(cc_request)

        assert mock_session.post.await_count == 2
        mock_sleep.assert_awaited()
        assert result["model"] == "gpt-5.4"

    @pytest.mark.asyncio()
    async def test_stream_request_retries_5xx_then_succeeds(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """A 5xx response should be retried in stream_request and then succeed."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        five_xx_resp = _make_mock_codex_response(status_code=502, text="bad gateway")
        ok_resp = _make_streaming_codex_response(
            status_code=200,
            chunks=[
                b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n',
                b"data: [DONE]\n\n",
            ],
        )

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(side_effect=[five_xx_resp, ok_resp])
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with (
                unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session),
                unittest.mock.patch("asyncio.sleep", new_callable=unittest.mock.AsyncMock) as mock_sleep,
            ):
                await adapter.stream_request(cc_request, mock_write)

        assert mock_session.post.await_count == 2
        mock_sleep.assert_awaited()
        assert written

    @pytest.mark.asyncio()
    async def test_make_request_does_not_retry_429(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """429 must not be retried."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        resp = _make_mock_codex_response(status_code=429, text="rate limited")
        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(return_value=resp)
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
                from kitty.providers.base import ProviderError

                with pytest.raises(ProviderError, match="rate limited"):
                    await adapter.make_request(cc_request)

        assert mock_session.post.await_count == 1

    @pytest.mark.asyncio()
    async def test_make_request_retries_transport_error_then_succeeds(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """Transport errors (timeout, connection failure) should be retried."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        sse_body = (
            b'data: {"type":"response.completed","response":'
            b'{"model":"gpt-5.4","status":"completed"}}\n\n'
            b"data: [DONE]\n\n"
        )
        ok_resp = _make_mock_codex_response(status_code=200, content=sse_body)

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(
            side_effect=[Exception("Connection timed out after 30s"), ok_resp],
        )
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with (
                unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session),
                unittest.mock.patch("asyncio.sleep", new_callable=unittest.mock.AsyncMock) as mock_sleep,
            ):
                result = await adapter.make_request(cc_request)

        assert mock_session.post.await_count == 2
        mock_sleep.assert_awaited()
        assert result["model"] == "gpt-5.4"

    @pytest.mark.asyncio()
    async def test_make_request_transport_error_exhausted(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """Transport errors are retried up to max attempts, then raise ProviderError."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(
            side_effect=Exception("Connection refused: chatgpt.com"),
        )
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with (
                unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session),
                unittest.mock.patch("asyncio.sleep", new_callable=unittest.mock.AsyncMock),
            ):
                from kitty.providers.base import ProviderError

                with pytest.raises(ProviderError, match="connection failed"):
                    await adapter.make_request(cc_request)

        # 5 total attempts (0..=4)
        assert mock_session.post.await_count == 5

    @pytest.mark.asyncio()
    async def test_stream_request_retries_transport_error_then_succeeds(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """Transport errors in stream_request should be retried."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        ok_resp = _make_streaming_codex_response(
            status_code=200,
            chunks=[
                b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n',
                b"data: [DONE]\n\n",
            ],
        )

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(
            side_effect=[Exception("Connection timed out after 30s"), ok_resp],
        )
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp_session = unittest.mock.MagicMock()
            mock_aiohttp_session.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp_session)
            mock_aiohttp_session.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            with (
                unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp_session),
                unittest.mock.patch("asyncio.sleep", new_callable=unittest.mock.AsyncMock) as mock_sleep,
            ):
                await adapter.stream_request(cc_request, mock_write)

        assert mock_session.post.await_count == 2
        mock_sleep.assert_awaited()
        assert written


class TestGetCloudflareSignature:
    """Test get_cloudflare_signature() returns the matched signature string."""

    def test_returns_cf_mitigated(self) -> None:
        """The cf-mitigated signature is detected first."""
        from kitty.cloudflare import get_cloudflare_signature

        assert get_cloudflare_signature("<html>cf-mitigated challenge</html>") == "cf-mitigated"

    def test_returns_cf_chl_opt(self) -> None:
        """The _cf_chl_opt signature is detected."""
        from kitty.cloudflare import get_cloudflare_signature

        assert get_cloudflare_signature("<html>_cf_chl_opt</html>") == "_cf_chl_opt"

    def test_returns_cloudflare_generic(self) -> None:
        """The generic cloudflare signature is detected."""
        from kitty.cloudflare import get_cloudflare_signature

        assert get_cloudflare_signature("<html>cloudflare ray id</html>") == "cloudflare"

    def test_returns_none_for_non_cf(self) -> None:
        """Non-CF bodies return None."""
        from kitty.cloudflare import get_cloudflare_signature

        assert get_cloudflare_signature("normal error message") is None

    def test_first_match_wins(self) -> None:
        """The first matching signature wins when multiple are present."""
        from kitty.cloudflare import get_cloudflare_signature

        assert get_cloudflare_signature("cf-mitigated and also cloudflare") == "cf-mitigated"


class TestLogCfCookies:
    """Test _log_cf_cookies() emits correct DEBUG log output."""

    @pytest.mark.asyncio()
    async def test_logs_cf_cookies_present(self, adapter: OpenAISubscriptionAdapter) -> None:
        """CF cookies in the jar are logged with truncated values."""
        session = curl_cffi.requests.AsyncSession()
        jar = session.cookies.jar
        jar.set_cookie(
            Cookie(
                version=0,
                name="__cf_bm",
                value="a" * 40,
                port=None,
                port_specified=False,
                domain="chatgpt.com",
                domain_specified=True,
                domain_initial_dot=False,
                path="/",
                path_specified=True,
                secure=True,
                expires=None,
                discard=True,
                comment=None,
                comment_url=None,
                rest={},
                rfc2109=False,
            )
        )

        with unittest.mock.patch("kitty.providers.openai_subscription.logger") as mock_logger:
            adapter._log_cf_cookies(session.cookies)
            mock_logger.debug.assert_called_once()
            assert "CF cookies present" in mock_logger.debug.call_args[0][0]
            assert "__cf_bm" in str(mock_logger.debug.call_args)

        await session.close()

    def test_logs_no_cf_cookies(self, adapter: OpenAISubscriptionAdapter) -> None:
        """Empty jar logs 'No CF cookies present'."""
        session = curl_cffi.requests.AsyncSession()

        with unittest.mock.patch("kitty.providers.openai_subscription.logger") as mock_logger:
            adapter._log_cf_cookies(session.cookies)
            mock_logger.debug.assert_called_once()
            assert "No CF cookies present" in mock_logger.debug.call_args[0][0]

    def test_logs_cookie_count(self, adapter: OpenAISubscriptionAdapter) -> None:
        """Multiple CF cookies include count."""
        session = curl_cffi.requests.AsyncSession()
        jar = session.cookies.jar
        for name in ("__cf_bm", "cf_clearance"):
            jar.set_cookie(
                Cookie(
                    version=0,
                    name=name,
                    value="short",
                    port=None,
                    port_specified=False,
                    domain="chatgpt.com",
                    domain_specified=True,
                    domain_initial_dot=False,
                    path="/",
                    path_specified=True,
                    secure=True,
                    expires=None,
                    discard=True,
                    comment=None,
                    comment_url=None,
                    rest={},
                    rfc2109=False,
                )
            )

        with unittest.mock.patch("kitty.providers.openai_subscription.logger") as mock_logger:
            adapter._log_cf_cookies(session.cookies)
            assert "2" in str(mock_logger.debug.call_args)


class TestSessionInitFilter:
    """Test that _filter_cloudflare_cookies runs at session initialization."""

    def test_filter_called_on_session_creation(self, adapter: OpenAISubscriptionAdapter) -> None:
        """When _curl_session creates a new AsyncSession, _filter_cloudflare_cookies is called."""
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_filter_cloudflare_cookies",
            return_value=0,
        ) as mock_filter:
            _ = adapter._curl_session
            mock_filter.assert_called_once()

    def test_filter_not_called_on_existing_session(self, adapter: OpenAISubscriptionAdapter) -> None:
        """When session already exists, _filter_cloudflare_cookies is not called again."""
        _ = adapter._curl_session  # create it first
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_filter_cloudflare_cookies",
            return_value=0,
        ) as mock_filter:
            _ = adapter._curl_session
            mock_filter.assert_not_called()


class TestAuthRefreshFilter:
    """Test that _filter_cloudflare_cookies runs after auth refresh in request methods."""

    @pytest.mark.asyncio()
    async def test_make_request_filters_after_auth(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """_filter_cloudflare_cookies is called in make_request after auth refresh."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        sse_body = (
            b'data: {"type":"response.completed","response"'
            b':{"model":"gpt-5.4","status":"completed"}}\n\n'
            b"data: [DONE]\n\n"
        )
        ok_resp = _make_mock_codex_response(status_code=200, content=sse_body)

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(return_value=ok_resp)
        with (
            unittest.mock.patch.object(
                OpenAISubscriptionAdapter,
                "_curl_session",
                new_callable=unittest.mock.PropertyMock,
                return_value=mock_session,
            ),
            unittest.mock.patch.object(
                OpenAISubscriptionAdapter,
                "_filter_cloudflare_cookies",
                return_value=0,
            ) as mock_filter,
        ):
            mock_aiohttp = _make_mock_oauth_http()
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp):
                await adapter.make_request(cc_request)

        # Filter should have been called (at least once — before the POST)
        assert mock_filter.call_count >= 1

    @pytest.mark.asyncio()
    async def test_stream_request_filters_after_auth(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """_filter_cloudflare_cookies is called in stream_request after auth refresh."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }

        sse_chunks = [
            b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n',
            b"data: [DONE]\n\n",
        ]
        ok_resp = _make_streaming_codex_response(status_code=200, chunks=sse_chunks)
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(return_value=ok_resp)
        with (
            unittest.mock.patch.object(
                OpenAISubscriptionAdapter,
                "_curl_session",
                new_callable=unittest.mock.PropertyMock,
                return_value=mock_session,
            ),
            unittest.mock.patch.object(
                OpenAISubscriptionAdapter,
                "_filter_cloudflare_cookies",
                return_value=0,
            ) as mock_filter,
        ):
            mock_aiohttp = _make_mock_oauth_http()
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp):
                await adapter.stream_request(cc_request, mock_write)

        # Filter should have been called (at least once — before the POST)
        assert mock_filter.call_count >= 1


# ── 401 auth recovery ────────────────────────────────────────────────────


class TestMakeRequest401Recovery:
    """Test 401 auth recovery in make_request matches Codex CLI's UnauthorizedRecovery.

    Codex CLI recovery steps:
    1. Reload auth from disk and retry
    2. Force-refresh token via OAuth and retry
    3. If still 401, raise error
    """

    @pytest.mark.asyncio()
    async def test_reloads_and_succeeds_on_second_attempt(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """First attempt returns 401, reload from disk, second attempt succeeds."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        unauthorized_resp = _make_mock_codex_response(
            status_code=401,
            text='{"error": {"message": "Unauthorized"}}',
        )
        sse_body = (
            b'data: {"type":"response.completed","response"'
            b':{"model":"gpt-5.4","status":"completed"}}\n\n'
            b"data: [DONE]\n\n"
        )
        ok_resp = _make_mock_codex_response(status_code=200, content=sse_body)

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(
            side_effect=[unauthorized_resp, ok_resp],
        )
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp = _make_mock_oauth_http()
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp):
                result = await adapter.make_request(cc_request)

        assert mock_session.post.await_count == 2
        assert result["model"] == "gpt-5.4"

    @pytest.mark.asyncio()
    async def test_force_refreshes_and_succeeds_on_third_attempt(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """First 401 triggers reload, second 401 triggers force-refresh, third succeeds."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        unauthorized_resp = _make_mock_codex_response(
            status_code=401,
            text='{"error": {"message": "Unauthorized"}}',
        )
        sse_body = (
            b'data: {"type":"response.completed","response"'
            b':{"model":"gpt-5.4","status":"completed"}}\n\n'
            b"data: [DONE]\n\n"
        )
        ok_resp = _make_mock_codex_response(status_code=200, content=sse_body)

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(
            side_effect=[unauthorized_resp, unauthorized_resp, ok_resp],
        )
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp = _make_mock_oauth_http()
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp):
                result = await adapter.make_request(cc_request)

        assert mock_session.post.await_count == 3
        assert result["model"] == "gpt-5.4"

    @pytest.mark.asyncio()
    async def test_raises_after_all_recovery_exhausted(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """All three attempts (initial + reload + force-refresh) return 401."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        unauthorized_resp = _make_mock_codex_response(
            status_code=401,
            text='{"error": {"message": "Token expired"}}',
        )

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(
            side_effect=[unauthorized_resp, unauthorized_resp, unauthorized_resp],
        )
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp = _make_mock_oauth_http()
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp):
                from kitty.providers.base import ProviderError

                with pytest.raises(ProviderError, match="auth failed|re-authenticate"):
                    await adapter.make_request(cc_request)

        # 3 attempts: initial + reload + force-refresh
        assert mock_session.post.await_count == 3

    @pytest.mark.asyncio()
    async def test_logs_recovery_steps(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """Each recovery step logs a warning."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
        }

        unauthorized_resp = _make_mock_codex_response(
            status_code=401,
            text='{"error": {"message": "Unauthorized"}}',
        )
        sse_body = (
            b'data: {"type":"response.completed","response"'
            b':{"model":"gpt-5.4","status":"completed"}}\n\n'
            b"data: [DONE]\n\n"
        )
        ok_resp = _make_mock_codex_response(status_code=200, content=sse_body)

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(
            side_effect=[unauthorized_resp, unauthorized_resp, ok_resp],
        )
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp = _make_mock_oauth_http()
            with (
                unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp),
                unittest.mock.patch("kitty.providers.openai_subscription.logger") as mock_logger,
            ):
                await adapter.make_request(cc_request)

        # Should log: reload warning, force-refresh warning
        warning_calls = [c for c in mock_logger.warning.call_args_list if "401" in str(c)]
        assert len(warning_calls) >= 2
        assert any("reloading" in str(c) for c in warning_calls)
        assert any("force" in str(c).lower() or "refresh" in str(c).lower() for c in warning_calls)


class TestStreamRequest401Recovery:
    """Test 401 auth recovery in stream_request matches Codex CLI's UnauthorizedRecovery."""

    @pytest.mark.asyncio()
    async def test_reloads_and_succeeds_on_second_attempt(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """First attempt returns 401, reload, second attempt streams successfully."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        unauthorized_resp = _make_mock_codex_response(
            status_code=401,
            text='{"error": {"message": "Unauthorized"}}',
        )
        ok_resp = _make_streaming_codex_response(
            status_code=200,
            chunks=[
                b'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n',
                b"data: [DONE]\n\n",
            ],
        )

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(
            side_effect=[unauthorized_resp, ok_resp],
        )
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp = _make_mock_oauth_http()
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp):
                await adapter.stream_request(cc_request, mock_write)

        assert mock_session.post.await_count == 2
        assert any(b'"delta":"Hi"' in chunk for chunk in written)

    @pytest.mark.asyncio()
    async def test_raises_after_all_recovery_exhausted(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """All three attempts return 401 — should raise ProviderError."""
        _, session_path = fresh_session
        cc_request = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
        }
        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        unauthorized_resp = _make_mock_codex_response(
            status_code=401,
            text='{"error": {"message": "Token expired"}}',
        )

        mock_session = unittest.mock.AsyncMock()
        mock_session.post = unittest.mock.AsyncMock(
            side_effect=[unauthorized_resp, unauthorized_resp, unauthorized_resp],
        )
        with unittest.mock.patch.object(
            OpenAISubscriptionAdapter,
            "_curl_session",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_session,
        ):
            mock_aiohttp = _make_mock_oauth_http()
            with unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp):
                from kitty.providers.base import ProviderError

                with pytest.raises(ProviderError, match="auth failed|re-authenticate"):
                    await adapter.stream_request(cc_request, mock_write)

        assert mock_session.post.await_count == 3
        assert written == []


# ── OAuth session cache tests ─────────────────────────────────────────────


class TestOAuthSessionCache:
    """Tests for the adapter-level session cache that prevents concurrent
    refresh_token_reused errors by sharing one OAuthSession instance (and
    its _refresh_lock) across requests for the same session file.
    """

    def test_load_session_returns_cached_instance_for_unchanged_file(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """Two consecutive loads return the same cached OAuthSession object."""
        _, session_path = fresh_session
        cc_req: dict = {"_resolved_key": str(session_path)}

        first = adapter._load_session(cc_req)
        second = adapter._load_session(cc_req)

        assert first is second, "Expected same OAuthSession instance from cache"
        assert first._refresh_lock is second._refresh_lock

    def test_load_session_reloads_when_file_changes_on_disk(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """Cache detects external session file changes and reloads."""
        original_session, session_path = fresh_session
        cc_req: dict = {"_resolved_key": str(session_path)}

        first = adapter._load_session(cc_req)
        assert first.access_token == "at_fresh"

        # Simulate external re-login: write a new session to the same file
        new_session = OAuthSession(
            client_id="app_test",
            access_token="at_externally_updated",
            refresh_token="rt_new",
            id_token=_make_id_token("acct-1234"),
            api_key=None,
            access_token_expires_at=time.time() + 3600,
            api_key_expires_at=time.time() + 3600,
            _file_path=str(session_path),
        )
        new_session.save()

        # Force mtime change (some filesystems have coarse granularity)
        os.utime(str(session_path), ns=(0, 0))

        reloaded = adapter._load_session(cc_req)

        assert reloaded is not first, "Expected a new instance after file change"
        assert reloaded.access_token == "at_externally_updated"

    @pytest.mark.asyncio
    async def test_concurrent_requests_share_one_refresh(
        self,
        adapter: OpenAISubscriptionAdapter,
        tmp_path: Path,
    ) -> None:
        """Two concurrent get_valid_api_key calls on cached sessions trigger
        only one OAuth refresh (the shared _refresh_lock serializes them).
        """
        now = time.time()
        id_token = _make_id_token("acct-1234")
        session = OAuthSession(
            client_id="app_test",
            access_token="at_expired",
            refresh_token="rt_shared",
            id_token=id_token,
            api_key=None,
            access_token_expires_at=now - 100,  # expired
            api_key_expires_at=now - 100,
            _file_path=str(tmp_path / "oauth_session.json"),
        )
        session.save()

        cc_req: dict = {"_resolved_key": str(tmp_path / "oauth_session.json")}

        s1 = adapter._load_session(cc_req)
        s2 = adapter._load_session(cc_req)
        assert s1 is s2

        refresh_count = 0

        async def _counting_refresh(self_refresh, http):
            nonlocal refresh_count
            refresh_count += 1
            s1.access_token = "at_refreshed"
            s1.access_token_expires_at = time.time() + 3600
            s1.api_key = "ak_refreshed"
            s1.api_key_expires_at = time.time() + 3600
            s1.refresh_token = "rt_rotated"

        with unittest.mock.patch.object(type(s1), "_refresh", _counting_refresh):
            import asyncio

            results = await asyncio.gather(
                s1.get_valid_api_key(unittest.mock.AsyncMock()),
                s2.get_valid_api_key(unittest.mock.AsyncMock()),
            )

        assert refresh_count == 1, f"Expected 1 refresh, got {refresh_count}"
        assert results[0] == "ak_refreshed"
        assert results[1] == "ak_refreshed"


class TestOAuthRefreshErrorClassification:
    """Tests for OAuthRefreshFailed being raised as ProviderError with
    http_status=401 so the bridge classifies auth failures correctly.
    """

    @pytest.mark.asyncio
    async def test_make_request_refresh_failure_sets_http_status_401(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """OAuthRefreshFailed in make_request raises ProviderError with http_status=401."""
        from kitty.providers.base import ProviderError

        _, session_path = fresh_session
        now = time.time()
        expired = OAuthSession(
            client_id="app_test",
            access_token="at_expired",
            refresh_token="rt_expired",
            id_token=_make_id_token("acct-1234"),
            api_key=None,
            access_token_expires_at=now - 100,
            api_key_expires_at=now - 100,
            _file_path=str(session_path),
        )
        expired.save()

        cc_req: dict = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": str(session_path),
            "_provider_config": {},
        }

        # Mock aiohttp that returns refresh_token_reused error from OAuth endpoint
        mock_aiohttp = unittest.mock.MagicMock()
        mock_aiohttp.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp)
        mock_aiohttp.__aexit__ = unittest.mock.AsyncMock(return_value=False)

        refresh_resp = unittest.mock.MagicMock()
        refresh_resp.status = 400
        refresh_resp.json = unittest.mock.AsyncMock(
            return_value={
                "error": "refresh_token_reused",
                "error_description": "Refresh token has already been used",
            }
        )
        refresh_cm = unittest.mock.MagicMock()
        refresh_cm.__aenter__ = unittest.mock.AsyncMock(return_value=refresh_resp)
        refresh_cm.__aexit__ = unittest.mock.AsyncMock(return_value=False)
        mock_aiohttp.post.return_value = refresh_cm

        with (
            unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp),
            pytest.raises(ProviderError) as exc_info,
        ):
            await adapter.make_request(cc_req)

        assert exc_info.value.http_status == 401, (
            f"Expected http_status=401 for auth failure, got {exc_info.value.http_status}"
        )
        assert "refresh_token_reused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_request_refresh_failure_sets_http_status_401(
        self,
        adapter: OpenAISubscriptionAdapter,
        fresh_session: tuple[OAuthSession, Path],
    ) -> None:
        """OAuthRefreshFailed in stream_request raises ProviderError with http_status=401."""
        from kitty.providers.base import ProviderError

        _, session_path = fresh_session
        now = time.time()
        expired = OAuthSession(
            client_id="app_test",
            access_token="at_expired",
            refresh_token="rt_expired",
            id_token=_make_id_token("acct-1234"),
            api_key=None,
            access_token_expires_at=now - 100,
            api_key_expires_at=now - 100,
            _file_path=str(session_path),
        )
        expired.save()

        cc_req: dict = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": str(session_path),
            "_provider_config": {},
        }

        mock_aiohttp = unittest.mock.MagicMock()
        mock_aiohttp.__aenter__ = unittest.mock.AsyncMock(return_value=mock_aiohttp)
        mock_aiohttp.__aexit__ = unittest.mock.AsyncMock(return_value=False)

        refresh_resp = unittest.mock.MagicMock()
        refresh_resp.status = 400
        refresh_resp.json = unittest.mock.AsyncMock(
            return_value={
                "error": "refresh_token_reused",
                "error_description": "Refresh token has already been used",
            }
        )
        refresh_cm = unittest.mock.MagicMock()
        refresh_cm.__aenter__ = unittest.mock.AsyncMock(return_value=refresh_resp)
        refresh_cm.__aexit__ = unittest.mock.AsyncMock(return_value=False)
        mock_aiohttp.post.return_value = refresh_cm

        written: list[bytes] = []

        async def mock_write(data: bytes) -> None:
            written.append(data)

        with (
            unittest.mock.patch("aiohttp.ClientSession", return_value=mock_aiohttp),
            pytest.raises(ProviderError) as exc_info,
        ):
            await adapter.stream_request(cc_req, mock_write)

        assert exc_info.value.http_status == 401
        assert "refresh_token_reused" in str(exc_info.value)
