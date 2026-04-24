"""Tests for OpenAI subscription provider adapter."""

from __future__ import annotations

import contextlib
import json
import time
import unittest.mock
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from kitty.auth.oauth_session import OAuthSession
from kitty.providers.openai_subscription import (
    _CODEX_BACKEND_URL,
    OpenAISubscriptionAdapter,
)

# ── Async mock helpers ────────────────────────────────────────────────────

async def _async_yield(value: object) -> AsyncIterator:
    """Yield a single value as an async context manager body."""
    yield value


def _make_mock_codex_response(
    *, status_code: int, text: str = "", content: bytes | None = None,
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
    *, status_code: int, chunks: list[bytes],
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
    *, status_code: int, chunks_before_error: list[bytes], error_message: str,
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

    def test_map_error_400_includes_status(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(400, {"error": {"message": "Bad request"}})
        assert "400" in str(error)
        assert "Bad request" in str(error)

    def test_map_error_401_mentions_reauth(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(401, {"error": {"message": "Unauthorized", "code": "invalid_api_key"}})
        assert "re-authenticate" in str(error)

    def test_map_error_403_cloudflare_html(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(
            403, {"error": {"message": "<html><head>cf-mitigated: challenge</head></html>"}}
        )
        msg = str(error).lower()
        assert "cloudflare" in msg
        assert "not an api key problem" in msg

    def test_map_error_403_non_cloudflare(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(403, {"error": {"message": "Forbidden"}})
        msg = str(error)
        assert "access denied" in msg
        assert "cloudflare" not in msg.lower()


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
        body = 'some text <script>window._cf_chl_opt = {};</script> more text'
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
    def test_user_agent_is_codex_cli_format(self, adapter: OpenAISubscriptionAdapter) -> None:
        headers = adapter._build_codex_headers("tok", _make_id_token())
        ua = headers["User-Agent"]
        assert ua.startswith("codex_cli_rs/")

    def test_no_browser_headers(self, adapter: OpenAISubscriptionAdapter) -> None:
        """Codex CLI does not send browser-like headers."""
        headers = adapter._build_codex_headers("tok", _make_id_token())
        assert "Origin" not in headers
        assert "Referer" not in headers
        assert "Accept" not in headers
        assert "OpenAI-Beta" not in headers

    def test_authorization_bearer(self, adapter: OpenAISubscriptionAdapter) -> None:
        headers = adapter._build_codex_headers("my-token", _make_id_token())
        assert headers["Authorization"] == "Bearer my-token"

    def test_account_id_from_jwt(self, adapter: OpenAISubscriptionAdapter) -> None:
        headers = adapter._build_codex_headers("tok", _make_id_token("acct-42"))
        assert headers["chatgpt-account-id"] == "acct-42"

    def test_no_account_id_without_jwt_claim(self, adapter: OpenAISubscriptionAdapter) -> None:
        headers = adapter._build_codex_headers("tok", _make_id_token(None))
        assert "chatgpt-account-id" not in headers

    def test_no_originator_header(self, adapter: OpenAISubscriptionAdapter) -> None:
        """The originator header triggers strict tool validation on the backend."""
        headers = adapter._build_codex_headers("tok", _make_id_token())
        assert "originator" not in headers


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
        assert OpenAISubscriptionAdapter._is_transient_stream_error(
            Exception("Recv error: connection reset")
        )

    def test_connection_closed_is_transient(self) -> None:
        assert OpenAISubscriptionAdapter._is_transient_stream_error(
            Exception("Connection closed by server")
        )

    def test_timeout_is_not_transient(self) -> None:
        assert not OpenAISubscriptionAdapter._is_transient_stream_error(
            Exception("Connection timed out after 30s")
        )

    def test_generic_error_is_not_transient(self) -> None:
        assert not OpenAISubscriptionAdapter._is_transient_stream_error(
            Exception("Something unexpected")
        )


# ── Session reuse ────────────────────────────────────────────────────────

class TestCurlSessionReuse:
    def test_same_instance_returned(self, adapter: OpenAISubscriptionAdapter) -> None:
        """The lazy _curl_session property returns the same instance."""
        s1 = adapter._curl_session
        s2 = adapter._curl_session
        assert s1 is s2


# ── Cloudflare detection in request methods ──────────────────────────────

class TestStreamRequestCloudflare:
    """Test that stream_request detects Cloudflare blocks and raises specific error."""

    @pytest.mark.asyncio()
    async def test_stream_request_raises_cloudflare_error(
        self, adapter: OpenAISubscriptionAdapter, fresh_session: tuple[OAuthSession, Path],
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
        self, adapter: OpenAISubscriptionAdapter, fresh_session: tuple[OAuthSession, Path],
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
        self, adapter: OpenAISubscriptionAdapter, fresh_session: tuple[OAuthSession, Path],
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
                b'data: [DONE]\n\n',
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
        self, adapter: OpenAISubscriptionAdapter, fresh_session: tuple[OAuthSession, Path],
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
        self, adapter: OpenAISubscriptionAdapter, fresh_session: tuple[OAuthSession, Path],
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
            b'data: [DONE]\n\n',
        ]
        mock_resp = _make_streaming_codex_response(status_code=200, chunks=sse_chunks)
        with _mock_curl_session(mock_resp):
            await adapter.stream_request(cc_request, mock_write)

        assert len(written) == 3
        assert b'"delta":"Hello"' in written[0]
        assert b'"delta":" world"' in written[1]

    @pytest.mark.asyncio()
    async def test_strips_bom_from_chunks(
        self, adapter: OpenAISubscriptionAdapter, fresh_session: tuple[OAuthSession, Path],
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
        self, adapter: OpenAISubscriptionAdapter, fresh_session: tuple[OAuthSession, Path],
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
            b'data: [DONE]\n\n'
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
        self, adapter: OpenAISubscriptionAdapter, fresh_session: tuple[OAuthSession, Path],
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
        self, adapter: OpenAISubscriptionAdapter, fresh_session: tuple[OAuthSession, Path],
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
            b'data: [DONE]\n\n'
        )
        mock_resp = _make_mock_codex_response(status_code=200, content=sse_body)
        with _mock_curl_session(mock_resp):
            await adapter.make_request(cc_request)

        mock_resp.close.assert_not_called()


class TestEmptyResponseBody:
    """Guard against empty SSE body silently producing a valid response."""

    @pytest.mark.asyncio()
    async def test_make_request_raises_on_empty_body(
        self, adapter: OpenAISubscriptionAdapter, fresh_session: tuple[OAuthSession, Path],
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
