"""Tests for pre-flight API key validation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kitty.egress import EgressConfig
from kitty.providers.base import ProviderAdapter
from kitty.validation import ValidationResult, validate_api_key


class MockProvider(ProviderAdapter):
    """Test provider with standard HTTP transport."""

    @property
    def provider_type(self) -> str:
        return "mock"

    @property
    def default_base_url(self) -> str:
        return "https://mock.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        return {"content": "test", "finish_reason": "stop", "usage": {}}

    def map_error(self, status_code: int, body: dict) -> Exception:
        return RuntimeError(f"Error {status_code}")


class MockCustomTransportProvider(ProviderAdapter):
    """Test provider that uses custom transport (e.g., boto3)."""

    @property
    def provider_type(self) -> str:
        return "custom"

    @property
    def default_base_url(self) -> str:
        return "https://custom.example.com/v1"

    @property
    def use_custom_transport(self) -> bool:
        return True

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        return {"content": "test", "finish_reason": "stop", "usage": {}}

    def map_error(self, status_code: int, body: dict) -> Exception:
        return RuntimeError(f"Error {status_code}")


def test_validation_result_defaults():
    r = ValidationResult(valid=True)
    assert r.valid is True
    assert r.reason is None
    assert r.warning is None


def test_validation_result_invalid():
    r = ValidationResult(valid=False, reason="bad key")
    assert r.valid is False
    assert r.reason == "bad key"


@pytest.mark.asyncio
async def test_validate_custom_transport_skipped():
    provider = MockCustomTransportProvider()
    result = await validate_api_key(provider, "any-key")
    assert result.valid is True
    assert result.warning is None


@pytest.mark.asyncio
@patch("kitty.validation.aiohttp.ClientSession")
async def test_validate_401_returns_invalid(mock_session_cls):
    provider = MockProvider()
    mock_response = AsyncMock()
    mock_response.status = 401
    mock_response.json = AsyncMock(return_value={"error": {"code": "401", "message": "token expired"}})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session_cls.return_value = mock_session

    result = await validate_api_key(provider, "expired-key")
    assert result.valid is False
    assert "token expired" in result.reason or "invalid" in result.reason.lower()


@pytest.mark.asyncio
@patch("kitty.validation.aiohttp.ClientSession")
async def test_validate_403_returns_invalid(mock_session_cls):
    provider = MockProvider()
    mock_response = AsyncMock()
    mock_response.status = 403
    mock_response.json = AsyncMock(return_value={"error": {"message": "forbidden"}})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session_cls.return_value = mock_session

    result = await validate_api_key(provider, "bad-key")
    assert result.valid is False
    assert "forbidden" in result.reason or "invalid" in result.reason.lower()


@pytest.mark.asyncio
@patch("kitty.validation.aiohttp.ClientSession")
async def test_validate_200_returns_valid(mock_session_cls):
    provider = MockProvider()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session_cls.return_value = mock_session

    result = await validate_api_key(provider, "valid-key")
    assert result.valid is True
    assert result.reason is None


@pytest.mark.asyncio
@patch("kitty.validation.aiohttp.ClientSession")
async def test_validate_timeout_returns_valid_with_warning(mock_session_cls):

    provider = MockProvider()
    mock_session = AsyncMock()
    mock_session.post = MagicMock(side_effect=asyncio.TimeoutError())
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session_cls.return_value = mock_session

    result = await validate_api_key(provider, "any-key")
    assert result.valid is True
    assert result.warning is not None
    assert "timed out" in result.warning


@pytest.mark.asyncio
@patch("kitty.validation.aiohttp.ClientSession")
async def test_validate_connection_error_returns_valid_with_warning(mock_session_cls):
    import aiohttp

    provider = MockProvider()
    mock_session = AsyncMock()
    mock_session.post = MagicMock(
        side_effect=aiohttp.ClientConnectorError(
            connection_key=MagicMock(),
            os_error=OSError("Connection refused"),
        )
    )
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session_cls.return_value = mock_session

    result = await validate_api_key(provider, "any-key")
    assert result.valid is True
    assert result.warning is not None


@pytest.mark.asyncio
@patch("kitty.validation.aiohttp.ClientSession")
async def test_validate_dirty_key_returns_invalid(mock_session_cls):
    """A key containing newlines/CR triggers a clear user-facing error."""
    provider = MockProvider()
    mock_session = AsyncMock()
    mock_session.post = MagicMock(side_effect=ValueError("Newline, carriage return, or null byte detected in headers."))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session_cls.return_value = mock_session

    result = await validate_api_key(provider, "key-with-newline\n")
    assert result.valid is False
    assert "invalid characters" in result.reason
    assert "kitty setup" in result.reason


@pytest.mark.asyncio
@patch("kitty.validation.aiohttp.ClientSession")
async def test_preflight_probes_the_normalised_url(mock_session_cls):
    """KBR-134 — pre-flight inherits the base-URL fix, because it composes the same way.

    ``validate_api_key`` builds its probe URL through ``build_base_url``, so the
    reporter's stored profile is probed at Mistral's real endpoint rather than at
    the doubled path that produced the original 404.  Nothing in
    ``validation.py`` changed to achieve this; the test pins the consequence so a
    later refactor cannot quietly undo it.
    """
    from kitty.providers.custom_openai import CustomOpenAIAdapter

    provider = CustomOpenAIAdapter()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session_cls.return_value = mock_session

    result = await validate_api_key(
        provider,
        "any-key",
        {"base_url": "https://api.mistral.ai/v1/chat/completions"},
    )

    assert result.valid is True
    assert mock_session.post.call_args.args[0] == "https://api.mistral.ai/v1/chat/completions"


@pytest.mark.asyncio
async def test_preflight_does_not_raise_on_an_unparseable_base_url():
    """A malformed stored base URL yields a result, not a traceback.

    ``validate_api_key`` builds its probe URL *before* its own ``try`` block, and
    ``launcher.py`` does not wrap the call, so anything ``build_base_url`` raises
    reaches the user as a Python traceback at launch.  Normalisation parses the
    URL where nothing did before, which made that reachable; this pins the
    graceful path.
    """
    from kitty.providers.custom_openai import CustomOpenAIAdapter

    result = await validate_api_key(CustomOpenAIAdapter(), "any-key", {"base_url": "https://[::1/v1"})

    assert result.valid is False


class TestPreflightBlamesTheUrlNotTheKey:
    """KBR-143 — a base URL problem must not be reported as a credential problem.

    ``aiohttp.InvalidURL`` subclasses ``ValueError``, and so does the whitespace-in-
    headers error this function already catches, so both a malformed URL and a dirty
    key landed in the same branch — and that branch told the key story.  The user was
    sent to replace a credential that was never the problem.
    """

    @pytest.mark.parametrize("base_url", ["https://[::1/v1", "https:///v1"], ids=["unparseable", "no-host"])
    @pytest.mark.parametrize(
        "egress",
        [None, EgressConfig("http://proxy.example:8080")],
        ids=["no-egress", "with-egress"],
    )
    @pytest.mark.asyncio
    async def test_a_bad_url_is_reported_as_a_bad_url(self, base_url: str, egress: EgressConfig | None):
        """The reason names the provider and the base URL, and never the key.

        Parametrised over egress because the two paths failed *differently*: without
        a proxy the user got the wrong message, and with one ``should_bypass`` parsed
        the same URL above the ``try`` and raised ``ValueError`` outright — a
        traceback at launch rather than any message at all.

        Args:
            base_url: A base URL no HTTP client can use.
            egress: The egress configuration in force, or ``None``.
        """
        from kitty.providers.custom_openai import CustomOpenAIAdapter

        result = await validate_api_key(CustomOpenAIAdapter(), "any-key", {"base_url": base_url}, egress=egress)

        assert result.valid is False
        assert "base URL" in result.reason
        assert "custom_openai" in result.reason
        assert "key" not in result.reason.lower()

    @pytest.mark.parametrize(
        "base_url",
        [
            "https:///v1?subscription-key=s3cret",
            "ftp://gw/v1?subscription-key=s3cret",
            "https://u:s3cret@[::1/v1",
            "//u:s3cret@[::1/v1",
        ],
        ids=["no-host", "rejected-scheme", "unparseable-userinfo", "schemeless-userinfo"],
    )
    @pytest.mark.asyncio
    async def test_the_reported_url_carries_no_credential(self, base_url: str):
        """The reason is printed at launch, so it is redacted like the 404 message.

        Parametrised across the branches that build it, because they do not all reach
        the URL the same way.  The ``ftp://`` case is the one that caught a real leak:
        ``build_base_url`` raises with the **raw** URL in its own message, and the
        reason quoted that exception verbatim — so redacting the URL alongside it was
        not enough.

        Args:
            base_url: A base URL that cannot produce a request, carrying a secret.
        """
        from kitty.providers.custom_openai import CustomOpenAIAdapter

        result = await validate_api_key(CustomOpenAIAdapter(), "any-key", {"base_url": base_url})

        assert result.valid is False
        assert "s3cret" not in result.reason, result.reason

    @pytest.mark.parametrize(
        "base_url",
        ["https://:8080/v1", "https://gw.example:99999/v1"],
        ids=["port-without-host", "port-out-of-range"],
    )
    @pytest.mark.asyncio
    async def test_a_url_with_an_unusable_authority_is_reported_as_a_url(self, base_url: str):
        """An empty host and an impossible port are URL faults, not key faults.

        ``urlsplit("https://:8080/v1").netloc`` is the truthy ``":8080"`` while its
        ``hostname`` is ``None``, so a check on ``netloc`` passed this through to
        ``aiohttp``, which raised ``InvalidURL`` — a ``ValueError``, and therefore the
        key message again.  An out-of-range port is the same class: ``urlsplit``
        accepts it and only ``.port`` objects.

        Args:
            base_url: A base URL whose authority no HTTP client can use.
        """
        from kitty.providers.custom_openai import CustomOpenAIAdapter

        result = await validate_api_key(CustomOpenAIAdapter(), "any-key", {"base_url": base_url})

        assert result.valid is False
        assert "base URL" in result.reason
        assert "key" not in result.reason.lower(), result.reason

    @pytest.mark.asyncio
    async def test_a_missing_provider_config_key_is_not_blamed_on_the_base_url(self):
        """Vertex's missing ``project_id`` is a configuration fault, not a URL one.

        Naming the base URL here would send the user to edit a value they never set,
        which is the class of misdirection this ticket exists to end.  No ``base_url``
        is configured for Vertex, so the message speaks of the configuration instead.
        """
        from kitty.providers.vertex import VertexAIAdapter

        result = await validate_api_key(VertexAIAdapter(), "any-key", {})

        assert result.valid is False
        assert "project_id" in result.reason
        assert "base URL" not in result.reason, result.reason

    @pytest.mark.asyncio
    async def test_an_empty_base_url_is_quoted_as_empty(self):
        """An empty configured value must not be reported as the provider's default.

        ``provider_config.get("base_url") or default`` swallowed it, so the message
        quoted ``https://api.openai.com/v1`` — a URL the user never typed and which
        would have worked.
        """
        from kitty.providers.custom_openai import CustomOpenAIAdapter

        result = await validate_api_key(CustomOpenAIAdapter(), "any-key", {"base_url": ""})

        assert result.valid is False
        assert "api.openai.com" not in result.reason, result.reason

    @pytest.mark.asyncio
    async def test_the_reason_has_no_doubled_period(self):
        """The reason is read by a human, so it is punctuated like a sentence."""
        from kitty.providers.custom_openai import CustomOpenAIAdapter

        result = await validate_api_key(CustomOpenAIAdapter(), "any-key", {"base_url": "ftp://x"})

        assert ".." not in result.reason, result.reason

    @pytest.mark.asyncio
    async def test_a_raising_build_base_url_becomes_a_result(self):
        """``custom_openai`` rejects a non-HTTP scheme by raising; launch must survive.

        ``launcher.py`` does not wrap this call, so the exception would reach the user
        as a traceback.
        """
        from kitty.providers.custom_openai import CustomOpenAIAdapter

        result = await validate_api_key(CustomOpenAIAdapter(), "any-key", {"base_url": "ftp://x"})

        assert result.valid is False
        assert "base URL" in result.reason

    @pytest.mark.asyncio
    async def test_a_provider_error_becomes_a_result(self):
        """Vertex raises ``ProviderError`` for a missing ``project_id``, not ``ValueError``.

        A different exception type from a different adapter, so the guard cannot be
        written against ``ValueError`` alone.  Vertex uses the default transport, so it
        is not skipped the way the custom-transport providers are.
        """
        from kitty.providers.vertex import VertexAIAdapter

        result = await validate_api_key(VertexAIAdapter(), "any-key", {})

        assert result.valid is False
        assert "project_id" in result.reason

    @pytest.mark.asyncio
    @patch("kitty.validation.aiohttp.ClientSession")
    async def test_a_dirty_key_still_reports_the_key(self, mock_session_cls):
        """The branch is narrowed, not removed: a real header fault still says so.

        Args:
            mock_session_cls: The patched ``ClientSession``.
        """
        session = AsyncMock()
        session.post = MagicMock(side_effect=ValueError("Newline, carriage return, or null byte detected in headers."))
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = session

        result = await validate_api_key(MockProvider(), "key-with-newline\n")

        assert result.valid is False
        assert "invalid characters" in result.reason
