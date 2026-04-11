"""Tests for pre-flight API key validation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
