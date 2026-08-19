"""Tests for fail-closed behaviour when an egress gateway is configured.

The feature's value is the guarantee, not the plumbing: if kitty cannot honour
the proxy it must stop, because carrying on means leaking the machine's own IP
to the upstream — silently, and exactly when the user believed otherwise.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from kitty.egress import EgressConfig, set_egress
from kitty.providers.base import ProviderAdapter
from kitty.providers.bedrock import BedrockAdapter

EGRESS = EgressConfig(proxy_url="http://proxy.example.com:12323", username="myuser", password="s3cr3tpw")


class _ConnError(aiohttp.ClientConnectorError):
    """Connection failure double.

    aiohttp's own class needs a real ConnectionKey to be stringifiable, and its
    shape varies between releases; this keeps the test independent of that.
    """

    def __init__(self, message: str) -> None:
        Exception.__init__(self, message)
        self._message = message

    def __str__(self) -> str:
        return self._message


class _PlainProvider(ProviderAdapter):
    @property
    def provider_type(self) -> str:
        return "plain"

    @property
    def default_base_url(self) -> str:
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(str(status_code))


class _Profile:
    """Minimal stand-in for a Profile in the guard's eyes."""

    def __init__(self, name: str, provider_config: dict | None = None) -> None:
        self.name = name
        self.provider_config = provider_config or {}


class TestSupportsEgress:
    """R9: adapters declare whether they can honour the proxy."""

    def test_default_adapter_supports_egress(self):
        assert _PlainProvider().supports_egress("some-key", {}) is True

    def test_bedrock_with_explicit_credentials_supports_egress(self):
        adapter = BedrockAdapter()
        key = "AKIAEXAMPLE:wJalrXUtnFEMI/K7MDENG"

        assert adapter.is_sso_mode(key) is False
        assert adapter.supports_egress(key, {}) is True

    def test_bedrock_in_sso_mode_does_not_support_egress(self):
        """botocore resolves SSO/STS/IMDS credentials outside the proxied client."""
        adapter = BedrockAdapter()
        sso_key = next(k for k in ("sso", "SSO", "") if adapter.is_sso_mode(k))

        assert adapter.supports_egress(sso_key, {}) is False


class TestLaunchGuard:
    """R10: a provider that cannot be proxied blocks the launch."""

    @staticmethod
    def _reason(provider, profile, key, backends=None):
        from kitty.egress_guard import egress_block_reason

        return egress_block_reason(provider, profile, key, backends)

    def test_no_egress_allows_anything(self):
        set_egress(None)
        adapter = BedrockAdapter()
        sso_key = next(k for k in ("sso", "SSO", "") if adapter.is_sso_mode(k))

        assert self._reason(adapter, _Profile("bedrock-1"), sso_key) is None

    def test_supported_provider_is_allowed(self):
        set_egress(EGRESS)

        assert self._reason(_PlainProvider(), _Profile("minimax-1"), "key") is None

    def test_unsupported_provider_is_blocked_and_named(self):
        set_egress(EGRESS)
        adapter = BedrockAdapter()
        sso_key = next(k for k in ("sso", "SSO", "") if adapter.is_sso_mode(k))

        reason = self._reason(adapter, _Profile("bedrock-sso"), sso_key)

        assert reason is not None
        assert "bedrock-sso" in reason
        assert EGRESS.masked() in reason
        assert EGRESS.password not in reason, "the proxy password must never reach a user-facing message"

    def test_every_balancing_member_is_checked_not_just_the_first(self):
        """One unproxyable member must not hide behind a compliant first member."""
        set_egress(EGRESS)
        adapter = BedrockAdapter()
        sso_key = next(k for k in ("sso", "SSO", "") if adapter.is_sso_mode(k))
        backends = [
            (_PlainProvider(), "key", _Profile("minimax-1")),
            (_PlainProvider(), "key", _Profile("minimax-2")),
            (adapter, sso_key, _Profile("bedrock-sso")),
        ]

        reason = self._reason(_PlainProvider(), _Profile("minimax-1"), "key", backends)

        assert reason is not None
        assert "bedrock-sso" in reason


class TestValidationFailsClosed:
    """R11: pre-flight validation stops failing open under egress."""

    @staticmethod
    async def _validate(egress, error):
        from kitty.validation import validate_api_key

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(side_effect=error)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            return await validate_api_key(_PlainProvider(), "key", {}, egress=egress)

    @pytest.mark.asyncio()
    async def test_connection_error_without_egress_still_proceeds(self):
        """Unchanged behaviour: a flaky network should not block a launch."""
        error = _ConnError("Cannot connect to proxy.example.com:12323")

        result = await self._validate(None, error)

        assert result.valid is True
        assert result.warning is not None

    @pytest.mark.asyncio()
    async def test_connection_error_with_egress_is_fatal(self):
        error = _ConnError("Cannot connect to proxy.example.com:12323")

        result = await self._validate(EGRESS, error)

        assert result.valid is False
        assert result.reason is not None
        assert EGRESS.masked() in result.reason
        assert EGRESS.password not in result.reason

    @pytest.mark.asyncio()
    async def test_generic_network_error_with_egress_is_fatal(self):
        result = await self._validate(EGRESS, aiohttp.ClientError("boom"))

        assert result.valid is False
        assert result.reason is not None
        assert EGRESS.masked() in result.reason
