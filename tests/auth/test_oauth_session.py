"""Tests for OAuthSession: token state machine and file persistence."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.auth.oauth_session import (
    OAUTH_TOKEN_URL,
    OAuthRefreshFailed,
    OAuthSession,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def fresh_tokens() -> dict:
    """Tokens that are freshly issued (far future expiry)."""
    return {
        "access_token": "access_abc",
        "refresh_token": "refresh_xyz",
        "id_token": "id_token_qrs",
        "api_key": "sk-openai-test123",
        "expires_in": 3600,
    }


@pytest.fixture()
def expired_tokens() -> dict:
    """Tokens that expired 10 minutes ago."""
    return {
        "access_token": "access_expired",
        "refresh_token": "refresh_xyz",
        "id_token": "id_token_qrs",
        "api_key": "sk-openai-expired",
        "expires_in": 3600,
        "created_at": time.time() - 3700,  # issued 61+ minutes ago
    }


@pytest.fixture()
def session_factory():
    """Factory that creates OAuthSession with configurable expiry."""

    def _make(
        expires_in: float = 3600,
        created_at: float | None = None,
        api_key_expires_in: float | None = None,
    ) -> OAuthSession:
        now = created_at or time.time()
        return OAuthSession(
            client_id="app_test",
            access_token="at_test",
            refresh_token="rt_test",
            id_token="id_test",
            api_key="sk_test",
            access_token_expires_at=now + expires_in,
            api_key_expires_at=now + (api_key_expires_in if api_key_expires_in else expires_in),
            _file_path=None,
        )

    return _make


# ── Serialization ───────────────────────────────────────────────────────────


class TestSerialization:
    def test_to_dict_round_trip_preserves_all_fields(self, session_factory) -> None:
        session = session_factory()
        data = session.to_dict()
        restored = OAuthSession.from_dict(data)
        assert restored.client_id == session.client_id
        assert restored.access_token == session.access_token
        assert restored.refresh_token == session.refresh_token
        assert restored.id_token == session.id_token
        assert restored.api_key == session.api_key
        assert restored.access_token_expires_at == session.access_token_expires_at
        assert restored.api_key_expires_at == session.api_key_expires_at
        # from_dict does not restore _file_path (use load() for that)
        assert restored._file_path is None

    def test_from_dict_parses_all_fields(self, fresh_tokens) -> None:
        created = time.time()
        data = {
            "client_id": "app_foo",
            "access_token": "at_bar",
            "refresh_token": "rt_baz",
            "id_token": "id_qux",
            "api_key": "sk_quux",
            "access_token_expires_at": created + 7200,
            "api_key_expires_at": created + 7200,
        }
        session = OAuthSession.from_dict(data)
        assert session.client_id == "app_foo"
        assert session.access_token == "at_bar"
        assert session.access_token_expires_at == created + 7200

    def test_from_token_response_sets_expiry_from_expires_in(self, fresh_tokens) -> None:
        session = OAuthSession.from_token_response(fresh_tokens, "app_xyz")
        assert session.client_id == "app_xyz"
        assert session.access_token == "access_abc"
        assert session.refresh_token == "refresh_xyz"
        assert session.id_token == "id_token_qrs"
        assert session.api_key == "sk-openai-test123"
        # expires_in was 3600; expiry should be ~now + 3600
        assert abs(session.access_token_expires_at - (time.time() + 3600)) < 5
        assert abs(session.api_key_expires_at - (time.time() + 3600)) < 5

    def test_from_token_response_missing_expires_in_uses_default(self) -> None:
        data = {
            "access_token": "at_noid",
            "refresh_token": "rt_noid",
            "id_token": "id_noid",
            "api_key": "sk_noid",
        }
        session = OAuthSession.from_token_response(data, "app_noid")
        # Should default to 3600
        expected_expiry = time.time() + 3600
        assert abs(session.access_token_expires_at - expected_expiry) < 5


# ── Expiry helpers ───────────────────────────────────────────────────────────


class TestExpiryProperties:
    def test_access_token_not_expired_when_fresh(self, session_factory) -> None:
        # Expires in 1 hour
        session = session_factory(expires_in=3600)
        assert not session.access_token_expired

    def test_access_token_expired_when_in_past(self, session_factory) -> None:
        # Expired 10 minutes ago
        session = session_factory(expires_in=-600)
        assert session.access_token_expired

    def test_api_key_expired_when_in_past(self, session_factory) -> None:
        session = session_factory(api_key_expires_in=-600)
        assert session.api_key_expired

    def test_api_key_not_expired_when_fresh(self, session_factory) -> None:
        session = session_factory(api_key_expires_in=7200)
        assert not session.api_key_expired


# ── get_valid_api_key ───────────────────────────────────────────────────────


class TestGetValidApiKey:
    @pytest.mark.asyncio
    async def test_returns_key_when_fresh_awaitable(self, session_factory) -> None:
        session = session_factory(expires_in=3600, api_key_expires_in=3600)
        mock_http = AsyncMock()
        result = await session.get_valid_api_key(mock_http)
        assert result == "sk_test"
        mock_http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_triggered_when_access_token_expired(self, session_factory) -> None:
        # Access token expired; API key not yet expired
        session = session_factory(expires_in=-100, api_key_expires_in=3600)
        with aioresponses() as m:
            # First POST: refresh_token grant → new tokens
            m.post(
                OAUTH_TOKEN_URL,
                status=200,
                payload={
                    "access_token": "at_new",
                    "refresh_token": "rt_new",
                    "id_token": "id_new",
                    "api_key": "sk_new",
                    "expires_in": 3600,
                },
            )
            # Second POST: token-exchange grant → new API key
            m.post(OAUTH_TOKEN_URL, status=200, payload={"openai_api_key": "sk_new"})

            async with aiohttp.ClientSession() as http:
                result = await session.get_valid_api_key(http)
            assert result == "sk_new"
            assert session.access_token == "at_new"
            assert session.refresh_token == "rt_new"
            assert session.api_key == "sk_new"

    @pytest.mark.asyncio
    async def test_refresh_triggered_when_api_key_expired(self, session_factory) -> None:
        session = session_factory(expires_in=-100, api_key_expires_in=-100)
        with aioresponses() as m:
            m.post(
                OAUTH_TOKEN_URL,
                status=200,
                payload={
                    "access_token": "at_new2",
                    "refresh_token": "rt_new2",
                    "id_token": "id_new2",
                    "api_key": "sk_new2",
                    "expires_in": 3600,
                },
            )
            m.post(OAUTH_TOKEN_URL, status=200, payload={"openai_api_key": "sk_new2"})

            async with aiohttp.ClientSession() as http:
                result = await session.get_valid_api_key(http)
            assert result == "sk_new2"

    @pytest.mark.asyncio
    async def test_proactive_refresh_within_60s_of_expiry(self, session_factory) -> None:
        # Access token expires in 30 seconds (within 60s margin) → proactive refresh
        session = session_factory(expires_in=30, api_key_expires_in=3600)
        with aioresponses() as m:
            m.post(
                OAUTH_TOKEN_URL,
                status=200,
                payload={
                    "access_token": "at_proactive",
                    "refresh_token": "rt_proactive",
                    "id_token": "id_proactive",
                    "api_key": "sk_proactive",
                    "expires_in": 3600,
                },
            )
            m.post(OAUTH_TOKEN_URL, status=200, payload={"openai_api_key": "sk_proactive"})

            async with aiohttp.ClientSession() as http:
                result = await session.get_valid_api_key(http)
            assert result == "sk_proactive"
            assert session.access_token == "at_proactive"

    @pytest.mark.asyncio
    async def test_refresh_raises_oauth_refresh_failed_on_invalid_grant(self, session_factory) -> None:
        session = session_factory(expires_in=-100, api_key_expires_in=3600)
        with aioresponses() as m:
            m.post(
                OAUTH_TOKEN_URL,
                status=400,
                payload={
                    "error": "invalid_grant",
                    "error_description": "Token revoked",
                },
            )

            async with aiohttp.ClientSession() as http:
                with pytest.raises(OAuthRefreshFailed) as exc_info:
                    await session.get_valid_api_key(http)
            assert "invalid_grant" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_refresh_raises_on_network_failure(self, session_factory) -> None:
        session = session_factory(expires_in=-100, api_key_expires_in=3600)
        mock_http = AsyncMock()
        # Simulate a transport-level failure — the mock doesn't support
        # `async with` properly but the exception still propagates as OAuthRefreshFailed.
        mock_http.post.side_effect = OSError("DNS failure")

        with pytest.raises(OAuthRefreshFailed):
            await session.get_valid_api_key(mock_http)


# ── File persistence ───────────────────────────────────────────────────────


class TestFilePersistence:
    def test_save_and_load_round_trip(self, tmp_path: Path, session_factory) -> None:
        session = session_factory()
        path = tmp_path / "session.json"
        session._file_path = str(path)
        session.save()
        loaded = OAuthSession.load(path)
        assert loaded.access_token == session.access_token
        assert loaded.refresh_token == session.refresh_token
        assert loaded.api_key == session.api_key
        assert loaded.access_token_expires_at == session.access_token_expires_at

    def test_load_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            OAuthSession.load(tmp_path / "nonexistent.json")

    def test_save_creates_file(self, tmp_path: Path, session_factory) -> None:
        session = session_factory()
        path = tmp_path / "new_session.json"
        session._file_path = str(path)
        session.save()
        assert path.exists()

    def test_create_session_file_sets_path_and_saves(self, tmp_path: Path, fresh_tokens) -> None:
        session = OAuthSession.from_token_response(fresh_tokens, "app_cid")
        config_dir = tmp_path
        auth_ref = "my-auth-ref-uuid"
        returned = OAuthSession.create_session_file(session, auth_ref, config_dir)
        expected_path = config_dir / "openai_oauth" / f"{auth_ref}.json"
        assert returned._file_path == str(expected_path)
        assert expected_path.exists()
        # Verify content
        loaded = OAuthSession.load(expected_path)
        assert loaded.api_key == fresh_tokens["api_key"]

    def test_create_session_file_creates_directory(self, tmp_path: Path, fresh_tokens) -> None:
        session = OAuthSession.from_token_response(fresh_tokens, "app_cid")
        config_dir = tmp_path / "nested"
        auth_ref = "auth-uuid"
        OAuthSession.create_session_file(session, auth_ref, config_dir)
        assert (config_dir / "openai_oauth").is_dir()
