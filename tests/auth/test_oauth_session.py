"""Tests for OAuthSession: token state machine and file persistence."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from kitty.auth.oauth_session import (
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


# ── get_valid_api_key ───────────────────────────────────────────


class _FakeTransport:
    """A scripted :class:`~kitty.auth.token_transport.TokenTransport`.

    The token state machine's claims -- when it refreshes, what it stores, how
    it reports failure -- are claims about *logic*, so they are proved against a
    scripted transport rather than a socket.  That the seam itself speaks the
    wire is a different claim, proved against a real local server in
    ``tests/auth/test_token_transport.py``.  Neither file is sufficient alone:
    a fake that accepted anything would let a broken seam ship green.
    """

    def __init__(self, *responses: tuple[int, str] | Exception) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def post_form(
        self,
        url: str,
        data: Mapping[str, str],
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float,
        proxies: Mapping[str, str] | None = None,
    ) -> tuple[int, str]:
        self.calls.append(
            {"url": url, "data": dict(data), "headers": dict(headers or {}), "timeout": timeout}
        )
        if not self._responses:
            raise AssertionError(f"unscripted POST to {url} carrying {dict(data)}")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _refresh_body(**overrides) -> str:
    """Serialize a successful refresh-grant response."""
    payload = {
        "access_token": "at_new",
        "refresh_token": "rt_new",
        "id_token": "id_new",
        "expires_in": 3600,
    }
    payload.update(overrides)
    return json.dumps(payload)


def _exchange_body(api_key: str = "sk_new") -> str:
    """Serialize a successful token-exchange response."""
    return json.dumps({"openai_api_key": api_key})


class TestGetValidApiKey:
    @pytest.mark.asyncio
    async def test_returns_key_without_posting_when_fresh(self, session_factory) -> None:
        session = session_factory(expires_in=3600, api_key_expires_in=3600)
        transport = _FakeTransport()

        result = await session.get_valid_api_key(transport)

        assert result == "sk_test"
        assert transport.calls == []

    @pytest.mark.asyncio
    async def test_refresh_triggered_when_access_token_expired(self, session_factory) -> None:
        session = session_factory(expires_in=-100, api_key_expires_in=3600)
        transport = _FakeTransport((200, _refresh_body()), (200, _exchange_body()))

        result = await session.get_valid_api_key(transport)

        assert result == "sk_new"
        assert session.access_token == "at_new"
        assert session.id_token == "id_new"

    @pytest.mark.asyncio
    async def test_refresh_triggered_when_api_key_expired(self, session_factory) -> None:
        session = session_factory(expires_in=-100, api_key_expires_in=-100)
        transport = _FakeTransport((200, _refresh_body()), (200, _exchange_body("sk_new2")))

        assert await session.get_valid_api_key(transport) == "sk_new2"

    @pytest.mark.asyncio
    async def test_proactive_refresh_within_60s_of_expiry(self, session_factory) -> None:
        """Expiry is 30 s out, inside ``_REFRESH_MARGIN_SECONDS``."""
        session = session_factory(expires_in=30, api_key_expires_in=3600)
        transport = _FakeTransport((200, _refresh_body()), (200, _exchange_body("sk_proactive")))

        assert await session.get_valid_api_key(transport) == "sk_proactive"

    @pytest.mark.asyncio
    async def test_the_refresh_token_is_rotated_when_the_server_returns_a_new_one(
        self, session_factory
    ) -> None:
        session = session_factory(expires_in=-100, api_key_expires_in=3600)
        transport = _FakeTransport((200, _refresh_body()), (200, _exchange_body()))

        await session.get_valid_api_key(transport)

        assert session.refresh_token == "rt_new"

    @pytest.mark.asyncio
    async def test_the_old_refresh_token_is_kept_when_the_server_returns_none(
        self, session_factory
    ) -> None:
        """Rotation is optional; dropping the old token would end the session."""
        session = session_factory(expires_in=-100, api_key_expires_in=3600)
        body = json.dumps({"access_token": "at_new", "id_token": "id_new", "expires_in": 3600})
        transport = _FakeTransport((200, body), (200, _exchange_body()))

        await session.get_valid_api_key(transport)

        assert session.refresh_token == "rt_test"

    @pytest.mark.asyncio
    async def test_a_successful_refresh_is_persisted(self, session_factory, tmp_path: Path) -> None:
        session = session_factory(expires_in=-100, api_key_expires_in=3600)
        session._file_path = str(tmp_path / "session.json")
        transport = _FakeTransport((200, _refresh_body()), (200, _exchange_body()))

        await session.get_valid_api_key(transport)

        assert json.loads((tmp_path / "session.json").read_text())["access_token"] == "at_new"

    @pytest.mark.asyncio
    async def test_a_failed_exchange_falls_back_to_the_access_token(self, session_factory) -> None:
        """Org accounts without Platform mapping have no API key to exchange for."""
        session = session_factory(expires_in=-100, api_key_expires_in=3600)
        transport = _FakeTransport((200, _refresh_body()), (400, json.dumps({"error": "no_org"})))

        result = await session.get_valid_api_key(transport)

        assert result == "at_new"
        assert session.api_key is None


class TestTheErrorContract:
    """The shapes ``get_valid_api_key`` is allowed to fail in.

    Moving this leg from ``aiohttp`` to ``curl_cffi`` (KBR-161) replaced an
    ``async with`` response object with a ``(status, text)`` pair, so every
    branch that used to read ``resp.status`` and ``await resp.json()`` was
    rewritten.  These pin the observable result of each one.
    """

    @pytest.mark.asyncio
    async def test_a_json_error_body_surfaces_its_error_and_description(
        self, session_factory
    ) -> None:
        session = session_factory(expires_in=-100, api_key_expires_in=3600)
        body = json.dumps({"error": "invalid_grant", "error_description": "Token revoked"})
        transport = _FakeTransport((400, body))

        with pytest.raises(OAuthRefreshFailed) as exc_info:
            await session.get_valid_api_key(transport)

        assert "invalid_grant" in str(exc_info.value)
        assert "Token revoked" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_an_unparseable_error_body_falls_back_to_the_status(
        self, session_factory
    ) -> None:
        """An HTML error page from a proxy must not become an opaque failure."""
        session = session_factory(expires_in=-100, api_key_expires_in=3600)
        transport = _FakeTransport((503, "<html>gateway timeout</html>"))

        with pytest.raises(OAuthRefreshFailed) as exc_info:
            await session.get_valid_api_key(transport)

        assert "HTTP 503" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_a_transport_exception_becomes_oauth_refresh_failed(
        self, session_factory
    ) -> None:
        session = session_factory(expires_in=-100, api_key_expires_in=3600)
        transport = _FakeTransport(OSError("DNS failure"))

        with pytest.raises(OAuthRefreshFailed) as exc_info:
            await session.get_valid_api_key(transport)

        assert "DNS failure" in str(exc_info.value)


class TestTheCodexIdentity:
    """KBR-161: this leg presents the same client as the API leg."""

    @pytest.mark.asyncio
    async def test_both_token_posts_carry_the_codex_user_agent(self, session_factory) -> None:
        """Both, not just the first -- ``_exchange_api_key`` is the site the
        ticket's own list of leaking call sites missed."""
        from kitty.codex_identity import build_codex_user_agent

        session = session_factory(expires_in=-100, api_key_expires_in=3600)
        transport = _FakeTransport((200, _refresh_body()), (200, _exchange_body()))

        await session.get_valid_api_key(transport)

        assert len(transport.calls) == 2
        assert [call["headers"].get("User-Agent") for call in transport.calls] == [
            build_codex_user_agent(),
            build_codex_user_agent(),
        ]

    @pytest.mark.asyncio
    async def test_both_token_posts_carry_an_explicit_timeout(self, session_factory) -> None:
        """The refresh lock is held across both, so neither may be unbounded."""
        session = session_factory(expires_in=-100, api_key_expires_in=3600)
        transport = _FakeTransport((200, _refresh_body()), (200, _exchange_body()))

        await session.get_valid_api_key(transport)

        assert all(call["timeout"] == 30 for call in transport.calls)


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
