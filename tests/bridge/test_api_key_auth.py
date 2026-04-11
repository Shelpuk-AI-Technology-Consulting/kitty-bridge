"""Tests for R4: API key authentication."""

from __future__ import annotations

from pathlib import Path

import aiohttp
import pytest

from kitty.bridge.keys import parse_keys_file
from kitty.bridge.server import BridgeServer
from kitty.providers.zai import ZaiRegularAdapter

# ---------------------------------------------------------------------------
# Keys file parser tests
# ---------------------------------------------------------------------------


class TestParseKeysFile:
    """parse_keys_file handles all keys.txt formats."""

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "keys.txt"
        p.write_text("")
        entries = parse_keys_file(p)
        assert entries == []

    def test_comments_and_blanks(self, tmp_path: Path):
        p = tmp_path / "keys.txt"
        p.write_text("# comment\n\n  \n# another comment\n")
        entries = parse_keys_file(p)
        assert entries == []

    def test_bare_key(self, tmp_path: Path):
        p = tmp_path / "keys.txt"
        p.write_text("sk-abc123\n")
        entries = parse_keys_file(p)
        assert len(entries) == 1
        assert entries[0].key == "sk-abc123"
        assert entries[0].profile is None

    def test_key_with_profile(self, tmp_path: Path):
        p = tmp_path / "keys.txt"
        p.write_text("sk-abc : my-profile\n")
        entries = parse_keys_file(p)
        assert len(entries) == 1
        assert entries[0].key == "sk-abc"
        assert entries[0].profile == "my-profile"

    def test_key_with_profile_no_spaces(self, tmp_path: Path):
        p = tmp_path / "keys.txt"
        p.write_text("sk-abc:my-profile\n")
        entries = parse_keys_file(p)
        assert entries[0].key == "sk-abc"
        assert entries[0].profile == "my-profile"

    def test_key_with_profile_extra_spaces(self, tmp_path: Path):
        p = tmp_path / "keys.txt"
        p.write_text("  sk-abc   :   my-profile  \n")
        entries = parse_keys_file(p)
        assert entries[0].key == "sk-abc"
        assert entries[0].profile == "my-profile"

    def test_multiple_entries(self, tmp_path: Path):
        p = tmp_path / "keys.txt"
        p.write_text("sk-abc : profile-a\nsk-def\n# comment\nsk-ghi : profile-c\n")
        entries = parse_keys_file(p)
        assert len(entries) == 3
        assert entries[0].key == "sk-abc"
        assert entries[0].profile == "profile-a"
        assert entries[1].key == "sk-def"
        assert entries[1].profile is None
        assert entries[2].key == "sk-ghi"
        assert entries[2].profile == "profile-c"

    def test_duplicate_keys_raises_error(self, tmp_path: Path):
        p = tmp_path / "keys.txt"
        p.write_text("sk-abc\nsk-abc\n")
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            parse_keys_file(p)

    def test_profile_with_colon(self, tmp_path: Path):
        """First colon is the separator; subsequent colons are part of profile name."""
        p = tmp_path / "keys.txt"
        p.write_text("sk-abc : my:profile:name\n")
        entries = parse_keys_file(p)
        assert entries[0].profile == "my:profile:name"

    def test_key_trimming(self, tmp_path: Path):
        p = tmp_path / "keys.txt"
        p.write_text("  sk-abc  \n")
        entries = parse_keys_file(p)
        assert entries[0].key == "sk-abc"

    def test_case_sensitive_keys(self, tmp_path: Path):
        """Keys are case-sensitive, so these are NOT duplicates."""
        p = tmp_path / "keys.txt"
        p.write_text("sk-ABC\nsk-abc\n")
        entries = parse_keys_file(p)
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# Server auth middleware integration tests
# ---------------------------------------------------------------------------


def _write_keys(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _make_server(keys_file: str | None = None, **kwargs):
    return BridgeServer(
        adapter=None,
        provider=ZaiRegularAdapter(),
        resolved_key="test-key",
        model="gpt-4o",
        provider_config={},
        keys_file=keys_file,
        **kwargs,
    )


class TestApiKeyAuthMiddleware:
    """Server rejects/allows based on keys file."""

    @pytest.mark.asyncio
    async def test_no_keys_file_allows_all(self):
        """Without keys file, all requests are allowed."""
        server = _make_server(keys_file=None)
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session, session.get(
                f"http://127.0.0.1:{port}/healthz"
            ) as resp:
                assert resp.status == 200
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_valid_key_allowed(self, tmp_path: Path):
        keys = _write_keys(tmp_path / "keys.txt", "sk-valid-key\n")
        server = _make_server(keys_file=str(keys))
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"http://127.0.0.1:{port}/healthz",
                    headers={"Authorization": "Bearer sk-valid-key"},
                ) as resp,
            ):
                assert resp.status == 200
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_invalid_key_rejected(self, tmp_path: Path):
        keys = _write_keys(tmp_path / "keys.txt", "sk-valid-key\n")
        server = _make_server(keys_file=str(keys))
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"http://127.0.0.1:{port}/healthz",
                    headers={"Authorization": "Bearer sk-wrong-key"},
                ) as resp,
            ):
                assert resp.status == 401
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_no_auth_header_rejected(self, tmp_path: Path):
        keys = _write_keys(tmp_path / "keys.txt", "sk-valid-key\n")
        server = _make_server(keys_file=str(keys))
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session, session.get(
                f"http://127.0.0.1:{port}/healthz"
            ) as resp:
                assert resp.status == 401
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_duplicate_keys_prevent_startup(self, tmp_path: Path):
        """Bridge refuses to start if keys file has duplicates."""
        keys = _write_keys(tmp_path / "keys.txt", "sk-dup\nsk-dup\n")
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            _make_server(keys_file=str(keys))

    @pytest.mark.asyncio
    async def test_key_id_in_access_log(self, tmp_path: Path):
        """Access log should show hashed key ID, not raw key."""
        keys = _write_keys(tmp_path / "keys.txt", "sk-secret-key\n")
        log_path = tmp_path / "access.log"
        server = _make_server(
            keys_file=str(keys),
            access_log_path=str(log_path),
        )
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"http://127.0.0.1:{port}/healthz",
                    headers={"Authorization": "Bearer sk-secret-key"},
                ) as resp,
            ):
                assert resp.status == 200
        finally:
            await server.stop_async()

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        fields = lines[0].split("\t")
        # Key ID should be first 8 chars of SHA-256, not "sk-secret-key"
        import hashlib

        expected_hash = hashlib.sha256(b"sk-secret-key").hexdigest()[:8]
        assert fields[2] == expected_hash
