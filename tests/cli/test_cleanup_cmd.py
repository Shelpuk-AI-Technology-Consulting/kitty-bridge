"""Tests for the kitty cleanup command."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from kitty.cli.cleanup_cmd import (
    _detect_stale_env,
    _display_value,
    _load_backup,
    run_cleanup,
    run_kilo_cleanup,
)
from kitty.launchers.claude import _atomic_write_text


def test_detect_stale_env_with_localhost_url():
    env = {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:12345",
        "ANTHROPIC_API_KEY": "test-key",
        "ANTHROPIC_MODEL": "glm-5.1",
        "API_TIMEOUT_MS": "3000000",
    }
    stale = _detect_stale_env(env)
    assert "ANTHROPIC_BASE_URL" in stale
    assert "ANTHROPIC_API_KEY" in stale
    assert "ANTHROPIC_MODEL" in stale
    assert "API_TIMEOUT_MS" not in stale


def test_detect_stale_env_with_localhost_hostname():
    env = {
        "ANTHROPIC_BASE_URL": "http://localhost:8080",
    }
    stale = _detect_stale_env(env)
    assert "ANTHROPIC_BASE_URL" in stale


def test_detect_stale_env_clean():
    env = {
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        "API_TIMEOUT_MS": "3000000",
    }
    stale = _detect_stale_env(env)
    assert stale == []


def test_detect_stale_env_no_base_url():
    env = {
        "API_TIMEOUT_MS": "3000000",
    }
    stale = _detect_stale_env(env)
    assert stale == []


def test_run_cleanup_removes_stale_values(tmp_path):
    settings_path = tmp_path / "settings.json"
    backup_path = tmp_path / "claude-settings-backup.json"
    settings_data = {
        "env": {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:32987",
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_MODEL": "glm-5.1",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5.1",
            "API_TIMEOUT_MS": "3000000",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        },
        "model": "opus",
    }
    settings_path.write_text(json.dumps(settings_data))

    with patch("kitty.cli.cleanup_cmd._get_backup_path", return_value=backup_path):
        exit_code = run_cleanup(settings_path=settings_path)
    assert exit_code == 0

    result = json.loads(settings_path.read_text())
    env = result["env"]
    assert "ANTHROPIC_BASE_URL" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_MODEL" not in env
    assert "ANTHROPIC_DEFAULT_OPUS_MODEL" not in env
    # User values preserved
    assert env["API_TIMEOUT_MS"] == "3000000"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


def test_run_cleanup_already_clean(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_data = {
        "env": {
            "API_TIMEOUT_MS": "3000000",
        },
    }
    settings_path.write_text(json.dumps(settings_data))

    exit_code = run_cleanup(settings_path=settings_path)
    assert exit_code == 0


def test_display_value_short_string():
    assert _display_value("hello") == "hello"


def test_display_value_long_string():
    assert _display_value("x" * 41) == "x" * 37 + "..."


def test_display_value_none():
    assert _display_value(None) == "None"


def test_display_value_int():
    assert _display_value(42) == "42"


def test_run_cleanup_no_settings_file(tmp_path):
    settings_path = tmp_path / "nonexistent.json"
    exit_code = run_cleanup(settings_path=settings_path)
    assert exit_code == 0


def test_run_cleanup_non_localhost_url_preserved(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_data = {
        "env": {
            "ANTHROPIC_BASE_URL": "https://my-proxy.example.com",
            "API_TIMEOUT_MS": "3000000",
        },
    }
    settings_path.write_text(json.dumps(settings_data))

    exit_code = run_cleanup(settings_path=settings_path)
    assert exit_code == 0

    result = json.loads(settings_path.read_text())
    env = result["env"]
    # Non-localhost URL should be preserved
    assert env["ANTHROPIC_BASE_URL"] == "https://my-proxy.example.com"


class TestAuthTokenCleanup:
    """Regression tests for ANTHROPIC_AUTH_TOKEN not being cleaned up."""

    def test_auth_token_detected_as_stale_with_kitty_value(self):
        """ANTHROPIC_AUTH_TOKEN = kitty-bridge-token should be detected as stale."""
        env = {
            "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
            "ANTHROPIC_API_KEY": "test-key",
        }
        stale = _detect_stale_env(env)
        assert "ANTHROPIC_AUTH_TOKEN" in stale
        assert "ANTHROPIC_API_KEY" in stale

    def test_auth_token_stale_without_base_url(self):
        """Kitty token alone (no localhost base URL) should still trigger cleanup."""
        env = {
            "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
        }
        stale = _detect_stale_env(env)
        assert "ANTHROPIC_AUTH_TOKEN" in stale

    def test_non_kitty_auth_token_not_stale(self):
        """A real user auth token must not be flagged as stale."""
        env = {
            "ANTHROPIC_AUTH_TOKEN": "sk-ant-real-user-token-12345",
        }
        stale = _detect_stale_env(env)
        assert stale == []

    def test_cleanup_removes_auth_token(self, tmp_path: Path):
        """Regression: cleanup must remove kitty-bridge-token, fixing 401 errors."""
        settings_path = tmp_path / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"
        settings_data = {
            "env": {
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:32987",
                "ANTHROPIC_API_KEY": "test-key",
                "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
                "ANTHROPIC_MODEL": "glm-5.1",
            },
        }
        settings_path.write_text(json.dumps(settings_data))

        with patch("kitty.cli.cleanup_cmd._get_backup_path", return_value=backup_path):
            exit_code = run_cleanup(settings_path=settings_path)
        assert exit_code == 0

        result = json.loads(settings_path.read_text())
        env = result["env"]
        assert "ANTHROPIC_AUTH_TOKEN" not in env
        assert "ANTHROPIC_BASE_URL" not in env
        assert "ANTHROPIC_API_KEY" not in env
        assert "ANTHROPIC_MODEL" not in env

    def test_cleanup_removes_kitty_token_without_base_url(self, tmp_path: Path):
        """When base URL was already removed, kitty token should still be cleaned."""
        settings_path = tmp_path / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"
        settings_data = {
            "env": {
                "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
                "ANTHROPIC_MODEL": "glm-5.1",
            },
        }
        settings_path.write_text(json.dumps(settings_data))

        with patch("kitty.cli.cleanup_cmd._get_backup_path", return_value=backup_path):
            exit_code = run_cleanup(settings_path=settings_path)
        assert exit_code == 0

        result = json.loads(settings_path.read_text())
        env = result["env"]
        assert "ANTHROPIC_AUTH_TOKEN" not in env
        assert "ANTHROPIC_MODEL" not in env

    def test_cleanup_removes_connector_opt_out(self, tmp_path: Path):
        """KBR-245 AC-R3a: the connector opt-out key is scrubbed alongside
        the other injected keys after a crashed legacy session."""
        settings_path = tmp_path / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"
        settings_data = {
            "env": {
                "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
                "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
                "ANTHROPIC_MODEL": "glm-5.1",
            },
        }
        settings_path.write_text(json.dumps(settings_data))

        with patch("kitty.cli.cleanup_cmd._get_backup_path", return_value=backup_path):
            exit_code = run_cleanup(settings_path=settings_path)
        assert exit_code == 0

        result = json.loads(settings_path.read_text())
        env = result["env"]
        assert "ANTHROPIC_AUTH_TOKEN" not in env
        assert "ENABLE_CLAUDEAI_MCP_SERVERS" not in env
        assert "ANTHROPIC_MODEL" not in env


class TestBackupRestore:
    """Tests for backup-based restore in run_cleanup."""

    def test_cleanup_restores_from_backup(self, tmp_path: Path):
        """When a backup exists, cleanup should restore exact original content."""
        settings_path = tmp_path / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"

        original_settings = {
            "env": {"ANTHROPIC_AUTH_TOKEN": "my-real-token", "API_TIMEOUT_MS": "3000000"},
            "model": "opus",
        }
        backup_path.write_text(json.dumps(original_settings))

        # Current settings have Kitty values
        current_settings = {
            "env": {
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:12345",
                "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
                "API_TIMEOUT_MS": "3000000",
            },
        }
        settings_path.write_text(json.dumps(current_settings))

        with patch("kitty.cli.cleanup_cmd._get_backup_path", return_value=backup_path):
            exit_code = run_cleanup(settings_path=settings_path)

        assert exit_code == 0
        result = json.loads(settings_path.read_text())
        assert result["env"]["ANTHROPIC_AUTH_TOKEN"] == "my-real-token"
        assert "ANTHROPIC_BASE_URL" not in result["env"]
        # Backup should be deleted after restore
        assert not backup_path.exists()

    def test_cleanup_restores_byte_exactly_from_crlf_backup(self, tmp_path: Path) -> None:
        """A CRLF backup is restored byte-exactly to the user-global settings file.

        Catches the universal-newlines read defect on POSIX today: the backup
        is staged with raw CRLF bytes (``Path.write_bytes``), the restore is
        driven through ``run_cleanup`` (not the free functions), and the
        result is asserted against the staged bytes verbatim. On Windows the
        two defects cancel for pure-CRLF content (read strips CR, write adds
        CR), so the Windows-visible byte-identity for the write defect is
        supplied by ``test_cleanup_restores_byte_exactly_from_lf_backup_on_windows``.

        Args:
            tmp_path: Per-test temp directory.
        """
        settings_path = tmp_path / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"

        backup_bytes = b'{\n  "model": "sonnet",\r\n  "env": {"API_TIMEOUT_MS": "999"}\r\n}\n'
        backup_path.write_bytes(backup_bytes)

        # Settings carry live kitty values so the restore path fires.
        settings_path.write_text(
            json.dumps(
                {
                    "env": {
                        "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
                        "ANTHROPIC_BASE_URL": "http://127.0.0.1:45678",
                    },
                },
            ),
            encoding="utf-8",
        )

        with patch("kitty.cli.cleanup_cmd._get_backup_path", return_value=backup_path):
            exit_code = run_cleanup(settings_path=settings_path)

        assert exit_code == 0, "run_cleanup failed during exact restore"
        assert settings_path.read_bytes() == backup_bytes, (
            f"restored settings are {settings_path.read_bytes()!r}; expected {backup_bytes!r}"
        )
        assert not backup_path.exists(), "kitty cleanup left the backup behind"

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="write-side CRLF translation only corrupts on Windows; CI covers it on the Windows leg",
    )
    def test_cleanup_restores_byte_exactly_from_lf_backup_on_windows(self, tmp_path: Path) -> None:
        """On Windows, an LF backup restores to LF bytes (the ticket's exact scenario).

        Claude Code writes LF, so the realistic scenario for any user is an
        LF backup being restored on Windows. Pre-fix this fails because
        ``_atomic_write_text`` translates ``\\n`` to ``\\r\\n`` on write.

        Args:
            tmp_path: Per-test temp directory.
        """
        settings_path = tmp_path / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"

        backup_bytes = b'{\n  "model": "sonnet",\n  "env": {"API_TIMEOUT_MS": "999"}\n}\n'
        backup_path.write_bytes(backup_bytes)

        settings_path.write_text(
            json.dumps(
                {
                    "env": {
                        "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
                        "ANTHROPIC_BASE_URL": "http://127.0.0.1:45678",
                    },
                },
            ),
            encoding="utf-8",
        )

        with patch("kitty.cli.cleanup_cmd._get_backup_path", return_value=backup_path):
            exit_code = run_cleanup(settings_path=settings_path)

        assert exit_code == 0, "run_cleanup failed during exact restore"
        assert settings_path.read_bytes() == backup_bytes, (
            f"restored settings are {settings_path.read_bytes()!r}; expected {backup_bytes!r}"
        )
        assert not backup_path.exists()

    def test_cleanup_heuristic_fallback_no_backup(self, tmp_path: Path):
        """Without backup, heuristic should still remove all Kitty keys."""
        settings_path = tmp_path / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"

        settings_data = {
            "env": {
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:32987",
                "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
                "ANTHROPIC_MODEL": "glm-5.1",
                "API_TIMEOUT_MS": "3000000",
            },
        }
        settings_path.write_text(json.dumps(settings_data))

        with patch("kitty.cli.cleanup_cmd._get_backup_path", return_value=backup_path):
            exit_code = run_cleanup(settings_path=settings_path)

        assert exit_code == 0
        result = json.loads(settings_path.read_text())
        env = result["env"]
        assert "ANTHROPIC_BASE_URL" not in env
        assert "ANTHROPIC_AUTH_TOKEN" not in env
        assert "ANTHROPIC_MODEL" not in env
        assert env["API_TIMEOUT_MS"] == "3000000"

    def test_stale_backup_not_restored_when_no_live_kitty_values(self, tmp_path: Path):
        """A backup left from an ended chain must not revert the user's settings.

        The settings file carries no kitty values, so no live/crashed session
        owns it — restoring an ancient backup would silently lose everything
        the user changed since."""
        settings_path = tmp_path / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"

        ancient = {"env": {"ANTHROPIC_AUTH_TOKEN": "token-from-weeks-ago"}, "model": "opus"}
        backup_path.write_text(json.dumps(ancient))
        current = {"env": {"ANTHROPIC_BASE_URL": "https://my-proxy.example.com", "API_TIMEOUT_MS": "3000000"}}
        settings_path.write_text(json.dumps(current))

        with patch("kitty.cli.cleanup_cmd._get_backup_path", return_value=backup_path):
            exit_code = run_cleanup(settings_path=settings_path)

        assert exit_code == 0
        result = json.loads(settings_path.read_text())
        # Current settings untouched, stale backup discarded.
        assert result["env"]["ANTHROPIC_BASE_URL"] == "https://my-proxy.example.com"
        assert result["env"]["API_TIMEOUT_MS"] == "3000000"
        assert not backup_path.exists()

    def test_missing_settings_file_resurrected_from_backup(self, tmp_path: Path):
        """When settings.json vanished mid-session, the backup is the only
        surviving original — cleanup must restore it, not delete it."""
        settings_path = tmp_path / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"

        original = {"env": {"ANTHROPIC_AUTH_TOKEN": "my-real-token"}, "model": "opus"}
        backup_path.write_text(json.dumps(original))

        with patch("kitty.cli.cleanup_cmd._get_backup_path", return_value=backup_path):
            exit_code = run_cleanup(settings_path=settings_path)

        assert exit_code == 0
        result = json.loads(settings_path.read_text())
        assert result["env"]["ANTHROPIC_AUTH_TOKEN"] == "my-real-token"
        assert not backup_path.exists()


class TestUndecodableSettings:
    """Issue #27: a recovery command must diagnose damage, never crash on it.

    ``read_text(encoding="utf-8")`` on a UTF-16 file raises ``UnicodeDecodeError``,
    which is a ``ValueError`` — neither ``OSError`` nor ``JSONDecodeError``.
    Notepad's "Unicode" save produces exactly such a file, and now that
    ``cleanup`` is reachable in a broken installation this crash is on the
    operator's path.
    """

    def test_undecodable_settings_reports_an_error(self, tmp_path: Path, capsys) -> None:
        """The heuristic path must exit 1 with a message, not a traceback."""
        settings_path = tmp_path / "settings.json"
        settings_path.write_bytes(json.dumps({"env": {}}).encode("utf-16"))

        with patch("kitty.cli.cleanup_cmd._get_backup_path", return_value=tmp_path / "absent.json"):
            exit_code = run_cleanup(settings_path=settings_path)

        assert exit_code == 1
        assert "Cannot read" in capsys.readouterr().out

    def test_undecodable_settings_still_restores_the_backup(self, tmp_path: Path) -> None:
        """An unreadable file means a crashed patch, so the exact backup wins."""
        settings_path = tmp_path / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"
        settings_path.write_bytes(json.dumps({"env": {}}).encode("utf-16"))
        backup_path.write_text(json.dumps({"model": "opus"}), encoding="utf-8")

        with patch("kitty.cli.cleanup_cmd._get_backup_path", return_value=backup_path):
            exit_code = run_cleanup(settings_path=settings_path)

        assert exit_code == 0
        assert json.loads(settings_path.read_text(encoding="utf-8")) == {"model": "opus"}
        assert not backup_path.exists()


# ── KBR-260 — byte-exact CRLF backup restore ─────────────────────────────────


class TestBackupReaders:
    """`_load_backup` must return the backup's bytes verbatim (KBR-260).

    The pre-fix reader used ``Path.read_text`` (universal newlines), which
    strips ``\\r\\n`` on every platform — silently breaking the byte-identity
    contract for any CRLF backup the user already has on disk.
    """

    def test_load_backup_preserves_crlf_backup_bytes(self, tmp_path: Path) -> None:
        """A CRLF backup's bytes are returned verbatim, no CR-strip.

        Args:
            tmp_path: Per-test temp directory.
        """
        backup_path = tmp_path / "backup.json"
        original_bytes = b'{\n  "model": "opus",\r\n  "env": {"API_TIMEOUT_MS": "3000000"}\r\n}\n'
        backup_path.write_bytes(original_bytes)

        result = _load_backup(backup_path)

        assert result is not None
        assert result.encode("utf-8") == original_bytes, (
            f"_load_backup returned bytes {result.encode('utf-8')!r}; expected {original_bytes!r}"
        )


class TestAtomicWriteText:
    """`_atomic_write_text` must preserve the destination's bytes (KBR-260).

    The pre-fix writer left the open's ``newline`` at the default ``None``,
    which makes CPython translate ``\\n`` to ``os.linesep`` on write. On
    Windows that is ``\\r\\n`` — silently corrupting any LF content into
    CRLF.
    """

    def test_opens_with_no_newline_translation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """`_atomic_write_text` opens the destination with ``newline=""``.

        The write-side defect is invisible behaviourally on POSIX because
        ``os.linesep == "\\n"`` there (CPython's ``TextIOWrapper`` is a no-op
        even when ``os.linesep`` is monkeypatched at runtime — empirically
        confirmed). The only POSIX-runnable regression guard is the open
        call's kwargs; behavioural coverage lives in the Windows-only test
        below.

        Args:
            tmp_path: Per-test temp directory.
            monkeypatch: Pytest's monkeypatch fixture (auto-restores).
        """
        captured: list[dict[str, object]] = []
        real_fdopen = os.fdopen

        def recording_fdopen(fd, mode, **kwargs):
            captured.append(kwargs)
            return real_fdopen(fd, mode, **kwargs)

        monkeypatch.setattr(os, "fdopen", recording_fdopen)

        target = tmp_path / "settings.json"
        _atomic_write_text(target, '{"a": 1}\n')

        assert len(captured) == 1, f"expected exactly one os.fdopen call, got {len(captured)}"
        assert captured[0].get("newline") == "", f"_atomic_write_text opened without newline=''; kwargs={captured[0]!r}"

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="write-side CRLF translation only fires on Windows; CI covers it on the Windows leg",
    )
    def test_writes_byte_exactly_on_windows(self, tmp_path: Path) -> None:
        """On Windows, `_atomic_write_text` preserves LF and CRLF content bytes.

        CI-verified on the Fast gate's Windows leg.

        Args:
            tmp_path: Per-test temp directory.
        """
        for content in (
            '{"a": 1}\n{"b": 2}\n',
            '{"a": 1}\r\n{"b": 2}\r\n',
        ):
            target = tmp_path / "settings.json"
            _atomic_write_text(target, content)
            assert target.read_bytes() == content.encode("utf-8"), (
                f"_atomic_write_text wrote {target.read_bytes()!r}; expected {content.encode('utf-8')!r}"
            )


class TestRunKiloCleanup:
    """The Kilo arm of ``kitty cleanup`` (KBR-268): backup-only exact restore.

    Mirrors ``TestBackupRestore`` for kilo.json. The backup is patched through
    ``cleanup_cmd._get_kilo_backup_path`` (the lazy single-source reader) and
    the config path is passed explicitly, so no test touches the real home.
    """

    @staticmethod
    def _patched_kilo_markers_config() -> dict:
        """Return a kilo.json body carrying both kitty markers."""
        return {
            "provider": {
                "kitty": {"options": {"baseURL": "http://127.0.0.1:18080/v1"}},
                "openai": {"options": {"apiKey": "user-key"}},
            },
            "model": "kitty/gpt-4o",
        }

    def test_restores_byte_exactly_and_removes_backup(self, tmp_path: Path) -> None:
        """SIGKILL-style staging: backup + patched config → exact restore (AC-R4)."""
        kilo_path = tmp_path / "kilo.json"
        backup_path = tmp_path / "kilo-config-backup.json"

        original = {
            "provider": {"openai": {"options": {"apiKey": "user-key"}}},
            "model": "opus",
        }
        backup_path.write_text(json.dumps(original))
        kilo_path.write_text(json.dumps(self._patched_kilo_markers_config()))

        with patch("kitty.cli.cleanup_cmd._get_kilo_backup_path", return_value=backup_path):
            exit_code = run_kilo_cleanup(settings_path=kilo_path)

        assert exit_code == 0
        assert json.loads(kilo_path.read_text()) == original
        assert not backup_path.exists(), "kitty cleanup left the Kilo backup behind"

    def test_restores_when_only_model_prefix_marker(self, tmp_path: Path) -> None:
        """A ``kitty/`` model prefix alone still triggers the restore (AC-R4).

        Kitty always writes the provider block and the model key as a pair, so
        a model-only file is hand-trimmed crash damage.
        """
        kilo_path = tmp_path / "kilo.json"
        backup_path = tmp_path / "kilo-config-backup.json"

        original = {"provider": {"openai": {"options": {"apiKey": "user-key"}}}}
        backup_path.write_text(json.dumps(original))
        kilo_path.write_text(json.dumps({"model": "kitty/gpt-4o"}))

        with patch("kitty.cli.cleanup_cmd._get_kilo_backup_path", return_value=backup_path):
            exit_code = run_kilo_cleanup(settings_path=kilo_path)

        assert exit_code == 0
        assert json.loads(kilo_path.read_text()) == original
        assert not backup_path.exists()

    def test_remote_url_provider_counts_as_stale(self, tmp_path: Path) -> None:
        """A remote-URL ``provider.kitty`` alone is NOT a live kitty marker (AC-R4).

        The stale-backup path fires instead: the backup is deleted, the user's
        config is left alone — a user who named their own provider ``kitty``
        must not have their config reverted.
        """
        kilo_path = tmp_path / "kilo.json"
        backup_path = tmp_path / "kilo-config-backup.json"

        backup_path.write_text(json.dumps({"model": "opus"}))
        remote_only = {
            "provider": {
                "kitty": {"options": {"baseURL": "https://api.openai.com/v1"}},
            },
        }
        kilo_path.write_text(json.dumps(remote_only))
        kilo_bytes_before = kilo_path.read_bytes()

        with patch("kitty.cli.cleanup_cmd._get_kilo_backup_path", return_value=backup_path):
            exit_code = run_kilo_cleanup(settings_path=kilo_path)

        assert exit_code == 0
        assert kilo_path.read_bytes() == kilo_bytes_before
        assert not backup_path.exists(), "stale Kilo backup was not removed"

    def test_resurrects_missing_kilo_json(self, tmp_path: Path) -> None:
        """Backup present + no kilo.json → the file is resurrected (AC-R5)."""
        kilo_path = tmp_path / "kilo.json"
        backup_path = tmp_path / "kilo-config-backup.json"

        original = {"provider": {"openai": {"options": {"apiKey": "user-key"}}}}
        backup_path.write_text(json.dumps(original))
        assert not kilo_path.exists()

        with patch("kitty.cli.cleanup_cmd._get_kilo_backup_path", return_value=backup_path):
            exit_code = run_kilo_cleanup(settings_path=kilo_path)

        assert exit_code == 0
        assert json.loads(kilo_path.read_text()) == original
        assert not backup_path.exists()

    def test_unreadable_kilo_json_restores_from_backup(self, tmp_path: Path) -> None:
        """An unparseable kilo.json counts as a crashed patch (AC-R6).

        ``kitty cleanup`` is an explicit repair request, so the exact backup
        wins — the deliberate opposite of the live session's
        ``cleanup_launch``, which must never guess.
        """
        kilo_path = tmp_path / "kilo.json"
        backup_path = tmp_path / "kilo-config-backup.json"

        original = {"provider": {"openai": {"options": {"apiKey": "user-key"}}}}
        backup_path.write_text(json.dumps(original))
        kilo_path.write_text("not valid json {{{")

        with patch("kitty.cli.cleanup_cmd._get_kilo_backup_path", return_value=backup_path):
            exit_code = run_kilo_cleanup(settings_path=kilo_path)

        assert exit_code == 0
        assert json.loads(kilo_path.read_text()) == original
        assert not backup_path.exists()

    def test_stale_backup_deleted_config_untouched(self, tmp_path: Path) -> None:
        """Backup + clean current config → stale backup removed, config kept (AC-R7)."""
        kilo_path = tmp_path / "kilo.json"
        backup_path = tmp_path / "kilo-config-backup.json"

        backup_path.write_text(json.dumps({"model": "opus"}))
        clean = {"provider": {"openai": {"options": {"apiKey": "user-key"}}}}
        kilo_path.write_text(json.dumps(clean))
        kilo_bytes_before = kilo_path.read_bytes()

        with patch("kitty.cli.cleanup_cmd._get_kilo_backup_path", return_value=backup_path):
            exit_code = run_kilo_cleanup(settings_path=kilo_path)

        assert exit_code == 0
        assert kilo_path.read_bytes() == kilo_bytes_before
        assert not backup_path.exists()

    def test_no_backup_is_noop_negative_control(self, tmp_path: Path) -> None:
        """No backup + marker-bearing config → strict no-op (AC-R8).

        The negative control for this harness (TEST_SUITE §6.3.2): pre-fix
        legacy damage has no backup to restore, and the backup-only design
        never strips without one — the config must be left exactly as staged.
        """
        kilo_path = tmp_path / "kilo.json"
        backup_path = tmp_path / "absent-backup.json"
        assert not backup_path.exists()

        staged = self._patched_kilo_markers_config()
        kilo_path.write_text(json.dumps(staged))
        kilo_bytes_before = kilo_path.read_bytes()

        with patch("kitty.cli.cleanup_cmd._get_kilo_backup_path", return_value=backup_path):
            exit_code = run_kilo_cleanup(settings_path=kilo_path)

        assert exit_code == 0
        assert kilo_path.read_bytes() == kilo_bytes_before

    def test_unreadable_backup_reports_error(self, tmp_path: Path, capsys) -> None:
        """A backup path that cannot be read → one Error line, exit 1 (AC-R9).

        Staged as a directory: ``open()`` raises ``IsADirectoryError`` (an
        ``OSError``) on every platform — ``chmod 000`` cannot stage this on
        the Windows leg.
        """
        kilo_path = tmp_path / "kilo.json"
        backup_path = tmp_path / "kilo-config-backup-dir"
        backup_path.mkdir()

        staged = self._patched_kilo_markers_config()
        kilo_path.write_text(json.dumps(staged))
        kilo_bytes_before = kilo_path.read_bytes()

        with patch("kitty.cli.cleanup_cmd._get_kilo_backup_path", return_value=backup_path):
            exit_code = run_kilo_cleanup(settings_path=kilo_path)

        assert exit_code == 1
        assert kilo_path.read_bytes() == kilo_bytes_before
        assert "Error" in capsys.readouterr().out

    def test_restore_write_failure_reports_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        """A failing restore write → one Error line, exit 1, backup kept (AC-R9).

        The restore goes through ``_atomic_write_text`` (mkstemp + rename, not
        ``Path.write_text``), so the seam is the source module attribute — the
        lazy import inside ``run_kilo_cleanup`` resolves it at call time.
        """
        import kitty.launchers.claude as claude_mod

        kilo_path = tmp_path / "kilo.json"
        backup_path = tmp_path / "kilo-config-backup.json"

        original = {"provider": {"openai": {"options": {"apiKey": "user-key"}}}}
        backup_path.write_text(json.dumps(original))
        kilo_path.write_text(json.dumps(self._patched_kilo_markers_config()))

        real_write = claude_mod._atomic_write_text

        def failing_write(path: Path, content: str) -> None:
            if path == kilo_path:
                raise OSError("simulated restore write failure")
            real_write(path, content)

        monkeypatch.setattr(claude_mod, "_atomic_write_text", failing_write)

        with patch("kitty.cli.cleanup_cmd._get_kilo_backup_path", return_value=backup_path):
            exit_code = run_kilo_cleanup(settings_path=kilo_path)

        assert exit_code == 1
        assert backup_path.exists(), "backup was deleted despite the restore failure"
        assert "Error" in capsys.readouterr().out

    def test_restores_byte_exactly_from_crlf_backup(self, tmp_path: Path) -> None:
        """A CRLF backup restores byte-exactly to kilo.json (AC-R11, KBR-262)."""
        kilo_path = tmp_path / "kilo.json"
        backup_path = tmp_path / "kilo-config-backup.json"

        backup_bytes = (
            b'{\r\n  "provider": {"openai": {"apiKey": "sk-original"}},\r\n'
            b'  "model": "opus",\r\n'
            b'  "comment": "user kept CRLF line endings"\r\n'
            b"}\r\n"
        )
        backup_path.write_bytes(backup_bytes)
        kilo_path.write_text(json.dumps(self._patched_kilo_markers_config()))

        with patch("kitty.cli.cleanup_cmd._get_kilo_backup_path", return_value=backup_path):
            exit_code = run_kilo_cleanup(settings_path=kilo_path)

        assert exit_code == 0
        assert kilo_path.read_bytes() == backup_bytes, (
            f"restored kilo.json is {kilo_path.read_bytes()!r}; expected {backup_bytes!r}"
        )
        assert not backup_path.exists()
