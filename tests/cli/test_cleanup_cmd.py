"""Tests for the kitty cleanup command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from kitty.cli.cleanup_cmd import _detect_stale_env, _display_value, run_cleanup


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
