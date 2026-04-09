"""Tests for ANTHROPIC_API_KEY injection and error handling in settings.json.

Claude Code's settings.json env block overrides process-level env vars,
so ANTHROPIC_API_KEY must be in the env block for Claude Code to authenticate
with the local bridge.
"""

import json
import uuid
from pathlib import Path

import pytest

from kitty.launchers.claude import ClaudeAdapter
from kitty.profiles.schema import Profile


def _make_profile(model: str = "minimax-m2.7") -> Profile:
    return Profile(
        name="test-profile",
        provider="minimax",
        model=model,
        auth_ref=str(uuid.uuid4()),
    )


def _write_settings(path: Path, env: dict | None = None, **extra: object) -> str:
    settings: dict = {**extra}
    if env is not None:
        settings["env"] = env
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(settings, indent=2)
    path.write_text(content, encoding="utf-8")
    return content


class TestApiKeyInSettingsEnv:
    """ANTHROPIC_API_KEY must be injected into settings.json env block so
    Claude Code authenticates with the bridge."""

    def test_injects_api_key_into_settings_env(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(settings_path, env={})

        adapter = ClaudeAdapter()
        env_overrides = {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:4242",
            "ANTHROPIC_API_KEY": "sk-test-key-123",
        }
        adapter.prepare_launch(env_overrides, settings_path=settings_path)

        patched = json.loads(settings_path.read_text(encoding="utf-8"))
        assert patched["env"]["ANTHROPIC_API_KEY"] == "sk-test-key-123"

    def test_does_not_remove_auth_token_from_settings_env(self, tmp_path: Path):
        """ANTHROPIC_AUTH_TOKEN must NOT be removed — it is not sent to the
        bridge; the bridge always uses Bearer {resolved_key}.  Removing it
        causes Claude Code to show 'Not logged in'."""
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(settings_path, env={
            "ANTHROPIC_AUTH_TOKEN": "secret-token",
            "ANTHROPIC_BASE_URL": "https://old.example.com",
        })

        adapter = ClaudeAdapter()
        adapter.prepare_launch(
            {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"},
            settings_path=settings_path,
        )

        patched = json.loads(settings_path.read_text(encoding="utf-8"))
        # ANTHROPIC_AUTH_TOKEN must still be present
        assert patched["env"]["ANTHROPIC_AUTH_TOKEN"] == "secret-token"

    def test_bridge_overrides_win_over_pre_existing_settings(self, tmp_path: Path):
        """Bridge env_overrides must win over pre-existing settings.json values."""
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(settings_path, env={
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:8080",
            "ANTHROPIC_API_KEY": "sk-from-settings",
            "ANTHROPIC_MODEL": "minimax-m2.7",
        })

        adapter = ClaudeAdapter()
        original = adapter.prepare_launch(
            {
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:9999",
                "ANTHROPIC_API_KEY": "sk-from-bridge",
                "ANTHROPIC_MODEL": "minimax-m2.7",
            },
            settings_path=settings_path,
        )

        patched = json.loads(settings_path.read_text(encoding="utf-8"))
        assert patched["env"]["ANTHROPIC_API_KEY"] == "sk-from-bridge"
        assert patched["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"

        adapter.cleanup_launch(original, settings_path=settings_path)

    def test_roundtrip_preserves_auth_token(self, tmp_path: Path):
        """After prepare_launch + cleanup_launch, ANTHROPIC_AUTH_TOKEN must be
        restored to its original value."""
        settings_path = tmp_path / ".claude" / "settings.json"
        original_content = _write_settings(
            settings_path,
            env={
                "ANTHROPIC_AUTH_TOKEN": "my-secret-token",
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
            },
        )

        adapter = ClaudeAdapter()
        saved = adapter.prepare_launch(
            {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242", "ANTHROPIC_API_KEY": "sk-test"},
            settings_path=settings_path,
        )

        adapter.cleanup_launch(saved, settings_path=settings_path)

        restored = settings_path.read_text(encoding="utf-8")
        assert restored == original_content
        parsed = json.loads(restored)
        assert parsed["env"]["ANTHROPIC_AUTH_TOKEN"] == "my-secret-token"


class TestMalformedSettings:
    """prepare_launch must handle malformed settings.json gracefully."""

    def test_malformed_json_returns_none_does_not_crash(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("not valid json {{{", encoding="utf-8")

        adapter = ClaudeAdapter()
        result = adapter.prepare_launch(
            {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"},
            settings_path=settings_path,
        )

        # Must return None so cleanup_launch is a no-op
        assert result is None
        # Original malformed content must be preserved (fail-closed)
        assert settings_path.read_text(encoding="utf-8") == "not valid json {{{"

    def test_settings_root_is_list_returns_none(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text('["some", "array"]', encoding="utf-8")

        adapter = ClaudeAdapter()
        result = adapter.prepare_launch(
            {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"},
            settings_path=settings_path,
        )

        assert result is None
        assert settings_path.read_text(encoding="utf-8") == '["some", "array"]'


class TestAtomicWrite:
    """File writes must not corrupt settings.json."""

    def test_write_produces_valid_json_and_preserves_existing_fields(self, tmp_path: Path):
        """After patching, the file must be valid JSON with no data loss."""
        settings_path = tmp_path / ".claude" / "settings.json"
        original_content = _write_settings(settings_path, env={"EXISTING": "value", "ANTHROPIC_AUTH_TOKEN": "keep-me"})

        adapter = ClaudeAdapter()
        original = adapter.prepare_launch(
            {
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:4242",
                "ANTHROPIC_API_KEY": "sk-test-key",
                "ANTHROPIC_MODEL": "minimax-m2.7",
                "ANTHROPIC_DEFAULT_OPUS_MODEL": "minimax-m2.7",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "minimax-m2.7",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "minimax-m2.7",
            },
            settings_path=settings_path,
        )

        # File must be valid JSON after patching
        patched = json.loads(settings_path.read_text(encoding="utf-8"))
        assert patched["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4242"
        assert patched["env"]["EXISTING"] == "value"  # not corrupted
        assert patched["env"]["ANTHROPIC_AUTH_TOKEN"] == "keep-me"  # not removed

        # Restore must be byte-identical to original content
        adapter.cleanup_launch(original, settings_path=settings_path)
        assert settings_path.read_text(encoding="utf-8") == original_content
