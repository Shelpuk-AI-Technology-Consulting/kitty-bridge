"""Tests for Claude Code settings.json override logic.

Claude Code's settings.json ``env`` block overrides process-level env vars,
so kitty must temporarily patch the file to inject the bridge URL and model.
"""

import json
import uuid
from pathlib import Path

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
    """Write a Claude Code settings.json and return its content."""
    settings: dict = {**extra}
    if env is not None:
        settings["env"] = env
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(settings, indent=2)
    path.write_text(content, encoding="utf-8")
    return content


class TestPrepareLaunch:
    def test_injects_base_url_into_settings_env(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(settings_path, env={"ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic"})

        adapter = ClaudeAdapter()
        env_overrides = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"}
        original = adapter.prepare_launch(env_overrides, settings_path=settings_path)

        patched = json.loads(settings_path.read_text(encoding="utf-8"))
        assert patched["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4242"
        assert original is not None

    def test_injects_model_vars_into_settings_env(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(settings_path, env={})

        adapter = ClaudeAdapter()
        env_overrides = {
            "ANTHROPIC_MODEL": "minimax-m2.7",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "minimax-m2.7",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "minimax-m2.7",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "minimax-m2.7",
        }
        adapter.prepare_launch(env_overrides, settings_path=settings_path)

        patched = json.loads(settings_path.read_text(encoding="utf-8"))
        assert patched["env"]["ANTHROPIC_MODEL"] == "minimax-m2.7"
        assert patched["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "minimax-m2.7"
        assert patched["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "minimax-m2.7"
        assert patched["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "minimax-m2.7"

    def test_auth_token_preserved_not_removed(self, tmp_path: Path):
        """ANTHROPIC_AUTH_TOKEN must NOT be removed from settings.json env.
        The bridge uses Bearer auth (Authorization header) independently of
        ANTHROPIC_AUTH_TOKEN, and removing it causes 'Not logged in' errors."""
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(
            settings_path,
            env={
                "ANTHROPIC_AUTH_TOKEN": "secret-token",
                "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
            },
        )

        adapter = ClaudeAdapter()
        env_overrides = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242", "ANTHROPIC_API_KEY": "sk-test"}
        adapter.prepare_launch(env_overrides, settings_path=settings_path)

        patched = json.loads(settings_path.read_text(encoding="utf-8"))
        assert patched["env"]["ANTHROPIC_AUTH_TOKEN"] == "secret-token"

    def test_preserves_other_settings_fields(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(
            settings_path,
            env={"ANTHROPIC_BASE_URL": "https://old.example.com"},
            model="opus[1m]",
            effortLevel="high",
        )

        adapter = ClaudeAdapter()
        adapter.prepare_launch({"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"}, settings_path=settings_path)

        patched = json.loads(settings_path.read_text(encoding="utf-8"))
        assert patched["model"] == "opus[1m]"
        assert patched["effortLevel"] == "high"

    def test_creates_env_block_if_missing(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(settings_path, model="opus[1m]")

        adapter = ClaudeAdapter()
        adapter.prepare_launch({"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"}, settings_path=settings_path)

        patched = json.loads(settings_path.read_text(encoding="utf-8"))
        assert "env" in patched
        assert patched["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4242"

    def test_returns_none_if_no_settings_file(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"

        adapter = ClaudeAdapter()
        original = adapter.prepare_launch({"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"}, settings_path=settings_path)

        assert original is None


class TestCleanupLaunch:
    def test_restores_original_settings(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        original_content = _write_settings(
            settings_path,
            env={"ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic"},
        )

        adapter = ClaudeAdapter()
        adapter.cleanup_launch(original_content, settings_path=settings_path)

        assert settings_path.read_text(encoding="utf-8") == original_content

    def test_noop_when_original_is_none(self, tmp_path: Path):
        adapter = ClaudeAdapter()
        # Should not raise
        adapter.cleanup_launch(None, settings_path=tmp_path / ".claude" / "settings.json")


class TestRoundTrip:
    def test_prepare_then_cleanup_restores_exactly(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(
            settings_path,
            env={
                "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
                "ANTHROPIC_AUTH_TOKEN": "secret-token",
                "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5.1",
            },
            model="opus[1m]",
            effortLevel="high",
        )
        original_content = settings_path.read_text(encoding="utf-8")

        adapter = ClaudeAdapter()
        env_overrides = {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:4242",
            "ANTHROPIC_MODEL": "minimax-m2.7",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "minimax-m2.7",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "minimax-m2.7",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "minimax-m2.7",
        }

        # Prepare
        original = adapter.prepare_launch(env_overrides, settings_path=settings_path)

        # Verify patched
        patched = json.loads(settings_path.read_text(encoding="utf-8"))
        assert patched["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4242"
        assert patched["env"]["ANTHROPIC_MODEL"] == "minimax-m2.7"
        assert patched["env"]["ANTHROPIC_AUTH_TOKEN"] == "secret-token"

        # Cleanup
        adapter.cleanup_launch(original, settings_path=settings_path)

        # Verify restored
        assert settings_path.read_text(encoding="utf-8") == original_content
