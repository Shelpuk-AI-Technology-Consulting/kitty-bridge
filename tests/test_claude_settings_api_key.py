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


@pytest.fixture(autouse=True)
def _backup_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the crash backup at a temp file so tests never touch the real one.

    Without this, prepare_launch/cleanup_launch read and write the developer's
    actual ~/.config/kitty/claude-settings-backup.json.
    """
    monkeypatch.setattr("kitty.launchers.claude._DEFAULT_BACKUP_PATH", tmp_path / "claude-settings-backup.json")


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


def _session_env(prepared: str | None) -> dict:
    """Return the ``env`` block of the session file ``prepare_launch`` wrote."""
    assert prepared is not None, "prepare_launch must produce a session settings file"
    return json.loads(Path(prepared).read_text(encoding="utf-8"))["env"]


class TestApiKeyInSettingsEnv:
    """The credential kitty hands Claude Code for this session.

    It travels in the per-session settings file rather than the user-global one
    (issue #22), because Claude Code's settings env outranks the process env we
    give the child.
    """

    def test_injects_api_key(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(settings_path, env={})

        adapter = ClaudeAdapter()
        env_overrides = {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:4242",
            "ANTHROPIC_API_KEY": "sk-test-key-123",
        }
        prepared = adapter.prepare_launch(env_overrides, settings_path=settings_path)

        assert _session_env(prepared)["ANTHROPIC_API_KEY"] == "sk-test-key-123"

    def test_auth_token_is_kittys_not_the_users(self, tmp_path: Path):
        """kitty always supplies its own token so Claude Code does not demand a
        login; the user's own value must not shadow it for this session."""
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(
            settings_path,
            env={
                "ANTHROPIC_AUTH_TOKEN": "secret-token",
                "ANTHROPIC_BASE_URL": "https://old.example.com",
            },
        )

        adapter = ClaudeAdapter()
        prepared = adapter.prepare_launch(
            {
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:4242",
                "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
            },
            settings_path=settings_path,
        )

        assert _session_env(prepared)["ANTHROPIC_AUTH_TOKEN"] == "kitty-bridge-token"

    def test_bridge_values_are_the_ones_written(self, tmp_path: Path):
        """Whatever the user has configured, the session file carries the
        bridge's values — it outranks user scope, so these are what apply."""
        settings_path = tmp_path / ".claude" / "settings.json"
        user_original = _write_settings(
            settings_path,
            env={
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:8080",
                "ANTHROPIC_API_KEY": "sk-from-settings",
                "ANTHROPIC_MODEL": "minimax-m2.7",
            },
        )

        adapter = ClaudeAdapter()
        prepared = adapter.prepare_launch(
            {
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:9999",
                "ANTHROPIC_API_KEY": "sk-from-bridge",
                "ANTHROPIC_MODEL": "minimax-m2.7",
            },
            settings_path=settings_path,
        )

        env = _session_env(prepared)
        assert env["ANTHROPIC_API_KEY"] == "sk-from-bridge"
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"
        # The user's own file is untouched throughout.
        assert settings_path.read_text(encoding="utf-8") == user_original

        adapter.cleanup_launch(prepared, settings_path=settings_path)
        assert settings_path.read_text(encoding="utf-8") == user_original

    def test_roundtrip_leaves_the_users_auth_token_alone(self, tmp_path: Path):
        """The user's own token must survive a kitty session untouched."""
        settings_path = tmp_path / ".claude" / "settings.json"
        original_content = _write_settings(
            settings_path,
            env={
                "ANTHROPIC_AUTH_TOKEN": "my-secret-token",
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
            },
        )

        adapter = ClaudeAdapter()
        prepared = adapter.prepare_launch(
            {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242", "ANTHROPIC_API_KEY": "sk-test"},
            settings_path=settings_path,
        )

        adapter.cleanup_launch(prepared, settings_path=settings_path)

        restored = settings_path.read_text(encoding="utf-8")
        assert restored == original_content
        assert json.loads(restored)["env"]["ANTHROPIC_AUTH_TOKEN"] == "my-secret-token"


class TestMalformedSettings:
    """A broken user settings.json must never block a launch.

    kitty only reads that file now (for the stale-value warning), so malformed
    content is at most a missed warning — the session still gets its own file.
    """

    def test_malformed_json_still_routes_the_session(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("{ this is not valid json", encoding="utf-8")

        adapter = ClaudeAdapter()
        prepared = adapter.prepare_launch(
            {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"},
            settings_path=settings_path,
        )

        assert _session_env(prepared)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4242"

    def test_settings_root_is_list_still_routes_the_session(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text('["not", "an", "object"]', encoding="utf-8")

        adapter = ClaudeAdapter()
        prepared = adapter.prepare_launch(
            {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"},
            settings_path=settings_path,
        )

        assert _session_env(prepared)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4242"


class TestAtomicWrite:
    """The session file must be well-formed and self-contained."""

    def test_session_file_is_valid_json_and_user_file_untouched(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        original = _write_settings(
            settings_path,
            env={"EXISTING_VAR": "keep-me"},
            model="opus",
            someOtherField={"nested": "value"},
        )

        adapter = ClaudeAdapter()
        prepared = adapter.prepare_launch(
            {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242", "ANTHROPIC_API_KEY": "sk-test"},
            settings_path=settings_path,
        )

        session = json.loads(Path(prepared).read_text(encoding="utf-8"))
        assert session["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4242"
        assert list(session) == ["env"], "the session file must only carry an env block"
        assert settings_path.read_text(encoding="utf-8") == original
