"""Tests for the env kitty hands Claude Code through its settings source.

Claude Code's settings ``env`` block overrides process-level env vars, so kitty
must supply a settings file to route the child at the bridge. Since issue #22
that is a **per-session** file passed via ``--settings``; the user-global
``~/.claude/settings.json`` is only read, never written.
"""

import json
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from kitty.launchers import claude as claude_mod
from kitty.launchers.claude import ClaudeAdapter
from kitty.profiles.schema import Profile

_BACKUP_NAME = "claude-settings-backup.json"


@pytest.fixture(autouse=True)
def _backup_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the crash backup at a temp file so tests never touch the real one.

    Without this, prepare_launch/cleanup_launch read and write the developer's
    actual ~/.config/kitty/claude-settings-backup.json.

    Args:
        tmp_path: Per-test temp directory.
        monkeypatch: Pytest's monkeypatch fixture (auto-restores).

    Returns:
        The hermetic backup path (same value patched into the module).
    """
    backup_path = tmp_path / _BACKUP_NAME
    monkeypatch.setattr("kitty.launchers.claude._DEFAULT_BACKUP_PATH", backup_path)
    return backup_path


def _make_profile(model: str = "minimax-m2.7") -> Profile:
    return Profile(
        name="test-profile",
        provider="minimax",
        model=model,
        auth_ref=str(uuid.uuid4()),
    )


def _session_env(prepared: str | None) -> dict:
    """Return the ``env`` block of the session file ``prepare_launch`` wrote."""
    assert prepared is not None, "prepare_launch must produce a session settings file"
    return json.loads(Path(prepared).read_text(encoding="utf-8"))["env"]


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
    """What kitty puts in front of Claude Code for this session."""

    def test_injects_base_url(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(settings_path, env={"ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic"})

        adapter = ClaudeAdapter()
        prepared = adapter.prepare_launch({"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"}, settings_path=settings_path)

        assert _session_env(prepared)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4242"

    def test_injects_model_vars(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(settings_path, env={})

        adapter = ClaudeAdapter()
        env_overrides = {
            "ANTHROPIC_MODEL": "minimax-m2.7",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "minimax-m2.7",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "minimax-m2.7",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "minimax-m2.7",
        }
        prepared = adapter.prepare_launch(env_overrides, settings_path=settings_path)

        env = _session_env(prepared)
        assert env["ANTHROPIC_MODEL"] == "minimax-m2.7"
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "minimax-m2.7"
        assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "minimax-m2.7"
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "minimax-m2.7"

    def test_kitty_auth_token_wins_over_the_users(self, tmp_path: Path):
        """kitty's token must be the one Claude Code uses for this session.

        The bridge uses Bearer auth independently of ANTHROPIC_AUTH_TOKEN, but
        Claude Code refuses to start without one ('Not logged in'). The session
        file outranks user scope, so the user's own token cannot leak in.
        """
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(settings_path, env={"ANTHROPIC_AUTH_TOKEN": "secret-token"})

        adapter = ClaudeAdapter()
        env_overrides = {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:4242",
            "ANTHROPIC_API_KEY": "sk-test",
            "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
        }
        prepared = adapter.prepare_launch(env_overrides, settings_path=settings_path)

        assert _session_env(prepared)["ANTHROPIC_AUTH_TOKEN"] == "kitty-bridge-token"

    def test_auth_token_present_when_user_is_logged_out(self, tmp_path: Path):
        """Without a token Claude Code demands an Anthropic login and ignores
        ANTHROPIC_API_KEY, so kitty always supplies its own."""
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(settings_path, env={})

        adapter = ClaudeAdapter()
        env_overrides = {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:4242",
            "ANTHROPIC_API_KEY": "sk-test",
            "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
        }
        prepared = adapter.prepare_launch(env_overrides, settings_path=settings_path)

        assert _session_env(prepared)["ANTHROPIC_AUTH_TOKEN"] == "kitty-bridge-token"

    def test_users_other_settings_are_left_alone(self, tmp_path: Path):
        """The user's top-level keys keep their values — kitty neither copies
        nor rewrites them; Claude Code inherits them from user scope."""
        settings_path = tmp_path / ".claude" / "settings.json"
        original = _write_settings(
            settings_path,
            env={"ANTHROPIC_BASE_URL": "https://old.example.com"},
            model="opus[1m]",
            effortLevel="high",
        )

        adapter = ClaudeAdapter()
        prepared = adapter.prepare_launch({"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"}, settings_path=settings_path)

        assert settings_path.read_text(encoding="utf-8") == original
        assert "model" not in json.loads(Path(prepared).read_text(encoding="utf-8"))

    def test_prepares_a_file_even_without_a_user_env_block(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(settings_path, model="opus[1m]")

        adapter = ClaudeAdapter()
        prepared = adapter.prepare_launch({"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"}, settings_path=settings_path)

        assert _session_env(prepared)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4242"

    def test_prepares_a_file_when_the_user_has_no_settings_at_all(self, tmp_path: Path):
        """Previously this returned None and the session was left unrouted."""
        settings_path = tmp_path / ".claude" / "settings.json"

        adapter = ClaudeAdapter()
        prepared = adapter.prepare_launch({"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"}, settings_path=settings_path)

        assert _session_env(prepared)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4242"
        assert not settings_path.exists()

    def test_injects_context_tokens(self, tmp_path: Path):
        """AC5.3: the context window must reach the child through the settings
        source, since that outranks the process env."""
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(settings_path, env={})

        adapter = ClaudeAdapter()
        env_overrides = {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:4242",
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1000000",
        }
        prepared = adapter.prepare_launch(env_overrides, settings_path=settings_path)

        assert _session_env(prepared)["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1000000"


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
    """A full session lifecycle must leave the user's file untouched."""

    def test_prepare_then_cleanup_leaves_user_settings_byte_identical(self, tmp_path: Path):
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

        prepared = adapter.prepare_launch(env_overrides, settings_path=settings_path)

        # The session's values live in the session file, not the user's.
        session_env = _session_env(prepared)
        assert session_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4242"
        assert session_env["ANTHROPIC_MODEL"] == "minimax-m2.7"
        assert settings_path.read_text(encoding="utf-8") == original_content

        adapter.cleanup_launch(prepared, settings_path=settings_path)

        assert settings_path.read_text(encoding="utf-8") == original_content
        assert not Path(prepared).exists()


class TestSettingsBackup:
    """Crash-recovery backup helpers.

    Since issue #22 the launch path no longer patches the user-global file, so
    nothing writes a backup during a normal session. These helpers stay for
    ``kitty cleanup``, which repairs files poisoned by *pre-fix* kitty versions
    during rollout, and are exercised directly rather than through
    ``prepare_launch``.
    """

    def test_prepare_launch_does_not_write_a_backup(self, tmp_path: Path):
        """The backup existed to undo a patch kitty no longer makes."""
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_settings(settings_path, env={"ANTHROPIC_AUTH_TOKEN": "my-real-token"}, model="opus")

        adapter = ClaudeAdapter()
        env_overrides = {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:4242",
            "ANTHROPIC_API_KEY": "sk-test",
            "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
        }

        with patch("kitty.launchers.claude.save_settings_backup") as mock_save:
            adapter.prepare_launch(env_overrides, settings_path=settings_path)

        mock_save.assert_not_called()

    def test_cleanup_launch_deletes_backup_on_the_legacy_path(self, tmp_path: Path):
        """A bare ``str`` still means "restore this verbatim and drop the
        backup" — the documented contract for legacy callers."""
        settings_path = tmp_path / ".claude" / "settings.json"
        original_content = _write_settings(settings_path, env={"ANTHROPIC_AUTH_TOKEN": "my-real-token"})

        adapter = ClaudeAdapter()
        with patch("kitty.launchers.claude.delete_settings_backup") as mock_delete:
            adapter.cleanup_launch(original_content, settings_path=settings_path)
            mock_delete.assert_called_once()

    def test_save_then_load_round_trips(self, tmp_path: Path, _backup_in_tmp: Path):
        """The helpers `kitty cleanup` depends on must round-trip exactly."""
        content = '{"env": {"ANTHROPIC_AUTH_TOKEN": "user-token"}}'

        claude_mod.save_settings_backup(content)

        assert claude_mod.load_settings_backup() == content
        claude_mod.delete_settings_backup()
        assert claude_mod.load_settings_backup() is None


class TestCleanupOwnership:
    """Ownership-checked restore — the legacy shared-file path.

    ``prepare_launch`` no longer produces a ``_SessionSnapshot`` (it returns a
    per-session file handle instead), so this machinery is unreachable from the
    launch path. It is retained as revert insurance and to keep ``kitty
    cleanup``'s repair of pre-fix leftovers meaningful, and is therefore driven
    directly here with a hand-built snapshot — going through ``prepare_launch``
    would silently exercise nothing.
    """

    @staticmethod
    def _snapshot(original_content: str, injected: dict[str, str]):
        """Build the snapshot a pre-#22 ``prepare_launch`` would have returned."""
        return claude_mod._SessionSnapshot(original_content, injected)

    def test_cleanup_restores_snapshot_when_backup_missing(self, tmp_path: Path, _backup_in_tmp: Path):
        """When the crash backup is gone, the owner still restores its snapshot."""
        settings_path = tmp_path / ".claude" / "settings.json"
        original_content = _write_settings(
            settings_path,
            env={"ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic"},
        )
        injected = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"}
        snapshot = self._snapshot(original_content, injected)

        # Stand in for the patch a pre-#22 prepare_launch would have written.
        _write_settings(settings_path, env=dict(injected))
        _backup_in_tmp.unlink(missing_ok=True)

        ClaudeAdapter().cleanup_launch(snapshot, settings_path=settings_path)

        assert settings_path.read_text(encoding="utf-8") == original_content

    def test_cleanup_prefers_the_backup_over_its_own_snapshot(self, tmp_path: Path, _backup_in_tmp: Path):
        """The backup holds the user's true pre-kitty content.

        A later session's own snapshot captured an earlier session's patch, so
        restoring it would reintroduce a dead bridge URL (issue #19).
        """
        settings_path = tmp_path / ".claude" / "settings.json"
        user_original = '{\n  "env": {\n    "CUSTOM_KEY": "keep-me"\n  }\n}'
        _backup_in_tmp.write_text(user_original, encoding="utf-8")

        injected = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:10002"}
        polluted_snapshot = self._snapshot(
            '{"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:10001"}}',
            injected,
        )
        _write_settings(settings_path, env=dict(injected))

        ClaudeAdapter().cleanup_launch(polluted_snapshot, settings_path=settings_path)

        assert settings_path.read_text(encoding="utf-8") == user_original

    def test_cleanup_leaves_file_when_user_hand_edited_kitty_key(self, tmp_path: Path, _backup_in_tmp: Path):
        """A non-owner must not overwrite whoever holds the file now.

        If the user (or a later session) changes a kitty-injected value while
        this session runs, this session no longer owns the file.
        """
        settings_path = tmp_path / ".claude" / "settings.json"
        injected = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"}
        snapshot = self._snapshot('{"env": {"CUSTOM_KEY": "keep-me"}}', injected)

        # The value on disk differs from what this session injected.
        _write_settings(settings_path, env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:5555"})

        ClaudeAdapter().cleanup_launch(snapshot, settings_path=settings_path)

        restored = json.loads(settings_path.read_text(encoding="utf-8"))
        assert restored["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:5555", (
            "cleanup overwrote another owner's value with its stale snapshot"
        )

    def test_cleanup_leaves_an_unreadable_file_alone(self, tmp_path: Path, _backup_in_tmp: Path):
        """Ownership is unknowable without a parseable file — never guess."""
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("{ broken", encoding="utf-8")
        snapshot = self._snapshot('{"env": {}}', {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4242"})

        ClaudeAdapter().cleanup_launch(snapshot, settings_path=settings_path)

        assert settings_path.read_text(encoding="utf-8") == "{ broken"
