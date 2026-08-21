"""Regression tests: overlapping kitty Claude sessions share ~/.claude/settings.json.

Claude Code hot-reloads settings.json and re-applies its ``env`` block to
running sessions on change, and every ``kitty claude`` session patches that one
user-global file. These tests pin the lifecycle invariants that must hold when
several sessions overlap on the same machine (e.g. Claude Code sessions in
different worktrees): any exit order restores the user's settings, a running
session keeps its own bridge routing, and crash recovery can always reconstruct
the pre-kitty content.

See ``.requirements/20260821T113955Z_multi_session_settings_race/REQUIREMENTS.md``.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kitty.launchers import claude as claude_mod
from kitty.launchers.claude import ClaudeAdapter

USER_SETTINGS = {
    "env": {"ANTHROPIC_AUTH_TOKEN": "user-token", "CUSTOM_KEY": "keep-me"},
    "model": "opus[1m]",
}


def _write_user_settings(settings_path: Path) -> str:
    """Write the pristine user settings.json and return its exact content."""
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(USER_SETTINGS, indent=2)
    settings_path.write_text(content, encoding="utf-8")
    return content


def _patch_backup_to_tmp(backup_path: Path):
    """Redirect the fixed global backup path to a temp file for hermeticity.

    Patches both the module constant ``_DEFAULT_BACKUP_PATH`` (so exists/read
    checks land on the temp file under the implementation that resolves the
    constant at call time) and the save/load/delete helpers (with fakes that
    delegate to the real implementations against the temp path, so real
    serialization and save-if-absent semantics are exercised). Under the
    pre-fix code (constant resolved at def-time) the constant patch is inert
    but the function fakes still keep tests hermetic.
    """
    real_save = claude_mod.save_settings_backup
    real_load = claude_mod.load_settings_backup
    real_delete = claude_mod.delete_settings_backup

    def fake_save(original: str) -> None:
        real_save(original, backup_path=backup_path)

    def fake_load() -> str | None:
        return real_load(backup_path=backup_path)

    def fake_delete() -> None:
        real_delete(backup_path=backup_path)

    return patch.multiple(
        claude_mod,
        _DEFAULT_BACKUP_PATH=backup_path,
        save_settings_backup=fake_save,
        load_settings_backup=fake_load,
        delete_settings_backup=fake_delete,
    )


def _session_env(port: int) -> dict[str, str]:
    """Return the env_overrides kitty injects for a bridge on ``port``."""
    return {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "ANTHROPIC_API_KEY": "sk-test",
        "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
    }


def _env(settings_path: Path) -> dict:
    """Return the current settings.json env block."""
    return json.loads(settings_path.read_text(encoding="utf-8")).get("env", {})


def _assert_user_settings_restored(settings_path: Path) -> None:
    """Assert settings.json is semantically equal to the user's pre-kitty content.

    Parsed-JSON equality, not byte equality: a surgical-unpatch fix legitimately
    re-serialises (key order, whitespace), and byte identity would wrongly imply
    that a user's own mid-session edits to settings.json must be clobbered.
    """
    assert json.loads(settings_path.read_text(encoding="utf-8")) == USER_SETTINGS, (
        "settings.json differs from the user's pre-kitty content after all sessions exited"
    )


class TestOverlappingExitOrder:
    """AC1/AC2: after all sessions exit, the user's settings are intact."""

    def test_first_started_session_exiting_first_restores_user_settings(self, tmp_path: Path):
        """Non-LIFO exit (A starts, B starts, A exits, B exits) must not leave
        A's dead bridge URL in the user's settings.json."""
        settings_path = tmp_path / ".claude" / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        with _patch_backup_to_tmp(backup_path):
            original_a = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
            original_b = adapter.prepare_launch(_session_env(10002), settings_path=settings_path)
            adapter.cleanup_launch(original_a, settings_path=settings_path)
            adapter.cleanup_launch(original_b, settings_path=settings_path)

        _assert_user_settings_restored(settings_path)
        # The last exit must also remove the crash backup; a stale one would
        # make `kitty cleanup` restore it over future user edits.
        assert not backup_path.exists(), "stale crash backup left behind after all sessions exited"

    def test_last_started_session_exiting_first_restores_user_settings(self, tmp_path: Path):
        """LIFO exit (A starts, B starts, B exits, A exits) is the ordering
        that works today and must keep working under any fix."""
        settings_path = tmp_path / ".claude" / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        with _patch_backup_to_tmp(backup_path):
            original_a = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
            original_b = adapter.prepare_launch(_session_env(10002), settings_path=settings_path)
            adapter.cleanup_launch(original_b, settings_path=settings_path)
            adapter.cleanup_launch(original_a, settings_path=settings_path)

        _assert_user_settings_restored(settings_path)


class TestRunningSessionIsolation:
    """AC3: another session's exit must not strip a running session's routing."""

    def test_earlier_session_exit_keeps_later_session_routing(self, tmp_path: Path):
        """While B runs, A's cleanup must not leave settings.json without B's
        bridge URL (Claude Code hot-reloads this file into running sessions)."""
        settings_path = tmp_path / ".claude" / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        with _patch_backup_to_tmp(backup_path):
            original_a = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
            adapter.prepare_launch(_session_env(10002), settings_path=settings_path)
            adapter.cleanup_launch(original_a, settings_path=settings_path)

        env = _env(settings_path)
        assert env.get("ANTHROPIC_BASE_URL") == "http://127.0.0.1:10002", (
            "session B is still running but settings.json no longer routes to B's bridge"
        )
        # The user's non-kitty keys MUST survive any other session's exit.
        assert env.get("CUSTOM_KEY") == "keep-me"
        # A's exit must not delete the shared crash backup — session B still
        # needs it for crash recovery.
        assert backup_path.exists(), "a not-owning session's exit deleted the shared crash backup"


class TestStartDirectionCrossTalk:
    """AC5: starting a second session must not reroute the first session's bridge.

    This is the defect `kitty doctor` cannot detect: a running session silently
    gets its `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` rewritten when another
    session starts. Under the current shared-file design the single
    `ANTHROPIC_BASE_URL` key can only hold one session's value, so any
    shared-file patch propagates to every running session — the only fix is
    per-session isolation (Option B). This test stays red on the current
    surface and documents the invariant a correct fix must satisfy.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Documents open issue #22: any shared-file design reroutes running sessions on "
            "each new launch (Claude Code hot-reloads the settings env block). Only per-session "
            "settings isolation fixes it. Strict so the marker is removed when #22 lands."
        ),
    )
    def test_second_session_start_keeps_first_session_routing(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        with _patch_backup_to_tmp(backup_path):
            adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
            adapter.prepare_launch(_session_env(10002), settings_path=settings_path)

        env = _env(settings_path)
        assert env.get("ANTHROPIC_BASE_URL") == "http://127.0.0.1:10001", (
            "session A is still running but settings.json now routes to B's bridge — "
            "Claude Code will hot-reload A's env and reroute A's traffic through B"
        )


class TestAnyNumberOfSessions:
    """AC6: representative higher-order interleaving (3 sessions, mixed exit order)."""

    def test_three_sessions_with_first_starter_exiting_first_restores_user_settings(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        with _patch_backup_to_tmp(backup_path):
            original_a = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
            original_b = adapter.prepare_launch(_session_env(10002), settings_path=settings_path)
            original_c = adapter.prepare_launch(_session_env(10003), settings_path=settings_path)
            adapter.cleanup_launch(original_a, settings_path=settings_path)
            adapter.cleanup_launch(original_c, settings_path=settings_path)
            adapter.cleanup_launch(original_b, settings_path=settings_path)

        _assert_user_settings_restored(settings_path)


class TestCrashBackupIntegrity:
    """AC4: crash recovery must reconstruct the user's settings, not a patch."""

    def test_second_session_start_keeps_user_original_in_backup(self, tmp_path: Path):
        """While sessions overlap, the backup must hold the user's pre-kitty
        content so `kitty cleanup` cannot restore a poisoned file."""
        settings_path = tmp_path / ".claude" / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"
        user_content = _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        with _patch_backup_to_tmp(backup_path):
            adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
            adapter.prepare_launch(_session_env(10002), settings_path=settings_path)
            backup_content = backup_path.read_text(encoding="utf-8") if backup_path.exists() else None

        assert backup_content == user_content, (
            "the crash-recovery backup holds a previous session's patched content — "
            "`kitty cleanup` after a crash would restore a dead bridge URL"
        )


class TestCleanupIdempotency:
    """AC10: a stray second cleanup call must not re-poison settings.json."""

    def test_second_cleanup_of_owning_session_is_noop(self, tmp_path: Path):
        """If a cleanup is invoked a second time (e.g. finally + atexit both fire),
        settings.json must stay at the user's pre-kitty content — not be rewritten
        with the now-stale in-memory snapshot."""
        settings_path = tmp_path / ".claude" / "settings.json"
        backup_path = tmp_path / "claude-settings-backup.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        with _patch_backup_to_tmp(backup_path):
            original_a = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
            original_b = adapter.prepare_launch(_session_env(10002), settings_path=settings_path)
            adapter.cleanup_launch(original_a, settings_path=settings_path)
            adapter.cleanup_launch(original_b, settings_path=settings_path)
            adapter.cleanup_launch(original_b, settings_path=settings_path)  # stray double cleanup

        _assert_user_settings_restored(settings_path)
