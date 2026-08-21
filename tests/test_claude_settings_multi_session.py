"""Overlapping kitty Claude sessions must not interfere with each other.

Several ``kitty claude`` sessions commonly run at once (e.g. one per worktree,
often on different profiles). Before issue #22 every session patched the one
user-global ``~/.claude/settings.json``, so the single ``ANTHROPIC_BASE_URL``
key could only describe whichever session started last. These tests pin the
replacement invariants: each session's routing lives in its own file, any
start/exit order leaves the user's settings untouched, and one session's exit
cannot disturb another's.

See ``.requirements/20260821T134223Z_fix_issue_22_start_direction_cross_talk/REQUIREMENTS.md``.
"""

import json
from pathlib import Path

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


def _session_env(port: int) -> dict[str, str]:
    """Return the env_overrides kitty injects for a bridge on ``port``."""
    return {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "ANTHROPIC_API_KEY": "sk-test",
        "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
    }


def _session_file_env(prepared: str) -> dict:
    """Return the ``env`` block of a prepared session settings file."""
    return json.loads(Path(prepared).read_text(encoding="utf-8"))["env"]


class TestOverlappingExitOrder:
    """AC1/AC2: any exit order leaves the user's settings untouched."""

    def test_first_started_session_exiting_first(self, tmp_path: Path):
        """Non-LIFO exit (A starts, B starts, A exits, B exits).

        This ordering used to leave A's dead bridge URL in the user's file
        permanently, because B's snapshot had captured A's patch.
        """
        settings_path = tmp_path / ".claude" / "settings.json"
        original = _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        prepared_a = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
        prepared_b = adapter.prepare_launch(_session_env(10002), settings_path=settings_path)
        adapter.cleanup_launch(prepared_a, settings_path=settings_path)
        adapter.cleanup_launch(prepared_b, settings_path=settings_path)

        assert settings_path.read_text(encoding="utf-8") == original
        assert not Path(prepared_a).exists()
        assert not Path(prepared_b).exists()

    def test_last_started_session_exiting_first(self, tmp_path: Path):
        """LIFO exit (A starts, B starts, B exits, A exits)."""
        settings_path = tmp_path / ".claude" / "settings.json"
        original = _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        prepared_a = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
        prepared_b = adapter.prepare_launch(_session_env(10002), settings_path=settings_path)
        adapter.cleanup_launch(prepared_b, settings_path=settings_path)
        adapter.cleanup_launch(prepared_a, settings_path=settings_path)

        assert settings_path.read_text(encoding="utf-8") == original


class TestRunningSessionIsolation:
    """AC3: another session's exit must not disturb a running session."""

    def test_earlier_session_exit_keeps_later_session_routing(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        prepared_a = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
        prepared_b = adapter.prepare_launch(_session_env(10002), settings_path=settings_path)
        adapter.cleanup_launch(prepared_a, settings_path=settings_path)

        assert Path(prepared_b).exists(), "a session's exit deleted another session's settings"
        assert _session_file_env(prepared_b)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:10002"


class TestStartDirectionCrossTalk:
    """AC5 / issue #22: a second session's start must not reroute the first.

    Under the shared-file design this was structurally impossible to satisfy —
    one ``ANTHROPIC_BASE_URL`` key cannot hold two sessions' values, so the
    second launch always overwrote the first session's routing. Per-session
    files remove the shared key entirely.
    """

    def test_second_session_start_keeps_first_session_routing(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        original = _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        prepared_a = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
        adapter.prepare_launch(_session_env(10002), settings_path=settings_path)

        assert _session_file_env(prepared_a)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:10001", (
            "session A is still running but its settings now route to B's bridge"
        )
        # And the shared file B would have rewritten is never touched at all.
        assert settings_path.read_text(encoding="utf-8") == original


class TestAnyNumberOfSessions:
    """AC6: representative higher-order interleaving (3 sessions, mixed order)."""

    def test_three_sessions_with_first_starter_exiting_first(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        original = _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        prepared = [adapter.prepare_launch(_session_env(10001 + i), settings_path=settings_path) for i in range(3)]
        for index in (0, 2, 1):
            adapter.cleanup_launch(prepared[index], settings_path=settings_path)

        assert settings_path.read_text(encoding="utf-8") == original
        assert not any(Path(item).exists() for item in prepared)

    def test_each_session_keeps_its_own_routing_while_all_run(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        prepared = [adapter.prepare_launch(_session_env(10001 + i), settings_path=settings_path) for i in range(3)]

        for index, item in enumerate(prepared):
            assert _session_file_env(item)["ANTHROPIC_BASE_URL"] == f"http://127.0.0.1:{10001 + index}"


class TestCleanupIdempotency:
    """Cleanup runs from both the finally block and the atexit handler."""

    def test_double_cleanup_is_safe(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        original = _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        prepared = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
        adapter.cleanup_launch(prepared, settings_path=settings_path)
        adapter.cleanup_launch(prepared, settings_path=settings_path)

        assert settings_path.read_text(encoding="utf-8") == original
