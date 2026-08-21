"""Per-session Claude Code settings isolation (issue #22).

Every ``kitty claude`` session used to patch the one user-global
``~/.claude/settings.json``, so a second session's start rewrote what the first
session's settings said — the single ``ANTHROPIC_BASE_URL`` key cannot hold two
sessions' values. These tests pin the replacement design: each session gets its
own settings file, passed to Claude Code via ``--settings``, and kitty never
writes the user-global file at all.

See ``.requirements/20260821T134223Z_fix_issue_22_start_direction_cross_talk/REQUIREMENTS.md``
and ``.system_design/steps/20260821_per_session_claude_settings.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kitty.launchers import claude as claude_mod
from kitty.launchers.base import LauncherAdapter
from kitty.launchers.claude import ClaudeAdapter

USER_SETTINGS = {
    "env": {"ANTHROPIC_AUTH_TOKEN": "user-token", "CUSTOM_KEY": "keep-me"},
    "model": "opus[1m]",
}


def _write_user_settings(settings_path: Path) -> str:
    """Write a pristine user settings.json and return its exact content."""
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
        "ANTHROPIC_MODEL": "glm-5.3",
    }


def _session_file_env(prepared: str) -> dict:
    """Return the ``env`` block written into a prepared session file."""
    return json.loads(Path(prepared).read_text(encoding="utf-8"))["env"]


@pytest.fixture
def _no_global_writes(monkeypatch: pytest.MonkeyPatch):
    """Make any write to the user-global settings path an immediate failure.

    AC1's primary assertion. Byte-identity alone would also hold for a
    patch-then-restore implementation — precisely the design being replaced —
    so the guard proves kitty never writes the file in the first place. Every
    write in ``claude.py`` goes through these two helpers (pinned by
    :class:`TestWriteHelperPrecondition`).
    """
    guarded: list[Path] = []

    def _install(global_path: Path) -> None:
        guarded.append(global_path)

    real_json = claude_mod._atomic_write_json
    real_text = claude_mod._atomic_write_text

    def _check(path: Path) -> None:
        for forbidden in guarded:
            if Path(path) == forbidden:
                raise AssertionError(f"kitty wrote the user-global settings file: {path}")

    def fake_json(path: Path, data: dict) -> None:
        _check(path)
        real_json(path, data)

    def fake_text(path: Path, content: str) -> None:
        _check(path)
        real_text(path, content)

    monkeypatch.setattr(claude_mod, "_atomic_write_json", fake_json)
    monkeypatch.setattr(claude_mod, "_atomic_write_text", fake_text)
    return _install


class TestNoGlobalWrites:
    """AC1: kitty must never write the user-global settings file."""

    def test_second_session_start_leaves_global_file_untouched(self, tmp_path: Path, _no_global_writes):
        """The issue-#22 invariant: starting B must not disturb A's state.

        Under the old shared-file design B's ``prepare_launch`` rewrote the one
        ``ANTHROPIC_BASE_URL`` key, so the file no longer described A.
        """
        settings_path = tmp_path / ".claude" / "settings.json"
        original = _write_user_settings(settings_path)
        _no_global_writes(settings_path)
        before = settings_path.stat().st_mtime_ns
        adapter = ClaudeAdapter()

        prepared_a = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
        prepared_b = adapter.prepare_launch(_session_env(10002), settings_path=settings_path)

        # Secondary checks: content and mtime are both untouched.
        assert settings_path.read_text(encoding="utf-8") == original
        assert settings_path.stat().st_mtime_ns == before
        # A's routing still describes A's bridge, unaffected by B's start.
        assert _session_file_env(prepared_a)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:10001"
        assert _session_file_env(prepared_b)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:10002"

    def test_cleanup_never_writes_global_file(self, tmp_path: Path, _no_global_writes):
        """AC5: neither cleanup call may touch the global file."""
        settings_path = tmp_path / ".claude" / "settings.json"
        original = _write_user_settings(settings_path)
        _no_global_writes(settings_path)
        adapter = ClaudeAdapter()

        prepared = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
        adapter.cleanup_launch(prepared, settings_path=settings_path)
        # Second call mirrors the finally + atexit double-invocation.
        adapter.cleanup_launch(prepared, settings_path=settings_path)

        assert settings_path.read_text(encoding="utf-8") == original


class TestWriteHelperPrecondition:
    """Pin the precondition :func:`_no_global_writes` relies on."""

    def test_settings_path_writes_go_through_the_atomic_helpers(self):
        """No code path writes ``settings_path`` except via the helpers.

        The AC1 guard works by intercepting ``_atomic_write_json`` /
        ``_atomic_write_text``. If a future edit wrote the settings file
        directly, the guard would silently stop guarding — so the invariant is
        asserted rather than assumed. ``prepare_launch`` legitimately writes
        the *session* file through an ``mkstemp`` handle; that path is a
        freshly created temp file, never ``settings_path``.
        """
        source = Path(claude_mod.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "settings_path" not in stripped:
                continue
            assert ".write_text(" not in stripped, f"direct write to settings_path: {stripped}"
            assert "open(" not in stripped or "read" in stripped, f"direct open of settings_path: {stripped}"


class TestSessionFileIsolation:
    """AC2/AC3: independent files carrying exactly kitty's keys."""

    def test_two_sessions_get_distinct_files(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        prepared_a = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
        prepared_b = adapter.prepare_launch(_session_env(10002), settings_path=settings_path)

        assert prepared_a != prepared_b
        assert Path(prepared_a).exists() and Path(prepared_b).exists()

    def test_session_file_carries_only_kitty_keys(self, tmp_path: Path):
        """The user's own env entries must NOT be copied in.

        Claude Code merges a ``--settings`` env block per variable, so omitted
        keys keep their user-scope values. Cloning them would freeze a snapshot
        of the user's env at launch time.
        """
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        env_overrides = _session_env(10001)
        prepared = adapter.prepare_launch(env_overrides, settings_path=settings_path)

        assert _session_file_env(prepared) == env_overrides
        assert "CUSTOM_KEY" not in _session_file_env(prepared)

    def test_only_injected_keys_are_written(self, tmp_path: Path):
        """Keys outside _SETTINGS_ENV_OVERRIDE_KEYS are not smuggled in."""
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        env_overrides = {**_session_env(10001), "UNRELATED_VAR": "nope"}
        prepared = adapter.prepare_launch(env_overrides, settings_path=settings_path)

        assert "UNRELATED_VAR" not in _session_file_env(prepared)


class TestRoutingWithoutUserSettings:
    """AC6: a machine with no ~/.claude/settings.json must still be routed."""

    def test_session_file_written_when_global_absent(self, tmp_path: Path):
        """Previously ``prepare_launch`` returned None here and the session fell
        back to process env — which the user's settings scopes outrank."""
        settings_path = tmp_path / ".claude" / "settings.json"
        adapter = ClaudeAdapter()

        prepared = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)

        assert prepared is not None
        assert _session_file_env(prepared)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:10001"
        assert not settings_path.exists(), "kitty must not create the user's settings file"

    @pytest.mark.parametrize(
        "content",
        [
            b"{not json",
            b'["a", "list"]',
            # UTF-16 with a BOM — what Notepad's "Save as -> Unicode" writes.
            # Decoding it as UTF-8 raises UnicodeDecodeError, which is a
            # ValueError, not an OSError: it would escape both this read and
            # the orchestrator's fail-closed handler, killing the launch with
            # a raw traceback and an orphaned bridge.
            b'\xff\xfe{\x00"\x00e\x00"\x00:\x001\x00}\x00',
        ],
        ids=["invalid-json", "list-root", "utf16-bom"],
    )
    def test_malformed_global_file_does_not_block_launch(self, tmp_path: Path, content: bytes):
        """AC10: R10's read is best-effort; R7 covers the session-file write only."""
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_bytes(content)
        adapter = ClaudeAdapter()

        prepared = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)

        assert prepared is not None
        assert _session_file_env(prepared)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:10001"
        assert settings_path.read_bytes() == content


class TestCleanup:
    """AC5: per-session cleanup, idempotent, nothing global touched."""

    def test_cleanup_removes_session_file(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        prepared = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
        assert Path(prepared).exists()
        adapter.cleanup_launch(prepared, settings_path=settings_path)

        assert not Path(prepared).exists()

    def test_cleanup_is_idempotent(self, tmp_path: Path):
        """finally + atexit both call it; the second must be a no-op."""
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        prepared = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
        adapter.cleanup_launch(prepared, settings_path=settings_path)
        adapter.cleanup_launch(prepared, settings_path=settings_path)

        assert not Path(prepared).exists()

    def test_one_session_cleanup_leaves_the_other_file(self, tmp_path: Path):
        """A's exit must not disturb B's still-running session."""
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        prepared_a = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
        prepared_b = adapter.prepare_launch(_session_env(10002), settings_path=settings_path)
        adapter.cleanup_launch(prepared_a, settings_path=settings_path)

        assert not Path(prepared_a).exists()
        assert Path(prepared_b).exists()
        assert _session_file_env(prepared_b)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:10002"

    def test_prepared_value_never_reaches_the_verbatim_restore_branch(self, tmp_path: Path, _no_global_writes):
        """R9: a bare ``str`` is written as the file's ENTIRE content.

        If the prepared value were a plain path string it would land in that
        legacy branch and overwrite the user's settings.json with a path.
        """
        settings_path = tmp_path / ".claude" / "settings.json"
        original = _write_user_settings(settings_path)
        _no_global_writes(settings_path)
        adapter = ClaudeAdapter()

        prepared = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
        assert isinstance(prepared, claude_mod._SessionSettingsFile), (
            "prepared value must be a distinct type, not a bare str"
        )
        adapter.cleanup_launch(prepared, settings_path=settings_path)

        assert settings_path.read_text(encoding="utf-8") == original


class TestSettingsCliArgs:
    """AC4/AC7: how the session file reaches the child command."""

    def test_claude_returns_settings_flag(self, tmp_path: Path):
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_user_settings(settings_path)
        adapter = ClaudeAdapter()

        prepared = adapter.prepare_launch(_session_env(10001), settings_path=settings_path)

        assert adapter.settings_cli_args(prepared) == ["--settings", str(prepared)]

    def test_no_args_when_nothing_prepared(self):
        """A skipped prepare must not produce a dangling flag."""
        assert ClaudeAdapter().settings_cli_args(None) == []

    def test_base_adapter_contributes_no_args(self):
        """AC7: adapters with no per-session settings are unaffected."""

        class _Stub(LauncherAdapter):
            """Minimal adapter exercising the inherited default."""

            @property
            def name(self) -> str:
                return "stub"

            @property
            def binary_name(self) -> str:
                return "stub"

            @property
            def bridge_protocol(self):
                return None

            def build_spawn_config(self, profile, bridge_port, resolved_key, *, context_tokens=None):
                raise NotImplementedError

        assert _Stub().settings_cli_args("anything") == []


class TestStaleGlobalWarning:
    """AC10: warn about a pre-fix poisoned global file, never modify it."""

    def test_warns_when_global_carries_kitty_signature(self, tmp_path: Path, caplog):
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        poisoned = json.dumps(
            {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9999", "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token"}},
            indent=2,
        )
        settings_path.write_text(poisoned, encoding="utf-8")
        adapter = ClaudeAdapter()

        with caplog.at_level("WARNING"):
            adapter.prepare_launch(_session_env(10001), settings_path=settings_path)

        assert "kitty cleanup" in caplog.text
        assert settings_path.read_text(encoding="utf-8") == poisoned

    def test_no_warning_for_a_users_own_local_proxy(self, tmp_path: Path, caplog):
        """A loopback URL alone is not a kitty signature.

        Users running LiteLLM/Ollama locally must not be nagged toward a
        command that would strip their own keys.
        """
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4000", "ANTHROPIC_API_KEY": "mine"}}, indent=2),
            encoding="utf-8",
        )
        adapter = ClaudeAdapter()

        with caplog.at_level("WARNING"):
            adapter.prepare_launch(_session_env(10001), settings_path=settings_path)

        assert "kitty cleanup" not in caplog.text


class TestPrepareFailure:
    """AC9 (adapter half): a failed write must propagate, not pass silently."""

    def test_write_failure_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """R7: the orchestrator turns this into a fail-closed abort.

        Silently continuing would run the session on the user's own
        credentials, bypassing the bridge and its usage logging.
        """
        settings_path = tmp_path / ".claude" / "settings.json"
        _write_user_settings(settings_path)

        def _boom(*_args, **_kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr(claude_mod.tempfile, "mkstemp", _boom)
        adapter = ClaudeAdapter()

        with pytest.raises(OSError):
            adapter.prepare_launch(_session_env(10001), settings_path=settings_path)
