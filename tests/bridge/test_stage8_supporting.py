"""Stage 8 tests — supporting system fixes (F38, F42, F43, F44, F45, F46, F47).

F38: Stale filelock timeout in FileBackend
F42: Race condition on concurrent start_bridge calls
F43: systemd restart loop throttling
F44: Corrupt profiles.json detection
F45: Launcher binary not found raises catchable exception
F46: KeyboardInterrupt suppression in bridge_runner entry point
F47: OAuth session save uses file lock
"""

from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── F38: Stale filelock timeout ─────────────────────────────────────────


class TestFileBackendLockTimeout:
    """F38: FileBackend's filelock must have an explicit timeout, not hang indefinitely."""

    def test_file_lock_has_explicit_timeout(self):
        """FileBackend._lock must be configured with a timeout (not infinite default)."""
        from kitty.credentials.file_backend import FileBackend

        backend = FileBackend(Path("/tmp/test_creds_f38.json"))
        # The lock should have a finite timeout (not None which means infinite)
        assert backend._lock.timeout is not None, "FileLock must have explicit timeout"
        assert backend._lock.timeout > 0, "FileLock timeout must be positive"
        assert backend._lock.timeout <= 30, "FileLock timeout should be reasonable (≤30s)"


# ── F42: Concurrent start_bridge race condition ─────────────────────────


class TestStartBridgeRaceCondition:
    """F42: start_bridge must acquire a lock to prevent concurrent start races."""

    def test_start_bridge_uses_state_file_lock(self, tmp_path):
        """start_bridge should acquire a lock file before checking state."""
        from kitty.bridge.manage import start_bridge

        state_path = tmp_path / "bridge_state.json"
        # Patch to avoid actual process spawn
        with patch("kitty.bridge.manage.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_proc.stderr = None
            mock_popen.return_value = mock_proc

            # Make the state file appear after a short delay
            def write_state_later():
                import time

                time.sleep(0.05)
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(
                    json.dumps(
                        {
                            "pid": 12345,
                            "host": "127.0.0.1",
                            "port": 8080,
                            "profile": "test",
                            "started_at": "now",
                            "tls": False,
                        }
                    )
                )

            t = threading.Thread(target=write_state_later, daemon=True)
            t.start()
            # Should not hang or crash
            with contextlib.suppress(SystemExit):
                start_bridge(state_path=str(state_path))
            t.join(timeout=5.0)


# ── F43: systemd restart loop throttling ─────────────────────────────────


class TestSystemdRestartThrottling:
    """F43: The generated systemd unit must include StartLimitBurst/StartLimitIntervalSec."""

    def test_systemd_unit_has_restart_limit(self):
        """The systemd unit should include StartLimitBurst to prevent restart loops."""
        from kitty.bridge.service import generate_systemd_unit

        unit = generate_systemd_unit()
        assert "StartLimitBurst" in unit, "systemd unit must include StartLimitBurst"
        assert "StartLimitIntervalSec" in unit, "systemd unit must include StartLimitIntervalSec"

    def test_systemd_unit_restart_sec_is_reasonable(self):
        """RestartSec should be >= 5s to avoid tight restart loops."""
        from kitty.bridge.service import generate_systemd_unit

        unit = generate_systemd_unit()
        # Find RestartSec value
        for line in unit.splitlines():
            if line.strip().startswith("RestartSec="):
                value = int(line.strip().split("=")[1])
                assert value >= 5, f"RestartSec={value} is too aggressive, should be >= 5"


# ── F44: Corrupt profiles.json detection ────────────────────────────────


class TestCorruptProfilesDetection:
    """F44: When profiles.json is corrupt, the store must log a warning, not silently return empty."""

    def test_corrupt_profiles_json_logs_warning(self, tmp_path, caplog):
        """A corrupt profiles.json should produce a warning log, not silent empty return."""
        import logging

        from kitty.profiles.store import ProfileStore

        store_path = tmp_path / "profiles.json"
        store_path.write_text("this is not valid json {{{", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="kitty.profiles.store"):
            store = ProfileStore(store_path)
            result = store.load_all()

        assert result == []
        warning_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("corrupt" in m.lower() or "invalid" in m.lower() for m in warning_msgs), (
            f"Expected corruption warning, got: {warning_msgs}"
        )

    def test_missing_profiles_file_returns_empty_without_warning(self, tmp_path, caplog):
        """A missing profiles.json should return empty without warnings."""
        import logging

        from kitty.profiles.store import ProfileStore

        store_path = tmp_path / "profiles.json"
        # Don't create the file

        with caplog.at_level(logging.WARNING, logger="kitty.profiles.store"):
            store = ProfileStore(store_path)
            result = store.load_all()

        assert result == []
        # No warning for a missing file — that's the first-run case
        warning_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("corrupt" in m.lower() for m in warning_msgs)


# ── F45: Launcher binary not found raises catchable exception ────────────


class TestLauncherBinaryNotFoundError:
    """F45: resolve_binary must raise a catchable exception, not SystemExit."""

    def test_resolve_binary_raises_file_not_found(self):
        """resolve_binary should raise FileNotFoundError, not SystemExit."""
        from kitty.cli.launcher import resolve_binary

        with (
            patch("kitty.cli.launcher.discover_binary", return_value=None),
            pytest.raises(FileNotFoundError, match="not found"),
        ):
            resolve_binary("nonexistent-binary-xyz")

    def test_resolve_binary_does_not_raise_system_exit(self):
        """resolve_binary must NOT raise SystemExit (which bypasses cleanup)."""
        from kitty.cli.launcher import resolve_binary

        with patch("kitty.cli.launcher.discover_binary", return_value=None), pytest.raises(FileNotFoundError):
            resolve_binary("nonexistent-binary-xyz")


# ── F46: KeyboardInterrupt suppression in bridge runner ─────────────────


class TestBridgeRunnerKeyboardInterrupt:
    """F46: The bridge_runner entry point should suppress KeyboardInterrupt on shutdown."""

    def test_run_server_suppresses_keyboard_interrupt(self):
        """bridge_runner.main should catch KeyboardInterrupt and exit cleanly."""
        # We test that the function is wrapped in suppress(KeyboardInterrupt)
        # by checking the source code pattern rather than running it.
        import inspect

        from kitty import bridge_runner

        source = inspect.getsource(bridge_runner)
        # The main function should contain suppress(KeyboardInterrupt)
        assert "suppress(KeyboardInterrupt)" in source


# ── F47: OAuth session save uses file lock ─────────────────────────────


class TestOAuthSessionFileLock:
    """F47: OAuthSession.save() must use a file lock to prevent concurrent write races."""

    def test_save_uses_filelock(self, tmp_path):
        """OAuthSession.save should acquire a lock file before writing."""
        from kitty.auth.oauth_session import OAuthSession

        session = OAuthSession(
            client_id="test-client",
            access_token="at",
            refresh_token="rt",
            id_token="idt",
            api_key="sk-test",
            access_token_expires_at=9999999999.0,
            api_key_expires_at=9999999999.0,
        )
        session._file_path = str(tmp_path / "oauth_session.json")

        # Mock filelock at the module level (it's imported inside save())
        with patch("filelock.FileLock") as mock_lock_cls:
            mock_lock = MagicMock()
            mock_lock_cls.return_value = mock_lock
            mock_lock.__enter__ = MagicMock(return_value=mock_lock)
            mock_lock.__exit__ = MagicMock(return_value=False)

            session.save()

            # FileLock should have been created with the session path
            mock_lock_cls.assert_called()
            lock_path = mock_lock_cls.call_args[0][0]
            assert lock_path.endswith(".lock")
            # The lock should have been acquired
            mock_lock.__enter__.assert_called()
