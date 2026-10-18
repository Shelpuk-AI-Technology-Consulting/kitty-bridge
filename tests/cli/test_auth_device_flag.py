"""Tests for ``kitty auth openai --device-auth`` / ``--no-device-auth``.

KBR-317 step 4. The flags are parsed inside ``_run_auth`` in ``main.py`` and
forwarded as a ``device_auth`` keyword through ``run_auth_openai`` to
``run_oauth_for_provider`` to ``run_oauth_flow``. This file pins the
argparse layer; the downstream threading is covered by the unit tests in
``test_auth_cmd.py`` and ``test_openai_oauth_dispatch.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from kitty.cli.main import _run_auth


@pytest.fixture
def patched_run(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Stub the downstream flow so the test exercises only the flag routing.

    The auth command is gated by ``_tty_required``; the fixture replaces it
    with a no-op so the test can run without a real terminal.
    """
    run_mock = AsyncMock()
    monkeypatch.setattr("kitty.cli.main._tty_required", lambda _name: _noop_ctx())
    monkeypatch.setattr("kitty.cli.auth_cmd.run_auth_openai", run_mock)
    return run_mock


class _noop_ctx:
    """Empty context manager that replaces ``_tty_required`` (sync protocol)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def _stores() -> tuple[object, object]:
    """The two opaque store arguments; _run_auth doesn't touch them."""
    return object(), object()


class TestAuthOpenaiFlags:
    def test_no_flag_forwards_none(self, patched_run: AsyncMock) -> None:
        _run_auth(*_stores(), ["openai"])
        _check_kwarg(patched_run, "device_auth", None)

    def test_device_auth_forwards_true(self, patched_run: AsyncMock) -> None:
        _run_auth(*_stores(), ["openai", "--device-auth"])
        _check_kwarg(patched_run, "device_auth", True)

    def test_no_device_auth_forwards_false(self, patched_run: AsyncMock) -> None:
        _run_auth(*_stores(), ["openai", "--no-device-auth"])
        _check_kwarg(patched_run, "device_auth", False)

    def test_mutually_exclusive_flags_systemexit(self, patched_run: AsyncMock) -> None:
        """argparse rejects --device-auth + --no-device-auth with SystemExit(2)."""
        with pytest.raises(SystemExit) as exc_info:
            _run_auth(*_stores(), ["openai", "--device-auth", "--no-device-auth"])
        assert exc_info.value.code == 2
        patched_run.assert_not_awaited()

    def test_unknown_provider_token_still_errors(
        self, patched_run: AsyncMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run_auth(*_stores(), ["unknown-provider"])
        assert exc_info.value.code == 1
        patched_run.assert_not_awaited()
        assert "Unknown auth provider" in capsys.readouterr().out

    def test_help_prints_and_exits(self, patched_run: AsyncMock) -> None:
        """``--help`` after ``openai`` prints the auth-command help and exits 0."""
        with pytest.raises(SystemExit) as exc_info:
            _run_auth(*_stores(), ["openai", "--help"])
        assert exc_info.value.code == 0
        patched_run.assert_not_awaited()


def _check_kwarg(mock: AsyncMock, name: str, value: object) -> None:
    """Assert that the last call to *mock* passed *value* for the *name* kwarg.

    Keeps the assertion off the positional store/provider args (whose types
    are opaque) and focused on the device_auth flag — the unit under test.
    """
    assert mock.await_count == 1, "run_auth_openai must be called exactly once"
    kwargs = mock.await_args.kwargs
    assert name in kwargs, f"missing kwarg {name!r} in call"
    assert kwargs[name] is value or kwargs[name] == value, (
        f"kwarg {name}: expected {value!r}, got {kwargs[name]!r}"
    )
