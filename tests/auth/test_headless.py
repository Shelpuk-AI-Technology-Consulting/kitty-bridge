"""Tests for the headless-environment detection predicate.

KBR-317. The predicate decides whether the OpenAI OAuth flow defaults to the
device-code path; it mirrors the variable list of codex-rs's
``is_headless_environment`` (revision ``1fc72c6``, cited in openai/codex#9253).

The autouse fixture establishes a **desktop baseline** (``DISPLAY=:0``, no
SSH/CI vars) so each test isolates one variable's effect. The tests run on a
Linux CI box where a bare environment *is* headless, so without the baseline
the "not headless" assertions would be untestable.
"""

from __future__ import annotations

import sys

import pytest

from kitty.auth.headless import is_headless_environment


@pytest.fixture(autouse=True)
def _desktop_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a desktop baseline: a display, no CI, no SSH."""
    for var in ("CI", "SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DISPLAY", ":0")


class TestIsHeadlessEnvironment:
    def test_no_relevant_vars_set_is_not_headless(self) -> None:
        """A desktop Linux session (DISPLAY set, no SSH/CI) is not headless."""
        assert is_headless_environment() is False

    def test_ci_set_is_headless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        assert is_headless_environment() is True

    def test_ci_empty_string_is_not_headless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty value counts as unset, matching codex's env_var_set."""
        monkeypatch.setenv("CI", "")
        assert is_headless_environment() is False

    def test_ci_whitespace_only_is_not_headless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "   ")
        assert is_headless_environment() is False

    @pytest.mark.parametrize("var", ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"])
    def test_ssh_vars_set_is_headless(self, monkeypatch: pytest.MonkeyPatch, var: str) -> None:
        monkeypatch.setenv(var, "1.2.3.4 5678 5.6.7.8 9012")
        assert is_headless_environment() is True

    def test_ssh_connection_empty_is_not_headless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SSH_CONNECTION", "")
        assert is_headless_environment() is False

    def test_display_set_is_not_headless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISPLAY", ":0")
        assert is_headless_environment() is False

    def test_wayland_display_set_is_not_headless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        assert is_headless_environment() is False

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason=(
            "Asserts the Linux-only DISPLAY branch of is_headless_environment: "
            "macOS/Windows must NOT flip to headless when DISPLAY is absent"
        ),
    )
    def test_neither_display_nor_wayland_is_headless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Linux with no X or Wayland socket has no usable GUI (the CLI-only VM case)."""
        monkeypatch.delenv("DISPLAY", raising=False)
        assert is_headless_environment() is True

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason=(
            "Asserts the Linux-only DISPLAY branch of is_headless_environment: "
            "an empty DISPLAY string on Linux still counts as headless"
        ),
    )
    def test_display_empty_string_is_headless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISPLAY", "")
        assert is_headless_environment() is True

    def test_wayland_only_is_not_headless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A Wayland session without X is still a usable GUI."""
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        assert is_headless_environment() is False

    def test_bare_env_on_darwin_is_not_headless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The DISPLAY check is Linux-only: macOS has no DISPLAY but has a GUI."""
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setattr(sys, "platform", "darwin")
        assert is_headless_environment() is False

    def test_bare_env_on_windows_is_not_headless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The DISPLAY check is Linux-only; Windows never sets DISPLAY."""
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setattr(sys, "platform", "win32")
        assert is_headless_environment() is False

    def test_ssh_connection_beats_display(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SSH into a machine with a display is still treated as headless (codex posture)."""
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 5678 5.6.7.8 9012")
        assert is_headless_environment() is True


class TestSelectDeviceFlow:
    """The §14.3 selection rule lives in one function, called by both
    ``run_oauth_flow`` and the CLI's "Opening browser" gate so the two
    cannot disagree.
    """

    def test_true_forces_device(self) -> None:
        from kitty.auth.headless import select_device_flow

        assert select_device_flow(True) is True

    def test_false_forces_browser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``False`` forces browser even on a headless box — the
        ``--no-device-auth`` escape hatch used behind an SSH tunnel."""
        from kitty.auth.headless import select_device_flow

        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("DISPLAY", "")  # would otherwise read headless
        assert select_device_flow(False) is False

    def test_none_defers_to_predicate_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kitty.auth.headless import select_device_flow

        monkeypatch.setenv("CI", "true")
        assert select_device_flow(None) is True

    def test_none_defers_to_predicate_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kitty.auth.headless import select_device_flow

        monkeypatch.setenv("DISPLAY", ":0")
        assert select_device_flow(None) is False
