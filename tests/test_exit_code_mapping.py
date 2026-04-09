"""Tests for exit code mapping — child process exit codes to kitty exit codes."""

from __future__ import annotations

import signal
import sys

import pytest

from kitty.cli.launcher import map_child_exit_code


class TestExitCodeMapping:
    def test_child_exit_0_maps_to_0(self) -> None:
        assert map_child_exit_code(0) == 0

    def test_child_exit_1_maps_to_1(self) -> None:
        assert map_child_exit_code(1) == 1

    def test_child_exit_2_maps_to_2(self) -> None:
        assert map_child_exit_code(2) == 2

    def test_child_sigterm_maps_to_143(self) -> None:
        # SIGTERM = 15 -> 128 + 15 = 143
        sigterm = signal.SIGTERM
        assert map_child_exit_code(-sigterm) == 128 + sigterm

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL not available on Windows")
    def test_child_sigkill_maps_to_137(self) -> None:
        # SIGKILL = 9 -> 128 + 9 = 137
        assert map_child_exit_code(-signal.SIGKILL) == 128 + signal.SIGKILL

    def test_child_exit_42_passes_through(self) -> None:
        assert map_child_exit_code(42) == 42

    def test_child_sigint_maps_to_130(self) -> None:
        # SIGINT = 2 -> 128 + 2 = 130
        assert map_child_exit_code(-signal.SIGINT) == 128 + signal.SIGINT

    def test_child_negative_one_maps_to_signal_base(self) -> None:
        # -1 signal number -> 128 + 1 = 129
        assert map_child_exit_code(-1) == 129

    def test_child_exit_255_passes_through(self) -> None:
        assert map_child_exit_code(255) == 255
