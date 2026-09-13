"""Tests for :mod:`kitty.bridge.stop_signals`, the stop-signal registration every bridge loop uses.

The Windows event loop does not implement ``add_signal_handler``: CPython's base
``AbstractEventLoop`` raises :exc:`NotImplementedError` and only the Unix loop
overrides it. A bridge that called it directly wrote its state file and then died
on Windows. These tests hand the helper a loop of each kind.
"""

from __future__ import annotations

import signal

from kitty.bridge.stop_signals import install_stop_handlers


class _UnixLikeLoop:
    """Record every handler registered, as a loop with signal support would."""

    def __init__(self) -> None:
        """Start with no handlers registered."""
        self.handlers: dict[int, object] = {}

    def add_signal_handler(self, sig: int, callback: object) -> None:
        """Record ``callback`` as the handler for ``sig``.

        Args:
            sig: The signal number.
            callback: The handler.
        """
        self.handlers[sig] = callback


class _WindowsLikeLoop:
    """Refuse signal handlers the way CPython's base event loop does."""

    def add_signal_handler(self, sig: int, callback: object) -> None:
        """Raise, as ``asyncio.AbstractEventLoop.add_signal_handler`` does.

        Args:
            sig: The signal number.
            callback: The handler.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError


def _stop() -> None:
    """Stand in for the stop callback."""


def test_install_stop_handlers_registers_interrupt_and_terminate_where_supported() -> None:
    """Both stop signals reach the callback on a loop that supports handlers."""
    loop = _UnixLikeLoop()

    assert install_stop_handlers(loop, _stop) is True
    assert loop.handlers == {signal.SIGINT: _stop, signal.SIGTERM: _stop}


def test_install_stop_handlers_declines_instead_of_raising_where_unsupported() -> None:
    """A loop without handler support is reported, not allowed to crash the bridge."""
    assert install_stop_handlers(_WindowsLikeLoop(), _stop) is False
