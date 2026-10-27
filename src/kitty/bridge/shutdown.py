"""Graceful-shutdown signal wiring shared by every bridge entry point."""

from __future__ import annotations

import asyncio
import contextlib
import signal


def install_stop_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    """Ask the loop to set ``stop_event`` when SIGINT or SIGTERM arrives.

    No event loop on Windows supports ``add_signal_handler``; it raises
    ``NotImplementedError``. There Ctrl+C reaches the main thread as
    ``KeyboardInterrupt`` instead, which ``asyncio.run`` turns into a task
    cancellation, so the caller's ``finally`` cleanup still runs. Doing nothing
    is therefore the correct fallback, not a silenced error.

    Args:
        loop: The running event loop.
        stop_event: Event that releases the caller's shutdown wait.
    """
    # Windows: shutdown is driven by KeyboardInterrupt instead, see above.
    with contextlib.suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGINT, stop_event.set)
        loop.add_signal_handler(signal.SIGTERM, stop_event.set)
