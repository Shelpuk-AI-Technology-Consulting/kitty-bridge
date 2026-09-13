"""Stop-signal registration for every loop that runs a bridge until it is told to stop.

The foreground ``kitty bridge`` (:mod:`kitty.cli.main`) and the background bridge
(:func:`kitty.bridge_runner.main`) both wait on an event that a stop signal sets.
``loop.add_signal_handler`` is Unix-only: CPython's base event loop raises
:exc:`NotImplementedError`, and the Windows loop does not override it. Called
directly, it killed the background bridge on Windows moments after it reported
ready.

It lives in :mod:`kitty.bridge` because both callers may already import that
package, so it needs no import contract of its own.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable


def install_stop_handlers(loop: asyncio.AbstractEventLoop, callback: Callable[[], object]) -> bool:
    """Run ``callback`` on SIGINT and SIGTERM, where the event loop supports it.

    Where it does not (Windows), nothing is registered and nothing is raised.
    Ctrl+C still raises :exc:`KeyboardInterrupt`, which :func:`asyncio.run`
    turns into cancellation of the waiting coroutine, and ``kitty bridge stop``
    ends a Windows process outright. So the caller's shutdown path still runs
    wherever a handler could have run.

    Args:
        loop: The running event loop.
        callback: Called with no arguments when a stop signal arrives.

    Returns:
        ``True`` when both handlers were registered, ``False`` when the loop
        does not support signal handlers.
    """
    try:
        loop.add_signal_handler(signal.SIGINT, callback)
        loop.add_signal_handler(signal.SIGTERM, callback)
    except NotImplementedError:
        return False
    return True
