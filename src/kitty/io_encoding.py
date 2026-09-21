"""Process-level output-stream handling, applied at every kitty entry point.

Kitty has two entry points — the ``kitty`` console script
(:func:`kitty.cli.main.main`) and the background bridge process
(:func:`kitty.bridge_runner.main`). Both must harden their streams before they
write anything, so the setting lives here rather than in either of them:
``kitty.bridge_runner`` importing ``kitty.cli`` would invert the dependency
direction that ``pyproject.toml``'s import contracts enforce everywhere else,
and would pull the whole CLI into every systemd, launchd and NSSM bridge
service. This module imports nothing but the standard library, so any layer
may depend on it.
"""

from __future__ import annotations

import contextlib
import os
import stat
import sys

__all__ = ["harden_output_streams", "relinquish_output_streams"]


def harden_output_streams() -> None:
    """Encode ``stdout`` and ``stderr`` as UTF-8 regardless of the machine's locale.

    On Windows, Python encodes a **non-console** ``stdout`` with the locale
    codepage — ``cp1252``, ``cp437``, ``cp932`` — and ships it with
    ``errors='strict'``. Piping, redirecting to a file, or being captured by a
    parent process is enough to select that path, so any character the codepage
    cannot represent killed the process with :exc:`UnicodeEncodeError`. That
    made ``kitty doctor > diagnostics.txt`` — the command the README tells a
    user to run when something is wrong — the invocation that crashed (KBR-10).

    The text at risk is not only kitty's own status glyphs: a Russian, Japanese
    or Chinese Windows install has a non-ASCII ``C:\\Users\\<name>``, and kitty
    prints paths.

    ``errors`` is passed explicitly and never left to default.
    :meth:`io.TextIOWrapper.reconfigure` silently resets it to ``'strict'``
    whenever ``encoding`` is given, which would *downgrade* ``stderr`` — Python
    ships that stream as ``'backslashreplace'`` — and UTF-8 with ``'strict'``
    still raises on a lone surrogate, the form :func:`os.fsdecode` produces from
    an undecodable filename. Only an explicit non-strict handler makes "kitty
    cannot die encoding its own output" true.

    A stream that cannot be reconfigured is left exactly as it was: under
    ``pythonw.exe`` ``sys.stdout`` is ``None``, a test may have substituted an
    :class:`io.StringIO`, and a closed or detached stream raises. Hardening
    output must never itself become the thing that fails.
    """
    for name in ("stdout", "stderr"):
        # getattr twice rather than a hasattr check: `sys.stdout` is `None`
        # under pythonw.exe, and a substituted stream may be any file-like
        # object, including one with no reconfigure at all. One expression
        # covers both.
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue

        # A closed or detached stream raises ValueError. That is a stream we
        # cannot help, not a reason to abort the command the user asked for.
        with contextlib.suppress(ValueError):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def relinquish_output_streams() -> None:
    """Point pipe-shaped ``stdout`` and ``stderr`` at ``os.devnull``.

    The background bridge child is spawned with ``stdout=PIPE`` and drained
    only while ``kitty bridge start``'s parent is still waiting (KBR-176).
    Once that parent gives up and exits, the pipe's read end closes, and
    every later write the child makes raises :exc:`BrokenPipeError` — the
    defect that killed healthy bridges before they reported ready (KBR-219).
    After the bridge has reported ready there is nothing left to say to a
    parent that may no longer exist, so the child moves its streams to
    ``os.devnull``: writes land nowhere and never raise.

    Only **pipes** are redirected. Service managers hand the bridge other
    things — systemd's journal (an ``AF_UNIX`` socket), launchd a file or
    ``/dev/null``, NSSM a file — and replacing those would swallow a
    deployment's log stream. A closed or unstatable fd is skipped for the
    same reason :func:`harden_output_streams` never fails: relinquishing
    output must not itself become the thing that breaks the bridge.

    A second call is a no-op by construction: after the first, the fds name
    a character device, not a pipe, so the guard skips them.
    """
    # Opened lazily: on a child whose streams are already non-pipes, no
    # devnull fd is created at all.
    devnull: int | None = None
    try:
        for fd in (1, 2):
            # A closed or invalid fd raises here; skipping it is the
            # contract, not an error to report.
            try:
                is_pipe = stat.S_ISFIFO(os.fstat(fd).st_mode)
            except OSError:
                continue
            if not is_pipe:
                continue
            if devnull is None:
                devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, fd)
    finally:
        if devnull is not None:
            os.close(devnull)
