"""Process-level output-stream encoding, applied at every kitty entry point.

Kitty has two entry points — the ``kitty`` console script
(:func:`kitty.cli.main.main`) and the background bridge process
(:func:`kitty.bridge_runner.main`). Both must harden their streams before they
write anything, so the setting lives here rather than in either of them:
``kitty.bridge_runner`` importing ``kitty.cli`` would invert the dependency
direction that ``pyproject.toml``'s import contracts enforce everywhere else,
and would pull the whole CLI into every systemd, launchd and NSSM bridge
service. This module imports nothing but :mod:`sys` and :mod:`contextlib`, so
any layer may depend on it.
"""

from __future__ import annotations

import contextlib
import sys

__all__ = ["harden_output_streams"]


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
