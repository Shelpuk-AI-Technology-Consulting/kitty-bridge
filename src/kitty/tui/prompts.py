"""TUI prompt utilities — text input, secret input, confirmation via questionary."""

from __future__ import annotations

import sys

import questionary
from prompt_toolkit import prompt as pt_prompt

__all__ = ["can_interact", "check_tty", "prompt_confirm", "prompt_secret", "prompt_text"]

# Windows standard-handle indices from the Win32 console API. Named because raw
# ``-10``/``-11`` reads like a mistake.
_STD_INPUT_HANDLE = -10
_STD_OUTPUT_HANDLE = -11


class NonTTYError(Exception):
    """Raised when a prompt is attempted in a non-TTY environment."""


def _handle_attached(raw_mode: int) -> bool:
    """Interpret a ``GetConsoleMode`` return value: nonzero means attached.

    The pure function is the decision the probe implements, and is the seam
    the unit tests patch on every leg. The ctypes call that produces
    ``raw_mode`` is Windows-only plumbing that lives inside
    :func:`_query_console_mode`.

    Args:
        raw_mode: The BOOL ``GetConsoleMode`` returned -- nonzero when the
            handle names a real console, zero on any failure (a pipe, a file,
            ``NUL``, no handle at all, a process started
            ``DETACHED_PROCESS``).

    Returns:
        ``True`` when the handle names a console the process can read keys from
        and draw on.
    """
    return bool(raw_mode)


def _query_console_mode(handle_value: int) -> int:
    """Ask the console API whether a standard handle names a real console.

    Args:
        handle_value: A standard-handle index, ``_STD_INPUT_HANDLE`` or
            ``_STD_OUTPUT_HANDLE``.

    Returns:
        The raw ``GetConsoleMode`` return value -- nonzero when the handle
        names a console, zero on any failure. Zero is also what a missing
        handle returns, so every refusal path funnels through the same
        interpretation (:func:`_handle_attached`).

    Raises:
        RuntimeError: On any platform other than Windows -- the ctypes names
            used here exist only there. The narrowing makes the rest of the
            function unreachable on the other legs at runtime and at mypy
            time, the same shape :func:`kitty.bridge.manage._probe_pid_windows`
            uses (whose comment records why).

    🔴 Windows-only: ``ctypes.WinDLL`` does not exist on POSIX, so the
    instantiation lives **inside** this function's win32-narrowed body,
    never at module scope -- hoisting it would ``AttributeError`` at import
    time and break every ``kitty.tui.prompts`` consumer on the other legs.
    """
    if sys.platform != "win32":  # pragma: no cover - guarded by can_interact
        raise RuntimeError("_query_console_mode is Windows-only")

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    # 🔴 restype MUST be declared (the trap bridge/manage.py records at its
    # own calls): ctypes defaults a return value to C `int`, truncating a
    # 64-bit HANDLE to 32 bits.
    kernel32.GetStdHandle.argtypes = (wintypes.DWORD,)
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.GetConsoleMode.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetConsoleMode.restype = wintypes.BOOL

    # ``GetConsoleMode`` writes the handle's mode flags into the out-parameter
    # and returns a BOOL -- nonzero when the handle names a console, zero on
    # any failure. The flags themselves are not consulted: the decision is
    # "is this a console at all", and the BOOL is that answer.
    mode = wintypes.DWORD()
    return int(kernel32.GetConsoleMode(kernel32.GetStdHandle(handle_value), ctypes.byref(mode)))


def can_interact() -> bool:
    """Report whether this process can host an interactive prompt or menu.

    Both streams are checked, not just stdin, and the second one is the point:
    an interactive UI must **read keys and draw**, so a console on one side is
    not enough.

    🔴 KBR-187, then KBR-204, then KBR-218. Reading ``sys.stdin.isatty()``
    alone is a POSIX reading of the question. On Windows ``isatty()`` is true
    for any *character device*, and that includes ``NUL`` -- so a child spawned
    with stdin redirected to ``NUL`` reported an interactive stdin, and died
    inside ``prompt_toolkit`` with ``NoConsoleScreenBufferError`` when it tried
    to build a Win32 screen buffer over a **piped stdout**. KBR-187 fixed the
    menus' copy of this guard; KBR-204 found :func:`check_tty` still held the
    old one, so ``kitty egress`` passed it and exited 0 when its menu declined.
    KBR-218 closes the hole the ``isatty``-AND reading cannot: on Windows
    ``NUL`` is a character device too, so ``kitty auth openai > NUL`` -- or any
    parent spawning an interactive command with ``stdin=DEVNULL,
    stdout=DEVNULL`` -- passes the guard and dies inside ``prompt_toolkit``
    with ``NoConsoleScreenBufferError``. On Windows the answer is now asked of
    the console API (:func:`_query_console_mode`, interpreted by
    :func:`_handle_attached`), which succeeds only for a real console and
    fails closed for ``NUL``, a pipe, a file, or a process started without
    one. On POSIX the same child reads ``/dev/null`` as not-a-tty, which is
    why no Linux run ever saw any of these -- and all of them are KBR-10's
    symptom, kitty misbehaving because its output is redirected, reached by
    different routes.

    ⚠️ This is the **only** answer to the question. :func:`check_tty` and
    :mod:`kitty.tui.menu` both consult it, so the paths cannot drift again.

    Returns:
        ``True`` when the process can host an interactive prompt or menu: on
        Windows, both standard handles are attached to a real console; on
        every other platform, both standard streams report ``isatty()``.
    """
    if sys.platform == "win32":
        return _handle_attached(_query_console_mode(_STD_INPUT_HANDLE)) and _handle_attached(
            _query_console_mode(_STD_OUTPUT_HANDLE)
        )
    return sys.stdin.isatty() and sys.stdout.isatty()


def check_tty() -> None:
    """Refuse to continue unless this process can interact.

    Raises:
        NonTTYError: If :func:`can_interact` is false -- on POSIX, stdin or
            stdout is not a terminal; on Windows, either standard handle is
            not attached to a console.
    """
    if not can_interact():
        raise NonTTYError("This command requires an interactive terminal (TTY)")


def prompt_text(label: str) -> str:
    """Prompt the user for text input.

    Args:
        label: The prompt label to display.

    Returns:
        The user's input string (may be empty if cancelled; callers should validate).

    Raises:
        NonTTYError: If stdin or stdout is not a TTY (see :func:`can_interact`).
    """
    check_tty()
    result = questionary.text(label).ask()
    return result if result is not None else ""


def prompt_secret(label: str) -> str:
    """Prompt the user for secret input (masked with asterisks).

    Shows one ``*`` per character so the user can see that input was accepted.

    Args:
        label: The prompt label to display.

    Returns:
        The user's secret input string.

    Raises:
        NonTTYError: If stdin or stdout is not a TTY (see :func:`can_interact`).
    """
    check_tty()
    try:
        return pt_prompt(label, is_password=True)
    except (KeyboardInterrupt, EOFError):
        return ""


def prompt_confirm(label: str, default: bool = True) -> bool:
    """Prompt the user for yes/no confirmation.

    Args:
        label: The question to display.
        default: Default value when user presses Enter without input or cancels.

    Returns:
        True for yes, False for no.

    Raises:
        NonTTYError: If stdin or stdout is not a TTY (see :func:`can_interact`).
    """
    check_tty()
    result = questionary.confirm(label, default=default).ask()
    # questionary returns None on Ctrl+C — fall back to the default
    return result if result is not None else default
