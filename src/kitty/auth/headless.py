"""Headless environment detection.

KBR-317. Decides whether the OpenAI OAuth flow should default to the
device-code path on a machine that has no usable GUI. Mirror of
``codex-rs/core/src/env.rs::is_headless_environment`` at codex revision
``1fc72c647fd52e3e73d4309c3b568d4d5fe012b5`` (the file has since moved on
main; the pinned revision is the one cited in openai/codex#9253).

The predicate is intentionally pure: it reads ``os.environ`` only, has no
dependency on aiohttp, asyncio, or any other module under ``kitty.auth.*``,
and is importable from any code path that needs to ask the question
(``run_oauth_flow`` here, and any future caller — e.g. tmux-wrap warnings —
without dragging in the OAuth machinery).
"""

from __future__ import annotations

import os
import sys

# Sentinel set used only in tests; the production code only consults os.environ.
_SSH_VARS: tuple[str, ...] = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")


def _env_var_set(name: str) -> bool:
    """Return True iff *name* is set in os.environ to a non-empty, non-whitespace value.

    Mirrors codex's ``env_var_set`` helper: an empty or whitespace-only value
    is treated as unset, so an explicit ``CI=`` (the form some CI runners use
    to mean "interactive") does not flip the predicate.
    """
    value = os.environ.get(name)
    return value is not None and value.strip() != ""


def is_headless_environment() -> bool:
    """Return True when the current process looks like it has no usable GUI.

    The check returns True when any of the following hold:

    - ``CI`` is set to a non-empty, non-whitespace value.
    - Any of ``SSH_CONNECTION``, ``SSH_CLIENT``, ``SSH_TTY`` is set to a
      non-empty, non-whitespace value.
    - On Linux: both ``DISPLAY`` and ``WAYLAND_DISPLAY`` are unset or empty.

    Returns False otherwise (a desktop with ``DISPLAY=:0`` and no SSH vars
    in the environment).

    Returns:
        bool: True when the device-code path should be the default.
    """
    if _env_var_set("CI"):
        return True
    for var in _SSH_VARS:
        if _env_var_set(var):
            return True
    # Linux-only branch, mirroring codex's #[cfg(target_os = "linux")]: macOS
    # has no DISPLAY by default (it uses WindowServer) but does have a GUI.
    if sys.platform.startswith("linux") and not _env_var_set("DISPLAY") and not _env_var_set("WAYLAND_DISPLAY"):
        return True
    return False


def select_device_flow(device_auth: bool | None) -> bool:
    """Resolve the explicit OpenAI login-path flag into a path decision.

    Single home of the selection rule (SYSTEM_DESIGN §14.3) so the flow
    dispatch inside :func:`kitty.auth.openai_oauth.run_oauth_flow` and the
    CLI layer's "Opening browser" prediction cannot drift apart: both call
    this one function.

    Args:
        device_auth: ``True`` forces the device-code flow, ``False`` forces
            the browser flow, ``None`` defers to
            :func:`is_headless_environment`.

    Returns:
        bool: True when the device-code path should run.
    """
    return device_auth if device_auth is not None else is_headless_environment()
