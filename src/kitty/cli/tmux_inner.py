"""Entry point for the kitty that runs inside a tmux session started by ``--tmux``.

:mod:`kitty.cli.tmux_wrap` starts ``python -m kitty.cli.tmux_inner <kitty args>``
in a new tmux session. A running tmux server gives that process the server's
environment, not the user's, so the outer kitty writes its own environment to a
private file. This module loads that file before anything else from kitty is
imported, then runs :func:`kitty.cli.main.main`. If kitty fails, it holds the
pane open so the error stays readable. Reasons: ``.system_design/SYSTEM_DESIGN.md``
§3.3, decisions T6, T7, T9 and T12.

Only the standard library is imported at module level: modules elsewhere in
kitty read the environment when imported (``Path.home()``), so importing them
before the swap would bake in the tmux server's values.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from collections.abc import Callable, MutableMapping, Sequence
from typing import TextIO

ENV_FILE_VARIABLE = "KITTY_TMUX_ENV_FILE"

# Set by tmux for this pane (or by the outer kitty for this session); the file holds the outer terminal's.
TMUX_OWNED_VARIABLES = ("TMUX", "TMUX_PANE", "TERM", "TERM_PROGRAM", "TERM_PROGRAM_VERSION", "KITTY_TMUX_WRAPPED")


class EnvironmentFileError(Exception):
    """The environment file named by ``KITTY_TMUX_ENV_FILE`` is missing or unusable."""


def load_environment(environ: MutableMapping[str, str]) -> None:
    """Replace ``environ`` with the outer kitty's environment and delete the file.

    Args:
        environ: The mapping to replace, normally :data:`os.environ`. The
            variables in :data:`TMUX_OWNED_VARIABLES` keep their current value,
            or stay unset.

    Raises:
        EnvironmentFileError: When no file is named, it cannot be read, or it
            does not hold a JSON object of strings.
    """
    path = environ.get(ENV_FILE_VARIABLE)
    if not path:
        raise EnvironmentFileError(f"{ENV_FILE_VARIABLE} is not set")
    try:
        with open(path, encoding="ascii") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError) as exc:
        raise EnvironmentFileError(f"cannot read the environment file: {exc}") from exc
    finally:
        # The file may hold API keys; remove it whether or not it parsed.
        with contextlib.suppress(OSError):
            os.remove(path)
    if not isinstance(loaded, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in loaded.items()):
        raise EnvironmentFileError("the environment file does not hold a JSON object of strings")

    kept = {name: environ[name] for name in TMUX_OWNED_VARIABLES if name in environ}
    replacement = {name: value for name, value in loaded.items() if name not in TMUX_OWNED_VARIABLES}
    replacement.update(kept)
    environ.clear()
    environ.update(replacement)


def run(
    argv: Sequence[str],
    *,
    environ: MutableMapping[str, str],
    main: Callable[[], object] | None = None,
    stdin: TextIO,
    stderr: TextIO,
) -> int:
    """Load the environment, run kitty, and hold the pane if kitty fails.

    Args:
        argv: Kitty's arguments, without the program name.
        environ: The environment to replace, normally :data:`os.environ`.
        main: Kitty's entry point; ``None`` imports :func:`kitty.cli.main.main`
            after the environment is in place.
        stdin: Where the hold prompt waits for Enter.
        stderr: Where errors and the prompt are written.

    Returns:
        Kitty's exit code: ``0`` on success, its ``sys.exit`` code, or ``1``
        for an exception or an unusable environment file.
    """
    code = 1
    try:
        load_environment(environ)
        if main is None:
            from kitty.cli.main import main as kitty_main

            main = kitty_main
        sys.argv = ["kitty", *argv]
        main()
        code = 0
    except SystemExit as exc:
        code = _exit_code(exc.code, stderr)
    except EnvironmentFileError as exc:
        stderr.write(f"Error: {exc}\n")
    except Exception:
        traceback.print_exc(file=stderr)

    # Without the hold, the session would close with the pane and take the error with it (T9).
    if code != 0:
        stderr.write(f"kitty exited with code {code} - press Enter to close\n")
        stderr.flush()
        stdin.readline()
    return code


def _exit_code(value: object, stderr: TextIO) -> int:
    """Turn a ``SystemExit`` payload into a process exit code, as the interpreter would.

    Args:
        value: ``SystemExit.code``.
        stderr: Where a message payload is written.

    Returns:
        ``0`` for ``None``, the integer itself, or ``1`` for a message.
    """
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    stderr.write(f"{value}\n")
    return 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:], environ=os.environ, stdin=sys.stdin, stderr=sys.stderr))
