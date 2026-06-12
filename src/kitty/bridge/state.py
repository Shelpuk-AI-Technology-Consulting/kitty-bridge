"""Bridge state file management."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BridgeState:
    """Runtime state of a running bridge instance."""

    pid: int
    host: str
    port: int
    profile: str
    started_at: str
    tls: bool


def write_state(path: Path | str, state: BridgeState) -> None:
    """Write bridge state to a JSON file atomically (F40).

    Uses temp file + ``os.replace()`` so a crash mid-write never leaves
    a corrupt file.  The ``os.replace()`` call is atomic on POSIX.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "pid": state.pid,
        "host": state.host,
        "port": state.port,
        "profile": state.profile,
        "started_at": state.started_at,
        "tls": state.tls,
    }
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=path.stem + ".",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2) + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def load_state(path: Path | str) -> BridgeState | None:
    """Load bridge state from a JSON file. Returns None if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return BridgeState(
            pid=data["pid"],
            host=data["host"],
            port=data["port"],
            profile=data["profile"],
            started_at=data["started_at"],
            tls=data["tls"],
        )
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Failed to parse bridge state file %s: %s", path, exc)
        return None


def remove_state(path: Path | str) -> None:
    """Remove the bridge state file. No-op if file doesn't exist."""
    path = Path(path)
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
