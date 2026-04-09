"""Binary discovery — find coding agent CLIs on PATH and platform-specific install directories."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

__all__ = ["discover_binary"]

_WIN_EXECUTABLE_EXTS = frozenset({".exe", ".cmd", ".bat", ".com"})

_COMMON_PATHS: dict[str, dict[str, list[str]]] = {
    "codex": {
        "linux": [
            "~/.nvm/versions/node/*/bin",
            "~/.npm-global/bin",
            "/usr/local/bin",
        ],
        "darwin": [
            "~/.nvm/versions/node/*/bin",
            "~/.npm-global/bin",
            "/usr/local/bin",
        ],
        "win32": [
            "%APPDATA%\\npm",
        ],
    },
    "claude": {
        "linux": [
            "~/.local/bin",
            "~/.nvm/versions/node/*/bin",
            "~/.npm-global/bin",
            "/usr/local/bin",
        ],
        "darwin": [
            "~/.local/bin",
            "~/.nvm/versions/node/*/bin",
            "~/.npm-global/bin",
            "/usr/local/bin",
        ],
        "win32": [
            "%USERPROFILE%\\.local\\bin",
            "%APPDATA%\\npm",
        ],
    },
    "kilo": {
        "linux": [
            "~/.nvm/versions/node/*/bin",
            "~/.npm-global/bin",
            "/usr/local/bin",
        ],
        "darwin": [
            "~/.nvm/versions/node/*/bin",
            "~/.npm-global/bin",
            "/usr/local/bin",
        ],
        "win32": [
            "%APPDATA%\\npm",
        ],
    },
}


def discover_binary(name: str) -> Path | None:
    """Search for a binary by name on PATH, then in platform-specific fallback directories.

    Returns the Path to the binary if found, or None.
    """
    # 1. PATH lookup via shutil.which
    found = shutil.which(name)
    if found is not None:
        return Path(found)

    # 2. Platform-specific fallback directories
    platform = sys.platform
    binary_paths = _COMMON_PATHS.get(name, {}).get(platform, [])
    if not binary_paths:
        return None

    expanded_dirs = _expand_all_dirs(binary_paths)

    for dir_path in expanded_dirs:
        if sys.platform == "win32":
            # On Windows, check common executable extensions
            for ext in ("", ".exe", ".cmd", ".bat"):
                candidate = dir_path / (name + ext)
                if _is_executable(candidate):
                    return candidate
        else:
            candidate = dir_path / name
            if _is_executable(candidate):
                return candidate

    return None


def _expand_all_dirs(dir_strings: list[str]) -> list[Path]:
    """Expand a list of directory path strings into resolved Path objects.

    Handles tilde expansion, environment variable expansion, and nvm glob patterns.
    """
    result: list[Path] = []
    for dir_str in dir_strings:
        expanded = os.path.expandvars(os.path.expanduser(dir_str))
        if "*" in dir_str:
            # Glob pattern — expand nvm-style versioned dirs
            prefix = expanded.split("*")[0]
            result.extend(_expand_nvm_dirs(Path(prefix)))
        elif Path(expanded).is_dir():
            result.append(Path(expanded))
    return result


def _expand_nvm_dirs(nvm_base: Path) -> list[Path]:
    """Glob-expand nvm version directories to find bin dirs.

    Returns sorted list of existing directories (not files).
    The caller is responsible for searching within them for the binary.
    """
    matches = list(nvm_base.glob("*/bin"))
    return sorted(d for d in matches if d.is_dir())


def _is_executable(path: Path) -> bool:
    """Platform-aware executable check.

    POSIX: checks execute permission bit.
    Windows: checks that the path is a regular file with a known executable extension.
    """
    try:
        if not path.is_file():
            return False
    except OSError:
        return False
    if sys.platform == "win32":
        return path.suffix.lower() in _WIN_EXECUTABLE_EXTS or path.suffix == ""
    return os.access(path, os.X_OK)
