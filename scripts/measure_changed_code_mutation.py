#!/usr/bin/env python3
"""Measure a changed-code mutation run against the fast gate's budget (T-H4 / KBR-92).

Maps a git rev-range onto the §6.1 mutation-scope registry
(``tests/mutmut_scope.py``) and derives the mutmut fnmatch patterns for exactly
the functions the range touches — so ``mutmut run`` can be scoped to a
changed-code subset instead of the nightly's full scope. ``TEST_SUITE.md`` §11
Q11 asks whether such a run fits the per-PR gate; this script is how the
number is produced — and, if the answer is yes, how a per-PR mutation job
would compute its own scope.

The pure mapping (diff → touched defs → registry-covered defs → function-level
patterns) is separated from the I/O edges (git, mutmut) so the unit tests in
``tests/test_measure_changed_code_mutation.py`` drive it without subprocesses.
The runner enforces a wall-clock budget because the design knows two
unbounded-stall failure modes (the ``ep_poll`` clean-test hang and the
KBR-266 CLOSE_WAIT stall) and an over-budget abort IS the Q11 answer
"doesn't fit per-PR" rather than a crash.

Usage::

    # dry-run: print touched modules/defs, patterns, and the exact command
    python scripts/measure_changed_code_mutation.py 30a91a0^1..30a91a0

    # run for real, teeing mutmut's output to a log, JSON summary on stdout
    python scripts/measure_changed_code_mutation.py 30a91a0^1..30a91a0 --run
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

# --- Mutmut positional-pattern CLI contract ---------------------------------
# `mutmut run` takes fnmatch patterns positionally; zero positional patterns
# means the FULL `only_mutate` scope — i.e. the nightly run, the exact
# opposite of a changed-code run. The empty-intersection refusal below
# guarantees the per-PR variant can never silently fall through to that.
_MUTMUT = "mutmut"

# --- Wall-clock cap defaults ----------------------------------------------
# The per-PR gate's default cap is the CI job's `timeout-minutes: 60` (see
# `.github/workflows/tests.yml`); runs that exceed it cannot be a per-PR
# gate, since GitHub would kill the job before reporting. Callers running a
# one-off measurement (where the true number is wanted, not the cap) pass a
# larger value explicitly — see `scripts/measure_changed_code_mutation.py
# --help`.
_DEFAULT_BUDGET_SECONDS = 3600.0
_DEFAULT_GRACE_SECONDS = 10.0

# Poll cadence for the budget loop. Short enough to honour sub-second
# budgets in tests; coarse enough that a 60-minute cap does ~36k cheap
# `poll()` calls (negligible overhead).
_POLL_SECONDS = 0.1

# `signal.SIGKILL` is missing from Python's `signal` module on Windows
# (Windows processes don't expose it). The script is documented as
# Linux/WSL-only (mutmut itself is Linux), but a numeric fallback keeps
# importing the module — and the test fakes — portable. POSIX SIGKILL
# is signal 9.
_KILL_SIGNAL: int = getattr(signal, "SIGKILL", 9)

# Default wall-clock cap for one L1 `pytest` invocation, used by the
# separately-timed clean-test command in the MUTATION_BASELINE.md breakdown.
_DEFAULT_CLEAN_TEST_SECONDS = 1500.0


# --------------------------------------------------------------------------
# Public exceptions
# --------------------------------------------------------------------------


class EmptyIntersectionError(RuntimeError):
    """The rev-range touches no §6.1-scope function; a scoped run is impossible.

    Distinguishing this from a generic error lets the caller (and `main`)
    emit an actionable message: "no §6.1-scope function was touched", rather
    than the generic "mutmut refused to run".
    """


# --------------------------------------------------------------------------
# Pure mapping — diff → module, diff → touched defs
# --------------------------------------------------------------------------


def module_for(repo_relative_path: str) -> str | None:
    """Map a repo-relative source path to its dotted module name.

    Args:
        repo_relative_path: Path relative to the repository root, POSIX or
            Windows form (``"src\\kitty\\bridge\\server.py"`` is accepted).

    Returns:
        The dotted module name (``"src/kitty/bridge/server.py"`` →
        ``"kitty.bridge.server"``), or ``None`` for paths outside
        ``src/kitty/`` or without a ``.py`` suffix. A package
        ``__init__.py`` maps to the package itself
        (``"src/kitty/providers/__init__.py"`` → ``"kitty.providers"``).

    Notes:
        Pure and total. The reverse mapping (module → path) belongs to
        ``pyproject.toml``'s ``[tool.mutmut] only_mutate``; this function is
        the inverse for *files only*.
    """
    normalised = repo_relative_path.replace("\\", "/")
    if not normalised.startswith("src/kitty/") or not normalised.endswith(".py"):
        return None
    parts = normalised[len("src/") : -len(".py")].split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def enclosing_defs(
    source: str, touched_lines: set[int]
) -> set[tuple[str | None, str]]:
    """Return the (class, function) pairs whose bodies overlap touched lines.

    Args:
        source: The post-image source text of a single file.
        touched_lines: Line numbers (1-based) touched by the diff.

    Returns:
        The set of ``(class_or_None, function_name)`` pairs whose body spans
        include at least one touched line. Top-level functions map to
        ``(None, name)``; class methods to ``(class_name, method_name)``.
        Nested defs are NOT enumerated — mutmut's mangler produces mutant
        keys only at module and class-method depth, so a change inside a
        nested def belongs to the enclosing top-level fn or class method
        (its body span covers the nested def's lines).

    Notes:
        The body span starts at the first decorator's line if any, and ends
        at ``end_lineno``. A module-level constant change touches no def
        (its lines are outside every span).
    """
    tree = ast.parse(source)
    spans: list[tuple[str | None, str, int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min(
                (d.lineno for d in node.decorator_list), default=node.lineno
            )
            end = getattr(node, "end_lineno", None) or node.lineno
            spans.append((None, node.name, start, end))
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = min(
                        (d.lineno for d in sub.decorator_list),
                        default=sub.lineno,
                    )
                    end = getattr(sub, "end_lineno", None) or sub.lineno
                    spans.append((node.name, sub.name, start, end))
    hits: set[tuple[str | None, str]] = set()
    for cls, fn, lo, hi in spans:
        if any(lo <= line <= hi for line in touched_lines):
            hits.add((cls, fn))
    return hits


# --------------------------------------------------------------------------
# Diff metadata — the seam between git subprocess and the pure mapping
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffMeta:
    """A diff, distilled into the minimum the registry intersection needs.

    Attributes:
        rev_range: The git rev-range this metadata was computed from (kept
            for diagnostics and the JSON summary).
        touched_modules: Dotted module names whose files appear in the diff.
        touched_defs: Mapping from module name to the (class, function) pairs
            whose bodies overlap the diff in that module. Empty frozenset for
            files whose hunks touch no def (module-level constants, comments).
    """

    rev_range: str
    touched_modules: frozenset[str]
    touched_defs: Mapping[str, frozenset[tuple[str | None, str]]] = field(
        default_factory=dict
    )


def _hunk_post_lines(rev_range: str, path: str, repo_root: str) -> set[int]:
    """Post-image line numbers touched by unified-0 diff hunks."""
    proc = subprocess.run(
        ["git", "diff", "--unified=0", rev_range, "--", path],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",  # cp1252 (Windows default) mojibakes non-ASCII
        check=True,
    )
    lines: set[int] = set()
    for ln in proc.stdout.splitlines():
        if not ln.startswith("@@"):
            continue
        # @@ -a,b +c,d @@  →  post span c..c+d-1 (single line if d omitted).
        post = ln.split("+", 1)[1].split("@@", 1)[0]
        start_s, _, count_s = post.partition(",")
        start = int(start_s)
        count = int(count_s) if count_s else 1
        if count > 0 and start > 0:
            lines.update(range(start, start + count))
    return lines


def _post_blob(rev: str, path: str, repo_root: str) -> str | None:
    """Return the post-image blob for ``path`` at ``rev``, or None if deleted."""
    proc = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",  # cp1252 (Windows default) mojibakes non-ASCII
    )
    if proc.returncode != 0 or proc.stdout is None:
        return None
    return proc.stdout


def diff_meta_from_git_range(rev_range: str, repo_root: str) -> DiffMeta:
    """Compute the diff metadata for ``rev_range`` against ``repo_root``.

    Args:
        rev_range: A git rev-range (``"A..B"`` or ``"A...B"``); the
            post-image is the right-hand side, from which blobs are read.
        repo_root: Absolute path to the repository root.

    Returns:
        A ``DiffMeta`` covering every changed ``src/kitty/`` file: which
        modules changed and which (class, function) bodies overlapped the
        diff. Files deleted in the range contribute no defs.

    Notes:
        The mapping is intentionally narrow: a touched file whose hunks
        fall outside any def (engine.py's data-table edit in KBR-285) is
        recorded in ``touched_modules`` but contributes nothing to
        ``touched_defs``, and therefore emits zero mutation patterns.
    """
    proc = subprocess.run(
        ["git", "diff", "--name-only", rev_range, "--", "src/kitty/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",  # cp1252 (Windows default) mojibakes server.py
        check=True,
    )
    # "A..B" splits on ".." to ['A', 'B']; "A...B" splits to ['A', '.B']
    # (because "..." = ".." + "."). Split on the run of 2–3 dots so both
    # forms resolve to the post-image revision correctly.
    post = re.split(r"\.{2,3}", rev_range)[-1]
    modules: set[str] = set()
    defs: dict[str, set[tuple[str | None, str]]] = {}
    for path in (ln.strip() for ln in proc.stdout.splitlines()):
        if not path:
            continue
        module = module_for(path)
        if module is None:
            continue
        modules.add(module)
        source = _post_blob(post, path, repo_root)
        if source is None:
            defs[module] = set()
            continue
        lines = _hunk_post_lines(rev_range, path, repo_root)
        defs[module] = set(enclosing_defs(source, lines))
    return DiffMeta(
        rev_range=rev_range,
        touched_modules=frozenset(modules),
        touched_defs={m: frozenset(s) for m, s in defs.items()},
    )


# --------------------------------------------------------------------------
# Registry intersection + pattern derivation
# --------------------------------------------------------------------------


# The registry's `Target` is duplicated locally to avoid an import-cycle
# hazard between `scripts/` and `tests/`; the shape is stable (KBR-88).
@dataclass(frozen=True)
class Target:
    """A scope row, in the shape ``mutmut_scope.Target`` carries.

    Attributes:
        module: Dotted module name, or ``"*"`` for cross-module rows.
        cls: Containing class name, ``None`` for top-level functions or
            whole-module scope, ``"*"`` for cross-module rows.
        function_or_method: Function or method name, ``None`` for whole
            module / whole class.
    """

    module: str
    cls: str | None
    function_or_method: str | None


# `mutmut_scope` separator is U+01C1, Latin "lateral click", doubled to keep
# the mangled name visually distinct from a real dotted path. Mirroring the
# registry module is the only honest way to derive the pattern without an
# import — the same constant lives in ``tests/mutmut_scope.py``.
_CLS = "ǁ"


def _mangled_patterns(target: Target) -> list[str]:
    """fnmatch pattern(s) for the synthetic per-def ``target``.

    The whole-module path (``Target(module, None, None)``) is NOT supported
    here — changed-code patterns are function-level by design (Q11 verbatim);
    a per-PR gate emits one pattern per touched def regardless of whether
    the registry row is whole-module or specific-method.
    """
    if target.function_or_method is None:
        raise ValueError(
            "changed-code patterns are function-level; "
            "whole-module rows are expansion sources, not patterns"
        )
    if target.cls is None:
        # Top-level function. Mangler prefixes `x_`; if the original name
        # starts with `_` the mangled key carries a double underscore.
        return [f"{target.module}.x_{target.function_or_method}__mutmut_*"]
    return [
        f"{target.module}.x{_CLS}{target.cls}{_CLS}{target.function_or_method}__mutmut_*"
    ]


def targets_covered(
    diff: DiffMeta,
    registry: Mapping[str, Sequence[Target]],
    registered_modules: frozenset[str] = frozenset(),
) -> list[tuple[str, Target]]:
    """Intersect the touched defs with the registry, expanding whole-module rows.

    Args:
        diff: The diff metadata.
        registry: Mapping from group name to its list of ``Target`` rows.
            Either the real ``mutmut_scope.TARGET_GROUPS`` or a test fixture.
        registered_modules: Dotted module names listed in ``pyproject.toml``'s
            ``[tool.mutmut] only_mutate``. Used only by cross-module rows
            (``module == "*"``) to filter which touched modules count;
            non-cross rows ignore it.

    Returns:
        A list of ``(group, Target)`` pairs, one per covered def, in
        declaration order. The ``Target`` is a SYNTHETIC per-def row
        (``Target(module, cls, fn)``), never the registry's whole-module
        row — patterns derive from the synthetic rows.
    """
    covered: list[tuple[str, Target]] = []
    seen_defs: set[tuple[str, str | None, str]] = set()

    def _add(group: str, module: str, cls: str | None, fn: str) -> None:
        key = (module, cls, fn)
        if key in seen_defs:
            return
        seen_defs.add(key)
        covered.append((group, Target(module, cls, fn)))

    for group, rows in registry.items():
        for row in rows:
            if row.module == "*":
                # Cross-module (provider_hooks) row. Covers `fn` in any
                # registered module that has a touched def with that name
                # (any class).
                if row.function_or_method is None or row.cls is None:
                    continue
                for mod, defs in diff.touched_defs.items():
                    if not _module_matches_registered(mod, registered_modules):
                        continue
                    for cls, fn in defs:
                        if fn == row.function_or_method:
                            _add(group, mod, cls, fn)
                continue

            # Non-cross rows. The registry names a package or a file module
            # (`kitty.bridge.messages`); the touched def's module is the
            # FILE's module (`kitty.bridge.messages.translator`). Match by
            # prefix — the same semantics mutmut's own `module.*` whole-
            # module pattern carries, where `*` spans dots.
            for mod, module_defs in diff.touched_defs.items():
                if mod != row.module and not mod.startswith(row.module + "."):
                    continue
                if row.function_or_method is None:
                    # Whole-module row. Covers every touched def in the module.
                    for cls, fn in module_defs:
                        _add(group, mod, cls, fn)
                    continue
                # Specific class/method row. Covers exactly that def.
                if row.cls is None:
                    target_key: tuple[str | None, str] = (
                        None,
                        row.function_or_method,
                    )
                else:
                    target_key = (row.cls, row.function_or_method)
                if target_key in module_defs:
                    _add(group, mod, target_key[0], target_key[1])
    return covered


def _module_matches_registered(
    module: str, registered_prefixes: frozenset[str]
) -> bool:
    """Whether ``module`` is covered by any prefix in ``registered_prefixes``.

    A prefix covers its exact name and any submodule (``kitty.providers``
    covers ``kitty.providers.openai``), because ``only_mutate`` uses
    package globs like ``"src/kitty/providers/*"`` which expand to every
    submodule.
    """
    return any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in registered_prefixes
    )


def patterns_for_covered(covered: list[tuple[str, Target]]) -> list[str]:
    """Flatten the covered defs into their function-level mutmut patterns.

    Dedupes while preserving first-seen order, so the final pattern list is
    deterministic and matches what ``mutmut run`` will receive positionally.
    """
    seen: set[str] = set()
    out: list[str] = []
    for _, target in covered:
        for pattern in _mangled_patterns(target):
            if pattern not in seen:
                seen.add(pattern)
                out.append(pattern)
    return out


def scoped_patterns(
    diff: DiffMeta,
    registry: Mapping[str, Sequence[Target]],
    registered_modules: frozenset[str],
) -> list[str]:
    """Derive the changed-code mutmut pattern list for ``diff``.

    Args:
        diff: The diff metadata.
        registry: The §6.1 scope registry.
        registered_modules: ``only_mutate`` modules from ``pyproject.toml``.

    Returns:
        The list of function-level mutmut fnmatch patterns to pass to
        ``mutmut run``. The list is empty iff the diff touches no
        §6.1-scope function.

    Raises:
        EmptyIntersectionError: If the covered set is empty AND the caller
            intends to run (this function is split from `main` precisely so
            the CLI can decide what to do; the pure helper raises so tests
            can pin the failure mode without re-implementing the CLI).
    """
    covered = targets_covered(diff, registry, registered_modules)
    if not covered:
        raise EmptyIntersectionError(
            "no §6.1-scope function was touched in the rev-range; "
            "a changed-code `mutmut run` would fall through to the full "
            "scope, which is exactly what the measurement must avoid"
        )
    return patterns_for_covered(covered)


# --------------------------------------------------------------------------
# Runner — subprocess with wall-clock cap and process-group teardown
# --------------------------------------------------------------------------


class _ProcLike(Protocol):
    """Minimal subprocess contract the runner needs.

    ``subprocess.Popen`` satisfies this structurally (wrapped by
    ``_PopenWrapper`` for group-kill semantics); the test fake implements it
    directly so no test spawns a real process.
    """

    def poll(self) -> int | None:
        """Return the exit code if the process has exited, else ``None``."""
        ...

    @property
    def returncode(self) -> int | None:
        """The exit code once set; ``None`` while still running."""
        ...

    def send_signal(self, sig: int) -> None:
        """Send ``sig`` to the process (and, by default, its group)."""
        ...


def _default_runner(
    command: list[str], *, repo_root: str, log_path: Path
) -> _ProcLike:
    """Spawn ``command`` in its own process group, redirecting stdout+stderr to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # The handle intentionally outlives this call — the subprocess writes to
    # it for the whole run — so a `with` block cannot scope it; the wrapper
    # closes it in the runner's `finally`.
    log_handle: IO[bytes] = open(log_path, "ab")  # noqa: SIM115
    try:
        proc = subprocess.Popen(  # noqa: S603 — command is checked upstream
            command,
            cwd=repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group so killpg() teardown works
        )
    except Exception:
        # Popen failed (mutmut missing, exec not found, ...): no subprocess
        # owns the handle, so the wrapper's `finally` will never close it.
        log_handle.close()
        raise
    return _PopenWrapper(proc, log_handle)


class _PopenWrapper(_ProcLike):
    """``_ProcLike`` adapter for a real ``Popen``.

    ``send_signal`` targets the process group via ``os.killpg``, matching the
    promise of ``start_new_session=True`` — a SIGTERM then SIGKILL in the
    cap loop terminates mutmut's forked test workers, not just the parent.
    """

    def __init__(self, proc: subprocess.Popen[Any], log_handle: IO[bytes]) -> None:
        self._proc = proc
        self._log = log_handle

    def poll(self) -> int | None:
        return self._proc.poll()

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    def send_signal(self, sig: int) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(self._proc.pid), sig)

    def close_log(self) -> None:
        self._log.close()


def build_command(patterns: Sequence[str]) -> list[str]:
    """Construct the ``mutmut run <patterns...>`` command line."""
    return [_MUTMUT, "run", *patterns]


def run_mutmut(
    command: Sequence[str],
    *,
    repo_root: str,
    log_path: Path,
    runner: Callable[..., _ProcLike] = _default_runner,
    budget_seconds: float = _DEFAULT_BUDGET_SECONDS,
    grace_seconds: float = _DEFAULT_GRACE_SECONDS,
) -> dict[str, object]:
    """Run ``command`` under a wall-clock cap, returning a structured summary.

    Args:
        command: The exact argv to spawn (typically from :func:`build_command`).
        repo_root: Working directory for the subprocess.
        log_path: File to tee stdout+stderr to.
        runner: Spawn callable — test seam, defaults to ``_default_runner``.
        budget_seconds: Wall-clock cap in seconds; ``0`` disables the cap
            (the per-PR gate's default is 3600 s, the CI job cap; the
            measurement run overrides to a much larger value to record the
            true number).
        grace_seconds: After SIGTERM, how long to wait before SIGKILL.

    Returns:
        A dict with keys ``status`` (``"ok"``, ``"over_budget"``,
        ``"error"``), ``exit_code``, ``wall_clock_seconds``,
        ``load_avg_{1m,5m,15m}``, ``cpu_count``, ``log_path``.
        The dict is JSON-serialisable.

    Notes:
        ``load_avg_*`` is ``None`` on platforms without ``os.getloadavg``
        (Windows). The keys remain present so downstream consumers can
        rely on the schema.
    """
    load1: float | None
    load5: float | None
    load15: float | None
    try:
        load1, load5, load15 = os.getloadavg()
    except (AttributeError, OSError):
        # Windows has no ``os.getloadavg``; report ``None`` so the
        # schema is uniform across platforms.
        load1 = load5 = load15 = None
    start = time.monotonic()
    # ``proc`` is initialised to ``None`` BEFORE the ``try`` so a runner
    # exception (mkdir, open, or a custom runner that opened a log and
    # then failed) still runs the ``finally`` and surfaces verbatim —
    # without this, the ``finally``'s ``getattr(proc, "close_log", None)``
    # would ``NameError`` and mask the runner's original exception. The
    # runner assigns the real handle inside the ``try`` once it returns.
    proc: _ProcLike | None = None
    status = "error"
    exit_code: int | None = None
    try:
        proc = runner(list(command), repo_root=repo_root, log_path=log_path)
        while True:
            exit_code = proc.poll()
            if exit_code is not None:
                status = "ok" if exit_code == 0 else "error"
                break
            # Poll first, then check the budget. A run that exits cleanly in
            # the small window `[budget, budget + _POLL_SECONDS)` is reported
            # as `ok` rather than `over_budget` — the safer ordering, since it
            # never SIGTERMs a clean run; the inverse would report bogus
            # `over_budget` for every long-but-successful run.
            if budget_seconds > 0 and (
                time.monotonic() - start >= budget_seconds
            ):
                proc.send_signal(signal.SIGTERM)
                grace_deadline = time.monotonic() + grace_seconds
                while proc.poll() is None and time.monotonic() < grace_deadline:
                    time.sleep(_POLL_SECONDS)
                if proc.poll() is None:
                    proc.send_signal(_KILL_SIGNAL)
                    # Bounded reap wait — the kernel sets `returncode` only
                    # after the child is reaped, so reading it immediately
                    # after SIGKILL would emit `null` in the JSON summary and
                    # lose the `-9` signal marker. A short poll loop bounds
                    # the wait without blocking forever on a stuck zombie.
                    reap_deadline = time.monotonic() + 2.0
                    while proc.poll() is None and time.monotonic() < reap_deadline:
                        time.sleep(_POLL_SECONDS)
                exit_code = proc.returncode
                status = "over_budget"
                break
            time.sleep(_POLL_SECONDS)
    finally:
        wall = time.monotonic() - start
        if proc is not None:
            close = getattr(proc, "close_log", None)
            if callable(close):
                close()
    return {
        "status": status,
        "exit_code": exit_code,
        "wall_clock_seconds": wall,
        "load_avg_1m": load1,
        "load_avg_5m": load5,
        "load_avg_15m": load15,
        "cpu_count": os.cpu_count(),
        "log_path": str(log_path),
    }


# --------------------------------------------------------------------------
# Pyproject loading — only_mutate → registered module set
# --------------------------------------------------------------------------


def _load_only_mutate_modules(repo_root: str) -> frozenset[str]:
    """Read ``[tool.mutmut] only_mutate`` from ``pyproject.toml`` and return its module set.

    Used to filter cross-module ``provider_hooks`` rows: a
    ``translate_to_upstream`` in ``kitty.bridge.engine`` is NOT covered by
    the provider_hooks row, because engine is not in ``only_mutate``.

    Tries stdlib ``tomllib`` first (3.11+) and falls back to the
    ``tomli`` dev extra for 3.10 — the pyproject's own dev-extras block
    declares the conditional.
    """
    pyproject = Path(repo_root) / "pyproject.toml"
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover — exercised on 3.10
        import tomli as tomllib  # type: ignore[no-redef, import-not-found]
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    entries: list[str] = data.get("tool", {}).get("mutmut", {}).get("only_mutate", [])
    modules: set[str] = set()
    for entry in entries:
        e = entry.rstrip("/")
        if e.endswith("/*"):
            # Package glob → the package's dotted name (also covers its submodules).
            e = e[:-2]
            if e.startswith("src/kitty/"):
                modules.add(e[len("src/") :].replace("/", "."))
            continue
        if e.endswith(".py"):
            mod = module_for(e)
            if mod is not None:
                modules.add(mod)
            continue
        # Bare directory entry (none today, but defensive).
        if e.startswith("src/kitty/"):
            modules.add(e[len("src/") :].replace("/", "."))
    return frozenset(modules)


def _load_registry(repo_root: str) -> dict[str, list[Target]]:
    """Load ``mutmut_scope.TARGET_GROUPS`` by filesystem path.

    Same loader pattern as the KBR-88 aggregator tests; avoids any
    package layout assumption about ``tests/`` (no ``__init__.py``).
    """
    import importlib.util

    scope_path = Path(repo_root) / "tests" / "mutmut_scope.py"
    spec = importlib.util.spec_from_file_location("mutmut_scope", scope_path)
    if spec is None or spec.loader is None:  # pragma: no cover — load guard
        raise ImportError(f"cannot load {scope_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mutmut_scope"] = module
    spec.loader.exec_module(module)
    # Convert the registry's NamedTuple rows into our dataclass rows so the
    # rest of this file never imports `mutmut_scope` (which lives in
    # ``tests/`` and is not on the script's normal import path).
    raw: dict[str, list[Any]] = module.TARGET_GROUPS
    out: dict[str, list[Target]] = {}
    for group, rows in raw.items():
        out[group] = [
            Target(module=row.module, cls=row.cls, function_or_method=row.function_or_method)
            for row in rows
        ]
    return out


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------


def _print_dry_run(
    diff: DiffMeta,
    patterns: list[str],
    command: list[str],
    *,
    out: Any = sys.stdout,
) -> None:
    """Emit the dry-run report (touched modules/defs, patterns, command)."""
    print(f"rev_range: {diff.rev_range}", file=out)
    print("touched_modules:", file=out)
    for module in sorted(diff.touched_modules):
        print(f"  - {module}", file=out)
    print("touched_defs (per module):", file=out)
    for module in sorted(diff.touched_defs):
        defs = sorted(diff.touched_defs[module], key=lambda cf: (cf[0] or "", cf[1]))
        print(f"  {module}:", file=out)
        for cls, fn in defs:
            print(f"    {cls or '-'} :: {fn}", file=out)
    print("patterns:", file=out)
    for pattern in patterns:
        print(f"  {pattern}", file=out)
    print("command:", file=out)
    print(f"  {' '.join(command)}", file=out)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code.

    Args:
        argv: Argument list (default: ``sys.argv[1:]``).

    Returns:
        ``0`` on a successful dry-run or a clean mutmut run; ``1`` when an
        empty intersection refuses ``--run`` or when mutmut exits non-zero.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Measure a changed-code mutation run against the fast gate's budget "
            "(T-H4 / KBR-92). Maps a git rev-range onto the §6.1 scope registry "
            "and runs `mutmut run` scoped to the touched functions only."
        )
    )
    parser.add_argument(
        "rev_range",
        help="Git rev-range to scope to (e.g. `30a91a0^1..30a91a0`).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute `mutmut run` and emit a JSON summary (default: dry-run).",
    )
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parent.parent),
        help="Repository root (default: this script's parent).",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for mutmut's log file (default: <repo>/mutants/logs/).",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=_DEFAULT_BUDGET_SECONDS,
        help=(
            "Wall-clock cap in seconds (default: 3600 — the CI job cap). "
            "Use 0 to disable the cap; use a much larger value to record the "
            "true number for an exploratory measurement."
        ),
    )
    parser.add_argument(
        "--grace-seconds",
        type=float,
        default=_DEFAULT_GRACE_SECONDS,
        help="Seconds to wait after SIGTERM before SIGKILL (default: 10).",
    )
    args = parser.parse_args(argv)

    log_dir = (
        Path(args.log_dir)
        if args.log_dir is not None
        else Path(args.repo) / "mutants" / "logs"
    )
    # `time.time_ns()` makes the filename unique across same-second
    # invocations (the alternative `int(time.time())` collides if two
    # `--run`s start inside one second; the second would append to the
    # first's log without warning).
    log_path = log_dir / f"changed_code_{time.time_ns()}.log"

    diff = diff_meta_from_git_range(args.rev_range, args.repo)
    registry = _load_registry(args.repo)
    registered = _load_only_mutate_modules(args.repo)
    command = build_command([])  # populated below; pre-built for the dry-run shape

    try:
        patterns = scoped_patterns(diff, registry, registered)
    except EmptyIntersectionError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        if args.run:
            return 1
        # Dry-run with empty intersection is informational, not a failure.
        print("rev_range:", args.rev_range, file=sys.stdout)
        print("covered patterns: (none — no §6.1-scope function touched)", file=sys.stdout)
        return 0

    command = build_command(patterns)

    if not args.run:
        _print_dry_run(diff, patterns, command)
        return 0

    summary = run_mutmut(
        command,
        repo_root=args.repo,
        log_path=log_path,
        budget_seconds=args.budget_seconds,
        grace_seconds=args.grace_seconds,
    )
    payload = {
        "rev_range": diff.rev_range,
        "patterns": patterns,
        "command": command,
        **summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
