"""Aggregate a mutmut run's per-mutant results into per-group scores.

T-H1 (KBR-88) recording step. Reads per-file ``.meta`` JSONs mutmut
writes under ``mutants/`` and buckets each mutant by the patterns in
``tests/mutmut_scope``. Prints a per-group table.

The formula is mutmut's own badge formula
(``mutmut/__main__.py::badge``):

    score = (killed + timeout) / (total - skipped)

Timeouts count as kills (a slow mutation is a behaviour change), and
``skipped`` drops out of the denominator (mutmut never tested those
mutants, so including them dilutes the score with noise). ``no_tests``
and ``suspicious`` stay in the denominator — a group whose mutants all
land in ``no_tests`` has a mis-scoped pattern, not a 0% score; the script
asserts ``no_tests == 0`` per group and fails loud if a group's
selection is empty.

Usage::

    .venv/bin/python scripts/aggregate_mutation_baseline.py

Reads from ``mutants/`` (cwd) and ``tests/mutmut_scope.py``. Prints a
markdown table; copy that into ``.system_design/MUTATION_BASELINE.md``.
"""

from __future__ import annotations

import fnmatch
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Resolve the repo root from this file's path so the script runs from
# any cwd. Both inputs (``mutants/``) and the ``tests/`` import below
# depend on a stable root; deriving it from ``__file__`` once is cheaper
# than threading flags through the CLI for two callers (record-the-
# baseline and the test that exercises ``bucket_mutants`` against a
# synthetic fixture).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests"))
from mutmut_scope import DEFERRED_GROUPS, TARGET_GROUPS, patterns_for  # noqa: E402

_MUTANTS_ROOT = _REPO_ROOT / "mutants"


# mutmut's `status_by_exit_code` (mutmut/stats.py) maps exit codes to
# status names. Try to import it from the installed mutmut (the only
# source of truth that follows mutmut's own version); fall back to the
# static table when the import fails (older mutmut 2.x did not export
# it, and a future 3.x release that re-numbers an exit code is a
# contract change the script's caller should notice, not be silently
# wrong about). ``mutmut`` is a dev extra, so the import can fail in a
# production install — the fallback is the right thing there.
def _build_status_by_exit_code() -> dict[int | None, str]:
    """Return mutmut's ``status_by_exit_code`` if importable, else a static table.

    The static table is verified against mutmut 3.8.0 (the version this
    ticket's baseline ran on). A future 3.x release that re-numbers an
    exit code will be picked up by the import branch when mutmut is
    installed; the static fallback preserves the script's behaviour when
    it isn't.
    """
    try:
        from mutmut.stats import status_by_exit_code as live  # type: ignore
    except ImportError:
        return {
            1: "killed",
            3: "killed",  # pytest internal error counts as a kill
            0: "survived",
            5: "no_tests",
            33: "no_tests",
            34: "skipped",
            35: "suspicious",
            36: "timeout",
            37: "caught by type check",
            None: "not_checked",
            -24: "timeout",
            24: "timeout",
            152: "timeout",
            255: "timeout",
            -11: "segfault",
            -9: "segfault",
        }
    return dict(live)


STATUS_BY_EXIT_CODE: dict[int | None, str] = _build_status_by_exit_code()


@dataclass
class Stat:
    killed: int = 0
    survived: int = 0
    no_tests: int = 0
    skipped: int = 0
    suspicious: int = 0
    timeout: int = 0
    caught_by_type_check: int = 0
    segfault: int = 0
    not_checked: int = 0

    @property
    def total(self) -> int:
        return (
            self.killed
            + self.survived
            + self.no_tests
            + self.skipped
            + self.suspicious
            + self.timeout
            + self.caught_by_type_check
            + self.segfault
            + self.not_checked
        )

    @property
    def tested(self) -> int:
        """Mutants that actually ran — denominator of the badge formula.

        Total minus ``skipped`` (skipped mutants were filtered out before
        running) and minus ``not_checked`` (still pending in an interrupted
        run; they are not kills because they have not been examined).
        """
        return self.total - self.skipped - self.not_checked

    @property
    def score(self) -> float:
        """mutmut's badge formula: (killed + timeout) / tested."""
        if self.tested <= 0:
            return 0.0
        return (self.killed + self.timeout) / self.tested


def collect_meta_files() -> list[Path]:
    """Find every per-file ``.meta`` JSON under ``mutants/``."""
    return sorted(_MUTANTS_ROOT.glob("src/**/*.meta"))


# Map mutmut's space-separated status names (whatever the installed
# version's ``status_by_exit_code`` returns) to ``Stat`` field names.
# Unknown names default to ``suspicious`` (the same default mutmut
# uses for unknown exit codes), keeping a future 3.x release that adds
# a new status name visible in the bucket counts.
_STATUS_TO_FIELD = {
    "killed": "killed",
    "survived": "survived",
    "no tests": "no_tests",
    "no_tests": "no_tests",  # underscore spelling, some mutmut versions
    "skipped": "skipped",
    "suspicious": "suspicious",
    "timeout": "timeout",
    "caught by type check": "caught_by_type_check",
    "segfault": "segfault",
    "not checked": "not_checked",
    "not_checked": "not_checked",  # underscore spelling, some mutmut versions
    # "check was interrupted by user" has no Stat field; those
    # mutants are unexamined (the run was stopped), so bucket them
    # alongside ``not_checked``.
    "check was interrupted by user": "not_checked",
}


def bucket_mutants(
    meta_files: Iterable[Path],
    group_patterns: dict[str, list[str]],
) -> tuple[dict[str, Stat], dict[int | None, int]]:
    """Read each meta file and tally each mutant into its target group.

    A mutant is bucketed into the first group whose patterns fnmatch it
    (groups are evaluated in declaration order in ``TARGET_GROUPS``).
    Mutants that match no group land in ``__unmatched__`` — those are
    out-of-scope (a §6.1 selection that excludes a file's every function
    would land here) and are reported separately so a mis-scope is
    visible rather than silent.

    Args:
        meta_files: Paths to mutmut's per-file ``.meta`` files.
        group_patterns: ``{group_name: [pattern, ...]}`` derived from
            ``patterns_for``.

    Returns:
        ``({group_name: Stat}, unknown_exit_codes)`` — the per-group
        buckets (plus ``__unmatched__`` and ``__total__`` keys for
        visibility into what did not match any group), and a map of
        exit codes the table did not recognise to the number of
        mutants that carried each one. ``main()`` prints a one-line
        warning when the second element is non-empty.
    """
    out: dict[str, Stat] = {g: Stat() for g in group_patterns}
    out["__unmatched__"] = Stat()
    out["__total__"] = Stat()
    unknown_exit_codes: dict[int | None, int] = {}

    for meta_path in meta_files:
        data = json.loads(meta_path.read_text())
        for mutant_key, exit_code in data.get("exit_code_by_key", {}).items():
            if exit_code not in STATUS_BY_EXIT_CODE:
                # Unknown exit code — a future mutmut 3.x release that
                # re-numbers an exit code lands here. Bucket as
                # suspicious so the count stays visible, and record the
                # code so ``main`` can warn rather than fail silently.
                unknown_exit_codes[exit_code] = (
                    unknown_exit_codes.get(exit_code, 0) + 1
                )
            raw_status = STATUS_BY_EXIT_CODE.get(exit_code, "suspicious")
            field = _STATUS_TO_FIELD.get(raw_status)
            if field is None:
                # A new mutmut status name we don't recognise; bucket
                # as suspicious so the count is visible and the script
                # does not silently drop the mutant.
                field = "suspicious"

            # Decide the group. Order matters: a mutant that matches
            # multiple groups is bucketed by the first. `mutmut run` is
            # also invoked with the *same* ordered pattern set, so the
            # "first match wins" semantics agree across run-time
            # selection and here-at-rest aggregation.
            matched_group: str | None = None
            for group_name in TARGET_GROUPS:
                if any(
                    fnmatch.fnmatch(mutant_key, p)
                    for p in group_patterns[group_name]
                ):
                    matched_group = group_name
                    break

            target = out[matched_group or "__unmatched__"]
            setattr(target, field, getattr(target, field) + 1)
            setattr(
                out["__total__"], field, getattr(out["__total__"], field) + 1
            )

    return out, unknown_exit_codes


def _warn_unknown_exit_codes(unknown: dict[int | None, int]) -> None:
    """Emit a single stderr summary of unknown exit codes.

    A future mutmut 3.x release that re-numbers an exit code lands in
    ``unknown``. The script continues (the count still gets bucketed as
    ``suspicious``), but the caller should know. One line per code; the
    script exits 0 — this is a heads-up, not a hard failure.
    """
    if not unknown:
        return
    parts = ", ".join(
        f"{code!r}: {count} mutant(s)"
        for code, count in sorted(unknown.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
    )
    print(
        f"warning: {len(unknown)} mutmut exit code(s) not in the "
        f"STATUS_BY_EXIT_CODE table ({parts}); bucketed as suspicious. "
        f"This usually means a mutmut version change — the table needs "
        f"to be updated.",
        file=sys.stderr,
    )


def render_markdown_table(stats: dict[str, Stat]) -> str:
    """Format the per-group table for ``MUTATION_BASELINE.md``."""
    lines = [
        "| Group | Total | Tested | Killed | Survived | Timeout | No tests | Suspicious | Score |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for group_name in TARGET_GROUPS:
        if group_name in DEFERRED_GROUPS:
            lines.append(
                f"| {group_name} | -- | -- | -- | -- | -- | -- | -- | "
                f"_deferred_ |"
            )
            continue
        s = stats[group_name]
        score = f"{s.score * 100:.1f}%"
        lines.append(
            f"| {group_name} | {s.total} | {s.tested} | {s.killed} "
            f"| {s.survived} | {s.timeout} | {s.no_tests} | "
            f"{s.suspicious} | {score} |"
        )
    s = stats["__total__"]
    lines.append(
        f"| **TOTAL (measured)** | {s.total} | {s.tested} | {s.killed} "
        f"| {s.survived} | {s.timeout} | {s.no_tests} | "
        f"{s.suspicious} | **{s.score * 100:.1f}%** |"
    )
    s = stats["__unmatched__"]
    if s.total:
        lines.append(
            f"| __unmatched__ | {s.total} | {s.tested} | {s.killed} "
            f"| {s.survived} | {s.timeout} | {s.no_tests} | "
            f"{s.suspicious} | {s.score * 100:.1f}% |"
        )
    return "\n".join(lines)


def main() -> int:
    meta_files = collect_meta_files()
    if not meta_files:
        print(
            "No .meta files found under mutants/src/ — has `mutmut run` "
            "completed?",
            file=sys.stderr,
        )
        return 2

    group_patterns = {g: patterns_for(g) for g in TARGET_GROUPS}
    stats, unknown_exit_codes = bucket_mutants(meta_files, group_patterns)
    _warn_unknown_exit_codes(unknown_exit_codes)

    # Sanity checks: every scoped group must have ≥ 1 mutant and zero
    # `no_tests`. A zero-total group means the registry or its pattern
    # derivation is wrong; a non-zero `no_tests` means the patterns
    # target functions whose trampoline is never hit by any L1 test,
    # which would mean the L1 suite is incomplete for that group.
    failures: list[str] = []
    for group_name in TARGET_GROUPS:
        if group_name in DEFERRED_GROUPS:
            continue  # deferred groups have no mutant budget; that's by design
        s = stats[group_name]
        if s.total == 0:
            failures.append(
                f"group {group_name!r}: zero mutants matched its patterns"
            )
        if s.no_tests > 0:
            failures.append(
                f"group {group_name!r}: {s.no_tests} mutants classified as "
                f"no_tests (their trampolines were never hit by the L1 "
                f"selection) -- the L1 suite is incomplete for this group"
            )
        if s.not_checked > 0:
            failures.append(
                f"group {group_name!r}: {s.not_checked} mutants are "
                f"not_checked (the run was interrupted or did not finish; "
                f"re-run `mutmut run` to resume)"
            )

    print(render_markdown_table(stats))
    print()
    if failures:
        print("FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
