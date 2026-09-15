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

sys.path.insert(0, "tests")
from mutmut_scope import DEFERRED_GROUPS, TARGET_GROUPS, patterns_for  # noqa: E402

# mutmut's `status_by_exit_code` (mutmut/stats.py), as a lookup table so we
# can translate the per-mutant exit codes the `.meta` files carry.
STATUS_BY_EXIT_CODE = {
    1: "killed",
    3: "killed",  # pytest internal error
    0: "survived",
    5: "no_tests",
    33: "no_tests",
    34: "skipped",
    35: "suspicious",
    36: "timeout",
    37: "caught_by_type_check",
    None: "not_checked",
    -24: "timeout",
    24: "timeout",
    152: "timeout",
    255: "timeout",
    -11: "segfault",
    -9: "segfault",
}


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
    return sorted(Path("mutants/src").rglob("*.meta"))


def bucket_mutants(
    meta_files: Iterable[Path],
    group_patterns: dict[str, list[str]],
) -> dict[str, Stat]:
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
        ``{group_name: Stat}``, plus ``__unmatched__`` and ``__total__``
        keys for visibility into what did not match any group.
    """
    out: dict[str, Stat] = {g: Stat() for g in group_patterns}
    out["__unmatched__"] = Stat()
    out["__total__"] = Stat()

    for meta_path in meta_files:
        data = json.loads(meta_path.read_text())
        for mutant_key, exit_code in data.get("exit_code_by_key", {}).items():
            status = STATUS_BY_EXIT_CODE.get(exit_code, "suspicious")

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
            setattr(target, status, getattr(target, status) + 1)
            setattr(
                out["__total__"], status, getattr(out["__total__"], status) + 1
            )

    return out


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
    stats = bucket_mutants(meta_files, group_patterns)

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
