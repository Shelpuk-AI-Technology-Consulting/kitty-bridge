"""KBR-216 — what CI supplies is stated in one place, and the workflows agree with it.

``.github/workflows/claude-code-review.yml`` runs Claude Code through a released
Kitty Bridge to review every same-repository pull request.  To do that the CI
environment was given a complete kitty installation: a version-pinned Claude Code
CLI, a profile store, a credential store, an egress gateway and two logs.  All of
it has been there since the automated reviewer landed (32d8f07, 2026-09-07).

``.system_design/TEST_SUITE.md`` did not know.  §11 carried **Q12** — *"How is a
pinned Claude Code binary supplied to CI?"* — as an open question to the product
owner, and §6.4.2 reasoned from the assumption that the answer might be "it
cannot be", while the answer sat in the workflow tree.  Three implementation
tasks were flagged ``blocked Q12`` against a blocker that had already cleared.

§8.6 now states the inventory, and this file is what stops the two drifting apart
again.  Five arms, because fewer would not have caught what this file was written
for — each of the six rows is covered by at least one *reverse* direction, so no
row can be deleted from the table with every arm green:

* **forward** — every binding §8.6 names is really used by the file §8.6 names it
  in, so the document cannot promise a capability CI does not have;
* **reverse** — every ``secrets.KITTY_*`` / ``vars.KITTY_*`` binding the workflows
  actually use appears as a row, so a capability cannot be added to CI and left
  out of the design;
* **version** — the CLI pin §8.6 quotes is the one CI installs, in both
  directions, since that pairing is maintained by hand against a floating action
  tag and is the arm most likely to fire in anger;
* **log** — every log the generated launcher writes is a row.  The two log rows
  bind neither a secret nor a version, so the reverse and version arms are both
  blind to them, and one of them carries the constraint that the debug log may
  never be uploaded as an artifact;
* **fork** — the guard §8.6's per-PR/nightly split rests on is still in place.
  Its removal changes no binding, so every other arm would stay green while the
  section's most load-bearing paragraph became false.

A forward-only check would go green on the day someone deletes a secret from the
workflow *and* from the table; the reverse arm is what makes the inventory a
statement about CI rather than a statement about itself.

**The comparison is a pure reporter, not an assertion.**
:func:`inventory_discrepancies` returns ``(arm, message)`` pairs and an empty list
means agreement; the two sibling reporters return bare messages.  That shape is what lets the falsification cases below
drive the *real* scan over a doctored document — watching a rule fail by hand is
not the same as shipping its negative control, and a matcher exercised only in
isolation proves nothing about its assembly.

Two artifacts edited separately that must agree, both readable statically — the
§6.2.3 "docs ⇄ code" case — so this file is ``l2`` and is named in
``tests/test_layer_selection.py``'s allowlist.

**Two stated limits.**  The reverse arm scans the ``KITTY_``-prefixed namespace
only: ``secrets.ANTHROPIC_AUTH_TOKEN`` and ``secrets.OPENROUTER_API_KEY`` also
appear in the tree and are deliberately not inventory rows, because neither is a
kitty capability and widening the arm to every secret would make the table a
mirror of the workflow's secret list rather than a design statement about what
the test suite may rely on.  And the forward arm is a substring scan for every
artifact *except* the launcher, so a row naming a binding that appears only in a
workflow's comments would pass — the launcher is special-cased because its own
prose demonstrably does exactly that.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

# L2: the subject of this file is an artifact outside `src/kitty` Python code --
# a design document and a tree of workflow YAML -- two things edited separately
# that must agree. It gates pull requests in the `l1 or l2` job; the marker
# records which half of that expression it answers to, and keeps a source-text
# scan out of the L1 set that mutation testing will judge.
pytestmark = pytest.mark.l2

ROOT = Path(__file__).resolve().parents[1]
TEST_SUITE = ROOT / ".system_design" / "TEST_SUITE.md"
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
REVIEW_WORKFLOW = WORKFLOWS_DIR / "claude-code-review.yml"

#: The §8.6 heading the inventory lives under. Matched as a prefix so the
#: section may be retitled without renumbering, but not silently renumbered.
_INVENTORY_HEADING = "### 8.6"

#: A markdown table row: `| Capability | `file` | `binding` |`.
_TABLE_ROW = re.compile(r"^\|(?P<cells>.+)\|\s*$")

#: A separator row, `|---|---|---|`, which carries no data.
_SEPARATOR_ROW = re.compile(r"^\|[\s:|-]+\|\s*$")

#: A kitty capability binding as GitHub Actions spells it, in either namespace.
#: The `KITTY_` prefix is the stated limit in this module's docstring.
_KITTY_BINDING = re.compile(r"\b(?:secrets|vars)\.(KITTY_[A-Z0-9_]+)\b")

#: How the review workflow pins the Claude Code CLI: the installer script is
#: handed the exact version as its one positional argument. Scoped to Claude's
#: own installer URL -- unscoped, any other project's `install.sh | bash -s --
#: 1.2.3` reports as "CI installs Claude Code 1.2.3", a red gate naming the
#: wrong tool, which is the mis-diagnosis class this repository's guards exist
#: to prevent.
_CLI_PIN = re.compile(
    r"claude\.ai/install\.sh\s*\|\s*bash\s+-s\s+--\s+(?P<version>[0-9][0-9A-Za-z.\-+]*)"
)

#: The shape of the row binding that quotes the CLI pin. The version arm reads
#: only rows of this shape: a future row binding `python-version: "3.12"` would
#: otherwise report as "§8.6 quotes Claude Code 3.12", naming the wrong tool.
#: It is also what keeps the comparison whole-token: the capture runs greedily
#: to the end of the version and the two sides are compared as strings, so a
#: document quoting `2.1.23` never satisfies a workflow installing `2.1.238`.
_PIN_BINDING = re.compile(r"\bbash\s+-s\s+--\s+(?P<version>[0-9][0-9A-Za-z.\-+]*)")

#: The condition that keeps a fork's pull request from ever starting the review
#: job. §8.6's whole per-PR/nightly split rests on this one line.
_FORK_GUARD = "github.event.pull_request.head.repo.full_name == github.repository"

#: The trigger that would undo it by running a fork's code in the base branch's
#: context, secrets included. The workflow refuses it in prose; this is the check.
_PULL_REQUEST_TARGET = "pull_request_target"

#: The action whose job the fork guard protects. Compared exactly once the tag is
#: split off, so a bump (`@v1` -> `@v2`) is not a discrepancy but
#: `anthropics/claude-code-action-fork` is not mistaken for it. The job's name is
#: never pinned.
_REVIEW_ACTION = "anthropics/claude-code-action"

#: A log file name as the generated launcher spells it.
_LOG_FILE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9.-]*\.log\b")

#: The script that writes the launcher. Its *generated* text is the authority
#: for the log rows, never its own source — see :func:`ci_artifacts`.
CONFIGURE_KITTY = ROOT / ".github" / "review" / "scripts" / "configure_kitty.py"


def _strip_cell(cell: str) -> str:
    """Return a markdown table cell's text without decoration.

    Strips surrounding whitespace and the backticks the document uses to mark a
    filename or a binding as literal, so a row's cells can be compared against
    real paths and real workflow text.

    Args:
        cell: One cell's raw text, as split from the table row.

    Returns:
        The cell's content with whitespace and backticks removed.
    """
    return cell.strip().strip("`").strip()


def inventory_rows(markdown: str) -> list[tuple[str, str, str]]:
    """Return the capability rows of the §8.6 inventory table.

    Reads the first markdown table that follows the §8.6 heading and stops at the
    first line that is not a table row, so prose after the table is not parsed as
    data.  The table's own header row and its separator are discarded.

    An absent section yields an empty list rather than an exception: this is a
    parser feeding a reporter, and :func:`inventory_discrepancies` is where a
    missing section becomes a reported discrepancy.

    Args:
        markdown: The full text of ``TEST_SUITE.md``.

    Returns:
        One ``(capability, artifact, binding)`` triple per row, in document
        order.  Empty when the section or its table is absent.
    """

    # Locate the section; everything before the heading is another section's table.
    lines = markdown.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(_INVENTORY_HEADING)),
        None,
    )
    if start is None:
        return []

    # Walk forward to the table, then consume it until the first non-row line.
    rows: list[tuple[str, str, str]] = []
    seen_table = False
    for line in lines[start + 1 :]:
        match = _TABLE_ROW.match(line)
        if match is None:
            if seen_table:
                break
            continue
        seen_table = True
        if _SEPARATOR_ROW.match(line):
            continue
        cells = [_strip_cell(cell) for cell in match.group("cells").split("|")]
        if len(cells) != 3:
            continue
        rows.append((cells[0], cells[1], cells[2]))

    # The header row is data-shaped and has to go by position, not by its wording.
    return rows[1:] if rows else rows


def workflow_bindings(text: str) -> set[str]:
    """Return the kitty capability bindings a workflow's text references.

    Args:
        text: The full text of one workflow file.

    Returns:
        The ``KITTY_``-prefixed names used through ``secrets.`` or ``vars.``,
        without their namespace prefix.
    """
    return set(_KITTY_BINDING.findall(text))


def pinned_cli_version(text: str) -> str | None:
    """Return the Claude Code CLI version a workflow's text installs.

    Args:
        text: The full text of one workflow file.

    Returns:
        The version handed to the official installer script, or ``None`` when the
        text installs no CLI.
    """
    match = _CLI_PIN.search(text)
    return match.group("version") if match else None


def inventory_discrepancies(markdown: str, artifacts: dict[str, str]) -> list[tuple[str, str]]:
    """Report every way the §8.6 inventory and the CI artifacts disagree.

    A pure reporter: it reads nothing from disk, raises nothing, and an empty
    result means the two agree.  That is what lets the falsification cases drive
    this same function over a document carrying a planted defect.

    Four arms, tagged rather than merely worded, so a caller can ask about one
    of them without matching on prose:

    * ``"section"`` — the section exists and has rows at all;
    * ``"forward"`` — each row's binding appears in the artifact the row names;
    * ``"reverse"`` — each kitty binding any artifact uses appears in some row;
    * ``"version"`` — the Claude Code pin agrees, in both directions.

    The log and fork directions are :func:`launcher_log_discrepancies` and
    :func:`fork_guard_discrepancies`, which read a different artifact each.

    Args:
        markdown: The full text of ``TEST_SUITE.md``.
        artifacts: Repository-relative path to full text, for every artifact a
            row may name and every workflow the reverse arm scans.

    Returns:
        ``(arm, message)`` pairs, one per problem, in a stable order.  Empty when
        the inventory and the artifacts agree.
    """

    # A blind parser reports nothing, which is indistinguishable from agreement.
    rows = inventory_rows(markdown)
    if not rows:
        return [
            (
                "section",
                f"TEST_SUITE.md has no {_INVENTORY_HEADING} inventory table, or it has no rows",
            )
        ]

    problems: list[tuple[str, str]] = []

    # Forward: the document may not name a binding its own artifact does not use.
    for capability, artifact, binding in rows:
        text = artifacts.get(artifact)
        if text is None:
            problems.append(
                ("forward", f"row {capability!r} names {artifact!r}, which is not a CI artifact")
            )
        elif binding not in text:
            problems.append(
                ("forward", f"row {capability!r} names {binding!r}, absent from {artifact}")
            )

    # Reverse: a capability may not reach CI without reaching the design document.
    # Compared as whole names, never as substrings: a new `secrets.KITTY_EGRESS`
    # would otherwise be absorbed by the documented `secrets.KITTY_EGRESS_JSON`
    # and arrive in CI undocumented with the guard green.
    documented = {
        name for _, _, binding in rows for name in _KITTY_BINDING.findall(binding)
    }
    for path, text in sorted(artifacts.items()):
        for used in sorted(workflow_bindings(text)):
            if used not in documented:
                problems.append(
                    ("reverse", f"{path} uses {used!r}, which no {_INVENTORY_HEADING} row names")
                )

    # Version: the CLI pin is a hand-maintained pairing, so both directions of it
    # belong in the reporter rather than in an inline assertion that no negative
    # control can drive.
    installed = {
        version for text in artifacts.values() if (version := pinned_cli_version(text))
    }
    quoted = {
        match.group("version")
        for _, _, binding in rows
        if (match := _PIN_BINDING.search(binding))
    }
    for version in sorted(installed - quoted):
        problems.append(
            ("version", f"CI installs Claude Code {version}, which no {_INVENTORY_HEADING} row quotes")
        )
    for version in sorted(quoted - installed):
        problems.append(
            ("version", f"{_INVENTORY_HEADING} quotes Claude Code {version}, which CI does not install")
        )

    return problems


def review_job_conditions(text: str) -> list[str]:
    """Return the ``if:`` of every job that runs the review action.

    Scoped to the jobs whose steps use :data:`_REVIEW_ACTION`, not to every job:
    otherwise a second job carrying the same comparison would satisfy the check
    after the guard was removed from the job that actually runs the review.  A
    review job with no ``if:`` contributes an empty string, so it is reported as
    unguarded rather than skipped.

    Args:
        text: The full text of one workflow file.

    Returns:
        One condition per review job, in document order.

    Raises:
        yaml.YAMLError: When ``text`` is not valid YAML.  Deliberately not caught:
            a workflow that will not parse must fail the fork arm, never pass it
            by having no conditions to check.
    """
    document = yaml.safe_load(text)
    jobs = document.get("jobs") if isinstance(document, dict) else None
    if not isinstance(jobs, dict):
        return []
    return [
        str(job.get("if", ""))
        for job in jobs.values()
        if isinstance(job, dict)
        and any(
            isinstance(step, dict)
            and str(step.get("uses", "")).split("@", 1)[0] == _REVIEW_ACTION
            for step in job.get("steps") or []
        )
    ]


def workflow_triggers(text: str) -> set[str]:
    """Return the event names a workflow's ``on:`` block declares.

    Handles PyYAML's quirk of parsing the bare key ``on`` as the boolean
    ``True``, the same way ``tests/test_github_actions.py`` does.

    Args:
        text: The full text of one workflow file.

    Returns:
        The declared trigger names.  Empty when the ``on:`` block is absent or is
        not one of the three shapes GitHub accepts — a mapping, a list or a
        single string.

    Raises:
        yaml.YAMLError: When ``text`` is not valid YAML.  Deliberately not caught,
            and the reason is specific: an empty result here makes the
            ``pull_request_target`` check pass, so swallowing a parse failure
            would turn a broken workflow into a green fork arm.
    """
    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        return set()
    block = document.get("on", document.get(True))
    if isinstance(block, dict):
        return set(block)
    if isinstance(block, list):
        return set(block)
    return {block} if isinstance(block, str) else set()


def fork_guard_discrepancies(markdown: str, review_workflow: str) -> list[str]:
    """Report any way the §8.6 fork claim and the review workflow disagree.

    §8.6's per-PR/nightly split rests on one line of YAML: the review job runs
    only when the pull request's head repository is this repository.  Nothing
    else in this module would notice its removal — no ``KITTY_*`` binding
    changes — so the claim needs its own arm or it is prose with no detector.

    **The trigger check reads the parsed ``on:`` block, never the file's text.**
    The workflow's own comments warn at length against switching to
    ``pull_request_target``; a substring scan is satisfied by that warning and
    reports the very document that forbids it.  Scope by what the workflow
    *declares*, not by what its prose happens to mention.

    Args:
        markdown: The full text of ``TEST_SUITE.md``.
        review_workflow: The full text of ``claude-code-review.yml``.

    Returns:
        Human-readable discrepancies.  Empty when the claim still holds.
    """

    problems: list[str] = []

    # Read the job's own condition, not the 124 KB of mostly-comment around it:
    # commenting the line out leaves it in the file and out of the `if:`.
    conditions = review_job_conditions(review_workflow)
    if not conditions:
        problems.append(f"no job runs {_REVIEW_ACTION!r}, so there is no review job to guard")

    # Each review job must carry the guard itself; a disjunct re-admits forks.
    for condition in conditions:
        if _FORK_GUARD not in condition:
            problems.append(
                f"a job running {_REVIEW_ACTION!r} does not carry the guard {_FORK_GUARD!r}"
            )
        elif "||" in condition:
            problems.append(
                f"the guard is weakened by a disjunct: {condition!r} — an `||` re-admits the "
                "fork runs §8.6 says are excluded"
            )

    if _FORK_GUARD not in markdown:
        problems.append(f"{_INVENTORY_HEADING} no longer quotes the guard {_FORK_GUARD!r}")

    # The documented way this breaks: reaching for the trigger that runs a fork's
    # pull request in the base branch's context, secrets included.
    if _PULL_REQUEST_TARGET in workflow_triggers(review_workflow):
        problems.append(
            f"the review workflow is triggered by {_PULL_REQUEST_TARGET!r}, which hands a fork "
            "run the secrets §8.6 says a fork run does not get"
        )

    return problems


def kitty_job_conditions(text: str) -> dict[str, str]:
    """Return the ``if:`` of every job whose definition binds a ``KITTY_*`` capability.

    The review job is not the only consumer any more: ``tmux-disconnect.yml``
    binds the same organisation secrets to test the pull request's own code.
    Selecting jobs by what they *bind*, rather than by which action they run, is
    what makes the next consumer covered before anyone remembers to add it.

    **Stated limits:** only the dotted spelling (``secrets.KITTY_X``) is seen,
    so ``secrets['KITTY_X']`` and ``secrets: inherit`` into a reusable workflow
    are not; no workflow here uses either.

    Args:
        text: The full text of one workflow file.

    Returns:
        Job name to condition (empty string for a job with no ``if:``), for
        every job whose own YAML references ``secrets.KITTY_*`` or ``vars.KITTY_*``.

    Raises:
        yaml.YAMLError: When ``text`` is not valid YAML; a workflow that will not
            parse must fail the arm, never pass it with no jobs to check.
    """
    document = yaml.safe_load(text)
    jobs = document.get("jobs") if isinstance(document, dict) else None
    if not isinstance(jobs, dict):
        return {}
    return {
        str(name): str(job.get("if", ""))
        for name, job in jobs.items()
        if isinstance(job, dict) and _KITTY_BINDING.search(yaml.safe_dump(job))
    }


def kitty_job_fork_discrepancies(workflow: str) -> list[str]:
    """Report every job binding a ``KITTY_*`` capability that a fork's pull request could start.

    §8.6 says a fork run never receives the kitty credentials. GitHub withholds
    secrets from a fork's ``pull_request`` run, but a ``vars.`` value is not
    documented as withheld, and ``pull_request_target`` hands a fork the secrets
    outright, so each such job must refuse forks itself.

    Args:
        workflow: The full text of one workflow file.

    Returns:
        Human-readable discrepancies. Empty when every binding job carries the
        guard without a disjunct and the workflow is not triggered by
        ``pull_request_target``.
    """
    problems: list[str] = []
    conditions = kitty_job_conditions(workflow)
    for name, condition in conditions.items():
        if _FORK_GUARD not in condition:
            problems.append(f"job {name!r} binds a KITTY_* capability and does not carry {_FORK_GUARD!r}")
        elif "||" in condition:
            problems.append(f"job {name!r} weakens the fork guard with a disjunct: {condition!r}")
    if conditions and _PULL_REQUEST_TARGET in workflow_triggers(workflow):
        problems.append(
            f"a workflow binding KITTY_* capabilities is triggered by {_PULL_REQUEST_TARGET!r}, "
            "which hands a fork run the secrets"
        )
    return problems


def launcher_log_discrepancies(markdown: str, launcher: str) -> list[str]:
    """Report any log the CI launcher writes that §8.6 does not name.

    The reverse direction for the two log rows, which nothing else covers.  A
    row whose binding is not a ``secrets.``/``vars.`` reference is invisible to
    the reverse arm and carries no version, so it could be deleted from the
    table with every other arm green — taking §8.6's "never upload the debug
    log" constraint out of the inventory with it.

    Args:
        markdown: The full text of ``TEST_SUITE.md``.
        launcher: The launcher text ``configure_kitty.wrapper_body`` generates.

    Returns:
        Human-readable discrepancies.  Empty when every log written is named.
    """
    documented = {binding for _, _, binding in inventory_rows(markdown)}
    return [
        f"the CI launcher writes {log!r}, which no {_INVENTORY_HEADING} row names"
        for log in sorted(set(_LOG_FILE.findall(launcher)))
        if log not in documented
    ]


def arm(problems: list[tuple[str, str]], name: str) -> list[str]:
    """Return the messages one arm of the reporter produced.

    Selecting by tag rather than by matching on message text is what keeps a
    test's claim stable when a diagnostic is reworded.

    Args:
        problems: The reporter's full result.
        name: The arm to select — ``"section"``, ``"forward"``, ``"reverse"``
            or ``"version"``.

    Returns:
        That arm's messages, in the reporter's order.
    """
    return [message for tag, message in problems if tag == name]


def unguarded_rows(markdown: str, artifacts: dict[str, str], launcher: str) -> list[str]:
    """Return the §8.6 rows whose deletion no arm would report.

    Drops each row from a copy of the document, one at a time, and asks every
    reporter whether anything changed.  A row nothing reports is a row the table
    can lose silently — which is how a capability, or a constraint attached to
    one, leaves the design with the guard green.

    Args:
        markdown: The full text of ``TEST_SUITE.md``.
        artifacts: Repository-relative path to authoritative text, as
            :func:`ci_artifacts` builds it.
        launcher: The launcher text ``configure_kitty.wrapper_body`` generates.

    Returns:
        The capability names of the unguarded rows, in document order.  Empty
        when every row is covered by at least one reverse direction.
    """
    lines = markdown.splitlines()
    unguarded: list[str] = []
    for capability, _, binding in inventory_rows(markdown):
        # Remove exactly this row's table line, never a prose mention of it.
        kept = [
            line
            for line in lines
            if not (line.startswith("|") and capability in line and f"`{binding}`" in line)
        ]
        assert len(kept) == len(lines) - 1, f"row {capability!r} did not match exactly one line"
        without = "\n".join(kept)

        if not inventory_discrepancies(without, artifacts) and not launcher_log_discrepancies(
            without, launcher
        ):
            unguarded.append(capability)
    return unguarded


@pytest.fixture(scope="module")
def suite_markdown() -> str:
    """Return the text of ``TEST_SUITE.md``.

    Returns:
        The design document's full contents.
    """
    return TEST_SUITE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ci_launcher() -> str:
    """Return the launcher ``configure_kitty.py`` generates.

    Imported and called rather than read, because what the module *says* and
    what it *writes* are two different things and only the second reaches a
    runner.  The script is import-safe: module level is constants and function
    definitions, with ``main()`` behind an ``if __name__`` guard.

    Returns:
        The full text of the generated launcher script.

    Raises:
        OSError: When ``configure_kitty.py`` is missing or unreadable.
        AssertionError: When no import spec can be built for it.
    """
    spec = importlib.util.spec_from_file_location("configure_kitty", CONFIGURE_KITTY)
    assert spec is not None and spec.loader is not None, f"cannot import {CONFIGURE_KITTY}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.wrapper_body("kitty")


@pytest.fixture(scope="module")
def ci_artifacts(ci_launcher: str) -> dict[str, str]:
    """Return every CI artifact the inventory may name, keyed by relative path.

    Covers the whole workflow directory rather than only the review workflow, so
    the reverse arm sees a kitty binding introduced in a *new* workflow — which
    is exactly how the next capability will arrive.  Both YAML spellings, because
    GitHub accepts either and a ``.yaml`` file added tomorrow would otherwise be
    invisible; ``tests/test_github_actions.py`` sets the same precedent.

    🔴 **`configure_kitty.py` maps to the launcher it generates, not to its own
    source.** Its docstring and comments discuss ``--debug-file`` and the log
    names at length, so a forward arm reading the module text is satisfied by
    the prose even after ``wrapper_body`` stops emitting either flag — measured.
    Scope by what the artifact *declares*, exactly as the fork arm reads the
    parsed ``on:`` block instead of the file that argues about it.

    Args:
        ci_launcher: The generated launcher text.

    Returns:
        Repository-relative POSIX path to the text that is authoritative for it.

    Raises:
        OSError: When a workflow file cannot be read.
    """
    paths = sorted(list(WORKFLOWS_DIR.glob("*.yml")) + list(WORKFLOWS_DIR.glob("*.yaml")))
    artifacts = {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8") for path in paths
    }
    artifacts[CONFIGURE_KITTY.relative_to(ROOT).as_posix()] = ci_launcher
    return artifacts


class TestTheInventoryIsReadable:
    """The parser finds real rows, so the agreement cases below are not vacuous."""

    def test_the_inventory_section_parses(self, suite_markdown: str) -> None:
        """🔴 The control.

        Without it, a parser that silently finds nothing would make every
        agreement case below pass over an empty set.  Asserts shape, never
        wording: a row's capability name is prose and a test that pinned it would
        fail on every legitimate revision.
        """
        rows = inventory_rows(suite_markdown)

        assert len(rows) >= 5, f"§8.6 parsed {len(rows)} rows; the inventory names at least five"
        for capability, artifact, binding in rows:
            assert capability, "a row has no capability name"
            assert artifact, f"row {capability!r} names no artifact"
            assert binding, f"row {capability!r} names no binding"


class TestTheInventoryAndTheWorkflowsAgree:
    """Both directions, over the real tree."""

    def test_every_binding_the_document_names_is_present_in_the_artifact_it_names(
        self, suite_markdown: str, ci_artifacts: dict[str, str]
    ) -> None:
        """Forward: §8.6 may not promise a capability CI does not have."""
        forward = arm(inventory_discrepancies(suite_markdown, ci_artifacts), "forward")

        assert forward == [], "§8.6 names something the CI artifacts do not carry:\n" + "\n".join(
            forward
        )

    def test_every_kitty_binding_the_artifacts_use_is_in_the_inventory(
        self, suite_markdown: str, ci_artifacts: dict[str, str]
    ) -> None:
        """Reverse: a capability may not reach CI without reaching the design."""
        reverse = arm(inventory_discrepancies(suite_markdown, ci_artifacts), "reverse")

        assert reverse == [], "a CI capability is undocumented in §8.6:\n" + "\n".join(reverse)

    def test_the_pinned_claude_cli_version_is_the_one_the_workflow_installs(
        self, suite_markdown: str, ci_artifacts: dict[str, str]
    ) -> None:
        """The document quotes the pin, and the pin is a hand-maintained pairing.

        Driven through the reporter, not asserted inline, so the negative control
        below exercises this same code path.  Matched as a whole token, so a
        document quoting ``2.1.23`` is not satisfied by a workflow installing
        ``2.1.238``.
        """
        version = arm(inventory_discrepancies(suite_markdown, ci_artifacts), "version")

        assert version == [], "§8.6 and the workflow disagree on the CLI pin:\n" + "\n".join(
            version
        )

    def test_every_log_the_launcher_writes_is_in_the_inventory(
        self, suite_markdown: str, ci_launcher: str
    ) -> None:
        """The reverse direction for the two log rows, which nothing else covers.

        Their bindings are not ``secrets.``/``vars.`` references and carry no
        version, so the reverse and version arms are both blind to them.
        """
        problems = launcher_log_discrepancies(suite_markdown, ci_launcher)

        assert problems == [], "the launcher writes a log §8.6 does not name:\n" + "\n".join(
            problems
        )

    def test_no_row_can_be_deleted_with_every_arm_green(
        self, suite_markdown: str, ci_artifacts: dict[str, str], ci_launcher: str
    ) -> None:
        """The property §8.6 and this module both claim, checked for every row.

        It was stated three times and tested for one row.  Enumerating the rows
        from the shipped table, rather than from the cases this file happened to
        write, is what extends it to the row nobody has added yet.
        """
        unguarded = unguarded_rows(suite_markdown, ci_artifacts, ci_launcher)

        assert unguarded == [], (
            "these §8.6 rows could be deleted with every arm green — add a reverse "
            f"direction that covers them: {unguarded}"
        )

    def test_the_review_job_still_refuses_a_fork_pull_request(self, suite_markdown: str) -> None:
        """§8.6's per-PR/nightly split rests on this guard, so the guard is checked.

        The fourth direction.  Neither the forward nor the reverse arm would
        notice the guard's removal — no ``KITTY_*`` binding changes — and the
        fork asymmetry is the most load-bearing sentence §8.6 adds.
        """
        problems = fork_guard_discrepancies(
            suite_markdown, REVIEW_WORKFLOW.read_text(encoding="utf-8")
        )

        assert problems == [], "§8.6's fork claim no longer matches the workflow:\n" + "\n".join(
            problems
        )

    def test_every_job_binding_kitty_capabilities_refuses_a_fork_pull_request(
        self, ci_artifacts: dict[str, str]
    ) -> None:
        """The fork claim, for every consumer of the kitty credentials, not only the review job."""
        workflows = {path: text for path, text in ci_artifacts.items() if path.endswith((".yml", ".yaml"))}
        binding_jobs = sum(len(kitty_job_conditions(text)) for text in workflows.values())
        problems = [
            f"{path}: {problem}" for path, text in workflows.items() for problem in kitty_job_fork_discrepancies(text)
        ]

        assert binding_jobs >= 2, f"the sweep found {binding_jobs} kitty-binding jobs; the tree has at least two"
        assert problems == [], "a job holding kitty credentials would run for a fork:\n" + "\n".join(problems)


class TestTheGuardCanFail:
    """🔴 Negative controls. Each arm is handed a planted defect and must report it.

    They doctor a **copy** of the real inputs, never the tree, so a control can
    never leave a defect behind.
    """

    def test_a_binding_the_named_artifact_lost_is_reported(
        self, suite_markdown: str, ci_artifacts: dict[str, str]
    ) -> None:
        """Forward arm: the real rows, against artifacts emptied of their bindings.

        Emptying the artifacts rather than inventing a row is what makes this a
        control on the *shipped* table: every row must be carried by something,
        so every row must be reported when nothing carries it.
        """
        emptied = dict.fromkeys(ci_artifacts, "")

        forward = arm(inventory_discrepancies(suite_markdown, emptied), "forward")

        assert len(forward) == len(inventory_rows(suite_markdown)), (
            f"the forward arm reported {len(forward)} of "
            f"{len(inventory_rows(suite_markdown))} rows against emptied artifacts"
        )

    def test_a_binding_absent_from_the_inventory_is_reported(
        self, suite_markdown: str, ci_artifacts: dict[str, str]
    ) -> None:
        """Reverse arm: a capability reaching CI without reaching the document."""
        planted = dict(ci_artifacts)
        planted["ci.yml"] = "value: ${{ secrets.KITTY_UNDOCUMENTED_CAPABILITY }}"

        reverse = arm(inventory_discrepancies(suite_markdown, planted), "reverse")

        assert any("KITTY_UNDOCUMENTED_CAPABILITY" in message for message in reverse), (
            f"the reverse arm missed an undocumented capability; it reported {reverse}"
        )

    def test_a_missing_inventory_section_is_reported(self, ci_artifacts: dict[str, str]) -> None:
        """Blindness: a document with no §8.6 must not read as agreement."""
        section = arm(
            inventory_discrepancies("# A design document with no inventory\n", ci_artifacts),
            "section",
        )

        assert section, "a document with no inventory table reported no discrepancy"

    def test_a_stale_pinned_version_is_reported(
        self, suite_markdown: str, ci_artifacts: dict[str, str]
    ) -> None:
        """Version arm: CI bumps the CLI and the document is not re-synced.

        The pairing is hand-maintained against a floating action tag, so this is
        the arm most likely to fire in anger.
        """
        bumped = dict(ci_artifacts)
        bumped[".github/workflows/claude-code-review.yml"] = (
            "curl -fsSL https://claude.ai/install.sh | bash -s -- 9.9.999"
        )

        version = arm(inventory_discrepancies(suite_markdown, bumped), "version")

        assert any("9.9.999" in message for message in version), (
            f"the version arm missed a bumped CLI pin; it reported {version}"
        )

    def test_a_document_quoting_a_version_ci_does_not_install_is_reported(
        self, suite_markdown: str, ci_artifacts: dict[str, str]
    ) -> None:
        """Version arm, the other direction — and the one that pins whole tokens.

        The forward arm cannot catch this: ``bash -s -- 2.1.238`` *contains*
        ``bash -s -- 2.1.23``, so a document quoting the shorter version passes
        it.  Only the ``quoted - installed`` direction reports it, and without
        this case that direction could be deleted with the suite green.
        """
        truncated = suite_markdown.replace("bash -s -- 2.1.238", "bash -s -- 2.1.23")
        assert truncated != suite_markdown, "the mutant did not change the quoted pin"

        version = arm(inventory_discrepancies(truncated, ci_artifacts), "version")

        assert any("2.1.23," in message or "2.1.23 " in message for message in version), (
            f"a document quoting a version CI does not install reported {version}"
        )

    def test_a_row_no_arm_covers_is_reported_as_unguarded(
        self, suite_markdown: str, ci_artifacts: dict[str, str], ci_launcher: str
    ) -> None:
        """Deletion arm: the arrival the property exists for.

        A toolchain row binding ``python-version`` passes the forward arm — the
        review workflow really does use it — but it is neither a ``KITTY_`` name,
        a pin nor a launcher log, so nothing reports its deletion.
        """
        pin_row = next(line for line in suite_markdown.splitlines() if "`bash -s -- " in line)
        extra = "| Python toolchain | `.github/workflows/claude-code-review.yml` | `python-version` |"
        planted = suite_markdown.replace(pin_row, f"{pin_row}\n{extra}", 1)

        unguarded = unguarded_rows(planted, ci_artifacts, ci_launcher)

        assert unguarded == ["Python toolchain"], (
            f"an uncovered row was not flagged as unguarded; the check reported {unguarded}"
        )

    def test_a_dotted_number_outside_the_pin_row_is_not_read_as_a_version(
        self, suite_markdown: str, ci_artifacts: dict[str, str]
    ) -> None:
        """Version arm, the false-positive direction: only the pin row is a pin.

        Reading every row's dotted numbers would report a toolchain version as a
        Claude Code version and name the wrong tool.
        """
        pin_row = next(line for line in suite_markdown.splitlines() if "`bash -s -- " in line)
        extra = "| Python toolchain | `.github/workflows/claude-code-review.yml` | `python 3.12` |"
        planted = suite_markdown.replace(pin_row, f"{pin_row}\n{extra}", 1)

        version = arm(inventory_discrepancies(planted, ci_artifacts), "version")

        assert version == [], f"a toolchain version was read as a Claude Code pin: {version}"

    def test_a_binding_absorbed_by_a_longer_documented_name_is_reported(
        self, suite_markdown: str, ci_artifacts: dict[str, str]
    ) -> None:
        """Reverse arm: a shorter name is not covered by a longer documented one.

        ``secrets.KITTY_EGRESS`` is a prefix of the documented
        ``secrets.KITTY_EGRESS_JSON``.  Under substring containment it would
        arrive in CI undocumented with the guard green — the same trap the
        version arm avoids by matching whole tokens, in the other direction.
        """
        planted = dict(ci_artifacts)
        planted["ci.yml"] = "value: ${{ secrets.KITTY_EGRESS }}"

        reverse = arm(inventory_discrepancies(suite_markdown, planted), "reverse")

        assert any("KITTY_EGRESS'" in message for message in reverse), (
            f"the reverse arm absorbed a shorter name into a longer one; it reported {reverse}"
        )

    def test_a_log_row_deleted_from_the_inventory_is_reported(
        self, suite_markdown: str, ci_launcher: str
    ) -> None:
        """Log arm: the row carrying the no-artifact constraint is removed.

        Its binding is neither a secret reference nor a version, so before this
        arm existed the deletion passed every other check — measured.
        """
        without = "\n".join(
            line for line in suite_markdown.splitlines() if "kitty-bridge-debug.log" not in line
        )

        problems = launcher_log_discrepancies(without, ci_launcher)

        assert any("kitty-bridge-debug.log" in problem for problem in problems), (
            f"deleting the debug-log row reported {problems}"
        )

    def test_a_launcher_that_stops_writing_a_log_is_reported(
        self, suite_markdown: str, ci_launcher: str
    ) -> None:
        """Forward arm over the launcher, not over the module that describes it.

        ``configure_kitty.py`` mentions both log names in its own prose, so a
        forward arm reading the module source stays green when ``wrapper_body``
        stops emitting them.  This drives the real reporter over a launcher with
        the flag removed.
        """
        crippled = re.sub(r'--debug-file\s+"[^"]*"', "", ci_launcher)
        assert "kitty-bridge-debug.log" not in crippled, "the mutant did not remove the flag"
        artifacts = {CONFIGURE_KITTY.relative_to(ROOT).as_posix(): crippled}

        forward = arm(inventory_discrepancies(suite_markdown, artifacts), "forward")

        assert any("kitty-bridge-debug.log" in message for message in forward), (
            f"a launcher that stopped writing the debug log reported {forward}"
        )

    def test_a_removed_fork_guard_is_reported(self, suite_markdown: str) -> None:
        """Fork arm: the guard the per-PR/nightly split rests on is deleted."""
        problems = fork_guard_discrepancies(
            suite_markdown,
            REVIEW_WORKFLOW.read_text(encoding="utf-8").replace(_FORK_GUARD, "true"),
        )

        assert any("does not carry the guard" in problem for problem in problems), (
            f"a removed fork guard reported {problems}"
        )

    def test_a_guard_kept_only_on_another_job_is_reported(self, suite_markdown: str) -> None:
        """Fork arm: the comparison survives, on a job that does not run the review.

        The plausible drift is a second job — a retry or a notice — carrying the
        same condition while the review job loses it.
        """
        workflow = """
jobs:
  notice:
    if: github.event.pull_request.head.repo.full_name == github.repository
    steps:
      - run: echo notice
  review:
    steps:
      - uses: anthropics/claude-code-action@v1
"""

        problems = fork_guard_discrepancies(suite_markdown, workflow)

        assert any("does not carry the guard" in problem for problem in problems), (
            f"a guard moved off the review job reported {problems}"
        )

    def test_an_unguarded_second_review_job_is_reported(self, suite_markdown: str) -> None:
        """Fork arm: a retry job runs the review too, and forgot the guard.

        Every review job must carry it — not merely one of them.  A check that
        passed when *any* review job was guarded would let this fork run through
        the second job.
        """
        workflow = """
jobs:
  review:
    if: github.event.pull_request.head.repo.full_name == github.repository
    steps:
      - uses: anthropics/claude-code-action@v1
  retry:
    steps:
      - uses: anthropics/claude-code-action@v1
"""

        problems = fork_guard_discrepancies(suite_markdown, workflow)

        assert any("does not carry the guard" in problem for problem in problems), (
            f"an unguarded retry job reported {problems}"
        )

    def test_an_unrelated_unguarded_job_is_not_reported(self, suite_markdown: str) -> None:
        """Fork arm, the false-positive direction: only review jobs need the guard.

        A ``build`` job that never runs the review has no reason to refuse forks.
        Flagging it would turn the gate red and name the wrong job — the
        mis-diagnosis this repository's guards exist to prevent.
        """
        workflow = """
jobs:
  build:
    steps:
      - run: make
  review:
    if: github.event.pull_request.head.repo.full_name == github.repository
    steps:
      - uses: anthropics/claude-code-action@v1
"""

        assert fork_guard_discrepancies(suite_markdown, workflow) == []

    def test_an_action_merely_sharing_the_prefix_is_not_the_review_action(
        self, suite_markdown: str
    ) -> None:
        """Fork arm, the false-positive direction: a prefix is not an identity.

        A fork or a local wrapper named ``claude-code-action-…`` is a different
        action; requiring the guard of its job would name the wrong one.
        """
        workflow = """
jobs:
  experiment:
    steps:
      - uses: anthropics/claude-code-action-fork@v1
  review:
    if: github.event.pull_request.head.repo.full_name == github.repository
    steps:
      - uses: anthropics/claude-code-action@v2
"""

        assert fork_guard_discrepancies(suite_markdown, workflow) == []

    def test_a_workflow_with_no_review_job_is_reported(self, suite_markdown: str) -> None:
        """Fork arm: nothing runs the review, so nothing is guarded.

        Without this, renaming or vendoring the action would leave zero review
        jobs, zero conditions to check, and a green arm.
        """
        problems = fork_guard_discrepancies(suite_markdown, "jobs:\n  build:\n    steps: []\n")

        assert any("no job runs" in problem for problem in problems), (
            f"a workflow with no review job reported {problems}"
        )

    def test_a_fork_guard_weakened_by_a_disjunct_is_reported(self, suite_markdown: str) -> None:
        """Fork arm: the condition is still there, and no longer excludes forks.

        The shape a substring scan over the workflow cannot see — the literal is
        present, so the arm stays green while the guard admits what §8.6 says it
        excludes.
        """
        weakened = REVIEW_WORKFLOW.read_text(encoding="utf-8").replace(
            _FORK_GUARD, f"{_FORK_GUARD} || github.actor == 'someone'"
        )

        problems = fork_guard_discrepancies(suite_markdown, weakened)

        assert any("weakened by a disjunct" in problem for problem in problems), (
            f"a guard re-admitting forks reported {problems}"
        )

    def test_a_document_that_stops_quoting_the_fork_guard_is_reported(self) -> None:
        """Fork arm, the other direction: the claim leaves §8.6.

        Every other fork case doctors the *workflow*; without this one the
        ``markdown`` half of the check could be deleted with the suite green.
        """
        problems = fork_guard_discrepancies(
            "# A design document that makes no fork claim\n",
            REVIEW_WORKFLOW.read_text(encoding="utf-8"),
        )

        assert any("no longer quotes the guard" in problem for problem in problems), (
            f"a document that dropped the fork claim reported {problems}"
        )

    def test_a_pull_request_target_trigger_is_reported(self, suite_markdown: str) -> None:
        """Fork arm: the documented wrong way to 'fix' a skipped fork run.

        ``pull_request_target`` runs a fork's pull request in the base branch's
        context **with** secrets, which would make §8.6's fork paragraph false
        while every binding still matched.
        """
        problems = fork_guard_discrepancies(
            suite_markdown,
            REVIEW_WORKFLOW.read_text(encoding="utf-8").replace(
                "on:\n  pull_request:", "on:\n  pull_request_target:", 1
            ),
        )

        assert any(_PULL_REQUEST_TARGET in problem for problem in problems), (
            f"a pull_request_target trigger reported {problems}"
        )

    def test_an_unguarded_job_binding_a_kitty_secret_is_reported(self) -> None:
        """Kitty-job fork arm: a new consumer of the secrets forgot the guard."""
        workflow = """
jobs:
  live:
    steps:
      - env:
          KITTY_CREDENTIALS_JSON: ${{ secrets.KITTY_CREDENTIALS_JSON }}
        run: configure
"""

        problems = kitty_job_fork_discrepancies(workflow)

        assert any("'live'" in problem and "does not carry" in problem for problem in problems), problems

    def test_a_kitty_job_guard_weakened_by_a_disjunct_is_reported(self) -> None:
        """Kitty-job fork arm: the literal is present and no longer excludes forks."""
        workflow = f"""
jobs:
  live:
    if: {_FORK_GUARD} || github.actor == 'someone'
    env:
      PROFILES: ${{{{ vars.KITTY_PROFILES_JSON }}}}
    steps: []
"""

        problems = kitty_job_fork_discrepancies(workflow)

        assert any("disjunct" in problem for problem in problems), problems

    def test_a_kitty_workflow_on_pull_request_target_is_reported(self) -> None:
        """Kitty-job fork arm: the trigger that gives a fork the secrets despite the guard."""
        workflow = f"""
on:
  pull_request_target:
jobs:
  live:
    if: {_FORK_GUARD}
    steps:
      - env:
          KITTY_EGRESS_JSON: ${{{{ secrets.KITTY_EGRESS_JSON }}}}
        run: configure
"""

        problems = kitty_job_fork_discrepancies(workflow)

        assert any(_PULL_REQUEST_TARGET in problem for problem in problems), problems

    def test_a_guarded_kitty_job_and_an_unrelated_job_are_not_reported(self) -> None:
        """Kitty-job fork arm, the false-positive direction: only binding jobs need the guard."""
        workflow = f"""
jobs:
  build:
    steps:
      - run: make
  live:
    if: >-
      github.event.pull_request.draft == false &&
      {_FORK_GUARD}
    steps:
      - env:
          KITTY_EGRESS_JSON: ${{{{ secrets.KITTY_EGRESS_JSON }}}}
        run: configure
"""

        assert kitty_job_conditions(workflow).keys() == {"live"}
        assert kitty_job_fork_discrepancies(workflow) == []
