"""KBR-216 — what CI supplies is stated in one place, and the workflows agree with it.

``.github/workflows/claude-code-review.yml`` runs Claude Code through a released
Kitty Bridge to review every same-repository pull request.  To do that the CI
environment was given a complete kitty installation: a version-pinned Claude Code
CLI, a profile store, a credential store, an egress gateway and two logs.  All of
it has been there for weeks.

``.system_design/TEST_SUITE.md`` did not know.  §11 carried **Q12** — *"How is a
pinned Claude Code binary supplied to CI?"* — as an open question to the product
owner, and §6.4.2 reasoned from the assumption that the answer might be "it
cannot be", while the answer sat in the workflow tree.  Three implementation
tasks were flagged ``blocked Q12`` against a blocker that had already cleared.

§8.6 now states the inventory, and this file is what stops the two drifting apart
again.  Four arms, because fewer would not have caught what this file was written
for:

* **forward** — every binding §8.6 names is really used by the file §8.6 names it
  in, so the document cannot promise a capability CI does not have;
* **reverse** — every ``secrets.KITTY_*`` / ``vars.KITTY_*`` binding the workflows
  actually use appears as a row, so a capability cannot be added to CI and left
  out of the design;
* **version** — the CLI pin §8.6 quotes is the one CI installs, in both
  directions, since that pairing is maintained by hand against a floating action
  tag and is the arm most likely to fire in anger;
* **fork** — the guard §8.6's per-PR/nightly split rests on is still in place.
  Its removal changes no binding, so every other arm would stay green while the
  section's most load-bearing paragraph became false.

A forward-only check would go green on the day someone deletes a secret from the
workflow *and* from the table; the reverse arm is what makes the inventory a
statement about CI rather than a statement about itself.

**The comparison is a pure reporter, not an assertion.**
:func:`inventory_discrepancies` returns a list of human-readable strings and an
empty list means agreement.  That shape is what lets the falsification cases below
drive the *real* scan over a doctored document — watching a rule fail by hand is
not the same as shipping its negative control, and a matcher exercised only in
isolation proves nothing about its assembly.

Two artifacts edited separately that must agree, both readable statically — the
§6.2.3 "docs ⇄ code" case — so this file is ``l2`` and is named in
``tests/test_layer_selection.py``'s allowlist.

**One stated limit.** The reverse arm scans the ``KITTY_``-prefixed namespace
only.  ``secrets.ANTHROPIC_AUTH_TOKEN`` and ``secrets.OPENROUTER_API_KEY`` also
appear in the tree and are deliberately not inventory rows: neither is a kitty
capability, and widening the arm to every secret would make the table a mirror of
the workflow's secret list rather than a design statement about what the test
suite may rely on.
"""

from __future__ import annotations

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
#: handed the exact version as its one positional argument.
_CLI_PIN = re.compile(r"install\.sh\s*\|\s*bash\s+-s\s+--\s+(?P<version>[0-9][0-9A-Za-z.\-+]*)")

#: A bare version token in the document, so `2.1.23` cannot satisfy a row
#: quoting `2.1.238`. Whole-token, never substring.
_VERSION_TOKEN = re.compile(r"\b[0-9]+(?:\.[0-9A-Za-z\-+]+)+\b")

#: The condition that keeps a fork's pull request from ever starting the review
#: job. §8.6's whole per-PR/nightly split rests on this one line.
_FORK_GUARD = "github.event.pull_request.head.repo.full_name == github.repository"

#: The trigger that would undo it by running a fork's code in the base branch's
#: context, secrets included. The workflow refuses it in prose; this is the check.
_PULL_REQUEST_TARGET = "pull_request_target"


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

    Three arms, tagged rather than merely worded, so a caller can ask about one
    of them without matching on prose:

    * ``"section"`` — the section exists and has rows at all;
    * ``"forward"`` — each row's binding appears in the artifact the row names;
    * ``"reverse"`` — each kitty binding any artifact uses appears in some row.

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
    quoted = {token for _, _, binding in rows for token in _VERSION_TOKEN.findall(binding)}
    for version in sorted(installed - quoted):
        problems.append(
            ("version", f"CI installs Claude Code {version}, which no {_INVENTORY_HEADING} row quotes")
        )
    for version in sorted(quoted - installed):
        problems.append(
            ("version", f"{_INVENTORY_HEADING} quotes Claude Code {version}, which CI does not install")
        )

    return problems


def workflow_triggers(text: str) -> set[str]:
    """Return the event names a workflow's ``on:`` block declares.

    Handles PyYAML's quirk of parsing the bare key ``on`` as the boolean
    ``True``, the same way ``tests/test_github_actions.py`` does.

    Args:
        text: The full text of one workflow file.

    Returns:
        The declared trigger names, empty when the block is absent or unparseable.
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

    # The workflow must still refuse a fork run, and §8.6 must quote the same line.
    if _FORK_GUARD not in review_workflow:
        problems.append(f"the review workflow no longer carries the guard {_FORK_GUARD!r}")
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


def arm(problems: list[tuple[str, str]], name: str) -> list[str]:
    """Return the messages one arm of the reporter produced.

    Selecting by tag rather than by matching on message text is what keeps a
    test's claim stable when a diagnostic is reworded.

    Args:
        problems: The reporter's full result.
        name: The arm to select — ``"section"``, ``"forward"`` or ``"reverse"``.

    Returns:
        That arm's messages, in the reporter's order.
    """
    return [message for tag, message in problems if tag == name]


@pytest.fixture(scope="module")
def suite_markdown() -> str:
    """Return the text of ``TEST_SUITE.md``.

    Returns:
        The design document's full contents.
    """
    return TEST_SUITE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ci_artifacts() -> dict[str, str]:
    """Return every CI artifact the inventory may name, keyed by relative path.

    Covers the whole workflow directory rather than only the review workflow, so
    the reverse arm sees a kitty binding introduced in a *new* workflow — which
    is exactly how the next capability will arrive.

    Returns:
        Repository-relative POSIX path to file text.
    """
    paths = sorted(WORKFLOWS_DIR.glob("*.yml"))
    paths.append(ROOT / ".github" / "review" / "scripts" / "configure_kitty.py")
    return {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8") for path in paths
    }


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

    def test_the_review_job_still_refuses_a_fork_pull_request(self, suite_markdown: str) -> None:
        """§8.6's per-PR/nightly split rests on this guard, so the guard is checked.

        The third direction.  Neither the forward nor the reverse arm would
        notice the guard's removal — no ``KITTY_*`` binding changes — and the
        fork asymmetry is the most load-bearing sentence §8.6 adds.
        """
        problems = fork_guard_discrepancies(
            suite_markdown, REVIEW_WORKFLOW.read_text(encoding="utf-8")
        )

        assert problems == [], "§8.6's fork claim no longer matches the workflow:\n" + "\n".join(
            problems
        )


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

    def test_a_removed_fork_guard_is_reported(self, suite_markdown: str) -> None:
        """Fork arm: the guard the per-PR/nightly split rests on is deleted."""
        problems = fork_guard_discrepancies(
            suite_markdown,
            REVIEW_WORKFLOW.read_text(encoding="utf-8").replace(_FORK_GUARD, "true"),
        )

        assert any("no longer carries the guard" in problem for problem in problems), (
            f"a removed fork guard reported {problems}"
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
