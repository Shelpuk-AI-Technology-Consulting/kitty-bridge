"""Contract guard — the committed corpus, the README and the code must agree.

``.system_design/TEST_SUITE.md`` §7.1 · plan task **T-W6** (KBR-29).

Three artifacts are edited separately from the corpus module and must agree with
it, so each gets an L2 guard — the division
:mod:`tests.harness.test_register_agreement` established:

* **the committed entries** — the lint plan §1.4 names, which fails the gate when
  a credential or a personal identifier reaches the repository;
* **the capture procedure** — ``tests/corpus/README.md`` describes the tool, and a
  procedure that has stopped describing it is how the tool stops being used
  correctly;
* **the source tree** — the corpus produces the *inbound* half of every oracle
  comparison, so §3.3.1's independent-oracle rule applies to it with full force.

Every test here reads a real file, which is what puts them at L2 rather than
beside the format tests in :mod:`tests.harness.test_corpus`.  The falsification
cases that need a deliberately broken corpus build one under ``tmp_path`` and
live there; what cannot be done there is asserting on what is *actually
committed*, which is this module's whole job.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from harness import corpus as k

pytestmark = pytest.mark.l2

#: The repository root, three levels up from this file.
ROOT = Path(__file__).resolve().parent.parent.parent

#: The committed corpus.
CORPUS = ROOT / "tests" / "corpus"

#: The capture procedure.
README = CORPUS / "README.md"


class TestTheCommittedCorpus:
    """The lint plan §1.4 names: an unscrubbed fixture fails CI."""

    def test_every_entry_loads(self) -> None:
        """A malformed manifest must fail here, not at the first test that reads it."""
        assert k.load_corpus(CORPUS)

    def test_no_entry_carries_a_credential_or_an_identifier(self) -> None:
        """The gate.

        `assert_corpus_clean` raises on an empty corpus as well as a dirty one,
        so this cannot pass by finding nothing to look at — which also makes a
        wrong `CORPUS` path a loud failure instead of a permanent green.
        """
        k.assert_corpus_clean(k.load_corpus(CORPUS))

    def test_the_corpus_collects_no_tests(self) -> None:
        """`testpaths = ["tests"]` includes this directory.

        A `.py` file here would be collected, would take `l1` by path default,
        and would put fixture data in the gate as a test module.
        """
        assert list(CORPUS.glob("**/*.py")) == []

    def test_every_body_file_belongs_to_an_entry(self) -> None:
        """An orphaned `.body` is a fixture nothing loads — and nothing lints.

        The lint reads bodies *through* the manifests, so a stray file is a
        credential-bearing file in the repository that `assert_corpus_clean`
        would never open.
        """
        claimed = {f"{entry.id}.body" for entry in k.load_corpus(CORPUS)}

        assert {path.name for path in CORPUS.glob("*.body")} == claimed


#: The three T-C4 entries (KBR-47). Their manifest-level claims are contract
#: claims — what the corpus format demands of a synthetic entry — so they sit
#: here at L2 with the rest of this module, not beside the behaviour-level
#: wiring tests in `tests/bridge/test_tc4_corpus_wiring.py` (that file is L1
#: and drives the entries through the real compactor).
TC4_ENTRY_IDS = (
    "m6_recovery_oversized_paired",
    "m5_irreducible_single_final_turn",
    "system_prompt_over_window_compacts_normally",
)

#: The five T-C1 entries (KBR-44). Four captured from real Claude Code 2.1.238
#: driven against the local recorder; the fifth (`no_output_config`) is the
#: P5f synthetic complement. Their wiring-level claims (what each entry's
#: triggers_met/triggers_absent declare, and how they were scrubbed) live in
#: this module's freshness + manifest tests, not in a per-task wiring module —
#: T-C1's entries are oracle *input*, not behaviour triggers, so the L2 lint
#: is the right home for their contract-level claims.
TC1_ENTRY_IDS = (
    "plain_turn",
    "tools_declared",
    "tool_use_and_tool_result",
    "effort_configured",
    "no_output_config",
)

#: The two P25 synthetic entries (KBR-185 / KBR-44's capture pass). The
#: `ALLOWLISTED_FIELD_IS_FALSY` trigger case and complement for a Responses-
#: wire body — synthetic because no Responses-format client exists in this
#: repository's capture environment.
P25_ENTRY_IDS = (
    "allowlisted_field_falsy",
    "allowlisted_field_absent",
)

#: The four T-C3 entries (KBR-46). Two synthetic pairs that pin the M3
#: tool-result-truncation boundary (50 000 / 50 001 chars) and the M5
#: compaction-budget boundary (the static `_COMPACTION_CHAR_THRESHOLD`).
#: Their boundary-position properties (CC-converted length exactly on the
#: claimed side) live in `tests/harness/test_corpus_thresholds.py`; what is
#: asserted here is only that the entry exists and is wired into the corpus.
TC3_ENTRY_IDS = (
    "tool_result_under_limit",
    "tool_result_over_limit",
    "compaction_budget_under",
    "compaction_budget_over",
)

#: Every committed entry must be owned by a documented task. `format_example`
#: is T-W6's worked example and stays (its own origin_note records why).
#: When a new task adds entries, add its IDs here AND to its wiring/lint
#: module — an entry in the corpus without an owner here is fixture data no
#: test loads, no lint scans by id, and no task owns.
OWNED_ENTRY_IDS = {
    "format_example",
    *TC4_ENTRY_IDS,
    *TC1_ENTRY_IDS,
    *P25_ENTRY_IDS,
    *TC3_ENTRY_IDS,
}


class TestTheTc4Entries:
    """KBR-47's manifest-level claims, per the ticket's second comment."""

    @pytest.mark.parametrize("entry_id", TC4_ENTRY_IDS)
    def test_each_is_synthetic_with_a_recorded_construction(self, entry_id: str) -> None:
        """`origin: synthetic`, and the note says both why and how constructed.

        A maintainer reading the manifest must not mistake 600 KB of filler
        for a capture, and must know what replaced the human-review step for
        a body too large to read by eye.
        """
        entry = k.load_entry(CORPUS / f"{entry_id}.json")

        assert entry.origin == "synthetic"
        note = entry.origin_note.lower()
        assert "constructed" in note, "the note must say the size is constructed"
        assert "review" in note, "the note must name the review-step replacement (§7.1.1)"

    @pytest.mark.parametrize("entry_id", TC4_ENTRY_IDS)
    def test_each_declares_no_triggers(self, entry_id: str) -> None:
        """M6 is loader-refused and M5's budget is profile-derived (§7.1).

        Silence is the honest state for these entries: the triggers are
        decided at the test that resolves them, which is the L1 wiring module
        and the L3 scripted-recorder test.
        """
        entry = k.load_entry(CORPUS / f"{entry_id}.json")

        assert entry.triggers_met == frozenset()
        assert entry.triggers_absent == frozenset()

    @pytest.mark.parametrize("entry_id", TC4_ENTRY_IDS)
    def test_each_is_an_inbound_messages_request(self, entry_id: str) -> None:
        """The corpus stores inbound requests — Claude Code's wire format."""
        entry = k.load_entry(CORPUS / f"{entry_id}.json")

        assert entry.request.path == "/v1/messages"
        assert entry.request.method == "POST"


class TestTheCorpusIsProtectedFromLineEndingTranslation:
    """`.gitattributes` is the first line of defence; the digest is the second."""

    def test_gitattributes_still_covers_the_corpus(self) -> None:
        """Nothing else notices a dropped pattern or a renamed directory.

        The digest catches a body rewritten *without* its manifest — but a
        checkout that rewrote both would pass everywhere, and a pattern silently
        no longer matching is exactly how that happens. Every other artifact that
        must agree with the corpus has a guard here; this one had none.

        Asserted against the paths rather than the file's presence, because a
        `.gitattributes` that exists and no longer names `tests/corpus` protects
        nothing while looking like it does.
        """
        text = (ROOT / ".gitattributes").read_text(encoding="utf-8")

        assert "tests/corpus/*.body -text" in text
        assert "tests/corpus/*.json -text" in text

    def test_the_patterns_name_the_directory_the_corpus_actually_uses(self) -> None:
        """The pair above is a constant unless something anchors it to reality.

        If the corpus ever moves, the patterns above still match themselves and
        the guard stays green over a directory nothing writes to.
        """
        assert CORPUS == ROOT / "tests" / "corpus"
        assert list(CORPUS.glob("*.body"))

    def test_no_committed_manifest_carries_crlf(self) -> None:
        """The checkout end is pinned; this pins the write end (KBR-261).

        `write_entry` used ``Path.write_text`` with default newline translation,
        which emits LF on Linux and CRLF on Windows regardless of
        ``.gitattributes``. A Windows contributor running the regen script
        produced CRLF manifests that CI then compared as a spurious
        LF-vs-committed byte diff — the failure the L1 roundtrip test
        documented as "rerun the regen script" when the real problem was
        the writer, not the bytes.

        The fix is platform-neutral: ``write_entry`` now passes
        ``newline="\\n"`` so its bytes are LF everywhere. This test is the
        direct counterpart — read every committed ``.json`` with
        ``read_bytes()`` (text mode would silently translate ``\\r\\n`` back
        to ``\\n`` on a Windows runner and mask the very drift it exists to
        catch) and refuse the file if any pair ever lands.
        """
        offenders = [
            path.name
            for path in sorted(CORPUS.glob("*.json"))
            if b"\r\n" in path.read_bytes()
        ]

        assert offenders == [], (
            f"committed corpus manifests carry CRLF line endings: {offenders}; "
            "the .gitattributes -text rule stops git from rewriting on checkout "
            "but write_entry must also emit LF — see KBR-261"
        )


class TestTheProcedureDescribesTheTool:
    """A capture procedure that has stopped describing the scrubber is a trap."""

    def test_every_pattern_class_is_named(self) -> None:
        """Renaming or adding a class without telling the operator fails here.

        The README's table is what a maintainer reads before deciding whether a
        capture is safe to commit; a class missing from it is a protection they
        will not know they do not have.
        """
        text = README.read_text(encoding="utf-8")
        missing = [name for name in (*k.PATTERN_NAMES, k.LITERAL_CLASS) if name not in text]

        assert missing == []

    def test_the_owner_and_cadence_are_recorded(self) -> None:
        """Plan §3 makes naming them part of this task's deliverable.

        §7.1's warning is that an un-refreshable corpus becomes a museum of a
        protocol nobody speaks; an unnamed owner is how it becomes one.
        """
        text = README.read_text(encoding="utf-8")

        assert "Owner:" in text
        assert "Cadence:" in text

    def test_both_capture_shapes_are_described(self) -> None:
        """§7.1 names two, and an operator following only the first captures half.

        The inbound request is what the oracle compares; the native baseline is
        what design channels C1b and C5 compare against. Without this, the
        native-baseline section could be deleted and nothing would notice.
        """
        text = README.read_text(encoding="utf-8")

        assert "ANTHROPIC_BASE_URL" in text
        assert "native baseline" in text.lower()
        assert "T-C7" in text

    def test_the_review_step_is_mandatory_in_the_procedure(self) -> None:
        """The scrubber is a net under the review, not a replacement for it."""
        assert "Read every `.body` file" in README.read_text(encoding="utf-8")

    def test_the_rotation_step_is_recorded(self) -> None:
        """Once a credential is in a public repository, scrubbing the tree is not the fix.

        The remedy is rotation, and the procedure has to say so where somebody
        who has just made the mistake will find it.
        """
        assert "rotate" in README.read_text(encoding="utf-8").lower()


class TestTheProcedureListsTheThresholdPairs:
    """KBR-46 (T-C3): the procedure must name every entry id it ships.

    AC-4 in ``.requirements/20260914T185833Z_kbr46_corpus_thresholds/REQUIREMENTS.md``
    guards the threshold-pair section by read — that "by read" is exactly the
    silent regression this guard replaces.
    """

    def test_every_threshold_pair_entry_is_named(self) -> None:
        """The four KBR-46 entry ids must each appear in the README's prose."""
        text = README.read_text(encoding="utf-8")
        for entry_id in (
            "tool_result_under_limit",
            "tool_result_over_limit",
            "compaction_budget_under",
            "compaction_budget_over",
        ):
            assert entry_id in text, (
                f"the README does not name the entry id {entry_id!r}; the T-C3 section "
                "documents each id and a future revision dropped one"
            )


class TestTheCommittedCorpusIsFresh:
    """The refresh cadence is enforced, not just written down.

    Two workflows install Claude Code (`claude-code-review.yml:785` and
    `tmux-disconnect.yml:98`); both spell the pin as ``bash -s -- X.Y.Z``. The
    guard fails when either drifts, when the two disagree, or when the
    committed corpus's captured entries no longer name that version. Reading
    the pin from the workflow file — not a second copy of ``2.1.238`` in the
    test — is what stops the two from diverging silently.
    """

    #: The exact shape the install line takes. Anchored on the installer URL
    #: and ``bash -s --`` so a generic ``pip install X.Y.Z`` elsewhere cannot
    #: be mistaken for the pin. The version arm is a strict semver triple
    #: (no pre-release / build tags) followed by a boundary
    #: (``\\s|"\\|'`` — the quote closes the shell's argument) so a relaxed
    #: match does not greedily consume ``2.1.238`` out of ``2.1.238-rc1``.
    _INSTALL_LINE = re.compile(
        r"claude\.ai/install\.sh\s*\|\s*bash\s+-s\s+--\s+(?P<version>\d+\.\d+\.\d+)(?=[\s\"']|\Z)"
    )

    def _workflow_paths(self) -> tuple[Path, Path]:
        """Both workflow files that pin the CLI. Two sites, one pin."""
        return (
            ROOT / ".github" / "workflows" / "claude-code-review.yml",
            ROOT / ".github" / "workflows" / "tmux-disconnect.yml",
        )

    def _pin_from(self, path: Path) -> str:
        """Return the pin named by ``path``'s install line, or raise.

        Args:
            path: A workflow file expected to contain the install line.

        Returns:
            The bare semver triple.

        Raises:
            AssertionError: When the install line is missing or its version
                is not a strict ``X.Y.Z`` triple.
        """
        text = path.read_text(encoding="utf-8")
        match = self._INSTALL_LINE.search(text)
        assert match is not None, (
            f"{path.name} no longer carries the Claude Code install line "
            "(`curl -fsSL https://claude.ai/install.sh | bash -s -- X.Y.Z`); "
            "the freshness guard cannot read its pin"
        )
        return match.group("version")

    def _assert_pins_agree(self, review_path: Path, tmux_path: Path) -> str:
        """Assert both workflow files name the same pin and return it.

        Args:
            review_path: One workflow file expected to carry the install line.
            tmux_path: The other.

        Returns:
            The pin both name.

        Raises:
            AssertionError: When either install line is missing or the two
                pins disagree.
        """
        review_pin = self._pin_from(review_path)
        tmux_pin = self._pin_from(tmux_path)

        assert review_pin == tmux_pin, (
            f"the two workflows disagree on the Claude Code pin: "
            f"{review_path.name} installs {review_pin!r}, {tmux_path.name} installs {tmux_pin!r}. "
            "Update both to the same version."
        )
        return review_pin

    def test_both_workflows_install_the_same_pinned_version(self) -> None:
        """One pin, two sites — the guard reads both and asserts they agree.

        A bump of one and not the other leaves the corpus "fresh" against a
        pin that no longer describes the CLI CI actually runs. A bump of
        **both** passes here by design — that case is the corpus guard's
        (`test_the_committed_corpus_passes_the_freshness_guard`), whose
        captured entries name the old pin. No version literal lives in this
        file: two copies of one value is how they drift, which is what this
        guard exists to prevent.
        """
        review, tmux = self._workflow_paths()
        self._assert_pins_agree(review, tmux)


    def test_a_workflow_whose_install_line_disappears_fails_loudly(self, tmp_path: Path) -> None:
        """A reformat that drops the literal must surface, not pass vacuously.

        The whole-token regex (``tests/test_ci_capability_inventory.py:103``)
        documents this same shape: a guard that stringified the version would
        quietly lose its anchor.
        """
        # Build a workflow whose pin line is gone (the version is now a comment).
        broken = tmp_path / "workflow.yml"
        broken.write_text(
            "# review installs claude-code but the literal is commented out\n"
            "# bash -s -- 2.1.238\n",
            encoding="utf-8",
        )

        with pytest.raises(AssertionError, match="install line"):
            self._pin_from(broken)

    def test_a_workflow_pinning_a_non_semver_version_is_refused(self, tmp_path: Path) -> None:
        r"""A relaxed version arm would let ``2.1.238-rc1`` slip through.

        The pin's contract is ``X.Y.Z``; the workflow's literal is the
        canonical form. A pre-release tag is a different pin — the corpus
        should not pretend otherwise. The strict ``\d+\.\d+\.\d+`` arm with
        a trailing boundary refuses to match it, which the helper reports
        as "no install line" because the install line *as parsed* does not
        exist.
        """
        relaxed = tmp_path / "workflow.yml"
        relaxed.write_text(
            "curl -fsSL https://claude.ai/install.sh | bash -s -- 2.1.238-rc1\n",
            encoding="utf-8",
        )

        with pytest.raises(AssertionError, match="install line"):
            self._pin_from(relaxed)

    def test_a_workflow_whose_pin_disagrees_with_its_twin_fails_loudly(self, tmp_path: Path) -> None:
        """The forward direction of the agreement check — the harder half.

        ``tests/test_ci_capability_inventory.py:838-843`` documents that a
        substring compare cannot catch this: ``bash -s -- 2.1.238`` contains
        ``bash -s -- 2.1.23``. The whole-token regex and the strict version
        arm together close that gap.

        Two ``tmp_path`` copies pin to disagreeing versions — no mutation of
        the real workflows, so a SIGKILL'd ``pytest`` cannot leave a drifted
        tree behind.
        """
        review = tmp_path / "claude-code-review.yml"
        review.write_text(
            "curl -fsSL https://claude.ai/install.sh | bash -s -- 2.1.238\n",
            encoding="utf-8",
        )
        tmux = tmp_path / "tmux-disconnect.yml"
        tmux.write_text(
            "curl -fsSL https://claude.ai/install.sh | bash -s -- 2.1.9\n",
            encoding="utf-8",
        )

        with pytest.raises(AssertionError, match="disagree"):
            self._assert_pins_agree(review, tmux)

    def test_the_committed_corpus_passes_the_freshness_guard(self) -> None:
        """Binds the workflow artifacts to the corpus — the guard's whole point.

        Red until the corpus ships its first captured entry (the guard raises
        "no captured entries" by design); green the moment T-C1 commits its
        five.
        """
        review, _ = self._workflow_paths()
        pin = self._pin_from(review)

        k.assert_captured_from_matches_pin(k.load_corpus(CORPUS), pin)


class TestTheCommittedCorpusHasNoOrphanEntries:
    """Every committed entry must be owned by a documented task.

    The per-task wiring tests (T-C1's L2 lint here, T-C4's
    ``tests/bridge/test_tc4_corpus_wiring.py``, P25's manifest checks)
    each assert *their* entries are committed. The complementary
    guarantee — that *no* other entry is — has to live somewhere with a
    cross-task allowlist, which is here.

    An entry in the corpus without an owner is fixture data no test loads,
    no lint scans by id, and no task owns — exactly the shape
    ``format_example``'s own origin_note warns about. ``OWNED_ENTRY_IDS``
    is the allowlist; adding a new task's entries goes there AND to that
    task's own wiring or lint module.
    """

    def test_every_committed_entry_is_owned(self) -> None:
        """No orphan entries — the corpus is the union of known task sets.

        The negation of this assertion (``unexpected: [...]``) names the
        orphans so the maintainer can decide whether to add them to the
        allowlist or delete them.
        """
        committed = {entry.id for entry in k.load_corpus(CORPUS)}

        orphans = sorted(committed - OWNED_ENTRY_IDS)
        assert orphans == [], (
            f"orphan corpus entries (no documented task owns them): {orphans}; "
            "either add them to OWNED_ENTRY_IDS in tests/harness/test_corpus_lint.py "
            "or delete them from tests/corpus/"
        )

    def test_every_owned_entry_is_committed(self) -> None:
        """No entry on the allowlist is missing from the corpus.

        The inverse check catches a task that documented entries in the
        allowlist but forgot to commit them — the wiring would still pass
        otherwise.
        """
        committed = {entry.id for entry in k.load_corpus(CORPUS)}

        missing = sorted(OWNED_ENTRY_IDS - committed)
        assert missing == [], (
            f"owned entries not committed: {missing}; an entry in OWNED_ENTRY_IDS "
            "but not in tests/corpus/ is a documented entry with no fixture"
        )


class TestTheCorpusIsAnIndependentOracle:
    """§3.3.1: the oracle must not be written in terms of the code under test."""

    def test_the_module_imports_nothing_from_kitty(self) -> None:
        """Asserted structurally, as `test_contract.py` asserts it for the contract.

        The corpus produces the *inbound* half of every comparison. A corpus that
        asked kitty how to read a body would inherit kitty's bugs, and I1 would
        prove only self-consistency. Prose in a docstring cannot enforce that;
        this can.
        """
        tree = ast.parse((ROOT / "tests" / "harness" / "corpus.py").read_text(encoding="utf-8"))

        walked = list(ast.walk(tree))
        imported = {
            node.module.split(".")[0] for node in walked if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name.split(".")[0]
            for node in walked
            if isinstance(node, ast.Import)
            for alias in node.names
        }

        assert "kitty" not in imported
