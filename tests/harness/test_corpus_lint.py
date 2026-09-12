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
