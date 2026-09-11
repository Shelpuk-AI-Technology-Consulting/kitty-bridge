"""Contract guard — the register data, the design document and the source tree must agree.

``.system_design/TEST_SUITE.md`` §6.2.3 · plan task **T-W3** (KBR-26).

Two artifacts must agree with the register data and are edited separately from
it, so each gets an L2 contract guard:

* **the design document** — §3.2.1 and §3.2.2 are the register's published form,
  and §3.2's own "register maintenance" paragraph makes adding a row the way a
  mutation gets introduced.  A reviewer reads the markdown; the oracle reads the
  data.  If those disagree the review is of a different artifact than the test.
* **the source tree** — every row names the site that performs its mutation.  A
  rename that does not reach the register leaves a row pointing at nothing, and
  the register goes on looking complete.

Every test here reads a real artifact, which is what puts them at L2 rather than
beside the schema tests in :mod:`tests.harness.test_register`.  What these guards
reconcile is **ids and conditionality only**; §3.2's tables carry no path column
and their Site cells are prose, so paths and sites are checked against the source
tree instead of against the document.  §3.2.4 records that division of authority.

Per §6.2 every guard asserts that its own scan finds known positives, and per
plan §1.4 each ships with a deliberate defect it must detect.  The functions
under test are pure — they take the markdown text and the row tuple as arguments
— precisely so a defect can be handed to them without editing a file on disk.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from harness import register as r

pytestmark = pytest.mark.l2

#: The repository root, three levels up from this file.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The design document the register is published in.
_DESIGN = _REPO_ROOT / ".system_design" / "TEST_SUITE.md"

#: The import roots the register's sites are addressed under.
_SRC = _REPO_ROOT / "src"


@pytest.fixture(scope="module")
def symbols() -> frozenset[str]:
    """Return every symbol defined under ``src``, addressed as a register site.

    Module-scoped because the scan parses 87 files and seven assertions below
    need it; per-test it was most of this file's runtime.

    Returns:
        The output of :func:`~harness.register.defined_symbols`.
    """
    return r.defined_symbols(_SRC)


@pytest.fixture(scope="module")
def markdown() -> str:
    """Return the design document's text.

    Returns:
        The full contents of ``.system_design/TEST_SUITE.md``.
    """
    return _DESIGN.read_text(encoding="utf-8")


class TestTheParserReadsTheDesignDocument:
    """The scan must find what it claims to find, or it is a no-op (§6.2)."""

    def test_the_parser_finds_every_row_shape_the_design_uses(self, markdown: str) -> None:
        """Pin the parser against ids no reformatting of §3.2 should lose.

        The first and last row of each table, plus one member of each
        sub-lettered family — the shapes a naive line-based parser gets wrong.
        """
        parsed = r.parse_register_markdown(markdown)

        for row_id in ("M1", "M14", "P1", "P2a", "P2b", "P5e", "P9c", "P19"):
            assert row_id in parsed.live_ids, f"{row_id} is published in §3.2 but the parser did not see it"

    def test_the_parser_reads_both_tables(self, markdown: str) -> None:
        """A parser that read only §3.2.1 would still look healthy on the M rows."""
        assert len([i for i in r.parse_register_markdown(markdown).live_ids if i.startswith("M")]) == 13
        assert len([i for i in r.parse_register_markdown(markdown).live_ids if i.startswith("P")]) == 28

    def test_the_parser_reads_the_unconditional_list(self, markdown: str) -> None:
        """§3.2.2's closing paragraph is the only place the exemption is written down."""
        parsed = r.parse_register_markdown(markdown)

        assert len(parsed.unconditional_ids) == 21
        assert {"M14", "P20", "P21"} <= set(parsed.unconditional_ids)

    def test_a_document_with_no_register_tables_is_an_error_not_an_empty_result(self) -> None:
        """An empty parse is the failure mode §6.2 exists to prevent.

        Returning ``()`` from a document whose headings moved would make every
        agreement check below pass over nothing.
        """
        with pytest.raises(r.RegisterMarkdownError):
            r.parse_register_markdown("# A document with no register in it\n")

    def test_a_table_that_has_lost_its_rows_is_an_error(self, markdown: str) -> None:
        """The heading can survive a reformatting that the row shape does not.

        Only §3.2.1's own rows are unpiped — anchored at line start with a full
        id cell — so §3.2.2 stays intact and the failure is attributable to the
        section the test names. A blanket ``replace("| M", "  M")`` also hit
        prose mentions of M-numbers, which made a failure here harder to read
        than the defect it was demonstrating.
        """
        defective = re.sub(r"^\| (M\d+[a-z]?) \|", r"  \1  ", markdown, flags=re.MULTILINE)

        with pytest.raises(r.RegisterMarkdownError, match="no live register rows"):
            r.parse_register_markdown(defective)

    def test_a_row_whose_id_cell_is_formatted_is_refused_not_skipped(self, markdown: str) -> None:
        """The falsification case for the "refuse what I cannot read" branch.

        §3.2's tables bold cells freely, so bolding a new row's id is a plausible
        way to highlight it — and a parser that simply ignored what it could not
        match would hand that row a silent pass from the one guard whose whole
        job is to notice it. §3.2's own "register maintenance" paragraph promises
        the opposite.
        """
        defective = markdown.replace(
            "| M14 |",
            "| **M15** | A new mutation | `X.y` | Always | because |\n| M14 |",
            1,
        )

        with pytest.raises(r.RegisterMarkdownError, match="id cell"):
            r.parse_register_markdown(defective)

    def test_a_plain_new_row_is_reported_rather_than_refused(self, markdown: str) -> None:
        """The complement, and the case that matters in practice.

        A parser that raised on every unfamiliar row would be useless: the normal
        way to add a mutation is a plain row, and that must surface as a *named
        disagreement* against the data, not as an unreadable-table error.
        """
        defective = markdown.replace(
            "| M14 |",
            "| M15 | A new mutation | `X.y` | Always | because |\n| M14 |",
            1,
        )

        problems = r.register_disagreements(r.REGISTER, defective)

        assert any("M15" in problem for problem in problems), problems

    def test_an_id_published_twice_is_refused(self, markdown: str) -> None:
        """An id is the register's addressing scheme, so a repeat makes it ambiguous.

        Caught here rather than left to :func:`register_disagreements`, which
        compares membership by *set*: a duplicate leaves the sets equal and the
        lengths unequal, so it either escaped entirely or surfaced as a
        nonsensical ordering complaint about an unrelated row.
        """
        defective = markdown.replace("| M7 |", "| M7 | a copy | `X.y` | Always | because |\n| M7 |", 1)

        with pytest.raises(r.RegisterMarkdownError, match="more than once"):
            r.parse_register_markdown(defective)

    def test_an_abbreviated_range_in_the_unconditional_list_is_refused(self, markdown: str) -> None:
        """The notation §3.2.2 used to carry, and why it had to go.

        ``P9a–c`` names three rows in one token. Expanding it is guesswork and
        skipping it silently drops two rows from the comparison — so the parser
        refuses it and says why, rather than reporting two false disagreements.
        """
        defective = markdown.replace("P9a, P9b, P9c,", "P9a–c,", 1)

        with pytest.raises(r.RegisterMarkdownError, match="range"):
            r.parse_register_markdown(defective)


class TestTheWithdrawnRow:
    """M13 is struck through in §3.2.1 and absent from the data (KBR-5)."""

    def test_the_only_struck_row_is_the_withdrawn_m13(self, markdown: str) -> None:
        """A new strike-through must be a deliberate decision, not a silent deletion."""
        assert r.parse_register_markdown(markdown).struck_ids == ("M13",)

    def test_a_struck_row_is_not_offered_as_live(self, markdown: str) -> None:
        """The oracle must never be handed a row that mutates nothing."""
        assert "M13" not in r.parse_register_markdown(markdown).live_ids

    def test_a_second_struck_row_is_caught(self, markdown: str) -> None:
        """The falsification case for the strike-through scan."""
        defective = markdown.replace("| M7 |", "| ~~M7~~ |", 1)

        assert r.parse_register_markdown(defective).struck_ids == ("M7", "M13")

    def test_a_row_struck_in_the_document_but_live_in_the_data_is_caught(self, markdown: str) -> None:
        """Withdrawing a row is a two-sided edit, and this is the side that can be forgotten."""
        defective = markdown.replace("| M7 |", "| ~~M7~~ |", 1)

        assert any("M7" in problem for problem in r.register_disagreements(r.REGISTER, defective))


class TestTheDataAndTheDesignNameTheSameRows:
    """§3.3.2 assertion 1 is a lookup into the register, so the two must match."""

    def test_the_shipped_pair_agrees(self, markdown: str) -> None:
        """The invariant. Every other test in this class is one of its falsifications."""
        assert r.register_disagreements(r.REGISTER, markdown) == ()

    def test_deleting_a_row_from_the_data_is_caught(self, markdown: str) -> None:
        """Plan §3's stated falsification case, from the data side."""
        without_p15 = tuple(row for row in r.REGISTER if row.id != "P15")

        problems = r.register_disagreements(without_p15, markdown)

        assert any("P15" in problem for problem in problems), problems

    def test_deleting_a_row_from_the_document_is_caught(self, markdown: str) -> None:
        """Plan §3's stated falsification case, from the document side."""
        defective = "\n".join(line for line in markdown.splitlines() if not line.startswith("| P15 |"))

        problems = r.register_disagreements(r.REGISTER, defective)

        assert any("P15" in problem for problem in problems), problems

    def test_reordering_the_data_is_caught(self, markdown: str) -> None:
        """The register is published in an order a reader holds beside the document.

        §3.2.2 interleaves P20 and P21 between P6 and P7, which is easy to
        "tidy" into numeric order and hard to notice afterwards.
        """
        reversed_rows = tuple(reversed(r.REGISTER))

        assert any("order" in problem for problem in r.register_disagreements(reversed_rows, markdown))

    def test_a_duplicated_row_in_the_data_is_reported_not_raised(self, markdown: str) -> None:
        """This function's contract is to *return* problems, never to raise.

        A copy-pasted row leaves the id sets equal while the lengths differ. The
        order comparison used to pair the two sequences strictly and died with
        ``ValueError: zip() argument 2 is longer than argument 1`` — an exception
        out of a function whose callers hand it damaged artifacts on purpose, and
        a message naming nothing a maintainer could act on.
        """
        duplicated = r.REGISTER + (r.REGISTER[-1],)

        problems = r.register_disagreements(duplicated, markdown)

        assert problems, "a duplicated row must be reported"
        assert all(isinstance(problem, str) for problem in problems)

    def test_flipping_a_conditional_flag_is_caught(self, markdown: str) -> None:
        """§3.2.2's unconditional list drives §3.3.2 assertion 2.

        A row the data calls unconditional and the document calls conditional is
        owed a corpus complement that nobody knows to write, and the
        disagreement is invisible without this check.
        """
        flipped = tuple(
            dataclasses.replace(row, conditional=not row.conditional) if row.id == "M3" else row for row in r.REGISTER
        )

        problems = r.register_disagreements(flipped, markdown)

        assert any("M3" in problem for problem in problems), problems

    def test_an_unconditional_list_that_omits_a_row_is_caught(self, markdown: str) -> None:
        """The defect this guard was written against.

        §3.2.2's list omitted M14, P20 and P21 — three rows whose own trigger
        cell reads ``Always``. §3.3.2 assertion 2 would then demand a complement
        case for a mutation that always fires.
        """
        defective = markdown.replace("M1, M2, M10, M14, P1,", "M1, M2, M10, P1,", 1)

        problems = r.register_disagreements(r.REGISTER, defective)

        assert any("M14" in problem for problem in problems), problems

    def test_an_unconditional_list_naming_a_row_that_does_not_exist_is_caught(self, markdown: str) -> None:
        """A stale entry left behind when a row is renamed or withdrawn."""
        defective = markdown.replace("M1, M2, M10, M14, P1,", "M1, M2, M10, M14, M99, P1,", 1)

        problems = r.register_disagreements(r.REGISTER, defective)

        assert any("M99" in problem for problem in problems), problems


class TestEverySiteResolvesInTheSource:
    """A row naming a symbol that no longer exists is a register that has rotted."""

    def test_every_site_symbol_exists(self, symbols: frozenset[str]) -> None:
        """The invariant: `site` is checkable data, not prose."""
        assert r.unresolved_sites(r.REGISTER, symbols) == ()

    def test_the_scan_resolves_a_known_positive(self, symbols: frozenset[str]) -> None:
        """The self-check §6.2 requires.

        A resolver that answered "everything exists" would make the test above
        pass forever. Both a method and a module-level constant, since a register
        row may name either.
        """
        assert "kitty/bridge/server.py:BridgeServer._normalize_model" in symbols
        assert "kitty/providers/base.py:ProviderAdapter._INTERNAL_KEYS" in symbols

    def test_the_scan_resolves_a_name_that_twelve_modules_define(self, symbols: frozenset[str]) -> None:
        """The positive control for *qualified* resolution, which is the whole point.

        ``build_upstream_headers`` is defined in twelve provider modules, so a
        leaf-name resolver would answer "present" for every one of P9a, P9b and
        P9c no matter which class had been renamed. Asserting that the name
        resolves on MiMo and does **not** resolve on Fireworks is what
        distinguishes a qualified scan from a name-level one.
        """
        assert "kitty/providers/mimo.py:MimoAdapter.build_upstream_headers" in symbols
        assert "kitty/providers/fireworks.py:FireworksAdapter.build_upstream_headers" not in symbols

    def test_the_scan_does_not_invent_symbols(self, symbols: frozenset[str]) -> None:
        """The complement: a resolver matching anything would also pass above."""
        assert "kitty/bridge/server.py:BridgeServer._method_that_does_not_exist" not in symbols

    def test_a_site_naming_a_missing_symbol_is_caught(self, symbols: frozenset[str]) -> None:
        """The falsification case for the site scan."""
        renamed = tuple(
            dataclasses.replace(row, site=("kitty/bridge/server.py:BridgeServer._renamed_away",))
            if row.id == "M1"
            else row
            for row in r.REGISTER
        )

        problems = r.unresolved_sites(renamed, symbols)

        assert any("M1" in problem for problem in problems), problems

    def test_a_site_naming_the_right_symbol_in_the_wrong_class_is_caught(self, symbols: frozenset[str]) -> None:
        """A rename that moves a method between adapters leaves the leaf name intact.

        This is the case a name-level scan cannot see, and the reason
        :func:`~harness.register.defined_symbols` addresses by file and
        qualified name.
        """
        moved = tuple(
            dataclasses.replace(row, site=("kitty/providers/fireworks.py:FireworksAdapter.build_upstream_headers",))
            if row.id == "P9b"
            else row
            for row in r.REGISTER
        )

        problems = r.unresolved_sites(moved, symbols)

        assert any("P9b" in problem for problem in problems), problems
