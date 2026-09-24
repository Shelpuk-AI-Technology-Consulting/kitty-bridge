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

import ast
import dataclasses
import re
from pathlib import Path

import pytest

from harness import contract as c
from harness import reader_chat_completions as cc_reader
from harness import reader_responses as reader
from harness import register as r

pytestmark = pytest.mark.l2

#: The repository root, three levels up from this file.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The design document the register is published in.
_DESIGN = _REPO_ROOT / ".system_design" / "TEST_SUITE.md"

#: The import roots the register's sites are addressed under.
_SRC = _REPO_ROOT / "src"

#: The adapter whose allowlist decides what P23 claims, and the name it spells it
#: under.  Read as **text**: §3.2.4's independence rule is why
#: :func:`~harness.register.defined_symbols` parses ``src/kitty`` with :mod:`ast`
#: rather than importing it, and a guard that imported this adapter to read its
#: allowlist would agree with the code instead of checking it.
_ALLOWLIST_MODULE = _SRC / "kitty" / "providers" / "openai_subscription.py"
_ALLOWLIST_NAME = "_ALLOWED_RESPONSES_PARAMS"
#: The CC-origin body builder on the same adapter; P24's derivation guard
#: reads its output-body literal keys to compute the "carries" set — the
#: keys the builder ships on the wire (rewritten, not dropped) — and P24
#: therefore does not claim.
_CC_TO_RESPONSES_NAME = "_cc_to_responses"


def _cc_to_responses_carries(source: str) -> frozenset[str]:
    """Read the keys ``_cc_to_responses`` writes into its output body.

    Reads the builder as text so the register, which imports nothing from
    ``src/kitty``, cannot accidentally agree with the code instead of
    checking it. Returns the subset of
    ``reader_chat_completions._PUBLISHED_EXTRA_KEYS`` that the builder ships
    on the wire — rewritten, not dropped, keys. P24's claim is the reader
    table minus this set.

    Args:
        source: The full source text of ``openai_subscription.py``.

    Returns:
        The extra-key table members the builder carries. Today: ``{"store"}``
        (forced to ``False``, claimed by P17 on the rewritten half).

    Raises:
        AssertionError: When ``source`` does not define ``_cc_to_responses`` —
            the guard must not pass for an empty read (plan §1.4).
    """
    tree = ast.parse(source)
    fn = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == _CC_TO_RESPONSES_NAME
        ),
        None,
    )
    assert fn is not None, (
        f"{_CC_TO_RESPONSES_NAME} is not defined in the adapter source — "
        "the derivation guard must not pass for an empty read (plan §1.4)"
    )

    carried: set[str] = set()

    def _str(node: ast.AST | None) -> str | None:
        """Return ``node.value`` if it is a string ``Constant``, else ``None``.

        A dict literal's key can be ``None`` (a ``**`` unpacking), so the
        annotation accepts it and this guard declines it the same way.
        """
        if node is None:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    # ``body: dict = {"model": ..., ...}`` — the initial literal.
    for stmt in fn.body:
        if (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "body"
            and isinstance(stmt.value, ast.Dict)
        ):
            for key_node in stmt.value.keys:
                name = _str(key_node)
                if name is not None:
                    carried.add(name)

    # ``body["X"] = ...`` — every later assignment to a string-keyed body
    # member, inside whichever ``if`` guards it. ``Subscript.slice`` is the
    # key expression directly on every Python this repo runs (3.9 removed
    # the ``ast.Index`` wrapper).
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "body"
        ):
            name = _str(node.targets[0].slice)
            if name is not None:
                carried.add(name)

    return frozenset(carried & cc_reader._PUBLISHED_EXTRA_KEYS)


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
def registry() -> dict[str, str]:
    """Return the AST-read provider registry (key → adapter class name).

    Module-scoped because the parse is a one-shot I/O and several assertions
    below need it; per-test it was redundant work.

    Returns:
        The output of :func:`~harness.register.provider_registry`.
    """
    return r.provider_registry(_SRC)


@pytest.fixture(scope="module")
def markdown() -> str:
    """Return the design document's text.

    Returns:
        The full contents of ``.system_design/TEST_SUITE.md``.
    """
    return _DESIGN.read_text(encoding="utf-8")


def _allowlisted_responses_params(source: str) -> frozenset[str]:
    """Read the Codex allowlist out of the adapter's source text.

    Takes the text rather than a path so the parse is pure and a deliberately
    damaged module can be handed to it, which is what plan §1.4 requires of a new
    guard.  It differs from :func:`~harness.register.defined_symbols` only in
    needing the assignment's *value* as well as its name.

    Args:
        source: The text of ``src/kitty/providers/openai_subscription.py``.

    Returns:
        The parameter names ``_ALLOWED_RESPONSES_PARAMS`` keeps.

    Raises:
        AssertionError: When the module no longer defines the allowlist, or
            spells it as something other than a ``frozenset`` of a literal.
            Returning an empty set instead would make every control field look
            dropped and leave this guard green for the wrong reason — the silent
            no-op §6.2 forbids.
    """
    # Walked rather than read off the class, so the guard survives the allowlist
    # moving to module level; the name is what identifies it.
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == _ALLOWLIST_NAME for target in node.targets):
            continue
        if isinstance(node.value, ast.Call) and node.value.args:
            # `literal_eval` raises on a computed argument -- `frozenset(_BASE | {...})`
            # parses as a Call with one arg and is not a literal. Re-raised as the
            # documented failure so both malformed spellings report alike.
            try:
                return frozenset(ast.literal_eval(node.value.args[0]))
            except ValueError as exc:
                raise AssertionError(f"{_ALLOWLIST_NAME} is no longer a frozenset built from a literal") from exc
        raise AssertionError(f"{_ALLOWLIST_NAME} is no longer a frozenset built from a literal")

    raise AssertionError(f"{_ALLOWLIST_MODULE.name} no longer defines {_ALLOWLIST_NAME}")


def published_row_cell(markdown: str, row_id: str, heading: str) -> str:
    """Return one cell of a §3.2 register row, located by its column heading.

    Positional indexing was the obvious spelling and is wrong in the quiet
    direction: §3.2.2's table has five columns today, so ``split("|")[2]`` is the
    Mutation cell, and a future edit that adds or removes a column would leave the
    assertion reading a *different* cell and passing.  Reading the heading row
    makes that edit a loud failure instead.

    One limitation, stated rather than discovered: a cell containing a literal
    ``|`` would split wrongly.  No register cell does, and the failure would be a
    mismatch rather than a silent pass.

    Args:
        markdown: The full text of ``.system_design/TEST_SUITE.md``.
        row_id: The row's id, e.g. ``P23``.
        heading: The column's heading exactly as the table spells it.

    Returns:
        The cell's text, stripped.

    Raises:
        RegisterMarkdownError: When the row, its heading row, or that column is
            absent — each of which would otherwise make this guard compare
            nothing.
    """
    lines = markdown.splitlines()

    # The heading row is the nearest `| # | ...` above the data row, so the two
    # tables of §3.2 cannot be crossed.
    row_index = next((i for i, line in enumerate(lines) if line.startswith(f"| {row_id} |")), None)
    if row_index is None:
        raise r.RegisterMarkdownError(f"§3.2 publishes no row {row_id}")
    heading_index = next((i for i in range(row_index, -1, -1) if lines[i].startswith("| # |")), None)
    if heading_index is None:
        raise r.RegisterMarkdownError(f"{row_id} sits under no table heading — §3.2's tables have changed shape")

    headings = [cell.strip() for cell in lines[heading_index].split("|")]
    if heading not in headings:
        raise r.RegisterMarkdownError(f"§3.2's table has no {heading!r} column; it has {headings[1:-1]}")

    cells = lines[row_index].split("|")
    return cells[headings.index(heading)].strip()


def _claimable_paths(control_fields: frozenset[str], allowlist: frozenset[str]) -> frozenset[str]:
    """Return the projection paths a row must claim for the allowlist's drops.

    The rule P23 encodes, as a function of the two artifacts that decide it, so
    the falsification cases can hand it a doctored one.

    Args:
        control_fields: The wire keys the Responses reader sends to
            ``envelope.extra`` — :data:`harness.reader_responses._EXTRA_KEYS`.
        allowlist: The parameters the Codex backend accepts.

    Returns:
        One ``envelope.extra[<wire key>]`` path per dropped control field.
    """
    return frozenset(c.extra_path(key) for key in control_fields - allowlist)


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
        assert len([i for i in r.parse_register_markdown(markdown).live_ids if i.startswith("M")]) == 32
        assert len([i for i in r.parse_register_markdown(markdown).live_ids if i.startswith("P")]) == 53

    def test_the_parser_reads_the_unconditional_list(self, markdown: str) -> None:
        """§3.2.2's closing paragraph is the only place the exemption is written down."""
        parsed = r.parse_register_markdown(markdown)

        assert len(parsed.unconditional_ids) == 53
        # KBR-55 + KBR-184 + KBR-305 + KBR-309: M27 and M29 are KBR-55's two
        # unconditional ids (M28 is conditional and so belongs on the
        # complement side, not in the unconditional sentence); P43 is
        # KBR-305's new unconditional id; M30 and M31 are KBR-309's two new
        # unconditional ids (id-renumbered from KBR-309's first-draft M27/M28
        # after the KBR-55 rebase). The long-standing spot-checks
        # (M14/P20/P21, the KBR-184 KBR-258 KBR-137 KBR-271 families) all
        # still hold.
        assert {
            "M14", "P20", "P21", "M26", "P24", "P31", "P32",
            "M27", "M29", "M30", "M31", "P43",
        } <= set(parsed.unconditional_ids)

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
            # M99, not the next free id: this fixture used `M16` until KBR-167
            # published a real M16, and the duplicate-id check then raised
            # before the assertion could run. An id no row will ever take is the
            # only spelling that cannot rot.
            "| **M99** | A new mutation | `X.y` | Always | because |\n| M14 |",
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
            # M99 for the reason given on the bolded-id case above.
            "| M99 | A new mutation | `X.y` | Always | because |\n| M14 |",
            1,
        )

        problems = r.register_disagreements(r.REGISTER, defective)

        assert any("M99" in problem for problem in problems), problems

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
        defective = markdown.replace(
            "M1, M2, M9a, M9b, M10, M14, M15, M16, P1,", "M1, M2, M9a, M9b, M10, M15, M16, P1,", 1
        )

        problems = r.register_disagreements(r.REGISTER, defective)

        assert any("M14" in problem for problem in problems), problems

    def test_an_unconditional_list_naming_a_row_that_does_not_exist_is_caught(self, markdown: str) -> None:
        """A stale entry left behind when a row is renamed or withdrawn."""
        defective = markdown.replace(
            "M1, M2, M9a, M9b, M10, M14, M15, M16, P1,", "M1, M2, M9a, M9b, M10, M14, M15, M16, M99, P1,", 1
        )

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


class TestP23ClaimsTheControlFieldsOutsideTheCodexAllowlist:
    """KBR-171 — P23's sixteen paths are recomputed here, never transcribed.

    §3.2.2's P23 enumerates sixteen ``envelope.extra[<wire key>]`` addresses
    rather than anchoring at a bare ``envelope.extra``, for the reason §3.3.1a
    gives.  An enumeration transcribed from a ticket would be a list nobody
    re-reads; recomputing it from the two artifacts that decide the set — T-A3's
    published control-field table and the adapter's allowlist — makes widening
    either a **deliberate** edit to the row rather than a silent divergence.

    ⚠️ **What this does not prove**, said here because the class name invites the
    stronger reading.  It does not make P23 independent of the code: once the
    code changes, the only route back to green is to edit the row to match, which
    is the trade §3.2.4 records.  And "dropped" means **never copied** —
    membership of the allowlist is *not* what carries a field through, since the
    literal feeds only a DEBUG log while the shipped body is an explicit ``if``
    chain testing truthiness.  An allowlisted field with a falsy value is dropped
    as well, sits outside P23 by construction, and is claimed by **P25**
    (`KBR-185` / G27).

    The reader's table is a sound left-hand side because it is itself pinned:
    ``test_reader_responses`` asserts it covers the published schema's 31 keys
    exactly, and that each key lands at the path the table claims.  Nothing here
    or there reads the vendor schema — §8's determinism rules forbid it — so a
    revision by OpenAI goes undetected, which is G24's shape rather than a solved
    problem.
    """

    def test_the_row_claims_every_control_field_outside_the_allowlist_and_nothing_else(self) -> None:
        """AC R4 — set equality, so over-claiming fails as loudly as under-claiming."""
        rows = [row for row in r.REGISTER if row.id == "P23"]
        assert rows, "P23 is not in the register data — the Codex allowlist's drops are unclaimed (KBR-171)"

        allowlist = _allowlisted_responses_params(_ALLOWLIST_MODULE.read_text(encoding="utf-8"))

        assert set(rows[0].paths) == _claimable_paths(reader._EXTRA_KEYS, allowlist)

    def test_widening_the_allowlist_would_unclaim_a_field(self) -> None:
        """The allowlist half of the derivation is live, not decoration.

        A guard that ignored its allowlist argument would pass the test above and
        go on passing after the adapter started shipping ``text`` upstream.
        """
        allowlist = _allowlisted_responses_params(_ALLOWLIST_MODULE.read_text(encoding="utf-8"))

        widened = _claimable_paths(reader._EXTRA_KEYS, allowlist | {"text"})

        assert c.extra_path("text") not in widened
        assert widened != _claimable_paths(reader._EXTRA_KEYS, allowlist)

    def test_dropping_a_control_field_from_the_reader_would_unclaim_it(self) -> None:
        """The other half: the reader's table decides which keys are addressable at all."""
        allowlist = _allowlisted_responses_params(_ALLOWLIST_MODULE.read_text(encoding="utf-8"))

        narrowed = _claimable_paths(reader._EXTRA_KEYS - {"truncation"}, allowlist)

        assert c.extra_path("truncation") not in narrowed
        assert narrowed != _claimable_paths(reader._EXTRA_KEYS, allowlist)

    def test_a_source_that_does_not_define_the_allowlist_is_an_error(self) -> None:
        """Plan §1.4's deliberate defect: an empty read must not pass for an empty allowlist.

        An allowlist read as ``frozenset()`` makes every control field look
        dropped, which is a state the set-equality test above would report as a
        *register* problem while the real fault was the parse.
        """
        with pytest.raises(AssertionError, match=_ALLOWLIST_NAME):
            _allowlisted_responses_params("UNRELATED = frozenset({'model'})\n")

    def test_the_allowlist_read_finds_the_real_one(self, symbols: frozenset[str]) -> None:
        """The positive control §6.2 requires beside the negative one above."""
        assert f"kitty/providers/openai_subscription.py:OpenAISubscriptionAdapter.{_ALLOWLIST_NAME}" in symbols

        allowlist = _allowlisted_responses_params(_ALLOWLIST_MODULE.read_text(encoding="utf-8"))

        assert {"model", "reasoning", "tool_choice"} <= allowlist
        assert "truncation" not in allowlist

    def test_the_published_row_names_the_same_sixteen_keys(self, markdown: str) -> None:
        """§3.2.2's P23 cell is a path list in all but spelling, so it is reconciled too.

        §3.2.4 declines to reconcile paths against the markdown because "the tables
        have no path column".  That stops being true for this one row: its Mutation
        cell enumerates the sixteen wire keys in backticks, which is a second copy
        of the data.  Measured during review — deleting ``truncation`` from the
        published cell, and changing "sixteen" to "fifteen", each left the whole
        suite green.
        """
        cell = published_row_cell(markdown, "P23", "Mutation")

        assert set(re.findall(r"`(\w+)`", cell)) == set(r._CODEX_DROPPED_CONTROL_FIELDS)
        assert "**sixteen**" in markdown, "§3.2.2's P23 cell no longer states the count it enumerates"

    def test_the_cell_is_found_by_its_heading_and_not_by_its_position(self, markdown: str) -> None:
        """The falsification case for the locator, since a wrong cell would pass quietly.

        A column inserted before ``Mutation`` shifts every index by one.  Under
        positional indexing the assertion above would then compare the *Site*
        cell's backticks against the sixteen keys — or, worse on a different
        edit, a cell that happens to match.  Three shapes are pinned: the column
        moves, the column goes, and the row goes.
        """
        # Both heading rows, because §3.2.1's table comes first and a `count=1`
        # replace shifts the wrong one -- which this test caught when written.
        shifted = markdown.replace("| # | Mutation |", "| # | Owner | Mutation |").replace(
            "| P23 | **Drop", "| P23 | someone | **Drop", 1
        )

        assert published_row_cell(shifted, "P23", "Mutation") == published_row_cell(markdown, "P23", "Mutation")

        with pytest.raises(r.RegisterMarkdownError, match="Mutation"):
            published_row_cell(markdown.replace("| # | Mutation |", "| # | Effect |"), "P23", "Mutation")

        with pytest.raises(r.RegisterMarkdownError, match="P23"):
            published_row_cell(markdown.replace("| P23 |", "| P99 |"), "P23", "Mutation")

    def test_a_malformed_allowlist_spelling_reports_as_a_parse_fault(self) -> None:
        """The second half of the deliberate defect above: present, but not a literal.

        ``frozenset(_BASE | {"model"})`` parses as a ``Call`` with one argument and
        reaches ``literal_eval``, which raises ``ValueError``.  Undressed, that
        surfaces as a stack trace rather than the documented failure.
        """
        with pytest.raises(AssertionError, match="frozenset built from a literal"):
            _allowlisted_responses_params('_ALLOWED_RESPONSES_PARAMS = frozenset(_BASE | {"model"})\n')

    def test_an_allowlisted_field_dropped_for_being_falsy_is_outside_this_row(self) -> None:
        """The boundary P25 owns, pinned so widening P23 cannot be accidental.

        ``_prepare_responses_body`` copies ``include`` only when it is truthy, so
        ``include: []`` — a legal ``CreateResponse`` body — is dropped while
        sitting *inside* the allowlist.  The delta is real and P23 does not claim
        it, which is deliberate: the mutation is conditional on the value, and
        the same shape at ``reasoning`` would swallow P22's address.  Asserted
        rather than left to a comment, because the cheapest wrong fix to that
        report is to add the key here.

        Reads the row only; the projection evidence lives in `KBR-185`, whose
        reproduction is two bodies through
        :meth:`~kitty.providers.openai_subscription.OpenAISubscriptionAdapter._prepare_responses_body`.
        """
        allowlist = _allowlisted_responses_params(_ALLOWLIST_MODULE.read_text(encoding="utf-8"))
        p23 = next(row for row in r.REGISTER if row.id == "P23")

        # Inside the allowlist, and therefore outside this row -- both halves, or
        # the test passes for a key the reader never classified in the first place.
        assert {"include", "reasoning"} <= allowlist & reader._EXTRA_KEYS
        assert not any(c.path_matches(path, c.extra_path("include")) for path in p23.paths)


class TestP24ClaimsTheDroppedNonSamplingControlFields:
    """KBR-184 / G26 — P24's thirteen paths are recomputed here, never transcribed.

    §3.2.2's P24 enumerates the CC-origin control fields ``_cc_to_responses``
    drops rather than anchoring at a bare ``envelope.extra``, for the reason
    §3.3.1a gives. The enumeration is the Chat Completions reader's
    ``_PUBLISHED_EXTRA_KEYS`` (T-A2 / KBR-34) minus what the builder ships on
    the wire — rewritten, not dropped, keys. Any change to the reader's
    table or to the builder's body literal widens or narrows P24's claim,
    and this guard catches it.

    ⚠️ **What this does not prove**, said here because the class name invites
    the stronger reading. It does not make P24 independent of the code: once
    the code changes, the only route back to green is to edit the row to
    match, which is the trade §3.2.4 records. And "dropped" means **never
    copied** — the reader projects CC ``metadata`` to ``envelope.extra[metadata]``
    and the builder does not write it, so the address is a delta P24 claims.
    The reader's table is the sound left-hand side because
    ``test_reader_chat_completions`` pins it against the published schema,
    and §8's determinism rules forbid reading the vendor schema directly —
    a vendor revision goes undetected, which is G24's shape.
    """

    def test_the_row_claims_every_dropped_key_and_nothing_else(self) -> None:
        """AC R2.1 — set equality, so over- and under-claim both fail loudly."""
        rows = [row for row in r.REGISTER if row.id == "P24"]
        assert rows, "P24 is not in the register data — KBR-184's CC-origin drops are unclaimed"

        source = _ALLOWLIST_MODULE.read_text(encoding="utf-8")
        carries = _cc_to_responses_carries(source)

        assert set(rows[0].paths) == {
            c.extra_path(key) for key in cc_reader._PUBLISHED_EXTRA_KEYS - carries
        }

    def test_widening_the_builder_to_carry_a_dropped_key_would_unclaim_it(self) -> None:
        """AC R2.2 — the builder half of the derivation is live, not decoration.

        A guard that ignored its builder argument would pass the set-equality
        above and go on passing after the adapter started shipping, say,
        ``metadata`` upstream — the loud under-claiming direction.
        """
        real_source = _ALLOWLIST_MODULE.read_text(encoding="utf-8")
        real_carries = _cc_to_responses_carries(real_source)

        # Plant: the builder adds ``metadata`` to its output body literal.
        # ``_prepare_responses_body`` also writes ``"store": False,`` earlier
        # in the file, so the first occurrence is the wrong builder's
        # literal — target the last one, which is ``_cc_to_responses``'s.
        needle = '"store": False,'
        plant = '"store": False,\n            "metadata": None,'
        idx = real_source.rfind(needle)
        assert idx >= 0, "the plant's needle is gone from the adapter source"
        widened_source = real_source[:idx] + plant + real_source[idx + len(needle):]
        assert widened_source != real_source, "the plant did not mutate the source"

        widened_carries = _cc_to_responses_carries(widened_source)

        assert "metadata" in widened_carries
        assert widened_carries != real_carries

        narrowed = {c.extra_path(key) for key in cc_reader._PUBLISHED_EXTRA_KEYS - widened_carries}
        assert c.extra_path("metadata") not in narrowed
        assert narrowed != {
            c.extra_path(key) for key in cc_reader._PUBLISHED_EXTRA_KEYS - real_carries
        }

    def test_dropping_a_control_field_from_the_reader_would_unclaim_it(self) -> None:
        """AC R2.3 — the reader's table decides which keys are addressable at all.

        If the reader stopped classifying ``metadata`` as an ``envelope.extra``
        key, the row's claim at ``envelope.extra[metadata]`` would no longer
        be derivable from the table — the row would over-claim.
        """
        real_source = _ALLOWLIST_MODULE.read_text(encoding="utf-8")
        real_carries = _cc_to_responses_carries(real_source)

        narrowed_table = cc_reader._PUBLISHED_EXTRA_KEYS - {"metadata"}
        narrowed = {c.extra_path(key) for key in narrowed_table - real_carries}

        assert c.extra_path("metadata") not in narrowed
        assert narrowed != {
            c.extra_path(key) for key in cc_reader._PUBLISHED_EXTRA_KEYS - real_carries
        }

    def test_a_source_that_does_not_define_the_builder_is_an_error(self) -> None:
        """Plan §1.4's deliberate defect — an empty read must not pass.

        A source with no ``_cc_to_responses`` makes every published key look
        dropped, which the set-equality test would report as a register
        problem while the real fault was the parse.
        """
        with pytest.raises(AssertionError, match=_CC_TO_RESPONSES_NAME):
            _cc_to_responses_carries("UNRELATED = frozenset({'model'})\n")

    def test_the_builder_read_finds_the_real_one(self, symbols: frozenset[str]) -> None:
        """The positive control §6.2 requires beside the negative one above."""
        assert (
            f"kitty/providers/openai_subscription.py:OpenAISubscriptionAdapter.{_CC_TO_RESPONSES_NAME}"
            in symbols
        )

        carries = _cc_to_responses_carries(_ALLOWLIST_MODULE.read_text(encoding="utf-8"))

        assert "store" in carries
        assert "metadata" not in carries
        assert "tool_choice" not in carries  # not in _PUBLISHED_EXTRA_KEYS (handled separately)
        assert "parallel_tool_calls" not in carries  # G36 moved it off extra

    def test_the_published_row_names_the_same_thirteen_keys(self, markdown: str) -> None:
        """§3.2.2's P24 cell is a path list in all but spelling, so it is reconciled too.

        §3.2.4 declines to reconcile paths against the markdown because "the
        tables have no path column". P24's Mutation cell enumerates the
        thirteen wire keys in backticks, the same shape P23 carries; a
        second copy of the data is reconciled or it rots in silence.
        """
        cell = published_row_cell(markdown, "P24", "Mutation")

        assert set(re.findall(r"`(\w+)`", cell)) == set(r._CC_DROPPED_CONTROL_FIELDS)
        assert "**thirteen**" in markdown, (
            "§3.2.2's P24 cell no longer states the count it enumerates"
        )


class TestP25ClaimsTheFalsyAllowlistedDrop:
    """KBR-185 — P25 claims the falsy-value drop the allowlist permits and the `if` chain loses.

    ``_ALLOWED_RESPONSES_PARAMS`` is read only by ``_prepare_responses_body``'s
    DEBUG log; the shipped body is an explicit ``if`` chain, and six of its
    branches test truthiness rather than presence (only ``parallel_tool_calls``
    tests presence).  An allowlisted field with a falsy value — ``include: []``,
    ``reasoning: {}``, both legal under ``CreateResponse`` — is therefore
    dropped, present inbound and absent upstream, and the projection's
    presence-based reader reports it as an ``envelope.extra`` delta nothing
    claimed until P25.  P23 excludes both keys by construction (they sit inside
    the allowlist); P14 reaches no ``extra`` path; P22's trigger is not met by
    this case.
    """

    def _p25(self) -> r.MutationRow:
        """Return the P25 row, failing with a named message when it is absent.

        Returns:
            The register row whose id is ``P25``.

        Raises:
            AssertionError: When P25 is not in the register data — the falsy
                allowlisted drops are then unclaimed and T-D5 reports a false
                I1 breach on the first corpus entry that carries one.
        """
        rows = [row for row in r.REGISTER if row.id == "P25"]
        assert rows, "P25 is not in the register data — the falsy allowlisted drops are unclaimed (KBR-185)"
        return rows[0]

    def test_the_row_exists_with_the_expected_shape(self, symbols: frozenset[str]) -> None:
        """The whole spec in one place: site, trigger, paths, conditionality."""
        row = self._p25()

        assert row.site == (
            "kitty/providers/openai_subscription.py:OpenAISubscriptionAdapter._prepare_responses_body",
        )
        assert row.trigger is r.Trigger.ALLOWLISTED_FIELD_IS_FALSY
        assert row.paths == (c.extra_path("include"), c.extra_path("reasoning"))
        assert row.conditional is True
        assert row.design_ref.startswith("§")
        assert r.row_shape_problems(row) == ()
        assert not r.unresolved_sites((row,), symbols)

    def test_the_trigger_member_is_in_the_closed_vocabulary_with_its_value(self) -> None:
        """T-W6 indexes the corpus by these names, so the spelling is load-bearing."""
        assert r.Trigger.ALLOWLISTED_FIELD_IS_FALSY.value == "allowlisted_field_is_falsy"

    def test_the_published_mutation_cell_names_no_key(self, markdown: str) -> None:
        """The §3.2.4 obligation travels with the shape, and this row's shape is prose.

        P23's Mutation cell enumerates its sixteen keys and therefore inherits
        both reconciliation exceptions.  P25's cell describes the row instead;
        the moment someone backticks ``include`` or ``reasoning`` into it, it
        becomes a second copy of the path data that nothing compares — so the
        absence itself is pinned here.
        """
        cell = published_row_cell(markdown, "P25", "Mutation")

        assert set(re.findall(r"`(\w+)`", cell)) == set()


class TestEveryScopeNamesRealProviders:
    """A scope entry naming a provider that does not exist is data nothing could contradict.

    KBR-139: the scope column claims knowledge of
    :data:`providers.registry._registry`, read by AST and never by import
    (§3.3.1's independent-oracle rule), plus the site↔scope sibling check that
    keeps a row's scope from silently drifting away from the files its sites
    name.
    """

    def test_every_row_scope_is_well_formed(self, registry: dict, symbols: frozenset) -> None:
        """The invariant: the shipped register's scopes all check out."""
        assert r.scope_problems(r.REGISTER, registry, symbols) == ()

    def test_the_registry_read_finds_the_known_providers(self, registry: dict) -> None:
        """The self-check §6.2 requires — a parser regression cannot pass silently.

        The count is pinned (a 24th adapter is a deliberate edit to this test,
        exactly like the register's 79-row pin) and three members are named,
        including the file-class-split case the site↔scope check depends on
        (``zai_coding`` -> ``ZaiAnthropicAdapter``, defined in ``zai_anthropic.py``).
        """
        assert len(registry) == 23
        assert registry["anthropic"] == "AnthropicAdapter"
        assert registry["openai_subscription"] == "OpenAISubscriptionAdapter"
        assert registry["zai_coding"] == "ZaiAnthropicAdapter"

    def test_a_scope_key_that_is_no_provider_is_named(self, registry: dict, symbols: frozenset) -> None:
        """The deliverable's falsification case: a bogus key fails, naming the row."""
        bad = dataclasses.replace(r.REGISTER[0], scope=("not_a_provider",))
        problems = r.scope_problems((bad,), registry, symbols)

        assert problems, "a scope naming a non-registry key must be reported"
        assert bad.id in problems[0]
        assert "not_a_provider" in problems[0]

    def test_the_sentinel_cannot_be_mixed_with_keys(self, registry: dict, symbols: frozenset) -> None:
        """A mixed tuple is two claims in one cell; whichever is honoured, the other lies."""
        bad = dataclasses.replace(r.REGISTER[0], scope=(r.ALL_PROVIDERS, "anthropic"))
        problems = r.scope_problems((bad,), registry, symbols)

        assert problems, "a mixed sentinel tuple must be reported"
        assert bad.id in problems[0]

    def test_an_empty_scope_is_reported_not_raised(self, registry: dict, symbols: frozenset) -> None:
        """An empty tuple stays constructable — the row_shape_problems pure-design reason."""
        bad = dataclasses.replace(r.REGISTER[0], scope=())
        problems = r.scope_problems((bad,), registry, symbols)

        assert problems, "an empty scope must be reported"
        assert bad.id in problems[0]

    def test_a_site_whose_file_the_scope_omits_is_named(self, registry: dict, symbols: frozenset) -> None:
        """The site-to-scope sibling check: P9b's site names mimo.py, so mimo must be in scope.

        P9b is the cheapest live row to damage: its site is one file, its scope
        one key. Dropping the key must turn the guard red — otherwise the check
        is decoration.
        """
        mimo_row = next(row for row in r.REGISTER if row.id == "P9b")
        bad = dataclasses.replace(mimo_row, scope=("kimi",))
        problems = r.scope_problems((bad,), registry, symbols)

        assert problems, "a scope omitting the site's own adapter must be reported"
        assert "mimo" in problems[0]

    def test_the_registry_reader_is_live_on_a_synthetic_source(self, tmp_path: Path) -> None:
        """The parser reads a hand-written mini-registry into key-class pairs.

        The live read above could agree with the guard by accident of both being
        wrong in the same direction; this positive control proves the reader
        itself parses a known input.
        """
        (tmp_path / "kitty" / "providers").mkdir(parents=True)
        (tmp_path / "kitty" / "providers" / "registry.py").write_text(
            "_registry: dict[str, type[ProviderAdapter]] = {\n"
            '    "alpha": AlphaAdapter,\n'
            '    "beta": BetaAdapter,\n'
            "}\n",
            encoding="utf-8",
        )

        assert r.provider_registry(tmp_path) == {"alpha": "AlphaAdapter", "beta": "BetaAdapter"}

    def test_a_registry_the_reader_cannot_parse_raises(self, tmp_path: Path) -> None:
        """An unreadable registry raises RegisterSourceError — never a silent empty dict.

        An empty result would make :func:`scope_problems` reject nothing, which
        is exactly the no-op §6.2 forbids.
        """
        (tmp_path / "kitty" / "providers").mkdir(parents=True)
        (tmp_path / "kitty" / "providers" / "registry.py").write_text(
            "_registry = _build_registry()\n", encoding="utf-8"
        )

        with pytest.raises(r.RegisterSourceError):
            r.provider_registry(tmp_path)

    def test_an_empty_registry_literal_raises(self, tmp_path: Path) -> None:
        """The third named refusal: a present-but-empty dict is no registry either.

        An empty key set would make every scope entry invalid in principle but
        validate nothing in practice — the same vacuous no-op, one step later.
        """
        (tmp_path / "kitty" / "providers").mkdir(parents=True)
        (tmp_path / "kitty" / "providers" / "registry.py").write_text(
            "_registry: dict[str, type[ProviderAdapter]] = {}\n", encoding="utf-8"
        )

        with pytest.raises(r.RegisterSourceError):
            r.provider_registry(tmp_path)

    def test_a_star_unpacking_entry_raises(self, tmp_path: Path) -> None:
        """The ``**``-unpacking refusal: a ``None`` key slot is no literal key.

        A registry assembled by merging another dict would let keys reach
        :func:`scope_problems` that the AST reader never saw — exactly the
        silent-widening shape the reader exists to refuse.
        """
        (tmp_path / "kitty" / "providers").mkdir(parents=True)
        (tmp_path / "kitty" / "providers" / "registry.py").write_text(
            "_OTHER = {}\n"
            "_registry: dict[str, type[ProviderAdapter]] = {**_OTHER, 'x': FooAdapter}\n",
            encoding="utf-8",
        )

        with pytest.raises(r.RegisterSourceError):
            r.provider_registry(tmp_path)

    def test_a_non_literal_key_raises(self, tmp_path: Path) -> None:
        """A bare name as a key is not a string literal — the ``literal_eval`` refusal."""
        (tmp_path / "kitty" / "providers").mkdir(parents=True)
        (tmp_path / "kitty" / "providers" / "registry.py").write_text(
            "KEY = 'x'\n"
            "_registry: dict[str, type[ProviderAdapter]] = {KEY: FooAdapter}\n",
            encoding="utf-8",
        )

        with pytest.raises(r.RegisterSourceError):
            r.provider_registry(tmp_path)

    def test_a_non_string_literal_key_raises(self, tmp_path: Path) -> None:
        """A literal key that is not a string (``{1: …}``) is refused, not coerced.

        The registry's keys are provider type strings; a numeric key would
        resolve to nothing in :func:`row_is_in_scope` and pass silently.
        """
        (tmp_path / "kitty" / "providers").mkdir(parents=True)
        (tmp_path / "kitty" / "providers" / "registry.py").write_text(
            "_registry: dict[str, type[ProviderAdapter]] = {1: FooAdapter}\n",
            encoding="utf-8",
        )

        with pytest.raises(r.RegisterSourceError):
            r.provider_registry(tmp_path)

    def test_a_non_class_value_raises(self, tmp_path: Path) -> None:
        """A call expression as a value is no bare class name — the site↔scope
        subset check needs the class's *name* to find its defining file."""
        (tmp_path / "kitty" / "providers").mkdir(parents=True)
        (tmp_path / "kitty" / "providers" / "registry.py").write_text(
            "def make_adapter():\n"
            "    return None\n"
            "\n"
            "_registry: dict[str, type[ProviderAdapter]] = {'x': make_adapter()}\n",
            encoding="utf-8",
        )

        with pytest.raises(r.RegisterSourceError):
            r.provider_registry(tmp_path)
