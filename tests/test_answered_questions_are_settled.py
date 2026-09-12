"""KBR-163 — a question §11 calls ANSWERED may not be called open anywhere else.

``.system_design/TEST_SUITE.md`` §11 collects the open questions for the product
owner and marks each answered one in place.  Every answer also lands somewhere
else: §11 is where the decision is recorded, but the passages that *waited* on it
— a table row without an acceptance oracle, a task carrying a ``blocked`` flag,
the "Blocked on decisions" table in the implementation plan — have to stop saying
so on the same day.

Q14 was named in **six** places across the two documents, and the enumeration
written by hand while answering it found five.  That is the defect this file
exists to make impossible: answering a question in §11 and leaving another
passage describing it as undecided is a design that contradicts itself, and it
reads green to every other test in the suite.

Two artifacts edited separately that must agree, both readable statically — the
``TEST_SUITE.md`` §6.2.3 "Register and docs ⇄ code" case — so this file is L2 and
is named in ``tests/test_layer_selection.py``'s allowlist.  It asserts nothing
about *what* any answer says: an answer is a product decision, and a test that
pinned its wording would fail on every legitimate revision.

**Two stated limits, so a green run is not read as more than it is.**  First, a
passage is seen only if it *names the question number* and uses one of the
phrases in :data:`_STILL_OPEN_PHRASES`; a paraphrase that describes a question as
open without naming it is invisible here and stays a reading job.  Second, an
answer's own §11 entry is excluded — it keeps the original question verbatim, so
the open-state wording survives there by design — and that exclusion ends at the
next question or the next ``##`` heading, never silently at end of file.
:func:`test_the_answer_exclusion_is_bounded` holds the second limit, because an
exclusion that quietly grows is indistinguishable from a guard that has gone
blind (§6.2.3).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.l2

_DESIGN = Path(__file__).resolve().parents[1] / ".system_design"
_TEST_SUITE = _DESIGN / "TEST_SUITE.md"
_IMPLEMENTATION_PLAN = _DESIGN / "TEST_SUITE_IMPLEMENTATION_PLAN.md"

_SECTION_ELEVEN_HEADING = "## 11. Open questions for the product owner"
_BLOCKED_ON_DECISIONS_HEADING = "## 15. Blocked on decisions"

# A §11 question opens a paragraph in bold: `**Q9 — ANSWERED by the product
# owner, 2026-09-07.**` or `**Q14 — What is a correct stream recovery ...?**`.
_QUESTION_HEADER = re.compile(r"^\*\*Q(\d+)\s*[–—-]\s*(.*)$")
_MARKDOWN_SECTION = re.compile(r"^## ")

_ANSWERED_MARKER = "ANSWERED by the product owner"

# A mention is either a bare `Q14` or a range like `Q10-Q14` / `Q5–Q7`, which is
# how §11's preamble and §15's closing prose refer to several at once.  The range
# form is the one the hand-written enumeration missed, so it is expanded.
_QUESTION_RANGE = re.compile(r"\bQ(\d+)\s*[–—-]\s*Q?(\d+)\b")
_QUESTION_MENTION = re.compile(r"\bQ(\d+)\b")

# Phrases that assert a question is still open, taken from the text that actually
# carried them rather than invented.  Deliberately narrow: bare "blocks" is far
# too common in an implementation plan to mean anything here, and a vocabulary
# that fired on it would be switched off within a week.  "blocked on q" is the
# targeted form that survives that exclusion.
_STILL_OPEN_PHRASES = (
    "undecided",
    "unanswered",
    "not decided",
    "until it is answered",
    "until this is decided",
    "waits on",
    "is a prerequisite",
    "are prerequisites",
    "blocked q",
    "blocked on q",
    "partial q",
    "still open",
    "is pending",
)


def _read(path: Path, override: str | None = None) -> str:
    """Return a document's text, or a caller-supplied stand-in for it.

    The override is what lets the falsification cases below drive the real scan
    over a document carrying a planted defect, rather than exercising the
    matchers in isolation and proving nothing about their assembly.

    Args:
        path: The document to read when no override is given.
        override: Text to use instead of the file's contents.

    Returns:
        The document text.
    """
    return path.read_text(encoding="utf-8") if override is None else override


def _question_header_lines(suite_text: str) -> dict[int, int]:
    """Return each §11 question number mapped to its 1-based header line.

    Args:
        suite_text: The full text of ``TEST_SUITE.md``.

    Returns:
        Question number to the line its header sits on.

    Raises:
        AssertionError: When §11 is absent, which would make this file vacuous.
    """
    lines = suite_text.splitlines()
    assert _SECTION_ELEVEN_HEADING in suite_text, "TEST_SUITE.md has no §11; this guard cannot run"
    first = next(i for i, line in enumerate(lines) if line.startswith(_SECTION_ELEVEN_HEADING))

    return {
        int(match.group(1)): lineno
        for lineno, line in enumerate(lines[first:], start=first + 1)
        if (match := _QUESTION_HEADER.match(line))
    }


def _questions(suite_text: str) -> dict[int, str]:
    """Return every §11 question number mapped to the remainder of its header.

    Args:
        suite_text: The full text of ``TEST_SUITE.md``.

    Returns:
        Question number to header text, which is what separates an answered
        question from an open one.
    """
    lines = suite_text.splitlines()
    return {
        number: _QUESTION_HEADER.match(lines[lineno - 1]).group(2)  # type: ignore[union-attr]
        for number, lineno in _question_header_lines(suite_text).items()
    }


def _answered_question_numbers(suite_text: str) -> set[int]:
    """Return the numbers of the §11 questions marked as answered.

    Args:
        suite_text: The full text of ``TEST_SUITE.md``.

    Returns:
        The question numbers whose header carries the answered marker.
    """
    return {number for number, header in _questions(suite_text).items() if _ANSWERED_MARKER in header}


def _answer_line_span(suite_text: str, number: int) -> tuple[int, int]:
    """Return the 1-based line span of one question's own §11 entry.

    The entry ends at the next question **or at the next ``##`` heading**.  The
    heading bound is load-bearing: §11 is the last section and the highest-
    numbered question is the newest, so without it that question's entry would
    run to end of file and swallow anything appended after it — which is exactly
    where the next section would land.

    Args:
        suite_text: The full text of ``TEST_SUITE.md``.
        number: The question whose entry is wanted.

    Returns:
        The first and last line of the entry, inclusive of the first.
    """
    lines = suite_text.splitlines()
    starts = _question_header_lines(suite_text)
    start = starts[number]

    later = [lineno for lineno in starts.values() if lineno > start]
    later += [
        lineno
        for lineno, line in enumerate(lines, start=1)
        if lineno > start and _MARKDOWN_SECTION.match(line)
    ]
    return start, min(later) if later else len(lines) + 1


def _blocks(text: str, name: str) -> list[tuple[str, int, str]]:
    """Split a design document into the units a question can be mentioned in.

    A table row is its own unit: the implementation plan states a task's blocking
    flag in one cell, and folding a table into a single paragraph would let one
    settled row mask an unsettled neighbour.

    Args:
        text: The document text.
        name: The document's file name, used to label each block.

    Returns:
        Triples of label, 1-based starting line, and block text.
    """
    blocks: list[tuple[str, int, str]] = []
    paragraph: list[str] = []
    paragraph_start = 0

    # Flush on a blank line and on crossing into a table, so a row is never glued
    # to the prose above it.
    def flush() -> None:
        """Close the paragraph under construction, if any."""
        if paragraph:
            blocks.append((name, paragraph_start, "\n".join(paragraph)))
            paragraph.clear()

    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.startswith("|"):
            flush()
            blocks.append((name, lineno, line))
        elif not line.strip():
            flush()
        else:
            if not paragraph:
                paragraph_start = lineno
            paragraph.append(line)
    flush()
    return blocks


def _questions_mentioned(block: str) -> set[int]:
    """Return every question number a block refers to, ranges expanded.

    Args:
        block: The block text to read.

    Returns:
        The question numbers mentioned, with ``Q10-Q14`` expanded to all five.
    """
    mentioned = {int(n) for n in _QUESTION_MENTION.findall(block)}
    for low, high in _QUESTION_RANGE.findall(block):
        mentioned.update(range(int(low), int(high) + 1))
    return mentioned


def _open_phrases_in(block: str) -> list[str]:
    """Return the still-open phrases a block contains.

    Args:
        block: The block text to read.

    Returns:
        The matching phrases, lower-cased, in vocabulary order.
    """
    lowered = block.lower()
    return [phrase for phrase in _STILL_OPEN_PHRASES if phrase in lowered]


def _unsettled_mentions(
    suite_text: str | None = None, plan_text: str | None = None
) -> list[tuple[str, int, int, list[str]]]:
    """Return every place an answered question is still described as open.

    Args:
        suite_text: Stand-in text for ``TEST_SUITE.md``; the file is read when
            omitted.
        plan_text: Stand-in text for the implementation plan; likewise.

    Returns:
        Tuples of document name, line, question number, and offending phrases.
    """
    suite = _read(_TEST_SUITE, suite_text)
    plan = _read(_IMPLEMENTATION_PLAN, plan_text)
    answered = _answered_question_numbers(suite)
    own_entries = {number: _answer_line_span(suite, number) for number in answered}

    findings: list[tuple[str, int, int, list[str]]] = []
    for name, text in ((_TEST_SUITE.name, suite), (_IMPLEMENTATION_PLAN.name, plan)):
        for _, lineno, block in _blocks(text, name):
            phrases = _open_phrases_in(block)
            if not phrases:
                continue
            for number in sorted(_questions_mentioned(block) & answered):
                # A question's own §11 entry keeps the original wording verbatim
                # and is not a stale claim about its state.
                start, end = own_entries[number]
                if name == _TEST_SUITE.name and start <= lineno < end:
                    continue
                findings.append((name, lineno, number, phrases))
    return findings


def _blocking_table_questions(plan_text: str) -> set[int]:
    """Return the question numbers §15's blocking table has a row for.

    The whole row is read, not its first cell: the docstring's claim that the
    block is stated by row *key* is true of the table as written and would stop
    being true after a column reorder.

    Args:
        plan_text: The full text of the implementation plan.

    Returns:
        Every question number named anywhere in a §15 table row.
    """
    start = plan_text.index(_BLOCKED_ON_DECISIONS_HEADING)
    end = plan_text.index("## 16.", start)

    return {
        number
        for line in plan_text[start:end].splitlines()
        if line.startswith("|")
        for number in _questions_mentioned(line)
    }


def test_the_open_questions_section_parses() -> None:
    """§11 must yield the questions this repository actually has.

    The answered set is pinned rather than merely required to be non-empty: Q9
    has been answered since 2026-09-07 and would satisfy a bare "something is
    answered" check on its own, so rewording a *later* question's marker would
    silently drop it out of the scan while every test stayed green.
    """
    questions = _questions(_read(_TEST_SUITE))

    assert len(questions) >= 14, f"§11 parsed only {sorted(questions)}; the header pattern has rotted"
    assert _answered_question_numbers(_read(_TEST_SUITE)) >= {9, 14}


def test_the_scan_finds_its_known_positives() -> None:
    """Each matcher must fire on text known to trip it.

    A vocabulary that matched nothing, or a mention pattern blind to the range
    form, would make the assertions below vacuously true.
    """
    assert _open_phrases_in("**The acceptance oracle here is undecided — Q14.**")
    assert _open_phrases_in("| **T-I7** | Streaming recovery | partial Q14 | ... |")
    assert _open_phrases_in("T-I7 is blocked on Q14 and cannot land its positives")
    assert not _open_phrases_in("T-I8 asserts the cross-attempt content is byte-identical.")

    # The range form is what the hand-written enumeration missed.
    assert _questions_mentioned("Q10-Q14 are prerequisites for the implementation work") == {
        10,
        11,
        12,
        13,
        14,
    }
    assert _questions_mentioned("**Q5–Q7** affect register rows") == {5, 6, 7}

    # The block splitter must keep table rows apart, or one settled row masks its
    # neighbour.
    plan = _read(_IMPLEMENTATION_PLAN)
    rows = [text for _, _, text in _blocks(plan, "plan") if text.startswith("|")]
    assert len(rows) > 50, "the implementation plan's tables collapsed into prose blocks"


def test_a_planted_stale_claim_is_reported() -> None:
    """The assembled scan, not merely its matchers, must detect a real defect.

    Three plants, one per shape the documents can carry it in: prose in the body
    of ``TEST_SUITE.md``, a table row in the implementation plan, and — the case
    that motivated :func:`_answer_line_span`'s heading bound — a passage placed
    **after** the last question, where the newest answer's own entry would
    otherwise have swallowed it.
    """
    suite = _read(_TEST_SUITE)
    plan = _read(_IMPLEMENTATION_PLAN)

    heading = "## 1. What the suite must prove"
    in_body = suite.replace(heading, f"The Q14 oracle is undecided.\n\n{heading}", 1)

    assert _unsettled_mentions(suite_text=in_body)
    assert _unsettled_mentions(plan_text=plan + "\n| **T-I7** | blocked on Q14 |\n")
    assert _unsettled_mentions(suite_text=suite + "\n## 12. Appendix\n\nThe Q14 oracle is undecided.\n")


def test_the_answer_exclusion_is_bounded() -> None:
    """No answer's own entry may swallow the rest of the document.

    The exclusion is the one place this guard deliberately looks away, so it
    carries its own assertion — §6.2.3's rule, since an exclusion that stops
    matching is indistinguishable from a guard that has gone blind.  Adding a
    ``## 12`` heading must shorten the last answer's entry, not extend it.
    """
    suite = _read(_TEST_SUITE)
    total_lines = len(suite.splitlines())

    for number in sorted(_answered_question_numbers(suite)):
        start, end = _answer_line_span(suite, number)
        assert start < end <= total_lines + 1

    # The highest-numbered question is the newest, and §11 is the last section:
    # its entry ends at EOF only until something is appended, and must then move.
    newest = max(_answered_question_numbers(suite))
    extended = suite + "\n## 12. Appendix\n\nUnrelated prose.\n"
    assert _answer_line_span(extended, newest)[1] < len(extended.splitlines()) + 1


def test_every_answered_question_is_settled_everywhere() -> None:
    """No passage may describe a question §11 has answered as still open.

    The assertion that earns this file's keep.  It fails the moment an answer is
    recorded in §11 while a dependent row, flag or table still waits on it.
    """
    unsettled = _unsettled_mentions()

    assert not unsettled, "answered questions still described as open:\n" + "\n".join(
        f"  {name}:{lineno} — Q{number} — {', '.join(phrases)}"
        for name, lineno, number, phrases in unsettled
    )


def test_the_blocked_on_decisions_table_names_no_answered_question() -> None:
    """§15's blocking table is keyed by question, and an answered one has no row.

    Checked separately from the phrase scan because a row states the block by
    position rather than by any word the vocabulary holds: ``| **Q14** | T-I7's
    positive assertions | ... |`` contains no still-open phrase at all.
    """
    plan = _read(_IMPLEMENTATION_PLAN)
    still_listed = sorted(_blocking_table_questions(plan) & _answered_question_numbers(_read(_TEST_SUITE)))

    assert not still_listed, f"§15 still lists answered questions as blocking: {still_listed}"
