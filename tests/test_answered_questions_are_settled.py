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
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.l2

_DESIGN = Path(__file__).resolve().parents[1] / ".system_design"
_TEST_SUITE = _DESIGN / "TEST_SUITE.md"
_IMPLEMENTATION_PLAN = _DESIGN / "TEST_SUITE_IMPLEMENTATION_PLAN.md"

# A §11 question opens a paragraph in bold: `**Q9 — ANSWERED by the product
# owner, 2026-09-07.**` or `**Q14 — What is a correct stream recovery ...?**`.
# The number is captured; the rest of the header decides answered from open.
_QUESTION_HEADER = re.compile(r"^\*\*Q(\d+)\s*[–—-]\s*(.*)$", re.MULTILINE)

_ANSWERED_MARKER = "ANSWERED by the product owner"

# A mention is either a bare `Q14` or a range like `Q10-Q14` / `Q5–Q7`, which is
# how §11's own preamble and §15's closing prose refer to several at once.  The
# range form is the one the hand-written enumeration missed, so it is matched
# first and expanded.
_QUESTION_RANGE = re.compile(r"\bQ(\d+)\s*[–—-]\s*Q?(\d+)\b")
_QUESTION_MENTION = re.compile(r"\bQ(\d+)\b")

# Phrases that assert a question is still open, taken from the text that actually
# carried them rather than invented.  Deliberately narrow: bare "blocks" is far
# too common in an implementation plan to mean anything here, and a vocabulary
# that fires on it would be turned off within a week.
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
    "partial q",
)


def _section_eleven() -> str:
    """Return the text of ``TEST_SUITE.md`` §11, the open-questions section.

    Returns:
        Everything from the §11 heading to the end of the document, which is where
        the section ends.

    Raises:
        AssertionError: When the §11 heading is not found, which would make every
            assertion in this file vacuous.
    """
    text = _TEST_SUITE.read_text(encoding="utf-8")
    marker = "## 11. Open questions for the product owner"
    assert marker in text, f"{_TEST_SUITE.name} has no §11 heading; this guard cannot run"
    return text[text.index(marker) :]


def _questions() -> dict[int, str]:
    """Return every §11 question number mapped to the remainder of its header.

    Returns:
        Question number to header text.  The header is what distinguishes an
        answered question from an open one.
    """
    return {int(number): header for number, header in _QUESTION_HEADER.findall(_section_eleven())}


def _answered_question_numbers() -> set[int]:
    """Return the numbers of the §11 questions marked as answered.

    Returns:
        The question numbers whose header carries the answered marker.
    """
    return {number for number, header in _questions().items() if _ANSWERED_MARKER in header}


def _answer_paragraph_bounds(number: int) -> tuple[int, int]:
    """Return the character span of one question's own paragraph inside §11.

    The paragraph is excluded from the scan below.  §11's answered form keeps the
    original question verbatim after an ``*Original question:*`` marker, so the
    text that described the question as open survives there **by design** — that
    is the record of what was asked, not a stale claim.

    Args:
        number: The question number whose paragraph is wanted.

    Returns:
        The start and end offsets of the paragraph within the §11 text.
    """
    section = _section_eleven()
    starts = {int(m.group(1)): m.start() for m in _QUESTION_HEADER.finditer(section)}
    start = starts[number]
    later = [offset for other, offset in starts.items() if offset > start]
    return start, min(later) if later else len(section)


def _blocks(path: Path) -> list[tuple[str, str]]:
    """Split a design document into the units a question can be mentioned in.

    A table row is its own unit: the implementation plan states a task's blocking
    flag in one cell, and folding the whole table into a single paragraph would
    let one settled row mask an unsettled neighbour.

    Args:
        path: The document to split.

    Returns:
        Pairs of a human-readable location and the block's text.
    """
    blocks: list[tuple[str, str]] = []
    paragraph: list[str] = []
    paragraph_start = 0

    # Flush on a blank line or on crossing into a table, so a row is never glued
    # to the prose above it.
    def flush() -> None:
        if paragraph:
            blocks.append((f"{path.name}:{paragraph_start}", "\n".join(paragraph)))
            paragraph.clear()

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.startswith("|"):
            flush()
            blocks.append((f"{path.name}:{lineno}", line))
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


def _unsettled_mentions() -> list[tuple[str, int, list[str]]]:
    """Return every place an answered question is still described as open.

    Returns:
        Triples of location, question number, and the offending phrases.
    """
    answered = _answered_question_numbers()
    section_eleven_offset = _TEST_SUITE.read_text(encoding="utf-8").index(
        "## 11. Open questions for the product owner"
    )
    own_paragraphs = {
        number: tuple(offset + section_eleven_offset for offset in _answer_paragraph_bounds(number))
        for number in answered
    }
    suite_text = _TEST_SUITE.read_text(encoding="utf-8")

    findings: list[tuple[str, int, list[str]]] = []
    for path in (_TEST_SUITE, _IMPLEMENTATION_PLAN):
        for location, block in _blocks(path):
            phrases = _open_phrases_in(block)
            if not phrases:
                continue
            for number in sorted(_questions_mentioned(block) & answered):
                # A question's own §11 paragraph keeps the original wording and is
                # not a stale claim about its state.
                if path is _TEST_SUITE:
                    start, end = own_paragraphs[number]
                    offset = suite_text.find(block)
                    if offset != -1 and start <= offset < end:
                        continue
                findings.append((location, number, phrases))
    return findings


def test_the_open_questions_section_parses() -> None:
    """§11 must yield questions, and at least one of them must be answered.

    Without this the whole file passes on an empty set — the vacuous-guard failure
    ``TEST_SUITE.md`` §6.2 names as this layer's characteristic defect.
    """
    questions = _questions()

    assert len(questions) >= 14, f"§11 parsed only {sorted(questions)}; the header pattern has rotted"
    assert _answered_question_numbers(), "no §11 question parses as answered; the marker has changed"


def test_the_scan_finds_its_known_positives() -> None:
    """Both halves of the scan must fire on text known to trip them.

    A still-open vocabulary that matched nothing, or a mention pattern blind to the
    range form, would make :func:`test_every_answered_question_is_settled_everywhere`
    vacuously true while the contradiction it hunts sat in the document.
    """
    # The exact sentence that carried Q14 before it was answered.
    assert _open_phrases_in("**The acceptance oracle here is undecided — Q14.**")
    assert _open_phrases_in("| **T-I7** | Streaming recovery | partial Q14 | ... |")
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
    assert _questions_mentioned("Q14 blocks T-I7") == {14}

    # The block splitter must keep table rows apart, or one settled row masks its
    # neighbour.
    rows = [text for _, text in _blocks(_IMPLEMENTATION_PLAN) if text.startswith("|")]
    assert len(rows) > 50, "the implementation plan's tables collapsed into prose blocks"


def test_every_answered_question_is_settled_everywhere() -> None:
    """No passage may describe a question §11 has answered as still open.

    The assertion that earns this file's keep.  It fails the moment an answer is
    recorded in §11 while a dependent row, flag or table still waits on it.
    """
    unsettled = _unsettled_mentions()

    assert not unsettled, "answered questions still described as open:\n" + "\n".join(
        f"  {location} — Q{number} — {', '.join(phrases)}" for location, number, phrases in unsettled
    )


def test_the_blocked_on_decisions_table_names_no_answered_question() -> None:
    """§15's blocking table is keyed by question, and an answered one has no row.

    Checked separately from the phrase scan because the cell that carries the claim
    is the row *key*: a row reading ``| **Q14** | T-I7's positive assertions | ... |``
    states the block by position rather than by any word the vocabulary above holds.
    """
    text = _IMPLEMENTATION_PLAN.read_text(encoding="utf-8")
    start = text.index("## 15. Blocked on decisions")
    end = text.index("## 16.", start)
    answered = _answered_question_numbers()

    still_listed = sorted(
        {
            number
            for line in text[start:end].splitlines()
            if line.startswith("|")
            for number in _questions_mentioned(line.split("|")[1]) & answered
        }
    )

    assert not still_listed, f"§15 still lists answered questions as blocking: {still_listed}"
