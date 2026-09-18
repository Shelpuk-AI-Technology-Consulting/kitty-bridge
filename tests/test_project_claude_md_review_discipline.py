"""Contract guard — KBR-286: the project ``CLAUDE.md`` must keep carrying the
PR-review resolution instruction.

``CLAUDE.md`` is the instruction every Claude Code session on this repository
acts on, yet nothing imports it, so it is the one artifact in the tree with no
other drift enforcement. Two artifacts edited separately must agree: the file
itself and git's view of it (tracked, not ignored). The guard fails loudly
when the file is missing, untracked, re-ignored, or rewritten without one of
the four mandated review-resolution behaviours (PR-level summary comment,
optional in-thread replies, enumerate-then-resolve fixed conversations, and
the declined-conversation disposition).

The anchors name the behaviours themselves, and are asserted inside the
review-resolution section only, so copy edits around them do not churn CI and
an unrelated future section cannot satisfy them. §6.2's two-artifacts shape;
the same guard idiom as ``test_custom_url_docs.py`` and the README table
guards.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.l2

_ROOT = Path(__file__).resolve().parents[1]
_CLAUDE_MD = _ROOT / "CLAUDE.md"
_REVIEW_SECTION_HEADING = "## Resolving GitHub code reviews"

# One entry per behaviour the ticket mandates. The anchors are phrases that
# only the behaviour's own wording contains; losing any one of them is losing
# the behaviour, which is exactly what this guard exists to report.
_CLAUSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "the PR-level summary comment (what was addressed and implemented, "
        "what was deferred)",
        ("PR-level comment", "deferred"),
    ),
    (
        "the optional in-thread replies with further detail",
        ("in-thread",),
    ),
    (
        "enumerate every open conversation and inline review thread, then "
        "resolve the conversations that were taken and fixed",
        (
            "every open conversation",
            "inline review thread",
            "Resolve every conversation that was taken and fixed",
        ),
    ),
    (
        "the declined-conversation disposition (never silently resolved, "
        "left open with a stated reason)",
        ("never silently resolved", "leave the conversation open"),
    ),
)


def _review_section() -> str:
    """Return the review-resolution section of ``CLAUDE.md``.

    Returns:
        The text between the review-resolution heading and the next
        same-level heading, whitespace-normalised, so clause anchors cannot
        be satisfied by an unrelated section and cannot be broken by a
        re-wrap of the same wording.

    Raises:
        AssertionError: When the heading is gone — a rename or deletion of
            the section is the first thing this guard must report.
    """
    # Read as UTF-8 explicitly: the file carries em-dashes, and a cp1252
    # default locale would otherwise break the read (KBR-266).
    text = _CLAUDE_MD.read_text(encoding="utf-8")
    start = text.find(_REVIEW_SECTION_HEADING)
    assert start != -1, (
        f"CLAUDE.md no longer contains the {_REVIEW_SECTION_HEADING!r} section"
    )
    rest = text[start + len(_REVIEW_SECTION_HEADING):]
    next_heading = rest.find("\n## ")
    body = rest if next_heading == -1 else rest[:next_heading]
    # Line wrapping is formatting, not behaviour: collapse all whitespace
    # runs so an anchor phrase survives a re-wrap of the same sentence.
    return re.sub(r"\s+", " ", body)


class TestProjectClaudeMdShips:
    """The file is present, tracked, and not gitignored.

    The ``.gitignore`` once carried a bare ``/CLAUDE.md`` line (commit
    ``b201181``, 2026-04-27, added without rationale in an unrelated
    change), which made the KBR-286 instruction unshippable; ``git
    check-ignore`` is the authoritative check because it evaluates every
    ignore pattern form, not just a literal line match.
    """

    def test_the_file_exists_at_the_repo_root(self) -> None:
        """The project CLAUDE.md is present at the repository root."""
        assert _CLAUDE_MD.exists(), (
            "CLAUDE.md is missing from the repository root; the KBR-286 "
            "instruction does not reach any session"
        )

    def test_the_file_is_tracked(self) -> None:
        """The project CLAUDE.md is committed, not a local-only artifact."""
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "CLAUDE.md"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            "CLAUDE.md exists on disk but is not tracked by git "
            f"(git ls-files --error-unmatch exited {result.returncode})"
        )

    def test_the_file_is_not_gitignored(self) -> None:
        """No ignore pattern matches the project CLAUDE.md."""
        result = subprocess.run(
            ["git", "check-ignore", "CLAUDE.md"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
        )
        # Exit 0 means "ignored", 1 means "not ignored", 128 means git
        # itself failed — only the middle state is acceptable.
        assert result.returncode == 1, (
            "CLAUDE.md is matched by .gitignore "
            f"(git check-ignore exited {result.returncode}); the KBR-286 "
            "instruction would silently stop shipping"
        )


class TestTheReviewResolutionSection:
    """Every mandated behaviour stays stated in the review-resolution section."""

    @pytest.mark.parametrize(("behaviour", "anchors"), _CLAUSES)
    def test_the_section_states_the_behaviour(
        self, behaviour: str, anchors: tuple[str, ...]
    ) -> None:
        """The review-resolution section still states one mandated behaviour.

        Args:
            behaviour: Human-readable name of the mandated behaviour, used
                in the failure message.
            anchors: Phrases that only the behaviour's own wording
                contains; all must be present in the section.
        """
        missing = [anchor for anchor in anchors if anchor not in _review_section()]
        assert not missing, (
            "CLAUDE.md's review-resolution section lost "
            f"{behaviour}: missing anchor(s) {missing!r}"
        )
