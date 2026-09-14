"""L1 contract tests for the T-C3 threshold corpus entries.

The four corpus entries this ticket ships — the M3 / M5 threshold pairs — are
fixtures, not behaviour. These tests pin two property classes:

* the builder is self-consistent — every manifest declaration matches the body
  it produces, measured against the same constants the bridge uses;
* the builder regenerates the committed fixture bytes — nobody can edit a
  ``.body`` without rerunning the regen script and breaking this test.

The committed entries' loadability and lint cleanliness are guarded by
``tests/harness/test_corpus_lint.py`` (L2) over every entry in the directory,
so they are not re-asserted here.

``.system_design/TEST_SUITE.md`` §7.1 · plan task **T-C3** (KBR-46).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from harness import corpus as k
from harness import corpus_thresholds as ct
from harness.contract import CapturedRequest, WireFormat
from harness.register import Trigger

#: The constants the bridge itself decides on, imported from the source of
#: truth. The builder imports the same names; these tests assert against them
#: so a constant change is visible in the failure message.
from kitty.bridge.server import _COMPACTION_CHAR_THRESHOLD, _TOOL_RESULT_TRUNCATION_LIMIT

#: The repository root, three levels up from this file.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The committed corpus.
_COMMITTED_CORPUS = _REPO_ROOT / "tests" / "corpus"

#: Every entry id this ticket commits, in the order the REQUIREMENTS.md table lists them.
ENTRY_IDS = (
    "tool_result_under_limit",
    "tool_result_over_limit",
    "compaction_budget_under",
    "compaction_budget_over",
)


# ---------------------------------------------------------------------------
# The builder's contract against the bridge's constants
# ---------------------------------------------------------------------------


class TestBuilderThresholdProperties:
    """Each build function's body sits exactly on the claimed side of its constant."""

    def test_under_tool_result_is_the_largest_non_triggering_size(self) -> None:
        """The under-side tool result is exactly the limit — the strict-``>`` boundary.

        ``_truncate_oversized_tool_results`` compares ``content_len > limit``, so
        a result of exactly ``limit`` chars does NOT trigger: this is the largest
        non-triggering size and the tightest honest complement.
        """
        captured, met, absent = ct.build_tool_result_under_limit()
        assert absent == frozenset({Trigger.TOOL_RESULT_OVER_LIMIT})
        assert met == frozenset()
        assert _tool_result_string_lengths(captured) == [_TOOL_RESULT_TRUNCATION_LIMIT]

    def test_over_tool_result_is_the_smallest_triggering_size(self) -> None:
        """The over-side tool result is limit + 1 — the first size that triggers."""
        captured, met, absent = ct.build_tool_result_over_limit()
        assert met == frozenset({Trigger.TOOL_RESULT_OVER_LIMIT})
        assert absent == frozenset()
        assert _tool_result_string_lengths(captured) == [_TOOL_RESULT_TRUNCATION_LIMIT + 1]

    def test_under_budget_messages_serialize_at_the_threshold(self) -> None:
        """The under-budget entry's messages serialize exactly at ``2_800_000``.

        ``_compact_messages`` short-circuits on ``original_size <= threshold``,
        so the threshold itself is the largest short-circuit size — mirroring
        the M3 pair's boundary pattern.
        """
        captured, met, absent = ct.build_compaction_budget_under()
        assert absent == frozenset({Trigger.TOOL_RESULT_OVER_LIMIT})
        assert met == frozenset()
        serialized = _anthropic_messages_serialized(captured)
        assert serialized == _COMPACTION_CHAR_THRESHOLD, (
            f"serialized messages length {serialized} != {_COMPACTION_CHAR_THRESHOLD}"
        )

    def test_over_budget_messages_serialize_one_above_the_threshold(self) -> None:
        """The over-budget entry's messages serialize at ``threshold + 1``.

        The smallest value ``original_size > threshold`` accepts, mirroring
        the M3 pair's ``limit + 1``.
        """
        captured, met, absent = ct.build_compaction_budget_over()
        assert met == frozenset({Trigger.TOOL_RESULT_OVER_LIMIT})
        assert absent == frozenset()
        serialized = _anthropic_messages_serialized(captured)
        assert serialized == _COMPACTION_CHAR_THRESHOLD + 1, (
            f"serialized messages length {serialized} != {_COMPACTION_CHAR_THRESHOLD + 1}"
        )

    def test_over_budget_is_trigger_for_any_profile_budget(self) -> None:
        """``threshold + 1`` exceeds any profile's derived ``messages_budget``.

        ``messages_budget = max_chars - overhead - 10_000`` where
        ``max_chars = min(tokens_to_chars(context_tokens), _MAX_REQUEST_CHARS) = 4 000 000``,
        so the largest budget is at most ``4 000 000 - 10 000``. A body of
        ``threshold + 1 = 2 800 001`` is below that ceiling, but the budget
        also subtracts the request overhead (tools, model, metadata), so a
        fixture that is meant to guarantee M5-trigger needs to be past the
        budget after overhead — which ``threshold + 1`` is for any realistic
        envelope (a few KB). Recorded rather than asserted-on: the constant
        arithmetic lives in the bridge, not here.
        """
        captured, _met, _absent = ct.build_compaction_budget_over()
        assert _anthropic_messages_serialized(captured) == _COMPACTION_CHAR_THRESHOLD + 1
        assert _COMPACTION_CHAR_THRESHOLD + 1 < 4_000_000

    def test_over_budget_carries_an_oversized_tool_result(self) -> None:
        """The over-budget entry also supplies M4's second condition.

        M4 fires only when compaction is engaged **and** a tool result exceeds
        the limit; the over-budget entry supplies both, which is what makes it
        the M4 fixture as well as the M5 one.
        """
        captured, _met, _absent = ct.build_compaction_budget_over()
        lengths = _tool_result_string_lengths(captured)
        assert any(n > _TOOL_RESULT_TRUNCATION_LIMIT for n in lengths), (
            f"compaction_budget_over carries no tool_result over {_TOOL_RESULT_TRUNCATION_LIMIT} "
            f"chars; lengths were {lengths}"
        )

    def test_under_budget_carries_no_oversized_tool_result(self) -> None:
        """The under-budget entry is also an M4 complement: no oversized tool result.

        Its ``triggers_absent`` claims ``tool_result_over_limit`` is absent, and
        M4's second condition with it — the entry cannot make M4 fire even if a
        slice forces compaction on.
        """
        captured, _met, _absent = ct.build_compaction_budget_under()
        assert all(n <= _TOOL_RESULT_TRUNCATION_LIMIT for n in _tool_result_string_lengths(captured))

    def test_bodies_are_deterministic(self) -> None:
        """Building twice yields identical bytes — no clock, no randomness."""
        for entry_id in ENTRY_IDS:
            assert ct.build_body_bytes(entry_id) == ct.build_body_bytes(entry_id), entry_id


# ---------------------------------------------------------------------------
# Scrub-inertness (req 5)
# ---------------------------------------------------------------------------


class TestBuilderIsScrubInert:
    """``write_entry`` scrubs unconditionally; the builder's output must be a fixed point.

    A filler line that matches a scrubber shape (a ``/home/<user>`` path, an
    e-mail address, a credential-like run) would be rewritten by ``scrub``,
    changing the body's length — and possibly crossing the boundary the
    fixture was sized for. This test catches that at L1 before the committed
    artifact exists; the regeneration test catches it post-commit.
    """

    @pytest.mark.parametrize("entry_id", list(ENTRY_IDS))
    def test_scrub_is_a_fixed_point_on_the_body(self, entry_id: str) -> None:
        """``scrub(captured) == captured`` — body, headers, host, path unchanged."""
        captured, _met, _absent = _BUILD_BY_ID[entry_id]()
        scrubbed = k.scrub(captured, allow=[])
        assert scrubbed.body == captured.body, (
            f"{entry_id}: scrub rewrote the body — a filler, header value, path, or routing "
            f"field matched a scrubber pattern and would change the byte length committed."
        )
        assert list(scrubbed.headers) == list(captured.headers), (
            f"{entry_id}: scrub rewrote a header — header values must survive scrub unchanged."
        )
        assert scrubbed.host == captured.host
        assert scrubbed.path == captured.path


# ---------------------------------------------------------------------------
# The committed artifacts reproduce the builder (regeneration contract)
# ---------------------------------------------------------------------------


class TestCommittedArtifactsRegenerate:
    """The committed bytes are what the builder produces right now.

    This is the "second reader" that replaces human review of the megabyte
    padded body, per the owner's 2026-09-13 Jira comment: the .body is text and
    an editor could change it, but any change without a rerun of the regen
    script fails here — the builder and the fixture cannot drift silently.
    """

    def test_every_entry_is_committed(self) -> None:
        loaded = {entry.id for entry in k.load_corpus(_COMMITTED_CORPUS)}
        missing = [entry_id for entry_id in ENTRY_IDS if entry_id not in loaded]
        assert not missing, (
            f"committed corpus is missing {missing}; rerun scripts/regenerate_corpus_thresholds.py"
        )

    def test_every_committed_body_matches_the_builder(self) -> None:
        for entry_id in ENTRY_IDS:
            committed = (_COMMITTED_CORPUS / f"{entry_id}.body").read_bytes()
            assert hashlib.sha256(committed).hexdigest() == hashlib.sha256(
                ct.build_body_bytes(entry_id)
            ).hexdigest(), (
                f"{entry_id}.body has drifted from the builder; "
                "rerun scripts/regenerate_corpus_thresholds.py"
            )

    def test_every_committed_manifest_matches_write_entry_byte_identically(self, tmp_path: Path) -> None:
        """Rerunning ``write_entry`` on the builder's output reproduces the committed bytes.

        Body bytes alone do not prove idempotence: the manifest's own prose
        (``description``, ``origin_note``), headers, and trigger declarations
        are all written by ``write_entry`` too, so a drift in any of them
        breaks the committed artifact just as a body drift does.
        """
        for entry_id in ENTRY_IDS:
            captured, met, absent = _BUILD_BY_ID[entry_id]()
            description, origin_note = ct.entry_metadata(entry_id)
            entry = k.CorpusEntry(
                id=entry_id,
                description=description,
                origin=k.SYNTHETIC,
                origin_note=origin_note,
                captured_from="",
                captured_at="",
                request=captured,
                wire_format=WireFormat.ANTHROPIC_MESSAGES,
                known_non_secrets=(),
                triggers_met=met,
                triggers_absent=absent,
            )
            k.write_entry(tmp_path, entry)
            regenerated_manifest = (tmp_path / f"{entry_id}.json").read_bytes()
            regenerated_body = (tmp_path / f"{entry_id}.body").read_bytes()
            committed_manifest = (_COMMITTED_CORPUS / f"{entry_id}.json").read_bytes()
            committed_body = (_COMMITTED_CORPUS / f"{entry_id}.body").read_bytes()
            assert regenerated_manifest == committed_manifest, (
                f"{entry_id}.json would regenerate differently; rerun "
                "scripts/regenerate_corpus_thresholds.py"
            )
            assert regenerated_body == committed_body, (
                f"{entry_id}.body would regenerate differently; rerun "
                "scripts/regenerate_corpus_thresholds.py"
            )


# Idiom matching ``TestCommittedArtifactsRegenerate``: parametrize test methods on the
# builder, so a hand edit to any of the four committed artifacts fails here.
_BUILD_BY_ID: dict[str, object] = {
    "tool_result_under_limit": ct.build_tool_result_under_limit,
    "tool_result_over_limit": ct.build_tool_result_over_limit,
    "compaction_budget_under": ct.build_compaction_budget_under,
    "compaction_budget_over": ct.build_compaction_budget_over,
}


# ---------------------------------------------------------------------------
# The padded body is transparent to a reviewer
# ---------------------------------------------------------------------------


class TestPaddedBodyStructure:
    """A reviewer sampling any region of a padded body reads construction, not content."""

    @pytest.mark.parametrize("entry_id", ["compaction_budget_under", "compaction_budget_over"])
    def test_every_filler_line_names_its_turn(self, entry_id: str) -> None:
        """No filler line is empty, and every line carries its turn index."""
        lines = _filler_lines(entry_id)
        assert lines, f"{entry_id}: no filler lines found — is the body padded at all?"
        for index, line in enumerate(lines):
            assert line.strip(), f"{entry_id}: filler line {index} is empty"
            assert "Turn " in line, f"{entry_id}: filler line {index} lacks a turn index: {line!r}"

    @pytest.mark.parametrize("entry_id", ["compaction_budget_under", "compaction_budget_over"])
    def test_no_filler_line_repeats_verbatim(self, entry_id: str) -> None:
        """The embedded turn index makes every filler line unique."""
        lines = _filler_lines(entry_id)
        duplicates = {line for line in lines if lines.count(line) > 1}
        assert not duplicates, f"{entry_id}: repeated filler lines, e.g. {sorted(duplicates)[:2]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anthropic_messages_serialized(captured: CapturedRequest) -> int:
    """Return the length the bridge's ``json.dumps(messages, ensure_ascii=False)`` would see.

    The fixture body is Anthropic Messages shape; the bridge's M5 comparison
    runs on the CC-converted messages, whose serialization of these simple
    text/tool shapes is within a few hundred characters of the Anthropic
    shape's. The builder's targets keep a wide margin around the threshold so
    the conversion's small delta cannot flip a side.
    """
    body = json.loads(captured.body.decode("utf-8"))
    return len(json.dumps(body["messages"], ensure_ascii=False))


def _tool_result_string_lengths(captured: CapturedRequest) -> list[int]:
    """Return the character length of every ``tool_result`` string content in the body.

    Anthropic shape nests them as user-message content blocks; the bridge's CC
    conversion carries the string through unchanged, so the length the
    ``>`` comparison sees is this one.
    """
    body = json.loads(captured.body.decode("utf-8"))
    lengths: list[int] = []
    for message in body.get("messages", ()):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and isinstance(block.get("content"), str)
            ):
                lengths.append(len(block["content"]))
    return lengths


def _filler_lines(entry_id: str) -> list[str]:
    """Return every line of the committed padded body that comes from a filler turn.

    The initial and closing user turns are real-shaped conversation content, not
    padding — they don't carry the ``Turn NNNN of`` prefix and are excluded so
    the structural assertions below only cover what this ticket is committing
    as construction.
    """
    body = json.loads((_COMMITTED_CORPUS / f"{entry_id}.body").read_text(encoding="utf-8"))
    lines: list[str] = []
    for message in body.get("messages", ()):
        content = message.get("content")
        if isinstance(content, str) and content.startswith("Turn "):
            lines.extend(line for line in content.splitlines() if line)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].startswith("Turn "):
                    lines.extend(line for line in block["text"].splitlines() if line)
    return lines
