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
from collections.abc import Callable
from pathlib import Path

import pytest

from harness import corpus as k
from harness import corpus_thresholds as ct
from harness.contract import CapturedRequest, WireFormat
from harness.register import Trigger

#: The bridge's CC translator — imported so the L1 tests can measure what
#: ``_compact_messages`` actually compares against.
from kitty.bridge.messages.translator import MessagesTranslator

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
        """The under-budget entry's CC-converted messages serialise to exactly the threshold.

        ``_compact_messages`` short-circuits on ``original_size <= threshold``,
        so the threshold itself is the largest short-circuit size — mirroring
        the M3 pair's boundary pattern. The CC-converted shape is what the
        bridge measures (server.py:6996 ``_safe_size``); pinning against the
        Anthropic-Messages shape would land the fixture 92 chars off the
        boundary the bridge actually compares.
        """
        captured, met, absent = ct.build_compaction_budget_under()
        assert absent == frozenset({Trigger.TOOL_RESULT_OVER_LIMIT})
        assert met == frozenset()
        cc_len = _cc_messages_serialized(captured)
        assert cc_len == _COMPACTION_CHAR_THRESHOLD, (
            f"CC-converted messages length {cc_len} != {_COMPACTION_CHAR_THRESHOLD}"
        )

    def test_over_budget_filler_alone_crosses_threshold(self) -> None:
        """The over-budget entry's CC-converted messages length exceeds the threshold.

        Captured WITHOUT the oversized tool_result's contribution, the filler
        is sized so the CC-converted length is exactly ``threshold + 1`` — the
        smallest value ``original_size > threshold`` accepts. Adding the 50 001-char
        tool_result on top pushes the total past ``threshold + 50 000``.
        """
        captured, met, absent = ct.build_compaction_budget_over()
        assert met == frozenset({Trigger.TOOL_RESULT_OVER_LIMIT})
        assert absent == frozenset()
        cc_len = _cc_messages_serialized(captured)
        assert cc_len > _COMPACTION_CHAR_THRESHOLD, (
            f"CC-converted messages length {cc_len} not above {_COMPACTION_CHAR_THRESHOLD}; "
            "the over entry must cross the threshold the bridge compares against"
        )

    def test_over_budget_still_over_threshold_after_m3_truncation(self) -> None:
        """After M3 truncates the 50 001-char tool_result, the body is still over budget.

        This is the failure mode the first iteration had: with the filler at
        exactly ``threshold + 1`` and the 50 001-char tool_result on top,
        M3's truncation drops the body ~50 000 chars and it falls below the
        threshold, so M5 short-circuits instead of firing the pruning step.
        The resized build keeps the filler alone over threshold so the
        post-truncation body remains above it.
        """
        captured, _met, _absent = ct.build_compaction_budget_over()
        cc = _translate_to_cc(captured)
        _apply_m3_truncation(cc["messages"])
        post_m3_cc = len(json.dumps(cc["messages"], ensure_ascii=False))
        assert post_m3_cc > _COMPACTION_CHAR_THRESHOLD, (
            f"post-M3 CC length {post_m3_cc} is not above {_COMPACTION_CHAR_THRESHOLD}; "
            "M5's pruning step would short-circuit instead of fire"
        )

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
_BUILD_BY_ID: dict[str, Callable[[], tuple[CapturedRequest, frozenset, frozenset]]] = {
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


def _cc_messages_serialized(captured: CapturedRequest) -> int:
    """Return ``len(json.dumps(messages, ensure_ascii=False))`` on the CC-converted shape.

    This is the property ``_compact_messages`` (``server.py:6996``) measures
    when deciding whether the budget is exceeded. The Anthropic-Messages shape
    the fixture commits differs from the CC-converted shape by a constant ~92
    chars for the no-tool-result layout; pinning against the CC shape is
    what makes the fixture land on the boundary the bridge actually compares.

    Args:
        captured: The fixture's captured request.

    Returns:
        The length the bridge's ``_safe_size`` would observe.
    """
    body = json.loads(captured.body.decode("utf-8"))
    cc = _TRANSLATOR.translate_request(body)
    return len(json.dumps(cc["messages"], ensure_ascii=False))


def _translate_to_cc(captured: CapturedRequest) -> dict:
    """Run ``MessagesTranslator.translate_request`` on the captured body.

    Returns the CC-converted request dict; callers mutate ``["messages"]``
    in place to simulate the bridge's mutation chain.
    """
    body = json.loads(captured.body.decode("utf-8"))
    return _TRANSLATOR.translate_request(body)


def _apply_m3_truncation(messages: list[dict]) -> int:
    """Run the M3 mutation sites against ``messages`` in place; return the count of truncations.

    Mirrors ``server.py:7267-7294`` (CC and Anthropic-native shapes) and
    ``server.py:7314-7325`` (Responses shape, which this fixture does not
    exercise) so the post-M3 length is exactly what the bridge would
    observe. Used by the post-M3 over-budget tests.
    """
    truncated = 0
    for msg in messages:
        if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
            content_len = len(msg["content"])
            if content_len > _TOOL_RESULT_TRUNCATION_LIMIT:
                msg["content"] = (
                    f"[Tool output truncated — original size: {content_len:,} chars]"
                )
                truncated += 1
            continue
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and isinstance(block.get("content"), str)
                    and len(block["content"]) > _TOOL_RESULT_TRUNCATION_LIMIT
                ):
                    original_len = len(block["content"])
                    block["content"] = (
                        f"[Tool output truncated — original size: {original_len:,} chars]"
                    )
                    truncated += 1
    return truncated


#: The bridge's CC translator — used by the CC-shape measurements below.
_TRANSLATOR = MessagesTranslator()


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
