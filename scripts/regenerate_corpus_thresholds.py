#!/usr/bin/env python3
"""Materialise the T-C3 threshold corpus entries into ``tests/corpus/``.

The four entries this writes are the M3 / M5 threshold pairs — see
``tests/harness/corpus_thresholds.py`` for the builders and
``tests/corpus/README.md`` §"Threshold-pair entries (T-C3)" for the product
decision this script records.

Run from the repository root::

    python scripts/regenerate_corpus_thresholds.py

The write goes through :func:`harness.corpus.write_entry`, so the reader's own
validation, scrubbing, and manifest building all run. The script is
idempotent: rerunning it on a tree whose entries are current leaves
``git status`` clean, which is the property
``tests/harness/test_corpus_thresholds.py::TestCommittedArtifactsRegenerate``
guards.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "tests"))

from harness import corpus as k  # noqa: E402
from harness import corpus_thresholds as ct  # noqa: E402
from harness.contract import WireFormat  # noqa: E402

#: The four entry ids this script writes. The description and origin_note for
#: each id live in :data:`corpus_thresholds.entry_metadata` — the single source
#: of truth, so a future edit to the manifest prose propagates to the committed
#: artifact through one path.
_ENTRY_IDS = (
    "tool_result_under_limit",
    "tool_result_over_limit",
    "compaction_budget_under",
    "compaction_budget_over",
)


def main() -> int:
    """Write the four entries and print their paths.

    Returns:
        0 on success; 1 on failure (any write raises).
    """
    corpus_root = _REPO_ROOT / "tests" / "corpus"
    for entry_id in _ENTRY_IDS:
        captured, met, absent = _BUILD(entry_id)()
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
        path = k.write_entry(corpus_root, entry)
        print(f"wrote {path}")
    return 0


def _BUILD(entry_id: str):
    """Return the builder function for ``entry_id``.

    Args:
        entry_id: One of the four ids the builders know.

    Returns:
        The builder function.

    Raises:
        KeyError: When ``entry_id`` is not one of the four.
    """
    builders: dict[str, object] = {
        "tool_result_under_limit": ct.build_tool_result_under_limit,
        "tool_result_over_limit": ct.build_tool_result_over_limit,
        "compaction_budget_under": ct.build_compaction_budget_under,
        "compaction_budget_over": ct.build_compaction_budget_over,
    }
    return builders[entry_id]


if __name__ == "__main__":
    raise SystemExit(main())
