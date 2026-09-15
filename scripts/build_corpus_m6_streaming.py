#!/usr/bin/env python3
"""Build the streaming twin of the M6 corpus entry (KBR-256).

Constructs ``m6_recovery_oversized_paired_streaming`` from the committed
``m6_recovery_oversized_paired`` entry by flipping ``stream: false`` to
``stream: true`` and swapping ``Accept: application/json`` for
``Accept: text/event-stream``. The conversation shape, the well-paired property,
the post-pre-flight oversized-gate size, and the deterministic filler all carry
over unchanged — the entry was already synthetic under the T-C4 blessing
(``design section 7.1's real-not-synthetic rationale does not apply to it``),
and an oversized well-paired *streaming* conversation is equally not capturable
on demand from a real Claude Code session.

Usage::

    PYTHONPATH=tests ./.venv/bin/python scripts/build_corpus_m6_streaming.py

The script is the "builder (this script)" the entry's ``origin_note`` names in
place of the human-review of every body step, per the corpus README's
§7.1.1 synthetic-entry exception. It writes via
:func:`harness.corpus.write_entry`, which scrubs credentials on the way out and
refuses any carry-over of compression or transfer-encoding headers.

The L2 corpus lint (``tests/harness/test_corpus_lint.py``) and the corpus-wiring
L1 tests (``tests/bridge/test_tc4_corpus_wiring.py``) hold the twin's shape on
commit. The owner action item (read every ``.body`` before commit, per corpus
README §7.1.1 step 5) is owned by the human merging the PR, not by this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "tests"))

from harness import corpus as k  # noqa: E402

ENTRY_ID = "m6_recovery_oversized_paired_streaming"
SOURCE_ID = "m6_recovery_oversized_paired"
CORPUS = WORKTREE / "tests" / "corpus"


def _flip_to_streaming(source: k.CorpusEntry) -> k.CorpusEntry:
    """Return ``source`` with the streaming twin's two wire-shape changes.

    The conversation body is byte-identical apart from the literal four-vs-five
    character change in the ``stream`` field; every header is preserved except
    ``Accept`` (which a streaming Claude Code request sets to ``text/event-stream``)
    and ``Content-Length`` (which :func:`harness.corpus.scrub` recomputes from
    the body anyway).

    Args:
        source: The non-streaming twin's loaded entry.

    Returns:
        A new :class:`~harness.corpus.CorpusEntry` for the streaming path.
    """
    body_obj = json.loads(source.request.body)
    body_obj["stream"] = True
    new_body = json.dumps(body_obj).encode("utf-8")

    new_headers: list[tuple[str, str]] = []
    for name, value in source.request.headers:
        # Drop the old Content-Length — scrub recomputes it; mirror for Accept.
        if name.lower() == "content-length":
            continue
        if name == "Accept":
            new_headers.append((name, "text/event-stream"))
            continue
        new_headers.append((name, value))

    return k.CorpusEntry(
        id=ENTRY_ID,
        description=(
            "Streaming twin of m6_recovery_oversized_paired: a well-paired "
            "conversation whose post-pre-flight serialized size exceeds the "
            "600,000-char oversized gate, issued with stream: true so the "
            "bridge's streaming Messages path engages. Exercises the KBR-256 "
            "compact-and-retry-same-backend recovery on the pre-byte 413 path."
        ),
        origin="synthetic",
        origin_note=(
            "Synthetic, not captured: an oversized well-paired streaming "
            "conversation does not occur in a real Claude Code session on "
            "demand (Claude Code ships stream: true, but the conversation "
            "shape that exceeds the 600,000-char gate only fires under "
            "test-side bodies; the same T-C4 rationale as the non-streaming "
            "twin applies). The size is constructed from the non-streaming "
            "twin's body with a single field flip — the deterministic filler "
            "and the well-paired shape carry over. The mandatory "
            "human-review step is replaced by code-review of the builder "
            "(scripts/build_corpus_m6_streaming.py), the scrubber's full-byte "
            "scan in CI, and a bounded head/tail human read of the committed "
            ".body file, per TEST_SUITE.md section 7.1.1."
        ),
        captured_from="",
        captured_at="",
        request=type(source.request)(
            method=source.request.method,
            scheme=source.request.scheme,
            host=source.request.host,
            path=source.request.path,
            query=source.request.query,
            headers=tuple(new_headers),
            body=new_body,
        ),
        wire_format=source.wire_format,
        known_non_secrets=source.known_non_secrets,
        triggers_met=frozenset(),
        triggers_absent=frozenset(),
    )


def main() -> None:
    """Load the source entry, build the twin, write it via ``write_entry``."""
    source_path = CORPUS / f"{SOURCE_ID}.json"
    if not source_path.is_file():
        raise SystemExit(
            f"{SOURCE_ID!r} is not committed under tests/corpus/ — cannot "
            "construct the streaming twin without the source."
        )

    source = k.load_entry(source_path)
    assert source.id == SOURCE_ID, (
        f"the m6 source entry id drifted; got {source.id!r}, expected {SOURCE_ID!r}"
    )

    twin = _flip_to_streaming(source)
    manifest_path = k.write_entry(CORPUS, twin)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
