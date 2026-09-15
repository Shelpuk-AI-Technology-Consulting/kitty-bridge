"""Builders for the T-C3 threshold corpus entries.

Plan task **T-C3** (KBR-46) ships four entries:

* ``tool_result_under_limit`` — one ``tool_result`` string of exactly
  ``_TOOL_RESULT_TRUNCATION_LIMIT`` (= 50 000) chars; the largest size that
  does NOT trigger M3.
* ``tool_result_over_limit`` — one ``tool_result`` string of
  ``_TOOL_RESULT_TRUNCATION_LIMIT + 1`` chars; the smallest size that DOES.
* ``compaction_budget_under`` — a transcript whose CC-converted messages
  serialise to exactly ``_COMPACTION_CHAR_THRESHOLD`` (= 2 800 000); the
  largest short-circuit size on the static fallback budget. M4 complement
  (no oversized tool result).
* ``compaction_budget_over`` — a transcript whose CC-converted messages
  serialise to ``_COMPACTION_CHAR_THRESHOLD + 1`` and that carries one
  oversized ``tool_result`` so M4's second condition is also present —
  and the filler ALONE crosses the threshold so M5 still fires after
  M3's truncation.

The bridge's M5 comparison (`_safe_size` inside `_compact_messages`) measures the
**CC-converted** messages length, not the Anthropic-Messages shape the
fixture commits. The two shapes differ by a constant ~92 chars for this
layout, and on the ``use_native_messages=True`` passthrough path the
Anthropic shape is preserved. The builder sizes filler against the CC shape
directly via ``MessagesTranslator.translate_request`` and iterates to the
target, so the fixture lands on the boundary the bridge actually compares.

The builder imports the threshold constants from :mod:`kitty.bridge.server`,
so a constant change flows into the fixture's byte lengths and the L1
regeneration test in :mod:`tests.harness.test_corpus_thresholds` reports the
drift.

``scripts/regenerate_corpus_thresholds.py`` materialises the four entries
into ``tests/corpus/`` through :func:`harness.corpus.write_entry`; the
regeneration test then pins the committed artifacts to the builder's output,
which is the "second reader" that replaces human review for the megabyte
padded body.

``.system_design/TEST_SUITE.md`` §7.1 · plan task **T-C3** (KBR-46).
"""

from __future__ import annotations

import json
from typing import Final

from harness.contract import CapturedRequest
from harness.register import Trigger

# Imported deliberately, not copied: a constant change in the source of truth
# must surface here and in the regeneration test, not as a silently stale
# fixture. The harness is free to import from ``kitty.*`` (the import-linter
# constrains ``kitty.*`` source modules, not ``tests/harness`` consumers).
from kitty.bridge.messages.translator import MessagesTranslator
from kitty.bridge.server import _COMPACTION_CHAR_THRESHOLD, _TOOL_RESULT_TRUNCATION_LIMIT

#: The MessagesTranslator instance the builder measures CC-shape lengths with.
#: Construction is cheap (no I/O) and stateless, so a single module-level
#: instance is correct.
_TRANSLATOR: Final = MessagesTranslator()

# ---------------------------------------------------------------------------
# Constants shared by every entry
# ---------------------------------------------------------------------------

#: The model every entry names. Matches :file:`tests/corpus/format_example.body`.
_MODEL: Final = "claude-sonnet-4-20250514"

#: The system prompt every entry carries — short, deterministic, reviewable.
_SYSTEM_PROMPT: Final = "You are Claude Code, Anthropic's official CLI for Claude."

#: The tool_use id and name shared by both M3-pair rounds and the
#: oversized tool_result embedded in the over-budget entry.
_TOOL_USE_ID: Final = "toolu_01"
_TOOL_NAME: Final = "Read"

#: The Anthropic-Messages headers every entry carries.
_HEADERS: Final[tuple[tuple[str, str], ...]] = (
    ("Host", "127.0.0.1"),
    ("Accept", "application/json"),
    ("Content-Type", "application/json"),
    ("X-Api-Key", "<redacted:credential_header>"),
    ("anthropic-version", "2023-06-01"),
    ("anthropic-beta", "claude-code-20250219"),
)

#: The number of filler user/assistant exchanges per budget entry.
#: Sized so each filler turn lands near ~7 KB of text — short enough that the
#: generator stays well under the threshold on the under side and comfortably
#: over it on the over side, given the rest of the skeleton.
_N_FILLER_EXCHANGES: Final = 200

#: A lorem-ipsum fragment used as filler body within each filler turn.
_LOREM: Final = "lorem ipsum dolor sit amet, consectetur adipiscing elit, "


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

#: The manifest metadata for each entry. This is the single source of truth —
#: the regen script and the regeneration test both read from here, which is
#: what makes the committed manifest's bytes a deterministic function of the
#: builder alone (the reviewer's finding: a manifest field drifting while the
#: body is unchanged breaks the "idempotent regen" claim just as much as a
#: body drift does).
_ENTRY_METADATA: Final[dict[str, tuple[str, str]]] = {
    "tool_result_under_limit": (
        "One tool_result of exactly 50000 chars — the largest non-triggering size "
        "against the bridge's strict `>` comparison. M3 complement.",
        "Synthetic, not captured: a captured tool result cannot be aimed at exactly the "
        "50000-char boundary and survive scrubbing at that size, since scrubbing shortens "
        "bodies. Calibrated against _TOOL_RESULT_TRUNCATION_LIMIT in "
        "src/kitty/bridge/server.py.",
    ),
    "tool_result_over_limit": (
        "One tool_result of 50001 chars — the smallest size the bridge's strict `>` "
        "comparison accepts. M3 trigger case.",
        "Synthetic, not captured: a capture cannot be aimed at 50001 chars and survive "
        "scrubbing at that size. Calibrated against _TOOL_RESULT_TRUNCATION_LIMIT in "
        "src/kitty/bridge/server.py.",
    ),
    "compaction_budget_under": (
        "Transcript whose CC-converted messages serialise to exactly "
        "_COMPACTION_CHAR_THRESHOLD (2800000) — the largest short-circuit size of "
        "_compact_messages on the static-fallback budget (when it is called with "
        "max_messages_chars=None in _compact_messages). M4 complement (no oversized "
        "tool_result). M5 status is profile-dependent; see tests/corpus/README.md "
        "§Threshold-pair entries.",
        "Synthetic, not captured: a capture cannot be aimed at exactly 2800000 chars of "
        "serialized CC-converted messages and survive scrubbing at that size. Padded "
        "construction: skeleton turns plus deterministic filler, generated by "
        "tests/harness/corpus_thresholds.py; human review replaced by the L1 regeneration "
        "test in tests/harness/test_corpus_thresholds.py. Calibrated against the "
        "CC-converted messages length the bridge measures with `_safe_size` "
        "(inside `_compact_messages`), not the Anthropic-Messages shape the fixture commits — the "
        "two shapes differ by a constant ~92 chars for this layout. The static "
        "_COMPACTION_CHAR_THRESHOLD is the threshold the bridge compares against only "
        "when _compact_messages is called without a profile-derived budget; on profiles "
        "with a larger derived budget this body also short-circuits, on profiles with a "
        "smaller budget this body DOES trigger M5, so oracle slices must resolve the "
        "profile before treating this as an M5 complement.",
    ),
    "compaction_budget_over": (
        "Transcript whose CC-converted messages — with the filler ALONE, before "
        "counting the embedded 50001-char oversized tool_result — serialise to exactly "
        "_COMPACTION_CHAR_THRESHOLD + 1 (2800001). Pre-M3 the total is ~2850122; "
        "post-M3-truncation (the oversized tool_result shrinks to a ~50-char notice) "
        "the body is still over the threshold, so M5's pruning step fires. Also the "
        "M4 request half (oversized tool result present). M5 status is "
        "profile-dependent; see tests/corpus/README.md §Threshold-pair entries.",
        "Synthetic, not captured: a capture cannot be aimed at exactly 2800001 chars of "
        "serialized CC-converted filler and survive scrubbing at that size. Padded "
        "construction generated by tests/harness/corpus_thresholds.py; human review "
        "replaced by the L1 regeneration test. Calibrated against the CC-converted "
        "messages length the bridge measures with `_safe_size` (inside `_compact_messages`), not "
        "the Anthropic-Messages shape the fixture commits. Sized so the filler ALONE "
        "(excluding the oversized tool_result) crosses the threshold, because an "
        "earlier build sized the total at threshold+1 and M3's truncation dropped the "
        "body back below — M5 short-circuited instead of firing. The oversized "
        "tool_result satisfies M4's request half; the compaction-engaged half is "
        "pipeline state and is supplied by an oracle slice at the call site that "
        "resolves the profile. On profiles whose derived budget exceeds this body's "
        "total (e.g. 1M-token models at ~3.99M), neither M5 nor M4 fires.",
    ),
}


def entry_metadata(entry_id: str) -> tuple[str, str]:
    """Return ``(description, origin_note)`` for ``entry_id``.

    Args:
        entry_id: One of the four ids this module ships.

    Returns:
        The manifest metadata pair.

    Raises:
        KeyError: When ``entry_id`` is not one of the four.
    """
    return _ENTRY_METADATA[entry_id]


def build_tool_result_under_limit() -> tuple[CapturedRequest, frozenset[Trigger], frozenset[Trigger]]:
    """Build the M3 complement: a request whose tool_result string is exactly the limit.

    Returns:
        ``(captured_request, triggers_met, triggers_absent)``.
    """
    messages = [
        _text_user("Read the README so I can ask follow-ups."),
        _text_assistant("Reading the file now."),
        _tool_result_message(_TOOL_RESULT_TRUNCATION_LIMIT),
    ]
    return _finalise(messages, met=frozenset(), absent=frozenset({Trigger.TOOL_RESULT_OVER_LIMIT}))


def build_tool_result_over_limit() -> tuple[CapturedRequest, frozenset[Trigger], frozenset[Trigger]]:
    """Build the M3 trigger case: a request whose tool_result string is ``limit + 1``.

    Returns:
        ``(captured_request, triggers_met, triggers_absent)``.
    """
    messages = [
        _text_user("Read the README so I can ask follow-ups."),
        _text_assistant("Reading the file now."),
        _tool_result_message(_TOOL_RESULT_TRUNCATION_LIMIT + 1),
    ]
    return _finalise(messages, met=frozenset({Trigger.TOOL_RESULT_OVER_LIMIT}), absent=frozenset())


def build_compaction_budget_under() -> tuple[CapturedRequest, frozenset[Trigger], frozenset[Trigger]]:
    """Build the M5 short-circuit case: CC-shape messages serialise to exactly the threshold.

    The bridge measures ``len(json.dumps(messages, ensure_ascii=False))`` on the
    **CC-converted** messages (``_safe_size``, inside ``_compact_messages``); this builder
    sizes the filler so the CC-converted length — not the Anthropic-shape
    length, which differs by a constant ~92 chars for this layout — is exactly
    ``_COMPACTION_CHAR_THRESHOLD``.

    The claim is about the **static fallback** budget only (when
    ``_compact_messages`` is called with ``max_messages_chars=None``,
    the static fallback branch). On profiles whose derived ``messages_budget`` is
    larger than the threshold (e.g. 1 M-token models give ~3.99 M), this body
    also short-circuits. On profiles with a smaller derived budget (e.g. the
    default 200 K-token model gives ~790 K), the body would trigger M5 — the
    README records that oracle slices must resolve the profile before
    treating this entry as an M5 complement.
    """
    messages = _padded_messages(target_cc_chars=_COMPACTION_CHAR_THRESHOLD)
    return _finalise(messages, met=frozenset(), absent=frozenset({Trigger.TOOL_RESULT_OVER_LIMIT}))


def build_compaction_budget_over() -> tuple[CapturedRequest, frozenset[Trigger], frozenset[Trigger]]:
    """Build the M5 trigger case whose filler alone crosses the threshold, plus M4's request half.

    The filler is sized so the CC-converted length **without** the embedded
    ``tool_use`` / ``tool_result`` pair is exactly ``_COMPACTION_CHAR_THRESHOLD + 1``
    — the smallest value the bridge's ``original_size > threshold`` accepts. The
    50 001-char oversized ``tool_result`` rides on top, so:

    * **pre-M3**, the CC-converted length is roughly ``threshold + 1 + 50 000``,
      comfortably over budget;
    * **post-M3**, the tool_result is truncated to a ~50-char notice, leaving
      the filler at ``threshold + 1`` — **still over budget**, so M5's pruning
      step runs after M3's truncation. (This is the failure mode the first
      implementation had: with the filler sized at ``threshold + 1`` counting
      the tool_result, M3's truncation dropped the body back under the
      threshold and M5 short-circuited instead of firing.)

    M4's second condition (oversized tool result present) is also met, and
    M5's trigger fires on the same body. On profiles whose derived budget is
    **larger** than this body's total (e.g. 1 M-token models at ~3.99 M),
    neither M5 nor M4 fires — the fixture is calibrated against the static
    fallback threshold, not against every profile.

    The ``tool_use`` block carries a ``/tmp/…`` path **deliberately**: a
    ``/home/<user>/`` path would match the scrubber's ``home_path`` rule, and
    ``write_entry`` would rewrite the committed bytes, breaking the
    regeneration test's byte-identity assertion.
    """
    # Filler alone at CC-shape threshold + 1: this is what guarantees the
    # post-M3-truncation length still exceeds the threshold, because M3
    # removes only the tool_result's contribution. The tool_use/tool_result
    # pair is inserted around tool_name = `_TOOL_NAME` to keep the
    # tool_result paired — an orphan would be silently dropped by the bridge's
    # pairing validation (register row M7) on an entry that does not declare
    # M7 in its manifest.
    messages = _padded_messages(target_cc_chars=_COMPACTION_CHAR_THRESHOLD + 1)
    tool_use_block = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": _TOOL_USE_ID,
                "name": _TOOL_NAME,
                "input": {"file_path": "/tmp/kitty-bridge-threshold-pair.md"},
            }
        ],
    }
    messages.insert(len(messages) - 1, tool_use_block)
    messages.insert(len(messages) - 1, _tool_result_message(_TOOL_RESULT_TRUNCATION_LIMIT + 1))
    return _finalise(messages, met=frozenset({Trigger.TOOL_RESULT_OVER_LIMIT}), absent=frozenset())


# ---------------------------------------------------------------------------
# Body byte accessors used by the regeneration test
# ---------------------------------------------------------------------------


_BUILDERS_BY_ID = {
    "tool_result_under_limit": build_tool_result_under_limit,
    "tool_result_over_limit": build_tool_result_over_limit,
    "compaction_budget_under": build_compaction_budget_under,
    "compaction_budget_over": build_compaction_budget_over,
}


def build_body_bytes(entry_id: str) -> bytes:
    """Return the body bytes ``write_entry`` would land for ``entry_id``.

    Args:
        entry_id: One of the four ids :data:`_BUILDERS_BY_ID` knows.

    Returns:
        The raw ``CapturedRequest.body`` bytes — what the committed
        ``<entry_id>.body`` file holds.

    Raises:
        KeyError: When ``entry_id`` is not one of the four this module ships.
    """
    return _BUILDERS_BY_ID[entry_id]()[0].body


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _finalise(
    messages: list[dict],
    *,
    met: frozenset[Trigger],
    absent: frozenset[Trigger],
) -> tuple[CapturedRequest, frozenset[Trigger], frozenset[Trigger]]:
    """Wrap ``messages`` in a full request body, serialise, and return the captured request.

    Args:
        messages: The Anthropic-Messages messages list; serialised verbatim as
            the body's ``messages`` field.
        met: The register triggers this request establishes.
        absent: The register triggers this request provably does not establish.

    Returns:
        ``(captured_request, triggers_met, triggers_absent)``.
    """
    body_obj = {
        "model": _MODEL,
        "max_tokens": 4096,
        "stream": True,
        "system": [{"type": "text", "text": _SYSTEM_PROMPT}],
        "messages": messages,
    }
    body_bytes = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
    return (
        CapturedRequest(
            method="POST",
            scheme="http",
            host="127.0.0.1",
            path="/v1/messages",
            query="",
            headers=_HEADERS,
            body=body_bytes,
        ),
        met,
        absent,
    )


def _text_user(text: str) -> dict:
    """Return an Anthropic-Messages user turn with a plain string content.

    Args:
        text: The user turn's content.

    Returns:
        The Anthropic-Messages user-turn dict.
    """
    return {"role": "user", "content": text}


def _text_assistant(text: str) -> dict:
    """Return an Anthropic-Messages assistant turn with a plain string content.

    Args:
        text: The assistant turn's content.

    Returns:
        The Anthropic-Messages assistant-turn dict.
    """
    return {"role": "assistant", "content": text}


def _tool_result_message(content_chars: int) -> dict:
    """Return an Anthropic-Messages user turn carrying a ``tool_result`` of ``content_chars`` chars.

    Uses string content (not a list of blocks) so the bridge's CC conversion
    passes the length through unchanged — the bridge's ``>`` comparison sees
    exactly the length this function declares. The CC conversion maps this
    Anthropic ``user`` message with a ``tool_result`` block to a CC
    ``role=="tool"`` message with the same string content; for the M3 threshold
    test the conversion is the bridge's, and for the M4 second condition the
    conversion is what makes the embedded turn discoverable by
    ``_compact_messages`` step 1.

    Args:
        content_chars: The exact character length of the ``tool_result`` content.

    Returns:
        The Anthropic-Messages user-turn dict carrying one tool_result block.
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": _TOOL_USE_ID,
                "content": "x" * content_chars,
            }
        ],
    }


def _padded_messages(
    *,
    target_cc_chars: int,
) -> list[dict]:
    """Return messages whose **CC-converted** ``json.dumps`` length equals ``target_cc_chars``.

    The bridge measures ``len(json.dumps(messages, ensure_ascii=False))`` on the
    CC-converted messages (``_safe_size``, inside ``_compact_messages``), not the
    Anthropic-Messages shape the fixture commits. The Anthropic and CC shapes
    differ by a constant offset (~92 chars for the no-tool-result layout)
    because the CC shape collapses ``[{"type":"text","text":T}]`` to a plain
    string and rewraps ``tool_result`` blocks. The builder measures the CC
    shape on every iteration so the fixture lands on the boundary the bridge
    actually compares against.

    The skeleton is an initial user turn, ``_N_FILLER_EXCHANGES`` user/assistant
    exchanges with deterministic filler text, and a closing user turn. The
    filler text embeds the turn index so a reviewer sampling any region sees
    construction, and the last filler turn absorbs the rounding error so the
    final CC-length lands on ``target_cc_chars`` exactly.

    Args:
        target_cc_chars: The exact ``len(json.dumps(cc_messages, ensure_ascii=False))``
            the returned messages must hit — what ``_compact_messages`` measures.

    Returns:
        The messages list. ``_cc_messages_serialized(returned) == target_cc_chars``.
    """

    initial_user = _text_user("Read the README so I can ask follow-ups.")
    closing_user = _text_user("Thanks, that's what I needed.")
    non_filler: list[dict] = [initial_user, closing_user]

    n_filler_turns = _N_FILLER_EXCHANGES * 2  # one user + one assistant per exchange
    total_turns = n_filler_turns + len(non_filler)

    # Build the messages with empty filler first, measure the CC-converted
    # skeleton length, then size the filler so the final CC-converted length
    # hits target_cc_chars. Iterate up to a small bound so the loop converges
    # even if the Filler-text shape interaction with the CC conversion ever
    # changes by a constant.
    messages: list[dict] = [initial_user]
    for i in range(n_filler_turns):
        messages.append(_text_user("") if i % 2 == 0 else _text_assistant(""))
    messages.append(closing_user)
    assert len(messages) == total_turns

    skeleton_cc_len = _cc_messages_serialized(_build_body(messages))
    filler_total = target_cc_chars - skeleton_cc_len
    if filler_total < 0:
        raise ValueError(
            f"target_cc_chars {target_cc_chars} is below the skeleton length "
            f"{skeleton_cc_len}; the threshold pair can only grow, not shrink"
        )

    per_turn = filler_total // n_filler_turns
    remainder = filler_total - per_turn * n_filler_turns
    for i in range(n_filler_turns):
        text_len = per_turn + (remainder if i == n_filler_turns - 1 else 0)
        text = _filler_text(turn_index=i + 1, total=total_turns, length=text_len)
        messages[i + 1] = _text_user(text) if i % 2 == 0 else _text_assistant(text)

    # Iterate to convergence: the filler-to-CC-length ratio is exactly 1 char
    # of filler → 1 char of CC length, but if a future change to the
    # translator broke that invariant we would land off-by-one. A second
    # round of measurement + adjustment handles the edge.
    for _attempt in range(3):
        actual_cc = _cc_messages_serialized(_build_body(messages))
        if actual_cc == target_cc_chars:
            return messages
        # Adjust the LAST filler turn's text length by the delta. The filler
        # text is plain ASCII, so 1 char of length delta == 1 char of CC
        # length delta.
        delta = target_cc_chars - actual_cc
        last_filler_idx = n_filler_turns  # the last filler turn sits here
        text = messages[last_filler_idx]["content"]
        if delta > 0:
            messages[last_filler_idx]["content"] = text + "x" * delta
        else:
            messages[last_filler_idx]["content"] = text[:delta]

    actual_cc = _cc_messages_serialized(_build_body(messages))
    raise AssertionError(
        f"CC-converted messages length {actual_cc} did not converge to target "
        f"{target_cc_chars} within 3 adjustment rounds; the bridge measurement "
        "property has changed in a way the builder does not account for."
    )


def _filler_text(*, turn_index: int, total: int, length: int) -> str:
    """Return a string of exactly ``length`` characters naming ``turn_index``.

    Format: ``"Turn NNNN of TOTAL: <lorem ipsum repeated>"``, padded or
    trimmed to exactly ``length`` chars. The embedded turn index is what
    makes the line transparent to a reviewer and what makes the L1
    regeneration test catch a hand-edited line that doesn't match the index.

    Args:
        turn_index: The 1-based turn number, embedded in the prefix so every
            line is unique and the index is visible at a glance.
        total: The total number of turns, embedded in the prefix.
        length: The exact length the returned string must have.

    Returns:
        A string of exactly ``length`` characters.
    """
    prefix = f"Turn {turn_index:04d} of {total:04d}: "
    body = _LOREM * max(0, (length - len(prefix)) // len(_LOREM))
    text = prefix + body
    if len(text) < length:
        return text + "x" * (length - len(text))
    return text[:length]


def _cc_messages_serialized(body_bytes: bytes) -> int:
    """Return ``len(json.dumps(messages, ensure_ascii=False))`` on the CC-converted shape.

    This is the property ``_safe_size`` inside ``_compact_messages`` measures
    when deciding whether the budget is exceeded. Pinning a fixture against
    this measurement — rather than the Anthropic-Messages shape the fixture
    commits — is what makes the boundary honest: the bridge compares CC shape
    with CC shape.

    Args:
        body_bytes: The raw Anthropic-Messages request body bytes.

    Returns:
        The length the bridge's ``_safe_size`` would observe.
    """
    body = json.loads(body_bytes.decode("utf-8"))
    cc_request = _TRANSLATOR.translate_request(body)
    return len(json.dumps(cc_request["messages"], ensure_ascii=False))


def _build_body(messages: list[dict]) -> bytes:
    """Return the raw Anthropic-Messages body bytes for ``messages`` alone.

    Wraps ``messages`` in the same envelope ``_finalise`` uses (model,
    system, stream, max_tokens) without the headers, so the CC-length
    measurement sees exactly the shape the bridge would translate.

    Args:
        messages: The Anthropic-Messages messages list.

    Returns:
        The UTF-8-encoded JSON body bytes.
    """
    return json.dumps(
        {
            "model": _MODEL,
            "max_tokens": 4096,
            "stream": True,
            "system": [{"type": "text", "text": _SYSTEM_PROMPT}],
            "messages": messages,
        },
        ensure_ascii=False,
    ).encode("utf-8")
