"""Builders for the T-C3 threshold corpus entries.

Plan task **T-C3** (KBR-46) ships four entries:

* ``tool_result_under_limit`` — one ``tool_result`` string of exactly the limit;
  the largest size that does NOT trigger M3.
* ``tool_result_over_limit`` — one ``tool_result`` string of ``limit + 1``;
  the smallest size that DOES.
* ``compaction_budget_under`` — a transcript whose Anthropic-shape messages
  serialise strictly below ``_COMPACTION_CHAR_THRESHOLD``; M5 complement.
* ``compaction_budget_over`` — a transcript whose Anthropic-shape messages
  serialise strictly above ``_COMPACTION_CHAR_THRESHOLD`` and that carries one
  oversized tool_result so M4's second condition is also present.

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
from kitty.bridge.server import _COMPACTION_CHAR_THRESHOLD, _TOOL_RESULT_TRUNCATION_LIMIT

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
        "Transcript whose Anthropic-shape messages serialise to exactly "
        "_COMPACTION_CHAR_THRESHOLD (2800000) — the largest short-circuit size of "
        "_compact_messages on the static-fallback budget. M4 complement (no oversized "
        "tool_result). M5 status is profile-dependent; see tests/corpus/README.md "
        "§Threshold-pair entries.",
        "Synthetic, not captured: a capture cannot be aimed at exactly 2800000 chars of "
        "serialized messages and survive scrubbing at that size. Padded construction: "
        "skeleton turns plus deterministic filler, generated by "
        "tests/harness/corpus_thresholds.py; human review replaced by the L1 regeneration "
        "test in tests/harness/test_corpus_thresholds.py. Calibrated against "
        "_COMPACTION_CHAR_THRESHOLD in src/kitty/bridge/server.py, which is the "
        "conservative upper bound of any profile's derived messages_budget but NOT the "
        "runtime trigger itself — on profiles with a smaller budget this body DOES "
        "trigger M5, so oracle slices must resolve the profile before treating this as "
        "an M5 complement.",
    ),
    "compaction_budget_over": (
        "Transcript whose Anthropic-shape messages serialise to exactly "
        "_COMPACTION_CHAR_THRESHOLD + 1 (2800001) — one past the short-circuit side. "
        "M5 trigger case for every profile (the body exceeds the worst-case budget "
        "ceiling). Also the M4 request half: it carries one oversized tool_result "
        "(50001 chars); the compaction-engaged half is supplied by an oracle slice at "
        "call time.",
        "Synthetic, not captured: a capture cannot be aimed at exactly 2800001 chars of "
        "serialized messages and survive scrubbing at that size. Padded construction "
        "generated by tests/harness/corpus_thresholds.py; human review replaced by the "
        "L1 regeneration test. Carries one oversized tool_result so M4's request half "
        "(oversized tool result present) is satisfied; the compaction-engaged half is "
        "pipeline state and is supplied by an oracle slice at the call site that "
        "resolves the profile, since the manifest cannot know whether the bridge will "
        "compact. Calibrated against _COMPACTION_CHAR_THRESHOLD in "
        "src/kitty/bridge/server.py; the body is past the worst-case budget ceiling so "
        "it triggers M5 on every profile.",
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
    """Build the M5 short-circuit case: messages serialise to exactly the threshold.

    The static ``_COMPACTION_CHAR_THRESHOLD`` is the conservative upper bound of
    any profile's derived ``messages_budget`` (which is always ``<= 4 000 000``).
    At exactly the threshold, ``_compact_messages`` short-circuits on every
    profile whose budget is ≥ 2 800 000. On profiles with smaller derived
    budgets (e.g. a 200 K-token default model gives a ~790 K budget), this
    body WOULD trigger M5 — the README records that oracle slices must check
    the resolved profile, not just the committed bytes.
    """
    messages = _padded_messages(
        target_chars=_COMPACTION_CHAR_THRESHOLD,
        oversized_tool_result_chars=None,
    )
    return _finalise(messages, met=frozenset(), absent=frozenset({Trigger.TOOL_RESULT_OVER_LIMIT}))


def build_compaction_budget_over() -> tuple[CapturedRequest, frozenset[Trigger], frozenset[Trigger]]:
    """Build the M5 trigger case at the smallest triggering size, carrying M4's request half.

    ``threshold + 1`` is the smallest value ``original_size > threshold`` accepts,
    and any profile's ``messages_budget`` is at most 4 M so the body crosses every
    profile's budget. The embedded oversized tool_result is the request-property
    half of M4 — the compaction-engaged half is supplied by an oracle slice at
    call time, since the manifest cannot know whether the bridge will compact.
    """
    messages = _padded_messages(
        target_chars=_COMPACTION_CHAR_THRESHOLD + 1,
        oversized_tool_result_chars=_TOOL_RESULT_TRUNCATION_LIMIT + 1,
    )
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
    """Wrap ``messages`` in a full request body, serialise, and return the captured request."""
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
    """Return an Anthropic-Messages user turn with a plain string content."""
    return {"role": "user", "content": text}


def _text_assistant(text: str) -> dict:
    """Return an Anthropic-Messages assistant turn with a plain string content."""
    return {"role": "assistant", "content": text}


def _tool_result_message(content_chars: int) -> dict:
    """Return an Anthropic-Messages user turn carrying a ``tool_result`` of ``content_chars`` chars.

    Uses string content (not a list of blocks) so the bridge's CC conversion
    passes the length through unchanged — the bridge's ``>`` comparison sees
    exactly the length this function declares.
    """
    # The CC conversion maps Anthropic `user` message with a `tool_result`
    # block to a CC `role=="tool"` message with the same string content.
    # For the M3 threshold test the conversion is the bridge's; for the M4
    # second condition the conversion is what makes the embedded turn
    # discoverable by ``_compact_messages`` step 1.
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
    target_chars: int,
    oversized_tool_result_chars: int | None,
) -> list[dict]:
    """Return Anthropic-Messages messages whose ``json.dumps`` length equals ``target_chars``.

    The skeleton is an initial user turn, ``_N_FILLER_EXCHANGES`` user/assistant
    exchanges with deterministic filler text, optionally a tool_use/tool_result
    pair (when ``oversized_tool_result_chars`` is given), and a closing user
    turn. The filler text embeds the turn index so a reviewer sampling any
    region sees construction, and the last filler turn absorbs the rounding
    error so the final length lands on ``target_chars`` exactly.

    Args:
        target_chars: The exact ``len(json.dumps(messages, ensure_ascii=False))``
            the returned messages must hit.
        oversized_tool_result_chars: When given, embed one ``tool_result`` of
            this many chars in the conversation, immediately after a
            corresponding tool_use.

    Returns:
        The messages list. ``len(json.dumps(returned, ensure_ascii=False)) == target_chars``.
    """
    initial_user = _text_user("Read the README so I can ask follow-ups.")
    closing_user = _text_user("Thanks, that's what I needed.")
    non_filler: list[dict] = [initial_user]
    if oversized_tool_result_chars is not None:
        non_filler.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": _TOOL_USE_ID,
                        "name": _TOOL_NAME,
                        # Deliberately avoids the scrubber's path patterns:
                        # ``/home/<user>`` triggers ``home_path`` and ``write_entry``
                        # would rewrite the committed bytes, breaking the
                        # regeneration test's byte-identity assertion. ``/tmp/``
                        # does not match any pattern.
                        "input": {"file_path": "/tmp/kitty-bridge-threshold-pair.md"},
                    }
                ],
            }
        )
        non_filler.append(_tool_result_message(oversized_tool_result_chars))
    non_filler.append(closing_user)

    n_filler_turns = _N_FILLER_EXCHANGES * 2  # one user + one assistant per exchange
    total_turns = n_filler_turns + len(non_filler)

    # Step 1: measure the skeleton with empty filler text. The roles MUST
    # match the final messages' role pattern (user/assistant/user/assistant…)
    # — an assistant wrapper is 5 chars longer than a user wrapper, so an
    # all-user skeleton would underestimate the wrapper overhead by 5 chars
    # per assistant turn and the final length would land at +5 × N_assistants
    # above the target. The arithmetic below assumes the wrappers are
    # accounted for here.
    skeleton: list[dict] = list(non_filler)
    for i in range(n_filler_turns):
        placeholder = _text_user("") if i % 2 == 0 else _text_assistant("")
        skeleton.insert(len(skeleton) - 1, placeholder)
    assert len(skeleton) == total_turns
    placeholder_len = len(json.dumps(skeleton, ensure_ascii=False))

    # Step 2: distribute the remainder across filler turns; the last one absorbs rounding.
    filler_total = target_chars - placeholder_len
    per_turn = filler_total // n_filler_turns
    remainder = filler_total - per_turn * n_filler_turns

    # Step 3: build the messages with real filler.
    messages: list[dict] = [initial_user]
    for i in range(n_filler_turns):
        text_len = per_turn + (remainder if i == n_filler_turns - 1 else 0)
        text = _filler_text(turn_index=i + 1, total=total_turns, length=text_len)
        messages.append(_text_user(text) if i % 2 == 0 else _text_assistant(text))
    messages.extend(non_filler[1:])  # everything after initial_user

    actual = len(json.dumps(messages, ensure_ascii=False))
    # Defence in depth: the arithmetic above is exact because the filler text
    # is constructed deterministically; if a future change breaks it, fail
    # loudly here rather than producing a fixture with a misleading length.
    if actual != target_chars:
        raise AssertionError(
            f"padded messages length {actual} != target {target_chars}; the padding "
            "arithmetic has drifted from the assumption that the placeholder "
            "skeleton and the final skeleton differ by exactly the filler text."
        )
    return messages


def _filler_text(*, turn_index: int, total: int, length: int) -> str:
    """Return a string of exactly ``length`` characters naming ``turn_index``.

    Format: ``"Turn NNNN of TOTAL: <lorem ipsum repeated>"``, padded or
    trimmed to exactly ``length`` chars. The embedded turn index is what
    makes the line transparent to a reviewer and what makes the L1
    regeneration test catch a hand-edited line that doesn't match the index.
    """
    prefix = f"Turn {turn_index:04d} of {total:04d}: "
    body = _LOREM * max(0, (length - len(prefix)) // len(_LOREM))
    text = prefix + body
    if len(text) < length:
        return text + "x" * (length - len(text))
    return text[:length]
