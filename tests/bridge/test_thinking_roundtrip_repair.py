"""Tests for the thinking round-trip repair on cross-backend transcript replay.

A backend that enforces a thinking-mode round-trip contract (DeepSeek, Kimi)
rejects a transcript whose assistant turns were authored by a provider that
emits no thinking blocks.  The bridge must recognise that rejection as a
request-shape problem of its own making, repair the transcript, and retry the
same backend without charging the failure to that backend's health.

Covers ``.requirements/20260828T153428Z_thinking_roundtrip_failover_repair``.
"""

from __future__ import annotations

import copy

from kitty.bridge.server import (
    _is_thinking_roundtrip_error,
    _repair_thinking_roundtrip,
)

# ── Real upstream error bodies ─────────────────────────────────────────────
# Verbatim from the incident report (kitty-bridge#32) and from DeepSeek's and
# Kimi's documented wording for the same contract.

DEEPSEEK_ANTHROPIC_400 = (
    '{"error":{"message":"The `content[].thinking` in the thinking mode must be '
    'passed back to the API.","type":"invalid_request_error","param":null,'
    '"code":"invalid_request_error"}}'
)

DEEPSEEK_CHAT_COMPLETIONS_400 = (
    '{"error":{"message":"The `reasoning_content` in the thinking mode must be '
    'passed back to the API.","type":"invalid_request_error"}}'
)

KIMI_400 = (
    '{"error":{"message":"thinking is enabled but reasoning_content is missing '
    'in assistant tool call message at index 63","type":"invalid_request_error"}}'
)


def _native_transcript() -> dict:
    """Anthropic Messages body whose assistant turns carry no thinking block."""
    return {
        "model": "deepseek-v4-flash",
        "stream": True,
        "system": [{"type": "text", "text": "You are helpful."}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Please run ls"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Running it now."},
                    {"type": "tool_use", "id": "call_1", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "a.txt"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "There is one file."}]},
        ],
        "tools": [{"name": "Bash", "description": "Run a command", "input_schema": {"type": "object"}}],
    }


def _cc_transcript() -> dict:
    """Chat Completions body whose assistant turns carry no reasoning_content."""
    return {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Please run ls"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": '{"command": "ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"},
            {"role": "assistant", "content": "There is one file."},
        ],
    }


# ── FR-1: recognising the rejection ────────────────────────────────────────


class TestIsThinkingRoundtripError:
    """AC-1.1 … AC-1.4 — the predicate that classifies the upstream 400."""

    def test_deepseek_anthropic_wording_matches(self):
        """AC-1.1 — the exact body from the incident report."""
        assert _is_thinking_roundtrip_error(400, DEEPSEEK_ANTHROPIC_400) is True

    def test_deepseek_chat_completions_wording_matches(self):
        """AC-1.1 — same contract, OpenAI-shaped field name."""
        assert _is_thinking_roundtrip_error(400, DEEPSEEK_CHAT_COMPLETIONS_400) is True

    def test_kimi_wording_matches(self):
        """AC-1.1 — Kimi states the same contract in different words."""
        assert _is_thinking_roundtrip_error(400, KIMI_400) is True

    def test_dict_body_matches(self):
        """AC-1.4 — the body may arrive already parsed."""
        body = {
            "error": {
                "message": "The `content[].thinking` in the thinking mode must be passed back to the API.",
            }
        }
        assert _is_thinking_roundtrip_error(400, body) is True

    def test_matching_is_case_insensitive(self):
        """AC-1.4 — providers are inconsistent about capitalisation."""
        assert _is_thinking_roundtrip_error(400, DEEPSEEK_ANTHROPIC_400.upper()) is True

    def test_500_does_not_match(self):
        """AC-1.2 — a 5xx is a backend fault and is already retried elsewhere."""
        assert _is_thinking_roundtrip_error(500, DEEPSEEK_ANTHROPIC_400) is False

    def test_502_does_not_match(self):
        """AC-1.2 — the failure that starts the incident must not be reclassified."""
        assert _is_thinking_roundtrip_error(502, DEEPSEEK_ANTHROPIC_400) is False

    def test_tool_use_format_error_does_not_match(self):
        """AC-1.3 — the other request-shape 400 keeps its own repair path."""
        body = '{"error":{"message":"invalid params, tool result\'s tool id(call_x) not found (2013)"}}'
        assert _is_thinking_roundtrip_error(400, body) is False

    def test_rate_limit_body_does_not_match(self):
        """AC-1.3 — no overlap with the quota classification."""
        assert _is_thinking_roundtrip_error(429, '{"error":{"message":"rate limit exceeded"}}') is False

    def test_context_too_large_does_not_match(self):
        """AC-1.3 — a genuinely oversized request must stay non-retryable."""
        body = '{"error":{"code":"1261","message":"Prompt exceeds max length"}}'
        assert _is_thinking_roundtrip_error(400, body) is False

    def test_unrelated_mention_of_thinking_does_not_match(self):
        """AC-1.3 — naming `thinking` is not enough; the contract phrase is required."""
        body = '{"error":{"message":"thinking blocks are not supported by this model"}}'
        assert _is_thinking_roundtrip_error(400, body) is False

    def test_none_body_does_not_match(self):
        """AC-1.4 — an absent body must not crash the classifier."""
        assert _is_thinking_roundtrip_error(400, None) is False

    def test_echoed_request_with_unrelated_missing_field_does_not_match(self):
        """AC-1.3 — the Kimi token pair must survive a gateway echoing the request.

        Some gateways include the offending request in the error body.  A
        transcript that legitimately carries ``reasoning_content`` would then
        put that word in the body, and "missing" is common in unrelated
        validation text — so the Kimi pattern requires the fuller phrase
        ``is missing in assistant``.  A false positive here would suppress the
        health penalty for a genuinely bad backend.
        """
        body = (
            '{"error":{"message":"missing required parameter: model"},'
            '"request":{"messages":[{"role":"assistant","reasoning_content":"real reasoning"}]}}'
        )
        assert _is_thinking_roundtrip_error(400, body) is False


# ── FR-2: repairing the transcript ─────────────────────────────────────────


class TestRepairThinkingRoundtripNative:
    """AC-2.1 … AC-2.6 on Anthropic-shaped bodies."""

    def test_assistant_content_list_gains_leading_thinking_block(self):
        """AC-2.1 — the carrier the target asks for is prepended."""
        body = _native_transcript()
        assert _repair_thinking_roundtrip(body, native=True) is True

        for msg in body["messages"]:
            if msg["role"] == "assistant":
                assert msg["content"][0] == {"type": "thinking", "thinking": ""}

    def test_existing_blocks_keep_order_and_values(self):
        """AC-2.1 — the repair adds, it never rewrites what is already there."""
        body = _native_transcript()
        original = copy.deepcopy(body["messages"][1]["content"])
        _repair_thinking_roundtrip(body, native=True)

        assert body["messages"][1]["content"][1:] == original

    def test_assistant_with_thinking_block_is_untouched(self):
        """AC-2.2 — no second carrier is added to a turn that already has one."""
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "already here", "signature": "sig"},
                        {"type": "text", "text": "Done"},
                    ],
                }
            ]
        }
        before = copy.deepcopy(body)
        assert _repair_thinking_roundtrip(body, native=True) is False
        assert body == before

    def test_assistant_string_content_becomes_thinking_plus_text(self):
        """AC-2.1 — a string-content assistant turn still needs the carrier."""
        body = {"messages": [{"role": "assistant", "content": "Hi there!"}]}
        assert _repair_thinking_roundtrip(body, native=True) is True

        assert body["messages"][0]["content"] == [
            {"type": "thinking", "thinking": ""},
            {"type": "text", "text": "Hi there!"},
        ]

    def test_non_assistant_messages_are_untouched(self):
        """AC-2.4 — the contract binds assistant turns only."""
        body = _native_transcript()
        before_users = copy.deepcopy([m for m in body["messages"] if m["role"] != "assistant"])
        _repair_thinking_roundtrip(body, native=True)

        assert [m for m in body["messages"] if m["role"] != "assistant"] == before_users

    def test_system_field_is_untouched(self):
        """AC-2.4 — the top-level Anthropic system block is not a message."""
        body = _native_transcript()
        _repair_thinking_roundtrip(body, native=True)

        assert body["system"] == [{"type": "text", "text": "You are helpful."}]

    def test_repair_is_idempotent(self):
        """AC-2.6 — a repaired body reports no further change."""
        body = _native_transcript()
        assert _repair_thinking_roundtrip(body, native=True) is True

        after_first = copy.deepcopy(body)
        assert _repair_thinking_roundtrip(body, native=True) is False
        assert body == after_first

    def test_original_messages_are_not_mutated(self):
        """AC-2.7 — the client's parsed request must survive the repair intact.

        The Messages handler forwards ``dict(body)``, a shallow copy sharing the
        client's ``messages`` list, and re-reads that list on later attempts.
        """
        body = _native_transcript()
        original_list = body["messages"]
        original_snapshot = copy.deepcopy(original_list)

        assert _repair_thinking_roundtrip(body, native=True) is True

        assert original_list == original_snapshot
        assert body["messages"] is not original_list

    def test_no_assistant_messages_reports_no_change(self):
        """AC-2.5 — nothing to repair means the caller must not retry."""
        body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
        assert _repair_thinking_roundtrip(body, native=True) is False

    def test_body_without_messages_reports_no_change(self):
        """AC-2.5 — a malformed body must not crash the repair."""
        assert _repair_thinking_roundtrip({"model": "x"}, native=True) is False


class TestRepairLeavesUnrepairableBodiesAlone:
    """Convergence guards — anything the carrier cannot attach to reports no change."""

    def test_assistant_without_usable_content_reports_no_change(self):
        """A message the carrier cannot attach to must not be reported as changed.

        The caller retries only on a change, so a message reported as changed
        on every pass would retry forever.  Falling through to the normal
        failover path is the correct outcome here.
        """
        body = {"messages": [{"role": "assistant", "content": None}]}
        assert _repair_thinking_roundtrip(body, native=True) is False
        assert body["messages"] == [{"role": "assistant", "content": None}]

    def test_body_without_messages_reports_no_change(self):
        """A Gemini-shaped body must pass through untouched.

        One bridge server can register all four protocol routes, so a backend
        in the sticky set may serialize a Gemini body, which has ``contents``
        rather than ``messages``.
        """
        body = {"contents": [{"role": "model", "parts": [{"text": "hi"}]}]}
        before = copy.deepcopy(body)
        assert _repair_thinking_roundtrip(body, native=True) is False
        assert body == before

    def test_partial_repair_still_reports_change(self):
        """One repairable turn beside an unrepairable one still earns a retry."""
        body = {
            "messages": [
                {"role": "assistant", "content": None},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ]
        }
        assert _repair_thinking_roundtrip(body, native=True) is True
        assert body["messages"][0] == {"role": "assistant", "content": None}
        assert body["messages"][1]["content"][0] == {"type": "thinking", "thinking": ""}


class TestRepairThinkingRoundtripChatCompletions:
    """AC-2.3 on Chat-Completions-shaped bodies."""

    def test_assistant_gains_empty_reasoning_content(self):
        """AC-2.3 — the OpenAI-shaped carrier is a sibling field, not a block."""
        body = _cc_transcript()
        assert _repair_thinking_roundtrip(body, native=False) is True

        for msg in body["messages"]:
            if msg["role"] == "assistant":
                assert msg["reasoning_content"] == ""

    def test_existing_reasoning_content_is_preserved(self):
        """AC-2.3 — a real chain of thought is never overwritten with an empty one."""
        body = {"messages": [{"role": "assistant", "content": "hi", "reasoning_content": "real reasoning"}]}
        assert _repair_thinking_roundtrip(body, native=False) is False
        assert body["messages"][0]["reasoning_content"] == "real reasoning"

    def test_tool_calls_are_preserved(self):
        """AC-2.3 — the repair must not disturb the tool-call round-trip."""
        body = _cc_transcript()
        before = copy.deepcopy(body["messages"][2]["tool_calls"])
        _repair_thinking_roundtrip(body, native=False)

        assert body["messages"][2]["tool_calls"] == before

    def test_tool_and_system_messages_are_untouched(self):
        """AC-2.4 — only assistant turns carry the contract."""
        body = _cc_transcript()
        _repair_thinking_roundtrip(body, native=False)

        assert body["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert body["messages"][3] == {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"}

    def test_repair_is_idempotent(self):
        """AC-2.6 — a repaired CC body reports no further change."""
        body = _cc_transcript()
        assert _repair_thinking_roundtrip(body, native=False) is True

        after_first = copy.deepcopy(body)
        assert _repair_thinking_roundtrip(body, native=False) is False
        assert body == after_first
