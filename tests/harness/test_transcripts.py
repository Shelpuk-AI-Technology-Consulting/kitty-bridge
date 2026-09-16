"""Tests for the shared transcript strategy library.

`.system_design/TEST_SUITE.md` §6.1 · plan **T-F1** (KBR-70).

The strategy library ships with the §1.4 harness rule applied: every property
the library promises is exercised here, and every flaw the reporter is
supposed to catch has at least one deliberate-defect case. The library is the
substrate for downstream property tests (T-F2 compaction, T-F3 pairing and
truncation, T-F6 translator), so a green run here is the prerequisite that
lets those tasks build without rediscovering the same boundaries.

**Layer.** No ``pytestmark``, so these take the ``l1`` path default — the
same default ``harness/test_bridge.py`` and ``harness/test_vertical_slice.py``
take. The L1 selection picks the file up on the Fast gate.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from hypothesis import given, settings

from harness import transcripts as t

# Hypothesis CI profile is registered in ``tests/conftest.py`` at session
# start, so this module needs no local registration — and downstream
# property-test modules (T-F2, T-F3, T-F6) inherit the same profile
# deterministically rather than by collection order.


# ── AC-2: the documented import surface ────────────────────────────────────
#
# Each export below is also exercised by the property tests below, so this
# section is the import-surface contract: a missing name fails here before it
# fails in a downstream property test.


def test_every_documented_export_is_importable() -> None:
    """The documented strategy and reporter names are importable from the module.

    AC-2: ``from harness.transcripts import …`` exposes the names the module
    docstring promises. Downstream property tests compose these directly with
    ``@given(messages_request())`` etc., so a missing name is a substrate
    regression that breaks every consumer at once.
    """
    expected = {
        "ANTHROPIC_MODELS",
        "OPENAI_MODELS",
        "CONTENT_SHAPES",
        "ContentShape",
        "messages_text_block",
        "messages_image_block",
        "messages_document_block",
        "messages_tool_definition",
        "messages_tool_use_block",
        "messages_tool_result_block",
        "messages_user_message",
        "messages_assistant_message",
        "messages_tool_pair",
        "messages_request",
        "messages_problems",
        "cc_text_part",
        "cc_image_part",
        "cc_tool_definition",
        "cc_tool_call",
        "cc_tool_message",
        "cc_user_message",
        "cc_assistant_message",
        "cc_tool_pair",
        "cc_request",
        "cc_problems",
        # T-F3 (KBR-72) addition — OpenAI Responses strategies and reporter.
        # The twins' L1 properties consume these the same way T-F2 consumes
        # the CC / Messages strategies: per-test mutations on top of
        # request-level composites; the substrate owns the validity claim.
        "responses_tool_definition",
        "responses_function_call_item",
        "responses_function_call_output_item",
        "responses_message_item",
        "responses_request",
        "responses_problems",
    }

    missing = expected - set(t.__all__)
    assert not missing, f"transcripts.__all__ is missing: {sorted(missing)}"


# ── AC-3: every generated body is valid ───────────────────────────────────
#
# R3's body: every body any **request-level** strategy yields passes the
# format's reporter with zero problems, on every supported Python (3.10–3.13),
# and round-trips through ``json.dumps(body, allow_nan=False)``. Two property
# tests, one per format, both with 200+ examples.


@given(t.messages_request())
@settings(max_examples=200)
def test_every_messages_request_body_reports_no_problems(body: object) -> None:
    """Every generated Anthropic Messages body is valid by ``messages_problems``."""
    assert t.messages_problems(body) == []


@given(t.messages_request())
@settings(max_examples=200)
def test_every_messages_request_body_is_json_strict(body: dict) -> None:
    """Every generated Anthropic Messages body round-trips through ``json.dumps(allow_nan=False)``.

    H1's failure mode: NaN / Inf in any numeric field would otherwise smuggle a
    body through the property tests only to be rejected by the real provider
    with a 400. The strategy library must not generate any.
    """
    serialised = json.dumps(body, allow_nan=False)
    assert isinstance(serialised, str)
    # Round-tripping must yield a body the reporter still finds valid — JSON
    # dumps then loads is a structural identity the strategies must survive.
    reloaded = json.loads(serialised)
    assert t.messages_problems(reloaded) == []


@given(t.cc_request())
@settings(max_examples=200)
def test_every_cc_request_body_reports_no_problems(body: object) -> None:
    """Every generated Chat Completions body is valid by ``cc_problems``."""
    assert t.cc_problems(body) == []


@given(t.cc_request())
@settings(max_examples=200)
def test_every_cc_request_body_is_json_strict(body: dict) -> None:
    """Every generated Chat Completions body round-trips through ``json.dumps(allow_nan=False)``."""
    serialised = json.dumps(body, allow_nan=False)
    assert isinstance(serialised, str)
    reloaded = json.loads(serialised)
    assert t.cc_problems(reloaded) == []


# ── AC-4: pairing invariant is owned by the conversation composite (R6) ────
#
# The reporter is the body-level oracle; the pairing invariant lives inside
# ``messages_request`` and ``cc_request``. Drawing bodies from the request-level
# strategy and asserting the reporter is empty pins the invariant at its
# owner. (M2/AC-4: the message-level strategy cannot own pairing because a
# standalone message has no knowledge of its neighbours — R6 names the
# coupling seam.)


@given(t.messages_request())
def test_messages_request_composite_holds_the_pairing_invariant(body: dict) -> None:
    """The Anthropic conversation composite owns the pairing invariant (R6).

    AC-4: drawing a body from the request-level strategy and asserting the
    reporter is empty pins the invariant where the design says it lives — in
    the composite, not in any standalone message strategy.
    """
    assert t.messages_problems(body) == []


@given(t.cc_request())
def test_cc_request_composite_holds_the_pairing_invariant(body: dict) -> None:
    """The Chat Completions conversation composite owns the pairing invariant (R6)."""
    assert t.cc_problems(body) == []


# ── AC-5: both ``content`` shapes round-trip through the reporter ─────────
#
# §6.1 ``_validate_tool_call_pairing``: "Output contains no ``tool_result``
# without a ``tool_use``, **in both message shapes**". The parameter on the
# request-level strategy varies ``content`` between the string and block-list
# shapes; this property asserts both pass the reporter.


@given(t.messages_request(content_shape="string"))
@settings(max_examples=100)
def test_messages_string_content_shape_is_valid(body: object) -> None:
    """A request strategy parameterised to ``content_shape="string"`` is still valid."""
    assert t.messages_problems(body) == []


@given(t.messages_request(content_shape="blocks"))
@settings(max_examples=100)
def test_messages_blocks_content_shape_is_valid(body: object) -> None:
    """A request strategy parameterised to ``content_shape="blocks"`` is still valid."""
    assert t.messages_problems(body) == []


@given(t.cc_request(content_shape="string"))
@settings(max_examples=100)
def test_cc_string_content_shape_is_valid(body: object) -> None:
    """A Chat Completions request parameterised to ``content_shape="string"`` is still valid."""
    assert t.cc_problems(body) == []


@given(t.cc_request(content_shape="blocks"))
@settings(max_examples=100)
def test_cc_blocks_content_shape_is_valid(body: object) -> None:
    """A Chat Completions request parameterised to ``content_shape="blocks"`` is still valid."""
    assert t.cc_problems(body) == []


# ── AC-6: the reporter catches deliberate defects ─────────────────────────
#
# §1.4 harness rule: the first working version of every harness ships with at
# least one falsification case — a deliberate defect it must detect, running in
# the suite. The positive control below enumerates the defects and asserts the
# reporter names every one; the negative control demonstrates the reporter
# does not fire on innocent text.

#: Every defect the Messages reporter claims to catch. Each is the minimum
#: shape that exposes the rule it tests, so a passing ``messages_problems``
#: that returns ``[]`` on any of these is a vacuous pass.
_MESSAGES_FALSIFICATION_CASES: list[tuple[str, dict, str]] = [
    (
        "missing required field",
        {
            "messages": [{"role": "user", "content": "hi"}],
            # 'model' is missing on purpose
            "max_tokens": 10,
        },
        "missing required field 'model'",
    ),
    (
        "missing max_tokens",
        {
            "model": "claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
            # 'max_tokens' is missing on purpose — wire-required for Messages
        },
        "missing required field 'max_tokens'",
    ),
    (
        "first turn is not user",
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": "I'm starting"},
            ],
        },
        "first turn must be a user turn",
    ),
    (
        "consecutive user turns",
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
            ],
        },
        "consecutive 'user' turns at messages[1]",
    ),
    (
        "consecutive assistant turns",
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "assistant", "content": "c"},
            ],
        },
        "consecutive 'assistant' turns at messages[2]",
    ),
    (
        "undeclared tool use",
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "tools": [{"name": "real_tool", "description": "x", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "ghost_tool",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call_1", "content": "ok"}
                    ],
                },
            ],
        },
        "undeclared tool 'ghost_tool'",
    ),
    (
        "orphan tool result",
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "tools": [{"name": "tool_a", "description": "x", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_orphan",
                            "content": "ghost",
                        }
                    ],
                },
            ],
        },
        "tool_result 'call_orphan' has no matching tool_use",
    ),
    (
        "unanswered tool use",
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "tools": [{"name": "tool_a", "description": "x", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_unanswered", "name": "tool_a", "input": {}}
                    ],
                },
            ],
        },
        "tool_use 'call_unanswered' has no matching tool_result",
    ),
    (
        "tool_use with no id",
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "tools": [{"name": "tool_a", "description": "x", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "tool_a", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "any", "content": "ok"}
                    ],
                },
            ],
        },
        "messages[1] tool_use has no 'id'",
    ),
    (
        "tool_result with no tool_use_id",
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]},
            ],
        },
        "messages[2] tool_result has no 'tool_use_id'",
    ),
    (
        "messages is not a list",
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "messages": {"role": "user", "content": "hi"},
        },
        "'messages' must be a list",
    ),
    (
        "messages is empty",
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "messages": [],
        },
        "'messages' must be a non-empty list",
    ),
    (
        "message with no role",
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "hi"},
                {"content": "no role"},
            ],
        },
        "messages[1] has no 'role'",
    ),
    (
        "message with no content",
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "messages": [
                {"role": "user"},
            ],
        },
        "messages[0] has no 'content'",
    ),
    (
        "message with content: None (user)",
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": None},
            ],
        },
        "messages[0] content must not be None",
    ),
    (
        "tool_result not in the immediately next turn",
        {
            "model": "claude-sonnet-5",
            "max_tokens": 10,
            "tools": [{"name": "tool_a", "description": "x", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_late", "name": "tool_a", "input": {}}
                    ],
                },
                {"role": "user", "content": "unrelated turn"},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call_late", "content": "ok"}
                    ],
                },
            ],
        },
        "tool_use 'call_late' answered too late at messages[3]",
    ),
]


@pytest.mark.parametrize(
    ("label", "body", "expected_problem"),
    _MESSAGES_FALSIFICATION_CASES,
    ids=[case[0] for case in _MESSAGES_FALSIFICATION_CASES],
)
def test_messages_problems_catches_every_listed_defect(
    label: str, body: dict, expected_problem: str
) -> None:
    """The Anthropic reporter names every violation it claims to catch (AC-6)."""
    problems = t.messages_problems(body)
    assert problems, f"reporter returned [] on a body with defect {label!r}"
    assert any(
        expected_problem in problem for problem in problems
    ), f"reporter did not name {expected_problem!r} on {label!r}: got {problems}"


#: The matching defect list for Chat Completions.
_CC_FALSIFICATION_CASES: list[tuple[str, dict, str]] = [
    (
        "missing required field",
        {"messages": [{"role": "user", "content": "hi"}]},
        "missing required field 'model'",
    ),
    (
        "first message is not user",
        {
            "model": "gpt-4o",
            "messages": [{"role": "assistant", "content": "I'm starting"}],
        },
        "first message must be a user message",
    ),
    (
        "consecutive user messages",
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
            ],
        },
        "consecutive 'user' messages at messages[1]",
    ),
    (
        "consecutive assistant messages",
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "assistant", "content": "c"},
            ],
        },
        "consecutive 'assistant' messages at messages[2]",
    ),
    (
        "undeclared tool call",
        {
            "model": "gpt-4o",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "real_tool",
                        "description": "x",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "ghost_tool", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ],
        },
        "undeclared tool 'ghost_tool'",
    ),
    (
        "orphan tool message",
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "tool", "tool_call_id": "call_orphan", "content": "ghost"},
            ],
        },
        "tool message 'call_orphan' has no matching tool_call",
    ),
    (
        "unanswered tool call",
        {
            "model": "gpt-4o",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "tool_a",
                        "description": "x",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_unanswered",
                            "type": "function",
                            "function": {"name": "tool_a", "arguments": "{}"},
                        }
                    ],
                },
            ],
        },
        "tool_call 'call_unanswered' has no matching tool message",
    ),
    (
        "tool_call with no id",
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "tool_a", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "anything", "content": "ok"},
            ],
        },
        "messages[1] tool_call has no 'id'",
    ),
    (
        "tool message with no tool_call_id",
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
                {"role": "tool", "content": "no id"},
            ],
        },
        "messages[2] tool message has no 'tool_call_id'",
    ),
    (
        "messages is not a list",
        {
            "model": "gpt-4o",
            "messages": {"role": "user", "content": "hi"},
        },
        "'messages' must be a list",
    ),
    (
        "messages is empty",
        {
            "model": "gpt-4o",
            "messages": [],
        },
        "'messages' must be a non-empty list",
    ),
    (
        "message with no role",
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {"content": "no role"},
            ],
        },
        "messages[1] has no 'role'",
    ),
    (
        "assistant without content and no tool_calls",
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant"},
            ],
        },
        "messages[1] assistant has no 'content' and no tool_calls",
    ),
    (
        "user with content: None",
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": None},
            ],
        },
        "messages[0] content must not be None",
    ),
    (
        "tool with content: None",
        {
            "model": "gpt-4o",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "tool_a",
                        "description": "x",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "tool_a", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": None},
            ],
        },
        "messages[2] content must not be None",
    ),
    (
        "tool message not in the immediately next messages",
        {
            "model": "gpt-4o",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "tool_a",
                        "description": "x",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_late",
                            "type": "function",
                            "function": {"name": "tool_a", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "user", "content": "unrelated"},
                {"role": "tool", "tool_call_id": "call_late", "content": "ok"},
            ],
        },
        "tool_call 'call_late' answered too late at messages[3]",
    ),
]


@pytest.mark.parametrize(
    ("label", "body", "expected_problem"),
    _CC_FALSIFICATION_CASES,
    ids=[case[0] for case in _CC_FALSIFICATION_CASES],
)
def test_cc_problems_catches_every_listed_defect(
    label: str, body: dict, expected_problem: str
) -> None:
    """The Chat Completions reporter names every violation it claims to catch (AC-6)."""
    problems = t.cc_problems(body)
    assert problems, f"reporter returned [] on a body with defect {label!r}"
    assert any(
        expected_problem in problem for problem in problems
    ), f"reporter did not name {expected_problem!r} on {label!r}: got {problems}"


def test_messages_problems_does_not_fire_on_innocent_text() -> None:
    """Negative control: the Anthropic reporter returns [] on a hand-crafted minimal valid body."""
    body = {
        "model": "claude-sonnet-5",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi."}]},
            {"role": "user", "content": "Bye."},
        ],
    }
    assert t.messages_problems(body) == []


def test_cc_problems_does_not_fire_on_innocent_text() -> None:
    """Negative control: the Chat Completions reporter returns [] on a hand-crafted minimal valid body."""
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": "Bye."},
        ],
    }
    assert t.cc_problems(body) == []


def test_messages_problems_rejects_nan_float() -> None:
    """A body with NaN is not JSON-strictly serialisable (R3, H1).

    The reporter must catch the NaN case even when the rest of the body is
    well-formed — the property tests above rely on this so the substrate does
    not smuggle non-wire floats through AC-3.
    """
    body = {
        "model": "claude-sonnet-5",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": float("nan"),
    }
    problems = t.messages_problems(body)
    assert any("JSON" in problem for problem in problems), (
        f"reporter did not flag the NaN: {problems}"
    )


def test_cc_problems_rejects_nan_float() -> None:
    """The Chat Completions reporter flags NaN too (R3, H1, AC-6 completeness)."""
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": float("nan"),
    }
    problems = t.cc_problems(body)
    assert any("JSON" in problem for problem in problems), (
        f"reporter did not flag the NaN: {problems}"
    )


# ── T-F3 addition: the Responses format (KBR-72) ──────────────────────────
#
# Mirrors the per-format sections above. The Responses substrate was added by
# KBR-72 to support the L1 properties for
# ``BridgeServer._truncate_oversized_responses_outputs`` and
# ``BridgeServer._drop_orphan_responses_tool_outputs`` (register rows M3 / M7),
# whose wire rules are stated by ``responses_problems`` and whose inputs are
# valid Responses request bodies drawn from ``responses_request()``. The
# composite stays valid by construction (paired call / output items, fixture
# role alternation, named tools).


@given(t.responses_request())
@settings(max_examples=200)
def test_every_responses_request_body_reports_no_problems(body: object) -> None:
    """Every generated OpenAI Responses body is valid by ``responses_problems``."""
    assert t.responses_problems(body) == []


@given(t.responses_request())
@settings(max_examples=200)
def test_every_responses_request_body_is_json_strict(body: dict) -> None:
    """Every generated Responses body round-trips through ``json.dumps(allow_nan=False)``."""
    serialised = json.dumps(body, allow_nan=False)
    assert isinstance(serialised, str)
    reloaded = json.loads(serialised)
    assert t.responses_problems(reloaded) == []


@given(t.responses_request())
def test_responses_request_composite_holds_the_pairing_invariant(body: dict) -> None:
    """The Responses conversation composite owns the pairing invariant (R6 analogue)."""
    assert t.responses_problems(body) == []


_RESPONSES_DEFECTS: list[tuple[str, dict, str]] = [
    (
        "missing model",
        {"input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]}]},
        "missing required field 'model'",
    ),
    (
        "input is a string instead of a list",
        {"model": "gpt-4o", "input": "hi"},
        "'input' must be a list",
    ),
    (
        "function_call_output with no declaring function_call anywhere",
        {
            "model": "gpt-4o",
            "tools": [{"type": "function", "name": "f", "parameters": {}}],
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]},
                {"type": "function_call_output", "call_id": "orphan-1", "output": "y"},
            ],
        },
        "function_call_output 'orphan-1' has no matching function_call",
    ),
    (
        "function_call with no answering output",
        {
            "model": "gpt-4o",
            "tools": [{"type": "function", "name": "f", "parameters": {}}],
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]},
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "f",
                    "arguments": "{}",
                },
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "y"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "z"}]},
            ],
        },
        "function_call 'call-1' has no matching function_call_output",
    ),
    (
        "unknown item type",
        {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]},
                {"type": "made_up_kind", "call_id": "x"},
            ],
        },
        "unknown input item type 'made_up_kind'",
    ),
    (
        "function_call targets undeclared tool name",
        {
            "model": "gpt-4o",
            "tools": [{"type": "function", "name": "declared", "parameters": {}}],
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]},
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "not_declared",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "y",
                },
            ],
        },
        "function_call targets undeclared tool 'not_declared'",
    ),
    (
        "forward-reference output before its declaring call",
        {
            "model": "gpt-4o",
            "tools": [{"type": "function", "name": "f", "parameters": {}}],
            "input": [
                {"type": "function_call_output", "call_id": "a", "output": "early"},
                {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "a", "output": "late"},
            ],
        },
        "function_call_output 'a' answered before its declaring function_call",
    ),
    (
        "consecutive message items repeat a role (fixture-rule violation)",
        {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "again"}]},
            ],
        },
        "consecutive message items with role 'user'",
    ),
    (
        "function_call_output 'output' is neither a string nor a list of parts",
        {
            "model": "gpt-4o",
            "tools": [{"type": "function", "name": "f", "parameters": {}}],
            "input": [
                {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "a", "output": 42},
            ],
        },
        "function_call_output 'output' must be a string or a list of parts",
    ),
    (
        "function_call_output 'output' list carries an unknown part type",
        {
            "model": "gpt-4o",
            "tools": [{"type": "function", "name": "f", "parameters": {}}],
            "input": [
                {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "a", "output": [{"type": "made_up", "text": "x"}]},
            ],
        },
        "unknown output part type 'made_up'",
    ),
]


@pytest.mark.parametrize(
    ("label", "body", "expected_problem"),
    _RESPONSES_DEFECTS,
    ids=[c[0] for c in _RESPONSES_DEFECTS],
)
def test_responses_problems_catches_every_listed_defect(
    label: str, body: dict, expected_problem: str
) -> None:
    """The Responses reporter names every violation it claims to catch."""
    problems = t.responses_problems(body)
    assert problems, f"reporter returned [] on a body with defect {label!r}"
    assert any(
        expected_problem in problem for problem in problems
    ), f"reporter did not name {expected_problem!r} on {label!r}: got {problems}"


def test_responses_problems_does_not_fire_on_innocent_text() -> None:
    """Negative control: the Responses reporter returns ``[]`` on a hand-crafted minimal valid body."""
    body = {
        "model": "gpt-4o",
        "tools": [{"type": "function", "name": "f", "parameters": {"type": "object", "properties": {}}}],
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello."}]},
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "f",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call-1", "output": "ok"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi."}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Bye."}]},
        ],
    }
    assert t.responses_problems(body) == []


def test_responses_problems_rejects_nan_float() -> None:
    """A Responses body with NaN is not JSON-strictly serialisable."""
    body = {
        "model": "gpt-4o",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "temperature": float("nan"),
    }
    problems = t.responses_problems(body)
    assert any("JSON" in problem for problem in problems), (
        f"reporter did not flag the NaN: {problems}"
    )


#: Hostile shapes the reporters must survive without raising (M1). Downstream
#: T-F3 mutates valid bodies to produce its orphan-tool_result scenario, and a
#: mutation that deletes a key or replaces a value with ``None`` or a wrong
#: type must yield a problem list, not a crash in the middle of a property
#: run. Each entry is (label, reporter, body, expected problem substring) —
#: every hostile shape is also a violation, so a report of ``[]`` fails.
_HOSTILE_INPUT_CASES: list[tuple[str, str, object, str]] = [
    (
        "messages contains a non-dict",
        "messages",
        {"model": "x", "messages": [1]},
        "messages[0] is not a dict",
    ),
    (
        "tools is None",
        "messages",
        {"model": "x", "messages": [{"role": "user", "content": "a"}], "tools": None},
        "'tools' must be a list when present",
    ),
    (
        "cc messages contains a non-dict",
        "cc",
        {"model": "x", "messages": [1]},
        "messages[0] is not a dict",
    ),
    (
        "cc tools is None",
        "cc",
        {"model": "x", "messages": [{"role": "user", "content": "a"}], "tools": None},
        "'tools' must be a list when present",
    ),
    (
        "cc tool_call function is None",
        "cc",
        {
            "model": "x",
            "messages": [
                {"role": "user", "content": "a"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "1", "function": None}],
                },
            ],
        },
        "'function' is not a dict",
    ),
    (
        "cc tool_call function is a string",
        "cc",
        {
            "model": "x",
            "messages": [
                {"role": "user", "content": "a"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "1", "function": "oops"}],
                },
            ],
        },
        "'function' is not a dict",
    ),
    (
        "cc tool_calls is not a list",
        "cc",
        {
            "model": "x",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": None, "tool_calls": 5},
            ],
        },
        "'tool_calls' must be a list when present",
    ),
    (
        "responses input contains a non-dict item",
        "responses",
        {"model": "x", "input": [1]},
        "input[0] is not a dict",
    ),
    (
        "responses function_call_output has missing call_id",
        "responses",
        {
            "model": "x",
            "tools": [{"type": "function", "name": "f", "parameters": {}}],
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]},
                {"type": "function_call_output", "output": "y"},
            ],
        },
        "input[1] function_call_output has no 'call_id'",
    ),
    (
        "responses model is not a string",
        "responses",
        {"model": 5, "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]}]},
        "'model' must be a non-empty string",
    ),
]


@pytest.mark.parametrize(
    ("label", "format_name", "body", "expected_problem"),
    _HOSTILE_INPUT_CASES,
    ids=[case[0] for case in _HOSTILE_INPUT_CASES],
)
def test_reporters_never_raise_on_hostile_input(
    label: str, format_name: str, body: object, expected_problem: str
) -> None:
    """Every reporter returns a problem list — never raise — on hostile shapes (M1).

    The docstring promise is scoped the way the sibling scopes it
    (``cache_breakpoints.request_problems``): *on any value handed in, the
    reporter returns a list*. A raise inside a property run crashes the test
    instead of reporting the violation, so the contract is enforced here.
    """
    reporter = {
        "messages": t.messages_problems,
        "cc": t.cc_problems,
        "responses": t.responses_problems,
    }[format_name]

    problems = reporter(body)

    assert isinstance(problems, list), f"{label}: reporter did not return a list"
    assert any(
        expected_problem in problem for problem in problems
    ), f"{label}: reporter did not flag {expected_problem!r}: got {problems}"


# ── AC-7: the module imports nothing from ``src/kitty`` ────────────────────
#
# R4 + the §3.3.1 independence rule: the substrate's strategies cannot share
# assumptions with the code under test. The grep guard below mirrors the
# pattern in ``harness/test_contract.py`` and ``harness/test_provider_aiohttp.py``
# (the third per-module instance) — nine concrete forms the guard claims to
# catch, and a positive/negative control pair that proves the guard fires
# exactly on what it claims.


#: Every form of reaching into the product package. The static half is
#: anchored to the start of a line (with leading whitespace allowed, so an
#: indented import still matches) rather than floating, so prose mentioning
#: an import in a comment or docstring cannot trip it. The two dynamic forms
#: stay unanchored, because they appear mid-expression.
_KITTY_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(?:src\.)?kitty\b"
    r"|import_module\(\s*[\"'](?:src\.)?kitty"
    r"|__import__\(\s*[\"'](?:src\.)?kitty"
)


def test_transcripts_imports_nothing_from_kitty() -> None:
    """The strategy module is independent of ``src/kitty`` (§3.3.1, R4, AC-7)."""
    source = Path(t.__file__).read_text(encoding="utf-8")

    # Read no meaningful source: a guard that passes on an empty string is
    # indistinguishable from one that cannot fail.
    assert len(source) > 1000, "read no meaningful source; the guard would pass vacuously"

    offending = [line.strip() for line in source.splitlines() if _KITTY_IMPORT.search(line)]
    assert offending == [], f"transcripts.py must not import kitty: {offending}"


def test_transcripts_import_guard_actually_fires_on_every_form_it_claims_to_catch() -> None:
    """Positive control: the guard catches every form it claims to (AC-7)."""
    forms = [
        "from kitty.bridge import server",
        "import kitty",
        "from  kitty import server",
        "import  kitty.bridge",
        "from src.kitty import server",
        'mod = importlib.import_module("kitty.bridge.server")',
        '__import__("kitty")',
        'mod = importlib.import_module("src.kitty.bridge.server")',
        '__import__("src.kitty")',
    ]

    undetected = [form for form in forms if not _KITTY_IMPORT.search(form)]
    assert undetected == [], f"the guard would miss these: {undetected}"


def test_transcripts_import_guard_does_not_fire_on_innocent_text() -> None:
    """Negative control: the guard does not fire on innocent text (AC-7)."""
    innocent = [
        "# kitty-bridge is the product under test",
        "from harness import transcripts",
        "kitty = 1",
        "# never write `import kitty` in this module",
    ]
    assert [line for line in innocent if _KITTY_IMPORT.search(line)] == []


# ── Determinism: the CI profile derandomises; the local default does not ──
#
# §1.3 rule: a strategy the property tests rely on must be reproducible. Under
# the CI profile (``os.environ["CI"]`` set) hypothesis derandomises every run
# so a failure replays deterministically; locally the developer gets the
# randomised profile and the on-disk example database. We assert the profile
# shape here rather than pinning a single example: the request-level
# strategies' example space is far too large for a single witness, and the
# AC-3 property tests with 200 examples are the deterministic regression
# anchor when ``CI`` is set. The control below also asserts the no-database
# half of the profile fires under CI, so a developer's failure does not
# leak into CI.


def test_ci_profile_disables_the_example_database_when_ci_is_set() -> None:
    """Under ``CI=1`` the example database is disabled (R8)."""
    if not os.environ.get("CI"):
        pytest.skip("only meaningful when CI is set")
    from hypothesis import settings as h_settings

    assert h_settings().database is None


def test_ci_profile_derandomises_when_ci_is_set() -> None:
    """Under ``CI=1`` hypothesis derandomises so a failure replays (R8)."""
    if not os.environ.get("CI"):
        pytest.skip("only meaningful when CI is set")
    from hypothesis import settings as h_settings

    assert h_settings().derandomize is True
