"""Driven tests for the response-direction transparency oracle.

`.system_design/TEST_SUITE.md` §3.3.1 (last paragraph), §3.3.2, §7.4 ·
plan task **T-D10** (KBR-59).

Tests run at ``l1`` (path-default per ``tests/layers.py``). They
exercise the response-direction comparison with real
:class:`~harness.contract.CapturedReply` bytes — the L4 equivalent
of the L1 oracle entry-point's seven assertions. The full L3
corpus-driven reply matrix is T-D9's scope.

The two driven cases shape the response oracle's contract from the
outside in:

* ``test_injected_unrecognised_field_fails_closed`` — the §3.3.1
  falsification analogue on the response side, end-to-end through the
  entry point (AC-FR-9).
* ``test_reply_oracle_raises_on_bedrock_converse_reply`` — the
  KBR-312 sibling gap surfaces at call time, not silently (AC-FR-4).
"""

from __future__ import annotations

import json

import pytest

from harness import contract as c
from harness import oracle
from harness import register as r
from harness.contract import CapturedReply, WireFormat


def _valid_messages_reply_body() -> bytes:
    """Return a minimal Anthropic Messages reply body.

    Used by the falsification case to construct a ``CapturedReply``
    whose body is recognised by ``AnthropicMessagesReplyProjection``
    without raising :class:`~harness.contract.UnreadableBodyError`.
    The body is minimal but well-formed: a single text block plus a
    ``stop_reason``.

    Returns:
        UTF-8 bytes of a JSON object the Anthropic Messages reply
        reader accepts.
    """
    return json.dumps(
        {
            "id": "msg_test_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ).encode("utf-8")


class TestDrivenReplyFalsification:
    """The response-direction oracle's first falsification case,
    end-to-end through the entry point (AC-FR-9)."""

    def test_injected_unrecognised_field_fails_closed(self) -> None:
        """A captured reply with an injected ``x-kitty-trace`` field
        fails the run at :class:`ResidualFieldsError` — unmodified
        (S6).

        Mirrors ``tests/harness/test_oracle.py``
        ``TestTotalityGate.test_injected_unrecognised_field_fails_closed``
        on the request side.
        """
        inbound_body = _valid_messages_reply_body()
        captured_body_dict = json.loads(_valid_messages_reply_body())
        captured_body_dict["x-kitty-trace"] = "injected"
        captured_body = json.dumps(captured_body_dict).encode("utf-8")

        with pytest.raises(c.ResidualFieldsError) as info:
            oracle.assert_no_unclaimed_reply_mutation(
                inbound=CapturedReply(status=200, headers=(), body=inbound_body),
                inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                captured=CapturedReply(status=200, headers=(), body=captured_body),
                captured_format=WireFormat.ANTHROPIC_MESSAGES,
                register=r.REGISTER,
                triggers_met=frozenset(),
            )
        assert "x-kitty-trace" in str(info.value)


class TestDrivenReplyRegistry:
    """The reply oracle raises at the registry boundary on a format
    with no reader (KBR-312 sibling, AC-FR-4)."""

    def test_reply_oracle_raises_on_bedrock_converse_reply(self) -> None:
        """Driving the reply oracle on ``WireFormat.BEDROCK_CONVERSE``
        — the format without a reply reader today — raises
        :class:`RuntimeError` naming the missing reader.

        The driven call surfaces the KBR-312 sibling gap as a hard
        failure rather than a vacuous pass. Lands the L1 contract in
        the suite.
        """
        body = _valid_messages_reply_body()
        with pytest.raises(RuntimeError):
            oracle.assert_no_unclaimed_reply_mutation(
                inbound=CapturedReply(status=200, headers=(), body=body),
                inbound_format=WireFormat.BEDROCK_CONVERSE,
                captured=CapturedReply(status=200, headers=(), body=body),
                captured_format=WireFormat.BEDROCK_CONVERSE,
                register=r.REGISTER,
                triggers_met=frozenset(),
            )
