"""Tests for reasoning effort extraction from API requests.

Covers Messages API (Anthropic), Responses API (OpenAI), and Chat Completions
pathways that feed normalized _reasoning_effort / _thinking_enabled into
provider adapters.
"""

from __future__ import annotations

from kitty.bridge.messages.translator import MessagesTranslator
from kitty.bridge.responses.translator import ResponsesTranslator


class TestMessagesTranslatorReasoning:
    """MessagesTranslator.translate_request() extracts thinking metadata."""

    def test_thinking_enabled_sets_reasoning_effort(self) -> None:
        t = MessagesTranslator()
        req = {
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = t.translate_request(req)
        assert result["_thinking_enabled"] is True
        assert result["_reasoning_effort"] == "high"

    def test_thinking_disabled_is_preserved(self) -> None:
        """Explicit ``thinking: {type: disabled}`` is a user configuration
        and must be preserved so Anthropic-compatible upstreams receive it.
        """
        t = MessagesTranslator()
        req = {
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = t.translate_request(req)
        assert result["_thinking_enabled"] is False
        assert "_reasoning_effort" not in result

    def test_effort_is_preserved(self) -> None:
        """Top-level Claude Code ``effort`` controls reasoning depth and
        must be passed to Anthropic-compatible upstreams.
        """
        t = MessagesTranslator()
        req = {
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "effort": "xhigh",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = t.translate_request(req)
        assert result["_effort"] == "xhigh"

    def test_thinking_adaptive_is_preserved(self) -> None:
        """Claude Code can send ``thinking: {type: adaptive}``; keep the
        original mode so AnthropicAdapter can restore it verbatim.
        """
        t = MessagesTranslator()
        req = {
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = t.translate_request(req)
        assert result["_thinking_adaptive"] is True
        assert result["_thinking_enabled"] is True
        assert result["_reasoning_effort"] == "high"

    def test_no_thinking_backward_compatible(self) -> None:
        t = MessagesTranslator()
        req = {
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = t.translate_request(req)
        assert "_thinking_enabled" not in result
        assert "_reasoning_effort" not in result

    def test_thinking_is_not_forwarded_as_top_level_key(self) -> None:
        t = MessagesTranslator()
        req = {
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = t.translate_request(req)
        assert "thinking" not in result


class TestResponsesTranslatorReasoning:
    """ResponsesTranslator.translate_request() extracts reasoning.effort."""

    def test_reasoning_effort_high(self) -> None:
        t = ResponsesTranslator()
        req = {
            "model": "gpt-5.4",
            "reasoning": {"effort": "high"},
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        }
        result = t.translate_request(req)
        assert result["_reasoning_effort"] == "high"
        assert result["_thinking_enabled"] is True

    def test_reasoning_effort_xhigh(self) -> None:
        t = ResponsesTranslator()
        req = {
            "model": "gpt-5.4",
            "reasoning": {"effort": "xhigh"},
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        }
        result = t.translate_request(req)
        assert result["_reasoning_effort"] == "xhigh"
        assert result["_thinking_enabled"] is True

    def test_reasoning_effort_low(self) -> None:
        t = ResponsesTranslator()
        req = {
            "model": "gpt-5.4",
            "reasoning": {"effort": "low"},
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        }
        result = t.translate_request(req)
        assert result["_reasoning_effort"] == "low"
        assert result["_thinking_enabled"] is True

    def test_reasoning_effort_none_value(self) -> None:
        t = ResponsesTranslator()
        req = {
            "model": "gpt-5.4",
            "reasoning": {"effort": "none"},
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        }
        result = t.translate_request(req)
        assert result["_reasoning_effort"] == "none"
        assert result["_thinking_enabled"] is False

    def test_no_reasoning_backward_compatible(self) -> None:
        t = ResponsesTranslator()
        req = {
            "model": "gpt-5.4",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        }
        result = t.translate_request(req)
        assert "_reasoning_effort" not in result
        assert "_thinking_enabled" not in result
