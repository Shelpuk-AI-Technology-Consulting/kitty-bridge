"""Tests for provider-level reasoning effort mapping.

Each provider translates the normalized _reasoning_effort / _thinking_enabled
into its upstream wire format.
"""

from __future__ import annotations

from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.openai import OpenAIAdapter
from kitty.providers.openai_subscription import OpenAISubscriptionAdapter
from kitty.providers.openrouter import OpenRouterAdapter
from kitty.providers.zai import ZaiRegularAdapter

# ── OpenAI ChatGPT Subscription (highest priority) ──────────────────────


class TestOpenAISubscriptionReasoning:
    def test_allowed_responses_params_includes_reasoning(self) -> None:
        assert "reasoning" in OpenAISubscriptionAdapter._ALLOWED_RESPONSES_PARAMS

    def test_prepare_responses_body_preserves_reasoning_from_original(self) -> None:
        original = {
            "model": "gpt-5.4",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "reasoning": {"effort": "high"},
        }
        result = OpenAISubscriptionAdapter._prepare_responses_body(original)
        assert result["reasoning"] == {"effort": "high"}

    def test_prepare_responses_body_injects_reasoning_from_cc_request(self) -> None:
        original = {
            "model": "gpt-5.4",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "_reasoning_effort": "high",
        }
        result = OpenAISubscriptionAdapter._prepare_responses_body(original)
        assert result["reasoning"] == {"effort": "high"}

    def test_cc_to_responses_injects_reasoning(self) -> None:
        adapter = OpenAISubscriptionAdapter()
        cc = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "_reasoning_effort": "high",
        }
        result = adapter._cc_to_responses(cc)
        assert result["reasoning"] == {"effort": "high"}

    def test_cc_to_responses_no_reasoning_when_absent(self) -> None:
        adapter = OpenAISubscriptionAdapter()
        cc = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = adapter._cc_to_responses(cc)
        assert "reasoning" not in result

    def test_cc_to_responses_no_reasoning_when_none(self) -> None:
        adapter = OpenAISubscriptionAdapter()
        cc = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "_reasoning_effort": "none",
        }
        result = adapter._cc_to_responses(cc)
        assert "reasoning" not in result

    def test_prepare_responses_body_no_reasoning_when_absent(self) -> None:
        original = {
            "model": "gpt-5.4",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        }
        result = OpenAISubscriptionAdapter._prepare_responses_body(original)
        assert "reasoning" not in result


# ── OpenAI (direct API) ─────────────────────────────────────────────────


class TestOpenAIReasoning:
    def test_translate_to_upstream_injects_reasoning_effort(self) -> None:
        adapter = OpenAIAdapter()
        cc = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "_reasoning_effort": "high",
            "_thinking_enabled": True,
        }
        result = adapter.translate_to_upstream(cc)
        assert result["reasoning_effort"] == "high"
        assert "_reasoning_effort" not in result
        assert "_thinking_enabled" not in result

    def test_translate_to_upstream_no_reasoning_when_absent(self) -> None:
        adapter = OpenAIAdapter()
        cc = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = adapter.translate_to_upstream(cc)
        assert "reasoning_effort" not in result
        assert "_reasoning_effort" not in result
        assert "_thinking_enabled" not in result

    def test_translate_to_upstream_strips_metadata(self) -> None:
        adapter = OpenAIAdapter()
        cc = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "_resolved_key": "some-key",
            "_provider_config": {"base_url": "http://example.com"},
        }
        result = adapter.translate_to_upstream(cc)
        assert "_resolved_key" not in result
        assert "_provider_config" not in result


# ── OpenRouter ──────────────────────────────────────────────────────────


class TestOpenRouterReasoning:
    def test_translate_to_upstream_injects_reasoning_object(self) -> None:
        adapter = OpenRouterAdapter()
        cc = {
            "model": "anthropic/claude-3-opus",
            "messages": [{"role": "user", "content": "hi"}],
            "_reasoning_effort": "high",
            "_thinking_enabled": True,
        }
        result = adapter.translate_to_upstream(cc)
        assert result["reasoning"] == {"effort": "high"}
        assert "_reasoning_effort" not in result
        assert "_thinking_enabled" not in result

    def test_translate_to_upstream_no_reasoning_when_absent(self) -> None:
        adapter = OpenRouterAdapter()
        cc = {
            "model": "anthropic/claude-3-opus",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = adapter.translate_to_upstream(cc)
        assert "reasoning" not in result
        assert "_reasoning_effort" not in result


# ── Z.AI ────────────────────────────────────────────────────────────────


class TestZAIReasoning:
    def test_translate_to_upstream_enables_thinking_on_effort(self) -> None:
        adapter = ZaiRegularAdapter()
        cc = {
            "model": "glm-4.7",
            "messages": [{"role": "user", "content": "hi"}],
            "_reasoning_effort": "high",
            "_thinking_enabled": True,
        }
        result = adapter.translate_to_upstream(cc)
        assert result["thinking"] == {"type": "enabled"}
        assert "_reasoning_effort" not in result
        assert "_thinking_enabled" not in result

    def test_translate_to_upstream_disables_thinking_when_none(self) -> None:
        adapter = ZaiRegularAdapter()
        cc = {
            "model": "glm-4.7",
            "messages": [{"role": "user", "content": "hi"}],
            "_reasoning_effort": "none",
            "_thinking_enabled": False,
        }
        result = adapter.translate_to_upstream(cc)
        assert result["thinking"] == {"type": "disabled"}
        assert "_reasoning_effort" not in result
        assert "_thinking_enabled" not in result

    def test_translate_to_upstream_disables_thinking_when_absent(self) -> None:
        adapter = ZaiRegularAdapter()
        cc = {
            "model": "glm-4.7",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = adapter.translate_to_upstream(cc)
        assert "thinking" not in result
        assert "_reasoning_effort" not in result


# ── Anthropic ───────────────────────────────────────────────────────────


class TestAnthropicReasoning:
    def test_translate_to_upstream_restores_thinking_on_enabled(self) -> None:
        adapter = AnthropicAdapter()
        cc = {
            "model": "claude-3-opus",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
            "_reasoning_effort": "high",
            "_thinking_enabled": True,
        }
        result = adapter.translate_to_upstream(cc)
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 10000}
        assert "_reasoning_effort" not in result
        assert "_thinking_enabled" not in result

    def test_translate_to_upstream_restores_thinking_on_disabled(self) -> None:
        adapter = AnthropicAdapter()
        cc = {
            "model": "claude-3-opus",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
            "_thinking_enabled": False,
        }
        result = adapter.translate_to_upstream(cc)
        assert result["thinking"] == {"type": "disabled"}

    def test_translate_to_upstream_no_thinking_when_absent(self) -> None:
        adapter = AnthropicAdapter()
        cc = {
            "model": "claude-3-opus",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
        }
        result = adapter.translate_to_upstream(cc)
        assert "thinking" not in result


# ── Base provider ───────────────────────────────────────────────────────


class TestBaseProviderStripsMetadata:
    def test_strips_internal_metadata_keys(self) -> None:
        """Default translate_to_upstream strips _reasoning_effort, _thinking_enabled,
        _resolved_key, _provider_config, _original_body."""
        adapter = OpenAIAdapter()  # uses base translate_to_upstream via super
        cc = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "_reasoning_effort": "high",
            "_thinking_enabled": True,
            "_resolved_key": "some-key",
            "_provider_config": {"base_url": "http://example.com"},
            "_original_body": {"model": "gpt-5.4"},
        }
        result = adapter.translate_to_upstream(cc)
        assert "_reasoning_effort" not in result
        assert "_thinking_enabled" not in result
        assert "_resolved_key" not in result
        assert "_provider_config" not in result
        assert "_original_body" not in result
