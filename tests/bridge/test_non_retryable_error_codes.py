"""Tests for non-retryable upstream error code detection (e.g. Z.AI 1211).

These tests verify:
  1. The _is_non_retryable_error_code helper detects permanent-failure codes.
  2. _should_retry_stream returns False for those codes (even on 5xx).
  3. Streaming and non-streaming paths do NOT retry on 1211.
  4. The upstream error message is surfaced to the client (not the empty-response fallback).
"""

from __future__ import annotations

import json

import pytest

from kitty.bridge.messages.translator import _EMPTY_ASSISTANT_FALLBACK_TEXT, MessagesTranslator
from kitty.bridge.server import BridgeServer

# ── Unit tests for _is_non_retryable_error_code ──────────────────────────────


class TestIsNonRetryableErrorCode:
    """Unit tests for BridgeServer._is_non_retryable_error_code."""

    @pytest.mark.parametrize(
        "status",
        [400, 422, 500, 502],
        ids=["400", "422", "500", "502"],
    )
    def test_1211_dict_body(self, status: int):
        """Code 1211 (Unknown Model) with dict body is non-retryable."""
        body = {"error": {"code": "1211", "message": "Unknown Model"}}
        assert BridgeServer._is_non_retryable_error_code(status, body) is True

    @pytest.mark.parametrize(
        "status",
        [400, 422, 500, 502],
        ids=["400", "422", "500", "502"],
    )
    def test_1211_string_body(self, status: int):
        """Code 1211 with raw JSON string body is non-retryable."""
        body = json.dumps({"error": {"code": "1211", "message": "Unknown Model"}})
        assert BridgeServer._is_non_retryable_error_code(status, body) is True

    def test_1211_integer_code(self):
        """Code 1211 as integer (not string) is still detected."""
        body = {"error": {"code": 1211, "message": "Unknown Model"}}
        assert BridgeServer._is_non_retryable_error_code(500, body) is True

    def test_1211_string_body_integer_code(self):
        """Code 1211 as integer in JSON string is still detected."""
        body = json.dumps({"error": {"code": 1211, "message": "Unknown Model"}})
        assert BridgeServer._is_non_retryable_error_code(500, body) is True

    def test_1234_network_failure_is_retryable(self):
        """Code 1234 (network failure) is still retryable."""
        body = {"error": {"code": "1234", "message": "Network error"}}
        assert BridgeServer._is_non_retryable_error_code(500, body) is False

    def test_500_no_code_is_retryable(self):
        """Generic 500 with no parseable error code is retryable."""
        assert BridgeServer._is_non_retryable_error_code(500, "Internal Server Error") is False

    def test_1261_context_too_large_is_non_retryable(self):
        """Code 1261 (context too large) is non-retryable."""
        body = {"error": {"code": "1261", "message": "Prompt exceeds max length"}}
        assert BridgeServer._is_non_retryable_error_code(400, body) is True

    def test_2013_context_exceeds_is_non_retryable(self):
        """Code 2013 (Minimax context exceeds) is non-retryable."""
        inner_error = json.dumps(
            {"type": "error", "error": {"code": "2013", "message": "context window exceeds limit"}}
        )
        body = {"error": {"message": inner_error}}
        assert BridgeServer._is_non_retryable_error_code(400, body) is True

    def test_unknown_code_is_retryable(self):
        """An unknown error code is retryable."""
        body = {"error": {"code": "9999", "message": "Something unexpected"}}
        assert BridgeServer._is_non_retryable_error_code(500, body) is False

    def test_none_body_is_retryable(self):
        """None body is retryable (no error code to check)."""
        assert BridgeServer._is_non_retryable_error_code(500, None) is False

    def test_empty_dict_is_retryable(self):
        """Empty dict body is retryable."""
        assert BridgeServer._is_non_retryable_error_code(500, {}) is False


# ── Unit tests for _should_retry_stream integration ───────────────────────────


class TestShouldRetryStreamWithNonRetryableCodes:
    """Verify _should_retry_stream respects non-retryable error codes."""

    def test_1211_on_500_not_retryable(self):
        """1211 with HTTP 500 is NOT retryable (even though 500 is in _RETRYABLE_STATUSES)."""
        body = json.dumps({"error": {"code": "1211", "message": "Unknown Model"}})
        assert BridgeServer._should_retry_stream(500, body) is False

    def test_1211_on_400_not_retryable(self):
        """1211 with HTTP 400 is NOT retryable."""
        body = json.dumps({"error": {"code": "1211", "message": "Unknown Model"}})
        assert BridgeServer._should_retry_stream(400, body) is False

    def test_generic_500_still_retryable(self):
        """Generic 500 without non-retryable code is still retryable."""
        body = json.dumps({"error": {"message": "Internal server error"}})
        assert BridgeServer._should_retry_stream(500, body) is True

    def test_429_still_retryable(self):
        """429 without non-retryable code is still retryable."""
        body = json.dumps({"error": {"message": "Rate limited"}})
        assert BridgeServer._should_retry_stream(429, body) is True

    def test_1234_on_500_still_retryable(self):
        """Code 1234 (network failure) on 500 is still retryable."""
        body = json.dumps({"error": {"code": "1234", "message": "Network error"}})
        assert BridgeServer._should_retry_stream(500, body) is True


# ── Fix 2: upstream error surfaced in fallback text ──────────────────────────


class TestFallbackTextWithUpstreamError:
    """Verify _fallback_assistant_text uses upstream error when available."""

    def test_no_context_returns_generic_fallback(self):
        """Without context, returns the generic empty-response text."""
        result = MessagesTranslator._fallback_assistant_text()
        assert result == _EMPTY_ASSISTANT_FALLBACK_TEXT

    def test_context_without_upstream_error_returns_generic(self):
        """With context but no upstream_error, returns generic text with metadata."""
        result = MessagesTranslator._fallback_assistant_text(
            context={"provider": "zai_coding", "model": "glm-5.1", "attempts": 5}
        )
        assert "Upstream model returned an empty response" in result
        assert "zai_coding" in result
        assert "glm-5.1" in result
        assert "after 5 attempts" in result

    def test_context_with_upstream_error_uses_error_message(self):
        """With upstream_error in context, uses that instead of generic text."""
        result = MessagesTranslator._fallback_assistant_text(
            context={
                "provider": "zai_coding",
                "model": "glm-5.1",
                "attempts": 5,
                "upstream_error": "[1211][Unknown Model, please check the model code.][abc123]",
            }
        )
        # Should NOT contain the generic "empty response" text
        assert "Upstream model returned an empty response" not in result
        # Should contain the actual upstream error
        assert "[1211]" in result
        assert "Unknown Model" in result
        # Should still have metadata
        assert "zai_coding" in result
        assert "glm-5.1" in result

    def test_upstream_error_without_metadata(self):
        """With upstream_error but no provider/model, still uses error message."""
        result = MessagesTranslator._fallback_assistant_text(
            context={"upstream_error": "Internal server error"}
        )
        assert "Internal server error" in result
        assert "Upstream model returned an empty response" not in result

    def test_refusal_takes_priority_over_upstream_error(self):
        """If the message has a refusal field, it takes priority."""
        result = MessagesTranslator._fallback_assistant_text(
            message={"refusal": "Content filtered"},
            context={"upstream_error": "some error"},
        )
        assert result == "Content filtered"
