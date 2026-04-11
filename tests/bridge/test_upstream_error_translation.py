"""Tests for upstream error translation — Z.AI code 1261 and related error handling."""

from __future__ import annotations

import json

import pytest

from kitty.bridge.server import BridgeServer


class TestTranslateUpstreamError:
    """Unit tests for BridgeServer._translate_upstream_error."""

    @staticmethod
    def _call(status: int, body: object) -> str:
        return BridgeServer._translate_upstream_error(status, body)

    # ── Z.AI code 1261: Prompt exceeds max length ─────────────────────────

    @pytest.mark.parametrize(
        "status",
        [400, 429, 500, 502],
        ids=["400", "429", "500", "502"],
    )
    def test_zai_1261_dict_body(self, status: int):
        """Z.AI 1261 error with dict body returns actionable /clear message."""
        body = {"error": {"code": "1261", "message": "Prompt exceeds max length"}}
        result = self._call(status, body)
        assert "/clear" in result
        assert "context" in result.lower()

    @pytest.mark.parametrize(
        "status",
        [400, 429, 500, 502],
        ids=["400", "429", "500", "502"],
    )
    def test_zai_1261_string_body(self, status: int):
        """Z.AI 1261 error with raw JSON string body returns actionable /clear message."""
        body = json.dumps({"error": {"code": "1261", "message": "Prompt exceeds max length"}})
        result = self._call(status, body)
        assert "/clear" in result
        assert "context" in result.lower()

    def test_zai_1261_integer_code(self):
        """Z.AI 1261 error with integer code (not string) still detected."""
        body = {"error": {"code": 1261, "message": "Prompt exceeds max length"}}
        result = self._call(400, body)
        assert "/clear" in result
        assert "context" in result.lower()

    def test_zai_1261_message_match_fallback(self):
        """Detection via message substring when code is missing."""
        body = {"error": {"message": "Prompt exceeds max length"}}
        result = self._call(400, body)
        assert "/clear" in result

    def test_zai_1261_string_body_integer_code(self):
        """Z.AI 1261 error with raw JSON string containing integer code."""
        body = json.dumps({"error": {"code": 1261, "message": "Prompt exceeds max length"}})
        result = self._call(400, body)
        assert "/clear" in result

    # ── Backward compatibility: non-1261 errors ────────────────────────────

    def test_400_non_1261_returns_details(self):
        """Non-1261 400 errors return raw details (backward compat)."""
        body = {"error": {"code": "1210", "message": "Incorrect API call parameters"}}
        result = self._call(400, body)
        # Should NOT contain /clear
        assert "/clear" not in result
        # Should contain the error info
        assert "1210" in result or "Incorrect" in result

    def test_400_string_body_non_json(self):
        """Non-JSON string body on 400 returns the raw string."""
        result = self._call(400, "Something went wrong")
        assert result == "Something went wrong"

    # ── Backward compatibility: auth errors ────────────────────────────────

    def test_401_returns_auth_error(self):
        """401 errors return authentication failure message."""
        body = {"error": "Unauthorized"}
        result = self._call(401, body)
        assert "authentication failed" in result.lower()
        assert "kitty setup" in result.lower()

    def test_403_returns_auth_error(self):
        """403 errors return authentication failure message."""
        body = {"error": "Forbidden"}
        result = self._call(403, body)
        assert "authentication failed" in result.lower()
        assert "kitty setup" in result.lower()

    # ── Backward compatibility: 500 with code 1234 ─────────────────────────

    def test_500_code_1234_network_failure(self):
        """500 with code 1234 returns network failure message."""
        body = {"error": {"code": "1234", "message": "Network error, error id: abc-123"}}
        result = self._call(500, body)
        assert "network" in result.lower()
        assert "retry" in result.lower()

    def test_500_generic(self):
        """Generic 500 returns internal failure message."""
        body = {"error": {"message": "Internal server error"}}
        result = self._call(500, body)
        assert "internal failure" in result.lower()
        assert "retry" in result.lower()

    def test_500_string_body(self):
        """500 with string body still works."""
        body = json.dumps({"error": {"code": "1234", "message": "Network error"}})
        result = self._call(500, body)
        assert "network" in result.lower()

    def test_500_network_failure_substring_only(self):
        """500 with network failure message but no code detected via substring."""
        body = {"error": {"message": "Network failure, please retry later"}}
        result = self._call(500, body)
        assert "network" in result.lower()
        assert "retry" in result.lower()

    def test_zai_1261_on_500_returns_clear_not_network_failure(self):
        """1261 on 500 returns /clear message, not generic network failure."""
        body = {"error": {"code": "1261", "message": "Prompt exceeds max length"}}
        result = self._call(500, body)
        assert "/clear" in result
        assert "network" not in result.lower()

    def test_none_body_returns_empty_details(self):
        """None body returns empty string."""
        result = self._call(400, None)
        assert result == ""

    def test_extract_error_fields_dict(self):
        """_extract_error_fields works with dict body."""
        code, message = BridgeServer._extract_error_fields({"error": {"code": "1261", "message": "Too big"}})
        assert code == "1261"
        assert message == "Too big"

    def test_extract_error_fields_string(self):
        """_extract_error_fields works with JSON string body."""
        body = json.dumps({"error": {"code": "1234", "message": "Network error"}})
        code, message = BridgeServer._extract_error_fields(body)
        assert code == "1234"
        assert message == "Network error"

    def test_extract_error_fields_invalid_json_string(self):
        """_extract_error_fields returns empty for non-JSON string."""
        code, message = BridgeServer._extract_error_fields("not json")
        assert code == ""
        assert message == ""

    def test_extract_error_fields_none(self):
        """_extract_error_fields returns empty for None."""
        code, message = BridgeServer._extract_error_fields(None)
        assert code == ""
        assert message == ""

    # ── Minimax code 2013: context window exceeds limit ──────────────────────

    def test_minimax_2013_dict_body(self):
        """Minimax 2013 error with double-nested JSON dict body returns actionable /clear message."""
        inner_error = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "bad_request_error",
                    "message": "invalid params, context window exceeds limit (2013)",
                    "http_code": "400",
                },
                "request_id": "abc123",
            }
        )
        body = {
            "type": "error",
            "error": {"type": "api_error", "message": inner_error},
        }
        result = self._call(400, body)
        assert "/clear" in result
        assert "context" in result.lower()

    def test_minimax_2013_string_body(self):
        """Minimax 2013 error with raw JSON string body returns actionable /clear message."""
        inner_error = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "bad_request_error",
                    "message": "invalid params, context window exceeds limit (2013)",
                    "http_code": "400",
                },
            }
        )
        body = json.dumps(
            {
                "type": "error",
                "error": {"type": "api_error", "message": inner_error},
            }
        )
        result = self._call(400, body)
        assert "/clear" in result
        assert "context" in result.lower()

    def test_minimax_context_window_exceeds_limit_text_only(self):
        """Detection via 'context window exceeds' text even without nested JSON."""
        body = {"error": {"message": "context window exceeds limit"}}
        result = self._call(400, body)
        assert "/clear" in result

    @pytest.mark.parametrize(
        "status",
        [400, 429, 500, 502],
        ids=["400", "429", "500", "502"],
    )
    def test_minimax_2013_on_various_statuses(self, status: int):
        """Minimax-style error detected regardless of HTTP status."""
        inner_error = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "bad_request_error",
                    "message": "invalid params, context window exceeds limit (2013)",
                    "http_code": "400",
                },
            }
        )
        body = {"error": {"message": inner_error}}
        result = self._call(status, body)
        assert "/clear" in result

    def test_extract_error_fields_nested_json_message(self):
        """_extract_error_fields parses inner JSON from message field for code."""
        inner = json.dumps({"type": "error", "error": {"code": "2013", "message": "context window exceeds limit"}})
        code, message = BridgeServer._extract_error_fields({"error": {"message": inner}})
        assert code == "2013"
        assert "context window exceeds limit" in message

    def test_extract_error_fields_nested_json_message_with_outer_code(self):
        """Outer code takes precedence over nested code."""
        inner = json.dumps({"error": {"code": "2013", "message": "context window"}})
        code, message = BridgeServer._extract_error_fields({"error": {"code": "5000", "message": inner}})
        assert code == "5000"
        assert "context window" in message
