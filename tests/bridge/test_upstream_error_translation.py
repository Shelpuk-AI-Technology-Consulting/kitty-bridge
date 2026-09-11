"""Tests for upstream error translation — Z.AI code 1261 and related error handling."""

from __future__ import annotations

import json

import pytest

from kitty.bridge.server import BridgeServer
from kitty.providers.base import ProviderAdapter
from kitty.providers.custom_openai import CustomOpenAIAdapter


class TestTranslateUpstreamError:
    """Unit tests for the static core, ``BridgeServer._translate_upstream_error_text``.

    The same-named instance method wraps this to supply the endpoint context; it
    is exercised in ``tests/bridge/test_custom_url_404.py``.
    """

    @staticmethod
    def _call(status: int, body: object) -> str:
        return BridgeServer._translate_upstream_error_text(status, body)

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

    # ── Minimax code 2013 reused for tool-call validation ───────────────

    def test_minimax_2013_tool_call_validation_dict_body(self):
        """Minimax 2013 with 'tool call result does not follow tool call' returns
        a tool-pairing-specific /clear message distinct from the context-window hint.
        """
        inner_error = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "bad_request_error",
                    "message": "invalid params, tool call result does not follow tool call (2013)",
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
        assert "tool" in result.lower() or "pairing" in result.lower()
        # The context-window hint must not be returned for the tool-call case
        assert "context has grown too large" not in result

    def test_minimax_2013_tool_call_validation_string_body(self):
        """Minimax 2013 tool-call validation with raw JSON string body."""
        inner_error = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "bad_request_error",
                    "message": "invalid params, tool call result does not follow tool call (2013)",
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
        assert "context has grown too large" not in result

    def test_minimax_2013_tool_call_validation_text_only(self):
        """The phrase 'tool call result does not follow tool call' is specific
        enough to detect on its own (no code 2013 required) and return the
        /clear hint. The user can recover from this state by clearing the
        conversation, regardless of which provider returned the error.
        """
        body = {"error": {"message": "tool call result does not follow tool call"}}
        result = self._call(400, body)
        assert "/clear" in result
        # Tool-call-specific phrasing, not the context-window hint
        assert "broken tool_use/tool_result pairing" in result


class TestCustomUrl404Message:
    """KBR-134 — a 404 from a user-configured endpoint must name the address.

    Claude Code renders any 404 on a model request as "that model may not
    exist", which sent the reporter hunting for a model problem that did not
    exist.  When the profile supplied the URL, the bridge says so instead.
    """

    # The text pinned in the requirements document, spelled out rather than
    # imported: an assertion that reads the implementation's own constant
    # proves only that the constant equals itself.
    EXPECTED = (
        'Upstream returned HTTP 404 for https://api.mistral.ai/v1/chat/completions/chat/completions. '
        "Either the model is not available at that endpoint, or this profile's base URL is wrong: "
        'Kitty appends "/chat/completions" to the base URL itself, so the base URL must end at the '
        'API root. Details: {"detail": "Not Found"}'
    )

    def test_names_the_requested_url_and_the_rule(self):
        """The reporter's exact failure produces the pinned message."""
        result = BridgeServer._translate_upstream_error_text(
            404,
            {"detail": "Not Found"},
            custom_url="https://api.mistral.ai/v1/chat/completions/chat/completions",
            appended_path="/chat/completions",
        )
        assert result == self.EXPECTED

    def test_omits_details_when_body_is_empty(self):
        """An empty upstream body drops the trailing clause rather than dangling."""
        result = BridgeServer._translate_upstream_error_text(
            404, None, custom_url="https://gw.example/v1/chat/completions", appended_path="/chat/completions"
        )
        assert result.endswith("must end at the API root.")
        assert "Details:" not in result

    def test_fixed_endpoint_provider_is_unaffected(self):
        """Without a custom URL the 404 body passes through exactly as before."""
        result = BridgeServer._translate_upstream_error_text(404, {"x": 1})
        assert result == json.dumps({"x": 1}, ensure_ascii=False)

    def test_context_window_error_still_wins(self):
        """A 404 carrying a context-window body keeps the more actionable /clear advice."""
        body = {"error": {"code": "1261", "message": "Prompt exceeds max length"}}
        result = BridgeServer._translate_upstream_error_text(
            404, body, custom_url="https://gw.example/v1/chat/completions", appended_path="/chat/completions"
        )
        assert "/clear" in result
        assert "base URL" not in result

    def test_auth_error_still_wins(self):
        """The 404 branch does not disturb the auth branch it sits below."""
        result = BridgeServer._translate_upstream_error_text(
            401, {"error": "Unauthorized"}, custom_url="https://gw.example/v1", appended_path="/chat/completions"
        )
        assert "authentication failed" in result.lower()


class TestCredentialRedaction:
    """A URL echoed into an error message must not carry credentials.

    KBR-134 redacted userinfo here through ``BridgeServer._redact_userinfo``.  That
    helper is superseded by ``ProviderAdapter.redact_url_for_display``, which also
    masks query values — a query could not reach an upstream before KBR-143, and is
    where gateways keep keys now that it can.  These cases are carried over rather
    than rewritten; the query ones are new.
    """

    def test_userinfo_is_removed(self):
        """The message travels into the agent transcript and the access log."""
        redacted = ProviderAdapter.redact_url_for_display("https://u:p@gw.example/v1/chat/completions")
        assert redacted == "https://gw.example/v1/chat/completions"

    def test_url_without_userinfo_is_unchanged(self):
        """Redaction is a no-op for the ordinary case."""
        url = "https://gw.example:8443/v1/chat/completions"
        assert ProviderAdapter.redact_url_for_display(url) == url

    def test_redacted_url_reaches_the_message(self):
        """Neither the password nor the ``user:pass@`` form survives into the text."""
        result = BridgeServer._translate_upstream_error_text(
            404,
            {},
            custom_url=ProviderAdapter.redact_url_for_display("https://u:hunter2@gw.example/v1/chat/completions"),
            appended_path="/chat/completions",
        )
        assert "hunter2" not in result
        assert "u:" not in result

    def test_a_query_credential_does_not_reach_the_message(self):
        """KBR-143 — the server composes the URL, so the server must redact it too.

        Built through the real instance method rather than handed a pre-redacted
        string: the claim is that *the bridge* redacts, not that a redactor exists.
        """
        server = BridgeServer(
            None,  # type: ignore[arg-type]
            CustomOpenAIAdapter(),
            "test-key",
            model="some-model",
            provider_config={"base_url": "https://gw.example/v1?subscription-key=s3cret"},
        )

        result = server._translate_upstream_error(404, {})

        assert "s3cret" not in result
        assert "subscription-key=****" in result
