"""Tests for rate limit / quota exhaustion detection and circuit breaker."""


from kitty.bridge.server import BridgeServer

# ── _is_rate_limit_error ───────────────────────────────────────────────────


class TestIsRateLimitError:
    """Test static method _is_rate_limit_error."""

    def test_code_1310_detected(self):
        body = {"error": {"code": "1310", "message": "Weekly/Monthly Limit Exhausted"}}
        assert BridgeServer._is_rate_limit_error(400, body) is True

    def test_code_1310_on_429(self):
        body = {"error": {"code": "1310", "message": "Limit Exhausted"}}
        assert BridgeServer._is_rate_limit_error(429, body) is True

    def test_limit_exhausted_pattern(self):
        body = {
            "error": {
                "code": "1310",
                "message": (
                    "Weekly/Monthly Limit Exhausted. "
                    "Your limit will reset at 2026-04-13 01:37:41"
                ),
            }
        }
        assert BridgeServer._is_rate_limit_error(400, body) is True

    def test_limit_exhausted_variant(self):
        body = {"error": {"code": "unknown", "message": "Your limit has been exhausted"}}
        assert BridgeServer._is_rate_limit_error(400, body) is True

    def test_rate_limit_pattern(self):
        body = {"error": {"code": "429", "message": "Rate limit exceeded"}}
        assert BridgeServer._is_rate_limit_error(429, body) is True

    def test_quota_exceeded_pattern(self):
        body = {"error": {"code": "xxx", "message": "Quota exceeded for this plan"}}
        assert BridgeServer._is_rate_limit_error(402, body) is True

    def test_usage_limit_pattern(self):
        body = {"error": {"code": "err", "message": "usage limit reached"}}
        assert BridgeServer._is_rate_limit_error(400, body) is True

    def test_context_too_large_not_rate_limit(self):
        """Context-too-large errors must NOT be treated as rate limits."""
        body = {"error": {"code": "2013", "message": "context window exceeds limit"}}
        assert BridgeServer._is_rate_limit_error(400, body) is False

    def test_code_1261_not_rate_limit(self):
        body = {"error": {"code": "1261", "message": "prompt exceeds max length"}}
        assert BridgeServer._is_rate_limit_error(400, body) is False

    def test_auth_401_not_rate_limit(self):
        body = {"error": {"code": "1310", "message": "limit exhausted"}}
        assert BridgeServer._is_rate_limit_error(401, body) is False

    def test_auth_403_not_rate_limit(self):
        body = {"error": {"code": "1310", "message": "limit exhausted"}}
        assert BridgeServer._is_rate_limit_error(403, body) is False

    def test_normal_400_not_rate_limit(self):
        body = {"error": {"code": "bad_request", "message": "Invalid parameter"}}
        assert BridgeServer._is_rate_limit_error(400, body) is False

    def test_normal_500_is_rate_limit_by_status(self):
        """500 is already retryable by status, but _is_rate_limit_error only checks error content."""
        body = {"error": {"code": "internal", "message": "Server error"}}
        assert BridgeServer._is_rate_limit_error(500, body) is False

    def test_string_body_with_code_1310(self):
        body = '{"error":{"code":"1310","message":"Limit Exhausted"}}'
        assert BridgeServer._is_rate_limit_error(400, body) is True

    def test_string_body_rate_limit_pattern(self):
        body = '{"error":{"code":"429","message":"rate limit hit"}}'
        assert BridgeServer._is_rate_limit_error(400, body) is True

    def test_empty_body(self):
        assert BridgeServer._is_rate_limit_error(400, None) is False

    def test_plain_string_body_no_json(self):
        assert BridgeServer._is_rate_limit_error(400, "something broke") is False


# ── _should_retry_stream ───────────────────────────────────────────────────


class TestShouldRetryStream:
    """Test static method _should_retry_stream."""

    def test_retryable_429(self):
        assert BridgeServer._should_retry_stream(429, "") is True

    def test_retryable_500(self):
        assert BridgeServer._should_retry_stream(500, "") is True

    def test_retryable_502(self):
        assert BridgeServer._should_retry_stream(502, "") is True

    def test_retryable_503(self):
        assert BridgeServer._should_retry_stream(503, "") is True

    def test_retryable_504(self):
        assert BridgeServer._should_retry_stream(504, "") is True

    def test_400_with_rate_limit_code(self):
        error_body = '{"error":{"code":"1310","message":"Limit Exhausted"}}'
        assert BridgeServer._should_retry_stream(400, error_body) is True

    def test_400_with_rate_limit_pattern(self):
        error_body = '{"error":{"code":"xxx","message":"quota exceeded"}}'
        assert BridgeServer._should_retry_stream(400, error_body) is True

    def test_400_normal_error_not_retryable(self):
        error_body = '{"error":{"code":"bad_request","message":"Invalid input"}}'
        assert BridgeServer._should_retry_stream(400, error_body) is False

    def test_400_context_too_large_not_retryable(self):
        error_body = '{"error":{"code":"2013","message":"context window exceeds limit"}}'
        assert BridgeServer._should_retry_stream(400, error_body) is False
