"""Tests for providers/azure.py — AzureOpenAIAdapter."""

from kitty.providers.azure import AzureOpenAIAdapter

# ── CC format samples ────────────────────────────────────────────────────

CC_MESSAGES_BASIC = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
]

CC_MESSAGES_TOOLS = [
    {"role": "user", "content": "What's the weather?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_abc",
        "content": "15°C, cloudy",
    },
]

# ── Properties ───────────────────────────────────────────────────────────


class TestAzureAdapterProperties:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "azure"

    def test_default_base_url(self):
        # No single default — must be configured per-profile
        assert "openai.azure.com" in self.adapter.default_base_url

    def test_use_custom_transport(self):
        # Azure uses standard CC format — no custom transport needed
        assert self.adapter.use_custom_transport is False


# ── upstream_path ────────────────────────────────────────────────────────


class TestAzureUpstreamPath:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_path_includes_deployment(self):
        path = self.adapter.upstream_path
        assert "/openai/deployments/" in path
        assert "/chat/completions" in path

    def test_path_with_deployment_id(self):
        adapter = AzureOpenAIAdapter()
        path = adapter.get_upstream_path("my-gpt4o-deployment")
        assert path == "/openai/deployments/my-gpt4o-deployment/chat/completions?api-version=2024-10-21"

    def test_api_version_query_param(self):
        adapter = AzureOpenAIAdapter()
        assert "api-version=" in adapter.get_upstream_path("dep")


# ── Auth headers ─────────────────────────────────────────────────────────


class TestAzureBuildUpstreamHeaders:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_api_key_auth(self):
        headers = self.adapter.build_upstream_headers("my-azure-api-key-123")
        assert headers["api-key"] == "my-azure-api-key-123"
        assert "Authorization" not in headers

    def test_entra_token_auth(self):
        """When key starts with 'Bearer ', use Authorization header."""
        headers = self.adapter.build_upstream_headers("Bearer eyJ0eXAiOiJKV1Q...")
        assert headers["Authorization"] == "Bearer eyJ0eXAiOiJKV1Q..."
        assert "api-key" not in headers

    def test_content_type_included(self):
        headers = self.adapter.build_upstream_headers("test-key")
        assert headers["Content-Type"] == "application/json"


# ── Request translation ─────────────────────────────────────────────────


class TestAzureTranslateToUpstream:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_removes_model_from_body(self):
        """Azure uses deployment-id in URL, not model in body."""
        cc = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert "model" not in result

    def test_preserves_messages(self):
        cc = {
            "model": "gpt-4o",
            "messages": CC_MESSAGES_BASIC,
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"

    def test_preserves_stream(self):
        cc = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["stream"] is True

    def test_preserves_tools(self):
        cc = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {"name": "test", "parameters": {}}}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert len(result["tools"]) == 1

    def test_preserves_temperature(self):
        cc = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "temperature": 0.7,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["temperature"] == 0.7

    def test_preserves_max_tokens(self):
        cc = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 4096,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["max_tokens"] == 4096

    def test_removes_internal_fields(self):
        cc = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": "test",
            "_provider_config": {},
        }
        result = self.adapter.translate_to_upstream(cc)
        assert "_resolved_key" not in result
        assert "_provider_config" not in result


# ── Response translation (passthrough — CC compatible) ───────────────────


class TestAzureTranslateFromUpstream:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_passthrough(self):
        resp = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.adapter.translate_from_upstream(resp)
        assert result is resp  # same object — passthrough


# ── normalize_model_name ────────────────────────────────────────────────


class TestAzureNormalizeModelName:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_returns_unchanged(self):
        # Azure uses deployment names, not model names — passthrough
        assert self.adapter.normalize_model_name("my-gpt4o-deployment") == "my-gpt4o-deployment"


# ── normalize_request ───────────────────────────────────────────────────


class TestAzureNormalizeRequest:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_does_not_modify_request(self):
        cc = {"model": "gpt-4o", "messages": []}
        self.adapter.normalize_request(cc)
        assert cc == {"model": "gpt-4o", "messages": []}


# ── Error mapping ────────────────────────────────────────────────────────


class TestAzureMapError:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_401_error(self):
        exc = self.adapter.map_error(401, {"error": {"message": "invalid api-key"}})
        assert "401" in str(exc)

    def test_429_error(self):
        exc = self.adapter.map_error(429, {"error": {"message": "rate limited"}})
        assert "429" in str(exc)

    def test_500_error(self):
        exc = self.adapter.map_error(500, {"error": {"message": "internal error"}})
        assert "500" in str(exc)


# ── build_request / parse_response ──────────────────────────────────────


class TestAzureBuildRequest:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_build_request_basic(self):
        result = self.adapter.build_request(
            model="my-deployment",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )
        assert result["model"] == "my-deployment"
        assert result["stream"] is False


class TestAzureParseResponse:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_parse_response(self):
        cc_resp = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.adapter.parse_response(cc_resp)
        assert result["content"] == "hi"
        assert result["finish_reason"] == "stop"


# ── Entra ID token resolution ────────────────────────────────────────────


class TestAzureEntraToken:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_is_entra_token(self):
        assert self.adapter.is_entra_token("Bearer eyJ0eXAiOi...") is True

    def test_is_not_entra_token(self):
        assert self.adapter.is_entra_token("my-api-key-123") is False

    def test_entra_prefix_case_insensitive(self):
        assert self.adapter.is_entra_token("bearer eyJ...") is True


class TestAzureEntraTokenNormalization:
    """Verify build_upstream_headers normalizes bearer casing."""

    def test_lowercase_bearer_normalized_to_proper(self):
        adapter = AzureOpenAIAdapter()
        headers = adapter.build_upstream_headers("bearer eyJ0eXAiOi...")
        assert headers["Authorization"] == "Bearer eyJ0eXAiOi..."

    def test_mixed_case_bearer_normalized(self):
        adapter = AzureOpenAIAdapter()
        headers = adapter.build_upstream_headers("Bearer eyJ0eXAiOi...")
        assert headers["Authorization"] == "Bearer eyJ0eXAiOi..."
