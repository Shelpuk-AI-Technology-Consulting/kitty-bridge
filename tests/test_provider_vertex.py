"""Tests for providers/vertex.py — VertexAIAdapter."""

import json

import pytest

from kitty.providers.vertex import VertexAIAdapter
from kitty.providers.base import ProviderError

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


class TestVertexAdapterProperties:
    def setup_method(self):
        self.adapter = VertexAIAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "vertex"

    def test_default_base_url_contains_aiplatform(self):
        assert "aiplatform.googleapis.com" in self.adapter.default_base_url

    def test_use_custom_transport(self):
        # Vertex AI uses standard CC format over aiohttp — no custom transport
        assert self.adapter.use_custom_transport is False


# ── upstream_path ────────────────────────────────────────────────────────


class TestVertexUpstreamPath:
    def setup_method(self):
        self.adapter = VertexAIAdapter()

    def test_path_includes_endpoints_openapi(self):
        path = self.adapter.upstream_path
        assert "/endpoints/openapi/" in path

    def test_path_includes_chat_completions(self):
        path = self.adapter.upstream_path
        assert "/chat/completions" in path

    def test_get_upstream_path_with_model(self):
        # Model is NOT in the path for Vertex AI — it's in the body
        path = self.adapter.get_upstream_path("google/gemini-2.0-flash-001")
        assert "/endpoints/openapi/" in path
        assert "/chat/completions" in path


# ── Base URL construction ────────────────────────────────────────────────


class TestVertexBuildUpstreamUrl:
    def setup_method(self):
        self.adapter = VertexAIAdapter()

    def test_default_base_url_uses_default_location(self):
        url = self.adapter.default_base_url
        assert "us-central1" in url

    def test_build_base_url_with_config(self):
        config = {"project_id": "my-project", "location": "us-central1"}
        url = self.adapter.build_base_url(config)
        assert "us-central1" in url
        assert "my-project" in url
        assert "aiplatform.googleapis.com" in url

    def test_build_base_url_global(self):
        config = {"project_id": "my-project", "location": "global"}
        url = self.adapter.build_base_url(config)
        assert "aiplatform.googleapis.com" in url
        assert "my-project" in url

    def test_build_base_url_default_location(self):
        config = {"project_id": "my-project"}
        url = self.adapter.build_base_url(config)
        assert "us-central1" in url

    def test_build_base_url_empty_location_falls_back(self):
        config = {"project_id": "my-project", "location": ""}
        url = self.adapter.build_base_url(config)
        assert "us-central1" in url

    def test_build_base_url_missing_project_raises(self):
        with pytest.raises(ProviderError, match="project_id"):
            self.adapter.build_base_url({})

    def test_build_base_url_empty_project_raises(self):
        with pytest.raises(ProviderError, match="project_id"):
            self.adapter.build_base_url({"project_id": ""})


# ── Auth headers ─────────────────────────────────────────────────────────


class TestVertexBuildUpstreamHeaders:
    def setup_method(self):
        self.adapter = VertexAIAdapter()

    def test_bearer_token_auth(self):
        headers = self.adapter.build_upstream_headers("ya29.a0AfH6SMB...")
        assert headers["Authorization"] == "Bearer ya29.a0AfH6SMB..."
        assert headers["Content-Type"] == "application/json"

    def test_no_api_key_header(self):
        # Vertex uses Authorization: Bearer, not api-key
        headers = self.adapter.build_upstream_headers("my-token")
        assert "api-key" not in headers
        assert "api_key" not in headers


# ── Request translation ─────────────────────────────────────────────────


class TestVertexTranslateToUpstream:
    def setup_method(self):
        self.adapter = VertexAIAdapter()

    def test_preserves_model(self):
        # Vertex AI uses model in body with "google/" prefix
        cc = {
            "model": "gemini-2.0-flash-001",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["model"] == "gemini-2.0-flash-001"

    def test_preserves_messages(self):
        cc = {
            "model": "google/gemini-2.0-flash-001",
            "messages": CC_MESSAGES_BASIC,
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"

    def test_preserves_stream(self):
        cc = {
            "model": "gemini-2.0-flash-001",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["stream"] is True

    def test_preserves_tools(self):
        cc = {
            "model": "gemini-2.0-flash-001",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {"name": "test", "parameters": {}}}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert len(result["tools"]) == 1

    def test_preserves_temperature(self):
        cc = {
            "model": "gemini-2.0-flash-001",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "temperature": 0.7,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["temperature"] == 0.7

    def test_preserves_max_tokens(self):
        cc = {
            "model": "gemini-2.0-flash-001",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 4096,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["max_tokens"] == 4096

    def test_removes_internal_fields(self):
        cc = {
            "model": "gemini-2.0-flash-001",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": "test",
            "_provider_config": {},
        }
        result = self.adapter.translate_to_upstream(cc)
        assert "_resolved_key" not in result
        assert "_provider_config" not in result


# ── Response translation (passthrough — CC compatible) ───────────────────


class TestVertexTranslateFromUpstream:
    def setup_method(self):
        self.adapter = VertexAIAdapter()

    def test_passthrough(self):
        resp = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.adapter.translate_from_upstream(resp)
        assert result is resp  # same object — passthrough


# ── Model name normalization ────────────────────────────────────────────


class TestVertexNormalizeModelName:
    def setup_method(self):
        self.adapter = VertexAIAdapter()

    def test_adds_google_prefix(self):
        # Vertex AI expects "google/{MODEL_ID}" format
        assert self.adapter.normalize_model_name("gemini-2.0-flash-001") == "google/gemini-2.0-flash-001"

    def test_does_not_double_prefix(self):
        # If already has "google/" prefix, don't add another
        assert self.adapter.normalize_model_name("google/gemini-2.0-flash-001") == "google/gemini-2.0-flash-001"

    def test_preserves_publishers_prefix(self):
        # Models from other publishers (e.g. meta) keep their prefix
        assert self.adapter.normalize_model_name("meta/llama-3.1-405b") == "meta/llama-3.1-405b"

    def test_preserves_other_publisher_prefix(self):
        # Third-party models on Vertex AI keep their prefix
        assert self.adapter.normalize_model_name("anthropic/claude-3-5-sonnet") == "anthropic/claude-3-5-sonnet"

    def test_empty_model(self):
        assert self.adapter.normalize_model_name("") == ""

    def test_publishers_path_models(self):
        # Some models use publishers path format
        assert self.adapter.normalize_model_name("google/gemini-2.5-pro") == "google/gemini-2.5-pro"


# ── normalize_request ───────────────────────────────────────────────────


class TestVertexNormalizeRequest:
    def setup_method(self):
        self.adapter = VertexAIAdapter()

    def test_does_not_modify_request(self):
        cc = {"model": "gemini-2.0-flash-001", "messages": []}
        self.adapter.normalize_request(cc)
        assert cc == {"model": "gemini-2.0-flash-001", "messages": []}


# ── Error mapping ────────────────────────────────────────────────────────


class TestVertexMapError:
    def setup_method(self):
        self.adapter = VertexAIAdapter()

    def test_401_error(self):
        exc = self.adapter.map_error(401, {"error": {"message": "unauthenticated"}})
        assert "401" in str(exc)

    def test_429_error(self):
        exc = self.adapter.map_error(429, {"error": {"message": "rate limited"}})
        assert "429" in str(exc)

    def test_500_error(self):
        exc = self.adapter.map_error(500, {"error": {"message": "internal error"}})
        assert "500" in str(exc)

    def test_non_dict_body(self):
        exc = self.adapter.map_error(500, "something broke")
        assert "500" in str(exc)


# ── build_request / parse_response ──────────────────────────────────────


class TestVertexBuildRequest:
    def setup_method(self):
        self.adapter = VertexAIAdapter()

    def test_build_request_basic(self):
        result = self.adapter.build_request(
            model="google/gemini-2.0-flash-001",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )
        assert result["model"] == "google/gemini-2.0-flash-001"
        assert result["stream"] is False


class TestVertexParseResponse:
    def setup_method(self):
        self.adapter = VertexAIAdapter()

    def test_parse_response(self):
        cc_resp = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.adapter.parse_response(cc_resp)
        assert result["content"] == "hi"
        assert result["finish_reason"] == "stop"

    def test_parse_response_null_usage(self):
        cc_resp = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": None,
        }
        result = self.adapter.parse_response(cc_resp)
        assert result["usage"] == {}

    def test_parse_response_with_tool_calls(self):
        cc_resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "test", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.adapter.parse_response(cc_resp)
        assert result["content"] is None
        assert result["finish_reason"] == "tool_calls"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "test"


# ── Credential validation ────────────────────────────────────────────────


class TestVertexCredentials:
    def setup_method(self):
        self.adapter = VertexAIAdapter()

    def test_valid_access_token(self):
        # OAuth2 access token from gcloud or ADC
        assert self.adapter.is_valid_access_token("ya29.a0AfH6SMBx...")
        assert self.adapter.is_valid_access_token("simple-token-string")

    def test_empty_token_invalid(self):
        assert not self.adapter.is_valid_access_token("")

    def test_whitespace_only_token_invalid(self):
        assert not self.adapter.is_valid_access_token("   ")

    def test_none_token_invalid(self):
        assert not self.adapter.is_valid_access_token(None)
