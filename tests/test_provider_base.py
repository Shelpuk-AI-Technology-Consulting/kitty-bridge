"""Tests for providers/base.py — ProviderAdapter interface."""

import pytest

from kitty.providers.base import ProviderAdapter


def _stub_adapter() -> ProviderAdapter:
    """Create a minimal concrete ProviderAdapter for testing defaults."""

    class _StubAdapter(ProviderAdapter):
        @property
        def provider_type(self) -> str:
            return "stub"

        @property
        def default_base_url(self) -> str:
            return "https://example.com/v1"

        def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
            return {}

        def parse_response(self, response_data: dict) -> dict:
            return {}

        def map_error(self, status_code: int, body: dict) -> Exception:
            return Exception("stub")

    return _StubAdapter()


class TestProviderAdapter:
    def test_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            ProviderAdapter()  # type: ignore[abstract]


class TestNormalizeModelNameDefault:
    """Default normalize_model_name returns model unchanged."""

    def test_default_returns_unchanged(self):
        adapter = _stub_adapter()
        assert adapter.normalize_model_name("gpt-4o") == "gpt-4o"
        assert adapter.normalize_model_name("minimax/minimax-m2.7") == "minimax/minimax-m2.7"


class TestNormalizeRequestDefault:
    """Default normalize_request does nothing."""

    def test_default_does_not_modify_request(self):
        adapter = _stub_adapter()
        cc = {"model": "gpt-4o", "messages": []}
        adapter.normalize_request(cc)
        assert cc == {"model": "gpt-4o", "messages": []}


class TestUpstreamPathDefault:
    """Default upstream_path returns /chat/completions."""

    def test_default_is_chat_completions(self):
        adapter = _stub_adapter()
        assert adapter.upstream_path == "/chat/completions"

    def test_get_upstream_path_ignores_model(self):
        adapter = _stub_adapter()
        assert adapter.get_upstream_path("gpt-4o") == "/chat/completions"


class TestBuildUpstreamHeadersDefault:
    """Default build_upstream_headers returns Bearer auth."""

    def test_returns_bearer_auth(self):
        adapter = _stub_adapter()
        headers = adapter.build_upstream_headers("sk-test-key-123")
        assert headers["Authorization"] == "Bearer sk-test-key-123"
        assert headers["Content-Type"] == "application/json"
        assert len(headers) == 2


class TestTranslateToUpstreamDefault:
    """Default translate_to_upstream returns the request unchanged."""

    def test_returns_same_dict(self):
        adapter = _stub_adapter()
        cc = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        result = adapter.translate_to_upstream(cc)
        assert result == cc  # same content — passthrough (filters internal metadata keys)

    def test_strips_internal_keys(self):
        """Internal metadata keys are stripped from the upstream body."""
        adapter = _stub_adapter()
        cc = {
            "model": "gpt-4o",
            "messages": [],
            "_reasoning_effort": "high",
            "_thinking_enabled": True,
            "_resolved_key": "sk-secret",
            "_provider_config": {"key": "val"},
            "_original_body": {"raw": "data"},
            "_native_messages_request": False,
        }
        result = adapter.translate_to_upstream(cc)
        assert "_reasoning_effort" not in result
        assert "_thinking_enabled" not in result
        assert "_resolved_key" not in result
        assert "_provider_config" not in result
        assert "_original_body" not in result
        assert "_native_messages_request" not in result
        assert result["model"] == "gpt-4o"

    def test_strips_base_url_defense_in_depth(self):
        """F15: base_url must never leak into the upstream body.

        The URL override is consumed by ``build_base_url()``/HTTP transport
        — it has no place in the JSON body.  The default
        ``translate_to_upstream`` strips it for any adapter that doesn't
        override ``translate_to_upstream``.
        """
        adapter = _stub_adapter()
        cc = {
            "model": "gpt-4o",
            "messages": [],
            "base_url": "https://attacker.example.com/v1",
        }
        result = adapter.translate_to_upstream(cc)
        assert "base_url" not in result
        assert result["model"] == "gpt-4o"


class TestTranslateFromUpstreamDefault:
    """Default translate_from_upstream returns the response unchanged."""

    def test_returns_same_dict(self):
        adapter = _stub_adapter()
        resp = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
        result = adapter.translate_from_upstream(resp)
        assert result is resp  # same object — passthrough


class TestTranslateUpstreamStreamEventDefault:
    """Default translate_upstream_stream_event returns the raw SSE event unchanged."""

    def test_returns_same_bytes(self):
        adapter = _stub_adapter()
        raw = b'data: {"choices":[]}\n\n'
        result = adapter.translate_upstream_stream_event(raw)
        assert result == [raw]  # wrapped in list, same bytes


class TestUseCustomTransportDefault:
    """Default use_custom_transport is False."""

    def test_default_is_false(self):
        adapter = _stub_adapter()
        assert adapter.use_custom_transport is False


class TestMakeRequestDefault:
    """Default make_request raises NotImplementedError."""

    @pytest.mark.asyncio
    async def test_raises(self):
        adapter = _stub_adapter()
        with pytest.raises(NotImplementedError):
            await adapter.make_request({"model": "test", "messages": []})


class TestStreamRequestDefault:
    """Default stream_request raises NotImplementedError."""

    @pytest.mark.asyncio
    async def test_raises(self):
        adapter = _stub_adapter()
        with pytest.raises(NotImplementedError):
            await adapter.stream_request({"model": "test", "messages": []}, lambda _: None)
