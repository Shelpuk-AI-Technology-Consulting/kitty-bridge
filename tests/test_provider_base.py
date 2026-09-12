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


class TestUpstreamWireIsMessagesApiForModelDefault:
    """The per-model wire-shape declaration defaults to the bare property.

    Guards the delegation KBR-7 introduces: an adapter that does not route by
    model must not have to override anything for the bridge's per-model reads
    to keep working.
    """

    def test_default_follows_the_property_for_any_model(self):
        adapter = _stub_adapter()
        assert adapter.upstream_wire_is_messages_api is False
        assert adapter.upstream_wire_is_messages_api_for_model("gpt-4o") is False
        assert adapter.upstream_wire_is_messages_api_for_model("") is False

    def test_follows_a_subclass_property_override(self):
        """A subclass declaring only the property is still answered correctly.

        The delegation is load-bearing for existing code, not only for a
        purpose-built double: ``CustomAnthropicAdapter`` declares the property
        and nothing else, and five tests in
        ``tests/bridge/test_thinking_roundtrip_failover.py`` go red if this
        method stops following it.
        """

        class _MessagesWireAdapter(type(_stub_adapter())):  # type: ignore[misc]  # concrete stub
            @property
            def upstream_wire_is_messages_api(self) -> bool:
                return True

        adapter = _MessagesWireAdapter()
        assert adapter.upstream_wire_is_messages_api_for_model("claude-opus-5") is True


class TestBuildUpstreamHeadersDefault:
    """Default build_upstream_headers returns Bearer auth."""

    def test_returns_bearer_auth(self):
        adapter = _stub_adapter()
        headers = adapter.build_upstream_headers("sk-test-key-123")
        assert headers["Authorization"] == "Bearer sk-test-key-123"
        assert headers["Content-Type"] == "application/json"
        assert len(headers) == 2


class TestBuildUpstreamHeadersForModelDefault:
    """The per-model header hook defaults to the model-independent one.

    Concrete on the base class rather than an optional hook the bridge finds
    with ``hasattr`` (KBR-127): an adapter that routes auth per model and
    forgets to define it would otherwise receive the default scheme in silence,
    which is the same shape of defect as resolving the route from the wrong
    model. The sibling ``upstream_wire_is_messages_api_for_model`` is concrete
    for the reason KBR-7 gives, and this follows it.
    """

    def test_default_ignores_the_model(self):
        adapter = _stub_adapter()
        for model in ("gpt-4o", "opencode/minimax-m2.5", ""):
            assert adapter.build_upstream_headers_for_model("sk-test-key-123", model) == adapter.build_upstream_headers(
                "sk-test-key-123"
            )

    def test_follows_a_subclass_header_override(self):
        """A subclass overriding only the model-independent form is still answered correctly.

        The delegation is what lets every adapter but OpenCode Go leave this
        alone; if it stopped following ``build_upstream_headers``, each of them
        would silently start sending Bearer auth.
        """

        class _XApiKeyAdapter(type(_stub_adapter())):  # type: ignore[misc]  # concrete stub
            def build_upstream_headers(self, api_key: str) -> dict[str, str]:
                return {"x-api-key": api_key}

        adapter = _XApiKeyAdapter()
        assert adapter.build_upstream_headers_for_model("sk-test", "any-model") == {"x-api-key": "sk-test"}


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

    def test_strips_internal_top_k(self):
        """KBR-178: `_top_k` is internal metadata and must never reach a provider.

        Chat Completions declares no ``top_k``, so the key exists only to carry
        the value to an Anthropic-family adapter.  Registering it here is what
        keeps it off the wire of the other twenty-two.
        """
        adapter = _stub_adapter()
        cc = {"model": "gpt-4o", "messages": [], "_top_k": 40}
        result = adapter.translate_to_upstream(cc)
        assert "_top_k" not in result
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
