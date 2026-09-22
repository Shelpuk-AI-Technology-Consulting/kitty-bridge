"""Tests for OllamaCloudAdapter."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kitty.providers.base import ProviderError
from kitty.providers.ollama_cloud import OllamaCloudAdapter

# ── Properties ───────────────────────────────────────────────────────────────


class TestOllamaCloudAdapterProperties:
    """Test basic adapter properties."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "ollama_cloud"

    def test_default_base_url(self):
        assert self.adapter.default_base_url == "https://ollama.com"

    def test_upstream_path(self):
        assert self.adapter.upstream_path == "/api/chat"

    def test_use_custom_transport(self):
        assert self.adapter.use_custom_transport is True


# ── Headers ──────────────────────────────────────────────────────────────────


class TestOllamaCloudHeaders:
    """Test upstream header construction."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_bearer_auth(self):
        headers = self.adapter.build_upstream_headers("sk-test-key")
        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["Content-Type"] == "application/json"

    def test_different_keys(self):
        h1 = self.adapter.build_upstream_headers("key-a")
        h2 = self.adapter.build_upstream_headers("key-b")
        assert h1["Authorization"] != h2["Authorization"]


# ── Model normalization ─────────────────────────────────────────────────────


class TestOllamaCloudNormalizeModelName:
    """Test model name normalization."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_strips_ollama_cloud_prefix(self):
        assert self.adapter.normalize_model_name("ollama_cloud/gpt-oss:120b") == "gpt-oss:120b"

    def test_strips_ollama_prefix(self):
        assert self.adapter.normalize_model_name("ollama/gpt-oss:120b") == "gpt-oss:120b"

    def test_no_prefix_passthrough(self):
        assert self.adapter.normalize_model_name("gpt-oss:120b") == "gpt-oss:120b"

    def test_slash_model_without_known_prefix_preserved(self):
        assert self.adapter.normalize_model_name("library/qwen:7b") == "library/qwen:7b"

    def test_colon_model_name_preserved(self):
        assert self.adapter.normalize_model_name("qwen3:8b") == "qwen3:8b"


# ── translate_to_upstream ───────────────────────────────────────────────────


class TestOllamaCloudTranslateToUpstream:
    """Test CC → Ollama request translation."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_basic_request(self):
        cc = {"model": "gpt-oss:120b", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        result = self.adapter.translate_to_upstream(cc)
        assert result["model"] == "gpt-oss:120b"
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        assert result["stream"] is False

    def test_system_message_extracted(self):
        cc = {
            "model": "gpt-oss:120b",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert any(m["role"] == "system" for m in result["messages"])

    def test_strips_internal_keys(self):
        cc = {
            "model": "gpt-oss:120b",
            "messages": [],
            "_resolved_key": "secret",
            "_provider_config": {},
            "_thinking_enabled": True,
            "_reasoning_effort": "high",
        }
        result = self.adapter.translate_to_upstream(cc)
        assert "_resolved_key" not in result
        assert "_provider_config" not in result
        assert "_thinking_enabled" not in result
        assert "_reasoning_effort" not in result

    def test_tools_forwarded(self):
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        cc = {"model": "gpt-oss:120b", "messages": [], "tools": tools}
        result = self.adapter.translate_to_upstream(cc)
        assert result["tools"] == tools

    def test_tool_result_message_translated(self):
        """CC tool results use tool_call_id; Ollama uses tool_name."""
        cc = {
            "model": "gpt-oss:120b",
            "messages": [
                {"role": "tool", "tool_call_id": "call_123", "name": "get_weather", "content": "sunny"},
            ],
        }
        result = self.adapter.translate_to_upstream(cc)
        tool_msg = result["messages"][0]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_name"] == "get_weather"
        assert tool_msg["content"] == "sunny"
        assert "tool_call_id" not in tool_msg

    def test_assistant_tool_calls_translated(self):
        """CC assistant tool_calls: arguments JSON string → Ollama dict."""
        cc = {
            "model": "gpt-oss:120b",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                        }
                    ],
                }
            ],
        }
        result = self.adapter.translate_to_upstream(cc)
        assistant_msg = result["messages"][0]
        assert assistant_msg["role"] == "assistant"
        # Arguments should be converted from JSON string to dict for Ollama
        tc = assistant_msg["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == {"city": "NYC"}

    def test_assistant_tool_calls_dict_args_preserved(self):
        """If arguments are already a dict, they pass through unchanged."""
        cc = {
            "model": "gpt-oss:120b",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "calc", "arguments": {"expr": "1+1"}},
                        }
                    ],
                }
            ],
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"expr": "1+1"}

    def test_options_from_temperature(self):
        """CC temperature maps to Ollama options.temperature."""
        cc = {"model": "gpt-oss:120b", "messages": [], "temperature": 0.7}
        result = self.adapter.translate_to_upstream(cc)
        assert result["options"]["temperature"] == 0.7

    def test_stop_mapped_to_options_stop(self):
        """KBR-178: the CC `stop` reaches Ollama as `options.stop`."""
        cc = {"model": "gpt-oss:120b", "messages": [], "stop": ["A", "B"]}
        result = self.adapter.translate_to_upstream(cc)
        assert result["options"]["stop"] == ["A", "B"]
        assert "stop" not in result

    def test_null_stop_creates_no_options_entry(self):
        """`stop: null` must not attach an `options` container carrying None.

        ``options`` is attached only when non-empty, so an unconditional write
        would put ``{"stop": None}`` on every request without stop sequences.
        See D6.
        """
        cc = {"model": "gpt-oss:120b", "messages": [], "stop": None}
        result = self.adapter.translate_to_upstream(cc)
        assert "options" not in result

    def test_empty_stop_creates_no_options_entry(self):
        """`stop: []` must not attach an `options` container either — see D6."""
        cc = {"model": "gpt-oss:120b", "messages": [], "stop": []}
        result = self.adapter.translate_to_upstream(cc)
        assert "options" not in result

    def test_options_empty_when_no_extras(self):
        cc = {"model": "gpt-oss:120b", "messages": []}
        result = self.adapter.translate_to_upstream(cc)
        assert "options" not in result

    def test_content_blocks_flattened_to_string(self):
        """CC content as list of text blocks → Ollama plain string."""
        cc = {
            "model": "gpt-oss:120b",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]},
            ],
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["messages"][0]["content"] == "hello\nworld"

    def test_none_content_becomes_empty_string(self):
        """CC content None → Ollama empty string (not None)."""
        cc = {
            "model": "gpt-oss:120b",
            "messages": [{"role": "assistant", "content": None, "tool_calls": []}],
        }
        result = self.adapter.translate_to_upstream(cc)
        # content is set only if not None, so it should be absent
        assert "content" not in result["messages"][0] or result["messages"][0].get("content") == ""


# ── _ollama_body (KBR-90 / T-H5, register row P19) ──────────────────────────


class TestOllamaCloudBody:
    """L1 — ``OllamaCloudAdapter._ollama_body`` is the pure payload builder (P19).

    Register row P19 ("Overwrite ``stream``") lives inside
    ``make_request`` / ``stream_request`` today, so ``mutmut`` cannot reach
    it from the L1 selection. ``_ollama_body`` extracts the body's
    translation plus the ``stream`` overwrite into a pure function; this
    class is what makes P19 a mutation-testable surface. The wire-capture
    characterisation of P19 itself ships at L2 in
    ``tests/test_wire_shape_honesty_wire.py`` (both halves of the ollama
    boundary are captured — the KBR-90 scope add closed the streaming-half
    stated limit flagged in KBR-80).
    """

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    # ── R1: pure builder — return shape ──────────────────────────────────

    def test_returns_body_dict_with_stream_false_for_make_request(self) -> None:
        """R1 / AC1 — ``make_request``'s builder sets ``stream`` to False.

        The function returns a dict (no ``(model_id, body)`` tuple like
        Bedrock — Ollama has no separate model-id argument). The probe
        carries a ``stream: True`` so the overwrite is observable, not
        just a set-on-absent.
        """
        cc = {
            "model": "gpt-oss:120b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        result = self.adapter._ollama_body(cc, streaming=False)
        assert isinstance(result, dict)
        assert result["stream"] is False

    def test_returns_body_dict_with_stream_true_for_stream_request(self) -> None:
        """R1 / AC1 — ``stream_request``'s builder sets ``stream`` to True."""
        cc = {
            "model": "gpt-oss:120b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        result = self.adapter._ollama_body(cc, streaming=True)
        assert isinstance(result, dict)
        assert result["stream"] is True

    # ── R1: pure builder — P19 overwrite is REAL, not vacuously true ────

    def test_overwrites_stream_even_when_translate_emits_it(self) -> None:
        """R1 / AC4 — the ``stream`` overwrite runs against an injected body.

        Unlike bedrock's defensive pop (which today is vacuously true —
        ``translate_to_upstream`` never emits ``stream``), the ollama P19
        overwrite is what the transport really applies: a translation that
        already set ``stream`` must still be overridden. A mutation that
        drops the assignment (or rewrites ``streaming`` to something that
        evaluates to the injected value) leaves ``stream`` at the sentinel
        and this test fails.

        Two separate patch contexts, deliberately: the builder mutates and
        returns the translate output dict, so a single ``return_value=``
        context would alias both calls to one dict and the second call's
        ``stream=True`` would leak into the first call's captured body.
        """
        injected = {
            "model": "gpt-oss:120b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": "sentinel-was-not-overwritten",
        }
        with patch.object(self.adapter, "translate_to_upstream", return_value=dict(injected)):
            body_false = self.adapter._ollama_body(
                {"model": "irrelevant", "messages": [{"role": "user", "content": "x"}]},
                streaming=False,
            )
        with patch.object(self.adapter, "translate_to_upstream", return_value=dict(injected)):
            body_true = self.adapter._ollama_body(
                {"model": "irrelevant", "messages": [{"role": "user", "content": "x"}]},
                streaming=True,
            )
        assert body_false["stream"] is False, "the P19 overwrite ran for streaming=False"
        assert body_true["stream"] is True, "the P19 overwrite ran for streaming=True"

    # ── R1: pure builder — no IO ─────────────────────────────────────────

    def test_does_not_open_an_aiohttp_session(self) -> None:
        """R1 / AC2 — the builder stays pure (no session, no network)."""
        cc = {
            "model": "gpt-oss:120b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        with patch.object(
            self.adapter,
            "_get_session",
            side_effect=AssertionError(
                "_ollama_body must stay pure; it must not open an aiohttp session"
            ),
        ):
            self.adapter._ollama_body(cc, streaming=False)
            self.adapter._ollama_body(cc, streaming=True)

    # ── R2: body equals translate output with stream overridden ──────────

    @pytest.mark.parametrize(
        "cc",
        [
            pytest.param(
                {
                    "model": "gpt-oss:120b",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False,
                },
                id="minimal",
            ),
            pytest.param(
                {
                    "model": "qwen3-coder-next",
                    "messages": [
                        {"role": "system", "content": "You are concise."},
                        {"role": "user", "content": "Hello"},
                    ],
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 256,
                    "stop": "END",
                },
                id="system-and-options-and-max-tokens",
            ),
            pytest.param(
                {
                    "model": "gpt-oss:120b",
                    "messages": [
                        {"role": "user", "content": "What's the weather?"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "London"}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_abc",
                            "content": "15°C",
                        },
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                },
                            },
                        }
                    ],
                    "stream": True,
                },
                id="tools-and-tool-call-round-trip",
            ),
        ],
    )
    @pytest.mark.parametrize("streaming", [False, True])
    def test_body_matches_translate_to_upstream_with_stream_overridden(
        self, cc: dict, streaming: bool
    ) -> None:
        """R2 / AC3 — body keys equal ``translate_to_upstream(cc)`` with
        ``stream`` replaced by the flag.

        The hook API is unchanged; the builder is the same body with P19's
        ``stream`` overwrite applied. Parametrised over a representative
        minimal request, a system+sampling-request, and a
        tools+assistant-tool-call round-trip, and over both streaming modes.
        """
        translated = self.adapter.translate_to_upstream(cc)
        body = self.adapter._ollama_body(cc, streaming=streaming)

        expected = {**translated, "stream": streaming}
        assert body == expected
        assert body["stream"] is streaming

    # ── R3: transports consume the builder's output ─────────────────────

    @pytest.mark.asyncio
    async def test_make_request_posts_builder_body_verbatim(self) -> None:
        """R3 / AC3 — ``make_request`` posts the builder's dict verbatim.

        Sentinel body patched at the builder seam: the transport adds no
        further body mutation and passes the dict through. Using a sentinel
        rather than a monkeypatched counter on ``translate_to_upstream``
        because the builder calls ``translate_to_upstream`` internally —
        the load-bearing seam is the builder's, not the hook's.
        """
        adapter = self.adapter
        cc = {
            "model": "gpt-oss:120b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "_resolved_key": "test-key",
            "_provider_config": {},
        }
        sentinel_body = {
            "model": "sentinel-model",
            "messages": [{"role": "user", "content": "sentinel"}],
            "stream": False,
        }
        sentinel_response = {
            "model": "gpt-oss:120b",
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 1,
        }
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=sentinel_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        with (
            patch.object(adapter, "_ollama_body", return_value=sentinel_body),
            patch.object(adapter, "_get_session", return_value=mock_session),
        ):
            await adapter.make_request(cc)
        kwargs = mock_session.post.call_args.kwargs
        assert kwargs["json"] is sentinel_body, (
            "the transport posted the builder's body verbatim (identity, not equality)"
        )
        assert kwargs["json"]["stream"] is False

    @pytest.mark.asyncio
    async def test_stream_request_posts_builder_body_verbatim(self) -> None:
        """R3 / AC3 — ``stream_request`` posts the builder's dict verbatim.

        Mirrors the non-streaming sibling. ``resp.content`` is stubbed
        with an empty async iterator so the streaming-half error-recovery
        path is not exercised; the assertion is on ``session.post``'s
        ``json=`` kwarg.
        """
        adapter = self.adapter
        cc = {
            "model": "gpt-oss:120b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "_resolved_key": "test-key",
            "_provider_config": {},
        }
        sentinel_body = {
            "model": "sentinel-model",
            "messages": [{"role": "user", "content": "sentinel"}],
            "stream": True,
        }

        async def mock_aiter():
            for chunk in []:  # pragma: no cover — empty by construction
                yield chunk

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content = mock_aiter()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        with (
            patch.object(adapter, "_ollama_body", return_value=sentinel_body),
            patch.object(adapter, "_get_session", return_value=mock_session),
        ):
            await adapter.stream_request(cc, AsyncMock())
        kwargs = mock_session.post.call_args.kwargs
        assert kwargs["json"] is sentinel_body, (
            "the transport posted the builder's body verbatim (identity, not equality)"
        )
        assert kwargs["json"]["stream"] is True


# ── translate_from_upstream ─────────────────────────────────────────────────


class TestOllamaCloudTranslateFromUpstream:
    """Test Ollama response → CC response translation."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_text_response(self):
        ollama_resp = {
            "model": "gpt-oss:120b",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        result = self.adapter.translate_from_upstream(ollama_resp)
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_usage_mapping(self):
        ollama_resp = {
            "model": "gpt-oss:120b",
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 50,
        }
        result = self.adapter.translate_from_upstream(ollama_resp)
        usage = result["usage"]
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_tool_calls_in_response(self):
        ollama_resp = {
            "model": "gpt-oss:120b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "NYC"}},
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
        }
        result = self.adapter.translate_from_upstream(ollama_resp)
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"

    def test_done_reason_length(self):
        ollama_resp = {
            "model": "gpt-oss:120b",
            "message": {"role": "assistant", "content": "cut off"},
            "done": True,
            "done_reason": "length",
        }
        result = self.adapter.translate_from_upstream(ollama_resp)
        assert result["choices"][0]["finish_reason"] == "length"

    def test_missing_usage_defaults_to_zero(self):
        ollama_resp = {
            "model": "gpt-oss:120b",
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
            "done_reason": "stop",
        }
        result = self.adapter.translate_from_upstream(ollama_resp)
        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 0

    def test_cc_response_structure(self):
        ollama_resp = {
            "model": "gpt-oss:120b",
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
            "done_reason": "stop",
        }
        result = self.adapter.translate_from_upstream(ollama_resp)
        assert result["object"] == "chat.completion"
        assert result["model"] == "gpt-oss:120b"
        assert "id" in result


# ── map_error ────────────────────────────────────────────────────────────────


class TestOllamaCloudMapError:
    """Test error mapping."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_dict_error(self):
        err = self.adapter.map_error(400, {"error": "model not found"})
        assert isinstance(err, ProviderError)
        assert "400" in str(err)
        assert "model not found" in str(err)

    def test_429_error(self):
        err = self.adapter.map_error(429, {"error": "too many requests"})
        assert isinstance(err, ProviderError)
        assert "429" in str(err)

    def test_500_error(self):
        err = self.adapter.map_error(500, {"error": "internal error"})
        assert isinstance(err, ProviderError)
        assert "500" in str(err)

    def test_non_dict_body(self):
        err = self.adapter.map_error(400, "bad request")
        assert isinstance(err, ProviderError)
        assert "bad request" in str(err)

    def test_nested_error_object(self):
        err = self.adapter.map_error(401, {"error": {"message": "invalid API key"}})
        assert isinstance(err, ProviderError)
        assert "invalid API key" in str(err)

    def test_http_status_set(self):
        err = self.adapter.map_error(429, {"error": "rate limited"})
        assert err.http_status == 429


# ── build_request / parse_response ──────────────────────────────────────────


class TestOllamaCloudBuildRequest:
    """Test build_request returns CC-format dict."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_basic(self):
        req = self.adapter.build_request("gpt-oss:120b", [{"role": "user", "content": "hi"}])
        assert req["model"] == "gpt-oss:120b"
        assert req["messages"] == [{"role": "user", "content": "hi"}]

    def test_with_stream(self):
        req = self.adapter.build_request("gpt-oss:120b", [], stream=True)
        assert req["stream"] is True

    def test_with_tools(self):
        tools = [{"type": "function", "function": {"name": "test"}}]
        req = self.adapter.build_request("gpt-oss:120b", [], tools=tools)
        assert req["tools"] == tools


class TestOllamaCloudParseResponse:
    """Test parse_response returns normalized CC dict."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_basic(self):
        resp = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = self.adapter.parse_response(resp)
        assert result["content"] == "hi"
        assert result["finish_reason"] == "stop"


# ── make_request (custom transport) ──────────────────────────────────────────


class TestOllamaCloudMakeRequest:
    """Test non-streaming custom transport."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    @pytest.mark.asyncio
    async def test_non_streaming_call(self):
        """make_request translates CC → Ollama, calls HTTP, translates back."""
        cc_request = {
            "model": "gpt-oss:120b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": "test-key",
            "_provider_config": {},
        }

        ollama_response = {
            "model": "gpt-oss:120b",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=ollama_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch.object(self.adapter, "_get_session", return_value=mock_session):
            result = await self.adapter.make_request(cc_request)

        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_upstream_error_raises(self):
        """make_request raises ProviderError on upstream failure."""
        cc_request = {
            "model": "gpt-oss:120b",
            "messages": [],
            "_resolved_key": "bad-key",
            "_provider_config": {},
        }

        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.json = AsyncMock(return_value={"error": "invalid API key"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with (
            patch.object(self.adapter, "_get_session", return_value=mock_session),
            pytest.raises(ProviderError, match="401"),
        ):
            await self.adapter.make_request(cc_request)


# ── stream_request (custom transport) ────────────────────────────────────────


class TestOllamaCloudStreamRequest:
    """Test streaming custom transport: NDJSON → CC SSE."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    @pytest.mark.asyncio
    async def test_streaming_yields_sse_events(self):
        """stream_request reads NDJSON lines and writes CC SSE chunks."""
        cc_request = {
            "model": "gpt-oss:120b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": "test-key",
            "_provider_config": {},
        }

        ndjson_lines = [
            b'{"model":"gpt-oss:120b","message":{"role":"assistant","content":"Hel"},"done":false}\n',
            b'{"model":"gpt-oss:120b","message":{"role":"assistant","content":"lo!"},"done":false}\n',
            b'{"model":"gpt-oss:120b","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":5}\n',
        ]

        async def mock_aiter():
            for line in ndjson_lines:
                yield line

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content = mock_aiter()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        written_chunks: list[bytes] = []

        async def mock_write(data: bytes):
            written_chunks.append(data)

        with patch.object(self.adapter, "_get_session", return_value=mock_session):
            await self.adapter.stream_request(cc_request, mock_write)

        # Should have written SSE chunks
        assert len(written_chunks) > 0

        # Parse written SSE data
        all_data = b"".join(written_chunks).decode()
        assert "data:" in all_data
        assert "[DONE]" in all_data

        # Verify text content made it through
        sse_events = [line for line in all_data.split("\n") if line.startswith("data:")]
        contents = []
        for event in sse_events:
            data_str = event[5:].strip()
            if data_str == "[DONE]":
                continue
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta and delta["content"]:
                contents.append(delta["content"])
        assert "Hel" in contents
        assert "lo!" in contents

    @pytest.mark.asyncio
    async def test_streaming_upstream_error(self):
        """stream_request raises ProviderError on upstream HTTP error."""
        cc_request = {
            "model": "gpt-oss:120b",
            "messages": [],
            "_resolved_key": "key",
            "_provider_config": {},
        }

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.json = AsyncMock(return_value={"error": "internal server error"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with (
            patch.object(self.adapter, "_get_session", return_value=mock_session),
            pytest.raises(ProviderError, match="500"),
        ):
            await self.adapter.stream_request(cc_request, AsyncMock())

    @pytest.mark.asyncio
    async def test_streaming_tool_calls(self):
        """stream_request translates Ollama tool_calls in NDJSON chunks to CC SSE."""
        cc_request = {
            "model": "qwen3-coder-next",
            "messages": [{"role": "user", "content": "2+2"}],
            "stream": True,
            "_resolved_key": "test-key",
            "_provider_config": {},
        }

        ndjson_lines = [
            b'{"model":"qwen3-coder-next","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_abc","function":{"index":0,"name":"calc","arguments":{"expr":"2+2"}}}]},"done":false}\n',
            b'{"model":"qwen3-coder-next","message":{"role":"assistant","content":""},"done":false}\n',
            b'{"model":"qwen3-coder-next","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}\n',
        ]

        async def mock_aiter():
            for line in ndjson_lines:
                yield line

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content = mock_aiter()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        written_chunks: list[bytes] = []

        async def mock_write(data: bytes):
            written_chunks.append(data)

        with patch.object(self.adapter, "_get_session", return_value=mock_session):
            await self.adapter.stream_request(cc_request, mock_write)

        all_data = b"".join(written_chunks).decode()
        # Verify tool_calls appear in SSE output
        sse_events = [line for line in all_data.split("\n") if line.startswith("data:")]
        found_tool_calls = False
        for event in sse_events:
            data_str = event[5:].strip()
            if data_str == "[DONE]":
                continue
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "tool_calls" in delta:
                found_tool_calls = True
                assert delta["tool_calls"][0]["function"]["name"] == "calc"
        assert found_tool_calls, "Expected tool_calls in streaming SSE output"


# ── parse_stream_to_cc_response ──────────────────────────────────────────────


class TestOllamaCloudParseStream:
    """Test CC SSE stream → CC response dict parsing."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_text_stream(self):
        raw = (
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"qwen3-coder-next",'
            b'"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"qwen3-coder-next",'
            b'"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"qwen3-coder-next",'
            b'"choices":[{"index":0,"delta":{"content":"lo!"},"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"qwen3-coder-next",'
            b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        result = self.adapter.parse_stream_to_cc_response(raw)
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["model"] == "qwen3-coder-next"

    def test_empty_stream(self):
        raw = b"data: [DONE]\n\n"
        result = self.adapter.parse_stream_to_cc_response(raw)
        assert result["choices"][0]["message"]["content"] is None

    def test_tool_calls_in_stream(self):
        raw = (
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"m",'
            b'"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","type":"function",'
            b'"function":{"name":"test","arguments":"{}"}}]},"finish_reason":null}]}\n\n'
            b"data: [DONE]\n\n"
        )
        result = self.adapter.parse_stream_to_cc_response(raw)
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1


# ── Registry ─────────────────────────────────────────────────────────────────


class TestOllamaCloudRegistry:
    """Test provider registry integration."""

    def test_registered(self):
        from kitty.providers.registry import get_provider

        adapter = get_provider("ollama_cloud")
        assert isinstance(adapter, OllamaCloudAdapter)

    def test_registry_key(self):
        from kitty.providers.registry import _registry

        assert "ollama_cloud" in _registry


# ── Session release (KBR-190) ────────────────────────────────────────────────


class TestACloseReleasesTheSession:
    """KBR-190 — the bridge closes the session this adapter owns."""

    async def test_it_closes_and_clears_a_built_session(self):
        """Both halves matter: `closed` is the leak, `None` is the recovery."""
        adapter = OllamaCloudAdapter()
        session = await adapter._get_session()

        await adapter.aclose()

        assert session.closed
        assert adapter._session is None

    async def test_it_is_a_no_op_when_no_session_was_built(self):
        """An adapter that never served a request is closed like any other."""
        adapter = OllamaCloudAdapter()

        await adapter.aclose()

        assert adapter._session is None

    async def test_it_is_idempotent(self):
        """`start_async`'s state-write failure path can reach `stop_async` twice."""
        adapter = OllamaCloudAdapter()
        await adapter._get_session()

        await adapter.aclose()
        await adapter.aclose()

        assert adapter._session is None

    async def test_the_next_request_builds_a_fresh_session(self):
        """A closed adapter still works — one instance can outlive one bridge."""
        adapter = OllamaCloudAdapter()
        first = await adapter._get_session()
        await adapter.aclose()

        second = await adapter._get_session()

        assert second is not first
        assert not second.closed
        await adapter.aclose()

    async def test_the_attribute_is_cleared_even_when_the_close_fails(self):
        """A failing close must not strand a dead session on the adapter.

        ``stop_async`` logs and contains a provider's teardown failure, so an
        attribute still pointing at a half-closed session would be handed back
        for the rest of the process's life, with one WARNING as the only trace.
        """
        adapter = OllamaCloudAdapter()
        session = await adapter._get_session()

        with (
            patch.object(session, "close", new=AsyncMock(side_effect=RuntimeError("close failed"))),
            pytest.raises(RuntimeError, match="close failed"),
        ):
            await adapter.aclose()

        assert adapter._session is None
        await session.close()
