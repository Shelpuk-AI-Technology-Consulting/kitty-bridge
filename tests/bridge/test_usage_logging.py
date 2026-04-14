"""Tests for LLM usage logging — --logging flag writes JSONL to ~/.cache/kitty/usage.log."""

import json
from datetime import datetime
from pathlib import Path

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# ── Stub adapters ────────────────────────────────────────────────────────────


class StubLauncher(LauncherAdapter):
    def __init__(self, protocol: BridgeProtocol = BridgeProtocol.MESSAGES_API):
        self._protocol = protocol

    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return self._protocol

    def build_spawn_config(self, profile: Profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class StubProvider(ProviderAdapter):
    @property
    def provider_type(self) -> str:
        return "stub"

    @property
    def default_base_url(self) -> str:
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Upstream error {status_code}: {body}")


# ── Helpers ──────────────────────────────────────────────────────────────────


UPSTREAM_RESPONSE = {
    "id": "chatcmpl-1",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
}

UPSTREAM_RESPONSE_NO_USAGE = {
    "id": "chatcmpl-2",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hi!"},
            "finish_reason": "stop",
        }
    ],
}


def _make_messages_request(stream: bool = False):
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1024,
        "stream": stream,
    }


def _make_responses_request(stream: bool = False):
    return {"model": "test-model", "input": [{"role": "user", "content": "hello"}], "stream": stream}


def _make_chat_completions_request(stream: bool = False):
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
    }


def _make_gemini_request():
    return {
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
    }


def _streaming_chunks():
    """Yield SSE-formatted streaming chunks with usage in the final chunk."""
    chunks = [
        (
            b'data: {"id":"chatcmpl-s","model":"test-model",'
            b'"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},'
            b'"finish_reason":null}]}\n\n'
        ),
        (
            b'data: {"id":"chatcmpl-s","model":"test-model",'
            b'"choices":[{"index":0,"delta":{"content":"!"},'
            b'"finish_reason":null}]}\n\n'
        ),
        (
            b'data: {"id":"chatcmpl-s","model":"test-model",'
            b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":200,"completion_tokens":20,'
            b'"total_tokens":220}}\n\n'
        ),
        b"data: [DONE]\n\n",
    ]
    return b"".join(chunks)


def _read_usage_log(log_path: Path) -> list[dict]:
    """Read and parse all JSONL entries from the usage log."""
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ── _log_usage unit tests ────────────────────────────────────────────────────


class TestLogUsageHelper:
    """Test the _log_usage method directly."""

    def test_writes_jsonl_entry(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            model="test-model",
            logging_enabled=True,
            _usage_log_path=log_path,
        )

        server._log_usage(
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

        entries = _read_usage_log(log_path)
        assert len(entries) == 1
        entry = entries[0]
        assert "timestamp" in entry
        assert entry["profile"] == "default"
        assert entry["provider"] == "stub"
        assert entry["model"] == "test-model"
        assert entry["input_tokens"] == 10
        assert entry["output_tokens"] == 5

    def test_multiple_calls_append(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            logging_enabled=True,
            _usage_log_path=log_path,
        )

        server._log_usage(usage={"prompt_tokens": 10, "completion_tokens": 5})
        server._log_usage(usage={"prompt_tokens": 20, "completion_tokens": 10})

        entries = _read_usage_log(log_path)
        assert len(entries) == 2
        assert entries[0]["input_tokens"] == 10
        assert entries[1]["input_tokens"] == 20

    def test_missing_usage_defaults_to_zero(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            logging_enabled=True,
            _usage_log_path=log_path,
        )

        server._log_usage(usage={})

        entries = _read_usage_log(log_path)
        assert len(entries) == 1
        assert entries[0]["input_tokens"] == 0
        assert entries[0]["output_tokens"] == 0

    def test_none_usage_defaults_to_zero(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            logging_enabled=True,
            _usage_log_path=log_path,
        )

        server._log_usage(usage=None)

        entries = _read_usage_log(log_path)
        assert entries[0]["input_tokens"] == 0
        assert entries[0]["output_tokens"] == 0

    def test_disabled_does_not_write(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            logging_enabled=False,
            _usage_log_path=log_path,
        )

        server._log_usage(usage={"prompt_tokens": 10, "completion_tokens": 5})

        assert not log_path.exists()

    def test_creates_directory_on_demand(self, tmp_path: Path):
        log_path = tmp_path / "subdir" / "usage.log"
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            logging_enabled=True,
            _usage_log_path=log_path,
        )

        server._log_usage(usage={"prompt_tokens": 1, "completion_tokens": 1})

        assert log_path.exists()

    def test_timestamp_is_iso8601_utc(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            logging_enabled=True,
            _usage_log_path=log_path,
        )

        server._log_usage(usage={"prompt_tokens": 1, "completion_tokens": 1})

        entries = _read_usage_log(log_path)
        ts = entries[0]["timestamp"]
        # Must be parseable as ISO 8601
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None

    def test_write_failure_does_not_raise(self, tmp_path: Path):
        """If the log file is not writable, _log_usage must not raise."""
        log_path = tmp_path / "readonly" / "usage.log"
        # Create directory but make it read-only
        log_path.parent.mkdir(parents=True)
        log_path.parent.chmod(0o444)

        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            logging_enabled=True,
            _usage_log_path=log_path,
        )

        # Must not raise
        server._log_usage(usage={"prompt_tokens": 1, "completion_tokens": 1})


# ── Non-streaming integration tests ─────────────────────────────────────────


class TestNonStreamingUsageLog:
    """Usage is logged for non-streaming LLM calls."""

    @pytest.mark.asyncio
    async def test_messages_api_non_streaming(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(
            adapter, provider, "test-key",
            model="test-model",
            logging_enabled=True,
            _usage_log_path=log_path,
        )
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json=_make_messages_request(stream=False),
                ) as resp:
                    assert resp.status == 200

            entries = _read_usage_log(log_path)
            assert len(entries) == 1
            assert entries[0]["input_tokens"] == 100
            assert entries[0]["output_tokens"] == 50
            assert entries[0]["provider"] == "stub"
            assert entries[0]["model"] == "test-model"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_responses_api_non_streaming(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(
            adapter, provider, "test-key",
            model="test-model",
            logging_enabled=True,
            _usage_log_path=log_path,
        )
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/responses",
                    json=_make_responses_request(stream=False),
                ) as resp:
                    assert resp.status == 200

            entries = _read_usage_log(log_path)
            assert len(entries) == 1
            assert entries[0]["input_tokens"] == 100
            assert entries[0]["output_tokens"] == 50
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_chat_completions_non_streaming(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        adapter = StubLauncher(BridgeProtocol.CHAT_COMPLETIONS_API)
        provider = StubProvider()
        server = BridgeServer(
            adapter, provider, "test-key",
            model="test-model",
            logging_enabled=True,
            _usage_log_path=log_path,
        )
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json=_make_chat_completions_request(stream=False),
                ) as resp:
                    assert resp.status == 200

            entries = _read_usage_log(log_path)
            assert len(entries) == 1
            assert entries[0]["input_tokens"] == 100
            assert entries[0]["output_tokens"] == 50
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_no_usage_block_defaults_to_zero(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(
            adapter, provider, "test-key",
            model="test-model",
            logging_enabled=True,
            _usage_log_path=log_path,
        )
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE_NO_USAGE)
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json=_make_messages_request(stream=False),
                ) as resp:
                    assert resp.status == 200

            entries = _read_usage_log(log_path)
            assert len(entries) == 1
            assert entries[0]["input_tokens"] == 0
            assert entries[0]["output_tokens"] == 0
        finally:
            await server.stop_async()


class TestNonStreamingLoggingDisabled:
    """No log written when logging_enabled=False."""

    @pytest.mark.asyncio
    async def test_no_log_when_disabled(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(
            adapter, provider, "test-key",
            model="test-model",
            logging_enabled=False,
            _usage_log_path=log_path,
        )
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json=_make_messages_request(stream=False),
                ) as resp:
                    assert resp.status == 200

            assert not log_path.exists()
        finally:
            await server.stop_async()


# ── Streaming integration tests ──────────────────────────────────────────────


class TestStreamingUsageLog:
    """Usage is logged for streaming LLM calls."""

    @pytest.mark.asyncio
    async def test_messages_api_streaming(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(
            adapter, provider, "test-key",
            model="test-model",
            logging_enabled=True,
            _usage_log_path=log_path,
        )
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body=_streaming_chunks(),
                    content_type="text/event-stream",
                )
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json=_make_messages_request(stream=True),
                ) as resp:
                    assert resp.status == 200
                    # Consume the stream
                    await resp.read()

            entries = _read_usage_log(log_path)
            assert len(entries) == 1
            assert entries[0]["input_tokens"] == 200
            assert entries[0]["output_tokens"] == 20
            assert entries[0]["provider"] == "stub"
            assert entries[0]["model"] == "test-model"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_responses_api_streaming(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(
            adapter, provider, "test-key",
            model="test-model",
            logging_enabled=True,
            _usage_log_path=log_path,
        )
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body=_streaming_chunks(),
                    content_type="text/event-stream",
                )
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/responses",
                    json=_make_responses_request(stream=True),
                ) as resp:
                    assert resp.status == 200
                    await resp.read()

            entries = _read_usage_log(log_path)
            assert len(entries) == 1
            assert entries[0]["input_tokens"] == 200
            assert entries[0]["output_tokens"] == 20
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_chat_completions_streaming(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        adapter = StubLauncher(BridgeProtocol.CHAT_COMPLETIONS_API)
        provider = StubProvider()
        server = BridgeServer(
            adapter, provider, "test-key",
            model="test-model",
            logging_enabled=True,
            _usage_log_path=log_path,
        )
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body=_streaming_chunks(),
                    content_type="text/event-stream",
                )
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json=_make_chat_completions_request(stream=True),
                ) as resp:
                    assert resp.status == 200
                    await resp.read()

            entries = _read_usage_log(log_path)
            assert len(entries) == 1
            assert entries[0]["input_tokens"] == 200
            assert entries[0]["output_tokens"] == 20
        finally:
            await server.stop_async()


class TestStreamingLoggingDisabled:
    """No log written when logging_enabled=False for streaming requests."""

    @pytest.mark.asyncio
    async def test_no_log_when_disabled_streaming(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(
            adapter, provider, "test-key",
            model="test-model",
            logging_enabled=False,
            _usage_log_path=log_path,
        )
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body=_streaming_chunks(),
                    content_type="text/event-stream",
                )
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json=_make_messages_request(stream=True),
                ) as resp:
                    assert resp.status == 200
                    await resp.read()

            assert not log_path.exists()
        finally:
            await server.stop_async()


# ── Profile name tracking ────────────────────────────────────────────────────


class TestProfileNameInLog:
    """Profile name is correctly logged."""

    @pytest.mark.asyncio
    async def test_profile_name_appears_in_log(self, tmp_path: Path):
        log_path = tmp_path / "usage.log"
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(
            adapter, provider, "test-key",
            model="test-model",
            logging_enabled=True,
            profile_name="my-dev-profile",
            _usage_log_path=log_path,
        )
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json=_make_messages_request(stream=False),
                ) as resp:
                    assert resp.status == 200

            entries = _read_usage_log(log_path)
            assert entries[0]["profile"] == "my-dev-profile"
        finally:
            await server.stop_async()
