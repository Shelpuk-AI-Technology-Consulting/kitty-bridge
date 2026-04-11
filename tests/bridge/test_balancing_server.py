"""Tests for bridge server round-robin balancing — backend selection per request."""

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol


class StubLauncher(LauncherAdapter):
    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.CHAT_COMPLETIONS_API

    def build_spawn_config(self, profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class StubProvider(ProviderAdapter):
    """Provider that records which URL/headers were used."""

    def __init__(self, provider_type: str = "stub", base_url: str = "https://api.example.com/v1"):
        self._provider_type = provider_type
        self._base_url = base_url

    @property
    def provider_type(self) -> str:
        return self._provider_type

    @property
    def default_base_url(self) -> str:
        return self._base_url

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Error {status_code}")


UPSTREAM_RESPONSE = {
    "id": "chatcmpl-1",
    "model": "test-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _make_backends(n: int):
    """Create n (provider, key, profile_dict) tuples for balancing tests."""
    import uuid

    from kitty.profiles.schema import Profile

    backends = []
    for i in range(n):
        provider = StubProvider(provider_type=f"stub-{i}", base_url=f"https://api{i}.example.com/v1")
        key = f"key-{i}"
        profile = Profile(name=f"profile-{i}", provider="openai", model=f"model-{i}", auth_ref=str(uuid.uuid4()))
        backends.append((provider, key, profile))
    return backends


class TestRoundRobinSelection:
    def test_single_backend_cycles_to_same(self):
        """With one backend, _get_next_backend always returns the same."""
        backends = _make_backends(1)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        first = server._get_next_backend()
        second = server._get_next_backend()
        assert first[1] == "key-0"
        assert second[1] == "key-0"

    def test_two_backends_round_robin(self):
        """With two backends, alternates between them."""
        backends = _make_backends(2)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        results = [server._get_next_backend() for _ in range(4)]
        keys = [r[1] for r in results]
        assert keys == ["key-0", "key-1", "key-0", "key-1"]

    def test_three_backends_round_robin(self):
        backends = _make_backends(3)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        results = [server._get_next_backend() for _ in range(6)]
        keys = [r[1] for r in results]
        assert keys == ["key-0", "key-1", "key-2", "key-0", "key-1", "key-2"]

    def test_no_backends_uses_single_profile(self):
        """When backends is None, falls back to single profile mode."""
        provider = StubProvider()
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=provider,
            resolved_key="single-key",
            model="single-model",
        )
        # _get_next_backend should return the single-profile tuple
        p, key, model, config = server._get_next_backend()
        assert key == "single-key"
        assert model == "single-model"

    def test_no_backends_no_model_returns_none(self):
        """When backends is None and no model set, returns None for model."""
        provider = StubProvider()
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=provider,
            resolved_key="single-key",
        )
        p, key, model, config = server._get_next_backend()
        assert key == "single-key"
        assert model is None


class TestRoundRobinIntegration:
    @pytest.mark.asyncio
    async def test_chat_completions_uses_round_robin(self):
        """Two chat completion requests should use different backends."""
        backends = _make_backends(2)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # Mock both upstream endpoints
            m.post("https://api0.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
            m.post("https://api1.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)

            async with aiohttp.ClientSession() as session:
                # First request → backend 0
                async with session.post(url, json=request_body) as resp:
                    assert resp.status == 200
                    await resp.json()

                # Second request → backend 1
                async with session.post(url, json=request_body) as resp:
                    assert resp.status == 200
                    await resp.json()

        await server.stop_async()
        # Verify both backends were called
        from yarl import URL

        requests_to_0 = list(m.requests.get(("POST", URL("https://api0.example.com/v1/chat/completions")), []))
        requests_to_1 = list(m.requests.get(("POST", URL("https://api1.example.com/v1/chat/completions")), []))
        assert len(requests_to_0) >= 1
        assert len(requests_to_1) >= 1

    @pytest.mark.asyncio
    async def test_chat_completions_single_backend(self):
        """Without backends, single profile mode works."""
        provider = StubProvider()
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=provider,
            resolved_key="my-key",
            model="my-model",
        )
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200

        await server.stop_async()


class TestProviderConfigPropagation:
    def test_provider_config_per_backend(self):
        """Each backend's provider_config is correctly set as _active_provider_config."""
        import uuid

        from kitty.profiles.schema import Profile

        backends = []
        for i in range(3):
            provider = StubProvider(provider_type=f"stub-{i}", base_url=f"https://api{i}.example.com/v1")
            profile = Profile(
                name=f"profile-{i}",
                provider="openai",
                model=f"model-{i}",
                auth_ref=str(uuid.uuid4()),
                provider_config={"custom_url": f"https://custom{i}.example.com"},
            )
            backends.append((provider, f"key-{i}", profile))

        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )

        # First backend → first provider_config
        server._select_backend()
        assert server._active_provider_config == {"custom_url": "https://custom0.example.com"}

        # Second backend → second provider_config
        server._select_backend()
        assert server._active_provider_config == {"custom_url": "https://custom1.example.com"}

        # Third backend → third provider_config
        server._select_backend()
        assert server._active_provider_config == {"custom_url": "https://custom2.example.com"}

    def test_single_profile_uses_init_provider_config(self):
        """Without backends, _active_provider_config uses the init parameter."""
        provider = StubProvider()
        config = {"base_url": "https://custom.example.com"}
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=provider,
            resolved_key="single-key",
            model="single-model",
            provider_config=config,
        )

        server._select_backend()
        assert server._active_provider_config == config
