"""Tests for per-request backend selection isolation via contextvars.

Concurrent aiohttp requests must not overwrite each other's
``_active_provider`` or ``_current_backend_idx`` when interleaved
across ``await`` points.
"""

from __future__ import annotations

import asyncio

import pytest

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.providers.base import ProviderAdapter


class _StubLauncher(LauncherAdapter):
    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self):
        from kitty.types import BridgeProtocol

        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(self, *args, **kwargs) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class _StubProvider(ProviderAdapter):
    @property
    def provider_type(self) -> str:
        return "stub"

    @property
    def default_base_url(self) -> str:
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages, "stream": False}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"err {status_code}")


def _make_server() -> BridgeServer:
    return BridgeServer(_StubLauncher(), _StubProvider(), "test-key")


class TestBackendContextIsolation:
    """Backend selection must survive across await points without cross-talk."""

    def test_select_backend_returns_context_dict(self):
        """_select_backend returns a dict with provider, key, model, idx."""
        server = _make_server()
        ctx = server._select_backend()
        assert isinstance(ctx, dict)
        assert "provider" in ctx
        assert "key" in ctx
        assert "model" in ctx
        assert "idx" in ctx
        assert ctx["provider"] is not None

    def test_select_backend_sets_context_var(self):
        """After _select_backend, the module-level context var is populated."""
        server = _make_server()
        from kitty.bridge.server import _backend_context

        ctx = server._select_backend()
        stored = _backend_context.get({})
        assert stored.get("provider") is ctx["provider"]

    @pytest.mark.asyncio
    async def test_context_survives_await(self):
        """Context var value persists across an await point."""
        server = _make_server()
        from kitty.bridge.server import _backend_context

        server._select_backend()
        ctx_before = _backend_context.get({})

        await asyncio.sleep(0.001)

        ctx_after = _backend_context.get({})
        assert ctx_after.get("idx") == ctx_before.get("idx")

    @pytest.mark.asyncio
    async def test_two_concurrent_tasks_isolated(self):
        """Two interleaved _select_backend calls produce isolated contexts."""
        server = _make_server()
        from kitty.bridge.server import _backend_context

        results = {}

        async def task_a():
            server._select_backend()
            await asyncio.sleep(0.003)
            results["a"] = _backend_context.get({})

        async def task_b():
            await asyncio.sleep(0.001)
            server._select_backend()
            results["b"] = _backend_context.get({})

        await asyncio.gather(task_a(), task_b())

        assert results["a"]["provider"] is not None
        assert results["b"]["provider"] is not None

    def test_properties_delegate_to_context_var(self):
        """self._active_provider reads from context var when set."""
        server = _make_server()
        ctx = server._select_backend()

        # Corrupting the instance dict should not affect reads
        # when the context var is set.
        server.__dict__["_provider"] = "CORRUPTED"
        assert server._active_provider is ctx["provider"]

    def test_current_backend_idx_reads_from_context(self):
        """self._current_backend_idx reads from context var."""
        server = _make_server()
        ctx = server._select_backend()

        server._backend_idx = 999
        assert server._current_backend_idx == ctx["idx"]


class TestProviderConfigContextIsolation:
    """_active_provider_config must be context-var-aware (same pattern as other properties)."""

    def test_select_backend_includes_provider_config_in_context(self):
        """_select_backend's result dict must include provider_config."""
        import uuid

        from kitty.profiles.schema import Profile

        profile = Profile(
            name="p1",
            provider="openai",
            model="m1",
            auth_ref=str(uuid.uuid4()),
            provider_config={"base_url": "https://example.com/v1", "region": "us-east"},
        )
        backends = [(_StubProvider(), "k1", profile)]
        server = BridgeServer(
            _StubLauncher(),
            _StubProvider(),
            "key-0",
            backends=backends,
            model="m0",
        )
        ctx = server._select_backend()
        assert ctx["provider_config"]["base_url"] == "https://example.com/v1"
        assert ctx["provider_config"]["region"] == "us-east"

    def test_active_provider_config_reads_from_context(self):
        """_active_provider_config returns the value from context var, not the instance field."""
        server = _make_server()
        server._select_backend()
        # Mutate the underlying dict to simulate concurrent overwrites
        server.__dict__["_provider_config"] = {"base_url": "CORRUPTED"}
        # The property should still return the value from the context var
        cfg = server._active_provider_config
        assert "CORRUPTED" not in cfg.get("base_url", "")

    @pytest.mark.asyncio
    async def test_concurrent_tasks_have_isolated_provider_config(self):
        """Two tasks selecting different backends see different provider_configs."""
        import uuid

        from kitty.profiles.schema import Profile

        profile_a = Profile(
            name="a",
            provider="openai",
            model="m",
            auth_ref=str(uuid.uuid4()),
            provider_config={"base_url": "https://backend-a.example.com"},
        )
        profile_b = Profile(
            name="b",
            provider="openai",
            model="m",
            auth_ref=str(uuid.uuid4()),
            provider_config={"base_url": "https://backend-b.example.com"},
        )
        backends = [
            (_StubProvider(), "k-a", profile_a),
            (_StubProvider(), "k-b", profile_b),
        ]
        server = BridgeServer(
            _StubLauncher(),
            _StubProvider(),
            "key",
            backends=backends,
            model="m",
        )

        results: dict = {}

        async def task(idx: int) -> None:
            server._select_backend()
            await asyncio.sleep(0.002)
            results[idx] = server._active_provider_config

        # Start tasks in order so they interleave
        t1 = asyncio.create_task(task(0))
        await asyncio.sleep(0.001)
        t2 = asyncio.create_task(task(1))
        await asyncio.gather(t1, t2)

        # Both must have valid configs from the context var
        assert results[0]
        assert results[1]
        # The two URLs must differ (proves each task saw its own selection)
        assert results[0].get("base_url") != results[1].get("base_url")
