"""Tests for bridge mode — standalone OpenAI-compatible API server."""

from __future__ import annotations

import uuid

import aiohttp
import pytest

from kitty.cli.router import BuiltinCommand, CLIRouter, RouteResult
from kitty.launchers.base import LauncherAdapter
from kitty.launchers.claude import ClaudeAdapter
from kitty.launchers.codex import CodexAdapter
from kitty.profiles.schema import Profile
from kitty.profiles.store import ProfileStore


# -- Fixtures -----------------------------------------------------------------


def _make_profile(
    name: str = "test-profile",
    provider: str = "zai_regular",
    model: str = "gpt-4o",
    is_default: bool = False,
) -> Profile:
    return Profile(
        name=name,
        provider=provider,
        model=model,
        auth_ref=str(uuid.uuid4()),
        is_default=is_default,
    )


@pytest.fixture()
def adapters() -> dict[str, LauncherAdapter]:
    return {"codex": CodexAdapter(), "claude": ClaudeAdapter()}


@pytest.fixture()
def populated_store(tmp_path: object) -> ProfileStore:
    store = ProfileStore(path=tmp_path / "profiles.json")  # type: ignore[arg-type]
    store.save(_make_profile("dev", is_default=True))
    store.save(_make_profile("glm", provider="zai_coding", is_default=False))
    return store


# -- Router Tests --------------------------------------------------------------


class TestBridgeBuiltinRouting:
    """Test routing of bridge builtin command."""

    def test_bridge_routes_to_builtin(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        """"bridge" should route to BuiltinCommand.BRIDGE."""
        router = CLIRouter(populated_store, adapters)
        result = router.route(["bridge"])
        assert result.builtin == BuiltinCommand.BRIDGE
        assert result.profile is not None
        assert result.profile.is_default is True  # uses default profile

    def test_bridge_case_insensitive(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        """Bridge builtin is case-insensitive."""
        router = CLIRouter(populated_store, adapters)
        result = router.route(["BRIDGE"])
        assert result.builtin == BuiltinCommand.BRIDGE

    def test_profile_bridge_routes_to_bridge_with_profile(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        """<profile> bridge should route to bridge with that profile."""
        router = CLIRouter(populated_store, adapters)
        result = router.route(["glm", "bridge"])
        assert result.builtin == BuiltinCommand.BRIDGE
        assert result.profile is not None
        assert result.profile.name == "glm"
        assert result.profile.provider == "zai_coding"

    def test_bridge_with_extra_args(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        """Bridge should pass through extra args."""
        router = CLIRouter(populated_store, adapters)
        result = router.route(["bridge", "--port", "8080"])
        assert result.builtin == BuiltinCommand.BRIDGE
        assert "--port" in result.extra_args
        assert "8080" in result.extra_args

    def test_profile_bridge_with_extra_args(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        """<profile> bridge should pass through extra args."""
        router = CLIRouter(populated_store, adapters)
        result = router.route(["glm", "bridge", "--port", "9000"])
        assert result.builtin == BuiltinCommand.BRIDGE
        assert result.profile is not None
        assert result.profile.name == "glm"
        assert "--port" in result.extra_args
        assert "9000" in result.extra_args


class TestBridgeBuiltinCommand:
    """Test BuiltinCommand enum includes BRIDGE."""

    def test_bridge_in_enum(self) -> None:
        """BuiltinCommand should include BRIDGE."""
        assert hasattr(BuiltinCommand, "BRIDGE")
        assert BuiltinCommand.BRIDGE == "bridge"

    def test_bridge_is_str_subclass(self) -> None:
        """BuiltinCommand.BRIDGE should be a string."""
        assert isinstance(BuiltinCommand.BRIDGE, str)


# -- Bridge Execution Tests ----------------------------------------------------


class TestBridgeExecution:
    """Test bridge mode execution (starting the server)."""

    @pytest.mark.asyncio
    async def test_bridge_starts_server(self) -> None:
        """Bridge mode should start BridgeServer."""
        from kitty.bridge.server import BridgeServer
        from kitty.providers.zai import ZaiRegularAdapter

        profile = _make_profile("test", provider="zai_regular")
        provider = ZaiRegularAdapter()
        resolved_key = "test-api-key"

        server = BridgeServer(
            adapter=None,  # No adapter for bridge mode
            provider=provider,
            resolved_key=resolved_key,
            model=profile.model,
            provider_config=profile.provider_config,
        )

        port = await server.start_async()
        assert port > 0  # Server starts on random available port

        # Test health check
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/healthz") as resp:
                assert resp.status == 200
                body = await resp.json()
                assert body["status"] == "ok"

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_bridge_exposes_chat_completions_endpoint(self) -> None:
        """Bridge server should expose /v1/chat/completions."""
        from kitty.bridge.server import BridgeServer
        from kitty.providers.zai import ZaiRegularAdapter

        profile = _make_profile("test", provider="zai_regular")
        provider = ZaiRegularAdapter()

        server = BridgeServer(
            adapter=None,
            provider=provider,
            resolved_key="test-key",
            model=profile.model,
            provider_config=profile.provider_config,
        )

        port = await server.start_async()
        assert port > 0

        # Test health check
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/healthz") as resp:
                assert resp.status == 200
                body = await resp.json()
                assert body["status"] == "ok"

        await server.stop_async()


# -- Integration Tests ---------------------------------------------------------


class TestBridgeIntegration:
    """Integration tests for bridge mode."""

    def test_route_result_for_bridge(self) -> None:
        """RouteResult should represent bridge mode correctly."""
        profile = _make_profile("test")
        result = RouteResult(
            builtin=BuiltinCommand.BRIDGE,
            profile=profile,
            extra_args=["--port", "8080"],
        )
        assert result.builtin == BuiltinCommand.BRIDGE
        assert result.adapter is None  # No launcher adapter for bridge
        assert result.profile == profile
        assert result.extra_args == ["--port", "8080"]
