"""Tests for the launch orchestrator — wires bridge + adapter + child process."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# ── Stub adapters ─────────────────────────────────────────────────────────────


class StubLauncher(LauncherAdapter):
    def __init__(self, protocol: BridgeProtocol = BridgeProtocol.RESPONSES_API):
        self._protocol = protocol

    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "echo"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return self._protocol

    def build_spawn_config(self, profile: Profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(
            cli_args=["-c", "import sys; sys.exit(0)"],
            env_overrides={"STUB_KEY": resolved_key},
            env_clear=[],
        )


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


def _make_profile() -> Profile:
    return Profile(
        name="test",
        provider="zai_regular",
        model="test-model",
        auth_ref=str(uuid.uuid4()),
    )


def _make_cred_store(key: str = "sk-test-key") -> CredentialStore:
    backend = MagicMock(spec=FileBackend)
    backend.get = MagicMock(return_value=key)
    return CredentialStore(backends=[backend])


# ── Test: map_child_exit_code ────────────────────────────────────────────────


class TestMapChildExitCode:
    def test_zero_passes_through(self):
        from kitty.cli.launcher import map_child_exit_code

        assert map_child_exit_code(0) == 0

    def test_positive_passes_through(self):
        from kitty.cli.launcher import map_child_exit_code

        assert map_child_exit_code(1) == 1
        assert map_child_exit_code(2) == 2
        assert map_child_exit_code(42) == 42

    def test_negative_maps_to_signal_convention(self):
        from kitty.cli.launcher import map_child_exit_code

        assert map_child_exit_code(-9) == 137  # SIGKILL
        assert map_child_exit_code(-15) == 143  # SIGTERM

    def test_negative_bounded_to_255(self):
        from kitty.cli.launcher import map_child_exit_code

        assert map_child_exit_code(-200) == 255


# ── Test: build_child_env ───────────────────────────────────────────────────


class TestBuildChildEnv:
    def test_copies_parent_env(self):
        from kitty.cli.launcher import build_child_env

        spawn_config = SpawnConfig()
        env = build_child_env(spawn_config)
        assert "PATH" in env or "path" in env or len(env) > 0

    def test_clears_specified_vars(self):
        from kitty.cli.launcher import build_child_env

        spawn_config = SpawnConfig(
            env_clear=["HOME"],
        )
        env = build_child_env(spawn_config)
        assert "HOME" not in env

    def test_applies_overrides(self):
        from kitty.cli.launcher import build_child_env

        spawn_config = SpawnConfig(
            env_overrides={"MY_VAR": "hello"},
        )
        env = build_child_env(spawn_config)
        assert env["MY_VAR"] == "hello"

    def test_clear_before_override(self):
        from kitty.cli.launcher import build_child_env

        spawn_config = SpawnConfig(
            env_overrides={"MY_VAR": "new"},
            env_clear=["MY_VAR"],
        )
        env = build_child_env(spawn_config)
        assert env["MY_VAR"] == "new"


# ── Test: resolve_binary ────────────────────────────────────────────────────


class TestResolveBinary:
    def test_returns_path_when_found(self):
        from kitty.cli.launcher import resolve_binary

        with patch("kitty.cli.launcher.discover_binary", return_value=Path("/usr/bin/codex")):
            result = resolve_binary("codex")
            assert result == Path("/usr/bin/codex")

    def test_raises_when_not_found(self):
        from kitty.cli.launcher import resolve_binary

        with patch("kitty.cli.launcher.discover_binary", return_value=None), pytest.raises(SystemExit):
            resolve_binary("codex")


# ── Test: launch async lifecycle ────────────────────────────────────────────


class TestLaunchLifecycle:
    @pytest.mark.asyncio
    async def test_launch_spawns_child_and_returns_exit_code(self):
        from kitty.cli.launcher import launch_async

        adapter = StubLauncher()
        provider = StubProvider()
        profile = _make_profile()
        cred_store = _make_cred_store()

        # Use "echo hello" as the child — always exits 0
        with patch("kitty.cli.launcher.discover_binary", return_value=Path(sys.executable)):
            exit_code = await launch_async(
                adapter=adapter,
                provider=provider,
                profile=profile,
                cred_store=cred_store,
                extra_args=[],
            )
        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_launch_passes_nonzero_exit_code(self):
        from kitty.cli.launcher import launch_async

        adapter = StubLauncher()
        provider = StubProvider()
        profile = _make_profile()
        cred_store = _make_cred_store()

        # Use python -c "exit(3)" as child
        adapter_name = adapter.name

        class FailLauncher(StubLauncher):
            @property
            def name(self) -> str:
                return adapter_name

            def build_spawn_config(self, profile, bridge_port, resolved_key):
                return SpawnConfig(
                    cli_args=["-c", "raise SystemExit(3)"],
                    env_overrides={},
                    env_clear=[],
                )

        fail_adapter = FailLauncher()
        with patch("kitty.cli.launcher.discover_binary", return_value=Path(sys.executable)):
            exit_code = await launch_async(
                adapter=fail_adapter,
                provider=provider,
                profile=profile,
                cred_store=cred_store,
                extra_args=[],
            )
        assert exit_code == 3

    @pytest.mark.asyncio
    async def test_launch_cleans_up_bridge(self):
        from kitty.cli.launcher import launch_async

        adapter = StubLauncher()
        provider = StubProvider()
        profile = _make_profile()
        cred_store = _make_cred_store()

        with patch("kitty.cli.launcher.discover_binary", return_value=Path(sys.executable)):
            await launch_async(
                adapter=adapter,
                provider=provider,
                profile=profile,
                cred_store=cred_store,
                extra_args=[],
            )
        # Bridge should be stopped — verify no lingering runners
        # (If bridge was not stopped, the test would hang on teardown)


class TestLaunchSync:
    def test_launch_sync_calls_async(self):
        from kitty.cli.launcher import launch

        adapter = StubLauncher()
        provider = StubProvider()
        profile = _make_profile()
        cred_store = _make_cred_store()

        with patch("kitty.cli.launcher.discover_binary", return_value=Path(sys.executable)):
            exit_code = launch(
                adapter=adapter,
                provider=provider,
                profile=profile,
                cred_store=cred_store,
                extra_args=[],
            )
        assert exit_code == 0
