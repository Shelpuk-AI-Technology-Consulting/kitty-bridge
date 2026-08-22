"""Tests for the launch orchestrator — wires bridge + adapter + child process."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kitty.bridge.server import BridgeServer
from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.launchers.claude import ClaudeAdapter
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

    def build_spawn_config(
        self,
        profile: Profile,
        bridge_port: int,
        resolved_key: str,
        *,
        context_tokens: int | None = None,
    ) -> SpawnConfig:
        del context_tokens  # launch_async always passes it; this stub ignores it
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


class _Sentinel(Exception):
    """Raised by stubbed ``start_async`` implementations to stop the flow."""


@pytest.fixture(autouse=True)
def _no_catalog_refresh(monkeypatch: pytest.MonkeyPatch):
    """Keep orchestrator tests offline: no network for the catalog sync.

    ``launch_async`` refreshes the model-context overrides catalog before
    starting the bridge (R3). This default keeps every lifecycle test
    deterministic and network-free; individual tests may re-patch the
    attribute with a recording stand-in.
    """
    from kitty.providers import model_context_sync

    async def _skip_refresh(**kwargs):
        return True

    monkeypatch.setattr(model_context_sync, "refresh_model_context_overrides", _skip_refresh)


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

        # F45: Now raises FileNotFoundError (catchable) instead of SystemExit
        with patch("kitty.cli.launcher.discover_binary", return_value=None), pytest.raises(FileNotFoundError):
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

            def build_spawn_config(self, profile, bridge_port, resolved_key, *, context_tokens=None):
                del context_tokens  # ignored by this stub
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


# ── Test: overrides-catalog refresh hook (AC3.4) ────────────────────────────


class TestLaunchRefreshesOverridesCatalog:
    """launch_async awaits the catalog refresh before the bridge starts."""

    @pytest.mark.asyncio
    async def test_refresh_awaited_before_bridge_start(self, monkeypatch: pytest.MonkeyPatch):
        from kitty.cli.launcher import launch_async
        from kitty.providers import model_context_sync

        order: list = []

        async def fake_refresh(**kwargs):
            order.append("refresh")
            return True

        async def fake_start_async(self):
            order.append("start")
            raise _Sentinel

        monkeypatch.setattr(model_context_sync, "refresh_model_context_overrides", fake_refresh)
        monkeypatch.setattr("kitty.bridge.server.BridgeServer.start_async", fake_start_async)

        with pytest.raises(_Sentinel):
            await launch_async(
                adapter=StubLauncher(),
                provider=StubProvider(),
                profile=_make_profile(),
                cred_store=_make_cred_store(),
                validate=False,
            )
        assert order == ["refresh", "start"]


# ── Test: context-window wiring (R6) ────────────────────────────────────────


class _RecordingLauncher(StubLauncher):
    """StubLauncher that records the ``context_tokens`` kwarg (R6)."""

    def __init__(self):
        """Initialize the protocol default and the recording list."""
        super().__init__()
        self.seen_context_tokens: list[int | None] = []

    def build_spawn_config(self, profile, bridge_port, resolved_key, *, context_tokens=None):
        """Record the kwarg, then build the normal stub spawn config."""
        self.seen_context_tokens.append(context_tokens)
        return super().build_spawn_config(profile, bridge_port, resolved_key, context_tokens=context_tokens)


class TestLaunchPassesContextTokens:
    """launch_async computes the context window and hands it to the adapter."""

    @pytest.mark.asyncio
    async def test_single_profile_passes_model_window(self, monkeypatch: pytest.MonkeyPatch):
        """AC6.1/AC6.3: single profile → the get_model_context_tokens result."""
        import kitty.providers.model_context as model_context
        from kitty.cli.launcher import launch_async

        seen_args: list[tuple] = []

        def fake_get(provider, model, provider_config=None):
            seen_args.append((provider, model, provider_config))
            return 999_999

        monkeypatch.setattr(model_context, "get_model_context_tokens", fake_get)
        adapter = _RecordingLauncher()

        with patch("kitty.cli.launcher.discover_binary", return_value=Path(sys.executable)):
            exit_code = await launch_async(
                adapter=adapter,
                provider=StubProvider(),
                profile=_make_profile(),
                cred_store=_make_cred_store(),
                extra_args=[],
                validate=False,
            )

        assert exit_code == 0
        assert adapter.seen_context_tokens == [999_999]
        assert seen_args == [("zai_regular", "test-model", {})]

    @pytest.mark.asyncio
    async def test_balancing_passes_min_across_backends(self, monkeypatch: pytest.MonkeyPatch):
        """AC6.2: balancing launch → get_balancing_min_context_tokens result."""
        import kitty.providers.model_context as model_context
        from kitty.cli.launcher import launch_async

        seen_entries: list[tuple] = []

        def fake_min(backends):
            seen_entries.extend(backends)
            return 123_456

        monkeypatch.setattr(model_context, "get_balancing_min_context_tokens", fake_min)
        adapter = _RecordingLauncher()

        member_a = Profile(name="a", provider="zai_regular", model="model-a", auth_ref=str(uuid.uuid4()))
        member_b = Profile(name="b", provider="zai_regular", model="model-b", auth_ref=str(uuid.uuid4()))
        backends = [
            (StubProvider(), "sk-a", member_a),
            (StubProvider(), "sk-b", member_b),
        ]

        with patch("kitty.cli.launcher.discover_binary", return_value=Path(sys.executable)):
            exit_code = await launch_async(
                adapter=adapter,
                provider=StubProvider(),
                profile=member_a,
                cred_store=_make_cred_store(),
                extra_args=[],
                validate=False,
                backends=backends,
            )

        assert exit_code == 0
        assert adapter.seen_context_tokens == [123_456]
        assert seen_entries == [
            ("stub", "model-a", {}),
            ("stub", "model-b", {}),
        ]


# ── Test: per-session settings reach the child command (issue #22) ───────────


class _RecordingClaudeAdapter(ClaudeAdapter):
    """ClaudeAdapter that records what ``prepare_launch`` handed back.

    The seam between ``prepare_launch`` and command construction is the one
    place where the #22 fix can silently fail: kitty could write a perfectly
    isolated session file and then never tell Claude Code about it, falling
    back to the user-global file with no test noticing.
    """

    def __init__(self, settings_path: Path):
        self._settings_path = settings_path
        self.prepared: str | None = None

    @property
    def binary_name(self) -> str:
        return "python"

    @property
    def default_settings_path(self) -> Path:
        return self._settings_path

    def build_spawn_config(self, profile, bridge_port, resolved_key, *, context_tokens=None):
        del context_tokens
        return SpawnConfig(
            cli_args=["-c", "import sys; sys.exit(0)"],
            env_overrides={"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{bridge_port}"},
            env_clear=[],
        )

    def prepare_launch(self, env_overrides, settings_path=None):
        self.prepared = super().prepare_launch(env_overrides, settings_path=settings_path)
        return self.prepared


class TestSessionSettingsReachTheChild:
    """AC11: the spawned argv must point at the file prepare_launch wrote."""

    @pytest.mark.asyncio
    async def test_spawned_command_carries_the_prepared_settings_path(self, tmp_path: Path):
        from kitty.cli.launcher import launch_async

        adapter = _RecordingClaudeAdapter(tmp_path / ".claude" / "settings.json")
        spawned: list[list[str]] = []

        # Record the argv and stand in for the child: the stub binary is the
        # Python interpreter, which would reject Claude Code's own --settings
        # flag. What matters here is the command kitty built.
        async def _record(*cmd, **kwargs):
            del kwargs
            spawned.append(list(cmd))
            proc = MagicMock()

            async def _wait():
                return 0

            proc.wait = _wait
            proc.pid = 4242
            proc.returncode = 0
            return proc

        with (
            patch("kitty.cli.launcher.discover_binary", return_value=Path(sys.executable)),
            patch("asyncio.create_subprocess_exec", side_effect=_record),
        ):
            exit_code = await launch_async(
                adapter=adapter,
                provider=StubProvider(),
                profile=_make_profile(),
                cred_store=_make_cred_store(),
                extra_args=[],
            )

        assert exit_code == 0
        assert adapter.prepared is not None
        argv = spawned[0]
        assert "--settings" in argv, "kitty never told Claude Code about the session file"
        assert argv[argv.index("--settings") + 1] == str(adapter.prepared)
        # Global flags must precede the binary's own args.
        assert argv.index("--settings") < argv.index("-c")

    @pytest.mark.asyncio
    async def test_settings_flag_precedes_a_user_supplied_subcommand(self, tmp_path: Path):
        """AC4: `kitty claude mcp list` must still be routed through the bridge.

        Claude Code's global options have to come before a subcommand, so a
        regression that appended the flag after ``extra_args`` would silently
        drop the session's routing for every subcommand invocation.
        """
        from kitty.cli.launcher import launch_async

        adapter = _RecordingClaudeAdapter(tmp_path / ".claude" / "settings.json")
        spawned: list[list[str]] = []

        async def _record(*cmd, **kwargs):
            del kwargs
            spawned.append(list(cmd))
            proc = MagicMock()

            async def _wait():
                return 0

            proc.wait = _wait
            proc.pid = 4242
            proc.returncode = 0
            return proc

        with (
            patch("kitty.cli.launcher.discover_binary", return_value=Path(sys.executable)),
            patch("asyncio.create_subprocess_exec", side_effect=_record),
        ):
            await launch_async(
                adapter=adapter,
                provider=StubProvider(),
                profile=_make_profile(),
                cred_store=_make_cred_store(),
                extra_args=["mcp", "list"],
            )

        argv = spawned[0]
        assert argv.index("--settings") < argv.index("mcp")

    @pytest.mark.asyncio
    async def test_launch_fails_closed_when_the_session_file_cannot_be_written(self, tmp_path: Path, capsys):
        """AC9: continuing would run the session on the user's own credentials,
        bypassing the bridge, the egress guard, and usage logging."""
        from kitty.cli import launcher as launcher_mod
        from kitty.cli.launcher import launch_async

        adapter = _RecordingClaudeAdapter(tmp_path / ".claude" / "settings.json")
        stopped: list[bool] = []

        def _boom(*_args, **_kwargs):
            raise OSError("no space left on device")

        real_stop = BridgeServer.stop_async

        async def _record_stop(self):
            stopped.append(True)
            return await real_stop(self)

        with (
            patch("kitty.cli.launcher.discover_binary", return_value=Path(sys.executable)),
            patch("kitty.launchers.claude.tempfile.mkstemp", side_effect=_boom),
            patch.object(BridgeServer, "stop_async", _record_stop),
            patch("asyncio.create_subprocess_exec") as mock_spawn,
        ):
            exit_code = await launch_async(
                adapter=adapter,
                provider=StubProvider(),
                profile=_make_profile(),
                cred_store=_make_cred_store(),
                extra_args=[],
            )

        assert exit_code == 1
        assert "session settings file" in capsys.readouterr().err
        mock_spawn.assert_not_called()
        assert stopped, "the bridge must be stopped when the launch aborts"
        assert launcher_mod._atexit_cleanup_state == []


class TestAdaptersWithoutSessionSettings:
    """AC7: adapters that own their config file keep their existing contract."""

    @pytest.mark.asyncio
    async def test_kilo_config_is_removed_even_though_nothing_was_prepared(self, tmp_path: Path):
        """``original is None`` means "I created this file — delete it".

        Gating cleanup on the prepared value would strand a Kilo config
        containing the real API key for anyone without a pre-existing one.
        """
        from kitty.cli.launcher import launch_async
        from kitty.launchers.kilo import KiloAdapter

        config_path = tmp_path / "kilo.json"

        class _TmpKilo(KiloAdapter):
            @property
            def binary_name(self) -> str:
                return "python"

            @property
            def default_settings_path(self) -> Path:
                return config_path

            def build_spawn_config(self, profile, bridge_port, resolved_key, *, context_tokens=None):
                super().build_spawn_config(profile, bridge_port, resolved_key, context_tokens=context_tokens)
                return SpawnConfig(cli_args=["-c", "import sys; sys.exit(0)"], env_overrides={}, env_clear=[])

        adapter = _TmpKilo()
        with patch("kitty.cli.launcher.discover_binary", return_value=Path(sys.executable)):
            exit_code = await launch_async(
                adapter=adapter,
                provider=StubProvider(),
                profile=_make_profile(),
                cred_store=_make_cred_store(),
                extra_args=[],
            )

        assert exit_code == 0
        assert not config_path.exists(), "kitty left a Kilo config holding the real API key"


# ── Test: session summary wiring (issue #26, R4) ────────────────────────────


class _RecordingServer(BridgeServer):
    """BridgeServer that records ``session_summary_path`` and aborts the launch."""

    seen: list = []

    def __init__(self, *args, **kwargs):
        """Record the nominated summary path before construction."""
        type(self).seen.append(kwargs.get("session_summary_path"))
        super().__init__(*args, **kwargs)

    async def start_async(self) -> int:
        """Abort the launch once the constructor has been observed."""
        raise _Sentinel


class TestSessionSummaryWiring:
    """The nominated summary path reaches the bridge that will write it."""

    @pytest.mark.asyncio
    async def test_launch_async_passes_the_summary_path_to_the_bridge(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """A4.2: the agent-launch path is the one CI actually uses."""
        from kitty.cli import launcher as launcher_mod

        _RecordingServer.seen = []
        monkeypatch.setattr(launcher_mod, "BridgeServer", _RecordingServer)

        with pytest.raises(_Sentinel):
            await launcher_mod.launch_async(
                adapter=StubLauncher(),
                provider=StubProvider(),
                profile=_make_profile(),
                cred_store=_make_cred_store(),
                validate=False,
                session_summary_path=tmp_path / "run.json",
            )

        assert _RecordingServer.seen == [tmp_path / "run.json"]

    @pytest.mark.asyncio
    async def test_launch_async_defaults_to_no_summary(self, monkeypatch: pytest.MonkeyPatch):
        """A4.4: an un-nominated launch leaves the bridge writing nothing."""
        from kitty.cli import launcher as launcher_mod

        _RecordingServer.seen = []
        monkeypatch.setattr(launcher_mod, "BridgeServer", _RecordingServer)

        with pytest.raises(_Sentinel):
            await launcher_mod.launch_async(
                adapter=StubLauncher(),
                provider=StubProvider(),
                profile=_make_profile(),
                cred_store=_make_cred_store(),
                validate=False,
            )

        assert _RecordingServer.seen == [None]


# ── Test: profile naming on the attribution surfaces (issue #26, R2/R5) ─────


class _NameRecordingServer(BridgeServer):
    """BridgeServer that records ``profile_name`` and aborts the launch."""

    seen: list = []

    def __init__(self, *args, **kwargs):
        """Record the profile name the launch path chose."""
        type(self).seen.append(kwargs.get("profile_name"))
        super().__init__(*args, **kwargs)

    async def start_async(self) -> int:
        """Abort the launch once the constructor has been observed."""
        raise _Sentinel


class TestProfileNameWiring:
    """An attribution record that calls every session "default" names nothing."""

    @pytest.mark.asyncio
    async def test_launch_names_the_profile_it_launched(self, monkeypatch: pytest.MonkeyPatch):
        """A5.5: the single-backend header reports the profile, not a placeholder."""
        from kitty.cli import launcher as launcher_mod

        _NameRecordingServer.seen = []
        monkeypatch.setattr(launcher_mod, "BridgeServer", _NameRecordingServer)

        with pytest.raises(_Sentinel):
            await launcher_mod.launch_async(
                adapter=StubLauncher(),
                provider=StubProvider(),
                profile=_make_profile(),
                cred_store=_make_cred_store(),
                validate=False,
            )

        assert _NameRecordingServer.seen == ["test"]

    @pytest.mark.asyncio
    async def test_a_balancing_launch_names_the_pool_not_its_first_member(self, monkeypatch: pytest.MonkeyPatch):
        """A2.2: the session ran as a pool; naming one member misdescribes it."""
        from kitty.cli import launcher as launcher_mod

        _NameRecordingServer.seen = []
        monkeypatch.setattr(launcher_mod, "BridgeServer", _NameRecordingServer)

        with pytest.raises(_Sentinel):
            await launcher_mod.launch_async(
                adapter=StubLauncher(),
                provider=StubProvider(),
                profile=_make_profile(),
                cred_store=_make_cred_store(),
                validate=False,
                backends=[(StubProvider(), "key", _make_profile())],
                profile_name="ci-pool",
            )

        assert _NameRecordingServer.seen == ["ci-pool"]
