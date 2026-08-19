"""Tests for launchers/base.py — LauncherAdapter interface, SpawnConfig, BridgeProtocol re-export."""

import pytest

from kitty.types import BridgeProtocol


class TestBridgeProtocol:
    def test_reexported_from_launchers(self):
        from kitty.launchers import BridgeProtocol as ReExported
        from kitty.types import BridgeProtocol as Original

        assert ReExported is Original

    def test_has_exactly_four_values(self):
        assert set(BridgeProtocol) == {
            BridgeProtocol.RESPONSES_API,
            BridgeProtocol.MESSAGES_API,
            BridgeProtocol.GEMINI_API,
            BridgeProtocol.CHAT_COMPLETIONS_API,
        }

    def test_values_are_strings(self):
        assert BridgeProtocol.RESPONSES_API == "responses"
        assert BridgeProtocol.MESSAGES_API == "messages"

    def test_is_str_subclass(self):
        """Enum members must be str instances (not just equal to strings)."""
        assert isinstance(BridgeProtocol.RESPONSES_API, str)

    def test_construct_from_string_value(self):
        """Must be constructable from its string value."""
        assert BridgeProtocol("responses") is BridgeProtocol.RESPONSES_API
        assert BridgeProtocol("messages") is BridgeProtocol.MESSAGES_API

    def test_usable_as_dict_key_and_value(self):
        """str-Enum must work transparently as string in dicts."""
        d = {BridgeProtocol.RESPONSES_API: "a"}
        assert d["responses"] == "a"
        assert d[BridgeProtocol.RESPONSES_API] == "a"


class TestSpawnConfig:
    def test_construct_with_all_fields(self):
        from kitty.launchers.base import SpawnConfig

        config = SpawnConfig(
            env_overrides={"FOO": "bar"},
            env_clear=["BAZ"],
            cli_args=["--flag"],
        )
        assert config.env_overrides == {"FOO": "bar"}
        assert config.env_clear == ["BAZ"]
        assert config.cli_args == ["--flag"]

    def test_defaults_are_empty(self):
        from kitty.launchers.base import SpawnConfig

        config = SpawnConfig()
        assert config.env_overrides == {}
        assert config.env_clear == []
        assert config.cli_args == []


class TestLauncherAdapter:
    def test_cannot_be_instantiated(self):
        from kitty.launchers.base import LauncherAdapter

        with pytest.raises(TypeError):
            LauncherAdapter()  # type: ignore[abstract]


class TestLaunchLifecycleHooks:
    """Every adapter must expose the launch lifecycle, not just Claude's.

    `cli/launcher.py` patches an agent's own settings file before spawning it and
    restores it afterwards, including from an atexit handler. Only ClaudeAdapter
    implemented those hooks, so the base class did not describe the interface the
    launcher actually relies on — a second adapter needing cleanup would have
    failed silently, because the call site swallows exceptions.
    """

    @staticmethod
    def _bare_adapter():
        """Build a minimal adapter that implements only the abstract members."""
        from kitty.launchers.base import LauncherAdapter, SpawnConfig

        class _Bare(LauncherAdapter):
            @property
            def name(self) -> str:
                return "bare"

            @property
            def binary_name(self) -> str:
                return "bare"

            @property
            def bridge_protocol(self) -> BridgeProtocol:
                return BridgeProtocol.CHAT_COMPLETIONS_API

            def build_spawn_config(self, profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
                return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])

        return _Bare()

    def test_prepare_launch_is_part_of_the_interface(self):
        adapter = self._bare_adapter()

        assert hasattr(adapter, "prepare_launch")

    def test_cleanup_launch_is_part_of_the_interface(self):
        adapter = self._bare_adapter()

        assert hasattr(adapter, "cleanup_launch")

    def test_prepare_launch_defaults_to_no_patching(self):
        """Returning None is what tells the launcher there is nothing to restore."""
        adapter = self._bare_adapter()

        assert adapter.prepare_launch({}) is None

    def test_cleanup_launch_defaults_to_a_no_op(self):
        """Must not raise: the atexit handler calls it without a hasattr guard."""
        adapter = self._bare_adapter()

        assert adapter.cleanup_launch(None) is None

    def test_claude_still_overrides_both(self):
        from kitty.launchers.base import LauncherAdapter
        from kitty.launchers.claude import ClaudeAdapter

        assert ClaudeAdapter.prepare_launch is not LauncherAdapter.prepare_launch
        assert ClaudeAdapter.cleanup_launch is not LauncherAdapter.cleanup_launch


class TestAdaptersAreCallableTheWayTheLauncherCallsThem:
    """Every registered adapter must accept the launcher's actual call shape.

    `cli/launcher.py` invokes the lifecycle hooks with the keyword
    ``settings_path=``. KiloAdapter named the same parameter ``config_path``,
    so `kitty <profile> kilo` raised TypeError before it ever spawned the agent
    — the target was unusable. Nothing caught it because the launch path is
    exercised per-adapter only for Claude.
    """

    @staticmethod
    def _registered_adapters():
        """Return the adapters the CLI actually offers, as (name, instance)."""
        from kitty.launchers.claude import ClaudeAdapter
        from kitty.launchers.codex import CodexAdapter
        from kitty.launchers.gemini import GeminiAdapter
        from kitty.launchers.kilo import KiloAdapter

        return [
            ("codex", CodexAdapter()),
            ("claude", ClaudeAdapter()),
            ("gemini", GeminiAdapter()),
            ("kilo", KiloAdapter()),
        ]

    @pytest.mark.parametrize("name,adapter", _registered_adapters.__func__())
    def test_prepare_launch_accepts_the_launcher_call_shape(self, name: str, adapter, tmp_path):
        """Mirrors cli/launcher.py exactly: positional env, keyword settings_path."""
        import inspect

        signature = inspect.signature(adapter.prepare_launch)
        signature.bind({"ANTHROPIC_BASE_URL": "http://127.0.0.1:1"}, settings_path=tmp_path / "cfg.json")

    @pytest.mark.parametrize("name,adapter", _registered_adapters.__func__())
    def test_cleanup_launch_accepts_the_launcher_call_shape(self, name: str, adapter, tmp_path):
        import inspect

        signature = inspect.signature(adapter.cleanup_launch)
        signature.bind(None, settings_path=tmp_path / "cfg.json")
