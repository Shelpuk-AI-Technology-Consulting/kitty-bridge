"""Tests for launchers/base.py — LauncherAdapter interface, SpawnConfig, BridgeProtocol re-export."""

import inspect

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


_REGISTERED_ADAPTERS = ("codex", "claude", "gemini", "kilo")


def _adapter(name: str):
    """Build one of the adapters the CLI registers.

    Args:
        name: Target name as typed by the user, e.g. ``"kilo"``.

    Returns:
        The adapter instance.
    """
    from kitty.launchers.claude import ClaudeAdapter
    from kitty.launchers.codex import CodexAdapter
    from kitty.launchers.gemini import GeminiAdapter
    from kitty.launchers.kilo import KiloAdapter

    return {"codex": CodexAdapter, "claude": ClaudeAdapter, "gemini": GeminiAdapter, "kilo": KiloAdapter}[name]()


class TestEachAdapterOwnsItsSettingsPath:
    """The launcher must patch each agent's own config file, and no other.

    It used to sniff for a module-level constant that was never an adapter
    attribute, so the lookup always failed and every agent was handed Claude
    Code's ``settings.json``. Kilo then wrote its provider block into that file
    while its own config went untouched — the agent stayed misconfigured and an
    unrelated file was modified. A signature-shape test cannot see any of that;
    these assert on the path actually resolved.
    """

    @pytest.mark.parametrize("name", _REGISTERED_ADAPTERS)
    def test_settings_path_is_declared_on_the_adapter(self, name: str):
        adapter = _adapter(name)

        assert hasattr(adapter, "default_settings_path"), "the launcher asks every adapter for this"

    def test_claude_resolves_to_claude_settings(self):
        path = _adapter("claude").default_settings_path

        assert path is not None
        assert path.name == "settings.json"
        assert ".claude" in path.parts

    def test_kilo_resolves_to_its_own_config_not_claudes(self):
        """The exact confusion that corrupted an unrelated file."""
        path = _adapter("kilo").default_settings_path

        assert path is not None
        assert path.name == "kilo.json"
        assert ".claude" not in path.parts

    @pytest.mark.parametrize("name", ["codex", "gemini"])
    def test_agents_without_a_settings_file_resolve_to_none(self, name: str):
        """None means 'do not patch anything', not 'use somebody else's file'."""
        assert _adapter(name).default_settings_path is None

    def test_no_two_adapters_share_a_settings_path(self):
        paths = [(n, _adapter(n).default_settings_path) for n in _REGISTERED_ADAPTERS]
        configured = [(n, p) for n, p in paths if p is not None]
        distinct = {p for _n, p in configured}

        assert len(distinct) == len(configured), f"adapters share a config file: {configured}"

    @pytest.mark.parametrize("name", _REGISTERED_ADAPTERS)
    def test_prepare_launch_accepts_the_launchers_call_shape(self, name: str, tmp_path):
        """Mirrors cli/launcher.py: positional env, keyword settings_path."""
        adapter = _adapter(name)

        # bind() raises TypeError if the keyword does not exist — that is the assertion.
        inspect.signature(adapter.prepare_launch).bind({"X": "1"}, settings_path=tmp_path / "cfg.json")
        inspect.signature(adapter.cleanup_launch).bind(None, settings_path=tmp_path / "cfg.json")


class TestBuildSpawnConfigContextTokens:
    """AC5.4: every adapter accepts the ``context_tokens`` keyword.

    ``launch_async`` always passes the resolved context window to
    ``build_spawn_config``; only ClaudeAdapter uses it, but the others must
    accept and ignore it so the orchestrator call site stays uniform.
    """

    @staticmethod
    def _profile():
        """Build a minimal valid profile for spawn-config construction."""
        import uuid

        from kitty.profiles.schema import Profile

        return Profile(name="t", provider="zai_regular", model="test-model", auth_ref=str(uuid.uuid4()))

    def test_base_declares_context_tokens_keyword_only(self):
        """The interface pins the kwarg as keyword-only with a None default."""
        from kitty.launchers.base import LauncherAdapter

        params = inspect.signature(LauncherAdapter.build_spawn_config).parameters
        assert "context_tokens" in params
        assert params["context_tokens"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["context_tokens"].default is None

    @pytest.mark.parametrize("name", _REGISTERED_ADAPTERS)
    def test_adapters_accept_and_survive_the_kwarg(self, name: str):
        """Every adapter accepts the kwarg; non-Claude ones must ignore it.

        Only ClaudeAdapter may emit CLAUDE_CODE_MAX_CONTEXT_TOKENS. The child
        reads it through the environment, so a leak from any other adapter
        would still reach Claude Code — absence is the observable contract.
        """
        adapter = _adapter(name)

        config = adapter.build_spawn_config(
            self._profile(),
            bridge_port=8080,
            resolved_key="sk-test",
            context_tokens=1_000_000,
        )

        assert config is not None
        if name != "claude":
            assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in config.env_overrides
