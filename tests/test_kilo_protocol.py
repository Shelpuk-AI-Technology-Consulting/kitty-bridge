"""Tests for Kilo Code CLI adapter — BridgeProtocol enum, KiloAdapter properties, spawn config."""

import uuid

from kitty.types import BridgeProtocol


def _make_profile(**overrides):
    from kitty.profiles.schema import Profile

    defaults = {
        "name": "test",
        "provider": "openrouter",
        "model": "test-model",
        "auth_ref": str(uuid.uuid4()),
    }
    defaults.update(overrides)
    return Profile(**defaults)


class TestBridgeProtocolChatCompletions:
    def test_chat_completions_api_value(self):
        assert BridgeProtocol.CHAT_COMPLETIONS_API.value == "chat_completions"

    def test_is_str_subclass(self):
        assert isinstance(BridgeProtocol.CHAT_COMPLETIONS_API, str)

    def test_construct_from_string(self):
        assert BridgeProtocol("chat_completions") is BridgeProtocol.CHAT_COMPLETIONS_API

    def test_enum_has_four_members(self):
        assert set(BridgeProtocol) == {
            BridgeProtocol.RESPONSES_API,
            BridgeProtocol.MESSAGES_API,
            BridgeProtocol.GEMINI_API,
            BridgeProtocol.CHAT_COMPLETIONS_API,
        }


class TestKiloAdapterProperties:
    def test_name(self):
        from kitty.launchers.kilo import KiloAdapter

        adapter = KiloAdapter()
        assert adapter.name == "kilo"

    def test_binary_name(self):
        from kitty.launchers.kilo import KiloAdapter

        adapter = KiloAdapter()
        assert adapter.binary_name == "kilo"

    def test_bridge_protocol(self):
        from kitty.launchers.kilo import KiloAdapter

        adapter = KiloAdapter()
        assert adapter.bridge_protocol is BridgeProtocol.CHAT_COMPLETIONS_API


class TestKiloAdapterSpawnConfig:
    def test_env_overrides_empty(self):
        """Config is set via opencode.json, not env vars."""
        from kitty.launchers.kilo import KiloAdapter

        adapter = KiloAdapter()
        profile = _make_profile()
        config = adapter.build_spawn_config(profile, 12345, "test-key")
        assert config.env_overrides == {}

    def test_cli_args_empty(self):
        from kitty.launchers.kilo import KiloAdapter

        adapter = KiloAdapter()
        config = adapter.build_spawn_config(_make_profile(), 12345, "test-key")
        assert config.cli_args == []

    def test_env_clear_empty(self):
        from kitty.launchers.kilo import KiloAdapter

        adapter = KiloAdapter()
        config = adapter.build_spawn_config(_make_profile(), 12345, "test-key")
        assert config.env_clear == []

    def test_resolved_key_not_in_env_overrides(self):
        """API key should NOT be in env_overrides — it goes in the config file instead."""
        from kitty.launchers.kilo import KiloAdapter

        adapter = KiloAdapter()
        config = adapter.build_spawn_config(_make_profile(), 12345, "secret-key-123")
        assert "apiKey" not in config.env_overrides
        assert all("secret" not in v for v in config.env_overrides.values())

    def test_stashes_bridge_port(self):
        """build_spawn_config stashes bridge_port for use in prepare_launch."""
        from kitty.launchers.kilo import KiloAdapter

        adapter = KiloAdapter()
        adapter.build_spawn_config(_make_profile(), 9999, "key")
        assert adapter._bridge_port == 9999

    def test_stashes_resolved_key(self):
        """build_spawn_config stashes resolved_key for use in prepare_launch."""
        from kitty.launchers.kilo import KiloAdapter

        adapter = KiloAdapter()
        adapter.build_spawn_config(_make_profile(), 12345, "my-secret-key")
        assert adapter._resolved_key == "my-secret-key"

    def test_stashes_model(self):
        """build_spawn_config stashes model for use in prepare_launch."""
        from kitty.launchers.kilo import KiloAdapter

        adapter = KiloAdapter()
        adapter.build_spawn_config(_make_profile(model="gpt-4o"), 12345, "key")
        assert adapter._model == "gpt-4o"
