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
