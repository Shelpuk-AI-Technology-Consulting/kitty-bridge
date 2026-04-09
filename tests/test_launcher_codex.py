"""Tests for launchers/codex.py — CodexAdapter spawn configuration."""

import uuid

from kitty.launchers.codex import CodexAdapter
from kitty.profiles.schema import Profile
from kitty.types import BridgeProtocol


def _make_profile(model: str = "gpt-4o") -> Profile:
    return Profile(
        name="test-profile",
        provider="zai_regular",
        model=model,
        auth_ref=str(uuid.uuid4()),
    )


class TestCodexAdapterProperties:
    def test_name(self):
        assert CodexAdapter().name == "codex"

    def test_binary_name(self):
        assert CodexAdapter().binary_name == "codex"

    def test_bridge_protocol(self):
        assert CodexAdapter().bridge_protocol == BridgeProtocol.RESPONSES_API


class TestCodexAdapterSpawnConfig:
    def test_cli_args_with_port_and_model(self):
        adapter = CodexAdapter()
        profile = _make_profile(model="gpt-5.4")
        config = adapter.build_spawn_config(profile, bridge_port=4242, resolved_key="sk-test")

        assert config.cli_args == [
            "-c",
            "model_provider=kitty",
            "-c",
            "model_providers.kitty.base_url=http://127.0.0.1:4242/v1",
            "-c",
            "model_providers.kitty.wire_api=responses",
            "-c",
            "model_providers.kitty.supports_websockets=false",
            "-c",
            "model_providers.kitty.name=Kitty Bridge",
            "-c",
            "model=gpt-5.4",
        ]

    def test_cli_args_interpolates_port(self):
        adapter = CodexAdapter()
        profile = _make_profile()
        config = adapter.build_spawn_config(profile, bridge_port=9999, resolved_key="key")

        base_url_arg = config.cli_args[3]
        assert base_url_arg == "model_providers.kitty.base_url=http://127.0.0.1:9999/v1"

    def test_env_overrides_empty(self):
        adapter = CodexAdapter()
        profile = _make_profile()
        config = adapter.build_spawn_config(profile, bridge_port=8080, resolved_key="key")
        assert config.env_overrides == {}

    def test_env_clear_empty(self):
        adapter = CodexAdapter()
        profile = _make_profile()
        config = adapter.build_spawn_config(profile, bridge_port=8080, resolved_key="key")
        assert config.env_clear == []

    def test_resolved_key_not_in_config(self):
        """CodexAdapter does not pass the API key via env or args — bridge handles auth."""
        adapter = CodexAdapter()
        profile = _make_profile()
        config = adapter.build_spawn_config(profile, bridge_port=8080, resolved_key="secret-key")

        for arg in config.cli_args:
            assert "secret-key" not in arg
        assert "secret-key" not in config.env_overrides.values()
