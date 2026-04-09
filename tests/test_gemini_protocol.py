"""Tests for the GeminiAdapter and Gemini bridge protocol."""

import uuid

from kitty.launchers.gemini import GeminiAdapter
from kitty.profiles.schema import Profile
from kitty.types import BridgeProtocol


def _make_profile(**overrides):
    defaults = dict(name="test", provider="openrouter", model="test-model", auth_ref=str(uuid.uuid4()))
    defaults.update(overrides)
    return Profile(**defaults)


class TestBridgeProtocolGemini:
    """Verify GEMINI_API exists in BridgeProtocol enum."""

    def test_gemini_api_value(self):
        assert BridgeProtocol.GEMINI_API.value == "gemini"

    def test_enum_has_four_members(self):
        assert set(BridgeProtocol) == {
            BridgeProtocol.RESPONSES_API,
            BridgeProtocol.MESSAGES_API,
            BridgeProtocol.GEMINI_API,
            BridgeProtocol.CHAT_COMPLETIONS_API,
        }


class TestGeminiAdapterProperties:
    """Verify GeminiAdapter basic properties."""

    def setup_method(self):
        self.adapter = GeminiAdapter()

    def test_name(self):
        assert self.adapter.name == "gemini"

    def test_binary_name(self):
        assert self.adapter.binary_name == "gemini"

    def test_bridge_protocol(self):
        assert self.adapter.bridge_protocol == BridgeProtocol.GEMINI_API


class TestGeminiAdapterSpawnConfig:
    """Verify GeminiAdapter.build_spawn_config sets the right env vars."""

    def setup_method(self):
        self.adapter = GeminiAdapter()

    def test_env_overrides_set_base_url(self):
        profile = _make_profile()
        config = self.adapter.build_spawn_config(profile, bridge_port=42421, resolved_key="key-123")
        assert config.env_overrides["GOOGLE_GEMINI_BASE_URL"] == "http://127.0.0.1:42421"

    def test_env_overrides_set_api_key(self):
        profile = _make_profile()
        config = self.adapter.build_spawn_config(profile, bridge_port=42421, resolved_key="key-123")
        assert config.env_overrides["GEMINI_API_KEY"] == "key-123"

    def test_cli_args_empty(self):
        profile = _make_profile()
        config = self.adapter.build_spawn_config(profile, bridge_port=42421, resolved_key="key-123")
        assert config.cli_args == []

    def test_env_clear_empty(self):
        profile = _make_profile()
        config = self.adapter.build_spawn_config(profile, bridge_port=42421, resolved_key="key-123")
        assert config.env_clear == []
