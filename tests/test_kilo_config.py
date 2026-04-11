"""Tests for KiloAdapter prepare_launch / cleanup_launch config file patching."""

import json
import uuid

import pytest

from kitty.launchers.kilo import KiloAdapter
from kitty.profiles.schema import Profile


def _make_profile(**overrides):
    defaults = {
        "name": "test",
        "provider": "openrouter",
        "model": "test-model",
        "auth_ref": str(uuid.uuid4()),
    }
    defaults.update(overrides)
    return Profile(**defaults)


@pytest.fixture()
def adapter():
    return KiloAdapter()


@pytest.fixture()
def config_dir(tmp_path):
    """Return a temporary config directory path."""
    d = tmp_path / "kilo-config"
    d.mkdir()
    return d


@pytest.fixture()
def config_path(config_dir):
    return config_dir / "kilo.json"


class TestPrepareLaunch:
    def test_creates_config_when_none_exists(self, adapter, config_path):
        adapter.build_spawn_config(_make_profile(), 18080, "my-api-key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, config_path=config_path)

        assert original is None  # No previous file
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert "kitty" in data["provider"]

    def test_config_contains_provider_with_base_url(self, adapter, config_path):
        adapter.build_spawn_config(_make_profile(), 18080, "my-api-key")
        adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, config_path=config_path)

        data = json.loads(config_path.read_text())
        kitty = data["provider"]["kitty"]
        assert kitty["options"]["baseURL"] == "http://127.0.0.1:18080/v1"

    def test_config_contains_provider_with_api_key(self, adapter, config_path):
        adapter.build_spawn_config(_make_profile(), 18080, "my-api-key")
        adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, config_path=config_path)

        data = json.loads(config_path.read_text())
        assert data["provider"]["kitty"]["options"]["apiKey"] == "my-api-key"

    def test_config_contains_model(self, adapter, config_path):
        adapter.build_spawn_config(_make_profile(model="gpt-4o"), 18080, "key")
        adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, config_path=config_path)

        data = json.loads(config_path.read_text())
        models = data["provider"]["kitty"]["models"]
        # Model key is the bare model name (no provider prefix)
        assert "gpt-4o" in models
        assert models["gpt-4o"]["id"] == "gpt-4o"
        # Active model uses provider/model format
        assert data["model"] == "kitty/gpt-4o"

    def test_patches_existing_config_preserving_other_providers(self, adapter, config_path):
        # Pre-existing config with another provider
        existing = {"provider": {"openai": {"options": {"apiKey": "existing-key"}}}}
        config_path.write_text(json.dumps(existing))

        adapter.build_spawn_config(_make_profile(), 18080, "my-key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, config_path=config_path)

        data = json.loads(config_path.read_text())
        assert "openai" in data["provider"]  # Preserved
        assert "kitty" in data["provider"]  # Added
        assert original is not None  # Saved original

    def test_returns_original_content(self, adapter, config_path):
        existing = {"provider": {"openai": {"options": {"apiKey": "sk-123"}}}}
        config_path.write_text(json.dumps(existing))

        adapter.build_spawn_config(_make_profile(), 18080, "key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, config_path=config_path)

        assert json.loads(original) == existing

    def test_handles_malformed_json(self, adapter, config_path):
        config_path.write_text("not valid json {{{")

        adapter.build_spawn_config(_make_profile(), 18080, "key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, config_path=config_path)

        # Should still create valid config, original saved as-is
        assert original == "not valid json {{{"
        data = json.loads(config_path.read_text())
        assert "kitty" in data["provider"]

    def test_config_has_npm_field(self, adapter, config_path):
        adapter.build_spawn_config(_make_profile(), 18080, "key")
        adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, config_path=config_path)

        data = json.loads(config_path.read_text())
        assert data["provider"]["kitty"]["npm"] == "@ai-sdk/openai-compatible"


class TestCleanupLaunch:
    def test_restores_original_config(self, adapter, config_path):
        existing = {"provider": {"openai": {"options": {"apiKey": "original"}}}}
        config_path.write_text(json.dumps(existing))

        adapter.build_spawn_config(_make_profile(), 18080, "key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, config_path=config_path)

        # Config now has kitty provider
        assert "kitty" in json.loads(config_path.read_text())["provider"]

        # Cleanup restores original
        adapter.cleanup_launch(original, config_path=config_path)
        assert json.loads(config_path.read_text()) == existing

    def test_removes_temporary_config_when_no_original(self, adapter, config_path):
        adapter.build_spawn_config(_make_profile(), 18080, "key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, config_path=config_path)

        assert config_path.exists()
        adapter.cleanup_launch(original, config_path=config_path)
        assert not config_path.exists()

    def test_cleanup_with_none_is_noop(self, adapter, config_path):
        # If prepare_launch was never called or returned None
        adapter.cleanup_launch(None, config_path=config_path)
        assert not config_path.exists()
