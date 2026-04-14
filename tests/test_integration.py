"""Integration tests — full routing, bridge lifecycle, and protocol round-trip."""

from __future__ import annotations

import uuid

from kitty.cli.launcher import map_child_exit_code
from kitty.cli.router import CLIRouter
from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
from kitty.launchers.claude import ClaudeAdapter
from kitty.launchers.codex import CodexAdapter
from kitty.profiles.schema import Profile
from kitty.profiles.store import ProfileStore


def _make_profile(
    name: str = "test-profile",
    provider: str = "zai_regular",
    model: str = "gpt-4o",
    is_default: bool = True,
) -> Profile:
    return Profile(
        name=name,
        provider=provider,
        model=model,
        auth_ref=str(uuid.uuid4()),
        is_default=is_default,
    )


# ── Router Integration ────────────────────────────────────────────────────────


class TestRouterIntegration:
    """Test that router correctly wires to adapters and profiles."""

    def test_codex_default_profile_routing(self, tmp_path: object) -> None:
        store = ProfileStore(path=tmp_path / "profiles.json")  # type: ignore[arg-type]
        store.save(_make_profile("dev", is_default=True))
        adapters = {"codex": CodexAdapter(), "claude": ClaudeAdapter()}
        router = CLIRouter(store, adapters)

        result = router.route(["codex"])
        assert result.adapter is not None
        assert result.adapter.name == "codex"
        assert result.adapter.bridge_protocol.value == "responses"
        assert result.profile is not None
        assert result.profile.name == "dev"

    def test_claude_default_profile_routing(self, tmp_path: object) -> None:
        store = ProfileStore(path=tmp_path / "profiles.json")  # type: ignore[arg-type]
        store.save(_make_profile("dev", is_default=True))
        adapters = {"codex": CodexAdapter(), "claude": ClaudeAdapter()}
        router = CLIRouter(store, adapters)

        result = router.route(["claude"])
        assert result.adapter is not None
        assert result.adapter.name == "claude"
        assert result.adapter.bridge_protocol.value == "messages"
        assert result.profile is not None
        assert result.profile.name == "dev"

    def test_explicit_profile_with_target(self, tmp_path: object) -> None:
        store = ProfileStore(path=tmp_path / "profiles.json")  # type: ignore[arg-type]
        store.save(_make_profile("staging", is_default=False))
        store.save(_make_profile("dev", is_default=True))
        adapters = {"codex": CodexAdapter(), "claude": ClaudeAdapter()}
        router = CLIRouter(store, adapters)

        result = router.route(["staging", "claude"])
        assert result.adapter is not None
        assert result.adapter.name == "claude"
        assert result.profile is not None
        assert result.profile.name == "staging"

    def test_profile_default_target_is_codex(self, tmp_path: object) -> None:
        store = ProfileStore(path=tmp_path / "profiles.json")  # type: ignore[arg-type]
        store.save(_make_profile("staging", is_default=True))
        adapters = {"codex": CodexAdapter(), "claude": ClaudeAdapter()}
        router = CLIRouter(store, adapters)

        result = router.route(["staging"])
        assert result.adapter is not None
        assert result.adapter.name == "codex"
        assert result.profile is not None
        assert result.profile.name == "staging"


# ── Adapter Spawn Config Integration ──────────────────────────────────────────


class TestAdapterSpawnConfig:
    """Test that adapters generate correct spawn configs with real profiles."""

    def test_codex_spawn_config(self, tmp_path: object) -> None:
        profile = _make_profile(model="gpt-4o")
        adapter = CodexAdapter()
        config = adapter.build_spawn_config(profile, bridge_port=12345, resolved_key="sk-test")

        assert "-c" in config.cli_args
        assert "model_provider=kitty" in config.cli_args
        assert "model_providers.kitty.base_url=http://127.0.0.1:12345/v1" in config.cli_args
        assert "model=gpt-4o" in config.cli_args
        assert config.env_overrides == {}
        assert config.env_clear == []

    def test_claude_spawn_config(self, tmp_path: object) -> None:
        profile = _make_profile(model="claude-3-opus")
        adapter = ClaudeAdapter()
        config = adapter.build_spawn_config(profile, bridge_port=54321, resolved_key="sk-test-key")

        assert config.cli_args == []
        assert config.env_overrides["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:54321"
        assert config.env_overrides["ANTHROPIC_API_KEY"] == "sk-test-key"
        assert config.env_overrides["ANTHROPIC_MODEL"] == "claude-3-opus"
        assert config.env_overrides["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "claude-3-opus"
        assert config.env_overrides["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "claude-3-opus"
        assert config.env_overrides["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "claude-3-opus"
        assert "ANTHROPIC_AUTH_TOKEN" not in config.env_clear
        assert config.env_overrides["ANTHROPIC_AUTH_TOKEN"] == "kitty-bridge-token"
        assert "ANTHROPIC_BEDROCK_BASE_URL" in config.env_clear
        assert "ANTHROPIC_VERTEX_BASE_URL" in config.env_clear
        assert "ANTHROPIC_FOUNDRY_BASE_URL" in config.env_clear


# ── Exit Code Integration ─────────────────────────────────────────────────────


class TestExitCodeIntegration:
    """Test exit code mapping in context of the launch lifecycle."""

    def test_success_exit_code_passthrough(self) -> None:
        assert map_child_exit_code(0) == 0

    def test_error_exit_code_passthrough(self) -> None:
        assert map_child_exit_code(1) == 1
        assert map_child_exit_code(2) == 2

    def test_signal_exit_code_mapping(self) -> None:
        import signal

        assert map_child_exit_code(-signal.SIGTERM) == 128 + signal.SIGTERM

    def test_arbitrary_exit_code_passthrough(self) -> None:
        assert map_child_exit_code(42) == 42


# ── Profile Store + Credential Store Integration ─────────────────────────────


class TestStoreIntegration:
    """Test that profile and credential stores work together correctly."""

    def test_profile_save_and_credential_round_trip(self, tmp_path: object) -> None:
        store = ProfileStore(path=tmp_path / "profiles.json")  # type: ignore[arg-type]
        cred_store = CredentialStore(backends=[FileBackend(path=tmp_path / "creds.json")])  # type: ignore[arg-type]

        profile = _make_profile()
        api_key = "sk-test-integration-key-12345"

        # Save
        cred_store.set(profile.auth_ref, api_key)
        store.save(profile)

        # Retrieve
        loaded = store.get(profile.name)
        assert loaded is not None
        assert loaded.auth_ref == profile.auth_ref

        resolved_key = cred_store.resolve(loaded)
        assert resolved_key == api_key

    def test_multiple_profiles_with_different_credentials(self, tmp_path: object) -> None:
        store = ProfileStore(path=tmp_path / "profiles.json")  # type: ignore[arg-type]
        cred_store = CredentialStore(backends=[FileBackend(path=tmp_path / "creds.json")])  # type: ignore[arg-type]

        p1 = _make_profile(name="dev", is_default=True)
        p2 = _make_profile(name="prod", is_default=False)

        cred_store.set(p1.auth_ref, "sk-dev-key")
        cred_store.set(p2.auth_ref, "sk-prod-key")

        store.save(p1)
        store.save(p2)

        assert cred_store.resolve(store.get("dev")) == "sk-dev-key"  # type: ignore[arg-type]
        assert cred_store.resolve(store.get("prod")) == "sk-prod-key"  # type: ignore[arg-type]

    def test_default_profile_single_invariant(self, tmp_path: object) -> None:
        """Only one profile can be default at a time."""
        store = ProfileStore(path=tmp_path / "profiles.json")  # type: ignore[arg-type]

        store.save(_make_profile(name="first", is_default=True))
        store.save(_make_profile(name="second", is_default=True))

        profiles = store.load_all()
        defaults = [p for p in profiles if p.is_default]
        assert len(defaults) == 1
        assert defaults[0].name == "second"
