"""Tests for launchers/claude.py — ClaudeAdapter spawn configuration."""

import uuid

import pytest

from kitty.launchers.claude import ClaudeAdapter
from kitty.profiles.schema import Profile
from kitty.types import BridgeProtocol


def _make_profile(model: str = "claude-sonnet-4-20250514") -> Profile:
    return Profile(
        name="test-profile",
        provider="zai_regular",
        model=model,
        auth_ref=str(uuid.uuid4()),
    )


class TestClaudeAdapterProperties:
    def test_name(self):
        assert ClaudeAdapter().name == "claude"

    def test_binary_name(self):
        assert ClaudeAdapter().binary_name == "claude"

    def test_bridge_protocol(self):
        assert ClaudeAdapter().bridge_protocol == BridgeProtocol.MESSAGES_API


class TestClaudeAdapterSpawnConfig:
    def test_env_overrides_base_url(self):
        adapter = ClaudeAdapter()
        profile = _make_profile()
        config = adapter.build_spawn_config(profile, bridge_port=4242, resolved_key="sk-test-key")

        assert config.env_overrides["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4242"

    def test_env_overrides_api_key(self):
        adapter = ClaudeAdapter()
        profile = _make_profile()
        config = adapter.build_spawn_config(profile, bridge_port=8080, resolved_key="my-secret-key")

        assert config.env_overrides["ANTHROPIC_API_KEY"] == "my-secret-key"

    def test_env_overrides_model(self):
        """ANTHROPIC_MODEL sets the startup model in Claude Code."""
        adapter = ClaudeAdapter()
        profile = _make_profile(model="minimax-m2.7")
        config = adapter.build_spawn_config(profile, bridge_port=8080, resolved_key="key")

        assert config.env_overrides["ANTHROPIC_MODEL"] == "minimax-m2.7"

    def test_env_overrides_model_aliases(self):
        """All three alias env vars are set to the profile model so Claude Code
        uses it regardless of which alias (opus/sonnet/haiku) it picks."""
        adapter = ClaudeAdapter()
        profile = _make_profile(model="minimax-m2.7")
        config = adapter.build_spawn_config(profile, bridge_port=8080, resolved_key="key")

        assert config.env_overrides["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "minimax-m2.7"
        assert config.env_overrides["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "minimax-m2.7"
        assert config.env_overrides["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "minimax-m2.7"

    def test_env_clear_lists_conflicting_vars(self):
        adapter = ClaudeAdapter()
        profile = _make_profile()
        config = adapter.build_spawn_config(profile, bridge_port=8080, resolved_key="key")

        assert set(config.env_clear) == {
            "ANTHROPIC_BEDROCK_BASE_URL",
            "ANTHROPIC_VERTEX_BASE_URL",
            "ANTHROPIC_FOUNDRY_BASE_URL",
        }

    def test_cli_args_empty(self):
        adapter = ClaudeAdapter()
        profile = _make_profile()
        config = adapter.build_spawn_config(profile, bridge_port=8080, resolved_key="key")
        assert config.cli_args == []

    def test_base_url_no_trailing_v1(self):
        """ANTHROPIC_BASE_URL must NOT include /v1 — Claude Code appends its own path."""
        adapter = ClaudeAdapter()
        profile = _make_profile()
        config = adapter.build_spawn_config(profile, bridge_port=8080, resolved_key="key")

        base_url = config.env_overrides["ANTHROPIC_BASE_URL"]
        assert not base_url.endswith("/v1")
        assert not base_url.endswith("/")

    def test_auth_token_in_env_overrides(self):
        """ANTHROPIC_AUTH_TOKEN must be in env_overrides so Claude Code does not
        require an Anthropic account login.  Without it, Claude Code checks for
        an OAuth token and prompts the user to log in, ignoring ANTHROPIC_API_KEY."""
        adapter = ClaudeAdapter()
        profile = _make_profile()
        config = adapter.build_spawn_config(profile, bridge_port=4242, resolved_key="sk-test")

        assert "ANTHROPIC_AUTH_TOKEN" in config.env_overrides
        assert config.env_overrides["ANTHROPIC_AUTH_TOKEN"]  # non-empty

    def test_auth_token_not_in_env_clear(self):
        """ANTHROPIC_AUTH_TOKEN must NOT be in env_clear because kitty sets its own
        value for it (to bypass the login gate)."""
        adapter = ClaudeAdapter()
        profile = _make_profile()
        config = adapter.build_spawn_config(profile, bridge_port=4242, resolved_key="sk-test")

        assert "ANTHROPIC_AUTH_TOKEN" not in config.env_clear


class TestClaudeAdapterContextTokens:
    """build_spawn_config propagates the resolved context window (R5)."""

    def test_context_tokens_sets_env_var(self):
        """AC5.1: an explicit window becomes CLAUDE_CODE_MAX_CONTEXT_TOKENS."""
        adapter = ClaudeAdapter()
        config = adapter.build_spawn_config(
            _make_profile(),
            bridge_port=8080,
            resolved_key="key",
            context_tokens=1_000_000,
        )

        assert config.env_overrides["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1000000"

    def test_no_context_tokens_omits_key(self):
        """AC5.2: omitting the kwarg leaves the child env untouched."""
        adapter = ClaudeAdapter()
        config = adapter.build_spawn_config(_make_profile(), bridge_port=8080, resolved_key="key")

        assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in config.env_overrides

    def test_none_context_tokens_omits_key(self):
        """AC5.2: an explicit None behaves exactly like the default."""
        adapter = ClaudeAdapter()
        config = adapter.build_spawn_config(
            _make_profile(),
            bridge_port=8080,
            resolved_key="key",
            context_tokens=None,
        )

        assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in config.env_overrides

    @pytest.mark.parametrize("tokens", [0, -5])
    def test_non_positive_context_tokens_omits_key(self, tokens: int):
        """Only a positive window is meaningful; zero/negative must not leak."""
        adapter = ClaudeAdapter()
        config = adapter.build_spawn_config(
            _make_profile(),
            bridge_port=8080,
            resolved_key="key",
            context_tokens=tokens,
        )

        assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in config.env_overrides

    def test_context_tokens_stringifies_large_windows_exactly(self):
        """The env value is the exact decimal string, even beyond 2**53.

        The value travels through JSON (settings.json env block), where
        consumers may parse numbers as floats; emitting ``str(int)`` keeps
        arbitrarily large windows exact end to end.
        """
        adapter = ClaudeAdapter()
        config = adapter.build_spawn_config(
            _make_profile(),
            bridge_port=8080,
            resolved_key="key",
            context_tokens=9_007_199_254_740_993,
        )

        assert config.env_overrides["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "9007199254740993"

    def test_context_tokens_set_for_non_claude_models(self):
        """R5: the window is set unconditionally — non-claude models are the point.

        Claude Code ignores the var for ``claude-*`` models but falls back to
        200K for everything else, so a guard emitting the key only for
        claude models would invert the documented rationale and silently
        reintroduce the early-compaction bug this task exists to fix.
        """
        adapter = ClaudeAdapter()
        config = adapter.build_spawn_config(
            _make_profile(model="qwen3.8-max"),
            bridge_port=8080,
            resolved_key="key",
            context_tokens=1_000_000,
        )

        assert config.env_overrides["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1000000"


class TestInjectedKeyListsInSync:
    """AC5.5: the context-window key must be injected AND cleaned up.

    ``_SETTINGS_ENV_OVERRIDE_KEYS`` makes ``prepare_launch`` write the key
    into settings.json; ``cleanup_cmd._KITTY_INJECTED_KEYS`` makes ``kitty
    cleanup`` remove it after a crashed session. A key present in only one
    list either never reaches Claude Code or survives a crash.
    """

    def test_context_key_in_settings_override_list(self):
        """The key must be injected into settings.json by prepare_launch."""
        from kitty.launchers.claude import _SETTINGS_ENV_OVERRIDE_KEYS

        assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" in _SETTINGS_ENV_OVERRIDE_KEYS

    def test_context_key_in_cleanup_list(self):
        """The key must be removed by ``kitty cleanup`` after a crash."""
        from kitty.cli.cleanup_cmd import _KITTY_INJECTED_KEYS

        assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" in _KITTY_INJECTED_KEYS
