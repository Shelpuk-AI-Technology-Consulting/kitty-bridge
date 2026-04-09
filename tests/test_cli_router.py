"""Tests for CLI router — command routing logic."""

from __future__ import annotations

import uuid

import pytest

from kitty.cli.router import BuiltinCommand, CLIRouter, RouteResult
from kitty.launchers.base import LauncherAdapter
from kitty.launchers.claude import ClaudeAdapter
from kitty.launchers.codex import CodexAdapter
from kitty.profiles.resolver import NoDefaultProfileError
from kitty.profiles.schema import Profile
from kitty.profiles.store import ProfileStore

# -- Fixtures -----------------------------------------------------------------


def _make_profile(
    name: str = "test-profile",
    provider: str = "zai_regular",
    model: str = "gpt-4o",
    is_default: bool = False,
) -> Profile:
    return Profile(
        name=name,
        provider=provider,
        model=model,
        auth_ref=str(uuid.uuid4()),
        is_default=is_default,
    )


@pytest.fixture()
def adapters() -> dict[str, LauncherAdapter]:
    return {"codex": CodexAdapter(), "claude": ClaudeAdapter()}


@pytest.fixture()
def populated_store(tmp_path: object) -> ProfileStore:
    store = ProfileStore(path=tmp_path / "profiles.json")  # type: ignore[arg-type]
    store.save(_make_profile("dev", is_default=True))
    store.save(_make_profile("staging", is_default=False))
    return store


@pytest.fixture()
def empty_store(tmp_path: object) -> ProfileStore:
    return ProfileStore(path=tmp_path / "profiles.json")  # type: ignore[arg-type]


# -- RouteResult data class ---------------------------------------------------


class TestRouteResult:
    def test_default_fields(self) -> None:
        result = RouteResult()
        assert result.adapter is None
        assert result.profile is None
        assert result.extra_args == []
        assert result.builtin is None

    def test_custom_fields(self) -> None:
        adapter = CodexAdapter()
        profile = _make_profile()
        result = RouteResult(adapter=adapter, profile=profile, extra_args=["--foo"])
        assert result.adapter is adapter
        assert result.profile is profile
        assert result.extra_args == ["--foo"]


# -- Built-in commands --------------------------------------------------------


class TestBuiltinCommands:
    def test_setup_routes_to_setup(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(populated_store, adapters)
        result = router.route(["setup"])
        assert result.builtin == BuiltinCommand.SETUP
        assert result.adapter is None

    def test_profile_routes_to_profile(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(populated_store, adapters)
        result = router.route(["profile"])
        assert result.builtin == BuiltinCommand.PROFILE

    def test_doctor_routes_to_doctor(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(populated_store, adapters)
        result = router.route(["doctor"])
        assert result.builtin == BuiltinCommand.DOCTOR

    def test_setup_case_insensitive(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(populated_store, adapters)
        result = router.route(["SETUP"])
        assert result.builtin == BuiltinCommand.SETUP


class TestBuiltinCommandIsStr:
    def test_is_str_subclass(self):
        assert isinstance(BuiltinCommand.SETUP, str)

    def test_construct_from_string_value(self):
        assert BuiltinCommand("setup") is BuiltinCommand.SETUP

    def test_usable_as_dict_key(self):
        d = {BuiltinCommand.SETUP: "a"}
        assert d["setup"] == "a"


# -- Launcher target routing --------------------------------------------------


class TestLauncherTargetRouting:
    def test_codex_routes_with_default_profile(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(populated_store, adapters)
        result = router.route(["codex"])
        assert result.adapter is not None
        assert result.adapter.name == "codex"
        assert result.profile is not None
        assert result.profile.is_default is True

    def test_claude_routes_with_default_profile(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(populated_store, adapters)
        result = router.route(["claude"])
        assert result.adapter is not None
        assert result.adapter.name == "claude"
        assert result.profile is not None
        assert result.profile.is_default is True

    def test_target_with_extra_args(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(populated_store, adapters)
        result = router.route(["codex", "--some-flag", "value"])
        assert result.adapter is not None
        assert result.adapter.name == "codex"
        assert result.extra_args == ["--some-flag", "value"]


# -- Profile name routing -----------------------------------------------------


class TestProfileRouting:
    def test_profile_name_routes_to_default_target(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(populated_store, adapters)
        result = router.route(["dev"])
        assert result.adapter is not None
        assert result.adapter.name == "codex"  # default target
        assert result.profile is not None
        assert result.profile.name == "dev"

    def test_profile_name_case_insensitive(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(populated_store, adapters)
        result = router.route(["DEV"])
        assert result.profile is not None
        assert result.profile.name == "dev"

    def test_profile_with_explicit_target(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(populated_store, adapters)
        result = router.route(["staging", "claude"])
        assert result.adapter is not None
        assert result.adapter.name == "claude"
        assert result.profile is not None
        assert result.profile.name == "staging"

    def test_profile_with_extra_args(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(populated_store, adapters)
        result = router.route(["dev", "--verbose"])
        assert result.profile is not None
        assert result.profile.name == "dev"
        assert result.extra_args == ["--verbose"]


# -- Unknown / error cases ----------------------------------------------------


class TestRoutingErrors:
    def test_unknown_word_fails(
        self,
        populated_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(populated_store, adapters)
        with pytest.raises(Exception, match="Unknown command or profile"):
            router.route(["nonexistent"])

    def test_no_default_profile_fails_for_target(
        self,
        tmp_path: object,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        store = ProfileStore(path=tmp_path / "profiles.json")  # type: ignore[arg-type]
        store.save(_make_profile("nondefault", is_default=False))
        router = CLIRouter(store, adapters)
        with pytest.raises(NoDefaultProfileError):
            router.route(["codex"])


# -- Auto-setup ---------------------------------------------------------------


class TestAutoSetup:
    def test_empty_store_triggers_setup(
        self,
        empty_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(empty_store, adapters)
        result = router.route([])
        assert result.builtin == BuiltinCommand.SETUP

    def test_empty_store_any_command_triggers_setup(
        self,
        empty_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(empty_store, adapters)
        result = router.route(["codex"])
        assert result.builtin == BuiltinCommand.SETUP

    def test_empty_store_profile_command_triggers_setup(
        self,
        empty_store: ProfileStore,
        adapters: dict[str, LauncherAdapter],
    ) -> None:
        router = CLIRouter(empty_store, adapters)
        result = router.route(["profile"])
        assert result.builtin == BuiltinCommand.SETUP
