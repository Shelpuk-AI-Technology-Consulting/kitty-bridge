"""Test that TUI provider lists are synced with registry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kitty.cli.profile_cmd import _create_profile_flow
from kitty.cli.setup_cmd import run_setup_wizard
from kitty.providers.registry import _registry


class _StopFlow(Exception):
    """Raised by test doubles to stop interactive flow after capturing menu options."""


def _capture_provider_options(monkeypatch: pytest.MonkeyPatch, module_prefix: str) -> list[str]:
    captured: list[str] = []

    class FakeSelectionMenu:
        def __init__(self, title: str, options: list[str]) -> None:
            assert title == "Select provider"
            captured.extend(options)

        def show(self) -> str:
            raise _StopFlow

    monkeypatch.setattr(f"{module_prefix}.SelectionMenu", FakeSelectionMenu)
    return captured


def test_setup_cmd_includes_all_registered_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup wizard provider list must include all providers from registry."""
    captured = _capture_provider_options(monkeypatch, "kitty.cli.setup_cmd")
    monkeypatch.setattr("kitty.cli.setup_cmd.check_tty", lambda: None)

    with pytest.raises(_StopFlow):
        run_setup_wizard(store=MagicMock(), cred_store=MagicMock())

    providers_in_ui = set(captured)
    providers_in_registry = set(_registry.keys())

    missing_in_ui = providers_in_registry - providers_in_ui
    extra_in_ui = providers_in_ui - providers_in_registry

    assert not missing_in_ui, f"Providers missing in setup_cmd.py TUI: {missing_in_ui}"
    assert not extra_in_ui, f"Extra providers in setup_cmd.py TUI (not in registry): {extra_in_ui}"


def test_profile_cmd_includes_all_registered_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Profile command provider list must include all providers from registry."""
    captured = _capture_provider_options(monkeypatch, "kitty.cli.profile_cmd")

    with pytest.raises(_StopFlow):
        _create_profile_flow(store=MagicMock(), cred_store=MagicMock())

    providers_in_ui = set(captured)
    providers_in_registry = set(_registry.keys())

    missing_in_ui = providers_in_registry - providers_in_ui
    extra_in_ui = providers_in_ui - providers_in_registry

    assert not missing_in_ui, f"Providers missing in profile_cmd.py TUI: {missing_in_ui}"
    assert not extra_in_ui, f"Extra providers in profile_cmd.py TUI (not in registry): {extra_in_ui}"
