"""Test that TUI provider lists are synced with registry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import questionary

from kitty.cli.profile_cmd import _create_profile_flow
from kitty.cli.setup_cmd import run_setup_wizard
from kitty.profiles.schema import PROVIDER_LABELS, PROVIDER_LIST, PROVIDER_SECTIONS


class _StopFlow(Exception):
    """Raised by test doubles to stop interactive flow after capturing menu options."""


def _capture_provider_options(monkeypatch: pytest.MonkeyPatch, module_prefix: str) -> list[str]:
    captured: list[str] = []

    class FakeSelectionMenu:
        def __init__(self, title: str, options: list) -> None:
            assert title == "Select provider"
            for opt in options:
                if isinstance(opt, questionary.Separator):
                    continue
                captured.append(opt.value if isinstance(opt, questionary.Choice) else opt)

        def show(self) -> str:
            raise _StopFlow

    monkeypatch.setattr(f"{module_prefix}.SelectionMenu", FakeSelectionMenu)
    return captured


def _expected_labels() -> set[str]:
    """Build the set of expected menu labels from PROVIDER_LIST + PROVIDER_LABELS."""
    return {PROVIDER_LABELS.get(p, p) for p in PROVIDER_LIST}


def test_setup_cmd_includes_all_registered_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup wizard provider list must show a label for every registered provider."""
    captured = _capture_provider_options(monkeypatch, "kitty.cli.setup_cmd")
    monkeypatch.setattr("kitty.cli.setup_cmd.check_tty", lambda: None)

    with pytest.raises(_StopFlow):
        run_setup_wizard(store=MagicMock(), cred_store=MagicMock())

    labels_in_ui = set(captured)
    expected = _expected_labels()

    missing_in_ui = expected - labels_in_ui
    extra_in_ui = labels_in_ui - expected

    assert not missing_in_ui, f"Providers missing in setup_cmd.py TUI: {missing_in_ui}"
    assert not extra_in_ui, f"Extra providers in setup_cmd.py TUI (not in registry): {extra_in_ui}"


def test_profile_cmd_includes_all_registered_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Profile command provider list must show a label for every registered provider."""
    captured = _capture_provider_options(monkeypatch, "kitty.cli.profile_cmd")

    with pytest.raises(_StopFlow):
        _create_profile_flow(store=MagicMock(), cred_store=MagicMock())

    labels_in_ui = set(captured)
    expected = _expected_labels()

    missing_in_ui = expected - labels_in_ui
    extra_in_ui = labels_in_ui - expected

    assert not missing_in_ui, f"Providers missing in profile_cmd.py TUI: {missing_in_ui}"
    assert not extra_in_ui, f"Extra providers in profile_cmd.py TUI (not in registry): {extra_in_ui}"


def test_provider_sections_cover_all_providers() -> None:
    """Every provider in PROVIDER_LIST must appear in exactly one section."""
    sectioned = [p for _, providers in PROVIDER_SECTIONS for p in providers]
    all_providers = set(PROVIDER_LIST)
    sectioned_set = set(sectioned)

    missing = all_providers - sectioned_set
    extra = sectioned_set - all_providers
    duplicates = [p for p in sectioned if sectioned.count(p) > 1]

    assert not missing, f"Providers not in any section: {missing}"
    assert not extra, f"Unknown providers in sections: {extra}"
    assert not duplicates, f"Duplicate providers in sections: {set(duplicates)}"


def test_provider_sections_sorted_alphabetically() -> None:
    """Providers within each section must be sorted alphabetically by display label."""
    for header, providers in PROVIDER_SECTIONS:
        sorted_providers = sorted(providers, key=lambda k: PROVIDER_LABELS.get(k, k))
        assert providers == sorted_providers, f"Section {header!r} not sorted: {providers} vs {sorted_providers}"


def test_provider_labels_covers_all() -> None:
    """Every provider in PROVIDER_LIST must have an entry in PROVIDER_LABELS."""
    unlabeled = [p for p in PROVIDER_LIST if p not in PROVIDER_LABELS]
    assert not unlabeled, f"Providers missing from PROVIDER_LABELS: {unlabeled}"
