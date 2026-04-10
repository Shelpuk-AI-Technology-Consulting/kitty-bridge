"""Test that TUI provider lists are synced with registry."""

from __future__ import annotations

from kitty.cli.profile_cmd import _create_profile_flow
from kitty.cli.setup_cmd import run_setup_wizard
from kitty.providers.registry import _registry


def test_setup_cmd_includes_all_registered_providers() -> None:
    """Setup wizard provider list must include all providers from registry."""
    # Extract provider list from setup_cmd source
    import inspect

    source = inspect.getsource(run_setup_wizard)
    # Find the provider list in the SelectionMenu call
    import re

    match = re.search(r'SelectionMenu\("Select provider",\s*\[(.*?)\]\)', source, re.DOTALL)
    assert match is not None, "Could not find provider list in setup_cmd.py"

    list_str = match.group(1)
    # Extract quoted strings
    providers_in_ui = set(re.findall(r'"([^"]+)"', list_str))
    providers_in_registry = set(_registry.keys())

    missing_in_ui = providers_in_registry - providers_in_ui
    extra_in_ui = providers_in_ui - providers_in_registry

    assert not missing_in_ui, f"Providers missing in setup_cmd.py TUI: {missing_in_ui}"
    assert not extra_in_ui, f"Extra providers in setup_cmd.py TUI (not in registry): {extra_in_ui}"


def test_profile_cmd_includes_all_registered_providers() -> None:
    """Profile command provider list must include all providers from registry."""
    import inspect
    import re

    source = inspect.getsource(_create_profile_flow)
    match = re.search(r'SelectionMenu\("Select provider",\s*\[(.*?)\]\)', source, re.DOTALL)
    assert match is not None, "Could not find provider list in profile_cmd.py"

    list_str = match.group(1)
    providers_in_ui = set(re.findall(r'"([^"]+)"', list_str))
    providers_in_registry = set(_registry.keys())

    missing_in_ui = providers_in_registry - providers_in_ui
    extra_in_ui = providers_in_ui - providers_in_registry

    assert not missing_in_ui, f"Providers missing in profile_cmd.py TUI: {missing_in_ui}"
    assert not extra_in_ui, f"Extra providers in profile_cmd.py TUI (not in registry): {extra_in_ui}"
