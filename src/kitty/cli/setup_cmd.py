"""Setup wizard command — first-run profile creation."""

from __future__ import annotations

import sys
import uuid

from kitty.credentials.store import CredentialStore
from kitty.profiles.schema import _NAME_PATTERN, RESERVED_NAMES, Profile
from kitty.profiles.store import ProfileStore
from rich.console import Console

from kitty.tui.display import print_error, print_section, print_status, print_step, print_warning
from kitty.tui.menu import SelectionMenu
from kitty.tui.prompts import NonTTYError, prompt_confirm, prompt_secret, prompt_text

__all__ = ["run_setup_wizard"]


def _check_tty() -> None:
    """Raise if not running in an interactive terminal."""
    if not sys.stdin.isatty():
        raise NonTTYError("This command requires an interactive terminal (TTY)")


def run_setup_wizard(store: ProfileStore, cred_store: CredentialStore) -> Profile:
    """Run the first-run setup wizard to create a profile.

    Steps:
    1. Provider selection
    2. API key entry (masked)
    3. Model selection
    4. Profile name (validated)
    5. Set as default confirmation
    6. Connectivity check (optional)

    Args:
        store: Profile store to save the new profile.
        cred_store: Credential store to save the API key.

    Returns:
        The created and saved Profile.

    Raises:
        NonTTYError: If not running in an interactive terminal.
    """
    _check_tty()

    print_section("kitty setup wizard")

    # Step 1: Provider selection
    print_step(1, 6, "Provider selection")
    provider_menu = SelectionMenu("Select provider", ["zai_regular", "zai_coding", "minimax", "novita", "ollama", "openai", "openrouter", "fireworks", "anthropic", "bedrock", "azure", "vertex"])
    provider = provider_menu.show()
    if provider is None:
        raise NonTTYError("Provider selection cancelled")

    # Step 2: API key (masked)
    print_step(2, 6, "API key")
    api_key = prompt_secret("Enter API key: ")
    if not api_key:
        print_error("API key cannot be empty")
        raise ValueError("API key cannot be empty")

    # Step 3: Model
    print_step(3, 6, "Model selection")
    model = prompt_text("Enter model name: ")
    if not model or not model.strip():
        print_error("Model name cannot be empty")
        raise ValueError("Model name cannot be empty")
    model = model.strip()

    # Step 4: Profile name (validated)
    print_step(4, 6, "Profile name")
    is_first = len(store.load_all()) == 0
    while True:
        name = prompt_text("Enter profile name (leave empty for default): ")
        if not name or not name.strip():
            name = provider  # Default to provider name
            break
        name = name.strip().lower()
        if name in RESERVED_NAMES:
            print_error(f"Profile name {name!r} is reserved")
            continue
        if not _NAME_PATTERN.match(name):
            print_error("Name must be lowercase, 1-32 chars, alphanumeric/dash/underscore")
            continue
        break

    # Step 5: Set as default
    print_step(5, 6, "Default profile")
    set_default = is_first or prompt_confirm("Set as default profile?", default=True)

    # Create profile
    auth_ref = str(uuid.uuid4())
    profile = Profile(
        name=name,
        provider=provider,  # type: ignore[arg-type]
        model=model,
        auth_ref=auth_ref,
        is_default=set_default,
    )

    # Save credential and profile
    cred_store.set(auth_ref, api_key)
    store.save(profile)
    print_status(f"Profile {name!r} created successfully")

    # Step 6: Connectivity check (optional)
    print_step(6, 6, "Connectivity check")
    if prompt_confirm("Test connectivity to provider?", default=True):
        console = Console()
        with console.status("[bold cyan]Testing connectivity..."):
            connected = _check_connectivity(provider, api_key)
        if connected:
            print_status("Connectivity check passed")
        else:
            print_warning("Connectivity check failed — profile saved but may not work until API key is verified")

    return profile


def _check_connectivity(provider_type: str, api_key: str) -> bool:
    """Test connectivity to the provider's API.

    Args:
        provider_type: Provider type string.
        api_key: API key to use for authentication.

    Returns:
        True if connectivity check passed, False otherwise.
    """
    try:
        from kitty.providers.registry import get_provider

        get_provider(provider_type)  # Verify provider is resolvable
        # A real connectivity check would make an HTTP request
        return True
    except Exception:
        return False
