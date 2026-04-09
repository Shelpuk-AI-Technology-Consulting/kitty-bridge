"""Profile menu command — interactive profile management."""

from __future__ import annotations

import sys
import uuid

from kitty.credentials.store import CredentialStore
from kitty.profiles.schema import RESERVED_NAMES, Profile
from kitty.profiles.store import ProfileStore
from kitty.tui.display import (
    clear_screen,
    print_error,
    print_info,
    print_section,
    print_status,
    print_table,
    print_warning,
)
from kitty.tui.menu import SelectionMenu
from kitty.tui.prompts import NonTTYError, prompt_confirm, prompt_secret, prompt_text

__all__ = ["run_profile_menu"]


def _check_tty() -> None:
    """Raise if not running in an interactive terminal."""
    if not sys.stdin.isatty():
        raise NonTTYError("This command requires an interactive terminal (TTY)")


def run_profile_menu(store: ProfileStore) -> None:
    """Interactive profile management menu.

    Args:
        store: Profile store to manage.

    Raises:
        NonTTYError: If not running in an interactive terminal.
    """
    _check_tty()

    while True:
        profiles = store.load_all()

        clear_screen()
        print_section("Profile Management")
        if profiles:
            rows = [
                [
                    p.name,
                    p.provider,
                    p.model,
                    "Yes" if p.is_default else "No",
                ]
                for p in profiles
            ]
            print_table(["Name", "Provider", "Model", "Default"], rows)
        else:
            print("  (no profiles)")

        menu = SelectionMenu(
            "Actions",
            [
                "Create new profile",
                "Delete profile",
                "Set default profile",
                "List profiles",
                "Back",
            ],
        )
        choice = menu.show()
        if choice is None or choice == "Back":
            break

        if choice == "Create new profile":
            cred_store = _make_credential_store(store)
            _create_profile_flow(store, cred_store)
        elif choice == "Delete profile":
            _delete_profile_flow(store)
        elif choice == "Set default profile":
            _set_default_flow(store)
        elif choice == "List profiles":
            # Already displayed above
            pass


def _make_credential_store(store: ProfileStore) -> CredentialStore:
    """Create a CredentialStore with FileBackend in the same directory as the profile store."""
    from kitty.credentials.file_backend import FileBackend

    creds_path = store._path.parent / "credentials.json"
    return CredentialStore(backends=[FileBackend(path=creds_path)])


def _create_profile_flow(store: ProfileStore, cred_store: CredentialStore) -> Profile:
    """Interactive flow for creating a new profile.

    Returns:
        The created and saved Profile.
    """
    # Step 1: Provider selection
    provider_menu = SelectionMenu("Select provider", ["zai_regular", "zai_coding", "minimax", "novita", "openai", "openrouter"])
    provider = provider_menu.show()
    if provider is None:
        raise NonTTYError("Provider selection cancelled")

    # Step 2: API key
    api_key = prompt_secret("Enter API key: ")
    if not api_key:
        print_error("API key cannot be empty")
        raise ValueError("API key cannot be empty")

    # Step 3: Model
    model = prompt_text("Enter model name: ")
    if not model or not model.strip():
        print_error("Model name cannot be empty")
        raise ValueError("Model name cannot be empty")
    model = model.strip()

    # Step 4: Profile name
    is_first = len(store.load_all()) == 0
    while True:
        name = prompt_text("Enter profile name: ")
        if not name or not name.strip():
            name = provider  # Default to provider name
            break
        name = name.strip().lower()
        if name in RESERVED_NAMES:
            print_error(f"Profile name {name!r} is reserved")
            continue
        from kitty.profiles.schema import _NAME_PATTERN

        if not _NAME_PATTERN.match(name):
            print_error("Name must be lowercase, 1-32 chars, alphanumeric/dash/underscore")
            continue
        break

    # Step 5: Set as default?
    set_default = is_first or prompt_confirm("Set as default profile?", default=True)

    # Step 6: Create and save
    auth_ref = str(uuid.uuid4())
    profile = Profile(
        name=name,
        provider=provider,  # type: ignore[arg-type]
        model=model,
        auth_ref=auth_ref,
        is_default=set_default,
    )

    cred_store.set(auth_ref, api_key)
    store.save(profile)
    print_status(f"Profile {name!r} created successfully")
    return profile


def _delete_profile_flow(store: ProfileStore) -> None:
    """Interactive flow for deleting a profile."""
    profiles = store.load_all()
    if not profiles:
        print_warning("No profiles to delete")
        return

    names = [p.name for p in profiles]
    menu = SelectionMenu("Select profile to delete", names)
    selected = menu.show()
    if selected is None:
        return

    if prompt_confirm(f"Delete profile {selected!r}?", default=False):
        store.delete(selected)
        print_status(f"Profile {selected!r} deleted")
    else:
        print_info("Cancelled")


def _set_default_flow(store: ProfileStore) -> None:
    """Interactive flow for setting the default profile."""
    profiles = store.load_all()
    if not profiles:
        print_warning("No profiles available")
        return

    names = [p.name for p in profiles]
    menu = SelectionMenu("Select default profile", names)
    selected = menu.show()
    if selected is None:
        return

    profile = store.get(selected)
    if profile is None:
        return

    updated = profile.model_copy(update={"is_default": True})
    store.save(updated)
    print_status(f"Profile {selected!r} set as default")
