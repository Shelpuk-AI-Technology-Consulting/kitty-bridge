"""Profile menu command — interactive profile management."""

from __future__ import annotations

import uuid

from kitty.credentials.store import CredentialStore
from kitty.profiles.schema import _NAME_PATTERN, PROVIDER_LABELS, PROVIDER_LIST, RESERVED_NAMES, BalancingProfile, Profile
from kitty.profiles.store import ProfileStore
from kitty.tui.display import (
    print_error,
    print_info,
    print_section,
    print_status,
    print_table,
    print_warning,
)
from kitty.tui.menu import CheckboxMenu, SelectionMenu
from kitty.tui.prompts import NonTTYError, check_tty, prompt_confirm, prompt_secret, prompt_text

__all__ = ["run_profile_menu"]


def run_profile_menu(store: ProfileStore) -> None:
    """Interactive profile management menu.

    Args:
        store: Profile store to manage.

    Raises:
        NonTTYError: If not running in an interactive terminal.
    """
    check_tty()
    cred_store = _make_credential_store(store)

    while True:
        backends = store.get_all_backends()
        regular = [b for b in backends if isinstance(b, Profile)]
        balancing = [b for b in backends if isinstance(b, BalancingProfile)]

        print_section("Profile Management")
        if backends:
            rows = []
            for b in backends:
                if isinstance(b, Profile):
                    rows.append([b.name, b.provider, b.model, "Yes" if b.is_default else "No"])
                else:
                    rows.append([b.name, f"balancing ({len(b.members)})", "—", "Yes" if b.is_default else "No"])
            print_table(["Name", "Provider", "Model", "Default"], rows)
        else:
            print("  (no profiles)")

        # Build context-aware action list
        actions: list[str] = ["Create new profile"]
        if len(regular) >= 2:
            actions.append("Create balancing profile")
        if regular:
            actions.append("Edit profile")
        if balancing:
            actions.append("Edit balancing profile")
        if backends:
            actions.extend(["Delete profile", "Set default profile"])
        actions.append("Back")

        choice = SelectionMenu("Actions", actions).show()
        if choice is None or choice == "Back":
            break

        if choice == "Create new profile":
            try:
                _create_profile_flow(store, cred_store)
            except (NonTTYError, ValueError, KeyboardInterrupt) as exc:
                print_info(f"Cancelled: {exc}")
        elif choice == "Create balancing profile":
            try:
                _create_balancing_flow(store)
            except (NonTTYError, ValueError, KeyboardInterrupt) as exc:
                print_info(f"Cancelled: {exc}")
        elif choice == "Edit profile":
            names = [b.name for b in regular]
            selected = SelectionMenu("Select profile to edit", names).show()
            if selected:
                _edit_profile_flow(store, cred_store, selected)
        elif choice == "Edit balancing profile":
            names = [b.name for b in balancing]
            selected = SelectionMenu("Select balancing profile to edit", names).show()
            if selected:
                _edit_balancing_flow(store, selected)
        elif choice == "Delete profile":
            _delete_profile_flow(store, cred_store)
        elif choice == "Set default profile":
            _set_default_flow(store)


def _make_credential_store(store: ProfileStore) -> CredentialStore:
    """Create a CredentialStore with FileBackend in the same directory as the profile store."""
    from kitty.credentials.file_backend import FileBackend

    creds_path = store._path.parent / "credentials.json"
    return CredentialStore(backends=[FileBackend(path=creds_path)])


def _find_reusable_auth_ref(store: ProfileStore, cred_store: CredentialStore, provider: str) -> str | None:
    """Find an existing auth_ref for the given provider with a resolvable key.

    Args:
        store: Profile store to search.
        cred_store: Credential store to verify the key exists.
        provider: Provider type to match.

    Returns:
        The auth_ref UUID string if found, or None.
    """
    for profile in store.load_all():
        if profile.provider == provider:
            key = cred_store.get(profile.auth_ref)
            if key is not None:
                return profile.auth_ref
    return None


def _create_profile_flow(store: ProfileStore, cred_store: CredentialStore) -> Profile:
    """Interactive flow for creating a new profile.

    Returns:
        The created and saved Profile.
    """
    # Step 1: Provider selection
    _label_to_type = {PROVIDER_LABELS.get(p, p): p for p in PROVIDER_LIST}
    provider = SelectionMenu("Select provider", list(_label_to_type)).show()
    if provider is None:
        raise NonTTYError("Provider selection cancelled")
    if provider in _label_to_type:
        provider = _label_to_type[provider]

    # Step 2: API key — offer reuse if a same-provider profile exists
    existing_auth_ref = _find_reusable_auth_ref(store, cred_store, provider)
    if existing_auth_ref is not None:
        reuse = prompt_confirm(f"Reuse existing API key for {provider!r}?", default=True)
        if reuse:
            auth_ref = existing_auth_ref
        else:
            api_key = prompt_secret("Enter new API key: ")
            if not api_key:
                print_error("API key cannot be empty")
                raise ValueError("API key cannot be empty")
            auth_ref = str(uuid.uuid4())
            cred_store.set(auth_ref, api_key)
    else:
        api_key = prompt_secret("Enter API key: ")
        if not api_key:
            print_error("API key cannot be empty")
            raise ValueError("API key cannot be empty")
        auth_ref = str(uuid.uuid4())
        cred_store.set(auth_ref, api_key)

    # Step 3: Model
    model = prompt_text("Model (OpenRouter or provider format, e.g. z-ai/glm-5): ")
    if not model or not model.strip():
        print_error("Model name cannot be empty")
        raise ValueError("Model name cannot be empty")
    model = model.strip()

    # Step 4: Profile name
    is_first = len(store.get_all_backends()) == 0
    while True:
        name = prompt_text("Enter profile name (leave empty for default): ")
        name = provider if not name or not name.strip() else name.strip().lower()
        if name in RESERVED_NAMES:
            print_error(f"Profile name {name!r} is reserved")
            continue
        if not _NAME_PATTERN.match(name):
            print_error("Name must be lowercase, 1-32 chars, alphanumeric/dash/underscore")
            continue
        if store.get_backend(name) is not None:
            print_error(f"Profile {name!r} already exists")
            continue
        break

    # Step 5: Set as default?
    set_default = is_first or prompt_confirm("Set as default profile?", default=True)

    # Step 6: Create and save
    profile = Profile(
        name=name,
        provider=provider,  # type: ignore[arg-type]
        model=model,
        auth_ref=auth_ref,
        is_default=set_default,
    )
    store.save(profile)
    print_status(f"Profile {name!r} created successfully")
    return profile


def _create_balancing_flow(store: ProfileStore) -> BalancingProfile:
    """Interactive flow for creating a balancing profile.

    Returns:
        The created and saved BalancingProfile.
    """
    profiles = store.load_all()
    if len(profiles) < 2:
        print_warning("Need at least 2 regular profiles to create a balancing profile")
        raise ValueError("Need at least 2 regular profiles")

    names = [p.name for p in profiles]
    selected = CheckboxMenu(
        "Select member profiles (select ≥ 2)",
        names,
    ).show()

    if not selected or len(selected) < 2:
        count = len(selected) if selected is not None else 0
        print_error(f"Balancing profile requires at least 2 members (got {count})")
        raise ValueError(f"Need at least 2 members, got {count}")

    # Profile name
    is_first = len(store.get_all_backends()) == 0
    while True:
        name = prompt_text("Enter balancing profile name (leave empty for 'balancer'): ")
        name = "balancer" if not name or not name.strip() else name.strip().lower()
        if name in RESERVED_NAMES:
            print_error(f"Profile name {name!r} is reserved")
            continue
        if not _NAME_PATTERN.match(name):
            print_error("Name must be lowercase, 1-32 chars, alphanumeric/dash/underscore")
            continue
        if store.get_backend(name) is not None:
            print_error(f"Profile {name!r} already exists")
            continue
        break

    # Set as default?
    set_default = is_first or prompt_confirm("Set as default profile?", default=True)

    bp = BalancingProfile(
        name=name,
        members=selected,
        is_default=set_default,
    )
    store.save(bp)
    print_status(f"Balancing profile {name!r} created with members: {', '.join(selected)}")
    return bp


def _edit_profile_flow(store: ProfileStore, cred_store: CredentialStore, profile_name: str) -> None:
    """Interactive flow for editing an existing regular profile.

    Allows changing the model name, the API key (copy-on-write), or both.

    Args:
        store: Profile store.
        cred_store: Credential store.
        profile_name: Name of the profile to edit.
    """
    profile = store.get(profile_name)
    if profile is None:
        print_error(f"Profile {profile_name!r} not found")
        return

    field = SelectionMenu(
        f"Edit profile {profile_name!r} — what to change?",
        ["Model", "API Key", "Both"],
    ).show()
    if field is None:
        return

    updates: dict = {}

    if field in ("Model", "Both"):
        new_model = prompt_text(f"New model name (current: {profile.model}): ")
        if new_model and new_model.strip():
            updates["model"] = new_model.strip()

    if field in ("API Key", "Both"):
        new_key = prompt_secret("Enter new API key: ")
        if new_key:
            # Copy-on-write: always create a fresh auth_ref for the edited profile.
            # The old credential entry is left untouched so other profiles sharing
            # the same auth_ref continue to work.
            new_auth_ref = str(uuid.uuid4())
            cred_store.set(new_auth_ref, new_key)
            updates["auth_ref"] = new_auth_ref

    if updates:
        updated = profile.model_copy(update=updates)
        store.save(updated)
        print_status(f"Profile {profile_name!r} updated")
    else:
        print_info("No changes made")


def _edit_balancing_flow(store: ProfileStore, balancing_name: str) -> None:
    """Interactive flow for editing a balancing profile's member list.

    Shows all regular profiles as a checkbox with current members pre-checked.
    Requires ≥2 members to save.

    Args:
        store: Profile store.
        balancing_name: Name of the balancing profile to edit.
    """
    backend = store.get_backend(balancing_name)
    if not isinstance(backend, BalancingProfile):
        print_error(f"Balancing profile {balancing_name!r} not found")
        return

    regular_profiles = store.load_all()
    all_names = [p.name for p in regular_profiles]

    new_members = CheckboxMenu(
        f"Edit members of {balancing_name!r} (select ≥ 2)",
        all_names,
        default_checked=list(backend.members),
    ).show()

    if new_members is None:
        print_info("Cancelled")
        return

    if len(new_members) < 2:
        print_error(f"Balancing profile requires at least 2 members (got {len(new_members)})")
        return

    updated = backend.model_copy(update={"members": new_members})
    store.save(updated)
    print_status(f"Balancing profile {balancing_name!r} updated with members: {', '.join(new_members)}")


def _delete_profile_flow(store: ProfileStore, cred_store: CredentialStore) -> None:
    """Interactive flow for deleting a profile.

    Cleans up the associated credential when no other profile shares the same auth_ref.
    """
    backends = store.get_all_backends()
    if not backends:
        print_warning("No profiles to delete")
        return

    names = [b.name for b in backends]
    selected = SelectionMenu("Select profile to delete", names).show()
    if selected is None:
        return

    # Check if any balancing profiles reference this one
    referencing = [b.name for b in backends if isinstance(b, BalancingProfile) and selected in b.members]
    if referencing:
        print_error(
            f"Cannot delete {selected!r} — it is a member of balancing profile(s): "
            f"{', '.join(referencing)}. Remove it from those profiles first."
        )
        return

    if prompt_confirm(f"Delete profile {selected!r}?", default=False):
        # Clean up credential if no other profile shares the same auth_ref
        target = store.get_backend(selected)
        store.delete(selected)
        if isinstance(target, Profile):
            remaining = [
                b for b in backends
                if isinstance(b, Profile) and b.name != selected and b.auth_ref == target.auth_ref
            ]
            if not remaining:
                cred_store.delete(target.auth_ref)
        print_status(f"Profile {selected!r} deleted")
    else:
        print_info("Cancelled")


def _set_default_flow(store: ProfileStore) -> None:
    """Interactive flow for setting the default profile."""
    backends = store.get_all_backends()
    if not backends:
        print_warning("No profiles available")
        return

    names = [b.name for b in backends]
    selected = SelectionMenu("Select default profile", names).show()
    if selected is None:
        return

    backend = store.get_backend(selected)
    if backend is None:
        return

    updated = backend.model_copy(update={"is_default": True})
    store.save(updated)
    print_status(f"Profile {selected!r} set as default")
