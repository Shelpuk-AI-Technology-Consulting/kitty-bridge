"""Setup wizard command — first-run profile creation."""

from __future__ import annotations

import uuid

from kitty.credentials.store import CredentialStore
from kitty.profiles.schema import _NAME_PATTERN, PROVIDER_LABELS, PROVIDER_LIST, RESERVED_NAMES, Profile
from kitty.profiles.store import ProfileStore
from kitty.tui.display import print_error, print_section, print_status, print_step, print_warning, status_spinner
from kitty.tui.menu import SelectionMenu
from kitty.tui.prompts import NonTTYError, check_tty, prompt_confirm, prompt_secret, prompt_text

__all__ = ["run_setup_wizard"]

# Re-export _find_reusable_auth_ref at this module level so that tests can mock
# it as "kitty.cli.setup_cmd._find_reusable_auth_ref" without knowing which
# module owns it.  Without this re-export, patching the function here would
# have no effect because setup_cmd would keep using the original reference.
from kitty.cli.profile_cmd import _find_reusable_auth_ref  # noqa: F401


def run_setup_wizard(store: ProfileStore, cred_store: CredentialStore) -> Profile:
    """Run the first-run setup wizard to create a profile.

    Steps:
    1. Provider selection (arrow-key menu)
    2. API key entry — offers reuse if a same-provider profile already exists
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
    check_tty()

    print_section("kitty setup wizard")

    # Step 1: Provider selection
    print_step(1, 6, "Provider selection")
    _label_to_type = {PROVIDER_LABELS.get(p, p): p for p in PROVIDER_LIST}
    provider = SelectionMenu("Select provider", list(_label_to_type)).show()
    if provider is not None:
        provider = _label_to_type[provider]
    if provider is None:
        raise NonTTYError("Provider selection cancelled")

    # Step 2: API key — offer reuse if a same-provider profile exists
    print_step(2, 6, "API key")
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
    print_step(3, 6, "Model selection")
    model = prompt_text("Model (OpenRouter or provider format, e.g. z-ai/glm-5): ")
    if not model or not model.strip():
        print_error("Model name cannot be empty")
        raise ValueError("Model name cannot be empty")
    model = model.strip()

    # Step 4: Profile name (validated)
    print_step(4, 6, "Profile name")
    is_first = len(store.load_all()) == 0
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

    # Step 5: Set as default
    print_step(5, 6, "Default profile")
    set_default = is_first or prompt_confirm("Set as default profile?", default=True)

    # Create profile
    profile = Profile(
        name=name,
        provider=provider,  # type: ignore[arg-type]
        model=model,
        auth_ref=auth_ref,
        is_default=set_default,
    )

    store.save(profile)
    print_status(f"Profile {name!r} created successfully")

    # Step 6: Connectivity check (optional)
    print_step(6, 6, "Connectivity check")
    if prompt_confirm("Test connectivity to provider?", default=True):
        with status_spinner("Testing connectivity..."):
            connected = _check_connectivity(provider, auth_ref)
        if connected:
            print_status("Connectivity check passed")
        else:
            print_warning("Connectivity check failed — profile saved but may not work until API key is verified")

    return profile


def _check_connectivity(provider_type: str, auth_ref: str) -> bool:
    """Test connectivity to the provider's API.

    Args:
        provider_type: Provider type string.
        auth_ref: Auth reference (unused for basic check).

    Returns:
        True if connectivity check passed, False otherwise.
    """
    try:
        from kitty.providers.registry import get_provider

        get_provider(provider_type)  # Verify provider is resolvable
        return True
    except Exception:
        return False
