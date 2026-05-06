"""CLI command: kitty auth openai — run the OAuth flow for ChatGPT subscription."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from kitty.auth.oauth_session import OAuthSession
from kitty.auth.openai_oauth import run_oauth_flow
from kitty.credentials.store import CredentialStore
from kitty.profiles.schema import Profile
from kitty.profiles.store import ProfileStore
from kitty.tui.display import print_error, print_section, print_status, print_warning
from kitty.tui.prompts import check_tty, prompt_confirm, prompt_text

if TYPE_CHECKING:
    pass

__all__ = ["run_auth_openai", "run_oauth_for_provider"]


async def run_oauth_for_provider(
    profile_store: ProfileStore, cred_store: CredentialStore, provider: str
) -> tuple[str, str]:
    """Run the OAuth flow for a provider and persist the session.

    Generic entry point that both the ``kitty auth`` command and the
    profile-creation wizards can call to handle browser-based SSO.

    Args:
        profile_store: Profile store (used to derive config directory).
        cred_store: Credential store to save the session file path.
        provider: Provider type string (currently only ``"openai_subscription"``).

    Returns:
        ``(auth_ref, session_file_path)`` tuple.

    Raises:
        NonTTYError: If not running in an interactive terminal.
        Exception: If the OAuth flow fails.
    """
    check_tty()
    print_status("Opening browser for ChatGPT login...")

    try:
        session = await run_oauth_flow()
    except Exception as exc:
        print_error(f"OpenAI OAuth flow failed: {exc}")
        raise

    # Generate auth_ref and persist session to disk
    auth_ref = str(uuid.uuid4())
    config_dir = profile_store._path.parent  # type: ignore[attr-defined]
    session = OAuthSession.create_session_file(session, auth_ref, config_dir)

    session_file_path = Path(session._file_path)  # type: ignore[assignment]
    cred_store.set(auth_ref, str(session_file_path))

    return auth_ref, str(session_file_path)


async def run_auth_openai(profile_store: ProfileStore, cred_store: CredentialStore) -> Profile:
    """Run the OpenAI OAuth flow and create a profile.

    Args:
        profile_store: Profile store to save the new profile.
        cred_store: Credential store to store the session file path.

    Returns:
        The created and saved Profile.

    Raises:
        NonTTYError: If not running in an interactive terminal.
    """
    check_tty()
    print_section("kitty auth openai")

    # Step 1: Run OAuth flow (shared helper)
    auth_ref, _session_path = await run_oauth_for_provider(profile_store, cred_store, "openai_subscription")

    # Step 2: Prompt for model name
    model = prompt_text("Model name (default: gpt-5.3-codex): ")
    model = "gpt-5.3-codex" if not model or not model.strip() else model.strip()

    # Step 3: Prompt for profile name
    profile_name = prompt_text("Profile name (default: openai-sub): ")
    profile_name = "openai-sub" if not profile_name or not profile_name.strip() else profile_name.strip().lower()

    # Step 4: Ask if should be set as default
    is_default = prompt_confirm("Set as default profile?", default=True)

    # Step 5: Create profile
    profile = Profile(
        name=profile_name,
        provider="openai_subscription",  # type: ignore[arg-type]
        model=model,
        auth_ref=auth_ref,
        is_default=is_default,
    )

    # Step 6: Validate (best-effort; custom-transport providers short-circuit)
    try:
        from kitty.providers.registry import get_provider
        from kitty.validation import validate_api_key

        provider = get_provider("openai_subscription")
        validation_result = await validate_api_key(provider, _session_path, {})
        if not validation_result.valid:
            print_warning(f"API key validation failed: {validation_result.reason}")
        elif validation_result.warning:
            print_warning(validation_result.warning)
    except Exception as exc:
        print_warning(f"API key validation skipped: {exc}")

    # Step 7: Save profile
    profile_store.save(profile)
    print_status(f"Profile {profile.name!r} created successfully")

    return profile
