"""Doctor command — diagnostics and health checks for the kitty environment."""

from __future__ import annotations

from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
from kitty.launchers.claude import ClaudeAdapter
from kitty.launchers.codex import CodexAdapter
from kitty.launchers.discovery import discover_binary
from kitty.launchers.kilo import KiloAdapter
from kitty.profiles.resolver import NoDefaultProfileError, ProfileResolver
from kitty.profiles.schema import Profile
from kitty.profiles.store import ProfileStore
from kitty.providers.registry import get_provider
from kitty.tui.display import print_error, print_section, print_status, print_warning

__all__ = ["run_doctor"]

_ADAPTERS = {
    "codex": CodexAdapter(),
    "claude": ClaudeAdapter(),
    "kilo": KiloAdapter(),
}


def run_doctor(
    store: ProfileStore,
    target_name: str | None = None,
    profile_name: str | None = None,
) -> int:
    """Run doctor diagnostics.

    Args:
        store: Profile store to validate.
        target_name: Optional specific launcher target to validate.
        profile_name: Optional specific profile to validate.

    Returns:
        0 if all checks pass, non-zero on failure.
    """
    failures = 0
    cred_store = CredentialStore(backends=[FileBackend()])

    print_section("kitty doctor")

    # Specific target validation
    if target_name is not None:
        failures += _check_target(target_name)
        return failures

    # Specific profile validation
    if profile_name is not None:
        failures += _check_profile(store, cred_store, profile_name)
        return failures

    # Full check: all targets
    for name in _ADAPTERS:
        failures += _check_target(name)

    # Check profiles
    resolver = ProfileResolver(store)
    profiles = resolver.list_profiles()

    if not profiles:
        print_warning("No profiles configured — run 'kitty setup' to get started")
        failures += 1
        return failures

    # Check default profile
    try:
        default = resolver.resolve_default()
        print_status(f"Default profile: {default.name}")
    except NoDefaultProfileError:
        print_error("No default profile set")
        failures += 1

    # Check each profile's credentials
    for profile in profiles:
        failures += _check_profile_credentials(cred_store, profile)

    if failures == 0:
        print_section("All checks passed")
    else:
        print_section(f"{failures} check(s) failed")

    return failures


def _check_target(name: str) -> int:
    """Check if a launcher target binary is available.

    Returns:
        0 if found, 1 if missing.
    """
    if name not in _ADAPTERS:
        print_error(f"Unknown target: {name!r}")
        return 1

    binary_path = discover_binary(name)
    if binary_path is not None:
        print_status(f"Target {name!r}: binary found at {binary_path}")
        return 0

    print_error(f"Target {name!r}: binary not found on PATH or common install directories")
    return 1


def _check_profile(store: ProfileStore, cred_store: CredentialStore, name: str) -> int:
    """Validate a specific profile.

    Returns:
        0 if valid, 1 if issues found.
    """
    failures = 0
    profile = store.get(name)
    if profile is None:
        print_error(f"Profile {name!r} not found")
        return 1

    print_status(f"Profile {name!r}: found (provider={profile.provider}, model={profile.model})")

    # Check provider is resolvable
    try:
        get_provider(profile.provider)
        print_status(f"  Provider {profile.provider!r}: resolvable")
    except KeyError:
        print_error(f"  Provider {profile.provider!r}: unknown provider type")
        failures += 1

    # Check credentials
    failures += _check_profile_credentials(cred_store, profile)

    return failures


def _check_profile_credentials(cred_store: CredentialStore, profile: Profile) -> int:
    """Check that a profile's credentials resolve.

    Returns:
        0 if credential found, 1 if not.
    """
    key = cred_store.get(profile.auth_ref)
    if key is not None:
        print_status(f"  Credentials for {profile.name!r}: resolved ({len(key)} chars)")
        return 0

    print_error(f"  Credentials for {profile.name!r}: auth_ref {profile.auth_ref!r} not found")
    return 1
