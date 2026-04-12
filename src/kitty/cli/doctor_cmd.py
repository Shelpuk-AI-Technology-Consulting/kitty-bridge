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
from kitty.tui.display import LiveChecklist, print_error, print_section, print_status, print_warning

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

    # Specific target validation — uses simple sequential output
    if target_name is not None:
        print_section("kitty doctor")
        failures += _check_target(target_name)
        return failures

    # Specific profile validation — uses simple sequential output
    if profile_name is not None:
        print_section("kitty doctor")
        failures += _check_profile(store, cred_store, profile_name)
        return failures

    # Full check — uses LiveChecklist for in-place updates
    resolver = ProfileResolver(store)
    profiles = resolver.list_profiles()

    checks = []

    # Target binary checks
    for name in _ADAPTERS:
        checks.append((f"Target {name!r}", _make_target_check(name)))

    # Early exit: no profiles
    if not profiles:
        print_warning("No profiles configured — run 'kitty setup' to get started")
        return 1

    # Default profile check
    checks.append(("Default profile", _make_default_profile_check(resolver)))

    # Per-profile credential checks
    for profile in profiles:
        checks.append(
            (f"Credentials for {profile.name!r}", _make_credential_check(cred_store, profile))
        )

    checklist = LiveChecklist("kitty doctor")
    failures = checklist.run_checks(checks)

    if failures == 0:
        print_section("All checks passed")
    else:
        print_section(f"{failures} check(s) failed")

    return failures


def _make_target_check(name: str):
    """Create a check function for a launcher target binary."""
    def check() -> tuple[bool, str]:
        if name not in _ADAPTERS:
            return False, f"Unknown target: {name!r}"
        binary_path = discover_binary(name)
        if binary_path is not None:
            return True, f"binary found at {binary_path}"
        return False, "binary not found on PATH or common install directories"
    return check


def _make_default_profile_check(resolver: ProfileResolver):
    """Create a check function for the default profile."""
    def check() -> tuple[bool, str]:
        try:
            default = resolver.resolve_default()
            return True, default.name
        except NoDefaultProfileError:
            return False, "No default profile set"
    return check


def _make_credential_check(cred_store: CredentialStore, profile: Profile):
    """Create a check function for a profile's credentials."""
    def check() -> tuple[bool, str]:
        key = cred_store.get(profile.auth_ref)
        if key is not None:
            return True, f"resolved ({len(key)} chars)"
        return False, f"auth_ref {profile.auth_ref!r} not found"
    return check


def _check_target(name: str) -> int:
    """Check if a launcher target binary is available (sequential output).

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
    """Validate a specific profile (sequential output).

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
    key = cred_store.get(profile.auth_ref)
    if key is not None:
        print_status(f"  Credentials for {profile.name!r}: resolved ({len(key)} chars)")
    else:
        print_error(f"  Credentials for {profile.name!r}: auth_ref {profile.auth_ref!r} not found")
        failures += 1

    return failures
