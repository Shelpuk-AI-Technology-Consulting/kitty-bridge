"""Egress gateway command — interactive setup, testing and removal.

``kitty egress`` opens a menu; ``kitty egress test`` and ``kitty egress show``
are the non-interactive equivalents for scripts.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from kitty.credentials.store import CredentialStore
from kitty.egress import EgressConfig, parse_proxy_url
from kitty.egress_store import EgressRecord, EgressStore, resolve_egress
from kitty.tui.display import (
    print_error,
    print_info,
    print_section,
    print_status,
    print_table,
    print_warning,
    status_spinner,
)
from kitty.tui.menu import SelectionMenu
from kitty.tui.prompts import NonTTYError, check_tty, prompt_confirm, prompt_secret, prompt_text

logger = logging.getLogger(__name__)

__all__ = ["run_egress_menu", "run_egress_show", "run_egress_test"]

#: Echo service used to report the address the upstream actually sees.
IP_ECHO_URL = "https://api.ipify.org"

_TEST_TIMEOUT_SECONDS = 15


def run_egress_menu(cred_store: CredentialStore, store: EgressStore | None = None) -> None:
    """Interactive gateway management.

    Args:
        cred_store: Credential store holding the proxy password.
        store: Gateway store. Defaults to the standard location.

    Raises:
        NonTTYError: If not running in an interactive terminal.
    """
    check_tty()
    store = store or EgressStore()

    while True:
        record = store.load()
        print_section("Egress Gateway")
        _print_status_block(record)

        actions = ["Configure gateway"]
        if record is not None:
            actions.extend(["Test connection", "Remove gateway"])
        actions.append("Back")

        choice = SelectionMenu("Actions", actions).show()
        if choice is None or choice == "Back":
            break

        try:
            if choice == "Configure gateway":
                _configure_flow(store, cred_store)
            elif choice == "Test connection":
                _run_test(_resolve_or_warn(store, cred_store))
            elif choice == "Remove gateway":
                _remove_flow(store, cred_store)
        except (NonTTYError, ValueError, KeyboardInterrupt) as exc:
            print_info(f"Cancelled: {exc}")


def run_egress_show(cred_store: CredentialStore, store: EgressStore | None = None) -> int:
    """Print the resolved gateway without prompting.

    Args:
        cred_store: Credential store holding the proxy password.
        store: Gateway store. Defaults to the standard location.

    Returns:
        Process exit code: 0 when a gateway is configured, 1 otherwise.
    """
    print_section("Egress Gateway")
    config = _resolve_or_warn(store or EgressStore(), cred_store)
    if config is None:
        return 1
    print_table(["Setting", "Value"], _config_rows(config))
    return 0


def run_egress_test(cred_store: CredentialStore, store: EgressStore | None = None) -> int:
    """Check the gateway and report the address upstreams will see.

    Args:
        cred_store: Credential store holding the proxy password.
        store: Gateway store. Defaults to the standard location.

    Returns:
        Process exit code: 0 if the gateway works, 1 otherwise.
    """
    print_section("Egress Gateway")
    config = _resolve_or_warn(store or EgressStore(), cred_store)
    if config is None:
        return 1
    return 0 if _run_test(config) else 1


# ── Internals ────────────────────────────────────────────────────────────


def _print_status_block(record: EgressRecord | None) -> None:
    """Render the current gateway, or a hint when none is configured."""
    if record is None:
        print_info("No gateway configured — all traffic leaves from this machine's own IP.")
        return
    auth = f"as {record.username}" if record.username else "no authentication"
    print_status(f"Gateway: {record.proxy_url} ({auth})")


def _config_rows(config: EgressConfig) -> list[list[str]]:
    """Build display rows for a resolved gateway, password masked."""
    return [
        ["Proxy", config.proxy_url],
        ["Username", config.username or "(none)"],
        ["Password", "****" if config.password else "(none)"],
    ]


def _resolve_or_warn(store: EgressStore, cred_store: CredentialStore) -> EgressConfig | None:
    """Resolve the effective gateway, reporting why it is unavailable.

    Args:
        store: Gateway store.
        cred_store: Credential store holding the proxy password.

    Returns:
        The resolved configuration, or None when none is usable.
    """
    try:
        config = resolve_egress(store=store, cred_store=cred_store)
    except ValueError as exc:
        print_error(str(exc))
        return None
    if config is None:
        print_warning("No egress gateway is configured.")
        print_info("Run 'kitty egress' and choose 'Configure gateway' to set one up.")
    return config


def _prompt_proxy_url() -> str:
    """Ask for the proxy endpoint until it parses.

    Returns:
        A normalised ``scheme://host:port`` URL.

    Raises:
        NonTTYError: If the prompt is cancelled.
    """
    while True:
        raw = prompt_text("Proxy address (e.g. proxy.iproyal.com:12323): ")
        if not raw or not raw.strip():
            print_error("Proxy address is required")
            continue
        candidate = raw.strip()
        # Most proxy vendors quote a bare host:port; accept that spelling.
        if "://" not in candidate:
            candidate = f"http://{candidate}"
        try:
            return parse_proxy_url(candidate).proxy_url
        except ValueError as exc:
            print_error(str(exc))


def _configure_flow(store: EgressStore, cred_store: CredentialStore) -> None:
    """Guide the user through setting up a gateway and verifying it.

    Args:
        store: Gateway store to write to.
        cred_store: Credential store to hold the password.
    """
    proxy_url = _prompt_proxy_url()

    username: str | None = None
    auth_ref: str | None = None
    password: str | None = None
    if prompt_confirm("Does this proxy require a username and password?", default=True):
        while True:
            entered = prompt_text("Proxy username: ")
            if entered and entered.strip():
                username = entered.strip()
                break
            print_error("Username is required when the proxy is authenticated")
        while True:
            password = prompt_secret("Proxy password: ")
            if password:
                break
            print_error("Password is required when the proxy is authenticated")
        auth_ref = str(uuid.uuid4())

    previous = store.load()
    if auth_ref is not None and password is not None:
        cred_store.set(auth_ref, password)
    store.save(EgressRecord(proxy_url=proxy_url, username=username, auth_ref=auth_ref))

    # Drop the superseded password so it does not linger in the store.
    if previous is not None and previous.auth_ref and previous.auth_ref != auth_ref:
        cred_store.delete(previous.auth_ref)

    print_status(f"Gateway saved: {proxy_url}")

    if prompt_confirm("Test the gateway now?", default=True):
        _run_test(EgressConfig(proxy_url=proxy_url, username=username, password=password))


def _remove_flow(store: EgressStore, cred_store: CredentialStore) -> None:
    """Delete the stored gateway and its password after confirmation.

    Args:
        store: Gateway store to clear.
        cred_store: Credential store holding the password.
    """
    record = store.load()
    if record is None:
        print_info("No gateway to remove")
        return
    if not prompt_confirm(f"Remove gateway {record.proxy_url}?", default=False):
        print_info("Cancelled")
        return

    store.delete()
    if record.auth_ref:
        cred_store.delete(record.auth_ref)
    print_status("Gateway removed — traffic will leave from this machine's own IP again.")


def _run_test(config: EgressConfig | None) -> bool:
    """Send one request through the gateway and report what upstream sees.

    Args:
        config: Gateway to exercise, or None to report nothing configured.

    Returns:
        True if the gateway responded successfully.
    """
    if config is None:
        return False

    with status_spinner("Contacting the gateway..."):
        observed_ip, elapsed_ms, error = asyncio.run(_probe(config))

    rows = _config_rows(config)
    if error is None:
        rows.append(["Connectivity", "OK"])
        rows.append(["Public IP", observed_ip or "(unknown)"])
        rows.append(["Latency", f"{elapsed_ms} ms"])
        print_table(["Setting", "Value"], rows)
        print_status(f"Upstream providers will see {observed_ip} for every kitty install using this gateway.")
        return True

    rows.append(["Connectivity", "FAILED"])
    print_table(["Setting", "Value"], rows)
    print_error(f"Could not reach {IP_ECHO_URL} through the gateway: {error}")
    print_info("Check the address, port and credentials, and that the proxy allows HTTPS CONNECT.")
    return False


async def _probe(config: EgressConfig) -> tuple[str | None, int, str | None]:
    """Fetch the observed public address through the proxy.

    Args:
        config: Gateway to use.

    Returns:
        ``(ip, elapsed_ms, error)`` where ``error`` is None on success.
    """
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=_TEST_TIMEOUT_SECONDS)
    started = time.monotonic()
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout, proxy=config.proxy_url, proxy_auth=config.auth) as session,
            session.get(IP_ECHO_URL) as resp,
        ):
            body = (await resp.text()).strip()
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if resp.status != 200:
                return None, elapsed_ms, f"HTTP {resp.status}"
            return body, elapsed_ms, None
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim to the user
        return None, int((time.monotonic() - started) * 1000), f"{type(exc).__name__}: {exc}"
