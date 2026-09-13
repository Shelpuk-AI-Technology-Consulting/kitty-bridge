"""Tests that the foreground ``kitty bridge`` loops stop through the shared stop-signal helper.

KBR-220 moved every bridge loop onto
:func:`kitty.bridge.stop_signals.install_stop_handlers`, because a direct
``loop.add_signal_handler`` call crashes on Windows.
``tests/test_stop_handler_call_sites.py`` proves no loop calls that API directly.
It cannot prove a loop still registers stop handlers at all: a loop that
dropped the call would pass it, and on POSIX would then die on SIGTERM without
running ``stop_async``. These tests drive ``_run_bridge`` and
``_run_bridge_balancing`` with a stubbed server, have the helper deliver a stop
at once, and check that shutdown ran. The background runner's loop is proven
end to end in ``tests/cli/test_bridge_state_location.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

# Only reached when a loop registers no stop handlers; long enough never to race a passing run.
_GIVE_UP_SECONDS = 2.0


def _member() -> SimpleNamespace:
    """Build a minimal profile stand-in for bridge construction.

    Returns:
        An object with the attributes the bridge loops read from a profile.
    """
    return SimpleNamespace(
        name="member", provider="zai_regular", model="test-model", auth_ref="ref-1", provider_config={}, backup=False
    )


@pytest.fixture
def stop_at_once(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub the server and deliver a stop signal the moment handlers are installed.

    Args:
        monkeypatch: pytest's monkeypatch fixture.

    Returns:
        The events in order: ``"handlers"`` when the helper was called with the
        running loop, ``"stopped"`` when ``stop_async`` ran.
    """
    import kitty.bridge.stop_signals as stop_signals
    import kitty.providers.model_context_sync as sync
    from kitty.bridge.server import BridgeServer

    events: list[str] = []

    def _install(loop: asyncio.AbstractEventLoop, callback) -> bool:
        """Record the call, then behave as if SIGTERM had just arrived."""
        if loop is asyncio.get_running_loop():
            events.append("handlers")
        callback()
        return True

    async def _no_refresh(**_kwargs) -> bool:
        """Skip the catalog fetch."""
        return True

    async def _start(self) -> int:
        """Pretend to listen, and end a loop no stop signal can reach instead of letting it hang."""
        task = asyncio.current_task()
        assert task is not None

        def _give_up() -> None:
            """Record that nothing stopped the loop, then cancel it."""
            events.append("no stop handlers")
            task.cancel()

        asyncio.get_running_loop().call_later(_GIVE_UP_SECONDS, _give_up)
        return 4000

    async def _stop(self) -> None:
        """Record that shutdown ran."""
        events.append("stopped")

    monkeypatch.setattr(stop_signals, "install_stop_handlers", _install)
    monkeypatch.setattr(sync, "refresh_model_context_overrides", _no_refresh)
    monkeypatch.setattr(BridgeServer, "start_async", _start)
    monkeypatch.setattr(BridgeServer, "stop_async", _stop)
    monkeypatch.setattr("kitty.egress_guard.egress_block_reason", lambda *a, **k: None)
    monkeypatch.setattr("kitty.providers.registry.get_provider", lambda *a, **k: object())
    return events


def test_the_single_profile_bridge_shuts_down_when_a_stop_signal_arrives(stop_at_once: list[str]) -> None:
    """``kitty bridge`` with one profile registers stop handlers, and a stop runs ``stop_async``."""
    import kitty.cli.main as cli_main

    with pytest.raises((SystemExit, asyncio.CancelledError)) as exited:
        cli_main._run_bridge(_member(), SimpleNamespace(get=lambda ref: "sk-test-key"), validate=False)

    assert (getattr(exited.value, "code", None), stop_at_once) == (0, ["handlers", "stopped"])


def test_the_balancing_bridge_shuts_down_when_a_stop_signal_arrives(
    stop_at_once: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``kitty bridge`` with a balancing profile registers stop handlers, and a stop runs ``stop_async``."""
    import kitty.cli.main as cli_main

    class _Resolver:
        """Resolve the balancing profile to one member."""

        def __init__(self, store) -> None:
            """Ignore the store."""

        def resolve_balancing(self, name):
            """Return the single member."""
            return [_member()]

    monkeypatch.setattr("kitty.profiles.store.ProfileStore", lambda *a, **k: object())
    monkeypatch.setattr("kitty.profiles.resolver.ProfileResolver", _Resolver)

    with pytest.raises((SystemExit, asyncio.CancelledError)) as exited:
        cli_main._run_bridge_balancing(
            SimpleNamespace(name="pool"), SimpleNamespace(get=lambda ref: "sk-test-key"), validate=False
        )

    assert (getattr(exited.value, "code", None), stop_at_once) == (0, ["handlers", "stopped"])
