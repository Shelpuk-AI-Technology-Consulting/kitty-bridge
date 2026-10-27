"""Shutdown-signal behaviour of the bridge entry points (issue #302).

On Windows no asyncio event loop supports ``loop.add_signal_handler``; it
raises ``NotImplementedError``. The bridge used to call it unguarded, so
``kitty bridge`` printed its banner and then died. CI runs on Linux only, so
these tests reproduce the Windows condition by making the running loop class
raise the same error, and they stand in for Ctrl+C by raising
``KeyboardInterrupt`` from a loop callback (which is how Windows delivers it).

Two groups:

* the shared helper ``install_stop_signal_handlers``, tested with fake loops;
* the three entry points (``_run_bridge``, ``_run_bridge_balancing``,
  ``bridge_runner.main``), tested end to end against a stubbed server.
"""

import asyncio
import signal
import sys
from types import SimpleNamespace

import pytest

from kitty.bridge.shutdown import install_stop_signal_handlers


class _UnsupportedLoop:
    """Loop double whose signal registration fails like a Windows loop."""

    def add_signal_handler(self, sig: int, callback: object) -> None:
        """Reject registration the way every Windows event loop does.

        Args:
            sig: The signal number being registered.
            callback: The callback that would have been registered.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError


class _RecordingLoop:
    """Loop double that records signal registrations like a POSIX loop."""

    def __init__(self) -> None:
        """Start with no registered handlers."""
        self.registrations: list[tuple[int, object]] = []

    def add_signal_handler(self, sig: int, callback: object) -> None:
        """Remember each registration, in order, including repeats.

        Args:
            sig: The signal number being registered.
            callback: The callback to run when the signal arrives.
        """
        self.registrations.append((sig, callback))


class TestInstallStopSignalHandlers:
    """The helper wires SIGINT/SIGTERM to the stop event where the loop allows it."""

    def test_install_on_unsupported_loop_does_not_raise(self) -> None:
        """R1: a Windows-style loop must not make registration fail."""
        install_stop_signal_handlers(_UnsupportedLoop(), asyncio.Event())  # type: ignore[arg-type]

    def test_install_on_supported_loop_registers_sigint_and_sigterm(self) -> None:
        """R3: POSIX behaviour is unchanged -- each signal is registered exactly once."""
        loop = _RecordingLoop()

        install_stop_signal_handlers(loop, asyncio.Event())  # type: ignore[arg-type]

        assert sorted(sig for sig, _ in loop.registrations) == sorted([signal.SIGINT, signal.SIGTERM])

    @pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
    def test_signal_callback_sets_the_stop_event(self, sig: int) -> None:
        """R3: delivering either signal is what releases the shutdown wait."""
        loop = _RecordingLoop()
        stop_event = asyncio.Event()
        install_stop_signal_handlers(loop, stop_event)  # type: ignore[arg-type]

        dict(loop.registrations)[sig]()  # type: ignore[operator]

        assert stop_event.is_set()


@pytest.fixture
def loop_without_signal_support(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the loop class ``asyncio.run`` will use behave like a Windows loop.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    probe = asyncio.new_event_loop()
    loop_class = type(probe)
    probe.close()

    def unsupported(self: object, *args: object, **kwargs: object) -> None:
        """Reject registration the way every Windows event loop does."""
        raise NotImplementedError

    monkeypatch.setattr(loop_class, "add_signal_handler", unsupported, raising=False)


def _press_ctrl_c_soon() -> None:
    """Schedule a ``KeyboardInterrupt`` for the next loop turn after ``start_async``.

    It fires once the entry point has finished its signal setup and is waiting
    -- the moment a user would press Ctrl+C. If the entry point already failed,
    the interrupt is skipped so that failure reaches the test: otherwise
    ``asyncio.run`` would run the pending callback while shutting down and
    ``suppress(KeyboardInterrupt)`` would hide the real error.
    """
    entry_point_task = asyncio.current_task()

    def press_ctrl_c() -> None:
        """Raise the interrupt unless the entry point has already failed."""
        if entry_point_task is not None and entry_point_task.done():
            return
        raise KeyboardInterrupt

    asyncio.get_running_loop().call_soon(press_ctrl_c)


def _profile_stub(name: str = "test") -> SimpleNamespace:
    """Build a minimal single-profile stand-in for bridge construction.

    Args:
        name: The profile name.

    Returns:
        An object carrying the attributes the entry points read.
    """
    return SimpleNamespace(
        name=name,
        provider="zai_regular",
        model="test-model",
        auth_ref="ref-1",
        provider_config={},
        backup=False,
    )


def _stub_shared_prerequisites(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise the catalog refresh, egress guard and provider lookup.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    import kitty.providers.model_context_sync as sync

    async def no_refresh(**kwargs: object) -> bool:
        """Skip the network call to the model-context catalog."""
        return True

    monkeypatch.setattr(sync, "refresh_model_context_overrides", no_refresh)
    monkeypatch.setattr("kitty.egress_guard.egress_block_reason", lambda *a, **k: None)
    monkeypatch.setattr("kitty.providers.registry.get_provider", lambda *a, **k: object())


def _stub_bridge_server_class(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace ``BridgeServer`` start/stop with recorders and a simulated Ctrl+C.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        A list that receives ``"start"`` and ``"stop"`` as they happen.
    """
    from kitty.bridge.server import BridgeServer

    calls: list[str] = []

    async def start_async(self: object) -> int:
        """Record the start and schedule the simulated Ctrl+C."""
        calls.append("start")
        _press_ctrl_c_soon()
        return 4242

    async def stop_async(self: object) -> None:
        """Record the cleanup."""
        calls.append("stop")

    monkeypatch.setattr(BridgeServer, "start_async", start_async)
    monkeypatch.setattr(BridgeServer, "stop_async", stop_async)
    return calls


@pytest.mark.usefixtures("loop_without_signal_support")
class TestEntryPointsOnLoopWithoutSignalSupport:
    """R1/R2: every bridge entry point survives a Windows-style loop and stops cleanly."""

    def test_single_profile_bridge_stops_cleanly_on_ctrl_c(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``kitty bridge`` keeps running until Ctrl+C, then cleans up and exits 0."""
        import kitty.cli.main as cli_main

        _stub_shared_prerequisites(monkeypatch)
        calls = _stub_bridge_server_class(monkeypatch)
        cred_store = SimpleNamespace(get=lambda ref: "sk-test-key")

        with pytest.raises(SystemExit) as exit_info:
            cli_main._run_bridge(_profile_stub(), cred_store, validate=False)

        assert (exit_info.value.code, calls) == (0, ["start", "stop"])

    def test_balancing_bridge_stops_cleanly_on_ctrl_c(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``kitty bridge`` with a balancing profile behaves the same way."""
        import kitty.cli.main as cli_main

        member = _profile_stub("member-1")

        class FakeProfileStore:
            """Stand-in for ``ProfileStore``; never touches disk."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                """Accept and ignore any construction arguments."""

        class FakeProfileResolver:
            """Stand-in resolver that returns one balancing member."""

            def __init__(self, store: object) -> None:
                """Accept and ignore the store.

                Args:
                    store: The profile store (unused).
                """

            def resolve_balancing(self, name: str) -> list[SimpleNamespace]:
                """Return the single member profile.

                Args:
                    name: The balancing profile name (unused).

                Returns:
                    A list holding the one member.
                """
                return [member]

        monkeypatch.setattr("kitty.profiles.store.ProfileStore", FakeProfileStore)
        monkeypatch.setattr("kitty.profiles.resolver.ProfileResolver", FakeProfileResolver)
        _stub_shared_prerequisites(monkeypatch)
        calls = _stub_bridge_server_class(monkeypatch)
        cred_store = SimpleNamespace(get=lambda ref: "sk-test-key")

        with pytest.raises(SystemExit) as exit_info:
            cli_main._run_bridge_balancing(SimpleNamespace(name="pool"), cred_store, validate=False)

        assert (exit_info.value.code, calls) == (0, ["start", "stop"])

    def test_background_bridge_runner_stops_cleanly_on_ctrl_c(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``kitty bridge start`` (``bridge_runner.main``) behaves the same way."""
        import kitty.bridge_runner as runner

        profile = _profile_stub()
        calls: list[str] = []

        class FakeBridgeServer:
            """Stand-in server recording start/stop and simulating Ctrl+C."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                """Accept and ignore any construction arguments."""

            async def start_async(self) -> int:
                """Record the start and schedule the simulated Ctrl+C.

                Returns:
                    A fake port number.
                """
                calls.append("start")
                _press_ctrl_c_soon()
                return 4242

            async def stop_async(self) -> None:
                """Record the cleanup."""
                calls.append("stop")

        class FakeProfileStore:
            """Stand-in store whose only backend is the profile stub."""

            def get_backend(self, name: str) -> SimpleNamespace:
                """Return the profile stub.

                Args:
                    name: The requested profile name (unused).

                Returns:
                    The single profile stub.
                """
                return profile

        class FakeProfileResolver:
            """Stand-in resolver that picks the profile stub as default."""

            def __init__(self, store: object) -> None:
                """Accept and ignore the store.

                Args:
                    store: The profile store (unused).
                """

            def resolve_default_backend(self) -> SimpleNamespace:
                """Return the profile stub.

                Returns:
                    The single profile stub.
                """
                return profile

        class FakeCredentialStore:
            """Stand-in credential store holding one fixed key."""

            def __init__(self, backends: object = None) -> None:
                """Accept and ignore the backends.

                Args:
                    backends: Credential backends (unused).
                """

            def get(self, ref: str) -> str:
                """Return the fixed test key.

                Args:
                    ref: The credential reference (unused).

                Returns:
                    A fixed key.
                """
                return "sk-test-key"

        monkeypatch.setattr(sys, "argv", ["kitty.bridge_runner"])
        monkeypatch.setattr(runner, "BridgeServer", FakeBridgeServer)
        monkeypatch.setattr("kitty.profiles.store.ProfileStore", FakeProfileStore)
        monkeypatch.setattr("kitty.profiles.resolver.ProfileResolver", FakeProfileResolver)
        monkeypatch.setattr("kitty.credentials.store.CredentialStore", FakeCredentialStore)
        monkeypatch.setattr("kitty.credentials.file_backend.FileBackend", object)
        monkeypatch.setattr("kitty.egress_store.resolve_egress", lambda cred_store=None: None)
        _stub_shared_prerequisites(monkeypatch)

        runner.main()

        assert calls == ["start", "stop"]
