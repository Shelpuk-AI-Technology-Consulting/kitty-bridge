"""Sentinel tests: bridge-only entry points refresh the catalog before start.

Every session-start entry point must await
``model_context_sync.refresh_model_context_overrides()`` before the bridge
server starts (AC3.4). These tests cover the three bridge-only modes that
never spawn an agent: ``cli/main.py _run_bridge``, ``cli/main.py
_run_bridge_balancing``, and the nested ``run()`` inside
``bridge_runner.main()`` (the background/service bridge). ``launch_async``
is covered in ``test_launch_orchestrator.py``.

The pattern: patch ``refresh_model_context_overrides`` at its source module
(entry points call it through module-attribute access) with a recorder, stub
``BridgeServer.start_async`` to record and raise a sentinel, then assert the
recorded order is ``["refresh", "start"]``. The sentinel guarantees the
assertion runs against the exact moment the server would have started — no
test ever runs a real server.
"""

from types import SimpleNamespace

import pytest


class _Sentinel(Exception):
    """Raised by stubbed ``start_async`` implementations to stop the flow."""


def _fake_refresh_factory(order: list) -> object:
    """Build a refresh stand-in that appends ``"refresh"`` to ``order``.

    Args:
        order: Shared list recording the call sequence.

    Returns:
        An async callable matching ``refresh_model_context_overrides``.
    """

    async def fake_refresh(**kwargs):
        order.append("refresh")
        return True

    return fake_refresh


def _fake_start_async_factory(order: list) -> object:
    """Build a ``start_async`` stub that appends ``"start"`` then raises.

    Args:
        order: Shared list recording the call sequence.

    Returns:
        An async function with the ``BridgeServer.start_async`` signature.
    """

    async def fake_start_async(self):
        order.append("start")
        raise _Sentinel

    return fake_start_async


def _make_profile_stub() -> SimpleNamespace:
    """Build a minimal single-profile stand-in for bridge construction."""
    return SimpleNamespace(
        name="test",
        provider="zai_regular",
        model="test-model",
        auth_ref="ref-1",
        provider_config={},
        backup=False,
    )


class TestRunBridgeRefresh:
    """``kitty bridge`` with a single profile refreshes before server start."""

    def test_refresh_awaited_before_server_start(self, monkeypatch: pytest.MonkeyPatch):
        """The single-profile bridge mode refreshes the catalog before start."""
        import kitty.cli.main as cli_main
        import kitty.providers.model_context_sync as sync
        from kitty.bridge.server import BridgeServer

        order: list = []
        monkeypatch.setattr(sync, "refresh_model_context_overrides", _fake_refresh_factory(order))
        monkeypatch.setattr(BridgeServer, "start_async", _fake_start_async_factory(order))
        monkeypatch.setattr("kitty.egress_guard.egress_block_reason", lambda *a, **k: None)
        monkeypatch.setattr("kitty.cli.main.egress_block_reason", lambda *a, **k: None, raising=False)
        monkeypatch.setattr("kitty.providers.registry.get_provider", lambda *a, **k: object())
        monkeypatch.setattr("kitty.cli.main.get_provider", lambda *a, **k: object(), raising=False)

        cred_store = SimpleNamespace(get=lambda ref: "sk-test-key")

        with pytest.raises(_Sentinel):
            cli_main._run_bridge(_make_profile_stub(), cred_store, validate=False)

        assert order == ["refresh", "start"]


class TestRunBridgeBalancingRefresh:
    """``kitty bridge`` with a balancing profile refreshes before start."""

    def test_refresh_awaited_before_server_start(self, monkeypatch: pytest.MonkeyPatch):
        """The balancing bridge mode refreshes the catalog before start."""
        import kitty.cli.main as cli_main
        import kitty.providers.model_context_sync as sync
        from kitty.bridge.server import BridgeServer

        order: list = []
        member = _make_profile_stub()
        member.name = "member-1"

        class FakeProfileStore:
            """Stand-in for ProfileStore — constructed inside the function."""

            def __init__(self, *args, **kwargs):
                pass

        class FakeProfileResolver:
            """Stand-in returning one balancing member."""

            def __init__(self, store):
                pass

            def resolve_balancing(self, name):
                """Return the single member profile."""
                return [member]

        monkeypatch.setattr(sync, "refresh_model_context_overrides", _fake_refresh_factory(order))
        monkeypatch.setattr(BridgeServer, "start_async", _fake_start_async_factory(order))
        monkeypatch.setattr("kitty.profiles.store.ProfileStore", FakeProfileStore)
        monkeypatch.setattr("kitty.profiles.resolver.ProfileResolver", FakeProfileResolver)
        monkeypatch.setattr("kitty.egress_guard.egress_block_reason", lambda *a, **k: None)
        monkeypatch.setattr("kitty.cli.main.egress_block_reason", lambda *a, **k: None, raising=False)
        monkeypatch.setattr("kitty.providers.registry.get_provider", lambda *a, **k: object())
        monkeypatch.setattr("kitty.cli.main.get_provider", lambda *a, **k: object(), raising=False)

        balancing = SimpleNamespace(name="balancing-profile")
        cred_store = SimpleNamespace(get=lambda ref: "sk-test-key")

        with pytest.raises(_Sentinel):
            cli_main._run_bridge_balancing(balancing, cred_store, validate=False)

        assert order == ["refresh", "start"]


class TestBridgeRunnerRefresh:
    """The background/service bridge (``bridge_runner.main``) refreshes too."""

    def test_refresh_awaited_before_server_start(self, monkeypatch: pytest.MonkeyPatch):
        """The background/service bridge refreshes the catalog before start."""
        import sys

        import kitty.bridge_runner as runner
        import kitty.providers.model_context_sync as sync

        order: list = []
        profile = _make_profile_stub()

        class FakeBridgeServer:
            """Stand-in server whose start_async raises the sentinel."""

            def __init__(self, *args, **kwargs):
                pass

            async def start_async(self):
                order.append("start")
                raise _Sentinel

        class FakeProfileStore:
            """Stand-in returning the profile as the default backend."""

            def __init__(self):
                pass

            def get_backend(self, name):
                """Return the single profile stub."""
                return profile

        class FakeProfileResolver:
            """Stand-in resolver returning the profile as default backend."""

            def __init__(self, store):
                pass

            def resolve_default_backend(self):
                """Return the single profile stub."""
                return profile

        class FakeCredentialStore:
            """Stand-in credential store with one fixed key."""

            def __init__(self, backends=None):
                pass

            def get(self, ref):
                """Return the fixed test key."""
                return "sk-test-key"

        monkeypatch.setattr(sys, "argv", ["kitty.bridge_runner"])
        monkeypatch.setattr(sync, "refresh_model_context_overrides", _fake_refresh_factory(order))
        monkeypatch.setattr(runner, "BridgeServer", FakeBridgeServer)
        monkeypatch.setattr("kitty.profiles.store.ProfileStore", FakeProfileStore)
        monkeypatch.setattr("kitty.profiles.resolver.ProfileResolver", FakeProfileResolver)
        monkeypatch.setattr("kitty.credentials.store.CredentialStore", FakeCredentialStore)
        monkeypatch.setattr("kitty.credentials.file_backend.FileBackend", object)
        monkeypatch.setattr("kitty.egress_store.resolve_egress", lambda cred_store=None: None)
        monkeypatch.setattr("kitty.egress_guard.egress_block_reason", lambda *a, **k: None)
        monkeypatch.setattr("kitty.bridge_runner.egress_block_reason", lambda *a, **k: None, raising=False)
        monkeypatch.setattr("kitty.providers.registry.get_provider", lambda *a, **k: object())
        monkeypatch.setattr("kitty.bridge_runner.get_provider", lambda *a, **k: object(), raising=False)

        with pytest.raises(_Sentinel):
            runner.main()

        assert order == ["refresh", "start"]
