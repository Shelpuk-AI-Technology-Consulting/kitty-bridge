"""``kitty bridge start`` applies the Claude-only eligibility gate.

The background bridge runner is the second process entry point; the
foreground ``kitty bridge`` paths gate subscription profiles before any
credential read (see ``tests/test_cli_main.py``), and this runner must not
stay a sibling arm that still answers a subscription profile with a
misleading "No API key". Like the foreground tests, the gate is pinned
before credential resolution: the fake store raises if consulted for an
ineligible profile.
"""

from __future__ import annotations

import sys
import uuid

import pytest

from kitty.profiles.schema import BalancingProfile, Profile

# ── Fixtures ───────────────────────────────────────────────────────────────


def _subscription_profile(name: str = "sub") -> Profile:
    return Profile(
        name=name,
        provider="anthropic_subscription",
        model="claude-sonnet-4-6",
        auth_ref=str(uuid.uuid4()),
    )


def _api_key_profile(name: str = "metered") -> Profile:
    return Profile(
        name=name,
        provider="zai_regular",
        model="glm-5.2",
        auth_ref="00000000-0000-4000-8000-000000000001",
    )


@pytest.fixture
def start_env(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch every collaborator ``bridge_runner.main`` reads.

    Mirrors the egress start-path fixture: egress resolves to a fixed config,
    the profile store hands back ``backend_holder["backend"]``, the credential
    store raises if consulted for a ref not in ``allowed_refs``, and the
    server is a spy recording construction kwargs.
    """
    from kitty import bridge_runner
    from kitty.egress import EgressConfig

    egress = EgressConfig(proxy_url="http://proxy.example:1234", username="u", password="s3cr3tpw")
    monkeypatch.setattr("kitty.egress_store.resolve_egress", lambda **kwargs: egress)
    monkeypatch.setattr("kitty.egress._egress", egress, raising=False)
    monkeypatch.setattr("kitty.egress_guard.egress_block_reason", lambda *a, **k: None, raising=False)

    captured: dict = {"server_kwargs": []}
    state: dict = {"backend": None, "members": [], "allowed_refs": []}

    class _FakeProfileStore:
        def get_backend(self, name: str | None):
            return state["backend"]

    class _FakeCredentialStore:
        def __init__(self, backends: object = None) -> None:
            pass

        def get(self, ref: str) -> str:
            if ref not in state["allowed_refs"]:
                raise AssertionError(f"credential store consulted for {ref!r}")
            return "sk-test-key"

    class _BridgeServerSpy:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["server_kwargs"].append(kwargs)
            raise _WiringSentinel

    class _FakeResolver:
        def __init__(self, store: object) -> None:
            pass

        def resolve_balancing(self, name: str):
            return state["members"]

        def resolve_default_backend(self):
            return None

    monkeypatch.setattr("kitty.profiles.store.ProfileStore", _FakeProfileStore)
    monkeypatch.setattr("kitty.credentials.store.CredentialStore", _FakeCredentialStore)
    monkeypatch.setattr("kitty.profiles.resolver.ProfileResolver", _FakeResolver)
    monkeypatch.setattr(bridge_runner, "BridgeServer", _BridgeServerSpy)
    monkeypatch.setattr(bridge_runner.asyncio, "run", lambda coro: coro.close())
    monkeypatch.setattr(sys, "argv", ["bridge_runner", "--profile", "any"])

    return {"captured": captured, "state": state}


class _WiringSentinel(Exception):
    """Abort the runner once the server is constructed."""


# ── Tests ──────────────────────────────────────────────────────────────────


class TestBridgeRunnerSubscriptionEligibility:
    def test_direct_subscription_profile_refused_before_credential_lookup(
        self, start_env: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A subscription profile is refused with the launcher message, no key read."""
        from kitty import bridge_runner

        start_env["state"]["backend"] = _subscription_profile()
        start_env["state"]["allowed_refs"] = []

        with pytest.raises(SystemExit) as exc_info:
            bridge_runner.main()

        assert exc_info.value.code == 1
        flattened = " ".join(capsys.readouterr().err.split())
        assert "requires the Claude Code launcher" in flattened
        assert "No API key" not in flattened
        assert start_env["captured"]["server_kwargs"] == []

    def test_all_subscription_pool_refused_with_the_no_members_message(
        self, start_env: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An all-subscription pool has no member for the background bridge."""
        from kitty import bridge_runner

        start_env["state"]["backend"] = BalancingProfile(name="bg-pool", members=["sub", "sub-2"])
        start_env["state"]["members"] = [_subscription_profile()]
        start_env["state"]["allowed_refs"] = []

        with pytest.raises(SystemExit) as exc_info:
            bridge_runner.main()

        assert exc_info.value.code == 1
        assert "no members eligible" in capsys.readouterr().err
        assert start_env["captured"]["server_kwargs"] == []

    def test_mixed_pool_wires_only_the_eligible_member(
        self, start_env: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Subscription members are skipped before their credentials are read."""
        from kitty import bridge_runner

        start_env["state"]["backend"] = BalancingProfile(name="bg-pool", members=["sub", "metered"])
        start_env["state"]["members"] = [_subscription_profile(), _api_key_profile()]
        start_env["state"]["allowed_refs"] = ["00000000-0000-4000-8000-000000000001"]
        monkeypatch.setattr(
            "kitty.providers.registry.get_provider", lambda *a, **k: object(), raising=False
        )

        with pytest.raises(_WiringSentinel):
            bridge_runner.main()

        kwargs = start_env["captured"]["server_kwargs"][0]
        wired = [backend[2].name for backend in kwargs["backends"]]
        assert wired == ["metered"]
