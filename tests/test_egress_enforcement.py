"""Each start path must enforce the egress guard, not merely consult it.

``egress_block_reason()`` *returns* a reason; it stops nothing. The enforcement
is the branch that follows — ``if _block: sys.exit(1)`` — in each of kitty's
five start paths. Plan task T-E6 (design: TEST_SUITE.md §6.2.3, plan §8): a
variant that keeps the guard call and discards its return value must make these
tests fail, or the suite cannot distinguish enforcement from decoration.

The mechanism: ``BridgeServer`` is replaced at every import site with a stub
whose ``__init__`` records the attempt and raises :class:`_EnforcementSentinel`.
Each enforcement test drives a real start path with a rejecting configuration
(egress gateway configured + a provider whose ``supports_egress()`` is
``False``) and asserts the path exited non-zero *and* never constructed a
server — no construction, no ``start_async()``, no listening socket.

Why the sentinel and not a socket scan: binding a port happens inside
``start_async()``, after construction, so "never constructed" is strictly
stronger than "no listening socket" and is portable across the Linux, Windows
and macOS legs (a live socket inventory is not).

Falsification. Plan §1.4 requires the harness to detect a deliberate defect
running in the suite, not just to argue the point.
``TestDiscardedGuardIsCaught`` (parametrised over all five paths) is that
test: it patches both guard bindings to a no-op — the runtime shape of "guard
call kept, return value discarded" — and asserts the sentinel fires on every
path. A variant that deleted the ``if _block:`` branch produces the same
effect at the seam: the AC1 tests fail because ``_EnforcementSentinel``
escapes ``pytest.raises(SystemExit)``.

That the sentinel is *reachable* — the AC1 negative assertion is not vacuously
green — is shown by the reachability controls (``TestFalsificationControl``,
proxyable provider → sentinel fires) for the launcher and background-runner
paths, and by the foreground ``kitty bridge`` paths' positive controls in
``tests/test_cli_main.py`` (``_WiringSentinel`` raised when the guard is told
to allow). ``TestDiscardedGuardIsCaught`` also reaches the sentinel on those
three paths, so its reachability proof is complete.

The unproxyable provider is a stub, not ``BedrockAdapter`` in SSO mode: which
adapters cannot be proxied is the guard's *decision* and is unit-tested in
``tests/test_egress_fail_closed.py``. This file proves that decision is
obeyed, so it needs only a provider that answers ``False``.

Nothing reaches upstream by construction — the guard fires before any client
exists — so no recorder and no network are involved (plan §8, T-E6 row).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
from kitty.egress import EgressConfig, set_egress
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import BalancingProfile, Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# L3: real start paths driven in-process, with only the server's construction
# stubbed out. Mutation testing does not judge this module (it is the
# enforcement check the mutants must survive), but the marker keeps it out of
# the L1 set whose commands assume millisecond units.
pytestmark = pytest.mark.l3

EGRESS = EgressConfig(proxy_url="http://proxy.example.com:12323", username="myuser", password="s3cr3tpw")

#: How a container or CI install configures egress (kitty.egress.ENV_PROXY).
#: ``bridge_runner`` resolves egress itself from this variable; the in-process
#: paths take ``set_egress`` directly.
EGRESS_ENV_VALUE = EGRESS.proxy_url


class _EnforcementSentinel(Exception):
    """Raised from a stubbed ``BridgeServer.__init__`` when enforcement is absent."""


class _UnproxyableProvider(ProviderAdapter):
    """A provider whose transport cannot honour the egress proxy."""

    @property
    def provider_type(self) -> str:
        return "unproxyable"

    @property
    def default_base_url(self) -> str:
        return "https://api.example.invalid/v1"

    def supports_egress(self, resolved_key: str, provider_config: dict) -> bool:
        return False

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(str(status_code))


class _ProxyableProvider(_UnproxyableProvider):
    """The falsification control's provider: the guard allows this one."""

    @property
    def provider_type(self) -> str:
        return "proxyable"

    def supports_egress(self, resolved_key: str, provider_config: dict) -> bool:
        return True


class _StubLauncher(LauncherAdapter):
    """Minimal launcher: the guard fires long before a child is spawned."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        """Never consulted before the guard fires."""
        return BridgeProtocol.RESPONSES_API

    @property
    def binary_name(self) -> str:
        return "echo"

    def build_spawn_config(
        self,
        profile: Profile,
        bridge_port: int,
        resolved_key: str,
        *,
        context_tokens: int | None = None,
    ) -> SpawnConfig:
        del profile, bridge_port, resolved_key, context_tokens
        return SpawnConfig(cli_args=[], env_overrides={}, env_clear=[])


def _patch_bridge_sentinel(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace ``BridgeServer`` at every import site with a recording sentinel.

    ``bridge_runner`` and ``cli/launcher`` hold module-level ``from ... import``
    bindings, so patching the defining module alone would not rebind them; the
    three sites are patched explicitly.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        A list receiving one kwargs dict per construction attempt; empty when
        no server was ever built.
    """
    seen: list[dict] = []

    class _SentinelServer:
        """Stand-in for ``BridgeServer`` that records and aborts."""

        def __init__(self, *args, **kwargs):
            """Record the attempt, then stop the flow."""
            seen.append(kwargs)
            raise _EnforcementSentinel

    monkeypatch.setattr("kitty.bridge.server.BridgeServer", _SentinelServer)
    monkeypatch.setattr("kitty.bridge_runner.BridgeServer", _SentinelServer, raising=False)
    monkeypatch.setattr("kitty.cli.launcher.BridgeServer", _SentinelServer, raising=False)
    return seen


def _profile_stub(name: str = "my-profile") -> SimpleNamespace:
    """Build a single-profile stand-in carrying what the start paths read.

    Args:
        name: Profile name; named in the guard's rejection message.

    Returns:
        A namespace shaped like :class:`kitty.profiles.schema.Profile`.
    """
    return SimpleNamespace(
        name=name,
        provider="unproxyable",
        model="test-model",
        auth_ref="ref-1",
        provider_config={},
        backup=False,
    )


def _cred_store() -> CredentialStore:
    """Return a credential store that resolves every reference to a key.

    Returns:
        A ``CredentialStore`` over a mocked file backend.
    """
    backend = MagicMock(spec=FileBackend)
    backend.get = MagicMock(return_value="sk-test-key")
    return CredentialStore(backends=[backend])


def _patch_profile_store(monkeypatch: pytest.MonkeyPatch, backend: object) -> None:
    """Make ``ProfileStore`` return ``backend`` without touching the disk.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        backend: What ``get_backend`` reports for any profile name.
    """

    class _FakeProfileStore:
        """Stand-in for ``ProfileStore`` — constructed inside the entry point."""

        def __init__(self, *args, **kwargs):
            """Accept and ignore the real store's arguments."""

        def get_backend(self, name):
            """Return the nominated backend."""
            return backend

    monkeypatch.setattr("kitty.profiles.store.ProfileStore", _FakeProfileStore)


def _patch_balancing_resolver(monkeypatch: pytest.MonkeyPatch, members: list) -> None:
    """Make ``ProfileResolver.resolve_balancing`` return ``members``.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        members: The member profiles the balancing path will build backends from.
    """

    class _FakeResolver:
        """Stand-in for ``ProfileResolver``."""

        def __init__(self, store):
            """Accept and ignore the store."""

        def resolve_balancing(self, name):
            """Return the balancing members."""
            return members

        def resolve_default_backend(self):
            """Never consulted — a profile name is always given."""
            return None

    monkeypatch.setattr("kitty.profiles.resolver.ProfileResolver", _FakeResolver)


def _patch_provider_registry(monkeypatch: pytest.MonkeyPatch, provider: ProviderAdapter) -> None:
    """Make every ``get_provider`` call return ``provider``.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        provider: The adapter the start paths will build backends from.
    """
    monkeypatch.setattr("kitty.providers.registry.get_provider", lambda *a, **k: provider)


def _patch_catalog_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``launch_async`` offline: its step 0 runs before the guard.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    from kitty.providers import model_context_sync

    async def _skip_refresh(**kwargs):
        """Report success without touching the network."""
        return True

    monkeypatch.setattr(model_context_sync, "refresh_model_context_overrides", _skip_refresh)


class TestLauncherEnforces:
    """`kitty <agent>` — the ``launch_async`` start path (cli/launcher.py)."""

    @pytest.mark.asyncio()
    async def test_rejecting_config_returns_nonzero_and_starts_no_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guard's rejection is obeyed: exit 1, no ``BridgeServer`` built."""
        from kitty.cli.launcher import launch_async

        set_egress(EGRESS)
        _patch_catalog_refresh(monkeypatch)
        seen = _patch_bridge_sentinel(monkeypatch)

        exit_code = await launch_async(
            adapter=_StubLauncher(),
            provider=_UnproxyableProvider(),
            profile=_profile_stub(),
            cred_store=_cred_store(),
            extra_args=[],
            validate=False,
        )

        assert exit_code == 1
        assert seen == [], "a server was constructed despite the guard rejecting the launch"


class TestForegroundBridgeEnforces:
    """`kitty bridge` — the two ``cli/main.py`` start paths."""

    def test_single_profile_rejecting_config_exits_nonzero_and_starts_no_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The single-profile path obeys the guard: ``sys.exit(1)``, no server."""
        import kitty.cli.main as cli_main

        set_egress(EGRESS)
        _patch_provider_registry(monkeypatch, _UnproxyableProvider())
        seen = _patch_bridge_sentinel(monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            cli_main._run_bridge(_profile_stub(), _cred_store(), validate=False)

        assert excinfo.value.code == 1
        assert seen == [], "a server was constructed despite the guard rejecting the launch"

    def test_balancing_rejecting_config_exits_nonzero_and_starts_no_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The balancing path obeys the guard: ``sys.exit(1)``, no server."""
        import kitty.cli.main as cli_main

        set_egress(EGRESS)
        _patch_provider_registry(monkeypatch, _UnproxyableProvider())
        # The store is constructed at the top of _run_bridge_balancing and its
        # __init__ mkdirs the real user config dir; both store and resolver
        # must be patched, as tests/test_cli_main.py's _balancing_stubs does.
        _patch_profile_store(monkeypatch, SimpleNamespace(name="ci-pool"))
        _patch_balancing_resolver(monkeypatch, [_profile_stub(name="member-1")])
        seen = _patch_bridge_sentinel(monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            cli_main._run_bridge_balancing(SimpleNamespace(name="ci-pool"), _cred_store(), validate=False)

        assert excinfo.value.code == 1
        assert seen == [], "a server was constructed despite the guard rejecting the launch"


class TestBackgroundRunnerEnforces:
    """`kitty bridge start` — the two ``bridge_runner.py`` start paths."""

    @staticmethod
    def _run_main(monkeypatch: pytest.MonkeyPatch, backend: object, members: list | None) -> None:
        """Drive ``bridge_runner.main()`` in-process for the given backend.

        Egress arrives through ``KITTY_EGRESS_PROXY`` — the environment path
        ``resolve_egress`` documents for containers and CI, and the only channel
        this entry point reads.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
            backend: What the profile store reports for ``--profile``.
            members: Balancing members to resolve, or ``None`` for the
                single-profile path.
        """
        import kitty.bridge_runner

        monkeypatch.setattr(sys, "argv", ["kitty.bridge_runner", "--profile", "ci-profile"])
        monkeypatch.setenv("KITTY_EGRESS_PROXY", EGRESS_ENV_VALUE)
        _patch_profile_store(monkeypatch, backend)
        if members is not None:
            _patch_balancing_resolver(monkeypatch, members)

        kitty.bridge_runner.main()

    @staticmethod
    def _patch_runner_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace the runner's stores so nothing reads the developer's config.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """

        class _FakeCredentialStore:
            """Stand-in for ``CredentialStore`` — no file backend, no keyring."""

            def __init__(self, *args, **kwargs):
                """Accept and ignore the real store's arguments."""

            def get(self, ref):
                """Resolve every reference to a key."""
                return "sk-test-key"

            def resolve(self, profile):
                """Resolve every profile to a key."""
                return "sk-test-key"

        monkeypatch.setattr("kitty.credentials.store.CredentialStore", _FakeCredentialStore)

    def test_single_profile_rejecting_config_exits_nonzero_and_starts_no_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The runner's single-profile path obeys the guard: no server starts."""
        self._patch_runner_credentials(monkeypatch)
        _patch_provider_registry(monkeypatch, _UnproxyableProvider())
        seen = _patch_bridge_sentinel(monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            self._run_main(monkeypatch, _profile_stub(), None)

        assert excinfo.value.code == 1
        assert seen == [], "a server was constructed despite the guard rejecting the launch"

    def test_balancing_rejecting_config_exits_nonzero_and_starts_no_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The runner's balancing path obeys the guard — every member is checked."""
        self._patch_runner_credentials(monkeypatch)
        _patch_provider_registry(monkeypatch, _UnproxyableProvider())
        seen = _patch_bridge_sentinel(monkeypatch)
        pool = BalancingProfile(name="ci-pool", members=["member-a", "member-b"])

        with pytest.raises(SystemExit) as excinfo:
            self._run_main(monkeypatch, pool, [_profile_stub(name="member-a"), _profile_stub(name="member-b")])

        assert excinfo.value.code == 1
        assert seen == [], "a server was constructed despite the guard rejecting the launch"


class TestFalsificationControl:
    """The enforcement tests' negative assertion is capable of failing.

    Driven with a *proxyable* provider — the guard returns ``None`` — each
    path falls through to ``BridgeServer(...)``, the sentinel fires, and the
    construction attempt is recorded. That is the detection the enforcement
    tests rely on: under a variant that discards the guard's return value the
    same fall-through happens with a *rejecting* configuration, and those
    tests fail. The foreground ``kitty bridge`` half of this control lives in
    ``tests/test_cli_main.py`` (``_WiringSentinel`` on the same seam).
    """

    @pytest.mark.asyncio()
    async def test_launcher_reaches_construction_when_the_guard_allows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``launch_async`` falls through to construction when nothing blocks it."""
        from kitty.cli.launcher import launch_async

        set_egress(EGRESS)
        _patch_catalog_refresh(monkeypatch)
        seen = _patch_bridge_sentinel(monkeypatch)

        with pytest.raises(_EnforcementSentinel):
            await launch_async(
                adapter=_StubLauncher(),
                provider=_ProxyableProvider(),
                profile=_profile_stub(),
                cred_store=_cred_store(),
                extra_args=[],
                validate=False,
            )

        assert len(seen) == 1, "the sentinel must be the thing that stops an allowed launch"

    def test_background_runner_reaches_construction_when_the_guard_allows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The runner falls through to construction when nothing blocks it."""
        TestBackgroundRunnerEnforces._patch_runner_credentials(monkeypatch)
        _patch_provider_registry(monkeypatch, _ProxyableProvider())
        seen = _patch_bridge_sentinel(monkeypatch)

        with pytest.raises(_EnforcementSentinel):
            TestBackgroundRunnerEnforces._run_main(monkeypatch, _profile_stub(), None)

        assert len(seen) == 1, "the sentinel must be the thing that stops an allowed launch"


def _drive_launcher_for_mutant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive ``launch_async`` synchronously so the dispatch stays uniform."""
    import asyncio

    from kitty.cli.launcher import launch_async

    asyncio.run(
        launch_async(
            adapter=_StubLauncher(),
            provider=_UnproxyableProvider(),
            profile=_profile_stub(),
            cred_store=_cred_store(),
            extra_args=[],
            validate=False,
        )
    )


def _drive_cli_main_single_for_mutant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive ``_run_bridge``."""
    import kitty.cli.main as cli_main

    cli_main._run_bridge(_profile_stub(), _cred_store(), validate=False)


def _drive_cli_main_balancing_for_mutant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive ``_run_bridge_balancing`` with one member."""
    import kitty.cli.main as cli_main

    # Store and resolver both patched: the store's __init__ mkdirs the real
    # user config dir, mirroring tests/test_cli_main.py's _balancing_stubs.
    _patch_profile_store(monkeypatch, SimpleNamespace(name="ci-pool"))
    _patch_balancing_resolver(monkeypatch, [_profile_stub(name="member-1")])
    cli_main._run_bridge_balancing(SimpleNamespace(name="ci-pool"), _cred_store(), validate=False)


def _drive_bridge_runner_single_for_mutant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive ``bridge_runner.main()`` for a single-profile launch."""
    TestBackgroundRunnerEnforces._patch_runner_credentials(monkeypatch)
    TestBackgroundRunnerEnforces._run_main(monkeypatch, _profile_stub(), None)


def _drive_bridge_runner_balancing_for_mutant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive ``bridge_runner.main()`` for a balancing launch."""
    TestBackgroundRunnerEnforces._patch_runner_credentials(monkeypatch)
    pool = BalancingProfile(name="ci-pool", members=["member-a", "member-b"])
    TestBackgroundRunnerEnforces._run_main(
        monkeypatch,
        pool,
        [_profile_stub(name="member-a"), _profile_stub(name="member-b")],
    )


class TestDiscardedGuardIsCaught:
    """Plan §1.4 / §6.2.3: a guard whose answer is discarded must not pass silently.

    The five enforcement tests above prove the honest answer is obeyed. This
    class proves the opposite: when the guard binding is patched to a no-op —
    the runtime shape of "call kept, return value discarded" — every start
    path falls through to ``BridgeServer(...)`` and the sentinel fires. A
    variant that deleted the ``if _block:`` branch would produce the same
    effect at the seam, and these tests would fail under it.

    Two guard-binding shapes need separate patches:
    - ``kitty.egress_guard.egress_block_reason`` covers the call-time imports
      in ``bridge_runner`` and ``cli/main``.
    - ``kitty.cli.launcher.egress_block_reason`` covers the module-level
      import in ``cli/launcher`` (its ``from kitty.egress_guard import``
      bound the name at import time).

    Both are patched in every case, so a single parametrisation covers all
    five paths and also discharges the bridge_runner balancing reachability
    that ``TestFalsificationControl`` does not cover.
    """

    @staticmethod
    def _patch_guard_noop(monkeypatch: pytest.MonkeyPatch) -> None:
        """Neutralise every guard binding the start paths consult.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.setattr(
            "kitty.egress_guard.egress_block_reason",
            lambda *a, **k: None,
        )
        monkeypatch.setattr(
            "kitty.cli.launcher.egress_block_reason",
            lambda *a, **k: None,
            raising=False,
        )

    @pytest.mark.parametrize(
        "drive",
        [
            pytest.param(_drive_launcher_for_mutant, id="launcher"),
            pytest.param(_drive_cli_main_single_for_mutant, id="cli_main_single"),
            pytest.param(_drive_cli_main_balancing_for_mutant, id="cli_main_balancing"),
            pytest.param(_drive_bridge_runner_single_for_mutant, id="bridge_runner_single"),
            pytest.param(_drive_bridge_runner_balancing_for_mutant, id="bridge_runner_balancing"),
        ],
    )
    def test_discarded_guard_falls_through_to_sentinel(
        self,
        drive,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every start path lets a neutered guard reach the sentinel."""
        seen = _patch_bridge_sentinel(monkeypatch)
        self._patch_guard_noop(monkeypatch)
        _patch_provider_registry(monkeypatch, _UnproxyableProvider())
        _patch_catalog_refresh(monkeypatch)
        set_egress(EGRESS)

        with pytest.raises(_EnforcementSentinel):
            drive(monkeypatch)

        assert len(seen) == 1, (
            f"the sentinel must catch a discarded guard on every start path; got {len(seen)} construction attempts"
        )
