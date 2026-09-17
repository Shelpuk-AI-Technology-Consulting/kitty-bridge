"""R2 — Fail-closed on a real start path (T-E8, KBR-68).

``.system_design/TEST_SUITE.md`` §6.2.3 (per-entry-point enforcement) and §5.5 claim 2 ·
plan task **T-E8** ·
`.requirements/20260916T120000Z_kbr68_t_e8_local_bypass_fail_closed_transport_asymmetry/REQUIREMENTS.md`.

**Scope (recorded here so a reader does not infer T-E8 over-delivers).** This module
exercises ONE representative ``egress_block_reason`` call site — the bridge-runner
single-profile start path at ``src/kitty/bridge_runner.py:179-183`` — on a Bedrock-SSO-shaped
profile. The five start paths §6.2.3 names
(``bridge_runner.py`` ×2, ``cli/launcher.py``, ``cli/main.py`` ×2) are T-E6's coverage;
T-E8's R2 is the user-facing message-naming claim on one of them.

**The user-facing claim.** With egress configured and a profile whose adapter reports
``supports_egress() == False`` (Bedrock in SSO mode), the launch is refused with a non-zero
exit, no ``BridgeServer`` listening socket is bound, and the stderr message names the
profile. The same path with a compliant adapter starts a server (control). The
falsification proves the test is sensitive to the exact defect §6.2.3 names:
the ``egress_block_reason(...)`` call is preserved but its return value is discarded.

**Layer.** No ``pytestmark``, so these take the ``l1`` path default. The drive is
process-level — ``bridge_runner.main`` parses argv, reads the profile store, prints to
stderr, and exits — and the assertion surfaces (SystemExit code, stderr text, presence
of a ``BridgeServer.__init__`` call) are observable from the test process with no live
dependency, which is what L1 is for. The same default-layer choice is what the T-E2
slice takes for its L3-shaped phases (§8.2: ``l3`` is in ``PENDING_ACTIVATION_LAYERS``,
so an ``l3`` marker today would leave this slice's correctness checked by no job at all).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from kitty.egress import EgressConfig
from kitty.providers.bedrock import BedrockAdapter

_PROFILE_NAME = "bedrock-sso"
"""Profile name used throughout the module; asserted in AC2.1's stderr check."""


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def start_path(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch every collaborator ``bridge_runner.main`` reads so one test can drive it.

    The collaborators are: ``ProfileStore`` (line 102), ``CredentialStore`` (line 103),
    ``BridgeServer`` (line 185), and ``asyncio.run`` (line 220) — the last replaced by a
    no-op so the test never tries to actually start the server. The fixture also pins
    process-wide egress and ``sys.argv`` so argparse reads a clean argv.

    Returns:
        A dict the tests consult: ``captured`` records ``BridgeServer.__init__`` calls;
        ``key_holder`` is a mutable whose ``["key"]`` is what the fake credential store
        returns (the control swaps the SSO marker for a non-SSO AWS key); ``monkeypatch``
        is re-exposed so the AC2.3 falsification can patch an extra symbol.
    """
    from kitty import bridge_runner

    egress = EgressConfig(proxy_url="http://proxy.example:1234", username="u", password="s3cr3tpw")
    # `main()` re-resolves egress from the environment or the real on-disk store
    # (line 113), so the fixture patches the resolver itself — a patch on
    # `kitty.egress._egress` is overwritten on the very next line of `main()`.
    monkeypatch.setattr("kitty.egress_store.resolve_egress", lambda **kwargs: egress)
    monkeypatch.setattr("kitty.egress._egress", egress, raising=False)

    captured = {"bridge_server_init_calls": 0}
    key_holder = {"key": "sso"}

    class _FakeProfileStore:
        def get_backend(self, name: str):
            return SimpleNamespace(
                name=_PROFILE_NAME,
                provider="bedrock",
                provider_config={"region": "us-east-1"},
                auth_ref="dummy-ref",
                model="anthropic.claude-3-sonnet",
            )

    class _FakeCredentialStore:
        def __init__(self, backends: object = None) -> None:
            pass

        def get(self, ref: str) -> str:
            return key_holder["key"]

    class _BridgeServerSpy:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["bridge_server_init_calls"] += 1

        async def start_async(self) -> int:
            raise SystemExit(0)

        async def stop_async(self) -> None:
            pass

    monkeypatch.setattr("kitty.profiles.store.ProfileStore", _FakeProfileStore)
    monkeypatch.setattr("kitty.credentials.store.CredentialStore", _FakeCredentialStore)
    # `BridgeServer` is imported at bridge_runner module level (line 16), so
    # patch the binding bridge_runner resolves against, not the source module.
    monkeypatch.setattr(bridge_runner, "BridgeServer", _BridgeServerSpy)
    monkeypatch.setattr(bridge_runner.asyncio, "run", lambda coro: coro.close())
    monkeypatch.setattr(sys, "argv", ["bridge_runner", "--profile", _PROFILE_NAME])

    return {"captured": captured, "key_holder": key_holder, "monkeypatch": monkeypatch, "egress": egress}


# ── AC2.1 — refusing profile exits non-zero, names profile, no socket ──────


class TestRefusingProfile:
    """AC2.1: a Bedrock-SSO-shaped profile under egress is refused, not started."""

    def test_bridge_runner_single_profile_path_exits_non_zero_no_socket_stderr_names_profile(
        self,
        start_path: dict,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """SystemExit, stderr names profile, ``BridgeServer.__init__`` never reached.

        The guard call at ``bridge_runner.py:180`` returns a string; the ``if _block:``
        branch on line 181 prints to stderr and ``sys.exit(1)``. The bridge is never
        constructed (line 185 is unreachable on this path).
        """
        from kitty import bridge_runner

        with pytest.raises(SystemExit) as exc_info:
            bridge_runner.main()

        assert exc_info.value.code != 0, (
            f"expected non-zero exit (refusing guard fires), got code={exc_info.value.code}"
        )

        err = capsys.readouterr().err
        assert _PROFILE_NAME in err, f"stderr does not name the refusing profile {_PROFILE_NAME!r}: {err!r}"
        assert start_path["egress"].password not in err, "the proxy password must never reach a user-facing message"

        assert start_path["captured"]["bridge_server_init_calls"] == 0, (
            f"BridgeServer.__init__ was reached {start_path['captured']['bridge_server_init_calls']} "
            "time(s) under a refusing profile: a listening socket would have been bound"
        )


# ── AC2.2 — compliant control proceeds past the guard ──────────────────────


class TestCompliantControl:
    """AC2.2: a non-SSO Bedrock key reaches ``BridgeServer.__init__`` and starts a server.

    The control proves the refusal in AC2.1 is the guard's branch and not a fixture
    defect (a fixture that refused everything would pass AC2.1 vacuously).
    """

    def test_non_sso_aws_key_proceeds_past_the_guard_to_bridge_server(self, start_path: dict) -> None:
        """A non-SSO AWS key reaches ``BridgeServer.__init__`` exactly once.

        Bedrock's :meth:`is_sso_mode` returns ``False`` for any AWS access-key-shaped
        credential, so ``supports_egress`` returns ``True``, so the guard returns
        ``None``, so the bridge is constructed.
        """
        start_path["key_holder"]["key"] = "AKIAEXAMPLE:wJalrXUtnFEMI/K7MDENG"

        from kitty import bridge_runner

        bridge_runner.main()

        assert start_path["captured"]["bridge_server_init_calls"] == 1, (
            "BridgeServer.__init__ was not reached with a compliant adapter — the guard "
            "refused a provider that should be allowed (AC2.2 control failed)"
        )


# ── AC2.3 — falsification: keep the call, discard the return value ──────────


class TestReturnDiscardedFalsification:
    """AC2.3: the suite detects the exact §6.2.3 defect (call present, return discarded)."""

    def test_discarding_the_guard_return_value_reaches_bridge_server(
        self,
        start_path: dict,
    ) -> None:
        """The mutation makes the assertion surface prove AC2.1 is sensitive.

        The bridge-runner code is patched so :func:`egress_block_reason` returns
        ``None`` — the call is present (the §6.2.3 mutation is "discard", not "delete"),
        but the ``if _block:`` predicate on line 181 never fires. ``BridgeServer.__init__``
        is then reached. This test asserts that fact; AC2.1's assertion
        (``init_calls == 0``) would fail under this same mutation, which is the
        relationship §6.2.3 asks the suite to demonstrate.
        """
        from kitty import bridge_runner

        # `egress_block_reason` is imported inside `main()` (line 109), so patch
        # the source module — `bridge_runner` does not bind the symbol at
        # module scope.
        start_path["monkeypatch"].setattr("kitty.egress_guard.egress_block_reason", lambda *a, **kw: None)

        bridge_runner.main()

        assert start_path["captured"]["bridge_server_init_calls"] == 1, (
            "BridgeServer.__init__ was not reached when the guard returned None — the "
            "AC2.1 assertion ('init_calls == 0') would not fail under this mutation, "
            "which is exactly the defect §6.2.3 asks the suite to detect"
        )

        # Sanity: the patch took effect on the symbol bridge_runner resolves against.
        assert BedrockAdapter().supports_egress("sso", {}) is False, (
            "the SSO-mode supports_egress check must still return False — the patch is "
            "on bridge_runner's *use*, not on BedrockAdapter"
        )
