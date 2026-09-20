"""Bind pytest-bdd's scenarios and define the acceptance step vocabulary (T-J1, KBR-107).

Two things live here, and one property of this file is load-bearing:

1. **The step definitions.** pytest-bdd's ``@given``/``@when``/``@then``
   decorators inject a pytest fixture into the *calling module's namespace*
   (``pytest-bdd``'s ``step()`` uses ``get_caller_module_locals``), and a
   scenario resolves a step by looking that fixture up through pytest's fixture
   manager. So the steps must live in a module **pytest collects** — this one,
   named ``test_*.py`` — not in a side module pytest never imports, where the
   injected fixtures are invisible.

2. **The ``scenarios()`` binding.** Every ``.feature`` under ``features/`` is
   bound recursively to a generated pytest test function. T-J2 and T-J3 add
   their ``.feature`` files and step definitions alongside this one's smoke; no
   further wiring is needed.

The relative order of the ``scenarios()`` call and the step decorators below it
is **not** load-bearing — both run during this module's import, before pytest
collects anything, and the first working version of this file had the
decorators first. What is load-bearing is the property in (1): a step defined
in a module pytest does not collect is a step pytest-bdd cannot find.

The step bodies bind to the L3 harness surface shipped in ``tests/harness/``
— specifically :mod:`harness.bridge`. ``tests/exemptions.py`` is **not**
imported here; the TR-1c exemption hook lands with T-J2 when its scenario
exists (§8.3 keeps the registry empty until then), and this file will be
extended in place rather than forked. The bodies bind to L3 symbols only;
they do not re-implement bridge behaviour — that is the rule §6.4.1 names
when it says "every scenario binds to an L3 harness rather than
re-implementing one."
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from harness.bridge import (
    BridgeFixture,
    InboundProtocol,
    assert_fixture_reached_its_recorder,
    inbound_path,
    minimal_inbound_body,
)
from harness.connect_proxy import unattributable_peer_ports
from harness.containment import BridgeAiohttpContainment, Phase1Result, SealedNetwork
from pytest_bdd import given, scenarios, then, when

from kitty.egress import EgressConfig

# Bind every ``.feature`` under this directory recursively. Runs during module
# import, before pytest collects; the step decorators below run in the same
# pass (see the module docstring on what is and is not load-bearing here).
scenarios("features")


# The type every step receives for the conftest's session fixture: the loop the
# fixture owns and the BridgeFixture started on it.
Session = tuple[asyncio.AbstractEventLoop, BridgeFixture]


# A single, scenario-stable marker: the request body the When step sends and the
# Then step asserts against. Defined here so the two steps cannot drift, and
# named so a future reviewer can grep one string across both.
SENT_MARKER = "wiring-smoke-marker"


@given("a started bridge session for the ANTHROPIC_MESSAGES protocol")
def bridge_session_is_ready(bridge_session: Session) -> Session:
    """Return the L3 bridge session the conftest already started.

    The conftest's ``bridge_session`` fixture owns the asyncio loop and a
    started :class:`~harness.bridge.BridgeFixture`. This step is a passthrough
    so a scenario can name the precondition in Gherkin without leaking the
    fixture plumbing into the step text.
    """
    return bridge_session


@when("Claude Code sends a minimal turn through the bridge", target_fixture="turn")
def send_minimal_turn(bridge_session: Session) -> dict[str, Any]:
    """Drive one minimal Anthropic Messages turn through the L3 bridge.

    Args:
        bridge_session: The tuple yielded by the conftest fixture — ``(loop,
            started BridgeFixture)``.

    Returns:
        A dict with the bridge's reply ``status``, raw ``body`` text, and the
        :data:`SENT_MARKER` the request carried, bound by pytest-bdd to the
        ``turn`` fixture so the ``Then`` step can hand them to the L3
        :func:`~harness.bridge.assert_fixture_reached_its_recorder` helper
        without re-driving the bridge.
    """
    loop, bridge = bridge_session
    body = minimal_inbound_body(InboundProtocol.MESSAGES, SENT_MARKER)
    status, response = loop.run_until_complete(bridge.post(inbound_path(InboundProtocol.MESSAGES), body))
    return {"status": status, "body": response, "marker": SENT_MARKER}


@then("the recording transport reports the turn")
def assert_recording_transport_saw_the_turn(
    bridge_session: Session,
    turn: dict[str, Any],
) -> None:
    """Confirm the L3 recording transport saw the request the bridge forwarded.

    Args:
        bridge_session: The conftest tuple; only ``bridge`` is read here.
        turn: The dict the ``When`` step returned.

    Raises:
        AssertionError: When the L3 helper
            :func:`~harness.bridge.assert_fixture_reached_its_recorder`
            reports any of the three defects it detects — a recorder that
            carried a different number of requests, a body that did not carry
            the sent marker, or a client that was not served. The helper's
            falsification suite lives in ``test_bridge_falsification.py``.
    """
    loop, bridge = bridge_session
    loop.run_until_complete(
        assert_fixture_reached_its_recorder(
            bridge,
            marker=turn["marker"],
            status=turn["status"],
            body=turn["body"],
        )
    )


# ── T-J3 (KBR-109) — EG acceptance scenarios ────────────────────────────────
# EG-0..EG-3 bind to the L3 sealed-network harness and the L3 start-path
# drive, never re-implementing bridge behaviour (TEST_SUITE.md §2.2 / §6.4.1).
# Step texts are intentionally distinct from the T-J1 wiring-smoke steps above.
# The falsification discipline is collective: T-J1's wiring smoke catches any
# step-text regression, T-E2 phase 3 catches an L3 negative-assertion
# regression, T-E8's AC-2.3 catches a guard-enforcement regression. EG-2
# reads ``sealed_network.recorder.connections`` and ``.requests`` directly
# so its own assertion cannot silently pass on a non-empty leak.


# The shape the EG "When a turn is sent through the bridge serving path" step
# binds to: ``(loop, SealedNetwork)`` plus a pre-bound egress config. Carried
# through pytest-bdd's ``target_fixture`` so the ``Then`` step reads the
# captured ``Phase1Result`` without re-driving the bridge.
SealedSession = tuple[asyncio.AbstractEventLoop, SealedNetwork, EgressConfig | None]
"""Type alias for the EG ``When`` step's bound state."""


def _drive_egress_off(
    loop: asyncio.AbstractEventLoop,
    net: SealedNetwork,
    monkeypatch: pytest.MonkeyPatch,
) -> Phase1Result:
    """Drive one turn with egress disabled — the EG-0 control shape.

    Thin wrapper over :meth:`BridgeAiohttpContainment.drive_phase_1` so the
    step body is a single line and the L3 harness is the only bridge
    surface exercised.

    Args:
        loop: The asyncio loop the sealed network is bound to.
        net: The started sealed network (proxy + recording upstream).
        monkeypatch: Pytest's monkeypatch fixture; the L3 drive's resolver
            patch reverts on its teardown.

    Returns:
        The :class:`~harness.containment.Phase1Result` the L3 drive
        produced — ``status == 200``, one capture, zero proxy attempts.
    """
    return loop.run_until_complete(BridgeAiohttpContainment().drive_phase_1(net, monkeypatch=monkeypatch))


def _drive_egress_on(
    loop: asyncio.AbstractEventLoop,
    net: SealedNetwork,
    egress: EgressConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> Phase1Result:
    """Drive one turn with the given egress config — the EG-1 / EG-2 shape.

    Args:
        loop: The asyncio loop the sealed network is bound to.
        net: The started sealed network (proxy + recording upstream).
        egress: The egress configuration to hand to ``BridgeServer``. The
            proxy URL the recorder expects is the proxy URL the bridge
            dials (the :func:`egress_for_sealed_network` fixture owns
            that derivation).
        monkeypatch: Pytest's monkeypatch fixture; the L3 drive's resolver
            patch reverts on its teardown.

    Returns:
        The :class:`~harness.containment.Phase1Result` the L3 drive
        produced.
    """
    return loop.run_until_complete(
        BridgeAiohttpContainment().drive_with_egress(net, egress=egress, monkeypatch=monkeypatch)
    )


# ── EG-0 / EG-1 / EG-2 share a "Given" set ──────────────────────────────────


@given("no egress gateway configured", target_fixture="egress_session")
def given_no_egress(
    sealed_network: tuple[asyncio.AbstractEventLoop, SealedNetwork],
) -> SealedSession:
    """EG-0: bind the ``(loop, net, None)`` session — egress off.

    Args:
        sealed_network: The ``(loop, net)`` tuple the :func:`sealed_network`
            fixture yields; the EG scenarios pass it through unchanged.

    Returns:
        The session tuple with ``egress=None``, signalling to the ``When``
        step that the drive shape is ``drive_phase_1`` (egress off).
    """
    loop, net = sealed_network
    return loop, net, None


@given("a configured egress gateway", target_fixture="egress_session")
def given_egress_configured(
    sealed_network: tuple[asyncio.AbstractEventLoop, SealedNetwork],
    egress_for_sealed_network: EgressConfig,
) -> SealedSession:
    """EG-1 / EG-2: bind the ``(loop, net, EgressConfig)`` session.

    Args:
        sealed_network: The ``(loop, net)`` tuple the :func:`sealed_network`
            fixture yields.
        egress_for_sealed_network: The :class:`~kitty.egress.EgressConfig`
            pointed at the harness proxy.

    Returns:
        The session tuple with the harness proxy URL pre-bound, ready for
        the ``When`` step to call ``drive_with_egress``. The proxy URL the
        recorder expects is the same proxy URL the bridge dials — the
        :func:`egress_for_sealed_network` fixture owns that derivation.
    """
    loop, net = sealed_network
    return loop, net, egress_for_sealed_network


@given("a configured egress gateway that has become unreachable", target_fixture="egress_session")
def given_egress_unreachable(
    sealed_network: tuple[asyncio.AbstractEventLoop, SealedNetwork],
    egress_for_sealed_network: EgressConfig,
) -> SealedSession:
    """EG-2 control: stop the proxy mid-fixture, then bind the session.

    Stopping the proxy *before* the ``When`` step drives is the only shape
    that exercises the §5.2.2 phase 2 negative assertion cleanly: the drive
    attempts a CONNECT against a closed listener, and the bridge's session
    builder surfaces the connection error rather than any retry path. The
    negative assertion's value comes from the upstream recorder seeing
    zero connections — :func:`unattributable_peer_ports` is therefore
    unused on this branch.

    Args:
        sealed_network: The ``(loop, net)`` tuple the :func:`sealed_network`
            fixture yields; the proxy is stopped on ``net`` before the yield.
        egress_for_sealed_network: The :class:`~kitty.egress.EgressConfig`
            pointed at the (now stopped) harness proxy.

    Returns:
        The session tuple, identical in shape to EG-1's — the ``When`` step
        treats both the same way and the drive itself surfaces the failure.
    """
    loop, net = sealed_network
    loop.run_until_complete(net.proxy.stop())
    return loop, net, egress_for_sealed_network


# ── EG-0 / EG-1 / EG-2 share a "When" step ──────────────────────────────────


@when("a turn is sent through the bridge serving path", target_fixture="eg_drive")
def when_a_turn_is_sent(
    egress_session: SealedSession,
    monkeypatch: pytest.MonkeyPatch,
) -> Phase1Result:
    """EG-0 ``When``: drive one turn with egress off (control).

    Kept as a separate decorator from the Claude-Code-shaped variants
    below so pytest-bdd can dispatch to the correct step text per
    scenario. The L3 drive is identical; only the Gherkin differs.

    Args:
        egress_session: Bound by the EG-0 ``Given``; the egress slot is
            ``None``, signalling the drive shape is :func:`_drive_egress_off`.
        monkeypatch: Pytest's monkeypatch fixture; the L3 drive's resolver
            patch reverts on its teardown.

    Returns:
        The :class:`~harness.containment.Phase1Result` the L3 drive produced.
    """
    loop, net, _ = egress_session
    return _drive_egress_off(loop, net, monkeypatch)


@when("Claude Code runs a session through kitty", target_fixture="eg_drive")
def when_claude_code_runs_session(
    egress_session: SealedSession,
    monkeypatch: pytest.MonkeyPatch,
) -> Phase1Result:
    """EG-1 ``When``: drive one Claude-Code-shaped session through kitty.

    Args:
        egress_session: Bound by the ``Given a configured egress gateway``
            step; the egress slot is non-None and points at the harness
            proxy.
        monkeypatch: Pytest's monkeypatch fixture; the L3 drive's resolver
            patch reverts on its teardown.

    Returns:
        The :class:`~harness.containment.Phase1Result` the L3 drive produced.
    """
    loop, net, egress = egress_session
    return _drive_egress_on(loop, net, egress, monkeypatch)


@when("Claude Code sends a turn through kitty", target_fixture="eg_drive")
def when_claude_code_sends_turn(
    egress_session: SealedSession,
    monkeypatch: pytest.MonkeyPatch,
) -> Phase1Result:
    """EG-2 ``When``: drive one Claude-Code-shaped turn with the proxy stopped.

    The ``Given a configured egress gateway that has become unreachable``
    step stopped the proxy before this ``When`` ran, so the drive's
    CONNECT attempt fails and the bridge surfaces an error to its
    caller. The L3 drive owns the recorder, the proxy (already stopped)
    and the resolver mapping; only the egress configuration is read here.

    Args:
        egress_session: Bound by the EG-2 ``Given``; the egress slot is
            non-None and points at the now-stopped proxy.
        monkeypatch: Pytest's monkeypatch fixture; the L3 drive's resolver
            patch reverts on its teardown.

    Returns:
        The :class:`~harness.containment.Phase1Result` the L3 drive produced.
    """
    loop, net, egress = egress_session
    return _drive_egress_on(loop, net, egress, monkeypatch)


# Kept here so a grep across the file finds both the EG-0 and EG-1/2
# when-steps and the routing helper that abstracts over their shared shape.
# The EG-0 step (``when_a_turn_is_sent``) calls ``_drive_egress_off``; the
# EG-1/2 steps call ``_drive_egress_on``. The helpers are the only bridge
# surface exercised; pytest-bdd dispatches per scenario on the Gherkin text.


# ── EG-0 Then steps ──────────────────────────────────────────────────────────


@then("the recording upstream records the connection")
def then_recorder_records_connection(eg_drive: Phase1Result) -> None:
    """EG-0 positive control: one capture arrived and the status was 200.

    Asserts the recorder saw exactly one capture with a 200 status — the
    preconditions for every EG-2 negative assertion (§5.3 trap 2):
    EG-2's "zero connections" only carries evidence because a working
    drive is reachable directly with egress disabled.

    Args:
        eg_drive: The :class:`~harness.containment.Phase1Result` the EG-0
            ``When`` step returned.
    """
    assert eg_drive.status == 200, f"bridge answered {eg_drive.status} with egress off: {eg_drive.text!r}"
    assert len(eg_drive.captures) == 1, (
        f"recorder saw {len(eg_drive.captures)} capture(s); the EG-0 reachability control requires exactly one"
    )


@then("the egress proxy saw no connection attempt")
def then_proxy_saw_nothing(eg_drive: Phase1Result) -> None:
    """EG-0: with egress off, the proxy's CONNECT log is empty.

    The proxy is up and listening (the harness started it) — it simply
    observed no traffic. A non-empty log here would mean the bridge
    contacted the proxy even though it was not configured to, which is
    the opposite of the EG-0 reachability story.

    Args:
        eg_drive: The :class:`~harness.containment.Phase1Result` the EG-0
            ``When`` step returned; ``attempts`` is what this step asserts on.
    """
    assert eg_drive.attempts == [], (
        f"proxy saw {len(eg_drive.attempts)} CONNECT attempt(s) with egress off: "
        f"{[a.target for a in eg_drive.attempts]}"
    )


# ── EG-1 Then step ──────────────────────────────────────────────────────────


@then("every connection the recording upstream accepted arrived through the gateway")
def then_every_connection_joined_a_tunnel(
    eg_drive: Phase1Result,
    sealed_network: tuple[asyncio.AbstractEventLoop, SealedNetwork],
) -> None:
    """EG-1: every recorder peer port joins a tunnel on the recorded source port.

    Per TEST_SUITE.md §5.2.1: one tunnel can carry many requests, one
    failed CONNECT can carry none, so the assertion is at the
    *connection* level — every peer port the recorder accepted must
    attribute to a tunnel the proxy opened. The L3 helper
    :func:`~harness.connect_proxy.unattributable_peer_ports` performs the
    join; an empty result is the green state.

    Args:
        eg_drive: The :class:`~harness.containment.Phase1Result` the EG-1
            ``When`` step returned; ``captures`` and ``connections`` feed the
            join.
        sealed_network: The ``(loop, net)`` tuple the :func:`sealed_network`
            fixture yields; ``net.proxy.attempts`` is the per-tunnel source-port
            log the join keys against.

    Raises:
        AssertionError: When any recorder peer port fails to join a
            proxy-recorded source port — a §5.2.2 phase 3 / T-E2 falsification
            shape, inherited.
    """
    _, net = sealed_network
    captures = list(eg_drive.captures)
    assert captures, "recorder accepted no captures; the drive never reached it"
    peer_ports = [c.peer_port for c in eg_drive.connections if c.peer_port]
    assert peer_ports, (
        "recorder's connection log has no entry with a real peer port: the §5.2.1 join has nothing to bind to"
    )
    unaccounted = unattributable_peer_ports(peer_ports, net.proxy.attempts)
    assert unaccounted == [], (
        f"recorder peer ports {peer_ports} not covered by tunnel source ports "
        f"{[a.source_port for a in net.proxy.attempts]}: §5.2.1's join broken"
    )


# ── EG-2 Then steps ──────────────────────────────────────────────────────────


@then("the turn fails with a clear error")
def then_turn_fails_with_clear_error(eg_drive: Phase1Result) -> None:
    """EG-2: with the proxy stopped, the bridge did not answer 2xx.

    The bridge surfaces a client-side error to its caller (``status < 0``
    in :class:`Phase1Result` semantics — the drive caught ``OSError`` or
    ``asyncio.TimeoutError`` and carried the exception repr in ``text``)
    or a protocol-level 5xx. The negative assertion EG-2 cares about is
    the upstream recorder holding zero connections, asserted next; this
    step establishes the user-visible failure shape.

    Args:
        eg_drive: The :class:`~harness.containment.Phase1Result` the EG-2
            ``When`` step returned; ``status`` is what this step asserts on
            (``!= 200`` covers both the negative client-side error case and a
            protocol-level 5xx).
    """
    assert eg_drive.status != 200, (
        f"bridge answered {eg_drive.status} with the proxy down: a 200 means "
        "the bridge reached the recorder anyway — the containment premise "
        "is broken"
    )


@then("the recording upstream receives nothing")
def then_recorder_received_nothing(
    sealed_network: tuple[asyncio.AbstractEventLoop, SealedNetwork],
) -> None:
    """EG-2 negative assertion: zero connections AND zero requests on the recorder.

    Reads ``sealed_network.recorder.connections`` and ``.requests`` **live**
    — not the drive's snapshot — so a regression in the harness that
    filtered or truncated the snapshot cannot hide a leak from this
    assertion. The same recorder-direct shape is what T-E2 phase 3's
    falsification reads (tests/harness/test_aiohttp_containment_slice.py
    ::TestPhase3Falsification), so the negative assertion carries the same
    evidence strength at L4 as at L3.

    Args:
        sealed_network: The ``(loop, net)`` tuple the :func:`sealed_network`
            fixture yields; the recorder is read live off ``net``.
    """
    _, net = sealed_network
    assert net.recorder.connections == [], (
        f"recorder accepted {len(net.recorder.connections)} connection(s) with the "
        "proxy stopped: the bridge fell back to a direct route — the §5.2.2 "
        "phase 2 defect"
    )
    assert net.recorder.requests == [], f"recorder saw {len(net.recorder.requests)} request(s) with the proxy stopped"


# ── EG-3 — the refusing-profile start path ──────────────────────────────────


@given("a profile whose transport cannot honour it", target_fixture="eg3_refusing")
def given_refusing_profile(
    refusing_profile_start_path: dict,
) -> dict:
    """EG-3: bind the patched ``bridge_runner`` collaborators.

    Mirrors :func:`tests.test_egress_start_path.start_path`. The patched
    collaborators (``BridgeServer`` spy, fake profile store, fake
    credential store returning the SSO marker) are the shape that makes
    :func:`kitty.egress_guard.egress_block_reason` return a non-empty
    string for the ``bedrock-sso`` profile — the §5.5 / §6.2.3 trigger.

    Args:
        refusing_profile_start_path: The fixture the conftest exposes;
            see its docstring for what it patches and why.

    Returns:
        The dict the EG-3 ``When`` step drives. Re-exposed as
        ``eg3_refusing`` so the ``Then`` steps read the
        ``bridge_server_init_calls`` counter without re-patching.
    """
    return refusing_profile_start_path


@when("the user runs kitty", target_fixture="eg3_outcome")
def when_user_runs_kitty(
    eg3_refusing: dict,
    capsys: pytest.CaptureFixture[str],
) -> dict:
    """EG-3: invoke ``bridge_runner.main()`` — the same path ``kitty`` ships.

    Captures the :class:`SystemExit` the refusing branch raises so the
    ``Then`` steps can assert on its code without pytest-bdd failing the
    scenario at the raise. It also drains capsys **here**: ``readouterr()``
    empties the buffer, so a ``Then`` step that called it would leave the
    second ``Then`` reading an empty stream — measured, not assumed: the
    first run of this scenario failed with ``stderr == ''`` for exactly
    that reason.

    Args:
        eg3_refusing: The patched-collaborators dict the ``Given`` bound.
        capsys: Pytest's capture fixture, drained once and carried in the
            outcome.

    Returns:
        A dict the ``Then`` steps read: ``exit_code`` (``None`` if no
        exit, otherwise the int from ``SystemExit.code``), ``stderr``
        (the captured stderr text), and ``refusing`` (the fixture dict).
        Re-bound through ``target_fixture="eg3_outcome"``.
    """
    from kitty import bridge_runner

    exit_code: int | None = None
    try:
        bridge_runner.main()
    except SystemExit as exc:
        # `SystemExit.code` may be int or str (per the Python spec). This
        # step normalises to int-or-None, which is *stricter* than the L3
        # reference (its `!= 0` accepts any non-zero code, str included).
        # The product path uses sys.exit(1) — int — so the normalisation
        # costs nothing today and keeps the Then-step assertion simple.
        exit_code = exc.code if isinstance(exc.code, int) else None
    return {
        "exit_code": exit_code,
        "stderr": capsys.readouterr().err,
        "refusing": eg3_refusing,
    }


@then("kitty refuses to start")
def then_kitty_refuses_to_start(
    eg3_outcome: dict,
) -> None:
    """EG-3 AC-1: non-zero SystemExit, ``BridgeServer.__init__`` never reached.

    A counter > 0 means the guard's return value was discarded — the
    §6.2.3 defect, which the L3 falsification in
    tests/test_egress_start_path.py::TestReturnDiscardedFalsification
    proves the suite catches.

    Args:
        eg3_outcome: The dict the EG-3 ``When`` step bound through
            ``target_fixture="eg3_outcome"``; ``stderr`` and ``refusing``
            are what this step asserts on.
    """
    err = eg3_outcome["stderr"]
    assert eg3_outcome["refusing"]["egress"].password not in err, (
        "the proxy password must never reach a user-facing message"
    )
    assert eg3_outcome["refusing"]["captured"]["bridge_server_init_calls"] == 0, (
        f"BridgeServer.__init__ was reached {eg3_outcome['refusing']['captured']['bridge_server_init_calls']} "
        "time(s) under a refusing profile: a listening socket would have been bound"
    )


@then("the refusal names the profile")
def then_refusal_names_profile(
    eg3_outcome: dict,
) -> None:
    """EG-3 AC-2: the user-facing stderr message names the refusing profile.

    The profile name comes from the fixture's returned dict — the
    ``Then`` step never imports the conftest directly, because ``tests``
    is not a package and bare ``pytest`` in CI does not add the
    checkout root to ``sys.path`` (per the kbr84 auto-memory note).

    Args:
        eg3_outcome: The dict the EG-3 ``When`` step bound through
            ``target_fixture="eg3_outcome"``; ``stderr``, ``exit_code``,
            and ``refusing.profile_name`` are what this step asserts on.
    """
    err = eg3_outcome["stderr"]
    profile_name = eg3_outcome["refusing"]["profile_name"]
    assert profile_name in err, f"stderr does not name the refusing profile {profile_name!r}: {err!r}"
    assert eg3_outcome["exit_code"] not in (None, 0), (
        f"bridge_runner.main() exited with code {eg3_outcome['exit_code']!r} (expected non-zero); stderr: {err!r}"
    )
