"""Bind pytest-bdd's scenarios and define the acceptance step vocabulary (T-J1, KBR-107).

Two things live here, in this order, and the order is load-bearing:

1. **The step definitions.** pytest-bdd's ``@given``/``@when``/``@then``
   decorators inject a pytest fixture into the *calling module's namespace*
   (``pytest-bdd``'s ``step()`` uses ``get_caller_module_locals``), and a
   scenario resolves a step by looking that fixture up through pytest's fixture
   manager. So the steps must live in a module **pytest collects** — this one —
   not in a side module pytest never imports, where the injected fixtures are
   invisible.

2. **The ``scenarios()`` binding.** Every ``.feature`` under ``features/`` is
   bound recursively to a generated pytest test function. T-J2 and T-J3 add
   their ``.feature`` files and step definitions alongside this one's smoke; no
   further wiring is needed.

The step bodies bind to the L3 harness surface shipped in ``tests/harness/`` and
``tests/exemptions.py``. They do not re-implement bridge behaviour — that is the
rule §6.4.1 names when it says "every scenario binds to an L3 harness rather than
re-implementing one."
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from harness.bridge import (
    BridgeFixture,
    InboundProtocol,
    assert_fixture_reached_its_recorder,
    inbound_path,
    minimal_inbound_body,
)
from pytest_bdd import given, scenarios, then, when

if TYPE_CHECKING:
    pass

# Register the step definitions below, then bind the feature files. Reversing
# the order reports the first step as "not found" at collection time.
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
    status, response = loop.run_until_complete(
        bridge.post(inbound_path(InboundProtocol.MESSAGES), body)
    )
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
            bridge, marker=turn["marker"], status=turn["status"]
        )
    )
