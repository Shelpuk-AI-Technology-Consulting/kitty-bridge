"""Shared fixtures for the L4 acceptance layer (T-J1, KBR-107).

The pytest-bdd generated tests are **synchronous** functions — pytest-bdd does
not natively run coroutine step functions — but the L3 harness
(:class:`~harness.bridge.BridgeFixture`) is async and its aiohttp session is
bound to the event loop that created it. To drive the L3 async surface from
sync steps without a per-step event loop (which would orphan the session), the
``bridge_session`` fixture here owns a single loop for the duration of one
scenario and yields the ``(loop, bridge)`` tuple to every step.

Steps consume the tuple and call ``loop.run_until_complete(coro)`` for each L3
op — most steps use ``bridge.post``. On teardown the fixture stops the bridge,
mirrors ``BridgeFixture.__aexit__``'s clean-path teardown-clean assertion, and
closes the loop. The outer ``try``/``finally`` closes the loop on **every**
exit path, including a raise from ``BridgeFixture.start()`` itself — reachable
per ``bridge.py:707-732`` (a profile-validation or bind failure), and the path
the round-2 review surfaced.

The conftest imports the L3 harness surface directly — the only behaviour the
acceptance layer owns is *how steps find L3*, never *how L3 behaves*. The
clean-path teardown helper itself lives in :mod:`teardown` so the conftest's
own falsification can call it with a deliberate defect.

The fixture accepts an optional ``request.param`` override: the smoke scenario
calls it without parametrising, and the conftest's own §1.4 falsification
parametrises it with a transport whose teardown-clean call fails, so the
acceptance layer inherits the same falsification culture the rest of the
harness follows.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from harness.bridge import BridgeFixture, UpstreamTransport, WireFormat, transport
from teardown import teardown_clean_path


@pytest.fixture
def bridge_session(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[asyncio.AbstractEventLoop, BridgeFixture]]:
    """Yield ``(loop, started BridgeFixture)`` for one acceptance scenario.

    Yields:
        A tuple of the asyncio loop and the :class:`BridgeFixture` the steps
        drive. The bridge is started on ``loop`` before the yield; on
        teardown — mirroring :meth:`BridgeFixture.__aexit__` exactly — the
        bridge is stopped, the transport's teardown is asserted only on the
        clean path, and the loop is closed last. The outer ``finally`` closes
        the loop on every exit, including ``BridgeFixture.start()`` raising.

    Args:
        request: The pytest fixture request. When the test parametrises
            ``bridge_session`` with ``indirect=True``, ``request.param`` is the
            transport to start instead of the default. The smoke scenario does
            not parametrise and gets the default; the conftest's own
            §1.4 falsification parametrises with a transport whose
            :meth:`assert_teardown_clean` fails, so the clean-path assertion is
            exercised.

    Notes:
        Function-scoped on purpose: pytest-bdd generates one test function per
        scenario, and a started ``BridgeFixture`` owns a real port the next
        scenario must not inherit.
    """
    loop = asyncio.new_event_loop()
    try:
        transport_instance: UpstreamTransport = (
            request.param
            if hasattr(request, "param")
            else transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)
        )
        bridge = BridgeFixture(transport_instance)
        loop.run_until_complete(bridge.start())
        try:
            yield loop, bridge
        except BaseException:
            # Test body raised. ``__aexit__`` skips the teardown-clean
            # assertion in this case, so the inner ``try`` mirrors it by
            # skipping too. Surface the test's own exception.
            loop.run_until_complete(bridge.stop())
            raise
        else:
            # Clean path. The shared ``teardown_clean_path`` is the single
            # implementation; the falsification test calls the same helper
            # with a deliberately broken transport.
            teardown_clean_path(bridge, transport_instance, loop)
    finally:
        # Closes the loop on every documented exit path:
        # - ``BridgeFixture.start()`` raising (reachable per bridge.py:707-732)
        # - the test body raising (after ``bridge.stop()`` runs)
        # - the clean path (after the teardown assertion runs)
        # - any future code that raises between fixture construction and the
        #   inner try block — the loop is always the conftest's, never the
        #   bridge's, and would strand itself otherwise.
        loop.close()
