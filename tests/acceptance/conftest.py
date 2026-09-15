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
closes the loop in that order; flipping them leaks the recorder port or
strands the aiohttp reader.

The conftest imports the L3 harness surface directly — the only behaviour the
acceptance layer owns is *how steps find L3*, never *how L3 behaves*.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from harness.bridge import BridgeFixture, WireFormat, transport


@pytest.fixture
def bridge_session() -> Iterator[tuple[asyncio.AbstractEventLoop, BridgeFixture]]:
    """Yield ``(loop, started BridgeFixture)`` for one acceptance scenario.

    Yields:
        A tuple of the asyncio loop and the :class:`BridgeFixture` the steps
        drive. The bridge is started on ``loop`` before the yield; on
        teardown — mirroring :meth:`BridgeFixture.__aexit__` exactly — the
        bridge is stopped, the transport's teardown is asserted only on the
        clean path, and the loop is closed last.

    Notes:
        Function-scoped on purpose: pytest-bdd generates one test function per
        scenario, and a started ``BridgeFixture`` owns a real port the next
        scenario must not inherit.
    """
    loop = asyncio.new_event_loop()
    transport_instance = transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)
    bridge = BridgeFixture(transport_instance)
    loop.run_until_complete(bridge.start())
    try:
        yield loop, bridge
    except BaseException:
        # The test body raised. ``__aexit__`` would skip the teardown-clean
        # assertion in this case, so the inner ``finally`` mirrors it by not
        # calling it either — surface the test's own exception, not ours.
        loop.run_until_complete(bridge.stop())
        loop.close()
        raise
    else:
        # Clean path. Mirror ``__aexit__``: stop, then assert teardown clean,
        # then close. Asserting before close lets the assertion shadow a
        # teardown-clean failure only on the path where there is no other
        # exception to surface — same as the L3 reference.
        loop.run_until_complete(bridge.stop())
        transport_instance.assert_teardown_clean()
        loop.close()
