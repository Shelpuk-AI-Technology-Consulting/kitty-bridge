"""The acceptance layer's clean-path teardown, shared by the conftest and its falsification.

The conftest's ``bridge_session`` fixture must mirror
:meth:`harness.bridge.BridgeFixture.__aexit__` on the clean path: stop the
bridge, then assert the transport's teardown is clean. That mirror lives here,
as a plain function rather than inside the fixture generator, so the
falsification test in ``test_bridge_session_falsification.py`` can call the
**same** helper with a deliberate defect and assert it raises — the §1.4
culture the rest of the harness follows. A helper only the fixture calls
cannot be falsified, because fixture teardown errors surface as test errors,
not as exceptions a test can wrap.

The transport is read off ``bridge.transport`` rather than passed as a
separate argument: there is exactly one transport per fixture, and threading
it through separately is a duplication the falsification has no use for.
"""

from __future__ import annotations

import asyncio

from harness.bridge import BridgeFixture


def teardown_clean_path(bridge: BridgeFixture, loop: asyncio.AbstractEventLoop) -> None:
    """Stop the bridge, then assert the transport's teardown is clean.

    The clean-path half of :meth:`BridgeFixture.__aexit__`, extracted so both
    the conftest and its falsification call one implementation.

    Args:
        bridge: The started fixture to stop. The transport is read off
            ``bridge.transport``; there is exactly one per fixture.
        loop: The loop the fixture owns; ``bridge.stop`` runs on it.

    Raises:
        Exception: Whatever the transport's
            :meth:`~harness.bridge.UpstreamTransport.assert_teardown_clean`
            raises — the same exception :meth:`BridgeFixture.__aexit__`
            would raise on this path, including
            :class:`~harness.bridge.MisdeclaredFormatError` for a transport
            whose declared format does not match what it captured.
    """
    loop.run_until_complete(bridge.stop())
    bridge.transport.assert_teardown_clean()
