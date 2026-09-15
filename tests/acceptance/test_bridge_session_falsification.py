"""§1.4 falsification for the acceptance layer's clean-path teardown.

The smoke scenario in ``test_acceptance.py`` exercises the happy path: a
test body drives ``BridgeFixture.post`` through a real transport, the bridge
serves 200, and the conftest's clean-path ``assert_teardown_clean()`` runs.
A regression that drops that assertion — or reorders it behind ``stop()`` —
would not surface in the smoke, because the body still passes and the loop
still closes. Plan §1.4: the first working version of every harness ships
with at least one deliberate defect it must detect.

The defect here mirrors the L3 harness's own ``_MisdeclaredTransport``
(``test_bridge_falsification.py``): a transport whose
:meth:`~harness.bridge.AiohttpTransport.assert_teardown_clean` raises
:class:`~harness.bridge.MisdeclaredFormatError`. The helper it is handed to
— :func:`teardown.teardown_clean_path` — is the *same* helper the conftest
calls; this test does not re-implement the teardown shape, it points the
shared implementation at a transport that must make it raise.

Coverage boundary, stated rather than hidden: this falsifies the helper
itself. A regression that *deletes the conftest's ``else`` call* to the
helper (rather than edits the helper) is not caught here — that call site
is one line and is what code review is for. The conftest's module docstring
records the same boundary.
"""

from __future__ import annotations

import asyncio

import pytest
from harness.bridge import (
    AiohttpTransport,
    BridgeFixture,
    MisdeclaredFormatError,
    WireFormat,
)
from teardown import teardown_clean_path


class _MisdeclaredTransport(AiohttpTransport):
    """An :class:`AiohttpTransport` whose teardown-clean call always raises.

    Subclasses ``AiohttpTransport`` so it satisfies the
    :class:`~harness.bridge.UpstreamTransport` protocol
    :class:`BridgeFixture` expects. Mirrors the spirit of the L3 harness's
    ``_MisdeclaredTransport`` (``test_bridge_falsification.py``) — same
    exception — without importing that module's private helper. The message
    names the transport and states that this is the deliberate defect, not a
    claim about a captured body: the scenario below never drives a request,
    so there is no body to mis-declare.
    """

    name = "acceptance-misdeclared"

    def assert_teardown_clean(self) -> None:
        """Raise unconditionally: this transport *is* the §1.4 defect."""
        raise MisdeclaredFormatError(
            f"transport {self.name!r} is the deliberate mis-declared-format defect: "
            "its teardown-clean assertion always raises"
        )


class TestTheConftestTeardownIsHonoured:
    """The clean-path teardown the conftest runs must be falsifiable."""

    def test_a_misdeclared_transport_fails_the_clean_path(self) -> None:
        """``teardown_clean_path`` surfaces a mis-declared format.

        This is the §1.4 detector: if a regression removes
        ``transport_instance.assert_teardown_clean()`` from the helper (or
        reorders the stop/assert so the assertion never runs), this test goes
        red — the ``raises`` is expected, so its absence is the failure.

        Sync on purpose: the helper runs on the conftest's own event loop via
        ``run_until_complete``, exactly as the conftest drives it. An
        ``async def`` test would put this call inside a running loop, where
        ``run_until_complete`` is itself an error — testing the wrong shape.
        """
        loop = asyncio.new_event_loop()
        try:
            transport_instance = _MisdeclaredTransport(WireFormat.ANTHROPIC_MESSAGES)
            bridge = BridgeFixture(transport_instance)
            loop.run_until_complete(bridge.start())

            with pytest.raises(MisdeclaredFormatError) as excinfo:
                teardown_clean_path(bridge, loop)

            assert "deliberate mis-declared-format defect" in str(excinfo.value)
        finally:
            loop.close()
