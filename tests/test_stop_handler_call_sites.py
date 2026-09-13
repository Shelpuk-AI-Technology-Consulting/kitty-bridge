"""Source guard: no kitty run loop registers stop signals except through the shared helper.

``loop.add_signal_handler`` raises :exc:`NotImplementedError` on Windows, and
calling it directly killed the background bridge there (KBR-220). The helper
:func:`kitty.bridge.stop_signals.install_stop_handlers` declines instead. Only the
Windows CI leg can show a direct call failing, and only for the loops a test
happens to start. The foreground ``kitty bridge`` loops are started by none, so
this guard reads the source instead.

**Layer.** ``l2``: it reads the source tree, not behaviour, so it stays out of
the ``l1`` set that mutation testing judges.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.l2

SRC = Path(__file__).resolve().parent.parent / "src" / "kitty"
HELPER = SRC / "bridge" / "stop_signals.py"


def test_only_the_shared_helper_calls_add_signal_handler() -> None:
    """Every ``add_signal_handler`` call in ``src/kitty`` lives in the helper."""
    offenders = [
        f"{path.relative_to(SRC.parent)}:{number}"
        for path in sorted(SRC.rglob("*.py"))
        if path != HELPER
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if ".add_signal_handler(" in line
    ]

    assert offenders == [], f"call kitty.bridge.stop_signals.install_stop_handlers instead: {offenders}"


def test_the_helper_is_where_the_guard_looks() -> None:
    """The guard's exemption names a file that really makes the call.

    A renamed helper would otherwise leave the scan above with nothing to find
    and nothing to exempt, and it would pass by construction.
    """
    assert ".add_signal_handler(" in HELPER.read_text(encoding="utf-8")
