"""Bridge-level tests: a custom-transport failure's body reaches the DEBUG log.

Live diagnosis 2026-10-29 (pool ``my``): the ``openai`` member failed its
stream with ``OpenAI subscription error 400: `` — the upstream body's
``message`` field was empty, and the structured body was dropped, so the
log carried no evidence for the cause. These tests pin the repair: the
bounded body rides the error (provider test) and every one of the four
route sites logs it at DEBUG next to the unchanged warning (server test).
Design of record: ``.requirements/20261029T184032Z_openai_400_diagnostics``.

**Layer.** Collects as ``l1`` alongside every other loopback test in this
directory (``tests/layers.py`` maps only ``tests/integration/`` away).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from kitty.bridge import server as server_module
from kitty.providers.base import ProviderError

_SERVER = Path(server_module.__file__)

#: The four custom-transport stream-failure sites — Responses, Messages,
#: Gemini, Chat Completions. One per route; a fifth means a new route
#: picked up the branch, a missing one means drift.
_EXPECTED_SITES = 4


class TestFailureBodyDebugLog:
    """The shared helper renders the body at DEBUG, and only when present."""

    def test_helper_logs_bounded_body_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """An error carrying a body gets it logged once, at DEBUG."""
        caplog.set_level("DEBUG", logger="kitty.bridge.server")
        exc = ProviderError("OpenAI subscription error 400: ")
        exc.upstream_body = '{"error": {"type": "invalid_request"}}'
        server_module.BridgeServer._log_custom_transport_failure_body(exc)
        rendered = [r.getMessage() for r in caplog.records if "failure body" in r.getMessage()]
        assert rendered == ["Custom-transport failure body: " + exc.upstream_body]

    def test_helper_is_silent_without_a_body(self, caplog: pytest.LogCaptureFixture) -> None:
        """No carriage (or a foreign exception type) logs nothing."""
        caplog.set_level("DEBUG", logger="kitty.bridge.server")
        server_module.BridgeServer._log_custom_transport_failure_body(ProviderError("boom"))
        server_module.BridgeServer._log_custom_transport_failure_body(RuntimeError("boom"))
        assert not [r for r in caplog.records if "failure body" in r.getMessage()]


class TestAllFourSitesLogTheBody:
    """Static drift guard: every warning site logs the body (R1's parity)."""

    def test_each_custom_transport_warning_is_followed_by_the_helper(self) -> None:
        """Every ``Custom-transport stream failed`` warning pairs with the helper call.

        The warning sites are spread across four stream branches; the
        textual pairing is checked here so a fifth branch (or a lost call)
        fails this test instead of shipping an opaque incident again.
        """
        tree = ast.parse(_SERVER.read_text(encoding="utf-8"))
        warning_sites = 0
        paired_sites = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                for stmt in handler.body:
                    if not _is_custom_transport_warning(stmt):
                        continue
                    warning_sites += 1
                    if any(_is_failure_body_log(s) for s in handler.body):
                        paired_sites += 1
        assert warning_sites == _EXPECTED_SITES, (
            f"expected {_EXPECTED_SITES} custom-transport warning sites, found {warning_sites}"
        )
        assert paired_sites == warning_sites, f"{warning_sites - paired_sites} warning site(s) skip the DEBUG body log"


def _is_custom_transport_warning(stmt: ast.stmt) -> bool:
    """Recognize the shared custom-transport warning line.

    Args:
        stmt: The statement to inspect.

    Returns:
        True when the statement is the ``Custom-transport stream failed`` warning.
    """
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    call = stmt.value
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr != "warning":
        return False
    if not call.args:
        return False
    first = call.args[0]
    return isinstance(first, ast.Constant) and "Custom-transport stream failed" in str(first.value)


def _is_failure_body_log(stmt: ast.stmt) -> bool:
    """Recognize the shared failure-body DEBUG call.

    Args:
        stmt: The statement to inspect.

    Returns:
        True when the statement calls ``_log_custom_transport_failure_body``.
    """
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    func = stmt.value.func
    if not isinstance(func, ast.Attribute):
        return False
    return func.attr == "_log_custom_transport_failure_body"
