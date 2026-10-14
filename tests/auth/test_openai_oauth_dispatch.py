"""Tests for the browser-vs-device dispatch inside ``run_oauth_flow``.

KBR-317 step 3. The selection rule lives in one place — ``run_oauth_flow`` —
so all four ``run_oauth_for_provider`` entry points inherit auto-detect, and
the one CLI surface (``kitty auth openai``) can force either path.
"""

from __future__ import annotations

import pytest

from kitty.auth import openai_oauth
from kitty.auth.openai_oauth import run_oauth_flow


class _SentinelPathError(RuntimeError):
    """Raised by a stubbed flow to prove which path the dispatch selected."""


@pytest.fixture
def stub_flows(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """Stub both flow functions to raise a marker naming the selected path."""
    called: dict[str, bool] = {}

    async def _device_stub(http: object | None = None) -> object:
        called["device"] = True
        raise _SentinelPathError("device path selected")

    async def _browser_stub(http: object | None = None) -> object:
        called["browser"] = True
        raise _SentinelPathError("browser path selected")

    monkeypatch.setattr(openai_oauth, "run_oauth_device_flow", _device_stub)

    # Patching run_oauth_flow itself would break the function under test, so
    # the browser path is marked by the callback-server builder never being
    # reached: instead patch is_headless_environment and let the browser
    # branch's first side effect (build_auth_url) prove it ran. Simpler and
    # honest: patch the private _start_callback_server to raise the marker.
    async def _server_stub(*args: object, **kwargs: object) -> object:
        called["browser"] = True
        raise _SentinelPathError("browser path selected")

    monkeypatch.setattr(openai_oauth, "_start_callback_server", _server_stub)
    return called


class TestRunOauthFlowDispatch:
    async def test_device_auth_true_selects_device_path(self, stub_flows: dict[str, bool]) -> None:
        with pytest.raises(_SentinelPathError, match="device path"):
            await run_oauth_flow(device_auth=True)
        assert stub_flows == {"device": True}

    async def test_device_auth_false_selects_browser_path(self, stub_flows: dict[str, bool]) -> None:
        with pytest.raises(_SentinelPathError, match="browser path"):
            await run_oauth_flow(device_auth=False)
        assert stub_flows == {"browser": True}

    async def test_none_defers_to_headless_true(
        self, stub_flows: dict[str, bool], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(openai_oauth, "is_headless_environment", lambda: True)
        with pytest.raises(_SentinelPathError, match="device path"):
            await run_oauth_flow()
        assert stub_flows == {"device": True}

    async def test_none_defers_to_headless_false(
        self, stub_flows: dict[str, bool], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(openai_oauth, "is_headless_environment", lambda: False)
        with pytest.raises(_SentinelPathError, match="browser path"):
            await run_oauth_flow()
        assert stub_flows == {"browser": True}

    def test_predicate_import_site_is_production(self) -> None:
        """The module under test must resolve the predicate through
        ``kitty.auth.headless`` — the same object the CLI layer sees, not a
        re-bind or a shadow."""
        import kitty.auth.headless as headless

        assert openai_oauth.is_headless_environment is headless.is_headless_environment
