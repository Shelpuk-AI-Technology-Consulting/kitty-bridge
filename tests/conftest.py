"""Shared fixtures for kitty tests."""

import socket
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_backend_context() -> None:
    """Reset the per-request backend selection context before every test.

    The module-level ``_backend_context`` ContextVar in ``server.py``
    persists across tests that run in the same thread.  This fixture
    ensures every test starts with a clean slate so properties like
    ``_active_provider`` and ``_current_backend_idx`` read from the
    instance fields, not a stale context-var value from a prior test.
    """
    from kitty.bridge.server import _backend_context

    _backend_context.set({})


@pytest.fixture(autouse=True)
def _reset_egress() -> None:
    """Clear the process-wide egress configuration before every test.

    ``kitty.egress`` holds the resolved proxy in a module global, because
    provider adapters have no other channel for bridge-level settings. Without
    this reset, a test that enables egress would silently proxy every later
    test in the same process.
    """
    from kitty.egress import set_egress

    set_egress(None)


@pytest.fixture(autouse=True)
def _session_settings_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep per-session agent settings files inside the test's temp directory.

    ``ClaudeAdapter.prepare_launch`` creates its file with ``mkstemp`` and no
    explicit ``dir``, which resolves to the real OS temp directory. Redirecting
    ``tempfile.tempdir`` keeps the suite from scattering session files (each
    holding a test credential) across the developer's machine.
    """
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add CLI flag to include slow tests."""
    parser.addoption("--runslow", action="store_true", default=False, help="run tests marked as slow")


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used by the suite."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip slow tests unless explicitly requested."""
    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory for profile/credential stores."""
    return tmp_path


@pytest.fixture()
def sample_profile_dict() -> dict:
    """Valid profile data dict for reuse across tests."""
    return {
        "name": "test-profile",
        "provider": "zai_regular",
        "model": "gpt-4o",
        "auth_ref": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "base_url": None,
        "provider_config": {},
        "is_default": False,
    }


@pytest.fixture()
def mock_provider_response() -> dict:
    """Sample Chat Completions response dict."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from the provider."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture()
def unused_tcp_port() -> int:
    """Find a free TCP port for bridge tests."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
