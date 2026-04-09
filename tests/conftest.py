"""Shared fixtures for kitty tests."""

import socket
from pathlib import Path

import pytest


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
