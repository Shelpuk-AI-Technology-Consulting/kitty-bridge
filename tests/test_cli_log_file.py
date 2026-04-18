"""Tests for --log-file CLI option: custom usage log path."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol


# ── Stubs ───────────────────────────────────────────────────────────────────


class StubLauncher(LauncherAdapter):
    def __init__(self, protocol: BridgeProtocol = BridgeProtocol.MESSAGES_API):
        self._protocol = protocol

    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return self._protocol

    def build_spawn_config(self, profile: Profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class StubProvider(ProviderAdapter):
    @property
    def provider_type(self) -> str:
        return "stub"

    @property
    def default_base_url(self) -> str:
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Upstream error {status_code}: {body}")


# ── BridgeServer custom path tests ─────────────────────────────────────────


class TestCustomUsageLogPath:
    """BridgeServer writes usage logs to a custom path when specified."""

    def test_custom_path_used(self, tmp_path: Path):
        custom_path = tmp_path / "custom" / "my_usage.jsonl"
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            model="test-model",
            logging_enabled=True,
            _usage_log_path=custom_path,
        )
        server._log_usage(usage={"prompt_tokens": 10, "completion_tokens": 5})

        assert custom_path.exists()
        entries = [json.loads(l) for l in custom_path.read_text().strip().splitlines()]
        assert len(entries) == 1
        assert entries[0]["input_tokens"] == 10

    def test_custom_path_creates_parent_dirs(self, tmp_path: Path):
        custom_path = tmp_path / "deep" / "nested" / "dir" / "usage.log"
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            logging_enabled=True,
            _usage_log_path=custom_path,
        )
        server._log_usage(usage={"prompt_tokens": 1, "completion_tokens": 1})

        assert custom_path.exists()

    def test_default_path_when_none_specified(self):
        from kitty.bridge.server import BridgeServer as _BS

        # Default _usage_log_path should be ~/.cache/kitty/usage.log
        server = _BS(
            StubLauncher(),
            StubProvider(),
            "test-key",
            logging_enabled=True,
        )
        expected = Path.home() / ".cache" / "kitty" / "usage.log"
        assert server._usage_log_path == expected


# ── CLI integration tests ──────────────────────────────────────────────────


class TestLogFileCLIArg:
    """--log-file flag is parsed and wired correctly."""

    def test_log_file_implies_logging_enabled(self):
        """When --log-file is given, logging_enabled should be True even without --logging."""
        import argparse

        from kitty.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--log-file", "/tmp/custom.log", "bridge"])
        assert args.log_file == Path("/tmp/custom.log")
        # Effective: logging_enabled = args.logging or args.log_file is not None
        effective = args.logging or args.log_file is not None
        assert effective is True

    def test_no_log_file_no_logging(self):
        import argparse

        from kitty.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["bridge"])
        assert args.log_file is None
        assert args.logging is False

    def test_log_file_with_explicit_logging(self):
        import argparse

        from kitty.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--logging", "--log-file", "/tmp/custom.log", "bridge"])
        assert args.log_file == Path("/tmp/custom.log")
        assert args.logging is True

    def test_logging_without_log_file(self):
        import argparse

        from kitty.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--logging", "bridge"])
        assert args.log_file is None
        assert args.logging is True
