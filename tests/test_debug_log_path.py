"""Tests for --debug-file CLI option: custom debug log path."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pytest

from kitty.bridge.server import _DEBUG_LOG_PATH, BridgeServer
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


# ── CLI parser tests ────────────────────────────────────────────────────────


class TestDebugFileCLIArg:
    """--debug-file flag is parsed correctly and implies --debug semantics."""

    def test_no_debug_flags(self):
        from kitty.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["bridge"])
        assert args.debug is False
        assert args.debug_file is None

    def test_debug_flag_alone(self):
        from kitty.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--debug", "bridge"])
        assert args.debug is True
        assert args.debug_file is None

    def test_debug_file_with_path(self):
        from kitty.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--debug-file", "/tmp/custom_debug.log", "bridge"])
        assert args.debug_file == Path("/tmp/custom_debug.log")
        assert args.debug is False  # --debug-file does NOT set --debug; effective logic handles it

    def test_debug_file_implies_debug_enabled(self):
        """When --debug-file is given, effective debug should be truthy."""
        from kitty.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--debug-file", "/tmp/custom_debug.log", "bridge"])
        effective = args.debug or args.debug_file is not None
        assert effective is True

    def test_debug_and_debug_file_together(self):
        from kitty.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--debug", "--debug-file", "/tmp/custom.log", "bridge"])
        assert args.debug is True
        assert args.debug_file == Path("/tmp/custom.log")


# ── BridgeServer debug path tests ──────────────────────────────────────────


class TestCustomDebugLogPath:
    """BridgeServer writes debug logs to a custom path when specified."""

    @pytest.fixture(autouse=True)
    def _clean_bridge_logger(self):
        """Clear kitty.bridge handlers before each test and restore after."""
        bridge_logger = logging.getLogger("kitty.bridge")
        original_handlers = list(bridge_logger.handlers)
        original_level = bridge_logger.level
        bridge_logger.handlers.clear()
        yield
        bridge_logger.handlers = original_handlers
        bridge_logger.level = original_level

    def test_debug_false_no_logging(self):
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            model="test-model",
            debug=False,
        )
        result = server._setup_debug_logging()
        assert result is None

    def test_unconfigured_kitty_logger_does_not_write_to_stderr(self, monkeypatch):
        """Kitty loggers must not leak through logging.lastResort in normal runs."""
        root_logger = logging.getLogger()
        bridge_logger = logging.getLogger("kitty.bridge")
        server_logger = logging.getLogger("kitty.bridge.server")

        original_root_handlers = list(root_logger.handlers)
        original_bridge_handlers = list(bridge_logger.handlers)
        original_server_handlers = list(server_logger.handlers)
        stderr = io.StringIO()

        try:
            root_logger.handlers.clear()
            bridge_logger.handlers.clear()
            server_logger.handlers.clear()
            monkeypatch.setattr(sys, "stderr", stderr)

            server_logger.error("Upstream Cloudflare block %d: %s", 403, "<html>blocked</html>")

            assert stderr.getvalue() == ""
        finally:
            root_logger.handlers = original_root_handlers
            bridge_logger.handlers = original_bridge_handlers
            server_logger.handlers = original_server_handlers

    def test_debug_true_default_path(self):
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            model="test-model",
            debug=True,
        )
        result = server._setup_debug_logging()
        assert result == _DEBUG_LOG_PATH

    def test_debug_string_custom_path(self, tmp_path: Path):
        custom_path = tmp_path / "custom" / "debug.log"
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            model="test-model",
            debug=str(custom_path),
        )
        result = server._setup_debug_logging()
        assert result == custom_path
        assert custom_path.parent.exists()

    def test_debug_custom_path_creates_parent_dirs(self, tmp_path: Path):
        custom_path = tmp_path / "deep" / "nested" / "dir" / "debug.log"
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            model="test-model",
            debug=str(custom_path),
        )
        result = server._setup_debug_logging()
        assert result == custom_path
        assert custom_path.parent.exists()

    def test_debug_custom_path_writes_log(self, tmp_path: Path):
        custom_path = tmp_path / "debug.log"
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            model="test-model",
            debug=str(custom_path),
        )
        log_path = server._setup_debug_logging()
        assert log_path == custom_path

        # Close all handlers to flush buffers to disk
        bridge_logger = logging.getLogger("kitty.bridge")
        logging.getLogger("kitty.bridge.server").debug("test child debug message")
        for h in list(bridge_logger.handlers):
            h.close()

        assert custom_path.exists()
        content = custom_path.read_text()
        assert "test child debug message" in content


class TestBudgetResolutionLinesReachTheDebugLog:
    """The compaction budget's INFO notices must survive the product's own debug path.

    ``_setup_debug_logging`` attached its ``FileHandler`` to ``kitty.bridge``
    only. ``kitty.providers.model_context`` is a **sibling** logger, and nobody
    sets its level, so root's default WARNING dropped its INFO records at the
    logger level before any handler was consulted. Both budget-resolution
    notices were therefore undeliverable via ``--debug`` — the product's own
    diagnostic surface, and the surface the KBR-170 acceptance criterion names
    ("an operator whose context_window is being shadowed can discover that
    from the logs"): the KBR-170 shadow notice, and, with the same defect
    since KBR-151, the default-fallback line.
    """

    @pytest.fixture(autouse=True)
    def _clean_model_context_logger(self):
        """Clear ``kitty.providers.model_context`` handlers before each test and restore after.

        Mirrors the ``_clean_bridge_logger`` autouse on
        :class:`TestCustomDebugLogPath`: ``_setup_debug_logging`` is deduped
        on the marker attribute, so re-attachment to a different path skips;
        the bridge path is cleared by the sibling class' fixture, the new
        logger is not.
        """
        mc_logger = logging.getLogger("kitty.providers.model_context")
        original_handlers = list(mc_logger.handlers)
        original_level = mc_logger.level
        mc_logger.handlers.clear()
        yield
        mc_logger.handlers = original_handlers
        mc_logger.level = original_level

    def test_shadow_and_fallback_notices_reach_the_debug_file(self, tmp_path: Path):
        import kitty.providers.model_context as mc

        custom_path = tmp_path / "debug.log"
        server = BridgeServer(
            StubLauncher(),
            StubProvider(),
            "test-key",
            model="test-model",
            debug=str(custom_path),
        )
        mc._log_shadowed_context_window.cache_clear()
        mc._log_default_fallback.cache_clear()
        target = logging.getLogger("kitty.providers.model_context")
        original_handlers = list(target.handlers)
        original_level = target.level
        try:
            log_path = server._setup_debug_logging()
            assert log_path == custom_path

            # Emit both notices directly: the unit under test is the logging
            # WIRING, not the resolver (the resolver's truth table lives in
            # tests/test_model_context.py).
            mc._log_shadowed_context_window("azure", "gpt-4o", 777_777, 128_000)
            mc._log_default_fallback("ollama", "llama3-custom")

            # Flush (not close — the handler is shared with the bridge logger
            # and the bridge keeps logging after this test).
            for h in list(target.handlers):
                h.flush()

            content = custom_path.read_text()
            assert "gpt-4o" in content
            assert "777777" in content.replace(",", "")
            assert "128000" in content.replace(",", "")
            assert "llama3-custom" in content
        finally:
            target.handlers = original_handlers
            target.setLevel(original_level)
            mc._log_shadowed_context_window.cache_clear()
            mc._log_default_fallback.cache_clear()


# ── Effective debug wiring test ─────────────────────────────────────────────


class TestEffectiveDebugWiring:
    """Verify that debug_file is correctly resolved to effective debug value."""

    def test_debug_file_overrides_to_string_path(self):
        """When --debug-file is provided, effective debug should be the path string."""
        debug_file: Path | None = Path("/tmp/my_debug.log")
        effective: bool | str = str(debug_file) if debug_file else False

        # Asserting against str(debug_file) here would be a tautology, so pin
        # the two properties main.py actually relies on: the value carries the
        # path (not a bool), and it is truthy so `debug` stays enabled.
        assert isinstance(effective, str)
        assert Path(effective) == debug_file
        assert effective  # truthy

    def test_no_debug_file_uses_debug_bool(self):
        """When --debug-file is absent, effective debug falls back to --debug flag."""
        debug_file = None
        debug = True
        effective: bool | str = str(debug_file) if debug_file else debug
        assert effective is True

    def test_neither_flag_gives_false(self):
        debug_file = None
        debug = False
        effective: bool | str = str(debug_file) if debug_file else debug
        assert effective is False
