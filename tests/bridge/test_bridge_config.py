"""Tests for R5: Bridge configuration file."""

from __future__ import annotations

from pathlib import Path

import pytest

from kitty.bridge.config import load_bridge_config


def _write_yaml(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestBridgeConfigParsing:
    """BridgeConfig loads from YAML and applies defaults."""

    def test_missing_file_returns_defaults(self, tmp_path: Path):
        config = load_bridge_config(tmp_path / "nonexistent.yaml")
        assert config.host == "127.0.0.1"
        assert config.port == 0
        assert config.profile is None
        assert config.log_access is None  # None means "use mode default"
        assert config.tls_cert is None
        assert config.tls_key is None

    def test_full_config(self, tmp_path: Path):
        _write_yaml(
            tmp_path / "bridge.yaml",
            """
host: "0.0.0.0"
port: 9090
profile: "my-zai"
keys_file: "/path/to/keys.txt"
log_access: true
log_dir: "/var/log/kitty"
tls_cert: "/path/to/cert.pem"
tls_key: "/path/to/key.pem"
""",
        )
        config = load_bridge_config(tmp_path / "bridge.yaml")
        assert config.host == "0.0.0.0"
        assert config.port == 9090
        assert config.profile == "my-zai"
        assert config.keys_file == "/path/to/keys.txt"
        assert config.log_access is True
        assert config.log_dir == "/var/log/kitty"
        assert config.tls_cert == "/path/to/cert.pem"
        assert config.tls_key == "/path/to/key.pem"

    def test_partial_config_uses_defaults(self, tmp_path: Path):
        _write_yaml(
            tmp_path / "bridge.yaml",
            """
port: 8080
""",
        )
        config = load_bridge_config(tmp_path / "bridge.yaml")
        assert config.host == "127.0.0.1"  # default
        assert config.port == 8080  # from file
        assert config.profile is None  # default
        assert config.tls_cert is None  # default

    def test_tilde_expansion(self, tmp_path: Path):
        _write_yaml(
            tmp_path / "bridge.yaml",
            """
keys_file: "~/keys.txt"
log_dir: "~/logs"
tls_cert: "~/cert.pem"
tls_key: "~/key.pem"
""",
        )
        config = load_bridge_config(tmp_path / "bridge.yaml")
        home = Path.home()
        assert config.keys_file == str(home / "keys.txt")
        assert config.log_dir == str(home / "logs")
        assert config.tls_cert == str(home / "cert.pem")
        assert config.tls_key == str(home / "key.pem")

    def test_empty_file_returns_defaults(self, tmp_path: Path):
        _write_yaml(tmp_path / "bridge.yaml", "")
        config = load_bridge_config(tmp_path / "bridge.yaml")
        assert config.host == "127.0.0.1"
        assert config.port == 0

    def test_invalid_yaml_raises_error(self, tmp_path: Path):
        _write_yaml(tmp_path / "bridge.yaml", ": invalid : yaml : [")
        with pytest.raises(ValueError, match="bridge.yaml"):
            load_bridge_config(tmp_path / "bridge.yaml")


# ---------------------------------------------------------------------------
# CLI override
# ---------------------------------------------------------------------------


class TestBridgeConfigCLIOverride:
    """CLI flags override config file values."""

    def test_cli_port_overrides_file(self, tmp_path: Path):
        _write_yaml(tmp_path / "bridge.yaml", "port: 8080")
        config = load_bridge_config(
            tmp_path / "bridge.yaml",
            cli_host="0.0.0.0",
            cli_port=9090,
        )
        assert config.port == 9090  # CLI overrides
        assert config.host == "0.0.0.0"  # CLI overrides

    def test_cli_flags_override_none_file_values(self, tmp_path: Path):
        """CLI flags override even when file doesn't set the value."""
        config = load_bridge_config(
            tmp_path / "nonexistent.yaml",
            cli_host="0.0.0.0",
            cli_port=9090,
        )
        assert config.host == "0.0.0.0"
        assert config.port == 9090

    def test_cli_none_uses_file_values(self, tmp_path: Path):
        """When CLI flags are None (not specified), file values are used."""
        _write_yaml(tmp_path / "bridge.yaml", "port: 8080\nhost: '0.0.0.0'")
        config = load_bridge_config(
            tmp_path / "bridge.yaml",
            cli_host=None,
            cli_port=None,
        )
        assert config.host == "0.0.0.0"  # from file
        assert config.port == 8080  # from file


# ---------------------------------------------------------------------------
# Resolved logging
# ---------------------------------------------------------------------------


class TestBridgeConfigResolvedLogging:
    """Resolved log_access depends on foreground/background mode."""

    def test_foreground_default_is_disabled(self, tmp_path: Path):
        config = load_bridge_config(tmp_path / "nonexistent.yaml")
        assert config.resolved_log_access(background=False) is False

    def test_background_default_is_enabled(self, tmp_path: Path):
        config = load_bridge_config(tmp_path / "nonexistent.yaml")
        assert config.resolved_log_access(background=True) is True

    def test_file_log_access_overrides_default(self, tmp_path: Path):
        _write_yaml(tmp_path / "bridge.yaml", "log_access: false")
        config = load_bridge_config(tmp_path / "bridge.yaml")
        # Even in background mode, explicit false in config disables it
        assert config.resolved_log_access(background=True) is False

    def test_file_log_access_true_enables_foreground(self, tmp_path: Path):
        _write_yaml(tmp_path / "bridge.yaml", "log_access: true")
        config = load_bridge_config(tmp_path / "bridge.yaml")
        assert config.resolved_log_access(background=False) is True
