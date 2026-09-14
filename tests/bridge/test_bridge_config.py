"""Tests for R5: Bridge configuration file."""

from __future__ import annotations

from pathlib import Path

import pytest

from kitty.bridge.config import load_bridge_config, resolve_keys_file


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
        # None, not the default path: the background bridge must tell "nothing
        # configured" (KBR-230: use the default file only if it exists) apart
        # from "configured and missing" (a clear startup error).
        assert config.keys_file is None

    @pytest.mark.parametrize("yaml_value", ['null', '""', "false", "0"])
    def test_falsy_keys_file_values_count_as_nothing_named(self, tmp_path: Path, yaml_value: str):
        _write_yaml(tmp_path / "bridge.yaml", f"keys_file: {yaml_value}")
        config = load_bridge_config(tmp_path / "bridge.yaml")
        # Every falsy scalar counts as nothing named (SYSTEM_DESIGN.md §1.6):
        # a typo like `keys_file: 0` yields auth off, not a crash on a path
        # named "False" or "0".
        assert config.keys_file is None

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
        # Path-valued fields pass through _expand_path, which renders them with
        # the host OS separator — compare against the same normalisation rather
        # than hard-coding POSIX separators.
        assert config.keys_file == str(Path("/path/to/keys.txt"))
        assert config.log_access is True
        assert config.log_dir == str(Path("/var/log/kitty"))
        assert config.tls_cert == str(Path("/path/to/cert.pem"))
        assert config.tls_key == str(Path("/path/to/key.pem"))

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


class TestResolveKeysFile:
    """The effective keys file: the named one, else the default when it exists."""

    def test_named_file_wins_even_when_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("kitty.bridge.config._DEFAULT_KEYS_FILE", str(tmp_path / "default.txt"))
        named = tmp_path / "named" / "keys.txt"
        _write_yaml(tmp_path / "bridge.yaml", f'keys_file: "{named}"')
        config = load_bridge_config(tmp_path / "bridge.yaml")
        # The named path is returned as-is even though it does not exist: deciding
        # how a named-but-missing file fails is the runner's job, not the resolver's.
        assert resolve_keys_file(config) == str(named)

    def test_unnamed_uses_the_default_when_it_exists(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        default = tmp_path / "default.txt"
        default.write_text("client-key\n", encoding="utf-8")
        monkeypatch.setattr("kitty.bridge.config._DEFAULT_KEYS_FILE", str(default))
        config = load_bridge_config(tmp_path / "nonexistent.yaml")
        assert resolve_keys_file(config) == str(default)

    def test_unnamed_without_a_default_is_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("kitty.bridge.config._DEFAULT_KEYS_FILE", str(tmp_path / "absent.txt"))
        config = load_bridge_config(tmp_path / "nonexistent.yaml")
        assert resolve_keys_file(config) is None
