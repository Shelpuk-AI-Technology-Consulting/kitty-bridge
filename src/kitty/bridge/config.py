"""Bridge configuration file loading and resolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 0
_DEFAULT_KEYS_FILE = str(Path.home() / ".config" / "kitty" / "bridge_keys.txt")
_DEFAULT_LOG_DIR = str(Path.home() / ".config" / "kitty" / "logs")


def _expand_path(value: str | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser())


@dataclass
class BridgeConfig:
    """Resolved bridge configuration (file values + CLI overrides + defaults).

    ``keys_file`` is the path named by configuration, or ``None`` when nothing is
    named — it is deliberately not defaulted; see :func:`resolve_keys_file` for
    the effective file.
    """

    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT
    profile: str | None = None
    keys_file: str | None = None  # None = nothing named in bridge.yaml
    log_access: bool | None = None  # None = use mode default
    log_dir: str = _DEFAULT_LOG_DIR
    tls_cert: str | None = None
    tls_key: str | None = None

    def resolved_log_access(self, *, background: bool) -> bool:
        """Resolve log_access based on mode.

        - If config file explicitly sets log_access, use that.
        - Otherwise: foreground=False, background=True.
        """
        if self.log_access is not None:
            return self.log_access
        return background


def load_bridge_config(
    config_path: Path | str,
    *,
    cli_host: str | None = None,
    cli_port: int | None = None,
    cli_profile: str | None = None,
    cli_log_access: bool | None = None,
    cli_tls_cert: str | None = None,
    cli_tls_key: str | None = None,
) -> BridgeConfig:
    """Load bridge config from YAML file, apply CLI overrides.

    Args:
        config_path: Path to bridge.yaml (may not exist).
        cli_host: CLI --host override.
        cli_port: CLI --port override.
        cli_profile: CLI --profile override.
        cli_log_access: CLI --log (True) or --no-log (False) override.
        cli_tls_cert: CLI --tls-cert override.
        cli_tls_key: CLI --tls-key override.

    Returns:
        Fully resolved BridgeConfig. ``keys_file`` holds what the file named
        (possibly ``None``), not the effective path — resolve it with
        :func:`resolve_keys_file`.
    """
    import yaml

    file_values: dict = {}
    config_path = Path(config_path)

    if config_path.exists():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

        if isinstance(raw, dict):
            file_values = raw
        elif raw is not None:
            raise ValueError(f"Invalid bridge config in {config_path}: expected mapping, got {type(raw).__name__}")

    # Build config: defaults → file → CLI
    def _get(key: str, cli_value: object, default: object) -> object:
        if cli_value is not None:
            return cli_value
        return file_values.get(key, default)

    config = BridgeConfig(
        host=str(_get("host", cli_host, _DEFAULT_HOST)),
        port=int(_get("port", cli_port, _DEFAULT_PORT)),  # type: ignore[call-overload]  # _get returns object
        profile=_get("profile", cli_profile, None),  # type: ignore[arg-type]
        # Falsy values — null, "", false, 0 — count as nothing named (KBR-230):
        # the bridge then starts with auth off rather than crashing.
        keys_file=_expand_path(str(v)) if (v := _get("keys_file", None, None)) else None,  # type: ignore[arg-type]
        log_access=_get("log_access", cli_log_access, None),  # type: ignore[arg-type]
        log_dir=str(_expand_path(str(_get("log_dir", None, _DEFAULT_LOG_DIR))) or _DEFAULT_LOG_DIR),  # type: ignore[arg-type]
        tls_cert=_expand_path(str(v) if v is not None else None)
        if (v := _get("tls_cert", cli_tls_cert, None)) is not None
        else None,  # type: ignore[arg-type]
        tls_key=_expand_path(str(v) if v is not None else None)
        if (v := _get("tls_key", cli_tls_key, None)) is not None
        else None,  # type: ignore[arg-type]
    )

    return config


def resolve_keys_file(config: BridgeConfig) -> str | None:
    """Resolve the keys file a background bridge authenticates with.

    A named ``keys_file`` wins even when it does not exist — deciding how a
    named-but-missing file fails is the runner's job (``kitty.bridge_runner``
    refuses to start). With nothing named, the default keys file is used when
    it exists; a fresh install with neither starts with authentication off
    (KBR-230).

    Args:
        config: The resolved bridge configuration.

    Returns:
        The effective keys file path, or ``None`` when the bridge should
        accept unauthenticated clients.
    """
    if config.keys_file is not None:
        return config.keys_file
    if Path(_DEFAULT_KEYS_FILE).exists():
        return _DEFAULT_KEYS_FILE
    return None
