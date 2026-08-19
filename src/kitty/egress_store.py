"""Persistent storage for the egress gateway configuration.

Mirrors :class:`kitty.profiles.store.ProfileStore` — same file lock, same atomic
write, same version field — because the gateway is configured through the TUI
and needs the same crash-safety as profiles.

The proxy password is deliberately **not** stored here. It goes to the existing
credential store under a UUID reference, exactly as profile API keys do, so that
no kitty config file ever holds a plaintext secret.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import filelock
from platformdirs import user_config_dir

from kitty.egress import ENV_PROXY, EgressConfig, parse_proxy_url

logger = logging.getLogger(__name__)

STORE_VERSION = 1

__all__ = ["STORE_VERSION", "EgressRecord", "EgressStore", "resolve_egress"]


class EgressRecord:
    """A stored egress gateway entry.

    Attributes:
        proxy_url: Proxy endpoint as ``scheme://host:port``, no credentials.
        username: Proxy username, or ``None`` for an unauthenticated proxy.
        auth_ref: UUID key of the password in the credential store, or ``None``.
    """

    __slots__ = ("auth_ref", "proxy_url", "username")

    def __init__(self, proxy_url: str, username: str | None = None, auth_ref: str | None = None) -> None:
        """Initialise the record.

        Args:
            proxy_url: Proxy endpoint as ``scheme://host:port``.
            username: Proxy username, if the proxy requires authentication.
            auth_ref: Credential-store reference holding the password.
        """
        self.proxy_url = proxy_url
        self.username = username
        self.auth_ref = auth_ref

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns:
            The record as a plain dict. Never contains the password.
        """
        return {"proxy_url": self.proxy_url, "username": self.username, "auth_ref": self.auth_ref}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EgressRecord:
        """Rebuild a record from stored JSON.

        Args:
            data: Mapping previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed record.

        Raises:
            KeyError: If ``proxy_url`` is missing.
        """
        return cls(
            proxy_url=data["proxy_url"],
            username=data.get("username"),
            auth_ref=data.get("auth_ref"),
        )

    def __eq__(self, other: object) -> bool:
        """Compare records field by field."""
        if not isinstance(other, EgressRecord):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"EgressRecord(proxy_url={self.proxy_url!r}, username={self.username!r}, auth_ref={self.auth_ref!r})"


class EgressStore:
    """Single-record JSON store for the egress gateway.

    Uses a file lock for concurrent-access safety and an atomic
    write-to-temp-then-rename for crash resilience, matching
    :class:`kitty.profiles.store.ProfileStore`.
    """

    def __init__(self, path: Path | None = None) -> None:
        """Initialise the store.

        Args:
            path: Location of ``egress.json``. Defaults to the kitty config
                directory, beside ``profiles.json``.
        """
        if path is None:
            config_dir = Path(user_config_dir("kitty"))
            config_dir.mkdir(parents=True, exist_ok=True)
            path = config_dir / "egress.json"
        self._path = path
        self._lock = filelock.FileLock(str(path) + ".lock", timeout=5)

    def load(self) -> EgressRecord | None:
        """Read the stored gateway.

        Returns:
            The stored record, or ``None`` when nothing is configured or the
            file is missing, unreadable, or written by a newer version.
        """
        try:
            with self._lock:
                raw = self._path.read_text(encoding="utf-8")
                data = json.loads(raw)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            # Loud, because a corrupt file silently disables egress — and a
            # silently disabled proxy is exactly the leak this feature prevents.
            logger.warning(
                "Egress config %s is corrupt (not valid JSON). Egress is disabled until it is reconfigured.",
                self._path,
            )
            return None
        except OSError:
            return None

        if not isinstance(data, dict) or data.get("version") != STORE_VERSION:
            logger.warning("Egress config %s has an unsupported version — ignoring", self._path)
            return None

        entry = data.get("egress")
        if not isinstance(entry, dict):
            return None
        try:
            return EgressRecord.from_dict(entry)
        except KeyError:
            logger.warning("Egress config %s is missing required fields — ignoring", self._path)
            return None

    def save(self, record: EgressRecord) -> None:
        """Persist the gateway, replacing any existing entry.

        Args:
            record: The record to store.
        """
        with self._lock:
            self._write({"version": STORE_VERSION, "egress": record.to_dict()})

    def delete(self) -> None:
        """Remove the stored gateway, leaving an empty store behind."""
        with self._lock:
            self._write({"version": STORE_VERSION, "egress": None})

    def _write(self, data: dict[str, Any]) -> None:
        """Write the store atomically (caller must hold the lock).

        Args:
            data: Complete file contents to serialise.
        """
        content = json.dumps(data, indent=2, ensure_ascii=False)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix=self._path.stem + ".",
            dir=self._path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(tmp_path, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise


def resolve_egress(
    *,
    cli_proxy: str | None = None,
    store: EgressStore | None = None,
    cred_store: Any = None,
) -> EgressConfig | None:
    """Resolve the effective egress gateway.

    Precedence is command line, then environment, then the stored gateway, so a
    scripted or containerised install can override what the TUI configured
    without editing files.

    Args:
        cli_proxy: Value of ``--egress-proxy``, if given.
        store: Store to read the persisted gateway from. Defaults to the
            standard location.
        cred_store: Credential store used to resolve the stored password. When
            omitted, a stored gateway that needs a password cannot be resolved.

    Returns:
        The effective configuration, or ``None`` when egress is disabled.

    Raises:
        ValueError: If a configured proxy URL is malformed, or a stored gateway
            references a password that can no longer be resolved.
    """
    # 1. Explicit flag wins over everything.
    if cli_proxy and cli_proxy.strip():
        return parse_proxy_url(cli_proxy)

    # 2. Environment, for containers and CI where no TUI is available.
    env_value = os.environ.get(ENV_PROXY, "").strip()
    if env_value:
        return parse_proxy_url(env_value)

    # 3. Whatever `kitty egress` last saved.
    record = (store or EgressStore()).load()
    if record is None:
        return None

    password: str | None = None
    if record.auth_ref:
        password = cred_store.get(record.auth_ref) if cred_store is not None else None
        if not password:
            raise ValueError(
                f"Egress gateway {record.proxy_url} needs a password but its stored credential "
                f"({record.auth_ref}) could not be resolved. Run 'kitty egress' to reconfigure it."
            )

    return EgressConfig(proxy_url=record.proxy_url, username=record.username, password=password)
