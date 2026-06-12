"""OpenAI OAuth session: token state, refresh, and file persistence."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

# OpenAI OAuth endpoints
OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
ID_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:id_token"

# Default expiry when not provided by the server (1 hour)
_DEFAULT_EXPIRES_IN = 3600

# Proactive refresh margin (seconds before expiry)
_REFRESH_MARGIN_SECONDS = 60

# HTTP timeout for OAuth token operations
_OAUTH_TIMEOUT = aiohttp.ClientTimeout(total=30)


class OAuthError(Exception):
    """Base exception for OAuth errors."""

    def __init__(self, error: str, error_description: str | None = None) -> None:
        self.error = error
        self.error_description = error_description or ""
        super().__init__(f"{error}: {error_description}" if error_description else error)


class OAuthRefreshFailed(OAuthError):
    """Raised when token refresh fails (e.g. revoked refresh token)."""

    def __init__(self, error: str, error_description: str | None = None) -> None:
        super().__init__(error, error_description)


class OAuthTokenExchangeFailed(OAuthError):
    """Raised when token-exchange grant fails (e.g. expired session)."""

    def __init__(self, error: str, error_description: str | None = None) -> None:
        super().__init__(error, error_description)


@dataclass
class OAuthSession:
    """OAuth session state for OpenAI Codex subscription auth.

    Encapsulates access_token, refresh_token, id_token, and the exchanged API key,
    along with expiry timestamps. Provides transparent auto-refresh and
    file-based persistence.
    """

    client_id: str
    access_token: str
    refresh_token: str
    id_token: str
    api_key: str | None
    access_token_expires_at: float  # epoch seconds
    api_key_expires_at: float  # epoch seconds
    _file_path: str | None = field(default=None, repr=False)
    _refresh_lock: asyncio.Lock = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._refresh_lock is None:
            object.__setattr__(self, "_refresh_lock", asyncio.Lock())

    # ── Factory from token endpoint response ─────────────────────────────────

    @classmethod
    def from_token_response(cls, payload: dict, client_id: str) -> OAuthSession:
        """Build a session from an OAuth token endpoint response.

        Args:
            payload: Parsed JSON from POST /oauth/token.
            client_id: The OAuth client ID.

        Returns:
            A new OAuthSession with expiry set from expires_in.
        """
        now = time.time()
        expires_in = payload.get("expires_in", _DEFAULT_EXPIRES_IN)
        return cls(
            client_id=client_id,
            access_token=payload["access_token"],
            refresh_token=payload["refresh_token"],
            id_token=payload["id_token"],
            api_key=payload.get("api_key"),
            access_token_expires_at=now + expires_in,
            api_key_expires_at=now + expires_in,
            _file_path=None,
        )

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict (excludes _file_path)."""
        return {
            "client_id": self.client_id,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "id_token": self.id_token,
            "api_key": self.api_key,
            "access_token_expires_at": self.access_token_expires_at,
            "api_key_expires_at": self.api_key_expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OAuthSession:
        """Reconstruct from a JSON dict (no _file_path)."""
        return cls(
            client_id=data["client_id"],
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            id_token=data["id_token"],
            api_key=data.get("api_key"),
            access_token_expires_at=data["access_token_expires_at"],
            api_key_expires_at=data["api_key_expires_at"],
            _file_path=data.get("_file_path"),  # may be absent in old files
        )

    # ── File persistence ───────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path) -> OAuthSession:
        """Load a session from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        session = cls.from_dict(data)
        session._file_path = str(path)
        return session

    def save(self) -> None:
        """Write the session to _file_path atomically (temp file + rename).

        F47: Uses filelock to prevent concurrent processes from overwriting
        each other's session data.
        """
        if self._file_path is None:
            raise ValueError("Cannot save: _file_path is not set")

        import filelock

        path = Path(self._file_path)
        lock_path = str(path) + ".lock"
        lock = filelock.FileLock(lock_path, timeout=5)
        with lock:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
            # Set restrictive permissions before rename to avoid a brief
            # world-readable window between the replace() and chmod().
            tmp.chmod(0o600)
            tmp.replace(path)
        logger.debug("OAuth session saved to %s", path)

    @classmethod
    def create_session_file(cls, session: OAuthSession, auth_ref: str, config_dir: Path) -> OAuthSession:
        """Set _file_path and persist the session.

        Args:
            session: The OAuthSession to save.
            auth_ref: UUIDv4 string used as the filename.
            config_dir: Kitty config directory (~/.config/kitty).

        Returns:
            The same session with _file_path set.
        """
        oauth_dir = config_dir / "openai_oauth"
        oauth_dir.mkdir(parents=True, exist_ok=True)
        session._file_path = str(oauth_dir / f"{auth_ref}.json")
        session.save()
        return session

    # ── Expiry helpers ───────────────────────────────────────────────────

    @property
    def access_token_expired(self) -> bool:
        """Return True if the access token is past its expiry time."""
        return time.time() >= self.access_token_expires_at

    @property
    def api_key_expired(self) -> bool:
        """Return True if the exchanged API key is past its expiry time."""
        return time.time() >= self.api_key_expires_at

    @property
    def _should_proactive_refresh(self) -> bool:
        """Return True if the access token expires within the refresh margin."""
        return time.time() >= (self.access_token_expires_at - _REFRESH_MARGIN_SECONDS)

    # ── Token refresh ─────────────────────────────────────────────────────

    async def _refresh(self, http: aiohttp.ClientSession) -> None:
        """Refresh tokens using the refresh_token grant.

        After successful refresh, updates access_token, refresh_token, id_token,
        and expiry times. Attempts to re-exchange for a new API key (best-effort;
        failure is logged but does not block the session). Persists the updated
        session to disk.

        Args:
            http: aiohttp session for HTTP calls.

        Raises:
            OAuthRefreshFailed: If the refresh grant fails.
        """
        # Step 1: Refresh tokens using refresh_token grant
        refresh_payload = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
        }
        async with http.post(OAUTH_TOKEN_URL, data=refresh_payload, timeout=_OAUTH_TIMEOUT) as resp:
            if resp.status >= 400:
                body = {}
                with contextlib.suppress(Exception):
                    body = await resp.json()
                raise OAuthRefreshFailed(
                    body.get("error", f"HTTP {resp.status}"),
                    body.get("error_description"),
                )
            tokens = await resp.json()

        self.access_token = tokens["access_token"]
        self.refresh_token = tokens.get("refresh_token", self.refresh_token)  # rotation
        self.id_token = tokens["id_token"]
        expires_in = tokens.get("expires_in", _DEFAULT_EXPIRES_IN)
        self.access_token_expires_at = time.time() + expires_in

        # Step 2: Re-exchange id_token for a new API key (best-effort).
        # Fails for org accounts without Platform API org mapping — in that
        # case we use the access_token directly as Bearer auth.
        try:
            api_key = await self._exchange_api_key(http)
            self.api_key = api_key
        except OAuthTokenExchangeFailed:
            logger.warning("API key exchange failed during refresh; using access_token directly")
            self.api_key = None
        self.api_key_expires_at = time.time() + expires_in

        # Persist immediately after successful refresh
        if self._file_path:
            self.save()

        logger.info("OAuth tokens refreshed successfully")

    async def _exchange_api_key(self, http: aiohttp.ClientSession) -> str:
        """Exchange id_token for an API key via token-exchange grant.

        Args:
            http: aiohttp session for HTTP calls.

        Returns:
            The new API key string.

        Raises:
            OAuthTokenExchangeFailed: If the token-exchange fails.
        """
        payload = {
            "grant_type": TOKEN_EXCHANGE_GRANT,
            "requested_token": "openai-api-key",
            "subject_token": self.id_token,
            "subject_token_type": ID_TOKEN_TYPE,
            "client_id": self.client_id,
        }
        async with http.post(OAUTH_TOKEN_URL, data=payload, timeout=_OAUTH_TIMEOUT) as resp:
            if resp.status >= 400:
                body = {}
                with contextlib.suppress(Exception):
                    body = await resp.json()
                raise OAuthTokenExchangeFailed(
                    body.get("error", "token_exchange_failed"),
                    body.get("error_description"),
                )
            result = await resp.json()
        return result["openai_api_key"]

    # ── Public API ─────────────────────────────────────────────────────────

    async def get_valid_api_key(self, http: aiohttp.ClientSession, *, force_refresh: bool = False) -> str:
        """Return a valid bearer token, refreshing if needed.

        Returns the exchanged API key if available; otherwise falls back to the
        access_token (used for ChatGPT subscription accounts where the
        API-key exchange is not available).

        Args:
            http: aiohttp session for HTTP calls.
            force_refresh: If True, skip expiry checks and force a token refresh
                (used when the API key has been rejected with 401).

        Returns:
            A valid bearer token string (api_key or access_token).

        Raises:
            OAuthRefreshFailed: If refresh fails and cannot be recovered.
        """
        async with self._refresh_lock:
            # Re-check after acquiring the lock — another coroutine may have refreshed
            if (
                not force_refresh
                and not self.access_token_expired
                and not self.api_key_expired
                and not self._should_proactive_refresh
            ):
                return self.api_key or self.access_token

            try:
                await self._refresh(http)
            except OAuthRefreshFailed:
                raise
            except Exception as exc:
                raise OAuthRefreshFailed(
                    "refresh_failed",
                    f"Network or unexpected error during token refresh: {exc}",
                ) from exc

            return self.api_key or self.access_token
