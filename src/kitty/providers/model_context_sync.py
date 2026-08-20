"""Runtime refresh of the model-context overrides catalog.

Fetches ``model_context_overrides.json`` from the kitty-bridge GitHub repo
once per session start and caches it under
``model_context.REMOTE_OVERRIDES_CACHE_PATH``. A cached copy younger than the
TTL suppresses the HTTP request; any failure degrades to the existing state
(cached copy, else packaged copy) without raising, so network problems never
block a session.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time

import aiohttp

from kitty.egress import aiohttp_session_kwargs
from kitty.providers import model_context

logger = logging.getLogger(__name__)

REMOTE_OVERRIDES_URL = (
    "https://raw.githubusercontent.com/Shelpuk-AI-Technology-Consulting/"
    "kitty-bridge/main/src/kitty/providers/model_context_overrides.json"
)

DEFAULT_TTL_SECONDS = 24 * 3600

_FETCH_TIMEOUT_SECONDS = 10


def _cache_is_fresh(ttl_seconds: float) -> bool:
    """Return True when the cached catalog exists and is younger than the TTL.

    Args:
        ttl_seconds: Maximum cache age in seconds.

    Returns:
        True when the cache file exists and its mtime is within the TTL.
    """
    path = model_context.REMOTE_OVERRIDES_CACHE_PATH
    try:
        # A non-file at the cache path (e.g. a directory) is not a cache.
        if not path.is_file():
            return False
        mtime = path.stat().st_mtime
    except OSError:
        return False
    return (time.time() - mtime) < ttl_seconds


def _body_is_valid(raw: str) -> bool:
    """Validate a fetched catalog body.

    Valid means: parses as a JSON object **and** contains at least one entry
    that passes :func:`model_context._coerce_context_tokens`. A syntactically
    valid but entry-wise garbage catalog (an upstream editing mistake) must
    not replace the existing cache — under wholesale-replace semantics it
    would wipe every override on every installation.

    Args:
        raw: The fetched response body.

    Returns:
        True when the body may be cached.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    return any(model_context._coerce_context_tokens(value) is not None for value in data.values())


async def refresh_model_context_overrides(*, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> bool:
    """Refresh the overrides catalog from the kitty-bridge GitHub repo.

    Best-effort by contract: never raises. On any failure the previous state
    (cached copy, else packaged copy) stays in effect and a warning is
    logged. Called at every session-start entry point before the bridge
    server starts.

    Args:
        ttl_seconds: Minimum cache age in seconds before a refetch is
            attempted. A fresher cache suppresses the HTTP request entirely;
            the TTL governs fetch attempts only — a valid cached copy older
            than the TTL still outranks the packaged catalog.

    Returns:
        True when the catalog is current after the call (fresh cache or
        successful fetch), False when a refresh was attempted and failed.
    """
    # A fresh cached copy suppresses the fetch entirely.
    if _cache_is_fresh(ttl_seconds):
        return True

    try:
        return await _fetch_and_store()
    except Exception:
        logger.warning("Failed to refresh model context overrides from %s", REMOTE_OVERRIDES_URL, exc_info=True)
        return False


async def _fetch_and_store() -> bool:
    """Fetch the remote catalog and atomically cache it.

    Unexpected errors propagate to :func:`refresh_model_context_overrides`,
    which converts them to ``False``.

    Returns:
        True on success, False on a non-200 response or an invalid body.
    """
    # Egress kwargs are mandatory: aiohttp ignores HTTP(S)_PROXY env vars.
    timeout = aiohttp.ClientTimeout(total=_FETCH_TIMEOUT_SECONDS)
    async with (
        aiohttp.ClientSession(timeout=timeout, **aiohttp_session_kwargs()) as session,
        session.get(REMOTE_OVERRIDES_URL) as response,
    ):
        if response.status != 200:
            logger.warning("Model context overrides refresh got HTTP %d", response.status)
            return False
        raw = await response.text()

    if not _body_is_valid(raw):
        logger.warning("Model context overrides refresh got an invalid body; keeping previous catalog")
        return False

    # Atomic replace so a crash mid-write cannot leave a truncated cache.
    cache_path = model_context.REMOTE_OVERRIDES_CACHE_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(cache_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(raw)
        os.replace(tmp_name, cache_path)
    finally:
        # After a successful replace the temp file no longer exists.
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    model_context._load_overrides.cache_clear()
    logger.debug("Model context overrides refreshed from %s", REMOTE_OVERRIDES_URL)
    return True
