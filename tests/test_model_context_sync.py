"""Tests for model_context_sync — runtime refresh of the overrides catalog.

The sync fetches the catalog from the kitty-bridge GitHub repo once per
session start, caches it under ``model_context.REMOTE_OVERRIDES_CACHE_PATH``
with a TTL, and never raises: any failure keeps the previous state (R3).
All HTTP is mocked with ``aioresponses``; cache freshness is controlled via
``os.utime``, never ``sleep``.
"""

import os
import time
from pathlib import Path

import aiohttp
import pytest
from aioresponses import aioresponses

EXPECTED_URL = (
    "https://raw.githubusercontent.com/Shelpuk-AI-Technology-Consulting/"
    "kitty-bridge/main/src/kitty/providers/model_context_overrides.json"
)

VALID_BODY = '{"new-model": 12345}'


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point REMOTE_OVERRIDES_CACHE_PATH at a tmp path and clear loader caches.

    The sync module reads the constant from ``model_context`` at call time,
    so patching the attribute there covers both the reader and the writer.
    """
    import kitty.providers.model_context as mc

    monkeypatch.setattr(
        mc,
        "REMOTE_OVERRIDES_CACHE_PATH",
        tmp_path / "cache" / "model_context_overrides.json",
    )
    mc._load_overrides.cache_clear()
    yield
    mc._load_overrides.cache_clear()


def _cache_path() -> Path:
    """Return the patched cache path."""
    import kitty.providers.model_context as mc

    return mc.REMOTE_OVERRIDES_CACHE_PATH


def _write_cache(body: str, *, age_seconds: float = 0.0) -> None:
    """Write a cache file and optionally backdate its mtime."""
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    if age_seconds:
        mtime = time.time() - age_seconds
        os.utime(path, (mtime, mtime))


# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------


class TestContract:
    """The fetch URL and TTL defaults are stable, documented contracts."""

    def test_url_points_at_repo_main_branch(self):
        import kitty.providers.model_context_sync as mcs

        assert mcs.REMOTE_OVERRIDES_URL == EXPECTED_URL

    def test_default_ttl_is_24_hours(self):
        import kitty.providers.model_context_sync as mcs

        assert mcs.DEFAULT_TTL_SECONDS == 24 * 3600

    async def test_session_uses_egress_kwargs_and_bounded_timeout(self, monkeypatch: pytest.MonkeyPatch):
        """The aiohttp session is built with egress kwargs and a 10 s timeout."""
        import kitty.providers.model_context_sync as mcs

        seen_kwargs: dict = {}
        real_session_cls = aiohttp.ClientSession

        def spy(**kwargs):
            seen_kwargs.update(kwargs)
            return real_session_cls(**kwargs)

        egress_calls = []

        def fake_egress_kwargs():
            egress_calls.append(1)
            return {"proxy": "http://egress.test:3128"}

        monkeypatch.setattr(mcs.aiohttp, "ClientSession", spy)
        monkeypatch.setattr(mcs, "aiohttp_session_kwargs", fake_egress_kwargs)

        with aioresponses() as mocked:
            mocked.get(mcs.REMOTE_OVERRIDES_URL, status=200, body=VALID_BODY)
            result = await mcs.refresh_model_context_overrides()

        assert result is True
        assert egress_calls == [1]
        assert seen_kwargs["timeout"].total == 10
        assert seen_kwargs["proxy"] == "http://egress.test:3128"


# ---------------------------------------------------------------------------
# TTL skip
# ---------------------------------------------------------------------------


class TestTtlSkip:
    """A cache younger than the TTL is kept without any HTTP request.

    No URL is registered with ``aioresponses`` in these tests: any attempted
    request would raise and the sync would return ``False`` — so asserting
    ``True`` proves no HTTP happened.
    """

    async def test_fresh_cache_skips_http(self):
        import kitty.providers.model_context_sync as mcs

        _write_cache('{"cached-model": 4242}')
        with aioresponses():
            result = await mcs.refresh_model_context_overrides()

        assert result is True
        assert _cache_path().read_text(encoding="utf-8") == '{"cached-model": 4242}'

    async def test_cache_just_inside_ttl_is_fresh(self):
        import kitty.providers.model_context_sync as mcs

        _write_cache('{"cached-model": 4242}', age_seconds=mcs.DEFAULT_TTL_SECONDS - 60)
        with aioresponses():
            result = await mcs.refresh_model_context_overrides()

        assert result is True

    async def test_custom_ttl_parameter_is_honoured(self):
        """A cache older than the given ttl_seconds is refetched."""
        import kitty.providers.model_context_sync as mcs

        _write_cache('{"cached-model": 4242}', age_seconds=120)
        with aioresponses() as mocked:
            mocked.get(mcs.REMOTE_OVERRIDES_URL, status=200, body=VALID_BODY)
            result = await mcs.refresh_model_context_overrides(ttl_seconds=60)

        assert result is True
        assert _cache_path().read_text(encoding="utf-8") == VALID_BODY


# ---------------------------------------------------------------------------
# Successful fetch
# ---------------------------------------------------------------------------


class TestFetch:
    """A stale or missing cache triggers a GET; success writes the body."""

    async def test_missing_cache_fetches_and_writes_body_exactly(self):
        import kitty.providers.model_context_sync as mcs

        with aioresponses() as mocked:
            mocked.get(mcs.REMOTE_OVERRIDES_URL, status=200, body=VALID_BODY)
            result = await mcs.refresh_model_context_overrides()

        assert result is True
        assert _cache_path().read_text(encoding="utf-8") == VALID_BODY

    async def test_stale_cache_refetches(self):
        import kitty.providers.model_context_sync as mcs

        _write_cache('{"stale-model": 1}', age_seconds=mcs.DEFAULT_TTL_SECONDS + 60)
        with aioresponses() as mocked:
            mocked.get(mcs.REMOTE_OVERRIDES_URL, status=200, body=VALID_BODY)
            result = await mcs.refresh_model_context_overrides()

        assert result is True
        assert _cache_path().read_text(encoding="utf-8") == VALID_BODY

    async def test_cache_at_exact_ttl_age_is_stale(self):
        import kitty.providers.model_context_sync as mcs

        _write_cache('{"stale-model": 1}', age_seconds=mcs.DEFAULT_TTL_SECONDS)
        with aioresponses() as mocked:
            mocked.get(mcs.REMOTE_OVERRIDES_URL, status=200, body=VALID_BODY)
            result = await mcs.refresh_model_context_overrides()

        assert result is True
        assert _cache_path().read_text(encoding="utf-8") == VALID_BODY

    async def test_success_clears_overrides_lru_cache(self):
        """Resolution sees the new catalog without a manual cache clear."""
        import kitty.providers.model_context as mc
        import kitty.providers.model_context_sync as mcs

        _write_cache('{"old-model": 111}', age_seconds=mcs.DEFAULT_TTL_SECONDS + 60)
        # Warm the loader cache with the stale catalog.
        assert mc.get_model_context_tokens("any", "old-model") == 111

        with aioresponses() as mocked:
            mocked.get(mcs.REMOTE_OVERRIDES_URL, status=200, body='{"new-model": 222}')
            result = await mcs.refresh_model_context_overrides()

        assert result is True
        assert mc.get_model_context_tokens("any", "new-model") == 222


# ---------------------------------------------------------------------------
# Failure modes — never raise, keep prior state
# ---------------------------------------------------------------------------


class TestFailureModes:
    """Every failure returns False and leaves an existing cache byte-identical."""

    @pytest.mark.parametrize(
        ("register", "body"),
        [
            ("network-error", None),
            ("http-500", None),
            ("invalid-json", "not json"),
            ("non-object-root", "[1, 2]"),
            ("zero-coercible-entries", '{"a": "1M", "b": -5}'),
        ],
    )
    async def test_failure_returns_false_and_keeps_cache(self, register, body):
        import kitty.providers.model_context_sync as mcs

        prior = '{"prior-model": 777}'
        _write_cache(prior, age_seconds=mcs.DEFAULT_TTL_SECONDS + 60)

        with aioresponses() as mocked:
            if register == "network-error":
                mocked.get(mcs.REMOTE_OVERRIDES_URL, exception=aiohttp.ClientConnectionError())
            elif register == "http-500":
                mocked.get(mcs.REMOTE_OVERRIDES_URL, status=500)
            else:
                mocked.get(mcs.REMOTE_OVERRIDES_URL, status=200, body=body)
            result = await mcs.refresh_model_context_overrides()

        assert result is False
        assert _cache_path().read_text(encoding="utf-8") == prior

    async def test_write_failure_returns_false_without_raising(self):
        """A cache path that is a directory breaks both read and write."""
        import kitty.providers.model_context_sync as mcs

        _cache_path().mkdir(parents=True, exist_ok=True)
        with aioresponses() as mocked:
            mocked.get(mcs.REMOTE_OVERRIDES_URL, status=200, body=VALID_BODY)
            result = await mcs.refresh_model_context_overrides()

        assert result is False


# ---------------------------------------------------------------------------
# Body validation boundary
# ---------------------------------------------------------------------------


class TestValidationBoundary:
    """One coercible entry is enough — garbage siblings don't sink the fetch."""

    async def test_at_least_one_coercible_entry_is_valid(self):
        import kitty.providers.model_context_sync as mcs

        mixed = '{"good": 1000, "bad": "1M"}'
        with aioresponses() as mocked:
            mocked.get(mcs.REMOTE_OVERRIDES_URL, status=200, body=mixed)
            result = await mcs.refresh_model_context_overrides()

        assert result is True
        # The raw fetched body is cached as-is; the loader drops "bad" later.
        assert _cache_path().read_text(encoding="utf-8") == mixed
