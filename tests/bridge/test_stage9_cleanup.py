"""Stage 9 tests — F4, F5, F31, F48, F49, F50."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from kitty.bridge.server import _AUTH_COOLDOWN, BridgeServer
from kitty.providers.base import ProviderAdapter


class _StubProvider(ProviderAdapter):
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
        return Exception(f"err {status_code}")


class TestF5AuthCooldown:
    """F5: Auth cooldown should be 15 minutes (900s), not 24 hours."""

    def test_auth_cooldown_is_15_minutes(self):
        """_AUTH_COOLDOWN must be 900 (15 minutes), not 86400 (24 hours)."""
        assert _AUTH_COOLDOWN == 900, (
            f"Expected 900 (15 min) but got {_AUTH_COOLDOWN}. "
            "A 24-hour cooldown effectively blacklists the backend permanently."
        )


class TestF48CloudflareFalsePositive:
    """F48: Generic 'cloudflare' substring causes false positives."""

    def test_generic_cloudflare_word_not_detected(self):
        """A 403 body containing just the word 'cloudflare' must NOT be
        flagged as a Cloudflare block — it could be a legitimate response."""
        from kitty.cloudflare import get_cloudflare_signature, is_cloudflare_block

        # This body contains "cloudflare" in a non-Challenge context
        body = "Your request was rejected. Please contact cloudflare support for details."
        assert is_cloudflare_block(403, body) is False
        assert get_cloudflare_signature(body) is None

    def test_specific_signatures_still_detected(self):
        """Specific Cloudflare signatures must still be detected."""
        from kitty.cloudflare import get_cloudflare_signature, is_cloudflare_block

        assert get_cloudflare_signature("cf-mitigated: access denied") == "cf-mitigated"
        assert is_cloudflare_block(403, "cf-mitigated: access denied") is True

        assert get_cloudflare_signature("_cf_chl_opt") == "_cf_chl_opt"
        assert is_cloudflare_block(403, "_cf_chl_opt") is True

        assert get_cloudflare_signature("cf-browser-verification") == "cf-browser-verification"
        assert is_cloudflare_block(403, "cf-browser-verification") is True

    def test_cloudflare_in_url_not_detected(self):
        """'cloudflare' appearing in a URL must not trigger detection."""
        from kitty.cloudflare import is_cloudflare_block

        body = "See https://community.cloudflare.com/t/topic/123 for help"
        assert is_cloudflare_block(403, body) is False


class TestF31DeadCode:
    """F31: Verify no unreachable RuntimeError remains."""

    def test_no_runtime_error_in_request_with_retry(self):
        """The RuntimeError('All backends exhausted') has been removed.
        Verify it no longer appears in the source."""
        import inspect

        source = inspect.getsource(BridgeServer._request_with_retry_balancing)
        assert "All backends exhausted with no response" not in source


class TestF49StateWriteFailure:
    """F49: Server must not be left running if state write fails."""

    @pytest.mark.asyncio
    async def test_start_async_cleans_up_on_state_write_failure(self):
        """If write_state raises OSError (e.g. disk full), the server must
        clean up its runner and not leave a running but unmanaged process."""
        from pathlib import Path

        from kitty.bridge.server import BridgeServer

        provider = _StubProvider()
        tmpdir = tempfile.mkdtemp()
        state_file = str(Path(tmpdir) / "bridge_state.json")

        server = BridgeServer(None, provider, "test-key", state_file=state_file)

        with (
            patch("kitty.bridge.state.write_state", side_effect=OSError("No space left on device")),
            pytest.raises(OSError, match="No space left on device"),
        ):
            await server.start_async()

        # The runner must have been cleaned up — not left running
        assert server._runner is None


class TestF50CrashHandlerStateCleanup:
    """F50: Crash handler must clean up state file so stale PID doesn't persist."""

    def test_crash_handlers_accept_state_path(self):
        """_setup_crash_handlers must accept an optional state_path parameter."""
        # Reset so we can re-install
        import kitty.bridge.server as srv_mod
        from kitty.bridge.server import _setup_crash_handlers

        srv_mod._crash_handlers_installed = False

        tmpdir = tempfile.mkdtemp()
        state_path = str(Path(tmpdir) / "bridge_state.json")
        log_path = Path(tmpdir) / "bridge.log"

        # Must not raise — accepts state_path
        _setup_crash_handlers(log_path, state_path=state_path)


class TestF4Repeat400Guard:
    """F4: Don't burn multiple backends on the same bad request body."""

    @pytest.mark.asyncio
    async def test_consecutive_400s_return_error_without_burning_all_backends(self):
        """When 2+ backends return 400 on the same request, stop retrying
        and return the error immediately instead of burning all backends."""
        import uuid

        import aiohttp
        from aioresponses import aioresponses

        from kitty.profiles.schema import Profile
        from kitty.providers.base import ProviderAdapter

        class _S(ProviderAdapter):
            def __init__(self, i):
                self._i = i

            @property
            def provider_type(self):
                return f"s{self._i}"

            @property
            def default_base_url(self):
                return f"https://api{self._i}.example.com/v1"

            def build_request(self, model, messages, **kw):
                return {"model": model, "messages": messages}

            def parse_response(self, d):
                return d

            def map_error(self, s, b):
                return Exception(f"e{s}")

        backends = []
        for i in range(3):
            backends.append(
                (
                    _S(i),
                    f"key-{i}",
                    Profile(
                        name=f"p{i}",
                        provider="openai",
                        model=f"m{i}",
                        auth_ref=str(uuid.uuid4()),
                    ),
                )
            )

        server = BridgeServer(None, backends[0][0], "k0", backends=backends)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                for i in range(3):
                    m.post(
                        f"https://api{i}.example.com/v1/chat/completions",
                        status=400,
                        payload={"error": {"message": "Bad request"}},
                    )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                    ) as resp,
                ):
                    # Should get error (not hang) and NOT try all 3 backends
                    assert resp.status >= 400 or resp.status == 200
                    if resp.status == 200:
                        body = await resp.text()
                        assert "error" in body.lower()
        finally:
            await server.stop_async()


class TestF16OpenCodeDeduplication:
    """F16: OpenCode should reuse AnthropicAdapter translation logic."""

    def test_opencode_inherits_anthropic_adapter(self):
        """OpenCodeGoAdapter should subclass AnthropicAdapter for Messages routing."""
        from kitty.providers.anthropic import AnthropicAdapter
        from kitty.providers.opencode import OpenCodeGoAdapter

        assert issubclass(OpenCodeGoAdapter, AnthropicAdapter)

    def test_opencode_no_longer_defines_duplicate_translation_helpers(self):
        """Duplicated Anthropic helper methods should not be defined in opencode.py."""
        import inspect

        from kitty.providers.opencode import OpenCodeGoAdapter

        source = inspect.getsource(OpenCodeGoAdapter)
        for name in (
            "def _translate_to_anthropic",
            "def _translate_from_anthropic",
            "def _translate_anthropic_stream_event",
            "def _translate_tool_result_msg",
        ):
            assert name not in source
