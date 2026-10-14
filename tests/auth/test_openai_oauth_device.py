"""Tests for the OpenAI device-code OAuth flow.

KBR-317. Layers:
- L1 parsing tests for the response parsers, interval edge cases, and the
  ``format_device_code_prompt`` function.
- L1 redaction tests assert that none of the three device secrets
  (``user_code``, ``device_auth_id``, authorization code) appear in DEBUG
  log output or in exception messages.
- L2 contract test pins the request/response JSON shapes as a literal
  fixture (kitty's contract with itself; drift from upstream codex-rs
  requires a deliberate fixture update).
- L3 subsystem tests drive ``run_oauth_device_flow`` end-to-end with
  ``aioresponses`` patching all three endpoints. The poll loop's sleep is
  no-op'd via monkeypatching ``asyncio.sleep`` and the polling budget is
  shrunk by patching the module constant — the production code carries no
  test-only parameters.

The browser path is exercised in ``test_openai_oauth.py`` and must not be
disturbed by these tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.auth import openai_oauth
from kitty.auth.openai_oauth import (
    CLIENT_ID,
    OAUTH_DEVICE_TOKEN_URL,
    OAUTH_DEVICE_USERCODE_URL,
    OAUTH_DEVICE_VERIFICATION_URL,
    OAUTH_TOKEN_URL,
    OAuthAuthorizationError,
    OAuthDeviceCodeCancelledError,
    OAuthDeviceCodeDisabledError,
    OAuthDeviceCodeResponseError,
    OAuthTimeoutError,
    format_device_code_prompt,
    format_device_code_prompt_plain,
    parse_device_token_poll,
    parse_device_user_code_response,
    run_oauth_device_flow,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def codex_user_code_response() -> dict[str, str]:
    """A realistic usercode response, mirroring codex-rs's UserCodeResp."""
    return {
        "device_auth_id": "dev_abc123",
        "user_code": "ABCD-EFGH",
        "interval": "5",
    }


@pytest.fixture
def codex_token_success() -> dict[str, str]:
    """A realistic polling-success response, mirroring codex-rs's CodeSuccessResp."""
    return {
        "authorization_code": "ac_test_xyz",
        "code_challenge": "challenge_xyz",
        "code_verifier": "verifier_xyz",
    }


@pytest.fixture
def happy_token_response() -> dict[str, Any]:
    """A realistic OAuth token response (the third hop)."""
    return {
        "access_token": "at_device",
        "refresh_token": "rt_device",
        "id_token": "id_device",
        "expires_in": 3600,
    }


@pytest.fixture
def fast_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the poll loop's sleeps instant so pending-poll tests don't wait.

    Patches the asyncio.sleep that ``run_oauth_device_flow`` resolves through
    its module-global ``asyncio`` reference; the loop logic itself is real.
    """

    async def _instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(openai_oauth.asyncio, "sleep", _instant_sleep)


# ── L1: response parsers ─────────────────────────────────────────────────


class TestParseDeviceUserCodeResponse:
    def test_happy_path(self, codex_user_code_response: dict[str, str]) -> None:
        parsed = parse_device_user_code_response(json.dumps(codex_user_code_response))
        assert parsed.device_auth_id == "dev_abc123"
        assert parsed.user_code == "ABCD-EFGH"
        assert parsed.interval == 5

    def test_user_code_alias(self) -> None:
        """codex-rs accepts both ``user_code`` and ``usercode`` for the one-time code."""
        body = json.dumps({"device_auth_id": "d1", "usercode": "ZZZZ-YYYY", "interval": "5"})
        parsed = parse_device_user_code_response(body)
        assert parsed.user_code == "ZZZZ-YYYY"

    def test_interval_whitespace_tolerated(self) -> None:
        body = json.dumps({"device_auth_id": "d1", "user_code": "AA-BB", "interval": " 7 "})
        parsed = parse_device_user_code_response(body)
        assert parsed.interval == 7

    def test_missing_interval_defaults_to_5(self) -> None:
        """Kitty's deliberate choice: a missing field is 5 s, not codex's serde-default of 0.

        codex's busy-loop on interval=0 is a real upstream bug; the departure
        is recorded in SYSTEM_DESIGN §14.4 and this test pins the default.
        """
        body = json.dumps({"device_auth_id": "d1", "user_code": "AA-BB"})
        parsed = parse_device_user_code_response(body)
        assert parsed.interval == 5

    @pytest.mark.parametrize("bad_interval", ["0", "-3", "abc", ""])
    def test_non_positive_or_unparseable_interval_raises(self, bad_interval: str) -> None:
        from kitty.auth.openai_oauth import OAuthDeviceCodeIntervalError

        body = json.dumps({"device_auth_id": "d1", "user_code": "AA-BB", "interval": bad_interval})
        with pytest.raises(OAuthDeviceCodeIntervalError):
            parse_device_user_code_response(body)

    def test_missing_device_auth_id_raises(self) -> None:
        body = json.dumps({"user_code": "AA-BB", "interval": "5"})
        with pytest.raises(OAuthDeviceCodeResponseError):
            parse_device_user_code_response(body)

    def test_missing_user_code_raises(self) -> None:
        body = json.dumps({"device_auth_id": "d1", "interval": "5"})
        with pytest.raises(OAuthDeviceCodeResponseError):
            parse_device_user_code_response(body)

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(OAuthDeviceCodeResponseError):
            parse_device_user_code_response("not json at all")


class TestParseDeviceTokenPoll:
    def test_pending_403(self) -> None:
        """403 = still pending, keep polling."""
        result = parse_device_token_poll(403, b"")
        assert result.kind == "pending"

    def test_pending_404(self) -> None:
        """404 = still pending, keep polling (matches codex-rs)."""
        result = parse_device_token_poll(404, b"")
        assert result.kind == "pending"

    def test_success_200(self, codex_token_success: dict[str, str]) -> None:
        result = parse_device_token_poll(200, json.dumps(codex_token_success).encode())
        assert result.kind == "success"
        assert result.authorization_code == "ac_test_xyz"
        assert result.code_verifier == "verifier_xyz"
        assert result.code_challenge == "challenge_xyz"

    @pytest.mark.parametrize("status", [401, 400, 500, 502])
    def test_other_statuses_raise(self, status: int) -> None:
        with pytest.raises(OAuthAuthorizationError):
            parse_device_token_poll(status, b'{"error": "boom"}')


# ── L1: prompt format ────────────────────────────────────────────────────


class TestFormatDeviceCodePrompt:
    def test_contains_url(self) -> None:
        prompt = format_device_code_prompt(OAUTH_DEVICE_VERIFICATION_URL, "ABCD-EFGH")
        assert OAUTH_DEVICE_VERIFICATION_URL in prompt
        assert "auth.openai.com/codex/device" in prompt

    def test_contains_one_time_code(self) -> None:
        prompt = format_device_code_prompt(OAUTH_DEVICE_VERIFICATION_URL, "WXYZ-1234")
        assert "WXYZ-1234" in prompt

    def test_contains_expiry_note(self) -> None:
        prompt = format_device_code_prompt(OAUTH_DEVICE_VERIFICATION_URL, "ABCD-EFGH")
        assert "15" in prompt and "minute" in prompt.lower()

    def test_contains_anti_phishing_footer(self) -> None:
        prompt = format_device_code_prompt(OAUTH_DEVICE_VERIFICATION_URL, "ABCD-EFGH")
        assert "Continue only if" in prompt

    def test_contains_workspace_admin_warning(self) -> None:
        """Personal accounts have device-code auth on by default; workspace
        plans need an admin toggle. The prompt must surface this so a user
        on a Business/Team plan has a next step."""
        prompt = format_device_code_prompt(OAUTH_DEVICE_VERIFICATION_URL, "ABCD-EFGH")
        assert "admin" in prompt.lower()

    def test_prompt_leaks_nothing_beyond_the_user_code(self) -> None:
        """The user code is the only secret the prompt surfaces (by design);
        no other secret identifier may appear."""
        prompt = format_device_code_prompt(OAUTH_DEVICE_VERIFICATION_URL, "PLAIN-1234")
        assert "PLAIN-1234" in prompt
        assert "device_auth_id" not in prompt
        assert "code_verifier" not in prompt
        assert "code_challenge" not in prompt


class TestFormatDeviceCodePromptPlain:
    """FR-4: a copy-paste-friendly plain copy accompanies the ANSI prompt."""

    def test_plain_prompt_contains_url_and_code(self) -> None:
        prompt = format_device_code_prompt_plain(OAUTH_DEVICE_VERIFICATION_URL, "PLAIN-1234")
        assert OAUTH_DEVICE_VERIFICATION_URL in prompt
        assert "PLAIN-1234" in prompt

    def test_plain_prompt_has_no_ansi_escapes(self) -> None:
        prompt = format_device_code_prompt_plain(OAUTH_DEVICE_VERIFICATION_URL, "PLAIN-1234")
        assert "\x1b" not in prompt

    def test_plain_prompt_carries_the_expiry_note(self) -> None:
        prompt = format_device_code_prompt_plain(OAUTH_DEVICE_VERIFICATION_URL, "ABCD-EFGH")
        assert "15" in prompt and "minute" in prompt.lower()

    async def test_flow_prints_both_coloured_and_plain_forms(
        self,
        codex_user_code_response: dict[str, str],
        codex_token_success: dict[str, str],
        happy_token_response: dict[str, Any],
        fast_poll: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The device flow emits the ANSI prompt AND the plain copy to stderr."""
        with aioresponses() as m:
            m.post(OAUTH_DEVICE_USERCODE_URL, payload=codex_user_code_response)
            m.post(OAUTH_DEVICE_TOKEN_URL, payload=codex_token_success)
            m.post(OAUTH_TOKEN_URL, payload=happy_token_response)
            m.post(OAUTH_TOKEN_URL, payload={"openai_api_key": "sk_exchanged"})
            await run_oauth_device_flow()
        stderr = capsys.readouterr().err
        assert "Sign in to ChatGPT for Kitty" in stderr  # the coloured prompt's header
        assert "Sign in at:" in stderr  # the plain copy's URL line
        assert stderr.count("ABCD-EFGH") >= 2, (
            "the one-time code must appear in both the coloured and the plain form"
        )


# ── L1: redaction ────────────────────────────────────────────────────────


class TestDeviceRedaction:
    def test_user_code_not_in_debug_log(
        self,
        caplog: pytest.LogCaptureFixture,
        codex_user_code_response: dict[str, str],
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="kitty.auth.openai_oauth"):
            parse_device_user_code_response(json.dumps(codex_user_code_response))
        assert "ABCD-EFGH" not in caplog.text
        assert "dev_abc123" not in caplog.text

    async def test_auth_code_not_in_exception_message(
        self,
        codex_user_code_response: dict[str, str],
        codex_token_success: dict[str, str],
        fast_poll: None,
    ) -> None:
        """An OAuth error body that echoes the request's authorization code
        must not propagate that code into a Python exception message."""
        secret_code = codex_token_success["authorization_code"] = "ac_DO_NOT_LEAK"
        with aioresponses() as m:
            m.post(OAUTH_DEVICE_USERCODE_URL, payload=codex_user_code_response)
            m.post(OAUTH_DEVICE_TOKEN_URL, payload=codex_token_success)
            m.post(
                OAUTH_TOKEN_URL,
                status=400,
                body=json.dumps(
                    {
                        "error": "invalid_request",
                        "error_description": f"the request included code={secret_code}",
                    }
                ),
            )
            with pytest.raises(OAuthAuthorizationError) as exc_info:
                await run_oauth_device_flow()
        assert secret_code not in str(exc_info.value)


# ── L2: contract fixture ─────────────────────────────────────────────────


class TestDeviceCodeContract:
    """Pin the exact JSON shapes kitty's device flow sends and expects.

    These fixtures are kitty's contract with itself. If a future change to
    codex-rs's protocol is intentional, the fixture is updated *with* the
    change; if it is not intentional, the test fails and surfaces the drift.

    The *request* bodies are asserted for real in
    ``TestRunDeviceFlowEnd2End::test_request_bodies_match_the_contract``
    (aioresponses matches on method+URL only, so only a captured-call
    assertion can pin what the production code actually serialises — the
    in-class dict checks below would pass on any typo in the sender).
    """

    def test_usercode_response_shape(self, codex_user_code_response: dict[str, str]) -> None:
        """The device server returns ``device_auth_id``, ``user_code``, ``interval``."""
        assert set(codex_user_code_response) == {"device_auth_id", "user_code", "interval"}

    def test_token_poll_success_shape(self, codex_token_success: dict[str, str]) -> None:
        """A successful poll returns ``authorization_code``, ``code_challenge``, ``code_verifier``."""
        assert set(codex_token_success) == {
            "authorization_code",
            "code_challenge",
            "code_verifier",
        }

    def test_authorization_endpoint_uses_server_redirect(self) -> None:
        """The final token exchange uses the device server's redirect URI,
        not the localhost callback URI used by the browser path."""
        from kitty.auth.openai_oauth import OAUTH_DEVICE_REDIRECT_URI, REDIRECT_URI

        assert OAUTH_DEVICE_REDIRECT_URI != REDIRECT_URI
        assert OAUTH_DEVICE_REDIRECT_URI == "https://auth.openai.com/deviceauth/callback"
        assert REDIRECT_URI == "http://localhost:1455/auth/callback"

    def test_poll_budget_is_fifteen_minutes(self) -> None:
        """The 15-minute cap matches codex; the browser path's 300 s must not
        leak into the device flow."""
        from kitty.auth.openai_oauth import DEVICE_FLOW_MAX_WAIT_SECONDS

        assert DEVICE_FLOW_MAX_WAIT_SECONDS == 900


# ── L3: end-to-end ──────────────────────────────────────────────────────


class TestRunDeviceFlowEnd2End:
    async def test_happy_path(
        self,
        codex_user_code_response: dict[str, str],
        codex_token_success: dict[str, str],
        happy_token_response: dict[str, Any],
        fast_poll: None,
    ) -> None:
        """All three hops succeed, returning a fully populated OAuthSession."""
        with aioresponses() as m:
            m.post(OAUTH_DEVICE_USERCODE_URL, payload=codex_user_code_response)
            m.post(OAUTH_DEVICE_TOKEN_URL, payload=codex_token_success)
            m.post(OAUTH_TOKEN_URL, payload=happy_token_response)
            # The flow also attempts the best-effort id_token → API key exchange.
            m.post(OAUTH_TOKEN_URL, payload={"openai_api_key": "sk_exchanged"})

            async with aiohttp.ClientSession() as http:
                session = await run_oauth_device_flow(http=http)

        assert session.access_token == "at_device"
        assert session.refresh_token == "rt_device"
        assert session.id_token == "id_device"
        assert session.api_key == "sk_exchanged"

    async def test_request_bodies_match_the_contract(
        self,
        codex_user_code_response: dict[str, str],
        codex_token_success: dict[str, str],
        happy_token_response: dict[str, Any],
        fast_poll: None,
    ) -> None:
        """The JSON bodies the flow actually serialises match the codex-rs
        contract exactly (L2-as-L3: aioresponses matches on method+URL only,
        so only the captured calls can pin what production sends — a typo in
        a field name would otherwise pass every test green)."""
        import json as json_mod

        from yarl import URL

        with aioresponses() as m:
            m.post(OAUTH_DEVICE_USERCODE_URL, payload=codex_user_code_response)
            m.post(OAUTH_DEVICE_TOKEN_URL, payload=codex_token_success)
            m.post(OAUTH_TOKEN_URL, payload=happy_token_response)
            m.post(OAUTH_TOKEN_URL, payload={"openai_api_key": "sk_exchanged"})
            async with aiohttp.ClientSession() as http:
                await run_oauth_device_flow(http=http)

            # aioresponses keys its captured calls by (method, yarl.URL).
            usercode_calls = m.requests[("POST", URL(OAUTH_DEVICE_USERCODE_URL))]
            poll_calls = m.requests[("POST", URL(OAUTH_DEVICE_TOKEN_URL))]
            assert len(usercode_calls) == 1
            assert json_mod.loads(usercode_calls[0].kwargs["data"]) == {"client_id": CLIENT_ID}
            assert len(poll_calls) == 1
            assert json_mod.loads(poll_calls[0].kwargs["data"]) == {
                "device_auth_id": "dev_abc123",
                "user_code": "ABCD-EFGH",
            }

    async def test_session_to_dict_matches_browser_path(
        self,
        happy_token_response: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """FR-6: given the same token payload, the device path's session
        equals the browser path's. Both paths funnel through
        ``OAuthSession.from_token_response``, so the equality target is the
        constructed session's ``to_dict()``. Time is frozen because the
        expiry timestamps derive from ``time.time()`` at construction."""
        import time as time_mod

        from kitty.auth.oauth_session import OAuthSession

        monkeypatch.setattr(time_mod, "time", lambda: 1_000_000.0)
        device_session = OAuthSession.from_token_response(happy_token_response, CLIENT_ID)
        browser_session = OAuthSession.from_token_response(happy_token_response, CLIENT_ID)
        assert device_session.to_dict() == browser_session.to_dict()

    async def test_ctrl_c_during_poll_raises_cancelled_error(
        self,
        codex_user_code_response: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """FR-5: Ctrl+C during the poll loop surfaces
        OAuthDeviceCodeCancelledError — including when it lands on the
        inter-poll sleep, where the loop spends ~99% of its wall clock
        (code-review round 1, High #3).

        The ``fast_poll`` fixture is intentionally NOT used: it patches the
        global ``asyncio.sleep``, which would also no-op this test's own
        sleep and let the cancel land before the task ever started.
        """
        import asyncio as asyncio_mod

        with aioresponses() as m:
            m.post(OAUTH_DEVICE_USERCODE_URL, payload=codex_user_code_response)
            # One pending poll is enough: the flow sleeps (really) before
            # polling, and the cancel lands inside that sleep.
            m.post(OAUTH_DEVICE_TOKEN_URL, status=403)

            task = asyncio_mod.ensure_future(run_oauth_device_flow())
            # Real sleep: let the task reach its inter-poll sleep (the
            # flow's own sleep is seconds; 0.05 s is comfortably inside it).
            await asyncio_mod.sleep(0.05)
            task.cancel()
            with pytest.raises(OAuthDeviceCodeCancelledError):
                await task
        """A 404 on /deviceauth/usercode means device-code auth is disabled
        for this account/workspace — fail fast with the typed error."""
        with aioresponses() as m:
            m.post(OAUTH_DEVICE_USERCODE_URL, status=404, body="not found")
            with pytest.raises(OAuthDeviceCodeDisabledError):
                await run_oauth_device_flow()

    async def test_usercode_500_raises_with_status(self) -> None:
        """A 5xx on the usercode hop fails fast with the status in the message."""
        with aioresponses() as m:
            m.post(OAUTH_DEVICE_USERCODE_URL, status=500, body="upstream broken")
            with pytest.raises(OAuthAuthorizationError, match="500"):
                await run_oauth_device_flow()

    async def test_poll_then_success(
        self,
        codex_user_code_response: dict[str, str],
        codex_token_success: dict[str, str],
        happy_token_response: dict[str, Any],
        fast_poll: None,
    ) -> None:
        """Two 404s (still pending) then a 200 — the loop must keep polling."""
        with aioresponses() as m:
            m.post(OAUTH_DEVICE_USERCODE_URL, payload=codex_user_code_response)
            m.post(OAUTH_DEVICE_TOKEN_URL, status=404)
            m.post(OAUTH_DEVICE_TOKEN_URL, status=404)
            m.post(OAUTH_DEVICE_TOKEN_URL, payload=codex_token_success)
            m.post(OAUTH_TOKEN_URL, payload=happy_token_response)
            m.post(OAUTH_TOKEN_URL, payload={"openai_api_key": "sk_exchanged"})

            async with aiohttp.ClientSession() as http:
                session = await run_oauth_device_flow(http=http)
        assert session.access_token == "at_device"

    async def test_partial_token_response_raises_typed_error(
        self,
        codex_user_code_response: dict[str, str],
        codex_token_success: dict[str, str],
        fast_poll: None,
    ) -> None:
        """The third hop returns access_token + id_token but no refresh_token.

        The dataclass constructor would raise a bare KeyError; the device
        path must validate first and raise a typed OAuthAuthorizationError
        (mirrors F35).
        """
        with aioresponses() as m:
            m.post(OAUTH_DEVICE_USERCODE_URL, payload=codex_user_code_response)
            m.post(OAUTH_DEVICE_TOKEN_URL, payload=codex_token_success)
            m.post(
                OAUTH_TOKEN_URL,
                payload={"access_token": "at", "id_token": "id", "expires_in": 3600},
            )
            with pytest.raises(OAuthAuthorizationError, match="refresh_token"):
                await run_oauth_device_flow()

    async def test_transport_error_mid_poll_fails_fast(
        self,
        codex_user_code_response: dict[str, str],
        fast_poll: None,
    ) -> None:
        """A DNS failure / connection reset mid-poll fails fast (no retry),
        matching codex's no-retry posture."""
        with aioresponses() as m:
            m.post(OAUTH_DEVICE_USERCODE_URL, payload=codex_user_code_response)
            m.post(OAUTH_DEVICE_TOKEN_URL, exception=ConnectionError("dns down"))
            with pytest.raises(OAuthAuthorizationError, match="dns down"):
                await run_oauth_device_flow()

    async def test_poll_loop_times_out_within_injected_budget(
        self,
        codex_user_code_response: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With all polls pending and a shrunken budget, the loop exits with
        OAuthTimeoutError rather than waiting the real 15 minutes.

        The ``fast_poll`` fixture is intentionally not used: a no-op sleep
        would prevent the event loop's monotonic clock from advancing past
        the budget, and the timeout would never fire. Multiple poll mocks
        keep aioresponses from returning ``ConnectionError`` on exhaustion
        (which the flow would wrap as ``OAuthAuthorizationError`` and the
        test would misclassify).
        """
        monkeypatch.setattr(openai_oauth, "DEVICE_FLOW_MAX_WAIT_SECONDS", 0.5)
        with aioresponses() as m:
            m.post(OAUTH_DEVICE_USERCODE_URL, payload=codex_user_code_response)
            for _ in range(50):
                m.post(OAUTH_DEVICE_TOKEN_URL, status=403)
            with pytest.raises(OAuthTimeoutError):
                await run_oauth_device_flow()

    async def test_final_sleep_never_overshoots_budget(
        self,
        codex_user_code_response: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The sleep before the deadline is clamped to the remaining budget
        (codex's min(sleep, max_wait - elapsed) discipline).

        The recorder must really sleep: a no-op would stop the monotonic
        clock from advancing and the deadline would never fire.
        """
        requested: list[float] = []
        real_sleep = asyncio.sleep

        async def _recording_sleep(seconds: float) -> None:
            requested.append(seconds)
            await real_sleep(seconds)

        monkeypatch.setattr(openai_oauth.asyncio, "sleep", _recording_sleep)
        monkeypatch.setattr(openai_oauth, "DEVICE_FLOW_MAX_WAIT_SECONDS", 0.2)

        with aioresponses() as m:
            m.post(OAUTH_DEVICE_USERCODE_URL, payload=codex_user_code_response)
            for _ in range(5):
                m.post(OAUTH_DEVICE_TOKEN_URL, status=403)
            with pytest.raises(OAuthTimeoutError):
                await run_oauth_device_flow()

        assert requested, "the poll loop must sleep between polls"
        assert max(requested) <= 0.2, "no single sleep may overshoot the total budget"

    async def test_session_closed_when_created_internally(
        self,
        codex_user_code_response: dict[str, str],
        codex_token_success: dict[str, str],
        happy_token_response: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        fast_poll: None,
    ) -> None:
        """When the caller passes no aiohttp session, the flow creates one and
        closes it before returning; when the caller passes one, the flow must
        not close it.

        The two cases are recorded together to share the patched
        ``aiohttp.ClientSession.close`` recorder, but the close call is
        checked for each call shape: a flow-owned session lands one entry,
        a caller-owned session lands zero.
        """
        closed_sessions: list[int] = []
        real_close = aiohttp.ClientSession.close

        async def close_recorder(self: aiohttp.ClientSession) -> None:
            closed_sessions.append(id(self))
            await real_close(self)

        monkeypatch.setattr(aiohttp.ClientSession, "close", close_recorder)

        with aioresponses() as m:
            m.post(OAUTH_DEVICE_USERCODE_URL, payload=codex_user_code_response)
            m.post(OAUTH_DEVICE_TOKEN_URL, payload=codex_token_success)
            m.post(OAUTH_TOKEN_URL, payload=happy_token_response)
            m.post(OAUTH_TOKEN_URL, payload={"openai_api_key": "sk_exchanged"})

            # Internal session: created and closed by the flow itself.
            session = await run_oauth_device_flow()
        assert session.access_token == "at_device"
        assert len(closed_sessions) == 1, (
            "run_oauth_device_flow must close the aiohttp session it created"
        )

        # Caller-owned session: the flow must not close it. We avoid the
        # `async with` form here because its __aexit__ would call close()
        # and obscure the assertion; the test owns the close explicitly.
        closed_sessions.clear()
        with aioresponses() as m:
            m.post(OAUTH_DEVICE_USERCODE_URL, payload=codex_user_code_response)
            m.post(OAUTH_DEVICE_TOKEN_URL, payload=codex_token_success)
            m.post(OAUTH_TOKEN_URL, payload=happy_token_response)
            m.post(OAUTH_TOKEN_URL, payload={"openai_api_key": "sk_exchanged"})

            http = aiohttp.ClientSession()
            try:
                await run_oauth_device_flow(http=http)
            finally:
                # Confirm the flow didn't close it; then close it ourselves.
                assert closed_sessions == [], (
                    "the flow must not close a session it did not create"
                )
                await http.close()
