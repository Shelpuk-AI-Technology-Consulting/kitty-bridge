"""KBR-328: Claude Code's startup quota probe, and bridge defects, park no healthy member.

Two separate defects from the same log lines (2026-11-04, pool ``my``):

1. **The crash.** A MiniMax Token Plan member answered Claude Code's one-token
   ``quota`` request with a Messages body whose ``content`` was ``null``. The
   translated path iterated it (``AnthropicAdapter.translate_from_upstream``)
   and raised ``'NoneType' object is not iterable``; the walk's generic
   ``except Exception`` arm then parked the member as ``hard`` for 300 s -- a
   bridge bug recorded as evidence against a healthy backend. MiniMax also
   reports some failures as HTTP 200 with a non-zero ``base_resp.status_code``,
   which the bridge translated as if it were a success.

2. **The walk.** The quota probe walked every member, and each one that
   answered 429 was parked, so one request could leave a pool all-unhealthy
   before the user's first turn.

Design of record: ``SYSTEM_DESIGN.md`` §19.

**Layer.** Collects as ``l1`` alongside every other loopback test in this
directory.
"""

from __future__ import annotations

import copy
import json
import logging
import uuid

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge import server as server_module
from kitty.bridge.server import BridgeServer, _is_quota_probe
from kitty.profiles.schema import Profile
from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.anthropic_subscription import AnthropicSubscriptionAdapter
from kitty.providers.base import ProviderAdapter, ProviderError
from kitty.providers.minimax import MiniMaxAdapter, minimax_embedded_error_status
from kitty.providers.minimax_token import MiniMaxTokenAnthropicAdapter

#: The Token Plan's global Anthropic-compatible endpoint (translated path).
_MINIMAX_URL = "https://api.minimax.io/anthropic/v1/messages"

#: The subscription relay's upstream.
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

#: Claude Code's startup quota probe, as the bridge receives it.
_QUOTA_PROBE = {
    "model": "MiniMax-M2",
    "max_tokens": 1,
    "messages": [{"role": "user", "content": "quota"}],
    "metadata": {"user_id": "session"},
}

#: An ordinary turn of the same size class.
_REAL_TURN = {
    "model": "MiniMax-M2",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Hello."}],
}

#: The MiniMax reply that crashed the translated path: a Messages body with null content.
_NULL_CONTENT_REPLY = {
    "id": "msg_null",
    "type": "message",
    "role": "assistant",
    "model": "MiniMax-M2",
    "content": None,
    "stop_reason": "max_tokens",
    "usage": None,
}

#: MiniMax's HTTP-200 rate-limit envelope.
_BASE_RESP_RATE_LIMIT = {
    "id": "",
    "type": "message",
    "content": None,
    "base_resp": {"status_code": 1002, "status_msg": "rate limit exceeded(RPM)"},
}

_OK_REPLY = {
    "id": "msg_ok",
    "type": "message",
    "role": "assistant",
    "model": "MiniMax-M2",
    "content": [{"type": "text", "text": "Answered."}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}

_RATE_LIMITED = {"type": "error", "error": {"type": "rate_limit_error", "message": "Error"}}

# Bounded: the relay loop advances only by parking, which the probe suppresses,
# so losing the probe's early return turns a test into a hang. Fail it instead.
pytestmark = pytest.mark.timeout(60)


@pytest.fixture
def fast_ladders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the retry ladders' real-time sleeps.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.01)
    monkeypatch.setattr(server_module, "_EMPTY_RETRY_DELAYS", [0.01, 0.01])
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])


def _minimax_member(name: str, *, backup: bool = False, native: bool = False) -> tuple:
    """Build a MiniMax Token Plan member keyed by its own name.

    Args:
        name: Profile name; also the member's API key, so calls can be attributed.
        backup: Whether the member is in the reserve tier.
        native: Whether the member uses the native Messages passthrough
            instead of the default translated path.

    Returns:
        A ``(provider, key, profile)`` balancing tuple.
    """
    return (
        MiniMaxTokenAnthropicAdapter(native_messages=native),
        f"key-{name}",
        Profile(
            name=name,
            provider="minimax_token",
            model="MiniMax-M2",
            auth_ref=str(uuid.uuid4()),
            backup=backup,
        ),
    )


class _BrokenStreamProvider(ProviderAdapter):
    """Custom-transport provider whose stream fails inside kitty's own code (a ``TypeError``)."""

    use_custom_transport = True

    @property
    def provider_type(self) -> str:
        """Name the provider in health attribution.

        Returns:
            The provider type string.
        """
        return "stub_broken_stream"

    @property
    def default_base_url(self) -> str:
        """Give the never-contacted base URL.

        Returns:
            A placeholder URL; the provider fails before any I/O.
        """
        return "https://broken-stream.example.test"

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        """Reflect the request through untouched.

        Args:
            model: The model name.
            messages: The conversation.
            **kwargs: Remaining request fields.

        Returns:
            The request fields as given.
        """
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        """Return the payload unchanged.

        Args:
            response_data: The upstream payload.

        Returns:
            The same payload.
        """
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        """Wrap a status into a provider error.

        Args:
            status_code: The upstream status.
            body: The upstream body.

        Returns:
            A provider error carrying both.
        """
        return ProviderError(f"{status_code}: {body}")

    async def stream_request(self, cc_request: dict, write) -> None:
        """Fail the stream the way a bridge bug would.

        Args:
            cc_request: The upstream request.
            write: The chunk sink; never called.

        Raises:
            TypeError: Always -- the defect class KBR-328 must not park on.
        """
        raise TypeError("'NoneType' object is not iterable")


def _broken_stream_member(name: str) -> tuple:
    """Build a custom-transport streaming member that always raises a bridge defect.

    Args:
        name: Profile name.

    Returns:
        A ``(provider, key, profile)`` balancing tuple.
    """
    return (
        _BrokenStreamProvider(),
        "",
        Profile(name=name, provider="openai_subscription", model="m", auth_ref=str(uuid.uuid4())),
    )


def _subscription_member(name: str, *, backup: bool = False) -> tuple:
    """Build a tokenless subscription member.

    Args:
        name: Profile name.
        backup: Whether the member is in the reserve tier.

    Returns:
        A ``(provider, key, profile)`` balancing tuple.
    """
    return (
        AnthropicSubscriptionAdapter(),
        "",
        Profile(
            name=name,
            provider="anthropic_subscription",
            model="claude-sonnet-4-6",
            auth_ref=str(uuid.uuid4()),
            backup=backup,
        ),
    )


def _pool(*backends: tuple, relay: bool = False) -> BridgeServer:
    """Build an unstarted balancing bridge over ``backends``.

    Args:
        backends: ``(provider, key, profile)`` tuples.
        relay: Whether the subscription relay is opted in.

    Returns:
        The bridge.
    """
    return BridgeServer(
        adapter=None,
        provider=backends[0][0],
        resolved_key=backends[0][1],
        backends=list(backends),
        allow_subscription_relay=relay,
    )


async def _send(server: BridgeServer, request: dict, routes: dict[str, object]) -> tuple[int, dict, list]:
    """Send one client request through the bridge against scripted upstreams.

    Args:
        server: An unstarted bridge.
        request: The inbound Messages body.
        routes: Maps an upstream URL to a responder ``(body, headers) -> (status, reply)``,
            or to an exception instance the upstream raises on every call.

    Returns:
        ``(client_status, client_body, upstream_calls)``; each call is ``(url, headers)``.
    """
    calls: list[tuple[str, dict]] = []

    def responder(url: str, answer):
        def respond(_url, **kwargs):
            headers = dict(kwargs.get("headers") or {})
            calls.append((url, headers))
            if isinstance(answer, BaseException):
                raise answer
            status, reply = answer(copy.deepcopy(kwargs.get("json")), headers)
            return CallbackResult(status=status, content_type="application/json", body=json.dumps(reply))

        return respond

    port = await server.start_async()
    try:
        with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
            for url, answer in routes.items():
                mocked.post(url, callback=responder(url, answer), repeat=True)
            async with aiohttp.ClientSession() as session, session.post(
                f"http://127.0.0.1:{port}/v1/messages",
                json=request,
                headers={"Authorization": "Bearer cc-token"},
            ) as resp:
                status = resp.status
                text = await resp.text()
    finally:
        await server.stop_async()
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        body = {"raw": text}
    return status, body, calls


def _parked(server: BridgeServer) -> dict[str, str | None]:
    """Return each parked member's name and failure kind.

    Args:
        server: The bridge after the request.

    Returns:
        ``{member_name: last_failure_kind}`` for every unhealthy member.
    """
    return {
        server._backends[i][2].name: health.get("last_failure_kind")
        for i, health in enumerate(server._backend_health)
        if not health["healthy"]
    }


def _always(status: int, reply: object):
    """Build a responder that always answers the same.

    Args:
        status: HTTP status.
        reply: JSON reply.

    Returns:
        The responder.
    """
    return lambda _body, _headers: (status, reply)


# ── Unit: the predicates ──────────────────────────────────────────────────


class TestQuotaProbeShape:
    """Only the exact probe shape is a probe; every real turn is not."""

    @pytest.mark.parametrize(
        "content",
        ["quota", [{"type": "text", "text": "quota"}], " quota\n"],
        ids=["string", "text-block", "whitespace"],
    )
    def test_probe_shapes(self, content: object) -> None:
        body = {**_QUOTA_PROBE, "messages": [{"role": "user", "content": content}]}
        assert _is_quota_probe(body)

    @pytest.mark.parametrize(
        "change",
        [
            {"max_tokens": 2},
            {"stream": True},
            {"messages": [{"role": "user", "content": "quota please"}]},
            {"messages": [{"role": "assistant", "content": "quota"}]},
            {"messages": [{"role": "user", "content": "quota"}, {"role": "user", "content": "quota"}]},
            {"messages": [{"role": "user", "content": [{"type": "image", "text": "quota"}]}]},
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "quota"}] * 2}]},
            {"messages": "quota"},
        ],
        ids=["max-tokens", "streaming", "other-text", "assistant", "two-messages", "image", "two-blocks", "bad-list"],
    )
    def test_near_misses_are_not_probes(self, change: dict) -> None:
        assert not _is_quota_probe({**_QUOTA_PROBE, **change})

    def test_non_dict_is_not_a_probe(self) -> None:
        assert not _is_quota_probe(["quota"])


class TestNullContentTranslation:
    """The failing line: a null ``content`` (or ``usage``) is an empty reply, not a crash."""

    def test_translate_from_upstream_tolerates_null_content_and_usage(self) -> None:
        result = AnthropicAdapter().translate_from_upstream(copy.deepcopy(_NULL_CONTENT_REPLY))
        assert result["choices"][0]["message"]["content"] is None
        assert result["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_minimax_token_inherits_the_tolerance(self) -> None:
        result = MiniMaxTokenAnthropicAdapter().translate_from_upstream(copy.deepcopy(_NULL_CONTENT_REPLY))
        assert BridgeServer._is_empty_cc_response(result)

    def test_null_text_and_non_dict_blocks_are_skipped(self) -> None:
        reply = {**_OK_REPLY, "content": [None, {"type": "text", "text": None}, {"type": "text", "text": "ok"}]}
        result = AnthropicAdapter().translate_from_upstream(reply)
        assert result["choices"][0]["message"]["content"] == "\nok"

    def test_native_empty_judge_tolerates_null_content(self) -> None:
        assert BridgeServer._is_empty_cc_response({"type": "message", "content": None})

    def test_bare_string_content_is_kept_as_text(self) -> None:
        """A non-conforming string ``content`` is the model's answer, not characters to skip."""
        result = AnthropicAdapter().translate_from_upstream({**_OK_REPLY, "content": "Plain answer."})
        assert result["choices"][0]["message"]["content"] == "Plain answer."
        assert not BridgeServer._is_empty_cc_response({"type": "message", "content": "Plain answer."})
        assert BridgeServer._is_empty_cc_response({"type": "message", "content": ""})

    def test_object_content_is_logged_and_treated_as_empty(self, caplog: pytest.LogCaptureFixture) -> None:
        """Any other non-list shape is a protocol violation: empty, but never silently."""
        with caplog.at_level(logging.WARNING, logger="kitty.providers.anthropic"):
            result = AnthropicAdapter().translate_from_upstream({**_OK_REPLY, "content": {"text": "x"}})
        assert result["choices"][0]["message"]["content"] is None
        assert any("not a list" in r.getMessage() for r in caplog.records)
        assert BridgeServer._is_empty_cc_response({"type": "message", "content": {"text": "x"}})


class TestMiniMaxEmbeddedError:
    """MiniMax's HTTP-200 ``base_resp`` errors map to the status they stand for."""

    @pytest.mark.parametrize(
        ("code", "status"),
        [(1002, 429), (1039, 429), (1004, 401), (1008, 402), (1027, 400), (2013, 400), (1013, 502), (9999, 502)],
    )
    def test_codes(self, code: int, status: int) -> None:
        assert minimax_embedded_error_status({"base_resp": {"status_code": code, "status_msg": "x"}}) == status

    @pytest.mark.parametrize(
        "body",
        [
            _OK_REPLY,
            {"base_resp": {"status_code": 0, "status_msg": "success"}},
            {"base_resp": {"status_code": "1002"}},
            {"base_resp": {"status_code": True}},
            {"base_resp": None},
            "text",
            None,
        ],
        ids=["no-base-resp", "success", "string-code", "bool-code", "null", "string-body", "none"],
    )
    def test_not_errors(self, body: object) -> None:
        assert minimax_embedded_error_status(body) is None

    def test_both_minimax_adapters_report_it(self) -> None:
        for adapter in (MiniMaxAdapter(), MiniMaxTokenAnthropicAdapter()):
            assert adapter.embedded_error_status(_BASE_RESP_RATE_LIMIT) == 429

    def test_other_providers_never_report_one(self) -> None:
        assert AnthropicAdapter().embedded_error_status(_BASE_RESP_RATE_LIMIT) is None


class TestDefectClassification:
    """A programming error raised inside the bridge is a ``defect``, never ``hard``."""

    @pytest.mark.parametrize(
        "exc", [TypeError("'NoneType' object is not iterable"), AttributeError("x"), KeyError("k")]
    )
    def test_bridge_bugs_are_defects(self, exc: Exception) -> None:
        assert BridgeServer._provider_error_failure_kind(exc) == "defect"

    def test_provider_errors_are_never_defects(self) -> None:
        class _Hybrid(ProviderError, TypeError):
            pass

        assert BridgeServer._provider_error_failure_kind(_Hybrid("upstream")) == "hard"
        assert BridgeServer._provider_error_failure_kind(ProviderError("upstream")) == "hard"

    def test_other_exceptions_stay_hard(self) -> None:
        assert BridgeServer._provider_error_failure_kind(RuntimeError("x")) == "hard"


# ── Bridge: done-when 1 and 2 ─────────────────────────────────────────────


class TestTheCrashParksNobody:
    """The reported reply no longer crashes, and a crash would park nobody anyway."""

    @pytest.mark.asyncio
    async def test_null_content_reply_parks_no_member(self, fast_ladders: None) -> None:
        """Done-when 1: the MiniMax-shaped reply is judged empty -- the empty
        ladder runs, nothing is parked, nothing crashes."""
        server = _pool(_minimax_member("mm-1"), _minimax_member("mm-2"))
        status, body, calls = await _send(server, _REAL_TURN, {_MINIMAX_URL: _always(200, _NULL_CONTENT_REPLY)})

        assert status == 502, body
        assert body["error"].get("reason") == "empty_response"
        assert calls, "the upstream must have been asked"
        assert _parked(server) == {}

    @pytest.mark.asyncio
    async def test_native_object_content_is_logged_not_silently_emptied(
        self, fast_ladders: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A native Messages reply skips the translator's warning, so the bridge logs its own."""
        reply = {**_OK_REPLY, "content": {"text": "Answered."}}
        server = _pool(_minimax_member("mm-1", native=True), _minimax_member("mm-2", native=True))
        with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
            status, _body, _calls = await _send(server, _REAL_TURN, {_MINIMAX_URL: _always(200, reply)})

        assert status == 502
        assert any("not a list" in r.getMessage() for r in caplog.records)
        assert _parked(server) == {}

    @pytest.mark.asyncio
    async def test_a_bridge_defect_parks_no_member(
        self, fast_ladders: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Done-when 2: any bug of this class -- not only the one fixed -- leaves
        the pool intact, logs the traceback, and skips the final ladder."""

        def broken(self, raw_response: dict) -> dict:
            raise TypeError("'NoneType' object is not iterable")

        monkeypatch.setattr(MiniMaxTokenAnthropicAdapter, "translate_from_upstream", broken)
        server = _pool(_minimax_member("mm-1"), _minimax_member("mm-2"))
        with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
            status, _body, calls = await _send(server, _REAL_TURN, {_MINIMAX_URL: _always(200, _OK_REPLY)})

        assert status == 500
        assert _parked(server) == {}
        # One attempt per member, and no 20 s + 40 s ladder re-running the same bug.
        assert len(calls) == 2
        defect_records = [r for r in caplog.records if "bridge defect" in r.getMessage()]
        assert defect_records, "the defect must be logged"
        assert all(r.exc_info is not None for r in defect_records), "with its traceback"

    @pytest.mark.asyncio
    async def test_a_streaming_custom_transport_defect_parks_no_member(self, fast_ladders: None) -> None:
        """The defect rule holds at a streaming custom-transport arm, not only the non-streaming walk.

        That arm classifies with its own ``_provider_error_failure_kind`` call; reverting it to a
        literal ``hard`` would park both members again.
        """
        server = _pool(_broken_stream_member("bs-1"), _broken_stream_member("bs-2"))
        await _send(server, {**_REAL_TURN, "stream": True}, {})

        assert _parked(server) == {}

    @pytest.mark.asyncio
    async def test_embedded_rate_limit_parks_as_rate_limit_and_fails_over(self, fast_ladders: None) -> None:
        """A ``base_resp`` 1002 takes the 429 path: parked as ``rate_limit``, the sibling serves.

        mm-2 is a backup member, so healthy mm-1 is the only primary and the
        arrival is pinned: the walk reaches mm-2 only through the 429's
        failover, which an unmapped null-content empty reply cannot produce.
        """
        replies = {"key-mm-1": (200, _BASE_RESP_RATE_LIMIT), "key-mm-2": (200, _OK_REPLY)}
        server = _pool(_minimax_member("mm-1"), _minimax_member("mm-2", backup=True))
        status, body, _calls = await _send(
            server, _REAL_TURN, {_MINIMAX_URL: lambda _b, headers: replies[headers.get("x-api-key")]}
        )

        assert status == 200, body
        assert body["content"][0]["text"] == "Answered."
        assert _parked(server) == {"mm-1": "rate_limit"}


# ── Bridge: done-when 4 ───────────────────────────────────────────────────


class TestQuotaProbeCannotEmptyThePool:
    """The probe gets one member, no walk, and parks only on deterministic evidence."""

    @pytest.mark.asyncio
    async def test_rate_limited_probe_tries_one_member_and_parks_none(self, fast_ladders: None) -> None:
        server = _pool(_minimax_member("mm-1"), _minimax_member("mm-2"), _minimax_member("mm-3"))
        status, _body, calls = await _send(server, _QUOTA_PROBE, {_MINIMAX_URL: _always(429, _RATE_LIMITED)})

        assert status == 429
        assert len(calls) == 1, "the probe must not walk the pool"
        assert _parked(server) == {}

    @pytest.mark.asyncio
    async def test_the_same_failure_on_a_real_turn_still_walks_and_parks(self, fast_ladders: None) -> None:
        """Regression guard: the probe rule is scoped to the probe."""
        server = _pool(_minimax_member("mm-1"), _minimax_member("mm-2"), _minimax_member("mm-3"))
        status, _body, calls = await _send(server, _REAL_TURN, {_MINIMAX_URL: _always(429, _RATE_LIMITED)})

        assert status == 429
        assert len({headers.get("x-api-key") for _url, headers in calls}) == 3
        assert set(_parked(server).values()) == {"rate_limit"}

    @pytest.mark.asyncio
    async def test_probe_still_parks_a_rejected_credential(self, fast_ladders: None) -> None:
        """``auth`` is deterministic: the first real turn would meet the same rejection."""
        server = _pool(_minimax_member("mm-1"), _minimax_member("mm-2"))
        status, _body, calls = await _send(
            server, _QUOTA_PROBE, {_MINIMAX_URL: _always(401, {"error": {"message": "invalid api key"}})}
        )

        assert status == 401
        assert len(calls) == 1
        assert list(_parked(server).values()) == ["auth"]

    @pytest.mark.asyncio
    async def test_probe_still_parks_an_entitlement_rejection(self, fast_ladders: None) -> None:
        """``entitlement`` is the other deterministic kind: a 403 plan rejection still parks on a probe."""
        rejection = {
            "type": "error",
            "error": {"type": "permission_error", "message": "This model requires a subscription"},
        }
        server = _pool(_subscription_member("sub"), _minimax_member("mm-1", backup=True), relay=True)
        status, _body, calls = await _send(
            server,
            {**_QUOTA_PROBE, "model": "claude-sonnet-4-6"},
            {_ANTHROPIC_URL: _always(403, rejection), _MINIMAX_URL: _always(200, _OK_REPLY)},
        )

        assert status == 403
        assert [url for url, _headers in calls] == [_ANTHROPIC_URL]
        assert _parked(server) == {"sub": "entitlement"}

    @pytest.mark.asyncio
    async def test_probe_on_the_subscription_member_neither_walks_nor_falls_over(self, fast_ladders: None) -> None:
        """The relay answers the probe with Anthropic's own 429: no sibling, no
        translated fall-over, no park -- the API-key member is never called."""
        server = _pool(_subscription_member("sub"), _minimax_member("mm-1", backup=True), relay=True)
        status, body, calls = await _send(
            server,
            {**_QUOTA_PROBE, "model": "claude-sonnet-4-6"},
            {_ANTHROPIC_URL: _always(429, _RATE_LIMITED), _MINIMAX_URL: _always(200, _OK_REPLY)},
        )

        assert status == 429, body
        assert [url for url, _headers in calls] == [_ANTHROPIC_URL]
        assert _parked(server) == {}

    @pytest.mark.asyncio
    async def test_probe_relay_transport_failure_surfaces_once_and_parks_none(
        self, fast_ladders: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A spent transport blip on the probe re-raises: no re-selection, no park, no hang.

        The relay loop advances only by parking, which the probe suppresses, so
        without the probe's ``raise`` this would re-select the same healthy
        member forever (the module's timeout turns that into a failure).
        """
        monkeypatch.setattr(server_module, "_TRANSPORT_GRACE_DELAYS", ())
        server = _pool(_subscription_member("sub"), _minimax_member("mm-1", backup=True), relay=True)
        status, _body, calls = await _send(
            server,
            {**_QUOTA_PROBE, "model": "claude-sonnet-4-6"},
            {
                _ANTHROPIC_URL: aiohttp.ClientConnectionError("connection reset"),
                _MINIMAX_URL: _always(200, _OK_REPLY),
            },
        )

        assert status >= 500
        assert [url for url, _headers in calls] == [_ANTHROPIC_URL]
        assert _parked(server) == {}

    @pytest.mark.asyncio
    async def test_real_turn_on_the_subscription_member_still_falls_over(self, fast_ladders: None) -> None:
        """Regression guard for the relay: a real turn keeps its §16 fall-over."""
        server = _pool(_subscription_member("sub"), _minimax_member("mm-1", backup=True), relay=True)
        status, body, calls = await _send(
            server,
            {**_REAL_TURN, "model": "claude-sonnet-4-6"},
            {_ANTHROPIC_URL: _always(429, _RATE_LIMITED), _MINIMAX_URL: _always(200, _OK_REPLY)},
        )

        assert status == 200, body
        assert [url for url, _headers in calls] == [_ANTHROPIC_URL, _MINIMAX_URL]
        assert _parked(server) == {"sub": "rate_limit"}

    @pytest.mark.asyncio
    async def test_probe_answered_empty_does_not_cross_into_the_relay(self, fast_ladders: None) -> None:
        """The empty terminal's last-resort crossing is a second member: not for the probe."""
        # The member in reserve keeps the arrival on mm-1; the crossing would still reach it.
        server = _pool(_minimax_member("mm-1"), _subscription_member("sub", backup=True), relay=True)
        status, _body, calls = await _send(
            server,
            _QUOTA_PROBE,
            {_MINIMAX_URL: _always(200, _NULL_CONTENT_REPLY), _ANTHROPIC_URL: _always(200, _OK_REPLY)},
        )

        assert status == 502
        assert [url for url, _headers in calls] == [_MINIMAX_URL]
        assert _parked(server) == {}
