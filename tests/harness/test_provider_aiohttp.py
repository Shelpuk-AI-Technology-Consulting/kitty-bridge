"""The provider-session recorder and its transport, against real sockets.

`.system_design/TEST_SUITE.md` §5.5, §7.2, §7.5 · plan task **T-B1** (KBR-40) ·
`.requirements/20260912T114422Z_provider_aiohttp_recorder/REQUIREMENTS.md`.

Two products are under observation here and they are driven differently, because
they are reached differently. ``ollama_cloud`` is driven through a real
``BridgeServer`` on T-W8's fixture; the OpenAI login OAuth leg is driven by
calling the product's own token-exchange coroutines, because that leg has no
adapter, never passes through the bridge, and its interactive half waits five
minutes for a browser.

Every conformance probe writes its request bytes itself, for the reason
``test_recorder.py`` records: a client library reorders, re-cases, adds and drops
headers, so a probe sent through one could not tell a recorder that *loses*
casing from a client that never *sent* mixed casing.

**Layer.** These bind real sockets and read as `l3`, but they carry the `l1` path
default deliberately: §8.2 states *"a test may not be moved to `l3` before the
Subsystem job exists"*, and no job selects `l3` today, so an `l3` marker would
remove them from every gate.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path

import aiohttp
import pytest
from exemptions import ratchet

import harness.provider_recorder as provider_recorder_module
from harness.bridge import (
    MODEL,
    Binding,
    BridgeFixture,
    InboundProtocol,
    UpstreamTransport,
    assert_transport_reaches_its_recorder,
    inbound_path,
    marker,
    minimal_inbound_body,
    registered_transports,
    transport,
)
from harness.connect_proxy import unattributable_peer_ports
from harness.contract import CapturedRequest, WireFormat
from harness.provider_aiohttp import ProviderAiohttpTransport, oauth_token_endpoint
from harness.provider_recorder import (
    OAUTH_TOKEN_SUFFIX,
    OLLAMA_CHAT_SUFFIX,
    ProviderRecordingUpstream,
    format_for_provider_path,
    is_oauth_token_path,
    oauth_token_body,
    ollama_success_body,
    ollama_success_stream,
)
from harness.recorder import Reply, UnmatchedPathError, minimal_success_body, minimal_success_stream
from harness.recorder_conformance import (
    PER_EXCHANGE_CHECKS,
    PER_SESSION_CHECKS,
    RICH_PROBE,
    correlate,
    probe,
    recording_of,
    send,
)
from harness.test_contract import _KITTY_IMPORT
from kitty.auth import openai_oauth
from kitty.auth.oauth_session import OAuthSession
from kitty.providers.ollama_cloud import OllamaCloudAdapter

#: The format this transport serves, named once so a case differs from its
#: neighbours only in the thing it is about.
FORMAT = WireFormat.OLLAMA_CHAT

#: The inbound route that exercises it. ``OLLAMA_CHAT`` has no inbound route of
#: its own — the bridge reaches it upstream and never serves it — so this is
#: named, not derived (§7.5.1).
ROUTE = InboundProtocol.CHAT_COMPLETIONS

#: T-W4's rich probe, re-aimed at this recorder's endpoint. Derived rather than
#: copied: what makes that constant valuable is its mixed header casing, its
#: duplicated header, its obs-text value and its two differently-cased percent
#: escapes, and a hand-written near-copy is a second source of truth that stays
#: green when T-W4 adds a hostile case to the original. Only the dispatch suffix
#: differs, and §7.2's own note on ``RICH_PROBE`` says why that matters: a probe
#: matching no suffix would silently exercise the wrong-format fallback in every
#: test that used it.
RICH_OLLAMA_PROBE = RICH_PROBE.replace(b"/chat/completions", OLLAMA_CHAT_SUFFIX.encode(), 1)


@pytest.fixture
async def recorder():
    """Start a provider recorder on an ephemeral loopback port, and stop it after.

    One instance per test, deliberately: a shared recorder mixes captures across
    tests, and the ordering claims are only meaningful per-instance.

    Yields:
        The running :class:`~harness.provider_recorder.ProviderRecordingUpstream`.
    """
    upstream = ProviderRecordingUpstream(default_format=FORMAT)
    await upstream.start()
    try:
        yield upstream
    finally:
        await upstream.stop()
        # The loud fallback, enforced structurally rather than by each test
        # remembering to ask: a wrong-format reply is not a loud failure on its
        # own, it is an apparently empty response and the retry ladder.
        upstream.assert_all_paths_matched()


def test_the_rich_probe_really_targets_this_recorder() -> None:
    """The derived probe must differ from the original and select this format.

    A ``replace`` that matched nothing would leave the probe aimed at the
    primary recorder's suffix, and every per-exchange check below would then be
    run over a request that took the fallback — the self-check §6.2 requires of
    anything that derives its own subject.
    """
    assert RICH_OLLAMA_PROBE != RICH_PROBE, "the suffix rewrite matched nothing"

    target = RICH_OLLAMA_PROBE.split(b" ")[1].split(b"?")[0].decode()
    assert format_for_provider_path(target) is FORMAT


class TestTheRecorderPassesEveryConformanceCheck:
    """R3 — this recorder satisfies the contract T-W4 judges all four against."""

    @pytest.mark.parametrize("check", PER_EXCHANGE_CHECKS, ids=lambda c: c.__name__)
    async def test_per_exchange_check(self, recorder: ProviderRecordingUpstream, check) -> None:
        """Assert one capture matches the request that produced it.

        Args:
            recorder: The running recorder.
            check: One of the ten per-exchange checks, ``check_peer_port``
                among them — the field T-E5's tunnel join reads.
        """
        sent = await send(recorder.host, recorder.port, RICH_OLLAMA_PROBE, marker="rich")
        captured = correlate(recording_of(recorder), sent)
        check(captured, sent, scheme=recorder.scheme)

    @pytest.mark.parametrize("check", PER_SESSION_CHECKS, ids=lambda c: c.__name__)
    async def test_per_session_check(self, recorder: ProviderRecordingUpstream, check) -> None:
        """Assert the recording as a whole is faithful.

        Args:
            recorder: The running recorder.
            check: One of the four per-session checks.
        """
        sent = [
            await send(recorder.host, recorder.port, probe(name, path=OLLAMA_CHAT_SUFFIX), marker=name)
            for name in ("first", "second")
        ]
        # KBR-188: exempt the WINDOWS CELL only -- this assertion gates normally
        # on the Linux and macOS legs, and fails the job the day Windows starts
        # passing. §8.3's parametrised-cell shape, as `test_recorder.py` does for
        # the same check against the primary recorder.
        exempt = sys.platform == "win32" and check.__name__ == "check_arrival_increases"
        with ratchet("provider-recorder-arrival-increases-per-session") if exempt else nullcontext():
            check(recording_of(recorder), sent, [s.source_port for s in sent])


class TestTheRepliesItSends:
    """R1, R2 — the three bodies, judged by what actually has to read them."""

    def test_the_non_streaming_success_reads_as_a_success_to_the_product(self) -> None:
        """The real adapter must get content and a finish reason out of it.

        Asserting the keys instead would assert this module agrees with itself.
        ``translate_from_upstream`` is what stands between this body and the
        bridge's emptiness judgement, so it is what decides whether the reply is
        a success or an 80-second retry ladder.
        """
        translated = OllamaCloudAdapter().translate_from_upstream(ollama_success_body())

        choice = translated["choices"][0]
        assert choice["message"]["content"], "the reply reads as empty, which costs the retry ladder"
        assert choice["finish_reason"] == "stop"

    def test_the_streamed_success_is_newline_delimited_json(self) -> None:
        """NDJSON, not SSE: the adapter splits on newlines and parses each line."""
        lines = [line for chunk in ollama_success_stream() for line in chunk.split(b"\n") if line.strip()]
        parsed = [json.loads(line) for line in lines]

        assert parsed[-1]["done"] is True, "nothing tells the adapter's loop to finish"
        assert any(obj.get("message", {}).get("content") for obj in parsed), "the stream carries no content"

    def test_the_oauth_body_carries_every_field_the_leg_requires(self) -> None:
        """Every field is judged by the code that reads it, not by this module.

        The two exchanges validate three fields between them. The other two are
        read one frame further out, where ``run_oauth_flow`` builds the session
        it returns — and ``from_token_response`` subscripts ``refresh_token``
        rather than defaulting it, so a body missing that field ends the login
        flow in a ``KeyError`` after both requests have already succeeded.
        """
        body = oauth_token_body()

        assert body["access_token"] and body["id_token"], "the code exchange rejects this"
        assert body["openai_api_key"], "the id_token exchange rejects this"

        session = OAuthSession.from_token_response(body, "client-id")
        assert session.refresh_token, "the flow's last step raises KeyError without it"
        assert session.access_token_expires_at > time.time(), "expires_in was not read"

    def test_a_format_this_recorder_does_not_serve_is_refused_at_construction(self) -> None:
        """Failing here beats replying in a format no adapter asked for."""
        with pytest.raises(ValueError, match="ollama_chat"):
            ProviderRecordingUpstream(default_format=WireFormat.CHAT_COMPLETIONS)

    async def test_the_oauth_endpoint_is_answered_and_is_not_an_unmatched_path(
        self, recorder: ProviderRecordingUpstream
    ) -> None:
        """R2 — a token exchange has no wire format and must not be reported as one.

        Args:
            recorder: The running recorder.
        """
        async with aiohttp.ClientSession() as session, session.post(
            f"{recorder.base_url}{OAUTH_TOKEN_SUFFIX}", data={"grant_type": "authorization_code"}
        ) as response:
            assert response.status == 200
            assert await response.json() == oauth_token_body()

        assert recorder.unmatched == []

    async def test_the_replies_carry_the_headers_a_provider_would_send(
        self, recorder: ProviderRecordingUpstream
    ) -> None:
        """A recorder impersonates a provider, and headers are part of that.

        Neither claim is visible to the adapter — it reads the body and ignores
        both — so nothing else in this module goes red if either is wrong.
        Ollama declares NDJSON on the streamed reply, and a complete JSON body
        arrives with a length rather than chunked.

        Args:
            recorder: The running recorder.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{recorder.base_url}{OLLAMA_CHAT_SUFFIX}", json={"stream": True}) as streamed:
                assert streamed.headers["Content-Type"] == "application/x-ndjson"
                await streamed.read()

            async with session.post(f"{recorder.base_url}{OLLAMA_CHAT_SUFFIX}", json={"stream": False}) as whole:
                assert whole.headers["Content-Type"] == "application/json"
                assert "Content-Length" in whole.headers
                assert "Transfer-Encoding" not in whole.headers

    async def test_a_path_matching_no_suffix_is_recorded(self, recorder: ProviderRecordingUpstream) -> None:
        """The teardown assertion has to have something to report.

        Args:
            recorder: The running recorder.
        """
        await send(recorder.host, recorder.port, probe("stray", path="/v1/chat/completions"), marker="stray")

        assert recorder.unmatched == ["/v1/chat/completions"]
        with pytest.raises(UnmatchedPathError, match="/v1/chat/completions"):
            recorder.assert_all_paths_matched()

        # Clear it, or the fixture's own teardown check reports this deliberate
        # miss as a failure.
        recorder.unmatched.clear()

    def test_the_oauth_suffix_and_the_format_suffix_are_distinct_questions(self) -> None:
        """Neither endpoint may answer for the other.

        The two lookups are separate functions rather than one table because
        only one of them yields a :class:`WireFormat`; asserting they do not
        overlap is what keeps that separation real.
        """
        assert is_oauth_token_path(f"/v1{OAUTH_TOKEN_SUFFIX}")
        assert format_for_provider_path(f"/v1{OAUTH_TOKEN_SUFFIX}") is None
        assert not is_oauth_token_path(OLLAMA_CHAT_SUFFIX)


class TestTheTransport:
    """R4 — what T-W8's interface asks of a transport, on this one."""

    def test_it_satisfies_the_extension_interface(self) -> None:
        """The check is structural: T-W8's protocol is ``runtime_checkable``."""
        assert isinstance(ProviderAiohttpTransport(FORMAT), UpstreamTransport)

    def test_importing_the_module_registers_it(self) -> None:
        """An Epic B transport is reachable by name once its module is imported."""
        assert ProviderAiohttpTransport.name in registered_transports()

    def test_binding_before_starting_raises(self) -> None:
        """There is no port to name until the recorder has one."""
        with pytest.raises(RuntimeError, match="not running"):
            ProviderAiohttpTransport(FORMAT).bind()

    async def test_it_binds_the_ollama_adapter_at_its_own_recorder(self) -> None:
        """``bind()`` is the seam: the fixture never builds this config itself."""
        subject = ProviderAiohttpTransport(FORMAT)
        await subject.start()
        try:
            adapter, config = subject.bind()

            assert isinstance(adapter, OllamaCloudAdapter)
            assert config == {"base_url": subject.recorder.base_url}
        finally:
            await subject.stop()

    async def test_every_bind_returns_the_one_adapter_whose_session_it_closes(self) -> None:
        """A fresh adapter per call would leave a session per call unreachable."""
        subject = ProviderAiohttpTransport(FORMAT)
        await subject.start()
        try:
            assert subject.bind()[0] is subject.bind()[0]
        finally:
            await subject.stop()

    async def test_the_session_closes_before_the_port_is_released(self) -> None:
        """The order is a requirement, so it is observed rather than asserted in prose.

        Closing the port first would reset a keep-alive connection the adapter's
        pool still holds. Swapping the two statements leaves every other test in
        this module green — measured — so without this the ordering claim is one
        no defect could falsify.
        """
        subject = ProviderAiohttpTransport(FORMAT)
        await subject.start()
        adapter, config = subject.bind()
        await adapter.make_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "_resolved_key": "harness-key",
                "_provider_config": config,
            }
        )

        # Wrap the recorder's own stop, which is the step the session must
        # precede, and record what the session looked like when it ran.
        closed_when_port_released: list[bool] = []
        original_stop = subject.recorder.stop

        async def _observe() -> None:
            """Record the session's state, then release the port."""
            closed_when_port_released.append(adapter._session is not None and adapter._session.closed)
            await original_stop()

        subject.recorder.stop = _observe  # type: ignore[method-assign]
        await subject.stop()

        assert closed_when_port_released == [True]

    async def test_stopping_closes_the_session_the_adapter_owns(self) -> None:
        """``BridgeServer.stop_async`` closes only the sessions it owns.

        Without this the ``ollama_cloud`` session outlives every test that
        started one: an "Unclosed client session" per test, and a real leak in a
        gate that runs thousands.
        """
        subject = ProviderAiohttpTransport(FORMAT)
        async with BridgeFixture(subject) as fixture:
            await fixture.post(inbound_path(ROUTE), minimal_inbound_body(ROUTE, marker()))
            adapter, _config = subject.bind()
            session = adapter._session

        assert session is not None, "the request did not go through the adapter's own session"
        assert session.closed


class TestThroughARealBridge:
    """R1, R4, R5 — the claim the whole delivery exists to support."""

    async def test_a_request_reaches_the_recorder_as_ollama_chat(self) -> None:
        """The user's text survives the bridge's CC → Ollama translation."""
        subject = ProviderAiohttpTransport(FORMAT)
        sent = marker()

        async with BridgeFixture(subject) as fixture:
            status, _text = await fixture.post(inbound_path(ROUTE), minimal_inbound_body(ROUTE, sent))
            captures = list(subject.captures)

        assert status == 200
        assert len(captures) == 1
        assert captures[0].path == OLLAMA_CHAT_SUFFIX
        assert sent.encode() in (captures[0].body or b"")

    async def test_the_capture_carries_the_peer_port_containment_joins_on(self) -> None:
        """§5.2.1's join key, required of every recorder and not only the primary."""
        subject = ProviderAiohttpTransport(FORMAT)

        async with BridgeFixture(subject) as fixture:
            await fixture.post(inbound_path(ROUTE), minimal_inbound_body(ROUTE, marker()))
            captures = list(subject.captures)
            connections = list(subject.connections)

        assert captures[0].peer_port
        assert [c.peer_port for c in connections] == [captures[0].peer_port]
        assert sum(c.requests for c in connections) == 1

        # The join is over **connections**, never requests (§5.2.1), and
        # `unattributable_peer_ports` raises rather than guess when a port is
        # unset -- so putting this transport's own connection log through it is
        # what proves the field satisfies T-E5's precondition, not merely that
        # something was recorded. With no tunnels to explain them, every port
        # comes back unattributed; that is the shape, and T-E5 supplies the
        # proxy that makes the list empty.
        assert unattributable_peer_ports([c.peer_port for c in connections], []) == [captures[0].peer_port]

    async def test_a_streamed_request_is_translated_back_to_chat_completions_sse(self) -> None:
        """The reply is NDJSON upstream and SSE downstream; both sides must hold."""
        subject = ProviderAiohttpTransport(FORMAT)
        sent = marker()

        async with BridgeFixture(subject) as fixture:
            status, text = await fixture.post(
                inbound_path(ROUTE), minimal_inbound_body(ROUTE, sent, stream=True)
            )
            captures = list(subject.captures)

        assert status == 200
        assert "data: [DONE]" in text
        assert '"content": "ok"' in text or '"content":"ok"' in text
        # The finish reason comes only from the final NDJSON object. Without
        # this the stream's last line could be malformed and the test would
        # still pass on the content chunk alone -- measured, not supposed.
        assert '"finish_reason": "stop"' in text or '"finish_reason":"stop"' in text
        assert json.loads(captures[0].body)["stream"] is True

    async def test_it_passes_the_integration_conformance_check(self) -> None:
        """R5 — the check every transport is judged by, on this one by name.

        The meta-test in ``test_bridge.py`` runs the same check from the
        ``CONFORMANCE_CASES`` row. Both are wanted: that one proves the row is
        wired, this one fails in this module when the transport is what broke.
        """
        await assert_transport_reaches_its_recorder(transport(ProviderAiohttpTransport.name, FORMAT), protocol=ROUTE)


class TestTheOAuthLoginLeg:
    """R6 — the leg with no adapter, and the seam that redirects it."""

    async def test_the_seam_swaps_the_endpoint_and_restores_it(
        self, recorder: ProviderRecordingUpstream
    ) -> None:
        """The constant is the leg's only channel; it must not stay swapped.

        Args:
            recorder: The running recorder.
        """
        original = openai_oauth.OAUTH_TOKEN_URL

        with oauth_token_endpoint(recorder) as url:
            assert url == openai_oauth.OAUTH_TOKEN_URL
            assert url.startswith(recorder.base_url)

        assert original == openai_oauth.OAUTH_TOKEN_URL

    async def test_the_seam_restores_the_endpoint_when_the_body_raises(
        self, recorder: ProviderRecordingUpstream
    ) -> None:
        """A failing test must not leave the next one posting at a dead port.

        Args:
            recorder: The running recorder.
        """
        original = openai_oauth.OAUTH_TOKEN_URL

        with pytest.raises(RuntimeError), oauth_token_endpoint(recorder):
            raise RuntimeError("boom")

        assert original == openai_oauth.OAUTH_TOKEN_URL

    async def test_the_seam_refuses_to_swap_a_name_nothing_defines(
        self, recorder: ProviderRecordingUpstream, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A silent no-op would leave a test passing over an empty capture list.

        If the leg is ever rewritten to read its endpoint from somewhere else,
        this is what says so — rather than the seam setting a name nothing
        consults while the test that used it captures nothing and asserts on it.

        Args:
            recorder: The running recorder.
            monkeypatch: Removes the attribute, standing in for that rewrite.
        """
        monkeypatch.delattr(openai_oauth, "OAUTH_TOKEN_URL")

        with pytest.raises(AttributeError), oauth_token_endpoint(recorder):
            pass  # pragma: no cover - the context manager raises on entry

    async def test_both_token_exchanges_are_captured_in_full(
        self, recorder: ProviderRecordingUpstream
    ) -> None:
        """The product's own coroutines, against the recorder, at startup fidelity.

        Driven directly rather than through ``run_oauth_flow``: that orchestrator
        binds a local callback server, opens a browser and waits five minutes for
        a human. These two coroutines are the whole of what reaches the network.

        Args:
            recorder: The running recorder.
        """
        with oauth_token_endpoint(recorder):
            async with aiohttp.ClientSession() as http:
                tokens = await openai_oauth._exchange_code_for_tokens("code", "verifier", "client-id", http)
                key = await openai_oauth._exchange_id_token_for_api_key(
                    tokens["id_token"], tokens["access_token"], "client-id", http
                )

        assert key == oauth_token_body()["openai_api_key"]

        captures = recorder.requests
        assert [c.method for c in captures] == ["POST", "POST"]
        assert [c.path for c in captures] == [OAUTH_TOKEN_SUFFIX, OAUTH_TOKEN_SUFFIX]
        assert all(c.peer_port for c in captures), "containment cannot join a capture with no peer port"

        grants = [c.body.decode().split("&")[0] for c in captures]
        assert grants[0] == "grant_type=authorization_code"
        assert grants[1].startswith("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Atoken-exchange")

    async def test_a_token_exchange_does_not_fail_the_declared_format_claim(self) -> None:
        """The exclusion is deliberate, so it is asserted rather than left to silence.

        The leg is answered before any format lookup — it has no
        :class:`WireFormat` and must not be reported as a fallback — which means
        ``assert_teardown_clean`` passes over it. That is a decision, and an
        undocumented, untested decision is indistinguishable from a hole.
        """
        subject = ProviderAiohttpTransport(FORMAT)
        await subject.start()
        try:
            with oauth_token_endpoint(subject.recorder):
                async with aiohttp.ClientSession() as http:
                    await openai_oauth._exchange_code_for_tokens("code", "verifier", "client-id", http)

            assert len(subject.captures) == 1
            subject.assert_teardown_clean()
        finally:
            await subject.stop()

    async def test_the_leg_is_recorded_as_a_request_like_any_other(
        self, recorder: ProviderRecordingUpstream
    ) -> None:
        """One recorder, two products: §7.2 assigns both to this row.

        Args:
            recorder: The running recorder.
        """
        with oauth_token_endpoint(recorder):
            async with aiohttp.ClientSession() as http:
                await openai_oauth._exchange_code_for_tokens("code", "verifier", "client-id", http)

        captured = recorder.requests[0]
        assert dict(captured.headers).get("Content-Type") == "application/x-www-form-urlencoded"
        assert recorder.connections[0].requests == 1


# ── Falsification (plan §1.4) ────────────────────────────────────────────────


class _BlindProviderRecorder(ProviderRecordingUpstream):
    """A recorder that counts a request and keeps none of its content."""

    def store(self, index: int, captured: CapturedRequest) -> None:
        """Store the capture with its body removed.

        ``store`` is T-W4's seam for exactly this: a defect that disturbs one
        thing and leaves the connection log and the ordering intact.

        Args:
            index: The reserved slot.
            captured: The capture, stored without its body.
        """
        super().store(index, replace(captured, body=b""))


@dataclass
class _BlindProviderTransport(ProviderAiohttpTransport):
    """Counts the request and records nothing about it.

    Plan §1.4's fourth named harness failure — "a projection that could not see
    the model name, in a product whose purpose is changing the model name" — in
    this transport's own terms.
    """

    name = "falsify-provider-blind"

    def __post_init__(self) -> None:
        """Use the body-dropping recorder instead of the real one."""
        self._recorder = _BlindProviderRecorder(default_format=self.format, responder=self.responder)


@dataclass
class _DecoyProviderTransport(ProviderAiohttpTransport):
    """Points the bridge at a *different* live recorder than the one it reports.

    §7.5.4's measured row 3, and the expensive failure: the request succeeds,
    the client gets 200, and this transport's capture list is empty — so every
    assertion built on it would be quantified over nothing and pass for free.
    """

    name = "falsify-provider-decoy"

    _decoy: ProviderRecordingUpstream = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build both recorders: the one reported, and the one actually reached."""
        super().__post_init__()
        self._decoy = ProviderRecordingUpstream(default_format=self.format)

    async def start(self) -> None:
        """Start both, so the decoy really answers rather than timing out."""
        await super().start()
        await self._decoy.start()

    async def stop(self) -> None:
        """Stop both, in a ``finally`` so neither leaks if the other raises."""
        try:
            await super().stop()
        finally:
            await self._decoy.stop()

    def bind(self) -> Binding:
        """Return a binding that reaches the decoy.

        Returns:
            This transport's adapter, pointed elsewhere.
        """
        adapter, _config = super().bind()
        return adapter, {"base_url": self._decoy.base_url}


async def _failure_message(subject: ProviderAiohttpTransport) -> str:
    """Run the conformance check against a defect and return why it failed.

    Args:
        subject: An unstarted defective transport.

    Returns:
        The failure message.

    Raises:
        Failed: When the check passed, which is the whole point of this section.
    """
    with pytest.raises(AssertionError) as excinfo:
        await assert_transport_reaches_its_recorder(subject, protocol=ROUTE)
    return str(excinfo.value)


class TestEachDefectIsCaught:
    """R7 — the deliberate defects this harness must detect, running in the suite."""

    async def test_a_binding_that_reaches_another_upstream_is_caught(self) -> None:
        """The decoy — the failure no status check can see."""
        message = await _failure_message(_DecoyProviderTransport(FORMAT))

        assert "0 capture" in message
        assert "quantified over nothing" in message

    async def test_a_capture_that_cannot_see_the_content_is_caught(self) -> None:
        """The blind capture — right count, right status, no evidence."""
        message = await _failure_message(_BlindProviderTransport(FORMAT))

        assert "marker" in message

    async def test_a_reply_in_the_fallback_format_is_caught_at_teardown(self) -> None:
        """The mis-declared format, in the shape this recorder can actually have.

        T-W8's own case for this binds an adapter posting to the *other* format
        the primary recorder serves, so the reply is correct and nothing is
        recorded. This recorder serves one format, so the same mistake takes the
        fallback instead and ``assert_teardown_clean`` is what reports it — which
        is why the transport has one teardown check and not two.
        """
        subject = ProviderAiohttpTransport(FORMAT)
        await subject.start()
        try:
            await send(
                subject.recorder.host,
                subject.recorder.port,
                probe("stray", path="/v1/chat/completions"),
                marker="stray",
            )
            with pytest.raises(UnmatchedPathError, match="/v1/chat/completions"):
                subject.assert_teardown_clean()
        finally:
            await subject.stop()

    async def test_the_defects_are_distinguishable(self) -> None:
        """Each must break its own assertion, or the suite has halved its evidence.

        Without this, a later tidy-up that collapsed two assertions into one
        would leave both cases above green while they tested one thing.
        """
        decoy = await _failure_message(_DecoyProviderTransport(FORMAT))
        blind = await _failure_message(_BlindProviderTransport(FORMAT))

        assert decoy != blind
        assert "capture" in decoy and "marker" not in decoy
        assert "marker" in blind

    async def test_the_blind_defect_still_reaches_its_own_recorder(self) -> None:
        """Stated positively, so the decoy's emptiness stays the decoy's alone.

        If a change made every defect produce an empty recording, both cases
        above would still pass and both would be testing the decoy.
        """
        blind = _BlindProviderTransport(FORMAT)
        with pytest.raises(AssertionError):
            await assert_transport_reaches_its_recorder(blind, protocol=ROUTE)

        assert len(blind.captures) == 1
        assert blind.captures[0].body == b""


class _CcBodyRecorder(ProviderRecordingUpstream):
    """Answers with the Chat Completions body the *parent* recorder would send.

    The defect a lost responder override would produce, and the one no
    inherited machinery can catch: T-W4's recorder and T-W8's fixture are both
    intact here, and the only thing wrong is the reply shape this task chose.
    """

    async def _default_responder(self, captured: CapturedRequest, response: Reply) -> None:
        """Reply in the format the primary recorder serves.

        Args:
            captured: The recorded request, unused.
            response: The unprepared response.
        """
        await self._reply_json(response, minimal_success_body(WireFormat.CHAT_COMPLETIONS))


class _SseStreamRecorder(ProviderRecordingUpstream):
    """Streams SSE frames where ``/api/chat`` speaks NDJSON."""

    async def _default_responder(self, captured: CapturedRequest, response: Reply) -> None:
        """Reply with the primary recorder's SSE stream.

        Args:
            captured: The recorded request, unused.
            response: The unprepared response.
        """
        await response.begin(200, {"Content-Type": "text/event-stream"})
        for chunk in minimal_success_stream(WireFormat.CHAT_COMPLETIONS):
            await response.write(chunk)
        await response.write_eof()


async def _through_the_adapter(recorder: ProviderRecordingUpstream, *, stream: bool) -> str:
    """Drive the real ``ollama_cloud`` adapter against ``recorder`` and return what it produced.

    The product is the judge here, not this module: whether a reply is a success
    or an empty response is ``translate_from_upstream``'s answer and the stream
    parser's, and those are the two things a wrong reply shape actually meets.

    Args:
        recorder: A started recorder.
        stream: Whether to drive the streaming path.

    Returns:
        The assistant text the adapter produced, empty when it could read none.
    """
    adapter = OllamaCloudAdapter()
    request = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
        "_resolved_key": "harness-key",
        "_provider_config": {"base_url": recorder.base_url},
    }
    try:
        if not stream:
            translated = await adapter.make_request(request)
            return translated["choices"][0]["message"]["content"] or ""

        written: list[bytes] = []

        async def _collect(chunk: bytes) -> None:
            """Collect one written chunk.

            Args:
                chunk: The bytes the adapter wrote downstream.
            """
            written.append(chunk)

        await adapter.stream_request(request, _collect)
        return "".join(
            json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            for line in b"".join(written).decode().splitlines()
            if line.startswith("data: ") and line[6:].strip() != "[DONE]"
        )
    finally:
        if adapter._session is not None:
            await adapter._session.close()


class TestTheReplyShapeIsLoadBearing:
    """R7 — a falsification case against what **this task** wrote.

    The two defects above ride on inherited machinery: ``store`` is T-W4's seam
    and ``bind`` is T-W8's. Neither goes red if this task's own reply overrides
    are wrong, and plan §1.4 asks for a defect *this* harness must detect.

    **These are judged through the adapter rather than through the bridge, and
    that is measured.** A defective reply driven through
    ``assert_transport_reaches_its_recorder`` was measured at **4 captures, a
    10-second timeout, and 72 seconds** including the teardown that waits for
    the retry ladder — against §8.2's 1.6-second budget for the whole T-W8 pair.
    The defect is real and caught either way; this way costs milliseconds and
    says which of the two shapes was wrong.
    """

    async def test_a_chat_completions_body_leaves_the_product_with_nothing(
        self, recorder: ProviderRecordingUpstream
    ) -> None:
        """Positive control first, then the defect, so neither can pass vacuously.

        Args:
            recorder: The running recorder, serving the real Ollama body.
        """
        assert await _through_the_adapter(recorder, stream=False) == "ok"

        defective = _CcBodyRecorder(default_format=FORMAT)
        await defective.start()
        try:
            assert await _through_the_adapter(defective, stream=False) == ""
        finally:
            await defective.stop()

    async def test_an_sse_stream_leaves_the_product_with_nothing(
        self, recorder: ProviderRecordingUpstream
    ) -> None:
        """The NDJSON choice is the delivery's, and this is what makes it falsifiable.

        Args:
            recorder: The running recorder, serving the real NDJSON stream.
        """
        assert await _through_the_adapter(recorder, stream=True) == "ok"

        defective = _SseStreamRecorder(default_format=FORMAT)
        await defective.start()
        try:
            assert await _through_the_adapter(defective, stream=True) == ""
        finally:
            await defective.stop()


class TestTheRecorderStaysIndependentOfKitty:
    """The rule ``contract.py`` enforces, extended to this recorder.

    Asserting the property in prose only is what ``contract.py``'s own docstring
    calls a defect: *"a rule that reads like a guarantee and guarantees
    nothing"*. The **transport** module is deliberately not guarded — binding an
    adapter is what it is for — and neither is this test module.
    """

    def test_the_recorder_imports_nothing_from_kitty(self) -> None:
        """A recorder that asked kitty how to read a request would inherit its bugs."""
        source = Path(provider_recorder_module.__file__).read_text(encoding="utf-8")
        assert len(source) > 1000, "read no meaningful source; the guard would pass vacuously"

        offending = [line.strip() for line in source.splitlines() if _KITTY_IMPORT.search(line)]
        assert offending == [], f"the provider recorder must not import kitty: {offending}"
