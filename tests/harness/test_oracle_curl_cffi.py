"""Driven end-to-end tests for the transparency oracle on the curl_cffi transport.

`.system_design/TEST_SUITE.md` §3.3, §3.3.2, §3.3.4, §3.3.5, §7.6 · plan task
**T-D5** (KBR-55).

Tests run at ``l1`` (path-default per ``tests/layers.py`` and the
vertical-slice precedent — ``tests/harness/test_oracle_driven.py`` for
T-D1, ``tests/harness/test_oracle_routing.py`` for T-D2). §3.4 calls
the oracle surface L3, and T-K6 owns the activation. Today an ``l3``
marker deselects the test in the Fast gate
(``two-pytest-gates-deselect-acceptance`` memory).

**Driven, end-to-end.** A real ``BridgeFixture`` is started against the
``curl_cffi`` recorder (KBR-41's
``tests/harness/curl_cffi.CurlCffiTransport``) with **harness TLS** —
the recorder terminates TLS with the harness CA, and the bridge's
``CODEX_CA_CERTIFICATE`` env wiring lets the adapter trust it. A
minimal Anthropic Messages body is posted to ``/v1/messages`` (the
CC-origin path through ``OpenAISubscriptionAdapter._cc_to_responses``),
and the oracle compares the captured OpenAI Responses body to the
Anthropic projection of the inbound.

A second driven case posts to ``/v1/responses`` so
``_prepare_responses_body`` runs against ``_original_body`` (the
``RESPONSES_ORIGIN_PATH`` carry).

**Why the driven body is deliberately minimal.** A minimal Anthropic
Messages body carries **no** ``cache_control`` / Anthropic
``metadata`` / ``top_k`` / non-empty ``stop_sequences`` /
string-form ``stop`` / Anthropic-defined tool / forcing
``tool_choice`` — so M16, M26, M27 (G28), M28 (G29), M29 (G30),
P34 and P35 do not fire on the happy path, and the run goes
green with only the unavoidable-for-the-row-set deltas
(``envelope.stream``, ``envelope.store``, ``envelope.model``).
The falsification tests add the trigger each row is supposed to
ignore or absorb.

**What T-D5 does NOT drive.** P33's trigger case (the bedrock
adapter's ``Converse`` rewrite of an absent ``tool_choice``) —
T-D6's deliverable. The bedrock slice owns it.
"""

from __future__ import annotations

import dataclasses
import json
import ssl
from pathlib import Path

import pytest

from harness import oracle
from harness import register as r
from harness.bridge import (
    BridgeFixture,
    InboundProtocol,
    inbound_path,
    minimal_inbound_body,
)
from harness.contract import CapturedRequest, WireFormat
from harness.curl_cffi import CurlCffiTransport, seed_oauth_session
from harness.curl_recorder import CODEX_RESPONSES_SUFFIX

#: Marker the oracle should find verbatim in the upstream body.
#: A short string survives every layer of JSON encoding.
_SENTINEL = "kbr55-t-d5-oracle-driven"

#: Trigger set for the CC-origin (CC→Responses via ``_cc_to_responses``)
#: driven case. ``Trigger.ALWAYS`` is required because P1 / M26 / P17
#: are unconditional — their ``envelope.*`` paths would show as
#: unclaimed deltas against a ``triggers_met`` that omitted it (P17
#: rewrites ``stream`` and forces ``store`` on every request; M1
#: rewrites the model on every request). ``CC_ORIGIN_PATH`` is met
#: when the adapter dispatches via ``_cc_to_responses``.
_CC_ORIGIN_TRIGGERS: frozenset[r.Trigger] = frozenset(
    {
        r.Trigger.ALWAYS,
        r.Trigger.NON_NATIVE_UPSTREAM_WIRE,
        r.Trigger.PROFILE_SETS_MODEL,
        r.Trigger.CC_ORIGIN_PATH,
    }
)

#: Trigger set for the Responses-origin driven case (the agent
#: posts to ``/v1/responses`` so ``_original_body`` is set on the
#: ``cc_request`` and ``_prepare_responses_body`` is the dispatch
#: site).
_RESPONSES_ORIGIN_TRIGGERS: frozenset[r.Trigger] = frozenset(
    {
        r.Trigger.ALWAYS,
        r.Trigger.NON_NATIVE_UPSTREAM_WIRE,
        r.Trigger.PROFILE_SETS_MODEL,
        r.Trigger.RESPONSES_ORIGIN_PATH,
    }
)

#: Triggers for the routing assertion only. Body obligations are
#: irrelevant to the routing check, so the lighter set avoids
#: order effects between body and routing failures.
_ROUTE_TRIGGERS: frozenset[r.Trigger] = frozenset(
    {r.Trigger.NON_NATIVE_UPSTREAM_WIRE, r.Trigger.PROFILE_SETS_MODEL}
)


# --------------------------------------------------------------------------
# Harness TLS fixtures — same shape as tests/harness/test_curl_cfi.py
# --------------------------------------------------------------------------

pytestmark = pytest.mark.usefixtures("certs")


@pytest.fixture
def server_context(certs) -> ssl.SSLContext:  # noqa: F811
    """Server-side TLS context the recorder terminates with.

    Args:
        certs: The session-scoped throwaway certificate set.

    Returns:
        The context the recorder presents.
    """
    from harness.connect_proxy import server_ssl_context

    return server_ssl_context(certs.target_cert, certs.target_key)


def _ca_path(tmp_path: Path, certs) -> Path:
    """Write the harness CA to a path the transport can read.

    Args:
        tmp_path: The pytest temp directory.
        certs: The session-scoped fixture, for the CA bytes.

    Returns:
        The file path the harness CA bytes were written to.
    """
    ca = tmp_path / "kbr55-ca.pem"
    ca.write_bytes(Path(str(certs.ca)).read_bytes())
    return ca


def _build_transport(server_context: ssl.SSLContext, tmp_path: Path, certs) -> CurlCffiTransport:
    """Build an unstarted ``curl_cffi`` transport with TLS and the harness CA wired.

    Args:
        server_context: TLS context the recorder terminates with.
        tmp_path: Where to write the CA file.
        certs: The session-scoped fixture, used to read the CA bytes.

    Returns:
        An unstarted transport; the caller owns its lifecycle.
    """
    return CurlCffiTransport(
        format=WireFormat.OPENAI_RESPONSES,
        ssl_context=server_context,
        ca_cert=_ca_path(tmp_path, certs),
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

async def _post_one(
    server_context: ssl.SSLContext,
    tmp_path: Path,
    certs,
    protocol: InboundProtocol,
    body: dict,
) -> CapturedRequest:
    """Drive one request through the bridge against the curl_cffi recorder.

    Posts ``body`` to the bridge route matching ``protocol``, against
    a started transport with a fresh OAuth session. Returns the
    single capture the recorder records — never the list, because
    the slice's assumption is exactly one.

    Args:
        server_context: TLS context for the recorder.
        tmp_path: Test-scoped temp dir; OAuth file lives here.
        certs: The session-scoped harness CA + key fixture.
        protocol: Which inbound protocol to drive (MESSAGES or
            RESPONSES).
        body: The inbound body the bridge sees (a dict that will be
            JSON-encoded by the bridge fixture).

    Returns:
        The single ``CapturedRequest`` the recorder recorded.
    """
    async with BridgeFixture(
        _build_transport(server_context, tmp_path, certs),
        key=seed_oauth_session(tmp_path),
    ) as fixture:
        status, response_body = await fixture.post(
            inbound_path(protocol), body
        )

        assert status == 200, (
            f"the recorder's minimal success reply must come back 200; "
            f"got {status}: {response_body[:200]!r}"
        )

        captures = list(fixture.captures)
        assert len(captures) == 1, (
            "exactly one upstream request — a second capture would mean "
            "the empty-response retry ladder fired"
        )
        return captures[0]


def _reconstructed_inbound(protocol: InboundProtocol, body: dict) -> CapturedRequest:
    """Synthesize the ``CapturedRequest`` the oracle needs for the inbound half.

    The Messages adapter does not alter the inbound scheme / host / path
    (those are the bridge's own), so the inbound projection is built from
    what Claude Code would have sent.

    Args:
        protocol: Which inbound protocol was driven.
        body: The JSON body the bridge saw.

    Returns:
        A ``CapturedRequest`` whose body is the JSON-encoded body bytes.
    """
    return CapturedRequest(
        method="POST",
        scheme="http",
        host="127.0.0.1",
        path=inbound_path(protocol),
        query="",
        body=json.dumps(body).encode("utf-8"),
    )


def _expected_route() -> oracle.ExpectedRoute:
    """Build the routing expectation the oracle checks.

    The ``codex_backend_url`` seam (KBR-41) rewrites the adapter's
    module constant to ``f"{recorder.base_url}{CODEX_RESPONSES_SUFFIX}"``,
    so the captured path is the **recorder's suffix**, not the
    published Codex pathname ``/backend-api/codex/responses``. The
    authority and scheme are rewritten to the recorder's own
    (``§3.3.5``'s mandatory normalisation; T-D2).

    Returns:
        The independent-route expectation, derived from
        ``CODEX_RESPONSES_SUFFIX`` (``"/responses"``), with method
        ``"POST"``, scheme and authority to be rewritten by the
        caller, and empty query.
    """
    return oracle.ExpectedRoute(
        method="POST",
        scheme="https",
        host="",
        path=CODEX_RESPONSES_SUFFIX,
        query="",
    )


# --------------------------------------------------------------------------
# The slice
# --------------------------------------------------------------------------

class TestCurlCffiOracleSlice:
    """End-to-end driven run on the curl_cffi transport for openai_subscription."""

    async def test_cc_origin_minimal_run_no_unclaimed_delta(
        self, server_context: ssl.SSLContext, tmp_path: Path, certs
    ) -> None:
        """A minimal Anthropic Messages body produces zero unclaimed deltas.

        Args:
            server_context: TLS context for the recorder.
            tmp_path: Test-scoped temp dir.
            certs: The session-scoped harness CA + key fixture.
        """
        body = minimal_inbound_body(InboundProtocol.MESSAGES, _SENTINEL)

        captured = await _post_one(
            server_context, tmp_path, certs, InboundProtocol.MESSAGES, body
        )

        report = oracle.assert_no_unclaimed_mutation(
            inbound=_reconstructed_inbound(InboundProtocol.MESSAGES, body),
            inbound_format=WireFormat.ANTHROPIC_MESSAGES,
            captured=captured,
            captured_format=WireFormat.OPENAI_RESPONSES,
            register=r.REGISTER,
            triggers_met=_CC_ORIGIN_TRIGGERS,
        )

        # A green run is "the call returned normally" — the oracle raises
        # on any assertion-1 / assertion-2 failure. ``report.deltas`` is
        # the FULL delta list, not the unclaimed subset, so on this route
        # it carries the three deltas the row set claims. Pinning them
        # turns a regression that widens the delta set red rather than
        # silent.
        assert report.deltas == (
            "envelope.stream",
            "envelope.store",
            "conversation.sampling[max_tokens]",
        ), (
            f"expected exactly the three claimed deltas; got {report.deltas!r}"
        )

    async def test_responses_origin_minimal_run_no_unclaimed_delta(
        self, server_context: ssl.SSLContext, tmp_path: Path, certs
    ) -> None:
        """The same minimal body via ``/v1/responses`` (the Responses-origin path).

        Args:
            server_context: TLS context for the recorder.
            tmp_path: Test-scoped temp dir.
            certs: The session-scoped harness CA + key fixture.
        """
        body = minimal_inbound_body(InboundProtocol.RESPONSES, _SENTINEL)

        captured = await _post_one(
            server_context, tmp_path, certs, InboundProtocol.RESPONSES, body
        )

        # The run asserts the absence of an exception; the comment near
        # ``_CC_ORIGIN_TRIGGERS`` records why this is the right check. A
        # future regression that widens the deltas set surfaces a
        # different number (handled by the dedicated CC-origin test).
        oracle.assert_no_unclaimed_mutation(
            inbound=_reconstructed_inbound(InboundProtocol.RESPONSES, body),
            inbound_format=WireFormat.OPENAI_RESPONSES,
            captured=captured,
            captured_format=WireFormat.OPENAI_RESPONSES,
            register=r.REGISTER,
            triggers_met=_RESPONSES_ORIGIN_TRIGGERS,
        )

    async def test_route_derivation_uses_the_recorder_suffix(
        self, server_context: ssl.SSLContext, tmp_path: Path, certs
    ) -> None:
        """``expected_route`` is derived from the recorder's suffix, not the published Codex pathname.

        The recorder seam rewrites the adapter's constant to
        ``f"{recorder.base_url}{CODEX_RESPONSES_SUFFIX}"`` and the
        captured path therefore is the suffix, not the published
        ``/backend-api/codex/responses``. This test pins the
        derivation shape (the BLOCKING-2 review finding): the
        assertion uses the recorder's own authority and scheme so
        the comparison passes (§3.3.5's mandatory normalisation).

        Args:
            server_context: TLS context for the recorder.
            tmp_path: Test-scoped temp dir.
            certs: The session-scoped harness CA + key fixture.
        """
        body = minimal_inbound_body(InboundProtocol.MESSAGES, _SENTINEL)
        captured = await _post_one(
            server_context, tmp_path, certs, InboundProtocol.MESSAGES, body
        )

        recorder_authority = captured.host  # ``127.0.0.1:<port>``
        recorder_scheme = captured.scheme  # ``"https"`` with harness TLS

        expected = dataclasses.replace(
            _expected_route(), scheme=recorder_scheme, host=recorder_authority
        )

        oracle.assert_no_unclaimed_mutation(
            inbound=_reconstructed_inbound(InboundProtocol.MESSAGES, body),
            inbound_format=WireFormat.ANTHROPIC_MESSAGES,
            captured=captured,
            captured_format=WireFormat.OPENAI_RESPONSES,
            register=r.REGISTER,
            triggers_met=frozenset(_CC_ORIGIN_TRIGGERS | _ROUTE_TRIGGERS),
            expected_route=expected,
        )

    async def test_rerouted_capture_fails_on_routing(
        self, server_context: ssl.SSLContext, tmp_path: Path, certs
    ) -> None:
        """Falsification (per plan §1.4): the routing assertion catches a body-identical reroute.

        The captured request is mutated to carry a different path
        (suffix kept identical to the recorder's so the body shape
        is unchanged — the body assertions still pass, the routing
        assertion must fire).

        Args:
            server_context: TLS context for the recorder.
            tmp_path: Test-scoped temp dir.
            certs: The session-scoped harness CA + key fixture.
        """
        body = minimal_inbound_body(InboundProtocol.MESSAGES, _SENTINEL)
        captured = await _post_one(
            server_context, tmp_path, certs, InboundProtocol.MESSAGES, body
        )

        # Mutate the path to a different value the recorder would also
        # serve — the body shape on the wire is unchanged; only the
        # routing assertion can fail.
        rerouted = dataclasses.replace(
            captured, path="/v1/chat/completions"
        )

        expected = dataclasses.replace(
            _expected_route(),
            scheme=captured.scheme,
            host=captured.host,
        )

        with pytest.raises(oracle.RoutingMismatchError) as info:
            oracle.assert_no_unclaimed_mutation(
                inbound=_reconstructed_inbound(InboundProtocol.MESSAGES, body),
                inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                captured=rerouted,
                captured_format=WireFormat.OPENAI_RESPONSES,
                register=r.REGISTER,
                triggers_met=frozenset(_CC_ORIGIN_TRIGGERS | _ROUTE_TRIGGERS),
                expected_route=expected,
            )

        # The mismatch must name a routing component, never a body
        # path — the body obligations have already passed by the
        # time the routing check fires (§3.3.5 fourth-obligation
        # order).
        assert any(p.startswith("route.") for p in info.value.paths), (
            f"routing mismatch must name route.* paths; got {info.value.paths!r}"
        )

    async def test_p17_override_fails_the_run(
        self, server_context: ssl.SSLContext, tmp_path: Path, certs
    ) -> None:
        """Falsification: removing P17's claim surfaces the forced `stream: True` / `store: False`.

        The CC-origin rebuilt body always injects ``stream: True``
        and ``store: False`` (the Codex backend is streaming-only).
        Removing P17 from the register leaves those two paths
        unclaimed.

        Args:
            server_context: TLS context for the recorder.
            tmp_path: Test-scoped temp dir.
            certs: The session-scoped harness CA + key fixture.
        """
        body = minimal_inbound_body(InboundProtocol.MESSAGES, _SENTINEL)
        captured = await _post_one(
            server_context, tmp_path, certs, InboundProtocol.MESSAGES, body
        )

        register_without_p17 = tuple(
            dataclasses.replace(row, paths=()) if row.id == "P17" else row
            for row in r.REGISTER
        )

        with pytest.raises(oracle.UnclaimedMutationError) as info:
            oracle.assert_no_unclaimed_mutation(
                inbound=_reconstructed_inbound(InboundProtocol.MESSAGES, body),
                inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                captured=captured,
                captured_format=WireFormat.OPENAI_RESPONSES,
                register=register_without_p17,
                triggers_met=_CC_ORIGIN_TRIGGERS,
            )

        assert {"envelope.stream", "envelope.store"} & set(info.value.paths), (
            f"P17's claim is the forced stream + store; the unclaimed "
            f"deltas must name envelope.stream and envelope.store. "
            f"Got: {info.value.paths!r}"
        )

    async def test_p25_override_fails_the_responses_run(
        self, server_context: ssl.SSLContext, tmp_path: Path, certs
    ) -> None:
        """Falsification: removing P25 surfaces the allowlisted-but-falsy drop.

        P25 (KBR-185) records that ``_prepare_responses_body``'s
        truthy-only branches drop an allowlisted Responses field
        whose value is empty (``include: []`` is the canonical
        example: legally valid under ``CreateResponse``, dropped
        because ``if original_body.get('include')`` is False-y for
        an empty list). Driving a Responses body with ``include: []``
        and removing P25 from the register surfaces the drop.

        Args:
            server_context: TLS context for the recorder.
            tmp_path: Test-scoped temp dir.
            certs: The session-scoped harness CA + key fixture.
        """
        body = minimal_inbound_body(InboundProtocol.RESPONSES, _SENTINEL)
        body["include"] = []

        captured = await _post_one(
            server_context, tmp_path, certs, InboundProtocol.RESPONSES, body
        )

        register_without_p25 = tuple(
            dataclasses.replace(row, paths=()) if row.id == "P25" else row
            for row in r.REGISTER
        )

        with pytest.raises(oracle.UnclaimedMutationError) as info:
            oracle.assert_no_unclaimed_mutation(
                inbound=_reconstructed_inbound(InboundProtocol.RESPONSES, body),
                inbound_format=WireFormat.OPENAI_RESPONSES,
                captured=captured,
                captured_format=WireFormat.OPENAI_RESPONSES,
                register=register_without_p25,
                triggers_met=_RESPONSES_ORIGIN_TRIGGERS,
            )

        # P25's anchor is ``envelope.extra[include]`` (§3.2.2). With
        # P25 removed, the absent-on-captured side surfaces as an
        # unclaimed delta at that path.
        assert "envelope.extra[include]" in info.value.paths, (
            f"P25's claim is envelope.extra[include]; the unclaimed "
            f"delta must surface there. Got: {info.value.paths!r}"
        )

    async def test_inbound_content_survives_while_introduced_is_caught(
        self, server_context: ssl.SSLContext, tmp_path: Path, certs
    ) -> None:
        """§3.3.3's distinguishing property, both halves.

        Half 1: an inbound body carrying the string ``"kitty-bridge"``
        reaches the upstream body **byte-identically** — a vendor
        token guard that strips it would break I1 in the act of
        defending I2, which §3.3.3 forbids.

        Half 2: a separate run where the captured body is mutated
        to insert a turn with no inbound counterpart must raise
        ``UnclaimedMutationError`` — that is what the
        distinguishing property exists to detect, and §3.3.3
        records the synthetic-history M13 string T-G5 inherits.
        The mutation uses ``input_text`` (the content type the
        Responses reader consumes) so the orphan turn reaches the
        reader's turns list and surfaces as a delta rather than
        a residual.

        Args:
            server_context: TLS context for the recorder.
            tmp_path: Test-scoped temp dir.
            certs: The session-scoped harness CA + key fixture.
        """
        body = minimal_inbound_body(
            InboundProtocol.MESSAGES, "explain how kitty-bridge works"
        )
        captured = await _post_one(
            server_context, tmp_path, certs, InboundProtocol.MESSAGES, body
        )

        # Half 1: the inbound substring survives byte-identical in
        # the captured body. Direct assertion on the capture, not via
        # the oracle — the KBR-5-historical M13 string shape T-G5
        # inherits is "the text reaches the wire unchanged";
        # assertion 1 (no unclaimed delta) covers the *absence* of
        # mutation, not the *presence* of the byte sequence.
        assert b"kitty-bridge" in captured.body, (
            "§3.3.3 / T-G5's regression contract: an inbound "
            "'kitty-bridge' string survives byte-identical in the "
            "captured body"
        )

        # Half 2: the captured body, with a turn carrying no inbound
        # counterpart, must trigger §3.3.2 assertion 1. The
        # synthetic-history M13 string T-G5 inherits is the
        # canonical example; this slice uses an anonymous marker so
        # the runtime signals carry no ambiguity.
        mutated = json.loads(captured.body.decode("utf-8"))
        mutated["input"] = list(mutated.get("input", [])) + [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "bridge-introduced sentinel for §3.3.3"}
                ],
            }
        ]
        mutated_capture = dataclasses.replace(
            captured, body=json.dumps(mutated).encode("utf-8")
        )

        with pytest.raises(oracle.UnclaimedMutationError) as info:
            oracle.assert_no_unclaimed_mutation(
                inbound=_reconstructed_inbound(InboundProtocol.MESSAGES, body),
                inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                captured=mutated_capture,
                captured_format=WireFormat.OPENAI_RESPONSES,
                register=r.REGISTER,
                triggers_met=_CC_ORIGIN_TRIGGERS,
            )

        # The added user message **merges** with the existing user turn
        # (§3.3.1b merge rule: consecutive same-role turns form one
        # turn, parts in order), so the part lands at
        # ``turns[0].parts[1]`` rather than a fresh ``turns[1]``.
        # Either path is a legitimate orphan; P13 / P17 / no other
        # triggered row claims it on this run.
        assert "conversation.turns[0].parts[1]" in info.value.paths, (
            f"the bridge-introduced part must surface as an unclaimed delta; "
            f"got {info.value.paths!r}"
        )
