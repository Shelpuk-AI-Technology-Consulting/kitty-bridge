"""Driven end-to-end tests for the transparency oracle.

`.system_design/TEST_SUITE.md` §3.3, §3.3.2, §3.3.4, §7.4 · plan task
**T-D1** (KBR-51).

Tests run at ``l1`` (path-default per ``tests/layers.py`` and the
vertical-slice precedent). ``l3`` activation is T-K6's job.

**Driven, end-to-end.** A real ``BridgeFixture`` is started against a
recording upstream that serves Chat Completions replies. A minimal
synthetic request is posted to the Messages route, the captured upstream
body is read off the recorder, and the oracle compares the two
projections.

**Why a synthetic body and not a corpus entry.** T-D1's scope is "both
§3.3.2 assertions on one adapter" — one default-transport adapter,
end-to-end, with one falsification case. The full corpus-driven run
against every corpus entry is **T-D4**'s deliverable (§4 of the plan
table). The corpus entries that exercise the Anthropic reader's
``output_config``, ``context_management``, and ``metadata`` keys
(``plain_turn``, ``tools_declared``, ``tool_use_and_tool_result``)
already surface real bridge-side gaps that are not in scope for T-D1
to close — the KBR-style register rows for those drops belong in their
own PRs. Driving with a minimal synthetic body keeps T-D1's surface
focussed on the oracle's claim matching, the totality gate, and the
native passthrough key-order check — exactly the three obligations the
oracle owns.

**The CC-upstream adapter is the load-bearing choice.** The KBR-75 reader
fix is on the Chat Completions reader; the ``aiohttp`` transport with
``WireFormat.CHAT_COMPLETIONS`` binds the ``custom_openai`` adapter (a
default-transport one), so the upstream wire is Chat Completions and the
KBR-75 fix gets end-to-end coverage.
"""

from __future__ import annotations

import json

from harness import oracle
from harness import register as r
from harness.bridge import (
    BridgeFixture,
    InboundProtocol,
    inbound_path,
    minimal_inbound_body,
    transport,
)
from harness.contract import CapturedRequest, WireFormat

#: The marker the oracle should find verbatim in the upstream body.
#: A short string survives every layer of JSON encoding without quoting
#: artifacts.
_SENTINEL = "kbr51-oracle-driven"


#: Triggers the route arranges: a Chat Completions default run meets both.
#: ``NON_NATIVE_UPSTREAM_WIRE`` because the upstream wire is CC, not the
#: agent's Messages; ``PROFILE_SETS_MODEL`` because the bridge rewrites the
#: model on the way out.
_ROUTE_TRIGGERS: frozenset[r.Trigger] = frozenset(
    {r.Trigger.NON_NATIVE_UPSTREAM_WIRE, r.Trigger.PROFILE_SETS_MODEL}
)


class TestDrivenDefaultRun:
    """End-to-end driven run on a minimal synthetic Messages body."""

    async def test_drive_minimal_body_through_default_adapter_no_unclaimed_delta(self) -> None:
        """End-to-end: minimal Messages body → Messages route → CC recorder.

        Posts a minimal Messages body through the bridge against a Chat
        Completions recording upstream. Asserts the oracle finds zero
        unclaimed deltas. The KBR-75 CC-reader fix gets end-to-end
        coverage here because the upstream wire is Chat Completions.
        """
        body = minimal_inbound_body(InboundProtocol.MESSAGES, _SENTINEL)

        async with BridgeFixture(transport("aiohttp", WireFormat.CHAT_COMPLETIONS)) as fixture:
            status, _ = await fixture.post(inbound_path(InboundProtocol.MESSAGES), body)
            assert status == 200, "the recorder's minimal success reply must come back 200"
            captures = list(fixture.captures)
            assert len(captures) == 1, (
                "exactly one upstream request — a second capture would mean "
                "the empty-response retry ladder fired"
            )

            captured = captures[0]

            # The Messages adapter does not change the inbound scheme /
            # host / path / headers (those are the bridge's own), so the
            # oracle's inbound projection is built from what Claude Code
            # would have sent. We reconstruct a CapturedRequest that
            # matches the driven shape.
            inbound = CapturedRequest(
                method="POST",
                scheme="http",
                host="127.0.0.1",
                path=inbound_path(InboundProtocol.MESSAGES),
                query="",
                body=json.dumps(body).encode("utf-8"),
            )

            report = oracle.assert_no_unclaimed_mutation(
                inbound=inbound,
                inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                captured=captured,
                captured_format=WireFormat.CHAT_COMPLETIONS,
                register=r.REGISTER,
                triggers_met=_ROUTE_TRIGGERS,
            )
            assert report.deltas == (), (
                f"§3.3.2 assertion 1 failed: {report.deltas!r} are unclaimed "
                f"by any triggered row"
            )

    async def test_changed_model_fails_oracle_end_to_end(self) -> None:
        """Falsification (plan §1.4): the diff sees the model field through a real bridge.

        The bridge does not know the model was changed downstream — the
        captured upstream body has the bridge's profile model written
        over the agent's. The oracle, run against the real driven
        captures, must surface this as an unclaimed delta when
        ``PROFILE_SETS_MODEL`` is deliberately omitted from
        ``triggers_met`` so M1 does not claim the model.
        """
        body = minimal_inbound_body(InboundProtocol.MESSAGES, _SENTINEL)

        # Distinct profile model so the bridge actually rewrites the
        # inbound model — without a difference, the diff has nothing
        # to find and the falsification is unreachable.
        profile_model = "profile-model-not-inbound-model"
        async with BridgeFixture(
            transport("aiohttp", WireFormat.CHAT_COMPLETIONS),
            model=profile_model,
        ) as fixture:
            status, _ = await fixture.post(inbound_path(InboundProtocol.MESSAGES), body)
            assert status == 200
            captures = list(fixture.captures)
            assert len(captures) == 1
            captured = captures[0]

            inbound = CapturedRequest(
                method="POST",
                scheme="http",
                host="127.0.0.1",
                path=inbound_path(InboundProtocol.MESSAGES),
                query="",
                body=json.dumps(body).encode("utf-8"),
            )

            # PROFILE_SETS_MODEL deliberately omitted. The bridge rewrote
            # the model on the way out, so the oracle sees a delta at
            # envelope.model — M1's trigger is unmet, no row claims the
            # delta, the oracle fails. This proves the diff sees the
            # model field through a real bridge.
            try:
                oracle.assert_no_unclaimed_mutation(
                    inbound=inbound,
                    inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                    captured=captured,
                    captured_format=WireFormat.CHAT_COMPLETIONS,
                    register=r.REGISTER,
                    triggers_met=frozenset({r.Trigger.NON_NATIVE_UPSTREAM_WIRE}),
                )
            except oracle.UnclaimedMutationError as exc:
                assert "envelope.model" in exc.paths, (
                    f"the changed-model falsification should name envelope.model; "
                    f"got {exc.paths!r}"
                )
            else:
                raise AssertionError(
                    "the changed-model falsification must fail the oracle; "
                    "a green run here would mean the oracle does not see the model"
                )

    async def test_native_passthrough_rejects_reordered_body(self) -> None:
        """§4.3 C2: native passthrough's key-order check fails on reorder.

        With ``NON_NATIVE_UPSTREAM_WIRE`` *not* in ``triggers_met`` (the
        route is native passthrough) and the captured body's keys
        reordered relative to the inbound's, the oracle's C2 key-order
        check fires. The KBR-75 fix does not apply here — the test
        exercises the *third* obligation the oracle owns (§4.3 C2),
        separately from claim matching.
        """
        import pytest

        body = minimal_inbound_body(InboundProtocol.MESSAGES, _SENTINEL)

        async with BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)) as fixture:
            # ANTHROPIC_MESSAGES upstream wire — native passthrough
            # (the Messages adapter translates to the same wire shape
            # upstream). The bridge's adapter identity on this path
            # routes body bytes through unchanged.
            status, _ = await fixture.post(inbound_path(InboundProtocol.MESSAGES), body)
            assert status == 200
            captured = list(fixture.captures)[0]

            inbound = CapturedRequest(
                method="POST",
                scheme="http",
                host="127.0.0.1",
                path=inbound_path(InboundProtocol.MESSAGES),
                query="",
                body=json.dumps(body).encode("utf-8"),
            )

            # Reorder the captured body's top-level keys. JSON key order
            # is exactly what a provider fingerprints, so the C2 check
            # must fire even though the projections are equal.
            reordered_body_dict = json.loads(captured.body)
            reordered = b"{" + b",".join(
                b'"' + k.encode("utf-8") + b'":' + json.dumps(reordered_body_dict[k]).encode("utf-8")
                for k in reversed(list(reordered_body_dict.keys()))
            ) + b"}"
            reordered_capture = CapturedRequest(
                method=captured.method,
                scheme=captured.scheme,
                host=captured.host,
                path=captured.path,
                query=captured.query,
                headers=captured.headers,
                body=reordered,
            )

            with pytest.raises(oracle.NativePassthroughKeyOrderError):
                oracle.assert_no_unclaimed_mutation(
                    inbound=inbound,
                    inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                    captured=reordered_capture,
                    captured_format=WireFormat.ANTHROPIC_MESSAGES,
                    register=r.REGISTER,
                    triggers_met=frozenset(),  # native passthrough
                )

    async def test_native_passthrough_preserves_key_order_end_to_end(self) -> None:
        """§4.3 C2, positive: the real bridge's native passthrough preserves
        JSON key order end to end.

        The negative test above manufactures the ordering difference by
        hand on the captured body; this one asserts the *unmanufactured*
        case — the oracle accepts the body that actually reaches the
        recorder, on a native route, with the real bridge between. Without
        this test a bridge defect that reordered JSON keys would pass the
        negative test (which constructs its own difference) and the unit
        tests (which fabricate their own equality) — green either way,
        for the exact bridge behaviour this obligation names.

        Scope note: this is the ``custom_anthropic`` route, the one
        native-passthrough adapter T-D1's one-adapter scope covers. Other
        native adapters are T-D4/T-D9's matrix.
        """
        body = minimal_inbound_body(InboundProtocol.MESSAGES, _SENTINEL)

        async with BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)) as fixture:
            status, _ = await fixture.post(inbound_path(InboundProtocol.MESSAGES), body)
            assert status == 200
            captured = list(fixture.captures)[0]

            inbound = CapturedRequest(
                method="POST",
                scheme="http",
                host="127.0.0.1",
                path=inbound_path(InboundProtocol.MESSAGES),
                query="",
                body=json.dumps(body).encode("utf-8"),
            )

            report = oracle.assert_no_unclaimed_mutation(
                inbound=inbound,
                inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                captured=captured,
                captured_format=WireFormat.ANTHROPIC_MESSAGES,
                register=r.REGISTER,
                triggers_met=frozenset(),  # native passthrough
            )
            assert report.deltas == ()

    async def test_native_model_rewrite_does_not_trip_the_c2_check(self) -> None:
        """The M1 model rewrite on the native route preserves key order.

        The native branch runs ``_normalize_model`` whenever the profile
        pins a model, rewriting the inbound ``model`` to the profile's.
        That is a registered, permitted mutation (register row M1) and
        changes the bytes — but it preserves key order exactly, which is
        §4.3 C2's actual claim. An earlier byte-equality implementation
        false-failed this case; the key-order check accepts it.

        The test is informative: it asserts the captured body's model is
        the profile's (proving M1 fired), then runs the oracle and asserts
        it passes (proving C2 does not false-fail on registered
        mutations).
        """
        body = minimal_inbound_body(InboundProtocol.MESSAGES, _SENTINEL)
        # Distinct profile model so the bridge actually rewrites the
        # inbound model — without a difference, the diff has nothing to
        # find and the test is uninformative.
        profile_model = "profile-model-not-inbound-model"
        async with BridgeFixture(
            transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES),
            model=profile_model,
        ) as fixture:
            status, _ = await fixture.post(inbound_path(InboundProtocol.MESSAGES), body)
            assert status == 200
            captured = list(fixture.captures)[0]

            # M1 fired: the captured body carries the profile model.
            captured_body = json.loads(captured.body)
            assert captured_body["model"] == profile_model, (
                "the native route's M1 rewrite must have changed the model; "
                "without it the test would be uninformative"
            )

            inbound = CapturedRequest(
                method="POST",
                scheme="http",
                host="127.0.0.1",
                path=inbound_path(InboundProtocol.MESSAGES),
                query="",
                body=json.dumps(body).encode("utf-8"),
            )

            # C2 compares key order, not values — M1's model rewrite
            # preserves the key sequence, so the oracle's C2 check passes.
            # M1's own trigger (PROFILE_SETS_MODEL) is met — the profile
            # pins a model — so M1 claims the envelope.model delta and
            # assertion 1 passes too. The oracle call returning is the
            # proof; ``report.deltas`` carries the (claimed) envelope.model
            # delta because it is the structural diff output, not the
            # unclaimed list.
            oracle.assert_no_unclaimed_mutation(
                inbound=inbound,
                inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                captured=captured,
                captured_format=WireFormat.ANTHROPIC_MESSAGES,
                register=r.REGISTER,
                triggers_met=frozenset({r.Trigger.PROFILE_SETS_MODEL}),
            )


class TestDrivenScopeEnforcement:
    """The scope filter carries through a real bridge (KBR-307).

    The L1 fixtures in ``test_oracle.py`` prove the filter's logic; this
    driven case proves the wiring: a real bridge run, its real capture,
    and the adapter key derived from the harness's own transport binding
    travel together into the runtime oracle surface. The openai_subscription
    companion stays projection-level (``test_oracle.py``) — the curl_cffi
    transport's binding needs TLS certs and OAuth seeding, and the per-adapter
    rule is pinned at the cheapest layer that can prove it.

    **Scope of this coverage: the wiring, not the adapter matrix.** One
    adapter route (``custom_anthropic`` on the aiohttp Messages transport)
    is what a driven slice can prove in isolation; the full per-adapter
    matrix is the corpus runner's job (KBR-55/56/57) once ``provider_key``
    moves into the harness, per §3.3.4's parametrised-over-transport rule.
    """

    async def test_drive_anthropic_route_stream_flip_is_unclaimed(self) -> None:
        """End-to-end: a synthetic stream flip on the Anthropic-Messages route
        is unclaimed, because P17 is scope-gated out of this adapter.

        The bridge does not touch ``stream`` on this route (no register row
        claims it there), so the flip is manufactured on the captured body —
        the same idiom the C2 reorder test uses. ``ALWAYS`` is met so P17 is
        trigger-eligible: the only thing keeping P17 from claiming the delta
        is its ``scope=("openai_subscription",)``. The adapter key comes from
        the harness's own binding — ``transport.bind()[0].provider_type`` —
        not a hardcoded literal, so the case exercises the derivation the
        corpus runner will use.
        """
        import pytest

        body = minimal_inbound_body(InboundProtocol.MESSAGES, _SENTINEL, stream=True)

        async with BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)) as fixture:
            status, _ = await fixture.post(inbound_path(InboundProtocol.MESSAGES), body)
            assert status == 200
            captured = list(fixture.captures)[0]

            # Derive the adapter key from the binding — the seam the corpus
            # runner uses. On this transport it resolves to custom_anthropic.
            adapter, _config = fixture.transport.bind()
            provider_key = adapter.provider_type

            # Flip the captured body's stream. One value change; everything
            # else matches the inbound projection.
            captured_body = json.loads(captured.body)
            assert captured_body["stream"] is True, (
                "the bridge must not touch stream on this route — a flipped "
                "capture here would mean the test is not manufacturing the delta"
            )
            captured_body["stream"] = False
            flipped_capture = CapturedRequest(
                method=captured.method,
                scheme=captured.scheme,
                host=captured.host,
                path=captured.path,
                query=captured.query,
                headers=captured.headers,
                body=json.dumps(captured_body).encode("utf-8"),
            )

            inbound = CapturedRequest(
                method="POST",
                scheme="http",
                host="127.0.0.1",
                path=inbound_path(InboundProtocol.MESSAGES),
                query="",
                body=json.dumps(body).encode("utf-8"),
            )

            # ALWAYS met → P17 trigger-eligible; PROFILE_SETS_MODEL met → M1
            # claims the model delta (the harness profile pins a model). The
            # scope gate is the only thing left to explain the raise.
            with pytest.raises(oracle.UnclaimedMutationError) as info:
                oracle.assert_no_unclaimed_mutation(
                    inbound=inbound,
                    inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                    captured=flipped_capture,
                    captured_format=WireFormat.ANTHROPIC_MESSAGES,
                    register=r.REGISTER,
                    triggers_met=frozenset(
                        {r.Trigger.ALWAYS, r.Trigger.PROFILE_SETS_MODEL}
                    ),
                    provider_key=provider_key,
                )
            assert "envelope.stream" in info.value.paths, (
                f"the scope-gated stream flip must surface as unclaimed; "
                f"got {info.value.paths!r}"
            )
