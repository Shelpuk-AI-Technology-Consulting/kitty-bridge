"""Driven end-to-end tests for the transparency oracle's botocore slice.

`.system_design/TEST_SUITE.md` §3.3, §3.3.2, §3.3.4, §3.2.3, §7.2.4,
§7.4 · plan task **T-D6** (KBR-56).

Tests run at ``l1`` (path-default per ``tests/layers.py`` and the T-D1
precedent in ``test_oracle_driven.py``). ``l3`` activation is T-K6's
job — the §3.4 table names this surface L3 for the same reason it does
the aiohttp recorder surface.

**Driven, end-to-end on the botocore transport.** A real ``BridgeFixture``
is started against a ``BedrockRecordingUpstream`` (T-B3, KBR-42); a
minimal Anthropic Messages body is posted to the Messages route; the
captured wire body is read off the recorder; and the oracle compares
the two projections. The Bedrock adapter is a CC-wire custom-transport
adapter (``use_custom_transport = True``, §3.3.4), so the default
aiohttp recorder never sees its body — this slice is what closes the
oracle's "every provider" claim for ``bedrock``.

**The §3.2.3 boundary, stated precisely.** ``modelId`` is a **URI
parameter** on the Converse wire (``POST /model/{modelId}/converse``),
not a body key — the harness reader reads
:attr:`envelope.model` from ``CapturedRequest.path``
(``reader_bedrock_converse.py``, PUBLISHED_TOP_LEVEL_KEYS) — and
``translate_to_upstream`` never emits ``stream`` into the body
(``bedrock.py:621-624``; the transport's ``stream`` pop is defensive).
The captured body therefore carries **neither** key, and the test
asserts that as a positive wire-shape observation documenting the
boundary the recorder sees — not as a P18 falsification. A P18
pop-regression cannot be observed through this slice at all: the
unpopped ``modelId`` collides with boto3's
``converse(modelId=…, **body)`` call as a duplicate keyword argument
(``bedrock.py:659``), the bridge 500s before the recorder captures
anything, and the drive's own ``status == 200`` / one-capture
assertions are what would surface it.

**The falsification is T-D1's changed-model pattern, adapted.** A
distinct profile model makes M1's rewrite observable — the inbound
body asks for one model, the profile pins another, and the captured
URI carries the profile's. With ``PROFILE_SETS_MODEL`` in
``triggers_met`` the resulting ``envelope.model`` delta is claimed and
the oracle is green with a non-empty ``deltas`` (the positive test
asserts exactly that, so the claim machinery is exercised, not
trivially green). With the trigger deliberately omitted the delta is
unclaimed and the oracle must raise :class:`UnclaimedMutationError`
naming ``envelope.model`` — the falsification test asserts exactly
that, and its ``else`` branch raises so a silently-green run cannot
pass.

**Why a minimal synthetic body and not a corpus entry.** T-D1's
docstring names the rationale: the full corpus-driven run against
every entry is T-D4's deliverable. The conditional rows reachable on
the bedrock route are out of scope for the same reason — P33
(``BEDROCK_FORCES_AUTO_TOOL_CHOICE``, bedrock-scoped) and P34
(Messages-translator-scoped, reachable on bedrock) owe their trigger
cases and §3.3.2 assertion-2 complements to T-D5's corpus entries.
The minimal body carries no ``tools``, no ``tool_choice`` and no
``disable_parallel_tool_use``, so neither conditional row's trigger is
met on this slice.
"""

from __future__ import annotations

import json

from harness import oracle
from harness import register as r
from harness.botocore import BotocoreTransport  # noqa: F401  -- registers the "botocore" transport
from harness.bridge import (
    BridgeFixture,
    InboundProtocol,
    inbound_path,
    minimal_inbound_body,
    transport,
)
from harness.contract import CapturedRequest, WireFormat

#: The marker the oracle should find verbatim in the upstream body. A short
#: string survives every layer of JSON encoding without quoting artifacts.
_SENTINEL = "kbr56-oracle-botocore"

#: A profile model distinct from the inbound body's, so M1's rewrite is
#: observable: the captured URI carries the profile's model, the delta at
#: ``envelope.model`` is real, and the claim machinery is exercised rather
#: than trivially green. Same lever as T-D1's falsification test.
_PROFILE_MODEL = "profile-model-not-inbound-model"

#: Triggers the route arranges. ``NON_NATIVE_UPSTREAM_WIRE`` because the
#: upstream wire is Bedrock Converse, not the agent's Messages;
#: ``PROFILE_SETS_MODEL`` because the fixture pins a profile model, so the
#: bridge rewrites ``envelope.model`` and M1 fires for real.
_ROUTE_TRIGGERS: frozenset[r.Trigger] = frozenset(
    {r.Trigger.NON_NATIVE_UPSTREAM_WIRE, r.Trigger.PROFILE_SETS_MODEL}
)


class TestBotocoreOracleSlice:
    """End-to-end driven run on a minimal synthetic Messages body through ``bedrock``."""

    async def test_drive_minimal_body_through_bedrock_no_unclaimed_delta(self) -> None:
        """Positive: minimal Messages body → Messages route → Converse recorder.

        Posts a minimal Messages body (asking for one model) through the
        bridge against a profile that pins a different model, and asserts:

        1. The recorder saw exactly one upstream request with a 200.
        2. **Wire-shape observation at the §3.2.3 boundary.** The captured
           Converse body carries neither ``modelId`` nor ``stream`` —
           ``modelId`` lives in the URI (``POST /model/{id}/converse``),
           read by the harness reader from the path, and ``stream`` is
           never emitted into the body.
        3. The oracle's §3.3.2 assertion 1 holds with the claim machinery
           genuinely exercised: the model rewrite produces a real
           ``envelope.model`` delta, M1 (``PROFILE_SETS_MODEL``) claims
           it, and the run is green. P18 is not active in this run — its
           ``ALWAYS`` trigger is not in ``triggers_met``, and must not
           be: the oracle's claim matching activates a row only when its
           trigger is in the set (``oracle.py``, ``_claim_matching``), so
           including ``ALWAYS`` would let P18 claim ``envelope.model``
           and disarm the falsification test's trigger omission.
        """
        body = minimal_inbound_body(InboundProtocol.MESSAGES, _SENTINEL)

        async with BridgeFixture(
            transport("botocore", WireFormat.BEDROCK_CONVERSE),
            model=_PROFILE_MODEL,
        ) as fixture:
            status, _ = await fixture.post(
                inbound_path(InboundProtocol.MESSAGES), body
            )
            assert status == 200, (
                "the recorder's minimal success reply must come back 200"
            )
            captures = list(fixture.captures)
            assert len(captures) == 1, (
                "exactly one upstream request — a second capture would mean "
                "the empty-response retry ladder fired"
            )

            captured = captures[0]

            # §3.2.3 boundary observation (see module docstring): the
            # captured body carries neither key. This documents the
            # boundary the recorder sees; it is not the P18 falsification
            # (see the falsification test and the module docstring for
            # why that regression is unobservable through this slice).
            captured_body = json.loads(captured.body)
            assert "modelId" not in captured_body, (
                "modelId is a URI parameter on the Converse wire, never a "
                f"body key; captured body carried {captured_body.get('modelId')!r}"
            )
            assert "stream" not in captured_body, (
                "translate_to_upstream never emits stream into the Converse "
                f"body; captured body carried {captured_body.get('stream')!r}"
            )

            # Reconstruct the inbound CapturedRequest the oracle reads
            # on the Messages side — the bridge does not change scheme /
            # host / path / headers for the Messages adapter.
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
                captured_format=WireFormat.BEDROCK_CONVERSE,
                register=r.REGISTER,
                triggers_met=_ROUTE_TRIGGERS,
            )
            assert report.deltas == ("envelope.model",), (
                f"the profile-model rewrite must produce exactly the "
                f"envelope.model delta; got {report.deltas!r}"
            )

    async def test_changed_model_fails_oracle_end_to_end(self) -> None:
        """Falsification (plan §1.4): the diff sees the model field through bedrock.

        Same drive as the positive — the captured URI carries the
        profile's model over the inbound body's — but
        ``PROFILE_SETS_MODEL`` is deliberately omitted from
        ``triggers_met`` so M1 does not claim the ``envelope.model``
        delta. The oracle must raise :class:`UnclaimedMutationError`
        naming ``envelope.model``; the ``else`` branch raises so a
        silently-green run cannot pass. This proves the oracle's diff
        sees the model field through the bedrock slice — §10 names "a
        projection that could not see the model name" as one of the
        four harnesses that would have passed while proving nothing.
        """
        body = minimal_inbound_body(InboundProtocol.MESSAGES, _SENTINEL)

        async with BridgeFixture(
            transport("botocore", WireFormat.BEDROCK_CONVERSE),
            model=_PROFILE_MODEL,
        ) as fixture:
            status, _ = await fixture.post(
                inbound_path(InboundProtocol.MESSAGES), body
            )
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

            # PROFILE_SETS_MODEL deliberately omitted: M1 does not claim
            # the envelope.model delta, no other triggered row does, and
            # the oracle must fail naming it.
            try:
                oracle.assert_no_unclaimed_mutation(
                    inbound=inbound,
                    inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                    captured=captured,
                    captured_format=WireFormat.BEDROCK_CONVERSE,
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
                    "a green run here would mean the oracle does not see the "
                    "model through the bedrock slice"
                )
