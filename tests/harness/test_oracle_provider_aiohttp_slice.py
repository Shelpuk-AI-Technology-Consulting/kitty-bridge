"""Corpus-driven oracle slice on the provider-aiohttp transport.

`.system_design/TEST_SUITE.md` §3.3.2, §3.3.4, §3.3.5, §7.2.2, §7.4 · plan
task **T-D7** (KBR-57). Requirements:
`.requirements/20260924T083854Z_td7_provider_aiohttp_oracle_slice/REQUIREMENTS.md`.

Drives the bridge end-to-end against the transparency oracle on **every**
inbound Anthropic-Messages corpus entry, using the `provider_aiohttp`
transport — the T-B1 recorder whose adapter is `ollama_cloud`, one of the
three `use_custom_transport = True` adapters the default aiohttp recording
upstream never sees (§3.3.4). With this slice the oracle's "every provider"
claim is no longer false for `ollama_cloud`; the other two halves are T-D5
(curl_cffi / openai_subscription) and T-D6 (botocore / bedrock, landed).

**All four oracle obligations are active** per entry: §3.3.1 totality,
§3.3.2 assertions 1 + 2, §3.3.5 routing. `provider_key` is **derived from
the binding** (`fixture.transport.bind()[0].provider_type`, resolving to
`"ollama_cloud"`), never hardcoded — the KBR-307 derivation seam. `Trigger.
ALWAYS` stays in `triggers_met` so register row **P19** is trigger-eligible
and the scope filter is the discriminating gate, not one masked by the
trigger filter (KBR-307 obligation 2 on this ticket).

**The §3.2.3 capture boundary, and P19's observation posture.** The recorder
captures *after* the mutation P19 names: `_ollama_body` runs inside
`make_request`/`stream_request`, before `session.post` — the same boundary
T-D6 records for P18. On a consistent drive the overwrite is
**value-preserving**: the transport writes `stream` from its own endpoint-mode
decision, and the inbound Messages body either carries the same value or none
(the readers' shared normalisation reads an absent request flag as its wire
default), so the projections agree at `envelope.stream` and no delta appears.
That is a wire fact, not a test gap — the same shape T-D6 recorded for P18's
pops. The row is therefore **armed, not observed by absence**: every driven
run keeps P19 eligible (ALWAYS met, in scope), so a transport whose
`envelope.stream` decision ever diverges observably turns red, and
`TestP19ClaimMachinery` proves the claim machinery genuinely fires by
flipping the captured value (the one mutation P19's site actually controls).

**Inherited T-D4 contracts** (`.system_design/steps/t_d4_default_transport_
slice.md`, acked on KBR-57): the routing derivation is the **literal** KBR-52
shape restricted to parameter-free `base_url`s (`ollama_cloud` composes
`{base}/api/chat` with no query; KBR-143's merge rule is not exercised), the
authority-and-scheme rewrite reads the **recorder's** base URL
(`urlsplit(fixture.transport.recorder.base_url)` — the bridge and the
recorder bind different ephemeral ports), the skip table is the module-local
`_CORPUS_SKIP_TABLE`, and `_SENTINEL_ROUTE_PATH` is pinned against the
adapter's published shape.

**The OAuth login leg** (`TestOAuthLoginLegThroughTheSlice`) is the slice's
other half per the ticket's Done-when. A form-encoded token grant is not an
LLM request and has no reader (§7.5's decided question), so the coverage is
recorder-level: both products captured, in order, on **one recorder
instance** — the §7.2.2 dual-product load the T-B1 tests (which cover the leg
in isolation) never exercise.

**Layer.** No `pytestmark`; harness tests default to `l1` per
`tests/layers.py` and the T-D1/T-D2/T-D4/T-D6 precedent. §3.4 calls this
surface L3 and T-K6 owns `l3` activation. The module binds a real
`BridgeServer` and the recorder (two sockets per run), so it is listed in
`tests/socket_binding_l1_modules.py` (§8.2).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
import pytest

from harness import oracle
from harness import register as r
from harness.bridge import BridgeFixture, InboundProtocol, inbound_path, transport
from harness.contract import CapturedRequest, WireFormat
from harness.corpus import load_corpus
from harness.provider_aiohttp import oauth_token_endpoint
from harness.provider_recorder import OAUTH_TOKEN_SUFFIX, OLLAMA_CHAT_SUFFIX
from kitty.auth import openai_oauth
from kitty.providers.ollama_cloud import OllamaCloudAdapter

#: The path the ``ollama_cloud`` adapter publishes as its endpoint
#: (``{base}/api/chat``, no query — the binding's ``base_url`` is
#: parameter-free, so the KBR-143 query-merge rule is not exercised). Pinned
#: so a future adapter rename fails the test loudly with both values in the
#: message (T-D4 contract 3).
_SENTINEL_ROUTE_PATH = "/api/chat"

#: A wrong path used to falsify the routing assertion. The body obligations
#: pass on the real route, so a RoutingMismatchError raised here proves the
#: routing assertion bites through the driven slice (T-D2's
#: reroute-with-byte-identical-body shape, applied to the L3-driven surface).
_WRONG_ROUTE_PATH = "/_intentionally_wrong_path_for_falsification"

#: The default harness profile's derived compaction budget, in chars —
#: `get_model_context_tokens("ollama_cloud", "harness-model", None)` resolves
#: to `DEFAULT_CONTEXT_TOKENS = 200_000` (the fallback in
#: `src/kitty/providers/model_context.py:15`) × `TOKENS_TO_CHARS_FACTOR = 4`.
#: A literal, not an import: the driver is a judge module and §3.3.1's
#: independent-oracle rule forbids importing `src/kitty` (same decision T-D4
#: and T-D2 made for their derivations).
_DEFAULT_PROFILE_BUDGET_CHARS = 800_000

#: Corpus entries whose empirical pass surfaced a real I1 finding, framing
#: gap, or calibration gap. Each value is the pytest.skip message; each
#: message names the unclaimed-delta paths (or the framing/calibration cause)
#: and the owner-tracking marker — a K57-F finding id for this slice's own
#: findings, T-D4's F3.d marker where the root cause is shared, or a framing/
#: calibration note that carries none (T-D4's pattern).
#:
#: Populated from the probe passes of 2026-09-24 (`.scratch/probe_corpus_
#: provider_aiohttp.py`, `-p19_claim.py`, `-full_deltas.py`). See
#: `REQUIREMENTS.md` §4 AC2 for the per-row rationale.
_CORPUS_SKIP_TABLE: dict[str, str] = {
    # K57-F1: the CC→Ollama translation drops the top-level control fields the
    # Messages body carries (context_management, output_config, thinking) and
    # merges the three system blocks into one whose text differs; the
    # multi-part user turn is flattened (`_flatten_content` joins with "\n"),
    # so parts[0].text differs and parts[1] is gone. Eight paths, probed
    # identical on both entries.
    "plain_turn": (
        "K57-F1 envelope.extra[context_management] + envelope.extra[output_config] "
        "+ envelope.extra[thinking] + conversation.system[0].text + "
        "conversation.system[1] + conversation.system[2] + "
        "conversation.turns[0].parts[0].text + conversation.turns[0].parts[1] "
        "unclaimed on the ollama_cloud route; new finding, KBR-57 scope addition"
    ),
    "effort_configured": (
        "K57-F1 envelope.extra[context_management] + envelope.extra[output_config] "
        "+ envelope.extra[thinking] + conversation.system[0].text + "
        "conversation.system[1] + conversation.system[2] + "
        "conversation.turns[0].parts[0].text + conversation.turns[0].parts[1] "
        "unclaimed on the ollama_cloud route; new finding, KBR-57 scope addition"
    ),
    # K57-F2 (reader gap): the adapter emits id/type on request-side
    # tool_calls; the published Ollama ChatRequest declares `function` only on
    # that sub-shape, so reader_ollama (T-A6, written against the published
    # schema) has no slot and the totality gate fails before any delta can be
    # classified. Go's decoder tolerates the extra keys upstream; whether the
    # reader learns the keys or the adapter stops sending them is the owner's.
    "tool_use_and_tool_result": (
        "K57-F2 reader gap: messages[N].tool_calls[*].id + "
        "messages[N].tool_calls[*].type residual — the adapter emits id/type, "
        "the published Ollama ChatRequest declares function only, reader_ollama "
        "has no slot (totality gate); new finding, KBR-57 scope addition"
    ),
    "compaction_budget_over": (
        "K57-F2 reader gap: messages[N].tool_calls[*].id + "
        "messages[N].tool_calls[*].type residual — the adapter emits id/type, "
        "the published Ollama ChatRequest declares function only, reader_ollama "
        "has no slot (totality gate; fires before M5's deltas can be claimed); "
        "new finding, KBR-57 scope addition"
    ),
    # K57-F3: the whole tool_result turn is dropped from the translated
    # request — T-D4's F3.d shape (likely M7 pairing-validation territory,
    # no register row), new on this route.
    "tool_result_under_limit": (
        "K57-F3 conversation.turns[2] dropped on the ollama_cloud route from a "
        "Messages body carrying a 50 000-char tool_result (F3.d shape, T-D4); "
        "KBR-57 scope addition"
    ),
    "tool_result_over_limit": (
        "K57-F3 conversation.turns[2] dropped on the ollama_cloud route from a "
        "Messages body carrying a 50 001-char tool_result (F3.d shape, T-D4); "
        "KBR-57 scope addition"
    ),
    # Framing gap, not a register gap: the entry's Messages body has role
    # 'system' inside messages (forbidden by the Anthropic Messages format —
    # system prompts are a top-level field). The Messages adapter rejects with
    # UnreadableBodyError before any capture; not a fidelity oracle input.
    "tools_declared": (
        "framing gap: entry body has role 'system' inside messages; Messages "
        "adapter rejects with UnreadableBodyError before capture"
    ),
    # Framing gap: the bridge returns 400 before any capture. Not a fidelity
    # oracle input.
    "system_prompt_over_window_compacts_normally": (
        "framing gap: bridge returns 400 before capture; entry body is rejected "
        "by the Messages adapter at ingress"
    ),
    # Calibration gap: the entry was calibrated to the 2.8 M-char static
    # threshold but the default profile's derived budget is 800 000 chars, so
    # M5 fires (body over budget); M5's pruned turn texts sit beneath M3's
    # coarse parts[*] anchor, and because M5's anchor is a proper prefix of
    # M3's, M5 does not specifically claim for M3 — assertion 2 flags M3
    # (ConditionalRowFiredWithoutTriggerError). Same shape as T-D4's
    # calibration gap on this entry.
    "compaction_budget_under": (
        "calibration gap: entry calibrated to 2.8 M-char static threshold; the "
        "default profile's 800 000-char budget makes M5 fire (body over budget) "
        "and M5's pruned turn texts sit beneath M3's coarse parts[*] anchor "
        "(M5's anchor is a proper prefix of M3's, so M5 does not specifically "
        "claim for M3) — assertion 2 flags M3; same shape as T-D4's"
    ),
    # KBR-315 / T-D4 F3.e: the typed-tool `type` drop. The Messages body carries
    # an Anthropic-defined (typed) tool declaration (`web_search_20250305`),
    # the CC→Ollama translation drops the `type` member (the Ollama
    # ChatRequest declares `function` only on the tool sub-shape), no register
    # row claims this drop on the `ollama_cloud` route. A register row for
    # the typed-tool `type` drop is the F3.e sibling ticket — same root cause
    # KBR-55 surfaced on the T-D4 judge, scoped to this route's adapter.
    "p35_tool_choice_omitted_forcing_anthropic_tool_trigger": (
        "F3.e conversation.tools[web_search].type dropped on the ollama_cloud "
        "route from a body carrying an Anthropic-defined (typed) tool "
        "declaration; no register row claims the typed-tool `type` drop — "
        "KBR-315 scope addition"
    ),
}


#: Per-entry expected-delta tuples for entries whose triggers legitimately
#: fire on the `ollama_cloud` route. Clean entries (everything not named
#: here) produce only M1's `PROFILE_SETS_MODEL` rewrite at `envelope.model`.
#: KBR-315 / KBR-55 scope addition: the same three entries T-D4 fixed
#: (`a8f8ca3`) — M28's empty-stop omission, P35's tool_choice omission —
#: plus P34's `parallel_tool_calls` omission on the complement body — all
#: fire here too because the bridge-level carry in
#: `MessagesTranslator.translate_request` + `carry_tool_choice_and_metadata`
#: runs on every translated adapter. The tuples below are the empirical
#: probe output from the merge of the KBR-55 entries + the `ollama_cloud`
#: transport binding on 2026-09-24 (the test failure surface); the
#: comment in `test_drive_entry_through_ollama_cloud` records the probe.
_EXPECTED_DELTAS: dict[str, tuple[str, ...]] = {
    # M28: empty `stop_sequences` omitted — `bridge/messages/translator.py`
    # joins them into nothing (no empty list reaches the CC wire).
    "m28_empty_stop_sequences_trigger": (
        "envelope.model",
        "conversation.sampling[stop]",
    ),
    # P35: `tool_choice` omitted when no tools are declared — the
    # `carry_tool_choice_and_metadata` early-return drops it (CC rejects
    # tool_choice beside no tools; the carry is intentionally silent).
    "p35_tool_choice_omitted_no_tools_trigger": (
        "envelope.model",
        "envelope.extra[tool_choice]",
    ),
    # P35 complement: same omission on a body that DOES declare tools
    # (the trigger fires when `tool_choice: false` is absent — a legal
    # Anthropic body, omitted per the legal-but-unsupported posture).
    "p35_tool_choice_omitted_complement": (
        "envelope.model",
        "envelope.extra[tool_choice]",
    ),
    # P34 trigger: this entry's body does NOT have `disable_parallel_tool_use`
    # set to `False` (the trigger for P34), so P34 is dormant here — only P35
    # fires (the body has tools + no tool_choice → P35 omits it).
    "p34_parallel_false_omitted_trigger": (
        "envelope.model",
        "envelope.extra[tool_choice]",
    ),
    # P34 complement: body has `disable_parallel_tool_use: false`, so the
    # carry omits `parallel_tool_calls: False` AND the tool_choice omission
    # fires — both deltas legitimately claimed.
    "p34_parallel_false_omitted_complement": (
        "envelope.model",
        "envelope.extra[parallel_tool_calls]",
        "envelope.extra[tool_choice]",
    ),
}


def _size_ordered_am_entries() -> list:
    """Return the inbound-Anthropic-Messages corpus entries, size-ordered.

    Called once at module collection time (the parametrize decorator below
    evaluates immediately). Loads the corpus and sorts by body length then id
    so the test collection order and failure messages are stable across runs.

    Returns:
        The corpus entries the slice drives, smallest body first.
    """
    entries = [
        e for e in load_corpus(Path(__file__).parent.parent / "corpus")
        if e.wire_format == WireFormat.ANTHROPIC_MESSAGES
    ]
    return sorted(entries, key=lambda e: (len(e.request.body), e.id))


def _entry_id(entry) -> str:
    """Return the pytest param id for one corpus entry.

    Args:
        entry: The corpus entry being parametrised.

    Returns:
        The entry id, prefixed with its body size so a failure message shows
        which shape failed at a glance.
    """
    return f"{len(entry.request.body):>10}B-{entry.id}"


def _expected_route(fixture: BridgeFixture) -> oracle.ExpectedRoute:
    """Return the §3.3.5 routing expectation for this fixture.

    Derived from `ollama_cloud`'s published URL shape (`/api/chat`, no query),
    with the recorder's scheme and host rewritten in. The rewrite reads the
    **recorder's** base URL, not the bridge's: the two bind different
    ephemeral ports, so an expectation built from `fixture.base_url` would
    mismatch `route.host` on every capture (T-D4 contract 1; §3.3.5's T-W4
    scope addition).

    Args:
        fixture: A started `BridgeFixture`; the recorder's base URL is read
            here.

    Returns:
        An `ExpectedRoute` whose path and query are the literal derivation
        and whose scheme and host are the recorder's.
    """
    rec = urlsplit(fixture.transport.recorder.base_url)
    return oracle.ExpectedRoute(
        method="POST",
        scheme=rec.scheme,
        host=rec.netloc,
        path=_SENTINEL_ROUTE_PATH,
        query="",
    )


def _triggers_met(entry) -> frozenset[r.Trigger]:
    """Return the trigger vocabulary for one corpus entry.

    Combines the entry's declared `triggers_met` (REQUEST-only triggers; the
    corpus loader refuses non-REQUEST triggers in both lists per KBR-186's
    classification) with the route's `NON_NATIVE_UPSTREAM_WIRE` trigger, the
    profile's `PROFILE_SETS_MODEL` trigger (both declared at the call site per
    the corpus README's PROFILE-decided-trigger rule) and **`ALWAYS`** — P19
    is ALWAYS-triggered, and KBR-307 obligation 2 on this ticket requires it
    trigger-eligible so the scope filter is the discriminating gate on this
    slice's observation target. For entries whose body exceeds the default
    profile's derived budget, `OVER_COMPACTION_BUDGET` is added (M5's trigger
    is PROFILE-decided and declared at the call site that resolves the
    profile).

    Args:
        entry: The corpus entry.

    Returns:
        The triggers the entry meets.
    """
    triggers = entry.triggers_met | {
        r.Trigger.NON_NATIVE_UPSTREAM_WIRE,
        r.Trigger.PROFILE_SETS_MODEL,
        r.Trigger.ALWAYS,
    }
    # The bridge measures the CC-converted messages (`_safe_size` inside
    # `_compact_messages`), which differs from the committed Anthropic shape
    # by a constant (~92 chars for this layout, per tests/corpus/README.md).
    # The raw body length is the approximation; no corpus entry sits within
    # that margin of the 800 K boundary (the closest are the ~605 KB m6 pair,
    # 24 % under).
    if len(entry.request.body) > _DEFAULT_PROFILE_BUDGET_CHARS:
        triggers = triggers | {r.Trigger.OVER_COMPACTION_BUDGET}
    return triggers


def _inbound_capture(entry) -> CapturedRequest:
    """Build the oracle's inbound `CapturedRequest` for `entry`.

    The host is rewritten to `127.0.0.1` (the bridge's loopback origin)
    because the entry's committed `host` is `api.anthropic.com` — the host the
    agent reached in the original capture. The oracle's body projection does
    not read the host; the routing assertion compares the *captured* route
    against the derived expectation, so the inbound host is irrelevant for
    both assertions.

    Args:
        entry: The corpus entry.

    Returns:
        The oracle-shaped inbound capture.
    """
    return CapturedRequest(
        method=entry.request.method,
        scheme=entry.request.scheme,
        host="127.0.0.1",
        path=entry.request.path,
        query=entry.request.query,
        headers=entry.request.headers,
        body=entry.request.body,
    )


def _flip_captured_stream_true(captured: CapturedRequest) -> CapturedRequest:
    """Return `captured` with its `stream` body value overwritten to `true`.

    This is the one mutation P19's site controls: `_ollama_body` writes
    `stream` from the transport's endpoint-mode decision, after
    `translate_to_upstream` has already copied any request-level value. The
    claim-machinery test uses the flip to give P19 a real delta to claim on a
    drive where the honest bytes produce none (see the module docstring's
    "P19 observation posture").

    Args:
        captured: The captured upstream request, body JSON-decoded and
            re-encoded with `stream: true`.

    Returns:
        The replaced capture, everything else byte-identical.
    """
    body = json.loads(captured.body)
    body["stream"] = True
    return replace(captured, body=json.dumps(body).encode("utf-8"))


def _provider_key(fixture: BridgeFixture) -> str:
    """Return the KBR-307 ``provider_key`` for the adapter the fixture bound.

    The transport caches its adapter in ``_adapter``, so every ``bind()`` call
    inside a single fixture returns the same instance — but the call shape
    repeats and is easy to drift; this helper is the one place the derivation
    lives, so a future change to how the runtime oracle's notion of "live on
    this adapter" is computed lands here once.

    Args:
        fixture: A started `BridgeFixture` whose transport is bound to a
            recorder.

    Returns:
        The bound adapter's ``provider_type`` (resolves to ``"ollama_cloud"``
        on this transport; the KBR-307 derivation seam).
    """
    return fixture.transport.bind()[0].provider_type


class TestCorpusDrivenProviderAiohttpSlice:
    """End-to-end oracle run on every inbound-Anthropic-Messages corpus entry.

    Parametrised over the corpus entries in size order (smallest first).
    Entries in `_CORPUS_SKIP_TABLE` are `pytest.skip`'d with their finding's
    message. Each entry that runs the oracle asserts:

    * `status == 200` and `len(captures) == 1` (no fired retry ladder).
    * `report.deltas == ("envelope.model",)` — the only permitted delta is
      M1's `PROFILE_SETS_MODEL` rewrite of `envelope.model`. The probe
      (2026-09-24) measured exactly this tuple on the five clean entries;
      P19's `envelope.stream` is absent because the transport's overwrite is
      value-preserving on a consistent drive (module docstring).
    * `expected_route` matches the captured route after the recorder-authority
      rewrite (§3.3.5 routing assertion).
    * `provider_key` is the binding's `provider_type` — the KBR-307 seam, so
      the scope filter judges the adapter the fixture actually bound.
    """

    @pytest.mark.parametrize(
        "entry",
        _size_ordered_am_entries(),
        ids=_entry_id,
    )
    async def test_drive_entry_through_ollama_cloud(self, entry) -> None:
        """Drive one corpus entry end to end; the oracle must accept it.

        Entries in `_CORPUS_SKIP_TABLE` skip with their rationale. Clean
        entries assert `report.deltas == ("envelope.model",)` (M1's
        PROFILE_SETS_MODEL rewrite is the only permitted delta).

        Args:
            entry: The corpus entry being driven.
        """
        skip_reason = _CORPUS_SKIP_TABLE.get(entry.id)
        if skip_reason is not None:
            pytest.skip(skip_reason)

        async with BridgeFixture(
            transport("provider_aiohttp", WireFormat.OLLAMA_CHAT)
        ) as fixture:
            provider_key = _provider_key(fixture)
            status, _text = await fixture.post(
                inbound_path(InboundProtocol.MESSAGES),
                json.loads(entry.request.body),
            )
            assert status == 200, (
                f"the recorder's minimal success reply must come back 200; "
                f"got {status}"
            )
            captures = list(fixture.captures)
            assert len(captures) == 1, (
                f"exactly one upstream request — a second capture would mean "
                f"the empty-response retry ladder fired (got {len(captures)})"
            )
            captured = captures[0]

            report = oracle.assert_no_unclaimed_mutation(
                inbound=_inbound_capture(entry),
                inbound_format=entry.wire_format,
                captured=captured,
                captured_format=WireFormat.OLLAMA_CHAT,
                register=r.REGISTER,
                triggers_met=_triggers_met(entry),
                expected_route=_expected_route(fixture),
                provider_key=provider_key,
            )
            assert report.deltas == _EXPECTED_DELTAS.get(
                entry.id, ("envelope.model",)
            ), (
                f"entry {entry.id!r}: expected the per-entry map's tuple "
                f"(KBR-315 scope addition) or M1's (envelope.model,) only; "
                f"got {report.deltas!r}"
            )


class TestRoutingFalsification:
    """§1.4 harness rule: the routing assertion must bite through the driven slice.

    The oracle's body obligations pass on the real captured body; a
    sentinel-wrong expected path must raise `RoutingMismatchError` naming
    `route.path`. The ordering (§3.3.5: routing runs last so the louder body
    diagnosis surfaces first) means the body obligations have already passed
    by the time routing raises — the routing error is the only thing that can
    fire. T-D2's reroute-with-byte-identical-body shape, applied to the
    L3-driven surface.
    """

    async def test_routing_mismatch_raises_on_wrong_expected_path(self) -> None:
        """Drive a small clean body; reroute to a sentinel path; oracle fails on routing."""
        # The smallest clean entry — the first not in the skip table — drives
        # the falsification. Selecting by "not skipped" rather than by id
        # keeps the body-obligations-pass-first ordering intact if a corpus
        # entry is renamed, removed, or moved into the skip table: the choice
        # is recomputed from the same table the driven slice consults, so the
        # two can never disagree (T-D4's shape).
        clean = next(
            e for e in _size_ordered_am_entries() if e.id not in _CORPUS_SKIP_TABLE
        )

        async with BridgeFixture(
            transport("provider_aiohttp", WireFormat.OLLAMA_CHAT)
        ) as fixture:
            rec = urlsplit(fixture.transport.recorder.base_url)
            status, _text = await fixture.post(
                inbound_path(InboundProtocol.MESSAGES),
                json.loads(clean.request.body),
            )
            assert status == 200
            captured = list(fixture.captures)[0]

            # 1. The real route passes every obligation.
            real_route = oracle.ExpectedRoute(
                method="POST",
                scheme=rec.scheme,
                host=rec.netloc,
                path=_SENTINEL_ROUTE_PATH,
                query="",
            )
            oracle.assert_no_unclaimed_mutation(
                inbound=_inbound_capture(clean),
                inbound_format=clean.wire_format,
                captured=captured,
                captured_format=WireFormat.OLLAMA_CHAT,
                register=r.REGISTER,
                triggers_met=_triggers_met(clean),
                expected_route=real_route,
                provider_key=_provider_key(fixture),
            )

            # 2. The same captured body on a sentinel-wrong path raises
            #    RoutingMismatchError. The body obligations have already
            #    passed on the real route above, so the routing error is the
            #    only thing that can fire (§3.3.5's ordering).
            wrong_route = oracle.ExpectedRoute(
                method="POST",
                scheme=rec.scheme,
                host=rec.netloc,
                path=_WRONG_ROUTE_PATH,
                query="",
            )
            with pytest.raises(oracle.RoutingMismatchError) as exc_info:
                oracle.assert_no_unclaimed_mutation(
                    inbound=_inbound_capture(clean),
                    inbound_format=clean.wire_format,
                    captured=captured,
                    captured_format=WireFormat.OLLAMA_CHAT,
                    register=r.REGISTER,
                    triggers_met=_triggers_met(clean),
                    expected_route=wrong_route,
                    provider_key=_provider_key(fixture),
                )
            assert "route.path" in exc_info.value.paths, (
                f"the routing falsification should name route.path; got "
                f"{exc_info.value.paths!r}"
            )


class TestExpectedRoutePathIsLiteralAndPinned:
    """The expected-path literal must match the adapter's published shape.

    A future adapter rename (or a derivation that drifts to a hardcoded wrong
    path) fails this test loudly with both values in the message — the kind of
    self-consistent green the design §1.4 forbids (T-D4 contract 3).
    """

    def test_sentinel_matches_adapter_upstream_path(self) -> None:
        """`_SENTINEL_ROUTE_PATH` equals `OllamaCloudAdapter().get_upstream_path(...)`."""
        actual = OllamaCloudAdapter().get_upstream_path("harness-model")
        assert actual == _SENTINEL_ROUTE_PATH, (
            f"the routing-derivation literal {_SENTINEL_ROUTE_PATH!r} must match "
            f"the adapter's published shape {actual!r}; a future adapter rename "
            f"must update the derivation, not pass silently"
        )


class TestP19ClaimMachinery:
    """The P19 claim must fire on the mutation its site controls.

    On a consistent drive the transport's `stream` overwrite is
    value-preserving (module docstring, "P19 observation posture"), so the
    honest bytes produce no `envelope.stream` delta and the row could be
    decorative without this class noticing. The flip gives P19 the one
    mutation `_ollama_body` actually controls — a `stream` value that differs
    from what `translate_to_upstream` left — and proves two things:

    * **Inclusion + claim** — with `ALWAYS` met and `provider_key` derived
      from the binding (`"ollama_cloud"`, in P19's scope), the flipped delta
      is claimed and the run is green with
      `("envelope.model", "envelope.stream")`.
    * **Trigger gate** — with `ALWAYS` omitted P19 loses eligibility, the
      flipped delta goes unclaimed, and the oracle raises naming
      `envelope.stream`: P19 is the *unique* ALWAYS-claimant for that address
      on this adapter, so the trigger omission is what disarms the claim.
      (The scope-**exclusion** half — P19 claiming nothing on a foreign
      adapter — is KBR-307 AC4's case, `test_oracle_driven.py::
      TestDrivenScopeEnforcement`; this slice cannot drive a foreign adapter
      because the transport binds `ollama_cloud` only.)

    The subject entry is the smallest **clean** entry whose inbound body does
    not ask for `stream: true` — flipped `true` then differs from what the
    inbound projection reads. Selecting from the corpus rather than naming an
    id keeps the test alive across corpus renames (the routing
    falsification's shape); filtering on `stream is not True` is what makes
    the flip a real mutation.
    """

    def _subject_entry(self):
        """Return the smallest clean entry whose inbound body does not stream.

        Returns:
            The corpus entry the P19 claim-machinery tests drive.

        Raises:
            AssertionError: When no clean entry qualifies — a bare
                ``next()`` would surface as a ``StopIteration`` traceback
                that names nothing; the failure must say what the corpus
                needs so the next author can act on it.
        """
        for entry in _size_ordered_am_entries():
            if entry.id in _CORPUS_SKIP_TABLE:
                continue
            if json.loads(entry.request.body).get("stream") is not True:
                return entry
        raise AssertionError(
            "no clean Anthropic-Messages corpus entry has a body without "
            "stream: true — the P19 claim-machinery tests need one (the flip "
            "must change the value to be a real mutation); add a corpus "
            "entry whose body omits stream or sets it false"
        )

    async def test_flipped_stream_is_claimed_when_always_is_met(self) -> None:
        """Positive: flip the captured `stream`; P19 claims it; the run is green."""
        entry = self._subject_entry()

        async with BridgeFixture(
            transport("provider_aiohttp", WireFormat.OLLAMA_CHAT)
        ) as fixture:
            provider_key = _provider_key(fixture)
            status, _text = await fixture.post(
                inbound_path(InboundProtocol.MESSAGES),
                json.loads(entry.request.body),
            )
            assert status == 200
            captured = _flip_captured_stream_true(list(fixture.captures)[0])

            report = oracle.assert_no_unclaimed_mutation(
                inbound=_inbound_capture(entry),
                inbound_format=entry.wire_format,
                captured=captured,
                captured_format=WireFormat.OLLAMA_CHAT,
                register=r.REGISTER,
                triggers_met=_triggers_met(entry),
                provider_key=provider_key,
            )
            assert report.deltas == ("envelope.model", "envelope.stream"), (
                f"the flipped stream must be claimed by P19 alongside M1's "
                f"model rewrite; got {report.deltas!r}"
            )

    async def test_flipped_stream_is_unclaimed_when_always_is_omitted(self) -> None:
        """Falsification: without ALWAYS the same flip raises naming envelope.stream."""
        entry = self._subject_entry()

        async with BridgeFixture(
            transport("provider_aiohttp", WireFormat.OLLAMA_CHAT)
        ) as fixture:
            provider_key = _provider_key(fixture)
            status, _text = await fixture.post(
                inbound_path(InboundProtocol.MESSAGES),
                json.loads(entry.request.body),
            )
            assert status == 200
            captured = _flip_captured_stream_true(list(fixture.captures)[0])

            # ALWAYS deliberately omitted: P19 loses eligibility, no other row
            # in scope claims envelope.stream on ollama_cloud, and the oracle
            # must fail naming it. The else-branch raises so a silently-green
            # run cannot pass.
            triggers = _triggers_met(entry) - {r.Trigger.ALWAYS}
            try:
                oracle.assert_no_unclaimed_mutation(
                    inbound=_inbound_capture(entry),
                    inbound_format=entry.wire_format,
                    captured=captured,
                    captured_format=WireFormat.OLLAMA_CHAT,
                    register=r.REGISTER,
                    triggers_met=triggers,
                    provider_key=provider_key,
                )
            except oracle.UnclaimedMutationError as exc:
                assert "envelope.stream" in exc.paths, (
                    f"the ALWAYS-omission falsification should name "
                    f"envelope.stream; got {exc.paths!r}"
                )
            else:
                raise AssertionError(
                    "the ALWAYS-omission falsification must fail the oracle; "
                    "a green run here would mean P19 is not the claimant of "
                    "envelope.stream on ollama_cloud"
                )


class TestOAuthLoginLegThroughTheSlice:
    """The slice's other half: the OAuth login leg on the same recorder.

    A form-encoded token grant is not an LLM request and has no reader
    (§7.5's decided question), so this is recorder-level coverage, not a
    projection comparison. `TestTheOAuthLoginLeg` in
    `test_provider_aiohttp.py` covers the leg in isolation — the seam, both
    exchanges, the teardown carve-out; none of its tests run a **bridge** and
    the leg concurrently. This test drives both products through one recorder
    instance in sequence and asserts neither disturbs the other: the
    §7.2.2 dual-product load, at bridge fidelity.
    """

    async def test_bridge_capture_and_oauth_leg_share_one_recorder(self) -> None:
        """Drive the bridge, then the login leg; both captures arrive, teardown is clean.

        The bridge's capture must land at the adapter's endpoint; the leg's at
        `OAUTH_TOKEN_SUFFIX`, form-encoded with the login grant type; and the
        fixture's clean-path teardown (`__aexit__`'s
        `assert_teardown_clean`) must pass over both — a token grant is
        answered before any format lookup and must not be reported as a
        fallback (the carve-out `test_provider_aiohttp.py` pins; the
        shared-recorder drive reaching teardown clean is the proof it did
        not break here).
        """
        subject = transport("provider_aiohttp", WireFormat.OLLAMA_CHAT)
        async with BridgeFixture(subject) as fixture:
            status, _text = await fixture.post(
                inbound_path(InboundProtocol.MESSAGES),
                json.loads(
                    next(
                        e
                        for e in _size_ordered_am_entries()
                        if e.id not in _CORPUS_SKIP_TABLE
                    ).request.body
                ),
            )
            assert status == 200

            with oauth_token_endpoint(subject.recorder):
                async with aiohttp.ClientSession() as http:
                    await openai_oauth._exchange_code_for_tokens(
                        "code", "verifier", "client-id", http
                    )

            captures = list(fixture.captures)

        assert [c.path for c in captures] == [
            OLLAMA_CHAT_SUFFIX,
            OAUTH_TOKEN_SUFFIX,
        ], (
            f"one recorder, two products: expected the bridge's request at "
            f"{OLLAMA_CHAT_SUFFIX!r} then the login leg at "
            f"{OAUTH_TOKEN_SUFFIX!r}; got {[c.path for c in captures]!r}"
        )

        oauth_capture = captures[1]
        assert oauth_capture.method == "POST"
        assert oauth_capture.body is not None
        assert oauth_capture.body.decode().split("&")[0] == (
            "grant_type=authorization_code"
        ), (
            f"the login leg's grant must be the authorization-code exchange; "
            f"got {oauth_capture.body.decode()[:120]!r}"
        )

        # The carve-out: the OAuth endpoint is outside the declared-format
        # claim by name, and a shared-recorder drive must not turn it into
        # a fallback hit. `BridgeFixture.__aexit__` runs the same check on
        # the clean path (tests/harness/bridge.py:825), so reaching the
        # post-block `assert [c.path ...]` line with the expected order is
        # the load-bearing proof — a second teardown pass would be the
        # unfalsifiable repeat §7.5.4 removed (and `provider_aiohttp.py`'s
        # own docstring records).
