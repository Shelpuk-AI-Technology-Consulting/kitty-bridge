"""Corpus-driven oracle slice on the default-transport adapter.

`.system_design/TEST_SUITE.md` §3.3.4 (triggers + complements across representative
models; parametrised over transport), §3.3.5 (routing), §7.4 (oracle / wire
projections) · plan task **T-D4** (KBR-54).

Drives the bridge end-to-end against the transparency oracle on **every** inbound
Anthropic-Messages corpus entry, using the `custom_openai` default-transport adapter
(Chat Completions upstream — the non-native route, where unannounced body changes are
most likely to hide), with all four oracle obligations active: §3.3.1 totality,
§3.3.2 assertions 1 + 2, §3.3.5 routing.

**One representative default-transport adapter, end to end, against the corpus, with
routing and both assertions active.** The full 20-adapter default matrix is **T-D9**
(KBR-58); the three custom-transport slices are **T-D5–T-D7** (KBR-55/56/57). T-D4's
routing-derivation restriction and skip-table pattern propagate to those tickets (see
`.system_design/steps/t_d4_default_transport_slice.md` for the inter-task contract).

**Routing-derivation restriction.** This slice's default-transport binding has
`base_url = recorder.base_url` with no query. The KBR-143 base-URL query-merge rule is
not exercised here; T-D5–T-D7 inherit the literal comparison and may need to extend
the derivation when their adapters carry endpoint-side queries (Azure, Vertex). The
authority-and-scheme rewrite uses the **recorder's** base URL
(`urlsplit(fixture.transport.recorder.base_url)`), not the bridge's — the two bind
different ephemeral ports (§3.3.5's T-W4 scope addition; same shape T-D2's
`_drive_azure` uses in `tests/harness/test_oracle_routing.py`).

**No special small-context profile.** The harness's default `"harness-model"` resolves
via `get_model_context_tokens` to `DEFAULT_CONTEXT_TOKENS = 200_000` (×
`TOKENS_TO_CHARS_FACTOR = 4`), giving an 800 000-char derived budget. The corpus's
compaction-pair entries were calibrated against the static 2.8 M-char
`_COMPACTION_CHAR_THRESHOLD`; on the default profile `compaction_budget_over` triggers
M5 (its 2.85 MB body is over budget) and the entry passes the oracle with all M5
deltas claimed. `compaction_budget_under` is no longer a clean M5 complement at this
budget — recorded as a calibration gap in the skip table.

**Skip table (`_CORPUS_SKIP_TABLE`)** — the slice's authored restriction. Every entry
whose empirical pass produced a real I1 finding is excluded with a one-line
`pytest.skip(...)` message naming the unclaimed-delta path and its owner-tracking
marker (F3 finding or sibling ticket id). Mirrors KBR-52's 2026-09-19
"parameter-free `base_url`s" pattern: an authored restriction, not a silent skip.
The table is local to this module; §8.3's `tests/exemptions.py` is platform skips, a
different subject.

**The module's one `src/kitty` import is the sentinel pin below, test infrastructure
and not part of the derivation** (T-D2's precedent in `test_oracle_routing.py`):
`TestExpectedRoutePathIsLiteralAndPinned` asserts the derivation's path literal
against `CustomOpenAIAdapter().get_upstream_path(...)` so an adapter rename fails
loudly. The oracle, the register, the corpus and the projections are all
`tests.harness` modules; §3.3.1's independent-oracle rule binds the judge, and this
module is a driver.

**Layer.** No `pytestmark`; harness tests default to `l1` per `tests/layers.py` and
the T-D1/T-D2 precedent. §3.4 calls this surface L3 and T-K6 owns `l3` activation.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from harness import oracle
from harness import register as r
from harness.bridge import BridgeFixture, InboundProtocol, inbound_path, transport
from harness.contract import CapturedRequest, WireFormat
from harness.corpus import load_corpus
from kitty.providers.custom_openai import CustomOpenAIAdapter

#: The path the CC adapter publishes as its endpoint. Pinned so a future adapter
#: rename fails the test loudly with both values in the message (F6.a).
_SENTINEL_ROUTE_PATH = "/chat/completions"

#: A wrong path used to falsify the routing assertion. The body obligations pass on
#: the real route, so a RoutingMismatchError raised here proves the routing
#: assertion bites through the driven slice (F6.b; T-D2's reroute-with-byte-identical-body
#: shape applied to the L3-driven surface).
_WRONG_ROUTE_PATH = "/_intentionally_wrong_path_for_falsification"

#: Corpus entries whose empirical pass surfaced a real I1 finding, framing gap, or
#: calibration gap. Each value is the pytest.skip message; each message names the
#: unclaimed-delta path (or the framing/calibration cause) and the owner-tracking
#: marker — a F3 finding, a sibling ticket id, or a `new finding` placeholder.
#:
#: Populated from the corrected `.scratch/probe_corpus.py` pass on `origin/main`
#: (2026-09-23). See `REQUIREMENTS.md` §3 F3 and §3 F4 for the per-row rationale.
#:
#: KBR-309 removed `plain_turn` and `effort_configured`: both entries now pass the
#: oracle with their full claimed-delta tuple pinned by
#: :data:`_EXPECTED_CLAIMED_DELTAS` below. The four deltas they used to drop —
#: `envelope.extra[context_management]`, `envelope.extra[metadata]`,
#: `conversation.turns[0].parts[0].text`, `conversation.turns[0].parts[1]` — are
#: claimed by M27, M26 (activated by `Trigger.ALWAYS` in `_triggers_met`),
#: M28's text anchor, and M28's bare-part anchor respectively.
_CORPUS_SKIP_TABLE: dict[str, str] = {
    # F3.d: the CC adapter drops the tool_result turn on a Messages body carrying a
    # 50 000-char tool_result (both under- and over-limit entries). Likely a pairing-
    # validation drop (M7 territory) but the trigger and claim are not in the register.
    "tool_result_under_limit": (
        "F3.d conversation.turns[2] dropped on CC adapter from Messages body "
        "carrying a 50 000-char tool_result; new finding, KBR-54 scope addition"
    ),
    "tool_result_over_limit": (
        "F3.d conversation.turns[2] dropped on CC adapter from Messages body "
        "carrying a 50 001-char tool_result; new finding, KBR-54 scope addition"
    ),
    # (F3.c — the CC reader's missing `reasoning_content` slot — resolved
    # by KBR-310: the request-direction grammar models the field on assistant
    # messages as a Thinking part, and `tool_use_and_tool_result` declares
    # `thinking_signalled_or_inferred` so P8's part-path claim is in scope.
    # The entry surfaces the remaining F3.a/F3.b drops — context_management
    # and metadata — which are KBR-309's territory on the CC adapter.)
    "tool_use_and_tool_result": (
        "F3.a envelope.extra[context_management] + F3.b envelope.extra[metadata] "
        "unclaimed on CC adapter; sibling of KBR-54 scope addition, KBR-309 owns"
    ),
    # Framing gap, not a register gap: the entry's Messages body has role 'system'
    # inside messages (forbidden by the Anthropic Messages format — system prompts
    # are a top-level field). The Messages adapter rejects with UnreadableBodyError
    # before any capture; not a fidelity oracle input.
    "tools_declared": (
        "framing gap: entry body has role 'system' inside messages; Messages "
        "adapter rejects with UnreadableBodyError before capture"
    ),
    # Framing gap: bridge returns 400 before any capture. Not a fidelity oracle input.
    "system_prompt_over_window_compacts_normally": (
        "framing gap: bridge returns 400 before capture; entry body is rejected "
        "by the Messages adapter at ingress"
    ),
    # Calibration gap: the entry was calibrated to the 2.8 M-char static threshold
    # (tests/corpus/README.md) but the default profile's derived budget is 800 000
    # chars, so M5 fires (body 2.8 MB > 800 K budget) and the entry is no longer a
    # clean M5 complement. The body carries no tool_result (verified empirically
    # against `compaction_budget_under.body`: zero tool blocks; manifest's
    # `triggers_absent: tool_result_over_limit` matches), so M3 has nothing to act
    # on — assertion 2's flag of M3 in the probe came from M3's coarse
    # `parts[*]` anchor matching M5's pruned turn texts (M5's anchor is a
    # proper prefix of M3's, so M5 does not specifically claim for M3). A
    # sibling ticket regenerating the entry against the 800 K-char budget is TBD.
    "compaction_budget_under": (
        "calibration gap: entry calibrated to 2.8 M-char static threshold; default "
        "profile's 800 000-char budget makes this a trigger case for M5 (body over "
        "budget); the entry has no oversized tool_result so M3 has nothing to act on"
    ),
    # New finding, KBR-55 scope addition (2026-09-24): the Messages→CC carry
    # drops an Anthropic-defined tool's `type` discriminator (the CC builder
    # writes `{name, description, parameters}` only) and the CC reader has no
    # `type` slot, so an inbound `web_search_20250305` projects
    # `conversation.tools[web_search].type = "web_search_20250305"` while the
    # captured projection carries `None` — a delta no register row claims
    # (M16 claims only `cache_control`). The entry's `description` half was a
    # fixture defect fixed in the entry itself (the carry always writes
    # `description`, defaulting to `""`); the `type` half is the finding. A
    # register row for the typed-tool `type` drop is a sibling ticket.
    "p35_tool_choice_omitted_forcing_anthropic_tool_trigger": (
        "F3.e conversation.tools[web_search].type dropped on CC adapter from a"
        " body carrying an Anthropic-defined (typed) tool declaration; no"
        " register row claims the typed-tool `type` drop — new finding, KBR-55"
        " scope addition"
    ),
}

#: Per-entry expected-delta tuples for entries whose triggers legitimately
#: fire on the default route. Clean entries (everything not named here)
#: produce only M1's `PROFILE_SETS_MODEL` rewrite at `envelope.model`; the
#: two KBR-55 entries below carry REQUEST triggers whose bridge-level sites
#: (`MessagesTranslator.translate_request` for M28, `carry_tool_choice_and_
#: metadata` for P35) are reachable on every translated adapter including
#: `custom_openai`, so the rows' claims fire here exactly as they do on the
#: curl_cffi route the entries were authored for. `compaction_budget_over`
#: is asserted inline (M5's tail is profile-dependent); these two are exact.
_EXPECTED_DELTAS: dict[str, tuple[str, ...]] = {
    "m28_empty_stop_sequences_trigger": (
        "envelope.model",
        "conversation.sampling[stop]",
    ),
    "p35_tool_choice_omitted_no_tools_trigger": (
        "envelope.model",
        "envelope.extra[tool_choice]",
    ),
}


#: Per-entry expected claimed-delta tuple — KBR-309. Entries whose bodies carry
#: Claude Code control fields (`output_config`, `thinking`, `context_management`,
#: `metadata`, multi-part user turns, multiple system blocks) project those fields
#: as deltas on the CC wire; the register rows M5b/M5d/M5f/M26/M27/M28 claim
#: them. This dict pins the exact set so a future drift — a new unclaimed field,
#: a claim withdrawn, a delta appearing or vanishing — fails the slice loudly
#: instead of passing silently. The walk order matches `_structural_diff`'s
#: traversal (envelope → envelope.extra → conversation.system → conversation.turns).
#:
#: `plain_turn` and `effort_configured` produce the same ten-delta tuple: both
#: bodies carry `model: "MiniMax-M3"` and the harness profile resolves to
#: `"harness-model"`, so the model lands as a delta on both. The captured body's
#: `MiniMax-M3`-shaped normalisation keeps them identical.
_EXPECTED_CLAIMED_DELTAS: dict[str, tuple[str, ...]] = {
    "plain_turn": (
        "envelope.model",
        "envelope.extra[context_management]",  # M27
        "envelope.extra[metadata]",            # M26 (activated by Trigger.ALWAYS)
        "envelope.extra[output_config]",      # P5f
        "envelope.extra[thinking]",           # P5d
        "conversation.system[0].text",        # P5b
        "conversation.system[1]",             # P5b
        "conversation.system[2]",             # P5b
        "conversation.turns[0].parts[0].text",  # M28 (join)
        "conversation.turns[0].parts[1]",     # M28 (join)
    ),
    "effort_configured": (  # same shape as plain_turn (10 deltas, same walk order)
        "envelope.model",
        "envelope.extra[context_management]",
        "envelope.extra[metadata]",
        "envelope.extra[output_config]",
        "envelope.extra[thinking]",
        "conversation.system[0].text",
        "conversation.system[1]",
        "conversation.system[2]",
        "conversation.turns[0].parts[0].text",
        "conversation.turns[0].parts[1]",
    ),
}


def _size_ordered_am_entries() -> list:
    """Return the inbound-Anthropic-Messages corpus entries, size-ordered.

    Called once at module collection time (the parametrize decorator below
    evaluates immediately). Loads the corpus and sorts by body length then id so
    the test collection order and failure messages are stable across runs.

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

    Derived from `custom_openai`'s published URL shape (`/chat/completions`, no query),
    with the recorder's scheme and host rewritten in (the recorder binds a different
    ephemeral port from the bridge; using `fixture.base_url` here would produce a
    `route.host` mismatch on every capture).

    Args:
        fixture: A started `BridgeFixture`; the recorder's base URL is read here.

    Returns:
        An `ExpectedRoute` whose path and query are the literal derivation and whose
        scheme and host are the recorder's.
    """
    rec = urlsplit(fixture.transport.recorder.base_url)
    return oracle.ExpectedRoute(
        method="POST",
        scheme=rec.scheme,
        host=rec.netloc,
        path=_SENTINEL_ROUTE_PATH,
        query="",
    )


#: The default harness profile's derived compaction budget, in chars. Derived from
#: `get_model_context_tokens("custom_openai", "harness-model", None)` resolving to
#: `DEFAULT_CONTEXT_TOKENS = 200_000` (the fallback in
#: `src/kitty/providers/model_context.py:15`) × `TOKENS_TO_CHARS_FACTOR = 4`. A
#: literal here, not an import: the driver is a judge module and the harness's
#: independent-oracle rule (§3.3.1) forbids importing `src/kitty` — same decision
#: T-D2 made for Azure's `api-version` literal.
_DEFAULT_PROFILE_BUDGET_CHARS = 800_000


def _triggers_met(entry) -> frozenset[r.Trigger]:
    """Return the trigger vocabulary for one corpus entry.

    Combines the entry's declared `triggers_met` (REQUEST-only triggers; the corpus
    loader refuses non-REQUEST triggers in both lists per KBR-186's classification)
    with the route's `NON_NATIVE_UPSTREAM_WIRE` trigger, the profile's
    `PROFILE_SETS_MODEL` trigger (both declared at the call site per the corpus
    README's PROFILE-decided-trigger rule, line 122), and `Trigger.ALWAYS` — the
    KBR-307 precedent (`test_oracle_driven.py` includes it at its driven-slice
    call site); `_claim_matching` keeps only rows whose trigger is in the set,
    so the unconditional rows M14 and M26 stay inert without it (KBR-309: M26
    is the metadata drop M26 was authored to claim). For entries whose body
    exceeds the default profile's derived budget, `OVER_COMPACTION_BUDGET` is added
    — M5's trigger is PROFILE-decided and declared at the call site that resolves
    the profile (corpus README, "Triggers have three states").

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
    # `_compact_messages`), which differs from the committed Anthropic shape by a
    # constant (~92 chars for this layout, per tests/corpus/README.md). The raw
    # body length is the approximation; no corpus entry sits within that margin
    # of the 800 K boundary (the closest are the ~605 KB m6 pair, 24 % under).
    if len(entry.request.body) > _DEFAULT_PROFILE_BUDGET_CHARS:
        triggers = triggers | {r.Trigger.OVER_COMPACTION_BUDGET}
    return triggers


def _inbound_capture(entry) -> CapturedRequest:
    """Build the oracle's inbound `CapturedRequest` for `entry`.

    The host is rewritten to `127.0.0.1` (the bridge's loopback origin) because the
    entry's committed `host` is `api.anthropic.com` — the host the agent reached in
    the original capture. The oracle's body projection does not read the host; the
    routing assertion compares the *captured* route against the derived expectation,
    so the inbound host is irrelevant for both assertions.

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


class TestRoutingFalsification:
    """§1.4 harness rule: the routing assertion must bite through the driven slice.

    The oracle's body obligations pass on the real captured body; a sentinel-wrong
    expected path must raise `RoutingMismatchError` naming `route.path`. The
    ordering (§3.3.5: routing runs last so the louder body diagnosis surfaces first)
    means the body obligations have already passed by the time routing raises — the
    routing error is the only thing that can fire. This is T-D2's
    reroute-with-byte-identical-body shape, applied to the L3-driven surface.
    """

    async def test_routing_mismatch_raises_on_wrong_expected_path(self) -> None:
        """Drive a small clean body; reroute to a sentinel path; oracle fails on routing."""
        # The smallest clean entry — the first not in the skip table — drives the
        # falsification. Selecting by "not skipped" rather than by id keeps the
        # body-obligations-pass-first ordering intact if a corpus entry is renamed,
        # removed, or moved into the skip table: the choice is recomputed from the
        # same table the driven slice consults, so the two can never disagree.
        clean = next(
            e for e in _size_ordered_am_entries() if e.id not in _CORPUS_SKIP_TABLE
        )

        async with BridgeFixture(transport("aiohttp", WireFormat.CHAT_COMPLETIONS)) as fixture:
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
                captured_format=WireFormat.CHAT_COMPLETIONS,
                register=r.REGISTER,
                triggers_met=_triggers_met(clean),
                expected_route=real_route,
            )

            # 2. The same captured body on a sentinel-wrong path raises
            #    RoutingMismatchError. The body obligations have already passed on
            #    the real route above, so the routing error is the only thing that
            #    can fire (§3.3.5's ordering; T-D2's falsification shape).
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
                    captured_format=WireFormat.CHAT_COMPLETIONS,
                    register=r.REGISTER,
                    triggers_met=_triggers_met(clean),
                    expected_route=wrong_route,
                )
            assert "route.path" in exc_info.value.paths, (
                f"the routing falsification should name route.path; got {exc_info.value.paths!r}"
            )


class TestExpectedRoutePathIsLiteralAndPinned:
    """F6.a: the expected-path literal must match the adapter's published shape.

    A future adapter rename (or a derivation that drifts to a hardcoded wrong path)
    fails this test loudly with both values in the message — the kind of self-consistent
    green the design §1.4 forbids.
    """

    def test_sentinel_matches_adapter_upstream_path(self) -> None:
        """`_SENTINEL_ROUTE_PATH` equals `CustomOpenAIAdapter().get_upstream_path(...)`."""
        actual = CustomOpenAIAdapter().get_upstream_path("harness-model")
        assert actual == _SENTINEL_ROUTE_PATH, (
            f"the routing-derivation literal {_SENTINEL_ROUTE_PATH!r} must match "
            f"the adapter's published shape {actual!r}; a future adapter rename "
            f"must update the derivation, not pass silently"
        )


class TestCorpusDrivenDefaultSlice:
    """End-to-end oracle run on every inbound-Anthropic-Messages corpus entry.

    Parametrised over the corpus entries in size order (smallest first). Each entry
    that runs the oracle asserts:

    * `len(captures) == 1` (no fired retry ladder).
    * `report.deltas` equals the expected tuple — `("envelope.model",)` for clean
      entries (the only delta is M1's `PROFILE_SETS_MODEL` rewrite of `envelope.model`);
      `compaction_budget_over` asserts `report.deltas[0] == "envelope.model"` and
      `len(report.deltas) >= 200` (the count from the empirical pass; the exact tail
      is profile-dependent so the test pins the head and a lower bound).
    * `verify_total` clean (totality is the oracle's first obligation; surfaces as a
      pass before the structural diff).
    * `expected_route` matches the captured route after the recorder-authority rewrite
      (§3.3.5 routing assertion).

    Entries in `_CORPUS_SKIP_TABLE` are `pytest.skip`'d with the F3 finding / framing
    rationale / calibration marker.
    """

    @pytest.mark.parametrize(
        "entry",
        _size_ordered_am_entries(),
        ids=_entry_id,
    )
    async def test_drive_entry_through_default_adapter(self, entry) -> None:
        """Drive one corpus entry end to end; the oracle must accept it.

        Entries in `_CORPUS_SKIP_TABLE` skip with their rationale. Clean entries
        assert `report.deltas == ("envelope.model",)` (M1's PROFILE_SETS_MODEL
        rewrite is the only permitted delta); `compaction_budget_over` asserts the
        M1 head plus a >=200 lower bound (M5's profile-dependent turn pruning).
        """
        skip_reason = _CORPUS_SKIP_TABLE.get(entry.id)
        if skip_reason is not None:
            pytest.skip(skip_reason)

        async with BridgeFixture(transport("aiohttp", WireFormat.CHAT_COMPLETIONS)) as fixture:
            status, _text = await fixture.post(
                inbound_path(InboundProtocol.MESSAGES),
                json.loads(entry.request.body),
            )
            assert status == 200, (
                f"the recorder's minimal success reply must come back 200; got {status}"
            )
            captures = list(fixture.captures)
            assert len(captures) == 1, (
                f"exactly one upstream request — a second capture would mean the "
                f"empty-response retry ladder fired (got {len(captures)})"
            )
            captured = captures[0]

            report = oracle.assert_no_unclaimed_mutation(
                inbound=_inbound_capture(entry),
                inbound_format=entry.wire_format,
                captured=captured,
                captured_format=WireFormat.CHAT_COMPLETIONS,
                register=r.REGISTER,
                triggers_met=_triggers_met(entry),
                expected_route=_expected_route(fixture),
            )

            # Per-method delta assertion (T-D2's positive-driven precedent).
            if entry.id == "compaction_budget_over":
                # M5 fires (body 2.85 MB > 800 K budget) and M1 rewrites the model.
                # The exact tail is profile-dependent; the empirical pass produced
                # ~200 deltas. Pin the head and a lower bound; pin the count so a
                # future regression where M5 stops firing is caught.
                assert report.deltas[0] == "envelope.model", (
                    f"compaction_budget_over's first delta must be envelope.model "
                    f"(M1's PROFILE_SETS_MODEL rewrite); got {report.deltas[0]!r}"
                )
                assert len(report.deltas) >= 200, (
                    f"compaction_budget_over must exercise M5 (>=200 turn deltas; "
                    f"empirical baseline 383 on 2026-09-23, threshold halved for "
                    f"margin against profile-driven variation while still catching "
                    f"the M5-stopped-firing regression); got {len(report.deltas)}"
                )
            elif entry.id in _EXPECTED_CLAIMED_DELTAS:
                # KBR-309 — entries whose bodies carry Claude Code control fields
                # produce claimed-delta tuples the register rows cover. The dict
                # pins the exact set so a future drift (a new unclaimed field, a
                # claim withdrawn, a delta appearing or vanishing) fails loudly
                # instead of passing silently. Adding/removing register rows
                # changes the tuple, so a row edit goes through this test on its
                # way to merge.
                expected = _EXPECTED_CLAIMED_DELTAS[entry.id]
                assert report.deltas == expected, (
                    f"{entry.id!r}: expected {expected!r}; got {report.deltas!r}. "
                    f"If a register row was added or removed, update "
                    f"_EXPECTED_CLAIMED_DELTAS to match."
                )
            else:
                # Per-entry expected deltas (KBR-55): two entries carry REQUEST
                # triggers whose sites are reachable on the default route (the
                # bridge-level carry / translator runs on every translated
                # adapter including `custom_openai`). The corpus entry names
                # its own expected tuple; clean entries get the same M1-only
                # assertion.
                expected = _EXPECTED_DELTAS.get(entry.id, ("envelope.model",))
                assert report.deltas == expected, (
                    f"entry {entry.id!r}: expected {expected!r} (per-entry map"
                    f" for KBR-55 / M1 only otherwise); got {report.deltas!r}"
                )
