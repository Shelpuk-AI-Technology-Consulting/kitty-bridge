"""KBR-47 (T-C4) — the committed corpus entries reach the paths they are named for.

``.system_design/TEST_SUITE.md`` §7.1 · plan task **T-C4** (KBR-47), requirements
``.requirements/20260914T184942Z_kbr47_tc4_corpus_entries/REQUIREMENTS.md``.

A corpus entry that names a code path but never reaches it is the F25 failure
KBR-5 measured: ``TestCompactionPostCondition``'s fixtures looked like coverage
and proved nothing, because a surviving user turn always defeated the
post-condition they were named for.  This module is the second reader for
T-C4's four entries: each one is loaded from ``tests/corpus/`` and run through
the real compactor on a server whose context budget matches the test's intent —
so an entry whose shape drifts goes red here, not silently.

What each entry claims, and the test that holds it:

* ``m6_recovery_oversized_paired`` — a well-paired conversation with many
  user/assistant turns, padded past 600,000 serialized chars.  The recovery
  gate (``_is_oversized_request``) opens only past that threshold, and the
  pre-flight must succeed so the request actually reaches the upstream.
* ``m5_irreducible_single_final_turn`` — a single user turn larger than the
  compaction budget: §6.1's *observed* behaviour is that compaction engages
  but cannot reduce the conversation further; the test pins that observation.
* ``system_prompt_over_window_compacts_normally`` — the post-KBR-5 positive
  control: an oversized system prompt plus an ordinary user turn **compacts
  normally**; the user turn survives.

The balancing scripted-recorder half — the test that answers 413 and is where
trigger M6 is declared met, per the ticket's own comment — lives in
:mod:`tests.bridge.test_tc4_corpus_recovery_l3` (separate L3 file, per the
layer-marker rule).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from harness import corpus as k

from kitty.bridge.server import (
    _OVERSIZED_INPUT_THRESHOLD,
    BridgeServer,
)
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

pytestmark = pytest.mark.l1

#: The committed corpus root.
CORPUS = Path(__file__).resolve().parents[1] / "corpus"

#: The four T-C4 entry ids (KBR-256 adds the streaming twin). Each maps to a
#: single AC the wiring tests hold.
TC4_ENTRY_IDS = (
    "m6_recovery_oversized_paired",
    "m6_recovery_oversized_paired_streaming",  # KBR-256: same wiring shape as the non-streaming twin.
    "m5_irreducible_single_final_turn",
    "system_prompt_over_window_compacts_normally",
)

#: Both M6 twins. The recovery's wiring properties (oversized gate, pre-flight
#: survival, no-orphan pairing) are pinned per twin: the streaming entry is a
#: builder output (scripts/build_corpus_m6_streaming.py), and a builder drift —
#: e.g. a re-compaction applied before write — would otherwise be caught by
#: nothing at this layer.
M6_ENTRY_IDS = (
    "m6_recovery_oversized_paired",
    "m6_recovery_oversized_paired_streaming",
)


# ── Stubs (kept local to keep this file independent) ─────────────────────────


class _StubLauncher(LauncherAdapter):
    """Minimal launcher adapter that performs no spawning."""

    def __init__(self, protocol: BridgeProtocol = BridgeProtocol.MESSAGES_API) -> None:
        """Initialize the stub launcher with the given bridge protocol."""
        self._protocol = protocol

    @property
    def name(self) -> str:
        """Return a stable name for diagnostic output."""
        return "tc4-stub-launcher"

    @property
    def binary_name(self) -> str:
        """Return a stable binary name for diagnostic output."""
        return "tc4-stub-launcher"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        """Return the protocol this stub advertises."""
        return self._protocol

    def build_spawn_config(
        self,
        profile: Profile,
        bridge_port: int,
        resolved_key: str,
        **kwargs: object,
    ) -> SpawnConfig:
        """Return an empty spawn config; this stub is never used to launch.

        ``**kwargs`` absorbs parameters the base class gained later (e.g. the
        ``context_tokens`` keyword the launcher passes today), so a signature
        mismatch cannot break a future reuse of this stub.
        """
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class _StubProvider(ProviderAdapter):
    """Minimal provider that returns request/response bodies unchanged."""

    @property
    def provider_type(self) -> str:
        """Return a stable provider-type name."""
        return "tc4-stub"

    @property
    def default_base_url(self) -> str:
        """Return a non-loopback placeholder URL — not exercised on these paths."""
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Return the request as-is."""
        return {"model": model, "messages": messages, "stream": kwargs.get("stream", False)}

    def parse_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Return the response as-is."""
        return response_data

    def map_error(self, status_code: int, body: dict[str, Any]) -> Exception:
        """Return a generic exception for an upstream error."""
        return Exception(f"Upstream error {status_code}: {body}")


def _server(context_chars: int) -> BridgeServer:
    """Return a ``BridgeServer`` whose compaction budget matches the test's intent.

    Args:
        context_chars: What ``_get_max_context_chars`` will return. The wiring
            tests need the compactor to engage on the entry bodies, which
            requires a context budget smaller than the entry's serialized size.

    Returns:
        The server.
    """
    server = BridgeServer(_StubLauncher(), _StubProvider(), "test-key")
    server._get_max_context_chars = lambda: context_chars  # type: ignore[method-assign]
    return server


def _load_entry(entry_id: str) -> k.CorpusEntry:
    """Load one committed entry by id.

    Args:
        entry_id: The manifest's filename stem.

    Returns:
        The entry, with its body bytes.

    Raises:
        AssertionError: When the entry is not committed under that id — the
            corpus and this module must agree on the names.
    """
    path = CORPUS / f"{entry_id}.json"
    assert path.is_file(), f"{entry_id} is not committed under tests/corpus/"
    return k.load_entry(path)


# ── Format-level claims ──────────────────────────────────────────────────────


class TestTheEntriesAreCommittedAndDeclared:
    """AC-1's count claim — the corpus is exactly what this task accounts for.

    The per-entry manifest claims (origin, origin_note, triggers, wire
    format) live at L2 in :mod:`tests.harness.test_corpus_lint`, where the
    corpus format's demands on a synthetic entry belong.
    """

    def test_every_tc4_entry_this_module_knows_is_committed(self) -> None:
        """AC-1: every TC4 entry this task accounts for is committed.

        The "no orphan entry anywhere in the corpus" guarantee is
        cross-task — it lives in
        :mod:`tests.harness.test_corpus_lint::TestTheCommittedCorpusHasNoOrphanEntries`
        where the allowlist can know about every documented task. This
        test asserts only its own half: TC4's four entries plus
        ``format_example`` are all present in the corpus.
        """
        ids = {entry.id for entry in k.load_corpus(CORPUS)}

        assert ids.issuperset({*TC4_ENTRY_IDS, "format_example"}), (
            f"TC4 entries missing from the corpus: "
            f"{set(TC4_ENTRY_IDS) - ids}; format_example present: {'format_example' in ids}"
        )


# ── Behaviour-level claims (the anti-F25 half) ───────────────────────────────


class TestTheEntriesReachTheirPaths:
    """The entries reach the paths they are named for, on a server whose budget fits."""

    @pytest.mark.parametrize("entry_id", M6_ENTRY_IDS)
    def test_the_m6_entry_exceeds_the_oversized_threshold_on_the_committed_body(self, entry_id: str) -> None:
        """The recovery gate opens only past 600,000 serialized characters.

        Measured on the committed body — the README rule that triggers are
        declared against what landed, never against the pre-scrub capture. The
        messages array is what ``_is_oversized_request`` serializes. AC-3's
        first half — the second (post-pre-flight) is the test below.

        Args:
            entry_id: Which M6 twin to measure — both must clear the gate.
        """
        entry = _load_entry(entry_id)
        body = json.loads(entry.request.body)

        serialized = len(json.dumps(body["messages"], ensure_ascii=False))

        assert serialized > _OVERSIZED_INPUT_THRESHOLD

    @pytest.mark.parametrize("entry_id", M6_ENTRY_IDS)
    def test_the_m6_entry_survives_preflight_compaction_still_oversized(self, entry_id: str) -> None:
        """The recovery entry is well-paired, so pre-flight succeeds.

        The budget is sized so pre-flight compaction's threshold short-circuit
        leaves the body unchanged — the body must still exceed the oversized
        gate so the upstream has a reason to 413 and the recovery path engages.

        Args:
            entry_id: Which M6 twin to measure — both must survive pre-flight.
        """
        entry = _load_entry(entry_id)
        cc_request = json.loads(entry.request.body)
        before = len(json.dumps(cc_request["messages"], ensure_ascii=False))

        # Budget larger than the entry's body so pre-flight short-circuits
        # below the threshold. A budget smaller than the body would engage
        # pre-flight compaction and the post-pre-flight size could fall below
        # the 600,000 gate.
        server = _server(context_chars=before + 200_000)
        server._apply_compaction(cc_request)

        after = len(json.dumps(cc_request["messages"], ensure_ascii=False))
        assert after == before, (
            "pre-flight compaction must short-circuit below the threshold; "
            f"got body {after} chars, expected {before} (no compaction)"
        )
        assert after > _OVERSIZED_INPUT_THRESHOLD, (
            "the M6 entry must remain over the recovery gate after pre-flight; "
            "otherwise the upstream has no reason to 413"
        )

    @pytest.mark.parametrize("entry_id", M6_ENTRY_IDS)
    def test_the_m6_entry_has_no_orphan_tool_messages(self, entry_id: str) -> None:
        """The well-paired property is structural, not enforced by test discipline.

        This is the anti-F25 half for the M6 entry: if an authoring drift added
        an orphan tool reference, pre-flight pairing validation would silently
        drop it (or, worse, send the conversation upstream as empty). The
        wiring test ``test_the_m6_entry_survives...`` uses a budget larger than
        the body so ``_compact_messages`` short-circuits and the orphan never
        gets a chance to be dropped — so the well-paired property would not be
        caught there. This test names the property structurally.

        Both orphan shapes the pairing rule handles are checked: the Chat
        Completions ``role: tool`` message with a ``tool_call_id``, and the
        Anthropic-native ``tool_result`` content block with a ``tool_use_id``
        whose ``tool_use`` no assistant turn carries. The entry is inbound
        Anthropic Messages, so the native shape is the one a real drift would
        introduce.

        Args:
            entry_id: Which M6 twin to check — both must be well-paired.
        """
        entry = _load_entry(entry_id)
        body = json.loads(entry.request.body)

        # Chat Completions shape: assistant ``tool_calls[].id`` declares, a
        # later ``role: tool`` message consumes.
        cc_call_ids: set[str] = set()
        orphan_cc_ids: list[str] = []
        # Anthropic-native shape: assistant ``tool_use.id`` declares, a later
        # user ``tool_result.tool_use_id`` consumes.
        native_use_ids: set[str] = set()
        orphan_native_ids: list[str] = []

        for message in body["messages"]:
            role = message.get("role")
            content = message.get("content")
            if role == "assistant":
                for call in message.get("tool_calls") or ():
                    if isinstance(call, dict) and isinstance(call.get("id"), str):
                        cc_call_ids.add(call["id"])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and isinstance(block.get("id"), str):
                            native_use_ids.add(block["id"])
            elif role == "tool":
                tool_call_id = message.get("tool_call_id")
                if isinstance(tool_call_id, str) and tool_call_id not in cc_call_ids:
                    orphan_cc_ids.append(tool_call_id)
            elif role == "user" and isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    tool_use_id = block.get("tool_use_id")
                    if isinstance(tool_use_id, str) and tool_use_id not in native_use_ids:
                        orphan_native_ids.append(tool_use_id)

        assert not orphan_cc_ids, (
            "the M6 entry must carry no orphan tool messages; "
            f"found orphan tool_call_ids={orphan_cc_ids}"
        )
        assert not orphan_native_ids, (
            "the M6 entry must carry no orphan tool_result blocks; "
            f"found orphan tool_use_ids={orphan_native_ids}"
        )

    def test_the_m5_irreducible_entry_survives_compaction_unchanged(self) -> None:
        """§6.1's observed behaviour: compaction engages but cannot reduce.

        The irreducible case is observed-but-undecided (G17); this test pins
        the current behaviour without pre-deciding the design question. The
        budget is sized smaller than the entry's body so pre-flight compaction
        engages — but the guaranteed-fit fallback preserves the user turn, so
        the conversation survives unchanged.
        """
        entry = _load_entry("m5_irreducible_single_final_turn")
        cc_request = json.loads(entry.request.body)
        before = json.loads(entry.request.body)["messages"]

        body_chars = len(json.dumps(before, ensure_ascii=False))
        # Budget smaller than the body so compaction engages; but a single
        # user turn is irreducible (guaranteed-fit preserves it).
        server = _server(context_chars=body_chars // 2)
        server._apply_compaction(cc_request)

        assert cc_request["messages"] == before, (
            "compaction must leave an irreducible conversation unchanged; "
            "any reduction here would mean the entry is not actually irreducible"
        )

    def test_the_system_prompt_over_window_entry_compacts_normally(self) -> None:
        """The post-KBR-5 positive control: oversized system, ordinary user turn.

        The guaranteed-fit fallback keeps the user turn, so this entry must
        compact without raising — and the user turn must still be there
        afterwards. A raise here would be a regression to the F3 hypothesis.
        """
        entry = _load_entry("system_prompt_over_window_compacts_normally")
        cc_request = json.loads(entry.request.body)
        before = json.loads(entry.request.body)["messages"]

        body_chars = len(json.dumps(before, ensure_ascii=False))
        # Budget smaller than the body so compaction engages; the user turn
        # must survive.
        server = _server(context_chars=body_chars // 2)
        server._apply_compaction(cc_request)

        # AC-6's meaningful claim is "compaction did not eat the conversation".
        # The ``role != "system"`` filter is decorative here: the entry is in
        # Anthropic Messages inbound shape, where ``system`` is a top-level
        # field rather than a ``role`` on a message — so the filter removes
        # nothing on this entry. The literal assertion is what actually fails
        # if compaction empties the list.
        assert cc_request["messages"], (
            "the user turn must survive the system-prompt-over-window shape"
        )
        assert any(m.get("role") == "user" for m in cc_request["messages"]), (
            "the user turn must survive the system-prompt-over-window shape; "
            f"got roles={[m.get('role') for m in cc_request['messages']]}"
        )


__all__ = (
    "CORPUS",
    "TC4_ENTRY_IDS",
    "_server",
    "_load_entry",
    "_StubLauncher",
    "_StubProvider",
)
