"""Property tests for ``BridgeServer._compact_messages`` (KBR-71, T-F2).

``.system_design/TEST_SUITE.md`` §6.1, second property row: identity below
budget · no orphaned pair · idempotent · **output ≤ budget unless the surviving
set is irreducible** (a small system message plus an oversized final turn, not
only an oversized system block — §6.1, lines 1722–1728).

These four properties exercise both message-content shapes (Chat Completions
and Anthropic-native) by compositing ``tests/harness/transcripts.py``
strategies. The properties do **not** re-prove budget *resolution* — that is
owned by ``tests/test_model_context.py`` and ``test_model_context_packaged_catalog.py``.
They exercise the compactor by passing an explicit ``max_messages_chars`` so
the pruning paths fire on demand.

**The irreducible arm is not optional wording.** P4 states the budget
property as ``size ≤ budget OR structure is irreducible OR raise``; without
the exception the property fails on every generated oversized final turn and
gets weakened. The structural predicate — system messages plus at most one
block — is checked directly on the output list, which avoids re-deriving the
compactor's internal block grouping.

**CompactionFailedError is a sanctioned end-state** (KBR-5's contract: only
corrupt conversations hit this). The strategies in ``transcripts.py`` never
emit orphan tool_results and always close the loop with the answering turn
(R6 of T-F1), so a valid input cannot have *every* non-system message dropped
by pairing validation. P4 asserts "no raise" to pin that contract directly:
a compactor bug that orphans a pair would fire the raise, and the property
catches it.
"""

from __future__ import annotations

import json

import hypothesis.strategies as st
from harness import transcripts as t
from hypothesis import given, settings

from kitty.bridge.server import BridgeServer, CompactionFailedError
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# ── Server stub ─────────────────────────────────────────────────────────────
#
# _compact_messages is an instance method: it calls
# ``self._validate_tool_call_pairing`` and reads ``self._active_model`` /
# ``self._backends`` (the latter only when ``_get_max_context_chars`` is on
# the path, which we do not take — we pass ``max_messages_chars`` explicitly).
# The minimal stub keeps the surface area small and consistent with the
# existing ``tests/bridge/test_compaction.py`` pattern.


class _StubLauncher(LauncherAdapter):
    """Minimal LauncherAdapter stub — no real child process, just satisfies the type."""

    @property
    def name(self) -> str:
        """Return the launcher name used to identify this adapter."""
        return "stub"

    @property
    def binary_name(self) -> str:
        """Return the on-disk name of the agent binary this adapter would spawn."""
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        """Return the bridge-side wire protocol this adapter assumes."""
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(
        self,
        profile: Profile,
        bridge_port: int,
        resolved_key: str,
        *,
        context_tokens: int | None = None,
    ) -> SpawnConfig:
        """Return a SpawnConfig with no env overrides — the compactor never spawns.

        Args:
            profile: The resolved profile (unused).
            bridge_port: Local port the bridge listens on (unused).
            resolved_key: The raw API key resolved from the credential store (unused).
            context_tokens: Optional model context window in tokens (unused).

        Returns:
            A SpawnConfig whose empty overrides mean "inherit the parent process".
        """
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class _StubProvider(ProviderAdapter):
    """Minimal ProviderAdapter stub — chat-completions shaped, no upstream call."""

    @property
    def provider_type(self) -> str:
        """Return the provider type identifier used in routing and config."""
        return "stub"

    @property
    def default_base_url(self) -> str:
        """Return the upstream URL placeholder (the compactor never calls it)."""
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        """Wrap the messages into a Chat-Completions request shape.

        Args:
            model: The model name.
            messages: The conversation list.
            **kwargs: Forwarded options; only ``stream`` is read.

        Returns:
            A CC-shaped request body, never sent on the wire.
        """
        return {"model": model, "messages": messages, "stream": kwargs.get("stream", False)}

    def parse_response(self, response_data: dict) -> dict:
        """Return the response data unchanged — no parsing logic exercised.

        Args:
            response_data: Whatever the upstream returned.

        Returns:
            The same dict, unchanged.
        """
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        """Wrap an upstream error in a generic Exception.

        Args:
            status_code: HTTP status the upstream returned.
            body: The parsed error body.

        Returns:
            An ``Exception`` carrying both fields.
        """
        return Exception(f"Upstream error {status_code}: {body}")


def _server() -> BridgeServer:
    """Return a fresh ``BridgeServer`` wired to the stub launcher/provider.

    A new instance is built per call so property tests do not leak state across
    examples (e.g. cache writes on ``self._active_model``).
    """
    return BridgeServer(_StubLauncher(), _StubProvider(), "test-key")


# ── Pure predicates for the properties ─────────────────────────────────────
#
# Each is a pure function of the messages list, with no dependence on
# ``_compact_messages`` or on the compactor's internal grouping. P2's oracle is
# a reimplementation of the pairing invariant *as seen by the wire*, not a
# call back into the production validator (which the compactor itself runs as
# its last step — testing against the same function proves self-consistency).


def _cc_assistant_tool_call_ids(messages: list[dict]) -> set[str]:
    ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and (tid := tc.get("id")):
                    ids.add(tid)
    return ids


def _cc_tool_result_ids(messages: list[dict]) -> set[str]:
    ids: set[str] = set()
    for m in messages:
        if m.get("role") == "tool" and (tid := m.get("tool_call_id")):
            ids.add(tid)
    return ids


def _native_assistant_tool_use_ids(messages: list[dict]) -> set[str]:
    ids: set[str] = set()
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and (tid := block.get("id")):
                ids.add(tid)
    return ids


def _native_user_tool_result_ids(messages: list[dict]) -> set[str]:
    ids: set[str] = set()
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tid = block.get("tool_use_id")
                if tid:
                    ids.add(tid)
    return ids


def _messages_are_pairing_clean(messages: list[dict]) -> bool:
    """The wire-side oracle P2 uses — the invariant in **both directions**.

    ``results ⊆ uses`` (no orphan tool_result) is the direction the
    compactor's own ``_validate_tool_call_pairing`` postcondition already
    guarantees; a property stating only that proves the validator, not the
    grouping. ``uses ⊆ results`` (no unanswered tool_call) is the direction
    non-atomic grouping breaks — head/tail pruning keeps the assistant's
    ``tool_calls`` and drops the answering message, the validator strips
    nothing (it only removes orphan *results*), and the upstream rejects the
    turn. The T-F1 substrate guarantees both directions on the input (R6), so
    a bidirectional check on the output isolates the compactor's grouping.
    """
    cc_calls = _cc_assistant_tool_call_ids(messages)
    cc_results = _cc_tool_result_ids(messages)
    native_uses = _native_assistant_tool_use_ids(messages)
    native_results = _native_user_tool_result_ids(messages)
    return (
        cc_results.issubset(cc_calls)
        and cc_calls.issubset(cc_results)
        and native_results.issubset(native_uses)
        and native_uses.issubset(native_results)
    )


def _serialize(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=False))


def _surviving_block_count(messages: list[dict]) -> int:
    """Count blocks in the compactor's sense: 0 or 1, the irreducible floor.

    A block is a single message, or a CC assistant+tools run, or a native
    assistant(tool_use) absorbing its single following user(tool_result). The
    compactor's guaranteed-fit loop ``break``s when only the system message and
    one such block remain — structural irreducibility is what the §6.1
    exception names.

    Args:
        messages: The compactor's output (or any list with the same shape).

    Returns:
        0 when the list is system-only, 1 when a single block survives, ≥2
        otherwise. The property P4 checks ``≤ 1``.
    """
    # System messages live at the front (the compactor only preserves the
    # first system block at index 0). A mid-conversation system message would
    # become a regular block — the T-F1 strategies never emit one, and the
    # properties prepend exactly zero or one.
    body = [m for m in messages if m.get("role") != "system"]
    if not body:
        return 0

    # First block spans an assistant(tool_calls) plus consecutive role=="tool".
    first = body[0]
    if first.get("role") == "assistant" and first.get("tool_calls"):
        j = 1
        while j < len(body) and body[j].get("role") == "tool":
            j += 1
        block = body[:j]
    # Native: assistant with tool_use content blocks absorbs the one following
    # user(tool_result).
    elif first.get("role") == "assistant" and isinstance(first.get("content"), list):
        if any(isinstance(b, dict) and b.get("type") == "tool_use" for b in first["content"]):
            j = 1
            next_is_tool_result_user = (
                j < len(body)
                and body[j].get("role") == "user"
                and isinstance(body[j].get("content"), list)
                and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in body[j]["content"])
            )
            if next_is_tool_result_user:
                j += 1
            block = body[:j]
        else:
            block = body[:1]
    else:
        block = body[:1]

    return 1 if block == body else 2  # "≥ 2" represented as 2 for the property's `> 1` check.


# ── Property: identity below budget ────────────────────────────────────────
#
# §6.1 first row: a conversation whose serialized size is at or below the
# budget passes through unchanged. The check is at the compactor's own
# threshold (step 0), so no truncation, no grouping, no fallback touches
# the messages. Both shapes are exercised: the substring property is purely
# a function of size, not of content shape.


class TestIdentityBelowBudget:
    """P1: messages whose serialized size is at or below the budget are untouched."""

    @given(t.cc_request())
    @settings(max_examples=200)
    def test_cc_bodies_below_budget_are_unchanged(self, body: dict) -> None:
        messages = body["messages"]
        size = _serialize(messages)
        result = _server()._compact_messages(list(messages), max_messages_chars=size)
        assert result == messages

    @given(t.messages_request())
    @settings(max_examples=200)
    def test_messages_bodies_below_budget_are_unchanged(self, body: dict) -> None:
        messages = body["messages"]
        size = _serialize(messages)
        result = _server()._compact_messages(list(messages), max_messages_chars=size)
        assert result == messages


# ── Property: no orphaned pair (P2) ─────────────────────────────────────────
#
# The compactor's guarantee is that the wire never carries an orphan
# tool_result. Pairing validation does the stripping, but it is the compactor's
# *grouping* that decides whether validation has work to do: a grouping bug
# would orphan a pair, validation would strip it, and the property would catch
# the bug at the wire. The forcing budget makes every generated conversation
# need compaction, which is when grouping choices matter.


_FORCING_BUDGET_CC = st.integers(min_value=10, max_value=200)
_FORCING_BUDGET_MESSAGES = st.integers(min_value=10, max_value=200)


class TestNoOrphanedPair:
    """P2: every output tool_result references a surviving tool_use."""

    @given(t.cc_request(), _FORCING_BUDGET_CC)
    @settings(max_examples=200)
    def test_cc_output_is_pairing_clean(self, body: dict, budget: int) -> None:
        messages = body["messages"]
        result = _server()._compact_messages(list(messages), max_messages_chars=budget)
        assert _messages_are_pairing_clean(result)

    @given(t.messages_request(), _FORCING_BUDGET_MESSAGES)
    @settings(max_examples=200)
    def test_messages_output_is_pairing_clean(self, body: dict, budget: int) -> None:
        messages = body["messages"]
        result = _server()._compact_messages(list(messages), max_messages_chars=budget)
        assert _messages_are_pairing_clean(result)


# ── Property: idempotence (P3) ──────────────────────────────────────────────
#
# Compacting the output of a compaction is a no-op (same list, modulo identity).
# Trivially true when the first compaction met its budget (the second call
# short-circuits at step 0). Non-trivially true in the irreducible arm: the
# second call re-runs the loop on an already-irrreducible set, and the loop
# must again ``break`` rather than drop the last block.


class TestIdempotence:
    """P3: ``compact(compact(x))`` is the same list as ``compact(x)``.

    Trivially true when the first compaction met its budget (the second call
    short-circuits at step 0). Non-trivially true in the irreducible arm: the
    second call re-runs the loop on an already-irreducible set, and the loop
    must again ``break`` rather than drop the last block. The invariant that
    makes this hold: the single surviving tail block's size exceeds
    ``int(0.2·budget) − system_size`` for any budget used here (it is, by
    irreducibility, larger than the whole budget), so the second pass
    re-classifies it as tail — never head — and the fallback has nothing to
    pop. A change to the head/tail accounting that breaks that placement
    breaks this property, by design.
    """

    @given(t.cc_request(), _FORCING_BUDGET_CC)
    @settings(max_examples=200)
    def test_cc_second_compaction_is_a_no_op(self, body: dict, budget: int) -> None:
        messages = body["messages"]
        first = _server()._compact_messages(list(messages), max_messages_chars=budget)
        second = _server()._compact_messages(list(first), max_messages_chars=budget)
        assert second == first

    @given(t.messages_request(), _FORCING_BUDGET_MESSAGES)
    @settings(max_examples=200)
    def test_messages_second_compaction_is_a_no_op(self, body: dict, budget: int) -> None:
        messages = body["messages"]
        first = _server()._compact_messages(list(messages), max_messages_chars=budget)
        second = _server()._compact_messages(list(first), max_messages_chars=budget)
        assert second == first


# ── Property: output ≤ budget unless the surviving set is irreducible (P4) ─
#
# The compactor exceeds the budget only when its guaranteed-fit loop has
# nothing left to drop. The property is stated as a disjunction: fits the
# budget, or is structurally irreducible (system messages plus at most one
# block), or raises the sanctioned CompactionFailedError (which T-F1's
# pairing-clean inputs never trigger, so the raise arm is pinned separately
# as "does not happen on valid input").


def _budget_or_irreducible_or_raises(messages_in: list[dict], budget: int) -> None:
    """Assert P4 holds for a given input + budget."""
    server = _server()
    try:
        result = server._compact_messages(list(messages_in), max_messages_chars=budget)
    except CompactionFailedError:
        # Sanctioned end-state: the input had non-system messages that
        # pairing validation would have to drop. T-F1 inputs are pairing-
        # clean, so this never fires under the property tests, but the
        # disjunction still names it so a future corruption is acceptable.
        return
    size = _serialize(result)
    if size <= budget:
        return
    # Over-budget arm — only the irreducible case is permitted.
    assert _surviving_block_count(result) <= 1, (
        f"compactor left an over-budget output with >1 surviving block(s); "
        f"size={size}, budget={budget}, output={result!r}"
    )


# ── Constructed irreducible case ────────────────────────────────────────────
#
# §6.1 warns that "a property stated without the exception fails on the first
# run and gets weakened by whoever is on the rota". The constructed case is
# the falsification control: small system + one oversized final user turn
# must land in the irreducible arm (over budget AND structure is system + 1
# block), exercising the exception by construction so it cannot be silently
# dropped from the property.


def _small_system_plus_oversized_turn() -> list[dict]:
    """The §6.1 example: a small system message and one oversized final user turn."""
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "x" * 500_000},
    ]


class TestBudgetOrIrreducibleOrRaises:
    """P4: output ≤ budget, or structurally irreducible, or raises (sanctioned)."""

    @given(t.cc_request(), _FORCING_BUDGET_CC)
    @settings(max_examples=200)
    def test_cc_output_is_within_budget_or_irreducible_or_raises(
        self, body: dict, budget: int
    ) -> None:
        _budget_or_irreducible_or_raises(body["messages"], budget)

    @given(t.messages_request(), _FORCING_BUDGET_MESSAGES)
    @settings(max_examples=200)
    def test_messages_output_is_within_budget_or_irreducible_or_raises(
        self, body: dict, budget: int
    ) -> None:
        _budget_or_irreducible_or_raises(body["messages"], budget)

    def test_constructed_oversized_final_turn_lands_in_the_irreducible_arm(self) -> None:
        """§6.1 falsification control: small system + one big final turn → irreducible."""
        messages = _small_system_plus_oversized_turn()
        budget = 10_000  # ~50× smaller than the final turn; well below the body size.
        server = _server()
        result = server._compact_messages(messages, max_messages_chars=budget)

        # The output must be over budget (otherwise P4's first arm hides the
        # irreducible arm).
        assert _serialize(result) > budget
        # And the structure must be system + at most one block.
        assert _surviving_block_count(result) <= 1
        # And the system message must survive.
        assert result and result[0].get("role") == "system"

    @given(t.cc_request(), _FORCING_BUDGET_CC)
    @settings(max_examples=50)
    def test_cc_valid_input_never_raises_compaction_failed(
        self, body: dict, budget: int
    ) -> None:
        """P4-raise-arm-pinning — CC shape: valid input never raises.

        T-F1 strategies never emit orphan tool_results; a grouping bug that
        let orphans slip through pairing validation would drop every non-
        system message and fire this raise. ``P4``'s helper accepts the raise
        (KBR-5's sanctioned end-state), so the no-raise guarantee must be
        pinned separately to catch that failure mode.
        """
        _server()._compact_messages(list(body["messages"]), max_messages_chars=budget)

    @given(t.messages_request(), _FORCING_BUDGET_MESSAGES)
    @settings(max_examples=50)
    def test_messages_valid_input_never_raises_compaction_failed(
        self, body: dict, budget: int
    ) -> None:
        """P4-raise-arm-pinning — native shape: valid input never raises."""
        _server()._compact_messages(list(body["messages"]), max_messages_chars=budget)
