"""Property tests for pairing and truncation at L1 (KBR-72 / T-F3).

`.system_design/TEST_SUITE.md` §6.1 (third and fourth property rows) with the
extended scope for the Responses-shape twins (the design-review finding the
owner decided on 2026-09-16). Owns four claims:

* ``BridgeServer._validate_tool_call_pairing`` — output contains no
  ``tool_result`` lacking a preceding declaring ``tool_use``, in the
  Chat Completions shape and the Anthropic-native shape; paired results and
  all other messages survive structurally identical and in input order.
* ``BridgeServer._truncate_oversized_tool_results`` — identity below the
  50,000-char limit, bounded above it (notice text exact, count accurate
  against a pre-snapshot oracle), non-tool content untouched.
* ``BridgeServer._drop_orphan_responses_tool_outputs`` — output contains
  no ``function_call_output`` without a preceding declaring
  ``function_call``; paired outputs and other items survive structurally
  identical and in input order.
* ``BridgeServer._truncate_oversized_responses_outputs`` — the same
  truncation rules on the Responses ``input`` item list (string outputs over
  the limit become the same notice; message items, ``reasoning`` items,
  and list-form outputs are untouched).

The substrate (KBR-70) provides valid Anthropic Messages and Chat
Completions request bodies; the T-F3 extension (KBR-72, this ticket)
provides valid OpenAI Responses request bodies. Local composite strategies
in this file inject orphans and over-limit content into otherwise valid
bodies — the substrate docstring carves this out explicitly: invalid bodies
are produced downstream by mutating valid bodies; the shared strategies
never emit them.

**Layer.** No ``pytestmark``, so this file falls back to ``l1`` via
``tests/layers.py:_FALLBACK_LAYER``. A future table addition that adds a
``tests/``-prefixed row would silently re-layer the file, so the gate's
layer-count guard is the tripwire. Same marker posture as the shipped
sibling ``test_compaction_properties.py``.

**Plan §1.4 harness rule.** Eleven constructed example tests
(``TestOracleFalsification``) feed each oracle a deliberately wrong output
and assert the property predicate or equality **rejects** it. Every
constructed case's expected output is hand-written (a literal list, not
derived from the oracle), and each case additionally asserts
``oracle(input) == hand_written_expected`` so it doubles as an *oracle*
falsification control — an oracle that mis-implements the wire rule cannot
agree with itself through the generated property and the constructed case.

**Terminology.** "Byte-identical" throughout this file means *structural
equality between a ``deepcopy`` snapshot taken immediately before the call
and the state after it* (``pre == post`` on the snapshot), not a byte-level
comparison.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any

import hypothesis.strategies as st
from harness import transcripts as t
from hypothesis import assume, given, settings

from kitty.bridge.server import (
    _TOOL_RESULT_TRUNCATION_LIMIT,
    BridgeServer,
)
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# ── Server stub ─────────────────────────────────────────────────────────────
#
# Both validators and the truncation pass are instance methods on
# ``BridgeServer``. The minimal stub keeps the surface area small and
# consistent with the sibling ``tests/bridge/test_compaction_properties.py``
# pattern (T-F2, KBR-71), which also defines its stubs at module level.


class _StubLauncher(LauncherAdapter):
    """Minimal ``LauncherAdapter`` — no real child process, just satisfies the type."""

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
        """Return a SpawnConfig with no env overrides — the validators never spawn.

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
    """Minimal ``ProviderAdapter`` — chat-completions shaped, no upstream call."""

    @property
    def provider_type(self) -> str:
        """Return the provider type identifier used in routing and config."""
        return "stub"

    @property
    def default_base_url(self) -> str:
        """Return the upstream URL placeholder (the validators never call it)."""
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

    A new instance per call so property tests do not leak state across
    examples (e.g. cache writes on ``self._active_model``). Mirror of
    the T-F2 sibling's ``_server()``.

    Returns:
        A ``BridgeServer`` constructed with no real launcher child process
        and no real upstream provider — sufficient to call the validators
        and the truncation pass, which use neither network nor file IO.
    """
    return BridgeServer(_StubLauncher(), _StubProvider(), "test-key")


# ── Helpers ─────────────────────────────────────────────────────────────────
#
# Light, local text strategies. The substrate's ``_short_ascii_text`` is
# private; the property tests need both *short* content (orphan injection)
# and *long* content (truncation over the limit). Importing private names
# would couple the property file to the substrate's internals (the
# substrate's own docstring forbids that for downstream consumers), so
# these are local.


def _short_text(min_size: int = 0, max_size: int = 16) -> st.SearchStrategy[str]:
    """Return a strategy over short JSON-safe text strings.

    Args:
        min_size: Minimum character count.
        max_size: Maximum character count.

    Returns:
        A strategy yielding plain ``str`` values, JSON-safe by construction.
    """
    return st.text(alphabet=st.characters(max_codepoint=0x7E), min_size=min_size, max_size=max_size)


def _tool_message(tool_call_id: str) -> st.SearchStrategy[dict[str, Any]]:
    """Return a Chat Completions ``role: tool`` message with the given id.

    Args:
        tool_call_id: The id of the assistant ``tool_calls`` entry this
            message answers (or, for orphan injection, a freshly drawn
            uuid that no assistant declared).

    Returns:
        A strategy yielding
        ``{"role": "tool", "tool_call_id": <id>, "content": <str>}``.
    """
    return st.builds(
        lambda text: {"role": "tool", "tool_call_id": tool_call_id, "content": text},
        text=_short_text(),
    )


def _native_orphan_message_factory(orphan_id: str) -> st.SearchStrategy[dict[str, Any]]:
    """Build a one-block user message carrying the given orphan id.

    Args:
        orphan_id: The ``tool_use_id`` on the orphan block. Must be a
            fresh UUID excluded from the declared set so the validator
            must drop the entire message.

    Returns:
        A strategy yielding
        ``{"role": "user", "content": [orphan_block]}``.
    """
    return st.builds(
        lambda text: {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": orphan_id, "content": text}],
        },
        text=_short_text(),
    )


def _native_mixed_message_with_id(text: str, orphan: str, orphan_id: str) -> dict[str, Any]:
    """Build a mixed user message carrying text and the given orphan id.

    Args:
        text: The text for the surviving (non-orphan) text block.
        orphan: The text for the orphan ``tool_result`` block.
        orphan_id: The ``tool_use_id`` on the orphan block.

    Returns:
        A user message whose ``content`` is a two-block list — one text
        block that survives pairing and one orphan ``tool_result`` block
        that the validator strips. The message itself survives because
        the rebuild is non-empty.
    """
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "tool_result", "tool_use_id": orphan_id, "content": orphan},
        ],
    }


# ── Local composite strategies ─────────────────────────────────────────────
#
# These mutate valid bodies from the substrate. They are local to this
# file on purpose — the substrate's docstring carves this out, and keeping
# the orphan vocabulary next to the properties that consume it lets a
# reader find the mutation shape where they find the property.


@st.composite
def _cc_with_orphan(draw: st.DrawFn) -> list[dict[str, Any]]:
    """Draw a valid CC transcript with one orphan tool message injected.

    The orphan's ``tool_call_id`` is a fresh UUID excluded from the
    declared set via ``assume``, so the validator must drop it.

    Returns:
        The transcript's ``messages`` list with one orphan tool message
        inserted at a drawn position.
    """
    body = draw(t.cc_request())
    messages: list[dict[str, Any]] = list(body["messages"])
    declared_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    declared_ids.add(tc["id"])
    orphan_id = str(uuid.uuid4())
    assume(orphan_id not in declared_ids)
    position = draw(st.integers(min_value=0, max_value=len(messages)))
    orphan_msg = draw(_tool_message(orphan_id))
    messages.insert(position, orphan_msg)
    return messages


@st.composite
def _native_with_orphan(draw: st.DrawFn) -> list[dict[str, Any]]:
    """Draw a valid native transcript with one orphan ``tool_result`` injected.

    Two variants are drawn per example; the run covers both arms many
    times. The orphan's ``tool_use_id`` is a fresh UUID excluded from the
    declared set via ``assume``.
    """
    body = draw(t.messages_request())
    messages: list[dict[str, Any]] = list(body["messages"])
    declared_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            for block in m["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and (tid := block.get("id"))
                ):
                        declared_ids.add(tid)
    orphan_id = str(uuid.uuid4())
    assume(orphan_id not in declared_ids)
    variant = draw(st.sampled_from(("new_message", "mixed")))
    position = draw(st.integers(min_value=0, max_value=len(messages)))
    if variant == "new_message":
        messages.insert(position, draw(_native_orphan_message_factory(orphan_id)))
    else:
        text = draw(_short_text())
        orphan = draw(_short_text())
        messages.insert(position, _native_mixed_message_with_id(text, orphan, orphan_id))
    return messages


@st.composite
def _responses_with_orphan(draw: st.DrawFn) -> dict[str, Any]:
    """Draw a valid Responses body with one orphan ``function_call_output`` injected.

    The orphan's ``call_id`` is a fresh UUID excluded from the declared
    set via ``assume``.
    """
    body = draw(t.responses_request())
    items: list[dict[str, Any]] = list(body["input"])
    declared_ids: set[str] = {
        item["call_id"]
        for item in items
        if isinstance(item, dict)
        and item.get("type") == "function_call"
        and isinstance(item.get("call_id"), str)
    }
    orphan_id = str(uuid.uuid4())
    assume(orphan_id not in declared_ids)
    orphan_item = {
        "type": "function_call_output",
        "call_id": orphan_id,
        "output": draw(_short_text()),
    }
    position = draw(st.integers(min_value=0, max_value=len(items)))
    items.insert(position, orphan_item)
    out = dict(body)
    out["input"] = items
    return out


# ── Truncation-local mutation strategies ──────────────────────────────────
#
# These build bodies whose tool-result string contents sit in a chosen
# length band: at-or-below the limit, or over it. They start from valid
# substrate bodies and rewrite content in place, leaving every other field
# intact so the property can compare structural equality against a
# pre-call snapshot.


def _length_in_band(lower: int, upper: int) -> st.SearchStrategy[int]:
    """Draw an integer length in the closed interval ``[lower, upper]``."""
    return st.integers(min_value=lower, max_value=upper)


def _resize_string_content(value: dict[str, Any], path: list[str], length: int) -> None:
    """Replace the string content at ``path`` inside ``value`` with ``"x" * length``.

    Used by the truncation composites to size content for the chosen band.
    Non-target fields are untouched; non-string content is left alone (the
    truncation function does not rewrite it; the constructed survival
    case pins this contract).
    """
    node: Any = value
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = "x" * length


# ── Oracles ─────────────────────────────────────────────────────────────────
#
# Each oracle states the wire rule in code. Equality against the oracle is
# the property assertion; the predicate ("no orphans", "all strings ≤
# limit") is stated redundantly for legible failures.


def _expected_cc(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the Chat Completions wire rule: drop orphan tool messages.

    The rule is: walk messages, accumulate declared tool_call ids from
    assistant ``tool_calls`` entries, drop every ``role=="tool"`` message
    whose ``tool_call_id`` is not in the seen set, keep everything else.

    Note: the oracle's ``isinstance(tid, str) and tid in seen`` is stricter
    than the SUT's ``tool_call_id and tool_call_id in seen_tool_use_ids``
    — a non-string ``tool_call_id`` the SUT would drop by falsiness but
    that happens to be in the declared set would be rejected here too.
    The generated space only emits uuid4 strings, so this divergence is
    unreachable today; if the substrate ever starts emitting non-string
    ids, the property would false-fail on the divergence, which is the
    desired early-warning behaviour.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and (tid := tc.get("id")):
                    seen.add(tid)
            out.append(m)
            continue
        if m.get("role") == "tool":
            tid = m.get("tool_call_id")
            if isinstance(tid, str) and tid in seen:
                out.append(m)
            continue
        out.append(m)
    return out


def _expected_native(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the Anthropic-native wire rule: strip orphan tool_result blocks.

    The rule: walk messages, accumulate declared tool_use ids from
    assistant content blocks, then for each user message with list content
    rebuild the content keeping only tool_result blocks whose tool_use_id
    is in the seen set. Drop the message entirely when the rebuild is empty.
    Other messages pass through unchanged.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            for block in m["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and (tid := block.get("id"))
                ):
                        seen.add(tid)
            out.append(m)
            continue
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            if not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"]
            ):
                out.append(m)
                continue
            rebuilt: list[dict[str, Any]] = []
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tid = block.get("tool_use_id")
                    if isinstance(tid, str) and tid in seen:
                        rebuilt.append(block)
                else:
                    rebuilt.append(block)
            if rebuilt:
                out.append({**m, "content": rebuilt})
            continue
        out.append(m)
    return out


def _expected_responses_paired(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the Responses wire rule: drop orphan ``function_call_output`` items.

    Mirrors ``_drop_orphan_response_outputs`` in the production code; the
    oracle is a literal restatement, not a call into the SUT.
    """
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        kind = item.get("type")
        if kind == "function_call":
            call_id = item.get("call_id")
            if isinstance(call_id, str):
                seen.add(call_id)
            kept.append(item)
        elif kind == "function_call_output":
            call_id = item.get("call_id")
            if isinstance(call_id, str) and call_id in seen:
                kept.append(item)
        else:
            kept.append(item)
    return kept


# ── Truncation oracles (apply the notice replacement on a deep copy) ─────


def _apply_cc_truncation(request: dict[str, Any]) -> dict[str, Any]:
    """Apply the Chat Completions truncation rule to a deep copy."""
    out = copy.deepcopy(request)
    for msg in out.get("messages") or []:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "tool"
            and isinstance(msg.get("content"), str)
            and len(msg["content"]) > _TOOL_RESULT_TRUNCATION_LIMIT
        ):
                msg["content"] = (
                    f"[Tool output truncated — original size: {len(msg['content']):,} chars]"
                )
    return out


def _apply_native_truncation(request: dict[str, Any]) -> dict[str, Any]:
    """Apply the Anthropic-native truncation rule to a deep copy."""
    out = copy.deepcopy(request)
    for msg in out.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and isinstance(block.get("content"), str)
                    and len(block["content"]) > _TOOL_RESULT_TRUNCATION_LIMIT
                ):
                    original = block["content"]
                    block["content"] = (
                        f"[Tool output truncated — original size: {len(original):,} chars]"
                    )
    return out


def _apply_responses_truncation(body: dict[str, Any]) -> dict[str, Any]:
    """Apply the Responses truncation rule to a deep copy."""
    out = copy.deepcopy(body)
    for item in out.get("input") or []:
        if (
            isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and isinstance(item.get("output"), str)
            and len(item["output"]) > _TOOL_RESULT_TRUNCATION_LIMIT
        ):
            original = item["output"]
            item["output"] = (
                f"[Tool output truncated — original size: {len(original):,} chars]"
            )
    return out


def _expected_truncation_count(request_or_body: dict[str, Any], shape: str) -> int:
    """Count over-limit string tool-result contents in the pre-snapshot.

    Args:
        request_or_body: The Chat Completions ``request`` (or Responses
            ``body``) snapshot.
        shape: One of ``"cc"``, ``"native"``, ``"responses"`` — selects
            which item shape to inspect.

    Returns:
        The number of string tool-result contents exceeding the limit.
        Independent of the SUT's bookkeeping, so a miscounting mutant
        cannot agree with itself.
    """
    if shape == "cc":
        n = 0
        for msg in request_or_body.get("messages") or []:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "tool"
                and isinstance(msg.get("content"), str)
                and len(msg["content"]) > _TOOL_RESULT_TRUNCATION_LIMIT
            ):
                    n += 1
        return n
    if shape == "native":
        n = 0
        for msg in request_or_body.get("messages") or []:
            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and isinstance(block.get("content"), str)
                        and len(block["content"]) > _TOOL_RESULT_TRUNCATION_LIMIT
                    ):
                        n += 1
        return n
    n = 0
    for item in request_or_body.get("input") or []:
        if (
            isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and isinstance(item.get("output"), str)
            and len(item["output"]) > _TOOL_RESULT_TRUNCATION_LIMIT
        ):
            n += 1
    return n
# ── Truncation composites (local mutation strategies) ─────────────────────
#
# The truncation property needs bodies whose tool-result string contents
# sit in a chosen length band. The substrate's request composites carry
# pairs with probability ½ per assistant turn (1–3 turns), so a drawn body
# has a small but nonzero chance of carrying no tool result. Rather than
# ``assume``-filter examples (which can trip Hypothesis's filter health
# checks under heavy mutation), every truncation composite *appends* a
# guaranteed-valid pair to the drawn body, so the function-under-test has
# something to truncate in every example.


_LIMIT = _TOOL_RESULT_TRUNCATION_LIMIT


def _cc_stub_tools() -> list[dict[str, Any]]:
    """Return a fresh one-tool Chat Completions ``tools`` list for constructed requests."""
    return [
        {
            "type": "function",
            "function": {
                "name": "stub",
                "description": "x",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _native_stub_tools() -> list[dict[str, Any]]:
    """Return a fresh one-tool Anthropic-native ``tools`` list for constructed requests."""
    return [
        {
            "name": "stub",
            "description": "x",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def _responses_stub_tools() -> list[dict[str, Any]]:
    """Return a fresh one-tool Responses ``tools`` list for constructed bodies."""
    return [
        {
            "type": "function",
            "name": "stub",
            "description": "x",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def _make_cc_pair(tool_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a fresh assistant + tool message pair addressed to ``tool_name``."""
    call_id = str(uuid.uuid4())
    return (
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "x"},
    )


def _make_native_pair(tool_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a fresh assistant + user(tool_result) pair for an Anthropic-native body."""
    call_id = str(uuid.uuid4())
    return (
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": call_id, "name": tool_name, "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": "x"}],
        },
    )


def _make_responses_pair(tool_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a fresh function_call + function_call_output pair."""
    call_id = str(uuid.uuid4())
    return (
        {
            "type": "function_call",
            "call_id": call_id,
            "name": tool_name,
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": "x",
        },
    )


def _declared_tool_name_cc(body: dict[str, Any]) -> str:
    """Return the name of the first declared tool in a CC body."""
    for tool in body.get("tools") or []:
        if isinstance(tool, dict):
            func = tool.get("function") or {}
            if isinstance(func, dict):
                name = func.get("name")
                if isinstance(name, str):
                    return name
    return "stub"


def _declared_tool_name_responses(body: dict[str, Any]) -> str:
    """Return the name of the first declared tool in a Responses body."""
    for tool in body.get("tools") or []:
        if isinstance(tool, dict):
            name = tool.get("name")
            if isinstance(name, str):
                return name
    return "stub"


@st.composite
def _cc_with_oversize_tool_results(draw: st.DrawFn) -> dict[str, Any]:
    """Draw a CC body with at least one tool-result content over the limit."""
    body = draw(t.cc_request())
    messages: list[dict[str, Any]] = list(body["messages"])
    tool_name = _declared_tool_name_cc(body)
    assistant_msg, tool_msg = _make_cc_pair(tool_name)
    length = draw(st.integers(min_value=_LIMIT + 1, max_value=150_000))
    tool_msg["content"] = "x" * length
    messages.append(assistant_msg)
    messages.append(tool_msg)
    out = dict(body)
    out["messages"] = messages
    return out


@st.composite
def _cc_below_limit(draw: st.DrawFn) -> dict[str, Any]:
    """Draw a CC body whose tool-result contents all sit in ``[0, _LIMIT]``."""
    body = draw(t.cc_request())
    messages: list[dict[str, Any]] = list(body["messages"])
    tool_name = _declared_tool_name_cc(body)
    assistant_msg, tool_msg = _make_cc_pair(tool_name)
    length = draw(st.integers(min_value=0, max_value=_LIMIT))
    tool_msg["content"] = "x" * length
    messages.append(assistant_msg)
    messages.append(tool_msg)
    out = dict(body)
    out["messages"] = messages
    return out


@st.composite
def _native_with_oversize_tool_results(draw: st.DrawFn) -> dict[str, Any]:
    """Draw a native body with at least one tool_result content over the limit."""
    body = draw(t.messages_request())
    messages: list[dict[str, Any]] = list(body["messages"])
    assistant_msg, tool_msg = _make_native_pair("StubTool")
    block = tool_msg["content"][0]
    length = draw(st.integers(min_value=_LIMIT + 1, max_value=150_000))
    block["content"] = "x" * length
    messages.append(assistant_msg)
    messages.append(tool_msg)
    out = dict(body)
    out["messages"] = messages
    return out


@st.composite
def _native_below_limit(draw: st.DrawFn) -> dict[str, Any]:
    """Draw a native body whose tool_result contents all sit in ``[0, _LIMIT]``."""
    body = draw(t.messages_request())
    messages: list[dict[str, Any]] = list(body["messages"])
    assistant_msg, tool_msg = _make_native_pair("StubTool")
    block = tool_msg["content"][0]
    length = draw(st.integers(min_value=0, max_value=_LIMIT))
    block["content"] = "x" * length
    messages.append(assistant_msg)
    messages.append(tool_msg)
    out = dict(body)
    out["messages"] = messages
    return out


@st.composite
def _responses_with_oversize_output(draw: st.DrawFn) -> dict[str, Any]:
    """Draw a Responses body with at least one ``function_call_output`` over the limit."""
    body = draw(t.responses_request())
    items: list[dict[str, Any]] = list(body["input"])
    tool_name = _declared_tool_name_responses(body)
    call_item, output_item = _make_responses_pair(tool_name)
    length = draw(st.integers(min_value=_LIMIT + 1, max_value=150_000))
    output_item["output"] = "x" * length
    items.append(call_item)
    items.append(output_item)
    out = dict(body)
    out["input"] = items
    return out


@st.composite
def _responses_below_limit(draw: st.DrawFn) -> dict[str, Any]:
    """Draw a Responses body whose outputs all sit in ``[0, _LIMIT]``."""
    body = draw(t.responses_request())
    items: list[dict[str, Any]] = list(body["input"])
    tool_name = _declared_tool_name_responses(body)
    call_item, output_item = _make_responses_pair(tool_name)
    length = draw(st.integers(min_value=0, max_value=_LIMIT))
    output_item["output"] = "x" * length
    items.append(call_item)
    items.append(output_item)
    out = dict(body)
    out["input"] = items
    return out


# ── Property: pairing (CC, native, Responses) ──────────────────────────────


class TestPairingCC:
    """R1: Chat Completions pairing — output contains no orphan tool message."""

    @given(_cc_with_orphan())
    @settings(max_examples=200)
    def test_cc_validator_output_matches_the_wire_rule_oracle(
        self, messages: list[dict[str, Any]]
    ) -> None:
        """Equality against the CC wire-rule oracle (R1c).

        The §6.1 row is "output contains no ``tool_result`` without a
        ``tool_use``"; the equality subsumes it and adds the preservation
        direction (paired results and other messages survive structurally
        identical and in input order).
        """
        result = _server()._validate_tool_call_pairing(list(messages))
        assert result == _expected_cc(messages), (
            f"validator output differs from oracle; "
            f"len(in)={len(messages)}, len(out)={len(result)}"
        )

    @given(_cc_with_orphan())
    @settings(max_examples=200)
    def test_cc_validator_output_has_no_orphan_tool_message(
        self, messages: list[dict[str, Any]]
    ) -> None:
        """§6.1 row stated verbatim: every tool message references a preceding call."""
        result = _server()._validate_tool_call_pairing(list(messages))
        seen: set[str] = set()
        for m in result:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    if isinstance(tc, dict) and (tid := tc.get("id")):
                        seen.add(tid)
            elif m.get("role") == "tool":
                tid = m.get("tool_call_id")
                assert isinstance(tid, str) and tid in seen, (
                    f"orphan tool message survived: tool_call_id={tid!r}"
                )


class TestPairingNative:
    """R2: Anthropic-native pairing — output contains no orphan tool_result block."""

    @given(_native_with_orphan())
    @settings(max_examples=200)
    def test_native_validator_output_matches_the_wire_rule_oracle(
        self, messages: list[dict[str, Any]]
    ) -> None:
        """Equality against the native wire-rule oracle (R2)."""
        result = _server()._validate_tool_call_pairing(list(messages))
        assert result == _expected_native(messages)

    @given(_native_with_orphan())
    @settings(max_examples=200)
    def test_native_validator_output_has_no_orphan_tool_result_block(
        self, messages: list[dict[str, Any]]
    ) -> None:
        """§6.1 row stated verbatim for the native shape."""
        result = _server()._validate_tool_call_pairing(list(messages))
        seen: set[str] = set()
        for m in result:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "assistant" and isinstance(m.get("content"), list):
                for block in m["content"]:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_use"
                        and (tid := block.get("id"))
                    ):
                        seen.add(tid)
            elif m.get("role") == "user" and isinstance(m.get("content"), list):
                for block in m["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tid = block.get("tool_use_id")
                        assert isinstance(tid, str) and tid in seen, (
                            f"orphan tool_result block survived: tool_use_id={tid!r}"
                        )


class TestPairingResponses:
    """R12: Responses pairing — output contains no orphan ``function_call_output``."""

    @given(_responses_with_orphan())
    @settings(max_examples=200)
    def test_responses_pairing_output_matches_the_wire_rule_oracle(
        self, body: dict[str, Any]
    ) -> None:
        """Equality against the Responses wire-rule oracle (R12)."""
        snapshot = copy.deepcopy(body)
        dropped = _server()._drop_orphan_responses_tool_outputs(body)
        assert dropped == 1, (
            f"exactly one orphan was injected; wrapper reported {dropped} dropped"
        )
        assert body["input"] == _expected_responses_paired(snapshot["input"])

    @given(_responses_with_orphan())
    @settings(max_examples=200)
    def test_responses_pairing_output_has_no_orphan_output_item(
        self, body: dict[str, Any]
    ) -> None:
        """§6.1 row extended to the Responses shape."""
        _server()._drop_orphan_responses_tool_outputs(body)
        seen: set[str] = set()
        for item in body["input"]:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "function_call":
                if isinstance(call_id := item.get("call_id"), str):
                    seen.add(call_id)
            elif kind == "function_call_output":
                call_id = item.get("call_id")
                assert isinstance(call_id, str) and call_id in seen, (
                    f"orphan function_call_output survived: call_id={call_id!r}"
                )


# ── Property: truncation identity below the limit (R3) ─────────────────────


class TestTruncationIdentity:
    """R3: tool-result contents at-or-below the limit survive byte-identical."""

    @given(_cc_below_limit())
    @settings(max_examples=200)
    def test_cc_below_limit_is_a_no_op(self, request: dict[str, Any]) -> None:
        """CC: every tool-result content ≤ limit ⇒ request unchanged, count 0."""
        snapshot = copy.deepcopy(request)
        count = _server()._truncate_oversized_tool_results(request)
        assert count == 0
        assert request == snapshot

    @given(_native_below_limit())
    @settings(max_examples=200)
    def test_native_below_limit_is_a_no_op(self, request: dict[str, Any]) -> None:
        """Native: every tool_result.content ≤ limit ⇒ request unchanged, count 0."""
        snapshot = copy.deepcopy(request)
        count = _server()._truncate_oversized_tool_results(request)
        assert count == 0
        assert request == snapshot

    @given(_responses_below_limit())
    @settings(max_examples=200)
    def test_responses_below_limit_is_a_no_op(self, body: dict[str, Any]) -> None:
        """Responses: every function_call_output.output ≤ limit ⇒ body unchanged, count 0."""
        snapshot = copy.deepcopy(body)
        count = _server()._truncate_oversized_responses_outputs(body)
        assert count == 0
        assert body == snapshot


# ── Property: truncation bounded above the limit (R4) ─────────────────────


class TestTruncationBounded:
    """R4: every tool-result content ≤ limit after; notice exact; count accurate."""

    @given(_cc_with_oversize_tool_results())
    @settings(max_examples=200)
    def test_cc_over_limit_truncated_to_notice(self, request: dict[str, Any]) -> None:
        """CC: at least one tool-result content > limit ⇒ rewritten to exact notice."""
        snapshot = copy.deepcopy(request)
        expected_count = _expected_truncation_count(snapshot, "cc")
        # Hold references to message dicts whose ``role=='tool'`` content was
        # over the limit BEFORE the call — proves in-place mutation (R4d).
        pre_targets = [
            msg
            for msg in request["messages"]
            if isinstance(msg, dict)
            and msg.get("role") == "tool"
            and isinstance(msg.get("content"), str)
            and len(msg["content"]) > _LIMIT
        ]
        count = _server()._truncate_oversized_tool_results(request)
        assert count == expected_count
        expected = _apply_cc_truncation(snapshot)
        assert request == expected
        # R4d — in-place mutation: the same dict objects now carry the notice.
        for msg in pre_targets:
            assert msg["content"].startswith("[Tool output truncated — original size: ")
            assert msg["content"].endswith(" chars]")

    @given(_native_with_oversize_tool_results())
    @settings(max_examples=200)
    def test_native_over_limit_truncated_to_notice(
        self, request: dict[str, Any]
    ) -> None:
        """Native: at least one tool_result.content > limit ⇒ rewritten to exact notice."""
        snapshot = copy.deepcopy(request)
        expected_count = _expected_truncation_count(snapshot, "native")
        pre_targets = [
            block
            for msg in request["messages"]
            for block in (msg.get("content") if isinstance(msg.get("content"), list) else [])
            if isinstance(block, dict)
            and block.get("type") == "tool_result"
            and isinstance(block.get("content"), str)
            and len(block["content"]) > _LIMIT
        ]
        count = _server()._truncate_oversized_tool_results(request)
        assert count == expected_count
        assert request == _apply_native_truncation(snapshot)
        for block in pre_targets:
            assert block["content"].startswith("[Tool output truncated — original size: ")
            assert block["content"].endswith(" chars]")

    @given(_responses_with_oversize_output())
    @settings(max_examples=200)
    def test_responses_over_limit_truncated_to_notice(
        self, body: dict[str, Any]
    ) -> None:
        """Responses: at least one function_call_output.output > limit ⇒ rewritten."""
        snapshot = copy.deepcopy(body)
        expected_count = _expected_truncation_count(snapshot, "responses")
        pre_targets = [
            item
            for item in body["input"]
            if isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and isinstance(item.get("output"), str)
            and len(item["output"]) > _LIMIT
        ]
        count = _server()._truncate_oversized_responses_outputs(body)
        assert count == expected_count
        assert body == _apply_responses_truncation(snapshot)
        for item in pre_targets:
            assert item["output"].startswith("[Tool output truncated — original size: ")
            assert item["output"].endswith(" chars]")


# ── Constructed: non-tool content untouched (R5) ──────────────────────────


def _oversize_non_tool_cc_request() -> dict[str, Any]:
    """CC request with an oversize non-tool user message and an under-limit tool message."""
    return {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "x" * 100_000},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "stub", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "short"},
        ],
        "tools": _cc_stub_tools(),
    }


def _oversize_non_tool_native_request() -> dict[str, Any]:
    """Native request: an oversize non-tool user text and a small structured tool_result.

    The structured payload moved to :class:`TestStructuredContentTruncated`
    (KBR-223): an oversized one truncates now, so this fixture keeps a small
    one and pins only the non-tool text arm.
    """
    return {
        "model": "claude-sonnet-5",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "x" * 100_000}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "a", "name": "stub", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "a",
                        "content": [{"type": "text", "text": "small"}],
                    }
                ],
            },
        ],
        "tools": _native_stub_tools(),
    }


def _oversize_non_tool_responses_body() -> dict[str, Any]:
    """Responses body: an oversize non-output item and a small ``output``.

    The list-form ``output`` moved to :class:`TestStructuredContentTruncated`
    (KBR-223): an oversized one truncates now, so this fixture keeps a small
    string ``output`` and pins only the non-output text arm.
    """
    return {
        "model": "gpt-4o",
        "tools": _responses_stub_tools(),
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x" * 100_000}]},
            {"type": "function_call", "call_id": "a", "name": "stub", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "a",
                "output": "small",
            },
        ],
    }


class TestNonToolContentUntouched:
    """R5: oversize non-tool text survives byte-identical (KBR-223).

    Pre-KBR-223 this class also pinned that structured tool_result content and
    list-form Responses output survived at any length — the gap the KBR-169
    truncation comment named. Those two assertions moved to
    :class:`TestStructuredContentTruncated`, where the new posture is asserted.
    """

    def test_cc_non_tool_oversize_text_survives(self) -> None:
        """R5 CC: a user message with a 100,000-char string survives the pass untouched."""
        request = _oversize_non_tool_cc_request()
        snapshot = copy.deepcopy(request)
        count = _server()._truncate_oversized_tool_results(request)
        assert count == 0
        assert request == snapshot

    def test_native_non_tool_oversize_text_survives(self) -> None:
        """R5 native: an oversize text block on a plain user message survives untouched.

        The fixture's structured ``tool_result`` was moved to
        :class:`TestStructuredContentTruncated` so the user-text arm pins
        alone (pre-KBR-223 the structured arm truncated too, breaking this
        test's ``count == 0`` assertion).
        """
        request = _oversize_non_tool_native_request()
        snapshot = copy.deepcopy(request)
        count = _server()._truncate_oversized_tool_results(request)
        assert count == 0
        # Pin only the user-text arm: the oversize text block on the plain
        # user message equals its pre-call content.
        assert request["messages"][0]["content"][0] == snapshot["messages"][0]["content"][0]

    def test_responses_message_oversize_text_survives(self) -> None:
        """R5 Responses: an oversize ``input_text`` part on a message item survives untouched.

        The fixture's list-form ``output`` was moved to
        :class:`TestStructuredContentTruncated` so the message-text arm pins
        alone (pre-KBR-223 the list output truncated too, breaking this
        test's ``body == snapshot`` assertion).
        """
        body = _oversize_non_tool_responses_body()
        snapshot = copy.deepcopy(body)
        count = _server()._truncate_oversized_responses_outputs(body)
        assert count == 0
        assert body == snapshot


# ── KBR-223: structured tool-result content is measured and truncated ──────


class TestStructuredContentTruncated:
    """KBR-223: oversized structured tool-result payloads now truncate.

    Pre-KBR-223 these sites keyed on string content only; an oversized
    structured ``tool_result`` (native) or list-form Responses ``output``
    shipped untruncated. The shared extractor ``_tool_result_content_size``
    measures both arms now.
    """

    def test_native_structured_tool_result_over_limit_truncates(self) -> None:
        request = {
            "model": "claude-sonnet-5",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "small"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "a", "name": "stub", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "a",
                            "content": [{"type": "text", "text": "x" * 100_000}],
                        }
                    ],
                },
            ],
            "tools": _native_stub_tools(),
        }
        snapshot = copy.deepcopy(request)
        count = _server()._truncate_oversized_tool_results(request)
        assert count == 1
        assert "Tool output truncated" in request["messages"][2]["content"][0]["content"]
        # The plain user text in messages[0] is untouched.
        assert request["messages"][0] == snapshot["messages"][0]

    def test_responses_list_form_output_over_limit_truncates(self) -> None:
        body = {
            "model": "gpt-4o",
            "tools": _responses_stub_tools(),
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "small"}]},
                {"type": "function_call", "call_id": "a", "name": "stub", "arguments": "{}"},
                {
                    "type": "function_call_output",
                    "call_id": "a",
                    "output": [{"type": "input_text", "text": "x" * 100_000}],
                },
            ],
        }
        snapshot = copy.deepcopy(body)
        count = _server()._truncate_oversized_responses_outputs(body)
        assert count == 1
        assert "Tool output truncated" in body["input"][2]["output"]
        # The message item in input[0] is untouched.
        assert body["input"][0] == snapshot["input"][0]

    def test_cc_compaction_step1_list_tool_result_over_limit_truncates(self) -> None:
        """M4 stays CC-shape only (KBR-223); ``role: "tool"`` list content truncates.

        A small ``max_messages_chars`` budget engages compaction regardless of
        the static threshold, so step 1 runs without a multi-megabyte fixture.
        """
        server = _server()
        messages = [
            {"role": "user", "content": "compact me"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "a", "type": "function", "function": {"name": "stub", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "x" * 100_000}]},
        ]
        compacted = server._compact_messages(messages, max_messages_chars=1000)
        tool_msgs = [m for m in compacted if m.get("role") == "tool"]
        assert tool_msgs, "compaction dropped the tool message entirely"
        assert "Tool output truncated" in tool_msgs[0]["content"]


# ── Constructed: boundary (R3 + R4 exact) ──────────────────────────────────


class TestBoundary:
    """Exact-boundary cases for the truncation limit (R3 / R4)."""

    def test_cc_exactly_50_000_chars_is_not_truncated(self) -> None:
        """R3 boundary, CC: exactly 50,000 chars is at-or-below the limit — untouched."""
        request = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "a", "type": "function", "function": {"name": "stub", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "a", "content": "x" * _LIMIT},
            ],
            "tools": _cc_stub_tools(),
        }
        snapshot = copy.deepcopy(request)
        count = _server()._truncate_oversized_tool_results(request)
        assert count == 0
        assert request == snapshot

    def test_cc_exactly_50_001_chars_is_truncated(self) -> None:
        """R3/R4 boundary, CC: the first char over the limit flips the predicate — truncated."""
        request = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "a", "type": "function", "function": {"name": "stub", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "a", "content": "x" * (_LIMIT + 1)},
            ],
            "tools": _cc_stub_tools(),
        }
        snapshot = copy.deepcopy(request)
        count = _server()._truncate_oversized_tool_results(request)
        assert count == 1
        assert request == _apply_cc_truncation(snapshot)

    def test_native_exactly_50_000_chars_is_not_truncated(self) -> None:
        """R3 boundary, native: exactly 50,000 chars is at-or-below the limit — untouched."""
        request = {
            "model": "claude-sonnet-5",
            "max_tokens": 64,
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "stub", "input": {}}]},
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "a", "content": "x" * _LIMIT}],
                },
            ],
            "tools": _native_stub_tools(),
        }
        snapshot = copy.deepcopy(request)
        count = _server()._truncate_oversized_tool_results(request)
        assert count == 0
        assert request == snapshot

    def test_native_exactly_50_001_chars_is_truncated(self) -> None:
        """R3/R4 boundary, native: the first char over the limit flips the predicate."""
        request = {
            "model": "claude-sonnet-5",
            "max_tokens": 64,
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "stub", "input": {}}]},
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "a", "content": "x" * (_LIMIT + 1)}],
                },
            ],
            "tools": _native_stub_tools(),
        }
        snapshot = copy.deepcopy(request)
        count = _server()._truncate_oversized_tool_results(request)
        assert count == 1
        assert request == _apply_native_truncation(snapshot)

    def test_responses_exactly_50_000_chars_is_not_truncated(self) -> None:
        """R3 boundary, Responses: exactly 50,000 chars is at-or-below the limit — untouched."""
        body = {
            "model": "gpt-4o",
            "tools": _responses_stub_tools(),
            "input": [
                {"type": "function_call", "call_id": "a", "name": "stub", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "a", "output": "x" * _LIMIT},
            ],
        }
        snapshot = copy.deepcopy(body)
        count = _server()._truncate_oversized_responses_outputs(body)
        assert count == 0
        assert body == snapshot

    def test_responses_exactly_50_001_chars_is_truncated(self) -> None:
        """R3/R4 boundary, Responses: the first char over the limit flips the predicate."""
        body = {
            "model": "gpt-4o",
            "tools": _responses_stub_tools(),
            "input": [
                {"type": "function_call", "call_id": "a", "name": "stub", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "a", "output": "x" * (_LIMIT + 1)},
            ],
        }
        snapshot = copy.deepcopy(body)
        count = _server()._truncate_oversized_responses_outputs(body)
        assert count == 1
        assert body == _apply_responses_truncation(snapshot)


# ── Constructed: falsy-id orphan dropped (all three shapes) ────────────────


class TestFalsyIdOrphanDropped:
    """Empty ``tool_call_id`` / ``tool_use_id`` / ``call_id`` is an undeclared id → dropped."""

    def test_cc_empty_tool_call_id_is_dropped(self) -> None:
        """Falsy id: a CC tool message with ``tool_call_id=""`` is undeclared — dropped."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "", "content": "orphan"},
        ]
        result = _server()._validate_tool_call_pairing(list(messages))
        assert all(m.get("role") != "tool" for m in result)

    def test_native_empty_tool_use_id_is_dropped(self) -> None:
        """Falsy id, native: ``tool_use_id=""`` makes the block an orphan — message dropped."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "", "content": "orphan"}],
            }
        ]
        result = _server()._validate_tool_call_pairing(list(messages))
        assert result == []

    def test_responses_empty_call_id_is_dropped(self) -> None:
        """Falsy id, Responses: ``call_id=""`` is undeclared — the output item is dropped."""
        body = {
            "model": "gpt-4o",
            "input": [{"type": "function_call_output", "call_id": "", "output": "orphan"}],
        }
        dropped = _server()._drop_orphan_responses_tool_outputs(body)
        assert dropped == 1
        assert body["input"] == []


# ── Constructed: forward-reference dropped (preceding-id reading) ──────────


class TestForwardReferenceDropped:
    """An output placed BEFORE its declaring call is an orphan by the preceding-id rule."""

    def test_cc_forward_reference_dropped(self) -> None:
        """Preceding-id reading, CC: a tool message placed before its declaring call is an orphan."""
        # Assistant declaring call id="a" appears AFTER the tool message that references it.
        messages = [
            {"role": "tool", "tool_call_id": "a", "content": "orphan"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "a", "type": "function", "function": {"name": "stub", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "a", "content": "valid"},
        ]
        result = _server()._validate_tool_call_pairing(list(messages))
        surviving = [m for m in result if m.get("role") == "tool"]
        assert len(surviving) == 1 and surviving[0]["content"] == "valid"

    def test_native_forward_reference_dropped(self) -> None:
        """Preceding-id reading, native: a tool_result block before its tool_use is an orphan."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "a", "content": "orphan"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "a", "name": "stub", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "a", "content": "valid"}],
            },
        ]
        result = _server()._validate_tool_call_pairing(list(messages))
        surviving_blocks = [
            block
            for m in result
            for block in (m.get("content") if isinstance(m.get("content"), list) else [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert len(surviving_blocks) == 1 and surviving_blocks[0]["content"] == "valid"

    def test_responses_forward_reference_dropped(self) -> None:
        """Preceding-id reading, Responses: an output before its call is an orphan."""
        body = {
            "model": "gpt-4o",
            "input": [
                {"type": "function_call_output", "call_id": "a", "output": "orphan"},
                {"type": "function_call", "call_id": "a", "name": "stub", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "a", "output": "valid"},
            ],
        }
        dropped = _server()._drop_orphan_responses_tool_outputs(body)
        # Only the orphan (first) is dropped — the second output references a call that preceded it.
        assert dropped == 1
        surviving = [
            item
            for item in body["input"]
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        assert len(surviving) == 1 and surviving[0]["output"] == "valid"


# ── Constructed: oracle falsification (R6 / plan §1.4) ────────────────────


class TestOracleFalsification:
    """Eleven constructed example tests (plan §1.4 harness rule) showing each oracle rejects a wrong output.

    Every constructed case's expected output is hand-written (a literal
    list, not derived from the oracle), and each case additionally asserts
    ``oracle(input) == hand_written_expected`` so it doubles as an *oracle*
    falsification control.
    """

    def test_cc_pairing_rejects_output_that_keeps_an_orphan(self) -> None:
        """R6.1: CC pairing — output re-inserts one orphan tool message."""
        messages = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "stub", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "valid"},
            {"role": "tool", "tool_call_id": "orphan-id", "content": "should be gone"},
        ]
        hand_expected = [
            messages[0], messages[1], messages[2],
        ]
        assert _expected_cc(messages) == hand_expected
        defective = hand_expected + [messages[3]]
        assert defective != _expected_cc(messages)

    def test_cc_pairing_rejects_output_that_drops_a_paired_result(self) -> None:
        """R6.2: CC pairing — output drops one *paired* tool message."""
        messages = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "stub", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "valid"},
        ]
        hand_expected = list(messages)
        assert _expected_cc(messages) == hand_expected
        defective = [messages[0], messages[1]]  # drops the paired tool message
        assert defective != _expected_cc(messages)

    def test_native_pairing_rejects_output_that_keeps_an_orphan_block(self) -> None:
        """R6.3: Native pairing — output re-inserts one orphan tool_result block."""
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "stub", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "valid"}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "tool_result", "tool_use_id": "orphan", "content": "should be gone"},
                ],
            },
        ]
        hand_expected = [
            messages[0],
            messages[1],
            {**messages[2], "content": [{"type": "text", "text": "hi"}]},
        ]
        assert _expected_native(messages) == hand_expected
        defective = [
            messages[0],
            messages[1],
            messages[2],  # the orphan block survives
        ]
        assert defective != _expected_native(messages)

    def test_native_pairing_rejects_output_that_strips_a_non_orphan_block(self) -> None:
        """R6.4: Native pairing — output strips one *non-orphan* block from a mixed message."""
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "stub", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "tool_result", "tool_use_id": "a", "content": "valid"},
                ],
            },
        ]
        hand_expected = list(messages)
        assert _expected_native(messages) == hand_expected
        defective = [
            messages[0],
            {**messages[1], "content": [messages[1]["content"][1]]},  # drops the text block
        ]
        assert defective != _expected_native(messages)

    def test_cc_truncation_rejects_output_that_truncates_a_sub_limit_result(self) -> None:
        """R6.5: CC truncation — output truncates a sub-limit tool result."""
        request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "stub", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "a", "content": "short"},
            ],
            "tools": _cc_stub_tools(),
        }
        hand_expected = copy.deepcopy(request)
        assert _apply_cc_truncation(request) == hand_expected
        defective = copy.deepcopy(request)
        defective["messages"][1]["content"] = "[Tool output truncated — original size: 5 chars]"
        assert defective != _apply_cc_truncation(request)

    def test_cc_truncation_rejects_output_that_leaves_over_limit_untouched(self) -> None:
        """R6.6: CC truncation — output leaves an over-limit string untouched."""
        request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "stub", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "a", "content": "x" * (_LIMIT + 1)},
            ],
            "tools": _cc_stub_tools(),
        }
        hand_expected = copy.deepcopy(request)
        hand_expected["messages"][1]["content"] = (
            f"[Tool output truncated — original size: {_LIMIT + 1:,} chars]"
        )
        assert _apply_cc_truncation(request) == hand_expected
        defective = copy.deepcopy(request)  # unchanged
        assert defective != _apply_cc_truncation(request)

    def test_cc_truncation_rejects_output_that_rewrites_non_tool_content(self) -> None:
        """R6.7: CC truncation — output rewrites non-tool content."""
        request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "x" * 100},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "stub", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "a", "content": "short"},
            ],
            "tools": _cc_stub_tools(),
        }
        hand_expected = copy.deepcopy(request)
        assert _apply_cc_truncation(request) == hand_expected
        defective = copy.deepcopy(request)
        defective["messages"][0]["content"] = "[Tool output truncated — original size: 100 chars]"
        assert defective != _apply_cc_truncation(request)

    def test_responses_pairing_rejects_output_that_keeps_an_orphan_item(self) -> None:
        """R6.8: Responses pairing — output re-inserts one orphan output item."""
        items = [
            {"type": "function_call", "call_id": "a", "name": "stub", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "valid"},
            {"type": "function_call_output", "call_id": "orphan", "output": "should be gone"},
        ]
        hand_expected = [items[0], items[1]]
        assert _expected_responses_paired(items) == hand_expected
        defective = list(items)
        assert defective != _expected_responses_paired(items)

    def test_responses_pairing_rejects_output_that_drops_a_paired_output(self) -> None:
        """R6.9: Responses pairing — output drops one paired output item."""
        items = [
            {"type": "function_call", "call_id": "a", "name": "stub", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "valid"},
        ]
        hand_expected = list(items)
        assert _expected_responses_paired(items) == hand_expected
        defective = [items[0]]
        assert defective != _expected_responses_paired(items)

    def test_responses_truncation_rejects_output_that_truncates_a_sub_limit_output(self) -> None:
        """R6.10: Responses truncation — output truncates a sub-limit output."""
        body = {
            "model": "gpt-4o",
            "input": [
                {"type": "function_call", "call_id": "a", "name": "stub", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "a", "output": "short"},
            ],
        }
        hand_expected = copy.deepcopy(body)
        assert _apply_responses_truncation(body) == hand_expected
        defective = copy.deepcopy(body)
        defective["input"][1]["output"] = "[Tool output truncated — original size: 5 chars]"
        assert defective != _apply_responses_truncation(body)

    def test_responses_truncation_rejects_output_that_leaves_over_limit_untouched(self) -> None:
        """R6.11: Responses truncation — output leaves an over-limit output untouched."""
        body = {
            "model": "gpt-4o",
            "input": [
                {"type": "function_call", "call_id": "a", "name": "stub", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "a", "output": "x" * (_LIMIT + 1)},
            ],
        }
        hand_expected = copy.deepcopy(body)
        hand_expected["input"][1]["output"] = (
            f"[Tool output truncated — original size: {_LIMIT + 1:,} chars]"
        )
        assert _apply_responses_truncation(body) == hand_expected
        defective = copy.deepcopy(body)
        assert defective != _apply_responses_truncation(body)
