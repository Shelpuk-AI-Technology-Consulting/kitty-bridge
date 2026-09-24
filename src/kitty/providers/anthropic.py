"""Anthropic provider adapter — translates between Chat Completions and Anthropic Messages API."""

from __future__ import annotations

import json
import logging
import uuid
from typing import cast

from kitty.providers.base import ProviderAdapter, ProviderError, WireShape

__all__ = ["AnthropicAdapter"]

logger = logging.getLogger(__name__)

_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096

# stop_reason → finish_reason
_STOP_REASON_MAP: dict[str | None, str] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
    None: "stop",
}

# CC string tool_choice → Anthropic tool_choice type (KBR-214)
_CC_TO_ANTHROPIC_TOOL_CHOICE: dict[str, str] = {"auto": "auto", "required": "any", "none": "none"}

# CC finish_reason → Anthropic stop_reason (for reverse mapping if needed)
_FINISH_TO_STOP: dict[str, str] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


def _anthropic_tool_choice(cc_request: dict) -> dict | None:
    """Translate a Chat Completions ``tool_choice`` into Anthropic's object form.

    ``"auto"``, ``"required"`` and ``"none"`` become ``{"type": "auto"}``,
    ``{"type": "any"}`` and ``{"type": "none"}``; the named form
    ``{"type": "function", "function": {"name": x}}`` becomes
    ``{"type": "tool", "name": x}``.  Any other value -- including Chat
    Completions' allowed-tools and custom forms, which Anthropic cannot express
    -- yields ``None`` so the caller writes nothing rather than a guess.

    ``parallel_tool_calls: False`` is carried as ``disable_parallel_tool_use:
    True`` on the returned object, because Anthropic has no top-level knob.
    ``none`` has no slot for it, and a bare ``parallel_tool_calls`` with no
    choice returns ``None``: carrying it would mean inventing a
    ``{"type": "auto"}`` the agent never sent (KBR-214 D4).

    Args:
        cc_request: The Chat Completions request being translated.

    Returns:
        The Anthropic ``tool_choice`` object, or ``None`` when there is nothing
        Anthropic can be told.
    """
    choice = cc_request.get("tool_choice")
    if isinstance(choice, str) and choice in _CC_TO_ANTHROPIC_TOOL_CHOICE:
        anthropic: dict = {"type": _CC_TO_ANTHROPIC_TOOL_CHOICE[choice]}
    elif (
        isinstance(choice, dict)
        and choice.get("type") == "function"
        and isinstance(choice.get("function"), dict)
        and isinstance(choice["function"].get("name"), str)
    ):
        anthropic = {"type": "tool", "name": choice["function"]["name"]}
    else:
        return None

    # Only a present, explicit `false` changes anything; `true` is the default.
    if anthropic["type"] != "none" and cc_request.get("parallel_tool_calls") is False:
        anthropic["disable_parallel_tool_use"] = True
    return anthropic


def _safe_json_load_args(arguments: str | None) -> dict:
    """Parse tool call arguments, falling back to ``{}`` on malformed JSON.

    Upstream models may produce syntactically invalid JSON strings for
    tool call arguments.  Without this guard, ``json.loads`` raises an
    unhandled ``JSONDecodeError`` that becomes a 500.
    """
    raw = arguments or "{}"
    try:
        # json.loads is typed as returning Any; the guard below covers the
        # only failure mode, and callers require a mapping.
        return cast(dict, json.loads(raw))
    except json.JSONDecodeError:
        return {}


def _image_source_from_url(url: str) -> dict:
    """Convert a CC ``image_url`` value into an Anthropic ``image`` source.

    A ``data:<media_type>;base64,<data>`` URI becomes a base64 source; anything
    else — an empty string, an ``http(s)`` URL, a malformed URI — becomes a URL
    source and reaches the upstream's validator. This function validates
    nothing itself: an upstream that cannot honour the value reports it better
    than a silent drop or repair would (KBR-222; hop 1's ``continue`` for
    source types it cannot spell is the other half of that stance).

    Args:
        url: The ``url`` member of a CC ``image_url`` content part.

    Returns:
        An Anthropic ``image.source`` object.
    """
    if url.startswith("data:") and ";base64," in url:
        media_type, _, data = url[len("data:"):].partition(";base64,")
        return {"type": "base64", "media_type": media_type, "data": data}
    return {"type": "url", "url": url}


class AnthropicCCStreamConverter:
    """Convert one Anthropic Messages SSE stream into Chat Completions chunks.

    Stateful where :meth:`AnthropicAdapter.translate_upstream_stream_event`
    cannot be: that method maps each event in isolation, so a ``tool_use``
    block's ``input_json_delta`` fragments have nowhere to land and every
    tool call was lost (KBR-232).  Here a block's Chat Completions
    ``tool_calls`` index is allocated when the block opens and every
    fragment of its JSON is forwarded under it, thinking deltas cross as
    ``reasoning_content``, and the finish chunk carries usage accumulated
    from both ends of the stream.

    One instance serves one upstream attempt: ``kitty.bridge`` creates it
    when the selected backend's upstream wire is Anthropic Messages for the
    routed model (:meth:`BridgeServer._serves_messages_wire`) and feeds it
    each ``data:`` line before the per-chunk logic it already runs for
    Chat Completions upstreams.  An ``error`` event passes through
    unchanged so the handlers' in-stream error detection sees it.
    """

    def __init__(self) -> None:
        # One id per stream: Chat Completions clients correlate a reply's
        # chunks by it, and the old per-event ids made every chunk an orphan.
        self._chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self._model = ""
        self._input_tokens = 0
        self._output_tokens = 0
        # Anthropic block index → Chat Completions tool_calls index.
        self._tool_indices: dict[int, int] = {}
        # Blocks whose arguments arrived whole in content_block_start; their
        # input_json_delta fragments are suppressed so arguments are not doubled.
        self._arguments_complete: set[int] = set()

    def feed(self, raw_bytes: bytes) -> list[bytes]:
        """Convert one upstream SSE line into Chat Completions SSE lines.

        Args:
            raw_bytes: One SSE line as the handler read it, e.g.
                ``b'event: content_block_delta\\ndata: {...}\\n\\n'``.

        Returns:
            Full ``data: `` SSE lines — Chat Completions chunks, the
            ``data: [DONE]`` sentinel on ``message_stop``, or the input
            unchanged for a non-JSON line and for an ``error`` event.
            Empty when the event has no Chat Completions counterpart.
        """
        raw_str = raw_bytes.decode("utf-8", errors="replace").strip()
        if not raw_str:
            return []

        # The handlers hand us one event's SSE, but an ``event:`` line may
        # precede the payload; only the ``data:`` line is interpreted.
        data_str = None
        for line in raw_str.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()

        if data_str is None:
            return []
        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            return [raw_bytes]
        if not isinstance(event, dict):
            return [raw_bytes]

        event_type = event.get("type", "")

        # An in-stream failure rides a 200 on Anthropic's wire; the handlers'
        # error detection keys on the ``type``, so the event must cross as-is.
        if event_type == "error":
            return [raw_bytes]

        if event_type == "message_start":
            message = event.get("message", {})
            self._model = message.get("model", "")
            self._input_tokens = message.get("usage", {}).get("input_tokens", 0)
            return [self._sse_chunk({"role": "assistant"})]

        if event_type == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") != "tool_use":
                return []
            index = event.get("index", 0)
            cc_index = len(self._tool_indices)
            self._tool_indices[index] = cc_index
            # A populated ``input`` at the block start is the whole arguments
            # object for providers that do not stream the JSON; Anthropic
            # itself starts empty and streams ``input_json_delta``.
            tool_input = block.get("input")
            if tool_input:
                self._arguments_complete.add(index)
                arguments = json.dumps(tool_input)
            else:
                arguments = ""
            return [
                self._sse_chunk(
                    {
                        "tool_calls": [
                            {
                                "index": cc_index,
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {"name": block.get("name", ""), "arguments": arguments},
                            }
                        ]
                    }
                )
            ]

        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "text_delta":
                return [self._sse_chunk({"content": delta.get("text", "")})]
            if delta_type == "thinking_delta":
                return [self._sse_chunk({"reasoning_content": delta.get("thinking", "")})]
            if delta_type == "input_json_delta":
                index = event.get("index", 0)
                # An unknown block has no allocated index; a completed one
                # must not grow a second copy of its arguments.
                if index not in self._tool_indices or index in self._arguments_complete:
                    return []
                return [
                    self._sse_chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": self._tool_indices[index],
                                    "function": {"arguments": delta.get("partial_json", "")},
                                }
                            ]
                        }
                    )
                ]
            # signature_delta and other block deltas have no CC counterpart.
            return []

        if event_type == "message_delta":
            # Anthropic's message_delta usage is the running output total.
            self._output_tokens = event.get("usage", {}).get("output_tokens", self._output_tokens)
            finish_reason = _STOP_REASON_MAP.get(event.get("delta", {}).get("stop_reason"), "stop")
            chunk = self._chunk({}, finish_reason=finish_reason)
            chunk["usage"] = {
                "prompt_tokens": self._input_tokens,
                "completion_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
            }
            return [f"data: {json.dumps(chunk)}\n\n".encode()]

        if event_type == "message_stop":
            return [b"data: [DONE]\n\n"]

        # ping, content_block_stop, anything unknown.
        return []

    def _chunk(self, delta: dict, finish_reason: str | None = None) -> dict:
        """Build one Chat Completions chunk payload for this stream.

        Args:
            delta: The choice delta.
            finish_reason: The finish reason, or ``None`` mid-stream.

        Returns:
            The chunk payload, not yet SSE-wrapped.
        """
        return {
            "id": self._chunk_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": self._model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    def _sse_chunk(self, delta: dict, finish_reason: str | None = None) -> bytes:
        """Wrap one chunk payload as a ``data: `` SSE line.

        Args:
            delta: The choice delta.
            finish_reason: The finish reason, or ``None`` mid-stream.

        Returns:
            The encoded SSE line, trailing blank line included.
        """
        return f"data: {json.dumps(self._chunk(delta, finish_reason))}\n\n".encode()


class AnthropicAdapter(ProviderAdapter):
    """Anthropic Messages API adapter.

    Translates between Kitty's internal Chat Completions format and
    Anthropic's Messages API (``POST /v1/messages``).  Anthropic uses
    ``x-api-key`` authentication and a content-block based request/response
    format rather than CC's message/content structure.

    Attributes:
        forwards_thinking_display: Whether :meth:`translate_to_upstream`
            restores the agent's thinking ``display`` onto ``thinking``.  True
            here, because Anthropic's Messages API defines the field.  A
            subclass whose upstream does not document it sets this to False, so
            an unknown field cannot turn every thinking request into a 400
            (KBR-203).
        injects_placeholder_thinking: Whether
            :meth:`_translate_assistant_msg` injects an empty unsigned
            thinking block into an assistant message that lacks one while
            thinking is active (register row P5e).  False here — the default,
            because this class *is* the ``anthropic`` provider: the live probe
            behind KBR-238 showed the unsigned block itself is rejected with
            ``400 ... thinking.signature: Field required`` while a history
            with no thinking block is accepted, so manufacturing one costs a
            rejected round-trip per turn and M17's strip has to remove it
            again (KBR-228 part C).  A subclass whose upstream has not been
            shown to reject the block sets this to True to keep the old wire.
        forwards_thinking_signature: Whether :meth:`translate_to_upstream`
            restores the agent's signed thinking blocks and original
            ``system`` value verbatim from the KBR-228 carriage.  True here:
            api.anthropic.com signature-binds thinking to the conversation
            that produced it, and only a byte-identical restore — signatures,
            ``redacted_thinking`` and cache breakpoints included — satisfies
            the check (KBR-228 part B).  A subclass whose upstream has never
            been shown to accept a ``signature`` or ``redacted_thinking``
            field sets this to False, which keeps today's wire: restoring
            unverified fields with no recovery pattern that recognises a
            foreign rejection would trade a verified fix for a hard failure.
        forwards_output_config: Whether :meth:`translate_to_upstream` restores
            the agent's ``output_config`` — the documented spelling of the
            effort control.  True here, because the field is on Anthropic's
            published Messages schema.  A subclass whose upstream rejects or
            does not document it sets this to False (KBR-224).
    """

    forwards_thinking_display: bool = True

    injects_placeholder_thinking: bool = False

    forwards_thinking_signature: bool = True

    forwards_output_config: bool = True

    @property
    def provider_type(self) -> str:
        return "anthropic"

    @property
    def default_base_url(self) -> str:
        return "https://api.anthropic.com"

    @property
    def upstream_path(self) -> str:
        return "/v1/messages"

    @property
    def upstream_wire_shape(self) -> WireShape:
        """:attr:`WireShape.MESSAGES` — this class's ``translate_to_upstream`` emits the Anthropic Messages API.

        Holds for subclasses that do **not** route by model: they either
        forward a native Messages body unchanged or fall back to this class's
        Chat Completions → Messages translation, so the wire shape does not
        depend on ``_native_messages_request``.

        It does **not** hold for a subclass that routes by model.
        :class:`~kitty.providers.opencode.OpenCodeGoAdapter` emits Chat
        Completions for every model outside its ``_MESSAGES_MODELS``, and
        inheriting this ``WireShape.MESSAGES`` would be KBR-7 in a new
        costume.  Such a subclass must override **both** this property,
        reporting its default route, and
        :meth:`upstream_wire_shape_for_model`, mirroring its own routing.
        """
        return WireShape.MESSAGES

    # ── Auth headers ─────────────────────────────────────────────────────

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        return {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    # ── CC → Anthropic request translation ───────────────────────────────

    def translate_to_upstream(self, cc_request: dict) -> dict:
        """Translate a Chat Completions request into an Anthropic Messages request."""
        anthropic: dict = {
            "model": cc_request["model"],
            "max_tokens": cc_request.get("max_tokens", _DEFAULT_MAX_TOKENS),
            "messages": [],
        }

        if cc_request.get("stream") is not None:
            anthropic["stream"] = cc_request["stream"]

        if "temperature" in cc_request and cc_request["temperature"] is not None:
            anthropic["temperature"] = cc_request["temperature"]

        if "top_p" in cc_request and cc_request["top_p"] is not None:
            anthropic["top_p"] = cc_request["top_p"]

        # KBR-178: the CC `stop` is this wire's `stop_sequences` under another
        # name.  Null and empty are both omitted -- `stop` is nullable in Chat
        # Completions and the bridge serves /v1/chat/completions directly, and
        # Anthropic rejects a null stop list.
        if cc_request.get("stop"):
            anthropic["stop_sequences"] = cc_request["stop"]

        # KBR-178: restored from internal metadata, because Chat Completions has
        # no `top_k` of its own in which to have carried it this far.
        if cc_request.get("_top_k") is not None:
            anthropic["top_k"] = cc_request["_top_k"]

        # Extract system messages → top-level system field. Two parallel
        # collections: ``system_parts`` (joined-string form, every body that
        # ships today produces) and ``system_blocks`` (the same parts as
        # blocks, with any ``cache_control`` marker in place). The emit
        # below picks one of three forms, gated on the
        # ``forwards_thinking_signature`` flag (KBR-228 part B) and the
        # G43/minimax scope-out (KBR-296).
        system_parts: list[str] = []
        system_blocks: list[dict] = []
        for msg in cc_request.get("messages", []):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
                    system_blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    # System content as list of blocks — extract text,
                    # keeping any cache marker on its block.
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            system_parts.append(text)
                            system_block: dict = {"type": "text", "text": text}
                            # KBR-308: keep the marker on the CC block so the
                            # blocks-form emit below can ship it; the join
                            # would lose it.
                            if block.get("cache_control") is not None:
                                system_block["cache_control"] = block["cache_control"]
                            system_blocks.append(system_block)
                        elif isinstance(block, str):
                            system_parts.append(block)
                            system_blocks.append({"type": "text", "text": block})

        # KBR-228 part B: the agent's system value (blocks and markers
        # included) on signature-binding routes — restored verbatim from
        # ``_anthropic_system`` instead of a join.
        carried_system = cc_request.get("_anthropic_system")
        if carried_system is not None and self.forwards_thinking_signature:
            if isinstance(carried_system, list):
                anthropic["system"] = [
                    dict(block) if isinstance(block, dict) else block for block in carried_system
                ]
            else:
                anthropic["system"] = carried_system
        # KBR-308: blocks form when any system part carries a marker AND
        # the upstream is verified — the same gate as the carriage restore
        # above. On ``minimax_token`` / ``opencode_go``'s Messages-routed
        # models (forwards_thinking_signature=False) the endpoint rejects
        # markers on system blocks, so the join stands and the G43 scope-out
        # is preserved on this route.
        elif any("cache_control" in block for block in system_blocks) and self.forwards_thinking_signature:
            anthropic["system"] = system_blocks
        # Joined-string form for everything else: unmarked system, or any
        # system on an unverified upstream.
        elif system_parts:
            anthropic["system"] = "\n".join(system_parts)

        # Translate messages.  KBR-222: a run of CC tool messages followed by
        # a user message re-joins into ONE Anthropic user message -- the
        # tool_result blocks first, then the translated sibling blocks -- which
        # is the shape Anthropic's docs prescribe and Claude Code sent inbound;
        # strict third-party Anthropic-compatible endpoints have rejected the
        # split.  Trailing tool messages with no user follower keep the
        # per-message shape this loop shipped before the fix.
        pending_tool_results: list[dict] = []
        for msg in cc_request.get("messages", []):
            role = msg.get("role")
            if role == "system":
                continue  # already handled above

            if role == "tool":
                pending_tool_results.append(self._tool_result_block(msg))
                continue

            if pending_tool_results and role == "user":
                # A string sibling needs block form to sit beside the
                # tool_result blocks; an empty or absent one contributes
                # nothing, which reproduces today's single tool_result message
                # exactly.
                sibling = self._translate_user_content(msg, cc_request)
                if not isinstance(sibling, list):
                    sibling = [{"type": "text", "text": sibling}] if sibling else []
                anthropic["messages"].append(
                    {
                        "role": "user",
                        "content": [*pending_tool_results, *sibling],
                    }
                )
                pending_tool_results = []
                continue

            anthropic["messages"].extend({"role": "user", "content": [b]} for b in pending_tool_results)
            pending_tool_results = []

            if role == "assistant":
                anthropic["messages"].append(self._translate_assistant_msg(msg, cc_request))
            elif role == "user":
                anthropic["messages"].append(
                    {
                        "role": "user",
                        "content": self._translate_user_content(msg, cc_request),
                    }
                )
            else:
                anthropic["messages"].append(
                    {
                        "role": role,
                        "content": msg.get("content", ""),
                    }
                )

        # A tool run at the very end of the body has no user turn to re-join.
        anthropic["messages"].extend({"role": "user", "content": [b]} for b in pending_tool_results)

        # Translate tools
        if "tools" in cc_request and cc_request["tools"]:
            anthropic["tools"] = self._translate_tools(
                cc_request["tools"], cc_request.get("_tool_cache_controls")
            )

        # KBR-214: this body is rebuilt from an allowlist, so an agent's tool
        # constraint is dropped here unless it is written back explicitly.
        tool_choice = _anthropic_tool_choice(cc_request)
        # A choice over no tools would describe tools this body does not declare.
        if tool_choice is not None and "tools" in anthropic:
            anthropic["tool_choice"] = tool_choice

        # KBR-214: restored from internal metadata, because Chat Completions has
        # no field of its own that means Anthropic's `metadata`.
        if cc_request.get("_metadata") is not None:
            anthropic["metadata"] = cc_request["_metadata"]

        # KBR-296: restored from internal metadata, because Chat Completions
        # has no top-level slot for Anthropic's automatic-caching form. The
        # same value already reached this upstream on attempt 0 (native
        # passthrough ships the raw body verbatim), so restoring it cannot
        # newly 400 — attempt-0 parity, no gating. KBR-308: on the
        # CC-origin path a raw ``cache_control`` key folds into the same
        # carriage when the carriage is absent (DQ-B: the carriage is
        # kitty's own translator's output and wins).
        top_cc = cc_request.get("_cache_control")
        if top_cc is None:
            top_cc = cc_request.get("cache_control")
        if top_cc is not None:
            anthropic["cache_control"] = top_cc

        # Restore thinking from normalized effort metadata
        if cc_request.get("_thinking_adaptive"):
            # Claude Code sent thinking: {type: "adaptive"} — forward it
            # verbatim to the Anthropic-compatible upstream.  This lets the
            # provider decide the budget automatically.
            anthropic["thinking"] = self._with_thinking_display({"type": "adaptive"}, cc_request)
        elif cc_request.get("_thinking_enabled"):
            # KBR-225: the agent's own budget, when the translator carried it,
            # ships verbatim — Anthropic renders the budget into the prompt, so
            # deriving it from max_tokens made two requests that differ only in
            # max_tokens miss each other's cache.  A carried budget is already
            # valid (int, >= 1024, < max_tokens), which implies
            # max_tokens >= 1025, so the fallback's raise below cannot trigger
            # on this branch and max_tokens ships as sent.
            if "_thinking_budget_tokens" in cc_request:
                anthropic["thinking"] = self._with_thinking_display(
                    {"type": "enabled", "budget_tokens": cc_request["_thinking_budget_tokens"]}, cc_request
                )
            else:
                # Fallback for an absent or invalid agent budget (the
                # translator carries only valid ones): derive from max_tokens.
                # Anthropic requires budget_tokens >= 1024 and budget_tokens < max_tokens.
                max_tokens = max(anthropic.get("max_tokens", _DEFAULT_MAX_TOKENS), 1025)
                anthropic["max_tokens"] = max_tokens
                anthropic["thinking"] = self._with_thinking_display(
                    {"type": "enabled", "budget_tokens": max_tokens - 1}, cc_request
                )
        elif cc_request.get("_thinking_enabled") is False:
            # No display here: Anthropic rejects `display` alongside `disabled`.
            anthropic["thinking"] = {"type": "disabled"}

        # Restore the effort parameter for Anthropic-compatible upstreams.
        # Claude Code sends this to control reasoning depth (e.g. "low",
        # "medium", "high", "xhigh").  The Anthropic Messages API accepts
        # it as a top-level parameter alongside thinking.
        if cc_request.get("_effort"):
            anthropic["effort"] = cc_request["_effort"]

        # KBR-224: restore the agent's `output_config` (the documented spelling
        # of the effort control) where this upstream documents the field.  Both
        # effort spellings ship side by side, unmerged: kitty has no authority
        # to arbitrate between two values the agent sent.
        if self.forwards_output_config and cc_request.get("_output_config") is not None:
            anthropic["output_config"] = cc_request["_output_config"]

        return anthropic

    def _with_thinking_display(self, thinking: dict, cc_request: dict) -> dict:
        """Add the agent's thinking ``display`` to *thinking* where this upstream documents it.

        Restoring ``display`` keeps the request faithful to what the agent sent;
        since KBR-227 (streamed replies are forwarded byte-for-byte) and
        KBR-228 part A (the non-streaming reply carries thinking through to
        ``MessagesTranslator``) the user actually sees the thinking it asks
        for.  Sending it to an upstream that does not document the field risks
        a 400 on every thinking request (KBR-203).  The value itself is checked
        by the translator, the only writer of ``_thinking_display``.

        Args:
            thinking: The ``adaptive`` or ``enabled`` thinking object being built.
            cc_request: The Chat Completions request, read for ``_thinking_display``.

        Returns:
            *thinking* itself, not a copy: ``display``, when added, is set in
            place.  Callers pass a freshly built dict, so nothing is aliased.
        """
        if self.forwards_thinking_display and "_thinking_display" in cc_request:
            thinking["display"] = cc_request["_thinking_display"]
        return thinking

    def _translate_assistant_msg(self, msg: dict, cc_request: dict | None = None) -> dict:
        """Translate an assistant message with optional tool_calls to Anthropic content blocks.

        Args:
            msg: CC-format assistant message dict.
            cc_request: Full CC request dict, used to check ``_thinking_enabled``.
                        Pass ``None`` to suppress empty thinking block injection.
        """
        content_blocks: list[dict] = []

        reasoning = msg.get("reasoning_content")
        thinking_enabled = (cc_request or {}).get("_thinking_enabled")
        # KBR-228 part B: the signed originals, verbatim and in wire order,
        # ahead of the rebuilt text and tool calls — thinking always precedes
        # both on this wire, so the original order survives.  A carriage with
        # the switch off, or none at all, rebuilds the unsigned block as
        # before.
        carried = msg.get("_thinking_blocks")
        if isinstance(carried, list) and carried and self.forwards_thinking_signature:
            content_blocks.extend(dict(block) for block in carried if isinstance(block, dict))
        elif reasoning:
            content_blocks.append({"type": "thinking", "thinking": reasoning})
        elif thinking_enabled and self.injects_placeholder_thinking:
            content_blocks.append({"type": "thinking", "thinking": ""})

        text = msg.get("content")
        if text:
            text_block: dict = {"type": "text", "text": text}
            # KBR-296: the joined text block's breakpoint, restored from the
            # message-level carriage (last-marked-wins on the carry side).
            # KBR-308: on the CC-origin path the raw assistant-message-object
            # ``cache_control`` folds in when the carriage is absent (DQ-B).
            cc_text = msg.get("_cache_control")
            if cc_text is None:
                cc_text = msg.get("cache_control")
            if cc_text is not None:
                text_block["cache_control"] = cc_text
            content_blocks.append(text_block)

        # KBR-296: per-tool_use breakpoints ride an index-keyed message-level
        # carriage; each rebuilt ``tool_use`` block reads its own position.
        # KBR-308: a raw ``tool_calls[i].cache_control`` on the CC body
        # folds in when the carriage lacks that index (DQ-B).
        tool_call_cache_controls = msg.get("_tool_call_cache_controls")
        for idx, tc in enumerate(msg.get("tool_calls", [])):
            func = tc.get("function", {})
            tool_use_block: dict = {
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": func.get("name", ""),
                "input": _safe_json_load_args(func.get("arguments")),
            }
            tc_cc = None
            if isinstance(tool_call_cache_controls, dict) and idx in tool_call_cache_controls:
                tc_cc = tool_call_cache_controls[idx]
            elif isinstance(tc, dict) and tc.get("cache_control") is not None:
                tc_cc = tc["cache_control"]
            if tc_cc is not None:
                tool_use_block["cache_control"] = tc_cc
            content_blocks.append(tool_use_block)

        return {"role": "assistant", "content": content_blocks or ""}

    @staticmethod
    def _tool_result_block(msg: dict) -> dict:
        """Build the Anthropic ``tool_result`` block for one CC tool message.

        Args:
            msg: A CC-format ``role: "tool"`` message dict.

        Returns:
            The ``tool_result`` content block. Anthropic requires tool_result
            to be inside a user message; the caller decides whether that is a
            message of its own or a re-joined tool run (KBR-222).
        """
        block: dict = {
            "type": "tool_result",
            "tool_use_id": msg.get("tool_call_id", ""),
            "content": msg.get("content", ""),
        }
        # KBR-296: the block's own ``cache_control`` breakpoint rides the
        # internal ``_cache_control`` message key (list-form content
        # ``tool_result.content`` is forwarded verbatim, so any nested
        # breakpoint rides inside the array without a carriage). KBR-308:
        # on the CC-origin path the raw tool-message-object
        # ``cache_control`` folds in when the carriage is absent (DQ-B).
        cc_msg = msg.get("_cache_control")
        if cc_msg is None:
            cc_msg = msg.get("cache_control")
        if cc_msg is not None:
            block["cache_control"] = cc_msg
        return block

    def _translate_user_content(self, msg: dict, cc_request: dict) -> list[dict] | str | None:
        """Translate a CC user message's content into Anthropic content blocks.

        Text and ``image_url`` parts get their Anthropic spellings; anything
        else forwards verbatim, as this hop did before KBR-222, so a
        CC-ingress client's unusual part reaches the upstream's validator
        instead of being silently repaired. A ``_documents`` entry addressed
        to *msg* contributes its blocks here too — by identity, so a compaction
        pass that rebuilt the message list forfeits the document rather than
        attaching it to a stranger.

        Args:
            msg: The CC-format user message dict.
            cc_request: The full CC request, read for ``_documents``.

        Returns:
            The content to ship: a list of blocks for list-form content or
            string turns carrying documents, otherwise the original string or
            ``None`` unchanged.
        """
        # KBR-222: documents ride the ``_documents`` internal key, addressed to
        # this message by identity. Collected first so both the string and the
        # list path splice them in. A malformed value is skipped, not raised:
        # like ``_top_k``, an internal key must never fail the request no
        # matter what it carries (the R5 suite injects a probe into every one).
        documents: list[dict] = []
        raw_documents = cc_request.get("_documents")
        if isinstance(raw_documents, list):
            for entry in raw_documents:
                if not isinstance(entry, dict) or entry.get("message") is not msg:
                    continue
                entry_blocks = entry.get("blocks")
                if isinstance(entry_blocks, list):
                    documents.extend(b for b in entry_blocks if isinstance(b, dict))

        content: object = msg.get("content", "")
        if isinstance(content, list):
            # Text and images get Anthropic spellings; unknown parts pass
            # through. A known part's other members survive the rebuild --
            # verbatim forwarding carried them before KBR-222, and a
            # CC-ingress client's ``cache_control`` on a text part is pinned
            # by the KBR-199 suite.
            blocks: list[dict] = []
            for part in content:
                if not isinstance(part, dict):
                    blocks.append(part)
                    continue
                kind = part.get("type")
                if kind == "text":
                    blocks.append({**part, "type": "text", "text": part.get("text", "")})
                elif kind == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    image_block: dict = {"type": "image", "source": _image_source_from_url(url)}
                    if "cache_control" in part:
                        image_block["cache_control"] = part["cache_control"]
                    blocks.append(image_block)
                else:
                    blocks.append(part)
            blocks.extend(documents)
            return blocks

        if isinstance(content, str) or content is None:
            if not documents:
                return content
            # A string turn carrying documents needs block form to hold them;
            # an empty or absent string contributes no text block.
            return ([{"type": "text", "text": content}] if content else []) + documents

        # A payload type no writer produces; stringify like the
        # ``_translate_message`` fallback rather than ship it raw.
        return str(content)

    def _translate_tools(
        self, cc_tools: list[dict], tool_cache_controls: dict[str, dict] | None = None
    ) -> list[dict]:
        """Translate CC tool definitions to Anthropic format.

        Args:
            cc_tools: The CC ``tools`` list.
            tool_cache_controls: KBR-296 carriage — per-tool ``cache_control``
                breakpoints, **name-keyed** to align with the register's P30
                vocabulary (``conversation.tools[<name>].cache_control``) and
                to survive any future normalisation that reorders tools.
                Restored onto the rebuilt declaration verbatim.
        """
        anthropic_tools = []
        for tool in cc_tools:
            func = tool.get("function", {})
            anthropic_tool: dict = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            }
            # KBR-296: a carried declaration breakpoint is restored by name;
            # a name nothing carried simply has no entry. KBR-308: on the
            # CC-origin path, a per-tool ``cache_control`` on the CC tool
            # object folds into the same lookup when the carriage lacks it
            # (DQ-B: carriage wins, raw CC shape is the fallback).
            name = func.get("name")
            tool_cc = None
            if tool_cache_controls and name in tool_cache_controls:
                tool_cc = tool_cache_controls[name]
            elif isinstance(tool, dict) and tool.get("cache_control") is not None:
                tool_cc = tool["cache_control"]
            if tool_cc is not None:
                anthropic_tool["cache_control"] = tool_cc
            anthropic_tools.append(anthropic_tool)
        return anthropic_tools

    # ── Anthropic response → CC translation ──────────────────────────────

    def translate_from_upstream(self, raw_response: dict) -> dict:
        """Translate an Anthropic Messages response into Chat Completions format."""
        content_blocks = raw_response.get("content", [])
        text_parts: list[str] = []
        tool_uses: list[dict] = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_uses.append(block)

        message: dict = {"role": "assistant", "content": "\n".join(text_parts) or None}

        if tool_uses:
            message["tool_calls"] = [
                {
                    "id": tu.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": tu.get("name", ""),
                        "arguments": json.dumps(tu.get("input", {})),
                    },
                }
                for tu in tool_uses
            ]

        # KBR-228 part A: the reply's thinking blocks ride the CC message under
        # an internal key, verbatim and in wire order, so the Messages client
        # sees the model's reasoning and the next turn has signed blocks to
        # send back.  ``thinking`` always precedes ``text``/``tool_use`` on
        # this wire, so prepending in ``translate_response`` preserves the
        # order the upstream produced.
        thinking_blocks = [
            dict(block)
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking")
        ]
        if thinking_blocks:
            message["_thinking_blocks"] = thinking_blocks

        stop_reason = raw_response.get("stop_reason")
        finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop")

        usage = raw_response.get("usage", {})
        return {
            "id": raw_response.get("id", ""),
            "object": "chat.completion",
            "created": 0,
            "model": raw_response.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
        }

    # ── SSE stream event translation ─────────────────────────────────────

    def translate_upstream_stream_event(self, raw_bytes: bytes) -> list[bytes]:
        """Translate an Anthropic SSE event into CC-format SSE chunks.

        Anthropic event flow:
        - message_start → emit CC chunk with role
        - content_block_start → ignored
        - content_block_delta (text_delta) → CC chunk with content delta
        - content_block_delta (input_json_delta) → buffered for tool_use
        - content_block_stop → flush tool buffer if tool_use block
        - message_delta → CC chunk with finish_reason
        - message_stop → emit [DONE]
        - ping → ignored
        """
        raw_str = raw_bytes.decode("utf-8", errors="replace").strip()
        if not raw_str:
            return []

        # Parse SSE lines
        data_str = None
        for line in raw_str.split("\n"):
            line = line.strip()
            if line.startswith("event:"):
                pass  # event type is communicated via JSON type field; no action needed
            elif line.startswith("data:"):
                data_str = line[5:].strip()

        if not data_str:
            return []

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return [raw_bytes]

        data_type = data.get("type", "")

        # Ignore non-content events (F10: "error" included to prevent silent passthrough)
        if data_type in ("ping", "content_block_start", "content_block_stop", "error"):
            return []

        if data_type == "message_start":
            msg = data.get("message", {})
            return self._make_cc_chunk({"role": "assistant"}, msg.get("model", ""))

        if data_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "text_delta":
                text = delta.get("text", "")
                return self._make_cc_chunk({"content": text})
            # input_json_delta — suppress (tool use streaming not needed for CC)
            return []

        if data_type == "message_delta":
            delta = data.get("delta", {})
            stop = delta.get("stop_reason")
            finish = _STOP_REASON_MAP.get(stop, "stop")
            return self._make_cc_chunk({}, finish_reason=finish)

        if data_type == "message_stop":
            return [b"data: [DONE]\n\n"]

        # Unknown event — passthrough
        return [raw_bytes]

    def _make_cc_chunk(
        self,
        delta: dict,
        model: str = "",
        finish_reason: str | None = None,
    ) -> list[bytes]:
        """Build a CC streaming chunk wrapped in SSE data."""
        chunk: dict = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return [f"data: {json.dumps(chunk)}\n\n".encode()]

    # ── Standard ProviderAdapter methods ─────────────────────────────────

    def normalize_model_name(self, model: str) -> str:
        """Strip provider prefix and normalize version separators (e.g. '4.6' -> '4-6')."""
        if "/" in model:
            model = model.split("/", 1)[1] or model
        return model.replace(".", "-")

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        request: dict = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
        }
        for key in ("temperature", "top_p", "max_tokens"):
            if key in kwargs and kwargs[key] is not None:
                request[key] = kwargs[key]
        if "tools" in kwargs and kwargs["tools"]:
            request["tools"] = kwargs["tools"]
        return request

    def parse_response(self, response_data: dict) -> dict:
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        result: dict = {
            "content": message.get("content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": response_data.get("usage", {}),
        }
        if "tool_calls" in message:
            result["tool_calls"] = message["tool_calls"]
        return result

    def map_error(self, status_code: int, body: dict) -> Exception:
        error_obj = body.get("error", body)
        msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
        return ProviderError(f"Anthropic error {status_code}: {msg}")
