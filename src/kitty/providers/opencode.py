"""OpenCode Go provider adapter — auto-routing by model across three endpoints.

OpenCode Go (https://opencode.ai) is a low-cost subscription ($10/month) that
provides reliable access to popular open coding models behind a single API key.
The provider serves its catalogue on **three** different endpoints, and the
adapter picks the right one from the model name so the user needs only one
profile:

- Anthropic Messages (``/v1/messages``) — ``_MESSAGES_MODELS``
- OpenAI Responses (``/v1/responses``) — ``_RESPONSES_MODELS`` (KBR-137)
- Chat Completions (``/v1/chat/completions``) — every other model, the default

Model names are deliberately **not** listed in this docstring.  The previous
version listed five, four of which the provider had since retired, and nothing
noticed (KBR-126).  The routing table's oracle is
``tests/data/opencode_go_endpoints.json`` — a snapshot of the provider's
published endpoint table with its source URL and verification date — and
``tests/test_opencode_endpoint_table.py`` asserts the sets below agree with it.

**KBR-137.**  Prior to this ticket the four ``/v1/responses`` models were
refused with a named :class:`~kitty.providers.base.UnsupportedModelError`
because no Responses body builder existed on the default aiohttp transport
(KBR-126's defensible workaround).  The adapter now builds the Responses
body in :meth:`OpenCodeGoAdapter._cc_to_responses`, decodes a Responses JSON
in :func:`_responses_to_cc`, and converts the Responses SSE stream to Chat
Completions chunks through :class:`OpenCodeGoResponsesCCStreamConverter`.
The wire-shape declaration became three-valued (per
``.system_design/TEST_SUITE.md`` §6.2.3) — the boolean ``False`` cannot
honestly express three wires.

**Thinking carriage on the Responses route.**  ``forwards_thinking_signature``
is intentionally ``False`` for the OpenCode Go adapter, so the KBR-228
``_thinking_blocks`` carriage carried on the Chat Completions message dict
is never restored on any route — including the Responses one.  The
Responses wire defines a ``reasoning`` item that *could* carry the
reasoning across, but translating the CC carriage into it is not done
here because (a) the four ``_RESPONSES_MODELS`` upstreams' tolerance for
``reasoning`` items in ``input`` is unverified, and (b) a live probe —
which this ticket cannot land for lack of a paid key — is the only honest
way to settle it.  KBR-246's evidence-gathering work is the documented
next step; until then the loss is deliberate, and a key-holder can land
the translation by populating ``_build_responses_input``'s assistant branch
with the corresponding ``reasoning`` item.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.base import ProviderAdapter, ProviderError, WireShape

__all__ = ["OpenCodeGoAdapter", "OpenCodeGoResponsesCCStreamConverter"]

logger = logging.getLogger(__name__)

# Models served via the Anthropic Messages API endpoint.  Mirrors the provider's
# published table; `tests/test_opencode_endpoint_table.py` holds them to it.
#
# These keys are matched against the NORMALIZED model name: since KBR-127 the
# URL path, the auth headers and the body all route on ``cc_request["model"]``,
# which has been through ``normalize_model_name``.  So ``normalize_model_name``
# below must preserve the dots — the base class's version replaces them with
# hyphens, and delegating to it would send every model here to
# ``/v1/chat/completions`` under Bearer auth, silently.  The two tests in
# ``tests/bridge/test_upstream_route_resolution.py`` that name a Messages model
# are what would go red.
#
# KBR-126 makes that warning load-bearing for far more names: six of the eight
# models below carry dots, as do all four in ``_RESPONSES_MODELS``.
_MESSAGES_MODELS: frozenset[str] = frozenset(
    {
        "minimax-m3",
        "minimax-m2.7",
        "minimax-m2.5",
        "qwen3.8-max",
        "qwen3.8-flash",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
    }
)

# Models served via the OpenAI Responses API endpoint.  Routed truthfully by
# `get_upstream_path`; sent via `_cc_to_responses` since KBR-137.
_RESPONSES_MODELS: frozenset[str] = frozenset(
    {
        "grok-4.6",
        "gpt-5.6-luna",
        "muse-spark-1.3-contributor",
        "muse-spark-1.2-contributor",
    }
)


def _is_messages_model(model: str) -> bool:
    """Return True if *model* should use the Anthropic Messages endpoint."""
    return model in _MESSAGES_MODELS


def _is_responses_model(model: str) -> bool:
    """Return True if *model* is served on the OpenAI Responses endpoint."""
    return model in _RESPONSES_MODELS


# ── Module-level helpers for the OpenAI Responses route (KBR-137) ──────────


def _responses_tool_choice(cc_tool_choice: Any) -> Any:
    """Rewrite a Chat Completions ``tool_choice`` into the Responses spelling.

    String modes (``"none"``, ``"auto"``, ``"required"``) pass through
    unchanged.  The named-function form unwraps from ``{"type":"function",
    "function":{"name":…}}`` to ``{"type":"function","name":…}}``.  Any other
    value is left verbatim — the Messages ingress cannot produce one and
    translating CC's allowed-tools and custom forms is not this ticket's to
    decide (same rationale as
    :func:`~kitty.providers.openai_subscription._responses_tool_choice`).

    Args:
        cc_tool_choice: A Chat Completions ``tool_choice`` value.

    Returns:
        The Responses spelling.
    """
    if (
        isinstance(cc_tool_choice, dict)
        and cc_tool_choice.get("type") == "function"
        and isinstance(cc_tool_choice.get("function"), dict)
        and isinstance(cc_tool_choice["function"].get("name"), str)
    ):
        return {"type": "function", "name": cc_tool_choice["function"]["name"]}
    return cc_tool_choice


def _response_format_to_text_format(response_format: dict) -> dict:
    """Move a CC ``response_format`` dict to the Responses ``text.format`` shape.

    CC nests the json_schema fields one level deeper than Responses:

    * CC:      ``{"type":"json_schema","json_schema":{"name","schema","strict"}}``
    * Responses: ``{"type":"json_schema","name","schema","strict"}`` (flat)

    ``text`` and ``json_object`` are spelled the same; passthrough for them.

    Args:
        response_format: The CC ``response_format`` dict; assumed already
            truthy (``isinstance(..., dict)``) by the caller.

    Returns:
        The Responses ``text.format`` value (without the wrapping ``text`` key).
    """
    rf_type = response_format.get("type")
    if rf_type == "json_schema":
        inner = response_format.get("json_schema")
        if isinstance(inner, dict):
            flat = {"type": "json_schema"}
            for key in ("name", "schema", "strict"):
                if key in inner:
                    flat[key] = inner[key]
            return flat
        return {"type": "json_schema"}
    if rf_type in ("text", "json_object"):
        return {"type": rf_type}
    # Unknown shape: carry verbatim so a future Responses field lands at its
    # documented address without this code reaching for it.
    return dict(response_format)


def _build_responses_input(messages: list[dict]) -> tuple[list[dict], str]:
    """Translate Chat Completions ``messages`` to Responses ``input`` + ``instructions``.

    System turns concatenate into the Responses ``instructions`` string.
    User turns become ``{"type":"message","role":"user","content":
    [{"type":"input_text","text":…}]``.  Assistant turns become
    ``{"type":"message","role":"assistant","content":
    [{"type":"output_text","text":…}]}`` with their text plus any
    ``tool_calls`` translated to ``function_call`` items.  Tool turns
    become ``{"type":"function_call_output","call_id":…,"output":…}``.

    The system text follows the same flattening KBR-222 applied to the
    Codex path: a list-form content flattens to its text parts.

    Args:
        messages: The CC ``messages`` list, assumed already extracted from the
            request body.

    Returns:
        A 2-tuple ``(input_items, instructions_text)`` ready to put on the
        Responses body.  Empty list and empty string when the input is
        empty.
    """
    input_items: list[dict] = []
    instructions_parts: list[str] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if isinstance(content, str):
                instructions_parts.append(content)
            elif isinstance(content, list):
                text = "\n".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
                if text:
                    instructions_parts.append(text)
            continue

        if role == "user":
            text = (
                content
                if isinstance(content, str)
                else (
                    "\n".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                    if isinstance(content, list)
                    else ""
                )
            )
            if text:
                input_items.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    }
                )
            continue

        if role == "assistant":
            text = content if isinstance(content, str) else (str(content) if content is not None else "")
            item: dict = {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}] if text else [],
            }
            has_content = bool(item["content"])
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                func = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", ""),
                    }
                )
            if has_content:
                input_items.append(item)
            continue

        if role == "tool":
            output = str(content) if content is not None else ""
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": output,
                }
            )

    return input_items, "\n\n".join(instructions_parts)


def _responses_to_cc(raw_response: dict) -> dict:
    """Translate an OpenAI Responses JSON body to a Chat Completions object.

    KBR-137 — the non-streaming reverse of :meth:`OpenCodeGoAdapter._cc_to_responses`.
    The Responses spec carries an ``output`` list of items, each either a
    ``message`` (text in ``content``) or a ``function_call`` (name, call_id,
    arguments).  Usage maps:

    * Responses ``input_tokens`` / ``output_tokens`` / ``total_tokens`` →
      CC ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``.

    Finish reason:

    * Any ``function_call`` in ``output`` → ``tool_calls``.
    * ``status == "incomplete"`` → ``length``.
    * Otherwise → ``stop``.

    Args:
        raw_response: The Responses JSON body, with the spec's spelling —
            ``output``, ``usage``, ``status``, ``model`` — or a recorded
            fixture that omits ``object``.

    Returns:
        A Chat Completions response object.
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    finish_reason = "stop"

    for index, item in enumerate(raw_response.get("output") or []):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("output_text", "text"):
                    text_parts.append(part.get("text", ""))
        elif item_type == "function_call":
            call_id = item.get("call_id") or f"call_{index}"
            tool_calls.append(
                {
                    "index": len(tool_calls),
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    },
                }
            )

    if tool_calls:
        finish_reason = "tool_calls"
    elif raw_response.get("status") == "incomplete":
        finish_reason = "length"

    message: dict = {"role": "assistant"}
    content = "".join(text_parts)
    message["content"] = content or None
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage_in = raw_response.get("usage") or {}
    cc_usage: dict = {}
    if usage_in:
        cc_usage = {
            "prompt_tokens": usage_in.get("input_tokens", 0),
            "completion_tokens": usage_in.get("output_tokens", 0),
            "total_tokens": usage_in.get(
                "total_tokens",
                usage_in.get("input_tokens", 0) + usage_in.get("output_tokens", 0),
            ),
        }

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": raw_response.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": cc_usage,
    }


class OpenCodeGoResponsesCCStreamConverter:
    """Convert one OpenAI Responses SSE stream into Chat Completions chunks.

    Mirrors the shape of
    :class:`~kitty.providers.anthropic.AnthropicCCStreamConverter`: one
    instance per upstream attempt, :meth:`feed` called once per
    upstream SSE line, output is one or more ``data: `` SSE lines.

    KBR-137 — the per-byte helper
    :meth:`OpenCodeGoAdapter.translate_upstream_stream_event` is stateless
    and therefore cannot track the output-item-ids the Responses stream
    emits, so on the Responses route the bridge allocates one of these
    converters per attempt (see
    :meth:`kitty.bridge.server.BridgeServer._stream_converter_for`) and
    feeds it each ``data:`` line before its own per-line logic.
    """

    #: Sentinel ``item_id`` for events that arrive without one — the
    #: delta/.done bookkeeping below must still see them as the same item,
    #: and an OpenAI Responses stream has only one message item in flight at
    #: a time, so conflating them is safe.
    _UNTRACKED = "<untracked>"

    def __init__(self) -> None:
        #: One id per stream — Chat Completions clients correlate a reply's
        #: chunks by it. A fresh UUID per attempt, mirroring the Anthropic
        #: converter.
        self._chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self._model = ""
        self._input_tokens = 0
        self._output_tokens = 0
        # Responses ``output_item`` id → Chat Completions tool_calls index.
        # The Responses stream carries items by id across ``output_item.added``
        # and ``function_call_arguments.delta`` events; the CC stream indexes
        # tool calls by positional index in document order.
        self._tool_indices: dict[str, int] = {}
        # Items whose arguments arrived whole in a ``.done`` event but no
        # earlier deltas were received; their function_call_arguments.delta
        # fragments would otherwise double the string under the same
        # ``index``. Mirrors ``AnthropicCCStreamConverter._arguments_complete``.
        self._arguments_complete: set[str] = set()
        # Items whose arguments crossed as at least one ``delta`` fragment.
        # The Responses spec's ``function_call_arguments.done`` carries the
        # FULL arguments precisely so a client that lost deltas can recover —
        # a client that received every delta must not also receive ``.done``'s
        # string, or the CC client concatenates both and the model sees its
        # tool arguments doubled.
        self._deltas_seen: set[str] = set()
        # Items whose text content has crossed as at least one ``delta`` —
        # symmetric to ``_deltas_seen``.  ``response.output_text.done`` carries
        # the full text so a client that lost deltas can recover; a client that
        # received every delta must not also receive ``.done``'s string.
        self._text_emitted: set[str] = set()

    def feed(self, raw_bytes: bytes) -> list[bytes]:
        """Convert one upstream SSE line into Chat Completions SSE lines.

        Args:
            raw_bytes: One SSE line as the handler read it, e.g.
                ``b'data: {...}\\n\\n'``.  Lines that do not start with a
                ``data:`` payload are ignored.

        Returns:
            ``data: `` SSE lines — Chat Completions chunks, the
            ``data: [DONE]`` sentinel on completion, or an empty list for
            events with no Chat Completions counterpart.
        """
        data_str: str | None = None
        raw_str = raw_bytes.decode("utf-8", errors="replace").strip()
        if not raw_str:
            return []
        for line in raw_str.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()
        if data_str is None:
            return []
        if data_str == "[DONE]":
            return [b"data: [DONE]\n\n"]
        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            return [raw_bytes]
        if not isinstance(event, dict):
            return [raw_bytes]

        event_type = event.get("type", "")

        if event_type == "response.created":
            response = event.get("response") or {}
            self._model = response.get("model", "")
            return [self._sse_chunk({"role": "assistant"})]

        if event_type == "response.in_progress":
            return []

        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") != "function_call":
                return []
            item_id = item.get("id") or f"fc_{len(self._tool_indices)}"
            cc_index = len(self._tool_indices)
            self._tool_indices[item_id] = cc_index
            arguments = item.get("arguments") or ""
            if arguments:
                self._arguments_complete.add(item_id)
            return [
                self._sse_chunk(
                    {
                        "tool_calls": [
                            {
                                "index": cc_index,
                                "id": item.get("call_id", ""),
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": arguments,
                                },
                            }
                        ]
                    }
                )
            ]

        if event_type in ("response.content_part.added", "response.content_part.done"):
            # The content part begins on the first ``output_text.delta``.
            return []

        if event_type == "response.output_text.delta":
            item_id = event.get("item_id") or self._UNTRACKED
            self._text_emitted.add(item_id)
            return [self._sse_chunk({"content": event.get("delta", "")})]

        if event_type == "response.output_text.done":
            # ``response.output_text.done`` carries the full text so a client
            # that lost the deltas can recover.  Two cases:
            #   * deltas already emitted — ``_text_emitted`` has the item id;
            #     .done is a no-op, otherwise the client concatenates both and
            #     the model's reply is doubled.
            #   * no deltas — some backends ship the full text only on .done.
            #     The carried text is the whole answer; emit it as a chunk or
            #     the client receives an empty reply.
            item_id = event.get("item_id") or self._UNTRACKED
            text = event.get("text")
            if text is None:
                return []
            if item_id in self._text_emitted:
                return []
            self._text_emitted.add(item_id)
            return [self._sse_chunk({"content": text})]

        if event_type == "response.function_call_arguments.delta":
            item_id = event.get("item_id") or event.get("id") or ""
            if item_id not in self._tool_indices or item_id in self._arguments_complete:
                return []
            self._deltas_seen.add(item_id)
            return [
                self._sse_chunk(
                    {
                        "tool_calls": [
                            {
                                "index": self._tool_indices[item_id],
                                "function": {"arguments": event.get("delta", "")},
                            }
                        ]
                    }
                )
            ]

        if event_type == "response.function_call_arguments.done":
            item_id = event.get("item_id") or event.get("id") or ""
            arguments = event.get("arguments") or ""
            if not item_id or item_id in self._arguments_complete:
                return []
            cc_index = self._tool_indices.get(item_id, -1)
            if cc_index == -1:
                return []
            # The deltas carried the same string the .done event carries; the
            # client already has it.  Skip the emission rather than duplicate.
            if item_id in self._deltas_seen:
                self._arguments_complete.add(item_id)
                return []
            self._arguments_complete.add(item_id)
            return [
                self._sse_chunk(
                    {
                        "tool_calls": [
                            {
                                "index": cc_index,
                                "function": {"arguments": arguments},
                            }
                        ]
                    }
                )
            ]

        if event_type in ("response.output_item.done",):
            return []  # tool-call state is settled by .done and the deltas above

        if event_type == "response.completed":
            response = event.get("response") or {}
            self._model = response.get("model", self._model)
            usage_in = (response.get("usage") or event.get("usage") or {})
            self._input_tokens = usage_in.get("input_tokens", 0)
            self._output_tokens = usage_in.get("output_tokens", 0)
            return [self._finish_chunk("stop"), b"data: [DONE]\n\n"]

        if event_type == "response.incomplete":
            # same shape as ``response.completed`` but finish_reason="length"
            return [self._finish_chunk("length"), b"data: [DONE]\n\n"]

        if event_type == "response.failed":
            # CC-shaped error chunk so the bridge's ``_is_upstream_stream_error``
            # detector (which keys on ``chunk["error"]``) sees it.  Mirrors the
            # Anthropic ``{"type":"error"}`` event's role.
            error = event.get("response", {}).get("error") or event.get("error") or {
                "message": "responses stream reported failure",
                "type": "upstream_error",
            }
            return [
                (
                    b'data: {"error": '
                    + json.dumps(error).encode()
                    + b"}\n\n"
                )
            ]

        if event_type == "error":
            # The Responses spec's ``{"type":"error"}`` event carries the
            # marker ``_is_upstream_stream_error`` keys on, so it crosses as-is.
            return [raw_bytes]

        # response.output_text.annotation_added, response.refusal.*,
        # response.reasoning_*, and the spec's other unmodelled events are
        # dropped — they are not Chat Completions chunks.
        return []

    def _finish_chunk(self, finish_reason: str) -> bytes:
        """Build the finish chunk (usage + finish_reason) followed by [DONE].

        Args:
            finish_reason: ``"stop"`` for completed, ``"length"`` for
                incomplete / hit ``max_output_tokens``.

        Returns:
            Two ``data: `` SSE lines: the finish chunk, then ``[DONE]``.
        """
        chunk: dict = {
            "id": self._chunk_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": self._model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": self._input_tokens,
                "completion_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
            },
        }
        return f"data: {json.dumps(chunk)}\n\n".encode()

    def _chunk(self, delta: dict) -> dict:
        """Build one Chat Completions chunk payload for this stream.

        Args:
            delta: The choice delta.

        Returns:
            The chunk payload, not yet SSE-wrapped.
        """
        return {
            "id": self._chunk_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": self._model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }

    def _sse_chunk(self, delta: dict) -> bytes:
        """Wrap one chunk payload as a ``data: `` SSE line.

        Args:
            delta: The choice delta.

        Returns:
            The encoded SSE line, trailing blank line included.
        """
        return f"data: {json.dumps(self._chunk(delta))}\n\n".encode()


class OpenCodeGoAdapter(AnthropicAdapter):
    """OpenCode Go adapter with automatic endpoint routing.

    Routes on the model name across the provider's three endpoints:
    ``/v1/messages`` (Anthropic Messages), ``/v1/chat/completions``
    (passthrough, the default route), and ``/v1/responses`` (OpenAI
    Responses).  All three are now servable since KBR-137 — the
    :class:`~kitty.providers.base.UnsupportedModelError` the KBR-126
    refusal raised for the Responses-routed models is gone.

    F16: Anthropic Messages translation is inherited from ``AnthropicAdapter``
    instead of duplicating the translation helpers here.
    """

    #: The Messages route serves MiniMax and Qwen models, whose upstreams do not
    #: document ``display``.  KBR-203, decision D2.
    forwards_thinking_display = False

    #: Keeps the empty placeholder thinking block (register row P5e): whether
    #: these upstreams accept a history with no thinking block is unverified,
    #: and the route must not change wire behaviour without evidence
    #: (KBR-228 part C).
    injects_placeholder_thinking = True

    #: Restores no signed thinking blocks and no verbatim system on the
    #: Messages route: these upstreams' tolerance for ``signature`` and
    #: ``redacted_thinking`` fields is unverified, and M17's recovery
    #: recognises only Anthropic's rejection wording (KBR-228 part B).
    forwards_thinking_signature = False

    #: The Messages route's upstreams (MiniMax rejects the field outright; Qwen
    #: does not document it) cannot safely receive ``output_config``.  KBR-224.
    forwards_output_config = False

    @property
    def provider_type(self) -> str:
        return "opencode_go"

    @property
    def default_base_url(self) -> str:
        return "https://opencode.ai/zen/go"

    @property
    def validation_model(self) -> str:
        """Use a known-valid model for key validation.

        OpenCode returns 401 for an unsupported model, which the bridge would
        report as an auth failure — so this must name a model the provider
        actually serves.  The previous value, ``glm-5``, had left the catalogue
        (KBR-126), which is the very failure this field exists to avoid.

        It must also be a model on the **Chat Completions** route:
        :func:`kitty.validation.validate_api_key` posts a Chat Completions body
        with the bare :meth:`build_upstream_headers` (Bearer) to
        ``get_upstream_path(normalize_model_name(validation_model))``.  Naming a
        Messages-routed model here — ``minimax-m2.7``, say — would send the wrong
        dialect with the wrong auth and fail every key check.
        ``tests/test_validation_model_routing.py`` enforces that for every
        adapter, not just this one.

        Distinct from the wire-shape guard's ``default_route_model``, which
        answers a different question; the two are deliberately not the same
        value.
        """
        return "mimo-v2.5"

    def normalize_model_name(self, model: str) -> str:
        """Strip provider prefix if present (e.g. ``opencode/glm-5.2``)."""
        if "/" in model:
            return model.rsplit("/", 1)[-1]
        return model

    # ── Per-model routing ─────────────────────────────────────────────────

    @property
    def upstream_path(self) -> str:  # noqa: D401 — overridden by get_upstream_path
        """Default path (Chat Completions).  ``get_upstream_path`` routes per model."""
        return "/v1/chat/completions"

    def get_upstream_path(self, model: str) -> str:
        """Return the endpoint the provider serves *model* on.

        Reports ``/v1/responses`` for the ``_RESPONSES_MODELS`` set, ``/v1/messages``
        for the Messages set, else the Chat Completions default.  Reporting the
        default route for the routed models would put back the lie in the
        routing table that KBR-126 exists to remove, as well as forcing an
        exemption list into the snapshot guard.

        Args:
            model: The model name, as ``translate_to_upstream`` reads it.

        Returns:
            One of ``/v1/messages``, ``/v1/responses`` or
            ``/v1/chat/completions``.
        """
        if _is_messages_model(model):
            return "/v1/messages"
        if _is_responses_model(model):
            return "/v1/responses"
        return "/v1/chat/completions"

    @property
    def upstream_wire_shape(self) -> WireShape:
        """Default wire shape (:attr:`WireShape.CHAT_COMPLETIONS`).  Routed per model below.

        Overrides ``AnthropicAdapter``'s unconditional ``WireShape.MESSAGES``,
        which was wrong for every model outside ``_MESSAGES_MODELS`` (KBR-7).
        Like ``upstream_path`` and ``build_upstream_headers`` above, this
        reports the adapter's default route; callers holding a model must ask
        :meth:`upstream_wire_shape_for_model`.

        KBR-137 added :attr:`WireShape.RESPONSES` for the four models served
        on ``/v1/responses`` — replacing the KBR-126 refusal with a real
        declaration, per §6.2.3's "replace, don't extend" rule.
        """
        return WireShape.CHAT_COMPLETIONS

    def upstream_wire_shape_for_model(self, model: str) -> WireShape:
        """Report the wire shape ``translate_to_upstream`` emits for *model*.

        Uses the routing predicates on the string it is given — exactly what
        ``translate_to_upstream`` routes on — deliberately without calling
        ``normalize_model_name`` again, so the two agree by construction
        rather than by a second copy of the rule.  Normalizing here would be
        more correct in isolation and less correct against the contract,
        which is agreement with the router.

        Args:
            model: The model name, exactly as ``translate_to_upstream`` reads it:
                the normalized model in ``cc_request``, since KBR-7 for the body
                and since KBR-127 for the path and auth headers too.

        Returns:
            :attr:`WireShape.MESSAGES` for ``_MESSAGES_MODELS``,
            :attr:`WireShape.RESPONSES` for ``_RESPONSES_MODELS``, else
            :attr:`WireShape.CHAT_COMPLETIONS`.
        """
        if _is_messages_model(model):
            return WireShape.MESSAGES
        if _is_responses_model(model):
            return WireShape.RESPONSES
        return WireShape.CHAT_COMPLETIONS

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Default headers (Chat Completions — Bearer auth)."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def build_upstream_headers_for_model(self, api_key: str, model: str) -> dict[str, str]:
        """Build auth headers appropriate for the model's endpoint."""
        if _is_messages_model(model):
            return AnthropicAdapter.build_upstream_headers(self, api_key)
        return self.build_upstream_headers(api_key)

    # ── Routed translation ─────────────────────────────────────────────────

    def translate_to_upstream(self, cc_request: dict) -> dict:
        """Serialize *cc_request* in the dialect this model's endpoint speaks.

        Every request path in ``server.py`` reaches this method — directly or
        through ``BridgeServer._upstream_body_for`` — because this adapter is
        neither a custom-transport nor a native-passthrough one, so nothing
        can ship a body without passing here.

        Args:
            cc_request: The normalized Chat Completions request.

        Returns:
            An Anthropic Messages body for a Messages-routed model, an OpenAI
            Responses body for a Responses-routed model, otherwise the Chat
            Completions body unchanged.
        """
        model = cc_request.get("model", "")
        if _is_messages_model(model):
            return AnthropicAdapter.translate_to_upstream(self, cc_request)
        if _is_responses_model(model):
            return self._cc_to_responses(cc_request)
        return ProviderAdapter.translate_to_upstream(self, cc_request)

    # ── OpenAI Responses route (KBR-137) ──────────────────────────────────

    #: Chat Completions fields absent from the OpenAI Responses create-request
    #: schema (verified 2026-09-16 against ``openai/openai-openapi`` master).
    #: Carried by register row P37 and dropped at DEBUG so a user can see why
    #: a setting silently does nothing on these four models.
    _RESPONSES_DROPPED_CC_FIELDS: frozenset[str] = frozenset(
        {
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "logit_bias",
            "n",
            "stop",
            "logprobs",
            "stream_options",
        }
    )

    def _cc_to_responses(self, cc_request: dict) -> dict:
        """Build an OpenAI Responses body from a Chat Completions request.

        Every Responses-routed model (``grok-4.6``, ``gpt-5.6-luna``,
        ``muse-spark-1.3-contributor``, ``muse-spark-1.2-contributor``)
        reaches this method.  The published Responses API spec is the
        authoritative source for what it accepts — verified 2026-09-16
        against ``openai/openai-openapi`` master.

        Register rows P36–P42 (added 2026-09-16 in ``tests/harness/register.py``
        and ``.system_design/TEST_SUITE.md`` §3.2.2) record each mutation
        below.  A change that drops or renames a field without a row turns
        the §6.2.3 register guard red.

        Args:
            cc_request: The normalized Chat Completions request.  Its ``model``
                is the route-deciding key — assumed to be one of
                ``_RESPONSES_MODELS`` by the caller.

        Returns:
            A body shaped for the OpenAI Responses create-request endpoint.
        """
        # Register row P37 — log dropped CC fields at DEBUG so a user can see
        # why a setting silently does nothing on these four models.
        dropped = set(cc_request) & self._RESPONSES_DROPPED_CC_FIELDS
        if dropped:
            logger.debug("OpenCode Go Responses unsupported parameters (dropped): %s", sorted(dropped))

        body: dict = {"model": cc_request.get("model", "")}

        # The bridge decides whether to stream; honour its choice.  ``store``
        # is a Codex-specific construct (P17) and is not injected here —
        # OpenCode Go's Responses backend accepts the spec default.
        if "stream" in cc_request and cc_request["stream"] is not None:
            body["stream"] = cc_request["stream"]

        # Register row P38 — the ceiling under its Responses spelling, with a
        # fixed precedence: max_output_tokens > max_completion_tokens >
        # max_tokens.  Only one is forwarded.
        for cc_key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
            value = cc_request.get(cc_key)
            if value is not None:
                body["max_output_tokens"] = value
                break

        for key in ("temperature", "top_p", "top_logprobs", "parallel_tool_calls"):
            value = cc_request.get(key)
            if value is not None:
                body[key] = value

        # Register row P42 — inject reasoning.effort from the normalized
        # metadata, in the target's own spelling (P3/P4 class).
        effort = cc_request.get("_reasoning_effort")
        if effort and effort != "none":
            body["reasoning"] = {"effort": effort}

        # P36 — tools envelope unwrap.  ``strict`` is carried:
        # the OpenAI Responses spec defines the field, and P15's Codex-specific
        # strip does not apply to this provider.
        cc_tools = cc_request.get("tools") or []
        if cc_tools:
            tools: list[dict] = []
            for tool in cc_tools:
                if not isinstance(tool, dict):
                    continue
                func = tool.get("function")
                if not isinstance(func, dict):
                    continue
                flat: dict = {"type": "function"}
                for src_key in ("name", "description", "parameters", "strict"):
                    if src_key in func:
                        flat[src_key] = func[src_key]
                tools.append(flat)
            body["tools"] = tools

        # P36 — tool_choice envelope unwrap.
        if cc_request.get("tool_choice") is not None:
            body["tool_choice"] = _responses_tool_choice(cc_request["tool_choice"])

        # P36 — response_format moves to text.format.
        response_format = cc_request.get("response_format")
        if isinstance(response_format, dict):
            body["text"] = {"format": _response_format_to_text_format(response_format)}

        # Register row P36 — hoist messages into ``instructions`` (system)
        # and ``input`` (the rest).
        body["input"], instructions = _build_responses_input(cc_request.get("messages") or [])
        if instructions:
            body["instructions"] = instructions

        return body

    def translate_from_upstream(self, raw_response: dict) -> dict:
        """Translate an upstream JSON response to a Chat Completions object.

        Routes on the response's shape:

        - Anthropic Messages responses carry ``type == "message"``; defer
          to the inherited translator.
        - OpenAI Responses responses (KBR-137) carry ``object ==
          "response"`` — the spec spelling — or, for recorded fixtures that
          omit it, an ``output`` list.  Both are translated.
        - Anything else is treated as Chat Completions and returned unchanged.
        """
        if raw_response.get("type") == "message":
            return AnthropicAdapter.translate_from_upstream(self, raw_response)
        if raw_response.get("object") == "response" or isinstance(raw_response.get("output"), list):
            return _responses_to_cc(raw_response)
        return raw_response

    def translate_upstream_stream_event(self, raw_bytes: bytes) -> list[bytes]:
        """Auto-detect SSE format and translate Anthropic events to CC chunks.

        Anthropic events have a ``"type"`` field (message_start,
        content_block_delta, etc.) while Chat Completions events have
        ``"object": "chat.completion.chunk"``.
        """
        raw_str = raw_bytes.decode("utf-8", errors="replace").strip()
        if not raw_str:
            return []

        for line in raw_str.split("\n"):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                return [raw_bytes]
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                return [raw_bytes]
            event_type = data.get("type", "")
            if event_type in (
                "message_start",
                "message_delta",
                "message_stop",
                "content_block_start",
                "content_block_stop",
                "content_block_delta",
                "ping",
                "error",
            ):
                return AnthropicAdapter.translate_upstream_stream_event(self, raw_bytes)
            return [raw_bytes]

        return [raw_bytes]

    def translate_upstream_stream_event_for_model(self, raw_bytes: bytes, model: str) -> list[bytes]:
        """Translate SSE events, routing based on model."""
        if _is_messages_model(model):
            return AnthropicAdapter.translate_upstream_stream_event(self, raw_bytes)
        return [raw_bytes]

    # ── Standard ProviderAdapter methods ───────────────────────────────────

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        request: dict = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
        }
        if "tools" in kwargs and kwargs["tools"]:
            request["tools"] = kwargs["tools"]
        for key in ("temperature", "top_p", "max_tokens"):
            if key in kwargs and kwargs[key] is not None:
                request[key] = kwargs[key]
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
        return ProviderError(f"OpenCode Go error {status_code}: {msg}")
