"""Anthropic provider adapter — translates between Chat Completions and Anthropic Messages API."""

from __future__ import annotations

import json
import logging
import uuid
from typing import cast

from kitty.providers.base import ProviderAdapter, ProviderError

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
    """

    forwards_thinking_display: bool = True

    injects_placeholder_thinking: bool = False

    forwards_thinking_signature: bool = True

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
    def upstream_wire_is_messages_api(self) -> bool:
        """True — this class's ``translate_to_upstream`` emits Messages API.

        Holds for subclasses that do **not** route by model: they either
        forward a native Messages body unchanged or fall back to this class's
        Chat Completions → Messages translation, so the wire shape does not
        depend on ``_native_messages_request``.

        It does **not** hold for a subclass that routes by model.
        :class:`~kitty.providers.opencode.OpenCodeGoAdapter` emits Chat
        Completions for every model outside its ``_MESSAGES_MODELS``, and
        inheriting this ``True`` was KBR-7.  Such a subclass must override
        **both** this property, reporting its default route, and
        ``upstream_wire_is_messages_api_for_model``, mirroring its own routing.
        """
        return True

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

        # Extract system messages → top-level system field
        system_parts: list[str] = []
        for msg in cc_request.get("messages", []):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    # System content as list of blocks — extract text
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_parts.append(block["text"])
                        elif isinstance(block, str):
                            system_parts.append(block)

        # KBR-228 part B: on the signature-binding routes the agent's own
        # system value — blocks and cache breakpoints included — is what the
        # thinking signatures are bound to, so it is restored verbatim from
        # the carriage instead of this joined string.  Where the carriage is
        # absent (a Chat Completions origin) or the upstream is unverified,
        # today's join stands.
        carried_system = cc_request.get("_anthropic_system")
        if carried_system is not None and self.forwards_thinking_signature:
            if isinstance(carried_system, list):
                anthropic["system"] = [
                    dict(block) if isinstance(block, dict) else block for block in carried_system
                ]
            else:
                anthropic["system"] = carried_system
        elif system_parts:
            anthropic["system"] = "\n".join(system_parts)

        # Translate messages
        for msg in cc_request.get("messages", []):
            role = msg.get("role")
            if role == "system":
                continue  # already handled above

            if role == "assistant":
                anthropic["messages"].append(self._translate_assistant_msg(msg, cc_request))
            elif role == "tool":
                anthropic["messages"].append(self._translate_tool_result_msg(msg))
            else:
                anthropic["messages"].append(
                    {
                        "role": role,
                        "content": msg.get("content", ""),
                    }
                )

        # Translate tools
        if "tools" in cc_request and cc_request["tools"]:
            anthropic["tools"] = self._translate_tools(cc_request["tools"])

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

        # Restore thinking from normalized effort metadata
        if cc_request.get("_thinking_adaptive"):
            # Claude Code sent thinking: {type: "adaptive"} — forward it
            # verbatim to the Anthropic-compatible upstream.  This lets the
            # provider decide the budget automatically.
            anthropic["thinking"] = self._with_thinking_display({"type": "adaptive"}, cc_request)
        elif cc_request.get("_thinking_enabled"):
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
            content_blocks.append({"type": "text", "text": text})

        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": func.get("name", ""),
                    "input": _safe_json_load_args(func.get("arguments")),
                }
            )

        return {"role": "assistant", "content": content_blocks or ""}

    def _translate_tool_result_msg(self, msg: dict) -> dict:
        """Translate a tool result message to Anthropic user message with tool_result block."""
        content = msg.get("content", "")
        # Anthropic requires tool_result to be inside a user message
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content,
                }
            ],
        }

    def _translate_tools(self, cc_tools: list[dict]) -> list[dict]:
        """Translate CC tool definitions to Anthropic format."""
        anthropic_tools = []
        for tool in cc_tools:
            func = tool.get("function", {})
            anthropic_tools.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                }
            )
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
