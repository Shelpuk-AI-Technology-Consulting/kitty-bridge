"""OpenCode Go provider adapter — auto-routing by model across three endpoints.

OpenCode Go (https://opencode.ai) is a low-cost subscription ($10/month) that
provides reliable access to popular open coding models behind a single API key.
The provider serves its catalogue on **three** different endpoints, and the
adapter picks the right one from the model name so the user needs only one
profile:

- Anthropic Messages (``/v1/messages``) — ``_MESSAGES_MODELS``
- OpenAI Responses (``/v1/responses``) — ``_RESPONSES_MODELS``, **refused**
- Chat Completions (``/v1/chat/completions``) — every other model, the default

Model names are deliberately **not** listed in this docstring.  The previous
version listed five, four of which the provider had since retired, and nothing
noticed (KBR-126).  The routing table's oracle is
``tests/data/opencode_go_endpoints.json`` — a snapshot of the provider's
published endpoint table with its source URL and verification date — and
``tests/test_opencode_endpoint_table.py`` asserts the sets below agree with it.

**Why Responses models are refused rather than served.**  Writing an OpenAI
Responses body is three translation hooks kitty does not have on this transport,
and it would require replacing the boolean wire-shape declaration with a
three-valued type (``.system_design/TEST_SUITE.md`` §6.2.3).  That is KBR-137.
Until it lands, :class:`UnsupportedModelError` fails the request with a message
naming the model and the ticket, instead of silently posting a Chat Completions
body to an endpoint that does not speak it — which is what this adapter did
before, and which the provider answers with a ``401`` the bridge then reports to
the user as a bad API key.
"""

from __future__ import annotations

import json
import logging

from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.base import ProviderAdapter, ProviderError, UnsupportedModelError

__all__ = ["OpenCodeGoAdapter"]

logger = logging.getLogger(__name__)

# Models served via the Anthropic Messages API endpoint.  Mirrors the provider's
# published table; `tests/test_opencode_endpoint_table.py` holds them to it.
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
# `get_upstream_path` and refused by `translate_to_upstream` until KBR-137.
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


class OpenCodeGoAdapter(AnthropicAdapter):
    """OpenCode Go adapter with automatic endpoint routing.

    Routes on the model name across the provider's three endpoints:
    ``/v1/messages`` (Anthropic Messages), ``/v1/chat/completions``
    (passthrough, the default route), and ``/v1/responses`` — which
    :meth:`get_upstream_path` reports truthfully and
    :meth:`translate_to_upstream` refuses with
    :class:`~kitty.providers.base.UnsupportedModelError`, because kitty cannot
    write a Responses body on this transport yet (KBR-137).

    F16: Anthropic Messages translation is inherited from ``AnthropicAdapter``
    instead of duplicating the translation helpers here.
    """

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

        Reports ``/v1/responses`` truthfully for models this adapter refuses to
        serialize.  Returning the default route for them instead would cost
        nothing at runtime — no request is ever built — and would put back the
        lie in the routing table that KBR-126 exists to remove, as well as
        forcing an exemption list into the snapshot guard.

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
    def upstream_wire_is_messages_api(self) -> bool:
        """Default wire shape (Chat Completions).  Routed per model below.

        Overrides ``AnthropicAdapter``'s unconditional ``True``, which was
        false for every model outside ``_MESSAGES_MODELS`` (KBR-7).  Like
        ``upstream_path`` and ``build_upstream_headers`` above, this reports the
        adapter's default route; callers holding a model must ask
        :meth:`upstream_wire_is_messages_api_for_model`.
        """
        return False

    def upstream_wire_is_messages_api_for_model(self, model: str) -> bool:
        """Report the wire shape ``translate_to_upstream`` emits for *model*.

        Uses ``_is_messages_model`` on the raw name, which is exactly what
        ``translate_to_upstream`` routes on — deliberately without
        ``normalize_model_name``, so the two agree by construction rather than
        by a second copy of the rule.  Normalizing here would be more correct in
        isolation and less correct against the contract, which is agreement with
        the router.

        Args:
            model: The model name, exactly as ``translate_to_upstream`` reads it.

        Returns:
            True for the models served on ``/v1/messages``.
        """
        return _is_messages_model(model)

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

        This is the choke point for the Responses refusal.  Every request path
        in ``server.py`` reaches this method — directly or through
        ``BridgeServer._upstream_body_for`` — because this adapter is neither a
        custom-transport nor a native-passthrough one, so nothing can ship a
        body without passing here.

        Args:
            cc_request: The normalized Chat Completions request.

        Returns:
            An Anthropic Messages body for a Messages-routed model, otherwise
            the Chat Completions body unchanged.

        Raises:
            UnsupportedModelError: When the model is served on
                ``/v1/responses``, which this adapter cannot write yet.
        """
        model = cc_request.get("model", "")
        # Refuse before anything is built: the alternative is a Chat Completions
        # body at a Responses endpoint, which the provider rejects with a 401
        # that the bridge then reports to the user as a bad API key.
        if _is_responses_model(model):
            raise UnsupportedModelError(
                f"OpenCode Go serves {model!r} on the OpenAI Responses API (/v1/responses), "
                f"which kitty cannot speak for this provider yet (KBR-137). "
                f"Choose a model served on /v1/messages or /v1/chat/completions."
            )
        if _is_messages_model(model):
            return AnthropicAdapter.translate_to_upstream(self, cc_request)
        return ProviderAdapter.translate_to_upstream(self, cc_request)

    def translate_from_upstream(self, raw_response: dict) -> dict:
        # Anthropic responses have a "type" field; CC responses have "object"
        if raw_response.get("type") == "message":
            return AnthropicAdapter.translate_from_upstream(self, raw_response)
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
