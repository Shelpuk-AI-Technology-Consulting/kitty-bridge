"""Wire-shape honesty at the serialization boundary — the T-G4 wire form.

The hook-level guard in ``test_wire_shape_honesty`` (KBR-7, extended by
KBR-137) asserts that an adapter's declared wire shape agrees with the
dialect its ``translate_to_upstream`` emits.  That guard observes the
hook's return value.  For three adapters that set ``use_custom_transport``,
the hook is not the boundary that ships:

* ``openai_subscription`` never invokes the hook on the request path —
  ``_cc_to_responses`` (CC-origin) and ``_prepare_responses_body``
  (Responses-origin) build the Responses body inside the curl_cffi
  transport (P13–P17).  The inherited declaration (``CHAT_COMPLETIONS``)
  is stale: the shipped body is Responses, so the declaration is
  corrected here to match — the KBR-7 atomic pattern.  The bridge's
  own request path bypasses this hook, so the correction has no consumer
  impact (``_make_upstream_request`` does the repair at server.py
  ~5075; ``_get_streaming_converter`` at 9849 is on the bridge-SSE path;
  ``_serves_messages_wire`` at 9777 returns False either way).
* ``bedrock`` calls the hook and the transport then pops ``modelId`` and
  ``stream`` (P18).  The pops are scalar and do not change the shape
  family (Converse remains OTHER).
* ``ollama_cloud`` calls the hook and the transport then overwrites
  ``stream`` (P19).  Scalar, does not change the shape family (Chat
  Completions remains Chat Completions).

This module also extends the sweep to adapters constructed with
``provider_config`` and to native-passthrough requests — both were
explicitly out of the hook sweep's coverage.

The classifier, the fixture bodies, and ``CUSTOM_TRANSPORT_ADAPTERS``
are imported from the hook module so the two files stay in lockstep on
what "shape" means.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from test_wire_shape_honesty import (  # sibling module via pytest prepend
    _MESSAGES_BODY,
    CUSTOM_TRANSPORT_ADAPTERS,
    _probe_request,
    classify_wire_shape,
)

from kitty.auth.oauth_session import OAuthSession
from kitty.providers.base import ProviderAdapter, WireShape
from kitty.providers.openai_subscription import OpenAISubscriptionAdapter
from kitty.providers.registry import get_provider

# Contract guard.  The L2 default for ``tests/*.py`` is L1 (mutation
# testing) per ``tests/layers.py``'s ``_FALLBACK_LAYER``; without this
# marker the file lands in the L1 set, and the hook module's comment
# above its own ``pytestmark`` names the consequence.
pytestmark = pytest.mark.l2

# Adapters this file captures the wire body for.  Maintained by hand and
# asserted equal to ``CUSTOM_TRANSPORT_ADAPTERS`` below — a fourth custom
# transport forces a decision (capture + set update).
_WIRE_CAPTURED_ADAPTERS = frozenset({"bedrock", "ollama_cloud", "openai_subscription"})


def test_every_custom_transport_adapter_has_a_wire_capture():
    """A fourth custom-transport adapter forces a decision here too.

    Mirrors the hook module's
    ``test_custom_transport_adapters_are_the_known_exempt_set``: the
    captured set is asserted, not narrated.
    """
    assert set(CUSTOM_TRANSPORT_ADAPTERS) == _WIRE_CAPTURED_ADAPTERS


# ── Minimal transport stubs ─────────────────────────────────────────────────


_BEDROCK_MINIMAL_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 1, "outputTokens": 1},
}

_OLLAMA_MINIMAL_RESPONSE = {
    "model": "kitty-test-model",
    "message": {"role": "assistant", "content": "ok"},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 1,
    "eval_count": 1,
}


# ── Bedrock (P18: transport pops modelId + stream) ──────────────────────────


async def test_bedrock_wire_body_classifies_as_other():
    """The body handed to botocore is a Converse body, as declared (OTHER).

    Drives ``make_request`` with a stubbed boto3 client (the pattern in
    ``tests/test_provider_bedrock.py:722-728``) and inspects the kwargs
    to ``client.converse`` after the P18 pops.  ``modelId`` is re-passed
    as a boto3 argument; the wire body is the kwargs minus ``modelId``.
    ``stream`` is popped with a default, so its absence is unconditional.
    """
    adapter = get_provider("bedrock")
    probe = copy.deepcopy(_probe_request("kitty-test-model"))

    mock_client = MagicMock()
    mock_client.converse.return_value = copy.deepcopy(_BEDROCK_MINIMAL_RESPONSE)

    with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
        await adapter.make_request(probe)

    wire_body = {k: v for k, v in mock_client.converse.call_args.kwargs.items() if k != "modelId"}

    assert "modelId" not in wire_body  # P18 pin: the transport popped it
    assert "stream" not in wire_body  # P18 pin: the transport popped it
    assert classify_wire_shape(wire_body) is adapter.upstream_wire_shape  # OTHER


async def test_bedrock_falsification_catches_a_dishonest_body():
    """A Messages-shaped Converse body is visible at the wire.

    The defect lives on the adapter side — monkeypatching
    ``translate_to_upstream`` to emit a Messages body — so the guard's
    sensitivity to adapter-side dishonesty is what is being tested.
    The Messages body carries ``modelId`` so the transport's pop does
    not crash.
    """
    adapter = get_provider("bedrock")
    dishonest = {
        "modelId": "kitty-test-model",
        "system": "You are a reviewer.",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "read_file", "description": "Read a file", "input_schema": {"type": "object"}}
        ],
    }

    mock_client = MagicMock()
    mock_client.converse.return_value = copy.deepcopy(_BEDROCK_MINIMAL_RESPONSE)

    with (
        patch.object(adapter, "translate_to_upstream", return_value=dishonest),
        patch.object(adapter, "_get_boto3_client", return_value=mock_client),
    ):
        await adapter.make_request(copy.deepcopy(_probe_request("kitty-test-model")))

    wire_body = {k: v for k, v in mock_client.converse.call_args.kwargs.items() if k != "modelId"}
    observed = classify_wire_shape(wire_body)

    assert observed is WireShape.MESSAGES
    assert observed is not adapter.upstream_wire_shape


# ── Ollama Cloud (P19: transport overwrites stream) ──────────────────────────


async def test_ollama_cloud_wire_body_classifies_as_chat_completions():
    """The body handed to aiohttp is Chat Completions, as declared.

    Drives ``make_request`` with a stubbed aiohttp session (the pattern
    in ``tests/test_provider_ollama_cloud.py:450-457``) and inspects the
    ``json=`` kwarg to ``session.post`` after the P19 overwrite.
    ``make_request`` sets ``stream=False`` for the non-streaming path.
    """
    adapter = get_provider("ollama_cloud")
    # ``stream=True`` so the P19 overwrite (transport sets False) is
    # observable, not just a set-on-absent.
    probe = copy.deepcopy(_probe_request("kitty-test-model"))
    probe["stream"] = True

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=copy.deepcopy(_OLLAMA_MINIMAL_RESPONSE))
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)

    with patch.object(adapter, "_get_session", return_value=mock_session):
        await adapter.make_request(probe)

    wire_body = mock_session.post.call_args.kwargs["json"]

    assert wire_body["stream"] is False  # P19 pin: the transport overwrote it
    assert classify_wire_shape(wire_body) is adapter.upstream_wire_shape  # CHAT_COMPLETIONS


async def test_ollama_cloud_falsification_catches_a_responses_body():
    """A Responses-shaped body survives the transport's stream overwrite.

    The defect is on the adapter side (monkeypatched translate); the
    transport's overwrite is scalar and does not change the shape
    family, so the dishonesty is observable at the wire.
    """
    adapter = get_provider("ollama_cloud")
    dishonest = {
        "model": "kitty-test-model",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ],
        "tools": [
            {"type": "function", "name": "read_file", "parameters": {"type": "object"}}
        ],
    }

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=copy.deepcopy(_OLLAMA_MINIMAL_RESPONSE))
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)

    with (
        patch.object(adapter, "translate_to_upstream", return_value=dishonest),
        patch.object(adapter, "_get_session", return_value=mock_session),
    ):
        await adapter.make_request(copy.deepcopy(_probe_request("kitty-test-model")))

    wire_body = mock_session.post.call_args.kwargs["json"]
    observed = classify_wire_shape(wire_body)

    assert observed is WireShape.RESPONSES
    assert observed is not adapter.upstream_wire_shape


# ── OpenAI Subscription (P13–P17: body built inside transport) ──────────────


def test_openai_subscription_responses_origin_body_classifies_as_responses():
    """``_prepare_responses_body`` emits a Responses body, as declared."""
    adapter = get_provider("openai_subscription")
    cc_request = {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]}
    original_body = {
        "model": "gpt-5.4",
        "instructions": "You are a reviewer.",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ],
        "tools": [
            {"type": "function", "name": "read_file", "description": "Read a file",
             "parameters": {"type": "object", "properties": {}}}
        ],
    }

    body = adapter._prepare_responses_body(cc_request, original_body)

    assert classify_wire_shape(body) is adapter.upstream_wire_shape  # RESPONSES


def test_openai_subscription_cc_origin_body_classifies_as_responses():
    """``_cc_to_responses`` emits a Responses body, as declared.

    The §3.2.3 table identifies the body builder as the ship boundary
    for this adapter; the transport is a transparent JSON carrier.  This
    test exercises that boundary directly.
    """
    adapter = get_provider("openai_subscription")

    body = adapter._cc_to_responses(copy.deepcopy(_probe_request("kitty-test-model")))

    assert classify_wire_shape(body) is adapter.upstream_wire_shape  # RESPONSES


def test_openai_subscription_falsification_catches_a_non_responses_body():
    """A Chat-Completions body from the builder trips the wire guard.

    The defect is on the builder (monkeypatched ``_cc_to_responses``).
    The CHAT_COMPLETIONS body carries a ``messages`` list with a system
    turn and ``function``-envelope tools so the classifier's
    CHAT_COMPLETIONS arm fires (it requires ``"system" in roles``).
    """
    adapter = get_provider("openai_subscription")
    cc_request = copy.deepcopy(_probe_request("kitty-test-model"))
    dishonest = {
        "model": cc_request["model"],
        "max_tokens": cc_request["max_tokens"],
        "messages": cc_request["messages"],  # _probe_request includes a system turn
        "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
    }

    with patch.object(adapter, "_cc_to_responses", return_value=dishonest):
        body = adapter._cc_to_responses(cc_request)

    observed = classify_wire_shape(body)
    assert observed is WireShape.CHAT_COMPLETIONS
    assert observed is not adapter.upstream_wire_shape  # declared RESPONSES


def _make_codex_id_token(account_id: str | None = None) -> str:
    """Build a minimal JWT id_token standing in for the Codex session."""
    header = "eyJhbGciOiJIUzI1NiJ9"
    payload_dict: dict = {}
    if account_id is not None:
        payload_dict["https://api.openai.com/auth"] = {"chatgpt_account_id": account_id}
    payload = base64.urlsafe_b64encode(json.dumps(payload_dict).encode()).rstrip(b"=").decode()
    return f"{header}.{payload}.fake_sig"


@pytest.fixture()
def fresh_codex_session(tmp_path: Path) -> Path:
    """A fresh, unexpired Codex OAuth session file on disk.

    Replicates the ``fresh_session`` fixture from
    ``tests/providers/test_openai_subscription.py:172-188`` so this module
    is self-contained when run in isolation — pytest's prepend import
    mode does not put a sibling test's directory on ``sys.path`` unless
    that sibling is also being collected.
    """
    now = time.time()
    session = OAuthSession(
        client_id="app_test",
        access_token="at_fresh",
        refresh_token="rt_fresh",
        id_token=_make_codex_id_token("acct-1234"),
        api_key=None,
        access_token_expires_at=now + 3600,
        api_key_expires_at=now + 3600,
        _file_path=str(tmp_path / "oauth_session.json"),
    )
    session.save()
    return Path(session._file_path)


@contextlib.contextmanager
def _mock_curl_capture_session(captured: dict):
    """Patch ``_curl_session`` with an AsyncMock that records its ``post`` call.

    Mirrors ``tests/providers/test_openai_subscription.py:91-110``'s
    ``_mock_curl_session``.  The captured ``json=`` is what the bridge
    ships upstream; the SSE response body is the minimal Codex success
    shape used by ``test_parses_sse_to_cc_response`` (lines 1425-1431).
    """
    sse_body = (
        b'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
        b'data: {"type":"response.completed","response":{"model":"gpt-5.4",'
        b'"status":"completed","usage":{"input_tokens":10,"output_tokens":5}}}\n\n'
        b"data: [DONE]\n\n"
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = ""
    mock_resp.content = sse_body

    mock_session = AsyncMock()

    def _record_post(*args, **kwargs):
        captured["json"] = kwargs.get("json")
        return mock_resp

    mock_session.post = AsyncMock(side_effect=_record_post)
    mock_session.close = MagicMock()

    with patch.object(
        OpenAISubscriptionAdapter,
        "_curl_session",
        new_callable=PropertyMock,
        return_value=mock_session,
    ):
        yield mock_session


async def test_openai_subscription_transport_passes_body_through_unchanged(fresh_codex_session):
    """The transport ships the body builder's output, byte-for-byte.

    Structural pin for the §3.2.3 boundary: between the body builder
    and the curl_cffi ``post`` call, the transport must not mutate the
    body.  A future transport-level body mutation — say, a retry
    adding a ``previous_response_id`` — would fail here and force a
    re-derivation of the wire shape.

    ``_cc_to_responses`` already emits ``stream=True`` internally
    (line 1142) and ``make_request``'s forced ``stream=True`` is
    idempotent; no overlay is needed.
    """
    adapter = get_provider("openai_subscription")
    cc_request = {
        "model": "gpt-5.4",
        "messages": [
            {"role": "system", "content": "You are a reviewer."},
            {"role": "user", "content": "Review this diff"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file", "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "stream": False,
        "_resolved_key": str(fresh_codex_session),
    }

    captured: dict = {}
    with _mock_curl_capture_session(captured):
        await adapter.make_request(cc_request)

    expected = adapter._cc_to_responses(copy.deepcopy(cc_request))

    assert captured["json"] == expected


# ── Provider_config-constructed adapters (R2) ──────────────────────────────


@pytest.mark.parametrize("native", [True, False])
def test_minimax_token_wire_shape_with_and_without_native_messages(native):
    """``provider_config`` construction does not break wire-shape honesty.

    ``minimax_token`` inherits ``MESSAGES`` from ``AnthropicAdapter``
    and does not route by model — the body builder's output IS the wire
    body for both constructions.  The native flag controls the bridge's
    forward path, not the adapter's translate.  This test pins that
    provider_config construction preserves the declaration across the
    flag's two values.
    """
    adapter = get_provider("minimax_token", {"native_messages": native})
    model = "kitty-test-model"

    assert adapter.use_native_messages is native

    body = adapter.translate_to_upstream(copy.deepcopy(_probe_request(model)))
    assert classify_wire_shape(body) is adapter.upstream_wire_shape_for_model(model)


def test_provider_config_falsification_flips_the_native_declaration():
    """A flipped declaration makes the wire sweep's pairing trip.

    The sweep pairs ``classify(captured)`` against
    ``adapter.upstream_wire_shape_for_model(model)`` — the same
    discipline the hook module's sweep uses.  Patching the declaration
    to a wrong value is the defect that the pairing is designed to catch.
    """
    adapter = get_provider("minimax_token", {"native_messages": True})
    model = "kitty-test-model"
    body = adapter.translate_to_upstream(copy.deepcopy(_probe_request(model)))

    with patch.object(
        type(adapter), "upstream_wire_shape_for_model", return_value=WireShape.CHAT_COMPLETIONS
    ):
        declared = adapter.upstream_wire_shape_for_model(model)

    observed = classify_wire_shape(body)
    assert declared is WireShape.CHAT_COMPLETIONS
    assert observed is WireShape.MESSAGES
    assert observed is not declared


# ── Native-passthrough wire shape (R3) ─────────────────────────────────────


def _native_probe() -> dict:
    """A /v1/messages request carrying the native-passthrough flag.

    Built from the hook module's ``_MESSAGES_BODY`` fixture — a Messages
    request shape — with ``_native_messages_request`` set so each
    adapter's native branch fires.
    """
    probe = copy.deepcopy(_MESSAGES_BODY)
    probe["_native_messages_request"] = True
    return probe


@pytest.mark.parametrize("provider_type", ["custom_anthropic", "minimax_token", "zai_coding"])
def test_native_passthrough_preserves_a_messages_body(provider_type):
    """The native branch preserves a Messages inbound, modulo internal keys.

    Three independent native-branch code paths are exercised
    (``custom_anthropic.py:91-97``, ``minimax_token.py:129-135``,
    ``zai_anthropic.py:85-91``) — each must hold.  The bridge's
    ``_serves_messages_wire`` relies on this preservation: the inbound
    is forwarded upstream verbatim.
    """
    provider_config = {"native_messages": True} if provider_type == "minimax_token" else None
    adapter = get_provider(provider_type, provider_config)
    assert adapter.use_native_messages is True

    probe = _native_probe()
    result = adapter.translate_to_upstream(copy.deepcopy(probe))

    assert classify_wire_shape(result) is WireShape.MESSAGES
    expected = {k: v for k, v in probe.items() if k not in ProviderAdapter._INTERNAL_KEYS}
    assert result == expected


def test_native_passthrough_falsification_a_cc_shaped_native_request_is_not_messages():
    """A Chat-Completions body under the native flag is classified CC.

    Demonstrates the classifier's system/tools axes are what does the
    separating — the R3 check is not vacuous.  In practice the bridge
    only sets ``_native_messages_request`` on the Messages-ingress path,
    so a CC body under the flag is a misconfiguration; the guard
    surfaces the misclassification.
    """
    adapter = get_provider("custom_anthropic")
    probe = {
        "model": "kitty-test-model",
        "max_tokens": 100,
        "messages": [
            {"role": "system", "content": "You are a reviewer."},
            {"role": "user", "content": "hi"},
        ],
        "tools": [
            {"type": "function", "function": {"name": "read_file", "parameters": {}}}
        ],
        "_native_messages_request": True,
    }

    result = adapter.translate_to_upstream(copy.deepcopy(probe))

    assert classify_wire_shape(result) is WireShape.CHAT_COMPLETIONS
    assert classify_wire_shape(result) is not WireShape.MESSAGES
