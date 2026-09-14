"""Bridge-level tests for the ChatGPT-subscription provider's Responses route.

These cover one claim that no unit test can make: that the model
:meth:`~kitty.bridge.server.BridgeServer._normalize_model` computes — register
row **M1**, the profile's model — actually survives the trip to
:class:`~kitty.providers.openai_subscription.OpenAISubscriptionAdapter` and onto
the wire.  It did not (KBR-160): the handler attaches the raw inbound body as
``cc_request["_original_body"]`` and the adapter built the shipped body out of
*that*, so M1 ran and its result was thrown away.

**Why the measurement is taken at the transport.** The wiring under test is
exactly what a unit test replaces, so stubbing the adapter's ``make_request`` —
the pattern :mod:`tests.bridge.test_bridge_server_bedrock` uses — would stub out
the defect.  These tests therefore drive the real adapter and mock one layer
lower, at the ``curl_cffi`` session it posts through, and read the shipped body
off the recorded call.

**Layer.** ``tests/layers.py`` maps only ``tests/integration/`` away from the
``l1`` fallback, so this file collects as ``l1`` alongside every other loopback
test in this directory.  Marking it ``l3`` would be worse than wrong: ``l3`` is
in ``PENDING_ACTIVATION_LAYERS``, so no CI job selects it and these tests would
never run.
"""

from __future__ import annotations

import contextlib
import json
import time
import unittest.mock
from collections.abc import Iterator
from pathlib import Path

import aiohttp
import pytest

from kitty.auth.oauth_session import OAuthSession
from kitty.bridge.server import BridgeServer
from kitty.providers.openai_subscription import OpenAISubscriptionAdapter

#: The model the profile sets, and therefore the only model that may ship.
_PROFILE_MODEL = "profile-model-XYZ"

#: The model the agent asks for.  Deliberately different from the profile's, and
#: deliberately not a real model name, so a passing assertion cannot be an
#: accident of the two happening to agree.
_CLIENT_MODEL = "client-model-ABC"

#: A well-formed inbound Responses body in the array form.
_INBOUND_INPUT = [
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Hello"}],
    }
]

#: A complete Codex SSE response, enough for the non-streaming collector to
#: produce a response rather than raise on an empty body.
_SSE_BODY = (
    b'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
    b'data: {"type":"response.completed","response":{"model":"' + _PROFILE_MODEL.encode() + b'",'
    b'"status":"completed","usage":{"input_tokens":10,"output_tokens":5}}}\n\n'
    b"data: [DONE]\n\n"
)


def _write_oauth_session(tmp_path: Path) -> Path:
    """Write an OAuth session file whose tokens are not due to refresh.

    An hour of headroom keeps the refresh path — and with it the clock — out of
    these tests entirely.

    Args:
        tmp_path: The pytest-provided temporary directory to write into.

    Returns:
        The path to the saved session file, which is what the bridge carries as
        its resolved key for this provider.
    """
    now = time.time()
    session = OAuthSession(
        client_id="app_test",
        access_token="at_fresh",
        refresh_token="rt_fresh",
        id_token="eyJhbGciOiJIUzI1NiJ9.e30.fake_sig",
        api_key=None,
        access_token_expires_at=now + 3600,
        api_key_expires_at=now + 3600,
        _file_path=str(tmp_path / "oauth_session.json"),
    )
    session.save()
    return Path(session._file_path)


@contextlib.contextmanager
def _recording_transport(response: object) -> Iterator[unittest.mock.MagicMock]:
    """Replace the adapter's curl session with one that records what it is sent.

    The adapter also opens a short-lived ``aiohttp`` session for OAuth refresh.
    That one is left real and unpatched: the fixture's token is fresh, so it is
    never used, and patching ``aiohttp.ClientSession`` globally would replace the
    test's own HTTP client along with it.

    Args:
        response: The mock ``curl_cffi`` response the recorded ``post`` returns.

    Yields:
        The mock session, whose ``post`` call carries the shipped body.
    """
    mock_session = unittest.mock.AsyncMock()
    mock_session.post = unittest.mock.AsyncMock(return_value=response)
    mock_session.close = unittest.mock.MagicMock()

    with unittest.mock.patch.object(
        OpenAISubscriptionAdapter,
        "_curl_session",
        new_callable=unittest.mock.PropertyMock,
        return_value=mock_session,
    ):
        yield mock_session


def _non_streaming_response() -> unittest.mock.MagicMock:
    """Build a mock curl response carrying a complete SSE body.

    Returns:
        A mock whose ``status_code``, ``text`` and ``content`` are what the
        non-streaming collector reads.
    """
    resp = unittest.mock.MagicMock()
    resp.status_code = 200
    resp.content = _SSE_BODY
    resp.text = _SSE_BODY.decode()
    return resp


def _streaming_response() -> unittest.mock.MagicMock:
    """Build a mock curl response that yields its SSE body in chunks.

    Returns:
        A mock exposing the async ``aiter_content`` the streaming path consumes.
    """
    resp = unittest.mock.MagicMock()
    resp.status_code = 200
    resp.content = _SSE_BODY
    resp.text = _SSE_BODY.decode()

    async def _aiter_content():
        yield _SSE_BODY

    resp.aiter_content = _aiter_content
    return resp


def _make_server(session_path: Path, provider_config: dict | None = None) -> BridgeServer:
    """Build a bridge-mode server whose profile sets a model of its own.

    Args:
        session_path: The OAuth session file standing in for the resolved key.
        provider_config: Profile provider configuration. The compaction tests
            use ``context_window`` — the product's own knob — to shrink the
            model-derived compaction budget so a small body crosses it.

    Returns:
        A server bound to an ephemeral port, not yet started.
    """
    return BridgeServer(
        adapter=None,
        provider=OpenAISubscriptionAdapter(),
        resolved_key=str(session_path),
        model=_PROFILE_MODEL,
        provider_config=provider_config or {},
        host="127.0.0.1",
        port=0,
    )


def _shipped_body(mock_session: unittest.mock.MagicMock) -> dict:
    """Return the body the adapter posted to the Codex backend.

    Args:
        mock_session: The recording session yielded by :func:`_recording_transport`.

    Returns:
        The ``json=`` payload of the single recorded ``post`` call.

    Raises:
        AssertionError: If the adapter never reached the transport, which would
            make any assertion about the shipped body vacuous.
    """
    assert mock_session.post.await_count == 1, (
        f"expected exactly one upstream POST, got {mock_session.post.await_count}"
    )
    return mock_session.post.await_args.kwargs["json"]


class TestTheProfileModelReachesTheProvider:
    """Register row M1 must hold on both of the Responses route's two paths."""

    @pytest.mark.asyncio()
    async def test_non_streaming_responses_ships_the_profile_model(self, tmp_path: Path) -> None:
        """Ship the profile's model, not the agent's, on the non-streaming path

        This is the path where ``_handle_responses`` attaches ``_original_body``
        before calling the provider.
        """
        server = _make_server(_write_oauth_session(tmp_path))
        with _recording_transport(_non_streaming_response()) as mock_session:
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as client,
                    client.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": _CLIENT_MODEL,
                            "input": _INBOUND_INPUT,
                            "stream": False,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
            finally:
                await server.stop_async()

        assert _shipped_body(mock_session)["model"] == _PROFILE_MODEL

    @pytest.mark.asyncio()
    async def test_streaming_responses_ships_the_profile_model(self, tmp_path: Path) -> None:
        """Ship the profile's model, not the agent's, on the streaming path

        Codex always streams — register row P17 forces ``stream: True`` — so
        this is the path a real subscription user actually takes, and it
        attaches ``_original_body`` at its own separate site.
        """
        server = _make_server(_write_oauth_session(tmp_path))
        with _recording_transport(_streaming_response()) as mock_session:
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as client,
                    client.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": _CLIENT_MODEL,
                            "input": _INBOUND_INPUT,
                            "stream": True,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
                    await resp.read()
            finally:
                await server.stop_async()

        assert _shipped_body(mock_session)["model"] == _PROFILE_MODEL

    @pytest.mark.asyncio()
    async def test_the_agents_model_is_the_only_thing_replaced(self, tmp_path: Path) -> None:
        """Carry the agent's own request through untouched apart from the model

        The fix takes one field from the normalized request.  Everything else on
        the shipped body must still come from the body the agent sent, or the
        cure is worse than KBR-160.
        """
        server = _make_server(_write_oauth_session(tmp_path))
        with _recording_transport(_streaming_response()) as mock_session:
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as client,
                    client.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": _CLIENT_MODEL,
                            "instructions": "You are helpful.",
                            "input": _INBOUND_INPUT,
                            "stream": True,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
                    await resp.read()
            finally:
                await server.stop_async()

        body = _shipped_body(mock_session)
        assert body["instructions"] == "You are helpful."
        assert json.dumps(body["input"]).count("Hello") == 1


#: An oversized tool output. Must exceed the bridge's 50,000-char truncation
#: limit (``_TOOL_RESULT_TRUNCATION_LIMIT``) — written as a literal, not derived
#: from the constant, so a fixture cannot silently shrink along with it.
_OVERSIZED_OUTPUT = "x" * 60_000

#: The notice the bridge substitutes for an oversized tool result, in the exact
#: spelling every route ships.
_TRUNCATION_NOTICE = "[Tool output truncated — original size: 60,000 chars]"


def _function_call(call_id: str) -> dict:
    """Build a Responses ``function_call`` input item.

    Args:
        call_id: The client-chosen call identifier outputs reference.

    Returns:
        The input item dict.
    """
    return {"type": "function_call", "call_id": call_id, "name": "report", "arguments": "{}"}


def _function_call_output(call_id: str, output: str) -> dict:
    """Build a Responses ``function_call_output`` input item.

    Args:
        call_id: The call identifier this output answers.
        output: The tool's output text.

    Returns:
        The input item dict.
    """
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _user_message(text: str) -> dict:
    """Build a Responses user-message input item.

    Args:
        text: The message text.

    Returns:
        The input item dict.
    """
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _reasoning_item(text: str) -> dict:
    """Build a Responses reasoning input item as a Codex client sends it.

    Args:
        text: The reasoning summary text.

    Returns:
        The input item dict.
    """
    return {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}


def _shipped_input(mock_session: unittest.mock.MagicMock) -> list:
    """Return the ``input`` array the adapter posted to the Codex backend.

    Args:
        mock_session: The recording session yielded by :func:`_recording_transport`.

    Returns:
        The shipped body's ``input`` list.

    Raises:
        AssertionError: If the shipped body carries no ``input`` at all.
    """
    body = _shipped_body(mock_session)
    assert "input" in body, f"shipped body carries no input: {sorted(body)}"
    return body["input"]


class TestTheOversizedRequestProtectionsReachTheProvider:
    """Register rows M3, M5 and M7 must hold on the wire, not only on the copy.

    KBR-169: the handler computes all three protections on the translated CC
    conversation and then attaches the raw inbound body as ``_original_body``,
    so the subscription adapter shipped the *unprotected* conversation. Each
    test reads the body at the transport — the wiring a stub would remove is
    the thing under test.
    """

    @pytest.mark.asyncio()
    async def test_oversized_tool_output_is_truncated_on_the_non_streaming_path(
        self, tmp_path: Path
    ) -> None:
        """Ship the truncation notice, not the oversized string, when not streaming."""
        server = _make_server(_write_oauth_session(tmp_path))
        with _recording_transport(_non_streaming_response()) as mock_session:
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as client,
                    client.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": _CLIENT_MODEL,
                            "input": [
                                _user_message("Run the report"),
                                _function_call("call_1"),
                                _function_call_output("call_1", _OVERSIZED_OUTPUT),
                                _user_message("Thanks"),
                            ],
                            "stream": False,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
            finally:
                await server.stop_async()

        shipped = _shipped_input(mock_session)
        outputs = [i for i in shipped if i.get("type") == "function_call_output"]
        assert len(outputs) == 1
        assert outputs[0]["output"] == _TRUNCATION_NOTICE
        assert _OVERSIZED_OUTPUT not in json.dumps(shipped)

    @pytest.mark.asyncio()
    async def test_oversized_tool_output_is_truncated_on_the_streaming_path(
        self, tmp_path: Path
    ) -> None:
        """Ship the truncation notice on the streaming path a Codex client takes."""
        server = _make_server(_write_oauth_session(tmp_path))
        with _recording_transport(_streaming_response()) as mock_session:
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as client,
                    client.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": _CLIENT_MODEL,
                            "input": [
                                _user_message("Run the report"),
                                _function_call("call_1"),
                                _function_call_output("call_1", _OVERSIZED_OUTPUT),
                                _user_message("Thanks"),
                            ],
                            "stream": True,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
                    await resp.read()
            finally:
                await server.stop_async()

        shipped = _shipped_input(mock_session)
        outputs = [i for i in shipped if i.get("type") == "function_call_output"]
        assert len(outputs) == 1
        assert outputs[0]["output"] == _TRUNCATION_NOTICE

    @pytest.mark.asyncio()
    async def test_orphan_tool_output_does_not_ship_but_a_paired_one_does(
        self, tmp_path: Path
    ) -> None:
        """Drop the orphan whose call was never declared; keep the paired output."""
        server = _make_server(_write_oauth_session(tmp_path))
        with _recording_transport(_non_streaming_response()) as mock_session:
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as client,
                    client.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": _CLIENT_MODEL,
                            "input": [
                                _user_message("hi"),
                                _function_call("call_1"),
                                _function_call_output("call_1", "paired result"),
                                _function_call_output("call_orphan", "orphan result"),
                                _user_message("bye"),
                            ],
                            "stream": False,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
            finally:
                await server.stop_async()

        shipped = _shipped_input(mock_session)
        shipped_ids = [i.get("call_id") for i in shipped if i.get("type") == "function_call_output"]
        assert shipped_ids == ["call_1"]
        assert "orphan result" not in json.dumps(shipped)

    @pytest.mark.asyncio()
    async def test_tool_output_without_call_id_is_dropped_not_a_500(
        self, tmp_path: Path
    ) -> None:
        """Treat a missing ``call_id`` as undeclared and drop the output.

        Before the orphan pass this item reached ``_translate_input_item``, which
        subscripts ``call_id``, and the handler rendered the ``KeyError`` as a 500.
        """
        server = _make_server(_write_oauth_session(tmp_path))
        with _recording_transport(_non_streaming_response()) as mock_session:
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as client,
                    client.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": _CLIENT_MODEL,
                            "input": [
                                _user_message("hi"),
                                {"type": "function_call_output", "output": "no id"},
                                _user_message("bye"),
                            ],
                            "stream": False,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
            finally:
                await server.stop_async()

        shipped = _shipped_input(mock_session)
        assert [i.get("type") for i in shipped if i.get("type") == "function_call_output"] == []
        assert len([i for i in shipped if i.get("role") == "user"]) == 2

    @pytest.mark.asyncio()
    async def test_below_threshold_input_ships_unchanged(self, tmp_path: Path) -> None:
        """Leave a small, well-formed conversation byte-identical on the wire.

        The three protections are conditional mutations; below their triggers
        the shipped ``input`` must equal what the agent sent. None of these
        items carry assistant content parts, so the registered content-type
        rewrite (P16) has nothing to flip either.
        """
        original = [
            _user_message("hello"),
            _reasoning_item("thinking about it"),
            _function_call("call_1"),
            _function_call_output("call_1", "small result"),
            _user_message("done"),
        ]
        server = _make_server(_write_oauth_session(tmp_path))
        with _recording_transport(_non_streaming_response()) as mock_session:
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as client,
                    client.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": _CLIENT_MODEL,
                            "input": original,
                            "stream": False,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
            finally:
                await server.stop_async()

        assert _shipped_input(mock_session) == original

    @pytest.mark.asyncio()
    async def test_over_budget_history_ships_compacted(self, tmp_path: Path) -> None:
        """Ship the compacted conversation when the history exceeds the budget.

        ``context_window`` is the product's own knob for the model-derived
        budget; 8,000 tokens puts the messages budget near 22K chars, so this
        ~44K-char body must compact. A tool hop in the middle of the history
        must go with its pruned turn — its riding reasoning item included —
        while a hop in the preserved tail ships whole, reasoning item too.
        """
        filler: list[dict] = []
        for i in range(50):
            filler.append(_user_message(f"user filler {i} " + "p" * 600))
            filler.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"assistant filler {i}"}],
                }
            )
        mid_hop = [
            _reasoning_item("mid reasoning"),
            _function_call("call_mid"),
            _function_call_output("call_mid", "mid result"),
        ]
        tail_hop = [
            _reasoning_item("tail reasoning"),
            _function_call("call_tail"),
            _function_call_output("call_tail", "tail result"),
        ]
        # The mid hop sits one third in, deep inside the window compaction
        # prunes; the tail hop rides the preserved end.
        sent = filler[:33] + mid_hop + filler[33:] + tail_hop

        server = _make_server(_write_oauth_session(tmp_path), {"context_window": 8000})
        with _recording_transport(_non_streaming_response()) as mock_session:
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as client,
                    client.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": _CLIENT_MODEL,
                            "instructions": "You are helpful.",
                            "input": sent,
                            "stream": False,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
            finally:
                await server.stop_async()

        shipped = _shipped_input(mock_session)
        dumped = json.dumps(shipped)
        # The conversation got smaller, and the shrink is compaction's, not a
        # total loss: head and tail survive.
        assert len(shipped) < len(sent)
        assert dumped.count("user filler 0 ") == 1
        assert "assistant filler 49" in dumped
        # The pruned mid hop is gone whole — call, output, and its riding
        # reasoning item.
        assert "mid reasoning" not in dumped
        assert "call_mid" not in dumped
        assert "mid result" not in dumped
        # The preserved tail keeps its hop complete, reasoning item included.
        assert "tail reasoning" in dumped
        assert "tail result" in dumped
        # Whatever survived pairs up: no shipped output without its call.
        call_ids = [i["call_id"] for i in shipped if i.get("type") == "function_call"]
        for item in shipped:
            if item.get("type") == "function_call_output":
                assert item["call_id"] in call_ids
