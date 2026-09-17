"""Per-route ingress 400s: the four other POST routes refuse `messages: "x"` / `contents: "x"`.

``.system_design/TEST_SUITE.md`` §6.2.1 (``not_a_server_error``) · Jira
**KBR-82** (T-G6) · cross-references KBR-159 (the five Responses 500s).

§6.2.1's contract is that the bridge must never 500 on a malformed body.
KBR-159 measured five such shapes on ``/v1/responses`` and this same task
fixed them in ``normalize_responses_request``. The conformance run drives
all five POST routes, so the same property must hold on each — the
pre-flight measurement that motivated F11 (in this task's
``REQUIREMENTS.md``) found four additional unguarded sites:

* ``/v1/messages`` iterating ``body["messages"]`` in
  ``_convert_native_to_cc_format`` (server.py:788).
* ``/v1/chat/completions`` iterating ``body["messages"]`` in
  ``_has_tool_use_blocks`` (server.py:175) and the CC translator.
* ``/v1beta/...:generateContent`` and ``:streamGenerateContent`` both
  iterating ``body["contents"]`` in the Gemini translator.

This module pins the post-fix behaviour for each of the four routes with a
single positive control per route: ``messages: "x"`` (or ``contents: "x"``
for Gemini) answers 400, not 500. Each route's error envelope matches its
dialect (Anthropic / OpenAI / Gemini), and the response carries no leaked
exception text.
"""

from __future__ import annotations

import json

import pytest
from harness.bridge import BridgeFixture, transport
from harness.contract import WireFormat

pytestmark = pytest.mark.l2


@pytest.fixture()
async def cc_bridge() -> BridgeFixture:
    """A bridge pointed at a recording Chat-Completions upstream."""
    async with BridgeFixture(transport("aiohttp", WireFormat.CHAT_COMPLETIONS)) as fixture:
        yield fixture


@pytest.fixture()
async def anthropic_bridge() -> BridgeFixture:
    """A bridge pointed at a recording Anthropic-Messages upstream."""
    async with BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)) as fixture:
        yield fixture


class TestMessagesRefusesMessagesNotList:
    """``/v1/messages`` with ``messages: "x"`` answers 400, not 500."""

    async def test_messages_string_answers_400(self, anthropic_bridge: BridgeFixture) -> None:
        status, text = await anthropic_bridge.post(
            "/v1/messages",
            {"model": "m", "messages": "x", "max_tokens": 1},
        )
        assert status == 400, f"expected 400, got {status}: {text}"
        envelope = json.loads(text)
        # Anthropic envelope: top-level `type: "error"` and `error.type` describing the kind.
        assert envelope.get("type") == "error"
        assert "Traceback" not in text


class TestChatCompletionsRefusesMessagesNotList:
    """``/v1/chat/completions`` with ``messages: "x"`` answers 400, not 500."""

    async def test_messages_string_answers_400(self, cc_bridge: BridgeFixture) -> None:
        status, text = await cc_bridge.post(
            "/v1/chat/completions",
            {"model": "m", "messages": "x"},
        )
        assert status == 400, f"expected 400, got {status}: {text}"
        envelope = json.loads(text)
        # OpenAI Chat-Completions error envelope: `error.code: "invalid_request"`.
        assert envelope["error"].get("code") == "invalid_request"
        assert "Traceback" not in text


class TestGeminiGenerateRefusesContentsNotList:
    """``/v1beta/...:generateContent`` with ``contents: "x"`` answers 400."""

    async def test_contents_string_answers_400(self, cc_bridge: BridgeFixture) -> None:
        status, text = await cc_bridge.post(
            "/v1beta/models/harness-model:generateContent",
            {"contents": "x"},
        )
        assert status == 400, f"expected 400, got {status}: {text}"
        envelope = json.loads(text)
        # Gemini error envelope: `error.code` is an integer and `error.status` describes the kind.
        assert isinstance(envelope["error"].get("code"), int)
        assert "Traceback" not in text


class TestGeminiStreamRefusesContentsNotList:
    """``/v1beta/...:streamGenerateContent`` with ``contents: "x"`` answers 400."""

    async def test_contents_string_answers_400(self, cc_bridge: BridgeFixture) -> None:
        status, text = await cc_bridge.post(
            "/v1beta/models/harness-model:streamGenerateContent",
            {"contents": "x"},
        )
        assert status == 400, f"expected 400, got {status}: {text}"
        envelope = json.loads(text)
        assert isinstance(envelope["error"].get("code"), int)
        assert "Traceback" not in text
