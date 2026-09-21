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


class TestGeminiToleratesScalarGenerationConfig:
    """A scalar ``generationConfig`` reaches the translator without a 500.

    Caught on CI (Python 3.13 and the Windows leg): the Gemini translator
    read ``gen_config = gemini_request.get("generationConfig", {})`` and
    then ``"temperature" in gen_config`` — a present-but-scalar value
    (``None``, an ``int``) crashed with ``TypeError: argument of type 'X'
    is not iterable``. The schema documents ``type: object`` so a real
    Gemini client never sends this, but the conformance fuzzer does, and
    §6.2.1 forbids a 500 on any malformed body.
    """

    @pytest.mark.parametrize("scalar", [None, 8364, "x", True])
    async def test_a_scalar_generation_config_does_not_500(
        self, cc_bridge: BridgeFixture, scalar: object
    ) -> None:
        status, text = await cc_bridge.post(
            "/v1beta/models/harness-model:generateContent",
            {"contents": [], "generationConfig": scalar},
        )
        assert status < 500, f"expected a non-5xx for generationConfig={scalar!r}, got {status}: {text}"


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


# ── KBR-288: the full Gemini required-chain shape set ────────────────────────


class TestGeminiRequiredChainShapes:
    """Every required-shape defect on the Gemini request path answers 400.

    KBR-288's split disposition (SYSTEM_DESIGN.md §12.3): required-chain
    members — ``contents`` → ``parts`` → per-part fields, ``tools`` →
    ``functionDeclarations`` → per-declaration fields — answer 400 in the
    Gemini dialect when their documented type is violated. Some rows
    crashed at base (the ``role`` unhashable, non-string ``text`` and
    present-null ``functionDeclarations`` classes); the rest were silently
    skipped and are tightened per the §12.3 policy (a silent drop inside a
    required chain is an I1 fidelity hit).
    """

    @staticmethod
    async def _post_and_expect_400(cc_bridge: BridgeFixture, body: dict) -> None:
        status, text = await cc_bridge.post(
            "/v1beta/models/harness-model:generateContent",
            body,
        )
        assert status == 400, f"expected 400, got {status}: {text}"
        envelope = json.loads(text)
        assert isinstance(envelope["error"].get("code"), int)
        assert envelope["error"].get("status") == "INVALID_ARGUMENT"
        assert "Traceback" not in text

    async def test_role_non_string_answers_400(self, cc_bridge: BridgeFixture) -> None:
        # Crashed at base: TypeError: unhashable type: 'list' (_ROLE_MAP.get).
        await self._post_and_expect_400(
            cc_bridge,
            {"contents": [{"role": ["user"], "parts": [{"text": "hi"}]}]},
        )

    async def test_contents_member_not_dict_answers_400(
        self, cc_bridge: BridgeFixture
    ) -> None:
        # KBR-82 closed this (400 already); pinned here so the whole To Be
        # table has a deterministic control on this route.
        await self._post_and_expect_400(cc_bridge, {"contents": [5]})

    async def test_parts_non_list_answers_400(self, cc_bridge: BridgeFixture) -> None:
        # Silently skipped at base (empty content, 200); tightened per §12.3.
        await self._post_and_expect_400(cc_bridge, {"contents": [{"parts": True}]})

    async def test_part_not_dict_answers_400(self, cc_bridge: BridgeFixture) -> None:
        # Silently filtered at base; tightened per §12.3.
        await self._post_and_expect_400(cc_bridge, {"contents": [{"parts": [5]}]})

    @pytest.mark.parametrize("bad_text", [{"a": 1}, [1], True, 5])
    async def test_part_text_non_string_answers_400(
        self, cc_bridge: BridgeFixture, bad_text: object
    ) -> None:
        # Crashed at base on every branch: TypeError: sequence item 0
        # (join over non-strings). Parametrised across the fuzzer's
        # scalar distribution.
        await self._post_and_expect_400(
            cc_bridge,
            {"contents": [{"parts": [{"text": bad_text}]}]},
        )

    async def test_assistant_text_non_string_answers_400(
        self, cc_bridge: BridgeFixture
    ) -> None:
        # The assistant thought branch joins text the same way; pin it
        # separately so the branch coverage is explicit.
        await self._post_and_expect_400(
            cc_bridge,
            {"contents": [{"role": "model", "parts": [{"text": [1], "thought": True}]}]},
        )

    async def test_assistant_plain_text_non_string_answers_400(
        self, cc_bridge: BridgeFixture
    ) -> None:
        # The plain assistant text branch (no ``thought`` flag) joins
        # unguarded text the same way; explicit pin keeps the branch
        # coverage honest.
        await self._post_and_expect_400(
            cc_bridge,
            {"contents": [{"role": "model", "parts": [{"text": [1]}]}]},
        )

    @pytest.mark.parametrize("bad_fc", ["f", 5, True, {"args": {}}, {"name": 5}])
    async def test_function_call_shape_defect_answers_400(
        self, cc_bridge: BridgeFixture, bad_fc: object
    ) -> None:
        # Non-dict and name-less dicts were silently skipped at base;
        # tightened per §12.3 (required-when-present member).
        await self._post_and_expect_400(
            cc_bridge,
            {"contents": [{"role": "model", "parts": [{"functionCall": bad_fc}]}]},
        )

    @pytest.mark.parametrize("bad_fr", ["r", 5, True, {"response": {}}, {"name": 5}])
    async def test_function_response_shape_defect_answers_400(
        self, cc_bridge: BridgeFixture, bad_fr: object
    ) -> None:
        # Mirror of the functionCall row on the tool-result side.
        await self._post_and_expect_400(
            cc_bridge,
            {"contents": [{"role": "tool", "parts": [{"functionResponse": bad_fr}]}]},
        )

    async def test_tool_not_dict_answers_400(self, cc_bridge: BridgeFixture) -> None:
        # Silently continued at base; tightened per §12.3.
        await self._post_and_expect_400(cc_bridge, {"contents": [], "tools": [5]})

    @pytest.mark.parametrize("bad_fds", [None, True, 5, "x", {"a": 1}])
    async def test_function_declarations_non_list_answers_400(
        self, cc_bridge: BridgeFixture, bad_fds: object
    ) -> None:
        # ``None`` and ``True`` crashed at base (TypeError: 'NoneType'/'bool'
        # object is not iterable — the .get default does not fire when the
        # key is present); the rest were silently skipped. Parametrised over
        # the same distribution as the generationConfig pin (KBR-82).
        await self._post_and_expect_400(
            cc_bridge,
            {"contents": [], "tools": [{"functionDeclarations": bad_fds}]},
        )

    @pytest.mark.parametrize("bad_fd", [5, "x", {"description": "d"}, {"name": 5}])
    async def test_function_declaration_member_defect_answers_400(
        self, cc_bridge: BridgeFixture, bad_fd: object
    ) -> None:
        # Silently skipped at base; tightened per §12.3.
        await self._post_and_expect_400(
            cc_bridge,
            {"contents": [], "tools": [{"functionDeclarations": [bad_fd]}]},
        )


class TestGeminiStreamSharedHandler:
    """The shared ``_handle_gemini`` serves both routes — pin the stream one.

    One tightening shape and one crash-class shape suffice: the stream route
    shares the handler and normalizer with ``:generateContent``, so the
    per-shape matrix lives on the non-streaming route only.
    """

    async def test_stream_tightening_shape_answers_400(
        self, cc_bridge: BridgeFixture
    ) -> None:
        status, text = await cc_bridge.post(
            "/v1beta/models/harness-model:streamGenerateContent",
            {"contents": [{"parts": True}]},
        )
        assert status == 400, f"expected 400, got {status}: {text}"
        envelope = json.loads(text)
        assert isinstance(envelope["error"].get("code"), int)
        assert envelope["error"].get("status") == "INVALID_ARGUMENT"
        assert "Traceback" not in text

    async def test_stream_crash_class_answers_400(self, cc_bridge: BridgeFixture) -> None:
        # Crashed at base: TypeError: unhashable type (role lookup).
        status, text = await cc_bridge.post(
            "/v1beta/models/harness-model:streamGenerateContent",
            {"contents": [{"role": {"a": 1}, "parts": [{"text": "hi"}]}]},
        )
        assert status == 400, f"expected 400, got {status}: {text}"
        envelope = json.loads(text)
        assert isinstance(envelope["error"].get("code"), int)
        assert envelope["error"].get("status") == "INVALID_ARGUMENT"
        assert "Traceback" not in text


class TestGeminiToleratedShapes:
    """Optional envelopes stay tolerant — malformed means absent, not 400.

    The §12.3 split: ``tools`` and ``systemInstruction`` are optional
    envelopes, so a wrong-typed envelope (or wrong-typed content inside a
    tolerated envelope) is treated as absent. The ``systemInstruction`` text
    row crashed at base and is closed with a translator-side guard — a
    tolerated envelope gets no boundary check, so the translator carries the
    leaf guard.
    """

    @pytest.mark.parametrize("non_list", [None, True, 5, "x", {"a": 1}])
    async def test_tools_non_list_tolerated(self, cc_bridge: BridgeFixture, non_list: object) -> None:
        status, text = await cc_bridge.post(
            "/v1beta/models/harness-model:generateContent",
            {"contents": [], "tools": non_list},
        )
        assert status < 500, f"expected non-5xx for tools={non_list!r}, got {status}: {text}"

    @pytest.mark.parametrize("bad_text", [{"a": 1}, [1]])
    async def test_system_instruction_text_non_string_tolerated(
        self, cc_bridge: BridgeFixture, bad_text: object
    ) -> None:
        # Crashed at base inside _extract_text's join; the tolerated
        # envelope now carries a translator-side isinstance guard.
        status, text = await cc_bridge.post(
            "/v1beta/models/harness-model:generateContent",
            {"contents": [], "systemInstruction": {"parts": [{"text": bad_text}]}},
        )
        assert status < 500, f"expected non-5xx, got {status}: {text}"

    async def test_function_declarations_empty_list_tolerated(
        self, cc_bridge: BridgeFixture
    ) -> None:
        # Negative control: `functionDeclarations: []` is a valid list
        # shape and must not 400 - catches a future over-tightening that
        # adds a non-empty-list guard.
        status, text = await cc_bridge.post(
            "/v1beta/models/harness-model:generateContent",
            {"contents": [], "tools": [{"functionDeclarations": []}]},
        )
        assert status < 500, f"expected non-5xx for empty list, got {status}: {text}"

