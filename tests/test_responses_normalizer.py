"""The measured ``/v1/responses`` bodies answer 400 — and the clean ones still pass.

``.system_design/TEST_SUITE.md`` §6.2.1 (``not_a_server_error``) · Jira
**KBR-82** (T-G6) · Jira **KBR-159** (the measured 500s) · Jira **KBR-144**
(the ingress normaliser these cases extend) · Jira **KBR-169** (the
orphan-drop pass that owns the ``function_call_output`` shape).

§6.2 requires every structural guard to be validated against **known
positives**, not trusted on its own reasoning. The four bodies in
:data:`FOUR_MEASURED_BODIES` are exactly that: each one was measured
raising against ``origin/main`` (KBR-159), and each must answer **400 with
``error.reason == "invalid_input"``** once the normaliser has hardened. If
a future change reintroduces the crash — or silently broadens the guard so
a legitimate body is rejected — one of these cases goes red.

The KBR-159 fifth shape (``{"input": [{"type": "function_call_output",
"output": "x"}]}``) is **deliberately absent** from this set. KBR-169's
``_drop_orphan_response_outputs`` pass (``src/kitty/bridge/server.py:388``)
handles missing ``call_id`` by silently dropping the unpaired item and
answering 200; that is the shipped contract and the conformance gate must
not contradict it. A dedicated regression test for KBR-169's behaviour
lives in ``tests/bridge/test_bridge_server_openai_subscription.py``.

The three bodies in :data:`THREE_CONFIRMED_CLEAN_BODIES` are the
complement: KBR-159 measured them passing through without raising, so the
schema and the normaliser must not begin to reject them. Each is pinned
with a **positive-control** assertion — the body reaches the recording
upstream and the bridge answers 200 — not with the permissive "neither
400 nor 500", which a 401 would also satisfy.
"""

from __future__ import annotations

import json

import pytest
from harness.bridge import BridgeFixture, transport
from harness.contract import WireFormat

pytestmark = pytest.mark.l2

#: The four bodies KBR-159 measured raising against ``origin/main`` *that
#: are still KBR-82's responsibility* (KBR-169 owns the fifth). Each
#: reaches ``ResponsesTranslator.translate_request``'s per-field ``.get``
#: and the handler's catch-all rendered it as a 500 before KBR-82 hardened
#: the normaliser.
FOUR_MEASURED_BODIES: list[dict] = [
    {"model": "m", "input": [], "tools": {"a": 1}},
    {"model": "m", "input": [], "tools": "web_search"},
    {"model": "m", "input": [], "tools": [1]},
    {"model": "m", "input": [{"type": "reasoning", "summary": "x"}]},
]

#: The three shapes KBR-159 measured passing through without raising. The
#: normaliser must keep accepting them — an over-eager guard here is the exact
#: failure mode the "why publish a schema" paragraph in KBR-82 names.
THREE_CONFIRMED_CLEAN_BODIES: list[dict] = [
    {"model": "m", "input": [], "instructions": {"tone": "concise"}},
    {"model": "m", "input": [], "reasoning": "x"},
    {
        "model": "m",
        "input": [{"type": "message", "role": "user", "content": 42}],
    },
]


@pytest.fixture()
async def bridge() -> BridgeFixture:
    """A real bridge in bridge mode, pointed at a recording Chat-Completions upstream."""
    async with BridgeFixture(transport("aiohttp", WireFormat.CHAT_COMPLETIONS)) as fixture:
        yield fixture


class TestTheFourMeasuredBodiesAnswer400:
    """Each of KBR-159's four measured-in-scope bodies answers 400 ``invalid_input``."""

    @pytest.mark.parametrize("body", FOUR_MEASURED_BODIES, ids=lambda b: json.dumps(b)[:60])
    async def test_a_measured_body_answers_400(self, bridge: BridgeFixture, body: dict) -> None:
        """A malformed body is refused at the ingress, never rendered as a 500.

        Args:
            bridge: The started bridge fixture.
            body: One of the four measured malformed bodies.
        """
        status, text = await bridge.post("/v1/responses", body)

        assert status == 400, f"expected 400 for {body!r}, got {status}: {text}"
        envelope = json.loads(text)
        assert envelope["error"]["reason"] == "invalid_input", f"envelope: {text}"


class TestConfirmedCleanBodiesStillPass:
    """The three shapes KBR-159 measured clean are still accepted."""

    @pytest.mark.parametrize("body", THREE_CONFIRMED_CLEAN_BODIES, ids=lambda b: json.dumps(b)[:60])
    async def test_a_clean_body_reaches_the_upstream(self, bridge: BridgeFixture, body: dict) -> None:
        """A legitimate body is forwarded to the upstream and answered 200.

        Args:
            bridge: The started bridge fixture.
            body: One of the three confirmed-clean bodies.
        """
        captures_before = len(bridge.captures)
        status, text = await bridge.post("/v1/responses", body)

        assert status == 200, f"expected 200 for {body!r}, got {status}: {text}"
        assert len(bridge.captures) > captures_before, (
            f"body {body!r} was accepted but never reached the upstream — "
            "the positive control failed"
        )
