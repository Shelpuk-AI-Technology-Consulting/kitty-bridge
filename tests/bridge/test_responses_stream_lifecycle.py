"""The translated /v1/responses stream opens with response.created (KBR-242).

Three contracts, one regression file:

- R1/R2/R3 — a non-empty stream opens the lifecycle exactly once, an
  empty failover publishes nothing across attempts, and ``sequence_number``
  is monotonic with no gaps across the failover (attempt 1 emitted zero
  bytes, so attempt 2 owns the entire numbering — ``0 .. len(events) - 1``).
- R4 — the error paths (cloudflare abort, upstream 5xx, timeout) write
  ``responses_format_error`` and never reach the lifecycle helper. These
  are negative controls: they pass on unfixed code (the helper does not
  exist yet) and continue to pass after the fix. They guard against a
  future over-broad "write the lifecycle everywhere" simplification.

The harness mirrors ``tests/bridge/test_responses_output_index_stream.py``
(self-contained, in-process ``BridgeServer`` + scripted ``aioresponses``
upstream + SSE walk), with one import for the timeout negative control:
``_Upstream`` from ``test_post_emission_no_failover`` (a local aiohttp
server that can stall mid-stream — aiohttp-server timeouts are not
scriptable with ``aioresponses`` alone). The ``fast_stall`` fixture
shrinks ``_STREAM_READ_TIMEOUT`` and ``_BACKOFF_BASE`` so the suite
stays fast, per the convention used by ``test_empty_response_retry.py``
and ``test_post_emission_no_failover.py``.

Path default: ``tests/bridge/`` → ``l1`` (see ``tests/layers.py``); this
file runs in the fast gate without registration.
"""

from __future__ import annotations

import json
import uuid

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge import server as server_module
from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.custom_openai import CustomOpenAIAdapter
from kitty.types import BridgeProtocol

from .test_post_emission_no_failover import _Upstream

CC_URL = "https://api.cc.test/v1/chat/completions"


class _StubLauncher(LauncherAdapter):
    """Minimal launcher that selects the Responses API bridge protocol."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.RESPONSES_API

    def build_spawn_config(self, profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


def _cc_chunk(delta: dict, finish: str | None = None) -> dict:
    """Build one Chat Completions streaming chunk.

    Args:
        delta: The ``choices[0].delta`` object of the chunk.
        finish: The chunk's ``finish_reason``, if it is the final chunk.

    Returns:
        A Chat Completions chunk dict.
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "MiniMax-M3",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _empty_stream() -> bytes:
    """A Chat Completions stream whose only payload is a finish-only chunk.

    The translator's finish branch sets ``response_was_empty`` and buffers a
    fallback ``response.completed`` event, then the empty-response ladder
    retries onto the next attempt (KBR-242).
    """
    body = "".join(f"data: {json.dumps(_cc_chunk({}, finish='stop'))}\n\n" for _ in (None,))
    return (body + "data: [DONE]\n\n").encode()


def _prose_stream() -> bytes:
    """A minimal Chat Completions stream: one content delta + finish."""
    chunks = [
        _cc_chunk({"content": "Recovered"}),
        _cc_chunk({}, finish="stop"),
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


def _parse_sse(body: bytes) -> list[tuple[str, dict]]:
    """Parse a Responses SSE byte stream into ``(event_name, data)`` pairs.

    Args:
        body: The raw response body the bridge wrote to the client.

    Returns:
        One pair per event, in wire order.
    """
    parsed: list[tuple[str, dict]] = []
    for block in body.decode().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
                parsed.append((data["type"], data))
    return parsed


def _make_server(base_url: str | None = None) -> BridgeServer:
    """Build a single-backend Responses server on the translated (CC) wire.

    Args:
        base_url: The CC upstream base URL (for tests that drive a local
            aiohttp server). Defaults to the KBR-240 walk's mock URL.

    Returns:
        The unstarted server.
    """
    provider_config = {"base_url": base_url} if base_url is not None else {"base_url": "https://api.cc.test/v1"}
    profile = Profile(
        name="pool-member-1",
        provider="custom_openai",
        model="MiniMax-M3",
        auth_ref=str(uuid.uuid4()),
        provider_config=provider_config,
    )
    return BridgeServer(
        adapter=_StubLauncher(),
        provider=CustomOpenAIAdapter(),
        resolved_key="key-1",
        model="MiniMax-M3",
        provider_config=profile.provider_config,
        host="127.0.0.1",
        port=0,
    )


def _client_request() -> dict:
    """A Codex-shaped Responses request."""
    return {
        "model": "MiniMax-M3",
        "stream": True,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "review"}],
            }
        ],
    }


async def _post(port: int) -> bytes:
    """POST one Responses request to a running server and return the body."""
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            f"http://127.0.0.1:{port}/v1/responses",
            json=_client_request(),
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp,
    ):
        assert resp.status == 200
        return await resp.read()


@pytest.fixture
def fast_stall(monkeypatch):
    """Shrink the sock-read timeout and every retry delay so the timeout test stays fast."""
    monkeypatch.setattr(server_module, "_STREAM_READ_TIMEOUT", 1)
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.001)


# ── R1/R2/R3: empty-response failover opens the lifecycle once ──────────────


@pytest.mark.asyncio
async def test_empty_failover_opens_the_lifecycle_once(monkeypatch):
    """Attempt 1 is empty and publishes nothing; attempt 2 is non-empty and
    opens the lifecycle exactly once, with ``sequence_number`` 0..len-1.

    The shape-2 regression (a duplicate ``response.created`` after the
    failover) and the write-time-materialization regression (decreasing
    ``sequence_number``) are both killed by the exact-equality assertion
    on the sequence numbers.
    """
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.001)
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.0, 0.0])

    server = _make_server()
    with aioresponses(passthrough=["http://127.0.0.1"]) as m:
        # No ``repeat=True``: the first registration matches the first POST
        # (attempt 1), the second matches the second POST (attempt 2).
        m.post(CC_URL, body=_empty_stream())
        m.post(CC_URL, body=_prose_stream())

        await server.start_async()
        try:
            body = await _post(server.port)
        finally:
            await server.stop_async()

    events = _parse_sse(body)
    names = [name for name, _ in events]

    # R1 + R2: exactly one lifecycle opening, both events precede the first
    # output-item event.
    assert names.count("response.created") == 1
    assert names.count("response.in_progress") == 1
    assert names[:2] == ["response.created", "response.in_progress"]
    assert names[2] == "response.output_item.added"

    # R3: monotonic, gapless, duplicate-free sequence numbers across the
    # whole client-visible stream. Attempt 1 emitted zero bytes, so the
    # numbering starts at 0 (the lifecycle on attempt 2).
    seqs = [d["sequence_number"] for _, d in events]
    assert seqs == list(range(len(events)))


# ── R4 negative controls: error paths do not open the lifecycle ─────────────


@pytest.mark.asyncio
async def test_cloudflare_abort_does_not_open_the_lifecycle():
    """A 403 with a Cloudflare signature writes one error event and no opening."""
    server = _make_server()
    with aioresponses(passthrough=["http://127.0.0.1"]) as m:
        m.post(
            CC_URL,
            status=403,
            body="cf-mitigated challenge page",
        )

        await server.start_async()
        try:
            body = await _post(server.port)
        finally:
            await server.stop_async()

    events = _parse_sse(body)
    names = [name for name, _ in events]

    assert "response.created" not in names
    assert "response.in_progress" not in names
    assert "error" in names


@pytest.mark.asyncio
async def test_upstream_5xx_does_not_open_the_lifecycle(monkeypatch):
    """An upstream HTTP 500 writes one error event and no opening."""
    # The 500 path retries up to ``max_attempts`` before writing the error
    # event; patch the retry delays so the test stays fast.
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.001)
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.0, 0.0])

    server = _make_server()
    with aioresponses(passthrough=["http://127.0.0.1"]) as m:
        m.post(
            CC_URL,
            status=500,
            body="internal server error",
            repeat=True,
        )

        await server.start_async()
        try:
            body = await _post(server.port)
        finally:
            await server.stop_async()

    events = _parse_sse(body)
    names = [name for name, _ in events]

    assert "response.created" not in names
    assert "response.in_progress" not in names
    assert "error" in names


@pytest.mark.asyncio
async def test_timeout_does_not_open_the_lifecycle(fast_stall):
    """An upstream that never sends the first byte times out and writes one
    error event; the lifecycle never opens.

    Uses ``_Upstream`` (local aiohttp server) because ``aioresponses`` cannot
    stall a connection without sending the first byte, and the standard-
    branch timeout fires on ``sock_read`` before any chunk arrives.
    """
    upstream = _Upstream("/v1/chat/completions", lambda req, _: upstream.send_then_stall(req, b""))
    async with upstream as base:
        server = _make_server(base_url=base)
        port = await server.start_async()
        try:
            body = await _post(port)
        finally:
            await server.stop_async()

    events = _parse_sse(body)
    names = [name for name, _ in events]

    assert "response.created" not in names
    assert "response.in_progress" not in names
    assert "error" in names
    error_events = [d for n, d in events if n == "error"]
    assert len(error_events) == 1
    assert error_events[0]["code"] == "timeout"
    assert error_events[0]["message"].startswith("Upstream provider timed out")
