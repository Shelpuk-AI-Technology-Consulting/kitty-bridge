"""Bridge-level tests: translated 429 arms honor the upstream Retry-After hint.

``SYSTEM_DESIGN.md`` §6.1 promises "``rate_limit`` from the provider's own
retry hint". The subscription relay already honors it
(``_mark_subscription_unhealthy``); these tests pin the same parity on the
translated streaming and non-streaming Messages arms, where a 429 currently
parks the member for the fixed default no matter what the provider asked for
(live observation 2026-10-29: MiniMax members flapping at 429 with the hint
ignored).

Every scenario parks BOTH members (or uses a single member), so the
assertions never depend on which member weighted selection picks first.

**Layer.** Collects as ``l1`` alongside every other loopback test in this
directory (``tests/layers.py`` maps only ``tests/integration/`` away).
"""

from __future__ import annotations

import json
import uuid

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge import server as server_module
from kitty.bridge.server import BridgeServer
from kitty.profiles.schema import Profile
from kitty.providers.custom_anthropic import CustomAnthropicAdapter


def _plain_member(name: str, host: str) -> tuple:
    """Build an ordinary Anthropic-wire member pointed at its own host.

    Args:
        name: Profile name for health attribution.
        host: Fake upstream host serving this member exclusively.

    Returns:
        A ``(provider, key, profile)`` balancing tuple.
    """
    return (
        CustomAnthropicAdapter(),
        "key-value",
        Profile(
            name=name,
            provider="custom_anthropic",
            model="claude-sonnet-4-6",
            auth_ref=str(uuid.uuid4()),
            provider_config={"base_url": f"https://{host}"},
        ),
    )


def _server(members: list[tuple]) -> BridgeServer:
    """Build a balancing server fronting the given members.

    Args:
        members: ``(provider, key, profile)`` balancing tuples.

    Returns:
        A server bound to an ephemeral port, not yet started.
    """
    return BridgeServer(
        adapter=None,
        provider=members[0][0],
        resolved_key=members[0][1],
        model="claude-sonnet-4-6",
        backends=members,
    )


def _rate_limit(status_headers: dict[str, str]) -> dict:
    """Build kwargs for a scripted 429 reply with arbitrary headers.

    Args:
        status_headers: Headers to send alongside the 429.

    Returns:
        Keyword arguments for ``aioresponses`` registration.
    """
    return {
        "status": 429,
        "content_type": "application/json",
        "body": json.dumps({"error": {"type": "rate_limit_error", "message": "slow down"}}),
        "headers": status_headers,
    }


def _cooldowns(server: BridgeServer) -> dict[str, tuple[float | int | None, str | None]]:
    """Read each member's ``(cooldown, failure_kind)`` by profile name.

    Args:
        server: The bridge server whose health records to inspect.

    Returns:
        One entry per balancing member, keyed by profile name.
    """
    return {
        member[2].name: (health.get("cooldown"), health.get("last_failure_kind"))
        for member, health in zip(server._backends, server._backend_health, strict=True)
    }


async def _post_messages(port: int, *, stream: bool) -> aiohttp.ClientResponse:
    """Post one Messages request to the bridge and return the response.

    Args:
        port: The bridge port to post to.
        stream: Whether to request a streaming reply.

    Returns:
        The completed client response (the caller reads/consumes it).
    """
    body: dict = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
    }
    if stream:
        body["stream"] = True
    session = aiohttp.ClientSession()
    try:
        resp = await session.post(f"http://127.0.0.1:{port}/v1/messages", json=body)
        await resp.read()
        return resp
    finally:
        await session.close()


class TestStreamingRetryAfter:
    """The translated streaming 429 arm parks for the hint's duration."""

    @pytest.mark.asyncio()
    async def test_429_parks_each_member_for_its_retry_after_hint(self, caplog) -> None:
        """Hints are honored per member and logged; the terminal status is unchanged."""
        caplog.set_level("DEBUG", logger="kitty.bridge.server")
        server = _server([_plain_member("plain-0", "api0.example.com"), _plain_member("plain-1", "api1.example.com")])
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api0.example.com/v1/messages", **_rate_limit({"Retry-After": "42"}))
                m.post("https://api1.example.com/v1/messages", **_rate_limit({"Retry-After": "84"}))
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json={
                        "model": "claude-sonnet-4-6",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 64,
                        "stream": True,
                    },
                ) as resp:
                    assert resp.status == 429, await resp.text()
        finally:
            await server.stop_async()

        cooldowns = _cooldowns(server)
        assert cooldowns["plain-0"] == (42, "rate_limit")
        assert cooldowns["plain-1"] == (84, "rate_limit")
        applied = [r for r in caplog.records if "Retry-After" in r.getMessage() and r.levelname == "DEBUG"]
        assert applied, "expected a DEBUG line naming the applied hint"

    @pytest.mark.asyncio()
    async def test_429_without_or_with_unparseable_hint_keeps_default_cooldown(self) -> None:
        """Absent and malformed hints both fall back to the fixed default."""
        server = _server([_plain_member("plain-0", "api0.example.com"), _plain_member("plain-1", "api1.example.com")])
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api0.example.com/v1/messages", **_rate_limit({}))
                m.post("https://api1.example.com/v1/messages", **_rate_limit({"Retry-After": "soon"}))
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json={
                        "model": "claude-sonnet-4-6",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 64,
                        "stream": True,
                    },
                ) as resp:
                    assert resp.status == 429, await resp.text()
        finally:
            await server.stop_async()

        cooldowns = _cooldowns(server)
        assert cooldowns["plain-0"] == (server._backend_cooldown, "rate_limit")
        assert cooldowns["plain-1"] == (server._backend_cooldown, "rate_limit")


class TestNonStreamingRetryAfter:
    """The translated non-streaming 429 classification carries the hint too."""

    @pytest.fixture
    def fast_empty_ladder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Collapse the post-exhaustion empty ladder's 20 s/40 s sleeps.

        Args:
            monkeypatch: pytest monkeypatch fixture.
        """
        monkeypatch.setattr(server_module, "_EMPTY_RETRY_DELAYS", [0.01, 0.01])
        monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])

    @pytest.mark.asyncio()
    async def test_429_parks_single_member_for_hint(self, fast_empty_ladder: None) -> None:
        """A hinted 429 parks the member for the hint's duration, not the default."""
        server = _server([_plain_member("solo", "api0.example.com")])
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api0.example.com/v1/messages", **_rate_limit({"Retry-After": "42"}), repeat=True)
                resp = await _post_messages(port, stream=False)
                assert resp.status == 429, await resp.text()
        finally:
            await server.stop_async()

        assert _cooldowns(server)["solo"] == (42, "rate_limit")

    @pytest.mark.asyncio()
    async def test_429_parks_single_member_for_default_without_hint(self, fast_empty_ladder: None) -> None:
        """Without the header the member parks for the backend default."""
        server = _server([_plain_member("solo", "api0.example.com")])
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api0.example.com/v1/messages", **_rate_limit({}), repeat=True)
                resp = await _post_messages(port, stream=False)
                assert resp.status == 429, await resp.text()
        finally:
            await server.stop_async()

        assert _cooldowns(server)["solo"][1] == "rate_limit"
        # Single-backend pools cap hard-failure cooldowns at 30 s (§6.1); the
        # hint path above must bypass that cap, the hintless path must not.
        assert _cooldowns(server)["solo"][0] == 30
