"""R1 — Local bypass, end-to-end (T-E8, KBR-68).

``.system_design/TEST_SUITE.md`` §5.5 claim 1 · plan task **T-E8** ·
`.requirements/20260916T120000Z_kbr68_t_e8_local_bypass_fail_closed_transport_asymmetry/REQUIREMENTS.md`.

**The user-facing claim.** With egress configured, a request to a loopback or ``localhost``
provider (a local Ollama) is delivered directly — a rented proxy cannot reach the caller's
LAN — and the proxy records zero CONNECT attempts. The positive-polarity assertion (the
recorder log IS the direct-arrival evidence) is its own evidence that direct delivery is
reachable on this destination; §5.3's phase-1 positive control exists to bracket
negative-polarity assertions, which this is not.

**Why a plain-HTTP listener, not the ``SealedNetwork`` recorder.** The delivered recorder
binds TLS and its leaf cert's SAN is ``HARNESS_UPSTREAM_HOST`` only — pointing the bridge
at ``localhost`` would fail TLS hostname verification. A plain-HTTP listener matches the
scenario (local Ollama is plain HTTP) and avoids the SAN mismatch.

**Layer.** No ``pytestmark``, so this takes the ``l1`` path default, following the T-E2
slice. The drive is plain HTTP on loopback, so the Python ≥3.11 TLS-in-TLS skip the T-E2
proxied phases carry does not apply; the test runs on every supported interpreter.

**Falsification cross-reference.** AC1.2's "validated by mutation-style falsification
during development, not asserted in CI" is the local equivalent of
``tests/harness/test_aiohttp_containment_slice.py::TestPhase3Falsification`` — the standing
CI demonstration that the harness's ``attempts == []`` surface is sensitive to a bypass.

The falsification *channel* here is "drive fails and the recorder saw nothing", not the
public-side's "proxy.attempts non-empty": aiohttp sends absolute-form requests for
plain-HTTP targets (the harness's CONNECT-only proxy answers 405), so a bypass that
forces the loopback request through the proxy is caught by ``status != 200 and
bodies == []``, not by ``proxy.attempts != []``. That is honest about what the bridge's
client does — the T-E2 falsification on the *https* side covers the CONNECT-channel
sensitivity.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import aiohttp
import pytest

from harness.connect_proxy import (
    CertFiles,
    ConnectProxy,
    proxy_config,
    server_ssl_context,
)

# ── A minimal plain-HTTP recorder for the loopback leg ─────────────────────
#
# The "local Ollama" in this scenario is plain HTTP — TlsTarget is the
# harness's plain-HTTP analogue, but a SealedNetwork-less drive does not
# need its full surface (cert generation, recorder stats). A 30-line
# aiohttp.web app is the smallest sound shape: it accepts POST
# /v1/messages, returns a minimal Anthropic Messages-shaped 200, and shuts
# down cleanly on context-manager exit.


@asynccontextmanager
async def plain_http_recorder() -> AsyncGenerator[tuple[int, list[str]]]:
    """Bind a plain-HTTP aiohttp recorder on loopback and yield its port + bodies seen.

    The recorder accepts any path under ``/v1/`` and answers with a 200 body in the
    Anthropic Messages shape (a content block with the inbound text echoed). The bodies
    list lets AC1.1 assert the recorder saw exactly one request — the positive-polarity
    evidence the test name promises — and lets the body marker be checked directly.

    Yields:
        ``(port, bodies)`` — the kernel-chosen loopback port and the list of echoed
        bodies the recorder received, in arrival order.
    """
    from aiohttp import web

    bodies: list[str] = []

    async def handle(request: web.Request) -> web.Response:
        """Answer one request with a minimal Anthropic-Messages-shaped 200."""
        payload = await request.json()
        text = ""
        for msg in payload.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                text += content
            else:
                for part in content or []:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text += part.get("text", "")
        echoed = f"echo: {text}"
        bodies.append(echoed)
        body = {
            "id": "msg-localbypass",
            "type": "message",
            "role": "assistant",
            "model": payload.get("model", "harness-model"),
            "content": [{"type": "text", "text": echoed}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        return web.json_response(body)

    app = web.Application()
    app.router.add_post("/v1/{tail:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield port, bodies
    finally:
        await runner.cleanup()


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
async def local_bypass_harness(
    certs: CertFiles, aiohttp_trusts_test_ca: None
) -> AsyncGenerator[tuple[int, list[str], ConnectProxy], None]:
    """A plain-HTTP loopback recorder and a TLS ConnectProxy, started together.

    The recorder is the "local Ollama" the bridge is pointed at; the proxy is the harness
    whose ``attempts`` log the assertion reads. Both share the test's lifetime and are
    torn down in reverse start order so a started proxy never holds an orphaned recorder
    if its tear-down raises.

    The ``aiohttp_trusts_test_ca`` parameter is a cheap belt for the AC1.2 falsification:
    the passing path never uses the proxy context, and under the mutation the bridge's
    request for a plain-HTTP target through the proxy travels in **absolute form** — not
    CONNECT — so the proxy answers 405 and no TLS hop to the proxy is attempted either
    way (see the module docstring's falsification note). Keeping the fixture in scope
    costs nothing on the passing path and keeps the drive's setup identical to the
    T-E2 slices'.

    Yields:
        ``(recorder_port, bodies, proxy)`` — the loopback port, the recorder's seen
        bodies, and the proxy whose ``attempts`` the assertion reads.
    """
    async with plain_http_recorder() as (recorder_port, bodies):
        proxy = ConnectProxy()
        await proxy.start(server_ssl_context(certs.proxy_cert, certs.proxy_key))
        try:
            yield recorder_port, bodies, proxy
        finally:
            await proxy.stop()


async def _drive(recorder_port: int, proxy: ConnectProxy, host_label: str) -> tuple[int, str]:
    """Drive one request through the bridge pointed at ``http://{host_label}:{recorder_port}``.

    Args:
        recorder_port: The plain-HTTP loopback port the recorder bound.
        proxy: The harness CONNECT proxy.
        host_label: The hostname portion of the provider's ``base_url`` — either
            ``"localhost"`` or ``"127.0.0.1"``.

    Returns:
        ``(status, text)`` — the response status and body the test client received from
        the bridge. A negative status never occurs here (the bridge never returns one);
        included for symmetry with the T-E2 slice's result shape.
    """
    from kitty.bridge.server import BridgeServer
    from kitty.providers.custom_anthropic import CustomAnthropicAdapter

    adapter = CustomAnthropicAdapter()
    server = BridgeServer(
        None,  # type: ignore[arg-type]
        adapter,
        resolved_key="harness-key",
        model="harness-model",
        provider_config={"base_url": f"http://{host_label}:{recorder_port}"},
        egress=proxy_config(proxy.port),
    )
    try:
        bridge_port = await server.start_async()
        body = {
            "model": "harness-model",
            "messages": [{"role": "user", "content": f"kbr68-{host_label}"}],
            "max_tokens": 16,
            "stream": False,
        }
        async with aiohttp.ClientSession() as client:
            response = await client.post(
                f"http://127.0.0.1:{bridge_port}/v1/messages",
                json=body,
                timeout=aiohttp.ClientTimeout(total=10.0),
            )
            return response.status, await response.text()
    finally:
        await server.stop_async()


# ── R1 AC1.1 — local bypass is end-to-end ─────────────────────────────────


class TestLocalBypass:
    """AC1.1: a loopback / ``localhost`` provider is reached directly under egress.

    Both branches of :func:`kitty.egress.should_bypass` are exercised end-to-end:
    the hostname branch (``localhost``) and the IP-literal branch (``127.0.0.1``). The
    harness's CONNECT proxy sits idle in both cases; the assertion is the proxy's
    ``attempts == []``.
    """

    @pytest.mark.parametrize("host_label", ["localhost", "127.0.0.1"])
    async def test_egress_configured_loopback_provider_is_reached_directly_with_zero_proxy_attempts(
        self,
        local_bypass_harness: tuple[int, list[str], ConnectProxy],
        host_label: str,
    ) -> None:
        """AC1.1: 200, exactly one capture with the expected body, zero CONNECT attempts.

        The bridge's ``_session_for`` consults :func:`kitty.egress.should_bypass` and
        returns the direct session for loopback / localhost URLs. The bridge then
        connects to ``http://{host_label}:{recorder_port}`` without tunnelling; the
        proxy is never asked to CONNECT.
        """
        recorder_port, bodies, proxy = local_bypass_harness

        status, _text = await _drive(recorder_port, proxy, host_label)

        assert status == 200, f"bridge answered {status} for {host_label}: expected 200"
        assert len(bodies) == 1, (
            f"recorder received {len(bodies)} request(s) for {host_label}, expected "
            "exactly one — the local-bypass property is a positive-polarity claim that "
            "the recorder observed the bridge's request directly"
        )
        assert bodies[0] == f"echo: kbr68-{host_label}", (
            f"recorder echoed {bodies[0]!r}, expected the marker for {host_label}: the "
            "wrong echo would mean the bridge addressed the recorder by a different "
            "hostname than the test set up"
        )
        assert proxy.attempts == [], (
            f"proxy saw {len(proxy.attempts)} CONNECT attempt(s) for {host_label}: "
            "the local-bypass property (a rented proxy cannot reach the caller's LAN) "
            "is broken — every CONNECT attempt would leak the user's IP"
        )
