"""The curl_cffi recorder and its socket-level connection log, against real TLS.

`.system_design/TEST_SUITE.md` §5.5, §7.2, §7.2.3 · plan task **T-B2** (KBR-41) ·
`.requirements/20260914T184459Z_kbr_41_curl_cffi_recorder/REQUIREMENTS.md`.

Every conformance probe writes its request bytes itself, for the reason
``test_recorder.py`` records: a client library reorders, re-cases, adds and
drops headers, so a probe sent through one could not tell a recorder that *loses*
casing from a client that never *sent* mixed casing. The driver is shared with
every other recorder; only the ``ssl_context`` and the probe's path suffix are
this one's.

**The socket-level rule has two falsification directions** (plan §1.4): a
recorder that logs only after the handshake misses the silent connection
entirely, and a recorder that logs at accept time *and* again in
``connection_made`` splits one request across two records. Both are shipped as
tests that name their defect.

**Layer.** No ``pytestmark``: the ``l1`` path default, for the reason
``test_provider_aiohttp.py`` records — these bind real sockets and read as
``l3``, but no job selects ``l3`` today, so an ``l3`` marker would remove them
from every gate.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import ssl
import time

import pytest

from harness.connect_proxy import server_ssl_context  # noqa: F401  (pytest fixtures)
from harness.contract import WireFormat
from harness.curl_recorder import (
    CODEX_RESPONSES_SUFFIX,
    OAUTH_REFRESH_SUFFIX,
    CurlRecordingUpstream,
    format_for_curl_path,
    is_oauth_refresh_path,
    oauth_refresh_body,
    responses_success_body,
    responses_success_stream,
)
from harness.recorder_conformance import (
    PER_EXCHANGE_CHECKS,
    PER_SESSION_CHECKS,
    RICH_BODY,
    recording_of,
    send,
)

pytestmark = pytest.mark.usefixtures("certs")

#: A probe that exercises the recorder's suffix table. The suffix
#: ``/responses`` selects ``WireFormat.OPENAI_RESPONSES``; the path's
#: percent-encoded segments and the duplicated / mixed-casing header set are
#: what the conformance checks read. The marker travels in the
#: ``X-Probe-Marker`` header (the conformance driver's correlation channel),
#: so a per-test probe overrides the literal ``rich``.
def _responses_probe(marker: str = "rich") -> bytes:
    """Build the recorder's probe with the given correlation marker.

    Args:
        marker: The value placed in the ``X-Probe-Marker`` header.

    Returns:
        The complete request bytes.
    """
    return (
        b"POST /backend-api/codex/responses?Beta=A%20B&key=k&beta=c HTTP/1.1\r\n"
        b"Host: Upstream.Example:443\r\n"
        b"x-api-key: secret\r\n"
        b"anthropic-version: 2023-06-01\r\n"
        b"anthropic-beta: two\r\n"
        b"X-Weird: caf\xe9\r\n"
        b"X-Probe-Marker: " + marker.encode() + b"\r\n"
        b"Content-Length: " + str(len(RICH_BODY)).encode() + b"\r\n"
        b"\r\n" + RICH_BODY
    )


_RESPONSES_PROBE = _responses_probe()


@pytest.fixture
def server_context(certs) -> ssl.SSLContext:  # noqa: F811
    """Return the server-side TLS context the recorder terminates with.

    Args:
        certs: The session-scoped throwaway certificate set.

    Returns:
        A context presenting the target leaf certificate, whose SAN covers
        ``127.0.0.1`` — the address every probe in this module uses.
    """
    return server_ssl_context(certs.target_cert, certs.target_key)


@pytest.fixture
def verifying_client_context(certs) -> ssl.SSLContext:
    """Return a client-side context trusting the harness CA.

    Args:
        certs: The session-scoped throwaway certificate set.

    Returns:
        A context whose trust store holds the throwaway CA. Hostname checking
        stays off because the recorder binds loopback.
    """
    context = ssl.create_default_context(cafile=str(certs.ca))
    context.check_hostname = False
    return context


@pytest.fixture
def probing_client_context() -> ssl.SSLContext:
    """Return a client-side context for probes that skip chain verification.

    Returns:
        A context with ``CERT_NONE`` and no hostname check, so the conformance
        driver's raw-socket probes work without coupling this module to the
        session-scoped ``certs`` fixture.
    """
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


@pytest.fixture
async def recorder(server_context: ssl.SSLContext):
    """Start one recorder per test, and stop it afterwards.

    Args:
        server_context: The TLS context to terminate with.

    Yields:
        The started recorder.
    """
    upstream = CurlRecordingUpstream(
        ssl_context=server_context, default_format=WireFormat.OPENAI_RESPONSES
    )
    await upstream.start()
    try:
        yield upstream
    finally:
        await upstream.stop()


class TestTheRecorderPassesEveryConformanceCheck:
    """The claim every recorder is judged against, over TLS."""

    @pytest.mark.parametrize("check", PER_EXCHANGE_CHECKS, ids=lambda c: c.__name__)
    async def test_per_exchange_check(self, recorder: CurlRecordingUpstream, check, probing_client_context) -> None:
        """One probe, driven raw over TLS, judged by one check.

        Args:
            recorder: The started recorder.
            check: One of the ten per-exchange checks.
            probing_client_context: A context that skips chain verification.
        """
        sent = await send(
            recorder.host, recorder.port, _RESPONSES_PROBE, marker="rich", ssl_context=probing_client_context
        )
        captures = recorder.requests
        assert len(captures) == 1, "the probe must produce exactly one capture"
        check(captures[0], sent, scheme=recorder.scheme)

    @pytest.mark.parametrize("check", PER_SESSION_CHECKS, ids=lambda c: c.__name__)
    async def test_per_session_check(self, recorder: CurlRecordingUpstream, check, probing_client_context) -> None:
        """A full session — several probes and one silent connection — judged whole.

        Args:
            recorder: The started recorder.
            check: One of the four per-session checks.
            probing_client_context: A context that skips chain verification.
        """
        first = await send(
            recorder.host, recorder.port, _responses_probe("one"), marker="one", ssl_context=probing_client_context
        )
        second = await send(
            recorder.host, recorder.port, _responses_probe("two"), marker="two", ssl_context=probing_client_context
        )
        silent = await asyncio.to_thread(_open_only, recorder, ssl_context=probing_client_context)
        recording = recording_of(recorder)
        check(recording, [first, second], [silent])


async def _wait_until(recorder: CurlRecordingUpstream, predicate, *, timeout: float = 2.0) -> bool:
    """Poll ``recorder`` until ``predicate(recorder)`` is true, or give up.

    The connection log's entries appear asynchronously — a failed handshake
    completes on a later loop iteration, and a request's count increments only
    once the handler has read the body — so a test asserting on the log waits
    on a **condition**, with a bounded timeout, rather than on a fixed sleep.

    Args:
        recorder: The recorder whose connection log the predicate reads.
        predicate: A callable taking the recorder and returning truthiness.
        timeout: Seconds to wait before giving up.

    Returns:
        Whether the predicate held before the timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate(recorder):
            return True
        await asyncio.sleep(0.005)
    return predicate(recorder)


def _open_only(recorder: CurlRecordingUpstream, *, ssl_context: ssl.SSLContext) -> int:
    """Open one TLS connection, send nothing, close it, and report its source port.

    Sync so it can run on a worker thread; :meth:`asyncio.to_thread` wraps it.

    Args:
        recorder: The started recorder.
        ssl_context: The client context, so the handshake completes and the
            recorder's ``connection_made`` fills in the peer port.

    Returns:
        The source port the driver's socket used.
    """
    sock = socket.create_connection((recorder.host, recorder.port), timeout=5)
    try:
        wrapped = ssl_context.wrap_socket(sock, server_hostname=recorder.host)
        port = int(wrapped.getsockname()[1])
        wrapped.close()
        return port
    finally:
        with contextlib.suppress(OSError):
            sock.close()


class TestTheVocabulary:
    """The three overrides, and the OAuth endpoint that sits outside them."""

    def test_a_format_this_recorder_does_not_serve_is_refused_at_construction(
        self, server_context: ssl.SSLContext
    ) -> None:
        """The base class would reject it too, for the wrong reason and message.

        Args:
            server_context: The TLS context to build the recorder with.
        """
        with pytest.raises(ValueError, match="openai_responses"):
            CurlRecordingUpstream(
                ssl_context=server_context, default_format=WireFormat.ANTHROPIC_MESSAGES
            )

    def test_the_format_suffix_selects_responses_only(self) -> None:
        """The serving leg's suffix; the OAuth suffix is a different question."""
        assert format_for_curl_path(f"/backend-api/codex{CODEX_RESPONSES_SUFFIX}") is WireFormat.OPENAI_RESPONSES
        assert format_for_curl_path(OAUTH_REFRESH_SUFFIX) is None
        assert format_for_curl_path("/v1/messages") is None

    def test_the_oauth_suffix_and_the_format_suffix_are_distinct_questions(self) -> None:
        """A token grant is answered before any format lookup, §7.2.2's rule."""
        assert is_oauth_refresh_path(OAUTH_REFRESH_SUFFIX)
        assert not is_oauth_refresh_path(f"/backend-api/codex{CODEX_RESPONSES_SUFFIX}")

    def test_the_non_streaming_success_reads_as_a_success_to_the_product(self) -> None:
        """Every key the bridge's reader needs, with non-empty content."""
        body = responses_success_body()
        assert body["status"] == "completed"
        texts = [
            part["text"]
            for item in body["output"]
            for part in item.get("content", [])
            if part["type"] == "output_text"
        ]
        assert texts and all(texts), "an empty reply costs the retry ladder"

    def test_the_streamed_success_carries_a_text_delta(self) -> None:
        """The ``output_text.delta`` is what emptiness judgements key on."""
        stream = responses_success_stream()
        decoded = [json.loads(chunk.split(b"data: ", 1)[1]) for chunk in stream]
        types = [event["type"] for event in decoded]
        assert "response.output_text.delta" in types
        deltas = [event["delta"] for event in decoded if event["type"] == "response.output_text.delta"]
        assert all(deltas), "an empty delta is an empty reply and costs the ladder"

    def test_the_oauth_body_carries_every_field_the_leg_requires(self) -> None:
        """`_refresh` reads four fields, `_exchange_api_key` reads the fifth."""
        body = oauth_refresh_body()
        assert {"access_token", "refresh_token", "id_token", "expires_in"} <= set(body)
        assert body["openai_api_key"]


class TestSocketLevelConnectionLogging:
    """§7.2.1's limitation, closed — and the double-record direction, falsified."""

    async def test_a_silent_connection_is_logged_with_its_peer_port(
        self, recorder: CurlRecordingUpstream, verifying_client_context: ssl.SSLContext
    ) -> None:
        """§5.2.1's bypass shape, over TLS.

        Args:
            recorder: The started recorder.
            verifying_client_context: A context that completes the handshake,
                so ``connection_made`` fills in the peer port.
        """
        opened = await asyncio.to_thread(_open_only_sync, recorder, verifying_client_context)
        assert len(recorder.connections) == 1
        assert recorder.connections[0].peer_port == opened
        assert recorder.connections[0].requests == 0

    async def test_a_failed_handshake_is_still_logged(self, recorder: CurlRecordingUpstream) -> None:
        """The connection §7.2.1 names — accepted, never negotiated.

        A raw socket that starts the handshake and abandons it produces a
        record with the placeholder peer port: the accept happened, no peer
        ever identified itself, and §5.2.1's bypass shape is a connection that
        carries no request regardless of what it sent.

        Args:
            recorder: The started recorder.
        """
        await asyncio.to_thread(_start_and_drop_handshake, recorder.host, recorder.port)

        assert await _wait_until(
            recorder, lambda r: len(r.connections) >= 1, timeout=2.0
        ), "the failed handshake must produce a record within the timeout"

        assert len(recorder.connections) == 1, (
            "a connection that failed the TLS handshake is §7.2.1's bypass shape "
            "and must still be logged"
        )
        assert recorder.connections[0].peer_port == -1, (
            "no handshake completed, so no peer ever identified itself"
        )
        assert recorder.connections[0].requests == 0

    async def test_a_completed_exchange_produces_exactly_one_record(
        self, recorder: CurlRecordingUpstream, verifying_client_context: ssl.SSLContext
    ) -> None:
        """The double-record direction: accept-time and handshake-time do not both append.

        Args:
            recorder: The started recorder.
            verifying_client_context: A context that completes the handshake.
        """
        sent = await send(
            recorder.host, recorder.port, _RESPONSES_PROBE, marker="one", ssl_context=verifying_client_context
        )

        assert await _wait_until(
            recorder,
            lambda r: len(r.connections) >= 1 and r.connections[0].requests >= 1,
            timeout=2.0,
        ), "the request must reach the recorder's body within the timeout"

        assert len(recorder.connections) == 1, (
            "one request-bearing connection must produce one record — an accept-time "
            "append followed by a connection_made append would split it in two"
        )
        assert recorder.connections[0].requests == 1
        # Both observation points name the same socket: §5.2.1's join key.
        assert recorder.connections[0].peer_port == sent.source_port


class TestTLSIsTerminated:
    """The recorder presents a certificate a real client verifies."""

    def test_the_scheme_is_https(self, server_context: ssl.SSLContext) -> None:
        """A client pointed at ``http://`` reaches a TLS listener and fails.

        Args:
            server_context: The TLS context to build the recorder with.
        """
        upstream = CurlRecordingUpstream(
            ssl_context=server_context, default_format=WireFormat.OPENAI_RESPONSES
        )
        assert upstream.scheme == "https"

    async def test_a_verifying_client_completes_the_handshake(
        self, recorder: CurlRecordingUpstream, verifying_client_context: ssl.SSLContext
    ) -> None:
        """The harness CA verifies the recorder's certificate.

        Args:
            recorder: The started recorder.
            verifying_client_context: A context that completes the handshake.
        """
        cipher = await asyncio.to_thread(
            _verify_handshake_sync, recorder.host, recorder.port, verifying_client_context
        )
        assert cipher is not None


def _open_only_sync(recorder, ssl_context) -> int:
    """Worker-thread counterpart of :func:`_open_only`.

    Returns:
        The driver's source port.
    """
    return _open_only(recorder, ssl_context=ssl_context)


def _start_and_drop_handshake(host, port) -> None:
    """Send a ClientHello prefix, then close — a handshake that never finishes.

    Args:
        host: The recorder's loopback host.
        port: The recorder's bound port.
    """
    sock = socket.create_connection((host, port), timeout=5)
    try:
        # A TLS ClientHello prefix — enough for the server's SSLProtocol to
        # start, not enough to complete the handshake.
        sock.sendall(b"\x16\x03\x01\x00\x05")
    finally:
        sock.close()


def _verify_handshake_sync(host, port, ssl_context):
    """Open a TLS connection whose cipher proves the verifier completed the handshake.

    Returns:
        The negotiated cipher tuple, or ``None`` if the handshake failed.
    """
    sock = socket.create_connection((host, port), timeout=5)
    try:
        wrapped = ssl_context.wrap_socket(sock, server_hostname="127.0.0.1")
        cipher = wrapped.cipher()
        wrapped.close()
        return cipher
    finally:
        with contextlib.suppress(OSError):
            sock.close()
