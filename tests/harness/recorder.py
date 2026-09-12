"""The primary aiohttp recording upstream.

`.system_design/TEST_SUITE.md` §7.2 · plan task **T-W4** (KBR-27).

A real HTTP server that stands in for a provider. It records what the bridge put
on the wire with enough fidelity to answer §4.3 C1 (the exact header set),
§3.3.5 (routing, which on Azure lives only in the URL) and §5.2.1 (which TCP
connection carried it), and it replies with a minimal valid success so that a
request driven through a real bridge completes in **one** upstream attempt.

**It imports nothing from ``src/kitty``, and must not.** §3.3.1's
independent-oracle rule governs everything the fidelity claim rests on, and a
recorder that asked kitty how to read a request would inherit kitty's bugs.
``test_recorder.py`` asserts the absence structurally, sharing
``test_contract.py``'s guard.

**Why the low-level ``web.Server`` and not ``web.Application``.** Two things this
recorder must do are unreachable from the high-level route:

* **Connections that carry no request.** ``Server.connection_made(handler,
  transport)`` is the only public seam that sees them, and §5.2.1 says an
  upstream connection with no matching tunnel *"is the only thing this assertion
  needs to catch"*. Wrapping the protocol object instead is impossible:
  ``RequestHandler`` defines ``__slots__``, so assigning to
  ``proto.connection_made`` raises, and the symptom is a bare
  ``ConnectionResetError`` at the client with no traceback.
* **Unbounded request bodies.** ``client_max_size`` is a ``BaseRequest``
  argument, not a ``Server`` one, so it is supplied through
  ``request_factory``. See :data:`_CLIENT_MAX_SIZE`.

Routing is not needed — a recorder answers every path — so the low-level server
costs nothing else.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from aiohttp import web

from harness.contract import CapturedRequest, WireFormat

__all__ = [
    "RecordingUpstream",
    "ConnectionRecord",
    "Responder",
    "Reply",
    "minimal_success_body",
    "minimal_success_stream",
    "format_for_path",
    "UnmatchedPathError",
]

#: Disables aiohttp's request-body ceiling outright. ``BaseRequest.read()``
#: guards on ``if self._client_max_size:``, so zero switches the check off —
#: deliberately, rather than picking a large number.
#:
#: The default is 1 MiB, and §7.1's corpus is specified to contain transcripts
#: *over the compaction budget*. At the default those arrive as a harness **413**,
#: which is M6's own trigger ("body re-compacted at half budget") — so the
#: harness would manufacture the mutation the oracle exists to detect. A fixed
#: ceiling has the same failure later, and the comparison is ``>=``, so a limit
#: equal to an entry's exact size still raises.
_CLIENT_MAX_SIZE = 0

#: Path suffixes that select the wire format of the reply. A **suffix** match,
#: because almost no adapter posts to a bare path: Azure sends
#: ``/openai/deployments/{id}/chat/completions``, vertex
#: ``/endpoints/openapi/chat/completions``, zai_anthropic
#: ``/api/anthropic/v1/messages``, and ollama ``/v1/chat/completions``. An exact
#: match would have selected correctly for three adapters and wrongly for the
#: rest. A bare ``/messages`` entry is deliberately absent: no shipped adapter
#: posts to one, and zai_anthropic's path already ends in ``/v1/messages``.
_FORMAT_SUFFIXES: tuple[tuple[str, WireFormat], ...] = (
    ("/chat/completions", WireFormat.CHAT_COMPLETIONS),
    ("/v1/messages", WireFormat.ANTHROPIC_MESSAGES),
)

#: The formats §7.2 assigns to the primary recorder. The other three — OpenAI
#: Responses, Bedrock Converse, Ollama ``/api/chat`` — belong to T-B1–T-B3,
#: whose transports this recorder never sees.
_SERVED_FORMATS = frozenset({WireFormat.ANTHROPIC_MESSAGES, WireFormat.CHAT_COMPLETIONS})

#: The text every minimal success carries. Non-empty deliberately: every
#: emptiness judgement in `server.py` keys on content being present, and an
#: empty reply costs 80 seconds of retry ladder.
_REPLY_TEXT = "ok"


class UnmatchedPathError(AssertionError):
    """Raised when a recorder is asked about requests that took the fallback.

    A request whose path matched no suffix is answered in the recorder's
    declared default format, which may be the wrong one. That failure is silent
    and expensive rather than loud: the adapter parses nothing out of the reply,
    the bridge judges the response empty, and the test pays the retry ladder. So
    the fallback is recorded and :meth:`RecordingUpstream.assert_all_paths_matched`
    turns it into a failure at teardown.
    """


@dataclass(frozen=True)
class ConnectionRecord:
    """One accepted TCP connection.

    Keyed on the ``RequestHandler`` the aiohttp seam already hands us rather
    than on the peer port, because §4.3 C5 counts *distinct connections* and a
    port-deduplicated count is wrong once the kernel reuses a source port. The
    port stays on the record as §5.2.1's cross-process join key against the
    proxy's tunnel log.

    Attributes:
        connection_id: Identity of the connection within this recorder,
            assigned in accept order.
        peer_port: The peer's port — the client's own source port.
        requests: How many requests rode on this connection. Zero is the
            interesting value: §5.2.1's bypass is a connection that carries
            none, and nothing in the request list can express it.
    """

    connection_id: int
    peer_port: int
    requests: int = 0


#: What a recorder replies with. Handed the capture and an unprepared
#: ``StreamResponse``, so it owns status, headers, body **and** the decision to
#: abort mid-stream. §7.2 requires every recorder to replay "error statuses,
#: Cloudflare blocks, empty responses, context-too-large rejections, and
#: disconnects at each of §6.3.1's four injection points"; a hook that returned a
#: finished response could express none of the last. T-B4 builds its failure
#: library on this signature rather than by editing this module.
Responder = Callable[[CapturedRequest, "Reply"], Awaitable[None]]


def minimal_success_body(fmt: WireFormat) -> dict[str, Any]:
    """Return the smallest non-streaming success body for ``fmt``.

    "Minimal" means the fewest keys that still read as a success to the bridge —
    not a faithful sample of a provider's response. What a *valid* body of each
    format is belongs to the wire readers (T-A1, T-A2), which are written against
    each format's published examples; duplicating that judgement here would
    create a second source of truth for it.

    Args:
        fmt: The wire format to answer in.

    Returns:
        The response body, ready to serialize.

    Raises:
        ValueError: When ``fmt`` is not one of the two formats §7.2 assigns to
            the primary recorder.
    """
    if fmt is WireFormat.ANTHROPIC_MESSAGES:
        return {
            "id": "msg_recorder",
            "type": "message",
            "role": "assistant",
            "model": "recorder-model",
            "content": [{"type": "text", "text": _REPLY_TEXT}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    if fmt is WireFormat.CHAT_COMPLETIONS:
        return {
            "id": "chatcmpl-recorder",
            "object": "chat.completion",
            "created": 0,
            "model": "recorder-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": _REPLY_TEXT},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    raise ValueError(f"the primary recorder serves {sorted(f.value for f in _SERVED_FORMATS)}, not {fmt.value}")


def minimal_success_stream(fmt: WireFormat) -> tuple[bytes, ...]:
    """Return the smallest valid SSE success stream for ``fmt``.

    Streaming is in scope because Claude Code sends ``"stream": true``, and a
    streaming request answered with a JSON body is not a success — it lands in
    the same retry ladder the non-streaming case would.

    For Anthropic Messages the sequence follows §6.2.2's grammar —
    ``message_start`` … ``content_block_start`` / ``content_block_delta`` /
    ``content_block_stop`` … ``message_delta``, ``message_stop``. That grammar
    is the *only* guard on this path: verified at ``server.py:3616``, a native
    Messages stream is forwarded to the client byte-for-byte and never reaches a
    translator, so no bridge-side emptiness judgement sees it.

    Args:
        fmt: The wire format to answer in.

    Returns:
        The SSE chunks, in order, each already terminated.

    Raises:
        ValueError: When ``fmt`` is not one of the two formats §7.2 assigns to
            the primary recorder.
    """
    if fmt is WireFormat.ANTHROPIC_MESSAGES:
        events = [
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_recorder", "type": "message", "role": "assistant",
                    "model": "recorder-model", "content": [], "stop_reason": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            }),
            ("content_block_start", {
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""},
            }),
            ("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": _REPLY_TEXT},
            }),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            }),
            ("message_stop", {"type": "message_stop"}),
        ]
        return tuple(
            f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode() for name, payload in events
        )

    if fmt is WireFormat.CHAT_COMPLETIONS:
        chunks = [
            {
                "id": "chatcmpl-recorder", "object": "chat.completion.chunk", "created": 0,
                "model": "recorder-model",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": _REPLY_TEXT},
                             "finish_reason": None}],
            },
            {
                "id": "chatcmpl-recorder", "object": "chat.completion.chunk", "created": 0,
                "model": "recorder-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        return tuple(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks) + (
            b"data: [DONE]\n\n",
        )

    raise ValueError(f"the primary recorder serves {sorted(f.value for f in _SERVED_FORMATS)}, not {fmt.value}")


def format_for_path(path: str) -> WireFormat | None:
    """Return the wire format a request at ``path`` selects, or ``None``.

    The pure half of :meth:`RecordingUpstream._format_for`: it answers the
    question and records nothing.  Split out for T-W8 (KBR-31), whose bridge
    fixture asserts at teardown that every captured path selected the format its
    transport declares.  That assertion cannot call the method — the method
    *appends to* :attr:`RecordingUpstream.unmatched` on a miss, so judging the
    captures with it would mutate the evidence being judged, on every fixture
    exit, and would fight the tests that deliberately clear that list.  Copying
    :data:`_FORMAT_SUFFIXES` into the fixture was the alternative and is worse:
    a second source of truth that stays green when the table moves.

    Args:
        path: The request's raw path.

    Returns:
        The format whose suffix ``path`` ends with, or ``None`` when no suffix
        matches.  ``None`` rather than a default, because "no rule applies" and
        "the fallback applies" are different facts and only the caller knows
        which it wants.
    """
    for suffix, fmt in _FORMAT_SUFFIXES:
        if path.endswith(suffix):
            return fmt
    return None


def _wants_stream(body: bytes) -> bool:
    """Return whether the request asked for a streamed reply.

    Args:
        body: The raw request body.

    Returns:
        True when the body is JSON with ``"stream": true``. A body that will not
        parse is not an error here — see :meth:`RecordingUpstream._format_for`.
    """
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return False
    return isinstance(parsed, dict) and parsed.get("stream") is True


@dataclass
class RecordingUpstream:
    """An HTTP server that records what reaches it and replies with a success.

    Bind it, point a bridge at :attr:`base_url`, then read :attr:`requests` and
    :attr:`connections`.

    One instance per test. A shared instance mixes captures across tests, and
    the ordering claim in the conformance suite is only meaningful per-instance.

    Attributes:
        default_format: The format used when a request's path matches no suffix.
            Required, with no implicit default: choosing one silently is how a
            wrong-format reply reaches a bridge, and that failure is not loud —
            the adapter parses nothing, the reply reads as empty, and the test
            pays the retry ladder.
        host: The bind address. Loopback is correct here and for T-W8; it is
            **not** a containment decision. §5.3 requires the sealed-network
            harness to address the upstream by a non-loopback name, and T-E2
            supplies that name and the per-transport resolver override.
        responder: What to reply with. Defaults to the minimal success.
        requests: Captures, in arrival order.
        connections: One :class:`ConnectionRecord` per accepted connection.
        unmatched: Paths that fell through to :attr:`default_format`.
    """

    default_format: WireFormat
    host: str = "127.0.0.1"
    responder: Responder | None = None
    connections: list[ConnectionRecord] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    #: Arrival-ordered slots. A slot is reserved at handler entry and filled
    #: once the body has been read, so that a short request cannot overtake a
    #: long one in the published order. A slot whose request never completed --
    #: a client that disconnected mid-body, which section 6.3.1 injects
    #: deliberately -- keeps its sentinel and is filtered out by
    #: :attr:`requests`; publishing it would put a request that never happened
    #: into the evidence every oracle reads.
    _slots: list[CapturedRequest] = field(default_factory=list, repr=False)

    #: The `web.Server` subclass to bind. A seam so a deliberately defective
    #: recorder can supply its own without patching a module global across an
    #: await -- which would leak the defect into any other recorder started
    #: concurrently on the same loop.
    _server_class: ClassVar[type[web.Server]]

    _runner: web.ServerRunner | None = field(default=None, init=False, repr=False)
    _port: int = field(default=0, init=False, repr=False)
    #: Connection id per ``RequestHandler``, keyed on the **object** and held for
    #: the recorder's lifetime. Never on ``id(handler)``: CPython reuses
    #: addresses after collection, so an id-keyed map that forgot entries could
    #: attribute a later connection's request to an earlier connection — which is
    #: the port-reuse failure R1.10 exists to eliminate, reintroduced one level
    #: down. A ``WeakKeyDictionary`` is not an option either: ``RequestHandler``
    #: omits ``__weakref__`` from ``__slots__``, so it cannot be weakly
    #: referenced (verified). The strong reference is bounded by this recorder
    #: being per-test.
    _by_handler: dict[Any, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Reject a format this recorder does not serve.

        Raises:
            ValueError: When ``default_format`` is outside the two §7.2 assigns
                to the primary recorder. Failing at construction beats replying
                in a format no adapter asked for.
        """
        if self.default_format not in _SERVED_FORMATS:
            raise ValueError(
                f"the primary recorder serves "
                f"{sorted(f.value for f in _SERVED_FORMATS)}, not {self.default_format.value}"
            )

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Bind an ephemeral port and begin accepting.

        Raises:
            RuntimeError: When called on an already-started recorder.
        """
        if self._runner is not None:
            raise RuntimeError("recorder already started")

        server = self._server_class(
            self._handle,
            recorder=self,
            auto_decompress=False,
            request_factory=_request_factory,
        )
        self._runner = web.ServerRunner(server)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, 0)
        await site.start()
        # Port 0 means the kernel chose one. `BaseRunner.addresses` is the
        # public way to read it back; the socket underneath is not.
        self._port = int(self._runner.addresses[0][1])

    async def stop(self) -> None:
        """Stop accepting and release the port."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def __aenter__(self) -> RecordingUpstream:
        """Start the recorder.

        Returns:
            The started recorder.
        """
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Stop the recorder.

        Args:
            *exc: The exception triple, unused; the port is released either way.
        """
        await self.stop()

    # -- addressing --------------------------------------------------------

    @property
    def requests(self) -> list[CapturedRequest]:
        """Return the completed captures, in arrival order.

        A slot reserved for a request that never finished — a client that
        disconnected mid-body, which §6.3.1 injects at four points — is not
        published. Publishing it would put a request that never happened into
        the evidence every oracle reads, beside real ones, and the failure would
        surface as a confusing "capture nobody sent" rather than as a disconnect.

        Returns:
            The captures whose requests completed.
        """
        return [c for c in self._slots if c is not _PENDING]

    @property
    def port(self) -> int:
        """Return the bound port.

        Returns:
            The ephemeral port chosen by the kernel.

        Raises:
            RuntimeError: When the recorder has not been started.
        """
        if self._runner is None:
            raise RuntimeError("recorder is not running; call start() first")
        return self._port

    @property
    def scheme(self) -> str:
        """Return the scheme this recorder serves.

        Returns:
            ``"http"``. Declared rather than assumed by the conformance suite,
            because §7.2's curl_cffi recorder terminates TLS.
        """
        return "http"

    @property
    def base_url(self) -> str:
        """Return the URL a provider adapter should be pointed at.

        Returns:
            The scheme, host and port, with no trailing slash.
        """
        return f"{self.scheme}://{self.host}:{self.port}"

    # -- assertions --------------------------------------------------------

    def assert_all_paths_matched(self) -> None:
        """Fail when any request was answered in the fallback format.

        Raises:
            UnmatchedPathError: When at least one path matched no suffix. Call
                this at teardown; a test that means to exercise the fallback
                clears :attr:`unmatched` instead.
        """
        if self.unmatched:
            raise UnmatchedPathError(
                f"answered in the fallback format {self.default_format.value} for "
                f"{self.unmatched}; the reply may be the wrong shape, and a "
                "wrong-shaped reply reads as an empty response rather than an error"
            )

    # -- request handling --------------------------------------------------

    def _format_for(self, path: str) -> WireFormat:
        """Return the wire format to answer a request at ``path`` in.

        Dispatches on the path because that is what a real provider does, and
        because bodies do not always distinguish the formats. Note this decides
        only what the recorder *replies*; the oracle still selects projections
        by the shape observed on the wire (§3.3.4), and nothing may route this
        decision into it.

        Args:
            path: The request's raw path.

        Returns:
            The matched format, or :attr:`default_format` — recording the miss.
        """
        matched = format_for_path(path)
        if matched is not None:
            return matched

        # The recording half, which is why this is a method and not the pure
        # function above: a miss is what `assert_all_paths_matched` reports.
        self.unmatched.append(path)
        return self.default_format

    async def _handle(self, request: web.BaseRequest) -> web.StreamResponse:
        """Record one request and reply to it.

        Args:
            request: The inbound aiohttp request.

        Returns:
            The response, already prepared and written by the responder.
        """
        # Stamp arrival and reserve the slot before anything can await. Reading
        # the body first would let a short later request overtake a long earlier
        # one, and the recorded order would stop being the arrival order.
        arrival = time.monotonic()
        index = len(self._slots)
        self._slots.append(_PENDING)

        connection_id = self._by_handler.get(request.protocol, -1)

        # Read the body before counting it. A client that disconnects mid-body
        # leaves its slot unfilled and therefore out of `requests` -- so counting
        # at handler entry would leave the connection log claiming a request the
        # request list does not have, and `check_connection_logged`'s
        # `carried == len(sent)` half would fail for a disconnect rather than for
        # a defect. Section 6.3.1 injects exactly that disconnect at four points.
        body = await request.read()

        self.count_on_connection(connection_id)

        captured = self.capture(request, body, arrival)
        self.store(index, captured)

        response = Reply(request)
        responder = self.responder or self._default_responder
        await responder(captured, response)
        return response

    def count_on_connection(self, connection_id: int) -> None:
        """Attribute one completed request to the connection that carried it.

        The second half of R1.10: §4.3 C5 counts distinct connections, and a
        capture has to be attributable to one of them. A seam of its own, beside
        :meth:`capture` and :meth:`store`, so a falsification recorder can
        miscount without also disturbing what was captured or where it was
        filed.

        Args:
            connection_id: The connection's id, or a negative value when the
                connection is already gone.
        """
        if not 0 <= connection_id < len(self.connections):
            return

        existing = self.connections[connection_id]
        # Read-modify-write with no `await` between the read and the write, so
        # the event loop cannot interleave another handler here.
        self.connections[connection_id] = ConnectionRecord(
            existing.connection_id, existing.peer_port, existing.requests + 1
        )

    def capture(
        self, request: web.BaseRequest, body: bytes, arrival: float
    ) -> CapturedRequest:
        """Build the record of one request.

        The single point where an observation becomes evidence, and therefore
        the single point a **defective** recorder has to override. The
        falsification suite subclasses this and nothing else, so a deliberate
        defect in what is captured cannot disturb the connection log or the
        recorded ordering — which is what lets each defect be attributed to one
        conformance check.

        Args:
            request: The inbound aiohttp request.
            body: The entity body, already read.
            arrival: The timestamp taken at handler entry.

        Returns:
            The capture.
        """
        return CapturedRequest(
            method=request.method,
            scheme=self.scheme,
            # Read the authority out of the headers. `request.host` falls back
            # to `socket.getfqdn()` when the client sent no Host, inventing the
            # machine's own name — and that fallback differs across the versions
            # pyproject allows.
            host=_host_header(request.raw_headers),
            # `rel_url.raw_path` / `raw_query_string`, never `path` /
            # `query_string`: the latter are percent-decoded.
            path=str(request.rel_url.raw_path),
            query=request.rel_url.raw_query_string,
            headers=_decode_headers(request.raw_headers),
            body=body,
            arrival=arrival,
            peer_port=_peer_port(request),
        )

    def store(self, index: int, captured: CapturedRequest) -> None:
        """File a capture in the slot reserved for it at handler entry.

        Separate from :meth:`capture` so that an ordering defect and a capture
        defect are distinct overrides, and so neither can be mistaken for the
        other when the falsification suite attributes a failure.

        Args:
            index: The slot reserved when the request arrived.
            captured: The capture to file.
        """
        self._slots[index] = captured

    async def _default_responder(
        self, captured: CapturedRequest, response: Reply
    ) -> None:
        """Reply with the minimal success for the request's format.

        Args:
            captured: The recorded request.
            response: The unprepared response to write through.
        """
        fmt = self._format_for(captured.path)

        if _wants_stream(captured.body):
            await response.begin(200, {"Content-Type": "text/event-stream"})
            for chunk in minimal_success_stream(fmt):
                await response.write(chunk)
            await response.write_eof()
            return

        payload = json.dumps(minimal_success_body(fmt)).encode()
        response.content_length = len(payload)
        await response.begin(200, {"Content-Type": "application/json"})
        await response.write(payload)
        await response.write_eof()


#: Placeholder occupying a request's slot between arrival and the body being
#: read. It is always overwritten before `_handle` returns; it exists so the
#: recorded order is arrival order even when two requests overlap.
_PENDING = CapturedRequest(method="", scheme="", host="", path="", query="")


def _request_factory(
    message: Any, payload: Any, protocol: Any, writer: Any, task: Any
) -> web.BaseRequest:
    """Build a request with the body ceiling disabled.

    ``client_max_size`` is a ``BaseRequest`` argument, not a ``Server`` one;
    passing it to ``web.Server`` forwards it to ``RequestHandler``, which does
    not accept it, and the only symptom is a connection reset at the client.

    Args:
        message: The parsed request line and headers.
        payload: The body stream.
        protocol: The owning ``RequestHandler``.
        writer: The payload writer.
        task: The handler task.

    Returns:
        The request, with no size ceiling.
    """
    return web.BaseRequest(
        message, payload, protocol, writer, task, protocol._loop,
        client_max_size=_CLIENT_MAX_SIZE,
    )


class _ConnectionLoggingServer(web.Server):
    """A ``web.Server`` that records every connection it accepts.

    Overriding ``Server.connection_made`` is the only public way to see a
    connection that carries **no** request — §5.2.1's bypass shape. Wrapping the
    ``RequestHandler`` is not an option: it defines ``__slots__``, so assigning
    to ``proto.connection_made`` raises ``AttributeError`` inside the protocol
    factory, and the client sees only a connection reset.

    A caveat worth stating where the next reader will find it: with a
    TLS-terminating site this fires **after** the handshake, so a connection
    that fails negotiation is not logged. §5.2.1 names a failed TLS negotiation
    as a real bypass, so T-B2 and T-E2 must observe at socket level or accept
    the limitation explicitly.
    """

    def __init__(self, handler: Any, *, recorder: RecordingUpstream, **kwargs: Any) -> None:
        """Bind the server to the recorder it logs into.

        Args:
            handler: The request handler coroutine.
            recorder: The recorder owning the connection log.
            **kwargs: Forwarded to ``web.Server``.
        """
        super().__init__(handler, **kwargs)
        self._recorder = recorder

    def connection_made(self, handler: Any, transport: Any) -> None:
        """Log the accepted connection, then hand off to aiohttp.

        Args:
            handler: The ``RequestHandler`` serving this connection; its
                identity is the connection's key, because ``BaseRequest.protocol``
                is the same object and a peer port is not unique over time.
            transport: The connection's transport.
        """
        peer = transport.get_extra_info("peername")
        connection_id = len(self._recorder.connections)
        self._recorder.connections.append(
            ConnectionRecord(connection_id, peer[1] if peer else -1)
        )
        self._recorder._by_handler[handler] = connection_id
        super().connection_made(handler, transport)


def _decode_headers(raw: Sequence[tuple[bytes, bytes]]) -> tuple[tuple[str, str], ...]:
    """Decode aiohttp's raw header bytes, preserving casing, order and repeats.

    ``latin-1`` rather than ``utf-8``: header values may carry obs-text, which
    ``utf-8`` refuses and ``latin-1`` never does, and T-W2's contract types the
    field ``str``. A decode error here would be a capture that fails rather than
    a capture that lies, but it would still lose the evidence.

    Args:
        raw: ``request.raw_headers``.

    Returns:
        The pairs, untouched but for the decode.
    """
    return tuple((name.decode("latin-1"), value.decode("latin-1")) for name, value in raw)


def _host_header(raw: Sequence[tuple[bytes, bytes]]) -> str:
    """Return the ``Host`` header as sent, or ``""`` when it was not sent.

    Args:
        raw: ``request.raw_headers``.

    Returns:
        The first ``Host`` value with its original casing, or the empty string.
    """
    for name, value in raw:
        if name.lower() == b"host":
            return value.decode("latin-1")
    return ""


def _peer_port(request: web.BaseRequest) -> int | None:
    """Return the peer port of the connection this request arrived on.

    Args:
        request: The inbound request.

    Returns:
        The port, or ``None`` when the transport is already gone.
    """
    transport = request.transport
    if transport is None:
        return None
    peer = transport.get_extra_info("peername")
    return peer[1] if peer else None


class Reply(web.StreamResponse):
    """A response that knows the request it answers.

    ``StreamResponse.prepare`` needs the request, and ``_req`` is only set *by*
    ``prepare`` — so a responder handed a bare ``StreamResponse`` has no way to
    start writing. Carrying the request on the response keeps the hook's
    signature to two arguments while leaving it full control of status, headers,
    body and abort.

    Attributes:
        request: The request being answered.
    """

    def __init__(self, request: web.BaseRequest) -> None:
        """Bind the response to its request.

        Args:
            request: The request being answered.
        """
        super().__init__()
        self.request = request

    async def begin(self, status: int = 200, headers: Mapping[str, str] | None = None) -> None:
        """Send the status line and headers, opening the body for writing.

        Args:
            status: The HTTP status to send.
            headers: Response headers, if any.
        """
        self.set_status(status)
        for name, value in (headers or {}).items():
            self.headers[name] = value
        await self.prepare(self.request)

    async def abort(self) -> None:
        """Drop the connection without finishing the response.

        §6.3.1 injects a disconnect at four points, one of them part-way through
        a stream. T-B4 builds those on this.
        """
        transport = self.request.transport
        if transport is not None:
            transport.abort()


# Bound after the class is defined, because the server subclass needs to refer
# to `RecordingUpstream` in its own annotations.
RecordingUpstream._server_class = _ConnectionLoggingServer
