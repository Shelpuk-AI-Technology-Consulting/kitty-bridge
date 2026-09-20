"""The curl_cffi recording upstream — TLS-terminating, socket-level logging.

`.system_design/TEST_SUITE.md` §5.5, §7.2, §7.2.3 · plan task **T-B2** (KBR-41).

§7.2 gives each of the bridge's five client configurations its own recorder.
The OpenAI subscription provider's two legs both ride ``curl_cffi.AsyncSession``
with a Chrome TLS fingerprint: the **serving** leg at ``/backend-api/codex/responses``
(OpenAI Responses wire format, always ``stream: true`` — register row P17) and
the **OAuth refresh** leg at ``auth.openai.com/oauth/token`` (form-encoded, which
KBR-161 moved off the aiohttp stack T-B1's recorder sees). Neither leg can reach
a plain-HTTP recorder, so this one terminates TLS with the harness certificate.

**Why this subclasses T-W4's recorder rather than being a second server.** §7.2.2
records the argument for T-B1 and it applies unchanged: §7.2.1 catalogues six
ways an aiohttp recorder can look correct and lie, and a second implementation is
a second chance to get each of them wrong. What differs here is vocabulary — the
format served, the suffix table, and what a minimal success is — so those are
overridden and nothing else. One thing is *added*, and it is the one thing this
recorder exists for:

**Connection logging at socket level (§7.2.1's limitation, owner decision).**
With a TLS-terminating site, ``Server.connection_made`` fires after the
handshake, so a connection that fails negotiation is invisible there — and §5.2.1
names failed TLS negotiation as a real bypass shape. This recorder logs the
accept in the **protocol factory**, which asyncio calls before the TLS handshake
begins, so every accepted connection produces a ``ConnectionRecord``. The peer
port is not visible to the factory (the accepted socket has not yet been handed
to anything), so the record is created with ``peer_port = -1`` and filled in by
``connection_made`` once the handshake completes. A record that keeps ``-1`` is
honest evidence that a connection existed and no peer ever identified itself.

**It imports nothing from ``src/kitty``, and must not**, for the reason
:mod:`harness.recorder` states. The product import this delivery needs lives in
:mod:`harness.curl_cffi`, which binds an adapter; nothing here asks kitty how to
read a request.
"""

from __future__ import annotations

import json
from typing import Any

from aiohttp import web

from harness.contract import CapturedRequest, WireFormat

# ``_wants_stream`` is private and deliberately imported anyway, for §7.2.2's
# reason: whether a body asked for a stream is one decision, and OpenAI's
# Responses API spells it the same way every other format does.
from harness.recorder import ConnectionRecord, RecordingUpstream, Reply, _request_factory, _wants_stream

__all__ = [
    "CurlRecordingUpstream",
    "CODEX_RESPONSES_SUFFIX",
    "OAUTH_REFRESH_SUFFIX",
    "format_for_curl_path",
    "is_oauth_refresh_path",
    "responses_success_body",
    "responses_success_stream",
    "oauth_refresh_body",
]

#: The suffix that selects an OpenAI Responses reply. The serving leg posts to
#: ``/backend-api/codex/responses``; a **suffix** match for §7.2's reason — a
#: recorder impersonates a provider, and providers dispatch on the URL.
CODEX_RESPONSES_SUFFIX = "/responses"

#: The suffix that selects an OAuth refresh reply. The refresh leg's one POST —
#: the ``refresh_token`` grant — and its id_token-for-API-key exchange both go to
#: the one endpoint, so one suffix covers the whole leg. **A different constant
#: from T-B1's** ``OAUTH_TOKEN_SUFFIX``: the two suffixes name the same path but
#: belong to different recorders, and swapping the two seams would point one leg
#: at a recorder that cannot see it.
OAUTH_REFRESH_SUFFIX = "/oauth/token"

#: The one wire format this recorder serves. §7.2 assigns it the OpenAI
#: Responses shape; the OAuth refresh leg is answered before any format lookup.
_SERVED_FORMAT = WireFormat.OPENAI_RESPONSES

#: The model and text every minimal success carries. Non-empty deliberately:
#: every emptiness judgement in ``server.py`` keys on content being present, and
#: an empty reply costs the 80-second retry ladder (§7.2.1).
_REPLY_MODEL = "recorder-model"
_REPLY_TEXT = "ok"


def format_for_curl_path(path: str) -> WireFormat | None:
    """Return the wire format a request at ``path`` selects, or ``None``.

    The counterpart of :func:`harness.recorder.format_for_path` for this
    recorder's own suffix table, and pure for the same reason: it answers the
    question and records nothing, so an assertion can use it without mutating
    the evidence it judges.

    Args:
        path: The request's raw path.

    Returns:
        :attr:`~harness.contract.WireFormat.OPENAI_RESPONSES` when ``path``
        names the Codex responses endpoint, and ``None`` when no suffix matches
        — ``None`` rather than a default, because "no rule applies" and "the
        fallback applies" are different facts and only the caller knows which it
        wants.
    """
    if path.endswith(CODEX_RESPONSES_SUFFIX):
        return _SERVED_FORMAT
    return None


def is_oauth_refresh_path(path: str) -> bool:
    """Return whether ``path`` addresses the OAuth refresh endpoint.

    Separate from :func:`format_for_curl_path` for §7.2.2's reason: a token
    exchange has no :class:`~harness.contract.WireFormat` and must not acquire
    one. §3.3.1 pairs every format with a wire reader, and no projection reads a
    form-encoded token grant.

    Args:
        path: The request's raw path.

    Returns:
        Whether the OAuth refresh reply applies.
    """
    return path.endswith(OAUTH_REFRESH_SUFFIX)


def responses_success_body() -> dict[str, Any]:
    """Return the smallest non-streaming OpenAI Responses success.

    The adapter always sends ``stream: true`` (P17), so this shape is only
    reached when a caller streams nothing — but the recorder's reply-format
    lookup is still total, and a body the bridge's reader cannot parse reads as
    an empty response rather than an error.

    Returns:
        The response body, ready to serialize.
    """
    return {
        "id": "resp_recorder",
        "object": "response",
        "model": _REPLY_MODEL,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_recorder",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": _REPLY_TEXT}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


def responses_success_stream() -> tuple[bytes, ...]:
    """Return the smallest valid OpenAI Responses SSE success stream.

    The event sequence follows the grammar ``bridge/responses/events.py`` emits
    inbound and ``ResponsesTranslator`` documents outbound: ``response.created``
    → ``response.in_progress`` → ``response.output_item.added`` →
    ``response.content_part.added`` → ``response.output_text.delta`` →
    ``response.output_text.done`` → ``response.content_part.done`` →
    ``response.output_item.done`` → ``response.completed``. The ``delta`` is
    load-bearing: without it the bridge judges the reply empty and pays the
    ladder.

    Returns:
        The SSE chunks, in order, each already terminated.
    """
    response_id = "resp_recorder"
    text_item_id = "msg_recorder"

    def _event(event_type: str, payload: dict[str, Any]) -> bytes:
        """Encode one SSE frame.

        Args:
            event_type: The SSE event name.
            payload: The data payload, already carrying ``type``.

        Returns:
            The frame, newline-terminated.
        """
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()

    return (
        _event("response.created", {
            "type": "response.created", "sequence_number": 0,
            "response": {"id": response_id, "object": "response", "status": "in_progress",
                         "model": _REPLY_MODEL, "output": [], "usage": None},
        }),
        _event("response.in_progress", {
            "type": "response.in_progress", "sequence_number": 1,
            "response": {"id": response_id, "object": "response", "status": "in_progress",
                         "model": _REPLY_MODEL, "output": [], "usage": None},
        }),
        _event("response.output_item.added", {
            "type": "response.output_item.added", "sequence_number": 2,
            "output_index": 0,
            "item": {"type": "message", "id": text_item_id, "status": "in_progress",
                     "role": "assistant", "content": []},
        }),
        _event("response.content_part.added", {
            "type": "response.content_part.added", "sequence_number": 3,
            "item_id": text_item_id, "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        }),
        _event("response.output_text.delta", {
            "type": "response.output_text.delta", "sequence_number": 4,
            "item_id": text_item_id, "output_index": 0, "content_index": 0,
            "delta": _REPLY_TEXT,
        }),
        _event("response.output_text.done", {
            "type": "response.output_text.done", "sequence_number": 5,
            "item_id": text_item_id, "output_index": 0, "content_index": 0,
            "text": _REPLY_TEXT,
        }),
        _event("response.content_part.done", {
            "type": "response.content_part.done", "sequence_number": 6,
            "item_id": text_item_id, "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": _REPLY_TEXT},
        }),
        _event("response.output_item.done", {
            "type": "response.output_item.done", "sequence_number": 7,
            "output_index": 0,
            "item": {"type": "message", "id": text_item_id, "status": "completed",
                     "role": "assistant",
                     "content": [{"type": "output_text", "text": _REPLY_TEXT}]},
        }),
        _event("response.completed", {
            "type": "response.completed", "sequence_number": 8,
            "response": {"id": response_id, "object": "response", "status": "completed",
                         "model": _REPLY_MODEL,
                         "output": [{"type": "message", "id": text_item_id,
                                     "status": "completed", "role": "assistant",
                                     "content": [{"type": "output_text",
                                                  "text": _REPLY_TEXT}]}],
                         "usage": {"input_tokens": 1, "output_tokens": 1,
                                   "total_tokens": 2}},
        }),
    )


def oauth_refresh_body() -> dict[str, Any]:
    """Return a token response the OAuth refresh leg can read.

    Carries every field :meth:`~kitty.auth.oauth_session.OAuthSession._refresh`
    reads from the refresh grant and every field
    :meth:`~kitty.auth.oauth_session.OAuthSession._exchange_api_key` reads from
    the token exchange — both POSTs address one endpoint, so both are served
    from one body.

    The ``id_token`` is a syntactically valid unsigned JWT with a trivial
    payload — the leg base64-decodes the payload without verifying the signature
    — and is not a credential of any kind.

    Returns:
        The token response, ready to serialize.
    """
    return {
        "access_token": "recorder-access-token",
        "refresh_token": "recorder-refresh-token",
        "id_token": "eyJhbGciOiJub25lIn0.eyJzdWIiOiJyZWNvcmRlciJ9.",
        "expires_in": 3600,
        "openai_api_key": "recorder-api-key",
    }


class CurlLoggingServer(web.Server):
    """A ``web.Server`` that logs every accept at socket level.

    Overriding ``__call__`` is the socket-level seam §7.2.1's limitation
    requires. asyncio's ``_accept_connection2`` calls the protocol factory
    **before** building the SSL transport, so a log written here sees every
    accepted connection — one that then fails the TLS handshake still has a
    record. The base class's ``connection_made`` (T-W4's
    :class:`~harness.recorder._ConnectionLoggingServer` pattern) fires only
    after the handshake, which is exactly the gap this closes; so this subclass
    writes its own and **replaces** T-W4's, whose append would otherwise
    double-record every successfully-handshaken connection.
    """

    def __init__(self, handler: Any, *, recorder: Any, **kwargs: Any) -> None:
        """Bind the server to the recorder it logs into.

        Args:
            handler: The request handler coroutine.
            recorder: The recorder owning the connection log.
            **kwargs: Forwarded to ``web.Server``.
        """
        super().__init__(handler, **kwargs)
        self._recorder = recorder

    def __call__(self) -> Any:
        """Log the accept, then build the request handler asyncio will serve.

        asyncio calls the protocol factory synchronously per accept, before the
        TLS transport exists, so the peer port is not readable here. The record
        is created with ``-1`` and filled in by :meth:`connection_made` when
        the handshake completes — the honest shape §5.2.1 needs, because a
        failed-handshake record still says "a connection was accepted".

        Returns:
            The fresh ``RequestHandler``, already mapped to its connection id.
        """
        handler = super().__call__()
        connection_id = len(self._recorder.connections)
        self._recorder.connections.append(ConnectionRecord(connection_id, -1))
        self._recorder._by_handler[handler] = connection_id
        return handler

    def connection_made(self, handler: Any, transport: Any) -> None:
        """Fill in the peer port once the TLS handshake has completed.

        Args:
            handler: The ``RequestHandler`` serving this connection.
            transport: The connection's transport, whose ``peername`` is the
                same socket's peer the accept-time record was created for.
        """
        peer = transport.get_extra_info("peername")
        connection_id = self._recorder._by_handler.get(handler)
        if connection_id is not None:
            existing = self._recorder.connections[connection_id]
            self._recorder.connections[connection_id] = ConnectionRecord(
                existing.connection_id,
                peer[1] if peer else existing.peer_port,
                existing.requests,
            )
        super().connection_made(handler, transport)


class CurlRecordingUpstream(RecordingUpstream):
    """T-W4's recorder, speaking the curl_cffi vocabulary over TLS.

    Everything the conformance contract judges — what is captured, in what
    order, on which connection, from which peer port — is inherited unchanged,
    so this recorder is judged by the same fourteen checks over the same capture
    path. Three things are overridden: the format it accepts, the suffix table
    it dispatches on, and what it replies with. Two things are added: the TLS
    context it terminates with, and the socket-level connection log.

    A plain subclass rather than a ``@dataclass`` one, following
    :class:`~harness.provider_recorder.ProviderRecordingUpstream`.
    """

    def __init__(self, ssl_context: Any, **kwargs: Any) -> None:
        """Build the recorder with the TLS context it will terminate with.

        Args:
            ssl_context: The server-side ``ssl.SSLContext`` to present. Built
                from the harness ``certs`` fixture through
                :func:`harness.connect_proxy.server_ssl_context`.
            **kwargs: Forwarded to :class:`~harness.recorder.RecordingUpstream`.
        """
        super().__init__(**kwargs)
        self._ssl_context = ssl_context

    @property
    def ssl_context(self) -> Any:
        """Return the TLS context this recorder terminates with.

        Returns:
            The ``ssl.SSLContext`` supplied at construction.
        """
        return self._ssl_context

    @property
    def scheme(self) -> str:
        """Return ``"https"``.

        Returns:
            The scheme this recorder serves. The primary recorder's ``"http"``
            is a constant; here it carries information — a client pointed at
            ``http://`` reaches a TLS listener and fails.
        """
        return "https"

    async def start(self) -> None:
        """Bind an ephemeral loopback port, terminate TLS, and begin accepting.

        Overrides the base only to hand :class:`~aiohttp.web.TCPSite` the TLS
        context; everything else — the runner setup, the ephemeral port read —
        is the base's own sequence.

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
        site = web.TCPSite(self._runner, self.host, 0, ssl_context=self._ssl_context)
        await site.start()
        # Port 0 means the kernel chose one. `BaseRunner.addresses` is the
        # public way to read it back; the socket underneath is not.
        self._port = int(self._runner.addresses[0][1])

    def __post_init__(self) -> None:
        """Reject a format this recorder does not serve.

        The base implementation is replaced, not extended, for §7.2.2's reason.

        Raises:
            ValueError: When ``default_format`` is not
                :attr:`~harness.contract.WireFormat.OPENAI_RESPONSES`.
        """
        if self.default_format is not _SERVED_FORMAT:
            raise ValueError(
                f"the curl_cffi recorder serves {_SERVED_FORMAT.value}, not {self.default_format.value}"
            )

    def _format_for(self, path: str) -> WireFormat:
        """Return the wire format to answer a request at ``path`` in.

        The recording half of the lookup, overridden rather than extended for
        §7.2.2's reason: the two recorders' suffix tables are disjoint.

        Args:
            path: The request's raw path.

        Returns:
            The matched format, or :attr:`default_format` — recording the miss,
            which :meth:`assert_all_paths_matched` turns into a failure at
            teardown.
        """
        matched = format_for_curl_path(path)
        if matched is not None:
            return matched

        self.unmatched.append(path)
        return self.default_format

    async def _default_responder(self, captured: CapturedRequest, response: Reply) -> None:
        """Reply with the minimal success for the request's endpoint.

        Args:
            captured: The recorded request.
            response: The unprepared response to write through.
        """
        # The OAuth leg is answered before any format lookup: it has no
        # `WireFormat` and must not be recorded as an unmatched path, which is
        # what consulting the format table would do.
        if is_oauth_refresh_path(captured.path):
            await self._reply_json(response, oauth_refresh_body())
            return

        # Called for its recording side effect as much as its answer: every
        # other reply this recorder sends is Responses-shaped, but a path that
        # selected none has to be reported at teardown.
        self._format_for(captured.path)

        if _wants_stream(captured.body):
            await response.begin(200, {"Content-Type": "text/event-stream"})
            for chunk in responses_success_stream():
                await response.write(chunk)
            await response.write_eof()
            return

        await self._reply_json(response, responses_success_body())

    @staticmethod
    async def _reply_json(response: Reply, payload: dict[str, Any]) -> None:
        """Send one complete JSON body.

        Args:
            response: The unprepared response.
            payload: The body to serialize.
        """
        encoded = json.dumps(payload).encode()
        # Set the length before preparing: aiohttp would otherwise chunk the
        # reply, and a chunked body is a difference from a real provider that
        # nothing in the harness would explain to the next reader.
        response.content_length = len(encoded)
        await response.begin(200, {"Content-Type": "application/json"})
        await response.write(encoded)
        await response.write_eof()


# Bound after the class is defined, because the server subclass refers to the
# recorder's fields through `recorder=` and the recorder refers to the server
# class through `_server_class`.
CurlRecordingUpstream._server_class = CurlLoggingServer  # type: ignore[assignment]
