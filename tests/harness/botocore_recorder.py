"""The botocore recording upstream.

`.system_design/TEST_SUITE.md` §3.2.3, §7.2, §7.2.3 · plan task **T-B3** (KBR-42).

§7.2 gives each of the bridge's five client configurations its own recorder.
The primary one (T-W4, :mod:`harness.recorder`) observes the sessions
:meth:`BridgeServer._session_for` owns, which is every default-transport
adapter; :mod:`harness.provider_recorder` observes ``ollama_cloud`` and the
OpenAI OAuth login leg. This one observes ``bedrock``, whose traffic does not
run through aiohttp at all: the adapter builds a boto3 client per request and
botocore owns the HTTP stack.

**Why an endpoint override and not a redirected base URL.**
``BedrockAdapter`` never calls ``build_base_url`` — §7.5.2 records it as one
of three custom-transport adapters the product's own redirection channel does
not reach. The product seam is ``provider_config["endpoint_url"]``, consumed
by ``_get_boto3_client`` and passed as ``endpoint_url=`` to
``session.client(...)``; :mod:`harness.botocore` is the transport that
supplies it.

**Why this subclasses T-W4's recorder rather than being a second server.**
§7.2.1 catalogues six ways an aiohttp recorder can look correct and lie; a
second implementation is a second chance to get each of them wrong. What
differs is vocabulary only — the format served, the suffix table that
dispatches the reply, and what a minimal success looks like — so this
subclass overrides those three things and nothing else. The capture path
(including ``peer_port``) is inherited unchanged.

**Why the streaming reply is AWS EventStream binary, not SSE.** Bedrock's
``converse_stream`` returns ``application/vnd.amazon.eventstream``: a stream
of length-prefixed frames, each carrying ``:message-type`` and ``:event-type``
headers plus a JSON payload, and closed by a CRC32. The pinned
``botocore.eventstream`` module has a parser but **no public serializer**
(verified against the installed source), so the harness ships its own
:func:`encode_eventstream` following the spec every AWS SDK implements, and
:func:`tests.harness.test_botocore` validates it by round-trip through the
pinned parser and through a real ``boto3.client(...).converse_stream(...)``
call.

**It imports nothing from ``src/kitty``, and must not**, for the reason
:mod:`harness.recorder` states: §3.3.1's independent-oracle rule. The product
import this delivery needs lives in :mod:`harness.botocore`, which binds an
adapter; nothing here asks kitty how to read a request. ``import boto3`` /
``import botocore`` are allowed — they are the provider family, not the
bridge.
"""

from __future__ import annotations

import json
import struct
from binascii import crc32
from typing import Any

from harness.contract import CapturedRequest, WireFormat

# No `_wants_stream` import: the Bedrock transport pops ``stream`` from
# the Converse payload before the HTTP request is built (register row
# **P18**), so the request body never carries that key — the stream
# decision has to come from the path suffix, not the body.
from harness.recorder import RecordingUpstream, Reply

__all__ = [
    "BedrockRecordingUpstream",
    "BEDROCK_CONVERSE_SUFFIX",
    "BEDROCK_CONVERSE_STREAM_SUFFIX",
    "format_for_bedrock_path",
    "bedrock_success_body",
    "bedrock_success_stream_events",
    "encode_eventstream",
    "encode_eventstream_frame",
]

#: The suffix that selects a non-streaming Converse reply. The Bedrock
#: service model (``botocore/data/bedrock-runtime/2023-09-30``) gives
#: ``Converse`` the request URI ``/model/{modelId}/converse``; a **suffix**
#: match survives any model id and any account-scoped prefix the way §7.2's
#: other recorders survive Azure's deployment path.
BEDROCK_CONVERSE_SUFFIX = "/converse"

#: The suffix that selects a ``ConverseStream`` reply —
#: ``/model/{modelId}/converse-stream`` in the service model. Distinct from
#: :data:`BEDROCK_CONVERSE_SUFFIX` because the two replies have nothing in
#: common: one is a complete JSON body, the other is EventStream binary.
BEDROCK_CONVERSE_STREAM_SUFFIX = "/converse-stream"

#: The one wire format this recorder serves. §7.2 assigns it the Bedrock
#: Converse shape; Ollama ``/api/chat`` is T-B1's.
_SERVED_FORMAT = WireFormat.BEDROCK_CONVERSE

#: The text every minimal success carries. Non-empty deliberately: every
#: emptiness judgement in ``server.py`` keys on content being present, and
#: an empty reply costs the 80-second retry ladder (§7.2.1).
_REPLY_TEXT = "ok"

#: Model id the replies name. Present in the JSON the adapter reads, but the
#: identifier itself never travels in the request body (P18 pops it); naming
#: one here keeps a reply readable by ``translate_from_upstream``.
_REPLY_MODEL = "recorder-model"

#: EventStream header names. Spelled out once: ``:message-type`` selects the
#: event vs exception branch in botocore's ``BaseEventStreamParser`` and
#: ``:event-type`` names the member the payload is parsed against; a typo in
#: either silently yields an unparseable stream rather than an error.
_MESSAGE_TYPE_HEADER = ":message-type"
_EVENT_TYPE_HEADER = ":event-type"
_CONTENT_TYPE_HEADER = ":content-type"
_EVENT_MESSAGE_TYPE = "event"
_EVENT_CONTENT_TYPE = "application/json"

#: Wire type codes of the EventStream header encoding (smithy spec; the
#: pinned botocore ``EventStreamHeaderParser._HEADER_TYPE_MAP`` is the
#: authority for the pinned version).
_HEADER_TYPE_STRING = 7


def format_for_bedrock_path(path: str) -> WireFormat | None:
    """Return the wire format a request at ``path`` selects, or ``None``.

    The counterpart of :func:`harness.recorder.format_for_path` for this
    recorder's own suffix table, and pure for the same reason: it answers the
    question and records nothing, so an assertion can use it without mutating
    the evidence it judges.

    Args:
        path: The request's raw path.

    Returns:
        :attr:`~harness.contract.WireFormat.BEDROCK_CONVERSE` when ``path``
        names one of the two Converse endpoints, and ``None`` when no suffix
        matches — ``None`` rather than a default, because "no rule applies"
        and "the fallback applies" are different facts and only the caller
        knows which it wants.
    """
    if path.endswith(BEDROCK_CONVERSE_STREAM_SUFFIX):
        return _SERVED_FORMAT
    if path.endswith(BEDROCK_CONVERSE_SUFFIX):
        return _SERVED_FORMAT
    return None


def bedrock_success_body() -> dict[str, Any]:
    """Return the smallest non-streaming Converse success.

    "Minimal" means the fewest keys that still read as a success to the
    bridge, not a faithful sample of a provider's response — the same rule
    :func:`harness.recorder.minimal_success_body` states. The keys are the
    ones the ``bedrock-runtime`` service model gives ``ConverseOutput`` and
    the ones ``BedrockAdapter.translate_from_upstream`` reads: an ``output``
    message with at least one ``text`` block, a ``stopReason`` and a
    ``usage`` pair.

    Returns:
        The response body, ready to serialize.
    """
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": _REPLY_TEXT}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }


def bedrock_success_stream_events() -> tuple[dict[str, Any], ...]:
    """Return the events of the smallest streamed Converse success.

    Each dict is one EventStream frame's payload keyed by its ``:event-type``
    member — the shape ``BedrockAdapter._translate_stream_event`` switches on.
    The sequence is the minimum that makes the adapter emit content **and** a
    finish reason: ``messageStart`` opens the turn, one ``contentBlockDelta``
    carries the text, ``contentBlockStop`` closes the block, and
    ``messageStop`` is the only event that produces a CC ``finish_reason`` —
    without it the stream ends mid-turn and the bridge judges the reply
    empty.

    Returns:
        The events, in wire order.
    """
    return (
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": _REPLY_TEXT}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
    )


def encode_eventstream_frame(event_type: str, payload: dict[str, Any]) -> bytes:
    """Encode one EventStream frame.

    Frame layout (smithy spec; pinned botocore ``DecodeUtils.unpack_prelude``
    reads it back): ``[total_length u32][headers_length u32][prelude_crc
    u32][headers][payload][message_crc u32]``, all big-endian. The prelude
    CRC covers the first eight bytes; the message CRC covers everything
    before it, and both are the standard ``binascii.crc32`` masked to 32
    bits — **not** CRC32C, which is a different polynomial and the one
    mistake this encoder could make invisibly.

    Headers carry ``:message-type: event``, ``:event-type:`` the member name
    and ``:content-type: application/json`` — the three botocore's
    ``BaseEventStreamParser`` consults, in the type-7 string encoding
    (``[name_len u8][name][type u8][value_len u16][value]``).

    Args:
        event_type: The ``:event-type`` member name, e.g. ``messageStop``.
        payload: The JSON body of the event.

    Returns:
        The complete frame bytes.
    """
    # Headers first: their length feeds the prelude, so the payload cannot be
    # laid out until they are.
    headers = b"".join(
        _encode_header(name, value)
        for name, value in (
            (_MESSAGE_TYPE_HEADER, _EVENT_MESSAGE_TYPE),
            (_EVENT_TYPE_HEADER, event_type),
            (_CONTENT_TYPE_HEADER, _EVENT_CONTENT_TYPE),
        )
    )
    payload_bytes = json.dumps(payload).encode()

    # `total` counts everything including the trailing message CRC, which is
    # why the +4 sits outside the header+payload sum.
    total = 12 + len(headers) + len(payload_bytes) + 4
    prelude = struct.pack("!III", total, len(headers), crc32(struct.pack("!II", total, len(headers))) & 0xFFFFFFFF)

    # The message CRC runs over prelude + headers + payload — the frame
    # length minus its own four bytes.
    body = prelude + headers + payload_bytes
    return body + struct.pack("!I", crc32(body) & 0xFFFFFFFF)


def encode_eventstream(events: tuple[dict[str, Any], ...]) -> bytes:
    """Encode a sequence of Converse stream events as EventStream binary.

    Args:
        events: One dict per frame, keyed by member name — the shape
            :func:`bedrock_success_stream_events` returns.

    Returns:
        The concatenated frames, ready to write as one
        ``application/vnd.amazon.eventstream`` body.
    """
    frames: list[bytes] = []
    for event in events:
        # One key per event dict by construction; the key *is* the
        # :event-type. A dict with zero or several keys is a caller defect,
        # and `next(iter(...))` on an empty dict fails loudly here rather
        # than emitting a frame botocore answers with a KeyError three
        # layers away.
        (event_type, payload), = event.items()
        frames.append(encode_eventstream_frame(event_type, payload))
    return b"".join(frames)


def _encode_header(name: str, value: str) -> bytes:
    """Encode one EventStream header in the type-7 string form.

    Args:
        name: The header name.
        value: The header value.

    Returns:
        The encoded header bytes.
    """
    name_bytes = name.encode()
    value_bytes = value.encode()
    return (
        struct.pack("!B", len(name_bytes))
        + name_bytes
        + struct.pack("!B", _HEADER_TYPE_STRING)
        + struct.pack("!H", len(value_bytes))
        + value_bytes
    )


class BedrockRecordingUpstream(RecordingUpstream):
    """T-W4's recorder, speaking the Bedrock Converse vocabulary.

    Everything the conformance contract judges — what is captured, in what
    order, on which connection, from which peer port — is inherited unchanged,
    so this recorder is judged by the same fourteen checks and cannot drift
    from the primary one on any of them. Three things are overridden: the
    format it accepts, the suffix table it dispatches on, and what it replies
    with.

    A plain subclass rather than a ``@dataclass`` one, because it adds no
    fields; :class:`~harness.provider_recorder.ProviderRecordingUpstream` is
    the same shape.
    """

    def __post_init__(self) -> None:
        """Reject a format this recorder does not serve.

        The base implementation is **replaced, not extended**: it is validation
        and nothing else today, and it would reject every format this recorder
        serves. ``start()`` builds the server and already sets
        ``auto_decompress=False``, so the capture path needs no override here —
        if T-W4 ever adds setup to its ``__post_init__`` beyond the
        validation, this override has to call it, which is the one reason to
        re-read this docstring.

        Raises:
            ValueError: When ``default_format`` is not
                :attr:`~harness.contract.WireFormat.BEDROCK_CONVERSE`. The
                base class would reject it too, but for the wrong reason and
                with the wrong message — it names the primary recorder's two
                formats, neither of which this one serves.
        """
        if self.default_format is not _SERVED_FORMAT:
            raise ValueError(
                f"the botocore recorder serves {_SERVED_FORMAT.value}, not {self.default_format.value}"
            )

    def _format_for(self, path: str) -> WireFormat:
        """Return the wire format to answer a request at ``path`` in.

        The recording half of the lookup, overridden rather than extended
        because the three recorders' suffix tables are disjoint: none serves
        a format another does, so a shared table would only ever answer a
        request one of them could not reply to. **Called from
        :meth:`_default_responder`** for the recording side effect, not for
        its return value (the responder dispatches on the path suffix
        directly, since P18 pops ``stream`` from the body).

        Args:
            path: The request's raw path.

        Returns:
            The matched format, or :attr:`default_format` — recording the
            miss, which :meth:`assert_all_paths_matched` turns into a failure
            at teardown.
        """
        matched = format_for_bedrock_path(path)
        if matched is not None:
            return matched

        self.unmatched.append(path)
        return self.default_format

    async def _default_responder(self, captured: CapturedRequest, response: Reply) -> None:
        """Reply with the minimal success for the request's endpoint.

        **The stream decision is made on the path, never on the body.** The
        other recorders read ``"stream": true`` out of the request body —
        but the Bedrock transport pops that key from the Converse payload
        before the HTTP request is even built (register row **P18**, whose
        observable effect is exactly this body): ``modelId`` becomes a call
        argument and ``stream`` becomes the choice between ``/converse`` and
        ``/converse-stream``. A body-based check would therefore always see
        no ``stream`` key, always answer JSON, and a streaming request would
        parse the JSON as an EventStream frame — a ``ChecksumMismatch`` deep
        inside boto3 (measured), which reads as a broken recorder rather
        than a broken dispatch rule.

        Args:
            captured: The recorded request.
            response: The unprepared response to write through.
        """
        # Called for its recording side effect as much as its answer: every
        # reply this recorder sends is Converse-shaped, but a path that
        # selected none has to be reported at teardown. The override on
        # ``_format_for`` does that recording; calling it here (rather than
        # re-implementing the lookup and the unmatched-path append) is what
        # keeps the recording half of the dispatch in one place — the same
        # recipe ``ProviderRecordingUpstream`` follows, and the reason this
        # subclass mirrors the override's signature.
        self._format_for(captured.path)

        if captured.path.endswith(BEDROCK_CONVERSE_STREAM_SUFFIX):
            encoded = encode_eventstream(bedrock_success_stream_events())
            # ``StreamResponse.force_close`` is a **method** that flips
            # ``_keep_alive`` to False (verified in aiohttp 3.13.5
            # ``web_response.py``); a no-arg call sets it. Without it
            # aiohttp's HTTP/1.1 keep-alive keeps the socket open and
            # botocore's ``StreamingBody`` waits for more frames until its
            # 60-second read timeout fires — the stdlib HTTP server used in
            # early prototyping was HTTP/1.0 and closed after the response,
            # which is why the shape never surfaced until this recorder met
            # a real client.
            response.force_close()
            await response.begin(200, {"Content-Type": "application/vnd.amazon.eventstream"})
            await response.write(encoded)
            await response.write_eof()
            return

        await self._reply_json(response, bedrock_success_body())

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
