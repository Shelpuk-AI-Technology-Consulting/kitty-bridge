"""The proven vertical slice: one request, bridge fixture to recorder to contract.

`.system_design/TEST_SUITE.md` §6.3.1, §7.2.1, §7.5.4, §7.6 · plan task **T-W9**
(KBR-32) · `.requirements/20260912T104328Z_proven_vertical_slice/REQUIREMENTS.md`.

Milestone 0 built three contracts and tested each **alone**: T-W2's capture and
projection types, T-W4's recording upstream, T-W8's bridge fixture. Nothing yet
drove a request through all three. This module does, and it is the first moment
the contracts are known to *compose* rather than merely to exist — six streams
build on that afterwards.

Four claims, plus the three obligations T-W4 and T-W8 could not discharge
because neither starts a real :class:`~kitty.bridge.server.BridgeServer`:

======================================  =========================================
Claim                                   Why only a driven bridge can show it
======================================  =========================================
Exactly one upstream request            A *second* capture is the only real
                                        evidence the empty-response retry ladder
                                        never fired
The capture satisfies T-W2's types      The recorder's output must be what a
                                        projection expects to read
A projection can actually read it       Type-compatible and *readable* are
                                        different claims; only the second
                                        composes
The query string survives               §3.3.5: on Azure the query carries the
                                        API version, so a lost query makes two
                                        routes indistinguishable
``has_content`` (§7.2.1)                A local flag in the streaming loop, not a
                                        callable — unreachable except by driving
                                        a bridge
A wall-clock bound                      80 s of real sleep must fail the suite,
                                        not merely slow it (commit ``691e974``)
======================================  =========================================

**Falsification (plan §1.4).** Two deliberate defects run here: a recorder that
drops the query string, and an upstream reply the bridge judges empty. T-W4
ships its **own** query-drop case against the recorder in isolation
(``DroppedQueryRecorder`` in ``test_recorder_falsification.py``); plan §3 says
the two must not be collapsed, so this one defines its own defect and drives a
real bridge with it. Defining rather than importing follows T-W8, whose
``_BlindRecorder`` is a local copy for the same reason.

**The transports defined here are never registered.** They are constructed
directly and handed to :func:`_drive`, following §7.5.4's own rule that a shared
registry must not hold things that are wrong on purpose. It is also load-bearing
for the gate: ``test_bridge.py``'s meta-test asserts
``set(registered_transports()) == EXPECTED_TRANSPORTS``, and registration happens
at **import time** on module-global state — so a stray ``register_transport``
here would leave this module green on its own and the full suite red, which is
an order-dependent failure and a direct hit on plan §1.3(5).

**Layer.** No ``pytestmark``, so these take the ``l1`` path default, following
T-W4 and T-W8. The reason is §8.2's and only §8.2's: ``l3`` is in
``PENDING_ACTIVATION_LAYERS``, so an ``l3`` marker today would leave the slice
checked by no job at all. §8.2 names this module so T-K6 inherits a list.

**Evidence base.** Every number quoted here was measured on Python 3.12.3. CI
across 3.10–3.13 is the authority; a version-dependent timing failure means
raising :data:`_SLICE_BUDGET_SECONDS`, never skipping a case.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from typing import Any

import pytest
from aiohttp import web

from harness.bridge import (
    AiohttpTransport,
    Binding,
    BridgeFixture,
    InboundProtocol,
    inbound_path,
    marker,
    minimal_inbound_body,
    protocol_for,
)
from harness.contract import (
    CapturedRequest,
    Conversation,
    Envelope,
    Projection,
    Request,
    Text,
    Turn,
    WireFormat,
    verify_total,
)
from harness.recorder import RecordingUpstream, Reply

#: The wall-clock ceiling on one driven request, 200× the healthy time measured
#: against ``main`` (0.01–0.02 s).
#:
#: **What this catches that the capture count does not.** Not "a ladder that
#: sleeps" — an earlier draft claimed that and it is wrong. Any ladder that fires
#: leaves an extra capture, so :func:`_assert_one_upstream_request` sees it
#: first and sees it deterministically. What is left for a wall-clock bound is
#: **slow with a single capture**: a connect grace period, a wedged handler, a
#: non-ladder regression. That is a narrow band and it is a real one.
#:
#: **There are two ladders, not one, and only one of them uses the named
#: delays.** ``_EMPTY_RETRY_DELAYS`` is read at exactly one site,
#: ``_request_with_retry_single`` — the *non-streaming* helper. Every streaming
#: path sleeps ``_BACKOFF_BASE * 2 ** (attempt % 4)``, i.e. 1, 2, 4, 8 s, and
#: reaches ``_EMPTY_FINAL_DELAYS`` only on its last two attempts. So a streaming
#: ladder can fire twice inside this budget. The capture count is what catches
#: that, which is the whole reason both assertions are kept.
#:
#: **Above ten seconds this never fires**, because ``BridgeFixture.post``'s own
#: ``DEFAULT_TIMEOUT`` raises :class:`~harness.bridge.TransportTimeout` first —
#: deliberately left alone, since that error already names the transport, the
#: elapsed time and both ladder costs, which is a better diagnostic than a
#: budget message.
#:
#: Asserted with :func:`time.monotonic`: ``pytest-timeout`` is not in the dev
#: extras and ``asyncio.timeout`` is 3.11+ against a 3.10–3.13 matrix. CI across
#: all four versions is the authority for this number — a version-dependent
#: failure means raising it, never skipping the case.
_SLICE_BUDGET_SECONDS = 4.0

#: The query the driven request carries upstream. Both properties are load-bearing
#: and an earlier draft had neither: ``%20`` would become ``+`` under a
#: ``parse_qsl``/``urlencode`` round trip (``providers/base.py`` says so), and
#: ``dup`` appears **twice** so a recorder that collapses duplicates is
#: distinguishable from one that preserves them. A plain ``a=b`` exercises
#: neither, which is the shape plan §1.4 objects to.
#:
#: **Byte-for-byte survival is specific to these adapters.**
#: ``compose_upstream_url`` *merges* when both sides carry a query and drops base
#: parameters whose name the endpoint also uses; ``custom_anthropic`` and
#: ``custom_openai`` contribute no endpoint query, so here the base query
#: survives verbatim. The same assertion against Azure would be false.
_QUERY = "kbr32=slice%20value&dup=1&dup=2"

#: A percent-encoded prefix on the upstream path, for the same reason the query
#: carries ``%20``. §7.2.1 names ``raw_path`` and ``raw_query_string`` as the two
#: percent-decoding traps, and the adapter's own path (``/v1/messages``) contains
#: nothing encoded — so without this, a recorder reading ``request.path`` instead
#: of ``rel_url.raw_path`` is **indistinguishable** from a correct one. Found by
#: running the mutation sweep against this module and getting nothing red.
#:
#: It survives because ``compose_upstream_url`` prepends the base URL's path
#: component to the endpoint's, and the recorder dispatches its reply format on
#: the path **suffix**, which the prefix leaves intact.
_PATH_PREFIX = "/kbr32%20prefix"

#: The upstream path the driven Anthropic Messages request lands on.
_UPSTREAM_PATH = f"{_PATH_PREFIX}/v1/messages"

#: The two reference costs a budget failure cites, so a reader can tell a fired
#: ladder from a slow machine without a rerun. §7.2.1's own numbers.
_EMPTY_LADDER_SECONDS = 80
_CONNECT_LADDER_SECONDS = 30


@dataclass
class _EncodedRouteTransport(AiohttpTransport):
    """A transport whose binding carries an encoded path prefix and a query.

    The default binding is ``{"base_url": recorder.base_url}``: the capture's
    query is then ``""`` and its path is the adapter's bare ``/v1/messages``.
    Against that, both a query-dropping recorder **and** a path-decoding one pass
    perfectly — measured, and the second was found only by running the mutation
    sweep. A claim is only non-vacuous if the driven request carries the thing
    the claim is about.

    Both travel through the **product's own channel**:
    ``provider_config["base_url"]`` carries them and
    :meth:`~kitty.providers.base.ProviderAdapter.compose_upstream_url` composes
    them with the adapter's endpoint path (the KBR-143 rule). Injecting either
    another way would prove the recorder reads ``raw_path`` and
    ``raw_query_string`` and nothing about whether the bridge preserves them.
    """

    name = "slice-encoded-route"

    def bind(self) -> Binding:
        """Return a binding carrying :data:`_PATH_PREFIX` and :data:`_QUERY`.

        Returns:
            The adapter this transport's format calls for, pointed at the
            recorder with an encoded path prefix and a query string attached.
        """
        adapter, config = super().bind()
        return adapter, {**config, "base_url": f"{config['base_url']}{_PATH_PREFIX}?{_QUERY}"}


class _QueryDroppingRecorder(RecordingUpstream):
    """A recorder that observes the query and keeps none of it.

    :meth:`~harness.recorder.RecordingUpstream.capture` is T-W4's seam for a
    deliberate defect: overriding it disturbs what was captured without touching
    the connection log or the recorded ordering, so the failure is attributable
    to one assertion.
    """

    def capture(self, request: web.BaseRequest, body: bytes, arrival: float) -> CapturedRequest:
        """Return the capture with its query string removed.

        Args:
            request: The inbound aiohttp request.
            body: The entity body, already read.
            arrival: The timestamp taken at handler entry.

        Returns:
            The deliberately wrong capture.
        """
        return replace(super().capture(request, body, arrival), query="")


@dataclass
class _QueryDroppingTransport(_EncodedRouteTransport):
    """Carries a query upstream and records none of it.

    Subclasses the encoded-route transport rather than the plain one: a defect
    that drops something never sent would be undetectable, and a case that
    cannot fail is what plan §1.4 objects to.
    """

    name = "slice-query-dropping"

    def __post_init__(self) -> None:
        """Use the query-dropping recorder instead of the real one."""
        self._recorder = _QueryDroppingRecorder(default_format=self.format, responder=self.responder)


class MessagesReader:
    """The smallest thing shaped like a :class:`~harness.contract.Projection`.

    Deliberately **not** ``harness.reader_anthropic_messages``, which T-A1
    (KBR-33) landed while this task was being written. T-W9's declared
    dependencies are T-W1, T-W2, T-W4 and T-W8 and **no reader task** (§7.5.4),
    and that is worth keeping now that a reader exists: a slice that went red
    because a reader had a bug would mis-attribute the failure, and the claim
    under test is only that a recorder's capture is readable **at all** by
    something shaped like a projection. This is therefore deliberately not a
    ``reader_<format>.py`` module either — §7.4.1 closes that set at **six**,
    one per :class:`~harness.contract.WireFormat` member, and this is evidence
    rather than a seventh projection. How Anthropic Messages *ought* to be
    projected is T-A1's question and is answered in T-A1's own tests.

    ``consumed`` is built from **literal** key names. Deriving it as
    ``frozenset(source)`` would make :func:`~harness.contract.verify_total`
    pass unconditionally, which is the assertion-nothing-can-kill shape. The
    deliberate consequence: if the bridge ever adds a new top-level key to the
    upstream body, this slice goes red — which is the signal worth having, since
    an unregistered addition is exactly what §3.2's register exists to catch.

    Attributes:
        wire_format: The format this reader claims, as the protocol requires.
    """

    wire_format = WireFormat.ANTHROPIC_MESSAGES

    def read_request(self, captured: CapturedRequest) -> Request:
        """Project a captured Anthropic Messages request.

        Args:
            captured: The request as observed on the wire.

        Returns:
            The wire-independent projection, with every top-level body key
            accounted for in ``consumed`` so that
            :func:`~harness.contract.verify_total` is a real check rather than a
            formality.
        """
        body: dict[str, Any] = json.loads(captured.body)

        # One turn per message, each carrying its text. The minimal inbound body
        # sends a plain string, which is the only content shape this must read.
        turns = tuple(Turn(role=m["role"], parts=(Text(m["content"]),)) for m in body["messages"])

        # Sampling keys are closed (§3.3.1b), so only the ones this body can
        # carry are lifted; anything else would raise rather than pass silently.
        sampling = {key: body[key] for key in ("max_tokens", "temperature") if key in body}

        return Request(
            envelope=Envelope(model=body.get("model"), stream=body.get("stream")),
            conversation=Conversation(turns=turns, sampling=sampling),
            consumed=frozenset({"model", "messages", "stream", *sampling}),
            source=body,
        )


async def _empty_reply(captured: CapturedRequest, reply: Reply) -> None:
    """Answer 200 with a well-formed Anthropic Messages body carrying no content.

    Well-formed on purpose. A malformed or unparseable body is a loud failure
    down a different path; what fires the empty-response ladder is a reply the
    bridge parses happily and judges contentless.

    Args:
        captured: The capture, unused.
        reply: The unprepared response.
    """
    payload = json.dumps(
        {
            "id": "msg_empty",
            "type": "message",
            "role": "assistant",
            "model": "recorder-model",
            "content": [],
            "stop_reason": "end_turn",
        }
    ).encode()
    await reply.begin(200, {"content-type": "application/json", "content-length": str(len(payload))})
    await reply.write(payload)


@dataclass(frozen=True)
class _Slice:
    """What one driven request produced.

    A record rather than a tuple because the upstream authority is only readable
    **while the transport is running** — a caller that reads
    ``transport.recorder.port`` after the fixture has torn down gets a
    ``RuntimeError``, which is how this field came to exist.

    Attributes:
        status: The status the bridge answered the agent with.
        text: The raw reply body.
        captures: What reached the upstream, in arrival order.
        sent: The marker this request carried, so a caller cannot assert against
            a different one than it sent.
        authority: The upstream's ``host:port``, read before teardown.
        peer_ports: The peer port of every connection the upstream accepted,
            read before teardown. §5.2.1 joins captures to tunnels on this.
    """

    status: int
    text: str
    captures: list[CapturedRequest]
    sent: str
    authority: str
    peer_ports: set[int | None]


async def _drive(
    subject: AiohttpTransport,
    *,
    stream: bool = False,
    budget: float = _SLICE_BUDGET_SECONDS,
) -> _Slice:
    """Drive one request end to end and return what the upstream saw.

    The wall-clock bound is enforced **here**, so no case in this module can
    silently cost the gate 80 seconds; a separate test proves this function
    enforces it, which is the §7.5.4 pattern — a guard that only proves a check
    exists does not prove anything calls it.

    Args:
        subject: An unstarted transport. Started and stopped by this function.
        stream: Whether to ask the bridge for a streamed reply.
        budget: The wall-clock ceiling on the request, in seconds.

    Returns:
        The record of the driven request.

    Raises:
        AssertionError: When the request outlived ``budget``.
    """
    route = protocol_for(subject.format)
    sent = marker()

    async with BridgeFixture(subject) as fixture:
        # Both read while the transport is still running: `recorder.port` raises
        # once it has stopped, and the connection log is the transport's.
        authority = f"{subject.recorder.host}:{subject.recorder.port}"
        started = time.monotonic()
        status, text = await fixture.post(
            inbound_path(route, stream=stream),
            minimal_inbound_body(route, sent, stream=stream),
        )
        elapsed = time.monotonic() - started
        captures = list(subject.captures)
        peer_ports = {connection.peer_port for connection in subject.connections}

    # The capture count is reported first and is the diagnosis, not decoration:
    # the gate runs ~18.5 minutes on contended runners, so "slow" must be
    # distinguishable from "a ladder fired" without a rerun.
    diagnosis = (
        "exactly one upstream request, so this is a slow runner and not a retry "
        "ladder — raise the budget rather than hunting a regression"
        if len(captures) == 1
        else f"{len(captures)} upstream requests, so a retry ladder fired"
    )
    assert elapsed <= budget, (
        f"the driven slice took {elapsed:.2f}s, over its {budget}s budget, and the "
        f"upstream saw {diagnosis}. Reference costs: the empty-response ladder is "
        f"~{_EMPTY_LADDER_SECONDS}s and an unreachable upstream ~{_CONNECT_LADDER_SECONDS}s"
    )
    return _Slice(
        status=status,
        text=text,
        captures=captures,
        sent=sent,
        authority=authority,
        peer_ports=peer_ports,
    )


def _assert_one_upstream_request(driven: _Slice) -> None:
    """Assert the driven request produced exactly one upstream request.

    A function rather than an inline assertion in each case, because the
    falsification case below must exercise **this** assertion and not a copy of
    it. A copy is an assertion no defect can kill: editing the real one would
    leave the falsification passing against text that no longer ships.

    Args:
        driven: The record of a driven request.

    Raises:
        AssertionError: When the upstream saw any number of requests but one.
    """
    assert len(driven.captures) == 1, (
        f"one inbound request produced {len(driven.captures)} upstream request(s); "
        f"more than one means the empty-response retry ladder fired, and zero means "
        f"the bridge reached some other upstream entirely and every assertion built "
        f"on this fixture would be quantified over nothing"
    )


def _assert_query_survived(driven: _Slice) -> None:
    """Assert the upstream saw the query string the bridge was pointed at.

    Shared with the falsification case for the reason
    :func:`_assert_one_upstream_request` records.

    Args:
        driven: The record of a driven request.

    Raises:
        AssertionError: When the captured query is not the one that was sent.
    """
    assert driven.captures[0].query == _QUERY, (
        f"the upstream saw query {driven.captures[0].query!r}, not {_QUERY!r}; "
        f"§3.3.5 puts routing in the request, so a lost or re-encoded query makes "
        f"two different routes indistinguishable"
    )


# -- R1, R5 — one request in, one request out --------------------------------


class TestOneRequestInOneRequestOut:
    """The product claim T-W8's conformance check deliberately does not make."""

    async def test_a_driven_request_reaches_the_upstream_exactly_once(self) -> None:
        """One inbound request leaves exactly one capture, and the client is served.

        §7.5.4 scopes T-W8's identical-looking assertion to the one binding its
        conformance check drives, because a check tolerating a retry ladder
        could not tell a working binding from a broken one. This is the product
        claim for a driven request; quantifying it over the corpus is T-D1's.
        """
        driven = await _drive(AiohttpTransport(WireFormat.ANTHROPIC_MESSAGES))

        _assert_one_upstream_request(driven)

        # Not redundant with the count, and not falsifiable by anything in this
        # module: §7.5.4's measured row 3 is an upstream 400, which produces one
        # correct capture and a failed client. T-W8's `_RefusingTransport` is the
        # defect that proves this assertion bites; re-shipping it here would
        # duplicate that case rather than add evidence.
        assert driven.status == 200

    async def test_a_chat_completions_stream_never_retries(self) -> None:
        """A streamed request is answered once, which is the ``has_content`` cell.

        §7.2.1 records which judgement guards which reply shape, and every one
        has T-W4-side evidence except this: ``has_content`` is a local flag
        inside ``BridgeServer._stream_chat_completions``, not a callable, so no
        unit test reaches it and T-W4's streams satisfy its precondition by
        construction. A single capture here is the evidence that the flag was
        set and the empty-response retry never fired. (Named by symbol, not by
        line: the plan's ``server.py:5145`` anchor is already stale.)

        This is a claim about one **pair** of axes and not about either alone
        (§7.5.1): inbound ``chat_completions`` over upstream
        ``CHAT_COMPLETIONS``, streaming. ``has_content`` exists only on that
        pass-through loop.
        """
        driven = await _drive(AiohttpTransport(WireFormat.CHAT_COMPLETIONS), stream=True)

        # A second capture here means `has_content` stayed false and the
        # empty-response retry fired on a stream that did carry content.
        #
        # The capture count is the ONLY assertion this case can make. A status
        # check would be unfalsifiable: `_stream_chat_completions` commits the
        # downstream 200 with `sr.prepare()` before it opens the upstream at
        # all, so every outcome on this route is a 200 — measured, a bridge with
        # `has_content` forced false does not answer non-200, it never completes.
        _assert_one_upstream_request(driven)


# -- R2 — the capture satisfies T-W2's declared contract ---------------------


class TestTheCaptureSatisfiesTheContract:
    """Every field T-W2 declares, at its declared type and its wire value."""

    async def test_the_capture_carries_every_contract_field(self) -> None:
        """The seven contract fields each hold their declared type and wire value.

        Type-compatibility is the half of T-W9 that makes the recorder's output
        usable by a projection at all. Asserting the values as well as the types
        is what stops a recorder satisfying this with seven well-typed blanks.
        """
        driven = await _drive(_EncodedRouteTransport(WireFormat.ANTHROPIC_MESSAGES))
        captured = driven.captures[0]

        assert isinstance(captured, CapturedRequest)
        assert captured.method == "POST"

        # The recorder's declared scheme, not something read off the wire — it
        # is a plain-HTTP loopback double. T-B2's TLS transport is where this
        # field starts carrying information.
        assert captured.scheme == "http"
        # The Host header the bridge sent, never `request.host`, which invents
        # the build machine's name when the header is absent (§7.2.1).
        assert captured.host == driven.authority
        # Read from `rel_url.raw_path`, never `request.path`: the prefix carries
        # a `%20`, so a percent-decoding recorder is distinguishable here.
        assert captured.path == _UPSTREAM_PATH
        assert captured.query == _QUERY
        assert isinstance(captured.body, bytes)
        assert driven.sent.encode() in captured.body

        # A non-empty sequence of `(str, str)` pairs and never a mapping, which
        # is the shape §4.3 C1's exact-header-set assertion needs. Whether
        # duplicates and casing survive is T-W4's conformance check's claim, not
        # this one's: the bridge sends no duplicate header on this route, so a
        # duplicate assertion here would be untestable.
        assert captured.headers
        assert all(isinstance(name, str) and isinstance(value, str) for name, value in captured.headers)

    async def test_the_capture_carries_the_connection_join_key(self) -> None:
        """``arrival`` and ``peer_port`` are populated and correctly typed.

        They are T-W4's to populate and §5.2.1's to consume — the join key
        against the proxy's tunnel log. No fidelity assertion reads them, so a
        capture that silently stopped carrying them would otherwise go unnoticed
        until T-E2 tried to join on a column of ``None``.

        The port is asserted to **join**, not merely to be an ``int``: a type
        check passes for a port matching no connection, which is the column T-E2
        would then be joining against nothing.
        """
        driven = await _drive(AiohttpTransport(WireFormat.ANTHROPIC_MESSAGES))
        captured = driven.captures[0]

        assert isinstance(captured.arrival, float)
        assert captured.peer_port in driven.peer_ports, (
            f"the capture's peer port {captured.peer_port!r} matches none of the "
            f"upstream's accepted connections {sorted(driven.peer_ports, key=str)}; "
            f"§5.2.1 joins captures to tunnels on exactly this key"
        )


# -- R3 — the capture is readable, not merely well-typed ---------------------


class TestTheCaptureIsReadable:
    """The composition claim: a projection can consume what the recorder made."""

    async def test_a_projection_reads_the_capture_totally(self) -> None:
        """A minimal reader projects the capture with nothing dropped or residual.

        Type-compatible and *readable* are different claims and only the second
        composes. :func:`~harness.contract.verify_total` is what makes this a
        real check: a reader that silently ignored a body key would leave the
        residual empty, so "the residual is empty" would pass it.
        """
        reader = MessagesReader()
        driven = await _drive(AiohttpTransport(WireFormat.ANTHROPIC_MESSAGES))

        # Member *presence* only — `contract.py` says so itself, and a protocol
        # `isinstance` never checks signatures. The real evidence is the two
        # lines below: the capture is read, and read totally.
        assert isinstance(reader, Projection)

        projected = reader.read_request(driven.captures[0])
        verify_total(projected)

        # The user's own text survived the whole slice, which is what makes this
        # a fidelity claim rather than a parse that happened not to raise.
        assert [part.text for turn in projected.conversation.turns for part in turn.parts] == [driven.sent]


# -- R4 — the query string survives ------------------------------------------


class TestTheQueryStringSurvives:
    """§3.3.5: routing lives in the request, and the body cannot show it."""

    async def test_the_query_string_survives_the_slice(self) -> None:
        """The capture's query is what the bridge sent, encoding and order intact.

        Read from ``raw_query_string`` (§7.2.1), so the percent-encoding and the
        duplicate parameter name in :data:`_QUERY` both survive. On Azure the
        query carries the API version while the path carries the deployment, and
        P6 removes ``model`` from the body — so two different routes with a lost
        query are indistinguishable.
        """
        driven = await _drive(_EncodedRouteTransport(WireFormat.ANTHROPIC_MESSAGES))

        _assert_query_survived(driven)


# -- R6 — the wall-clock bound -----------------------------------------------


class TestTheWallClockBound:
    """The bound is enforced on every driven request, not merely defined."""

    async def test_the_budget_is_enforced_on_every_driven_request(self) -> None:
        """A request over budget fails, and the failure names the ladder costs.

        This is the §7.5.4 pattern rather than a unit test on a comparison: plan
        §1.4's own list of past harness failures includes "a guard proving a
        function was *called* when the enforcement was the branch after it". A
        budget of zero is over-budget for any real request, so what this proves
        is that :func:`_drive` — the one path every case in this module takes —
        actually applies it.
        """
        with pytest.raises(AssertionError) as excinfo:
            await _drive(AiohttpTransport(WireFormat.ANTHROPIC_MESSAGES), budget=0.0)

        message = str(excinfo.value)
        assert "budget" in message
        assert str(_EMPTY_LADDER_SECONDS) in message


# -- R7 — falsification (plan §1.4) ------------------------------------------


class TestTheDefectsAreCaught:
    """Two deliberate defects, each of which the slice must detect."""

    async def test_a_query_dropping_recorder_fails_the_slice(self) -> None:
        """A recorder that drops the query fails the end-to-end query claim.

        T-W4 ships its own query-drop case against the recorder **in isolation**
        (``DroppedQueryRecorder``). Plan §3 says the two must not be collapsed:
        that one proves the recorder reads the field, this one proves the claim
        survives a real bridge in front of it — which is the only version that
        could catch the bridge dropping the query before the recorder ever saw
        it.
        """
        driven = await _drive(_QueryDroppingTransport(WireFormat.ANTHROPIC_MESSAGES))

        # The defect must have taken effect, or this case proves nothing.
        assert driven.captures[0].query == ""

        # The **shipped** assertion, run against the defect — not a copy of it.
        with pytest.raises(AssertionError) as excinfo:
            _assert_query_survived(driven)

        assert _QUERY in str(excinfo.value)

    async def test_a_fired_ladder_leaves_more_than_one_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty upstream reply fires the ladder, breaking the one-capture claim.

        Without this, "exactly one capture" could be an assertion that passes by
        construction. The delays are flattened rather than waited out: the ladder
        is 80 seconds of real sleep, and a falsification case that cost the gate
        80 seconds is the defect commit ``691e974`` fixed once already. What is
        under test is that the ladder **fired**, which the capture count shows
        and the sleeping does not.

        This drives R1's own pair — inbound ``messages`` over upstream
        ``ANTHROPIC_MESSAGES``, non-streaming — which is what makes it a
        falsification of R1 rather than of some neighbouring path. That route
        reaches ``_request_with_retry_single``, the one site that reads
        ``_EMPTY_RETRY_DELAYS``, which is why patching those two lists bites at
        all. Measured with the delays flattened: five captures in 0.008 s.

        **Values flattened, lengths unchanged.** ``max_attempts`` is computed as
        ``len(_EMPTY_RETRY_DELAYS) + len(_EMPTY_FINAL_DELAYS) + 1``, so an empty
        list would collapse the ladder to a single attempt and the defect would
        stop being a defect.

        Args:
            monkeypatch: Pytest's monkeypatch fixture, which reverts the delays.
        """
        monkeypatch.setattr("kitty.bridge.server._EMPTY_RETRY_DELAYS", [0.0, 0.0])
        monkeypatch.setattr("kitty.bridge.server._EMPTY_FINAL_DELAYS", [0.0, 0.0])

        subject = AiohttpTransport(WireFormat.ANTHROPIC_MESSAGES, responder=_empty_reply)
        driven = await _drive(subject)

        # The defect must have taken effect, or this case proves nothing.
        assert len(driven.captures) > 1, (
            "an empty upstream reply did not fire the retry ladder, so the "
            "one-capture assertion is not known to detect one"
        )

        # The **shipped** assertion, run against the defect — not a copy of it.
        with pytest.raises(AssertionError) as excinfo:
            _assert_one_upstream_request(driven)

        assert str(len(driven.captures)) in str(excinfo.value)


# -- The inbound routes this module drives -----------------------------------


class TestTheDrivenRoutes:
    """Pins which inbound route each format is driven through."""

    def test_each_driven_format_uses_its_matching_route(self) -> None:
        """The two formats this module drives map to the routes it assumes.

        §7.5.1: inbound protocol and upstream format are independent axes, so
        the pairing is a convenience and not an equivalence. Pinned here because
        the path and body assertions above are written against these two routes
        and would become confusing rather than red if the mapping changed.
        """
        assert protocol_for(WireFormat.ANTHROPIC_MESSAGES) is InboundProtocol.MESSAGES
        assert protocol_for(WireFormat.CHAT_COMPLETIONS) is InboundProtocol.CHAT_COMPLETIONS
