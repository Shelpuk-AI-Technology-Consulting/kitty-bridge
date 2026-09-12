"""The bridge fixture: a real ``BridgeServer`` in front of a recording upstream.

`.system_design/TEST_SUITE.md` §7.5 · plan task **T-W8** (KBR-31).

§7.2 says what a recording upstream must *observe*; this module says how the
product is put in front of one. Of the 62 test modules under ``tests/bridge/``,
38 start a real :class:`~kitty.bridge.server.BridgeServer` and most build their
own stub adapter, fake upstream and ``post()`` helper; around fifteen planned
tasks across Epics D, E, G, I, J and K need one shared version, and three more
(T-B1–T-B3) need to plug a **different transport** into it.

**This is the one module in this package that imports the product, and it must.**
``contract.py`` and ``recorder.py`` each carry a structural guard forbidding any
``kitty`` import, because §3.3.1's independent-oracle rule says a reader that
asked kitty how to parse a body would inherit kitty's bugs. That governs what
*judges* a request. This module *starts* the thing under test. Its own guard is a
different claim — that it registers no Epic B transport — and lives in
``test_bridge.py`` beside :data:`REGISTERED_HERE`.

**No pytest fixture, and not in ``pytest_plugins``.** T-W5's proxy is published
that way because a test asks for ``connect_proxy`` by name and cannot construct
one. This module is different on both counts: a test must choose a wire format
and a bridge shape anyway, so ``BridgeFixture(transport("aiohttp", fmt))`` says
more than a fixture name would — and importing this module pulls in
``kitty.bridge.server``, measured at **0.72 s**, which a global plugin would
charge to every pytest invocation including those that never start a bridge.

**The failure this module exists to prevent.** A bridge pointed at the *wrong
live* upstream answers **200**; the test sees success and this fixture's recorder
holds **nothing**. Every assertion built on it — "no unclaimed delta" (§3.3),
"zero unattributable connections" (§5.2.1), every §6.3.1 lifecycle claim — is
then quantified over an empty list and passes vacuously. That is plan §1.4's own
shape, so :func:`assert_transport_reaches_its_recorder` makes the wiring a
checkable claim, and ``test_bridge_falsification.py`` ships the four defects it
must catch.
"""

from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import pytest

from harness.contract import CapturedRequest, WireFormat
from harness.recorder import ConnectionRecord, RecordingUpstream, Responder, format_for_path
from kitty.bridge.server import BridgeServer
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.providers.custom_anthropic import CustomAnthropicAdapter
from kitty.providers.custom_openai import CustomOpenAIAdapter

__all__ = [
    "Backend",
    "Binding",
    "BridgeFixture",
    "InboundProtocol",
    "MisdeclaredFormatError",
    "REGISTERED_HERE",
    "TransportTimeout",
    "UpstreamTransport",
    "AiohttpTransport",
    "assert_transport_reaches_its_recorder",
    "backend_for",
    "inbound_path",
    "marker",
    "minimal_inbound_body",
    "pin_backend_order",
    "profile_for",
    "protocol_for",
    "redirected",
    "register_transport",
    "registered_transports",
    "transport",
]

#: The transports **this module** registers. Pinned by ``test_bridge.py`` against
#: this constant and never against :func:`registered_transports`, whose contents
#: depend on which modules the pytest session imported and which grows the moment
#: an Epic B module is imported anywhere in the run.
REGISTERED_HERE: tuple[str, ...] = ("aiohttp",)

#: Client timeout for :meth:`BridgeFixture.post`. aiohttp's own default is
#: ``total=300``, so a mis-wired binding would cost five minutes per test — and a
#: closed upstream port was measured at 30 seconds even so. A test that means to
#: observe the 80-second empty-response ladder (§7.2.1) raises this deliberately.
DEFAULT_TIMEOUT = 10.0

#: The two elapsed times a timeout message cites, so a reader knows which failure
#: they are probably looking at rather than guessing from a bare cancellation.
_CONNECT_LADDER_SECONDS = 30
_EMPTY_LADDER_SECONDS = 80

#: A valid UUIDv4. ``tests/conftest.py``'s `sample_profile_dict` carries a UUIDv7
#: and therefore cannot build a ``Profile`` at all (KBR-175).
_AUTH_REF = "6f1d4f3e-2b9a-4c1d-8f7e-5a2b3c4d5e6f"

#: The resolved key every fixture profile carries. Never a real credential shape.
_KEY = "harness-key"

#: The model a fixture uses when the caller names none.
MODEL = "harness-model"

#: Which adapter serves which upstream format. Both read
#: ``provider_config["base_url"]``, so the default transport uses the product's
#: own redirection channel and needs no subclass — :func:`redirected` exists for
#: the ~14 adapters that do not (§7.5.2).
_ADAPTER_FOR_FORMAT: dict[WireFormat, type[ProviderAdapter]] = {
    WireFormat.ANTHROPIC_MESSAGES: CustomAnthropicAdapter,
    WireFormat.CHAT_COMPLETIONS: CustomOpenAIAdapter,
}


class MisdeclaredFormatError(AssertionError):
    """Raised when a transport answered in a format it does not declare.

    §7.2's recorder chooses its reply format by path **suffix** and only records
    a miss when no suffix matches. So a transport whose bound adapter posts to
    another format's path is answered correctly, the adapter parses the reply,
    and nothing is recorded as unmatched — the declared format was never the
    thing under test. This is the only signal that says so.
    """


class TransportTimeout(AssertionError):
    """Raised when an inbound request outlived :data:`DEFAULT_TIMEOUT`.

    An ``AssertionError`` rather than a bare ``TimeoutError`` so the failure
    carries the transport, the elapsed time and the captures so far. A
    cancellation on its own says only that something took too long, which is the
    one thing the reader already knows.
    """


class InboundProtocol(Enum):
    """The bridge routes an agent posts to.

    Deliberately **not** :class:`~harness.contract.WireFormat`, which describes
    what a recorder serves. The two axes are independent — translating between
    them is what the bridge does and what §3.3's oracle exists to check — and
    two of the six wire formats have no inbound route at all. One parameter for
    both would work only for the pass-through pairs T-W8 happens to ship.
    """

    CHAT_COMPLETIONS = "chat_completions"
    MESSAGES = "messages"
    RESPONSES = "responses"
    GEMINI = "gemini"


#: The inbound route whose dialect matches an upstream format, for helpers that
#: need a sensible default. **Not** an equivalence: §7.5.1 — the two axes are
#: independent, and translating between them is what the bridge is.
_PROTOCOL_FOR_FORMAT: dict[WireFormat, InboundProtocol] = {
    WireFormat.ANTHROPIC_MESSAGES: InboundProtocol.MESSAGES,
    WireFormat.CHAT_COMPLETIONS: InboundProtocol.CHAT_COMPLETIONS,
    WireFormat.OPENAI_RESPONSES: InboundProtocol.RESPONSES,
    WireFormat.GEMINI: InboundProtocol.GEMINI,
}

#: What a transport hands the fixture: the adapter to run, and the provider
#: configuration that points it at that transport's own recorder.
Binding = tuple[ProviderAdapter, dict[str, Any]]

#: One member of a balancing pool, in the shape ``BridgeServer(backends=…)`` takes.
Backend = tuple[ProviderAdapter, str, Profile]


@runtime_checkable
class UpstreamTransport(Protocol):
    """What the bridge fixture needs from a recording upstream it did not write.

    T-B1–T-B3 satisfy this and call :func:`register_transport`; nothing in this
    module needs to change for them. Attributes rather than a base class so a
    transport can be any object, including one wrapping a client library that
    owns its own server.

    Attributes:
        name: The registry key.
        format: The **upstream** wire format this transport serves.
    """

    name: str
    format: WireFormat

    async def start(self) -> None:
        """Begin accepting, on a port of the transport's choosing."""

    async def stop(self) -> None:
        """Stop accepting and release the port."""

    def bind(self) -> Binding:
        """Return the adapter and provider config that reach this transport.

        Returns:
            The pair a bridge is constructed with. This is the single seam: the
            fixture never reads a recorder's base URL itself, because botocore
            takes an endpoint override, curl_cffi terminates TLS, and the OAuth
            leg is a different session entirely (§7.2).
        """

    @property
    def captures(self) -> Sequence[CapturedRequest]:
        """Return what reached this transport, in arrival order."""

    @property
    def connections(self) -> Sequence[ConnectionRecord]:
        """Return every accepted connection.

        §5.2.1's bypass is a connection carrying **no** request, which no
        capture list can express, and `connect_proxy.unattributable_peer_ports`
        joins on exactly these ports.
        """

    def assert_teardown_clean(self) -> None:
        """Assert every request was answered in the format this transport declares.

        How a transport knows that is its own business — none of the other three
        can consult :class:`~harness.recorder.RecordingUpstream`'s suffix table.

        Raises:
            AssertionError: When some request was not.
        """


#: Builds a transport for one upstream format. ``responder`` is T-W4's
#: failure-library seam (T-B4, T-I7, T-I8, T-I11, T-G7): scripted replies that
#: change between attempts are **stateful closures** over it, which is why no
#: setter is added to a Protocol fifteen tickets consume.
TransportFactory = Callable[..., UpstreamTransport]

_TRANSPORTS: dict[str, TransportFactory] = {}


def register_transport(name: str, factory: TransportFactory) -> None:
    """Register a transport under ``name``.

    Args:
        name: The registry key, e.g. ``"curl_cffi"``.
        factory: Called as ``factory(fmt, responder=…)``.

    Raises:
        ValueError: When ``name`` is already registered. Silently replacing it
            would let two Epic B modules disagree about which recorder a name
            means, and the loser would be whichever imported first.
    """
    if name in _TRANSPORTS:
        raise ValueError(f"transport {name!r} is already registered; registered: {registered_transports()}")

    _TRANSPORTS[name] = factory


def registered_transports() -> tuple[str, ...]:
    """Return every registered transport name, sorted.

    Returns:
        The names registered **in this process**, which is a function of which
        modules were imported. A guard pinning what one module registers must
        read :data:`REGISTERED_HERE` instead; a meta-test iterating this must
        import its subjects explicitly and assert set equality, or it reports
        success for a category it never ran.
    """
    return tuple(sorted(_TRANSPORTS))


def transport(name: str, fmt: WireFormat, *, responder: Responder | None = None) -> UpstreamTransport:
    """Build a registered transport.

    Args:
        name: A registered transport name.
        fmt: The upstream wire format it should serve.
        responder: What to reply with; the transport's own default when omitted.

    Returns:
        An unstarted transport.

    Raises:
        LookupError: When ``name`` is not registered.
    """
    try:
        factory = _TRANSPORTS[name]
    except KeyError:
        raise LookupError(f"no transport named {name!r}; registered: {registered_transports()}") from None

    return factory(fmt, responder=responder)


def marker() -> str:
    """Return a marker unique to one request.

    Returns:
        A string to carry as the user's text and look for in the captured body.
        Regenerated per call so a capture left by an earlier fixture cannot
        satisfy an assertion about this one.
    """
    return f"kbr31-{uuid.uuid4().hex}"


def redirected(adapter: ProviderAdapter, origin: str, provider_config: dict[str, Any] | None = None) -> ProviderAdapter:
    """Return ``adapter`` re-hosted onto ``origin``, keeping path and query.

    For the ~14 default-transport adapters that honour no configuration key,
    :meth:`~kitty.providers.base.ProviderAdapter.build_base_url` is the only
    seam: it is the single method the bridge calls for the destination.
    Overriding ``default_base_url`` instead — the convention ``tests/bridge/``
    established by hand — is **not** equivalent, because the six adapters that
    override ``build_base_url`` ignore it.

    **Scheme and authority are substituted; path, query and fragment survive.**
    Replacing the whole URL is the obvious implementation and it is wrong:
    Vertex builds ``/{version}/projects/{project_id}/locations/{location}`` into
    its base URL, and §3.3.5 calls that "the account being billed". Deleting it
    would make P21 report a routing mismatch against a request that went exactly
    where the harness sent it. Azure survives a whole-URL replacement — its
    routing is in ``get_upstream_path`` — so the check most people would run
    passes while the row below it breaks. §3.3.5 specifies this same
    substitution from the comparison side, so both sides are one rule.

    Args:
        adapter: The adapter to redirect. Its type, and any state its
            constructor resolved, are preserved.
        origin: Scheme and authority to re-host onto, e.g. a recorder's
            ``base_url``. Anything it carries beyond those is ignored.
        provider_config: Passed to the adapter's real ``build_base_url``.

    Returns:
        An instance of a subclass of ``type(adapter)`` whose ``build_base_url``
        returns the re-hosted URL for any configuration.

    Raises:
        Exception: Whatever the adapter's own ``build_base_url`` raises — it is
            called here, once. That method is also pre-flight's **validator**
            (Vertex raises ``ProviderError`` for a missing ``project_id`` from
            inside it), so a redirected adapter raises at redirect time and no
            longer per request. A test exercising pre-flight validation must use
            the adapter itself, not the redirected copy.
    """
    # Compute against the real implementation, so whatever the adapter derives
    # from its configuration is kept rather than guessed at.
    original = urlsplit(adapter.build_base_url(provider_config or {}))
    target = urlsplit(origin)
    rehosted = urlunsplit((target.scheme, target.netloc, original.path, original.query, original.fragment))

    def _build_base_url(self: ProviderAdapter, provider_config: dict | None = None) -> str:
        """Return the re-hosted base URL.

        Args:
            provider_config: Ignored; the URL was resolved at redirect time.

        Returns:
            The re-hosted URL.
        """
        return rehosted

    subclass = type(
        f"Redirected{type(adapter).__name__}",
        (type(adapter),),
        {"build_base_url": _build_base_url, "__doc__": f"{type(adapter).__name__} re-hosted onto {origin}."},
    )

    # Copy state rather than re-running `__init__`: `MiniMaxTokenAdapter`
    # resolves a flag in its constructor, and a fresh no-arg instance would
    # silently discard it. No adapter defines `__slots__`.
    clone = object.__new__(subclass)
    clone.__dict__.update(adapter.__dict__)
    return clone


@dataclass
class AiohttpTransport:
    """The default transport: T-W4's recorder, reached over plain HTTP.

    The only transport this module registers. It binds ``custom_anthropic`` or
    ``custom_openai`` — the two adapters that honour
    ``provider_config["base_url"]`` — so the default path uses the product's own
    redirection channel rather than a subclass.

    Attributes:
        format: The upstream wire format served.
        responder: What to reply with; the recorder's minimal success when
            omitted. A closure over mutable state is how a reply is scripted to
            change between attempts.
    """

    #: A class attribute, not a field: every instance answers to one registry key.
    name = "aiohttp"

    format: WireFormat
    responder: Responder | None = None
    _recorder: RecordingUpstream = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the recorder eagerly, so an unserved format fails here.

        Raises:
            ValueError: When ``format`` is not one this recorder serves.
                Constructing now rather than at :meth:`start` puts that check at
                construction (R3.8) without this module keeping a second copy of
                the served-format set.
        """
        self._recorder = RecordingUpstream(default_format=self.format, responder=self.responder)

    @property
    def recorder(self) -> RecordingUpstream:
        """Return the underlying recorder.

        Returns:
            The :class:`~harness.recorder.RecordingUpstream`, for tests that
            need the parts of it this interface does not expose.
        """
        return self._recorder

    async def start(self) -> None:
        """Bind an ephemeral loopback port and begin recording."""
        await self._recorder.start()

    async def stop(self) -> None:
        """Release the port."""
        await self._recorder.stop()

    def bind(self) -> Binding:
        """Return the adapter and config that reach this recorder.

        Returns:
            A fresh adapter for this transport's format, and the provider
            configuration naming the recorder's base URL.

        Raises:
            RuntimeError: When the transport has not been started, because the
                recorder has no port until then.
        """
        return _ADAPTER_FOR_FORMAT[self.format](), {"base_url": self._recorder.base_url}

    @property
    def captures(self) -> Sequence[CapturedRequest]:
        """Return the completed captures, in arrival order.

        Returns:
            What the recorder holds. Spelled ``captures`` on a transport and
            ``requests`` on the recorder; one list, §7.2's name unchanged.
        """
        return self._recorder.requests

    @property
    def connections(self) -> Sequence[ConnectionRecord]:
        """Return every accepted connection.

        Returns:
            One record per connection, including any that carried no request.
        """
        return self._recorder.connections

    def assert_teardown_clean(self) -> None:
        """Assert every request was answered in the format this transport declares.

        Raises:
            UnmatchedPathError: When a request took the recorder's fallback.
            MisdeclaredFormatError: When a request's path selected some *other*
                format. That case leaves ``unmatched`` empty — the recorder
                dispatches by suffix and found one — so nothing else reports it.
        """
        self._recorder.assert_all_paths_matched()

        # The pure lookup, never `_format_for`: that one appends to `unmatched`,
        # and an assertion must not mutate the evidence it judges.
        wrong = [c.path for c in self.captures if format_for_path(c.path) is not self.format]
        if wrong:
            raise MisdeclaredFormatError(
                f"transport {self.name!r} declares {self.format.value} but "
                f"{wrong} select another format; the reply was chosen by the path, "
                f"so the declared format was never under test"
            )


register_transport(AiohttpTransport.name, AiohttpTransport)


def protocol_for(fmt: WireFormat) -> InboundProtocol:
    """Return the inbound route whose dialect matches ``fmt``.

    A convenience for helpers that must pick one, **not** a claim that the axes
    are the same (§7.5.1); a test asserting on translation names both itself.

    Args:
        fmt: An upstream wire format.

    Returns:
        The inbound protocol in the same dialect.

    Raises:
        KeyError: When no inbound route speaks ``fmt``. Exactly two: Bedrock
            Converse and Ollama ``/api/chat``, which the bridge reaches upstream
            but never serves inbound. A transport on either — T-B3's botocore,
            half of T-B1's — must name its own inbound route, because the bridge
            translates and nothing can derive it from the upstream format.
    """
    return _PROTOCOL_FOR_FORMAT[fmt]


def inbound_path(protocol: InboundProtocol, *, model: str = MODEL, stream: bool = False) -> str:
    """Return the bridge route that serves ``protocol``.

    Args:
        protocol: The inbound protocol.
        model: The model name. Only Gemini puts it in the path.
        stream: Whether the request streams. Only Gemini puts it in the path;
            every other protocol carries it in the body — so a Gemini caller
            must pass the same value here *and* to
            :func:`minimal_inbound_body`, which ignores it. Nothing detects a
            mismatch, because to the bridge neither spelling is malformed.

    Returns:
        The path to POST to.
    """
    if protocol is InboundProtocol.GEMINI:
        verb = "streamGenerateContent" if stream else "generateContent"
        return f"/v1beta/models/{model}:{verb}"

    return {
        InboundProtocol.CHAT_COMPLETIONS: "/v1/chat/completions",
        InboundProtocol.MESSAGES: "/v1/messages",
        InboundProtocol.RESPONSES: "/v1/responses",
    }[protocol]


def minimal_inbound_body(
    protocol: InboundProtocol, text: str, *, model: str = MODEL, stream: bool = False
) -> dict[str, Any]:
    """Return the smallest request ``protocol`` accepts, carrying ``text``.

    Args:
        protocol: The inbound protocol.
        text: The user's text — a :func:`marker` when the caller means to find
            it again in the captured upstream body.
        model: The model to ask for.
        stream: Whether to ask for a stream.

    Returns:
        The request body.
    """
    if protocol is InboundProtocol.GEMINI:
        # Gemini carries neither model nor stream in the body; both are in the path.
        return {"contents": [{"role": "user", "parts": [{"text": text}]}]}

    if protocol is InboundProtocol.RESPONSES:
        return {
            "model": model,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}],
            "stream": stream,
        }

    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "stream": stream,
    }
    if protocol is InboundProtocol.MESSAGES:
        # `max_tokens` is required by the Messages API and by nothing else.
        body["max_tokens"] = 16
    return body


def profile_for(
    transport: UpstreamTransport,
    *,
    name: str = "harness",
    model: str = MODEL,
    backup: bool = False,
    binding: Binding | None = None,
) -> Profile:
    """Return a valid profile pointed at ``transport``.

    ``provider_config`` is the channel the bridge reads — ``profile`` in
    balancing mode, the constructor keyword in single-backend mode.
    ``Profile.base_url`` is neither: the resolver never reads it, and it is typed
    ``HttpsUrl``, which a loopback recorder serving ``http://`` cannot satisfy.

    Args:
        transport: A **started** transport; its binding is read here.
        name: The profile name.
        model: The model the profile pins.
        backup: Whether this is a reserve-tier member.
        binding: A binding already obtained from ``transport``, so a caller
            building several profiles does not pay for it repeatedly.

    Returns:
        A profile the shipped schema accepts — including a UUIDv4 ``auth_ref``,
        which ``tests/conftest.py``'s `sample_profile_dict` is not (KBR-175).
    """
    adapter, provider_config = binding if binding is not None else transport.bind()
    return Profile(
        name=name,
        provider=adapter.provider_type,  # type: ignore[arg-type]
        model=model,
        auth_ref=_AUTH_REF,
        provider_config=provider_config,
        backup=backup,
    )


def backend_for(
    transport: UpstreamTransport,
    *,
    name: str = "harness",
    model: str = MODEL,
    key: str = _KEY,
    backup: bool = False,
    binding: Binding | None = None,
) -> Backend:
    """Return one balancing-pool member pointed at ``transport``.

    Args:
        transport: A **started** transport.
        name: The profile name; distinct per member.
        model: The model this member pins.
        key: The resolved credential.
        backup: Whether this member is reserve tier.
        binding: A binding already obtained from ``transport``. Passed in when
            building several members, because ``bind()`` is not guaranteed cheap
            — §7.5.2 says it may be an OAuth leg (T-B1) or a TLS session (T-B2),
            so re-deriving it per member would multiply real work.

    Returns:
        The ``(adapter, key, profile)`` triple ``BridgeServer(backends=…)`` takes.
    """
    resolved = binding if binding is not None else transport.bind()
    return resolved[0], key, profile_for(transport, name=name, model=model, backup=backup, binding=resolved)


def pin_backend_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make balancing selection round-robin instead of weighted-random.

    ``_get_next_backend`` picks with ``random.choices(tier, weights=…)``, so a
    two-backend test without this may never try the backend it is about to
    assert on. That is measured, not supposed:
    ``tests/bridge/test_opencode_responses_refusal.py`` records that its own
    two-backend test "would pass with the refusal handler deleted, which is how
    this was found". Shipped here so four Epic I tickets do not re-derive it.

    Args:
        monkeypatch: Pytest's monkeypatch fixture, which reverts the patch.
    """
    calls = itertools.count()

    def _round_robin(tier: Sequence[Any], weights: Sequence[float] | None = None, k: int = 1) -> list[Any]:
        """Return the next member of ``tier``, cycling.

        Args:
            tier: The candidates.
            weights: Ignored; determinism is the point.
            k: Ignored; the caller always takes one.

        Returns:
            A one-element list, as ``random.choices`` returns.
        """
        # A true cycle rather than "first unseen, then stick": the sticky form
        # starves every backend but one once the pool has been walked.
        return [tier[next(calls) % len(tier)]]

    # `kitty.bridge.server.random` is the stdlib module object, so this patches
    # `random.choices` PROCESS-WIDE for the duration — every caller in the
    # interpreter gets round-robin, not only the bridge. `monkeypatch` reverting
    # it is what keeps that contained, which is why this takes the fixture
    # rather than patching by hand.
    monkeypatch.setattr("kitty.bridge.server.random.choices", _round_robin)


@dataclass
class BridgeFixture:
    """A real ``BridgeServer`` started against a transport, as a context manager.

    Attributes:
        transport: The upstream this bridge is pointed at. Started and stopped
            by this fixture.
        model: The profile's model, or ``None`` for a profile that sets none.
            ``None`` is not speculative: register row **M1** ("replace `model`
            with the profile's model") is *conditional* on the profile setting
            one, so proving it needs the trigger **and** its complement, and the
            complement is only reachable this way. `tests/bridge/` already builds
            it by hand for the same reason.
        backend_models: One model per balancing-pool member, or ``None`` for the
            single-backend shape. Distinct models make it observable which
            member served a request, since all members share this transport's
            recorder; a test needing members on *separate* recorders builds its
            own ``backends`` from :func:`backend_for`.
    """

    transport: UpstreamTransport
    model: str | None = MODEL
    backend_models: Sequence[str] | None = None
    server: BridgeServer | None = field(init=False, default=None)
    _port: int = field(init=False, default=0, repr=False)

    async def start(self) -> None:
        """Start the transport, then a real bridge pointed at it.

        Raises:
            BaseException: Whatever the binding or the bridge raised, after
                releasing the transport. ``__aenter__`` propagates a failure here
                and Python then never calls ``__aexit__``, so without this the
                recorder stays bound and only the *next* test's ephemeral port
                allocation would notice. Reachable: a profile validation error in
                :func:`backend_for`, or a bind failure.
        """
        await self.transport.start()
        try:
            await self._start_bridge()
        except BaseException:
            await self.stop()
            raise

    async def _start_bridge(self) -> None:
        """Build and start the bridge against the already-started transport."""
        adapter, provider_config = self.transport.bind()

        # The `None` launcher adapter and its ignore are encapsulated here
        # rather than repeated in each of the 38 modules that need them.
        if self.backend_models is None:
            self.server = BridgeServer(
                None,  # type: ignore[arg-type]
                adapter,
                _KEY,
                model=self.model,
                provider_config=provider_config,
            )
        else:
            backends = [
                backend_for(self.transport, name=f"harness{i}", model=m, binding=(adapter, provider_config))
                for i, m in enumerate(self.backend_models)
            ]
            self.server = BridgeServer(
                None,  # type: ignore[arg-type]
                adapter,
                _KEY,
                model=self.model,
                backends=backends,
            )

        self._port = await self.server.start_async()

    async def stop(self) -> None:
        """Stop the bridge, then the transport, releasing both ports.

        The transport is stopped in a ``finally``: a bridge that fails to shut
        down must not leave a recorder bound, because the next test's ephemeral
        port allocation is the only thing that would notice.

        **This waits for in-flight upstream handlers.** ``stop_async`` cleans up
        the runner and closes the bridge's sessions; it does not abort live
        connections. So a test that deliberately leaves a request hanging — to
        observe a timeout, say — must release it before teardown, or it pays the
        full hold time in the gate. Aborting here instead would mean reaching
        into T-W4's recorder, and §7.3's proxy is the place that decision was
        already made for the case that needs it.
        """
        try:
            if self.server is not None:
                await self.server.stop_async()
                self.server = None
        finally:
            # Forget the port, so a post-teardown `port` or `post()` fails with
            # this class's own "not running" error rather than a connection
            # refusal from a port that now belongs to nobody.
            self._port = 0
            await self.transport.stop()

    async def __aenter__(self) -> BridgeFixture:
        """Start the transport and the bridge.

        Returns:
            The started fixture.
        """
        await self.start()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        """Release both ports, and check the transport only on the clean path.

        Args:
            exc_type: The exception type escaping the block, if any.
            *_: The value and traceback, unused.

        An exception raised here would **supersede** one raised in the block, so
        running the teardown check over a failing body would replace the
        assertion the test was making — and the falsification cases assert on
        message content.
        """
        await self.stop()
        if exc_type is None:
            self.transport.assert_teardown_clean()

    @property
    def port(self) -> int:
        """Return the bridge's bound port.

        Returns:
            The ephemeral port chosen by the kernel.

        Raises:
            RuntimeError: When the fixture has not been started.
        """
        if not self._port:
            raise RuntimeError("bridge fixture is not running; use it as a context manager")
        return self._port

    @property
    def base_url(self) -> str:
        """Return the URL an agent would post to.

        Returns:
            The bridge's loopback origin, with no trailing slash.
        """
        return f"http://127.0.0.1:{self.port}"

    @property
    def captures(self) -> Sequence[CapturedRequest]:
        """Return what reached the upstream, in arrival order.

        Returns:
            The transport's captures.
        """
        return self.transport.captures

    async def post(self, path: str, body: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT) -> tuple[int, str]:
        """POST ``body`` to the bridge and return the reply.

        Args:
            path: The inbound route, e.g. from :func:`inbound_path`.
            body: The JSON body, e.g. from :func:`minimal_inbound_body`.
            timeout: Total client timeout in seconds.

        Returns:
            The status and the raw response text. Text, not JSON: three of the
            inbound surfaces answer with SSE, which a JSON decode cannot read.

        Raises:
            TransportTimeout: When the reply did not arrive in ``timeout``.
        """
        started = time.monotonic()
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        # A session per call, deliberately, not an optimisation waiting to be
        # made: it is what guarantees every client session is closed before
        # `stop_async` runs. Hoisting it to fixture level would invert that
        # ordering, and nothing here would go red.
        try:
            async with (
                aiohttp.ClientSession(timeout=client_timeout) as session,
                session.post(f"{self.base_url}{path}", json=body) as response,
            ):
                return response.status, await response.text()
        except (TimeoutError, asyncio.TimeoutError) as exc:
            # A bare cancellation says only that something was slow, which the
            # reader already knows. The two ladders are named so they can tell
            # a mis-wired binding from a wrong-format reply without a rerun.
            raise TransportTimeout(
                f"no reply from the bridge on {path} within {timeout}s "
                f"(elapsed {time.monotonic() - started:.1f}s); transport "
                f"{self.transport.name!r} holds {len(self.captures)} capture(s). "
                f"An unreachable upstream costs ~{_CONNECT_LADDER_SECONDS}s and a "
                f"wrong-format reply ~{_EMPTY_LADDER_SECONDS}s; raise timeout= to observe either"
            ) from exc


async def assert_transport_reaches_its_recorder(
    subject: UpstreamTransport, *, protocol: InboundProtocol | None = None
) -> None:
    """Assert a registered transport actually carries a bridge's request.

    Drives **one** request through a real :class:`BridgeFixture` and makes four
    assertions, each of which a defect the other three pass can break — which is
    the only argument that establishes they are not redundant. The defects are
    in ``test_bridge_falsification.py``, one per assertion.

    T-B1–T-B3 call this against their own registration; the meta-test in
    ``test_bridge.py`` calls it over every registered transport, so a new one
    inherits the check by registering rather than by remembering.

    It takes an **unstarted transport**, not a registry name, so that the
    deliberate defects in ``test_bridge_falsification.py`` never have to be
    registered: a shared registry carrying four things that are wrong on purpose
    is a hazard, and it would also make the registry-completeness meta-test
    assert over them. The meta-test resolves names to instances itself.

    Args:
        subject: An unstarted transport. Started and stopped by this function.
        protocol: The inbound route to drive; the one matching the transport's
            declared format by default.

    Raises:
        AssertionError: When the bridge did not reach *this* transport's
            recorder, exactly once, with this request's content, successfully,
            in the declared format.
    """
    name = subject.name
    route = protocol if protocol is not None else protocol_for(subject.format)
    sent = marker()

    async with BridgeFixture(subject) as fixture:
        status, text = await fixture.post(inbound_path(route), minimal_inbound_body(route, sent))
        captures = list(subject.captures)

        # 1. Reached *this* recorder, once. An empty list is the decoy — a
        #    binding pointed at somebody else's live upstream, which answers 200.
        assert len(captures) == 1, (
            f"transport {name!r} holds {len(captures)} capture(s), expected exactly 1: "
            f"0 means the bridge reached some other upstream and every assertion built "
            f"on this fixture would be quantified over nothing; more than 1 means a retry "
            f"ladder fired, so the binding cannot be told from a broken one"
        )

        # 2. Carried *this* request. A count alone passes for a recorder that
        #    fabricates captures or stores no body.
        assert sent.encode() in (captures[0].body or b""), (
            f"transport {name!r} captured a request whose body does not carry the "
            f"marker {sent!r}; the capture cannot see the content under test"
        )

        # 3. And the client was actually served. A correct capture with a failed
        #    client is real — an upstream 400 produces exactly that.
        assert status == 200, (
            f"transport {name!r} captured the request correctly but the bridge answered {status}: {text[:200]!r}"
        )

    # 4. And in the format this transport declares — asserted by the fixture's
    #    own teardown, above, on the clean path. It is deliberately *not*
    #    repeated here: it would then be an assertion no defect could falsify,
    #    since `__aexit__` raises first either way, and an assertion nothing can
    #    kill is the thing plan section 1.4 objects to. `_MisdeclaredTransport`
    #    falsifies the teardown call instead.
