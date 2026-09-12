"""The bridge fixture core and its transport extension interface.

`.system_design/TEST_SUITE.md` §7.5 · plan task **T-W8** (KBR-31) ·
`.requirements/20260911T233557Z_bridge_fixture_core/REQUIREMENTS.md`.

The deliberate defects this module's subject must catch live in
``test_bridge_falsification.py``; what is here is the fixture's own behaviour.

**Layer.** No ``pytestmark``, so these take the ``l1`` path default, following
T-W4 and T-W5. The reason is §8.2's and only §8.2's: a test may not be moved to
``l3`` before the Subsystem job exists, and CI runs ``-m "l1 or l2"``, so an
``l3`` marker today would leave the fixture's own correctness checked by no job
at all while fifteen tickets build on it. §8.2 lists this module so T-K6
inherits a list rather than a search.
"""

from __future__ import annotations

import ast
import asyncio
import json
import socket
import uuid
from pathlib import Path

import aiohttp
import pytest

from harness.bridge import (
    DEFAULT_TIMEOUT,
    MODEL,
    REGISTERED_HERE,
    AiohttpTransport,
    BridgeFixture,
    InboundProtocol,
    MisdeclaredFormatError,
    TransportTimeout,
    UpstreamTransport,
    assert_transport_reaches_its_recorder,
    backend_for,
    inbound_path,
    marker,
    minimal_inbound_body,
    pin_backend_order,
    profile_for,
    protocol_for,
    redirected,
    register_transport,
    registered_transports,
    transport,
)
from harness.contract import WireFormat
from harness.recorder import Reply, UnmatchedPathError, minimal_success_body
from kitty.bridge.server import BridgeServer
from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.base import ProviderAdapter
from kitty.providers.minimax_token import MiniMaxTokenAnthropicAdapter
from kitty.providers.vertex import VertexAIAdapter

#: One row per registered transport: the module that registers it, its registry
#: name, the upstream format to construct it with, and **the inbound route that
#: exercises it**. Each of T-B1 (KBR-40), T-B2 (KBR-41) and T-B3 (KBR-42) adds
#: one row, and inherits the conformance check by doing so.
#:
#: The inbound route is a separate field and not derived, because the bridge
#: *translates*: `BEDROCK_CONVERSE` (T-B3) and `OLLAMA_CHAT` (half of T-B1) have
#: no inbound route of their own, so nothing can infer one from the format. A
#: single hardcoded format here — the first draft — would have raised
#: ``ValueError`` at construction for all three, since none of them serves
#: Anthropic Messages.
#:
#: This lives here and not in ``bridge.py`` on purpose: the structural guard
#: forbids that module importing an Epic B one, and registry completeness is a
#: property of the suite, not of the core.
CONFORMANCE_CASES: tuple[tuple[str, str, WireFormat, InboundProtocol], ...] = (
    ("harness.bridge", "aiohttp", WireFormat.ANTHROPIC_MESSAGES, InboundProtocol.MESSAGES),
    ("harness.provider_aiohttp", "provider_aiohttp", WireFormat.OLLAMA_CHAT, InboundProtocol.CHAT_COMPLETIONS),
)

#: What the modules above are expected to have registered between them. Derived
#: from the rows, **not** from ``REGISTERED_HERE``: that tuple is the core's own
#: registration and is pinned to one entry, so deriving from it would mean an
#: Epic B author's single new row still failed the completeness check.
EXPECTED_TRANSPORTS: frozenset[str] = frozenset(name for _module, name, _fmt, _route in CONFORMANCE_CASES)

#: Both formats the primary recorder serves, with the inbound route that matches
#: each — the pair every end-to-end case below is parametrised over.
SERVED = [
    pytest.param(WireFormat.ANTHROPIC_MESSAGES, id="anthropic_messages"),
    pytest.param(WireFormat.CHAT_COMPLETIONS, id="chat_completions"),
]


def _fixture(fmt: WireFormat = WireFormat.ANTHROPIC_MESSAGES, **kwargs: object) -> BridgeFixture:
    """Build an unstarted fixture over a fresh aiohttp transport.

    Args:
        fmt: The upstream wire format to serve.
        **kwargs: Passed to :class:`~harness.bridge.BridgeFixture`.

    Returns:
        The unstarted fixture.
    """
    return BridgeFixture(transport("aiohttp", fmt), **kwargs)  # type: ignore[arg-type]


async def _drive(fixture: BridgeFixture, fmt: WireFormat, text: str) -> tuple[int, str]:
    """POST one minimal request in the dialect matching ``fmt``.

    Args:
        fixture: A started fixture.
        fmt: The upstream format, which selects the inbound route.
        text: The user's text.

    Returns:
        The status and raw reply text.
    """
    route = protocol_for(fmt)
    return await fixture.post(inbound_path(route), minimal_inbound_body(route, text))


def _closed(port: int) -> bool:
    """Return whether nothing is listening on ``port``.

    Args:
        port: A loopback TCP port.

    Returns:
        ``True`` when a connection is refused.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) != 0


class _TeardownWasChecked(AssertionError):
    """Raised by :class:`_CheckedTransport` to prove its check was reached."""


class _CheckedTransport(AiohttpTransport):
    """A transport whose teardown check is unmistakable when it runs."""

    def assert_teardown_clean(self) -> None:
        """Announce that the check ran.

        Raises:
            _TeardownWasChecked: Always.
        """
        raise _TeardownWasChecked("the fixture ran the transport's teardown check")


async def _raw_post(base_url: str, path: str, body: bytes = b'{"probe": true}') -> None:
    """POST straight to a recorder, bypassing the bridge.

    Used where the subject is what the *upstream* received at a given path. The
    bridge chooses the upstream path from its adapter, so it cannot be steered
    by choosing an inbound route.

    Args:
        base_url: The recorder's origin.
        path: The upstream path to hit.
        body: The request body.
    """
    async with (
        aiohttp.ClientSession() as session,
        session.post(f"{base_url}{path}", data=body, headers={"content-type": "application/json"}) as response,
    ):
        await response.read()


# ── R3.5, R3.7 — the registry ────────────────────────────────────────────────


class TestTheRegistry:
    """Registration, lookup, and the pin on what this module itself registers."""

    def test_the_default_transport_is_registered(self) -> None:
        """The one transport T-W8 ships is reachable by name."""
        assert "aiohttp" in registered_transports()

    def test_registering_a_name_twice_raises(self) -> None:
        """Silently replacing would let the loser be whichever imported first.

        Two Epic B modules disagreeing about what a name means is a conflict
        that must be visible, not resolved by import order.
        """
        with pytest.raises(ValueError, match="already registered"):
            register_transport("aiohttp", AiohttpTransport)

    def test_an_unknown_name_raises_and_names_what_is_registered(self) -> None:
        """The message has to be actionable: the likely cause is a missing import."""
        with pytest.raises(LookupError, match="aiohttp") as excinfo:
            transport("curl_cffi", WireFormat.CHAT_COMPLETIONS)
        assert "curl_cffi" in str(excinfo.value)

    def test_this_module_registers_exactly_one_transport(self) -> None:
        """Pin ``bridge.py``'s own registration, not the process-wide registry.

        Asserting ``registered_transports() == ("aiohttp",)`` would be the
        obvious form and would break on the first Epic B registration — in a
        module five streams are told not to edit. The claim that stays true is
        about what *this* module registers.
        """
        assert REGISTERED_HERE == ("aiohttp",)

    def test_the_core_imports_no_epic_b_module(self) -> None:
        """Structurally, not by convention.

        T-B1–T-B3 register into the core; the core must not reach back, or the
        extension interface is decorative and the import graph is a cycle
        waiting to happen.
        """
        source = Path(__import__("harness.bridge", fromlist=["__file__"]).__file__).read_text(encoding="utf-8")
        assert len(source) > 1000, "read no meaningful source; the guard would pass vacuously"

        tree = ast.parse(source)
        from_imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]

        # A *relative* import would slip past the name check entirely: its target
        # lives in `node.level`, and `node.module` is the tail or None — so
        # `from .epic_b import X` presents as "epic_b" and matches no prefix.
        # This package has an `__init__.py`, so the form is available.
        relative = [f".{node.module or ''}" for node in from_imports if node.level]
        assert relative == [], f"the core uses relative imports, which the name check cannot see: {relative}"

        imported = {node.module or "" for node in from_imports} | {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        }
        epic_b = {name for name in imported if name.startswith("harness.") and name != "harness.contract"}

        assert epic_b == {"harness.recorder"}, f"the core imports harness modules it should not: {epic_b}"


# ── R3.1–R3.4, R3.8, R3.9 — the extension interface ──────────────────────────


class TestTheExtensionInterface:
    """What a transport must provide, exercised on the one that ships."""

    def test_the_default_transport_satisfies_the_protocol(self) -> None:
        """The shipped transport is an instance of the interface it defines.

        A `Protocol` nobody is checked against is documentation; this is what
        makes it a contract T-B1–T-B3 can be judged by.

        **What it does not buy:** ``isinstance`` against a ``runtime_checkable``
        Protocol checks attribute *presence* only, never signatures, and
        ``mypy`` runs on ``src/kitty`` alone (plan §1.3) — so a transport whose
        ``bind()`` takes the wrong arguments passes this. The conformance check
        is what catches that, by calling it.
        """
        assert isinstance(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES), UpstreamTransport)

    @pytest.mark.parametrize(
        "fmt",
        [WireFormat.OPENAI_RESPONSES, WireFormat.GEMINI, WireFormat.BEDROCK_CONVERSE, WireFormat.OLLAMA_CHAT],
    )
    def test_a_format_this_transport_does_not_serve_raises_at_construction(self, fmt: WireFormat) -> None:
        """Not at request time.

        A wrong-format reply is not a loud failure — the adapter parses nothing,
        the reply reads as empty, and the test pays the retry ladder. The four
        formats here belong to T-B1–T-B3, whose transports this one never is.

        Args:
            fmt: A format outside the two the primary recorder serves.
        """
        with pytest.raises(ValueError, match="primary recorder serves"):
            transport("aiohttp", fmt)

    def test_the_factory_takes_no_host_keyword(self) -> None:
        """A seam is paid for by a named consumer, and this one had none.

        §7.3 gives T-E1 non-loopback addressing by *resolution* — the proxy's
        ``resolve`` map — not by a bind address, and a CI runner may have no
        non-loopback address to bind. Pinned so it is not re-added absently.
        """
        with pytest.raises(TypeError):
            transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, host="127.0.0.2")  # type: ignore[call-arg]

    async def test_bind_points_an_adapter_at_this_transport_s_recorder(self) -> None:
        """The seam returns a product adapter and the config that reaches it."""
        subject = transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)
        await subject.start()
        try:
            adapter, config = subject.bind()

            assert isinstance(adapter, ProviderAdapter)
            assert config["base_url"] == subject.recorder.base_url  # type: ignore[attr-defined]
            # The product's own channel, not a subclass: this adapter reads it.
            assert adapter.build_base_url(config) == subject.recorder.base_url  # type: ignore[attr-defined]
        finally:
            await subject.stop()

    def test_binding_before_the_transport_starts_raises(self) -> None:
        """A recorder has no port until it is bound, so neither has a binding.

        Documented on ``bind`` and otherwise untested; the failure it prevents is
        a bridge pointed at ``http://127.0.0.1:0``.
        """
        with pytest.raises(RuntimeError, match="not running"):
            transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES).bind()

    async def test_a_stateful_responder_scripts_successive_replies(self) -> None:
        """Scripted failures are closures over the hook, not a setter on the Protocol.

        Every consumer named for ``responder`` — §6.3.1's four injection points,
        T-I8's blip-then-success — needs the reply to change between attempts.
        This pins that the hook is enough, so no one adds a setter to an
        interface fifteen tickets consume.
        """
        replies = iter(
            [
                (400, {"type": "error", "error": {"type": "invalid_request_error", "message": "no"}}),
                (200, minimal_success_body(WireFormat.ANTHROPIC_MESSAGES)),
            ]
        )

        async def scripted(captured: object, reply: Reply) -> None:
            """Answer with the next scripted status and body.

            The body matters as much as the status: an empty or unparseable one
            is not a loud failure but 80 seconds of retry ladder, which would
            exhaust this script and report a ``StopIteration`` instead of the
            reply sequence under test.

            Args:
                captured: The capture, unused.
                reply: The unprepared response.
            """
            status, body = next(replies)
            await reply.begin(status, {"content-type": "application/json"})
            await reply.write(json.dumps(body).encode())

        subject = transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=scripted)
        async with BridgeFixture(subject) as fixture:
            first, _ = await _drive(fixture, WireFormat.ANTHROPIC_MESSAGES, marker())
            second, _ = await _drive(fixture, WireFormat.ANTHROPIC_MESSAGES, marker())

        assert (first, second) == (400, 200)

    async def test_connections_are_reported_alongside_captures(self) -> None:
        """§5.2.1's bypass is a connection carrying no request.

        No capture list can express it, and ``unattributable_peer_ports`` joins
        on exactly these ports — so the interface carries them, rather than
        T-E1 and T-I9 each reaching past it.
        """
        async with _fixture() as fixture:
            await _drive(fixture, WireFormat.ANTHROPIC_MESSAGES, marker())
            records = list(fixture.transport.connections)

        assert len(records) == 1
        assert records[0].requests == 1
        assert records[0].peer_port > 0


# ── R2.4 — redirection ───────────────────────────────────────────────────────


class TestRedirection:
    """Pointing an adapter that honours no configuration key at a recorder."""

    def test_it_re_hosts_and_keeps_the_path(self) -> None:
        """Vertex builds the billed account into the base URL's path.

        §3.3.5: for Vertex "the account being billed is a URL component" (P21).
        A whole-URL replacement deletes ``/projects/{id}/locations/{loc}`` while
        T-D2's independently-derived expectation still carries it, so P21 would
        report a routing mismatch against a request that went exactly where the
        harness sent it.
        """
        config = {"project_id": "p1", "location": "europe-west4"}
        rehosted = redirected(VertexAIAdapter(), "http://127.0.0.1:9", config).build_base_url(config)

        assert rehosted.startswith("http://127.0.0.1:9/")
        assert rehosted.endswith("/projects/p1/locations/europe-west4")

    def test_a_whole_url_replacement_would_have_lost_that_path(self) -> None:
        """The difference asserted, so the re-host rule cannot be quietly undone.

        Without this, a later simplification to "return the origin" passes every
        other test here — the two adapters T-W8 binds have empty base-URL paths,
        so nothing else in this file would notice.
        """
        config = {"project_id": "p1"}
        original = VertexAIAdapter().build_base_url(config)
        rehosted = redirected(VertexAIAdapter(), "http://127.0.0.1:9", config).build_base_url(config)

        from urllib.parse import urlsplit

        assert urlsplit(original).path == urlsplit(rehosted).path != ""

    def test_it_redirects_an_adapter_that_honours_no_configuration_key(self) -> None:
        """The ~14 adapters ``provider_config["base_url"]`` cannot reach.

        ``AnthropicAdapter`` does not override ``build_base_url`` at all, so the
        product's own channel is a no-op for it — which is the whole reason this
        helper exists.
        """
        assert AnthropicAdapter().build_base_url({"base_url": "http://127.0.0.1:9"}) != "http://127.0.0.1:9"
        assert redirected(AnthropicAdapter(), "http://127.0.0.1:9").build_base_url({}) == "http://127.0.0.1:9"

    def test_the_redirected_adapter_keeps_its_type_and_behaviour(self) -> None:
        """Everything but the destination is inherited, so the shipped code runs.

        The upstream **path** is the one asserted because it is what the
        destination is composed with: a helper that moved it as well as the
        origin would send a correct body to the wrong endpoint, which on Azure
        is a different request entirely (§3.3.5).
        """
        original = AnthropicAdapter()
        redirect = redirected(original, "http://127.0.0.1:9")

        assert isinstance(redirect, AnthropicAdapter)
        assert redirect.upstream_path == original.upstream_path
        assert redirect.build_upstream_headers("k") == original.build_upstream_headers("k")

    def test_constructor_resolved_state_survives_the_redirect(self) -> None:
        """``MiniMaxTokenAnthropicAdapter`` resolves a flag in ``__init__``.

        Re-running the constructor with no arguments would discard it silently,
        and the redirected adapter would translate differently from the one the
        test built.
        """
        adapter = MiniMaxTokenAnthropicAdapter(native_messages=True)
        redirect = redirected(adapter, "http://127.0.0.1:9", {})

        assert redirect.use_native_messages == adapter.use_native_messages is True

    def test_the_adapter_s_own_validation_still_runs_at_redirect_time(self) -> None:
        """``build_base_url`` is also pre-flight's validator.

        Documented rather than discovered: the redirected copy raises here and
        no longer per request, so a test exercising pre-flight validation must
        use the adapter itself.
        """
        with pytest.raises(Exception, match="project_id"):
            redirected(VertexAIAdapter(), "http://127.0.0.1:9", {})

    async def test_a_redirected_adapter_reaches_a_real_recorder(self) -> None:
        """End to end, because the three tests above are all about a string."""
        subject = transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)
        await subject.start()
        try:
            adapter = redirected(AnthropicAdapter(), subject.recorder.base_url)
            server = BridgeServer(None, adapter, "k", model=MODEL)  # type: ignore[arg-type]
            port = await server.start_async()
            sent = marker()
            route = protocol_for(WireFormat.ANTHROPIC_MESSAGES)
            try:
                # A plain client rather than `BridgeFixture.post`: this bridge is
                # built by hand to isolate `redirected`, and borrowing the
                # fixture's helper would mean assigning its private port field.
                timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
                async with (
                    aiohttp.ClientSession(timeout=timeout) as session,
                    session.post(
                        f"http://127.0.0.1:{port}{inbound_path(route)}",
                        json=minimal_inbound_body(route, sent),
                    ) as response,
                ):
                    status = response.status
                    await response.read()
            finally:
                await server.stop_async()

            assert status == 200
            assert len(subject.captures) == 1
            assert sent.encode() in subject.captures[0].body
        finally:
            await subject.stop()


# ── R2.1–R2.3, R2.5 — profiles and backends ──────────────────────────────────


class TestTheProfileFactory:
    """Valid profiles, and the balancing triple ``BridgeServer`` consumes."""

    async def test_a_profile_carries_the_binding_and_is_schema_valid(self) -> None:
        """Including a UUIDv4 ``auth_ref``, which `sample_profile_dict` is not.

        KBR-175: the existing shared fixture carries a UUIDv7 and cannot build a
        ``Profile`` at all, which is why this factory exists rather than reusing it.
        """
        subject = transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)
        await subject.start()
        try:
            profile = profile_for(subject)

            assert profile.provider_config["base_url"] == subject.recorder.base_url  # type: ignore[attr-defined]
            assert uuid.UUID(profile.auth_ref).version == 4
            assert profile.base_url is None, "the resolver never reads this, and it is HTTPS-only"
        finally:
            await subject.stop()

    async def test_a_backend_is_the_triple_the_bridge_takes(self) -> None:
        """Asserted by construction rather than by shape, so the pin is real."""
        subject = transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)
        await subject.start()
        try:
            adapter, key, profile = backend_for(subject, model="m0")

            assert isinstance(adapter, ProviderAdapter)
            assert isinstance(key, str)
            assert profile.model == "m0"
            BridgeServer(None, adapter, key, backends=[(adapter, key, profile)])  # type: ignore[arg-type]
        finally:
            await subject.stop()

    async def test_a_reserve_member_is_marked_backup(self) -> None:
        """Reserve-tier selection is unreachable otherwise."""
        subject = transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)
        await subject.start()
        try:
            assert backend_for(subject, backup=True)[2].backup is True
        finally:
            await subject.stop()


# ── R1.6, R1.1–R1.3, R5 — the fixture itself ─────────────────────────────────


class TestTheFixtureCore:
    """Starting, addressing, posting and stopping."""

    async def test_it_starts_the_shipped_bridge_class(self) -> None:
        """The subject is the product, not a stand-in for it."""
        async with _fixture() as fixture:
            assert isinstance(fixture.server, BridgeServer)
            assert fixture.port > 0
            assert fixture.base_url == f"http://127.0.0.1:{fixture.port}"

    @pytest.mark.parametrize("fmt", SERVED)
    async def test_one_request_reaches_the_upstream_once(self, fmt: WireFormat) -> None:
        """The whole point, for each format the primary recorder serves.

        Args:
            fmt: The upstream wire format.
        """
        sent = marker()
        async with _fixture(fmt) as fixture:
            status, _ = await _drive(fixture, fmt, sent)
            captures = list(fixture.captures)

        assert status == 200
        assert len(captures) == 1
        assert sent.encode() in captures[0].body

    @pytest.mark.parametrize("route", list(InboundProtocol))
    async def test_every_inbound_protocol_reaches_the_upstream(self, route: InboundProtocol) -> None:
        """All four routes, driven through a real bridge.

        ``inbound_path`` and ``minimal_inbound_body`` are otherwise asserted
        only against string constants, which proves the helpers agree with
        themselves and nothing about whether the bridge accepts what they
        build. For a module whose whole subject is harnesses that pass while
        proving nothing, that is the one gap not worth leaving — the Responses
        and Gemini bodies are used by no other case here.

        Args:
            route: The inbound protocol to drive.
        """
        sent = marker()
        async with _fixture() as fixture:
            status, _ = await fixture.post(inbound_path(route), minimal_inbound_body(route, sent))
            captures = list(fixture.captures)

        assert status == 200
        assert len(captures) == 1
        assert sent.encode() in captures[0].body, "the user's text must survive translation to the upstream"

    async def test_a_streamed_reply_comes_back_as_text(self) -> None:
        """Three inbound surfaces answer with SSE, which a JSON decode cannot read."""
        route = protocol_for(WireFormat.ANTHROPIC_MESSAGES)
        async with _fixture() as fixture:
            status, text = await fixture.post(inbound_path(route), minimal_inbound_body(route, marker(), stream=True))

        assert status == 200
        assert "event:" in text

    async def test_the_port_is_refused_before_the_fixture_starts(self) -> None:
        """Reading an address that does not exist yet is a mistake, not a zero."""
        with pytest.raises(RuntimeError, match="not running"):
            _ = _fixture().port

    async def test_a_timeout_names_the_transport_and_the_two_ladders(self) -> None:
        """A bare cancellation says only that something was slow.

        The reader already knows that. What identifies the defect is which
        ladder the elapsed time resembles, so the message carries both.

        The responder is released explicitly rather than left sleeping, because
        teardown waits for in-flight upstream handlers: ``stop_async`` cleans up
        the runner but does not abort live connections, so a responder that
        slept for 30 seconds would charge the fast gate all 30 of them. A test
        that means to leave a request hanging must free it before teardown.
        """
        route = protocol_for(WireFormat.ANTHROPIC_MESSAGES)
        released = asyncio.Event()

        async def waits_to_be_released(captured: object, reply: Reply) -> None:
            """Hold the request open until the test lets go.

            Args:
                captured: The capture, unused.
                reply: The unprepared response, never prepared.
            """
            await released.wait()

        subject = transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=waits_to_be_released)
        fixture = BridgeFixture(subject)
        await fixture.start()
        try:
            with pytest.raises(TransportTimeout) as excinfo:
                await fixture.post(inbound_path(route), minimal_inbound_body(route, marker()), timeout=0.3)
        finally:
            released.set()
            await fixture.stop()

        message = str(excinfo.value)
        assert "aiohttp" in message
        assert "30" in message and "80" in message, "the message must name both ladders"
        # Deterministically one, not zero: the recorder stores its capture after
        # reading the body and *before* calling the responder, so a reply held
        # open still leaves a complete capture. Pinning the value is what makes
        # the count in the message worth carrying at all.
        assert "1 capture" in message

    def test_the_default_timeout_is_asserted_as_a_value(self) -> None:
        """Not by waiting for it. aiohttp's own default is ``total=300``."""
        from harness.bridge import DEFAULT_TIMEOUT

        assert DEFAULT_TIMEOUT == 10.0


class TestTeardown:
    """Releasing both ports, and never substituting an exception."""

    async def test_both_ports_are_released_on_the_clean_path(self) -> None:
        """A leaked recorder is noticed only by the next test's port allocation."""
        async with _fixture() as fixture:
            bridge_port = fixture.port
            upstream_port = fixture.transport.recorder.port  # type: ignore[attr-defined]
            await _drive(fixture, WireFormat.ANTHROPIC_MESSAGES, marker())

        assert _closed(bridge_port)
        assert _closed(upstream_port)

    async def test_a_bridge_that_fails_to_start_still_releases_the_recorder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure path ``__aexit__`` never sees.

        When ``start()`` raises, ``__aenter__`` propagates and Python does not
        call ``__aexit__`` — so nothing else would release the recorder, and the
        leak would surface as an unrelated test failing to bind much later.
        Reachable in earnest: a profile validation error while building
        backends, or a bind failure.

        Args:
            monkeypatch: Used to make the bridge's own start fail.
        """
        fixture = _fixture()
        boom = RuntimeError("the bridge could not bind")
        ports: list[int] = []

        async def explode(_self: BridgeServer) -> int:
            """Fail the way a bind failure would, once the recorder is up.

            Args:
                _self: The bridge, unused.

            Returns:
                Never; always raises.

            Raises:
                RuntimeError: Always.
            """
            ports.append(fixture.transport.recorder.port)
            raise boom

        monkeypatch.setattr(BridgeServer, "start_async", explode)

        with pytest.raises(RuntimeError) as excinfo:
            await fixture.start()

        assert excinfo.value is boom
        assert _closed(ports[0]), "the recorder outlived a bridge that never started"

    async def test_a_transport_that_fails_to_start_is_released_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The other half of the failure ``__aexit__`` never sees.

        ``RecordingUpstream.start`` assigns its runner, awaits ``setup()`` and
        only then binds the site — so a failure in the bind leaves a runner that
        still needs ``cleanup()``, and it happens before there is any bridge for
        the second guard to unwind. Two guards, because one cannot cover both.

        Args:
            monkeypatch: Used to fail the transport's own start.
        """
        fixture = _fixture()
        boom = RuntimeError("the recorder could not bind")
        stopped: list[bool] = []

        original_stop = type(fixture.transport).stop

        async def spy_stop(transport_self: AiohttpTransport) -> None:
            """Record that the transport was released, then release it.

            Args:
                transport_self: The transport being stopped.
            """
            stopped.append(True)
            await original_stop(transport_self)

        async def explode(_self: AiohttpTransport) -> None:
            """Fail the way a bind failure inside the recorder would.

            Args:
                _self: The transport, unused.

            Raises:
                RuntimeError: Always.
            """
            raise boom

        monkeypatch.setattr(AiohttpTransport, "stop", spy_stop)
        monkeypatch.setattr(AiohttpTransport, "start", explode)

        with pytest.raises(RuntimeError) as excinfo:
            await fixture.start()

        assert excinfo.value is boom
        assert stopped == [True], "a transport that failed part-way through start was never released"

    async def test_a_stopped_fixture_forgets_its_port(self) -> None:
        """A dead port is worse than no port.

        Left in place, a post-teardown ``post()`` fails with a connection
        refusal from a port that now belongs to nobody — or, worse, to whatever
        bound it next.
        """
        fixture = _fixture()
        async with fixture:
            assert fixture.port > 0

        with pytest.raises(RuntimeError, match="not running"):
            _ = fixture.port

    async def test_a_raising_body_still_releases_both_ports(self) -> None:
        """Teardown that only runs on success is teardown that does not run."""
        fixture = _fixture()
        sentinel = RuntimeError("the test's own failure")

        with pytest.raises(RuntimeError) as excinfo:
            async with fixture:
                bridge_port = fixture.port
                upstream_port = fixture.transport.recorder.port  # type: ignore[attr-defined]
                raise sentinel

        assert excinfo.value is sentinel, "the original failure must reach the reporter"
        assert _closed(bridge_port)
        assert _closed(upstream_port)

    async def test_the_teardown_check_does_not_replace_the_body_s_failure(self) -> None:
        """An exception from ``__aexit__`` supersedes one from the block.

        So a teardown check run over a failing body would report the wrong
        thing, and the falsification cases — which assert on message content —
        would fail for the wrong reason. Here the body fails *and* leaves an
        unmatched path, which the teardown check would otherwise raise on.
        """
        fixture = _fixture()
        sentinel = RuntimeError("the test's own failure")

        with pytest.raises(RuntimeError) as excinfo:
            async with fixture:
                await fixture.post("/not/an/api", {"model": MODEL})
                raise sentinel

        assert excinfo.value is sentinel


class TestTeardownChecks:
    """What ``assert_teardown_clean`` catches, who calls it, and what it must not disturb.

    These drive the **recorder** directly rather than through the bridge. The
    subject is what the *upstream* received, and the bridge decides the upstream
    path from its adapter — an inbound route of ``/not/an/api`` is a bridge 404
    and produces no upstream request at all. That is §7.5.1's two axes, and
    getting it wrong here would have made every case below vacuous.
    """

    async def test_an_unmatched_path_fails_the_transport(self) -> None:
        """A fallback-format reply is silent and costs the retry ladder.

        Inherited from T-W4's recorder and routed through the transport, so
        every consumer of the fixture gets it.
        """
        subject = transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)
        await subject.start()
        try:
            await _raw_post(subject.recorder.base_url, "/not/an/api")
            with pytest.raises(UnmatchedPathError):
                subject.assert_teardown_clean()
        finally:
            await subject.stop()

    async def test_a_path_selecting_another_format_fails_and_names_the_declaration(self) -> None:
        """The case ``unmatched`` cannot report.

        The recorder dispatches by path **suffix**, so a request to the *other*
        format's path is answered in that format, the adapter parses it happily,
        and nothing is recorded as unmatched. The transport's declared format was
        simply never under test — this is the only signal that says so.
        """
        subject = transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)
        await subject.start()
        try:
            await _raw_post(subject.recorder.base_url, "/chat/completions")

            assert subject.recorder.unmatched == [], "the suffix matched, so nothing is unmatched"
            with pytest.raises(MisdeclaredFormatError, match="anthropic_messages"):
                subject.assert_teardown_clean()
        finally:
            await subject.stop()

    async def test_the_check_leaves_the_recorder_s_unmatched_list_alone(self) -> None:
        """An assertion must not mutate the evidence it judges.

        ``RecordingUpstream._format_for`` appends to ``unmatched`` on a miss, so
        judging captures with it would grow that list on every fixture exit and
        fight the tests that deliberately clear it. This is why T-W4's pure
        ``format_for_path`` was split out, and this is what would notice if the
        split were undone.
        """
        subject = transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)
        await subject.start()
        try:
            await _raw_post(subject.recorder.base_url, "/not/an/api")
            subject.recorder.unmatched.clear()

            with pytest.raises(MisdeclaredFormatError):
                subject.assert_teardown_clean()

            assert subject.recorder.unmatched == []
        finally:
            await subject.stop()

    async def test_the_fixture_runs_the_check_on_the_clean_path(self) -> None:
        """Otherwise the guard is enforced by nothing.

        A unit test on ``assert_teardown_clean`` proves that function raises; it
        does not prove anything **calls** it. Plan §1.4's own list of past
        harness failures includes "a guard proving a function was called when
        the enforcement was the branch after it", and this is its mirror.
        """
        with pytest.raises(_TeardownWasChecked):
            async with BridgeFixture(_CheckedTransport(WireFormat.ANTHROPIC_MESSAGES)):
                pass

    async def test_the_fixture_skips_the_check_when_the_body_raised(self) -> None:
        """An exception from ``__aexit__`` supersedes one from the block.

        So a teardown check run over a failing body would report the wrong
        thing, and the falsification cases — which assert on message content —
        would fail for the wrong reason. The transport here raises a distinctive
        error from its check, so a regression is unmistakable rather than
        inferred from an absence.
        """
        sentinel = RuntimeError("the test's own failure")

        with pytest.raises(RuntimeError) as excinfo:
            async with BridgeFixture(_CheckedTransport(WireFormat.ANTHROPIC_MESSAGES)):
                raise sentinel

        assert excinfo.value is sentinel


# ── R1.4, R1.5 — both bridge shapes ──────────────────────────────────────────


class TestBothBridgeShapes:
    """Single-backend and balancing, which share no selection code."""

    async def test_a_balancing_pool_serves_a_request(self) -> None:
        """``_select_backend`` is unreachable through the single-backend constructor."""
        sent = marker()
        async with _fixture(backend_models=["m0", "m1"]) as fixture:
            status, _ = await _drive(fixture, WireFormat.ANTHROPIC_MESSAGES, sent)
            captures = list(fixture.captures)

        assert status == 200
        assert len(captures) == 1
        assert sent.encode() in captures[0].body

    def test_the_product_really_selects_at_random(self) -> None:
        """The half of R1.5 the round-robin test cannot assert.

        ``pin_backend_order`` is worth shipping only because the shipped
        selection is weighted-random. If the product ever became deterministic,
        the seam would be dead weight and every balancing test would be pinning
        something that no longer moves — visibly, here, rather than never.
        """
        source = (Path(__file__).resolve().parents[2] / "src" / "kitty" / "bridge" / "server.py").read_text(
            encoding="utf-8"
        )

        assert "random.choices(tier, weights=weights, k=1)" in source

    async def test_pinning_makes_selection_round_robin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Weighted-random selection makes an undisciplined balancing test vacuous.

        Measured, not supposed: ``test_opencode_responses_refusal.py`` records a
        two-backend test that "would pass with the refusal handler deleted".
        Distinct models per member are what make the choice observable, since
        both members share this transport's recorder.

        Args:
            monkeypatch: Reverts the patch on the ambient random source.
        """
        pin_backend_order(monkeypatch)

        async with _fixture(backend_models=["m0", "m1"]) as fixture:
            await _drive(fixture, WireFormat.ANTHROPIC_MESSAGES, marker())
            await _drive(fixture, WireFormat.ANTHROPIC_MESSAGES, marker())
            models = [c.body.decode() for c in fixture.captures]

        assert len(models) == 2
        assert '"model": "m0"' in models[0]
        assert '"model": "m1"' in models[1]


# ── R1.6 — the inbound vocabulary ────────────────────────────────────────────


class TestTheInboundHelpers:
    """Routes and bodies per inbound protocol — the axis that is not ``WireFormat``."""

    @pytest.mark.parametrize(
        ("protocol", "expected"),
        [
            (InboundProtocol.CHAT_COMPLETIONS, "/v1/chat/completions"),
            (InboundProtocol.MESSAGES, "/v1/messages"),
            (InboundProtocol.RESPONSES, "/v1/responses"),
        ],
    )
    def test_each_protocol_has_its_own_route(self, protocol: InboundProtocol, expected: str) -> None:
        """Pinned, because a wrong route is a 404 rather than a translation bug.

        Args:
            protocol: The inbound protocol.
            expected: Its route.
        """
        assert inbound_path(protocol) == expected

    @pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
    def test_gemini_carries_the_model_and_the_verb_in_the_path(self, stream: bool) -> None:
        """The one protocol whose route is not a constant.

        Args:
            stream: Whether the request streams.
        """
        path = inbound_path(InboundProtocol.GEMINI, model="m", stream=stream)
        assert path.startswith("/v1beta/models/m:")
        assert path.endswith("streamGenerateContent" if stream else "generateContent")

    @pytest.mark.parametrize("protocol", list(InboundProtocol))
    def test_every_body_carries_the_marker(self, protocol: InboundProtocol) -> None:
        """The marker is how a capture is tied to the request that produced it.

        Args:
            protocol: The inbound protocol.
        """
        assert "MARK" in str(minimal_inbound_body(protocol, "MARK"))

    def test_the_messages_body_carries_the_field_only_it_requires(self) -> None:
        """``max_tokens`` is required by the Messages API and by nothing else."""
        assert "max_tokens" in minimal_inbound_body(InboundProtocol.MESSAGES, "x")
        assert "max_tokens" not in minimal_inbound_body(InboundProtocol.CHAT_COMPLETIONS, "x")

    @pytest.mark.parametrize("fmt", SERVED)
    def test_the_matching_route_is_a_convenience_not_an_equivalence(self, fmt: WireFormat) -> None:
        """Two axes, and only the pass-through pairs coincide.

        The formats with no inbound route at all are what make the point:
        ``protocol_for`` must refuse them rather than invent one.

        Args:
            fmt: A format the primary recorder serves.
        """
        assert isinstance(protocol_for(fmt), InboundProtocol)

    @pytest.mark.parametrize(
        ("fmt", "expected"),
        [
            (WireFormat.ANTHROPIC_MESSAGES, InboundProtocol.MESSAGES),
            (WireFormat.CHAT_COMPLETIONS, InboundProtocol.CHAT_COMPLETIONS),
            (WireFormat.OPENAI_RESPONSES, InboundProtocol.RESPONSES),
            (WireFormat.GEMINI, InboundProtocol.GEMINI),
        ],
    )
    def test_every_format_with_an_inbound_route_resolves(self, fmt: WireFormat, expected: InboundProtocol) -> None:
        """All four, not just the two the primary recorder serves.

        Parametrising over ``SERVED`` cannot see a missing entry for a format
        some *other* transport declares — and T-B2's is ``OPENAI_RESPONSES``,
        which has a route of its own and was missing from the map.

        Args:
            fmt: A format with an inbound route.
            expected: That route.
        """
        assert protocol_for(fmt) is expected

    @pytest.mark.parametrize("fmt", [WireFormat.BEDROCK_CONVERSE, WireFormat.OLLAMA_CHAT])
    def test_the_two_formats_with_no_inbound_route_refuse(self, fmt: WireFormat) -> None:
        """Exactly two, and they must refuse rather than invent one.

        The bridge reaches both upstream and serves neither inbound, so a
        transport on either names its own route. Guessing would drive a request
        the product never receives.

        Args:
            fmt: A format with no inbound route.
        """
        with pytest.raises(KeyError):
            protocol_for(fmt)


# ── R4 — the conformance check ───────────────────────────────────────────────


class TestTheConformanceCheck:
    """The claim every transport, present and future, is judged by."""

    @pytest.mark.parametrize("fmt", SERVED)
    async def test_the_registered_transport_passes(self, fmt: WireFormat) -> None:
        """The positive control.

        Args:
            fmt: A format the primary recorder serves.
        """
        await assert_transport_reaches_its_recorder(transport("aiohttp", fmt))

    @pytest.mark.parametrize(
        ("module", "name", "fmt", "route"),
        list(CONFORMANCE_CASES),
        ids=[name for _module, name, _fmt, _route in CONFORMANCE_CASES],
    )
    async def test_every_registered_transport_passes(
        self, module: str, name: str, fmt: WireFormat, route: InboundProtocol
    ) -> None:
        """The meta-test: a transport inherits the check by registering.

        Each row carries its own format and inbound route, so an Epic B author
        adds a row and is done. A single hardcoded format here would raise
        ``ValueError`` at construction for every one of T-B1, T-B2 and T-B3 —
        none of them serves Anthropic Messages — which would make "inherits by
        registering" false for all three transports it was written for.

        The row's module is imported here, not assumed imported. A name enters
        the registry only when something imports the module that registers it,
        and this module imports only ``harness.bridge`` — so every row but the
        core's own would fail with ``LookupError`` under a selective run, and
        under a full run would depend on collection order. That is the same
        reasoning the completeness test below states, applied to the test that
        actually drives the check.

        Args:
            module: The module that registers this transport.
            name: A registered transport name.
            fmt: The upstream format to construct it with.
            route: The inbound route that exercises it.
        """
        import importlib

        importlib.import_module(module)

        await assert_transport_reaches_its_recorder(transport(name, fmt), protocol=route)

    def test_the_meta_test_iterates_a_complete_set_not_whatever_was_imported(self) -> None:
        """ "Non-empty" would report success for a category it never ran.

        A name enters the registry only when something imports the registering
        module, so under a selective run the meta-test above could iterate a set
        of one and pass. Every row's module is imported explicitly and the
        registered set is asserted **equal** — the principle `tests/conftest.py`
        states: "a job that reports success without running a category it claims
        is worse than one that fails".
        """
        import importlib

        for module, _name, _fmt, _route in CONFORMANCE_CASES:
            importlib.import_module(module)

        assert set(registered_transports()) == EXPECTED_TRANSPORTS, (
            "a transport is registered that no CONFORMANCE_CASES row covers, or a row "
            "names a transport nothing registered; add the row beside the registration"
        )

    def test_two_markers_differ(self) -> None:
        """A capture left by an earlier fixture must not satisfy a later assertion."""
        assert marker() != marker()
