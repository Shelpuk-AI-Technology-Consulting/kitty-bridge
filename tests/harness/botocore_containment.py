"""The botocore containment transport: the T-E4 slice's drive surface.

`.system_design/TEST_SUITE.md` §5.2.1, §5.2.2, §5.3, §5.5 · plan task **T-E4**
([KBR-64](https://shelpuk.atlassian.net/browse/KBR-64)).

What this module owns:

* **`BotocoreContainment`** — the ``ContainmentTransport`` implementation for
  the ``bedrock`` adapter's botocore client. Its ``drive_phase_1`` /
  ``drive_with_egress`` shape mirrors
  :class:`harness.containment.BridgeAiohttpContainment`, and every method is
  deliberately the botocore twin of its aiohttp sibling.

The **botocore-specific facts** the twins diverge on:

* ``BedrockAdapter._get_boto3_client`` reads the process-wide egress from
  ``kitty.egress.get_egress()`` — not ``BridgeServer._egress`` — so the
  egress-on drives call :func:`kitty.egress.set_egress`, not the bridge
  constructor's ``egress=`` keyword. ``tests/conftest.py``'s autouse
  ``_reset_egress`` clears it after every test either way.
* The direct leg for botocore is the same ``socket.getaddrinfo`` patch
  T-E1 shipped for bridge-aiohttp: botocore's ``urllib3`` reaches
  ``socket.create_connection`` → ``socket.getaddrinfo`` for the harness
  hostname, and the harness maps it to the recorder's loopback port. The
  proxied leg needs no client-side resolution (§5.3: the harness is the
  proxy and owns the resolver), so the patch is only applied around the
  direct leg.
* The TLS hop to the recorder needs the harness CA in the boto3 client's
  trust store. ``AWS_CA_BUNDLE`` is the botocore-documented channel, so the
  drives set it around the drive; the recorder's leaf certificate carries
  ``HARNESS_UPSTREAM_HOST`` in its SAN (see
  :attr:`SealedNetwork.upstream_base_url`).
* The falsification control patches ``kitty.providers.bedrock.get_egress``,
  not ``kitty.egress``: the bedrock adapter resolves its own module binding
  at import time, so a patch on the egress module is silently ignored — the
  exact fail-by-silence shape
  ``test_aiohttp_containment_slice.py::TestPhase3Falsification`` documents
  for ``should_bypass`` on the aiohttp leg.
"""

from __future__ import annotations

import asyncio
import contextlib
import ssl
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar

import aiohttp

from harness.botocore_recorder import BedrockRecordingUpstream
from harness.containment import (
    _DRIVE_TIMEOUT,
    Phase1Result,
    SealedNetwork,
    monkeypatched_aiohttp_resolver,
    register_containment_transport,
)
from harness.contract import WireFormat

if TYPE_CHECKING:
    import pytest

    from kitty.bridge.server import BridgeServer
    from kitty.egress import EgressConfig

__all__ = ["BotocoreContainment"]


class BotocoreContainment:
    """The ``ContainmentTransport`` for the ``bedrock`` adapter's botocore client.

    Attributes:
        name: The registry key, ``"botocore"``.
    """

    #: A class attribute, ``"botocore"``. Matches
    #: :attr:`harness.bridge.BotocoreTransport.name` and the capability
    #: report's fourth row.
    name = "botocore"

    #: Same model the bridge-aiohttp slice uses; the bedrock adapter's
    #: ``translate_to_upstream`` pops it into ``modelId``, so the exact value
    #: does not enter the containment assertions.
    _MODEL: ClassVar[str] = "harness-model"

    #: The inbound route the botocore leg is driven through. The bridge's
    #: ``/v1/chat/completions`` route reaches the bedrock adapter (the bedrock
    #: adapter is a Chat-Completions-wire adapter upstream, and the bridge
    #: never serves Bedrock Converse inbound — §7.5.1). Same value
    #: ``harness.test_botocore`` uses.
    _ROUTE: ClassVar[str] = "/v1/chat/completions"

    @contextlib.contextmanager
    def direct_route(self, harness: SealedNetwork) -> Iterator[None]:
        """Yield once — the socket-level resolver patch lives on the drives.

        The bridge-aiohttp twin applies its ``socket.getaddrinfo`` patch
        inside ``drive_phase_1`` / ``drive_with_egress`` so the
        ``resolver_port`` falsification seam stays on the method that uses
        it; this module does the same, so the direct route itself has nothing
        to apply. T-E3..T-E5's twins (curl_cffi's ``--resolve``,
        botocore's ``endpoint_url``) reach through their ``bind()``
        mechanisms instead.

        Args:
            harness: The sealed network the request will be driven against.

        Yields:
            ``None``.
        """
        yield

    @contextlib.contextmanager
    def _botocore_trusts_test_ca(self, monkeypatch: pytest.MonkeyPatch, ca_path: str) -> Iterator[None]:
        """Point the botocore client at the harness CA on **both** TLS legs.

        botocore verifies two TLS hops when an ``https://`` proxy fronts an
        ``https://`` target, and they read their CA bundle from different
        places:

        * **Target hop** (the recorder at ``https://upstream.kitty-test.invalid``)
          reads the ``verify`` value that
          ``session.create_client`` resolved from the ``AWS_CA_BUNDLE``
          environment variable at client construction. ``monkeypatch.setenv``
          covers this leg.
        * **Proxy hop** (the harness CONNECT proxy) reads
          ``proxies_config={'proxy_ca_bundle': ...}``, a completely separate
          Config setting (``botocore/httpsession.py:397-419``,
          ``_setup_proxy_ssl_context``). The default is ``None``, which means
          the proxy hop falls back to urllib3's default SSL context — certifi,
          not the harness CA — and the TLS handshake to the harness proxy
          fails with ``SSL: WRONG_VERSION_NUMBER``. The wrapper below injects
          the bundle whenever ``proxies=`` is set.

        Args:
            monkeypatch: Pytest's monkeypatch fixture.
            ca_path: The path of the harness CA certificate to trust on both
                hops.

        Yields:
            ``None``.
        """
        monkeypatch.setenv("AWS_CA_BUNDLE", ca_path)
        with self._patched_botocore_config(monkeypatch, ca_path=ca_path):
            yield

    @contextlib.contextmanager
    def _patched_botocore_config(
        self, monkeypatch: pytest.MonkeyPatch, *, ca_path: str | None = None
    ) -> Iterator[None]:
        """Patch ``botocore.config.Config`` to add a one-attempt retry bound.

        botocore's default legacy retry policy makes up to five attempts with
        exponential backoff — worst-case ~15 s of sleeps per failed request
        (measured on the resolved botocore (1.43.94):
        ``botocore/data/_retry.json`` pins
        ``max_attempts: 5`` in legacy mode and
        ``botocore/retryhandler.py``'s ``delay_exponential`` produces
        ``rand() * 2^attempt`` delays). The phase-2 negative assertion (proxy
        stopped mid-test) drives exactly that refusal shape: a five-attempt
        backoff would spend the phase's budget testing botocore's backoff
        rather than containment. One attempt is the product's own honest
        failure, and the drive timeout bounds it.

        The patch site is ``botocore.config.Config`` itself, because
        ``BedrockAdapter._get_boto3_client`` imports
        ``from botocore.config import Config as _BotoConfig`` **inside the
        method body** — a fresh lookup per call — so patching the module
        attribute reaches it. A subclass override would have to rebuild the
        whole client; this patch does not.

        Args:
            monkeypatch: Pytest's monkeypatch fixture; the patch reverts on
                its teardown.
            ca_path: When set, injects ``proxies_config={'proxy_ca_bundle':
                ca_path}`` into every ``Config`` that carries ``proxies=``,
                so the **proxy** TLS hop trusts the harness CA. See
                :meth:`_botocore_trusts_test_ca` for why this is a separate
                setting from ``AWS_CA_BUNDLE``.

        Yields:
            ``None``.
        """
        import botocore.config

        original_config = botocore.config.Config

        def _bounded_config(*args: object, **kwargs: object) -> object:
            """Return a ``Config`` with the retry bound and proxy CA applied.

            botocore resolves the retry count through
            ``Client._register_retries`` (``botocore/client.py:262``) by
            reading ``config.retries.get('total_max_attempts')``. The
            legacy key ``max_attempts`` is transformed into
            ``total_max_attempts = max_attempts + 1`` by
            ``_transform_legacy_retries`` (``botocore/client.py:302-311``),
            so ``max_attempts: 1`` becomes ``total_max_attempts: 2`` — exactly
            one initial attempt and one retry, matching the harness's intent.

            The bound must also keep ``mode``: ``botocore/client.py:249``
            reads ``config.retries['mode']`` directly, and botocore's
            internal ``Config(...)`` call (the second of two during
            ``create_client``) carries ``retries={'total_max_attempts': 2,
            'mode': 'legacy'}``. Dropping ``total_max_attempts`` from the
            merged retries is therefore the seam that makes our bound win:
            with both keys present, botocore prefers ``total_max_attempts``
            and the bound is silently ignored.

            Args:
                *args: Positional arguments for ``botocore.config.Config``.
                **kwargs: Keyword arguments; the caller's ``retries`` (if
                    any) is preserved except for ``total_max_attempts``,
                    which is replaced with the bound.

            Returns:
                The constructed Config.
            """
            existing_retries = dict(kwargs.get("retries") or {})
            existing_retries.pop("total_max_attempts", None)
            bounded_retries = {**existing_retries, "max_attempts": 1}
            merged = {**kwargs, "retries": bounded_retries}
            if ca_path is not None and merged.get("proxies"):
                existing_proxies_config = dict(merged.get("proxies_config") or {})
                existing_proxies_config.setdefault("proxy_ca_bundle", ca_path)
                merged["proxies_config"] = existing_proxies_config
            return original_config(*args, **merged)  # type: ignore[arg-type]

        monkeypatch.setattr(botocore.config, "Config", _bounded_config)
        yield

    async def drive_phase_1(
        self,
        harness: SealedNetwork,
        *,
        monkeypatch: pytest.MonkeyPatch,
        resolver_port: int | None = None,
    ) -> Phase1Result:
        """Drive one request through the bridge's botocore leg with egress off.

        Args:
            harness: The sealed network the request is driven against.
            monkeypatch: Pytest's monkeypatch fixture; the resolver patch
                reverts on its teardown.
            resolver_port: The port the direct-leg resolver maps the harness
                hostname to. The default — ``harness.upstream_port`` — sends
                the bridge to the recorder. The falsification case passes a
                closed port.

        Returns:
            A :class:`Phase1Result` with the bridge's status, the recorder's
            captures and connections after the request, and the proxy's
            attempts. With egress off and the default resolver port,
            ``status == 200``, ``len(captures) == 1``, ``attempts == []``.
        """
        from kitty.bridge.server import BridgeServer
        from kitty.egress import set_egress
        from kitty.providers.bedrock import BedrockAdapter

        target_port = harness.upstream_port if resolver_port is None else resolver_port

        # Egress is cleared explicitly rather than assumed: `_reset_egress`'s
        # autouse reset in `tests/conftest.py` runs *before* the test, and a
        # test that enables egress later in the same test would leak into the
        # drive. Clearing here is the same one-line seam
        # `BridgeAiohttpContainment.drive_phase_1` relies on implicitly.
        set_egress(None)

        with (
            self.direct_route(harness),
            monkeypatched_aiohttp_resolver(monkeypatch, harness.upstream_host, target_port),
            self._botocore_trusts_test_ca(monkeypatch, harness.ca_path),
        ):
            # The bedrock adapter is constructed directly rather than through
            # `BridgeFixture`: `BotocoreTransport.bind()` returns the
            # `BedrockAdapter` and the `provider_config` naming the recorder
            # by its loopback URL. §5.3's whole point is that the bridge
            # reaches the harness **by its non-loopback name**, so the
            # `provider_config["endpoint_url"]` is spelled with the harness
            # hostname, not the recorder's own.
            adapter = BedrockAdapter()
            server = BridgeServer(
                None,  # type: ignore[arg-type]
                adapter,
                resolved_key="harness-access-key:harness-secret-key",
                model=self._MODEL,
                provider_config={
                    "endpoint_url": harness.upstream_base_url,
                    "region": "us-east-1",
                },
            )
            return await self._drive(server, harness)

    async def drive_with_egress(
        self,
        harness: SealedNetwork,
        *,
        egress: EgressConfig,
        monkeypatch: pytest.MonkeyPatch,
        resolver_port: int | None = None,
    ) -> Phase1Result:
        """Drive one request through the bridge's botocore leg with egress on.

        Args:
            harness: The sealed network the request is driven against.
            egress: The configuration to install process-wide; the bedrock
                adapter's ``_get_boto3_client`` reads it back at client
                construction and turns it into ``botocore.config.Config(proxies=...)``.
            monkeypatch: Pytest's monkeypatch fixture; the resolver patch
                reverts on its teardown.
            resolver_port: The port the direct-leg resolver maps the harness
                hostname to. Default is ``harness.upstream_port``.

        Returns:
            A :class:`Phase1Result` with the recorder's observations. The
            caller is responsible for asserting on them.
        """
        from kitty.bridge.server import BridgeServer
        from kitty.egress import set_egress
        from kitty.providers.bedrock import BedrockAdapter

        target_port = harness.upstream_port if resolver_port is None else resolver_port

        # The egress must be in scope before the drive, because the bedrock
        # adapter reads it at client construction. The `monkeypatch` fixture
        # does not know about the process-wide egress global —
        # `tests/conftest.py`'s autouse `_reset_egress` clears it at the
        # next test's start, which is the only teardown this drive needs.
        set_egress(egress)

        with (
            self.direct_route(harness),
            monkeypatched_aiohttp_resolver(monkeypatch, harness.upstream_host, target_port),
            self._botocore_trusts_test_ca(monkeypatch, harness.ca_path),
        ):
            adapter = BedrockAdapter()
            server = BridgeServer(
                None,  # type: ignore[arg-type]
                adapter,
                resolved_key="harness-access-key:harness-secret-key",
                model=self._MODEL,
                provider_config={
                    "endpoint_url": harness.upstream_base_url,
                    "region": "us-east-1",
                },
            )
            return await self._drive(server, harness)

    async def _drive(self, server: BridgeServer, harness: SealedNetwork) -> Phase1Result:
        """POST one minimal chat-completions turn through the started bridge.

        Args:
            server: The started bridge.
            harness: The sealed network the request is driven against.

        Returns:
            A :class:`Phase1Result` with the bridge's status, the recorder's
            captures and connections after the request, and the proxy's
            attempts.
        """
        status: int = -1
        text: str = ""
        try:
            bridge_port = await server.start_async()

            # A chat-completions body, the route the bedrock adapter serves.
            body = {
                "model": self._MODEL,
                "messages": [{"role": "user", "content": f"kbr64-{uuid.uuid4().hex}"}],
                "stream": False,
            }

            async with aiohttp.ClientSession() as client:
                response = await client.post(
                    f"http://127.0.0.1:{bridge_port}{self._ROUTE}",
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=_DRIVE_TIMEOUT),
                )
                text = await response.text()
                status = response.status
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            status = -1
            text = repr(exc)
        finally:
            await server.stop_async()

        return Phase1Result(
            status=status,
            text=text,
            captures=list(harness.recorder.requests),
            connections=list(harness.recorder.connections),
            attempts=list(harness.proxy.attempts),
        )


# ── Recorder lifecycle adapter ─────────────────────────────────────────────
#
# :class:`SealedNetwork`'s factory shape (main's T-E3) calls
# ``await recorder.start()`` with no arguments on a factory-built recorder.
# :class:`BedrockRecordingUpstream` inherits
# ``start(ssl_context: SSLContext | None = None)`` from
# :class:`RecordingUpstream`, whose TLS context is supplied at start time,
# not at construction. The wrapper below captures the harness's
# ``SSLContext`` at construction so the harness's factory contract and the
# bedrock recorder's lifecycle signature stay in agreement without touching
# the upstream botocore recorder module.


class _TlsBedrockRecordingUpstream(BedrockRecordingUpstream):
    """A ``BedrockRecordingUpstream`` whose TLS context is bound at construction.

    The parent class takes the ``SSLContext`` at ``start()``; the harness's
    factory shape passes the context at construction and expects
    ``start()`` to take no arguments. The wrapper closes the gap by
    capturing the context here and forwarding it inside ``start()`` — the
    recording path, the harness's TLS-handling path and the
    ``bedrock-recorder-as-recordable-target`` contract all stay unchanged.

    Attributes:
        ssl_context: The captured ``SSLContext`` the harness built from
            ``certs.target_cert`` / ``certs.target_key``; applied at
            ``start()``.
    """

    ssl_context: ssl.SSLContext | None = None

    def __init__(self, *, ssl_context: ssl.SSLContext, default_format: WireFormat) -> None:
        """Store ``ssl_context`` for later application at ``start()``.

        Args:
            ssl_context: The harness's TLS context; applied at start time.
            default_format: The served format, forwarded to the parent
                constructor — which validates it against this recorder's
                served set. The wrapper deliberately does not silently
                override the caller's value: the parent's ``__post_init__``
                rejects any format the bedrock recorder does not serve, so
                a caller passing an unexpected format fails loudly here
                rather than being quietly redirected.
        """
        super().__init__(default_format=default_format)
        self.ssl_context = ssl_context

    async def start(self, *args: object, **kwargs: object) -> None:
        """Apply the captured ``SSLContext`` and start the listener.

        Args:
            *args: Unused; required only to satisfy the harness's
                ``recorder.start()`` (no-arguments) contract.
            **kwargs: Unused; ``ssl_context`` is taken from
                :attr:`ssl_context` instead.
        """
        await super().start(ssl_context=self.ssl_context)  # type: ignore[arg-type]


# ── Registration ───────────────────────────────────────────────────────────
#
# Following the same pattern :mod:`harness.botocore` uses for the bridge
# fixture's transport registry: the class is registered where it is defined
# so a slice that imports :mod:`harness.botocore_containment` directly
# receives the registered transport without needing a side-effecting import
# of :mod:`harness.containment`. A circular import would otherwise arise
# because this module already imports :class:`SealedNetwork`,
# :func:`monkeypatched_aiohttp_resolver` and :class:`Phase1Result` from it.


__all__ = ["BotocoreContainment", "_TlsBedrockRecordingUpstream"]


register_containment_transport(BotocoreContainment.name, BotocoreContainment)
