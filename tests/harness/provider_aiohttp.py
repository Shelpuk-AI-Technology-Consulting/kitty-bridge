"""The provider-aiohttp transport, and the seam that redirects the OAuth leg.

`.system_design/TEST_SUITE.md` §5.5, §7.2, §7.5 · plan task **T-B1** (KBR-40).

:mod:`harness.provider_recorder` is the server; this module is how the product
is pointed at it. Two products, two mechanisms, because the two paths differ in
what they expose:

* ``ollama_cloud`` is a :class:`~kitty.providers.base.ProviderAdapter` the
  bridge drives, so it plugs into T-W8's ``UpstreamTransport`` interface and
  inherits :func:`~harness.bridge.assert_transport_reaches_its_recorder` by
  registering. It honours ``provider_config["base_url"]`` — resolved inside its
  own request path rather than through ``build_base_url``, which is why
  :func:`~harness.bridge.redirected` does not apply to it and ``bind()`` does.
* The OpenAI **login** OAuth token leg has no adapter and no
  ``provider_config``: it is driven from the CLI at login time and reaches a
  module constant. :func:`oauth_token_endpoint` is its equivalent of ``bind()``.

**Why the OAuth leg is not an ``UpstreamTransport`` (§7.5's open question).**
``UpstreamTransport.bind()`` must return ``(adapter, provider_config)`` and the
conformance check drives one request through a real ``BridgeServer``; the login
leg has neither an adapter nor a bridge, so a transport for it could not satisfy
the interface it joined. The other rejected option was a seventh
:class:`~harness.contract.WireFormat`, which would mutate a contract six Epic A
readers consume (T-W2, KBR-25) to add a value **no projection can read** — a
form-encoded token grant is not an LLM request, and §3.3.1 pairs every format
with a wire reader. Serving the leg by path suffix and redirecting it here costs
no shared contract at all.

**Scope, after KBR-161.** §7.2 assigns this recorder "the ``openai_subscription``
OAuth token legs", written when both ran on aiohttp. Since KBR-161 the
**refresh** leg runs on the adapter's impersonating ``curl_cffi`` session
(:class:`kitty.auth.token_transport.CurlTokenTransport`, the only
``TokenTransport`` implementation), which an aiohttp recorder cannot observe;
§5.5 records the split. So this module covers the **login** leg, and the refresh
leg belongs to T-B2's TLS-terminating recorder.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from harness.bridge import Binding, register_transport
from harness.contract import CapturedRequest, WireFormat
from harness.provider_recorder import (
    OAUTH_TOKEN_SUFFIX,
    ProviderRecordingUpstream,
)
from harness.recorder import ConnectionRecord, Responder
from kitty.auth import openai_oauth
from kitty.providers.ollama_cloud import OllamaCloudAdapter

__all__ = [
    "ProviderAiohttpTransport",
    "oauth_token_endpoint",
]


@dataclass
class ProviderAiohttpTransport:
    """The transport for adapters that own their aiohttp session.

    Attributes:
        format: The upstream wire format served. Only
            :attr:`~harness.contract.WireFormat.OLLAMA_CHAT`; the recorder
            rejects anything else at construction.
        responder: What to reply with; the recorder's minimal success when
            omitted. A closure over mutable state is how a reply is scripted to
            change between attempts.
    """

    #: A class attribute, not a field: every instance answers to one registry key.
    name = "provider_aiohttp"

    format: WireFormat
    responder: Responder | None = None
    _recorder: ProviderRecordingUpstream = field(init=False, repr=False)
    _adapter: OllamaCloudAdapter | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Build the recorder eagerly, so an unserved format fails here.

        Raises:
            ValueError: When ``format`` is not one this recorder serves.
        """
        self._recorder = ProviderRecordingUpstream(default_format=self.format, responder=self.responder)

    @property
    def recorder(self) -> ProviderRecordingUpstream:
        """Return the underlying recorder.

        Returns:
            The :class:`~harness.provider_recorder.ProviderRecordingUpstream`,
            for tests that need the parts of it the transport interface does not
            expose — the OAuth captures among them.
        """
        return self._recorder

    async def start(self) -> None:
        """Bind an ephemeral loopback port and begin recording."""
        await self._recorder.start()

    async def stop(self) -> None:
        """Release the recorder's port, and close the adapter's own session.

        ``BridgeServer.stop_async`` closes the sessions **it** owns and knows
        nothing about an adapter that built its own, so without this the
        ``ollama_cloud`` session outlives every test that starts one — an
        "Unclosed client session" warning per test, and a real leak in a gate
        that runs thousands. Closing it here rather than in the product is
        deliberate: the product's session is a per-process pool whose lifetime is
        the process, and shortening it would be a change to the product to suit
        a test.

        The recorder is stopped in a ``finally``: a session that fails to close
        must not leave a port bound, because the next test's ephemeral port
        allocation is the only thing that would notice.
        """
        try:
            if self._adapter is not None:
                session = self._adapter._session
                if session is not None and not session.closed:
                    await session.close()
        finally:
            await self._recorder.stop()

    def bind(self) -> Binding:
        """Return the adapter and config that reach this recorder.

        **One adapter per transport, reused.** The primary transport returns a
        fresh adapter per call because its adapters are stateless; this one's
        owns a connection pool that :meth:`stop` has to close, and a fresh
        adapter per call would mean a session per call with only the last one
        reachable. Every balancing member sharing one adapter is the shape
        :func:`~harness.bridge.backend_for` already expects, which is why it
        takes a ``binding`` to pass around.

        Returns:
            The ``ollama_cloud`` adapter, and the provider configuration naming
            the recorder's base URL — the product's own channel, honoured by
            ``OllamaCloudAdapter._build_url``.

        Raises:
            RuntimeError: When the transport has not been started, because the
                recorder has no port until then.
        """
        base_url = self._recorder.base_url
        if self._adapter is None:
            self._adapter = OllamaCloudAdapter()
        return self._adapter, {"base_url": base_url}

    @property
    def captures(self) -> Sequence[CapturedRequest]:
        """Return the completed captures, in arrival order.

        Returns:
            What the recorder holds — including any OAuth token exchange, which
            arrives on the same server and is evidence of the same kind.
        """
        return self._recorder.requests

    @property
    def connections(self) -> Sequence[ConnectionRecord]:
        """Return every accepted connection.

        Returns:
            One record per connection, including any that carried no request —
            §5.2.1's bypass shape, and the peer ports T-E5's tunnel join reads.
        """
        return self._recorder.connections

    def assert_teardown_clean(self) -> None:
        """Assert every request was answered in the format this transport declares.

        **One check, not two, and that is measured rather than assumed.** The
        primary transport follows ``assert_all_paths_matched`` with a second
        pass catching a path that selected some *other* format — §7.5.4's row 4,
        whose whole point is that the capture list comes out complete and
        correct and nothing else reports it. That shape needs **two** served
        formats. This recorder serves one, so the same mistake takes the
        fallback instead: measured against a transport whose adapter posted
        elsewhere, the bridge could parse nothing out of the reply, retried, and
        the conformance check failed on **4 captures and a timeout** — loudly,
        13 seconds in, naming this transport. A second pass here would be an
        assertion no defect could falsify, which is exactly what §7.5.4 found
        and removed when the same repeat appeared in T-W8.

        **The OAuth endpoint is outside this claim, by name rather than by
        silence.** A token grant is answered before any format lookup (it has no
        :class:`~harness.contract.WireFormat` and must not be reported as a
        fallback), so a capture at that endpoint is not evidence for or against
        the declared format. Nothing a *bridge* drives can land there — the
        adapter's upstream path is a constant — and a transport whose adapter
        did was measured at the same 4 captures and a timeout.

        Raises:
            UnmatchedPathError: When a request took the recorder's fallback.
        """
        self._recorder.assert_all_paths_matched()


register_transport(ProviderAiohttpTransport.name, ProviderAiohttpTransport)


@contextlib.contextmanager
def oauth_token_endpoint(recorder: ProviderRecordingUpstream) -> Iterator[str]:
    """Point the OAuth login leg's token endpoint at ``recorder``.

    The login leg reaches ``https://auth.openai.com/oauth/token`` through a
    module constant, with no configuration channel of any kind — so this is the
    only place it can be redirected, and it is this leg's ``bind()``.

    **Two constants of that name exist, and this swaps one.**
    :mod:`kitty.auth.openai_oauth` holds the login leg's, and
    :mod:`kitty.auth.oauth_session` holds the **refresh** leg's. They are
    identical strings and unrelated variables. Redirecting the refresh leg
    through this function would swap a name that leg never reads and send its
    request to the real host; it runs on ``curl_cffi`` since KBR-161 and is
    T-B2's to redirect, over a recorder that terminates TLS.

    Reading the constant before writing it is what makes the seam falsifiable:
    if the leg is ever rewritten to read the endpoint from somewhere else, this
    raises :class:`AttributeError` rather than swapping a name nothing consults
    and leaving a test to pass with an empty capture list.

    Args:
        recorder: A **started** recorder; its base URL is read here.

    Yields:
        The URL the leg will now post to.

    Raises:
        AttributeError: When :mod:`kitty.auth.openai_oauth` no longer defines
            ``OAUTH_TOKEN_URL``.
    """
    url = f"{recorder.base_url}{OAUTH_TOKEN_SUFFIX}"
    original: Any = openai_oauth.OAUTH_TOKEN_URL
    openai_oauth.OAUTH_TOKEN_URL = url
    try:
        yield url
    finally:
        # Restored in a `finally` rather than by pytest's monkeypatch, so the
        # seam is usable from anything — a containment slice driving the leg
        # outside a test function included.
        openai_oauth.OAUTH_TOKEN_URL = original
