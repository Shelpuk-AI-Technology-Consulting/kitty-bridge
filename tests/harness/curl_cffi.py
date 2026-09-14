"""The curl_cffi transport, and the two seams that reach its recorder.

`.system_design/TEST_SUITE.md` §5.5, §7.2, §7.2.3, §7.5 · plan task **T-B2**
(KBR-41).

:mod:`harness.curl_recorder` is the server; this module is how the product is
pointed at it, and it is a different job from T-B1's for a structural reason:
``OpenAISubscriptionAdapter`` is one of the three custom-transport adapters that
never read ``provider_config["base_url"]`` (§7.5.2). Its upstream path is a
module constant, so pointing it at the recorder is a **module-constant swap**,
not a ``build_base_url`` override — the same shape as T-B1's OAuth seam, applied
to a constant that lives on :mod:`kitty.providers.openai_subscription`.

Two constants are swapped, and they are deliberately kept in two seams so that
neither swap can reach the other leg:

* :func:`codex_backend_url` — the serving leg's ``_CODEX_BACKEND_URL``;
* :func:`oauth_refresh_endpoint` — the refresh leg's
  ``kitty.auth.oauth_session.OAUTH_TOKEN_URL``, a **different** constant from
  the login leg's (§7.2.2 records the trap).

Both follow T-B1's rule: read the constant before writing it, restore in a
``finally``, and raise ``AttributeError`` when the name is gone — so a leg
rewritten to read its endpoint elsewhere fails loudly rather than leaving a test
to pass with an empty capture list.

**The transport is also where body redaction lives** (KBR-25's scope question,
decided per-transport): the refresh leg carries ``client_secret`` and
``refresh_token`` in its body, and a capture rendered in an assertion diff or a
CI log would publish them. :attr:`CurlCffiTransport.captures` returns masked
copies; :attr:`~harness.recorder.RecordingUpstream.requests` stays raw, and the
three transports that already implement the protocol are unaffected.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode

from harness.bridge import Binding, register_transport
from harness.contract import CapturedRequest, WireFormat
from harness.curl_recorder import (
    CODEX_RESPONSES_SUFFIX,
    OAUTH_REFRESH_SUFFIX,
    CurlRecordingUpstream,
)
from harness.recorder import ConnectionRecord, Responder
from kitty.auth import oauth_session
from kitty.providers import openai_subscription
from kitty.providers.openai_subscription import OpenAISubscriptionAdapter

__all__ = [
    "CurlCffiTransport",
    "REDACTED_FORM_FIELDS",
    "codex_backend_url",
    "oauth_refresh_endpoint",
    "redact_oauth_form_body",
]


#: The form fields the OAuth refresh leg sends whose values are credentials.
#: Anything else in a captured token grant body — ``grant_type``,
#: ``client_id`` — is not secret and stays visible, so a redacted capture still
#: reads as the request it was.
REDACTED_FORM_FIELDS = frozenset({"client_secret", "refresh_token"})

#: What a redacted field's value is replaced with. Chosen so a diff line
#: carrying it cannot be confused with the original: the length changes with
#: the redaction, and no real credential is four asterisks long.
_REDACTED = "***"


def redact_oauth_form_body(body: bytes) -> bytes:
    """Mask OAuth form credentials in a captured token-grant body.

    Form-encoded is the only shape the refresh leg sends, and the only shape
    that carries credentials here — so a body that does not parse as a form
    passes through **unchanged**, and a form whose fields are all public passes
    through too. Malformed input is never an error: T-C6's malformed body must
    reach a reader untouched, and this function is on that path.

    Args:
        body: The captured body bytes.

    Returns:
        The body with every :data:`REDACTED_FORM_FIELDS` value replaced by
        :data:`_REDACTED`, or the original bytes when nothing matched.
    """
    text = body.decode("utf-8", errors="replace")
    pairs = parse_qsl(text, keep_blank_values=True)
    if not pairs:
        return body

    changed = False
    out: list[tuple[str, str]] = []
    for name, value in pairs:
        if name in REDACTED_FORM_FIELDS and value:
            out.append((name, _REDACTED))
            changed = True
        else:
            out.append((name, value))

    if not changed:
        return body

    # Preserve the original encoding choices as closely as `urlencode` allows:
    # the fields a reader cares about are the names, not the quoting. The
    # `safe="*"` keeps the redaction marker literal — a diff that renders
    # `%2A%2A%2A` reads as corruption, not as a credential that was removed.
    return urlencode(out, quote_via=quote, safe="*").encode("utf-8")


@dataclass
class CurlCffiTransport:
    """The transport for the adapter that owns the impersonating curl session.

    Attributes:
        format: The upstream wire format served. Only
            :attr:`~harness.contract.WireFormat.OPENAI_RESPONSES`; the recorder
            rejects anything else at construction.
        ssl_context: The server-side TLS context the recorder terminates with,
            built from the harness ``certs`` fixture through
            :func:`harness.connect_proxy.server_ssl_context`.
        responder: What to reply with; the recorder's minimal success when
            omitted. A closure over mutable state is how a reply is scripted to
            change between attempts.
        ca_cert: Path to the harness CA the recorder's certificate is signed
            by, exported to the adapter through its own ``CODEX_CA_CERTIFICATE``
            seam. ``None`` when the recorder is used without TLS, which only a
            test driving raw HTTP would do.
    """

    #: A class attribute, not a field: every instance answers to one registry key.
    name = "curl_cffi"

    format: WireFormat
    ssl_context: Any = None
    responder: Responder | None = None
    ca_cert: Path | None = None
    _recorder: CurlRecordingUpstream = field(init=False, repr=False)
    _adapter: OpenAISubscriptionAdapter | None = field(init=False, default=None, repr=False)
    _codex_seam: Any = field(init=False, default=None, repr=False)
    _refresh_seam: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Build the recorder eagerly, so an unserved format fails here.

        Raises:
            ValueError: When ``format`` is not one this recorder serves.
        """
        self._recorder = CurlRecordingUpstream(
            ssl_context=self.ssl_context, default_format=self.format, responder=self.responder
        )

    @property
    def recorder(self) -> CurlRecordingUpstream:
        """Return the underlying recorder.

        Returns:
            The :class:`~harness.curl_recorder.CurlRecordingUpstream`, for tests
            that need the parts of it the transport interface does not expose.
        """
        return self._recorder

    async def start(self) -> None:
        """Bind an ephemeral loopback port and begin recording.

        **The two seams are entered here**, not left to a test to drive, for
        the reason §7.5.4 records: ``assert_transport_reaches_its_recorder``
        starts the transport itself and cannot know about a transport's
        redirect mechanism. With the seams inside ``start``, the conformance
        check works unchanged, and so does every fixture consumer.

        Raises:
            RuntimeError: When called on an already-started recorder.
            Exception: Whatever the seams' entry raised, after releasing the
                recorder. A partially-entered pair of context managers must
                not leave a constant swapped when the transport fails to start.
        """
        await self._recorder.start()
        try:
            self._codex_seam = codex_backend_url(self._recorder)
            self._refresh_seam = oauth_refresh_endpoint(self._recorder)
            self._codex_seam.__enter__()
            self._refresh_seam.__enter__()
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Close the adapter's sessions, restore the env, then release the port.

        The OpenAI subscription adapter owns **two** connection pools — the
        serving leg's and the OAuth refresh leg's
        (:attr:`~kitty.providers.openai_subscription.OpenAISubscriptionAdapter._oauth_curl_session`)
        — and :meth:`~kitty.providers.base.ProviderAdapter.aclose` releases
        both. The ``CODEX_CA_CERTIFICATE`` environment variable the adapter
        reads is unset so a test that runs after this one is not poisoned by
        the harness CA. The recorder is stopped in a ``finally``: a transport
        that fails to close must not leave a port bound, because the next
        test's ephemeral port allocation is the only thing that would notice.
        """
        # The seams exit before anything can observe a half-restored state.
        for seam in (self._refresh_seam, self._codex_seam):
            if seam is not None:
                seam.__exit__(None, None, None)
        self._refresh_seam = None
        self._codex_seam = None
        try:
            if self._adapter is not None:
                await self._adapter.aclose()
        finally:
            _set_env("CODEX_CA_CERTIFICATE", None)
            await self._recorder.stop()

    def bind(self) -> Binding:
        """Return the adapter and config that reach this recorder.

        **One adapter per transport, reused**, for T-B1's reason: it owns two
        session pools that :meth:`stop` has to close.

        The adapter reads its CA from the ``CODEX_CA_CERTIFICATE`` environment
        variable (Codex CLI's ``custom_ca.rs`` precedence: it wins over
        ``SSL_CERT_FILE``), not from ``provider_config`` — so ``bind()`` exports
        the harness CA to that variable before the adapter's first request.
        The upstream URL itself is **not** in the config (the adapter ignores
        ``base_url``) and is pointed at the recorder by
        :func:`codex_backend_url`, which the test drives for the lifetime of
        the transport.

        Returns:
            The ``openai_subscription`` adapter and an empty provider config —
            the adapter resolves the harness CA from the
            ``CODEX_CA_CERTIFICATE`` environment variable above, and ignores
            ``provider_config["base_url"]`` (§7.5.2's custom-transport rule),
            so there is nothing for a config dict to carry.

        Raises:
            RuntimeError: When the transport has not been started, because the
                recorder has no port until then.
        """
        if self.ca_cert is not None:
            _set_env("CODEX_CA_CERTIFICATE", str(self.ca_cert))
        if self._adapter is None:
            self._adapter = OpenAISubscriptionAdapter()
        return self._adapter, {}

    @property
    def captures(self) -> Sequence[CapturedRequest]:
        """Return the completed captures, with OAuth credentials redacted.

        Returns:
            One :class:`~harness.contract.CapturedRequest` per completed
            request, in arrival order. A capture whose body carries a
            redactable form field is replaced by a **masked copy** — the
            recorder's own list, which
            :attr:`~harness.recorder.RecordingUpstream.requests` publishes, is
            never aliased, so it stays raw by construction. A capture whose
            body carries no redactable field is returned unchanged.
        """
        return [
            _with_redacted_body(captured) if _body_carries_a_credential(captured.body) else captured
            for captured in self._recorder.requests
        ]

    @property
    def connections(self) -> Sequence[ConnectionRecord]:
        """Return every accepted connection.

        Returns:
            One record per connection, including any that carried no request —
            §5.2.1's bypass shape, and the peer ports T-E5's tunnel join reads.
            A record whose peer port is ``-1`` is a connection whose TLS
            handshake never completed: §7.2.1's socket-level rule keeps it
            visible.
        """
        return self._recorder.connections

    def assert_teardown_clean(self) -> None:
        """Assert every request was answered in the format this transport declares.

        One check, not two, for T-B1's measured reason: this recorder serves one
        format, so a mis-declared format takes the fallback instead of silently
        dispatching to another suffix, and the conformance check catches it at
        4 captures and a timeout.

        Raises:
            UnmatchedPathError: When a request took the recorder's fallback.
        """
        self._recorder.assert_all_paths_matched()


def _body_carries_a_credential(body: bytes | None) -> bool:
    """Return whether a captured body carries a redactable OAuth form field.

    A cheap name-based pre-filter, so the parse-and-rebuild cost of
    :func:`redact_oauth_form_body` is paid only by the captures that need it.

    Args:
        body: The captured body bytes, or ``None`` for a request that had none.

    Returns:
        Whether any :data:`REDACTED_FORM_FIELDS` name appears in the body.
    """
    if not body:
        return False
    return any(name.encode() in body for name in REDACTED_FORM_FIELDS)


def _with_redacted_body(captured: CapturedRequest) -> CapturedRequest:
    """Return a copy of ``captured`` whose body has credentials masked.

    Args:
        captured: The capture to copy.

    Returns:
        A fresh ``CapturedRequest``; the argument is never mutated, so the
        recorder's own list stays raw by construction.
    """
    return replace(captured, body=redact_oauth_form_body(captured.body))


def _set_env(name: str, value: str | None) -> None:
    """Set ``name`` to ``value`` in the process environment.

    ``None`` deletes the variable — used by ``stop`` to undo what ``bind``
    did, so a test that runs after this one is not poisoned by the harness CA.

    Args:
        name: The variable name.
        value: The value, or ``None`` to unset.
    """
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


register_transport(CurlCffiTransport.name, CurlCffiTransport)


@contextlib.contextmanager
def codex_backend_url(recorder: CurlRecordingUpstream) -> Iterator[str]:
    """Point the serving leg's upstream constant at ``recorder``.

    ``OpenAISubscriptionAdapter`` reaches
    ``https://chatgpt.com/backend-api/codex/responses`` through a module
    constant with no configuration channel of any kind (§7.5.2's
    custom-transport rule) — so this is the only place it can be redirected,
    and it is the serving leg's ``bind()``.

    Reading the constant before writing it is what makes the seam falsifiable:
    if the adapter is ever rewritten to read the URL from somewhere else, this
    raises :class:`AttributeError` rather than swapping a name nothing consults
    and leaving a test to pass with an empty capture list.

    Args:
        recorder: A **started** recorder; its base URL is read here.

    Yields:
        The URL the leg will now post to.

    Raises:
        AttributeError: When
            :mod:`kitty.providers.openai_subscription` no longer defines
            ``_CODEX_BACKEND_URL``.
    """
    url = f"{recorder.base_url}{CODEX_RESPONSES_SUFFIX}"
    original: Any = openai_subscription._CODEX_BACKEND_URL
    openai_subscription._CODEX_BACKEND_URL = url
    try:
        yield url
    finally:
        # Restored in a `finally` rather than by pytest's monkeypatch, so the
        # seam is usable from anything — a containment slice driving the leg
        # outside a test function included.
        openai_subscription._CODEX_BACKEND_URL = original


@contextlib.contextmanager
def oauth_refresh_endpoint(recorder: CurlRecordingUpstream) -> Iterator[str]:
    """Point the OAuth refresh leg's token endpoint at ``recorder``.

    **A different constant from T-B1's seam** (§7.2.2): this swaps
    :mod:`kitty.auth.oauth_session`'s ``OAUTH_TOKEN_URL``, the refresh leg's;
    T-B1's :func:`~harness.provider_aiohttp.oauth_token_endpoint` swaps
    :mod:`kitty.auth.openai_oauth`'s, the login leg's. The two strings are
    identical and the variables are unrelated — swapping the wrong one sends a
    real request to ``auth.openai.com``, which is the failure this guard exists
    to prevent.

    Reading the constant before writing it is what makes the seam falsifiable:
    if the leg is ever rewritten to read the endpoint from somewhere else, this
    raises :class:`AttributeError` rather than swapping a name nothing consults
    and leaving a test to pass with an empty capture list.

    Args:
        recorder: A **started** recorder; its base URL is read here.

    Yields:
        The URL the leg will now post to.

    Raises:
        AttributeError: When :mod:`kitty.auth.oauth_session` no longer defines
            ``OAUTH_TOKEN_URL``.
    """
    url = f"{recorder.base_url}{OAUTH_REFRESH_SUFFIX}"
    original: Any = oauth_session.OAUTH_TOKEN_URL
    oauth_session.OAUTH_TOKEN_URL = url
    try:
        yield url
    finally:
        # Restored in a `finally` rather than by pytest's monkeypatch, so the
        # seam is usable from anything — a containment slice driving the leg
        # outside a test function included.
        oauth_session.OAUTH_TOKEN_URL = original
