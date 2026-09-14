"""The curl_cffi transport: registration, binding, redaction, and the two seams.

`.system_design/TEST_SUITE.md` §5.5, §7.2, §7.2.3, §7.5 · plan task **T-B2**
(KBR-41) · `.requirements/20260914T184459Z_kbr_41_curl_cffi_recorder/`.

The integration conformance check, the OAuth refresh leg's redirect, and the
real-bridge round-trip live in this module. The recorder's own conformance and
TLS termination live in :mod:`test_curl_recorder`. The two test modules share
the recorder's lifecycle through its public surface.

**Layer.** No ``pytestmark``: the ``l1`` path default, for the reason
``test_provider_aiohttp.py`` records.
"""

from __future__ import annotations

import ssl
from pathlib import Path

import pytest

from harness.bridge import (
    BridgeFixture,
    InboundProtocol,
    UpstreamTransport,
    inbound_path,
    marker,
    minimal_inbound_body,
)
from harness.contract import CapturedRequest, WireFormat
from harness.curl_cffi import (
    REDACTED_FORM_FIELDS,
    CurlCffiTransport,
    codex_backend_url,
    oauth_refresh_endpoint,
    redact_oauth_form_body,
)

pytestmark = pytest.mark.usefixtures("certs")


@pytest.fixture
def server_context(certs) -> ssl.SSLContext:  # noqa: F811
    """Return the server-side TLS context the transport's recorder terminates with.

    Args:
        certs: The session-scoped throwaway certificate set.

    Returns:
        The context the recorder presents.
    """
    from harness.connect_proxy import server_ssl_context

    return server_ssl_context(certs.target_cert, certs.target_key)


@pytest.fixture
async def transport_(
    server_context: ssl.SSLContext, tmp_path: Path, certs
) -> CurlCffiTransport:
    """Build a started curl_cffi transport, one per test.

    Args:
        server_context: The TLS context the recorder terminates with.
        tmp_path: Where the harness CA file lives, so the adapter's sessions
            can trust it through ``CODEX_CA_CERTIFICATE``.
        certs: The session-scoped fixture, used to read the CA bytes.

    Yields:
        The started transport; stopped after the test.
    """
    ca_path = tmp_path / "kbr41-ca.pem"
    ca_path.write_bytes(_ca_bytes(certs))
    t = CurlCffiTransport(
        format=WireFormat.OPENAI_RESPONSES,
        ssl_context=server_context,
        ca_cert=ca_path,
    )
    await t.start()
    try:
        yield t
    finally:
        await t.stop()


def _ca_bytes(certs: object) -> bytes:
    """Return the harness CA's PEM bytes.

    Args:
        certs: The session-scoped fixture.

    Returns:
        The CA file's bytes — the same bytes the connecting adapter must
        trust for its ``verify=``.
    """
    return Path(str(certs.ca)).read_bytes()


class TestTheTransport:
    """Registration, lifecycle, and the seam's surface."""

    def test_the_default_transport_registers_curl_cffi(self) -> None:
        """``import harness.curl_cffi`` is enough to register the transport."""
        import harness.curl_cffi  # noqa: F401

        assert "curl_cffi" in registered()

    def test_a_format_this_transport_does_not_serve_is_refused_at_construction(
        self, server_context: ssl.SSLContext, tmp_path: Path
    ) -> None:
        """The base class would refuse it too, for the wrong reason.

        Args:
            server_context: The TLS context.
            tmp_path: Unused, fixture-scoped for parity.
        """
        with pytest.raises(ValueError, match="openai_responses"):
            CurlCffiTransport(
                format=WireFormat.ANTHROPIC_MESSAGES,
                ssl_context=server_context,
                ca_cert=tmp_path / "kbr41-ca.pem",
            )

    def test_bind_returns_one_adapter_for_the_transport_lifetime(
        self, transport_: CurlCffiTransport
    ) -> None:
        """Every ``bind()`` returns the same instance — ``stop()`` closes what ``bind()`` built.

        The OpenAI subscription adapter owns two connection pools, and
        :meth:`stop` closes both via :meth:`aclose` on the cached adapter. A
        regression that returned a fresh adapter per call would leak a session
        per call.

        Args:
            transport_: A started transport.
        """
        first, _ = transport_.bind()
        second, _ = transport_.bind()
        assert first is second


def registered() -> tuple[str, ...]:
    """Return the registered transport names.

    Returns:
        Every name currently in the transport registry. Imported lazily so the
        assertion does not require a top-level import.
    """
    from harness.bridge import registered_transports
    return registered_transports()


class TestRedaction:
    """R5: the transport's ``captures`` masks OAuth credentials, the recorder does not."""

    def test_form_fields_in_redacted_set_are_masked(self) -> None:
        body = b"grant_type=refresh_token&refresh_token=rt_abc&client_id=app_x&client_secret=cs_secret"
        out = redact_oauth_form_body(body)
        for name in REDACTED_FORM_FIELDS:
            assert f"{name}=***" in out.decode("utf-8")
        # Non-credential fields stay readable.
        assert "grant_type=refresh_token" in out.decode("utf-8")
        assert "client_id=app_x" in out.decode("utf-8")

    def test_public_form_is_unchanged(self) -> None:
        """A body without credential fields passes through byte-identical."""
        body = b"grant_type=refresh_token&client_id=app_x"
        assert redact_oauth_form_body(body) == body

    def test_malformed_body_is_unchanged(self) -> None:
        """T-C6's malformed body must reach a reader untouched."""
        body = b"not-form=%XX&garbage"
        assert redact_oauth_form_body(body) == body

    def test_captures_returns_fresh_objects_not_aliases(self, transport_: CurlCffiTransport) -> None:
        """``captures`` cannot share state with the recorder's own list.

        Args:
            transport_: A started transport.
        """
        # A capture filed directly in the recorder's slot list, so the test
        # exercises the transport's `captures` view against a known body
        # without dragging a bridge or an OAuth session into it.
        transport_.recorder._slots.append(CapturedRequest(  # noqa: SLF001 — the recorder's evidence store
            method="POST",
            scheme="https",
            host="127.0.0.1",
            path="/oauth/token",
            query="",
            headers=(),
            body=b"refresh_token=rt_visible",
            arrival=0.0,
            peer_port=None,
        ))
        assert len(transport_.captures) == 1
        masked = transport_.captures[0]
        # The recorder's own list is unchanged.
        assert b"rt_visible" in transport_.recorder.requests[0].body
        # The transport's view masks credentials.
        assert b"rt_visible" not in masked.body
        assert b"refresh_token=***" in masked.body
        # And it's a new object — mutating it does not feed back.
        assert masked is not transport_.recorder.requests[0]


class TestTheSeams:
    """Both constant swaps, and the trap §7.2.2 records."""

    async def test_the_serving_leg_seam_swaps_and_restores(
        self, transport_: CurlCffiTransport
    ) -> None:
        """`_CODEX_BACKEND_URL` points at the recorder inside, and out.

        Args:
            transport_: A started transport.
        """
        from kitty.providers import openai_subscription

        original = openai_subscription._CODEX_BACKEND_URL
        with codex_backend_url(transport_.recorder) as url:
            assert url == openai_subscription._CODEX_BACKEND_URL
        assert original == openai_subscription._CODEX_BACKEND_URL

    async def test_the_refresh_leg_seam_swaps_and_restores(
        self, transport_: CurlCffiTransport
    ) -> None:
        """`oauth_session.OAUTH_TOKEN_URL` points at the recorder inside, and out.

        Args:
            transport_: A started transport.
        """
        from kitty.auth import oauth_session

        original = oauth_session.OAUTH_TOKEN_URL
        with oauth_refresh_endpoint(transport_.recorder) as url:
            assert url == oauth_session.OAUTH_TOKEN_URL
        assert original == oauth_session.OAUTH_TOKEN_URL

    async def test_a_failing_body_still_restores_the_swap(
        self, transport_: CurlCffiTransport
    ) -> None:
        """The seam's ``finally`` is non-negotiable.

        Args:
            transport_: A started transport.
        """
        from kitty.auth import oauth_session

        original = oauth_session.OAUTH_TOKEN_URL
        with pytest.raises(RuntimeError), oauth_refresh_endpoint(transport_.recorder):
            raise RuntimeError("boom")
        assert original == oauth_session.OAUTH_TOKEN_URL

    async def test_a_seam_refuses_to_swap_a_name_nothing_defines(
        self, transport_: CurlCffiTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A leg rewritten to read its endpoint elsewhere fails loudly.

        Args:
            transport_: A started transport; the seam reads its ``base_url``,
                and the ``monkeypatch`` removes the constants the seams read.
            monkeypatch: Removes the constants the seams read.
        """
        from kitty.auth import oauth_session
        from kitty.providers import openai_subscription

        recorder = transport_.recorder
        monkeypatch.delattr(oauth_session, "OAUTH_TOKEN_URL")
        monkeypatch.delattr(openai_subscription, "_CODEX_BACKEND_URL")

        with pytest.raises(AttributeError), oauth_refresh_endpoint(recorder):
            pass  # pragma: no cover
        with pytest.raises(AttributeError), codex_backend_url(recorder):
            pass  # pragma: no cover


class TestThroughARealBridge:
    """The transport, integrated with a real ``BridgeServer`` and a real recorder."""

    async def test_it_satisfies_the_extension_interface(
        self, transport_: CurlCffiTransport
    ) -> None:
        """The eight members T-B1 checked are inherited unchanged.

        Args:
            transport_: A started transport.
        """
        assert isinstance(transport_, UpstreamTransport)
        assert transport_.name == "curl_cffi"
        assert transport_.format is WireFormat.OPENAI_RESPONSES

    async def test_a_request_reaches_the_recorder_as_openai_responses(
        self, server_context: ssl.SSLContext, tmp_path: Path, certs
    ) -> None:
        """One request, one capture, with the marker in the body.

        ``BridgeFixture`` starts its own transport, so this test builds a
        fresh transport it can hand over. The transport's own ``start``
        redirects both legs, so the fixture sees the redirect without the
        test driving the seams itself.

        Args:
            server_context: The TLS context.
            tmp_path: Where the OAuth session file lives.
            certs: The session-scoped fixture, for the real CA bytes.
        """
        sent = marker()
        oauth_key = _seed_oauth_session(tmp_path)
        ca_path = tmp_path / "kbr41-ca.pem"
        ca_path.write_bytes(_ca_bytes(certs))
        t = CurlCffiTransport(
            format=WireFormat.OPENAI_RESPONSES,
            ssl_context=server_context,
            ca_cert=ca_path,
        )

        async with BridgeFixture(t, key=oauth_key) as fixture:
            status, body = await fixture.post(
                inbound_path(InboundProtocol.RESPONSES),
                minimal_inbound_body(InboundProtocol.RESPONSES, sent),
            )

        assert status == 200, f"inbound response: {status} {body[:200]!r}"
        assert any(sent.encode() in c.body for c in t.captures), "the marker must reach the recorder's body"

    async def test_a_request_carries_marker_in_path_even_when_redacted(
        self, server_context: ssl.SSLContext, tmp_path: Path, certs
    ) -> None:
        """A non-credential body is unchanged in the transport's view.

        Args:
            server_context: The TLS context.
            tmp_path: Where the OAuth session file lives.
            certs: The session-scoped fixture, for the real CA bytes.
        """
        sent = marker()
        oauth_key = _seed_oauth_session(tmp_path)
        ca_path = tmp_path / "kbr41-ca.pem"
        ca_path.write_bytes(_ca_bytes(certs))
        t = CurlCffiTransport(
            format=WireFormat.OPENAI_RESPONSES,
            ssl_context=server_context,
            ca_cert=ca_path,
        )

        async with BridgeFixture(t, key=oauth_key) as fixture:
            await fixture.post(
                inbound_path(InboundProtocol.RESPONSES),
                minimal_inbound_body(InboundProtocol.RESPONSES, sent),
            )

        for capture in t.captures:
            assert sent.encode() in capture.body

    async def test_a_stale_token_refresh_reaches_the_recorder(
        self, server_context: ssl.SSLContext, tmp_path: Path, certs
    ) -> None:
        """A stale token forces the refresh leg to post onto the recorder.

        The seam-swap tests prove the constant is rewritten; this proves the
        product actually reads it. A regression where the URL moved into
        another module's namespace (a static import into ``openai_oauth``,
        say) would leave the seam swapping a name nothing consults, and every
        other test would still pass with an empty capture list.

        Args:
            server_context: The TLS context.
            tmp_path: Where the OAuth session file lives.
            certs: The session-scoped fixture, for the real CA bytes.
        """
        sent = marker()
        oauth_key = _seed_oauth_session(tmp_path, stale=True)
        ca_path = tmp_path / "kbr41-ca.pem"
        ca_path.write_bytes(_ca_bytes(certs))
        t = CurlCffiTransport(
            format=WireFormat.OPENAI_RESPONSES,
            ssl_context=server_context,
            ca_cert=ca_path,
        )

        async with BridgeFixture(t, key=oauth_key) as fixture:
            status, _body = await fixture.post(
                inbound_path(InboundProtocol.RESPONSES),
                minimal_inbound_body(InboundProtocol.RESPONSES, sent),
            )

        assert status == 200, "the serving request must reach the recorder"
        refresh_captures = [c for c in t.captures if c.path.endswith("/oauth/token")]
        assert refresh_captures, (
            "a stale token must have forced the refresh leg onto the recorder"
        )
        grant = refresh_captures[0]
        assert b"grant_type=refresh_token" in grant.body, "the request must be a refresh grant"
        # The transport's view masks credentials; the recorder's own list does not.
        assert b"rt_original" in t.recorder.requests[0].body
        assert b"refresh_token=***" in grant.body


def _seed_oauth_session(tmp_path: Path, *, stale: bool = False) -> str:
    """Write a seeded OAuth session file and return its path.

    Args:
        tmp_path: Where to write it.
        stale: When true, the access token is already expired, which forces
            :meth:`~kitty.auth.oauth_session.OAuthSession.get_valid_api_key`
            down the refresh path — and with it the refresh POST the recorder
            captures.

    Returns:
        The absolute path; the bridge carries this as its resolved key.
    """
    import time

    from kitty.auth.oauth_session import OAuthSession

    now = time.time()
    session = OAuthSession(
        client_id="app_test",
        access_token="at_fresh",
        refresh_token="rt_original",
        id_token="eyJhbGciOiJIUzI1NiJ9.e30.fake_sig",
        api_key=None,
        access_token_expires_at=now + (-(1 if stale else -3600)),
        api_key_expires_at=now + (-(1 if stale else -3600)),
        _file_path=str(tmp_path / "oauth_session.json"),
    )
    session.save()
    return session._file_path
