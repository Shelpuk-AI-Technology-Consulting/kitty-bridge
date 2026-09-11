"""The transport the OAuth token endpoint is reached over.

KBR-161. The OpenAI subscription provider used to talk to OpenAI as **two**
clients: the API leg over ``curl_cffi`` carrying an impersonated Codex CLI
identity, and the OAuth token leg over a plain ``aiohttp`` session carrying
none. Two clients for one account, alternating for the life of a session, is a
shape no genuine Codex installation produces.

This module is the seam that let the recurring half of that OAuth traffic move
onto the adapter's existing impersonating session without :mod:`kitty.auth`
importing :mod:`kitty.providers` -- which would be a cycle, since the providers
layer already imports this one. :class:`OAuthSession` depends on the
:class:`TokenTransport` protocol; the adapter supplies the concrete
:class:`CurlTokenTransport` built over the session it already has.

The protocol is deliberately narrow -- one form POST, returning a status and a
body -- because that is the entire surface the token endpoint needs, and a
narrow seam is one a test can implement in four lines.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import curl_cffi.requests


class TokenTransport(Protocol):
    """How :class:`~kitty.auth.oauth_session.OAuthSession` reaches the token endpoint."""

    async def post_form(
        self,
        url: str,
        data: Mapping[str, str],
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float,
    ) -> tuple[int, str]:
        """POST *data* as an HTML form and return the response.

        Args:
            url: The token endpoint to post to.
            data: Form fields, sent as ``application/x-www-form-urlencoded``.
            headers: Extra request headers, such as the impersonated Codex
                ``User-Agent``.
            timeout: Total seconds allowed for the request. Required rather
                than defaulted: the caller holds a refresh lock across two of
                these, so an unbounded request stalls every concurrent request
                on the session.

        Returns:
            A ``(status_code, body_text)`` pair. The body is returned as text
            rather than parsed JSON because the token endpoint's error bodies
            are not reliably JSON, and the caller decides how to treat that.
        """
        ...


class CurlTokenTransport:
    """A :class:`TokenTransport` over an impersonating ``curl_cffi`` session.

    Wraps a session the OpenAI subscription adapter builds with
    ``impersonate="chrome136"`` and the egress ``proxies`` mapping, so the
    refresh leg presents the same TLS fingerprint and the same identity as the
    API leg it is interleaved with.

    The adapter passes a session **dedicated to this leg**, not the one serving
    completions. An ``AsyncSession`` owns a bounded pool of curl handles
    (``max_clients``, 10 by default) and a streaming response holds its handle
    until the stream ends, so a token refresh sharing that pool would queue
    behind in-flight completions -- while holding
    :attr:`~kitty.auth.oauth_session.OAuthSession._refresh_lock`, blocking every
    other request for that account. The two legs address different hosts, so
    they would never share a connection or a cookie anyway: sharing the session
    would buy nothing observable and cost that.
    """

    def __init__(self, session: curl_cffi.requests.AsyncSession) -> None:
        """Store the session to post over.

        Args:
            session: The adapter's long-lived ``curl_cffi`` session. Passed in
                rather than constructed here so that the refresh leg cannot
                drift onto a differently-fingerprinted client than the API leg,
                which is the defect this module exists to close.
        """
        self._session = session

    async def post_form(
        self,
        url: str,
        data: Mapping[str, str],
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float,
    ) -> tuple[int, str]:
        """POST *data* as a form over the impersonating session.

        Args:
            url: The token endpoint to post to.
            data: Form fields, sent as ``application/x-www-form-urlencoded``.
            headers: Extra request headers.
            timeout: Total seconds allowed for the request.

        Returns:
            A ``(status_code, body_text)`` pair.
        """
        response = await self._session.post(
            url,
            data=dict(data),
            headers=dict(headers) if headers else None,
            timeout=timeout,
        )
        return response.status_code, response.text
