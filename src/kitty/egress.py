"""Static egress proxy configuration shared by every outbound HTTP client.

Kitty can route all provider-bound traffic through a single authenticated HTTP
CONNECT proxy so that many installations present one egress IP upstream. This
module is the one place that knows what that proxy is.

It deliberately lives at the top level, beside :mod:`kitty.types` and
:mod:`kitty.cloudflare`, because the configuration has to be visible to the
bridge, the provider adapters, credential validation, the OAuth flows and the
CLI. Every other candidate location would violate one of the import-layering
contracts in ``pyproject.toml``.

Kitty never sets ``HTTP_PROXY``/``HTTPS_PROXY`` in its own environment. The
three HTTP stacks in use disagree about those variables — aiohttp ignores them
unless ``trust_env=True`` while curl_cffi and botocore honour them — so relying
on the environment would silently leave the main serving path unproxied while
appearing to work elsewhere. Each client is configured explicitly instead.
"""

from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

if TYPE_CHECKING:
    import aiohttp

logger = logging.getLogger(__name__)

__all__ = [
    "ENV_PROXY",
    "EgressConfig",
    "aiohttp_session_kwargs",
    "get_egress",
    "parse_proxy_url",
    "set_egress",
    "should_bypass",
]

#: Environment variable holding a full proxy URL, credentials optional.
ENV_PROXY = "KITTY_EGRESS_PROXY"

_ALLOWED_SCHEMES = ("http", "https")

_MASK = "****"


@dataclass(frozen=True)
class EgressConfig:
    """A resolved egress proxy.

    Attributes:
        proxy_url: Proxy endpoint as ``scheme://host:port``, with any
            credentials stripped out. aiohttp wants credentials supplied
            separately rather than embedded in the URL.
        username: Proxy username, or ``None`` for an unauthenticated proxy.
        password: Proxy password, or ``None`` for an unauthenticated proxy.
    """

    proxy_url: str
    username: str | None = None
    password: str | None = None

    def __post_init__(self) -> None:
        """Reject a half-specified credential pair.

        Raises:
            ValueError: If exactly one of ``username``/``password`` is set.
        """
        # A username with no password authenticates as an empty string, which
        # fails at the proxy in a way that is hard to diagnose. Catch it here.
        if (self.username is None) != (self.password is None):
            raise ValueError("egress proxy username and password must be provided together")

    @property
    def auth(self) -> aiohttp.BasicAuth | None:
        """Return proxy credentials in the form aiohttp expects.

        Returns:
            A :class:`aiohttp.BasicAuth`, or ``None`` when the proxy needs no
            authentication.
        """
        if self.username is None or self.password is None:
            return None
        import aiohttp

        return aiohttp.BasicAuth(self.username, self.password)

    def url_with_credentials(self) -> str:
        """Return the proxy URL with credentials embedded in the userinfo.

        Needed by clients that take a single proxy URL rather than separate
        credentials — curl_cffi and botocore both work this way.

        Returns:
            The proxy URL, including ``user:password@`` when authenticated.
        """
        if self.username is None or self.password is None:
            return self.proxy_url
        scheme, _, rest = self.proxy_url.partition("://")
        return f"{scheme}://{self.username}:{self.password}@{rest}"

    def proxies_dict(self) -> dict[str, str]:
        """Return a ``{scheme: url}`` mapping for curl_cffi and botocore.

        Returns:
            The same proxy for both ``http`` and ``https`` destinations.
        """
        url = self.url_with_credentials()
        return {"http": url, "https": url}

    def __repr__(self) -> str:
        """Return a debug representation with the password masked.

        Overridden because the dataclass default would print the password into
        any log line, traceback frame dump, or failing assertion diff.
        """
        return f"EgressConfig({self.masked()!r})"

    __str__ = __repr__

    def masked(self) -> str:
        """Return a display form of the proxy that never reveals the password.

        Returns:
            The proxy URL with the password replaced by asterisks.
        """
        if self.username is None:
            return self.proxy_url
        scheme, _, rest = self.proxy_url.partition("://")
        return f"{scheme}://{self.username}:{_MASK}@{rest}"


def parse_proxy_url(raw: str) -> EgressConfig:
    """Parse a proxy URL into an :class:`EgressConfig`.

    Credentials embedded in the userinfo are split out, because aiohttp takes
    them as a separate ``BasicAuth`` rather than inside the URL. Percent-encoded
    credentials are decoded, so a password containing ``@`` or ``:`` survives.

    Args:
        raw: Proxy URL, for example ``http://user:pass@proxy.example.com:3128``.

    Returns:
        The parsed configuration.

    Raises:
        ValueError: If the URL is empty, has no host, or uses a scheme other
            than ``http``/``https``.
    """
    candidate = (raw or "").strip()
    if not candidate:
        raise ValueError("egress proxy URL must not be empty")

    parts = urlsplit(candidate)
    if parts.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"unsupported egress proxy scheme {parts.scheme or '(none)'!r}: "
            f"expected one of {', '.join(_ALLOWED_SCHEMES)}"
        )
    if not parts.hostname:
        raise ValueError(f"egress proxy URL {candidate!r} has no host")

    # urlsplit strips the brackets from an IPv6 literal; put them back so the
    # rebuilt URL stays parseable.
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError(f"egress proxy URL {candidate!r} has an invalid port") from exc
    netloc = host if port is None else f"{host}:{port}"

    return EgressConfig(
        proxy_url=f"{parts.scheme}://{netloc}",
        username=unquote(parts.username) if parts.username else None,
        password=unquote(parts.password) if parts.password else None,
    )


def should_bypass(url: str) -> bool:
    """Return True when a destination must be reached without the proxy.

    Loopback, private and link-local destinations never leave the machine or the
    local network, so routing them through a rented proxy is both pointless and
    broken — the proxy cannot reach the caller's LAN. Bypassing also keeps local
    Ollama, the bridge's own health poll and the AWS instance metadata service
    working while egress is enabled.

    Args:
        url: Destination URL.

    Returns:
        True if the request should go out directly, False to use the proxy.
    """
    host = urlsplit(url).hostname
    if not host:
        return False

    lowered = host.lower()
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return True

    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        # A routable hostname. Resolving it here would be a DNS round trip on
        # every request, and a public name is the overwhelmingly common case.
        return False

    return address.is_loopback or address.is_private or address.is_link_local


def aiohttp_session_kwargs(config: EgressConfig | None = None) -> dict:
    """Return ``ClientSession`` kwargs that route through the egress proxy.

    Args:
        config: Configuration to use. Defaults to the process-wide one, which
            is how provider adapters reach it.

    Returns:
        ``{"proxy": ..., "proxy_auth": ...}`` when egress is configured, or an
        empty dict otherwise, so callers can splat it unconditionally.
    """
    effective = config if config is not None else get_egress()
    if effective is None:
        return {}
    return {"proxy": effective.proxy_url, "proxy_auth": effective.auth}


_egress: EgressConfig | None = None


def set_egress(config: EgressConfig | None) -> None:
    """Install the process-wide egress configuration.

    Called once at startup. Provider adapters have no other channel for
    bridge-level settings: :func:`kitty.providers.registry.get_provider` passes
    only per-profile configuration, so the adapters that own their own transport
    read this global instead.

    Args:
        config: The configuration to install, or ``None`` to disable egress.
    """
    global _egress
    _egress = config


def get_egress() -> EgressConfig | None:
    """Return the process-wide egress configuration, if any.

    Returns:
        The installed configuration, or ``None`` when egress is disabled.
    """
    return _egress
