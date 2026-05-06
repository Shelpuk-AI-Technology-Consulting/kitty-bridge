"""PKCE (RFC 7636) helpers for OAuth authorization code flow."""

from __future__ import annotations

import base64
import hashlib
import secrets

# RFC 3986 unreserved characters
_UNRESERVED = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"


def generate_code_verifier(length: int = 64) -> str:
    """Generate a cryptographically random code verifier (RFC 7636).

    Args:
        length: Number of random characters (default 64, must be 43-128).

    Returns:
        A string of *length* random characters from the unreserved set.
    """
    if not (43 <= length <= 128):
        raise ValueError("code verifier length must be between 43 and 128")
    # secrets.choice picks uniformly from the string
    return "".join(secrets.choice(_UNRESERVED) for _ in range(length))


def generate_code_challenge(verifier: str) -> str:
    """Compute the S256 code challenge for a verifier (RFC 7636).

    Args:
        verifier: A valid code verifier string.

    Returns:
        Base64URL(SHA256(verifier)) with no padding.
    """
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
