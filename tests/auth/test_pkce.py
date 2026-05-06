"""Tests for PKCE (RFC 7636) helpers."""

from __future__ import annotations

import base64
import hashlib

from kitty.auth.pkce import generate_code_challenge, generate_code_verifier


class TestCodeVerifier:
    def test_length_is_64_chars(self) -> None:
        v = generate_code_verifier()
        assert len(v) == 64

    def test_only_unreserved_chars(self) -> None:
        """RFC 3986 unreserved: A-Z a-z 0-9 - . _ ~"""
        v = generate_code_verifier()
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
        assert set(v) <= allowed

    def test_different_each_call(self) -> None:
        v1 = generate_code_verifier()
        v2 = generate_code_verifier()
        assert v1 != v2

    def test_within_rfc_length_range(self) -> None:
        """RFC 7636: verifier must be 43-128 chars."""
        v = generate_code_verifier()
        assert 43 <= len(v) <= 128


class TestCodeChallenge:
    def test_is_deterministic(self) -> None:
        verifier = "test-verifier-value-12345678901234567890123456789012"
        c1 = generate_code_challenge(verifier)
        c2 = generate_code_challenge(verifier)
        assert c1 == c2

    def test_is_base64url_no_padding(self) -> None:
        """Base64URL encoding: no +, /, or = characters."""
        verifier = generate_code_verifier()
        challenge = generate_code_challenge(verifier)
        assert "+" not in challenge
        assert "/" not in challenge
        assert "=" not in challenge

    def test_matches_manual_s256_computation(self) -> None:
        """Verify the challenge is Base64URL(SHA256(verifier))."""
        verifier = "my-test-verifier-1234567890123456789012345678"
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        assert generate_code_challenge(verifier) == expected

    def test_challenge_length(self) -> None:
        """SHA-256 digest is 32 bytes → Base64URL is 43 chars (no padding)."""
        verifier = generate_code_verifier()
        challenge = generate_code_challenge(verifier)
        assert len(challenge) == 43
