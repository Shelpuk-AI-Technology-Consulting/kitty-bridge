"""Tests for kitty.egress — proxy URL parsing, bypass rules, and masking."""

from __future__ import annotations

import pytest

from kitty.egress import (
    EgressConfig,
    get_egress,
    parse_proxy_url,
    set_egress,
    should_bypass,
)

# ---------------------------------------------------------------------------
# R1: parse_proxy_url
# ---------------------------------------------------------------------------


class TestParseProxyUrl:
    def test_splits_credentials_out_of_the_url(self):
        """aiohttp wants credentials as BasicAuth, not embedded in the URL."""
        cfg = parse_proxy_url("http://myuser:mypass@proxy.example.com:12323")

        assert cfg.proxy_url == "http://proxy.example.com:12323"
        assert cfg.username == "myuser"
        assert cfg.password == "mypass"

    def test_url_without_credentials(self):
        cfg = parse_proxy_url("http://proxy.example.com:3128")

        assert cfg.proxy_url == "http://proxy.example.com:3128"
        assert cfg.username is None
        assert cfg.password is None

    def test_https_scheme_is_accepted(self):
        assert parse_proxy_url("https://proxy.example.com:443").proxy_url == "https://proxy.example.com:443"

    def test_url_without_port(self):
        cfg = parse_proxy_url("http://proxy.example.com")

        assert cfg.proxy_url == "http://proxy.example.com"

    def test_percent_encoded_password_is_decoded(self):
        """A password containing @ or : must survive the round trip."""
        cfg = parse_proxy_url("http://user:p%40ss%3Aword@proxy.example.com:3128")

        assert cfg.password == "p@ss:word"

    def test_ipv6_literal_keeps_its_brackets(self):
        cfg = parse_proxy_url("http://[2001:db8::1]:3128")

        assert cfg.proxy_url == "http://[2001:db8::1]:3128"

    def test_surrounding_whitespace_is_ignored(self):
        assert parse_proxy_url("  http://proxy.example.com:3128\n").proxy_url == "http://proxy.example.com:3128"

    @pytest.mark.parametrize(
        "raw",
        ["socks5://proxy.example.com:1080", "ftp://proxy.example.com", "proxy.example.com:3128"],
    )
    def test_rejects_unsupported_schemes(self, raw: str):
        """SOCKS is not supported; a bare host:port has no scheme at all."""
        with pytest.raises(ValueError, match="scheme"):
            parse_proxy_url(raw)

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_rejects_empty(self, raw: str):
        with pytest.raises(ValueError, match="empty"):
            parse_proxy_url(raw)

    def test_rejects_url_without_host(self):
        with pytest.raises(ValueError, match="no host"):
            parse_proxy_url("http://")

    def test_username_without_password_is_rejected(self):
        """R5: a username alone authenticates as an empty password."""
        with pytest.raises(ValueError, match="together"):
            parse_proxy_url("http://useronly@proxy.example.com:3128")


# ---------------------------------------------------------------------------
# R5: paired credential validation
# ---------------------------------------------------------------------------


class TestCredentialPairing:
    def test_username_without_password_rejected(self):
        with pytest.raises(ValueError, match="together"):
            EgressConfig(proxy_url="http://h:1", username="u")

    def test_password_without_username_rejected(self):
        with pytest.raises(ValueError, match="together"):
            EgressConfig(proxy_url="http://h:1", password="p")

    def test_neither_is_fine(self):
        assert EgressConfig(proxy_url="http://h:1").auth is None

    def test_both_is_fine(self):
        auth = EgressConfig(proxy_url="http://h:1", username="u", password="p").auth

        assert auth is not None
        assert (auth.login, auth.password) == ("u", "p")


# ---------------------------------------------------------------------------
# R2: should_bypass
# ---------------------------------------------------------------------------


class TestShouldBypass:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:11434",           # default Ollama
            "http://LOCALHOST:11434",           # case-insensitive
            "http://foo.localhost/v1",
            "http://127.0.0.1:8080",
            "http://127.255.1.2/v1",
            "http://[::1]:8080",
            "http://10.0.0.5/v1",
            "http://172.16.0.1/v1",
            "http://172.31.255.254/v1",
            "http://192.168.1.50:8000",
            "http://169.254.169.254/latest/meta-data",  # AWS IMDS
        ],
    )
    def test_local_destinations_bypass(self, url: str):
        assert should_bypass(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.anthropic.com/v1/messages",
            "https://api.minimax.io/anthropic",
            "https://openrouter.ai/api/v1",
            "https://8.8.8.8/v1",
            "https://172.32.0.1/v1",   # just outside 172.16/12
            "https://11.0.0.1/v1",     # just outside 10/8
        ],
    )
    def test_public_destinations_use_the_proxy(self, url: str):
        assert should_bypass(url) is False

    def test_url_without_host_does_not_bypass(self):
        """Fail towards the proxy rather than towards a direct connection."""
        assert should_bypass("not-a-url") is False


# ---------------------------------------------------------------------------
# R3: masking and client-specific renderings
# ---------------------------------------------------------------------------


class TestMasking:
    def test_masked_hides_the_password(self):
        cfg = EgressConfig(proxy_url="http://proxy.example.com:12323", username="myuser", password="s3cr3t")

        masked = cfg.masked()

        assert masked == "http://myuser:****@proxy.example.com:12323"
        assert "s3cr3t" not in masked

    def test_masked_without_credentials_is_the_plain_url(self):
        cfg = EgressConfig(proxy_url="http://proxy.example.com:3128")

        assert cfg.masked() == "http://proxy.example.com:3128"

    def test_url_with_credentials_round_trips(self):
        """curl_cffi and botocore take one URL, so credentials go back inline."""
        cfg = parse_proxy_url("http://u:p@proxy.example.com:3128")

        assert cfg.url_with_credentials() == "http://u:p@proxy.example.com:3128"

    def test_proxies_dict_covers_both_schemes(self):
        cfg = EgressConfig(proxy_url="http://proxy.example.com:3128", username="u", password="p")

        assert cfg.proxies_dict() == {
            "http": "http://u:p@proxy.example.com:3128",
            "https": "http://u:p@proxy.example.com:3128",
        }

    def test_proxies_dict_without_credentials(self):
        cfg = EgressConfig(proxy_url="http://proxy.example.com:3128")

        assert cfg.proxies_dict() == {
            "http": "http://proxy.example.com:3128",
            "https": "http://proxy.example.com:3128",
        }


# ---------------------------------------------------------------------------
# Process-wide accessor
# ---------------------------------------------------------------------------


class TestProcessGlobal:
    def test_defaults_to_none(self):
        assert get_egress() is None

    def test_set_and_get(self):
        cfg = EgressConfig(proxy_url="http://proxy.example.com:3128")
        set_egress(cfg)

        assert get_egress() is cfg

    def test_can_be_cleared(self):
        set_egress(EgressConfig(proxy_url="http://proxy.example.com:3128"))
        set_egress(None)

        assert get_egress() is None
