"""R3 — The transport asymmetry, pinned per proxy-application site (T-E8, KBR-68).

``.system_design/TEST_SUITE.md`` §5.5 claim 3 (and consequence 1: the local-bypass
assertion is *false* for the custom-transport paths) · plan task **T-E8** ·
`.requirements/20260916T120000Z_kbr68_t_e8_local_bypass_fail_closed_transport_asymmetry/REQUIREMENTS.md`.

**The user-facing claim.** ``_session_for`` and ``should_bypass`` govern **only**
``BridgeServer``'s own aiohttp sessions. Every other outbound path in §5.5's table applies
the proxy **unconditionally** — including loopback and private destinations. That asymmetry
is deliberate (arguably safer) and §5.5 consequence 1 requires the design to say so rather
than imply uniformity. These tests are the pin: if a future "fix" quietly adds a
``should_bypass`` check to one of these paths, they fail.

**Layer limit (recorded so a reader knows what is *not* pinned here).** R3 pins the
**construction** layer only: a future bypass added at *request time* on these sessions —
or a session-level ``CURLOPT_NOPROXY`` with a matching host pattern — would leave
construction-layer kwargs unchanged and this module green. Request-time and session-level
bypass additions are caught by T-E3–T-E5's per-transport containment slices, which prove
every upstream connection joins a tunnel. This module is the cheapest guard on the
cheapest-to-change surface; it is not the last word.

**Layer.** No ``pytestmark``, so these take the ``l1`` path default, matching
``tests/test_egress.py`` and the rest of the fast gate: the sites are construction-time
calls with no live dependency, which is exactly what L1 is for.

**The inventory.** Three distinct construction functions cover §5.5's five rows plus the
sixth site the catalogue names (``model_context_sync``):

1. ``OpenAISubscriptionAdapter._new_curl_session`` — the curl_cffi ``proxies=`` mapping
   and the KBR-161 ``CURLOPT_NOPROXY`` remedy (§5.5 rows 1–2);
2. ``kitty.egress.aiohttp_session_kwargs`` — the aiohttp helper the OAuth-login leg
   (§5.5 row 3), ``ollama_cloud`` (§5.5 row 5) and ``model_context_sync`` (sixth site)
   share;
3. ``BedrockAdapter._get_boto3_client`` — the botocore ``Config(proxies=…)`` (§5.5 row 4).
"""

from __future__ import annotations

import sys

import pytest

from kitty.egress import EgressConfig, aiohttp_session_kwargs
from kitty.providers.bedrock import BedrockAdapter

#: Distinctive proxy credentials — the masking check in the companion module
#: (`test_egress_start_path.py`) already covers the "no password in user-facing text"
#: shape; here the password is only a fixture detail.
EGRESS = EgressConfig(proxy_url="http://proxy.example.com:12323", username="myuser", password="s3cr3tpw")


# ── Site 1 — curl_cffi: proxies= + CURLOPT_NOPROXY (§5.5 rows 1–2) ─────────


class TestCurlCffiSite:
    """The curl_cffi session builder applies the proxy unconditionally."""

    def test_new_curl_session_sets_proxies_and_noproxy_when_egress_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC3.1 + AC3.2: ``proxies=`` present and ``CURLOPT_NOPROXY`` set empty.

        The builder takes no destination URL — that *is* the pin: there is no
        per-destination decision to bypass. A future bypass that consulted
        ``should_bypass`` would have to add the URL parameter first, which this
        test's spy shape (kwargs only, no URL) makes visible.
        """
        from curl_cffi import CurlOpt

        import kitty.providers.openai_subscription as subscription

        captured: dict = {}

        class _Spy:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)
                self.cookies = None

        monkeypatch.setattr(subscription.curl_cffi.requests, "AsyncSession", _Spy)
        monkeypatch.setattr(subscription, "get_egress", lambda: EGRESS)

        subscription.OpenAISubscriptionAdapter._new_curl_session(None)

        assert captured.get("proxies") == EGRESS.proxies_dict(), (
            f"_new_curl_session did not apply the egress mapping: got "
            f"proxies={captured.get('proxies')!r}; a bypass has been added to the "
            "curl_cffi serving and refresh legs"
        )
        assert captured.get("curl_options", {}).get(CurlOpt.NOPROXY) == "", (
            f"_new_curl_session did not set CURLOPT_NOPROXY: got "
            f"curl_options={captured.get('curl_options')!r}; the KBR-161 remedy has "
            "been dropped, and an ambient NO_PROXY would silently defeat proxies="
        )


# ── Site 2 — aiohttp: the shared session-kwargs helper (§5.5 rows 3 & 5) ───


class TestAiohttpSessionKwargsSite:
    """The aiohttp helper returns proxy-bearing kwargs with egress configured.

    Three outbound paths share this helper — the OAuth-login leg
    (``auth/openai_oauth.py:434``), ``ollama_cloud`` (``providers/ollama_cloud.py:325``)
    and the catalog fetch (``providers/model_context_sync.py:121``). Pinning the helper
    pins all three at once; the per-file structural check in R4
    (``tests/test_egress_coverage.py``) pins that each file still *uses* it.
    """

    def test_helper_returns_proxy_kwargs_when_egress_is_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC3.1: the returned kwargs carry ``proxy`` and ``proxy_auth``.

        The helper takes a config, not a URL — destination-blind by signature, which
        is the construction-layer shape of "no bypass path".
        """
        monkeypatch.setattr("kitty.egress._egress", EGRESS, raising=False)

        kwargs = aiohttp_session_kwargs()

        assert kwargs.get("proxy") == EGRESS.proxy_url, (
            f"aiohttp_session_kwargs() did not return the proxy URL: got "
            f"{kwargs!r}; a bypass has been added to the shared aiohttp helper"
        )
        assert kwargs.get("proxy_auth") == EGRESS.auth, (
            f"aiohttp_session_kwargs() did not return proxy credentials: got {kwargs!r}"
        )

    def test_helper_returns_empty_kwargs_without_egress(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The complement: no egress, no proxy kwargs. Guards against a stale global."""
        monkeypatch.setattr("kitty.egress._egress", None, raising=False)

        assert aiohttp_session_kwargs() == {}


# ── Site 3 — botocore: Config(proxies=…) (§5.5 row 4) ──────────────────────


class TestBedrockBotoConfigSite:
    """The bedrock client factory applies the proxy to the boto3 client."""

    def test_boto3_client_receives_config_with_proxies_when_egress_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC3.1: ``Config(proxies=…)`` lands on a **loopback** destination too.

        AC3.1's one destination-argument site: the ``endpoint_url`` override in
        ``provider_config`` is the only per-destination knob this factory takes, so the
        drive passes a loopback endpoint and asserts the egress mapping arrives
        **alongside** it. A future bypass keyed on "endpoint_url is loopback" would have
        to drop the ``config`` kwarg for that destination, and this spy makes the shape
        visible.
        """
        captured: dict = {}

        class _FakeClient:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        class _FakeSession:
            def __init__(self, **kwargs: object) -> None:
                pass

            def client(self, service_name: str, **kwargs: object) -> _FakeClient:
                return _FakeClient(**kwargs)

        fake_boto3 = type(sys)("boto3")
        fake_boto3.Session = _FakeSession
        monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
        monkeypatch.setattr("kitty.egress._egress", EGRESS, raising=False)

        # The adapter's `_get_boto3_client` performs `import boto3` lazily; the
        # sys.modules entry above satisfies it with the fake.
        BedrockAdapter()._get_boto3_client(
            "AKIAEXAMPLE:wJalrXUtnFEMI/K7MDENG",
            {"region": "us-east-1", "endpoint_url": "http://127.0.0.1:9"},
        )

        config = captured.get("config")
        assert config is not None, (
            "the bedrock client was constructed without a botocore Config: the egress "
            "mapping has been dropped from the boto3 path"
        )
        proxies = getattr(config, "proxies", None)
        assert proxies == EGRESS.proxies_dict(), (
            f"the bedrock client's botocore Config does not carry the egress mapping: "
            f"got proxies={proxies!r}; a bypass has been added to the botocore path — "
            "even for the loopback endpoint_url the drive passed"
        )
        assert captured.get("endpoint_url") == "http://127.0.0.1:9", (
            f"the loopback endpoint_url did not reach the client: got "
            f"{captured.get('endpoint_url')!r} — the drive must address a loopback "
            "destination to pin the mapping's destination-independence"
        )
