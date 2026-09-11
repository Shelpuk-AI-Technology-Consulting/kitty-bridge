"""KBR-143 — the one rule that turns a base URL and an endpoint path into an address.

Three sites used to compose the upstream URL by string concatenation, so a base URL
carrying a query string put the endpoint *after* the query and the request went
somewhere the user never asked for.  :meth:`~kitty.providers.base.ProviderAdapter.compose_upstream_url`
replaces all three.

L1 by the path default in ``tests/layers.py``: the rule is a pure function of two
strings, which is the lowest layer that can prove it (`TEST_SUITE.md` §2.2).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlsplit

import pytest

from kitty.bridge.server import BridgeServer
from kitty.providers.base import ProviderAdapter
from kitty.providers.custom_openai import CustomOpenAIAdapter
from kitty.providers.ollama_cloud import OllamaCloudAdapter
from kitty.providers.registry import _registry
from kitty.validation import validate_api_key

# The two endpoint paths that exercise both shapes an adapter can supply: a bare
# path, and Azure's, which carries a query of its own (register row P20).
_CC = "/chat/completions"
_AZURE = "/openai/deployments/d/chat/completions?api-version=2024-10-21"


def _compose(base_url: str, endpoint_path: str) -> str:
    """Compose an upstream URL through the helper under test.

    Args:
        base_url: The value a profile's ``provider_config["base_url"]`` holds.
        endpoint_path: The path the adapter appends, leading slash included.

    Returns:
        The address the bridge would request.
    """
    return ProviderAdapter.compose_upstream_url(base_url, endpoint_path)


class TestComposeUpstreamUrlPath:
    """The endpoint joins the base URL's path component (R2)."""

    @pytest.mark.parametrize(
        ("base_url", "expected"),
        [
            ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1/chat/completions"),
            ("https://api.deepseek.com/v1/", "https://api.deepseek.com/v1/chat/completions"),
            ("https://api.deepseek.com", "https://api.deepseek.com/chat/completions"),
            ("https://api.deepseek.com/", "https://api.deepseek.com/chat/completions"),
            ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
            ("http://u:p@gw.example:8443/v1", "http://u:p@gw.example:8443/v1/chat/completions"),
        ],
    )
    def test_appends_the_endpoint(self, base_url: str, expected: str):
        """A correctly configured base URL composes exactly as it always did.

        Args:
            base_url: The configured base URL.
            expected: The address the bridge must request.
        """
        assert _compose(base_url, _CC) == expected

    def test_collapses_every_trailing_slash(self):
        """A doubled trailing slash composes as it did before KBR-143.

        The three concatenation sites this replaces all called ``.rstrip("/")``.
        Preserving the second slash would be more faithful to what the user typed
        and would break a base URL that works today, for a shape that is a typo
        rather than an intention.
        """
        assert _compose("https://gw.example/v1//", _CC) == "https://gw.example/v1/chat/completions"


class TestComposeUpstreamUrlQuery:
    """A query on the base URL survives, and does not displace the endpoint (R2, R3)."""

    def test_query_moves_behind_the_endpoint(self):
        """The reported defect: the endpoint used to land inside the query value."""
        assert _compose("https://gw.example/v1?tenant=x", _CC) == "https://gw.example/v1/chat/completions?tenant=x"

    def test_fragment_is_preserved(self):
        """Same defect through the fragment.  Preserved, not honoured — see D5."""
        assert _compose("https://gw.example/v1#frag", _CC) == "https://gw.example/v1/chat/completions#frag"

    def test_query_and_fragment_together(self):
        """Both components keep their place relative to the composed path."""
        assert _compose("https://gw.example/v1?t=x#f", _CC) == "https://gw.example/v1/chat/completions?t=x#f"

    def test_endpoint_query_survives_a_base_without_one(self):
        """Azure's ``api-version`` reaches the wire from the endpoint path alone."""
        composed = _compose("https://res.openai.azure.com", _AZURE)

        assert composed == "https://res.openai.azure.com/openai/deployments/d/chat/completions?api-version=2024-10-21"

    def test_both_queries_merge_with_the_endpoint_last(self):
        """A user parameter and a provider parameter coexist (R3)."""
        composed = _compose("https://res.openai.azure.com/?tenant=x", _AZURE)

        assert urlsplit(composed).query == "tenant=x&api-version=2024-10-21"

    def test_provider_wins_a_name_clash(self):
        """D2, decided by the product owner: the adapter's own value is the one sent.

        The clashing base-URL parameter is dropped and the user's other parameters
        are kept, because the adapter's translation code is written against the
        API version it names.
        """
        composed = _compose("https://res.openai.azure.com/?api-version=2024-02-01&tenant=x", _AZURE)

        assert urlsplit(composed).query == "tenant=x&api-version=2024-10-21"

    def test_a_name_that_merely_contains_the_endpoint_name_survives(self):
        """The clash test is on the whole parameter name, not a substring."""
        composed = _compose("https://gw.example/v1?xapi-version=1", _AZURE)

        assert urlsplit(composed).query == "xapi-version=1&api-version=2024-10-21"

    def test_a_surviving_parameter_is_copied_as_written(self):
        """D3 — no re-encoding: ``?a`` does not gain ``=`` and ``%20`` stays ``%20``.

        ``parse_qsl`` + ``urlencode`` would rewrite both, which silently changes a
        value the user pasted and breaks a signed URL outright.
        """
        assert urlsplit(_compose("https://gw.example/v1?a&b=%20c", _CC)).query == "a&b=%20c"


class TestComposeUpstreamUrlPreservesTheDestination:
    """Nothing but the path and query may change (R2, and the security property)."""

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.deepseek.com/v1",
            "http://localhost:8000/v1",
            "http://u:p@gw.example:8443/v1",
            "https://gw.example/v1?tenant=x#frag",
            "https://gw.example",
        ],
    )
    @pytest.mark.parametrize("endpoint_path", [_CC, _AZURE])
    def test_scheme_host_and_fragment_come_from_the_base_url(self, base_url: str, endpoint_path: str):
        """The destination host is the configured one, whatever the endpoint says.

        Args:
            base_url: The configured base URL.
            endpoint_path: The adapter's endpoint path.
        """
        before, after = urlsplit(base_url), urlsplit(_compose(base_url, endpoint_path))

        assert (after.scheme, after.netloc, after.fragment) == (before.scheme, before.netloc, before.fragment)

    def test_an_endpoint_path_cannot_redirect_the_request(self):
        """A protocol-relative endpoint path must not become the host.

        ``urlsplit("//evil.example/x")`` parses ``evil.example`` as the netloc, so a
        composer that spliced strings could be steered to another host by the path
        side.  Only the endpoint's *path* is read, so the configured host stands.
        No adapter ships such a path; the assertion is what keeps that true.
        """
        composed = _compose("https://gw.example/v1", "//evil.example/x")

        assert urlsplit(composed).netloc == "gw.example"


class TestComposeUpstreamUrlIsTotal:
    """The helper never raises, because two callers cannot afford it to (D6)."""

    def test_an_unparseable_base_url_falls_back_to_concatenation(self):
        """A malformed IPv6 literal yields today's broken-but-graceful address.

        ``kitty.validation.validate_api_key`` composes outside its own ``try`` and
        ``BridgeServer._translate_upstream_error`` composes while *formatting* an
        upstream error.  Raising here would turn a bad stored profile into a
        traceback at launch and an upstream 404 into one mid-session.
        """
        assert _compose("https://[::1/v1", _CC) == "https://[::1/v1/chat/completions"

    def test_an_empty_base_url_yields_the_endpoint(self):
        """A degenerate input still returns a string rather than raising."""
        assert _compose("", _CC) == _CC


class TestRedactUrlForDisplay:
    """A composed URL that reaches a log or a transcript carries no secrets.

    KBR-143 made query-bearing base URLs work, and a query is the standard place a
    gateway puts a credential — ``?api-key=``, ``?subscription-key=``, ``?code=``, a
    SAS ``?sig=``.  The bridge echoes the composed URL into its 404 diagnostic and
    pre-flight echoes it into a launch error, so both need a display form.
    """

    def test_userinfo_is_removed(self):
        """The case KBR-134's redactor covered, carried over unchanged."""
        redacted = ProviderAdapter.redact_url_for_display("https://u:p@gw.example/v1/chat/completions")

        assert redacted == "https://gw.example/v1/chat/completions"

    def test_a_url_with_nothing_to_hide_is_unchanged(self):
        """Redaction is a no-op for the ordinary case, so the message reads normally."""
        url = "https://gw.example:8443/v1/chat/completions"

        assert ProviderAdapter.redact_url_for_display(url) == url

    def test_every_query_value_is_masked(self):
        """Values go, names stay: the name is the diagnostic, the value is the risk."""
        redacted = ProviderAdapter.redact_url_for_display("https://gw.example/v1/chat?sig=abc&api-key=s3cret")

        assert redacted == "https://gw.example/v1/chat?sig=****&api-key=****"

    def test_a_valueless_parameter_keeps_its_shape(self):
        """``?debug`` carries no value to mask and must not gain one."""
        assert ProviderAdapter.redact_url_for_display("https://gw.example/v1?debug") == "https://gw.example/v1?debug"

    def test_masking_is_indiscriminate(self):
        """Even a harmless value is masked, because the alternative fails open.

        Telling a credential from a routing parameter means guessing from its name,
        and a guess that is wrong once leaks a key.  Azure's ``api-version`` is
        masked along with everything else; its *name* still tells the reader the
        parameter was sent, which is what the diagnostic needs.
        """
        redacted = ProviderAdapter.redact_url_for_display("https://res.openai.azure.com/d?api-version=2024-02-01")

        assert redacted == "https://res.openai.azure.com/d?api-version=****"

    def test_an_unparseable_url_keeps_its_shape_and_loses_its_query(self):
        """The malformed case is the one the new pre-flight message has to report.

        There is no structure to edit, so the redaction is textual and deliberately
        over-broad.  Withholding the URL entirely would be safe and useless: a message
        that names no address diagnoses nothing.
        """
        redacted = ProviderAdapter.redact_url_for_display("https://[::1/v1?key=s3cret")

        assert redacted == "https://[::1/v1?****"

    def test_an_unparseable_url_also_loses_its_userinfo(self):
        """Over-redacting an unparseable URL covers userinfo as well as the query."""
        assert ProviderAdapter.redact_url_for_display("https://u:p@[::1/v1") == "https://[::1/v1"


class TestEverySiteAgreesWithTheHelper:
    """Each of the three composition sites produces what the helper produces (R1).

    ``tests/test_upstream_url_single_rule.py`` pins the *set* of modules that resolve a
    base URL; these cases pin what each of them does with one.  Neither is sufficient
    alone: the scan cannot see ``OllamaCloudAdapter._build_url``, which reads
    ``provider_config["base_url"]`` directly, and these cases cannot see a fourth site.

    L1, not L2: an unstarted server and a mocked session are not "two artifacts
    compared statically", and `TEST_SUITE.md` §2.1 puts behavioural assertions where
    mutation testing judges them.
    """

    # A base URL carrying a query: the shape all three used to break on.
    _BASE = "https://gw.example/v1?tenant=x"

    def test_bridge_server(self):
        """``BridgeServer._build_upstream_url`` composes through the helper."""
        provider = CustomOpenAIAdapter()
        server = BridgeServer(
            None,  # type: ignore[arg-type]
            provider,
            "test-key",
            model="some-model",
            provider_config={"base_url": self._BASE},
        )

        expected = ProviderAdapter.compose_upstream_url(self._BASE, provider.get_upstream_path("some-model"))
        assert server._build_upstream_url() == expected

    @pytest.mark.asyncio
    @patch("kitty.validation.aiohttp.ClientSession")
    async def test_preflight(self, mock_session_cls):
        """``validate_api_key`` probes the composed address.

        Args:
            mock_session_cls: The patched ``ClientSession``, so no socket opens.
        """
        provider = CustomOpenAIAdapter()
        response = AsyncMock()
        response.status = 200
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        session = AsyncMock()
        session.post = MagicMock(return_value=response)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = session

        await validate_api_key(provider, "any-key", {"base_url": self._BASE})

        model = provider.normalize_model_name(provider.validation_model)
        expected = ProviderAdapter.compose_upstream_url(self._BASE, provider.get_upstream_path(model))
        assert session.post.call_args.args[0] == expected

    def test_ollama_cloud_transport(self):
        """``OllamaCloudAdapter._build_url`` owns its transport and must agree too.

        It reads ``provider_config["base_url"]`` directly rather than through
        ``build_base_url``, so the static scan cannot see it — which is why this case
        exists rather than being left to the guard.
        """
        adapter = OllamaCloudAdapter()

        expected = ProviderAdapter.compose_upstream_url(self._BASE, adapter.upstream_path)
        assert adapter._build_url({"base_url": self._BASE}) == expected


class TestNoAdapterDefaultAddressMoves:
    """R7 — the change is invisible to every provider's own configuration."""

    @pytest.mark.parametrize("provider_type", sorted(_registry))
    def test_composed_default_equals_the_old_concatenation(self, provider_type: str):
        """Each adapter's default address is byte-identical to the pre-KBR-143 one.

        The sweep is over the whole registry rather than a sample: finding F4 in
        `TEST_SUITE.md` §4.4 was first written from a six-provider sample and was
        wrong about the population, and this is the same shape of claim.

        Args:
            provider_type: The registry key for the adapter under test.
        """
        adapter = _registry[provider_type]()
        base, path = adapter.default_base_url, adapter.get_upstream_path("some-model")

        assert adapter.compose_upstream_url(base, path) == base.rstrip("/") + path
