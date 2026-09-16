"""Tests for providers/azure.py — AzureOpenAIAdapter."""

import pytest

from kitty.providers.azure import AzureOpenAIAdapter

# ── CC format samples ────────────────────────────────────────────────────

CC_MESSAGES_BASIC = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
]

CC_MESSAGES_TOOLS = [
    {"role": "user", "content": "What's the weather?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_abc",
        "content": "15°C, cloudy",
    },
]

# ── Properties ───────────────────────────────────────────────────────────


class TestAzureAdapterProperties:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "azure"

    def test_default_base_url(self):
        # No single default — must be configured per-profile
        assert "openai.azure.com" in self.adapter.default_base_url

    def test_use_custom_transport(self):
        # Azure uses standard CC format — no custom transport needed
        assert self.adapter.use_custom_transport is False

    def test_requires_custom_url(self):
        """KBR-153 — the wizard must prompt for the resource's base URL.

        The setup wizard branches on this flag (``kitty.cli.setup_cmd`` and
        ``kitty.cli.profile_cmd``); with it False nothing ever wrote
        ``provider_config["base_url"]`` and the adapter asked DNS to resolve
        the ``{resource}`` placeholder.
        """
        assert self.adapter.requires_custom_url is True


# ── upstream_path ────────────────────────────────────────────────────────


class TestAzureUpstreamPath:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_path_includes_deployment(self):
        path = self.adapter.upstream_path
        assert "/openai/deployments/" in path
        assert "/chat/completions" in path

    def test_path_with_deployment_id(self):
        adapter = AzureOpenAIAdapter()
        path = adapter.get_upstream_path("my-gpt4o-deployment")
        assert path == "/openai/deployments/my-gpt4o-deployment/chat/completions?api-version=2024-10-21"

    def test_api_version_query_param(self):
        adapter = AzureOpenAIAdapter()
        assert "api-version=" in adapter.get_upstream_path("dep")


# ── Auth headers ─────────────────────────────────────────────────────────


class TestAzureBuildUpstreamHeaders:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_api_key_auth(self):
        headers = self.adapter.build_upstream_headers("my-azure-api-key-123")
        assert headers["api-key"] == "my-azure-api-key-123"
        assert "Authorization" not in headers

    def test_entra_token_auth(self):
        """When key starts with 'Bearer ', use Authorization header."""
        headers = self.adapter.build_upstream_headers("Bearer eyJ0eXAiOiJKV1Q...")
        assert headers["Authorization"] == "Bearer eyJ0eXAiOiJKV1Q..."
        assert "api-key" not in headers

    def test_content_type_included(self):
        headers = self.adapter.build_upstream_headers("test-key")
        assert headers["Content-Type"] == "application/json"


# ── Request translation ─────────────────────────────────────────────────


class TestAzureTranslateToUpstream:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_removes_model_from_body(self):
        """Azure uses deployment-id in URL, not model in body."""
        cc = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert "model" not in result

    def test_preserves_messages(self):
        cc = {
            "model": "gpt-4o",
            "messages": CC_MESSAGES_BASIC,
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"

    def test_preserves_stream(self):
        cc = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["stream"] is True

    def test_preserves_tools(self):
        cc = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {"name": "test", "parameters": {}}}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert len(result["tools"]) == 1

    def test_preserves_temperature(self):
        cc = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "temperature": 0.7,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["temperature"] == 0.7

    def test_preserves_max_tokens(self):
        cc = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 4096,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["max_tokens"] == 4096

    def test_removes_internal_fields(self):
        cc = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": "test",
            "_provider_config": {},
        }
        result = self.adapter.translate_to_upstream(cc)
        assert "_resolved_key" not in result
        assert "_provider_config" not in result


# ── Response translation (passthrough — CC compatible) ───────────────────


class TestAzureTranslateFromUpstream:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_passthrough(self):
        resp = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.adapter.translate_from_upstream(resp)
        assert result is resp  # same object — passthrough


# ── normalize_model_name ────────────────────────────────────────────────


class TestAzureNormalizeModelName:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_strips_prefix(self):
        assert self.adapter.normalize_model_name("azure/my-gpt4o-deployment") == "my-gpt4o-deployment"

    def test_no_prefix(self):
        assert self.adapter.normalize_model_name("my-gpt4o-deployment") == "my-gpt4o-deployment"


# ── normalize_request ───────────────────────────────────────────────────


class TestAzureNormalizeRequest:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_does_not_modify_request(self):
        cc = {"model": "gpt-4o", "messages": []}
        self.adapter.normalize_request(cc)
        assert cc == {"model": "gpt-4o", "messages": []}


# ── Error mapping ────────────────────────────────────────────────────────


class TestAzureMapError:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_401_error(self):
        exc = self.adapter.map_error(401, {"error": {"message": "invalid api-key"}})
        assert "401" in str(exc)

    def test_429_error(self):
        exc = self.adapter.map_error(429, {"error": {"message": "rate limited"}})
        assert "429" in str(exc)

    def test_500_error(self):
        exc = self.adapter.map_error(500, {"error": {"message": "internal error"}})
        assert "500" in str(exc)


# ── build_request / parse_response ──────────────────────────────────────


class TestAzureBuildRequest:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_build_request_basic(self):
        result = self.adapter.build_request(
            model="my-deployment",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )
        assert result["model"] == "my-deployment"
        assert result["stream"] is False


class TestAzureParseResponse:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_parse_response(self):
        cc_resp = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.adapter.parse_response(cc_resp)
        assert result["content"] == "hi"
        assert result["finish_reason"] == "stop"


# ── Entra ID token resolution ────────────────────────────────────────────


class TestAzureEntraToken:
    def setup_method(self):
        self.adapter = AzureOpenAIAdapter()

    def test_is_entra_token(self):
        assert self.adapter.is_entra_token("Bearer eyJ0eXAiOi...") is True

    def test_is_not_entra_token(self):
        assert self.adapter.is_entra_token("my-api-key-123") is False

    def test_entra_prefix_case_insensitive(self):
        assert self.adapter.is_entra_token("bearer eyJ...") is True


class TestAzureEntraTokenNormalization:
    """Verify build_upstream_headers normalizes bearer casing."""

    def test_lowercase_bearer_normalized_to_proper(self):
        adapter = AzureOpenAIAdapter()
        headers = adapter.build_upstream_headers("bearer eyJ0eXAiOi...")
        assert headers["Authorization"] == "Bearer eyJ0eXAiOi..."

    def test_mixed_case_bearer_normalized(self):
        adapter = AzureOpenAIAdapter()
        headers = adapter.build_upstream_headers("Bearer eyJ0eXAiOi...")
        assert headers["Authorization"] == "Bearer eyJ0eXAiOi..."


class TestAzureBuildBaseUrl:
    """KBR-153 — Azure must honour ``provider_config["base_url"]``.

    Before this fix the adapter inherited ``build_base_url``, which returned
    the literal ``https://{resource}.openai.azure.com`` placeholder whatever
    the profile said, and ``requires_custom_url`` was False so no wizard ever
    wrote the key.  Now: the key is read, a scheme check sits in front, and a
    pasted endpoint is cut back to the resource root because the deployment
    segment is dynamic (the model, register row P20).

    The cut searches the parsed *path* for the first ``/openai/deployments/``
    marker.  Anything resembling the marker in a query, a fragment, or in
    percent-encoded form is a different shape and passes through untouched —
    that property, plus the address the bridge then composes, is what the
    "claim the customer cares about" test pins at the end.
    """

    #: The endpoint Microsoft's documentation shows, with the adapter's
    #: pinned api-version, so the cut's effect on the wire is observable.
    _DOCUMENTED = (
        "https://res.openai.azure.com/openai/deployments/d/chat/completions"
        "?api-version=2024-10-21"
    )

    @staticmethod
    def _build(url: str) -> str:
        """Return the base URL the adapter derives from ``url``.

        Args:
            url: The value stored in ``provider_config["base_url"]``.

        Returns:
            The normalised base URL.
        """
        return AzureOpenAIAdapter().build_base_url({"base_url": url})

    # ── A configured resource root passes through ─────────────────────────

    def test_resource_root_passes_through(self):
        assert self._build("https://res.openai.azure.com") == "https://res.openai.azure.com"

    def test_resource_root_with_trailing_slash_keeps_it(self):
        assert self._build("https://res.openai.azure.com/") == "https://res.openai.azure.com/"

    def test_resource_root_with_port_passes_through(self):
        assert (
            self._build("https://res.openai.azure.com:8443")
            == "https://res.openai.azure.com:8443"
        )

    def test_userinfo_is_preserved(self):
        """A gateway with embedded credentials is returned verbatim past the cut."""
        url = "https://u:p@res.openai.azure.com/openai/deployments/d/chat/completions"
        assert self._build(url) == "https://u:p@res.openai.azure.com"

    # ── The documented endpoint is cut back to the resource root ──────────

    def test_documented_endpoint_is_cut_to_the_resource_root(self):
        assert self._build(self._DOCUMENTED) == (
            "https://res.openai.azure.com?api-version=2024-10-21"
        )

    def test_documented_endpoint_with_path_trailing_slash_is_cut(self):
        """A trailing slash on the path (no query) is cut back to the root.

        The path keeps its component-level trailing slash before the cut, and
        the cut replaces it with the resource root — clean, with no leftover
        separator.
        """
        url = "https://res.openai.azure.com/openai/deployments/d/chat/completions/"
        assert self._build(url) == "https://res.openai.azure.com"

    def test_documented_endpoint_on_a_port_is_cut(self):
        url = "https://res.openai.azure.com:8443/openai/deployments/d/chat/completions"
        assert self._build(url) == "https://res.openai.azure.com:8443"

    def test_doubled_marker_cuts_at_the_first_one(self):
        """Two pasted deployments still resolve to the resource root.

        Find- (not rfind-)ing the marker keeps a doubled pasted path
        consistent with the unwritten one: the cut is at the first marker,
        and the bridge then appends the deployment segment from the request
        model.
        """
        url = (
            "https://res.openai.azure.com/openai/deployments/d"
            "/openai/deployments/e/chat/completions"
        )
        assert self._build(url) == "https://res.openai.azure.com"

    # ── A gateway root with no marker passes through unchanged ────────────

    def test_gateway_root_without_the_marker_passes_through(self):
        assert self._build("https://gw.example/azure") == "https://gw.example/azure"

    def test_through_a_prefix_path_with_no_marker(self):
        """A path-shaped URL that does not contain the marker is untouched."""
        url = "https://gw.example/azure/v1"
        assert self._build(url) == url

    # ── Shapes a literal substring match would corrupt ────────────────────

    def test_marker_in_the_query_is_not_a_cut(self):
        """The marker must be in the *path*; a query mention is a parameter."""
        url = "https://res.openai.azure.com/root?x=/openai/deployments/d"
        assert self._build(url) == url

    def test_marker_in_the_fragment_is_not_a_cut(self):
        url = "https://res.openai.azure.com/root#/openai/deployments/d"
        assert self._build(url) == url

    def test_percent_encoded_marker_is_not_a_cut(self):
        """``urlsplit`` does not decode the path, so an encoded marker is not a marker."""
        url = "https://res.openai.azure.com/openai%2Fdeployments%2Fd"
        assert self._build(url) == url

    def test_marker_at_root_with_a_query_does_not_drop_the_query(self):
        """Cutting back to the root must keep the base URL's query intact."""
        url = "https://res.openai.azure.com/openai/deployments/d/chat/completions?tenant=x"
        assert self._build(url) == "https://res.openai.azure.com?tenant=x"

    # ── A missing URL raises, so launch names what is missing ─────────────

    def test_missing_url_raises(self):
        with pytest.raises(ValueError, match="base_url"):
            AzureOpenAIAdapter().build_base_url({})

    def test_none_config_raises(self):
        with pytest.raises(ValueError, match="base_url"):
            AzureOpenAIAdapter().build_base_url(None)

    def test_empty_url_raises(self):
        with pytest.raises(ValueError, match="base_url"):
            AzureOpenAIAdapter().build_base_url({"base_url": ""})

    # ── Scheme validation, matching ``custom_openai``'s contract ──────────

    def test_non_http_scheme_raises(self):
        with pytest.raises(ValueError, match="Invalid base_url"):
            self._build("ftp://x")

    def test_schemeless_url_raises(self):
        with pytest.raises(ValueError, match="Invalid base_url"):
            self._build("res.openai.azure.com/openai/deployments/d")

    # ── The full composition — the claim the customer cares about ─────────

    def test_pasted_documented_endpoint_is_requested_verbatim(self):
        """What the user pastes is what the bridge requests, end to end.

        Normalisation (``build_base_url``) and composition
        (``compose_upstream_url``) together, with no server: the acceptance
        criterion of KBR-153 at the lowest layer that can carry it.
        """
        adapter = AzureOpenAIAdapter()
        composed = adapter.compose_upstream_url(
            self._build(self._DOCUMENTED),
            adapter.get_upstream_path("d"),
        )
        assert composed == self._DOCUMENTED

    def test_a_pasted_api_version_is_dropped_in_favour_of_the_adapters(self):
        """KBR-143's name-clash rule: the endpoint's ``api-version`` wins,
        the user's other parameters survive.

        The doc pins the third ticket item this way: a pasted
        ``api-version=…`` is dropped in favour of the adapter's, while
        ``tenant=x`` — a parameter the user *did* want — comes through.  No
        ``provider_config["api_version"]`` override is added in this ticket;
        the rule is enforced by the composition helper, not by the adapter.

        The ``tenant=x`` half is what catches a future cut that drops the
        query wholesale rather than only the clashing parameter.
        """
        pasted = (
            "https://res.openai.azure.com/openai/deployments/d"
            "/chat/completions?tenant=x&api-version=2024-02-01"
        )
        adapter = AzureOpenAIAdapter()
        composed = adapter.compose_upstream_url(
            self._build(pasted), adapter.get_upstream_path("d")
        )
        assert "tenant=x" in composed
        assert "api-version=2024-02-01" not in composed
        assert "api-version=2024-10-21" in composed
