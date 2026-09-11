"""Tests for Custom OpenAI-compatible provider adapter."""

from kitty.providers.custom_openai import CustomOpenAIAdapter, ProviderError


class TestCustomOpenAIAdapter:
    """Test suite for CustomOpenAIAdapter."""

    def test_instantiation(self):
        adapter = CustomOpenAIAdapter()
        assert adapter is not None

    def test_provider_type(self):
        adapter = CustomOpenAIAdapter()
        assert adapter.provider_type == "custom_openai"

    def test_default_base_url(self):
        adapter = CustomOpenAIAdapter()
        assert adapter.default_base_url == "https://api.openai.com/v1"

    def test_upstream_path(self):
        adapter = CustomOpenAIAdapter()
        assert adapter.upstream_path == "/chat/completions"

    def test_requires_custom_url(self):
        adapter = CustomOpenAIAdapter()
        assert adapter.requires_custom_url is True

    def test_use_custom_transport_false(self):
        adapter = CustomOpenAIAdapter()
        assert adapter.use_custom_transport is False


class TestCustomOpenAIBuildBaseUrl:
    def test_returns_config_url(self):
        adapter = CustomOpenAIAdapter()
        url = adapter.build_base_url({"base_url": "https://api.deepseek.com/v1"})
        assert url == "https://api.deepseek.com/v1"

    def test_returns_config_url_http(self):
        """HTTP URLs allowed via provider_config (unlike Profile.base_url which is HTTPS-only)."""
        adapter = CustomOpenAIAdapter()
        url = adapter.build_base_url({"base_url": "http://localhost:8000/v1"})
        assert url == "http://localhost:8000/v1"

    def test_falls_back_to_default(self):
        adapter = CustomOpenAIAdapter()
        url = adapter.build_base_url({})
        assert url == "https://api.openai.com/v1"

    def test_falls_back_on_none_config(self):
        adapter = CustomOpenAIAdapter()
        url = adapter.build_base_url(None)
        assert url == "https://api.openai.com/v1"


class TestCustomOpenAIHeaders:
    def test_bearer_auth(self):
        adapter = CustomOpenAIAdapter()
        headers = adapter.build_upstream_headers("sk-test-key-123")
        assert headers["Authorization"] == "Bearer sk-test-key-123"
        assert headers["Content-Type"] == "application/json"

    def test_different_keys_different_headers(self):
        adapter = CustomOpenAIAdapter()
        h1 = adapter.build_upstream_headers("key-a")
        h2 = adapter.build_upstream_headers("key-b")
        assert h1["Authorization"] != h2["Authorization"]


class TestCustomOpenAIBuildRequest:
    def test_basic_request(self):
        adapter = CustomOpenAIAdapter()
        req = adapter.build_request("deepseek-chat", [{"role": "user", "content": "hi"}])
        assert req == {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

    def test_streaming(self):
        adapter = CustomOpenAIAdapter()
        req = adapter.build_request("deepseek-chat", [], stream=True)
        assert req["stream"] is True

    def test_optional_params(self):
        adapter = CustomOpenAIAdapter()
        req = adapter.build_request(
            "deepseek-chat",
            [],
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
        )
        assert req["temperature"] == 0.7
        assert req["top_p"] == 0.9
        assert req["max_tokens"] == 100

    def test_tools(self):
        adapter = CustomOpenAIAdapter()
        tools = [{"type": "function", "function": {"name": "test"}}]
        req = adapter.build_request("deepseek-chat", [], tools=tools)
        assert req["tools"] == tools

    def test_omits_none_params(self):
        adapter = CustomOpenAIAdapter()
        req = adapter.build_request("deepseek-chat", [], temperature=None, max_tokens=None)
        assert "temperature" not in req
        assert "max_tokens" not in req


class TestCustomOpenAIParseResponse:
    def test_text_response(self):
        adapter = CustomOpenAIAdapter()
        resp = {
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = adapter.parse_response(resp)
        assert result["content"] == "Hello!"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10

    def test_tool_calls(self):
        adapter = CustomOpenAIAdapter()
        tool_calls = [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]
        resp = {
            "choices": [{"message": {"role": "assistant", "tool_calls": tool_calls}, "finish_reason": "tool_calls"}],
        }
        result = adapter.parse_response(resp)
        assert result["tool_calls"] == tool_calls

    def test_missing_choices(self):
        adapter = CustomOpenAIAdapter()
        result = adapter.parse_response({})
        assert result["content"] is None

    def test_empty_choices(self):
        adapter = CustomOpenAIAdapter()
        result = adapter.parse_response({"choices": []})
        assert result["content"] is None


class TestCustomOpenAITranslation:
    def test_translate_to_upstream_strips_internal_keys(self):
        adapter = CustomOpenAIAdapter()
        req = {
            "model": "deepseek-chat",
            "messages": [],
            "_resolved_key": "secret",
            "_provider_config": {},
            "_original_body": {},
            "_reasoning_effort": "high",
            "_thinking_enabled": True,
        }
        result = adapter.translate_to_upstream(req)
        for key in adapter._INTERNAL_KEYS:
            assert key not in result
        assert "model" in result
        assert "messages" in result

    def test_translate_to_upstream_preserves_other_fields(self):
        adapter = CustomOpenAIAdapter()
        req = {"model": "m", "messages": [], "temperature": 0.5, "tools": []}
        result = adapter.translate_to_upstream(req)
        assert result == {"model": "m", "messages": [], "temperature": 0.5, "tools": []}

    def test_translate_to_upstream_injects_empty_reasoning_content_when_thinking_enabled(self):
        adapter = CustomOpenAIAdapter()
        req = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "assistant", "content": "done", "reasoning_content": "thoughts"},
            ],
            "_thinking_enabled": True,
        }
        result = adapter.translate_to_upstream(req)
        assert result["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning_content": ""},
            {"role": "assistant", "content": "done", "reasoning_content": "thoughts"},
        ]

    def test_translate_to_upstream_injects_when_reasoning_effort_only(self):
        """Activation via _reasoning_effort without explicit _thinking_enabled."""
        adapter = CustomOpenAIAdapter()
        req = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "assistant", "content": "hello"}],
            "_reasoning_effort": "high",
        }
        result = adapter.translate_to_upstream(req)
        assert result["messages"] == [{"role": "assistant", "content": "hello", "reasoning_content": ""}]

    def test_translate_to_upstream_does_not_inject_reasoning_content_when_thinking_disabled(self):
        adapter = CustomOpenAIAdapter()
        req = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "assistant", "content": "hello"}],
            "_reasoning_effort": "none",
        }
        result = adapter.translate_to_upstream(req)
        assert result["messages"] == [{"role": "assistant", "content": "hello"}]

    def test_translate_to_upstream_no_injection_without_thinking_signal(self):
        """No injection when neither _thinking_enabled nor _reasoning_effort is set,
        and no assistant message carries reasoning_content."""
        adapter = CustomOpenAIAdapter()
        req = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "assistant", "content": "hello"}],
        }
        result = adapter.translate_to_upstream(req)
        assert result["messages"] == [{"role": "assistant", "content": "hello"}]

    def test_translate_to_upstream_auto_detects_thinking_from_reasoning_content(self):
        """When no explicit thinking flag is set but conversation history already
        contains assistant messages with reasoning_content, auto-detect that
        thinking mode is active and inject empty reasoning_content where missing."""
        adapter = CustomOpenAIAdapter()
        req = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "first reply"},
                {"role": "user", "content": "more"},
                {"role": "assistant", "content": "second reply", "reasoning_content": "deep thoughts"},
                {"role": "assistant", "content": "third reply"},
            ],
        }
        result = adapter.translate_to_upstream(req)
        assert result["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "first reply", "reasoning_content": ""},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "second reply", "reasoning_content": "deep thoughts"},
            {"role": "assistant", "content": "third reply", "reasoning_content": ""},
        ]

    def test_translate_to_upstream_auto_detect_does_not_trigger_from_user_message(self):
        """Auto-detection only looks at role=assistant messages, not user messages
        that might happen to have a reasoning_content key."""
        adapter = CustomOpenAIAdapter()
        req = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "hi", "reasoning_content": "not real"},
                {"role": "assistant", "content": "reply"},
            ],
        }
        result = adapter.translate_to_upstream(req)
        assert result["messages"] == [
            {"role": "user", "content": "hi", "reasoning_content": "not real"},
            {"role": "assistant", "content": "reply"},
        ]

    def test_detect_thinking_from_messages_empty_list(self):
        """Empty message list returns False."""
        adapter = CustomOpenAIAdapter()
        assert adapter._detect_thinking_from_messages([]) is False

    def test_translate_from_upstream_passthrough(self):
        adapter = CustomOpenAIAdapter()
        resp = {"choices": [{"message": {"content": "hi"}}]}
        assert adapter.translate_from_upstream(resp) is resp

    def test_translate_upstream_stream_event_passthrough(self):
        adapter = CustomOpenAIAdapter()
        chunk = b'data: {"choices": []}\n\n'
        assert adapter.translate_upstream_stream_event(chunk) == [chunk]


class TestCustomOpenAIErrorMapping:
    def test_dict_error(self):
        adapter = CustomOpenAIAdapter()
        err = adapter.map_error(401, {"error": {"message": "invalid api key"}})
        assert isinstance(err, ProviderError)
        assert "401" in str(err)
        assert "invalid api key" in str(err)

    def test_nested_error(self):
        adapter = CustomOpenAIAdapter()
        err = adapter.map_error(429, {"error": {"message": "rate limited", "type": "rate_limit"}})
        assert isinstance(err, ProviderError)
        assert "429" in str(err)
        assert "rate limited" in str(err)

    def test_non_dict_body(self):
        adapter = CustomOpenAIAdapter()
        err = adapter.map_error(500, "internal server error")
        assert isinstance(err, ProviderError)
        assert "500" in str(err)


class TestCustomOpenAINormalizeModel:
    def test_passthrough(self):
        adapter = CustomOpenAIAdapter()
        assert adapter.normalize_model_name("deepseek-chat") == "deepseek-chat"
        assert adapter.normalize_model_name("Qwen/Qwen3-235B-A22B") == "Qwen/Qwen3-235B-A22B"

    def test_no_prefix_stripping(self):
        """Custom provider does not strip prefixes — model names are user-specified."""
        adapter = CustomOpenAIAdapter()
        assert adapter.normalize_model_name("custom/deepseek-chat") == "custom/deepseek-chat"

    def test_normalize_request_noop(self):
        adapter = CustomOpenAIAdapter()
        req = {"model": "deepseek-chat", "messages": []}
        adapter.normalize_request(req)
        assert req == {"model": "deepseek-chat", "messages": []}


class TestCustomOpenAIBaseUrlEndpointSuffix:
    """KBR-134 — a base URL that already carries ``/chat/completions``.

    The bridge composes ``base_url + upstream_path``, so a user who pastes the
    full endpoint (which is what every provider's documentation shows) gets a
    doubled path and a 404.  ``build_base_url`` strips the redundant suffix.
    """

    @staticmethod
    def _build(url: str) -> str:
        """Return the base URL the adapter derives from ``url``.

        Args:
            url: The value stored in ``provider_config["base_url"]``.

        Returns:
            The normalised base URL.
        """
        return CustomOpenAIAdapter().build_base_url({"base_url": url})

    # Every shape the two properties below sweep: KBR-134's enumeration, which the
    # new query and fragment cases join rather than replace.
    _CORPUS = tuple(
        f"{scheme}://{host}{path}"
        for scheme in ("http", "https")
        for host in ("gw", "gw.example", "gw.example:8443", "u:p@gw.example")
        for path in (
            "",
            "/",
            "/v1",
            "/v1/",
            "/chat/completions",
            "/chat/completions/",
            "/v1/chat/completions",
            "/v1/chat/completions/",
            "/v1/chat/completions/chat/completions",
            "//chat/completions",
            "/v1/chat/completions;x=1",
            "/v1/chat/completions?q=1",
            "/v1/chat/completions#f",
            "/v1/chat/completions?q=1#f",
            "/openai/deployments/d/chat/completions?api-version=2024-02-01",
        )
    )

    # ── The reported defect ────────────────────────────────────────────────

    def test_strips_endpoint_suffix(self):
        """The reporter's exact input resolves to Mistral's API root."""
        assert self._build("https://api.mistral.ai/v1/chat/completions") == "https://api.mistral.ai/v1"

    def test_strips_endpoint_suffix_with_trailing_slash(self):
        """A trailing slash belongs to the match, not to the returned value."""
        assert self._build("https://api.mistral.ai/v1/chat/completions/") == "https://api.mistral.ai/v1"

    def test_strips_when_suffix_is_the_whole_path(self):
        """A base URL that is nothing but the endpoint leaves a bare origin."""
        assert self._build("https://api.example.com/chat/completions") == "https://api.example.com"

    def test_strips_exactly_one_occurrence(self):
        """Stripping once keeps the composed URL identical to the input (D4)."""
        assert self._build("https://gw/chat/completions/chat/completions") == "https://gw/chat/completions"

    # ── Correctly configured URLs are untouched ────────────────────────────

    def test_leaves_api_root_untouched(self):
        """The documented form is returned byte-identical."""
        assert self._build("https://api.deepseek.com/v1") == "https://api.deepseek.com/v1"

    def test_leaves_local_http_url_untouched(self):
        """A loopback vLLM or LM Studio endpoint is unaffected."""
        assert self._build("http://localhost:8000/v1") == "http://localhost:8000/v1"

    def test_leaves_trailing_slash_untouched(self):
        """A non-matching URL keeps its trailing slash (R5)."""
        assert self._build("https://gw.example:8443/api/v1/") == "https://gw.example:8443/api/v1/"

    # ── Shapes where a naive strip would corrupt the address ───────────────

    def test_does_not_consume_the_host(self):
        """``https://chat/completions`` ends with the suffix as a *string* only."""
        assert self._build("https://chat/completions") == "https://chat/completions"

    def test_leaves_path_parameters_untouched(self):
        """``urlsplit`` keeps ``;x=1`` in the path, so the match correctly fails."""
        assert self._build("https://host/v1/chat/completions;x=1") == "https://host/v1/chat/completions;x=1"

    # ── Shapes KBR-134 refused, two of which are now handled (KBR-143) ──────
    #
    # KBR-134's self-check compared whole URLs, so it refused every shape where
    # stripping moved anything at all -- a query, a fragment, and a doubled slash
    # alike (its decision D10, filed as KBR-143).  The check now compares *paths*,
    # which is the component the strip actually edits.  A query and a fragment stop
    # blocking it, because they are no longer part of the comparison.  A doubled
    # slash still does, because it is part of the path and collapsing it would
    # request a different address -- see `test_still_leaves_a_doubled_slash_alone`.

    def test_strips_the_suffix_and_keeps_the_query(self):
        """The query stays with the base URL; only the endpoint leaves the path."""
        assert self._build("https://gw/v1/chat/completions?tenant=x") == "https://gw/v1?tenant=x"

    def test_strips_the_suffix_and_keeps_the_fragment(self):
        """Same for a fragment, which no HTTP client puts on the wire anyway."""
        assert self._build("https://gw/v1/chat/completions#frag") == "https://gw/v1#frag"

    def test_still_leaves_a_doubled_slash_alone(self):
        """An empty path segment is part of the address, so it is not collapsed.

        ``//chat/completions`` and ``/chat/completions`` are different paths to
        nginx, to S3 and to most gateways.  Stripping here would compose back to the
        single-slash form, which is a different address rather than a cleaner one —
        the same reason KBR-134 gave for stripping exactly once, applied to the same
        URL from the other side.  KBR-134's behaviour is kept deliberately.
        """
        assert self._build("https://gw//chat/completions") == "https://gw//chat/completions"

    def test_strips_azures_documented_endpoint(self):
        """The URL Microsoft's own documentation shows, normalised (KBR-143)."""
        azure = "https://res.openai.azure.com/openai/deployments/d/chat/completions?api-version=2024-02-01"

        assert self._build(azure) == "https://res.openai.azure.com/openai/deployments/d?api-version=2024-02-01"

    def test_azure_endpoint_composes_back_to_itself(self):
        """The claim the customer cares about: pasted verbatim, requested verbatim.

        This is the acceptance criterion of KBR-143 at the lowest layer that can
        carry it — normalisation and composition together, with no server.
        """
        azure = "https://res.openai.azure.com/openai/deployments/d/chat/completions?api-version=2024-02-01"
        adapter = CustomOpenAIAdapter()

        composed = adapter.compose_upstream_url(self._build(azure), adapter.upstream_path)

        assert composed == azure

    def test_match_is_case_sensitive(self):
        """URL paths are case-sensitive by specification, so an upper-case path is left alone."""
        assert self._build("https://gw/V1/CHAT/COMPLETIONS") == "https://gw/V1/CHAT/COMPLETIONS"

    # ── The property the strip must never violate ──────────────────────────

    def test_normalisation_touches_only_the_path(self):
        """Normalisation may edit the path.  It may not touch anything else.

        KBR-134 stated this as "the composed URL never changes", which was the
        honest property of a fix that composed by concatenation and therefore could
        not handle a query at all.  That property is retired deliberately (KBR-143):
        a query-bearing URL *must* now compose differently, because before it
        composed to an address the user never asked for.

        What replaces it is structural and narrower — the strip rewrites the path
        component and ``urlunsplit``s the rest untouched — so a future edit that
        reaches the host, the query or the fragment fails here.
        """
        from urllib.parse import urlsplit

        suffix = CustomOpenAIAdapter().upstream_path
        for url in self._CORPUS:
            # No `try` around `urlsplit`: every corpus entry is parseable, and an
            # unparseable one added later should raise loudly rather than be skipped.
            before, after = urlsplit(url), urlsplit(self._build(url))

            assert (before.scheme, before.netloc, before.query, before.fragment) == (
                after.scheme,
                after.netloc,
                after.query,
                after.fragment,
            ), url
            trimmed = before.path.rstrip("/")
            allowed = {before.path}
            # Offer the stripped form only where the suffix is actually there. Computing
            # it unconditionally would let a mutant that blindly truncated the last 17
            # characters satisfy this assertion on a path that never matched.
            if trimmed.endswith(suffix):
                allowed.add(trimmed[: -len(suffix)])
            assert after.path in allowed, url

    def test_composing_the_endpoint_back_reproduces_the_pasted_address(self):
        """The claim a user would make: what they pasted is what Kitty requests.

        This is KBR-143's replacement for the retired property above, and it is the
        one that carries the ticket's acceptance criterion.  It holds only for a URL
        that was actually normalised; a URL left alone keeps whatever address it
        always composed to, which may well be wrong, and saying otherwise would
        claim this change fixes shapes it does not touch.
        """
        from urllib.parse import urlsplit

        adapter = CustomOpenAIAdapter()
        for url in self._CORPUS:
            normalised = self._build(url)
            if normalised == url:
                continue
            composed = adapter.compose_upstream_url(normalised, adapter.upstream_path)
            assert urlsplit(composed).path == urlsplit(url).path.rstrip("/"), url
            assert urlsplit(composed).query == urlsplit(url).query, url

    def test_scheme_and_host_are_preserved(self):
        """Normalisation touches the path and nothing else."""
        from urllib.parse import urlsplit

        for url in (
            "https://api.mistral.ai/v1/chat/completions",
            "http://localhost:8000/v1",
            "https://chat/completions",
            "https://gw.example:8443/api/v1/",
        ):
            before, after = urlsplit(url), urlsplit(self._build(url))
            assert (after.scheme, after.netloc) == (before.scheme, before.netloc), url

    def test_unparseable_url_is_returned_untouched(self):
        """A URL ``urlsplit`` cannot read must not become an exception.

        ``urlsplit`` raises on a malformed IPv6 literal, and this helper runs
        inside ``build_base_url`` — which ``kitty.validation.validate_api_key``
        calls *outside* its own ``try``.  Raising here would turn a bad stored
        profile into a traceback at launch instead of an error message, which is
        a regression this normalisation introduced and must not reintroduce.
        """
        assert self._build("https://[::1/v1") == "https://[::1/v1"
        assert self._build("https://[::1/v1/chat/completions") == "https://[::1/v1/chat/completions"

    # ── Existing validation is unchanged ───────────────────────────────────

    def test_rejects_empty_url(self):
        """An empty base URL still raises, with the existing message."""
        import pytest

        with pytest.raises(ValueError, match="Invalid base_url"):
            self._build("")

    def test_rejects_non_http_scheme(self):
        """A non-HTTP scheme still raises before normalisation is reached."""
        import pytest

        with pytest.raises(ValueError, match="Invalid base_url"):
            self._build("ftp://x")

    def test_rejects_schemeless_url(self):
        """A bare host still raises before normalisation is reached."""
        import pytest

        with pytest.raises(ValueError, match="Invalid base_url"):
            self._build("api.mistral.ai/v1")
