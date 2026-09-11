"""Provider adapter interface — stateless request/response builders for upstream APIs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

# Stands in for anything withheld from a message or a log. Spelled the same as
# `kitty.egress._MASK`, which masks a proxy password, so one convention covers both.
_MASK = "****"


class ProviderAdapter(ABC):
    """Interface for upstream Chat Completions API providers.

    Implementations are stateless: they build request payloads and parse
    response payloads but do not perform HTTP calls themselves.
    """

    # Internal metadata keys that must never be sent upstream.
    _INTERNAL_KEYS = frozenset(
        {
            "_reasoning_effort",
            "_thinking_enabled",
            # KBR-6: written by MessagesTranslator.translate_request alongside
            # the two above, and missed when they were added.  AnthropicAdapter
            # reads both from cc_request before rebuilding its own body, so
            # stripping them here does not disturb adaptive thinking or effort.
            "_effort",
            "_thinking_adaptive",
            "_resolved_key",
            "_provider_config",
            "_original_body",
            "_native_messages_request",
            "base_url",  # F15 defense-in-depth — URL override goes through build_base_url(),
            # not the CC request body.  Stripping it here protects
            # adapters that rely on the default translate_to_upstream().
        }
    )

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Unique provider type identifier (e.g. ``"zai_regular"``)."""

    @property
    @abstractmethod
    def default_base_url(self) -> str:
        """Default upstream API base URL for this provider."""

    def normalize_model_name(self, model: str) -> str:
        """Normalize a model name for this provider.

        Strips OpenRouter-style prefixes (e.g. ``"minimax/"``) when using
        a direct provider.  OpenRouter itself overrides this to pass names
        through unchanged.

        Args:
            model: Raw model name, possibly with an OpenRouter prefix.

        Returns:
            The provider-native model name.
        """
        return model

    def normalize_request(self, cc_request: dict) -> None:
        """Normalize a Chat Completions request for this provider.

        Mutates ``cc_request`` in place to add or adjust provider-specific
        parameters.  Default implementation does nothing.

        Args:
            cc_request: Chat Completions request dict to normalize.
        """
        return None

    @property
    def upstream_path(self) -> str:
        """Upstream API endpoint path appended to ``default_base_url``.

        Default is ``"/chat/completions"``.  Override for providers that use
        a different endpoint (e.g. Anthropic Messages API).
        """
        return "/chat/completions"

    def build_base_url(self, provider_config: dict) -> str:
        """Build the base URL using provider-specific configuration.

        Default returns ``default_base_url``.  Override for providers
        where the URL depends on config parameters (e.g. Vertex AI needs
        project_id and location in the URL).

        Args:
            provider_config: Provider-specific configuration dict.

        Returns:
            Base URL string (without the endpoint path).
        """
        return self.default_base_url

    @staticmethod
    def compose_upstream_url(base_url: str, endpoint_path: str) -> str:
        """Join a base URL and an endpoint path into the address to request.

        The bridge used to compose this by string concatenation, which put the
        endpoint *after* any query string the base URL carried and sent the request
        somewhere the profile never named (KBR-143).  Azure OpenAI's documented
        endpoint always carries ``?api-version=``, so that class of base URL could
        not be served at all.  This method is the one composition rule: every site
        that needs an upstream address calls it.

        The endpoint joins the **path** component.  Scheme, host, port and fragment
        are the base URL's, untouched — only the endpoint's path and query are read,
        so a protocol-relative ``endpoint_path`` cannot move the request to another
        host.  Queries merge: when both sides carry one, the base URL's parameters
        come first, minus any whose name the endpoint also uses, followed by the
        endpoint's query verbatim.

        The endpoint wins a name clash because the adapter that supplied it is
        written against the API version it names, while the user's other parameters
        survive because carrying them is why a query was pasted at all.  Surviving
        parameters are copied as written rather than re-encoded: ``parse_qsl`` plus
        ``urlencode`` would turn ``?a`` into ``a=`` and ``%20`` into ``+``, silently
        rewriting a value the caller chose.

        Args:
            base_url: The provider's base URL, as :meth:`build_base_url` returns it.
            endpoint_path: The endpoint to append, leading slash included — and
                possibly carrying a query of its own, as Azure's does.

        Returns:
            The full URL to request.  A ``base_url`` that cannot be parsed falls
            back to plain concatenation instead of raising: both
            :func:`kitty.validation.validate_api_key` and
            ``BridgeServer._translate_upstream_error`` compose where an exception
            would replace a readable message with a traceback.
        """
        # `urlsplit` rejects a malformed IPv6 literal such as "https://[::1/v1".
        # Such a URL is unusable either way; returning the old concatenation keeps
        # the failure a message from the HTTP client rather than a crash here.
        try:
            base = urlsplit(base_url)
            endpoint = urlsplit(endpoint_path)
        except ValueError:
            return base_url.rstrip("/") + endpoint_path

        # `rstrip`, collapsing every trailing slash, because that is what the three
        # concatenation sites this replaces did. Keeping a doubled trailing slash
        # would be more faithful to what the user typed and would break a base URL
        # that works today, for a shape that is a typo rather than an intention.
        query = ProviderAdapter._merge_query(base.query, endpoint.query)
        return urlunsplit(base._replace(path=base.path.rstrip("/") + endpoint.path, query=query))

    @staticmethod
    def redact_url_for_display(url: str) -> str:
        """Return a form of ``url`` safe to put in a log, a message or a transcript.

        Userinfo is dropped and every query **value** is masked, the parameter names
        surviving.  Before KBR-143 a query-bearing base URL could not reach an
        upstream at all, so no working profile carried one; now that they work, a
        query is the standard place a gateway keeps a credential —
        ``?subscription-key=``, ``?code=``, a SAS ``?sig=``.  Both places the bridge
        echoes a composed URL reach a durable record: the HTTP 404 diagnostic travels
        into the agent transcript and the access log, and pre-flight's failure reason
        is printed at launch.

        Values are masked indiscriminately rather than by name, because telling a
        credential from a routing parameter means guessing, and a guess that is wrong
        once leaks a key.  The parameter *name* is what the diagnostic needs — it says
        the parameter was sent — so nothing useful is lost.

        Args:
            url: The URL about to be shown to a human.

        A **valueless** parameter is left as written: ``?debug`` has no value to mask,
        and its text is a name by this rule.  A bare token used as a parameter name
        would therefore survive, which is accepted — masking names as well would cost
        every parameter name in the diagnostic to protect a shape no API uses.

        The fragment is masked whole rather than per-parameter, because an HTTP client
        never sends one: there is nothing to diagnose in a component the provider does
        not see, so none of it is worth keeping.

        Returns:
            The redacted URL.  An unparseable one — the very case a malformed-profile
            message has to report — is redacted textually instead of being withheld
            entirely, because a message that shows nothing diagnoses nothing.
        """
        try:
            parts = urlsplit(url)
        except ValueError:
            return ProviderAdapter._redact_unparseable_url(url)

        # A URL with no authority cannot be redacted structurally: "u:p@host/v1" parses
        # as scheme "u" with the credentials in the PATH, where no netloc rule reaches
        # them. Such a URL can never name a host, so it is only ever shown as an error.
        if not parts.netloc:
            return ProviderAdapter._redact_unparseable_url(url)

        # Split on "&" rather than parsing: `parse_qsl` would decode the names, and
        # re-encoding them could alter a name the reader needs to recognise.
        if parts.query:
            masked = [p if "=" not in p else f"{p.split('=', 1)[0]}={_MASK}" for p in parts.query.split("&")]
            parts = parts._replace(query="&".join(masked))

        if parts.fragment:
            parts = parts._replace(fragment=_MASK)

        # The common case has no userinfo, which keeps an ordinary URL byte-identical.
        if "@" in parts.netloc:
            parts = parts._replace(netloc=parts.netloc.rsplit("@", 1)[1])

        return urlunsplit(parts)

    @staticmethod
    def _redact_unparseable_url(url: str) -> str:
        """Redact a URL :func:`~urllib.parse.urlsplit` cannot read, by text alone.

        There is no structure to edit, so this over-redacts deliberately: everything
        from the first ``"?"`` is treated as query and dropped, and anything before an
        ``"@"`` is treated as userinfo and dropped.  What survives is the scheme, host
        and path — the part that names the address, which is what the message needs.

        A path legitimately containing ``"@"`` loses its head under this rule. That is
        an acceptable price on a URL that is already malformed, and the alternative —
        showing the value whole — is how a credential reaches a log.

        Args:
            url: The unparseable URL.

        Returns:
            The textually redacted form.
        """
        head, query_separator, _ = url.partition("?")

        # `partition` returns the whole string as its FIRST element when the separator
        # is absent, so a URL with no "://" would otherwise be read as all scheme and
        # no host -- leaving the userinfo strip below nothing to work on.
        scheme, scheme_separator, rest = head.partition("://")
        if not scheme_separator:
            scheme, rest = "", head

        if "@" in rest:
            rest = rest.rsplit("@", 1)[1]

        shown = f"{scheme}://{rest}" if scheme else rest
        return f"{shown}?{_MASK}" if query_separator else shown

    @staticmethod
    def _merge_query(base_query: str, endpoint_query: str) -> str:
        """Combine two query strings, letting the endpoint's parameters win.

        Args:
            base_query: The query component of the configured base URL.
            endpoint_query: The query component of the adapter's endpoint path.

        Returns:
            The merged query, with every base parameter whose name the endpoint does
            not also use, followed by ``endpoint_query`` unchanged.  When either
            side is empty the other is returned byte-for-byte.
        """
        # The common case by far: one side has no query, so nothing is parsed and
        # the surviving string cannot be altered.
        if not endpoint_query:
            return base_query
        if not base_query:
            return endpoint_query

        # Names are compared as written, because percent-decoding them would mean
        # re-encoding the values this split exists to leave alone.
        endpoint_names = {pair.split("=", 1)[0] for pair in endpoint_query.split("&")}
        kept = [pair for pair in base_query.split("&") if pair.split("=", 1)[0] not in endpoint_names]
        return "&".join([*kept, endpoint_query])

    @staticmethod
    def _strip_endpoint_suffix(url: str, suffix: str) -> str:
        """Remove one trailing copy of ``suffix`` from a base URL's path.

        The bridge appends the endpoint path to the base URL, so a base URL that
        already ends in that endpoint produces a doubled path and a 404 from the
        upstream.  Users paste the full endpoint routinely — it is the form every
        provider's documentation shows — so the redundant tail is removed here
        rather than rejected (KBR-134).

        The suffix is a **parameter rather than** :attr:`upstream_path` so that a
        caller always supplies the same path composition will use.  Adapters
        that route per model through :meth:`get_upstream_path` (Azure, Vertex,
        OpenCode) would otherwise be normalised against a path they never
        request.

        Args:
            url: The configured base URL, already validated as ``http(s)``.
            suffix: The endpoint path that will be appended to the result,
                leading slash included — for example ``"/chat/completions"``.

        Returns:
            ``url`` with one trailing ``suffix`` removed from its path, or ``url``
            unchanged when removing it would change the **path** the bridge ends up
            requesting, or when it cannot be parsed at all.
        """
        # Match the parsed path, never the raw string: "https://chat/completions"
        # ends with "/chat/completions" as text, and stripping that eats the host.
        # `urlsplit`, not `urlparse` -- the latter splits a trailing ";params" off
        # the last path segment and reattaches it to whichever segment ends up last.
        #
        # An unparseable URL is returned untouched rather than allowed to raise:
        # `urlsplit` rejects a malformed IPv6 literal such as "https://[::1/v1", and
        # this helper runs inside `build_base_url`, which pre-flight validation calls
        # OUTSIDE its own try block. Raising here would turn a bad stored profile into
        # a traceback at launch, where it previously produced an error message.
        try:
            parts = urlsplit(url)
        except ValueError:
            return url
        path = parts.path.rstrip("/")
        if not path.endswith(suffix):
            return url

        # Self-check, on the PATH rather than the whole URL (KBR-143). Keep the
        # candidate only if composing the endpoint back onto it reproduces the path
        # the caller gave. This still rejects a doubled slash, whose empty segment is
        # part of the address -- "//chat/completions" and "/chat/completions" are
        # different paths to nginx and to S3 -- without enumerating shapes.
        #
        # KBR-134 compared whole URLs here, which also rejected a query or a fragment,
        # because it composed by concatenation and a query genuinely could not be
        # handled. `compose_upstream_url` joins the path component instead, so neither
        # belongs in the comparison any more: that is what makes Azure's documented
        # endpoint reachable.
        candidate_path = path[: -len(suffix)]
        if candidate_path.rstrip("/") + suffix != path:
            return url
        return urlunsplit(parts._replace(path=candidate_path))

    def get_upstream_path(self, model: str) -> str:
        """Build the upstream path for a specific model.

        Default returns ``upstream_path`` (ignores model).  Override for
        providers where the model is part of the URL path (e.g. Azure OpenAI
        uses ``/openai/deployments/{deployment-id}/chat/completions``).

        Args:
            model: The normalized model identifier.

        Returns:
            URL path to append to ``default_base_url``.
        """
        return self.upstream_path

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Build HTTP headers for the upstream request.

        Default uses ``Authorization: Bearer``.  Override for providers
        with different auth schemes (e.g. ``x-api-key`` for Anthropic).

        Args:
            api_key: Resolved API key for the upstream provider.
        """
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def build_upstream_headers_for_model(self, api_key: str, model: str) -> dict[str, str]:
        """Build HTTP headers for the upstream request carrying *model*.

        The per-model form of :meth:`build_upstream_headers`, for adapters
        that authenticate differently depending on which endpoint the model
        routes to — the same pairing as :attr:`upstream_path` /
        :meth:`get_upstream_path` and :attr:`upstream_wire_is_messages_api` /
        :meth:`upstream_wire_is_messages_api_for_model`.

        Concrete here, and deliberately not an optional hook the bridge
        reaches for with ``hasattr``: an adapter that routes auth per model
        and forgets to define it would then silently receive the default
        scheme, which is the shape of the defect KBR-127 fixed on the model
        itself.  KBR-7 made the wire-shape sibling concrete for the same
        reason.  The default ignores *model* and answers as
        :meth:`build_upstream_headers` does, so an adapter with one auth
        scheme overrides nothing.

        Args:
            api_key: Resolved API key for the upstream provider.
            model: The model the request will be sent with, exactly as
                ``translate_to_upstream`` reads it: the normalized model in
                ``cc_request``.

        Returns:
            The headers to send upstream.
        """
        return self.build_upstream_headers(api_key)

    def translate_to_upstream(self, cc_request: dict) -> dict:
        """Translate a normalized CC request into the upstream wire format.

        Default strips internal metadata keys and returns the rest unchanged
        (passthrough).  Override for providers whose upstream API differs from
        Chat Completions.

        Args:
            cc_request: Fully normalized Chat Completions request dict.

        Returns:
            Dict to send as JSON body to the upstream endpoint.
        """
        return {k: v for k, v in cc_request.items() if k not in self._INTERNAL_KEYS}

    def _inject_empty_reasoning_content(self, messages: list[dict]) -> list[dict]:
        """Inject empty reasoning_content into assistant messages that lack it.

        Some providers (Kimi, Z.AI) require all assistant messages to have
        ``reasoning_content`` when thinking mode is enabled.  Returns the
        original list reference if no injection was needed.
        """
        modified = False
        new_messages = []
        for msg in messages:
            if msg.get("role") == "assistant" and "reasoning_content" not in msg:
                new_messages.append({**msg, "reasoning_content": ""})
                modified = True
            else:
                new_messages.append(msg)
        return new_messages if modified else messages

    def _detect_thinking_from_messages(self, messages: list[dict], *, require_non_empty: bool = False) -> bool:
        """Return True if any assistant message has ``reasoning_content``.

        When the upstream provider defaults thinking to enabled (e.g. DeepSeek,
        Kimi) but the agent does not send an explicit ``thinking`` signal, the
        presence of ``reasoning_content`` in any assistant message proves
        thinking is active and reasoning echo-back is required.

        Args:
            messages: Chat Completions message list.
            require_non_empty: When True, only matches non-empty
                ``reasoning_content`` (Kimi's requirement).  When False,
                matches the key's mere presence (DeepSeek's requirement).
        """
        if require_non_empty:
            return any(msg.get("role") == "assistant" and msg.get("reasoning_content") for msg in messages)
        return any(msg.get("role") == "assistant" and "reasoning_content" in msg for msg in messages)

    def translate_from_upstream(self, raw_response: dict) -> dict:
        """Translate an upstream JSON response into Chat Completions format.

        Default returns the response unchanged (passthrough).  Override for
        providers whose response format differs from Chat Completions.

        Args:
            raw_response: Parsed JSON response from the upstream provider.

        Returns:
            Dict in Chat Completions response format.
        """
        return raw_response

    def translate_upstream_stream_event(self, raw_bytes: bytes) -> list[bytes]:
        """Translate a raw upstream SSE chunk into downstream SSE chunks.

        Default wraps the raw bytes in a single-element list (passthrough).
        Override for providers whose SSE event format differs.

        Args:
            raw_bytes: Raw bytes received from the upstream SSE stream.

        Returns:
            List of raw byte chunks to forward to the downstream client.
        """
        return [raw_bytes]

    @abstractmethod
    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        """Build a Chat Completions API request payload.

        Args:
            model: Model identifier string.
            messages: Normalized message list (role, content, tool_calls, etc.).
            **kwargs: Additional request parameters (stream, tools, temperature, etc.).
        """

    @abstractmethod
    def parse_response(self, response_data: dict) -> dict:
        """Parse an upstream Chat Completions response into a normalized dict."""

    @abstractmethod
    def map_error(self, status_code: int, body: dict) -> Exception:
        """Map an HTTP error status and body to a typed exception."""

    # ── Custom transport ─────────────────────────────────────────────────

    @property
    def validation_model(self) -> str:
        """Model name used for API key validation.

        Override in providers that require a specific model name for the
        validation request to succeed (e.g. providers that reject unknown
        models with 401 instead of 400).
        """
        return "test"

    @property
    def requires_custom_url(self) -> bool:
        """Whether this provider requires a custom base URL from the user.

        When True, the profile creation flow prompts for a base URL and
        stores it in ``provider_config["base_url"]``.
        """
        return False

    @property
    def requires_oauth(self) -> bool:
        """Whether this provider uses OAuth instead of a static API key.

        Override in providers that authenticate via a browser-based OAuth flow
        (e.g. OpenAI ChatGPT subscription).  When True, profile creation
        wizards launch the OAuth flow instead of prompting for an API key.
        """
        return False

    @property
    def use_native_messages(self) -> bool:
        """Whether this provider accepts Anthropic Messages API bodies directly.

        When True, the bridge's ``/v1/messages`` path skips the Messages →
        Chat Completions translation layer and forwards Messages API request,
        response, and SSE event formats directly through this adapter.
        """
        return False

    @property
    def upstream_wire_is_messages_api(self) -> bool:
        """Whether ``translate_to_upstream`` emits an Anthropic Messages body.

        Distinct from :attr:`use_native_messages`, which says whether the
        bridge may skip its own translation layer.  This one describes the
        shape that actually goes on the wire, and the two can disagree: an
        Anthropic-wire adapter whose ``_native_messages_request`` flag was
        cleared (by the bridge's ``tool_use`` format fallback) still emits an
        Anthropic body, built from the Chat Completions request.

        Anything shaping the serialized body — notably the bridge's thinking
        round-trip repair — must branch on the declared wire shape, never on
        the request flag.

        On an adapter that routes by model this answers only for the **default**
        route.  A caller holding a model must ask
        :meth:`upstream_wire_is_messages_api_for_model` instead: branching on
        this property with a routed adapter in hand is what KBR-7 was.
        """
        return False

    def upstream_wire_is_messages_api_for_model(self, model: str) -> bool:
        """Whether ``translate_to_upstream`` emits a Messages body for *model*.

        The per-model form of :attr:`upstream_wire_is_messages_api`, for
        adapters that route to different endpoints depending on the model —
        the same pairing as :attr:`upstream_path` / :meth:`get_upstream_path`
        and ``build_upstream_headers`` / ``build_upstream_headers_for_model``.

        Callers that have a model in hand must use this rather than the bare
        property, and must read the model from the request being serialized so
        the two agree by construction (KBR-7).

        An adapter that routes by model overrides **both**: this, mirroring its
        own routing predicate, and the property, reporting its default route.

        Args:
            model: The model name the request will be sent with, exactly as
                ``translate_to_upstream`` reads it: the normalized model in
                ``cc_request``.  Since KBR-127 the URL and header helpers
                resolve from that same key, so an implementation must not
                normalize again — it would disagree with its own router.

        Returns:
            True when the body for *model* is an Anthropic Messages body.
        """
        return self.upstream_wire_is_messages_api

    @property
    def use_custom_transport(self) -> bool:
        """Whether this provider handles its own HTTP transport.

        When True, the bridge delegates upstream HTTP calls to
        ``make_request`` / ``stream_request`` instead of using its
        own aiohttp client.  Providers that require a specialized
        HTTP client (e.g. AWS SigV4 via boto3) should override
        this to return True.
        """
        return False

    def supports_egress(self, resolved_key: str, provider_config: dict) -> bool:
        """Whether this provider can route its traffic through an egress proxy.

        Adapters that use the bridge's HTTP session inherit egress for free.
        Only adapters owning their transport can be unable to honour it, and
        they must say so here — when egress is configured, kitty refuses to
        start rather than let a provider connect from the machine's own IP.

        Args:
            resolved_key: The credential this provider will use, since the
                answer can depend on the auth mode.
            provider_config: Per-profile provider configuration.

        Returns:
            True if all of this provider's traffic will honour the proxy.
        """
        return True

    async def make_request(self, cc_request: dict) -> dict:
        """Perform a non-streaming upstream request using custom transport.

        Override when ``use_custom_transport`` is True.  The ``cc_request``
        is a fully normalized Chat Completions request dict — the provider
        must translate it to the upstream format, make the HTTP call, and
        return a Chat Completions response dict.

        Raises:
            NotImplementedError: If not overridden by a custom-transport provider.
        """
        raise NotImplementedError("Custom transport provider must implement make_request()")

    async def stream_request(
        self,
        cc_request: dict,
        write: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Perform a streaming upstream request using custom transport.

        Override when ``use_custom_transport`` is True.  The provider must
        translate the request, open a streaming connection, and call
        ``write`` with CC-format SSE chunks as they arrive.

        Args:
            cc_request: Fully normalized Chat Completions request dict.
            write: Async callback to send bytes to the downstream client.

        Raises:
            NotImplementedError: If not overridden by a custom-transport provider.
        """
        raise NotImplementedError("Custom transport provider must implement stream_request()")


class ProviderError(Exception):
    """Base exception for upstream provider adapter errors."""

    is_cloudflare = False
    http_status: int = 0
    retry_after: int | None = None


__all__ = ["ProviderAdapter", "ProviderError"]
