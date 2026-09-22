"""Routing expectation tests for the transparency oracle.

`.system_design/TEST_SUITE.md` §3.3.5 · plan task **T-D2** (KBR-52).

§3.3.5: routing is part of the request, and the body cannot show it. On Azure
the deployment id lives in the URL (P20) while P6 deliberately removes
``model`` from the body, so two requests to two different deployments have
**byte-identical bodies** — a body-only oracle cannot tell them apart, and a
misrouted request is a wrong model billed to a wrong account. The oracle
therefore takes a caller-derived routing expectation and compares it against
the captured request, component by component.

**The expectation is derived independently** — the same rule §3.3.1 applies to
bodies. :func:`_expected_azure_route` computes the route from the configured
profile using Azure's *published* URL shape: asking the code under test where
it meant to go and confirming it went there proves nothing. The derivation
helpers reference no ``src/kitty`` symbol, and the only occurrence of the
imported adapter symbol in this module is inside the binding's ``bind()``
override — a helper calling ``AzureOpenAIAdapter().get_upstream_path(...)``
would evade a naive import-line grep and is forbidden by construction. The
module's one ``src/kitty`` import is the binding's adapter below, test
infrastructure and not part of the derivation. Two behaviours are reimplemented
rather than observed, because §3.3.5 names both as the derivation's
obligation:

* the cut of a pasted full endpoint at the ``/openai/deployments/`` marker
  (Azure's ``_cut_deployment_segment`` rule — the Azure-specific operation
  behind §3.3.5's generic trailing-suffix paragraph);
* the model half: the deployment is the prefix-stripped profile model when the
  profile names one, else the prefix-stripped model the agent asked for
  (``_route_model``'s rule, KBR-127).

**Authority and scheme are rewritten before comparing, and this is not
optional** (§3.3.5, T-W4's scope addition). A published URL shape is
``https://`` on the provider's own hostname; the harness serves
``http://127.0.0.1:<ephemeral>`` — the recorder captures the ``Host`` header
verbatim, port included, precisely so two recorders stay distinguishable. The
derivation therefore builds the published route first and then rewrites
scheme and host with the captured request's own. Path and query need no such
treatment, and they are where the Azure falsification lives.

**The falsification (plan §1.4 harness rule).** A captured request whose
deployment path segment is replaced while the body stays byte-identical must
fail the oracle — and fail on routing: the body obligations pass first, so
the ``RoutingMismatchError`` type itself proves the failure is the route and
not a content delta.

**Scope (out).** §3.3.5's seventh falsification case (KBR-127 — prefixed
model, with path *and* auth scheme *and* body shape agreeing) is T-D3's: the
auth-scheme leg is a header claim and the body-shape leg is the projection
diff, both outside the routing assertion. The driven positive case below uses
a prefixed profile model so the derivation's prefix-strip is exercised
through the bridge and not only in the pure derivation tests.

Tests run at ``l1`` (path-default per ``tests/layers.py`` and the T-D1
precedent in ``test_oracle_driven.py``). ``l3`` activation is T-K6's job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from harness import oracle
from harness import register as r
from harness.bridge import (
    AiohttpTransport,
    Binding,
    BridgeFixture,
    InboundProtocol,
    inbound_path,
    minimal_inbound_body,
    redirected,
)
from harness.contract import CapturedRequest, WireFormat
from kitty.providers.azure import AzureOpenAIAdapter
from kitty.providers.opencode import OpenCodeGoAdapter

#: The marker the oracle should find verbatim in the upstream body. A short
#: string survives every layer of JSON encoding without quoting artifacts.
_SENTINEL = "kbr52-oracle-routing"

#: Azure's api-version, as the published shape pins it. Duplicated from the
#: adapter's constant on purpose: deriving it from ``src/kitty`` would violate
#: the independent-derivation rule. A bump of the adapter's version is a
#: routing-visible wire change (P20 claims ``route.query``), and this literal
#: is what notices it.
_AZURE_PUBLISHED_API_VERSION = "2024-10-21"

#: The deployment marker Azure's published endpoint carries. The derivation
#: cuts a pasted ``base_url`` here, exactly as the adapter is documented to.
_DEPLOYMENT_MARKER = "/openai/deployments/"

#: A synthetic published resource host, for the pure derivation tests. Never
#: resolved: the derivation reads it out of the profile's ``base_url`` and the
#: authority rewrite replaces it before any comparison.
_PUBLISHED_RESOURCE = "my-resource.openai.azure.com"

#: A minimal Chat Completions body, identical on both sides of the unit tests
#: so the structural diff yields no body deltas and the routing assertion is
#: the only thing that can fire.
_BODY_CC = json.dumps(
    {"model": "x", "messages": [{"role": "user", "content": "x"}]}
).encode("utf-8")

#: Triggers the Azure route arranges for the positive and falsification runs.
#: ``ALWAYS`` is met on every request and is what activates the always-on
#: Azure rows (P6 removes the model, P20 encodes it in the path) — the
#: claim-matching loop keeps a row active only when its trigger is in this
#: set, so omitting ``ALWAYS`` would leave the model delta unclaimed and
#: assertion 1 would fail a correct run. ``NON_NATIVE_UPSTREAM_WIRE`` because
#: the upstream wire is CC, not the agent's Messages; ``PROFILE_SETS_MODEL``
#: because the profile pins a model (M1's trigger).
_ROUTE_TRIGGERS: frozenset[r.Trigger] = frozenset(
    {
        r.Trigger.ALWAYS,
        r.Trigger.NON_NATIVE_UPSTREAM_WIRE,
        r.Trigger.PROFILE_SETS_MODEL,
    }
)

#: The opencode_go model §3.3.5's seventh case (T-D3, KBR-53) drives. The
#: profile model carries the provider prefix; ``minimax-m2.5`` is published
#: on ``/v1/messages`` per the endpoint table below (verified 2026-09-11).
#: KBR-127's ticket names this exact case.
_PREFIXED_MESSAGES_MODEL = "opencode/minimax-m2.5"

#: The published endpoint table, read as data. The seventh case's path leg
#: looks the bare model up here — it never imports ``get_upstream_path``.
_ENDPOINTS_TABLE: dict = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "opencode_go_endpoints.json").read_text(
        encoding="utf-8"
    )
)


# --------------------------------------------------------------------------
# The independent derivation
# --------------------------------------------------------------------------


def _strip_provider_prefix(model: str) -> str:
    """Strip a ``provider/model`` prefix, reimplementing Azure's rule.

    The bridge resolves the routing model through the adapter's
    ``normalize_model_name`` (KBR-127); the derivation must reach the same
    answer without calling it. Azure's rule: split on the first ``/`` and
    keep the tail, keeping the input itself when the tail would be empty.

    Args:
        model: The model name as the profile or the agent wrote it.

    Returns:
        The deployment name the published URL shape addresses.
    """
    if "/" in model:
        return model.split("/", 1)[1] or model
    return model


def _cut_deployment_segment(base_url: str) -> str:
    """Cut a pasted full endpoint back to the resource root.

    Users paste the endpoint the provider's documentation shows; the bridge
    cuts it at the deployment marker so the composed path is not doubled. The
    derivation reproduces the rule rather than observing it (§3.3.5): the cut
    is what makes a pasted full endpoint equivalent to the resource root,
    because the pasted deployment segment is removed before the published
    path is composed from the model.

    The adapter's own cut also guards ``urlsplit``'s ``ValueError`` on a
    malformed IPv6 literal; the derivation omits that guard **deliberately**
    — a profile that reaches a driven run has already been validated as an
    ``http(s)://`` URL by the adapter (``build_base_url`` raises before any
    request is composed), so the derivation only ever sees a parseable one,
    and a second guard here would be dead code for an impossible input.

    Args:
        base_url: The profile's configured base URL, in either form.

    Returns:
        The URL with its path truncated at the marker, or unchanged when the
        marker is absent.
    """
    parts = urlsplit(base_url)
    offset = parts.path.find(_DEPLOYMENT_MARKER)
    if offset == -1:
        return base_url
    return urlunsplit(parts._replace(path=parts.path[:offset]))


def _expected_azure_route(
    *,
    profile_model: str | None,
    inbound_model: str,
    base_url: str,
    captured: CapturedRequest,
) -> oracle.ExpectedRoute:
    """Derive the expected route from the profile, independently of kitty.

    Computes Azure's published URL shape —
    ``<base_url>/openai/deployments/<deployment>/chat/completions?api-version=<v>``
    — from the configured profile: the deployment is the prefix-stripped
    profile model when the profile names one, else the prefix-stripped model
    the agent asked for. The scheme and host of the published shape are then
    **rewritten with the captured request's own** (§3.3.5's mandatory
    normalisation): the recorder serves ``http`` on an ephemeral loopback
    port, so a published authority would mismatch by construction on every
    comparison. Path and query are compared exactly as derived.

    Args:
        profile_model: The profile's model, or ``None`` when it names none.
        inbound_model: The model the agent's request carried.
        base_url: The profile's configured base URL, in either accepted form.
        captured: The captured request, whose scheme and host replace the
            published authority.

    Returns:
        The routing expectation for :func:`oracle.assert_no_unclaimed_mutation`.
    """
    # The model half: profile override first, then the agent's ask — the same
    # precedence `_route_model` documents, reached without importing it.
    routed = profile_model if profile_model else inbound_model
    deployment = _strip_provider_prefix(routed)

    # The published shape, scheme and host taken from the (cut) base URL.
    root = _cut_deployment_segment(base_url)
    parts = urlsplit(root)
    published = oracle.ExpectedRoute(
        method="POST",
        scheme=parts.scheme,
        host=parts.netloc,
        path=f"{_DEPLOYMENT_MARKER}{deployment}/chat/completions",
        query=f"api-version={_AZURE_PUBLISHED_API_VERSION}",
    )

    # §3.3.5: rewrite the authority with the harness's own before comparing.
    # Kept as an explicit step so the rule is visible where a reader would
    # otherwise conclude the recorder was wrong and "fix" it.
    return replace(published, scheme=captured.scheme, host=captured.host)


# --------------------------------------------------------------------------
# The Azure binding
# --------------------------------------------------------------------------


@dataclass
class _AzureAiohttpTransport(AiohttpTransport):
    """The default recorder reached through ``AzureOpenAIAdapter``.

    ``_ADAPTER_FOR_FORMAT`` cannot express this binding: that map is keyed by
    wire format and Azure's format is Chat Completions, already taken by
    ``custom_openai``. ``bind()`` is the seam the harness documents for
    exactly this — the fixture never reads a recorder's base URL itself, so a
    transport author supplies the adapter/config pair. Not registered in the
    transport registry: the name is local to this module, and the registry
    exists so two Epic modules cannot disagree about what a name means.
    """

    #: A class attribute, not a field, matching the parent's convention.
    name = "azure-aiohttp"

    def bind(self) -> Binding:
        """Return the Azure adapter pointed at this recorder.

        Returns:
            A fresh ``AzureOpenAIAdapter`` and the provider configuration
            naming the recorder's base URL — the resource root, with no query
            string, so the captured query is exactly the derivation's literal.
        """
        return AzureOpenAIAdapter(), {"base_url": self._recorder.base_url}


@dataclass
class _OpencodeGoAiohttpTransport(AiohttpTransport):
    """The default recorder reached through ``OpenCodeGoAdapter``.

    Same registration reason as the Azure binding: ``_ADAPTER_FOR_FORMAT``
    is keyed by wire format and opencode_go's Messages-routed models speak
    ``ANTHROPIC_MESSAGES``, already taken by ``custom_anthropic``. Unlike
    Azure's, this binding's ``redirected()`` seam is load-bearing:
    ``OpenCodeGoAdapter`` reads no ``provider_config`` key, so handing the
    config ``{"base_url": ...}`` alone would leave ``build_base_url``
    returning the real opencode.ai — the bridge would post where the
    harness cannot see.
    """

    #: A class attribute, not a field, matching the parent's convention.
    name = "opencode-go-aiohttp"

    def bind(self) -> Binding:
        """Return the opencode_go adapter re-hosted onto this recorder.

        Returns:
            A ``redirected()`` copy of ``OpenCodeGoAdapter`` — the seam
            overrides ``build_base_url``, the one method the bridge calls
            for the destination — and the provider configuration naming the
            recorder's base URL.
        """
        return (
            redirected(OpenCodeGoAdapter(), self._recorder.base_url),
            {"base_url": self._recorder.base_url},
        )


def _inbound_capture(path: str, body: dict) -> CapturedRequest:
    """Reconstruct the agent's inbound request as the bridge received it.

    The Messages adapter does not change the inbound scheme/host/path (those
    are the bridge's own), so the oracle's inbound projection is built from
    what Claude Code would have sent — the T-D1 driven precedent.

    Args:
        path: The inbound route, from :func:`inbound_path`.
        body: The posted JSON body.

    Returns:
        The inbound :class:`CapturedRequest` for the oracle's left side.
    """
    return CapturedRequest(
        method="POST",
        scheme="http",
        host="127.0.0.1",
        path=path,
        query="",
        body=json.dumps(body).encode("utf-8"),
    )


async def _drive_azure(
    fixture_model: str | None,
    inbound_model: str,
) -> tuple[CapturedRequest, CapturedRequest, dict, str]:
    """Post a minimal Messages body through the bridge on the Azure binding.

    Args:
        fixture_model: The profile model the fixture is constructed with
            (``None`` for a profile that names none).
        inbound_model: The model the inbound body asks for.

    Returns:
        The inbound capture, the captured upstream request, the inbound body
        dict (for the derivation's model half), and the binding's actual
        ``base_url`` — the derivation reads the profile value the bridge was
        really given, not one reconstructed from the capture.

    Raises:
        AssertionError: When the drive does not produce exactly one upstream
            capture — a second would mean the empty-response retry ladder
            fired, and the routing comparison would judge the wrong request.
    """
    body = minimal_inbound_body(InboundProtocol.MESSAGES, _SENTINEL, model=inbound_model)
    transport = _AzureAiohttpTransport(format=WireFormat.CHAT_COMPLETIONS)
    async with BridgeFixture(transport, model=fixture_model) as fixture:
        status, _ = await fixture.post(inbound_path(InboundProtocol.MESSAGES), body)
        assert status == 200, "the recorder's minimal success reply must come back 200"
        captures = list(fixture.captures)
        assert len(captures) == 1, (
            "exactly one upstream request — a second capture would mean "
            "the empty-response retry ladder fired"
        )
        inbound = _inbound_capture(inbound_path(InboundProtocol.MESSAGES), body)
        return inbound, captures[0], body, transport.recorder.base_url


async def _drive_opencode() -> tuple[CapturedRequest, CapturedRequest]:
    """Post a minimal Messages body through the bridge on the opencode binding.

    The profile model carries the provider prefix
    (``opencode/minimax-m2.5``) so ``normalize_model_name``'s prefix strip
    is exercised through the bridge — the KBR-127 shape.

    Returns:
        The inbound capture and the captured upstream request.

    Raises:
        AssertionError: When the drive does not produce exactly one upstream
            capture — a second would mean a retry ladder fired, and the
            three-leg assertions would judge the wrong request.
    """
    body = minimal_inbound_body(InboundProtocol.MESSAGES, _SENTINEL, model="agent-model")
    transport = _OpencodeGoAiohttpTransport(format=WireFormat.ANTHROPIC_MESSAGES)
    async with BridgeFixture(transport, model=_PREFIXED_MESSAGES_MODEL) as fixture:
        status, _ = await fixture.post(inbound_path(InboundProtocol.MESSAGES), body)
        assert status == 200, "the recorder's minimal success reply must come back 200"
        captures = list(fixture.captures)
        assert len(captures) == 1, (
            "exactly one upstream request — a second capture would mean "
            "a retry ladder fired"
        )
        inbound = _inbound_capture(inbound_path(InboundProtocol.MESSAGES), body)
        return inbound, captures[0]


# --------------------------------------------------------------------------
# Pure derivation tests
# --------------------------------------------------------------------------


class TestTheIndependentDerivation:
    """The derivation computes the published shape without importing kitty."""

    def test_resource_root_yields_the_published_shape(self) -> None:
        """A resource-root base_url derives the documented Azure endpoint."""
        base_url = f"https://{_PUBLISHED_RESOURCE}"
        captured = CapturedRequest(
            method="POST", scheme="http", host="127.0.0.1:1", path="/x", query=""
        )
        route = _expected_azure_route(
            profile_model="azure/my-gpt4o",
            inbound_model="whatever",
            base_url=base_url,
            captured=captured,
        )
        assert route.path == "/openai/deployments/my-gpt4o/chat/completions"
        assert route.query == f"api-version={_AZURE_PUBLISHED_API_VERSION}"
        assert route.method == "POST"
        # The published authority, before the rewrite...
        assert _PUBLISHED_RESOURCE in base_url
        # ...is rewritten with the captured request's own (§3.3.5).
        assert route.scheme == "http"
        assert route.host == "127.0.0.1:1"

    def test_prefixed_model_resolves_to_the_bare_deployment(self) -> None:
        """A ``provider/deployment`` profile model addresses the bare name."""
        route = _expected_azure_route(
            profile_model="azure/my-gpt4o",
            inbound_model="ignored",
            base_url=f"https://{_PUBLISHED_RESOURCE}",
            captured=CapturedRequest(
                method="POST", scheme="http", host="127.0.0.1:1", path="/x", query=""
            ),
        )
        assert "/deployments/my-gpt4o/" in route.path

    def test_no_profile_model_uses_the_agent_model(self) -> None:
        """§3.3.5's other half: without a profile model, the agent's ask routes."""
        route = _expected_azure_route(
            profile_model=None,
            inbound_model="agent-deployment",
            base_url=f"https://{_PUBLISHED_RESOURCE}",
            captured=CapturedRequest(
                method="POST", scheme="http", host="127.0.0.1:1", path="/x", query=""
            ),
        )
        assert "/deployments/agent-deployment/" in route.path

    def test_full_endpoint_base_url_yields_the_same_route(self) -> None:
        """A pasted full endpoint derives the same route as the resource root.

        §3.3.5's explicit warning: a profile whose ``base_url`` already ends
        in ``/chat/completions`` reaches the same destination as one that does
        not, so the derivation must apply the cut or that profile reports a
        mismatch against a request that went exactly where it should. The
        pasted deployment segment and query are *replaced*, not appended.
        """
        captured = CapturedRequest(
            method="POST", scheme="http", host="127.0.0.1:1", path="/x", query=""
        )
        from_root = _expected_azure_route(
            profile_model="my-gpt4o",
            inbound_model="ignored",
            base_url=f"https://{_PUBLISHED_RESOURCE}",
            captured=captured,
        )
        from_full = _expected_azure_route(
            profile_model="my-gpt4o",
            inbound_model="ignored",
            base_url=(
                f"https://{_PUBLISHED_RESOURCE}/openai/deployments/pasted-deployment"
                "/chat/completions?api-version=1999-01-01"
            ),
            captured=captured,
        )
        assert from_full == from_root


# --------------------------------------------------------------------------
# Oracle comparison unit tests
# --------------------------------------------------------------------------


class TestTheRoutingComparison:
    """The oracle compares every route component when an expectation is given."""

    @staticmethod
    def _pair(
        path: str = "/openai/deployments/my-gpt4o/chat/completions",
    ) -> tuple[CapturedRequest, oracle.ExpectedRoute]:
        """Build a captured request and the expectation that matches it.

        The body is the shared minimal CC body so the body obligations pass
        and the routing assertion is isolated.

        Args:
            path: The captured path; tests mutate one component at a time.

        Returns:
            The (captured, expected) pair, equal on every component.
        """
        expected = oracle.ExpectedRoute(
            method="POST",
            scheme="http",
            host="127.0.0.1:1",
            path=path,
            query="api-version=2024-10-21",
        )
        captured = CapturedRequest(
            method=expected.method,
            scheme=expected.scheme,
            host=expected.host,
            path=expected.path,
            query=expected.query,
            body=_BODY_CC,
        )
        return captured, expected

    @staticmethod
    def _mutated(expected: oracle.ExpectedRoute, component: str) -> CapturedRequest:
        """Rewrite exactly one route component of an otherwise-matching capture.

        Args:
            expected: The expectation whose components seed the capture.
            component: The one component to replace with a wrong value.

        Returns:
            A capture equal to the expectation except at ``component``.
        """
        overrides: dict[str, str] = {
            "method": "GET",
            "scheme": "https",
            "host": "wrong.example:9",
            "path": "/other",
            "query": "x=1",
        }
        values = {
            name: overrides[name] if name == component else getattr(expected, name)
            for name in ("method", "scheme", "host", "path", "query")
        }
        return CapturedRequest(body=_BODY_CC, **values)

    def test_a_matching_route_passes_and_is_reported(self) -> None:
        """All five components equal: the oracle passes and records it."""
        captured, expected = self._pair()
        report = oracle.assert_no_unclaimed_mutation(
            inbound=captured,
            inbound_format=WireFormat.CHAT_COMPLETIONS,
            captured=captured,
            captured_format=WireFormat.CHAT_COMPLETIONS,
            register=r.REGISTER,
            triggers_met=frozenset(),
            expected_route=expected,
        )
        assert report.expected_route is expected

    def test_a_changed_path_fails_naming_route_path(self) -> None:
        """The Azure falsification's unit form: a changed path is named."""
        captured, expected = self._pair()
        rerouted = self._mutated(expected, "path")
        try:
            oracle.assert_no_unclaimed_mutation(
                inbound=captured,
                inbound_format=WireFormat.CHAT_COMPLETIONS,
                captured=rerouted,
                captured_format=WireFormat.CHAT_COMPLETIONS,
                register=r.REGISTER,
                triggers_met=frozenset(),
                expected_route=expected,
            )
        except oracle.RoutingMismatchError as exc:
            assert "route.path" in exc.paths
            assert "path" in str(exc)
        else:
            raise AssertionError("a changed path must fail the routing assertion")

    def test_each_component_is_compared(self) -> None:
        """Every route component mismatches on its own when rewritten."""
        for component in ("method", "scheme", "host", "path", "query"):
            captured, expected = self._pair()
            mutated = self._mutated(expected, component)
            try:
                oracle.assert_no_unclaimed_mutation(
                    inbound=captured,
                    inbound_format=WireFormat.CHAT_COMPLETIONS,
                    captured=mutated,
                    captured_format=WireFormat.CHAT_COMPLETIONS,
                    register=r.REGISTER,
                    triggers_met=frozenset(),
                    expected_route=expected,
                )
            except oracle.RoutingMismatchError as exc:
                assert f"route.{component}" in exc.paths, (
                    f"component {component!r} must be named in the failure"
                )
            else:
                raise AssertionError(
                    f"a changed {component!r} must fail the routing assertion"
                )

    def test_a_credential_in_query_is_redacted_in_the_failure_message(self) -> None:
        """§4.3 C2-adjacent: a query mismatch must not leak the credential.

        ``CapturedRequest.__repr__`` routes the ``query`` component through
        :func:`~harness.contract._redact_query`, which masks values whose key
        is in :data:`~harness.contract.REDACTED_QUERY_KEYS` (``key``,
        ``api_key``, ``access_token``). The routing failure message has to
        apply the same redaction — a profile whose ``base_url`` carries
        ``?api_key=SECRET`` would otherwise print the secret into the pytest
        failure message and CI log when a routing mismatch fires. KBR-143's
        merge is the path that brings the credential into the composed
        upstream query; this test pins the unmasked surface as closed.
        """
        # The credential key spelling is the contract's own
        # (:data:`~harness.contract.REDACTED_QUERY_KEYS` = ``key``,
        # ``api_key``, ``access_token``) — the test uses one the contract
        # recognises, since redacting an unrecognised key is not the contract.
        captured = CapturedRequest(
            method="POST", scheme="http", host="127.0.0.1:1",
            path="/v1", query="api_key=SECRET&api-version=2024-10-21", body=_BODY_CC,
        )
        expected = oracle.ExpectedRoute(
            method="POST", scheme="http", host="127.0.0.1:1",
            path="/v2", query="api_key=OTHER&api-version=2024-10-21",
        )
        try:
            oracle.assert_no_unclaimed_mutation(
                inbound=captured,
                inbound_format=WireFormat.CHAT_COMPLETIONS,
                captured=captured,
                captured_format=WireFormat.CHAT_COMPLETIONS,
                register=r.REGISTER,
                triggers_met=frozenset(),
                expected_route=expected,
            )
        except oracle.RoutingMismatchError as exc:
            assert "SECRET" not in str(exc), (
                "credential must not appear in the routing failure message"
            )
            assert "OTHER" not in str(exc), (
                "expected-side credential must not appear either — both sides "
                "are masked, not just the captured"
            )
        else:
            raise AssertionError("a mismatching query must fail the routing assertion")

    def test_none_disables_the_routing_check(self) -> None:
        """``expected_route=None`` keeps T-D1's behaviour: no routing claim."""
        captured, expected = self._pair()
        # A capture that would fail the comparison if one were made: the
        # routing check is absent, not vacuous, when no expectation is given.
        unrouted = self._mutated(expected, "path")
        report = oracle.assert_no_unclaimed_mutation(
            inbound=captured,
            inbound_format=WireFormat.CHAT_COMPLETIONS,
            captured=unrouted,
            captured_format=WireFormat.CHAT_COMPLETIONS,
            register=r.REGISTER,
            triggers_met=frozenset(),
        )
        assert report.expected_route is None


# --------------------------------------------------------------------------
# Driven end-to-end tests
# --------------------------------------------------------------------------


class TestTheAzureRouteDriven:
    """End-to-end: the real bridge's Azure route judged by the derived route."""

    async def test_prefixed_profile_model_routes_to_the_derived_path(self) -> None:
        """Positive: the capture matches the independently derived route.

        The profile model carries a provider prefix (``azure/<deployment>``)
        so the derivation's prefix-strip is exercised through the bridge: the
        captured path must address the bare deployment. The profile's
        ``base_url`` is the recorder's authority — the same string the
        binding handed the bridge — which the derivation reads and then
        rewrites with the captured scheme and host (§3.3.5).
        """
        inbound, captured, body, base_url = await _drive_azure(
            fixture_model="azure/my-gpt4o", inbound_model="agent-model"
        )
        expected = _expected_azure_route(
            profile_model="azure/my-gpt4o",
            inbound_model=body["model"],
            base_url=base_url,
            captured=captured,
        )
        report = oracle.assert_no_unclaimed_mutation(
            inbound=inbound,
            inbound_format=WireFormat.ANTHROPIC_MESSAGES,
            captured=captured,
            captured_format=WireFormat.CHAT_COMPLETIONS,
            register=r.REGISTER,
            triggers_met=_ROUTE_TRIGGERS,
            expected_route=expected,
        )
        # Exactly one delta, and it is registered: the model the bridge
        # moved from the body into the deployment path (P6 + P20, claimed
        # under ALWAYS; M1 also claims it under PROFILE_SETS_MODEL). Reaching
        # the report at all proves every delta was claimed — assertion 1
        # raises otherwise — so pinning the exact tuple documents the route's
        # whole expected difference.
        assert report.deltas == ("envelope.model",)
        assert report.expected_route is expected

    async def test_changed_deployment_segment_with_identical_body_fails(self) -> None:
        """Falsification (§3.3.5's sixth case): the body cannot show it.

        The captured request is rerouted to another deployment with the body
        byte-identical — exactly what a misroute on Azure looks like on the
        wire. The body obligations pass (the body is untouched), so the
        ``RoutingMismatchError`` type itself proves the failure is the route.
        """
        inbound, captured, body, base_url = await _drive_azure(
            fixture_model="azure/my-gpt4o", inbound_model="agent-model"
        )
        expected = _expected_azure_route(
            profile_model="azure/my-gpt4o",
            inbound_model=body["model"],
            base_url=base_url,
            captured=captured,
        )
        # Reroute: swap the deployment segment, keep every byte of the body.
        misrouted = CapturedRequest(
            method=captured.method,
            scheme=captured.scheme,
            host=captured.host,
            path=captured.path.replace("/my-gpt4o/", "/other-deployment/"),
            query=captured.query,
            headers=captured.headers,
            body=captured.body,
        )
        assert misrouted.body == captured.body, "the falsification needs a byte-identical body"
        try:
            oracle.assert_no_unclaimed_mutation(
                inbound=inbound,
                inbound_format=WireFormat.ANTHROPIC_MESSAGES,
                captured=misrouted,
                captured_format=WireFormat.CHAT_COMPLETIONS,
                register=r.REGISTER,
                triggers_met=_ROUTE_TRIGGERS,
                expected_route=expected,
            )
        except oracle.RoutingMismatchError as exc:
            assert "route.path" in exc.paths, (
                f"the deployment falsification must name the path; got {exc.paths!r}"
            )
        else:
            raise AssertionError(
                "a changed deployment segment with a byte-identical body must "
                "fail the oracle — a green run here means the routing "
                "assertion is wired to nothing"
            )

    async def test_no_profile_model_routes_to_the_agent_model(self) -> None:
        """§3.3.5's model-half complement, driven: no profile model set.

        ``PROFILE_SETS_MODEL`` is deliberately absent — the profile names no
        model — so M1 is inactive and P6 (under ``ALWAYS``) is what claims
        the model delta. The deployment in the path comes from the agent's
        ask, which is the derivation's other half.
        """
        inbound, captured, body, base_url = await _drive_azure(
            fixture_model=None, inbound_model="agent-deployment"
        )
        expected = _expected_azure_route(
            profile_model=None,
            inbound_model=body["model"],
            base_url=base_url,
            captured=captured,
        )
        report = oracle.assert_no_unclaimed_mutation(
            inbound=inbound,
            inbound_format=WireFormat.ANTHROPIC_MESSAGES,
            captured=captured,
            captured_format=WireFormat.CHAT_COMPLETIONS,
            register=r.REGISTER,
            triggers_met=frozenset({r.Trigger.ALWAYS, r.Trigger.NON_NATIVE_UPSTREAM_WIRE}),
            expected_route=expected,
        )
        assert report.deltas == ("envelope.model",)


# --------------------------------------------------------------------------
# §3.3.5 seventh falsification case — T-D3 (KBR-53, KBR-127)
# --------------------------------------------------------------------------


class TestPrefixedProfileModel:
    """The prefixed-model conjunction: path, auth scheme and body shape.

    §3.3.5's seventh case, from KBR-127: a profile whose model carries a
    provider prefix must route the same normalized model into the path,
    the auth scheme and the body. The defect showed a correct body
    reaching a correct-looking host at the wrong path under the wrong
    auth — each of the three checks alone passes on it; only their
    conjunction catches it, which is why all three live in one test.
    """

    async def test_prefixed_messages_model_routes_to_messages_on_all_three_legs(self) -> None:
        """All three legs agree on the prefixed Messages model.

        The path comes from the published endpoint table read as data; the
        auth scheme from the published Messages header shape (``x-api-key``,
        no ``Authorization``); the body shape from the Messages reader
        accepting the capture. The oracle run underneath carries
        ``_ROUTE_TRIGGERS`` so the M1 model rewrite is claimed — the
        body-shape leg proves the reader succeeds, not that the oracle
        passes vacuously.
        """
        inbound, captured = await _drive_opencode()

        # Leg 1 — path: the published table, read independently of src/kitty.
        # The published operation path composes with the published base's
        # own path (opencode.ai/zen/go) — the full published URL for this
        # model is base + endpoint, and `redirected()` preserves the base
        # path when it re-hosts the adapter onto the recorder (§3.3.5's
        # scheme/authority-only substitution, the Vertex precedent).
        bare_model = _PREFIXED_MESSAGES_MODEL.rsplit("/", 1)[1]
        published_base_path = urlsplit(_ENDPOINTS_TABLE["base_url"]).path.rstrip("/")
        expected_path = published_base_path + _ENDPOINTS_TABLE["models"][bare_model]
        assert captured.path == expected_path == "/zen/go/v1/messages"

        # Leg 2 — auth scheme: Messages-routed models take Anthropic's headers.
        # The recorder stores headers verbatim as (name, value) pairs; the
        # names are matched exactly as the adapter spells them.
        header_names = {name for name, _ in captured.headers}
        assert "x-api-key" in header_names, (
            f"a Messages-routed model must carry x-api-key; got {sorted(header_names)}"
        )
        assert "Authorization" not in header_names, (
            "a Messages-routed model must not carry Bearer auth; "
            f"got {sorted(header_names)}"
        )

        # Leg 3 — body shape: the Messages reader accepts the capture, and
        # the oracle's full run passes with every delta claimed.
        oracle._reader_for(WireFormat.ANTHROPIC_MESSAGES).read_request(captured)
        report = oracle.assert_no_unclaimed_mutation(
            inbound=inbound,
            inbound_format=WireFormat.ANTHROPIC_MESSAGES,
            captured=captured,
            captured_format=WireFormat.ANTHROPIC_MESSAGES,
            register=r.REGISTER,
            triggers_met=_ROUTE_TRIGGERS,
        )
        assert "envelope.model" in report.deltas
