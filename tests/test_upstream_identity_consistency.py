"""Regression guard: one impersonated client version, from one source.

KBR-8.  :class:`~kitty.providers.openai_subscription.OpenAISubscriptionAdapter`
impersonates the Codex CLI and stated its version **twice, from two different
sources, in the same request**: ``User-Agent`` was built as
``codex_cli_rs/{kitty.__version__}`` while the ``version`` header carried the
module constant ``_CODEX_CLI_VERSION``.  Measured on the base revision, one
request claimed to be ``codex_cli_rs/1.9.1`` and ``version: 0.128.0`` at once.

That is two defects, and this module asserts against both — the two further C1
assertions ``TEST_SUITE.md`` §4.3 draws out of finding F1:

* **A self-contradiction.**  No genuine Codex CLI disagrees with itself, so a
  provider comparing the two fields has a one-line rule that identifies bridge
  traffic with no false positives.  A breach of I2 — and unlike the rest of F1
  it is not *missing* information but *contradictory* information, which only an
  intermediary can produce.
* **A version oracle.**  A user-agent derived from ``kitty.__version__`` changes
  when kitty ships and at no other time.  KBR-8 was filed when it read
  ``1.9.0``; it read ``1.9.1`` by the time the fix was written, so the ticket's
  own text was falsified by a kitty release.

**Where this asserts.**  Over the headers each adapter actually puts on the
wire, enumerated by :func:`wire_routes`.  Two decisions make that real rather
than nominal:

1. :func:`dispatch_headers` **mirrors** ``BridgeServer._build_upstream_headers``
   (``server.py:6612-6615``) instead of calling ``build_upstream_headers``
   directly, so an adapter that routes headers by model — ``opencode_go`` today
   — is swept on the builder the server would actually call.  Calling the base
   hook directly would inspect a dict that never ships.
2. ``openai_subscription`` is swept on ``_build_codex_headers`` as well.  It does
   not override ``build_upstream_headers`` at all, so **the KBR-8 defect is
   invisible without that entry**: its custom transport builds wire headers at
   ``openai_subscription.py:564`` and ``:736`` and hands them to curl_cffi.

Both are §6.2.3's "the hook is not the wire", which already caught this repo out
in KBR-7; here it is the difference between a guard and a decoration.

**Coverage of the three custom-transport adapters,** stated explicitly because
two of the three are the cases where a sweep can look busy and see nothing:

* ``ollama_cloud`` — its transport calls the hook (``ollama_cloud.py:374`` and
  ``:403``), so for it the hook genuinely is the wire.  Covered.
* ``openai_subscription`` — covered via ``_build_codex_headers``, as above.
* ``bedrock`` — **not covered.**  It builds no headers of its own; boto3 signs
  the request and botocore sets its own ``User-Agent``.  The dict the inherited
  hook returns never ships.  Residual recorded against T-G9 / KBR-78.

That these three are the *only* custom-transport adapters is not re-asserted
here: ``tests/test_wire_shape_honesty.py`` pins the set mechanically, on the
``use_custom_transport`` property that makes this sweep blind in the first
place.  A fourth adapter forces a decision there, and this list is then stale by
the same test.

**Why behavioural rather than a source scan.**  The checks patch
``kitty.__version__`` and read the headers that come out, instead of grepping
``src/`` for ``__version__``.  A source scan asserts what the code *says*; the
sentinel asserts what an adapter *emits*, and so catches a derivation written in
a form no pattern anticipated.  Adapters are constructed **inside**
:meth:`Route.headers`, under the patch, for the same reason: an adapter that
bound the version in ``__init__`` would escape a sweep that built it first.

**What this does not cover**, so the next reader does not over-trust it:

* The sentinel finds a version *copied* into a header, not one *transformed*
  into it.  ``__version__.split(".")[0]``, or a hash of it, is still a kitty
  version oracle and still passes.  No behavioural sentinel can close that; it
  is named here rather than left to be discovered.
* Adapters branch on their inputs, and only the branches :func:`wire_routes`
  names are swept — both of ``opencode_go``'s model routes and both of
  ``azure``'s credential shapes, one branch of anything added later.
* The token-exchange leg at ``auth/openai_oauth.py:310-312`` sends only
  ``Authorization`` — the same impersonated client making a request under a
  different identity.  That is Q1's territory, not KBR-8's.
* The exact-set header contract — names, absences, casing, value shape per
  adapter — is **T-G9 / KBR-78**, not this file.  This is the defect-scoped
  guard ``TEST_SUITE_IMPLEMENTATION_PLAN.md`` §16 asks for on the KBR-8 row's
  "atomic fix + test now" route, and T-G9 subsumes it the way T-G5 subsumes
  KBR-5's ``tests/bridge/test_vendor_token_guard.py``.

The identity *policy* — what each adapter should claim to be — is Q1, and is
untouched here: this file only requires that whatever an adapter claims, it
claims it once and does not read it from kitty's release train.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import NamedTuple

import pytest

from kitty.providers.base import ProviderAdapter
from kitty.providers.openai import OpenAIAdapter
from kitty.providers.openai_subscription import _CODEX_CLI_VERSION, OpenAISubscriptionAdapter
from kitty.providers.registry import _registry, get_provider

pytestmark = pytest.mark.l2

#: Stand-in for ``kitty.__version__`` during the sweep.  Deliberately not a
#: plausible version string: it must be impossible to confuse with a real value
#: an adapter might legitimately hard-code.
_SENTINEL = "0.0.0-kitty-version-sentinel"

#: Credentials for the builders.  Values are irrelevant — no request is made —
#: but they must be non-empty, since an adapter may branch on a missing key.
_API_KEY = "test-key"
_ACCESS_TOKEN = "test-access-token"

#: ``AzureOpenAIAdapter.build_upstream_headers`` returns a different header set
#: for a Microsoft Entra bearer token than for an API key, so both are swept.
_ENTRA_KEY = "Bearer test-entra-token"

#: ``_extract_account_id`` parses this as a JWT inside a bare ``except``, so an
#: empty string yields no ``ChatGPT-Account-Id`` header rather than raising.
_EMPTY_ID_TOKEN = ""

#: The model every route is swept on unless the adapter routes by model.
_DEFAULT_MODEL = "claude-sonnet-4-5"

#: ``OpenCodeGoAdapter`` picks its wire shape — and so its header set — from the
#: model name, so one model per route.  See ``opencode._MESSAGES_MODELS``.
_OPENCODE_MODELS = ("claude-sonnet-4-5", "minimax-m2.5")

#: A user-agent's leading ``<token>/<version>`` product, as RFC 9110 §5.5.3
#: defines it.  Only the first product is read: Codex CLI's trailing
#: ``(<os> <release>; <arch>)`` comment carries version-like numbers that are the
#: operating system's, not the client's.
_USER_AGENT_PRODUCT = re.compile(r"^(?P<token>[^/\s]+)/(?P<version>[^\s]+)")


def header_value(headers: Mapping[str, str], name: str) -> str | None:
    """Look a header up without regard to the casing it was written in.

    Args:
        headers: The header set an adapter produced.
        name: The header name to find, in any casing.

    Returns:
        The matching header's value, or ``None`` when the set has no such
        header.  The first match wins, which is arbitrary only if a set carries
        the same name in two casings (``Version`` beside ``version`` are distinct
        dict keys); no adapter does, and T-G9 owns the exact set.

    HTTP header names are case-insensitive (RFC 9110 §5.1), and this codebase
    exercises that freely — ``OpenAISubscriptionAdapter`` sends ``User-Agent``
    beside a lowercase ``version``, and ``ZaiAnthropicAdapter`` a capitalised
    ``Authorization`` beside lowercase ``content-type``.  A case-sensitive lookup
    would let this guard pass by simply failing to find the pair it compares.
    """
    lowered = name.lower()

    return next((value for key, value in headers.items() if key.lower() == lowered), None)


def kitty_version_leaks(headers: Mapping[str, str], sentinel: str) -> list[str]:
    """Return every header that carries kitty's version.

    Args:
        headers: The header set an adapter produced while ``kitty.__version__``
            was patched to ``sentinel``.
        sentinel: The stand-in value that was patched in.

    Returns:
        One ``"name: value"`` line per offending header, sorted; empty when no
        header is derived from kitty's version.

    Names are scanned as well as values.  A header *named* for the bridge's
    version would be as good a fingerprint as one valued by it, and costs one
    ``or`` to rule out.
    """
    return sorted(
        f"{name}: {value}"
        for name, value in headers.items()
        if sentinel in name or sentinel in value
    )


def version_disagreement(headers: Mapping[str, str]) -> tuple[str, str] | None:
    """Return the two client versions a header set states, when they differ.

    Args:
        headers: The header set an adapter produced.

    Returns:
        ``(user-agent version, version header)`` when the set states both and
        they disagree; ``None`` when they agree, or when the set does not state
        both and so cannot contradict itself.

    The ``version`` header is matched by its **exact** name, not by any header
    whose name contains "version".  ``anthropic-version: 2023-06-01`` — which
    ``AnthropicAdapter``, ``ZaiAnthropicAdapter`` and ``OpenCodeGoAdapter``'s
    Messages route all send — declares the *API* version, not the client's, and
    comparing it against a user-agent would manufacture a failure out of two
    fields that were never claiming the same thing.
    """
    user_agent = header_value(headers, "user-agent")
    declared = header_value(headers, "version")

    # Neither header alone can contradict anything: this returns None for every
    # adapter that sends one, the other, or neither.
    if user_agent is None or declared is None:
        return None

    product = _USER_AGENT_PRODUCT.match(user_agent)
    if product is None:
        return None

    found = product.group("version")

    return None if found == declared else (found, declared)


def dispatch_headers(provider: ProviderAdapter, api_key: str, model: str) -> dict[str, str]:
    """Build a provider's headers the way the bridge itself builds them.

    Args:
        provider: The adapter to ask.
        api_key: The resolved upstream credential.
        model: The active model, which model-routing adapters select on.

    Returns:
        The header set the bridge would send for this provider and model.

    This mirrors ``BridgeServer._build_upstream_headers``
    (``server.py:6612-6615``) deliberately, rather than calling
    ``build_upstream_headers`` directly.  The server prefers an adapter's
    per-model builder when it has one, so a sweep that skipped that branch would
    inspect, for ``opencode_go``, a dict the server never sends.  The duplication
    is the point: ``TestTheSweepCatchesWhatItClaimsTo`` plants an adapter that
    leaks only through the per-model builder, so dropping this branch fails.
    """
    # `hasattr`, matching the server: the hook is optional and defined on only
    # one adapter, so there is no base-class method to override.
    if hasattr(provider, "build_upstream_headers_for_model"):
        return provider.build_upstream_headers_for_model(api_key, model)  # type: ignore[attr-defined]  # optional provider hook, as in server.py

    return provider.build_upstream_headers(api_key)


class Route(NamedTuple):
    """One header-producing path through an adapter.

    Attributes:
        label: Human-readable identifier, used as the parametrised test id.
        provider_type: Registry key of the adapter to construct.
        api_key: Credential to build with — the shape some adapters branch on.
        model: Active model, which model-routing adapters select on.
        codex_transport: When true, build through
            ``OpenAISubscriptionAdapter._build_codex_headers`` instead of the
            bridge's header dispatch, because that adapter's custom transport
            bypasses the dispatch entirely.
    """

    label: str
    provider_type: str
    api_key: str = _API_KEY
    model: str = _DEFAULT_MODEL
    codex_transport: bool = False

    def headers(self) -> dict[str, str]:
        """Construct the adapter and return the headers this route produces.

        Returns:
            The header set as the bridge would send it for this route.

        The adapter is constructed **here**, not when the route is enumerated,
        so that it is built while ``kitty.__version__`` is patched.  An adapter
        that read the version in ``__init__`` would otherwise bind the real
        value before the sentinel landed, and the sweep would see nothing.
        """
        provider = get_provider(self.provider_type, {})

        if self.codex_transport:
            return provider._build_codex_headers(_ACCESS_TOKEN, _EMPTY_ID_TOKEN)  # type: ignore[attr-defined]  # subscription-only custom transport

        return dispatch_headers(provider, self.api_key, self.model)


def wire_routes() -> list[Route]:
    """Enumerate every header-producing route that reaches an upstream provider.

    Returns:
        One :class:`Route` per adapter, plus the extra routes named below.

    Every registry entry contributes its default route.  Three adapters
    contribute more, and each addition exists because the default route would
    otherwise miss something real:

    * ``openai_subscription`` — its wire headers come from
      ``_build_codex_headers``, which the header dispatch never reaches, so
      **the KBR-8 defect is invisible without this route**;
    * ``opencode_go`` — routes by model, so one model on each of its two routes;
    * ``azure`` — returns ``Authorization`` for a Microsoft Entra bearer token
      and ``api-key`` for an API key, so both credential shapes are swept.
    """
    routes = [
        Route(label=f"{provider_type}", provider_type=provider_type)
        for provider_type in sorted(_registry)
    ]

    routes.append(
        Route(
            label="openai_subscription[_build_codex_headers]",
            provider_type="openai_subscription",
            codex_transport=True,
        )
    )
    routes.extend(
        Route(label=f"opencode_go[{model}]", provider_type="opencode_go", model=model)
        for model in _OPENCODE_MODELS
    )
    routes.append(Route(label="azure[entra]", provider_type="azure", api_key=_ENTRA_KEY))

    return routes


_WIRE_ROUTES = wire_routes()
_ROUTE_IDS = [route.label for route in _WIRE_ROUTES]


class _RegressedSubscriptionAdapter(OpenAISubscriptionAdapter):
    """The KBR-8 defect, restored, so the checks can be shown to catch it.

    ``_build_user_agent`` here reproduces the *defect* this change removed: the
    version comes from ``kitty.__version__`` while the inherited
    ``_build_codex_headers`` still sends ``_CODEX_CLI_VERSION`` in the ``version``
    header, so one object exhibits both defects at once.  The OS suffix is frozen
    rather than read from :mod:`platform`, so the control can assert the emitted
    header by equality instead of by pattern.

    The positive control is the *historical* defect rather than an invented one,
    so it cannot drift away from what the guard claims to prevent — the same
    choice T-G5 makes with M13's synthetic string.
    """

    @staticmethod
    def _build_user_agent() -> str:
        """Build the pre-fix user-agent, derived from kitty's own version.

        Returns:
            A Codex CLI user-agent whose version is ``kitty.__version__``.

        The import is inside the function on purpose.  Read at module scope it
        would bind before ``monkeypatch`` lands and the control would go green
        against a defect it never reproduced.
        """
        from kitty import __version__

        return f"codex_cli_rs/{__version__} (Linux 6.0.0; x86_64)"


class _PlantedVersionOracle(OpenAIAdapter):
    """A registered adapter carrying both defects, for falsifying the sweep.

    Exhibits a user-agent derived from ``kitty.__version__`` *and* a ``version``
    header that disagrees with it, through the ordinary
    :meth:`build_upstream_headers` hook.

    Subclasses a concrete adapter rather than :class:`ProviderAdapter` itself,
    which leaves ``build_request``, ``parse_response`` and ``map_error``
    abstract — three stubs unrelated to headers would be noise here.
    """

    @property
    def provider_type(self) -> str:
        """Return the registry key this planted adapter answers to.

        Returns:
            The planted provider type.
        """
        return "_planted_oracle"

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Build headers exhibiting both KBR-8 defects.

        Args:
            api_key: Resolved upstream credential, echoed into ``Authorization``.

        Returns:
            A header set whose user-agent is versioned from kitty's own version
            and disagrees with the ``version`` header beside it.
        """
        from kitty import __version__

        return {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": f"planted_cli/{__version__}",
            "version": _CODEX_CLI_VERSION,
        }


class _PlantedPerModelOracle(_PlantedVersionOracle):
    """The same defects, reachable only through the per-model header builder.

    The plain hook is clean, so this adapter is caught **only** if the sweep
    follows the bridge's own dispatch into
    :meth:`build_upstream_headers_for_model`.  It is the falsification case for
    :func:`dispatch_headers`: delete that branch and the controls below fail.
    """

    @property
    def provider_type(self) -> str:
        """Return the registry key this planted adapter answers to.

        Returns:
            The planted provider type.
        """
        return "_planted_per_model"

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Build a clean header set, so only the per-model route offends.

        Args:
            api_key: Resolved upstream credential.

        Returns:
            A header set with no user-agent and no version.
        """
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def build_upstream_headers_for_model(self, api_key: str, model: str) -> dict[str, str]:
        """Build headers exhibiting both KBR-8 defects, per model.

        Args:
            api_key: Resolved upstream credential.
            model: Active model, ignored — every model offends.

        Returns:
            A header set whose user-agent is versioned from kitty's own version
            and disagrees with the ``version`` header beside it.
        """
        return _PlantedVersionOracle.build_upstream_headers(self, api_key)


class TestNoHeaderIsDerivedFromKittyVersion:
    """R2: no adapter reads kitty's release train into an upstream header."""

    @pytest.mark.parametrize("route", _WIRE_ROUTES, ids=_ROUTE_IDS)
    def test_route_emits_no_kitty_version(
        self, route: Route, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No header an adapter emits carries kitty's version."""
        monkeypatch.setattr("kitty.__version__", _SENTINEL)
        headers = route.headers()

        # An adapter returning {} would satisfy every assertion below by having
        # emitted nothing at all.
        assert headers, f"{route.label} produced no headers"

        leaks = kitty_version_leaks(headers, _SENTINEL)

        assert not leaks, (
            f"{route.label} derives an upstream header from kitty.__version__, "
            f"which makes the header change with every kitty release and with "
            f"nothing else — a version oracle for the bridge (KBR-8, "
            f"TEST_SUITE.md §4.3 C1). Offending headers: {leaks}"
        )


class TestClientVersionIsStatedOnce:
    """R3: an adapter stating its version twice states the same version twice."""

    @pytest.mark.parametrize("route", _WIRE_ROUTES, ids=_ROUTE_IDS)
    def test_route_does_not_contradict_itself(self, route: Route) -> None:
        """A user-agent version and a ``version`` header agree, where both exist."""
        disagreement = version_disagreement(route.headers())

        assert disagreement is None, (
            f"{route.label} sends two different client versions in one request: "
            f"User-Agent says {disagreement[0]!r} and the version header says "  # type: ignore[index]
            f"{disagreement[1]!r}. No genuine client disagrees with itself, so "  # type: ignore[index]
            f"this is a one-line detection rule for bridge traffic (KBR-8, "
            f"TEST_SUITE.md §4.3 C1)."
        )

    def test_the_agreement_check_is_not_vacuous(self) -> None:
        """Some swept route actually states both versions.

        Exactly one does today — ``openai_subscription``'s Codex transport — so
        the sweep above is a true statement about an empty set everywhere else.
        Without this assertion, renaming the ``version`` header or changing the
        user-agent's shape would empty the population the check inspects, and
        every assertion in this class would stay green while stating nothing.
        """
        pair_bearing = []
        for route in _WIRE_ROUTES:
            headers = route.headers()
            if (
                header_value(headers, "user-agent") is not None
                and header_value(headers, "version") is not None
            ):
                pair_bearing.append(route.label)

        assert "openai_subscription[_build_codex_headers]" in pair_bearing, (
            "No swept route states both a user-agent version and a version "
            f"header, so the agreement check compares nothing. Found: {pair_bearing}"
        )


class TestTheChecksCatchTheDefectTheyDescribe:
    """§1.4 harness rule, part one: the checks fail on the real defect."""

    def test_version_oracle_is_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The sentinel check flags a user-agent derived from kitty's version."""
        monkeypatch.setattr("kitty.__version__", _SENTINEL)
        regressed = _RegressedSubscriptionAdapter()

        leaks = kitty_version_leaks(
            regressed._build_codex_headers(_ACCESS_TOKEN, _EMPTY_ID_TOKEN), _SENTINEL
        )

        assert leaks == [f"User-Agent: codex_cli_rs/{_SENTINEL} (Linux 6.0.0; x86_64)"]

    def test_self_contradiction_is_detected(self) -> None:
        """The agreement check flags the two versions the pre-fix adapter sent."""
        from kitty import __version__

        regressed = _RegressedSubscriptionAdapter()

        disagreement = version_disagreement(
            regressed._build_codex_headers(_ACCESS_TOKEN, _EMPTY_ID_TOKEN)
        )

        assert disagreement == (__version__, _CODEX_CLI_VERSION)

    def test_the_check_distinguishes_agreeing_from_disagreeing_pairs(self) -> None:
        """The agreement check fires on a mismatch and stays silent on a match.

        Both directions in one test because they are one logical fact.  Without
        the agreeing half, a ``version_disagreement`` that returned ``None``
        unconditionally would satisfy every other assertion in this file;
        without the disagreeing half, one that always reported a conflict would.
        """
        assert version_disagreement({"User-Agent": "codex_cli_rs/1.2.3", "version": "1.2.3"}) is None
        assert version_disagreement({"User-Agent": "codex_cli_rs/1.2.3", "version": "9.9.9"}) == (
            "1.2.3",
            "9.9.9",
        )

    def test_an_api_version_header_is_not_read_as_a_client_version(self) -> None:
        """``anthropic-version`` is the API's version and must not be compared."""
        headers = {"User-Agent": "claude-code/1.0", "anthropic-version": "2023-06-01"}

        assert version_disagreement(headers) is None


class TestTheSweepCatchesWhatItClaimsTo:
    """§1.4 harness rule, part two: the *enumeration* is falsified, not just the checks.

    A guard whose check function is proven and whose enumeration is not is the
    failure §1.4 names — proof that a function was called, where the enforcement
    was the branch after it.  Each test here plants a defective adapter in the
    registry and asserts the sweep itself surfaces it.
    """

    def test_a_planted_adapter_is_swept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A registered adapter with both defects is found by the sweep."""
        monkeypatch.setitem(_registry, "_planted_oracle", _PlantedVersionOracle)
        monkeypatch.setattr("kitty.__version__", _SENTINEL)

        planted = [route for route in wire_routes() if route.provider_type == "_planted_oracle"]

        assert planted, "the sweep did not enumerate a newly registered adapter"
        for route in planted:
            headers = route.headers()
            assert kitty_version_leaks(headers, _SENTINEL)
            assert version_disagreement(headers) == (_SENTINEL, _CODEX_CLI_VERSION)

    def test_a_planted_per_model_adapter_is_swept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An adapter that offends only via its per-model builder is still found.

        This is what pins :func:`dispatch_headers` to the server's own dispatch.
        Sweeping ``build_upstream_headers`` directly would read this adapter's
        clean hook and report it innocent.
        """
        monkeypatch.setitem(_registry, "_planted_per_model", _PlantedPerModelOracle)
        monkeypatch.setattr("kitty.__version__", _SENTINEL)

        planted = [route for route in wire_routes() if route.provider_type == "_planted_per_model"]

        assert planted, "the sweep did not enumerate a newly registered adapter"
        for route in planted:
            headers = route.headers()
            assert kitty_version_leaks(headers, _SENTINEL)
            assert version_disagreement(headers) == (_SENTINEL, _CODEX_CLI_VERSION)


class TestTheSweepLooksAtEverything:
    """A sweep that enumerates nothing passes forever; assert what it reaches."""

    def test_every_registered_adapter_is_covered(self) -> None:
        """The sweep reaches every registry entry, not merely some."""
        swept = {route.provider_type for route in _WIRE_ROUTES}

        assert swept == set(_registry)

    def test_the_subscription_wire_builder_is_covered(self) -> None:
        """The one route that makes this guard real is in the enumeration.

        ``openai_subscription`` does not override ``build_upstream_headers``, so
        without this route the sweep is blind to KBR-8 and passes unfixed.
        """
        assert "openai_subscription[_build_codex_headers]" in _ROUTE_IDS

    def test_both_model_routes_of_the_routing_adapter_are_covered(self) -> None:
        """``opencode_go``'s swept models select *different* header sets.

        Asserting the labels alone would be self-referential — this file builds
        both the labels and the list it looks them up in, so the assertion holds
        even if both models land on the same route and one branch goes unswept.
        Comparing the emitted sets is what makes the claim about the adapter.
        ``_MESSAGES_MODELS`` is slated to change under KBR-126, so this is a live
        risk rather than a theoretical one.
        """
        by_model = {
            model: Route(label=model, provider_type="opencode_go", model=model).headers()
            for model in _OPENCODE_MODELS
        }

        for model in _OPENCODE_MODELS:
            assert f"opencode_go[{model}]" in _ROUTE_IDS

        # Compared by header *name*, not by value: the routes are built with the
        # same credential, so a value comparison would pass on two identical
        # branches only by accident, and fail to notice the day they collapse.
        distinct = {tuple(sorted(headers)) for headers in by_model.values()}
        assert len(distinct) == len(_OPENCODE_MODELS), (
            f"the swept models no longer select different header sets, so one "
            f"of opencode_go's routes is unswept: {by_model}"
        )

    def test_both_azure_credential_shapes_are_covered(self) -> None:
        """``azure`` returns a *different* header set per credential shape.

        Same reasoning as above: ``_ENTRA_KEY`` must actually satisfy
        ``AzureOpenAIAdapter.is_entra_token``, or the sweep covers the ``api-key``
        branch twice and the label assertion never notices.
        """
        assert "azure[entra]" in _ROUTE_IDS

        api_key_headers = Route(label="k", provider_type="azure", api_key=_API_KEY).headers()
        entra_headers = Route(label="e", provider_type="azure", api_key=_ENTRA_KEY).headers()

        # By header *name*. The two routes carry different credentials, so their
        # values differ even when both take the same branch -- comparing values
        # would report success for a sweep that covered `api-key` twice.
        assert sorted(api_key_headers) != sorted(entra_headers), (
            f"both azure routes produced the same header names, so "
            f"_ENTRA_KEY no longer reaches the Entra branch and one credential "
            f"shape is unswept: {sorted(api_key_headers)}"
        )
