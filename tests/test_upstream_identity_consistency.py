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
   instead of calling ``build_upstream_headers`` directly, so an adapter that
   routes headers by model — ``opencode_go`` today — is swept on the builder
   the server would actually call.  Calling the base hook directly would
   inspect a dict that never ships.
2. ``openai_subscription`` is swept on ``_build_codex_headers`` as well.  It does
   not override ``build_upstream_headers`` at all, so **the KBR-8 defect is
   invisible without that entry**: its custom transport builds wire headers in
   ``_build_codex_headers`` and hands them to curl_cffi.

Both are §6.2.3's "the hook is not the wire", which already caught this repo out
in KBR-7; here it is the difference between a guard and a decoration.

**Coverage of the three custom-transport adapters,** stated explicitly because
two of the three are the cases where a sweep can look busy and see nothing:

* ``ollama_cloud`` — its transport calls the hook (``make_request`` and
  ``stream_request`` both go through it), so for it the hook genuinely is the
  wire.  Covered.
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
* The OAuth token legs are **not** swept here, and structurally cannot be:
  this file enumerates *adapter header builders*, and those requests are built
  by free functions in :mod:`kitty.auth` that never reach an adapter.  As of
  KBR-161 they carry the same impersonated identity as the API leg, from the
  same source — :mod:`kitty.codex_identity` — and
  ``tests/test_oauth_leg_identity.py`` is the sibling that proves the two legs
  agree **and**, since T-G9, carries their exact-set contract.
* **T-G9 / KBR-78 landed here** (this file was extended, not replaced): the
  exact-set contract — names, absences, casing, value shape per route — is
  asserted below over the same :func:`wire_routes` enumeration, and the
  positive controls above are reused as its falsification fixtures.  What the
  exact-set work added that the KBR-8 sweep could not see: a header *added*,
  *removed*, or *re-cased*; a version *transformed* rather than copied (the
  substring sentinel cannot catch ``__version__.split(".")[0]`` — the
  value-shape comparison can); and ``bedrock``, whose hook never ships and
  whose contract is observed on the botocore request instead.

The identity *policy* — what each adapter should claim to be — is Q1, and is
untouched here: this file only requires that whatever an adapter claims, it
claims it once and does not read it from kitty's release train.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import inspect
import json
import re
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import NamedTuple

import pytest

from kitty.codex_identity import CODEX_CLI_VERSION as _CODEX_CLI_VERSION
from kitty.codex_identity import build_codex_user_agent as _build_codex_user_agent
from kitty.providers.anthropic import _ANTHROPIC_VERSION
from kitty.providers.base import ProviderAdapter
from kitty.providers.bedrock import BedrockAdapter
from kitty.providers.openai import OpenAIAdapter
from kitty.providers.openai_subscription import OpenAISubscriptionAdapter
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

#: The synthetic account id baked into the openai_subscription account-bearing
#: route (T-G9).  The ``_account_id`` route below exercises the conditional
#: ``ChatGPT-Account-Id`` arm of ``_build_codex_headers`` (P9d).
_ACCOUNT_ID = "account-test-1234"

#: The model every route is swept on unless the adapter routes by model.
_DEFAULT_MODEL = "claude-sonnet-4-5"

#: ``OpenCodeGoAdapter`` picks its wire shape — and so its header set — from the
#: model name, so one model per route.  Pairs chosen so the two arms of the
#: ``_is_messages_model`` branch are both exercised — ``claude-sonnet-4-5`` is
#: **outside** the routing table (default arm), ``minimax-m2.5`` is **inside**
#: (Messages arm).
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

    This mirrors ``BridgeServer._build_upstream_headers`` deliberately, rather
    than calling ``build_upstream_headers`` directly.  The server unconditionally
    prefers the per-model builder, so a sweep that called the bare hook would
    inspect, for ``opencode_go``, a dict the server never sends.  The duplication
    is the point: ``TestTheSweepCatchesWhatItClaimsTo`` plants an adapter that
    leaks only through the per-model builder, so dropping this dispatch fails.

    The base class makes ``build_upstream_headers_for_model`` concrete (it
    delegates to ``build_upstream_headers``), so ``hasattr`` is always true and
    the earlier conditional is dead — the mirror now matches the server.
    """
    return provider.build_upstream_headers_for_model(api_key, model)


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
        id_token: JWT id_token passed to ``_build_codex_headers`` — its
            ``chatgpt_account_id`` claim triggers the conditional
            ``ChatGPT-Account-Id`` arm (P9d).  Default is empty, which exercises
            the no-account complement.
    """

    label: str
    provider_type: str
    api_key: str = _API_KEY
    model: str = _DEFAULT_MODEL
    codex_transport: bool = False
    id_token: str = _EMPTY_ID_TOKEN

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
            return provider._build_codex_headers(_ACCESS_TOKEN, self.id_token)  # type: ignore[attr-defined]  # subscription-only custom transport

        return dispatch_headers(provider, self.api_key, self.model)


def wire_routes() -> list[Route]:
    """Enumerate every header-producing route that reaches an upstream provider.

    Returns:
        One :class:`Route` per adapter, plus the extra routes named below.

    Every registry entry contributes its default route.  Four adapters
    contribute more, and each addition exists because the default route would
    otherwise miss something real:

    * ``openai_subscription`` — its wire headers come from
      ``_build_codex_headers``, which the header dispatch never reaches, so
      **the KBR-8 defect is invisible without this route**.  Two routes:
      the unconditional one and the account-bearing one that exercises the
      conditional ``ChatGPT-Account-Id`` arm (P9d).
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
    routes.append(
        Route(
            label="openai_subscription[_build_codex_headers+account]",
            provider_type="openai_subscription",
            codex_transport=True,
            id_token=_id_token_with_account(_ACCOUNT_ID),
        )
    )
    routes.extend(
        Route(label=f"opencode_go[{model}]", provider_type="opencode_go", model=model)
        for model in _OPENCODE_MODELS
    )
    routes.append(Route(label="azure[entra]", provider_type="azure", api_key=_ENTRA_KEY))

    return routes


def _id_token_with_account(account_id: str) -> str:
    """Build a synthetic JWT id_token carrying the OpenAI account claim.

    Mirrors the parsing path of
    :meth:`OpenAISubscriptionAdapter._extract_account_id` — which reads the
    payload section, base64url-decodes, and reads
    ``payload["https://api.openai.com/auth"]["chatgpt_account_id"]``.  The
    test never sends this to OpenAI; it exists only to drive the conditional
    ``ChatGPT-Account-Id`` arm (P9d) under the sweep.
    """
    payload = json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}})
    encoded = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    return f"header.{encoded}.sig"


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


# ── T-G9 / KBR-78 — the exact-set header contract ──────────────────────────


#: Header names exempt from the exact-set table — the bridge's hook-level
#: dict is never what ships (boto3/botocore signs the request, sets its own
#: ``User-Agent``, etc.).  The bedrock wire contract is asserted separately
#: below, on the botocore-prepared request.  A future custom-transport adapter
#: that also delegates headers must move to this set AND add a wire
#: observation of its own; the test ``test_only_bedrock_is_exempt_from_the
#: exact_set_table`` pins that boundary.
_WIRE_OBSERVED_ELSEWHERE: frozenset[str] = frozenset({"bedrock"})

#: The exact header set each wire route must carry — names (casing-exact) and
#: values, both.  Values are read off the same single sources the adapters
#: use, so a future drift in either side turns red:
#:
#: * ``codex_identity.CODEX_CLI_VERSION`` / :func:`build_codex_user_agent`
#: * ``kitty.providers.anthropic._ANTHROPIC_VERSION``
#: * the credential strings the route was built with (``_API_KEY``,
#:   ``_ACCESS_TOKEN``, ``_ENTRA_KEY``)
#: * frozen literals ``"claude-code/1.0"``, ``"application/json"``,
#:   ``"text/event-stream"``
#:
#: ``"Content-Type"`` is lowercase (``"content-type"``) on the Anthropic
#: family — that is the register's documented deviation (P9e) and is part of
#: the assertion.  The wire casing is aiohttp's, which is unaddressable; this
#: contract is over the dict the adapter constructs (``TEST_SUITE.md`` §4.3
#: C1 scope rule).
_BEARER_TEMPLATE = "Bearer {key}"


def _build_expected_sets() -> dict[str, dict[str, str]]:
    """Resolve all expected header sets from their single sources.

    Built once at module load — values reference in-process constants whose
    resolution is deterministic, so the table is stable across the run.
    """
    base = {"Authorization": _BEARER_TEMPLATE.format(key=_API_KEY), "Content-Type": "application/json"}
    auth_family = {
        "x-api-key": _API_KEY,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    coding_agent_ua = {
        "Authorization": _BEARER_TEMPLATE.format(key=_API_KEY),
        "Content-Type": "application/json",
        "User-Agent": "claude-code/1.0",
    }
    codex_default = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": _BEARER_TEMPLATE.format(key=_ACCESS_TOKEN),
        "User-Agent": _build_codex_user_agent(),
        "version": _CODEX_CLI_VERSION,
    }

    # The bare registry route — the per-adapter default builder.  The wire
    # for ``openai_subscription`` actually goes through ``_build_codex_headers``
    # (the two extra routes below); this base set still pins the inherited
    # default hasn't been overridden with something unrelated.  ``bedrock`` is
    # exempt (the hook never ships — see TestBedrockWireContract).
    table: dict[str, dict[str, str]] = {label: dict(base) for label in [
        "custom_anthropic",  # inherits the Anthropic family from AnthropicAdapter
        "custom_openai",
        "fireworks",
        "google_aistudio",
        "minimax",
        "minimax_token",  # also inherits the Anthropic family
        "novita",
        "ollama_cloud",
        "openai",
        "openai_subscription",
        "openrouter",
        "vertex",
        "zai_coding_cc",
        "zai_regular",
    ]}

    # Anthropic family (P9e) — overrides return the auth-family set.
    table["anthropic"] = dict(auth_family)
    table["custom_anthropic"] = dict(auth_family)
    table["minimax_token"] = dict(auth_family)

    # Coding-agent User-Agent (P9a / P9c) — base + User-Agent.
    table["kimi"] = dict(coding_agent_ua)
    table["byteplus"] = dict(coding_agent_ua)

    # Mimo (P9b) — drops Authorization, adds api-key, keeps the User-Agent.
    table["mimo"] = {
        "Content-Type": "application/json",
        "User-Agent": "claude-code/1.0",
        "api-key": _API_KEY,
    }

    # Ollama (P9h) — no auth header at all.
    table["ollama"] = {"Content-Type": "application/json"}

    # Azure (P9g) — credential-shape branch.
    table["azure"] = {"api-key": _API_KEY, "Content-Type": "application/json"}
    table["azure[entra]"] = dict(base)
    # The entra path normalises the bearer token's casing to "Bearer <token>".
    # ``_ENTRA_KEY = "Bearer test-entra-token"`` → token = "test-entra-token" →
    # Authorization: "Bearer test-entra-token" — the input verbatim, since the
    # split strips the leading prefix and re-prefixes with a canonical "Bearer ".
    table["azure[entra]"]["Authorization"] = _ENTRA_KEY

    # OpenCode Go (P9e by delegation on its Messages models).  The bare
    # registry route sweeps with the default model, which is outside
    # ``_MESSAGES_MODELS`` → default arm (base).
    table["opencode_go"] = dict(base)
    # ``claude-sonnet-4-5`` is outside ``_MESSAGES_MODELS`` → default arm.
    table["opencode_go[claude-sonnet-4-5]"] = dict(base)
    # ``minimax-m2.5`` is inside ``_MESSAGES_MODELS`` → Messages arm.
    table["opencode_go[minimax-m2.5]"] = dict(auth_family)

    # Z.AI coding (P9f) — Anthropic version + lowercase content-type, but keeps
    # Bearer Authorization (NOT the x-api-key swap that is P9e).
    table["zai_coding"] = {
        "Authorization": _BEARER_TEMPLATE.format(key=_API_KEY),
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    # OpenAI subscription — ``_build_codex_headers`` is the actual wire
    # (covered by the KBR-8 sweep above); the exact-set contract adds the
    # trigger + complement pair per §3.3.4.
    table["openai_subscription[_build_codex_headers]"] = dict(codex_default)
    table["openai_subscription[_build_codex_headers+account]"] = {
        **codex_default,
        "ChatGPT-Account-Id": _ACCOUNT_ID,
    }

    return table


_EXPECTED_SETS: dict[str, dict[str, str]] = _build_expected_sets()


def header_set_violation(headers: Mapping[str, str], expected: Mapping[str, str]) -> str | None:
    """Compare a route's emitted header dict against its expected one.

    Pure over its arguments so the falsification cases can hand it a
    deliberate defect (per ``tests/test_opencode_endpoint_table.py``'s
    ``check_routing`` precedent).  Returns ``None`` when the names match
    **and** every value matches; otherwise a one-line message naming the
    offending axes.  Casing is part of the assertion
    (``TEST_SUITE.md`` §4.3 C1).

    Args:
        headers: The dict an adapter produced.
        expected: The exact name→value map the route must produce.

    Returns:
        A violation message, or ``None`` if everything matches.
    """
    extra = sorted(set(headers) - set(expected))
    missing = sorted(set(expected) - set(headers))
    if extra or missing:
        return (
            f"header set differs: extra={extra}, missing={missing}, "
            f"expected exactly {sorted(expected)}, got {sorted(headers)}"
        )

    value_mismatches = sorted(
        name for name in expected if headers[name] != expected[name]
    )
    if value_mismatches:
        return (
            f"values differ on {value_mismatches}: "
            + ", ".join(f"{name}={headers[name]!r} vs {expected[name]!r}" for name in value_mismatches)
        )

    return None


def forbidden_violation(headers: Mapping[str, str]) -> str | None:
    """Check the bridge-introduced-content forbidden set.

    One prohibition, scanned twice (per ``TEST_SUITE.md`` §4.3 C1): no
    header **name** or **value** may contain ``kitty`` in any casing — a
    bridge-introduced literal would be the cheap fingerprint
    ``X-Kitty-Foo`` would be too, since ``X-Kitty-*`` is the downstream-
    only attribution header family and contains ``kitty`` as a substring,
    so the same scan catches both.  User-supplied credentials can in
    principle contain ``kitty``; this assertion is run over the *fixed*
    test fixtures, where the swept values are the constant credential
    strings and the frozen literals — so a real hit is a real defect, not
    a false positive on user content.

    Args:
        headers: The dict an adapter produced.

    Returns:
        A violation message, or ``None`` if no prohibited name or value
        appears.
    """
    name_hits = sorted(name for name in headers if "kitty" in name.lower())
    value_hits = sorted(
        name for name, value in headers.items() if isinstance(value, str) and "kitty" in value.lower()
    )
    if not name_hits and not value_hits:
        return None

    return (
        f"forbidden bridge-introduced content in headers: names={name_hits}, "
        f"values={value_hits} (TEST_SUITE.md §4.3 C1)"
    )


class TestTheExactSetPerRoute:
    """T-G9: every route emits **exactly** its registered header dict.

    The KBR-8 sweep catches *copies* of ``kitty.__version__`` and *self-
    contradictions* in user-agent/version.  T-G9 adds: *extras* (an adapter
    that adds a header), *omissions* (one that drops one), *casing* drift, and
    *transformed* version derivations.  Casing is exact-set equality on names
    and equality on values against the single source each value must come
    from.
    """

    @pytest.mark.parametrize("route", _WIRE_ROUTES, ids=_ROUTE_IDS)
    def test_route_emits_exactly_its_registered_set(self, route: Route) -> None:
        """Names are exact (casing included) and values match the single source."""
        if route.label in _WIRE_OBSERVED_ELSEWHERE:
            pytest.skip(
                f"{route.label} is observed on the botocore-prepared request, "
                f"not on the adapter hook — see test_bedrock_wire_contract below"
            )

        headers = route.headers()
        expected = _EXPECTED_SETS[route.label]

        violation = header_set_violation(headers, expected)

        assert violation is None, (
            f"{route.label}: {violation} "
            f"(TEST_SUITE.md §4.3 C1 / KBR-78 exact-set contract)"
        )

    @pytest.mark.parametrize("route", _WIRE_ROUTES, ids=_ROUTE_IDS)
    def test_route_emits_no_bridge_introduced_kitty_content(self, route: Route) -> None:
        """No header name or value contains ``kitty`` in any casing."""
        if route.label in _WIRE_OBSERVED_ELSEWHERE:
            pytest.skip(f"{route.label} is observed separately — see below")

        violation = forbidden_violation(route.headers())

        assert violation is None, f"{route.label}: {violation}"


class _PlantedExtraHeader(OpenAIAdapter):
    """A registered adapter with a planted extra header.

    Used by AC-1's "added header" falsification.  Picks ``X-Sentinel-Extra``
    so the name is visibly not in any other adapter's set; the value is
    constant so the message can name it.
    """

    @property
    def provider_type(self) -> str:
        """Return the planted registry key."""
        return "_planted_extra"

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Return the base set with a planted extra header."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Sentinel-Extra": "planted",
        }


class _PlantedRecasedHeader(OpenAIAdapter):
    """A registered adapter with a planted re-cased header.

    Used by AC-1's "casing change" falsification.  Spells ``content-type``
    as ``Content-Type`` on a base-default builder — the Anthropic family
    differs from the base by casing, and this asserts a reverse move (base
    → Anthropic casing) is caught as a deviation.
    """

    @property
    def provider_type(self) -> str:
        """Return the planted registry key."""
        return "_planted_recased"

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Return the base set re-cased to the Anthropic spelling."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",  # Anthropic family spelling
        }


class _PlantedDroppedHeader(OpenAIAdapter):
    """A registered adapter that silently stops sending ``Content-Type``.

    Used by AC-1's "removed-header" falsification.  A dropped header is the
    refactoring-shaped defect — a builder rewritten from scratch that
    forgets one of the two base headers — and the exact-set equality is
    what catches it.
    """

    @property
    def provider_type(self) -> str:
        """Return the planted registry key."""
        return "_planted_dropped"

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Return only ``Authorization`` — ``Content-Type`` is gone."""
        return {"Authorization": f"Bearer {api_key}"}


class _PlantedForbiddenHeader(OpenAIAdapter):
    """A registered adapter that ships ``X-Kitty-Foo``.

    Used by AC-1's forbidden-name falsification.
    """

    @property
    def provider_type(self) -> str:
        """Return the planted registry key."""
        return "_planted_forbidden"

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Return the base set with a planted ``X-Kitty-*`` header."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Kitty-Foo": "planted",
        }


class _PlantedForbiddenValue(OpenAIAdapter):
    """A registered adapter whose ``User-Agent`` carries ``kitty`` in the value.

    Used by AC-1's forbidden-value falsification.  Carries a User-Agent whose
    value contains the substring — distinct from the existing KBR-8 sentinel
    check, which only fires when the value contains the *sentinel* (a copy of
    ``__version__``).
    """

    @property
    def provider_type(self) -> str:
        """Return the planted registry key."""
        return "_planted_kitty_ua"

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Return the base set with a planted kitty-bearing User-Agent."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "kitty-bridge/1.0",
        }


class _PlantedUnsweptArm(OpenAIAdapter):
    """A registered adapter whose builder has a branch no swept route reaches.

    Used by AC-3(b)'s falsification: the branch condition is a credential
    value :func:`wire_routes` never passes, so the sweep cannot execute the
    arm — exactly the shape of residual 3, "an adapter that branches on an
    input not enumerated there has one branch unasserted".  A future
    adapter adding a credential shape (or model route) without the sweep
    learning about it reproduces this shape.
    """

    @property
    def provider_type(self) -> str:
        """Return the planted registry key."""
        return "_planted_unswept_arm"

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Branch on a credential shape no swept route carries."""
        if api_key.startswith("sso:"):
            # This arm is unreachable from the sweep: no Route passes an
            # ``sso:``-prefixed key.
            return {"Content-Type": "application/json"}
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }


class _PlantedTransformedVersion(OpenAIAdapter):
    """A registered adapter that transforms ``__version__`` rather than copying.

    Used by AC-2's "transformed derivation" falsification.  Derives
    ``User-Agent`` from ``__version__.split(".")[0]`` — under the sentinel
    this yields ``"0"``, which does NOT contain the sentinel substring and
    therefore passes the KBR-8 sweep's substring check.  The value-shape
    comparison (``User-Agent == "claude-code/1.0"`` for the planted route's
    expectation) catches it.  This is the defect the substring sentinel
    cannot close and the literal-equality value shape can; KBR-78's residual 2.
    """

    @property
    def provider_type(self) -> str:
        """Return the planted registry key."""
        return "_planted_transformed"

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Derive User-Agent from a *transformed* ``__version__``."""
        from kitty import __version__

        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"planted_cli/{__version__.split('.')[0]}",
        }


class TestTheExactSetCatchesEveryPlantedDefect:
    """§1.4: every falsification per claim the exact-set contract makes.

    Each test plants one defect on a registered adapter and asserts the
    :func:`header_set_violation` / :func:`forbidden_violation` check itself
    rejects it.  Driving the checker with the planted shape proves the
    *assertion logic* catches the defect, not merely that the patch was
    applied (the failure mode ``inline assert has no negative control``
    records).
    """

    def test_an_extra_header_is_caught(self) -> None:
        """A registered adapter with a planted extra header fails the check."""
        headers = _PlantedExtraHeader().build_upstream_headers(_API_KEY)
        expected = {"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"}

        violation = header_set_violation(headers, expected)

        assert violation is not None, (
            "an extra header must fail the exact-set check; if it passes, "
            "the assertion is subset-shaped and T-G9 has regressed"
        )
        assert "X-Sentinel-Extra" in violation, (
            "the violation message must name the offending header so a "
            "maintainer diagnoses the failure without opening the test"
        )

    def test_a_recased_header_is_caught(self) -> None:
        """A re-cased header fails the check against the base expectation.

        ``header_set_violation`` compares by exact name, not case-folded
        name.  The Anthropic family's lowercase ``content-type`` differs
        from the base's ``Content-Type``, and a reverse move is a
        deviation — the assertion must not silently accept it.
        """
        # Take the expectation for ``anthropic`` (lowercase content-type)
        # and hand the base builder a dict in its casing.  The planted
        # class is the source of truth for the defect shape.
        headers = _PlantedRecasedHeader().build_upstream_headers(_API_KEY)
        expected = _EXPECTED_SETS["anthropic"]

        violation = header_set_violation(headers, expected)

        assert violation is not None
        # Pin the specific names: ``Content-Type`` is in the planted set but
        # absent from the expected set (extra), and ``content-type`` is
        # expected but absent from the planted set (missing).
        assert "'Content-Type'" in violation, (
            f"the violation must name the extra header; got {violation!r}"
        )
        assert "'content-type'" in violation, (
            f"the violation must name the missing header; got {violation!r}"
        )

    def test_a_dropped_header_is_caught(self) -> None:
        """A missing header fails the check, naming the missing name.

        AC-1's "removed-header" falsification.  An adapter that stops
        sending ``Content-Type`` is a plausible refactor — exactly the
        regression the exact-set equality guards against.
        """
        headers = _PlantedDroppedHeader().build_upstream_headers(_API_KEY)
        expected = _EXPECTED_SETS["openai"]  # a base route for shape reference

        violation = header_set_violation(headers, expected)

        assert violation is not None
        # The specific missing name must appear — readers diagnose from the
        # message, and the literal-word "missing" check is too broad.
        assert "'Content-Type'" in violation, (
            f"the violation must name the missing header; got {violation!r}"
        )

    def test_a_forbidden_x_kitty_header_name_is_caught(self) -> None:
        """An ``X-Kitty-*`` name is forbidden in the upstream header set."""
        headers = _PlantedForbiddenHeader().build_upstream_headers(_API_KEY)

        violation = forbidden_violation(headers)

        assert violation is not None
        assert "X-Kitty-Foo" in violation

    def test_a_forbidden_kitty_value_is_caught(self) -> None:
        """A User-Agent carrying ``kitty`` in its value is forbidden."""
        headers = _PlantedForbiddenValue().build_upstream_headers(_API_KEY)

        violation = forbidden_violation(headers)

        assert violation is not None
        assert "User-Agent" in violation

    def test_a_transformed_version_derivation_is_caught_by_value_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``__version__.split('.')[0]`` passes the substring sentinel — not this."""
        monkeypatch.setattr("kitty.__version__", _SENTINEL)

        headers = _PlantedTransformedVersion().build_upstream_headers(_API_KEY)

        # First, prove the existing KBR-8 substring sentinel is blind to it:
        # the transformed derivation produces "planted_cli/0", which contains
        # no sentinel substring, so ``kitty_version_leaks`` (defined above in
        # this module) reports nothing — and that's the whole point of the
        # residual the value-shape comparison closes.
        leaks = kitty_version_leaks(headers, _SENTINEL)
        assert leaks == [], (
            "the existing substring sentinel must miss the transformed case; "
            f"if it doesn't, the test setup is wrong (got {leaks!r})"
        )

        # The value-shape comparison catches it: the planted builder's
        # expected shape is the base, with no User-Agent, so the extra
        # User-Agent is a header-set deviation.  In the realistic
        # regression shape (a future openai_subscription whose
        # ``_build_user_agent`` derives from ``__version__.split('.')[0]``),
        # the value would compare against ``build_codex_user_agent()`` and
        # fail by value mismatch.
        violation = header_set_violation(
            headers, {"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"}
        )
        assert violation is not None
        assert "User-Agent" in violation


# ── The branch-arm enumeration guard ─────────────────────────────────────────


def _arm_lines_in_block(stmts: list[ast.stmt], out: set[int]) -> None:
    """Collect branch-arm entry lines from one statement block, recursively.

    Three statement shapes count as arms (the guard-style header builders in
    ``providers/`` use exactly these):

    * the first statement of an ``If`` body — the condition-true path;
    * the first statement of an ``If`` orelse — the explicit else path;
    * the statement **following** an ``If`` at the same level — the
      fall-through path.  Python parses ``if c: return X`` + a following
      statement as siblings, not as orelse, even though the following
      statement only executes on the not-condition path (every guard in
      this codebase returns from inside the ``if``).  Treating the sibling
      as the fall-through arm is what makes both azure's credential shapes
      and opencode_go's model routes enumerable.

    Conditional *expressions* are enumerated too: every ``IfExp`` (ternary)
    contributes the lines of both its branches, and ``match`` statements
    contribute each case's first statement.  Nested ``If``s recurse into
    their bodies and orelses.
    """
    for i, stmt in enumerate(stmts):
        if not isinstance(stmt, ast.If):
            continue
        if stmt.body:
            out.add(stmt.body[0].lineno)
            _arm_lines_in_block(stmt.body, out)
        if stmt.orelse:
            out.add(stmt.orelse[0].lineno)
            _arm_lines_in_block(stmt.orelse, out)
        # The fall-through sibling — see the docstring for why it is an arm.
        if i + 1 < len(stmts):
            out.add(stmts[i + 1].lineno)
        # Conditional expressions inside the condition and inside every
        # statement of both arms are enumerated by the expression walk below
        # (they carry no stmt of their own).
        for node in ast.walk(stmt.test):
            if isinstance(node, ast.IfExp):
                out.add(node.body.lineno)
                out.add(node.orelse.lineno)


def _ifexp_arm_lines(fn: ast.FunctionDef) -> set[int]:
    """Return the entry lines of every ``IfExp`` branch anywhere in *fn*.

    A ternary's branches are expressions, so the statement-level walk in
    :func:`_arm_lines_in_block` misses them; they are walked here over the
    whole function.  One shipped builder uses a ternary today
    (``azure.build_upstream_headers``'s token split), and both of its
    branches execute inside the ``if`` arm — which is why this walk, not
    the block walk, owns them.
    """
    out: set[int] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.IfExp):
            out.add(node.body.lineno)
            out.add(node.orelse.lineno)
    return out


def _builder_source_lines(cls: type[ProviderAdapter]) -> dict[str, set[int]]:
    """Collect the AST-derived branch-arm entry lines for each builder.

    Scoped to methods defined **on** the class (not inherited), because an
    inherited builder's arms are enumerated under the ancestor's class —
    and ancestors in the registry are swept in their own right.  The base
    class defines both methods but is not itself in the registry; its
    concrete default is captured by every adapter that does not override.

    Args:
        cls: The adapter class to enumerate.

    Returns:
        Mapping ``method_name -> set of branch-arm entry lines`` (absolute
        source line numbers).  Empty mapping when the class defines no
        overriding builder.
    """
    import textwrap

    sources: dict[str, set[int]] = {}
    for method_name in ("build_upstream_headers", "build_upstream_headers_for_model", "_build_codex_headers"):
        if method_name not in cls.__dict__:
            continue
        method = cls.__dict__[method_name]
        try:
            raw_lines, start_lineno = inspect.getsourcelines(method)
        except (OSError, TypeError):
            continue
        # ``getsourcelines`` returns the method body with its original
        # indentation — a class method is indented one level.  ``ast.parse``
        # needs top-level syntax; ``textwrap.dedent`` strips the common
        # leading whitespace so the ``def`` statement sits at column 0.
        source = textwrap.dedent("".join(raw_lines))
        tree = ast.parse(source)
        # The parsed tree's single top-level FunctionDef is the method.
        fn = tree.body[0]
        # AST line numbers are 1-based inside the dedented snippet;
        # convert to absolute file lines by adding the snippet's offset.
        snippet_to_file = start_lineno - 1
        snippet_arms: set[int] = set()
        _arm_lines_in_block(fn.body, snippet_arms)
        snippet_arms |= _ifexp_arm_lines(fn)
        sources[method_name] = {snippet_to_file + line for line in snippet_arms}

    return sources


def _traced_lines(routes: list[Route]) -> set[tuple[str, int]]:
    """Run ``route.headers()`` under a line tracer scoped to providers/.

    Records every line hit whose file lives under ``src/kitty/providers/``.
    The previous trace function is restored on exit so any outer tracing
    (a debugger, coverage) keeps working.  Nothing is filtered by method —
    the intersection with the AST-derived arm lines (in
    :func:`uncovered_builder_arms`) only looks at lines **inside** builders,
    so the extra hits are ignored by construction.

    Args:
        routes: The routes to drive the trace over.

    Returns:
        The set of ``(filename, line_number)`` tuples hit during the sweep.
    """
    providers_root = str(Path(__file__).resolve().parent.parent / "src" / "kitty" / "providers")
    previous_trace = sys.gettrace()
    hits: set[tuple[str, int]] = set()

    def _trace(frame, event, _arg):  # noqa: ANN001 — settrace callback signature
        if event == "call":
            filename = frame.f_code.co_filename
            # Trace only frames from the providers tree; everything else
            # (pytest, the test file itself, kitty.bridge) is skipped.
            return _trace if filename.startswith(providers_root) else None
        if event == "line":
            filename = frame.f_code.co_filename
            if filename.startswith(providers_root):
                hits.add((filename, frame.f_lineno))
        return _trace

    sys.settrace(_trace)
    try:
        for route in routes:
            route.headers()
    finally:
        sys.settrace(previous_trace)

    return hits


def uncovered_builder_arms(routes: list[Route]) -> list[str]:
    """List every builder branch arm no route in *routes* executed.

    Args:
        routes: The routes to drive.  The full sweep is what the shipped
            claim is about; a filtered list is how the falsification cases
            prove the check bites.

    Returns:
        One message per uncovered arm — ``"{provider_type}.{method} line {L}"``
        — sorted.  Empty when every arm is hit.
    """
    all_lines = _traced_lines(routes)
    uncovered: list[str] = []

    for provider_type in sorted(_registry):
        cls = get_provider(provider_type, {}).__class__
        try:
            source_file = inspect.getsourcefile(cls)
        except (OSError, TypeError):
            # A dynamically-defined class has no source to enumerate; the
            # builder walk below would find nothing either.  Skip cleanly —
            # the same treatment _builder_source_lines gives its methods.
            continue
        for method_name, arm_lines in _builder_source_lines(cls).items():
            for line in arm_lines:
                hit = any(filename == source_file and lineno == line for filename, lineno in all_lines)
                if not hit:
                    uncovered.append(f"{provider_type}.{method_name} line {line}")

    return sorted(uncovered)


class TestTheSweepCoversEveryBuilderArm:
    """§1.4 / residual 3: branches are derived from the source, not a hand list.

    AST-enumerates every ``If`` arm of every overriding builder, runs the
    sweep, and asserts every arm is hit.  A future adapter that adds an
    ``if`` branch in its header builder without the sweep learning about it
    turns this test red, naming the arm.
    """

    def test_no_uncovered_builder_arms(self) -> None:
        """The shipped routes hit every builder arm in every registered adapter."""
        uncovered = uncovered_builder_arms(_WIRE_ROUTES)

        assert uncovered == [], (
            "the sweep did not exercise every header-builder branch in every "
            f"registered adapter: {uncovered}. The sweep's route matrix must "
            f"enumerate every conditional input (KBR-78 / TEST_SUITE.md §4.3 C1 "
            f"residual 3)."
        )

    def test_dropping_the_azure_entra_route_is_caught(self) -> None:
        """Removing the entra route leaves azure's ``if is_entra_token(...)``
        body-arm unhit — the falsification case for the meta-check."""
        without_entra = [r for r in _WIRE_ROUTES if r.label != "azure[entra]"]

        uncovered = uncovered_builder_arms(without_entra)

        assert uncovered != [], (
            "dropping the entra route must leave at least one arm unhit; if "
            "nothing changed, the meta-check is not actually measuring what "
            "it claims"
        )
        assert any("azure" in msg for msg in uncovered), (
            f"the message must name azure as the affected site; got {uncovered}"
        )

    def test_dropping_an_opencode_route_is_caught(self) -> None:
        """Removing every opencode_go route that exercises the default arm leaves it unhit.

        Two routes cover the default arm today — the bare registry route
        (default model, ``claude-sonnet-4-5``) and ``opencode_go[claude-sonnet-4-5]``
        — so both must go for the arm to go dark.  Dropping only one is
        equivalent to keeping the arm (a single route can satisfy it).
        """
        without_default = [
            r for r in _WIRE_ROUTES
            if r.label not in ("opencode_go", "opencode_go[claude-sonnet-4-5]")
        ]

        uncovered = uncovered_builder_arms(without_default)

        assert uncovered != [], (
            "dropping both default-arm opencode_go routes must leave the "
            f"default branch unhit; got {uncovered}"
        )
        assert any("opencode" in msg for msg in uncovered), (
            "the message must name opencode as the affected site; got "
            f"{uncovered}"
        )

    def test_dropping_the_account_bearing_codex_route_is_caught(self) -> None:
        """Removing the account-bearing codex route leaves its arm unhit."""
        without_account = [r for r in _WIRE_ROUTES if r.label != "openai_subscription[_build_codex_headers+account]"]

        uncovered = uncovered_builder_arms(without_account)

        assert any("openai_subscription" in msg for msg in uncovered), (
            "dropping the account-bearing codex route must leave the "
            f"conditional ChatGPT-Account-Id branch unhit; got {uncovered}"
        )

    def test_a_planted_unswept_arm_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An adapter that branches on an input the sweep never passes is named.

        AC-3(b): the enumeration guard is derived from the adapter's own
        source, so a *new* conditional in a *newly registered* adapter —
        one no route in :func:`wire_routes` exercises — turns the meta-test
        red and names the adapter and the uncovered line.  This is
        residual 3's exact failure mode, caught rather than sampled.
        """
        monkeypatch.setitem(_registry, "_planted_unswept_arm", _PlantedUnsweptArm)

        uncovered = uncovered_builder_arms(_WIRE_ROUTES)

        assert uncovered, (
            "a planted adapter whose builder branches on a credential shape "
            "no swept route passes must leave an arm unhit — the enumeration "
            "is not reading the registry, or the planted arm was reached"
        )
        assert any("_planted_unswept_arm" in msg for msg in uncovered), (
            f"the message must name the planted adapter; got {uncovered}"
        )


# ── bedrock — the wire observed, not the hook ────────────────────────────────


class _BedrockShortCircuit:
    """Capture a botocore ``before-send`` request and short-circuit the send.

    Botocore fires ``before-send.<service>.<operation>`` with the fully
    signed ``AWSRequest``; returning an :class:`AWSResponse` skips the
    actual HTTP send.  This is botocore's documented test seam.  The
    capture runs with fake credentials (``AKIAFAKE``) — no network — and
    always returns a 400 ``ValidationException`` body so both
    ``make_request`` and ``stream_request`` exit uniformly via
    :class:`ProviderError`.
    """

    def __init__(self) -> None:
        self.headers: dict[str, str] | None = None
        self.url: str | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch, streaming: bool) -> BedrockAdapter:
        """Build a real adapter and inject the capture into its boto3 client."""

        adapter = BedrockAdapter()
        original = adapter._get_boto3_client
        capture = self

        def patched(resolved_key: str, provider_config: dict):
            client = original(resolved_key, provider_config)
            operation = "ConverseStream" if streaming else "Converse"
            client.meta.events.register_first(f"before-send.bedrock-runtime.{operation}", capture._on_request)
            return client

        monkeypatch.setattr(adapter, "_get_boto3_client", patched)
        return adapter

    def _on_request(self, request, **kwargs):  # noqa: ANN001 — botocore hook signature
        """Record the request, return a 400 to short-circuit without networking."""
        from botocore.awsrequest import AWSResponse

        self.headers = {
            name: (value.decode() if isinstance(value, bytes) else str(value))
            for name, value in request.headers.items()
        }
        self.url = request.url
        # 400 ValidationException — uniform exit for both legs.
        return AWSResponse(
            url=request.url,
            status_code=400,
            headers={},
            raw=_BytesBody(b'{"__type":"ValidationException","message":"probe"}'),
        )


class _BytesBody:
    """A minimal file-like body botocore can stream-read.

    Botocore's response parser reads ``raw.stream()`` (a generator of bytes
    chunks) or ``raw.read()`` (a bytes-returning method).  ``BytesIO`` is
    close but lacks ``stream`` on the installed botocore; ``_BytesBody``
    implements both shapes.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        """Return the full body bytes."""
        return self._data

    def stream(self, chunk_size: int = 1024) -> Iterator[bytes]:
        """Yield the body in ``chunk_size`` chunks, generator-style."""
        yield self._data


def _drive_bedrock_wire(monkeypatch: pytest.MonkeyPatch, *, streaming: bool) -> dict[str, str]:
    """Run one bedrock call and return the captured headers.

    The ``before-send`` hook short-circuits the actual send, so this opens
    no socket; the adapter wraps the botocore call in a thread-pool
    executor and raises :class:`ProviderError` on the 400 response.  The
    captured ``headers`` attribute is read after the exception propagates.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        streaming: ``True`` for the ``converse_stream`` leg, ``False`` for ``converse``.

    Returns:
        The exact header dict botocore prepared.
    """
    import asyncio

    capture = _BedrockShortCircuit()
    adapter = capture.install(monkeypatch, streaming=streaming)
    cc_request = {
        "model": "anthropic.claude-3-sonnet",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "_resolved_key": "AKIAFAKE:fake",
        "_provider_config": {"region": "us-east-1"},
    }

    loop = asyncio.new_event_loop()
    try:
        if streaming:

            async def run():
                await adapter.stream_request(cc_request, write=_echo)

            # 400 → ProviderError; we only wanted the request.
            with contextlib.suppress(Exception):
                loop.run_until_complete(run())
        else:

            async def run():
                return await adapter.make_request(cc_request)

            with contextlib.suppress(Exception):
                loop.run_until_complete(run())
    finally:
        loop.close()

    assert capture.headers is not None, (
        "before-send never fired — botocore event registration did not take"
    )
    return capture.headers


async def _echo(_chunk: bytes) -> None:
    """A no-op write callback for ``stream_request``."""
    return None


#: Botocore's measured header name set for a Converse call (botocore 1.43.93,
#: measured 2026-09-15).  Per ``TEST_SUITE.md`` §6.2.4 ("a contract pins
#: what the code reads, never what it merely tolerates"), these are
#: botocore's, not the bridge's — a botocore upgrade that renames one
#: turns this test red with a clear message.  This is the bedrock §6.2.4
#: dependency contract; ``tests/test_curl_cffi_transport_contract.py``
#: carries the same shape for ``curl_cffi``.
#:
#: **Note on the version pin.**  This file pins the *contract* but does not
#: pin botocore's version in ``pyproject.toml`` — that is T-G11's scope
#: (declare ``botocore`` as an explicit dependency, with the version
#: floor this contract was measured against).  Until T-G11 lands, a
#: botocore upgrade is a known cause of red on this test.
_BEDROCK_EXPECTED_NAMES: frozenset[str] = frozenset(
    {
        "Authorization",
        "Content-Length",
        "Content-Type",
        "User-Agent",
        "X-Amz-Date",
        "amz-sdk-invocation-id",
        "amz-sdk-request",
    }
)

_BEDROCK_SIGV4_PREFIX = "AWS4-HMAC-SHA256 "
_BEDROCK_USER_AGENT_PREFIX = "Boto3/"
_BEDROCK_AMZ_DATE_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")
_BEDROCK_INVOCATION_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def bedrock_wire_violation(headers: Mapping[str, str]) -> str | None:
    """Assert bedrock's botocore-prepared request against the §6.2.4 contract.

    Names are pinned to botocore's measured set; values are pinned by
    prefix / regex so a botocore version bump that changes content does
    not turn red, but a rename does.

    Args:
        headers: The headers botocore captured from ``before-send``.

    Returns:
        A violation message, or ``None`` if every check passes.
    """
    missing = sorted(_BEDROCK_EXPECTED_NAMES - set(headers))
    extra = sorted(set(headers) - _BEDROCK_EXPECTED_NAMES)
    if missing or extra:
        return f"name set differs: missing={missing}, extra={extra}"

    problems: list[str] = []
    if not headers["Authorization"].startswith(_BEDROCK_SIGV4_PREFIX):
        problems.append(f"Authorization={headers['Authorization']!r} lacks SigV4 prefix")
    if not headers["User-Agent"].startswith(_BEDROCK_USER_AGENT_PREFIX):
        problems.append(f"User-Agent={headers['User-Agent']!r} lacks Boto3/ prefix")
    if not _BEDROCK_AMZ_DATE_PATTERN.match(headers["X-Amz-Date"]):
        problems.append(f"X-Amz-Date={headers['X-Amz-Date']!r} does not match the SigV4 timestamp shape")
    if not _BEDROCK_INVOCATION_ID_PATTERN.match(headers["amz-sdk-invocation-id"]):
        problems.append(f"amz-sdk-invocation-id={headers['amz-sdk-invocation-id']!r} is not a UUID")
    if not headers["amz-sdk-request"].startswith("attempt="):
        problems.append(f"amz-sdk-request={headers['amz-sdk-request']!r} lacks the attempt= prefix")
    if headers["Content-Type"] != "application/json":
        problems.append(f"Content-Type={headers['Content-Type']!r} (expected application/json)")

    if problems:
        return "; ".join(problems)

    return forbidden_violation(headers)


class TestBedrockWireContract:
    """KBR-78 residual 1: the bedrock wire is observed, the hook is decoration."""

    @pytest.mark.parametrize("streaming", [False, True], ids=["converse", "converse_stream"])
    def test_bedrock_ships_exactly_botocores_signed_set(self, streaming: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """Botocore's prepared request carries exactly its measured name set."""
        headers = _drive_bedrock_wire(monkeypatch, streaming=streaming)

        violation = bedrock_wire_violation(headers)

        assert violation is None, f"bedrock wire violates its §6.2.4 contract: {violation}"

    def test_bedrock_sentinel_does_not_leak_into_botocores_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under ``kitty.__version__`` patched to the sentinel, no header carries it."""
        monkeypatch.setattr("kitty.__version__", _SENTINEL)

        headers = _drive_bedrock_wire(monkeypatch, streaming=False)

        leaks = kitty_version_leaks(headers, _SENTINEL)
        assert leaks == [], (
            "botocore's request leaked the sentinel — either the bridge set "
            f"a derived header or the contract check is wrong: {leaks}"
        )

    def test_a_user_agent_extra_with_kitty_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A planted branding header carrying ``kitty`` is caught by the forbidden scan.

        This is the bedrock-shaped "someone branded the bridge" defect.
        A future change that adds a header — say, a ``User-Agent`` carrying
        the bridge name, or a custom ``X-Kitty-*`` attribution — turns the
        bridge into a one-line fingerprint for the bedrock leg.  The
        forbidden scan on the captured headers catches it.

        The planted handler is the simplest realistic shape: a ``before-sign``
        event handler that mutates ``request.headers`` to add a
        ``User-Agent`` carrying ``kitty``.  ``before-sign`` fires after
        ``request-created`` and before signing; ``before-send`` then sees
        the modified set, which is what the wire contract asserts.
        """
        adapter = BedrockAdapter()
        original = adapter._get_boto3_client
        capture = _BedrockShortCircuit()

        def patched(resolved_key: str, provider_config: dict):
            client = original(resolved_key, provider_config)

            def branding_handler(request, **kwargs):  # noqa: ANN001 — botocore hook signature
                # ``before-send`` (not ``before-sign``): by the time botocore
                # reaches before-send, its own user-agent builder has run
                # and ``request.headers`` is now an aiohttp-style HeadersDict
                # that supports direct assignment.  At before-sign the
                # builder hasn't run yet, but botocore re-injects the UA
                # AFTER before-sign, so a mutation there is overwritten.
                # The planted defect shape — "someone's branding hook runs
                # late" — is exactly what the contract must catch.
                request.headers["User-Agent"] = b"planted_kitty/1.0"

            # Botocore's ``register_first`` *prepends* to the head each call —
            # registration order therefore matches execution order at the
            # head.  The branding handler must run before the capture so the
            # capture sees the planted UA, not botocore's.
            client.meta.events.register_first("before-send.bedrock-runtime.Converse", branding_handler)
            client.meta.events.register_first("before-send.bedrock-runtime.Converse", capture._on_request)
            return client

        monkeypatch.setattr(adapter, "_get_boto3_client", patched)

        cc_request = {
            "model": "anthropic.claude-3-sonnet",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "_resolved_key": "AKIAFAKE:fake",
            "_provider_config": {"region": "us-east-1"},
        }
        import asyncio

        loop = asyncio.new_event_loop()
        try:

            async def run():
                return await adapter.make_request(cc_request)

            with contextlib.suppress(Exception):
                loop.run_until_complete(run())
        finally:
            loop.close()

        assert capture.headers is not None, (
            "before-send never fired — the capture wiring did not take"
        )
        violation = forbidden_violation(capture.headers)
        assert violation is not None, (
            "a planted User-Agent carrying 'kitty' must trip the forbidden "
            f"scan; observed headers: {sorted(capture.headers)}"
        )
        assert "User-Agent" in violation


# ── The boundary guard ──────────────────────────────────────────────────────


class TestTheBoundaryBetweenHookAndWireIsPinned:
    """Only ``bedrock`` is exempt from the hook-level exact-set table.

    A future custom-transport adapter that also delegates headers must
    move to the exemption set AND add a wire observation of its own; this
    guard pins the boundary so that does not happen silently.
    """

    def test_only_bedrock_is_exempt_from_the_exact_set_table(self) -> None:
        """The exemption set is exactly ``{bedrock}`` — no more, no less."""
        # The wire observation test must exist for bedrock — otherwise the
        # exemption is decoration (the KBR-8 anti-pattern).
        test_method_names = {name for name in dir(TestBedrockWireContract) if name.startswith("test_")}
        assert "test_bedrock_ships_exactly_botocores_signed_set" in test_method_names, (
            "the bedrock exemption requires a live wire observation; "
            "without it the exemption is decoration"
        )

        # The expected set covers every route *except* the exempt set.
        assert set(_EXPECTED_SETS).union(_WIRE_OBSERVED_ELSEWHERE) == set(_ROUTE_IDS), (
            "every swept route is either in the exact-set table or in the "
            f"wire-observed-elsewhere exemption set; got table="
            f"{sorted(_EXPECTED_SETS)} exempt={sorted(_WIRE_OBSERVED_ELSEWHERE)} "
            f"routes={sorted(_ROUTE_IDS)}"
        )
        assert frozenset({"bedrock"}) == _WIRE_OBSERVED_ELSEWHERE, (
            "a custom-transport adapter cannot join the exemption set "
            "without a wire observation of its own; if it did, this "
            f"guard would fail it. Got {_WIRE_OBSERVED_ELSEWHERE}"
        )
