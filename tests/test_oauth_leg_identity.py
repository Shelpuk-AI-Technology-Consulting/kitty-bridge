"""Regression guard: both legs present one client, from one source.

KBR-161.  :class:`~kitty.providers.openai_subscription.OpenAISubscriptionAdapter`
reached OpenAI over **two clients with two identities**, in one session, to one
vendor: the API leg over ``curl_cffi`` carrying a full impersonated Codex CLI
identity, and the OAuth token leg over a plain ``aiohttp`` session carrying —
across three of its four call sites — no headers at all.  ``get_valid_api_key``
runs on every Codex request, so a long-running session showed a provider a
stream of Chrome-fingerprinted, Codex-identified API calls punctuated at each
token lifetime by an anonymous call to the same vendor's auth host, correlatable
by account.  A genuine Codex CLI refreshes with the client it uses for the API
(``codex-rs/login/src/auth/default_client.rs``); two clients for one account is
a shape no real installation produces.

**Why this is a sibling of** ``tests/test_upstream_identity_consistency.py``
**rather than an extension of it.**  That file is KBR-8's guard and it sweeps
*adapter header builders*, enumerated by its ``wire_routes()``.  The OAuth
requests are built by free functions in :mod:`kitty.auth` that never reach an
adapter, so that sweep is structurally blind to them — no amount of adding rows
to it would see this defect.  The two files divide as: KBR-8's asks "does each
adapter contradict itself?", this one asks "do the two legs agree with each
other?".

**What this does not cover.**  The *transport* half is only half closed, by
decision.  The recurring refresh leg moved onto an impersonating ``curl_cffi``
session, so it now matches the API leg's TLS fingerprint.  The interactive login
leg (sites 3 and 4) stays on ``aiohttp``: it runs from ``kitty auth openai``,
which has no adapter to borrow a session from, and a regression there blocks
sign-in.  That leaves one non-Codex TLS handshake, bound to the account, at
signup — recorded as a named residual in ``TEST_SUITE.md`` §4.5, not fixed here.
This file asserts **identity**, which both legs now share; it asserts nothing
about the fingerprint.

**Why the enumeration guards itself** (§6.2).  The defect this ticket found was
that a hand-written list of token POSTs had three entries where the code had
four — the missing one, ``_exchange_api_key``, sits on the recurring path.  A
guard whose own subject is another hand-written list rots exactly the same way,
so :func:`token_post_sites` derives the list from the AST instead, and
:class:`TestTheEnumerationCatchesANewSite` plants a fifth site to prove it
notices.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty import codex_identity
from kitty.auth.oauth_session import OAUTH_TOKEN_URL, OAuthSession
from kitty.auth.openai_oauth import (
    _exchange_code_for_tokens,
    _exchange_id_token_for_api_key,
)
from kitty.providers.openai_subscription import OpenAISubscriptionAdapter

pytestmark = pytest.mark.l2

SRC = Path(__file__).resolve().parent.parent / "src" / "kitty"

#: Deliberately not a plausible version: it must be impossible to confuse with a
#: value some code might legitimately hard-code.
_SENTINEL = "0.0.0-codex-version-sentinel"

#: The methods a token POST can be issued through: ``post`` on the ``aiohttp``
#: login leg, ``post_form`` on the ``curl_cffi`` refresh leg.  Both are swept,
#: because a scan that knew only one would go quietly blind the next time a leg
#: changes transport — which is precisely what this ticket did.
_POST_METHODS = frozenset({"post", "post_form"})


# ── The enumeration ────────────────────────────────────────────────────────


def token_post_sites(source: str, *, url_names: frozenset[str]) -> list[int]:
    """Find every POST to the OAuth token endpoint in *source*.

    Pure over source text so a deliberate defect can be handed to it without
    writing a module into the package — the shape
    ``tests/test_upstream_route_source_of_truth.py`` already uses.

    Args:
        source: Python source to scan.
        url_names: Names that refer to the token endpoint in this module.

    Returns:
        The line number of each matching call, ascending.
    """
    tree = ast.parse(source)
    found: list[int] = []
    for node in ast.walk(tree):
        # A POST is `<something>.post(...)` or `<something>.post_form(...)`; the
        # receiver is irrelevant, since it is a session on one leg and a
        # transport on the other.
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _POST_METHODS:
            continue
        target = node.args[0] if node.args else None
        if isinstance(target, ast.Name) and target.id in url_names:
            found.append(node.lineno)
    return sorted(found)


def _modules_naming_the_token_url() -> dict[Path, frozenset[str]]:
    """Map each source file to the names it binds the token URL under."""
    modules: dict[Path, frozenset[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value == OAUTH_TOKEN_URL
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        if names:
            modules[path] = frozenset(names)
    return modules


# ── Capturing what each leg actually sends ─────────────────────────────────


class _CapturingTransport:
    """A transport that records headers and answers a full refresh."""

    def __init__(self) -> None:
        self.headers: list[dict[str, str]] = []

    async def post_form(
        self,
        url: str,
        data: Mapping[str, str],
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float,
    ) -> tuple[int, str]:
        self.headers.append(dict(headers or {}))
        return 200, json.dumps(
            {
                "access_token": "at",
                "refresh_token": "rt",
                "id_token": "it",
                "expires_in": 3600,
                "openai_api_key": "sk",
            }
        )


async def _refresh_leg_user_agents() -> list[str]:
    """Drive sites 1 and 2 and return the user-agent each sent."""
    session = OAuthSession(
        client_id="cid",
        access_token="at",
        refresh_token="rt",
        id_token="it",
        api_key="sk",
        access_token_expires_at=0.0,  # expired: forces the refresh
        api_key_expires_at=0.0,
        _file_path=None,
    )
    transport = _CapturingTransport()
    await session.get_valid_api_key(transport)
    return [h.get("User-Agent", "") for h in transport.headers]


async def _login_leg_user_agents() -> list[str]:
    """Drive sites 3 and 4 and return the user-agent each sent."""
    captured: list[dict[str, str]] = []

    def capture(url, **kw):
        captured.append(dict(kw.get("headers") or {}))

    with aioresponses() as m:
        m.post(
            OAUTH_TOKEN_URL,
            callback=capture,
            payload={"access_token": "at", "id_token": "it", "refresh_token": "rt", "expires_in": 3600},
        )
        m.post(OAUTH_TOKEN_URL, callback=capture, payload={"openai_api_key": "sk"})
        async with aiohttp.ClientSession() as http:
            await _exchange_code_for_tokens("code", "verifier", "cid", http)
            await _exchange_id_token_for_api_key("id", "acc", "cid", http)

    return [h.get("User-Agent", "") for h in captured]


def _api_leg_headers() -> dict[str, str]:
    """The headers the API leg puts on the wire."""
    return OpenAISubscriptionAdapter()._build_codex_headers("token", "")


async def _every_user_agent() -> list[str]:
    """All five user-agents: the API leg's, then the four token POSTs'."""
    return [
        _api_leg_headers()["User-Agent"],
        *await _refresh_leg_user_agents(),
        *await _login_leg_user_agents(),
    ]


# ── The claims ─────────────────────────────────────────────────────────────


class TestBothLegsPresentOneIdentity:
    @pytest.mark.asyncio
    async def test_every_leg_sends_the_same_user_agent(self) -> None:
        """Five requests, one client."""
        agents = await _every_user_agent()

        assert len(agents) == 5, agents
        assert len(set(agents)) == 1, agents

    @pytest.mark.asyncio
    async def test_no_leg_omits_the_user_agent(self) -> None:
        """The defect was *absence*, so assert presence explicitly.

        Equality alone would be satisfied by five empty strings.
        """
        agents = await _every_user_agent()

        assert all(agent.startswith("codex_cli_rs/") for agent in agents), agents

    @pytest.mark.asyncio
    async def test_one_source_moves_every_field_together(self, monkeypatch) -> None:
        """Patching one name moves all six version-bearing fields.

        Six, not five: the API leg's ``version`` header is a consumer of the
        same constant, and KBR-8 was precisely those two disagreeing.  A
        module-level alias anywhere would bind at import and fail this.
        """
        monkeypatch.setattr(codex_identity, "CODEX_CLI_VERSION", _SENTINEL)

        agents = await _every_user_agent()
        version_header = _api_leg_headers()["version"]

        assert all(_SENTINEL in agent for agent in agents), agents
        assert version_header == _SENTINEL


class TestTheChecksCatchTheDefectTheyDescribe:
    """§6.2: a guard must be shown to fail on the defect it claims to catch."""

    @pytest.mark.asyncio
    async def test_a_leg_that_sends_no_user_agent_is_caught(self, monkeypatch) -> None:
        """The pre-fix shape: the token POSTs carried no identity."""
        monkeypatch.setattr("kitty.auth.oauth_session.token_request_headers", dict)

        agents = [
            _api_leg_headers()["User-Agent"],
            *await _refresh_leg_user_agents(),
        ]

        assert len(set(agents)) > 1, "stripping the token identity must break agreement"

    @pytest.mark.asyncio
    async def test_a_second_version_source_is_caught(self, monkeypatch) -> None:
        """The KBR-8 shape: one leg reading a different constant."""
        monkeypatch.setattr(
            "kitty.auth.oauth_session.token_request_headers",
            lambda: {"User-Agent": "codex_cli_rs/0.0.0 (Other 1; x86_64)"},
        )

        agents = await _every_user_agent()

        assert len(set(agents)) > 1, "a divergent source must break agreement"


class TestTheEnumerationCatchesANewSite:
    """The list of swept sites is derived, not written down.

    KBR-161's own ticket listed three token POSTs where the code had four. This
    is the check that stops this guard repeating that mistake.
    """

    def test_the_token_url_is_defined_in_exactly_the_expected_modules(self) -> None:
        """A fifth site elsewhere would escape a scan scoped to two files."""
        modules = {path.relative_to(SRC).as_posix() for path in _modules_naming_the_token_url()}

        assert modules == {"auth/oauth_session.py", "auth/openai_oauth.py"}

    def test_the_scan_finds_exactly_the_four_sites_that_are_swept(self) -> None:
        """The AST count and the behavioural sweep must agree.

        If someone adds a fifth POST, this fails even though every existing
        assertion still passes — which is the whole point.
        """
        total = sum(
            len(token_post_sites(path.read_text(encoding="utf-8"), url_names=names))
            for path, names in _modules_naming_the_token_url().items()
        )

        assert total == 4, "four token POSTs are swept behaviourally; the source has a different number"

    def test_the_scan_finds_the_known_sites(self) -> None:
        """A scan that found nothing would satisfy the count check by accident."""
        by_module = {
            path.relative_to(SRC).as_posix(): token_post_sites(path.read_text(encoding="utf-8"), url_names=names)
            for path, names in _modules_naming_the_token_url().items()
        }

        assert len(by_module["auth/oauth_session.py"]) == 2
        assert len(by_module["auth/openai_oauth.py"]) == 2

    def test_a_planted_fifth_site_is_found(self) -> None:
        """Written the way a developer plausibly would, not to suit the scanner."""
        planted = '''
OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"


async def revoke(http, token):
    """Revoke a refresh token."""
    status, body = await http.post_form(
        OAUTH_TOKEN_URL,
        {"token": token},
        headers={},
        timeout=30.0,
    )
    return status
'''

        assert token_post_sites(planted, url_names=frozenset({"OAUTH_TOKEN_URL"})) != []

    def test_a_planted_aiohttp_site_is_found(self) -> None:
        """The login leg's shape, so the scan cannot go blind on one transport."""
        planted = '''
OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"


async def introspect(http, token):
    """Introspect a token."""
    async with http.post(OAUTH_TOKEN_URL, data={"token": token}) as resp:
        return resp.status
'''

        assert token_post_sites(planted, url_names=frozenset({"OAUTH_TOKEN_URL"})) != []

    def test_an_unrelated_post_is_not_counted(self) -> None:
        """So the scan is about the token endpoint, not about POSTs in general."""
        planted = '''
OTHER_URL = "https://example.invalid/x"
OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"


async def send(http):
    """Post somewhere else entirely."""
    return await http.post(OTHER_URL, data={})
'''

        assert token_post_sites(planted, url_names=frozenset({"OAUTH_TOKEN_URL"})) == []


class TestTheVersionHasOneSource:
    """R1, mechanically: the constant that makes all of the above possible."""

    def test_the_impersonated_version_literal_appears_once_in_src(self) -> None:
        """A copy would satisfy every agreement check above and still be KBR-8."""
        version = codex_identity.CODEX_CLI_VERSION
        hits = [
            path.relative_to(SRC).as_posix()
            for path in sorted(SRC.rglob("*.py"))
            for line in path.read_text(encoding="utf-8").splitlines()
            if f'"{version}"' in line or f"'{version}'" in line
        ]

        assert hits == ["codex_identity.py"], hits
