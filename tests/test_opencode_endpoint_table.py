"""Contract guard — kitty's OpenCode Go routing must match the provider's published table.

``.system_design/TEST_SUITE.md`` §6.2.3, "Register and docs ⇄ code".  Closes gap
**G20** / **KBR-126**: ``OpenCodeGoAdapter`` routed two models to
``/v1/messages`` while the provider served eight there, and had no route at all
for the four it serves on ``/v1/responses``.  Six models therefore received a
Chat Completions body at a Messages endpoint, which the provider answers with
``401`` — so the defect reached the user as a *bogus authentication failure*,
not as a routing error.

**What the oracle is, and what it is not.**
``tests/data/opencode_go_endpoints.json`` records what the **provider
publishes**; the frozensets in :mod:`kitty.providers.opencode` record what
**kitty routes**.  They are deliberately two artifacts, not one: deriving the
routing sets from the JSON at import would remove the duplication and add a
worse failure mode, since a missing or corrupt data file would silently route
every model to Chat Completions — this bug, reintroduced, invisibly.  The
duplication is the point, and this module is the assertion that the two agree.

**The honest limit.**  Snapshot and constants are written by the same person in
the same commit, so a green run proves self-consistency at one moment, not
agreement with the provider.  No stronger evidence is reachable without a paid
key: an unauthenticated probe of both endpoints returns ``401 AuthError`` from
each, because auth is evaluated before dialect.  The snapshot's ``probe_note``
records exactly which probe a key-holder could add.  There is consequently **no
staleness alarm** — refreshing the snapshot is a human act, recorded as a gap in
§9.2 rather than papered over with a networked test.

Every check asserts its own subject set, in the style of
``tests/test_egress_coverage.py``, so none can rot into a no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kitty.providers.opencode import _MESSAGES_MODELS, _RESPONSES_MODELS, OpenCodeGoAdapter

# L2: a contract guard between two artifacts that are edited separately — the
# provider's published table and this repository's routing constants.
pytestmark = pytest.mark.l2

_SNAPSHOT_PATH = Path(__file__).parent / "data" / "opencode_go_endpoints.json"

MESSAGES_PATH = "/v1/messages"
RESPONSES_PATH = "/v1/responses"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# The three dialects the provider serves. Asserted rather than narrated: a
# fourth endpoint appearing in the snapshot must force a decision here, not be
# absorbed silently into the default route.
KNOWN_ENDPOINTS = frozenset({MESSAGES_PATH, RESPONSES_PATH, CHAT_COMPLETIONS_PATH})


def load_snapshot() -> dict:
    """Load the provider's published endpoint table from disk.

    Returns:
        The parsed snapshot document, including its provenance fields.
    """
    return json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))


def models_on(snapshot: dict, endpoint: str) -> frozenset[str]:
    """Return the models the snapshot assigns to *endpoint*.

    Args:
        snapshot: A snapshot document, real or synthetic.
        endpoint: One of the paths in :data:`KNOWN_ENDPOINTS`.

    Returns:
        The model names served on that endpoint.
    """
    return frozenset(name for name, path in snapshot.get("models", {}).items() if path == endpoint)


def check_routing(
    snapshot: dict,
    messages_models: frozenset[str],
    responses_models: frozenset[str],
) -> list[str]:
    """Return one problem per disagreement between the snapshot and kitty's routing.

    Pure by design, so the negative tests below can hand it a deliberate defect
    and ask whether it notices — the rule ``tests/layers.py`` states and the
    reason a guard entangled with file IO cannot be trusted.

    Args:
        snapshot: A snapshot document, real or synthetic.
        messages_models: The adapter's Anthropic Messages routing set.
        responses_models: The adapter's OpenAI Responses routing set.

    Returns:
        Human-readable problems, empty when the two agree.  Every problem is
        reported rather than the first: a maintainer fixing a stale table needs
        the whole diff, not one line of it per CI round.
    """
    problems: list[str] = []
    models = snapshot.get("models", {})

    # An empty snapshot satisfies every set comparison below perfectly, so it
    # must be rejected before them. "No disagreements found" is exactly what a
    # guard that stopped looking also reports.
    if not models:
        return ["the snapshot lists no models — the guard would pass by having nothing to check"]

    unknown = {path for path in models.values() if path not in KNOWN_ENDPOINTS}
    if unknown:
        problems.append(f"snapshot uses endpoints this guard does not know: {sorted(unknown)}")

    # Each dialect must be represented, or the snapshot proves nothing about the
    # route it omits while still comparing equal on the two it keeps.
    for endpoint in sorted(KNOWN_ENDPOINTS):
        if not models_on(snapshot, endpoint):
            problems.append(f"snapshot has no model on {endpoint} — that route is unproven")

    # Set equality in BOTH directions. A one-directional check ("every snapshot
    # Messages model is in the constant") would miss a constant naming a model
    # the provider has since moved off /v1/messages — which is the precise shape
    # of the defect this guard closes.
    for endpoint, routed, label in (
        (MESSAGES_PATH, messages_models, "_MESSAGES_MODELS"),
        (RESPONSES_PATH, responses_models, "_RESPONSES_MODELS"),
    ):
        published = models_on(snapshot, endpoint)
        missing = published - routed
        extra = routed - published
        if missing:
            problems.append(f"provider serves {sorted(missing)} on {endpoint}; {label} does not route them there")
        if extra:
            problems.append(f"{label} routes {sorted(extra)} to {endpoint}; the provider does not serve them there")

    return problems


# ── The guard ──────────────────────────────────────────────────────────────


def test_routing_constants_match_the_published_table():
    """R9 — kitty routes each model where the provider says it is served.

    This is the assertion that fails on the unfixed revision: six models the
    provider serves on ``/v1/messages`` and four it serves on ``/v1/responses``
    are routed to Chat Completions.
    """
    problems = check_routing(load_snapshot(), _MESSAGES_MODELS, _RESPONSES_MODELS)

    assert not problems, "kitty's OpenCode Go routing disagrees with the provider:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("model", sorted(load_snapshot()["models"]))
def test_every_published_model_routes_to_its_published_endpoint(model: str):
    """R9 — the behavioural half: ``get_upstream_path`` agrees, not just the sets.

    Asserted against the method rather than the constants because a defect can
    live in the branch itself — a reversed test, or a route checked in the wrong
    order — leaving both frozensets correct and every request misrouted.
    """
    expected = load_snapshot()["models"][model]

    assert OpenCodeGoAdapter().get_upstream_path(model) == expected


# ── Guards on the guard ────────────────────────────────────────────────────


def test_the_snapshot_carries_its_provenance():
    """R8 — a table with no source and no date is folklore, not an oracle."""
    snapshot = load_snapshot()

    assert snapshot["source_url"].startswith("https://")
    assert snapshot["verified_utc"]
    assert snapshot["probe_note"]


def test_the_snapshot_covers_the_whole_catalogue():
    """R8 — pins the size, so a truncated refresh is visible rather than silent."""
    snapshot = load_snapshot()

    assert len(snapshot["models"]) == 28
    assert {path for path in snapshot["models"].values()} == KNOWN_ENDPOINTS


def test_the_checker_rejects_an_empty_snapshot():
    """R9 — the guard must not pass by having nothing to check."""
    assert check_routing({"models": {}}, _MESSAGES_MODELS, _RESPONSES_MODELS)


def test_the_checker_rejects_a_snapshot_missing_an_endpoint_class():
    """R9 — a snapshot that omits a route proves nothing about it.

    Without this, dropping every ``/v1/responses`` row would leave the two
    remaining comparisons equal and the guard green.
    """
    snapshot = load_snapshot()
    snapshot["models"] = {m: p for m, p in snapshot["models"].items() if p != RESPONSES_PATH}

    problems = check_routing(snapshot, _MESSAGES_MODELS, frozenset())

    assert any(RESPONSES_PATH in problem for problem in problems)


def test_the_checker_notices_a_routing_set_the_provider_has_moved_on_from():
    """R9 — the reverse direction, which is the shape of the bug being fixed.

    A model kitty still routes to ``/v1/messages`` after the provider stopped
    serving it there is exactly as wrong as one it never started routing, and a
    one-directional check would see only the second.
    """
    problems = check_routing(load_snapshot(), _MESSAGES_MODELS | {"a-model-the-provider-retired"}, _RESPONSES_MODELS)

    assert any("a-model-the-provider-retired" in problem for problem in problems)


def test_the_checker_notices_a_model_added_to_the_snapshot_but_not_routed():
    """R9 — the forward direction: a refreshed table with no code change fails."""
    snapshot = load_snapshot()
    snapshot["models"]["minimax-m4"] = MESSAGES_PATH

    problems = check_routing(snapshot, _MESSAGES_MODELS, _RESPONSES_MODELS)

    assert any("minimax-m4" in problem for problem in problems)


def test_the_checker_notices_an_endpoint_it_does_not_know():
    """R9 — a fourth dialect must force a decision, not inherit the default route."""
    snapshot = load_snapshot()
    snapshot["models"]["some-future-model"] = "/v2/converse"

    problems = check_routing(snapshot, _MESSAGES_MODELS, _RESPONSES_MODELS)

    assert any("/v2/converse" in problem for problem in problems)


def test_the_validation_model_is_in_the_published_catalogue():
    """R7 — the field that rotted last time, pinned against the oracle.

    ``glm-5`` was chosen because it was "always available" and then left the
    catalogue, which is how ``kitty auth`` came to risk reporting a bogus auth
    failure.  Membership in the snapshot is the check that would have caught it.
    """
    adapter = OpenCodeGoAdapter()

    assert adapter.normalize_model_name(adapter.validation_model) in load_snapshot()["models"]


def test_every_unserved_model_is_named_in_the_readme():
    """R17 — a user must be able to find out why four models do not work.

    Guarded rather than trusted because the prose that just rotted — this
    adapter's module docstring — rotted exactly this way, and the README cell
    names four more perishable model ids.
    """
    # Without this the check passes by having nothing to look for — which it did
    # on first run, while `_RESPONSES_MODELS` was still empty. The exact no-op
    # this module's docstring promises none of its checks can become.
    assert _RESPONSES_MODELS, "no unserved models to look for — this check has nothing to guard"

    readme = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    row = next((line for line in readme.splitlines() if "`opencode_go`" in line), None)

    assert row is not None, "no OpenCode Go row found in the README provider table"
    missing = sorted(model for model in _RESPONSES_MODELS if model not in row)
    assert not missing, f"README's OpenCode Go row does not name the unserved models: {missing}"

    # The README is the package's PyPI long description. An internal tracker id
    # means nothing to the person reading it there, and the first one admitted
    # is what makes the second one look normal. Asserted rather than trusted,
    # because a criterion with no guard is what this whole fix is about.
    assert "KBR-" not in readme, "the README is the PyPI long description — no internal tracker ids"
