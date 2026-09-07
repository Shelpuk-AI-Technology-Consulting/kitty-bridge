"""Structural guard: the bridge must not name itself where a provider can see it.

Kitty Bridge exists to be invisible to the upstream provider. KBR-5 was a
literal breach — a ``[Kitty Bridge: ...]`` message written into the request body
— and a behavioural test only covers the path someone remembered to test. This
file enumerates every vendor-named string literal in the modules that can build
an upstream request, and fails when a new one appears that has not been
reviewed.

If this test fails, you added a string naming the product. Either prove it
cannot reach an upstream request body or header and add it to the allowlist
below with that reason, or do not add it.

**This is a defect-scoped stand-in, not the general guard.**
``.system_design/TEST_SUITE.md`` §6.2.3 requires the real one to be scoped by a
projection diff, because a flat scan of a *serialized body* would fail on a user
who legitimately writes "kitty" — and "fixing" that by stripping their text
would breach message fidelity in the act of defending indistinguishability. This
file avoids that trap by scanning **source literals**, never traffic: agent
content is never inspected here. The general guard is T-G5 (KBR-81) and needs
the oracle; this stands in until then, and hands T-G5 its positive control.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

BRIDGE = Path(__file__).resolve().parent.parent.parent / "src" / "kitty" / "bridge"

#: Modules that can put bytes into an upstream request, and are therefore scanned.
_IN_SCOPE = [
    "server.py",
    "tool_audit.py",
    "messages/translator.py",
    "messages/events.py",
    "responses/translator.py",
    "responses/events.py",
    "gemini/translator.py",
    "gemini/events.py",
]

#: Modules deliberately out of scope, and why none can reach a request.
#:
#: Recorded rather than silently skipped: if one of these ever gains a code path
#: that builds request content, this list is the thing that is now wrong.
_OUT_OF_SCOPE: dict[str, str] = {
    "config.py": "config file/dir paths only; never request content",
    "manage.py": "start/stop/status of a local process; builds no upstream request",
    "service.py": "generates systemd/launchd/NSSM service files for the OS",
    "engine.py": "no vendor literals; kept listed so the split is explicit",
    "keys.py": "bridge API key handling; downstream only",
    "state.py": "on-disk bridge state",
}

#: Every vendor-named literal in the in-scope modules, and why it is safe.
#:
#: Keyed by the literal itself rather than by line number, so moving code does
#: not churn this list while *changing* a message forces re-classification —
#: which is exactly the review a vendor-named string deserves.
_ALLOWLIST: dict[str, str] = {
    # Logging — never serialized into a request.
    "kitty.bridge": "logger name",
    "_kitty_bridge_log": "attribute marking the bridge's own log handler",
    # Filesystem and environment — local only.
    "kitty": "path component under ~/.config or ~/.cache",
    "KITTY_SESSION_SUMMARY": "environment variable read at startup",
    "KITTY_BRIDGE_CONN_LIMIT": "environment variable read at startup",
    # Downstream response headers. _attribution_headers() is applied only to
    # responses; no build_upstream_headers() can reach these names.
    "X-Kitty-Backend": "downstream response header (attribution)",
    "X-Kitty-Tier": "downstream response header (attribution)",
    "X-Kitty-Model": "downstream response header (attribution)",
    # Downstream response bodies — what the agent reads, not what the provider sees.
    "kitty-bridge": "owned_by field in the /v1/models downstream response",
    "An internal error occurred. Check kitty logs for details.": "downstream 500 body",
    "kitty auth openai": "downstream error body: how the user re-authenticates",
    "): API key is invalid, expired, or lacks permission. Update your API key with 'kitty setup'.": (
        "downstream error body: how the user fixes their key"
    ),
    # KBR-5's replacement. Downstream only, by construction: it is returned by
    # _compaction_failed_response(), and the compaction path now RAISES rather
    # than writing anything into cc_request["messages"].
    (
        "Kitty Bridge could not reduce this conversation to something the model can accept: "
        "after compaction no messages were left to send. This usually means the conversation "
        "contains a tool result whose matching tool call was lost. Start a new conversation "
        "(/clear), or switch to a model with a larger context window."
    ): "downstream 400 body for an irreducible conversation (KBR-5)",
}

#: The string this ticket deleted. The scanner must still be able to see it.
_HISTORICAL_M13_STRING = (
    "[Kitty Bridge: Unable to compact conversation — the system prompt "
    "is too large relative to the model's context window. "
    "Use /clear to reset the conversation.]"
)


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Return the ids of every docstring node in ``tree``.

    Docstrings are excluded from the scan: they are documentation, cannot be
    serialized into a request, and including them would bury the handful of
    literals that matter under every mention of the product in prose.

    Args:
        tree: A parsed module.

    Returns:
        Ids of the ``ast.Constant`` nodes that are docstrings.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def _scan_source(source: str) -> list[str]:
    """Return every non-docstring string literal in ``source`` naming the product.

    Args:
        source: Python source text.

    Returns:
        The matching literal values, in traversal order, with duplicates kept so
        a caller can count occurrences.
    """
    tree = ast.parse(source)
    docs = _docstring_ids(tree)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docs
        and "kitty" in node.value.lower()
    ]


def _scan_in_scope() -> dict[str, list[str]]:
    """Scan every in-scope module.

    Returns:
        Mapping of relative module path to the vendor literals it contains.
    """
    found: dict[str, list[str]] = {}
    for rel in _IN_SCOPE:
        path = BRIDGE / rel
        if not path.exists():  # A module was renamed; the list below is now stale.
            continue
        hits = _scan_source(path.read_text(encoding="utf-8"))
        if hits:
            found[rel] = hits
    return found


class TestNoUnreviewedVendorToken:
    """Every vendor-named literal that could reach a request is accounted for."""

    def test_every_vendor_named_literal_is_allowlisted(self):
        """A new string naming the product fails until someone classifies it."""
        unexpected: list[tuple[str, str]] = []
        for rel, hits in _scan_in_scope().items():
            unexpected.extend((rel, hit) for hit in hits if hit not in _ALLOWLIST)
        assert not unexpected, (
            "New vendor-named literal(s) in a module that can build an upstream request. "
            "Prove each cannot reach a request body or header and add it to _ALLOWLIST "
            f"with that reason: {unexpected}"
        )

    def test_allowlist_has_no_stale_entries(self):
        """A removed literal must be removed from the allowlist too.

        Without this the list silently rots into a permission slip for strings
        nobody has looked at in a year.
        """
        live = {hit for hits in _scan_in_scope().values() for hit in hits}
        stale = sorted(set(_ALLOWLIST) - live)
        assert not stale, f"_ALLOWLIST entries no longer present in the source: {stale}"

    def test_every_in_scope_module_exists(self):
        """A rename must not silently shrink the scan's coverage."""
        missing = [rel for rel in _IN_SCOPE if not (BRIDGE / rel).exists()]
        assert not missing, f"_IN_SCOPE names modules that no longer exist: {missing}"

    def test_out_of_scope_modules_exist_and_are_deliberate(self):
        """The exclusions are a recorded decision, not an oversight."""
        missing = [rel for rel in _OUT_OF_SCOPE if not (BRIDGE / rel).exists()]
        assert not missing, f"_OUT_OF_SCOPE names modules that no longer exist: {missing}"


class TestTheScanActuallyFindsSomething:
    """A guard that cannot fail is worse than no guard.

    These are the positive controls. The first is load-bearing beyond this
    ticket: once M13 is deleted there is no live bridge-introduced vendor string
    left, so T-G5 (KBR-81) inherits this synthetic fixture as its positive.
    """

    def test_the_scan_finds_the_historical_m13_string(self):
        """The deleted KBR-5 message, reintroduced, is caught."""
        source = f'def f():\n    return [{{"role": "user", "content": {_HISTORICAL_M13_STRING!r}}}]\n'
        assert _scan_source(source) == [_HISTORICAL_M13_STRING]
        assert _HISTORICAL_M13_STRING not in _ALLOWLIST, "the deleted message must never be allowlisted"

    def test_the_scan_is_case_insensitive(self):
        """``KITTY``/``Kitty``/``kitty`` are the same fingerprint to a provider."""
        assert _scan_source('x = "KITTY BRIDGE"') == ["KITTY BRIDGE"]

    def test_the_scan_ignores_docstrings(self):
        """Prose about the product is not a wire risk, and must not drown the signal."""
        assert _scan_source('"""About kitty bridge."""\nx = 1\n') == []

    def test_the_scan_finds_something_in_the_real_source(self):
        """If this returns nothing the scan has silently stopped working."""
        assert _scan_in_scope(), "the scan found no vendor literals at all — it is broken"

    @pytest.mark.parametrize("rel", ["server.py"])
    def test_the_deleted_message_is_gone_from_the_source(self, rel):
        """KBR-5's regression assertion, stated directly."""
        source = (BRIDGE / rel).read_text(encoding="utf-8")
        assert "Unable to compact conversation" not in source
        assert "[Kitty Bridge:" not in source
