"""Structural guard: every DEBUG log site that could carry a credential routes through the redaction helpers.

`.system_design/SYSTEM_DESIGN.md` §9 — DEBUG-log redaction policy.  The
behavioural tests in ``test_egress_properties.py`` prove the helpers
redact; this file proves **every** offending call site uses them, so a
future handler that adds a raw ``logger.debug("Upstream POST → %s", url)``
or ``logger.debug("... headers: %s", dict(h))`` line cannot ship.

If this test fails, you added a DEBUG log call that could carry a
credential. Either route it through ``BridgeServer._debug_url`` /
``BridgeServer._debug_headers`` and the assertion below updates itself
from the file's known-positives table, or explain in the table why the
new call must not be redacted.

**Why the L2 marker is explicit, not path-default.**  ``tests/layers.py``
assigns ``tests/test_*.py`` the L1 default, so without the marker this
file's structural guard would land in the fast gate's L1 selection and
be invisible to ``pytest -m l2 -q``.  The pattern is the one
``test_egress_coverage.py`` and ``test_ipaddress_contract.py`` use.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# L2: this file asserts what an artifact outside Python code — the bridge's
# DEBUG-log call sites — must do, and gates pull requests independently of
# the L1 selection.
pytestmark = pytest.mark.l2

SRC = Path(__file__).resolve().parent.parent / "src" / "kitty" / "bridge" / "server.py"

#: Patterns that match a DEBUG log call whose argument could carry a
#: credential.  The first is the URL-dump form; the second is the
#: header-dump form.  Both regexes are anchored on ``logger.debug(`` and
#: the format string's literal prefix so the scan does not match a
#: coincidentally similar phrase in a comment or docstring.
_REDACT_SITE_PATTERNS: dict[str, re.Pattern[str]] = {
    "url-dump": re.compile(r"""logger\.debug\(\s*["']Upstream POST → %s["']"""),
    "header-dump": re.compile(r"""logger\.debug\(\s*["'][^"']*headers: %s["']"""),
}

#: The number of call sites per pattern kind that must exist today.
#: Updated when the codebase gains or loses a DEBUG-log site; a regression
#: here is a regression in the bridge, not in the test.
_EXPECTED_COUNTS: dict[str, int] = {
    "url-dump": 4,
    "header-dump": 2,
}

#: Per-kind witness: the exact line text the scan expects to find at
#: each known site, so a silent rewrite fails this test before the
#: broader regex tests can pass with a different line.  ``header-dump``
#: has one entry per call site; ``url-dump`` has one because all four
#: sites share the same form — the count guard above catches a change
#: in the *number* of occurrences, which is what would matter.
_KNOWN_SITE_TEXTS: dict[str, tuple[str, ...]] = {
    "url-dump": (
        'logger.debug("Upstream POST → %s", BridgeServer._debug_url(url))',
    ),
    "header-dump": (
        'logger.debug("Request headers: %s", BridgeServer._debug_headers(request.headers))',
        'logger.debug("Upstream response headers: %s", BridgeServer._debug_headers(upstream.headers))',
    ),
}


def _iter_redact_sites() -> list[tuple[str, str, int]]:
    """Find every DEBUG log call whose argument could carry a credential.

    Returns:
        ``(kind, line_text, line_number)`` triples, one per offending
        call site in ``bridge/server.py``.  ``kind`` is the pattern key.
    """
    found: list[tuple[str, str, int]] = []
    text = SRC.read_text(encoding="utf-8").splitlines()
    for kind, pattern in _REDACT_SITE_PATTERNS.items():
        for lineno, line in enumerate(text, start=1):
            if pattern.search(line):
                found.append((kind, line.strip(), lineno))
    return found


class TestDebugRedactionSitesAreCovered:
    """R8: every credential-bearing DEBUG log call routes through the helpers."""

    def test_no_offending_site_logs_a_url_or_headers_directly(self) -> None:
        """Every offending DEBUG site uses the helper, not a raw ``url`` or ``dict(h)``.

        A bare ``url`` or ``dict(headers)`` argument on a matching format
        string is the failure mode this guard exists to catch.
        """
        offenders: list[str] = []
        for kind, line, lineno in _iter_redact_sites():
            expected_substrings = {
                "url-dump": "BridgeServer._debug_url(",
                "header-dump": "BridgeServer._debug_headers(",
            }[kind]
            if expected_substrings not in line:
                offenders.append(f"{SRC.name}:{lineno} ({kind}) -> {line}")

        assert not offenders, (
            "These DEBUG log calls could carry a credential and do not "
            "route through the redaction helper:\n  "
            + "\n  ".join(offenders)
            + "\n\nRoute them through BridgeServer._debug_url / "
            "BridgeServer._debug_headers."
        )

    def test_every_known_site_kind_is_still_present(self) -> None:
        """Every pattern kind still matches at least one site in the source.

        Complements the previous test: the negative form asserts every
        match uses the helper; this one asserts no *kind* has vanished —
        a kind gone from the source means either the sites were removed
        (the count guard catches that) or the regex stopped matching
        (this test names the regex as the thing to fix).
        """
        actual_kinds = {kind for kind, _line, _lineno in _iter_redact_sites()}
        expected_kinds = set(_KNOWN_SITE_TEXTS)
        missing = expected_kinds - actual_kinds

        assert not missing, (
            f"A DEBUG-log redaction pattern is no longer matched in {SRC.name}: "
            f"{sorted(missing)}.  The corresponding kind is either gone "
            "(update _KNOWN_SITE_TEXTS) or no longer matches the regex "
            "(update _REDACT_SITE_PATTERNS)."
        )

    def test_the_scan_actually_finds_something(self) -> None:
        """Guard against a broken regex quietly passing the suite."""
        actual_counts = _iter_redact_sites()
        assert len(actual_counts) >= sum(_EXPECTED_COUNTS.values()), (
            f"Scan found {len(actual_counts)} sites; expected at least "
            f"{sum(_EXPECTED_COUNTS.values())} across "
            f"{sorted(_KNOWN_SITE_TEXTS)}."
        )

    def test_no_new_site_hides_inside_an_already_listed_form(self) -> None:
        """A new call to ``logger.debug("Upstream POST → %s", ...)`` cannot sneak in.

        Mirrors ``test_egress_coverage.py::test_no_new_client_hides_inside_an_already_listed_file``:
        a per-kind count table makes a regression in the number of call
        sites — *added* or *removed* — visible.
        """
        actual: dict[str, int] = {}
        for kind, _line, _lineno in _iter_redact_sites():
            actual[kind] = actual.get(kind, 0) + 1

        added = {k: v for k, v in actual.items() if v != _EXPECTED_COUNTS.get(k, 0)}
        removed = {k: _EXPECTED_COUNTS[k] for k in _EXPECTED_COUNTS if actual.get(k, 0) == 0}

        detail: list[str] = []
        for kind, count in sorted(added.items()):
            detail.append(f"{kind}: expected {_EXPECTED_COUNTS.get(kind, 0)}, found {count}")
        for kind, count in sorted(removed.items()):
            detail.append(f"{kind}: expected {count}, found none")

        assert not detail, (
            "DEBUG-log redaction site counts changed — review each, then update "
            "_KNOWN_SITE_TEXTS in tests/test_egress_log_redaction.py: "
            + "; ".join(detail)
        )

    def test_every_known_site_text_appears(self) -> None:
        """The known-positives table is a witness to today's exact line text.

        A site removed or rewritten without updating the table fails
        here, even if the regex still matches a different line — the
        *exact text* is the contract.
        """
        text = SRC.read_text(encoding="utf-8")
        missing = [
            f"{kind}: {line!r}"
            for kind, lines in _KNOWN_SITE_TEXTS.items()
            for line in lines
            if line not in text
        ]

        assert not missing, (
            "These known-positive DEBUG-log sites no longer appear verbatim in "
            f"{SRC.name}:\n  " + "\n  ".join(missing) + "\n\nEither the site "
            "was edited (update _KNOWN_SITE_TEXTS) or removed (file a ticket)."
        )
