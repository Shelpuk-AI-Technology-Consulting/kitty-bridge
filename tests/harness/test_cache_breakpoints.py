"""L1 tests for the cache-breakpoint request builder and detector.

`.system_design/TEST_SUITE.md` §3.4 · **KBR-198** (CB-1, epic KBR-197).

:mod:`harness.cache_breakpoints` is test data other tests trust. The translator's
characterisation tests assert that a breakpoint is *absent* from kitty's output,
and an absence proves nothing unless the input really carried the breakpoint, at
the site the test names. So the builder is tested here against an independently
written table of paths, never against its own.

**Every checker ships with the defect it must detect.** Plan §1.4's harness rule:
:func:`~harness.cache_breakpoints.find_breakpoints` is falsified by its positive
cases (an always-empty detector fails every one of them), and
:func:`~harness.cache_breakpoints.request_problems` by one deliberately broken
body per rule, in :class:`TestRequestProblemsFalsification`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from harness import cache_breakpoints as cb
from harness.test_contract import _KITTY_IMPORT

# Written independently of the module, so a wrong path table in the builder
# cannot agree with itself.
EXPECTED_SITE_PATHS: dict[str, tuple[str | int, ...]] = {
    "tool": ("tools", 0),
    "system": ("system", 0),
    "document": ("messages", 0, "content", 0),
    "image": ("messages", 0, "content", 1),
    "user_text": ("messages", 0, "content", 2),
    "assistant_text": ("messages", 1, "content", 0),
    "tool_use": ("messages", 1, "content", 1),
    "tool_result": ("messages", 2, "content", 0),
    "tool_result_nested": ("messages", 2, "content", 0, "content", 0),
    "top_level": (),
}


def _walk(body: dict[str, Any], path: tuple[str | int, ...]) -> Any:
    """Follow a path of keys and indexes into a request body.

    Args:
        body: The request body to descend into.
        path: Dict keys and list indexes, outermost first.

    Returns:
        The node the path ends at.
    """
    node: Any = body
    for step in path:
        node = node[step]
    return node


# --------------------------------------------------------------------------
# find_breakpoints
# --------------------------------------------------------------------------


class TestFindBreakpoints:
    """The detector every absence assertion rests on."""

    def test_finds_a_top_level_key(self) -> None:
        """A top-level ``cache_control`` is automatic caching, and is found."""
        assert cb.find_breakpoints({"model": "m", "cache_control": cb.BREAKPOINT}) == [cb.BREAKPOINT]

    def test_finds_a_key_nested_in_a_list_inside_a_dict(self) -> None:
        """A block-level breakpoint deep in ``messages`` is found."""
        body = {"messages": [{"content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]}]}

        assert cb.find_breakpoints(body) == [{"type": "ephemeral"}]

    def test_finds_a_private_prefixed_key(self) -> None:
        """An unregistered private key containing ``cache_control`` is still found.

        KBR-296 registered ``_cache_control`` / ``_tool_cache_controls`` /
        ``_tool_call_cache_controls`` as kitty carriage keys (the mirror in
        ``_KITTY_CARRIAGE_KEYS``), so a literal ``_cache_control`` is now
        skipped — see the mirror-falsification tests below. The detector's
        ``"cache_control" in key`` substring match still catches a different
        unregistered private key, which is what this test now covers: a
        carry-through on an unknown private key would turn the absence
        tests red.
        """
        assert cb.find_breakpoints({"_my_cache_control": {"type": "ephemeral"}}) == [
            {"type": "ephemeral"}
        ]

    def test_skips_top_level_registered_carriage_keys(self) -> None:
        """KBR-296: request-level carriage keys are cargo, not findings.

        The mirror in ``_KITTY_CARRIAGE_KEYS`` keeps the KBR-198/KBR-199
        absence characterisations honest: a carried value on a registered
        carriage must not surface as a phantom breakpoint, the way an
        unregistered private key still would.
        """
        assert cb.find_breakpoints(
            {
                "_cache_control": cb.BREAKPOINT,
                "_tool_cache_controls": {"read_file": cb.BREAKPOINT},
            }
        ) == []

    def test_skips_message_level_registered_carriage_keys(self) -> None:
        """KBR-296: message-level carriage keys are cargo, not findings.

        The detector walks message dicts by recursion, so a message-level
        key containing ``cache_control`` matches the substring rule — the
        mirror covers the registered names so their carried values ride as
        cargo on the intermediate.
        """
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "hi",
                    "_cache_control": cb.BREAKPOINT,
                    "_tool_call_cache_controls": {0: cb.BREAKPOINT},
                },
                {
                    "role": "tool",
                    "tool_call_id": "x",
                    "content": "ok",
                    "_cache_control": cb.BREAKPOINT,
                },
            ]
        }

        assert cb.find_breakpoints(body) == []

    def test_finds_the_breakpoint_value_under_an_unrelated_key(self) -> None:
        """The exact breakpoint value is found under a key that does not say ``cache_control``."""
        assert cb.find_breakpoints({"meta": [{"_prefix_marker": cb.BREAKPOINT}]}) == [cb.BREAKPOINT]

    def test_finds_the_breakpoint_value_as_a_bare_list_item(self) -> None:
        """The exact breakpoint value is found when it sits directly in a list, such as ``_breakpoints: [...]``."""
        assert cb.find_breakpoints({"meta": [cb.BREAKPOINT]}) == [cb.BREAKPOINT]

    def test_returns_nothing_for_a_body_without_breakpoints(self) -> None:
        """A body with no breakpoint yields an empty list, not an error."""
        body = {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}

        assert cb.find_breakpoints(body) == []

    def test_does_not_descend_into_a_matched_value(self) -> None:
        """A breakpoint is counted once, however its value is shaped inside."""
        assert cb.find_breakpoints({"cache_control": {"cache_control": "inner"}}) == [{"cache_control": "inner"}]


# --------------------------------------------------------------------------
# build_request
# --------------------------------------------------------------------------


class TestBuildRequest:
    """The builder puts exactly one breakpoint exactly where the site says."""

    def test_sites_are_exactly_the_ten_the_requirements_name(self) -> None:
        """No site is missing from the builder, and it offers none the tests do not cover."""
        assert set(cb.SITES) == set(EXPECTED_SITE_PATHS)

    @pytest.mark.parametrize("site", sorted(EXPECTED_SITE_PATHS))
    def test_carries_exactly_one_breakpoint(self, site: str) -> None:
        """The built body carries the one-hour breakpoint once and nowhere else.

        Args:
            site: The breakpoint site under test.
        """
        assert cb.find_breakpoints(cb.build_request(site)) == [cb.BREAKPOINT]

    @pytest.mark.parametrize(("site", "path"), sorted(EXPECTED_SITE_PATHS.items()))
    def test_puts_the_breakpoint_at_the_named_site(self, site: str, path: tuple[str | int, ...]) -> None:
        """The breakpoint sits on the block the site names, not merely somewhere.

        Args:
            site: The breakpoint site under test.
            path: The independently written path to that site's carrier.
        """
        assert _walk(cb.build_request(site), path)["cache_control"] == cb.BREAKPOINT

    def test_an_unknown_site_is_rejected(self) -> None:
        """A misspelt site raises rather than building a body with no breakpoint."""
        with pytest.raises(ValueError, match="nope"):
            cb.build_request("nope")

    def test_mutating_a_built_body_leaves_the_shared_breakpoint_alone(self) -> None:
        """A downstream test that edits its body cannot corrupt :data:`BREAKPOINT` for the next test."""
        body = cb.build_request("system")

        body["system"][0]["cache_control"]["ttl"] = "5m"

        assert cb.BREAKPOINT == {"type": "ephemeral", "ttl": "1h"}

    def test_every_body_survives_a_json_round_trip(self) -> None:
        """The bodies are plain JSON, so a wire-level test can send them unchanged."""
        bodies = [cb.build_request(site) for site in cb.SITES]

        assert json.loads(json.dumps(bodies)) == bodies

    @pytest.mark.parametrize("site", sorted(EXPECTED_SITE_PATHS))
    def test_is_well_formed(self, site: str) -> None:
        """The built body breaks none of the rules :func:`request_problems` checks.

        Args:
            site: The breakpoint site under test.
        """
        assert cb.request_problems(cb.build_request(site)) == []

    def test_every_prefix_clears_twice_the_largest_cacheable_minimum(self) -> None:
        """Anthropic silently skips caching below 4,096 tokens on some models, so no body may fall short.

        The floor is written here rather than read from :data:`MIN_PREFIX_WORDS`,
        which the module also sizes its padding by, so shrinking both together
        still fails.
        """
        largest_minimum_tokens = 4096  # Haiku 4.5, Opus 4.5/4.6 — prompt-caching docs, retrieved 2026-09-13

        short = {
            site: words
            for site in cb.SITES
            if (words := len(cb.build_request(site)["tools"][0]["description"].split())) < 2 * largest_minimum_tokens
        }

        assert short == {}

    def test_the_module_imports_nothing_from_kitty(self) -> None:
        """The builder cannot share the translator's assumptions (§3.3.1's independence rule)."""
        source = Path(cb.__file__).read_text(encoding="utf-8")
        # A guard that passes on an empty file is indistinguishable from one that cannot fail.
        assert len(source) > 1000, "read no meaningful source; the guard would pass vacuously"

        offending = [line.strip() for line in source.splitlines() if _KITTY_IMPORT.search(line)]

        assert offending == []


# --------------------------------------------------------------------------
# request_problems — one deliberate defect per rule
# --------------------------------------------------------------------------


def _drop_max_tokens(body: dict[str, Any]) -> None:
    """Remove a field Anthropic requires.

    Args:
        body: The request body to break in place.
    """
    del body["max_tokens"]


def _unanswer_the_tool_use(body: dict[str, Any]) -> None:
    """Replace the tool result with plain text, leaving the ``tool_use`` unanswered.

    Args:
        body: The request body to break in place.
    """
    body["messages"][2]["content"] = [{"type": "text", "text": "never mind"}]


def _call_an_undeclared_tool(body: dict[str, Any]) -> None:
    """Rename the called tool to one the request does not declare.

    Args:
        body: The request body to break in place.
    """
    body["messages"][1]["content"][1]["name"] = "delete_everything"


def _truncate_the_png(body: dict[str, Any]) -> None:
    """Cut the image data short, keeping the PNG signature intact.

    Args:
        body: The request body to break in place.
    """
    source = body["messages"][0]["content"][1]["source"]
    source["data"] = source["data"][:24]


def _add_four_more_breakpoints(body: dict[str, Any]) -> None:
    """Mark four more blocks, exceeding Anthropic's limit of four breakpoints.

    Args:
        body: The request body to break in place.
    """
    for block in (*body["messages"][0]["content"], body["system"][0]):
        block["cache_control"] = {"type": "ephemeral"}


def _empty_a_text_block(body: dict[str, Any]) -> None:
    """Blank the user's text block; Anthropic cannot cache an empty text block.

    Args:
        body: The request body to break in place.
    """
    body["messages"][0]["content"][2]["text"] = ""


def _add_a_second_user_turn(body: dict[str, Any]) -> None:
    """Append a user turn directly after the last user turn.

    Args:
        body: The request body to break in place.
    """
    body["messages"].append({"role": "user", "content": [{"type": "text", "text": "and another thing"}]})


def _shrink_the_padding(body: dict[str, Any]) -> None:
    """Shorten the tool description below the cacheable-prefix word floor.

    Args:
        body: The request body to break in place.
    """
    body["tools"][0]["description"] = "Reads a file."


DEFECTS: list[tuple[Callable[[dict[str, Any]], None], str]] = [
    (_drop_max_tokens, "max_tokens"),
    (_unanswer_the_tool_use, "unanswered"),
    (_call_an_undeclared_tool, "undeclared"),
    (_truncate_the_png, "PNG"),
    (_add_four_more_breakpoints, "breakpoints"),
    (_empty_a_text_block, "empty"),
    (_add_a_second_user_turn, "consecutive"),
    (_shrink_the_padding, "words"),
]


class TestRequestProblemsFalsification:
    """Each rule reports the defect it exists to catch (plan §1.4)."""

    @pytest.mark.parametrize(("defect", "named"), DEFECTS, ids=[d.__name__.lstrip("_") for d, _ in DEFECTS])
    def test_reports_the_defect(self, defect: Callable[[dict[str, Any]], None], named: str) -> None:
        """A body broken one way yields a problem that names the break.

        Args:
            defect: Breaks a well-formed body in place, in exactly one way.
            named: A word the resulting problem message must contain.
        """
        body = cb.build_request("tool")
        defect(body)

        problems = cb.request_problems(body)

        assert any(named in problem for problem in problems), problems
