"""L1 — native-capable adapters leave prompt-cache breakpoints untouched.

Jira: `KBR-200 <https://shelpuk.atlassian.net/browse/KBR-200>`_ (CB-3, epic
`KBR-197 <https://shelpuk.atlassian.net/browse/KBR-197>`_). Design: row M16 of
``.system_design/TEST_SUITE.md`` §3.2.1.

Native passthrough (``use_native_messages == True``) is the only route on
which a Claude Code user's prompt cache survives today. Three adapters qualify:
``zai_anthropic`` and ``custom_anthropic`` (always native) and
``minimax_token`` (opt-in). This module guards the two adapter hooks a request
passes through on that route — :meth:`ProviderAdapter.normalize_request` and
:meth:`ProviderAdapter.translate_to_upstream` — against a future change that
silently drops or downgrades the agent's breakpoints.

**This proves the field survives, not that the cache hits.** Anthropic
invalidates the cache when the thinking configuration or
``output_config.effort`` changes; that defect is `CB-6 (KBR-203)`_'s scope,
not this one's. A reader of green tests here must not conclude the native
route's caching is fully proven.

**One breakpoint per body, full value asserted.** The fixture
(:mod:`harness.cache_breakpoints`) places exactly one ``{"type": "ephemeral",
"ttl": "1h"}`` at a chosen site, because Anthropic returns 400 above four.
``find_breakpoints`` matches by value as well as by key name, so a guard
"preserving the key" while re-encoding the value would still go red.

**Tool order and tool content are pinned.** Anthropic's cache hierarchy is
``tools → system → messages``: any reorder or edit to the tool array
invalidates that level **and every level after it**. ``normalize_request``
is per-adapter and free to reshape tools, so a stable breakpoint on a
reordered array is a guaranteed miss — exactly the change a well-meaning
normaliser makes.

**Two falsification cases per the harness rule** (``TEST_SUITE_IMPLEMENTATION_PLAN.md``
§1.4). The guard must be shown to fail when broken: a stand-in
adapter whose ``normalize_request`` deletes every ``cache_control`` anywhere
in the body, and a stand-in whose ``_INTERNAL_KEYS`` claims ``cache_control``.
Their sibling tests assert the deletion fires — proving the positive claims
in this file are not vacuous.

Why L1: every claim here is a property of an adapter or a constant, with no
server, no transport, no agent involved. The companion server-level proof
(``tests/bridge/test_native_passthrough_cache_breaks.py``, L2) drives the
end-to-end branch + adapter seam.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
from harness import cache_breakpoints as cb

from kitty.providers.base import ProviderAdapter
from kitty.providers.custom_anthropic import CustomAnthropicAdapter
from kitty.providers.minimax_token import MiniMaxTokenAnthropicAdapter
from kitty.providers.zai_anthropic import ZaiAnthropicAdapter

#: The three native-capable adapters. ``minimax_token`` is opt-in: its
#: ``use_native_messages`` defaults to ``False`` and is set via the
#: ``native_messages`` kwarg (or ``provider_config["native_messages"]``).
_NATIVE_ADAPTER_FACTORIES: dict[str, Callable[[], ProviderAdapter]] = {
    "zai_anthropic": ZaiAnthropicAdapter,
    "custom_anthropic": CustomAnthropicAdapter,
    "minimax_token": lambda: MiniMaxTokenAnthropicAdapter(native_messages=True),
}


@pytest.fixture(params=_NATIVE_ADAPTER_FACTORIES.keys(), ids=_NATIVE_ADAPTER_FACTORIES.keys())
def native_adapter(request: pytest.FixtureRequest) -> ProviderAdapter:
    """A native-capable adapter for each entry in ``_NATIVE_ADAPTER_FACTORIES``."""
    return _NATIVE_ADAPTER_FACTORIES[request.param]()


# ── normalize_request is a no-op for cache_control ────────────────────────


@pytest.mark.parametrize("site", cb.SITES)
def test_normalize_request_preserves_breakpoint_full_value(native_adapter: ProviderAdapter, site: str) -> None:
    """``normalize_request`` leaves the breakpoint at *site* at its full value.

    The fixture places one breakpoint at the named site; after the call
    ``find_breakpoints`` must return the same dict — equal by value, not only
    by key name. A future normaliser that strips ``ttl`` and leaves a
    five-minute-only breakpoint would still go red here.

    **Tripwire framing.** This test passes trivially today because every
    native-capable adapter inherits the base no-op (none overrides
    ``normalize_request``). The assertion's job is to go red the day one of
    them does and the override touches a breakpoint.
    """
    body = cb.build_request(site)
    before = cb.find_breakpoints(body)
    assert len(before) == 1, f"fixture should place one breakpoint at {site!r}"

    native_adapter.normalize_request(body)

    assert cb.find_breakpoints(body) == before


def test_normalize_request_does_not_reorder_or_edit_tools(native_adapter: ProviderAdapter) -> None:
    """``normalize_request`` does not reorder or edit the tool declarations.

    Anthropic's cache hierarchy makes tool edits the most destructive change a
    normaliser could make — it invalidates every level after ``tools``.
    Asserted against the fixture's tool array; the breakpoint site is
    irrelevant (the tools array is the same for every site).

    **Tripwire framing.** Same as
    :func:`test_normalize_request_preserves_breakpoint_full_value`: passes
    trivially today because no native adapter overrides
    ``normalize_request``. The assertion's job is to go red the day one
    does and the override touches the tools array.
    """
    body = cb.build_request("tool")
    tools_before = copy.deepcopy(body["tools"])

    native_adapter.normalize_request(body)

    assert body["tools"] == tools_before


# ── translate_to_upstream keeps the breakpoint on the native flag ─────────


def test_translate_to_upstream_keeps_top_level_breakpoint_on_native_flag(native_adapter: ProviderAdapter) -> None:
    """The native passthrough of ``translate_to_upstream`` keeps a top-level breakpoint.

    The bridge sets ``_native_messages_request=True`` before the adapter sees
    the body; the adapter's override returns the body as-is minus
    ``_INTERNAL_KEYS``. A top-level ``cache_control`` is the agent's, not
    kitty's, so it must survive the strip.
    """
    body = cb.build_request("top_level")
    body["_native_messages_request"] = True

    sent = native_adapter.translate_to_upstream(body)

    assert sent["cache_control"] == dict(cb.BREAKPOINT)
    # The native flag is internal; it must not reach the wire.
    assert "_native_messages_request" not in sent


def test_cache_control_is_not_a_member_of_internal_keys() -> None:
    """Structural pair-tests for the strip test above.

    The behavioural strip test would still go red if a future change
    inlined ``cache_control`` handling inside ``translate_to_upstream``; this
    one fails the day someone adds it to ``_INTERNAL_KEYS`` in the most
    likely (and only) way a future contributor would.
    """
    assert "cache_control" not in ProviderAdapter._INTERNAL_KEYS


# ── Falsification: the guard can fail ─────────────────────────────────────


class _BreakpointDropper(ZaiAnthropicAdapter):
    """Stand-in adapter whose ``normalize_request`` deletes every breakpoint.

    Per ``TEST_SUITE_IMPLEMENTATION_PLAN.md`` §1.4: a guard never shown to
    fail is indistinguishable from one that cannot. This sibling proves the
    positive claims above are not vacuous — driving the same inputs through
    this dropper must remove every breakpoint.
    """

    def normalize_request(self, cc_request: dict) -> None:
        def strip(node: object) -> None:
            if isinstance(node, dict):
                node.pop("cache_control", None)
                for value in node.values():
                    strip(value)
            elif isinstance(node, list):
                for value in node:
                    strip(value)

        strip(cc_request)


@pytest.mark.parametrize("site", cb.SITES)
def test_falsification_normalize_can_drop_a_breakpoint(site: str) -> None:
    """A ``normalize_request`` that drops breakpoints is detectable.

    This is the broken-side characterisation that proves the sibling guard
    tests are not vacuous: driving the fixture through ``_BreakpointDropper``
    removes the breakpoint from each site.
    """
    body = cb.build_request(site)
    _BreakpointDropper().normalize_request(body)
    assert cb.find_breakpoints(body) == []


class _CacheControlInternalKey(ZaiAnthropicAdapter):
    """Stand-in adapter whose ``_INTERNAL_KEYS`` claims ``cache_control``.

    Sibling falsification for the strip test: a future contributor adding
    ``"cache_control"`` to the set would silently break the wire shape. This
    adapter exercises that exact change so the test can prove it is detected.
    """

    _INTERNAL_KEYS = ProviderAdapter._INTERNAL_KEYS | {"cache_control"}


def test_falsification_internal_key_membership_strips_breakpoint() -> None:
    """Adding ``cache_control`` to ``_INTERNAL_KEYS`` removes it from the wire."""
    body = cb.build_request("top_level")
    body["_native_messages_request"] = True

    sent = _CacheControlInternalKey().translate_to_upstream(body)

    assert "cache_control" not in sent
