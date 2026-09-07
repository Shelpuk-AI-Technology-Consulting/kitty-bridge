"""Regression guard: no kitty-internal key reaches the upstream request body.

KBR-6. ``MessagesTranslator.translate_request`` wrote ``_effort`` and
``_thinking_adaptive`` into the Chat Completions request; neither was a member
of ``ProviderAdapter._INTERNAL_KEYS``, so both were forwarded to the provider by
sixteen adapters.  Underscore-prefixed fields appear in no vendor's public API,
which makes them an unambiguous signature of an intermediary — a breach of I1
(message fidelity) and I2 (bridge indistinguishability).

**Where this asserts.** At :meth:`BridgeServer._upstream_body_for`, the bridge's
serialization boundary, not at ``adapter.translate_to_upstream``.  Every handler
re-serializes through it and it applies ``_repair_thinking_roundtrip`` to the
body *after* the internal-key strip, so a test at the adapter hook would be
blind to anything that repair introduces.  See ``TEST_SUITE.md`` §6.2.3, which
requires the serialization boundary.

**What this does not cover.** Three adapters override ``use_custom_transport``
and build their real wire body inside their own transport, downstream of this
boundary: ``bedrock``, ``ollama_cloud`` and ``openai_subscription``.  For those,
this file proves the bridge-side body is clean, not the bytes on the socket.
Their coverage is carried by T-G2 over T-D4–T-D9's captures; recorded as the
residual on gap G15 in ``TEST_SUITE.md`` §9.

**Why the input is derived, not hand-written.** The key set comes from the AST
scan in ``tests/internal_key_scan.py`` — the same scan the completeness guard
uses.  A hand-written list here would drift from the code the day someone mints
a new key, and the drift would be silent.
"""

from __future__ import annotations

import pytest
from internal_key_scan import discovered_keys

from kitty.bridge.messages.translator import MessagesTranslator
from kitty.bridge.server import BridgeServer
from kitty.providers.base import ProviderAdapter
from kitty.providers.registry import _registry, get_provider

#: A model on the Chat Completions route for every adapter.
_CC_MODEL = "claude-sonnet-4-5"

#: ``OpenCodeGoAdapter`` picks its wire shape from the model name, so one model
#: exercises one route.  See ``opencode._MESSAGES_MODELS``.
_EXTRA_MODELS: dict[str, tuple[str, ...]] = {"opencode_go": ("minimax-m2.5",)}

#: Placeholder values by key, so an adapter that reads a key finds the shape it
#: expects rather than raising on a string where it wanted a dict.
_PLACEHOLDERS: dict[str, object] = {
    "_provider_config": {},
    "_original_body": {"model": _CC_MODEL},
    "_resolved_key": "test-key",
    "base_url": "https://upstream.kitty-test.invalid",
}


def _messages_body(model: str) -> dict:
    """Build a Messages API request that triggers every translator branch.

    ``effort`` and ``thinking`` are what make the translator mint ``_effort``,
    ``_thinking_adaptive``, ``_thinking_enabled`` and ``_reasoning_effort`` —
    the four keys at the heart of KBR-6.

    Args:
        model: Model name, which selects the wire route on model-routing adapters.

    Returns:
        A Messages API request body.
    """
    return {
        "model": model,
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hello"}],
        "effort": "high",
        "thinking": {"type": "adaptive"},
    }


def _translator_only_cc(model: str) -> dict:
    """Build a CC request from the real translator, with nothing added.

    R5 exists to reject a "fix" that makes every other test pass by deleting the
    keys from the translator. It can only do that if its input is the
    translator's own output: the topped-up dict from :func:`_cc_request`
    re-supplies ``_effort`` and ``_thinking_adaptive`` from ``_INTERNAL_KEYS``,
    so an adapter reads them and emits the feature whether the translator still
    mints them or not.

    That is not hypothetical. The first version of this file used the topped-up
    input for R5, and ``test_adaptive_thinking_survives_the_strip`` passed
    against the deletion it was written to catch.

    Args:
        model: Model name to translate for.

    Returns:
        The translator's output, unmodified.
    """
    return MessagesTranslator().translate_request(_messages_body(model))


def _cc_request(model: str, *, native: bool) -> dict:
    """Build a CC request carrying every internal key the source can write.

    The translator supplies the keys it mints; the rest — the ones
    ``BridgeServer`` attaches during request handling — are added from the AST
    scan's discovered set, unioned with ``_INTERNAL_KEYS`` so a registered key
    that nothing currently writes is still exercised.

    Args:
        model: Model name to route on.
        native: Value for ``_native_messages_request``. Three adapters branch on
            it, taking the strip path when it is set and the Anthropic rebuild
            when it is not, so both values are exercised.

    Returns:
        A normalized CC request dict.
    """
    cc = MessagesTranslator().translate_request(_messages_body(model))
    for key in discovered_keys() | set(ProviderAdapter._INTERNAL_KEYS):
        cc.setdefault(key, _PLACEHOLDERS.get(key, True))
    cc["_native_messages_request"] = native
    return cc


def _routes() -> list[tuple[str, str, bool]]:
    """Enumerate every (provider, model, native-flag) route to assert on.

    §6.2.3 is explicit that one fixed request is not sufficient and that
    adapters routing by model need one input per route.

    Returns:
        ``(provider_type, model, native)`` triples covering the registry.
    """
    return [
        (provider_type, model, native)
        for provider_type in sorted(_registry)
        for model in (_CC_MODEL, *_EXTRA_MODELS.get(provider_type, ()))
        for native in (False, True)
    ]


def _upstream_body(provider_type: str, cc: dict, *, thinking_repair: bool = False) -> dict:
    """Serialize *cc* through the bridge's own serialization boundary.

    Args:
        provider_type: Registry key of the adapter under test.
        cc: The normalized CC request.
        thinking_repair: Arm ``_thinking_repair_backends`` for the selected
            backend, so ``_repair_thinking_roundtrip`` runs on the body after
            the internal-key strip. Off by default because it is a per-backend
            recovery state, not the ordinary path.

    Returns:
        The body the bridge would send as JSON.
    """
    server = BridgeServer(
        adapter=None,
        provider=get_provider(provider_type, {}),
        resolved_key="test-key",
        model=cc["model"],
    )
    if thinking_repair:
        server._thinking_repair_backends = {server._current_backend_idx}
    return server._upstream_body_for(cc)


def _leaked(body: dict) -> list[str]:
    """Return the internal keys present in an upstream body.

    Both clauses matter. ``_INTERNAL_KEYS`` contains ``base_url``, which is not
    underscore-prefixed, so a ``^_`` check alone cannot see it regress.

    Args:
        body: The upstream request body.

    Returns:
        Sorted offending keys; empty when the body is clean.
    """
    internal = set(ProviderAdapter._INTERNAL_KEYS)
    return sorted(key for key in body if key.startswith("_") or key in internal)


class TestNoInternalKeyReachesUpstream:
    """R2: the strip works on every adapter, on every route it selects."""

    @pytest.mark.parametrize(
        ("provider_type", "model", "native"),
        _routes(),
        ids=lambda value: str(value),
    )
    def test_no_internal_key_reaches_upstream(self, provider_type: str, model: str, native: bool):
        body = _upstream_body(provider_type, _cc_request(model, native=native))

        assert not _leaked(body), (
            f"{provider_type} forwards kitty-internal key(s) {_leaked(body)} to the provider. "
            "Every key kitty adds to the request must be a member of "
            "ProviderAdapter._INTERNAL_KEYS (src/kitty/providers/base.py)."
        )

    @pytest.mark.parametrize("provider_type", ["anthropic", "openai", "zai_regular"])
    def test_no_internal_key_reaches_upstream_under_thinking_repair(self, provider_type: str):
        """The post-strip stage this boundary was chosen for must also be clean.

        `_upstream_body_for` was picked over `translate_to_upstream` precisely
        because `_repair_thinking_roundtrip` runs after the strip. That branch
        is dormant unless the backend is in `_thinking_repair_backends`, so
        without this case the choice of boundary is asserted but never
        exercised — and a repair that started minting a key would slip past the
        test placed there to catch it.

        One native-wire adapter and two CC-wire ones, since the repair writes a
        different carrier for each dialect.
        """
        cc = _cc_request(_CC_MODEL, native=False)
        body = _upstream_body(provider_type, cc, thinking_repair=True)

        assert not _leaked(body)

    def test_the_input_carries_every_discovered_key(self):
        """R6: the two guards share one key list, so neither can drift.

        Without this, a key the AST scan starts reporting could go untested
        here and nobody would notice.
        """
        cc = _cc_request(_CC_MODEL, native=False)
        expected = discovered_keys() | set(ProviderAdapter._INTERNAL_KEYS)

        assert expected <= set(cc), f"input is missing {sorted(expected - set(cc))}"

    def test_the_translator_still_mints_the_defect_keys(self):
        """A translator that stopped minting them would make R2 vacuous.

        Asserted against the translator's own output. Against the topped-up
        input this could only fail if the keys left ``_INTERNAL_KEYS``, which is
        not what it claims to guard.
        """
        cc = _translator_only_cc(_CC_MODEL)

        assert {"_effort", "_thinking_adaptive"} <= set(cc)


class TestTheStripPreservesWhatTheKeysCarry:
    """R5: cleanliness must not be bought by deleting the feature.

    The cheapest way to make every test above pass is to stop writing the keys
    in the translator. That would silently remove adaptive thinking and effort
    control from every Anthropic-compatible provider. These two tests reject
    that fix.

    They take :func:`_translator_only_cc`, never the topped-up input — see that
    function for why the distinction is the whole point of this class.
    """

    def test_adaptive_thinking_survives_the_strip(self):
        body = _upstream_body("anthropic", _translator_only_cc(_CC_MODEL))

        assert body["thinking"] == {"type": "adaptive"}

    def test_effort_survives_the_strip(self):
        body = _upstream_body("anthropic", _translator_only_cc(_CC_MODEL))

        assert body["effort"] == "high"

    def test_the_agents_own_content_is_untouched(self):
        """I1: stripping kitty's keys must not disturb the conversation."""
        body = _upstream_body("openai", _cc_request(_CC_MODEL, native=False))

        assert body["messages"][-1]["content"] == "hello"
