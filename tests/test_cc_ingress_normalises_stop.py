"""Structural guard: the Chat Completions ingress normalises a string ``stop``.

``StopConfiguration`` in ``openai/openai-openapi`` declares ``stop`` as ``oneOf``
a string or an array of one to four strings, which makes ``"END"`` and
``["END"]`` the same request.  Every wire kitty writes downstream accepts only
the array form -- Anthropic declares ``Array<string>``, Converse a list, Ollama
an array -- so :func:`kitty.bridge.server._normalize_cc_stop` wraps a non-empty
string once, at the ingress, before the inbound body forks.

**Why this file exists separately.** Every behavioural case for that helper calls
it directly, so all of them would still pass if the call at the ingress were
deleted -- and a string ``stop`` would go back to reaching Anthropic as
``stop_sequences: "END"``, which it rejects.  Nothing else in the suite notices
an unwired normaliser.

Placing a string-form field's normalisation late and partially is a defect this
project has already shipped once: KBR-144, where the second path iterated the
Responses ``input`` string and sent the user's text as a list of its own
letters.  Register row M15 records the resulting placement rule, and KBR-178's
R11 follows it.  This guard is what holds the placement.

See ``.system_design/TEST_SUITE.md`` §6.2.3 and gap G30.
"""

from __future__ import annotations

import inspect

import pytest

from kitty.bridge.server import BridgeServer

# L2: a structural scan of source text -- the shipped call site and the rule it
# must satisfy are edited separately and must agree. The marker keeps a source
# scan out of the L1 set that mutation testing judges, matching
# `tests/test_internal_key_completeness.py`.
pytestmark = pytest.mark.l2


class TestTheCCIngressNormalisesStop:
    """R11's placement, which no behavioural test can observe."""

    def test_the_handler_calls_the_normaliser(self) -> None:
        """``_handle_chat_completions`` must call ``_normalize_cc_stop``."""
        source = inspect.getsource(BridgeServer._handle_chat_completions)

        assert "_normalize_cc_stop(cc_request)" in source, (
            "The Chat Completions ingress no longer normalises `stop`. A string-form "
            "`stop` is legal (StopConfiguration is a oneOf) and every wire kitty writes "
            "takes only the array form, so without this call it reaches Anthropic as "
            '`stop_sequences: "END"` and is rejected. See KBR-178 R11, register row M15 '
            "for the placement rule, and KBR-144 for what late normalisation cost."
        )

    def test_it_is_called_before_the_body_forks(self) -> None:
        """The call must precede the model and per-adapter normalisation hooks.

        M15's rule is "before the inbound body forks".  ``_normalize_model`` and
        the adapter's ``normalize_request`` are the first two things that read
        the body, so the wrap has to come before both of them or a hook could
        branch on the un-normalised form.
        """
        source = inspect.getsource(BridgeServer._handle_chat_completions)

        wrap = source.index("_normalize_cc_stop(cc_request)")
        model = source.index("self._normalize_model(cc_request)")
        adapter = source.index("normalize_request(cc_request)")

        assert wrap < model < adapter, (
            "`_normalize_cc_stop` must run before `_normalize_model` and the adapter's "
            "`normalize_request`, so neither can branch on a string-form `stop`."
        )
