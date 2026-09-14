"""Unit tests for the Responses-body protection machinery (KBR-169).

The end-to-end claims live in
:mod:`tests.bridge.test_bridge_server_openai_subscription`, where they belong:
the wire either carries the protected conversation or it does not. What no
end-to-end test can pin is the *selection policy* inside
:meth:`BridgeServer._prune_compacted_responses_input` — a wire hop split by
compaction in either direction, an orphan left behind by the split, and the
item-ownership walk those decisions read. The Responses API rejects a
``function_call`` whose riding reasoning is missing and a ``reasoning`` item
whose following item is gone, so each split direction is asserted separately.

``walk_input_items`` is the single implementation of
``translate_request``'s conversation loop, so the corpus test pins it to that
method, and the ownership test pins it against the naive item-index mapping it
must not be.
"""

from __future__ import annotations

import pytest

from kitty.bridge.responses.translator import ResponsesTranslator
from kitty.bridge.server import BridgeServer
from kitty.providers.openai_subscription import OpenAISubscriptionAdapter


def _user_message(text: str) -> dict:
    """Build a Responses user-message input item.

    Args:
        text: The message text.

    Returns:
        The input item dict.
    """
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _assistant_message(text: str) -> dict:
    """Build a Responses assistant-message input item.

    Args:
        text: The message text.

    Returns:
        The input item dict.
    """
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _function_call(call_id: str) -> dict:
    """Build a Responses ``function_call`` input item.

    Args:
        call_id: The client-chosen call identifier outputs reference.

    Returns:
        The input item dict.
    """
    return {"type": "function_call", "call_id": call_id, "name": "tool", "arguments": "{}"}


def _function_call_output(call_id: str, output: str) -> dict:
    """Build a Responses ``function_call_output`` input item.

    Args:
        call_id: The call identifier this output answers.
        output: The tool's output text.

    Returns:
        The input item dict.
    """
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _reasoning_item(text: str) -> dict:
    """Build a Responses reasoning input item.

    Args:
        text: The reasoning summary text.

    Returns:
        The input item dict.
    """
    return {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}


def _make_server() -> BridgeServer:
    """Build a bridge server for direct selector calls.

    No request is ever served; the selector reads no provider state.

    Returns:
        A server whose mutation helpers are exercisable in isolation.
    """
    return BridgeServer(
        adapter=None,
        provider=OpenAISubscriptionAdapter(),
        resolved_key="kbr169-unit-tests",
        model="profile-model",
        provider_config={},
        host="127.0.0.1",
        port=0,
    )


class TestWalkInputItems:
    """The ownership walk must agree with the translation it mirrors."""

    def test_walk_messages_equal_translate_request_across_the_corpus(self) -> None:
        """Every corpus body translates to the same messages both ways.

        The corpus carries each discriminating shape: reasoning folding,
        reasoning skipping a non-assistant message, consecutive reasoning
        items accumulating onto one assistant, a reasoning item with no
        following assistant, an unknown item type, and a system merge that
        the instructions message joins.
        """
        translator = ResponsesTranslator()
        bodies = [
            {
                "instructions": "Be helpful.",
                "input": [_user_message("a"), _assistant_message("b"), _user_message("c")],
            },
            {
                "input": [_reasoning_item("r"), _function_call("c1"), _function_call_output("c1", "o")],
            },
            {
                "input": [_reasoning_item("r"), _user_message("u"), _assistant_message("a")],
            },
            {
                "input": [
                    _reasoning_item("r1"),
                    _reasoning_item("r2"),
                    _function_call("c1"),
                ],
            },
            {"input": [_reasoning_item("r"), _user_message("u")]},
            {"input": [{"type": "web_search_call"}, _assistant_message("a")]},
            {
                "instructions": "Base.",
                "input": [
                    {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "s1"}]},
                    {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "s2"}]},
                    _user_message("u"),
                ],
            },
        ]
        for body in bodies:
            messages, _owners = translator.walk_input_items(body)
            assert messages == translator.translate_request(dict(body))["messages"]

    def test_walk_ownership_is_not_the_naive_item_index(self) -> None:
        """A leading reasoning item is owned by the message it rides forward to.

        ``[reasoning, function_call, function_call_output]`` translates to two
        messages; the naive item-k mapping (item k owns message k) would point
        the reasoning at message 0 and the call at message 1, which does not
        exist. This is the falsification case for the mapping this walk
        replaces.
        """
        translator = ResponsesTranslator()
        body = {
            "input": [
                _reasoning_item("r"),
                _function_call("c1"),
                _function_call_output("c1", "o"),
            ]
        }
        messages, owners = translator.walk_input_items(body)
        assert len(messages) == 2
        assert owners == [0, 0, 1]

    def test_reasoning_with_no_following_assistant_has_no_owner(self) -> None:
        """A reasoning item before a non-assistant tail owns nothing.

        ``translate_request`` discards accumulated reasoning that never meets
        an assistant message, so the riding item must own nothing — the
        selector will drop it with a pruned head instead of shipping it
        stranded.
        """
        translator = ResponsesTranslator()
        messages, owners = translator.walk_input_items(
            {"input": [_reasoning_item("r"), _user_message("u")]}
        )
        assert [m["role"] for m in messages] == ["user"]
        assert owners == [None, 0]

    def test_reasoning_skips_a_non_assistant_message_to_its_owner(self) -> None:
        """The owner is the next assistant message, not the next message.

        ``[reasoning, user, assistant]`` — the reasoning rides past the user
        message, exactly where the translation attaches it.
        """
        translator = ResponsesTranslator()
        _messages, owners = translator.walk_input_items(
            {"input": [_reasoning_item("r"), _user_message("u"), _assistant_message("a")]}
        )
        assert owners == [1, 0, 1]

    def test_system_merge_remaps_ownership_to_the_survivor(self) -> None:
        """Items merged into one system message are owned by the survivor.

        The instructions message and two system items merge; owner indices
        must point into the merged list, not the pre-merge one.
        """
        translator = ResponsesTranslator()
        body = {
            "instructions": "Base.",
            "input": [
                {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "s1"}]},
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "s2"}]},
                _user_message("u"),
            ],
        }
        messages, owners = translator.walk_input_items(body)
        assert [m["role"] for m in messages] == ["system", "user"]
        assert owners == [0, 0, 1]


class TestPruneCompactedResponsesInput:
    """A wire hop split by compaction must stay whole, reasoning and all."""

    @pytest.fixture()
    def server(self) -> BridgeServer:
        """Provide a server for direct selector calls.

        Returns:
            The server under test.
        """
        return _make_server()

    @staticmethod
    def _prune(
        server: BridgeServer,
        body: dict,
        pre: list[dict],
        compacted: list[dict],
    ) -> int:
        """Run the selector against one walk of the body.

        The snapshot must be walked exactly once: each ``walk_input_items``
        call builds fresh message dicts, and the selector matches survival by
        identity — mirroring the production handler, whose snapshot is the
        translated list itself and whose compacted list shares those objects.

        Args:
            server: The server under test.
            body: The Responses request; pruned in place.
            pre: The pre-compaction snapshot, as one walk built it.
            compacted: The surviving message objects, picked from ``pre``.

        Returns:
            The number of items the selector removed.
        """
        return server._prune_compacted_responses_input(
            body, pre, compacted, ResponsesTranslator()
        )

    def test_head_split_hop_keeps_its_riding_reasoning(self, server: BridgeServer) -> None:
        """A hop whose first CC block was pruned ships whole.

        ``[user, R1, F1, F2, O1, O2, user2]``: the CC grouper puts F1 alone in
        one block and F2+O1+O2 in the next. When compaction prunes F1's block,
        the group repair must resurrect F1 — and the riding R1 with it, or the
        wire ships a call without the reasoning item the API requires before
        it.
        """
        body = {
            "input": [
                _user_message("u0"),
                _reasoning_item("R1"),
                _function_call("f1"),
                _function_call("f2"),
                _function_call_output("f1", "o1"),
                _function_call_output("f2", "o2"),
                _user_message("u2"),
            ]
        }
        translator = ResponsesTranslator()
        pre, _ = translator.walk_input_items(body)
        # F1's lone block pruned; F2's block (with both outputs) survives.
        compacted = [pre[0], pre[2], pre[3], pre[4], pre[5]]
        removed = self._prune(server, body, pre, compacted)
        assert removed == 0
        assert [i.get("type") or i.get("role") for i in body["input"]] == [
            "message",
            "reasoning",
            "function_call",
            "function_call",
            "function_call_output",
            "function_call_output",
            "message",
        ]
        assert body["input"][1] == _reasoning_item("R1")

    def test_tail_split_hop_keeps_its_leading_reasoning(self, server: BridgeServer) -> None:
        """A hop whose second CC block was pruned ships whole.

        Same wire hop, opposite pruning: F1's block survives, F2's is pruned.
        The group repair must resurrect F2 and its outputs, and R1 — which the
        translation folded into F1's message — must precede both calls.
        """
        body = {
            "input": [
                _user_message("u0"),
                _reasoning_item("R1"),
                _function_call("f1"),
                _function_call("f2"),
                _function_call_output("f1", "o1"),
                _function_call_output("f2", "o2"),
                _user_message("u2"),
            ]
        }
        translator = ResponsesTranslator()
        pre, _ = translator.walk_input_items(body)
        compacted = [pre[0], pre[1], pre[5]]
        removed = self._prune(server, body, pre, compacted)
        assert removed == 0
        assert body["input"] == [
            _user_message("u0"),
            _reasoning_item("R1"),
            _function_call("f1"),
            _function_call("f2"),
            _function_call_output("f1", "o1"),
            _function_call_output("f2", "o2"),
            _user_message("u2"),
        ]

    def test_wholly_pruned_hop_leaves_no_stranded_items(self, server: BridgeServer) -> None:
        """A hop with nothing surviving goes whole, riding reasoning included."""
        body = {
            "input": [
                _user_message("u0"),
                _reasoning_item("R1"),
                _function_call("f1"),
                _function_call_output("f1", "o1"),
                _user_message("u2"),
            ]
        }
        translator = ResponsesTranslator()
        pre, _ = translator.walk_input_items(body)
        compacted = [pre[0], pre[3]]
        removed = self._prune(server, body, pre, compacted)
        assert removed == 3
        assert body["input"] == [_user_message("u0"), _user_message("u2")]

    def test_orphan_output_from_an_earlier_group_is_swept(self, server: BridgeServer) -> None:
        """An output whose call was pruned from an earlier group does not ship.

        The orphan rule is global, so an output can survive selection while
        the group that declared its call did not; the selector's final sweep
        must drop it, matching what pairing validation does to the CC copy.
        """
        body = {
            "input": [
                _user_message("u0"),
                _function_call("f1"),
                _function_call_output("f1", "o1"),
                _function_call("f2"),
                _function_call_output("f1", "o2-references-f1"),
            ]
        }
        translator = ResponsesTranslator()
        pre, _ = translator.walk_input_items(body)
        compacted = [pre[0], pre[3], pre[4]]
        removed = self._prune(server, body, pre, compacted)
        assert removed == 3
        assert body["input"] == [_user_message("u0"), _function_call("f2")]

    def test_message_item_does_not_join_a_following_hop(self, server: BridgeServer) -> None:
        """A pruned message next to a surviving hop stays pruned.

        The atomic group is riding items + calls + outputs; a plain message
        item before the hop is its own group and must not be resurrected by
        the hop's survival.
        """
        body = {
            "input": [
                _user_message("u0"),
                _function_call("f1"),
                _function_call_output("f1", "o1"),
                _user_message("u2"),
            ]
        }
        translator = ResponsesTranslator()
        pre, _ = translator.walk_input_items(body)
        compacted = [pre[0], pre[1], pre[2]]
        removed = self._prune(server, body, pre, compacted)
        assert removed == 1
        assert body["input"] == [_user_message("u0"), _function_call("f1"), _function_call_output("f1", "o1")]

    def test_riding_item_with_no_owner_is_dropped_on_selection(
        self, server: BridgeServer
    ) -> None:
        """A reasoning item the translation never attached goes with the prune."""
        body = {
            "input": [
                _user_message("u0"),
                _reasoning_item("R1"),
                _user_message("u2"),
            ]
        }
        translator = ResponsesTranslator()
        pre, _ = translator.walk_input_items(body)
        compacted = [pre[1]]
        removed = self._prune(server, body, pre, compacted)
        assert removed == 2
        assert body["input"] == [_user_message("u2")]

    def test_an_edited_message_fails_conservative(self, server: BridgeServer) -> None:
        """A message replaced since the snapshot owns nothing — items drop.

        Survival is matched by identity, not position or value: a dict that
        merely *equals* a snapshot message matches nothing, so the failure
        direction is over-compaction, never an unprotected body.
        """
        body = {
            "input": [
                _user_message("u0"),
                _user_message("u2"),
            ]
        }
        translator = ResponsesTranslator()
        pre, _ = translator.walk_input_items(body)
        replaced = [{"role": "user", "content": "u0"}]
        removed = self._prune(server, body, pre, replaced)
        assert removed == 2
        assert body["input"] == []

    def test_no_pruning_is_a_no_op(self, server: BridgeServer) -> None:
        """A compaction that shrank nothing leaves the body untouched."""
        body = {
            "input": [
                _user_message("u0"),
                _function_call("f1"),
                _function_call_output("f1", "o1"),
            ]
        }
        translator = ResponsesTranslator()
        pre, _ = translator.walk_input_items(body)
        snapshot = list(pre)
        removed = self._prune(server, body, pre, snapshot)
        assert removed == 0
        assert len(body["input"]) == 3
