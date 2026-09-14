"""What the translated Messages route ships for pasted images and documents (KBR-222).

Until KBR-222 the Messages→CC translator kept only ``text`` blocks, so a
pasted screenshot or document never reached the upstream and a tool_result's
sibling text was dropped.  The fix carries images as CC ``image_url`` parts
and documents on the ``_documents`` internal key, and the Anthropic family
restores both — re-joining a tool run and its trailing user message into the
one Anthropic user message the wire prescribes.

Every test here is driven from an agent's **Messages body** through
:meth:`~kitty.bridge.messages.translator.MessagesTranslator.translate_request`
and then the adapter's ``translate_to_upstream`` — the translated route as the
bridge runs it, the same harness
``tests/providers/test_anthropic_output_config.py`` uses.  Assertions are on
exact wire structures: §3.3.1b's merge pipeline is order- and split-blind, so
only structural asserts carry the re-join claim.
"""

from __future__ import annotations

from kitty.bridge.messages.translator import MessagesTranslator
from kitty.providers.anthropic import AnthropicAdapter

_PNG_SOURCE = {"type": "base64", "media_type": "image/png", "data": "aWNvbg=="}
_PDF_BLOCK = {
    "type": "document",
    "source": {"type": "base64", "media_type": "application/pdf", "data": "cGRm"},
}


def _messages_body(*messages):
    """Build a minimal Messages API request carrying *messages*."""
    return {"model": "claude-sonnet-5", "max_tokens": 64, "messages": list(messages)}


def _route(body):
    """Run the translated route: hop 1 (Messages→CC) then hop 2 (CC→Anthropic)."""
    return AnthropicAdapter().translate_to_upstream(MessagesTranslator().translate_request(body))


def _assistant_tool_calls(*tool_ids):
    """Build an assistant message that opened the given tool calls."""
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": tool_id, "name": "t", "input": {}} for tool_id in tool_ids
        ],
    }


def _tool_result(tool_id, content="ok"):
    """Build a user tool_result block."""
    return {"type": "tool_result", "tool_use_id": tool_id, "content": content}


class TestImageRestore:
    """CC ``image_url`` parts come back as Anthropic ``image`` blocks."""

    def test_base64_image_reaches_the_wire_as_an_image_block(self):
        """A pasted screenshot survives both hops with its bytes intact."""
        wire = _route(
            _messages_body(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is in this screenshot?"},
                        {"type": "image", "source": dict(_PNG_SOURCE)},
                    ],
                }
            )
        )
        assert wire["messages"][0]["content"] == [
            {"type": "text", "text": "what is in this screenshot?"},
            {"type": "image", "source": _PNG_SOURCE},
        ]

    def test_url_image_reaches_the_wire_as_a_url_source(self):
        """A URL-source image keeps its URL as a ``url`` source."""
        wire = _route(
            _messages_body(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.com/cat.png"},
                        }
                    ],
                }
            )
        )
        assert wire["messages"][0]["content"] == [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}},
        ]

    def test_image_only_turn_is_not_empty_on_the_wire(self):
        """The image-only turn that used to collapse to ``""`` now carries the image."""
        wire = _route(
            _messages_body({"role": "user", "content": [{"type": "image", "source": dict(_PNG_SOURCE)}]})
        )
        assert wire["messages"][0]["content"] == [{"type": "image", "source": _PNG_SOURCE}]

    def test_image_url_part_carries_its_cache_control_onto_the_block(self):
        """The one member the image rebuild carries besides the source is ``cache_control``.

        A rebuilt ``text`` part keeps every member verbatim (the KBR-199 suite
        pins that); an ``image_url`` part changes shape, so its cross-wire
        ``cache_control`` is carried explicitly, the CC ``image_url`` object
        is not, and nor is any other part member — ``detail``, OpenAI's own,
        is in the input to pin that drop empirically. The exact-equality
        assert is the pin: anything extra appearing on the block fails it.
        """
        part = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aWNvbg==", "detail": "high"},
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
        cc = {
            "model": "claude-sonnet-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [dict(part)]}],
        }
        wire = AnthropicAdapter().translate_to_upstream(cc)
        assert wire["messages"][0]["content"] == [
            {
                "type": "image",
                "source": _PNG_SOURCE,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]


class TestDocumentRestore:
    """``_documents`` entries are spliced back into their own turn."""

    def test_document_reaches_the_wire_inside_its_own_turn(self):
        """Two documents in two turns each land in the turn that carried them."""
        other_document = {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "cGRmMg=="},
        }
        wire = _route(
            _messages_body(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "first"}, dict(_PDF_BLOCK)],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "second"}, other_document],
                },
            )
        )
        assert wire["messages"][0]["content"] == [
            {"type": "text", "text": "first"},
            _PDF_BLOCK,
        ]
        assert wire["messages"][1]["content"] == [
            {"type": "text", "text": "second"},
            other_document,
        ]

    def test_document_is_forfeited_when_its_message_is_gone(self):
        """An entry addressed to a message the body no longer carries attaches nowhere.

        The compaction case: identity addressing may forfeit the document but
        must never attach it to a different turn.
        """
        translator = MessagesTranslator()
        cc = translator.translate_request(
            _messages_body({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        )
        # Simulate compaction rebuilding the message list: the addressed dict
        # identity no longer appears in cc["messages"].
        cc["_documents"] = [{"message": {"role": "user", "content": "rebuilt"}, "blocks": [dict(_PDF_BLOCK)]}]
        wire = AnthropicAdapter().translate_to_upstream(cc)
        assert not any(
            isinstance(block, dict) and block.get("type") == "document"
            for message in wire["messages"]
            for block in (message["content"] if isinstance(message["content"], list) else ())
        )

    def test_unknown_part_forwards_verbatim(self):
        """A part type this hop cannot name reaches the upstream unchanged."""
        part = {"type": "custom_thing", "x": 1}
        cc = {
            "model": "claude-sonnet-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [dict(part)]}],
        }
        wire = AnthropicAdapter().translate_to_upstream(cc)
        assert wire["messages"][0]["content"] == [part]

    def test_malformed_documents_value_is_skipped_not_raised(self):
        """The R5 contract: an internal key carrying junk must never fail the request.

        ``tests/test_internal_keys_not_sent_upstream.py`` injects a probe into
        every discovered key; ``_documents`` holding a non-list must degrade to
        "no documents", the way ``_top_k`` degrades to a verbatim copy.
        """
        cc = {
            "model": "claude-sonnet-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "_documents": True,
        }
        wire = AnthropicAdapter().translate_to_upstream(cc)
        assert wire["messages"][0]["content"] == [{"type": "text", "text": "hi"}]


class TestToolRunRejoin:
    """A tool run and its trailing user turn re-join into one Anthropic user message."""

    def test_tool_run_and_sibling_text_rejoin_into_one_user_message(self):
        """The system reminder beside a tool_result ships inside the same user message."""
        wire = _route(
            _messages_body(
                {"role": "user", "content": "go"},
                _assistant_tool_calls("a", "b"),
                {"role": "user", "content": [_tool_result("a"), _tool_result("b")]},
                {"role": "user", "content": [{"type": "text", "text": "keep going"}]},
            )
        )
        assert wire["messages"][-1] == {
            "role": "user",
            "content": [
                _tool_result("a"),
                _tool_result("b"),
                {"type": "text", "text": "keep going"},
            ],
        }

    def test_document_beside_tool_result_rejoins_into_one_user_message(self):
        """The empty-text trailing message hop 1 emits still receives its document."""
        wire = _route(
            _messages_body(
                {"role": "user", "content": "go"},
                _assistant_tool_calls("a"),
                {"role": "user", "content": [_tool_result("a"), dict(_PDF_BLOCK)]},
            )
        )
        assert wire["messages"][-1] == {
            "role": "user",
            "content": [_tool_result("a"), _PDF_BLOCK],
        }

    def test_image_beside_tool_result_rejoins_into_one_user_message(self):
        """An image sibling ships inside the re-joined user message, bytes intact."""
        wire = _route(
            _messages_body(
                {"role": "user", "content": "go"},
                _assistant_tool_calls("a"),
                {
                    "role": "user",
                    "content": [
                        _tool_result("a"),
                        {"type": "image", "source": dict(_PNG_SOURCE)},
                    ],
                },
            )
        )
        assert wire["messages"][-1] == {
            "role": "user",
            "content": [
                _tool_result("a"),
                {"type": "image", "source": _PNG_SOURCE},
            ],
        }

    def test_bare_parallel_tool_results_keep_two_messages(self):
        """Without a trailing user turn, today's per-result messages are unchanged."""
        wire = _route(
            _messages_body(
                {"role": "user", "content": "go"},
                _assistant_tool_calls("a", "b"),
                {"role": "user", "content": [_tool_result("a"), _tool_result("b")]},
            )
        )
        assert wire["messages"][-2:] == [
            {"role": "user", "content": [_tool_result("a")]},
            {"role": "user", "content": [_tool_result("b")]},
        ]

    def test_trailing_tool_run_keeps_per_message_shape(self):
        """A tool run with no user follower is untouched."""
        wire = _route(
            _messages_body(
                {"role": "user", "content": "go"},
                _assistant_tool_calls("a"),
                {"role": "user", "content": [_tool_result("a")]},
            )
        )
        assert wire["messages"][-1] == {"role": "user", "content": [_tool_result("a")]}

    def test_string_user_content_is_unchanged(self):
        """A text-only turn still ships a string content, as before the fix."""
        wire = _route(_messages_body({"role": "user", "content": "hello"}))
        assert wire["messages"][0] == {"role": "user", "content": "hello"}
