"""L1 tests for the Ollama ``/api/chat`` reader — T-A6 / KBR-38.

``.system_design/TEST_SUITE.md`` §3.3.1, §3.3.1a, §3.3.1b, §7.4, §7.4.1,
§7.4.2 · plan task **T-A6**.

These tests assert the reader projects every published Ollama ``/api/chat``
example with an empty residual (§3.3.1's totality rule), respects the
six cross-reader normalisations (§3.3.1b), and obeys the seven
register-mapping decisions each later reader must answer the same way
(§7.4.2). Where no published example exercises a wire shape the
parameters document, the test carries an **assembled-from-the-published**
body in the same style as the Gemini test file
(``test_reader_gemini.py``).
"""

from __future__ import annotations

import base64
import inspect
import json
import re
from collections.abc import Mapping
from typing import Any

import pytest

from harness import contract as c
from harness import reader_ollama as r

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


PROJECTOR = r.OllamaChatProjection()


def _captured(body: bytes) -> c.CapturedRequest:
    """Wrap raw body bytes in a :class:`CapturedRequest` with empty headers."""
    return c.CapturedRequest(
        method="POST",
        scheme="http",
        host="localhost",
        path="/api/chat",
        query="",
        headers=[],
        body=body,
    )


def _project(body: Mapping[str, Any] | str | bytes) -> c.Request:
    """Project a body (dict, str, or bytes) through the reader."""
    if isinstance(body, Mapping):
        raw = json.dumps(body, separators=(",", ":")).encode()
    elif isinstance(body, str):
        raw = body.encode()
    else:
        raw = body
    return PROJECTOR.read_request(_captured(raw))


_VALID_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 64).decode()


class TestProtocolConformance:
    """The reader conforms to the §7.4 ``Projection`` protocol."""

    def test_it_is_a_projection(self) -> None:
        assert isinstance(PROJECTOR, c.Projection)

    def test_its_wire_format_is_ollama_chat(self) -> None:
        assert PROJECTOR.wire_format is c.WireFormat.OLLAMA_CHAT

    def test_it_imports_nothing_from_kitty(self) -> None:
        source = inspect.getsource(r)
        offenders = re.findall(r"^\s*(?:from|import)\s+kitty\b", source, re.MULTILINE)
        assert not offenders, (
            "reader_ollama.py must not import from kitty.* — independent-"
            "oracle rule (§3.3.1); offenders: " + repr(offenders)
        )


class TestPublishedExamples:
    """Each ``#### Chat request ...`` example in ``docs/api.md`` round-trips."""

    def test_streaming(self) -> None:
        body = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "why is the sky blue?"}],
        }
        req = _project(body)
        assert req.residual == {}
        assert req.consumed == frozenset(body.keys())
        assert req.envelope.model == "llama3.2"
        assert req.envelope.stream is None
        assert len(req.conversation.turns) == 1
        assert req.conversation.turns[0].role == "user"

    def test_streaming_with_tools(self) -> None:
        body = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "what is the weather in tokyo?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather in a given city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "The city",
                                }
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
            "stream": True,
        }
        req = _project(body)
        assert req.residual == {}
        assert req.consumed == frozenset(body.keys())
        assert req.envelope.stream is True
        assert len(req.conversation.tools) == 1
        assert req.conversation.tools[0].name == "get_weather"

    def test_no_streaming(self) -> None:
        body = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "why is the sky blue?"}],
            "stream": False,
        }
        req = _project(body)
        assert req.residual == {}
        assert req.envelope.stream is False

    def test_no_streaming_with_tools(self) -> None:
        body = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "what is the weather in tokyo?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather in a given city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "The city",
                                }
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
            "stream": False,
        }
        req = _project(body)
        assert req.residual == {}
        assert req.envelope.stream is False
        assert len(req.conversation.tools) == 1

    def test_structured_outputs(self) -> None:
        body = {
            "model": "llama3.1",
            "messages": [
                {
                    "role": "user",
                    "content": "Ollama is 22 years old and busy saving the world. "
                    "Return a JSON object with the age and availability.",
                }
            ],
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                    "available": {"type": "boolean"},
                },
                "required": ["age", "available"],
            },
            "options": {"temperature": 0},
        }
        req = _project(body)
        assert req.residual == {}
        assert req.envelope.extra["format"] == body["format"]
        assert req.conversation.sampling["temperature"] == 0

    def test_with_history(self) -> None:
        body = {
            "model": "llama3.2",
            "messages": [
                {"role": "user", "content": "why is the sky blue?"},
                {"role": "assistant", "content": "due to rayleigh scattering."},
                {"role": "user", "content": "how is that different than mie scattering?"},
            ],
        }
        req = _project(body)
        assert req.residual == {}
        assert [t.role for t in req.conversation.turns] == [
            "user",
            "assistant",
            "user",
        ]

    def test_with_history_with_tools(self) -> None:
        body = {
            "model": "llama3.2",
            "messages": [
                {"role": "user", "content": "what is the weather in Toronto?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Toronto"},
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "11 degrees celsius",
                    "tool_name": "get_weather",
                },
            ],
            "stream": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather in a given city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "The city",
                                }
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
        }
        req = _project(body)
        assert req.residual == {}
        assert [t.role for t in req.conversation.turns] == [
            "user",
            "assistant",
            "user",
        ]
        assistant_turn = req.conversation.turns[1]
        assert [type(p).__name__ for p in assistant_turn.parts] == ["ToolUse"]
        tool_result_turn = req.conversation.turns[2]
        assert [type(p).__name__ for p in tool_result_turn.parts] == ["ToolResult"]
        result = tool_result_turn.parts[0]
        assert result.tool_use_id is None
        assert result.is_error is False
        tool_use = assistant_turn.parts[0]
        assert tool_use.name == "get_weather"
        assert tool_use.id is None
        assert tool_use.arguments == {"city": "Toronto"}

    def test_with_images(self) -> None:
        body = {
            "model": "llava",
            "messages": [
                {
                    "role": "user",
                    "content": "what is in this image?",
                    "images": [_VALID_PNG],
                }
            ],
        }
        req = _project(body)
        assert req.residual == {}
        parts = req.conversation.turns[0].parts
        assert [type(p).__name__ for p in parts] == ["Text", "Image"]
        image = parts[1]
        assert image.digest == c.image_digest(base64.b64decode(_VALID_PNG))
        assert image.media_type is None
        assert image.ref is None

    def test_reproducible_outputs(self) -> None:
        body = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hello!"}],
            "options": {"seed": 101, "temperature": 0},
        }
        req = _project(body)
        assert req.residual == {}
        assert req.conversation.sampling["seed"] == 101
        assert req.conversation.sampling["temperature"] == 0

    def test_with_tools_plain(self) -> None:
        body = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "What is the weather today in Paris?"}],
            "stream": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The location to get the weather for, e.g. San Francisco, CA",
                                },
                                "format": {
                                    "type": "string",
                                    "description": "The format to return the weather in, "
                                    "e.g. 'celsius' or 'fahrenheit'",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "format"],
                        },
                    },
                }
            ],
        }
        req = _project(body)
        assert req.residual == {}
        assert len(req.conversation.tools) == 1
        tool = req.conversation.tools[0]
        assert tool.name == "get_current_weather"
        assert tool.description == "Get the current weather for a location"
        assert tool.strict is None


class TestAssistantThinking:
    """Synthetic bodies for assistant-message ``thinking`` fields."""

    def test_assistant_thinking_part(self) -> None:
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": "answer",
                    "thinking": "let me reason through this",
                }
            ],
        }
        req = _project(body)
        assert req.residual == {}
        parts = req.conversation.turns[0].parts
        assert [type(p).__name__ for p in parts] == ["Text", "Thinking"]
        assert parts[1].text == "let me reason through this"
        assert parts[1].signature is None

    def test_request_level_think_extra(self) -> None:
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "think": "medium",
        }
        req = _project(body)
        assert req.residual == {}
        assert req.envelope.extra["think"] == "medium"

    def test_think_boolean_extra(self) -> None:
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "think": True,
        }
        req = _project(body)
        assert req.residual == {}
        assert req.envelope.extra["think"] is True


_CANONICAL_OPTION_PARAMS = [
    ("temperature", 0.42, "temperature"),
    ("top_p", 0.95, "top_p"),
    ("top_k", 40, "top_k"),
    ("seed", 7, "seed"),
    ("stop", ["\n", "user:"], "stop"),
    ("num_predict", 256, "max_tokens"),
    ("frequency_penalty", 0.5, "frequency_penalty"),
    ("presence_penalty", 0.5, "presence_penalty"),
]


class TestCanonicalSampling:
    """Each canonical ``options.*`` lands on the closed sampling set."""

    @pytest.mark.parametrize(("wire_key", "value", "sampling_key"), _CANONICAL_OPTION_PARAMS)
    def test_canonical_mapping(self, wire_key: str, value: Any, sampling_key: str) -> None:
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {wire_key: value},
        }
        req = _project(body)
        assert req.residual == {}
        assert req.conversation.sampling[sampling_key] == value
        assert wire_key not in req.envelope.extra


_NON_CANONICAL_OPTION_KEYS = [
    "num_ctx",
    "repeat_penalty",
    "repeat_last_n",
    "min_p",
    "num_keep",
    "draft_num_predict",
    "penalize_newline",
    "numa",
    "num_batch",
    "num_gpu",
    "main_gpu",
    "use_mmap",
    "num_thread",
]


class TestNonCanonicalOptions:
    """Each non-canonical recognised option lands in ``envelope.extra`` bare-keyed."""

    @pytest.mark.parametrize("key", _NON_CANONICAL_OPTION_KEYS)
    def test_lands_in_extra_under_bare_key(self, key: str) -> None:
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {key: 1},
        }
        req = _project(body)
        assert req.residual == {}
        assert req.envelope.extra[key] == 1


class TestUnknownOptionsResidualise:
    """An unknown ``options.*`` key residualises at its wire path."""

    def test_unknown_options_key_residualises(self) -> None:
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"min_p2": 0.3, "temperature": 0.7},
        }
        req = _project(body)
        assert req.residual == {"options.min_p2": 0.3}
        assert req.conversation.sampling["temperature"] == 0.7


class TestDeclaredControls:
    """``format``, ``think``, ``keep_alive`` land in ``envelope.extra``."""

    def test_format_extra(self) -> None:
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "format": "json",
        }
        req = _project(body)
        assert req.residual == {}
        assert req.envelope.extra["format"] == "json"

    def test_keep_alive_extra(self) -> None:
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "keep_alive": "5m",
        }
        req = _project(body)
        assert req.residual == {}
        assert req.envelope.extra["keep_alive"] == "5m"


class TestToolCalls:
    """The ``tool_calls`` array projects into ``ToolUse`` parts."""

    def test_id_on_wire_residualises(self) -> None:
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "future-id",
                            "function": {"name": "f", "arguments": {}},
                        }
                    ],
                }
            ],
        }
        req = _project(body)
        assert req.residual == {"messages[0].tool_calls[0].id": "future-id"}

    def test_arguments_object_decodes(self) -> None:
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "f",
                                "arguments": {"a": 1, "b": "two"},
                            }
                        }
                    ],
                }
            ],
        }
        req = _project(body)
        assert req.residual == {}
        assert req.conversation.turns[0].parts[0].arguments == {
            "a": 1,
            "b": "two",
        }

    def test_arguments_absent(self) -> None:
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "f"}}],
                }
            ],
        }
        req = _project(body)
        assert req.residual == {}
        assert req.conversation.turns[0].parts[0].arguments == {}

    def test_arguments_null(self) -> None:
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "f", "arguments": None}}],
                }
            ],
        }
        req = _project(body)
        assert req.residual == {}
        assert req.conversation.turns[0].parts[0].arguments == {}

    def test_arguments_non_object_residualises_raw(self) -> None:
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "f",
                                "arguments": "this is not a JSON object",
                            }
                        }
                    ],
                }
            ],
        }
        req = _project(body)
        assert req.residual == {"messages[0].tool_calls[0].function.arguments": "this is not a JSON object"}
        assert req.conversation.turns[0].parts[0].arguments == {}

    def test_function_absent_raises(self) -> None:
        """A tool_calls entry with no ``function`` raises — the position cannot be vacated."""
        body = {
            "model": "m",
            "messages": [{"role": "assistant", "tool_calls": [{"name": "f"}]}],
        }
        with pytest.raises(c.UnreadableBodyError):
            _project(body)

    def test_function_null_raises(self) -> None:
        """A tool_calls entry with ``function: null`` raises (§7.4.2 rule 7 row 4)."""
        body = {
            "model": "m",
            "messages": [{"role": "assistant", "tool_calls": [{"function": None}]}],
        }
        with pytest.raises(c.UnreadableBodyError):
            _project(body)

    def test_function_scalar_raises(self) -> None:
        """A tool_calls entry with a scalar ``function`` raises."""
        body = {
            "model": "m",
            "messages": [{"role": "assistant", "tool_calls": [{"function": 7}]}],
        }
        with pytest.raises(c.UnreadableBodyError):
            _project(body)

    def test_position_not_vacated_by_null_function(self) -> None:
        """Convergence pin: the two-call example projects both ToolUse parts —
        a ``continue`` on ``function: null`` would shift later indices.
        """
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "f1", "arguments": {}}},
                        {"function": {"name": "f2", "arguments": {}}},
                    ],
                }
            ],
        }
        req = _project(body)
        names = [p.name for p in req.conversation.turns[0].parts]
        assert names == ["f1", "f2"]

    def test_entry_unknown_key_residualises(self) -> None:
        """An unrecognised key on a tool_calls entry residualises at its path."""
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "f", "arguments": {}},
                        }
                    ],
                }
            ],
        }
        req = _project(body)
        # Ollama publishes no ``type`` on tool_calls entries — the CC
        # spelling is an unrecognised wire key here and residualises.
        assert req.residual == {"messages[0].tool_calls[0].type": "function"}

    def test_non_mapping_entry_residualises(self) -> None:
        """A scalar tool_calls entry residualises at its indexed path."""
        body = {
            "model": "m",
            "messages": [{"role": "assistant", "tool_calls": ["not a dict"]}],
        }
        req = _project(body)
        assert req.residual == {"messages[0].tool_calls[0]": "not a dict"}


class TestToolDeclarations:
    """``tools`` entries fail closed on every unmodelled field (review round 1)."""

    def _tools_body(self, tool_entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [tool_entry],
        }

    def test_well_formed_tool_projects(self) -> None:
        req = _project(
            self._tools_body(
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            )
        )
        assert req.residual == {}
        assert req.conversation.tools[0].name == "get_weather"
        assert req.conversation.tools[0].strict is None

    def test_entry_unknown_key_residualises(self) -> None:
        """An unknown field on the tool entry residualises at ``tools[0].<key>``."""
        body = self._tools_body(
            {
                "type": "function",
                "vendor_marker": 1,
                "function": {"name": "f", "parameters": {}},
            }
        )
        req = _project(body)
        assert req.residual == {"tools[0].vendor_marker": 1}
        assert req.conversation.tools[0].name == "f"

    def test_function_unknown_key_residualises(self) -> None:
        """An unknown field inside ``function`` residualises at ``tools[0].function.<key>``."""
        body = self._tools_body(
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {},
                    "strict": True,
                },
            }
        )
        req = _project(body)
        # Ollama publishes no ``strict`` — the Anthropic/CC spelling is
        # an unrecognised wire key here and residualises at its path.
        assert req.residual == {"tools[0].function.strict": True}
        assert req.conversation.tools[0].name == "f"

    def test_non_function_type_residualises_whole_entry(self) -> None:
        """``type`` other than ``"function"`` residualises the whole entry at ``tools[0]``.

        A non-function declaration is one the product cannot faithfully
        forward; naming it (Chat Completions' answer,
        ``reader_chat_completions.py:920-967``) is the honest answer.
        """
        body = self._tools_body(
            {
                "type": "web_search",
                "function": {"name": "f", "parameters": {}},
            }
        )
        req = _project(body)
        assert req.residual == {"tools[0]": {"type": "web_search", "function": {"name": "f", "parameters": {}}}}
        assert req.conversation.tools == ()

    def test_function_absent_residualises_whole_entry(self) -> None:
        """A tool entry with no ``function`` object residualises whole at ``tools[0]``."""
        body = self._tools_body({"type": "function"})
        req = _project(body)
        assert req.residual == {"tools[0]": {"type": "function"}}
        assert req.conversation.tools == ()

    def test_function_null_residualises_whole_entry(self) -> None:
        """``function: null`` on a tool entry residualises whole — tools are
        name-addressed (§3.3.1a), so vacating the slot shifts no indexed path.
        """
        body = self._tools_body({"type": "function", "function": None})
        req = _project(body)
        assert req.residual == {"tools[0]": {"type": "function", "function": None}}
        assert req.conversation.tools == ()


class TestToolResults:
    """``role: "tool"`` messages yield ``ToolResult`` parts."""

    def test_tool_name_consumed_not_residualised(self) -> None:
        body = {
            "model": "m",
            "messages": [{"role": "tool", "content": "r", "tool_name": "f"}],
        }
        req = _project(body)
        assert req.residual == {}
        result = req.conversation.turns[0].parts[0]
        assert result.tool_use_id is None
        assert result.is_error is False
        assert result.content[0].text == "r"
        assert req.source["messages"][0]["tool_name"] == "f"

    def test_unknown_tool_message_key_residualises(self) -> None:
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "tool",
                    "content": "r",
                    "tool_name": "f",
                    "vendor_marker": 1,
                }
            ],
        }
        req = _project(body)
        assert req.residual == {"messages[0].vendor_marker": 1}

    def test_unknown_system_message_key_residualises(self) -> None:
        """A ``role: system`` message with an unknown key residualises at its path."""
        body = {
            "model": "m",
            "messages": [
                {"role": "system", "content": "be terse", "vendor_marker": 1},
                {"role": "user", "content": "hi"},
            ],
        }
        req = _project(body)
        assert req.residual == {"messages[0].vendor_marker": 1}


class TestAssistantContent:
    """Empty content + ``tool_calls`` converges with Chat Completions' canonical exchange."""

    def test_empty_content_with_tool_calls_no_text_part(self) -> None:
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "f", "arguments": {}}}],
                }
            ],
        }
        req = _project(body)
        assert req.residual == {}
        parts = req.conversation.turns[0].parts
        assert [type(p).__name__ for p in parts] == ["ToolUse"]

    def test_null_content_with_tool_calls_no_text_part(self) -> None:
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": "f", "arguments": {}}}],
                }
            ],
        }
        req = _project(body)
        assert req.residual == {}
        parts = req.conversation.turns[0].parts
        assert [type(p).__name__ for p in parts] == ["ToolUse"]

    def test_empty_content_without_tool_calls_projects_empty_text(self) -> None:
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": ""}],
        }
        req = _project(body)
        assert req.residual == {}
        assert req.conversation.turns[0].parts[0].text == ""

    def test_assistant_empty_content_without_tool_calls_projects_empty_text(
        self,
    ) -> None:
        """An assistant message with ``content == ""`` and no ``tool_calls``
        projects ``Text("")`` — §3.3.1's "empty block is a part" rule
        applies on the assistant path too, not only for user messages.
        """
        body = {
            "model": "m",
            "messages": [{"role": "assistant", "content": ""}],
        }
        req = _project(body)
        assert req.residual == {}
        assert req.conversation.turns[0].parts[0].text == ""


class TestPartOrder:
    """Indexed-path convergence: the part order is fixed."""

    def test_assistant_text_thinking_tool_use_order(self) -> None:
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": "answer",
                    "thinking": "thinking",
                    "tool_calls": [
                        {"function": {"name": "f1", "arguments": {}}},
                        {"function": {"name": "f2", "arguments": {}}},
                    ],
                }
            ],
        }
        req = _project(body)
        parts = req.conversation.turns[0].parts
        assert [type(p).__name__ for p in parts] == [
            "Text",
            "Thinking",
            "ToolUse",
            "ToolUse",
        ]
        assert parts[0].text == "answer"
        assert parts[1].text == "thinking"
        assert parts[2].name == "f1"
        assert parts[3].name == "f2"


class TestMergeRule:
    """§3.3.1b's merge rule — tool-results first, run + following merge, same-role merge."""

    def test_tool_results_then_following_user(self) -> None:
        body = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "f", "arguments": {}}}],
                },
                {"role": "tool", "content": "r1", "tool_name": "f"},
                {"role": "tool", "content": "r2", "tool_name": "f"},
                {"role": "user", "content": "thanks"},
            ],
        }
        req = _project(body)
        assert [t.role for t in req.conversation.turns] == [
            "user",
            "assistant",
            "user",
        ]
        last_turn = req.conversation.turns[2]
        assert [type(p).__name__ for p in last_turn.parts] == [
            "ToolResult",
            "ToolResult",
            "Text",
        ]

    def test_consecutive_same_role_merge(self) -> None:
        body = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
            ],
        }
        req = _project(body)
        assert len(req.conversation.turns) == 1
        assert req.conversation.turns[0].role == "user"
        assert [p.text for p in req.conversation.turns[0].parts] == [
            "first",
            "second",
        ]

    def test_system_message_lifted(self) -> None:
        body = {
            "model": "m",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
        }
        req = _project(body)
        assert [t.text for t in req.conversation.system] == ["be terse"]
        assert [t.role for t in req.conversation.turns] == ["user"]


class TestImages:
    """``images`` entries project ``Image`` parts with the canonical digest recipe."""

    def test_valid_png_digests_decoded_bytes(self) -> None:
        body = {
            "model": "llava",
            "messages": [{"role": "user", "content": "x", "images": [_VALID_PNG]}],
        }
        req = _project(body)
        assert req.residual == {}
        image = req.conversation.turns[0].parts[1]
        assert image.digest == c.image_digest(base64.b64decode(_VALID_PNG))

    def test_undecodable_image_residualises_and_digests_raw(self) -> None:
        bad_b64 = "not-valid-base64!!!"
        body = {
            "model": "llava",
            "messages": [{"role": "user", "content": "x", "images": [bad_b64]}],
        }
        req = _project(body)
        image = req.conversation.turns[0].parts[1]
        assert image.digest == c.image_digest(bad_b64.encode("utf-8"))
        assert req.residual == {"messages[0].images[0]": bad_b64}

    def test_lenient_decode_only_string_residualises(self) -> None:
        """A base64 string whose only flaw is a non-alphabet char residualises.

        ``base64.b64decode`` with ``validate=False`` (the default)
        silently discards non-alphabet characters before decoding: a
        lenient decode of ``"aGVsbG8h*"`` discards the ``*`` and
        decodes the rest successfully, producing a different image
        than the wire intends. ``validate=True`` rejects the
        non-alphabet char with ``binascii.Error``. A mutation that
        drops ``validate=True`` survives on a pure-bad-string input
        (the ``*``-discarded remainder also raises there), but not on
        this mixed input — that is why this test exists.
        """
        mixed = "aGVsbG8h*"  # b64('hello!') + a non-alphabet char
        body = {
            "model": "llava",
            "messages": [{"role": "user", "content": "x", "images": [mixed]}],
        }
        req = _project(body)
        assert req.residual == {"messages[0].images[0]": mixed}

    def test_non_string_image_entry_residualises(self) -> None:
        """A non-string image entry (e.g. an int) residualises at its indexed path."""
        body = {
            "model": "llava",
            "messages": [{"role": "user", "content": "x", "images": [7]}],
        }
        req = _project(body)
        assert req.residual == {"messages[0].images[0]": 7}


class TestStreamAbsent:
    """``stream`` is read from the wire record — absent → ``None``."""

    def test_stream_absent_projects_none(self) -> None:
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        req = _project(body)
        assert req.envelope.stream is None


class TestTypedLeafWrongTypeResidualises:
    """A wrong-typed value for a grammar-named leaf residualises at the leaf's path."""

    def test_model_as_int_residualises(self) -> None:
        body = {"model": 7, "messages": [{"role": "user", "content": "x"}]}
        req = _project(body)
        assert req.residual == {"model": 7}
        assert req.envelope.model is None

    def test_stream_as_string_residualises(self) -> None:
        body = {
            "model": "m",
            "stream": "yes",
            "messages": [{"role": "user", "content": "x"}],
        }
        req = _project(body)
        assert req.residual == {"stream": "yes"}
        assert req.envelope.stream is None


_MAXIMAL_BODY: dict[str, Any] = {
    "model": "m",
    "messages": [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "x", "images": [_VALID_PNG]},
        {
            "role": "assistant",
            "content": "",
            "thinking": "thinking",
            "tool_calls": [
                {"function": {"name": "f1", "arguments": {"a": 1}}},
                {"function": {"name": "f2", "arguments": {}}},
            ],
        },
        {"role": "tool", "content": "r", "tool_name": "f1"},
        {"role": "user", "content": "y"},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "g",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ],
    "format": {"type": "object"},
    "think": "high",
    "keep_alive": "5m",
    "stream": False,
    "options": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "seed": 1,
        "stop": ["\n"],
        "num_predict": 100,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
        "num_ctx": 2048,
        "repeat_penalty": 1.1,
        "repeat_last_n": 64,
        "min_p": 0.05,
        "num_keep": 5,
        "draft_num_predict": 4,
        "penalize_newline": True,
        "numa": False,
        "num_batch": 2,
        "num_gpu": 1,
        "main_gpu": 0,
        "use_mmap": True,
        "num_thread": 8,
    },
}


class TestDisjointExtraKeys:
    """§7.4.2 rule 2 — the reader's full ``envelope.extra`` set is pairwise disjoint."""

    def test_full_extra_set_is_pairwise_disjoint(self) -> None:
        req = _project(_MAXIMAL_BODY)
        extra_keys = set(req.envelope.extra.keys())
        assert "format" in extra_keys
        assert "think" in extra_keys
        assert "keep_alive" in extra_keys
        for key in _NON_CANONICAL_OPTION_KEYS:
            assert key in extra_keys, f"missing extra[{key!r}]"
        assert len(extra_keys) == len(_NON_CANONICAL_OPTION_KEYS) + 3


class TestResidualKeySpelling:
    """§7.4.1 residual-key convention — top-level keys bare, nested keys indexed."""

    def test_top_level_unrecognised_key_is_bare(self) -> None:
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "x-kitty-trace": 1,
        }
        req = _project(body)
        assert req.residual == {"x-kitty-trace": 1}

    def test_unknown_message_key_is_indexed(self) -> None:
        body = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "x", "vendor_marker": 1},
                {"role": "user", "content": "y", "vendor_marker": 2},
            ],
        }
        req = _project(body)
        assert req.residual == {
            "messages[0].vendor_marker": 1,
            "messages[1].vendor_marker": 2,
        }


def _mutate_option(body: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    return {**body, "options": {**body["options"], key: value}}


@pytest.mark.parametrize(
    ("label", "mutator", "expected_key"),
    [
        ("messages as string", lambda b: {**b, "messages": "not a list"}, "messages"),
        ("options as string", lambda b: {**b, "options": "not a dict"}, "options"),
        ("options as list", lambda b: {**b, "options": [1, 2]}, "options"),
        (
            "unknown options key",
            lambda b: _mutate_option(b, "vendor_marker", 1),
            "options.vendor_marker",
        ),
        (
            "unknown message key (user)",
            lambda b: {
                **b,
                "messages": [
                    b["messages"][0],
                    {**b["messages"][1], "vendor_marker": 1},
                    *b["messages"][2:],
                ],
            },
            "messages[1].vendor_marker",
        ),
        (
            "unknown system message key",
            lambda b: {
                **b,
                "messages": [
                    {**b["messages"][0], "vendor_marker": 1},
                    *b["messages"][1:],
                ],
            },
            "messages[0].vendor_marker",
        ),
        (
            "tool_calls arguments as string",
            lambda b: {
                **b,
                "messages": [
                    *b["messages"][:2],
                    {
                        **b["messages"][2],
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "f",
                                    "arguments": "not an object",
                                }
                            }
                        ],
                    },
                    *b["messages"][3:],
                ],
            },
            "messages[2].tool_calls[0].function.arguments",
        ),
        (
            "tool id on wire",
            lambda b: {
                **b,
                "messages": [
                    *b["messages"][:2],
                    {
                        **b["messages"][2],
                        "tool_calls": [
                            {
                                "id": "future",
                                "function": {"name": "f", "arguments": {}},
                            }
                        ],
                    },
                    *b["messages"][3:],
                ],
            },
            "messages[2].tool_calls[0].id",
        ),
        ("model as int", lambda b: {**b, "model": 7}, "model"),
        ("stream as string", lambda b: {**b, "stream": "yes"}, "stream"),
    ],
)
class TestInjectionProbeResidualises:
    """Mutations the reader accounts for by residualising, never by raising.

    Each mutation names the exact residual key it must produce — a
    mutation that residualises at a different (or wider) path fails,
    not just one that leaves the residual empty.
    """

    def test_mutation_residualises(self, label: str, mutator: Any, expected_key: str) -> None:
        mutated = mutator(_MAXIMAL_BODY)
        req = _project(mutated)
        assert expected_key in req.residual, (
            f"{label}: expected residual key {expected_key!r}, got {sorted(req.residual)!r}"
        )


@pytest.mark.parametrize(
    ("label", "mutator"),
    [
        (
            "tool_calls function as string",
            lambda b: {
                **b,
                "messages": [
                    *b["messages"][:2],
                    {
                        **b["messages"][2],
                        "tool_calls": [{"function": "not a dict"}],
                    },
                    *b["messages"][3:],
                ],
            },
        ),
        (
            "tool_calls as string",
            lambda b: {
                **b,
                "messages": [
                    *b["messages"][:2],
                    {**b["messages"][2], "tool_calls": "not a list"},
                    *b["messages"][3:],
                ],
            },
        ),
        (
            "content as int (assistant)",
            lambda b: {
                **b,
                "messages": [
                    *b["messages"][:2],
                    {**b["messages"][2], "content": 7},
                    *b["messages"][3:],
                ],
            },
        ),
        (
            "role unknown",
            lambda b: {
                **b,
                "messages": [
                    {**b["messages"][0], "role": "vendor"},
                    *b["messages"][1:],
                ],
            },
        ),
        (
            "missing role",
            lambda b: {
                **b,
                "messages": [
                    {"content": "x"},
                    *b["messages"][1:],
                ],
            },
        ),
    ],
)
class TestInjectionProbeRaises:
    """A wrong-typed value that *is* the part raises ``UnreadableBodyError``."""

    def test_mutation_raises(self, label: str, mutator: Any) -> None:
        mutated = mutator(_MAXIMAL_BODY)
        with pytest.raises(c.UnreadableBodyError):
            _project(mutated)


@pytest.mark.parametrize(
    ("label", "mutator", "expected"),
    [
        ("format as int", lambda b: {**b, "format": 1}, 1),
        ("think as int", lambda b: {**b, "think": 1}, 1),
    ],
)
class TestExtraCarriedWhole:
    """A declared control's value is carried whole in ``envelope.extra``."""

    def test_mutation_carried_whole(self, label: str, mutator: Any, expected: Any) -> None:
        mutated = mutator(_MAXIMAL_BODY)
        req = _project(mutated)
        key = label.split(" as ")[0]
        assert req.envelope.extra[key] == expected
        assert req.residual == {}
