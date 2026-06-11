"""Tests for context compaction in BridgeServer.

Tests verify that _compact_messages correctly prunes large conversation
histories to prevent 400 "context too large" errors from upstream providers.
"""

import json

from kitty.bridge.server import (
    _COMPACTION_GUARANTEED_MESSAGES_MAX,
    _MAX_REQUEST_CHARS,
    BridgeServer,
)
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.providers.model_context import DEFAULT_CONTEXT_TOKENS, tokens_to_chars
from kitty.types import BridgeProtocol

# ── Stub adapters for testing ───────────────────────────────────────────────


class StubLauncher(LauncherAdapter):
    def __init__(self, protocol: BridgeProtocol = BridgeProtocol.MESSAGES_API):
        self._protocol = protocol

    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return self._protocol

    def build_spawn_config(self, profile: Profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class StubProvider(ProviderAdapter):
    @property
    def provider_type(self) -> str:
        return "stub"

    @property
    def default_base_url(self) -> str:
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages, "stream": kwargs.get("stream", False)}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Upstream error {status_code}: {body}")


def _make_server() -> BridgeServer:
    adapter = StubLauncher()
    provider = StubProvider()
    return BridgeServer(adapter, provider, "test-key")


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_messages(n: int, start_idx: int = 0) -> list[dict]:
    """Create n alternating user/assistant messages."""
    msgs = []
    for i in range(start_idx, start_idx + n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Message {i}"})
    return msgs


def _make_tool_call_block(call_id: str, tool_name: str, result_content: str) -> list[dict]:
    """Create an atomic tool_call + tool_result block."""
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "content": result_content,
            "tool_call_id": call_id,
        },
    ]


def _char_size(messages: list[dict]) -> int:
    """Estimate serialized character size of messages list."""
    return len(json.dumps(messages, ensure_ascii=False))


def _make_large_messages(count: int, content_multiplier: int = 2000) -> list[dict]:
    """Create messages large enough to exceed the 2.8M char compaction threshold."""
    messages = [{"role": "system", "content": "System"}]
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"Message {i} " * content_multiplier})
    return messages


# ── Tests ───────────────────────────────────────────────────────────────────


class TestCompactMessagesBelowThreshold:
    """When messages are below the compaction threshold, nothing should change."""

    def test_short_history_unchanged(self):
        server = _make_server()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        original = json.dumps(messages, ensure_ascii=False)

        result = server._compact_messages(messages.copy())

        assert json.dumps(result, ensure_ascii=False) == original

    def test_empty_messages_unchanged(self):
        server = _make_server()
        result = server._compact_messages([])
        assert result == []


class TestCompactMessagesPreservesCriticalMessages:
    """System message and last user message must always be preserved."""

    def test_system_message_always_preserved(self):
        server = _make_server()
        system_msg = {"role": "system", "content": "Critical system prompt " * 1000}
        # Build a large message list that exceeds threshold
        messages = [system_msg] + _make_messages(100)

        result = server._compact_messages(messages.copy())

        # System message must be first
        assert result[0]["role"] == "system"
        assert result[0]["content"] == system_msg["content"]

    def test_oversized_system_message_still_preserved(self):
        """System message must be preserved even if it exceeds head_budget alone."""
        server = _make_server()
        # System prompt > 560K chars (20% of 2.8M threshold)
        huge_system = {"role": "system", "content": "S" * 600_000}
        messages = [huge_system]
        for i in range(120):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i} " * 2000})

        assert _char_size(messages) > 2_800_000

        result = server._compact_messages(messages.copy())

        # System message must survive even though it's larger than head_budget
        assert result[0]["role"] == "system"
        assert len(result[0]["content"]) == 600_000

    def test_last_user_message_preserved(self):
        server = _make_server()
        last_user = {"role": "user", "content": "This is the critical last question"}
        messages = _make_messages(100) + [last_user]

        result = server._compact_messages(messages.copy())

        # Find the last user message in result
        user_msgs = [m for m in result if m["role"] == "user"]
        assert any(m["content"] == "This is the critical last question" for m in user_msgs)

    def test_system_and_last_user_both_preserved(self):
        server = _make_server()
        system_msg = {"role": "system", "content": "System prompt"}
        last_user = {"role": "user", "content": "Final question"}
        messages = [system_msg] + _make_messages(100) + [last_user]

        result = server._compact_messages(messages.copy())

        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System prompt"
        user_msgs = [m for m in result if m["role"] == "user"]
        assert any(m["content"] == "Final question" for m in user_msgs)


class TestCompactMessagesHeadTail:
    """Head+Tail compaction: keep head (system+initial) and tail (recent)."""

    def test_reduces_message_count(self):
        server = _make_server()
        messages = _make_large_messages(200)

        original_size = _char_size(messages)
        assert original_size > 2_800_000, f"Test data too small: {original_size}"

        result = server._compact_messages(messages.copy())
        result_size = _char_size(result)

        assert result_size < original_size
        assert len(result) < len(messages)

    def test_head_preserves_initial_context(self):
        server = _make_server()
        system_msg = {"role": "system", "content": "System"}
        first_user = {"role": "user", "content": "This is my initial task description " * 100}

        messages = [system_msg, first_user] + _make_messages(200, start_idx=2)
        # Make messages large enough to trigger compaction
        for m in messages[2:]:
            m["content"] = m["content"] + " " * 2000

        result = server._compact_messages(messages.copy())

        # First two messages (system + first user) should be preserved
        assert result[0]["role"] == "system"
        assert result[1]["content"].startswith("This is my initial task description")

    def test_tail_preserves_recent_messages(self):
        server = _make_server()
        recent_messages = [
            {"role": "user", "content": "Recent question 1"},
            {"role": "assistant", "content": "Recent answer 1"},
            {"role": "user", "content": "Recent question 2"},
            {"role": "assistant", "content": "Recent answer 2"},
        ]
        messages = [{"role": "system", "content": "System"}]
        for i in range(150):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Old message {i} " * 300})
        messages.extend(recent_messages)

        result = server._compact_messages(messages.copy())

        # Last 4 messages should match the recent_messages
        for rm, actual in zip(recent_messages, result[-4:], strict=True):
            assert actual["content"] == rm["content"]


class TestCompactMessagesToolResultTruncation:
    """Large tool results should be truncated to save space."""

    def test_large_tool_result_truncated(self):
        server = _make_server()
        large_content = "X" * 100_000

        # Build messages large enough to exceed compaction threshold
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": "Let me check",
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
                ],
            },
            {
                "role": "tool",
                "content": large_content,
                "tool_call_id": "call_123",
            },
        ]
        # Add filler to push total over the 2.8M threshold
        for i in range(50):
            messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": "Filler " * 10000})

        assert _char_size(messages) > 2_800_000

        result = server._compact_messages(messages.copy())

        # The tool result should be truncated
        tool_results = [m for m in result if m["role"] == "tool"]
        assert len(tool_results) == 1
        assert len(tool_results[0]["content"]) < 100_000
        assert "truncated" in tool_results[0]["content"].lower()
        assert "original size" in tool_results[0]["content"].lower()

    def test_truncation_preserves_tool_call_id(self):
        server = _make_server()
        large_content = "Y" * 100_000
        # Properly paired assistant(tool_calls) + tool(result) — the test
        # verifies that after truncation the tool_call_id is still preserved.
        tool_call_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc_456",
                    "type": "function",
                    "function": {"name": "t", "arguments": "{}"},
                }
            ],
        }
        tool_msg = {
            "role": "tool",
            "content": large_content,
            "tool_call_id": "call_abc_456",
        }

        messages = [{"role": "system", "content": "System"}, tool_call_msg, tool_msg]
        for i in range(50):
            messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": "Filler " * 10000})

        result = server._compact_messages(messages.copy())

        tool_results = [m for m in result if m["role"] == "tool"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_call_id"] == "call_abc_456"

    def test_small_tool_result_unchanged(self):
        server = _make_server()
        small_content = "File contents here"
        tool_msg = {
            "role": "tool",
            "content": small_content,
            "tool_call_id": "call_123",
        }

        messages = [
            {"role": "system", "content": "System"},
            tool_msg,
        ]

        result = server._compact_messages(messages.copy())

        tool_results = [m for m in result if m["role"] == "tool"]
        assert tool_results[0]["content"] == small_content


class TestCompactMessagesToolCallAtomicity:
    """Tool-call + tool-result pairs must never be split by pruning."""

    def test_tool_call_result_pair_kept_together_in_tail(self):
        """If a tool result is in the tail, its assistant tool_call must also be there."""
        server = _make_server()
        block = _make_tool_call_block("call_tail", "read_file", "File content here")

        messages = [{"role": "system", "content": "System"}]
        for i in range(200):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Filler {i} " * 2000})
        messages.extend(block)

        assert _char_size(messages) > 2_800_000

        result = server._compact_messages(messages.copy())

        # Find the tool result in the output
        tool_result_idx = None
        for i, m in enumerate(result):
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_tail":
                tool_result_idx = i
                break

        assert tool_result_idx is not None, "Tool result should be present in compacted output"
        # The immediately preceding message must be the assistant tool_call
        assert tool_result_idx > 0
        assert result[tool_result_idx - 1]["role"] == "assistant"
        assert result[tool_result_idx - 1].get("tool_calls") is not None

    def test_tool_call_result_pair_kept_together_in_head(self):
        """If an assistant tool_call is in the head, its tool result must also be there."""
        server = _make_server()
        block = _make_tool_call_block("call_head", "list_files", "file1.py\nfile2.py")

        messages = [
            {"role": "system", "content": "System"},
        ]
        messages.extend(block)
        # Add enough filler after to force compaction
        for i in range(200):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Filler {i} " * 2000})

        assert _char_size(messages) > 2_800_000

        result = server._compact_messages(messages.copy())

        # Find the assistant tool_call in the output
        tool_call_idx = None
        for i, m in enumerate(result):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if tc["id"] == "call_head":
                        tool_call_idx = i
                        break

        assert tool_call_idx is not None, "Tool call should be preserved in compacted output"
        # The immediately next message must be the tool result
        assert tool_call_idx + 1 < len(result)
        assert result[tool_call_idx + 1]["role"] == "tool"
        assert result[tool_call_idx + 1]["tool_call_id"] == "call_head"


class TestCompactMessagesToolCallPairingValidation:
    """Orphan tool results (no matching tool_use) must be dropped by compaction.

    The bridge's atomic-block grouping keeps paired tool_use/tool_result
    blocks together, but a corrupt conversation may still have a `tool`
    message whose ``tool_call_id`` has no preceding ``assistant(tool_calls)``
    with a matching ``tool_use.id``. The upstream rejects these requests
    with code 2013 ("tool call result does not follow tool call"). The
    compaction post-condition must drop these orphans before the request
    is sent.
    """

    def test_orphan_tool_result_dropped(self):
        """Input has a tool message with no matching tool_use; the tool is dropped."""
        server = _make_server()
        messages = [{"role": "system", "content": "System"}]
        # Add large filler to force compaction
        for i in range(200):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Filler {i} " * 2000})
        # Orphan tool result at the end (lands in tail)
        messages.append(
            {
                "role": "tool",
                "content": "orphan result",
                "tool_call_id": "call_orphan_1",
            },
        )

        assert _char_size(messages) > 2_800_000

        result = server._compact_messages(messages.copy())

        tool_results = [m for m in result if m.get("role") == "tool"]
        assert tool_results == [], f"Orphan tool result must be dropped, got {tool_results}"
        # Non-tool messages preserved
        assert any(m.get("role") == "system" for m in result)

    def test_paired_tool_result_kept(self):
        """Properly paired assistant(tool_calls) + tool both survive."""
        server = _make_server()
        block = _make_tool_call_block("call_paired", "read_file", "file contents")

        messages = [{"role": "system", "content": "System"}]
        messages.extend(block)
        messages.append({"role": "user", "content": "Thanks"})

        result = server._compact_messages(messages.copy())

        # Both the assistant tool_call and the tool result should survive
        assert any(
            m.get("role") == "assistant" and m.get("tool_calls") and m["tool_calls"][0]["id"] == "call_paired"
            for m in result
        )
        assert any(m.get("role") == "tool" and m.get("tool_call_id") == "call_paired" for m in result)

    def test_multiple_orphan_tool_results_all_dropped(self):
        """Multiple orphan tool messages are all dropped, no others lost."""
        server = _make_server()
        messages = [{"role": "system", "content": "System"}]
        # Add large filler to force compaction
        for i in range(200):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Filler {i} " * 2000})
        # Three orphans at the end (in the tail region)
        messages.extend(
            [
                {"role": "tool", "content": "orphan A", "tool_call_id": "call_orphan_a"},
                {"role": "tool", "content": "orphan B", "tool_call_id": "call_orphan_b"},
                {"role": "tool", "content": "orphan C", "tool_call_id": "call_orphan_c"},
                {"role": "assistant", "content": "OK"},
            ]
        )

        assert _char_size(messages) > 2_800_000

        result = server._compact_messages(messages.copy())

        # No orphans survive
        assert not any(m.get("role") == "tool" for m in result)
        # System message preserved
        assert any(m.get("role") == "system" for m in result)

    def test_orphan_in_tail_block_dropped_after_large_compaction(self):
        """An orphan tool message in the tail region is dropped during compaction."""
        server = _make_server()
        # Valid pair at the start (likely in head)
        valid_block = _make_tool_call_block("call_valid", "list_files", "a.py\nb.py")

        messages = [{"role": "system", "content": "System"}]
        messages.extend(valid_block)
        # Large filler in the middle
        for i in range(200):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Filler {i} " * 2000})
        # Orphan tool result at the very end (in the tail region)
        messages.append(
            {
                "role": "tool",
                "content": "orphan tail result",
                "tool_call_id": "call_orphan_tail",
            }
        )

        assert _char_size(messages) > 2_800_000

        result = server._compact_messages(messages.copy())

        # Orphan in tail must be dropped
        assert not any(m.get("tool_call_id") == "call_orphan_tail" for m in result)
        # Valid pair must survive
        assert any(m.get("role") == "tool" and m.get("tool_call_id") == "call_valid" for m in result)

    def test_existing_paired_messages_survive_regression(self):
        """Regression: previously valid atomic blocks still survive the validation pass."""
        server = _make_server()
        block1 = _make_tool_call_block("call_a", "read", "a-content")
        block2 = _make_tool_call_block("call_b", "write", "b-content")
        block3 = _make_tool_call_block("call_c", "list", "c-content")

        messages = [{"role": "system", "content": "System"}]
        messages.extend(block1)
        messages.append({"role": "assistant", "content": "thinking..."})
        messages.extend(block2)
        messages.append({"role": "user", "content": "next question"})
        messages.extend(block3)

        result = server._compact_messages(messages.copy())

        # All three tool_use ids and all three tool result ids must be present
        kept_tool_use_ids = {
            tc["id"] for m in result if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])
        }
        kept_tool_result_ids = {m.get("tool_call_id") for m in result if m.get("role") == "tool"}
        assert {"call_a", "call_b", "call_c"}.issubset(kept_tool_use_ids)
        assert {"call_a", "call_b", "call_c"}.issubset(kept_tool_result_ids)


class TestValidateToolCallPairingHelper:
    """Unit tests for the standalone _validate_tool_call_pairing helper."""

    def test_valid_input_unchanged(self):
        server = _make_server()
        block = _make_tool_call_block("call_x", "tool", "result")
        messages = [{"role": "system", "content": "S"}, *block, {"role": "user", "content": "Q"}]

        result = server._validate_tool_call_pairing(messages)

        # Order and content preserved
        assert result == messages

    def test_orphan_dropped_with_warning(self, caplog):
        import logging

        server = _make_server()
        messages = [
            {"role": "user", "content": "Q"},
            {
                "role": "tool",
                "content": "orphan result",
                "tool_call_id": "call_orphan_z",
            },
        ]

        with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
            result = server._validate_tool_call_pairing(messages)

        # Orphan dropped
        assert not any(m.get("role") == "tool" for m in result)
        # Non-tool messages preserved
        assert any(m.get("role") == "user" for m in result)
        # Warning emitted
        assert any("orphan" in r.message.lower() or "tool_call_id" in r.message for r in caplog.records)

    def test_no_assistant_messages_keeps_input(self):
        """Input with only user/system messages returns unchanged."""
        server = _make_server()
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Q1"},
            {"role": "user", "content": "Q2"},
        ]

        result = server._validate_tool_call_pairing(messages)

        assert result == messages

    def test_tool_result_with_matching_id_after_multiple_assistants(self):
        """A tool result whose id appears in any preceding assistant is preserved."""
        server = _make_server()
        messages = [
            {"role": "user", "content": "Q1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_first",
                        "type": "function",
                        "function": {"name": "t1", "arguments": "{}"},
                    }
                ],
            },
            {"role": "user", "content": "Q2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_second",
                        "type": "function",
                        "function": {"name": "t2", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "second result", "tool_call_id": "call_second"},
        ]

        result = server._validate_tool_call_pairing(messages)

        # The tool result for call_second must survive (matches the second assistant)
        assert any(m.get("role") == "tool" and m.get("tool_call_id") == "call_second" for m in result)
        # Order preserved
        assert len(result) == len(messages)

    def test_assistant_turn_with_multiple_tool_calls_keeps_all_matching_results(self):
        """An assistant with N tool_calls keeps N corresponding tool results."""
        server = _make_server()
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "t1", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "t2", "arguments": "{}"}},
                    {"id": "call_3", "type": "function", "function": {"name": "t3", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "content": "r1", "tool_call_id": "call_1"},
            {"role": "tool", "content": "r2", "tool_call_id": "call_2"},
            {"role": "tool", "content": "r3", "tool_call_id": "call_3"},
        ]

        result = server._validate_tool_call_pairing(messages)

        kept_ids = {m.get("tool_call_id") for m in result if m.get("role") == "tool"}
        assert kept_ids == {"call_1", "call_2", "call_3"}


class TestApplyCompactionWiresValidation:
    """Wiring tests: _apply_compaction invokes the pairing validator."""

    def test_apply_compaction_drops_orphan_tool_results(self):
        """After _apply_compaction, no orphan tool results remain in messages."""
        server = _make_server()
        # Build a request large enough to trigger compaction
        messages = [{"role": "system", "content": "System"}]
        for i in range(200):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Filler {i} " * 2000})
        # Add an orphan at the end
        messages.append(
            {
                "role": "tool",
                "content": "orphan result",
                "tool_call_id": "call_orphan_apply",
            }
        )
        cc_request = {"model": "test-model", "messages": messages}

        server._apply_compaction(cc_request)

        # No orphan tool results should remain
        orphans = [
            m
            for m in cc_request["messages"]
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_orphan_apply"
        ]
        assert orphans == []

    def test_apply_compaction_preserves_paired_messages(self):
        """Properly paired tool_use/tool_result messages survive _apply_compaction."""
        server = _make_server()
        block = _make_tool_call_block("call_preserved", "read_file", "file data")

        messages = [{"role": "system", "content": "System"}]
        messages.extend(block)
        for i in range(200):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Filler {i} " * 2000})
        messages.extend(block)  # second pair

        cc_request = {"model": "test-model", "messages": messages}

        server._apply_compaction(cc_request)

        # Both blocks' tool results should still be present (atomic-block
        # grouping keeps them; validation does not drop paired results).
        kept_tool_result_ids = {m.get("tool_call_id") for m in cc_request["messages"] if m.get("role") == "tool"}
        assert "call_preserved" in kept_tool_result_ids


class TestCompactMessagesLogging:
    """Compaction should log when triggered."""

    def test_logs_compaction_summary(self, caplog):
        import logging

        server = _make_server()
        messages = _make_large_messages(200)

        assert _char_size(messages) > 2_800_000

        with caplog.at_level(logging.INFO, logger="kitty.bridge.server"):
            server._compact_messages(messages.copy())

        assert any("compaction" in r.message.lower() for r in caplog.records)


class TestCompactMessagesEdgeCases:
    """Edge cases for the compaction logic."""

    def test_single_message_unchanged(self):
        server = _make_server()
        messages = [{"role": "user", "content": "Hello"}]
        result = server._compact_messages(messages.copy())
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_all_messages_fit_unchanged(self):
        """If all messages fit under threshold, return as-is."""
        server = _make_server()
        messages = [
            {"role": "system", "content": "Short system"},
            {"role": "user", "content": "Short user"},
            {"role": "assistant", "content": "Short response"},
        ]
        original = json.dumps(messages, ensure_ascii=False)

        result = server._compact_messages(messages.copy())

        assert json.dumps(result, ensure_ascii=False) == original

    def test_messages_with_only_system_and_user(self):
        """Minimal conversation should pass through unchanged."""
        server = _make_server()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        result = server._compact_messages(messages.copy())
        assert len(result) == 2

    def test_no_overlap_between_head_and_tail(self):
        """Head and tail must not contain duplicate messages."""
        server = _make_server()
        messages = _make_large_messages(200)

        result = server._compact_messages(messages.copy())

        # Check no message appears twice (by identity)
        assert len(result) == len({id(m) for m in result})


class TestCompactMessagesGuaranteedFit:
    """Compaction must guarantee the serialized messages fit under _COMPACTION_GUARANTEED_MESSAGES_MAX.

    After head+tail pruning, if the result is still too large, the method must
    iteratively drop oldest blocks (head first, then tail front) until it fits.
    System messages and at least one tail block must always be preserved.
    """

    def test_guaranteed_fit_reduces_oversized_result(self):
        """Even enormous messages must be reduced below the guaranteed max."""
        server = _make_server()
        # Build messages where each block is huge, so head+tail pruning still
        # leaves a result above the guaranteed budget.
        messages = [{"role": "system", "content": "System prompt"}]
        for i in range(30):
            role = "user" if i % 2 == 0 else "assistant"
            # Each message ~200K chars; 30 of them = ~6M total
            messages.append({"role": role, "content": f"Message {i} " * 50_000})

        original_size = _char_size(messages)
        assert original_size > _COMPACTION_GUARANTEED_MESSAGES_MAX

        result = server._compact_messages(messages.copy())
        result_size = _char_size(result)

        assert result_size <= _COMPACTION_GUARANTEED_MESSAGES_MAX, (
            f"Compacted size {result_size} exceeds guaranteed max {_COMPACTION_GUARANTEED_MESSAGES_MAX}"
        )

    def test_guaranteed_fit_drops_head_before_tail(self):
        """When shrinking, oldest head blocks are dropped before any tail blocks."""
        server = _make_server()
        # Small system, distinct head messages, distinct tail messages
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "HEAD_FIRST " * 100_000},  # ~1.2M
            {"role": "assistant", "content": "HEAD_SECOND " * 100_000},  # ~1.3M
            {"role": "user", "content": "MIDDLE " * 100_000},  # ~0.7M
            {"role": "assistant", "content": "TAIL_SECOND " * 100_000},  # ~1.3M
            {"role": "user", "content": "TAIL_FIRST " * 100_000},  # ~1.2M
        ]

        result = server._compact_messages(messages.copy())
        result_text = json.dumps(result, ensure_ascii=False)

        # System must survive
        assert result[0]["role"] == "system"

        # Most recent tail content must survive (it's most valuable)
        assert "TAIL_FIRST" in result_text

        # Head content should be dropped to make room
        assert "HEAD_FIRST" not in result_text

    def test_guaranteed_fit_trims_tail_keeping_latest(self):
        """When head is fully dropped, tail is trimmed from oldest but latest block survives."""
        server = _make_server()
        # No head — only system + many huge tail blocks
        messages = [{"role": "system", "content": "S" * 100_000}]  # ~100K system
        for i in range(25):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Block_{i:02d}_" * 50_000})  # ~350K each

        result = server._compact_messages(messages.copy())
        result_text = json.dumps(result, ensure_ascii=False)

        # System survives
        assert result[0]["role"] == "system"
        assert len(result[0]["content"]) == 100_000

        # Latest block (Block_24) must survive
        assert "Block_24" in result_text

        # Result must be under guaranteed max
        assert _char_size(result) <= _COMPACTION_GUARANTEED_MESSAGES_MAX

    def test_guaranteed_fit_preserves_system_message(self):
        """System message must survive aggressive shrinking even when very large."""
        server = _make_server()
        huge_system = "SYS" * 500_000  # ~1.5M chars
        messages = [{"role": "system", "content": huge_system}]
        for i in range(15):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Msg{i} " * 50_000})

        result = server._compact_messages(messages.copy())

        # System message must be first and unchanged
        assert result[0]["role"] == "system"
        assert result[0]["content"] == huge_system

    def test_guaranteed_fit_preserves_atomic_blocks(self):
        """Tool-call/result pairs must not be split during fallback shrinking."""
        server = _make_server()
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Do task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "T" * 200_000, "tool_call_id": "call_1"},
            {"role": "assistant", "content": "A" * 200_000},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "write", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "R" * 200_000, "tool_call_id": "call_2"},
            {"role": "assistant", "content": "Final answer"},
        ]

        result = server._compact_messages(messages.copy())

        # If a tool_call_id appears, its assistant tool_call must be adjacent
        tool_results = [(i, m) for i, m in enumerate(result) if m.get("role") == "tool"]
        for idx, tr in tool_results:
            assert idx > 0, f"Tool result at index {idx} has no preceding assistant"
            prev = result[idx - 1]
            assert prev["role"] == "assistant"
            assert prev.get("tool_calls") is not None
            call_ids = {tc["id"] for tc in prev["tool_calls"]}
            assert tr["tool_call_id"] in call_ids


class TestRequestSizeAccountingForTools:
    """Regression tests for the full-request size check including tools.

    The bridge's compaction only shrinks messages, but _check_request_size
    checks the ENTIRE request (tools + system + model + metadata + messages).
    If the tools array is large, the request can exceed _MAX_REQUEST_CHARS
    even after messages are compacted.  This test class validates the fix.
    """

    def test_check_request_size_rejects_when_tools_push_over_limit(self):
        """A request with compacted messages but large tools must still be rejected."""
        server = _make_server()
        # Messages are small (well under threshold)
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        # But tools are enormous — push the whole request over _MAX_REQUEST_CHARS
        large_tools = [
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": f"Tool {i}: " + "x" * 100_000,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for i in range(50)
        ]
        cc_request = {
            "model": "test-model",
            "messages": messages,
            "tools": large_tools,
            "stream": True,
        }
        # Sanity: the full request should exceed the limit
        full_size = len(json.dumps(cc_request, ensure_ascii=False))
        assert full_size > _MAX_REQUEST_CHARS

        result = server._check_request_size(cc_request)
        assert result is not None, "Expected _check_request_size to reject the oversized request"
        assert result.status == 400

    def test_compaction_does_not_shrink_tools(self):
        """_apply_compaction only touches messages — tools are never trimmed."""
        server = _make_server()
        # Build a request where messages are small but tools are huge
        original_tools = [
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": "x" * 50_000,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for i in range(20)
        ]
        tools_size_before = len(json.dumps(original_tools, ensure_ascii=False))
        cc_request = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "A" * 3_000_000},
                {"role": "assistant", "content": "B" * 3_000_000},
            ],
            "tools": original_tools,
        }
        server._apply_compaction(cc_request)

        # Tools should be unchanged
        assert cc_request["tools"] == original_tools
        tools_size_after = len(json.dumps(cc_request["tools"], ensure_ascii=False))
        assert tools_size_before == tools_size_after

    def test_full_request_fits_after_compaction_with_tools(self):
        """The happy path: messages are compacted, tools are preserved, whole request fits."""
        server = _make_server()
        # Large messages that trigger compaction
        messages = [{"role": "system", "content": "System"}]
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Msg{i} " * 50_000})

        # Moderate tools — together with compacted messages should fit
        tools = [
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": f"Tool {i} description",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for i in range(10)
        ]
        cc_request = {
            "model": "test-model",
            "messages": messages,
            "tools": tools,
            "stream": True,
        }

        server._apply_compaction(cc_request)
        result = server._check_request_size(cc_request)
        assert result is None, (
            f"Expected request to fit after compaction. "
            f"Full size: {len(json.dumps(cc_request, ensure_ascii=False))} chars, "
            f"limit: {_MAX_REQUEST_CHARS}"
        )

    def test_request_rejected_when_tools_plus_compacted_messages_exceed_limit(self):
        """BUG: compacted messages fit, but tools push the full request over the limit.

        This reproduces the /compact failure scenario: Claude Code's compaction
        produces a request where messages are small enough, but the tools array
        (which is never compacted) causes the full request to exceed the limit.
        The bridge must handle this gracefully instead of hard-rejecting.
        """
        server = _make_server()
        # Build messages that are just barely under the compaction threshold
        # so compaction still kicks in but doesn't shrink much
        messages = [{"role": "system", "content": "System prompt"}]
        for i in range(15):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Msg{i} " * 100_000})

        # Tools that are large but not enormous — together with compacted
        # messages they push the full request over _MAX_REQUEST_CHARS
        tools = [
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": "A tool that does something useful. " + "x" * 20_000,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for i in range(40)  # 40 tools × ~20K each ≈ 800K total
        ]

        cc_request = {
            "model": "test-model",
            "messages": messages,
            "tools": tools,
            "stream": True,
        }

        # Before compaction: the full request would exceed the limit.
        # Verify this by checking the raw messages size.
        raw_msg_size = len(json.dumps(cc_request["messages"], ensure_ascii=False))
        assert raw_msg_size > _COMPACTION_GUARANTEED_MESSAGES_MAX, (
            f"Raw messages should exceed guaranteed max: got {raw_msg_size:,}"
        )

        # Apply compaction — now accounts for the non-message payload (tools, model)
        server._apply_compaction(cc_request)

        full_size = len(json.dumps(cc_request, ensure_ascii=False))
        # After tools-aware compaction, the full request should fit.
        result = server._check_request_size(cc_request)
        assert result is None, (
            f"Full request should fit after tools-aware compaction. "
            f"Full size: {full_size:,}, limit: {_MAX_REQUEST_CHARS:,}"
        )


# ---------------------------------------------------------------------------
# _get_max_context_chars — context-aware char budget
# ---------------------------------------------------------------------------


class TestGetMaxContextChars:
    """_get_max_context_chars derives the char budget from model metadata."""

    def test_no_model_returns_absolute_cap(self):
        """When no model is configured, fall back to _MAX_REQUEST_CHARS."""
        server = _make_server()
        assert server._get_max_context_chars() == _MAX_REQUEST_CHARS

    def test_known_model_returns_context_derived_chars(self):
        """When a model with known context is set, return tokens_to_chars(context)."""
        server = _make_server()
        server._active_provider = StubProvider()
        server._active_model = "openai/gpt-4o"
        server._active_provider_config = {}
        server._backends = None
        # gpt-4o has 128k context in metadata → 128000 * 4 = 512000
        assert server._get_max_context_chars() == 512_000

    def test_unknown_model_returns_default_chars(self):
        """When the model is set but metadata is missing, use DEFAULT_CONTEXT_TOKENS."""
        server = _make_server()
        server._active_provider = StubProvider()
        server._active_model = "nonexistent-model-xyz"
        server._active_provider_config = {}
        server._backends = None
        expected = tokens_to_chars(DEFAULT_CONTEXT_TOKENS)  # 800k
        assert server._get_max_context_chars() == expected

    def test_balancing_uses_min_context(self):
        """For balancing profiles, use the smallest context across all backends."""
        server = _make_server()
        # Simulate balancing with 2 backends: gpt-4o (128k) and deepseek-chat (163_840)
        p1 = StubProvider()
        profile1 = Profile(
            name="p1",
            provider="openrouter",
            model="openai/gpt-4o",
            auth_ref="dd7f361b-3794-4343-a917-906760d3cde4",
        )
        p2 = StubProvider()
        profile2 = Profile(
            name="p2",
            provider="openrouter",
            model="deepseek/deepseek-chat",
            auth_ref="76df9193-5b84-4824-8c35-a1dce9c01d64",
        )
        server._backends = [
            (p1, "key1", profile1),
            (p2, "key2", profile2),
        ]
        # Min: gpt-4o 128000 → 128000 * 4 = 512000
        assert server._get_max_context_chars() == 512_000

    def test_provider_config_override(self):
        """Manual context_window in provider_config overrides metadata."""
        server = _make_server()
        server._active_provider = StubProvider()
        server._active_model = "openai/gpt-4o"
        server._active_provider_config = {"context_window": 50_000}
        server._backends = None
        # Override: 50000 * 4 = 200000
        assert server._get_max_context_chars() == 200_000

    def test_huge_context_capped_at_max_request_chars(self):
        """Models with very large context are capped at _MAX_REQUEST_CHARS."""
        server = _make_server()
        server._active_provider = StubProvider()
        server._active_model = "google/gemini-2.5-flash"
        server._active_provider_config = {}
        server._backends = None
        # gemini has 1M context → ~4.2M chars, capped at _MAX_REQUEST_CHARS (4M)
        assert server._get_max_context_chars() == _MAX_REQUEST_CHARS


# ---------------------------------------------------------------------------
# Context-aware compaction and size checking
# ---------------------------------------------------------------------------


class TestContextAwareCompaction:
    """Compaction and request-size checks use model-derived char limits."""

    def test_compaction_uses_model_context(self):
        """Compaction threshold is derived from model context, not hardcoded."""
        server = _make_server()
        server._active_provider = StubProvider()
        server._active_model = "openai/gpt-4o"  # 128k → 512k chars
        server._active_provider_config = {}
        server._backends = None

        # Create messages that exceed 70% of 512k (~358k) but are under 70% of 4M (~2.8M)
        messages = [{"role": "system", "content": "System"}]
        for i in range(30):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i} " * 3000})
        raw_size = len(json.dumps(messages, ensure_ascii=False))
        assert raw_size > 358_000, f"Need >358k chars, got {raw_size:,}"

        cc_request = {"model": "gpt-4o", "messages": messages}
        server._apply_compaction(cc_request)
        result_size = len(json.dumps(cc_request["messages"], ensure_ascii=False))
        # After compaction, messages should be smaller
        assert result_size < raw_size, "Compaction should have reduced messages"

    def test_size_check_uses_model_context(self):
        """_check_request_size rejects based on model-derived limit, not hardcoded."""
        server = _make_server()
        server._active_provider = StubProvider()
        server._active_model = "openai/gpt-4o"  # 128k → 512k chars
        server._active_provider_config = {}
        server._backends = None

        # Create a request that exceeds 512k but is under 4M
        large_content = "A" * 600_000
        cc_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": large_content}],
        }
        full_size = len(json.dumps(cc_request, ensure_ascii=False))
        assert full_size > 512_000, f"Need >512k chars, got {full_size:,}"
        assert full_size < _MAX_REQUEST_CHARS, f"Should be under 4M, got {full_size:,}"

        result = server._check_request_size(cc_request)
        assert result is not None, "Should reject request exceeding model context"
        assert result.status == 400


class NativePassthroughProvider(StubProvider):
    """Stub provider that simulates a provider forwarding raw Anthropic Messages."""

    @property
    def use_native_messages(self) -> bool:
        return True


class TestNativePassthroughCompactionWarning:
    """Safety-net warning when a provider using ``use_native_messages=True``
    sends a request large enough to trigger compaction.

    The compaction grouping in ``_compact_messages`` is Chat-Completions-format
    aware. When the active provider forwards the raw Anthropic Messages body,
    ``tool_use`` is a content block on the assistant message and the
    following message is ``user`` with a ``tool_result`` block. The grouping
    misses the pairing and the pruner can drop the ``assistant(tool_use)``
    while keeping the following ``user(tool_result)``. MiniMax then returns
    ``invalid params, tool call result does not follow tool call (2013)``.

    The bridge logs a warning in this case so operators see a signal if a
    future provider re-introduces this combination.
    """

    def _make_server_with_native_provider(self) -> BridgeServer:
        server = _make_server()
        server._active_provider = NativePassthroughProvider()
        server._active_model = None
        server._active_provider_config = {}
        server._backends = None
        return server

    def test_warning_logged_for_large_native_request(self, caplog):
        """A request above the compaction threshold with use_native_messages=True
        triggers a warning that names the provider class and the size."""
        import logging

        server = self._make_server_with_native_provider()
        large_content = "A" * 3_000_000  # > 70% of _MAX_REQUEST_CHARS (4M * 0.7 = 2.8M)
        cc_request = {
            "model": "m",
            "messages": [{"role": "user", "content": large_content}],
        }
        with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
            server._apply_compaction(cc_request)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("Native passthrough provider" in r.getMessage() for r in warnings), (
            f"Expected a 'Native passthrough provider' warning, got: {[r.getMessage() for r in warnings]}"
        )

    def test_no_warning_for_small_native_request(self, caplog):
        """A request below the compaction threshold with use_native_messages=True
        does not trigger the warning."""
        import logging

        server = self._make_server_with_native_provider()
        cc_request = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
            server._apply_compaction(cc_request)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("Native passthrough provider" in r.getMessage() for r in warnings)

    def test_no_warning_for_large_translated_request(self, caplog):
        """A large request on the default (translated) path does not trigger
        the warning — the translator produces a CC messages list whose
        pairing is recognized by the compaction grouping."""
        import logging

        server = _make_server()  # default provider: use_native_messages=False
        large_content = "A" * 3_000_000
        cc_request = {
            "model": "m",
            "messages": [{"role": "user", "content": large_content}],
        }
        with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
            server._apply_compaction(cc_request)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("Native passthrough provider" in r.getMessage() for r in warnings)
