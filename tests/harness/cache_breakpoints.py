"""Anthropic request bodies carrying one prompt-cache breakpoint at a named site.

`.system_design/TEST_SUITE.md` §3.4 · **KBR-198** (CB-1, epic KBR-197).

Claude Code marks stable prompt prefixes with ``cache_control`` breakpoints, and
Anthropic bills a cache read at 0.1x base input. A breakpoint that dies in transit
re-bills the prefix at roughly ten times that on every turn, silently. This module
supplies the inputs and the detector for tests that pin where breakpoints die.

**One breakpoint per body, never all sites at once.** Anthropic rejects a request
with more than four breakpoints (HTTP 400), so only a per-site body can be reused by
wire-level tests (T-D1, T-J2). Every site shares one conversation and only the
breakpoint's position changes, so two tests' outcomes can differ only by site.

**The prefix is sized to be cacheable.** Below a per-model minimum (512 to 4,096
tokens) Anthropic skips caching *and returns no error*, which would make a wire test
vacuous. A breakpoint caches everything up to *and including* its block, and in
Anthropic's ``tools -> system -> messages`` order the tool definition comes first,
so the tool description is inside every site's prefix (at the ``tool`` site it is
the marked block itself). It holds at least :data:`MIN_PREFIX_WORDS` words. That is
a stand-in, not a token count: it assumes ordinary English prose is at least one
token per word and doubles the largest minimum for margin. The real proof is
``usage.cache_creation_input_tokens > 0`` on the wire.

**The detector matches by key name and by value.** :func:`find_breakpoints` finds any
key containing ``cache_control`` (so a carry-through on a private ``_cache_control``
key is seen) and any value equal to :data:`BREAKPOINT` under any key. A fix that
re-encodes the value under an unrelated key is not seen.

**Well-formed is partly the fixture's own shape.** :func:`request_problems` checks
rules Anthropic enforces (required fields, answered tool calls, the breakpoint limit)
alongside rules that only keep this fixture honest: Anthropic merges consecutive
same-role turns rather than rejecting them, so "roles alternate" is a fixture rule.

**It imports nothing from ``src/kitty``**, so the bodies cannot share the
translator's assumptions (§3.3.1's independence rule).
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

__all__ = ["BREAKPOINT", "MIN_PREFIX_WORDS", "SITES", "build_request", "find_breakpoints", "request_problems"]

#: The one-hour breakpoint. A one-hour cache write bills at 2x base input and a
#: five-minute one at 1.25x, so tests assert this *value*: a presence check would
#: pass a carry-through that silently downgraded the TTL. Never mutated: every
#: body gets its own copy.
BREAKPOINT: dict[str, str] = {"type": "ephemeral", "ttl": "1h"}

#: Twice the largest per-model minimum cacheable prefix Anthropic lists (4,096 tokens).
MIN_PREFIX_WORDS = 2 * 4096

#: Anthropic allows at most four breakpoints per request, the top-level one included.
_MAX_BREAKPOINTS = 4

_MODEL = "claude-sonnet-5"
_TOOL_NAME = "read_file"
_TOOL_USE_ID = "toolu_01CacheBreakpointFixture"

# A real 1x1 RGBA PNG: its chunk CRCs were verified by hand, so the API can decode it.
_PNG_1X1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"

_PADDING_SENTENCES = (
    "Open the file at the given path and return its contents as text.",
    "Paths are resolved against the workspace root and may not escape it.",
    "Binary files are refused with a short message naming the detected type.",
    "Large files are returned in pages so a single call never floods the context.",
    "Line numbers start at one and are included when the caller asks for them.",
    "Symbolic links are followed only when their target stays inside the workspace.",
)


def _padding() -> str:
    """Build the tool description that makes every breakpoint's prefix cacheable.

    Cycles through varied sentences, each numbered, until the text holds at least
    :data:`MIN_PREFIX_WORDS` whitespace-separated words.

    Returns:
        The padding text.
    """
    sentences: list[str] = []
    words = 0
    while words < MIN_PREFIX_WORDS:
        sentence = f"Rule {len(sentences) + 1}. {_PADDING_SENTENCES[len(sentences) % len(_PADDING_SENTENCES)]}"
        sentences.append(sentence)
        words += len(sentence.split())
    return " ".join(sentences)


_PADDING = _padding()

# Where each site's breakpoint goes: dict keys and list indexes into the skeleton.
_SITE_PATHS: dict[str, tuple[str | int, ...]] = {
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

#: Every site Anthropic permits a breakpoint at, plus the block nested inside a
#: ``tool_result``'s content, which the SDK schema accepts.
SITES: tuple[str, ...] = tuple(_SITE_PATHS)


def _skeleton() -> dict[str, Any]:
    """Build the shared conversation, without any breakpoint.

    Returns:
        A fresh Anthropic Messages request body that no other caller holds.
    """
    return {
        "model": _MODEL,
        "max_tokens": 1024,
        "tools": [
            {
                "name": _TOOL_NAME,
                "description": _PADDING,
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            }
        ],
        "system": [{"type": "text", "text": "You are a careful coding assistant."}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "Project notes: keep it simple.",
                        },
                    },
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _PNG_1X1}},
                    {"type": "text", "text": "Please read README.md and summarise it."},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Reading the file now."},
                    {"type": "tool_use", "id": _TOOL_USE_ID, "name": _TOOL_NAME, "input": {"path": "README.md"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": _TOOL_USE_ID,
                        "content": [{"type": "text", "text": "# Kitty Bridge\nRoutes agent traffic."}],
                    }
                ],
            },
        ],
    }


def build_request(site: str) -> dict[str, Any]:
    """Build a request carrying exactly one :data:`BREAKPOINT`, at ``site``.

    Args:
        site: One of :data:`SITES`.

    Returns:
        A fresh, JSON-serialisable Anthropic Messages request body.

    Raises:
        ValueError: When ``site`` is not one of :data:`SITES`.
    """
    if site not in _SITE_PATHS:
        raise ValueError(f"unknown breakpoint site {site!r}; expected one of {SITES}")
    body = _skeleton()
    carrier: Any = body
    for step in _SITE_PATHS[site]:
        carrier = carrier[step]
    carrier["cache_control"] = dict(BREAKPOINT)
    return body


def find_breakpoints(node: Any) -> list[Any]:
    """Find every breakpoint anywhere in a body, in document order.

    A breakpoint is the value of any key whose name contains ``cache_control``, or
    any value equal to :data:`BREAKPOINT`, wherever it sits. A matched value is not
    searched further, so each breakpoint is counted once.

    Args:
        node: A JSON-shaped structure of dicts, lists and scalars.

    Returns:
        The breakpoint values found; empty when there are none.
    """
    found: list[Any] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if (isinstance(key, str) and "cache_control" in key) or value == BREAKPOINT:
                found.append(value)
            else:
                found.extend(find_breakpoints(value))
    elif isinstance(node, list):
        for item in node:
            if item == BREAKPOINT:
                found.append(item)
            else:
                found.extend(find_breakpoints(item))
    return found


def _content_blocks(body: dict[str, Any]) -> list[dict[str, Any]]:
    """List the system blocks, message blocks and blocks nested in tool results.

    Args:
        body: An Anthropic Messages request body.

    Returns:
        Every content block, outermost first.
    """
    blocks = [b for b in body.get("system", []) if isinstance(b, dict)]
    for message in body.get("messages", []):
        content = message.get("content")
        for block in content if isinstance(content, list) else []:
            blocks.append(block)
            nested = block.get("content")
            if block.get("type") == "tool_result" and isinstance(nested, list):
                blocks.extend(nested)
    return blocks


def _png_problem(data: str) -> str | None:
    """Say what is wrong with base64 image data that should be a complete PNG.

    Args:
        data: The block's base64 ``source.data``.

    Returns:
        A problem message, or ``None`` when the data is a complete PNG.
    """
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return "image data is not valid base64"
    # The fixed IEND trailer is what a truncated image loses, and the signature alone keeps.
    if not (raw.startswith(_PNG_SIGNATURE) and raw.endswith(_PNG_IEND)):
        return "image data is not a complete PNG"
    return None


def request_problems(body: dict[str, Any]) -> list[str]:
    """Report every way a body breaks the rules this fixture promises to keep.

    A pure reporter: on a body shaped like a Messages request it raises nothing, and
    an empty list means well-formed.

    Args:
        body: An Anthropic Messages request body.

    Returns:
        One message per violation.
    """
    problems = [f"missing required field {key!r}" for key in ("model", "max_tokens") if key not in body]

    # Conversation shape: Anthropic needs a user turn first and last; alternation is the fixture's own rule.
    messages = body.get("messages", [])
    roles = [message.get("role") for message in messages]
    if not roles or roles[0] != "user" or roles[-1] != "user":
        problems.append(f"conversation must start and end on a user turn, got roles {roles}")
    problems.extend(
        f"consecutive {a!r} turns at messages[{i}]"
        for i, (a, b) in enumerate(zip(roles, roles[1:], strict=False))
        if a == b
    )

    # Every tool call is answered in the very next turn, and names a tool the request declares.
    declared = {tool.get("name") for tool in body.get("tools", [])}
    for i, message in enumerate(messages):
        content = message.get("content")
        blocks = content if isinstance(content, list) else []
        following = messages[i + 1].get("content") if i + 1 < len(messages) else None
        answered = (
            {b.get("tool_use_id") for b in following if isinstance(b, dict)} if isinstance(following, list) else set()
        )
        for block in blocks:
            if block.get("type") != "tool_use":
                continue
            if block.get("id") not in answered:
                problems.append(f"unanswered tool_use {block.get('id')!r} at messages[{i}]")
            if block.get("name") not in declared:
                problems.append(f"tool_use calls undeclared tool {block.get('name')!r}")

    for block in _content_blocks(body):
        if block.get("type") == "text" and not block.get("text"):
            problems.append("empty text block, which Anthropic cannot cache")
        if block.get("type") == "image":
            png_problem = _png_problem(str(block.get("source", {}).get("data", "")))
            if png_problem:
                problems.append(png_problem)

    breakpoints = len(find_breakpoints(body))
    if breakpoints > _MAX_BREAKPOINTS:
        problems.append(f"{breakpoints} breakpoints, above Anthropic's limit of {_MAX_BREAKPOINTS}")

    words = sum(len(str(tool.get("description", "")).split()) for tool in body.get("tools", []))
    if words < MIN_PREFIX_WORDS:
        problems.append(f"tool descriptions hold {words} words, below the cacheable-prefix floor of {MIN_PREFIX_WORDS}")

    return problems
