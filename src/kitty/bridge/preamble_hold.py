"""Preamble hold for the native Anthropic Messages passthrough (KBR-155).

The native passthrough forwards upstream SSE bytes to the client unmodified.
Before this module existed it forwarded them the instant they arrived, so an
upstream reply with no content reached Claude Code as a blank turn before
anything could judge it — and once a byte has reached the client the bridge
neither retries nor fails over (``.system_design/TEST_SUITE.md`` §11 Q14(a)).

:class:`PreambleHold` keeps an empty reply out of that situation instead of
recovering from inside it (Q14(b)). It withholds the stream's leading bytes
until the stream proves it carries content, then hands every held byte back
**verbatim** — never re-serialised, because §4.3 C2 holds the passthrough to
byte identity — and passes the rest through untouched. A stream that ends
while still held has written nothing, so the caller can still retry it.

The release rule, with the reasons recorded in Q14(b) and its amendments:

- a ``content_block_delta`` on a block that is not thinking, unless it is a
  ``text_delta`` with empty text (D6: a blank text reply is still empty);
- a ``content_block_start`` whose block already carries content — any type
  other than ``text``, ``thinking`` and ``redacted_thinking``, or a ``text``
  block started with non-empty text (D1: such blocks may send no delta);
- an ``error`` event, which is the provider's to deliver (D2) — recognised by
  its SSE ``event:`` name as well as its ``data.type``, because the official
  SDK raises on the name alone;
- more than :data:`MAX_HELD_BYTES` held, where the hold fails open (D5).
"""

from __future__ import annotations

import json

__all__ = ["MAX_HELD_BYTES", "PreambleHold"]

# D5: the affected adapters serve reasoning models, so the hold can span a
# whole thinking phase. Past this bound it releases rather than grow.
MAX_HELD_BYTES = 10 * 1024 * 1024

# Block types that are thinking, so their start and deltas release nothing.
_THINKING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})

# Bound on remembered thinking-block indices, as the auditor bounds its open blocks: a
# hostile stream of distinct thinking starts must not grow the set without limit.
_MAX_THINKING_BLOCKS = 256

# Delta types that belong to a thinking block even without a recorded start.
_THINKING_DELTA_TYPES = frozenset({"thinking_delta", "signature_delta"})


class PreambleHold:
    """Withhold a native Messages stream's leading bytes until content arrives.

    One instance judges one upstream attempt; a retry needs a fresh one, since
    a held partial line or a thinking index must not leak into the next
    attempt. The judgement reads only ``data:`` lines and never alters the
    bytes it returns.

    Attributes:
        stop_reason: The last ``stop_reason`` a ``message_delta`` carried while
            the stream was held, or ``None``. After the stream ends unreleased,
            ``"max_tokens"`` or ``"model_context_window_exceeded"`` means the
            reply was truncated before any content.
    """

    def __init__(self, *, max_held_bytes: int = MAX_HELD_BYTES) -> None:
        """Initialise a hold for one upstream attempt.

        Args:
            max_held_bytes: Bound on withheld bytes; exceeding it releases. The
                bound is checked per upstream chunk, so it can be overshot by
                at most the chunk that crossed it.
        """
        self._max_held_bytes = max_held_bytes
        self._held = bytearray()
        # Offset in `_held` of the first byte not yet scanned for a newline.
        self._scanned = 0
        self._thinking_indices: set[int] = set()
        # The current event's SSE `event:` name, which stands in for a missing `data.type`.
        self._event_name: str | None = None
        self._released = False
        self.stop_reason: str | None = None

    @property
    def released(self) -> bool:
        """bool: Whether the hold has released and bytes now pass straight through."""
        return self._released

    @property
    def held_size(self) -> int:
        """int: How many bytes are withheld; 0 once released."""
        return len(self._held)

    def head(self, limit: int) -> bytes:
        """Return the first withheld bytes without copying the whole buffer.

        Args:
            limit: The most bytes to return.

        Returns:
            Up to ``limit`` withheld bytes; empty once released.
        """
        return bytes(self._held[:limit])

    def feed(self, chunk: bytes) -> bytes:
        """Consume one upstream chunk and return the bytes to write to the client.

        Args:
            chunk: Raw bytes exactly as received from upstream.

        Returns:
            ``b""`` while holding; on release, every byte held so far including
            this chunk; after release, ``chunk`` itself.
        """
        if self._released:
            return chunk
        self._held.extend(chunk)

        # Judge only complete lines: a trigger split across chunks counts once whole.
        while (newline := self._held.find(b"\n", self._scanned)) != -1:
            line = bytes(self._held[self._scanned : newline])
            self._scanned = newline + 1
            if self._is_release_line(line):
                return self._release()

        # D5: bound the hold rather than grow it without limit.
        if len(self._held) > self._max_held_bytes:
            return self._release()
        return b""

    def _release(self) -> bytes:
        """Stop holding and hand back every withheld byte, in arrival order.

        Returns:
            The withheld bytes, verbatim.
        """
        self._released = True
        out = bytes(self._held)
        self._held = bytearray()
        return out

    def _is_release_line(self, line: bytes) -> bool:
        """Decide whether one complete SSE line ends the hold.

        Args:
            line: One line without its newline; a trailing ``\\r`` is tolerated.

        Returns:
            ``True`` when the line is an event the release rule counts.
        """
        if line.startswith(b"event:"):
            self._event_name = line[6:].strip().decode("utf-8", errors="replace")
            # D2: the SDK dispatches and raises on `event: error` even with no data line at all.
            return self._event_name == "error"
        if not line.strip():
            self._event_name = None
            return False
        if not line.startswith(b"data:"):
            return False
        try:
            event = json.loads(line[5:].strip().decode("utf-8", errors="replace"))
        except (ValueError, RecursionError):
            # Untrusted bytes: a 4300-digit integer or deep nesting must not escape as an error.
            return False
        if not isinstance(event, dict):
            return False

        event_type = event.get("type", self._event_name)
        if event_type == "content_block_start":
            return self._block_start_releases(event)
        if event_type == "content_block_delta":
            index = event.get("index")
            delta = event.get("delta")
            if not isinstance(delta, dict):
                delta = {}
            delta_type = delta.get("type")
            thinking = isinstance(index, int) and index in self._thinking_indices
            if thinking or delta_type in _THINKING_DELTA_TYPES:
                return False
            # D6: an empty text chunk is no more content than an empty text block start.
            return not (delta_type == "text_delta" and delta.get("text") == "")
        if event_type == "message_delta":
            delta = event.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("stop_reason"), str):
                self.stop_reason = delta["stop_reason"]
            return False
        return event_type == "error"

    def _block_start_releases(self, event: dict) -> bool:
        """Judge a ``content_block_start``, remembering thinking blocks by index.

        Args:
            event: The parsed ``content_block_start`` payload.

        Returns:
            ``True`` when the block arrives already carrying content (D1).
        """
        block = event.get("content_block")
        if not isinstance(block, dict):
            return False
        block_type = block.get("type")
        if block_type in _THINKING_BLOCK_TYPES:
            index = event.get("index")
            if isinstance(index, int):
                self._thinking_indices.add(index)
            # Fail open, as the byte cap does (D5), rather than track an unbounded set.
            return len(self._thinking_indices) > _MAX_THINKING_BLOCKS
        # D6's twin: a text block that starts empty is not content either.
        if block_type == "text":
            text = block.get("text")
            return isinstance(text, str) and text != ""
        return True
