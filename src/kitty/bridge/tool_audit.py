"""Response-side ``tool_use`` auditing for the Messages API bridge.

Makes visible a defect class the bridge previously forwarded in silence: an
upstream returns a ``tool_use`` block whose ``input`` is the wrong shape, the
client's own validator rejects it, and nothing in kitty's log records what was
forwarded (kitty-bridge#33).  Response payloads were logged only on the custom
transport, so on every other transport the operator could not tell a bridge
translation bug from an upstream serialization bug from a client-side schema
error.

Two pieces:

- :func:`describe_tool_input_anomaly` — a pure detector that compares a
  returned ``input`` against the tool's **own declared schema** and describes
  the mismatch.  It names no cause: it reports shape, so it stays correct
  whatever the underlying reason turns out to be.
- :class:`ToolUseAuditor` — assembles ``tool_use`` inputs from the Anthropic
  SSE the bridge writes to the client, and reports each one.

Every log line this module emits is prefixed :data:`AUDIT_MARKER`, so an
incident report can be answered with a single ``grep`` — the evidence in #33
was literally a grep that returned zero hits.

Two invariants hold throughout.  **Auditing never alters the response**: it
observes the bytes being written, rewrites nothing, and cannot influence
backend health or routing.  **The WARNING carries key names only, never
values**: it is the one line here that can reach a non-debug handler if an
embedder configures logging, and tool inputs routinely contain file contents
and user data.  Full inputs stay at DEBUG, which only ever reaches the opt-in
debug file.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

__all__ = [
    "AUDIT_MARKER",
    "MAX_LOGGED_INPUT_CHARS",
    "ToolUseAuditor",
    "collect_tool_schemas",
    "describe_tool_input_anomaly",
    "report_tool_use",
]

logger = logging.getLogger(__name__)

# Fixed, greppable prefix on every line this module emits.
AUDIT_MARKER = "tool_use audit:"

# Bound on the JSON rendering written to the DEBUG log.  A structured-output
# payload can be large, and a debug log that is mostly one tool input is
# unreadable.
MAX_LOGGED_INPUT_CHARS = 2000

# Bound on accumulated ``input_json_delta`` text per tool call.
_MAX_BUFFERED_ARG_CHARS = 1_000_000

# Bound on simultaneously open tool_use blocks.  Per-block argument bounds do
# not bound the total: an upstream emitting `content_block_start` without
# matching stops would otherwise retain one entry per index for the whole
# response.  The native path held no per-stream state before this auditor, so
# every dimension of it has to be capped.
_MAX_OPEN_BLOCKS = 256

# Bound on the auditor's own line buffer.  The native path has no line buffer
# today, so this one is introduced here; an upstream that never sends a newline
# must not be able to grow it without limit.
_MAX_LINE_BYTES = 10 * 1024 * 1024

# Schema keywords whose presence means the top-level ``properties``/``required``
# are not the authoritative constraint.  Composition and ``$ref`` put the real
# schema somewhere this detector does not resolve — and in draft-07, which the
# Claude Code SDK validates against, the siblings of ``$ref`` are ignored
# entirely.  Judging such a schema from its root keys would call valid input
# malformed, so the detector stands down instead.
_COMPOSITION_KEYWORDS = ("oneOf", "anyOf", "allOf", "not", "if", "$ref")


def collect_tool_schemas(messages_request: dict) -> dict[str, dict]:
    """Map each declared tool name to its input schema.

    The client tells the bridge exactly what shape it expects back, in the
    ``tools`` array of its own request.  That declaration is the only ground
    truth available here, and it is what makes the detector schema-driven
    rather than provider-specific.

    Args:
        messages_request: The Anthropic Messages request body from the client.

    Returns:
        Tool name to ``input_schema``. Empty when the request declares no
        tools, which disables anomaly detection while leaving logging active.
    """
    tools = messages_request.get("tools")
    if not isinstance(tools, list):
        return {}

    schemas: dict[str, dict] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        schema = tool.get("input_schema")
        if isinstance(name, str) and isinstance(schema, dict):
            schemas[name] = schema
    return schemas


def _summarize_keys(keys: list[str], limit: int = 10) -> str:
    """Render a bounded key list for the WARNING line.

    The DEBUG rendering is capped by :data:`MAX_LOGGED_INPUT_CHARS`, but the
    WARNING is the line that can reach an embedder's handler, and its key lists
    come straight from upstream bytes — a path-keyed input can carry thousands.
    Bounding here keeps the one escaping line small.

    Args:
        keys: Key names to render.
        limit: How many to show before summarising the remainder.

    Returns:
        A bracketed list, with a ``(+N more)`` suffix when truncated.
    """
    if len(keys) <= limit:
        return str(keys)
    return f"{keys[:limit]} (+{len(keys) - limit} more)"


def _looks_nested(tool_input: dict, missing: list[str]) -> bool:
    """Return True when the missing properties all sit one level down.

    The specific fingerprint of a payload wrapped one level too deep: a single
    root key whose value is an object containing every property the schema
    required and the root lacks.  Far more precise than "one unexpected key",
    and it still fires when the wrapper name collides with a declared property
    — the documented wrapper key rotates, and one of its values is the tool's
    own name.

    Args:
        tool_input: The returned input object.
        missing: Required property names absent from the root.

    Returns:
        True when every missing name is present inside the single root value.
    """
    if len(tool_input) != 1 or not missing:
        return False
    inner = next(iter(tool_input.values()))
    return isinstance(inner, dict) and all(name in inner for name in missing)


def describe_tool_input_anomaly(name: str, tool_input: object, schema: dict | None) -> str | None:
    """Describe how a returned tool input contradicts its declared schema.

    Reports on either of two findings:

    - **both** at least one root key the schema does not declare **and** at
      least one required property missing.  Either alone is ordinary traffic —
      a model omitting a field is a commoner failure that belongs to the
      client's validator, and a harmless extra key is not evidence of anything
      — so requiring both is what keeps this quiet enough to be worth reading;
    - the nesting fingerprint of :func:`_looks_nested`, which catches a wrapped
      payload even when the wrapper key collides with a declared property and
      the first rule therefore cannot fire.

    A root ``required`` name absent from the input is proof the input violates
    the declared schema, unconditionally and regardless of
    ``additionalProperties``.  That is what makes the rule precise rather than
    merely conservative, and why it must not later be relaxed into an "or".

    Deliberately describes shape and names no cause.  The reported incident is
    *suspected* to come from an upstream wrapping tool arguments one level too
    deep, but that was unconfirmed when this was written, and a detector that
    asserted a cause would be wrong the moment the suspicion was.

    Args:
        name: The tool's name, for the description.
        tool_input: The ``input`` value returned by the upstream.
        schema: The tool's declared ``input_schema``, or None when the client
            never declared this tool.

    Returns:
        A description of the mismatch naming **key names only, never values**,
        or None when the input is consistent with the schema, when there is
        nothing to check against, or when the schema is composed and its
        constraints cannot be read directly.
    """
    if not isinstance(schema, dict) or not isinstance(tool_input, dict):
        return None

    # A composed or referencing schema keeps its real constraints somewhere
    # this detector does not resolve; every valid key would look unexpected.
    if any(keyword in schema for keyword in _COMPOSITION_KEYWORDS):
        return None

    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not properties:
        return None
    if not isinstance(required, list) or not required:
        return None

    unexpected = sorted(key for key in tool_input if key not in properties)
    # `required` comes from the client's request, so a malformed entry (an
    # unhashable dict, say) must not raise: containment would disable auditing
    # for the whole session, and the same schema recurs every turn.
    missing = sorted(key for key in required if isinstance(key, str) and key not in tool_input)
    nested = _looks_nested(tool_input, missing)
    if not missing or (not unexpected and not nested):
        return None

    if unexpected:
        description = (
            f"tool_use {name!r} input has unexpected root key(s) {_summarize_keys(unexpected)} "
            f"and is missing required propert(ies) {_summarize_keys(missing)}"
        )
    else:
        # Only the nesting rule fired: the wrapper key is itself a declared
        # property, so reporting an empty "unexpected" list would read as a bug.
        description = f"tool_use {name!r} input is missing required propert(ies) {_summarize_keys(missing)}"
    if nested:
        description += " — every missing property is present one level down, so the payload looks wrapped (envelope)"
    return description


def report_tool_use(
    name: str,
    tool_input: object,
    schemas: dict[str, dict],
    *,
    backend: str | None = None,
    on_anomaly: Callable[[], None] | None = None,
) -> None:
    """Log one returned ``tool_use`` block, and warn if its shape contradicts its schema.

    Always emits the DEBUG line: the point of kitty-bridge#33's first ask is
    that the forwarded payload be recoverable from kitty's own log even when
    nothing looks wrong.

    Args:
        name: The tool's name.
        tool_input: The ``input`` value returned by the upstream.
        schemas: Declared schemas from :func:`collect_tool_schemas`.
        backend: Identifier of the backend that served the response, so a
            warning points at a pool member.
        on_anomaly: Called once when an anomaly is found, so the caller can
            count it somewhere an operator will see without ``--debug``.
    """
    # Full input at DEBUG only — the debug log is opt-in and file-backed.
    logger.debug("%s name=%s input=%s", AUDIT_MARKER, name, _render(tool_input))

    anomaly = describe_tool_input_anomaly(name, tool_input, schemas.get(name))
    if anomaly is None:
        return

    # Key names only. This line can reach an embedder's own handler, and tool
    # inputs routinely carry file contents and user data.
    logger.warning(
        "%s upstream returned a malformed tool_use (backend=%s): %s",
        AUDIT_MARKER,
        backend or "unknown",
        anomaly,
    )
    if on_anomaly is not None:
        on_anomaly()


def _render(tool_input: object) -> str:
    """Render a tool input for the DEBUG log, bounded and never raising.

    Args:
        tool_input: The value to render.

    Returns:
        A JSON rendering truncated to :data:`MAX_LOGGED_INPUT_CHARS`, or a
        repr-based fallback when the value is not JSON-serialisable.
    """
    try:
        text = json.dumps(tool_input, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(tool_input)
    if len(text) > MAX_LOGGED_INPUT_CHARS:
        return f"{text[:MAX_LOGGED_INPUT_CHARS]}… ({len(text)} chars total)"
    return text


class ToolUseAuditor:
    """Assembles ``tool_use`` inputs from the Anthropic SSE written to the client.

    Every Messages path — native passthrough, CC-translated and custom
    transport — ends by writing Anthropic SSE to the client, so that write
    boundary is the one place all of them share.  Observing there means one
    parser and one state machine, and it means an abandoned upstream attempt is
    never audited, because its events are never written.

    Chunks arrive on arbitrary byte boundaries, so lines are buffered and split
    the way the bridge's own SSE handling does.

    Every public method is total.  A leaked exception here would be
    indistinguishable from an upstream stream failure — it would inject an SSE
    ``error`` event into a stream that already delivered valid content and
    leave the serving backend unmarked — so parse failures and unexpected
    shapes are contained, and after any containment event the auditor disables
    itself for the rest of the response rather than failing once per chunk.
    """

    def __init__(
        self,
        schemas: dict[str, dict],
        *,
        backend: str | None = None,
        on_anomaly: Callable[[], None] | None = None,
    ) -> None:
        """Initialise an auditor for one upstream attempt.

        A new instance is required per attempt: carrying one across a failover
        would splice the previous attempt's trailing partial line onto the next
        attempt's first chunk and merge two independent blocks at the same
        index, fabricating a warning about bytes that never existed together.

        Args:
            schemas: Declared tool schemas from :func:`collect_tool_schemas`.
            backend: Identifier of the serving backend, for warnings.
            on_anomaly: Called once per anomaly found.
        """
        self._schemas = schemas
        self._backend = backend
        self._on_anomaly = on_anomaly
        self._line_buffer = bytearray()
        # Open tool_use blocks by content-block index.
        self._open: dict[int, dict] = {}
        self._disabled = False

    def feed(self, chunk: bytes) -> None:
        """Consume one chunk of the SSE being written to the client.

        Args:
            chunk: Raw bytes, exactly as written downstream.
        """
        if self._disabled:
            return
        try:
            self._feed(chunk)
        except Exception as exc:
            self._disable(f"{type(exc).__name__}: {exc}")

    def finish(self) -> None:
        """Report any ``tool_use`` block left open when the response ended.

        An upstream that truncates mid-tool-call is exactly the case worth
        seeing in the log, so the partial input is reported rather than
        dropped.
        """
        if self._disabled:
            return
        try:
            for index in list(self._open):
                self._close(index)
        except Exception as exc:
            self._disable(f"{type(exc).__name__}: {exc}")

    def _disable(self, reason: str) -> None:
        """Stop auditing this response after a containment event.

        Args:
            reason: What went wrong, for the single DEBUG line.
        """
        self._disabled = True
        self._open.clear()
        self._line_buffer = bytearray()
        logger.debug("%s auditing failed, disabled for this response: %s", AUDIT_MARKER, reason)

    def _feed(self, chunk: bytes) -> None:
        """Buffer bytes and dispatch each complete SSE data line.

        Args:
            chunk: Raw bytes, exactly as written downstream.
        """
        self._line_buffer.extend(chunk)
        if len(self._line_buffer) > _MAX_LINE_BYTES:
            self._disable(f"SSE line exceeded {_MAX_LINE_BYTES} bytes")
            return
        while b"\n" in self._line_buffer:
            raw_line, _, rest = bytes(self._line_buffer).partition(b"\n")
            self._line_buffer = bytearray(rest)
            line = raw_line.decode("utf-8", errors="replace").strip()
            # Cheap pre-check: most lines are `event:` or blank, and skipping
            # them avoids a JSON parse per line.
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                self._handle(event)

    def _handle(self, event: dict) -> None:
        """Track one Anthropic SSE event.

        Args:
            event: The parsed ``data:`` payload.
        """
        event_type = event.get("type")

        if event_type == "content_block_start":
            block = event.get("content_block")
            index = event.get("index")
            if isinstance(block, dict) and block.get("type") == "tool_use" and isinstance(index, int):
                # Anthropic always opens with an empty input and streams
                # deltas, but a non-Anthropic shim handing back a pre-parsed
                # tool call typically populates it here and sends no deltas at
                # all — precisely the traffic this audit exists to see. Seed
                # from it; any deltas that follow win.
                if len(self._open) >= _MAX_OPEN_BLOCKS:
                    self._disable(f"more than {_MAX_OPEN_BLOCKS} open tool_use blocks")
                    return
                seeded = block.get("input")
                self._open[index] = {
                    "name": block.get("name", ""),
                    "args": [],
                    "seeded": seeded if isinstance(seeded, dict) and seeded else None,
                }
            return

        if event_type == "content_block_delta":
            index = event.get("index")
            delta = event.get("delta")
            if not isinstance(index, int) or not isinstance(delta, dict):
                return
            state = self._open.get(index)
            if state is None or delta.get("type") != "input_json_delta":
                return
            partial = delta.get("partial_json")
            if isinstance(partial, str):
                state["args"].append(partial)
                state["size"] = state.get("size", 0) + len(partial)
                # Abandon this block rather than the whole audit: a runaway
                # tool call is still a stream the client is entitled to.
                if state["size"] > _MAX_BUFFERED_ARG_CHARS:
                    logger.debug("%s buffer exceeded for index %d, abandoning block", AUDIT_MARKER, index)
                    del self._open[index]
            return

        if event_type == "content_block_stop":
            index = event.get("index")
            if isinstance(index, int) and index in self._open:
                self._close(index)

    def _close(self, index: int) -> None:
        """Finish one ``tool_use`` block and report it.

        Args:
            index: The content-block index that just closed.
        """
        state = self._open.pop(index, None)
        if state is None:
            return

        text = "".join(state["args"])
        if not text:
            # No deltas: either a shim that populated content_block_start, or
            # a genuinely argument-less tool call.
            report_tool_use(
                state["name"],
                state["seeded"] if state["seeded"] is not None else {},
                self._schemas,
                backend=self._backend,
                on_anomaly=self._on_anomaly,
            )
            return

        try:
            tool_input = json.loads(text)
        except json.JSONDecodeError:
            # Invalid accumulated JSON is itself worth seeing — it is the
            # serialization defect the issue's corroborating report describes.
            # Length only, never the bytes: they may be a truncated payload.
            logger.warning(
                "%s upstream tool_use %r arguments are not valid JSON (backend=%s, %d chars)",
                AUDIT_MARKER,
                state["name"],
                self._backend or "unknown",
                len(text),
            )
            if self._on_anomaly is not None:
                self._on_anomaly()
            return

        report_tool_use(
            state["name"],
            tool_input,
            self._schemas,
            backend=self._backend,
            on_anomaly=self._on_anomaly,
        )
