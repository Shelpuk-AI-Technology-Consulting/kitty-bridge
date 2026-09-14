#!/usr/bin/env python3
"""Capture the KBR-44 (T-C1) corpus entries from real Claude Code 2.1.238.

Drives four scripted sessions against per-session :class:`RecordingUpstream`
instances per ``tests/corpus/README.md``'s capture procedure, inspects each
captured body, and writes the entries via
:func:`harness.corpus.write_entry` (which scrubs on the way out).

The C5 check is run by the caller, not here: this script prints, for each
session, whether the body carries ``output_config`` and whether it
co-occurs with ``thinking`` / ``effort``. The caller (the T-C1 operator)
reads each ``.body`` end-to-end before committing and decides the entry
declarations per ``tests/corpus/README.md``'s "declare against the
committed artifact" rule.

Usage::

    CC_BIN=/tmp/kbr44/cc-2.1.238/claude \\
        PYTHONPATH=tests ./.venv/bin/python scripts/capture_corpus_t_c1.py

Environment:
    CC_BIN: Path to the Claude Code binary to drive. Defaults to
        ``/tmp/kbr44/cc-2.1.238/claude`` (the workflow pin, staged per
        the README). The binary must report ``2.1.238`` — the freshness
        guard refuses any other version at the lint.

The script never touches the user's ``~/.claude``: ``HOME`` and
``CLAUDE_CONFIG_DIR`` are redirected to a scratch directory for every
session.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "tests"))

from harness.contract import WireFormat  # noqa: E402
from harness.recorder import RecordingUpstream  # noqa: E402

CC_BIN = os.environ.get("CC_BIN", "/tmp/kbr44/cc-2.1.238/claude")

#: The streaming Anthropic Messages reply the recorder gives for a plain turn.
#: Same event grammar the recorder's default responder uses (§6.2.2); spelled
#: out here so a reader sees what the agent was told without opening the
#: recorder module, and so session B's second slot has a concrete value.
_PLAIN_REPLY: tuple[bytes, ...] = tuple(
    f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()
    for name, payload in [
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_capture", "type": "message", "role": "assistant",
                "model": "recorder-model", "content": [], "stop_reason": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }),
        ("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "ok"},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        }),
        ("message_stop", {"type": "message_stop"}),
    ]
)


def _tool_use_events() -> tuple[bytes, ...]:
    """A streaming reply whose only block is a ``Bash`` tool_use."""
    import hashlib

    tool_id = "toolu_" + hashlib.sha256(b"kbr44-capture").hexdigest()[:16]
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_capture", "type": "message", "role": "assistant",
                    "model": "recorder-model", "content": [], "stop_reason": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "tool_use", "id": tool_id, "name": "Bash", "input": {}},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "input_json_delta",
                          "partial_json": json.dumps({"command": "echo ok"})},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return tuple(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode() for name, payload in events)


def _queue_responder(replies: list[tuple[bytes, ...]]):
    """Return a responder that hands out ``replies`` in order, then repeats the last."""
    state = {"index": 0}

    async def respond(captured, response) -> None:  # type: ignore[no-untyped-def]
        idx = min(state["index"], len(replies) - 1)
        state["index"] += 1
        await response.begin(200, {"Content-Type": "text/event-stream"})
        for chunk in replies[idx]:
            await response.write(chunk)
        await response.write_eof()

    return respond


def _run_cc(session_home: Path, port: int, prompt: str, *, effort: str | None,
            bare: bool, timeout: int) -> subprocess.CompletedProcess[str]:
    """Run one Claude Code session against the recorder on ``port``."""
    env = os.environ.copy()
    env["HOME"] = str(session_home)
    env["CLAUDE_CONFIG_DIR"] = str(session_home / ".claude")
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    env["ANTHROPIC_API_KEY"] = "kbr-44-capture-throwaway"
    env["ANTHROPIC_AUTH_TOKEN"] = "kitty-bridge-token"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["NO_COLOR"] = "1"
    # The effort dial: CC emits `output_config` only when an effort is set,
    # and an inherited CLAUDE_EFFORT would configure it for every session —
    # so it is unset here and re-set explicitly for the effort session.
    env.pop("CLAUDE_EFFORT", None)
    if effort is not None:
        env["CLAUDE_EFFORT"] = effort

    cmd = [CC_BIN]
    if bare:
        cmd.append("--bare")
    cmd += ["-p", "--no-session-persistence", "--permission-mode", "bypassPermissions"]
    if effort is not None:
        cmd += ["--effort", effort]
    cmd.append(prompt)
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)


async def _capture_session(
    name: str,
    prompt: str,
    *,
    effort: str | None = None,
    bare: bool = True,
    scripted: list[tuple[bytes, ...]] | None = None,
) -> list:
    """Drive one session and return its captured requests."""
    upstream = RecordingUpstream(default_format=WireFormat.ANTHROPIC_MESSAGES)
    if scripted is not None:
        upstream.responder = _queue_responder(list(scripted))
    await upstream.start()
    scratch = Path(tempfile.mkdtemp(prefix=f"kbr44-{name}-"))
    try:
        proc = _run_cc(scratch, upstream._port, prompt, effort=effort, bare=bare, timeout=180)
        captured = list(upstream.requests)
        print(f"[{name}] exit={proc.returncode} stdout_lines={len(proc.stdout.splitlines())} "
              f"captured={len(captured)}")
        if proc.returncode != 0:
            head = (proc.stderr or proc.stdout).splitlines()[:6]
            for line in head:
                print(f"    {line}")
        return captured
    finally:
        await upstream.stop()


def _summarise(label: str, captured: list) -> None:
    """Print what each captured body carries — the C5 evidence and the entry shapes."""
    for i, req in enumerate(captured):
        try:
            body = json.loads(req.body)
        except json.JSONDecodeError:
            print(f"  [{label} #{i}] non-JSON body ({len(req.body)} bytes)")
            continue
        has_output_config = "output_config" in body
        has_thinking = "thinking" in body
        has_effort = "effort" in body
        turns = len(body.get("messages", []))
        tools = len(body.get("tools", []))
        max_tokens = body.get("max_tokens")
        stream = body.get("stream")
        block_kinds = sorted(
            {
                part.get("type")
                for turn in body.get("messages", [])
                for part in (turn.get("content") if isinstance(turn.get("content"), list) else [])
                if isinstance(part, dict) and "type" in part
            }
        )
        co_occurs = has_output_config and (has_thinking or has_effort)
        print(
            f"  [{label} #{i}] turns={turns} tools={tools} max_tokens={max_tokens} "
            f"stream={stream} block_kinds={block_kinds} "
            f"output_config={has_output_config} thinking={has_thinking} effort={has_effort} "
            f"co-occurs={co_occurs}"
        )


async def main() -> None:
    # Session A — the baseline plain turn. Bare, no effort, text-only prompt.
    a = await _capture_session("plain", "Reply with the word ok")
    _summarise("plain", a)

    # Session B — the tool_use / tool_result pair. The recorder's first reply
    # carries a Bash tool_use; CC executes it and sends the paired turn; the
    # second reply is plain text so CC stops (the queue repeats the last when
    # the index overruns, which is the desired behaviour — any extra request
    # CC sends gets the same plain text).
    b = await _capture_session(
        "tools",
        "Read the file README.md at the repository root and reply with its first line.",
        scripted=[_tool_use_events(), _PLAIN_REPLY],
    )
    _summarise("tools", b)

    # Session C — effort configured, so the body carries output_config.
    c = await _capture_session("effort", "Reply with the word ok", effort="medium")
    _summarise("effort", c)

    # Session D — a non-bare run, whose tools array is the full set.
    d = await _capture_session("full", "Reply with the word ok", bare=False)
    _summarise("full", d)

    print("\nNOTE: this script writes no entries. The operator reads every .body "
          "end-to-end, declares triggers per tests/corpus/README.md, and calls "
          "harness.corpus.write_entry per entry.")


if __name__ == "__main__":
    asyncio.run(main())
