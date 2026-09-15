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
    """Return a responder that hands out ``replies`` in order to ``/v1/messages`` POSTs.

    Claude Code fires a ``HEAD /api/hello`` preflight before the real POST;
    consumed by the queue, that preflight would burn the first scripted reply
    (the tool_use) and the real request would land on the second slot. The
    responder short-circuits the preflight — anything that is not a non-empty
    ``/v1/messages`` POST gets a 204 No Content and does not advance the queue.
    """
    state = {"index": 0}

    async def respond(captured, response) -> None:  # type: ignore[no-untyped-def]
        if not (captured.method == "POST"
                and captured.path == "/v1/messages"
                and captured.body):
            response.content_length = 0
            await response.begin(204, {})
            await response.write_eof()
            return
        idx = min(state["index"], len(replies) - 1)
        state["index"] += 1
        await response.begin(200, {"Content-Type": "text/event-stream"})
        for chunk in replies[idx]:
            await response.write(chunk)
        await response.write_eof()

    return respond


def _build_cc_cmd(prompt: str, *, effort: str | None, bare: bool) -> list[str]:
    """Build the Claude Code command line for one session."""
    cmd = [CC_BIN]
    if bare:
        cmd.append("--bare")
    cmd += ["-p", "--no-session-persistence", "--permission-mode", "bypassPermissions"]
    if effort is not None:
        cmd += ["--effort", effort]
    cmd.append(prompt)
    return cmd


def _cc_env(session_home: Path, port: int, *, effort: str | None) -> dict[str, str]:
    """Build the environment for one Claude Code session against the recorder."""
    env = os.environ.copy()
    env["HOME"] = str(session_home)
    env["CLAUDE_CONFIG_DIR"] = str(session_home / ".claude")
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    env["ANTHROPIC_API_KEY"] = "kbr-44-capture-throwaway"
    env["ANTHROPIC_AUTH_TOKEN"] = "kitty-bridge-token"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["NO_COLOR"] = "1"
    # The effort dial: CLAUDE_EFFORT, if inherited from the operator's
    # shell, would configure every session identically and blind the
    # plain vs effort comparison the captures support. Pop it here and
    # re-set explicitly for the effort session. CC 2.1.238 itself emits
    # `output_config` in every body regardless — verified 2026-09-14, the
    # plain_turn body carries `output_config: {"effort": "high"}` even
    # with this unset and with --bare — so the comment must not say
    # otherwise. See memory `confirmed_integrations_cc_always_emits_output_config`.
    env.pop("CLAUDE_EFFORT", None)
    if effort is not None:
        env["CLAUDE_EFFORT"] = effort
    return env


async def _run_cc(session_home: Path, port: int, prompt: str, *,
                  effort: str | None, bare: bool, timeout: int) -> subprocess.CompletedProcess[str]:
    """Run one Claude Code session against the recorder on ``port``.

    Async: ``subprocess.run`` would block the event loop, and the recorder's
    aiohttp server runs on the same loop — a blocked loop means the recorder
    never answers, and CC waits out its whole timeout on a silent socket.
    """
    cmd = _build_cc_cmd(prompt, effort=effort, bare=bare)
    env = _cc_env(session_home, port, effort=effort)
    proc = await asyncio.create_subprocess_exec(
        *cmd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        out, err = await proc.communicate()
        return subprocess.CompletedProcess(cmd, -9, out.decode(errors="replace"),
                                           err.decode(errors="replace"))
    return subprocess.CompletedProcess(cmd, proc.returncode or 0,
                                       out.decode(errors="replace"), err.decode(errors="replace"))


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
    with tempfile.TemporaryDirectory(prefix=f"kbr44-{name}-") as home_str:
        scratch = Path(home_str)
        await upstream.start()
        try:
            proc = await _run_cc(scratch, upstream.port, prompt,
                                  effort=effort, bare=bare, timeout=180)
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
    # (on the real /v1/messages POST) carries a Bash tool_use; CC executes it and
    # sends the paired turn; the second reply is plain text so CC stops. The
    # responder skips the `HEAD /api/hello` preflight (returns 204) so the queue
    # is not consumed before the real request.
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
