"""Agent startup smoke — a pinned real Claude Code binary reaches the bridge.

`.system_design/TEST_SUITE.md` §6.4.2 · plan task **T-I5** (KBR-97) ·
`.requirements/20260921T114643Z_agent_startup_smoke/REQUIREMENTS.md`.

One test, one claim, and the claim is TEST_SUITE.md §6.4.2's own sentence:
a pinned real Claude Code binary runs one non-interactive turn against the
scripted recorder — no live provider, no credentials, no network — and
*connectivity* is what it proves. It does not prove the settings-precedence
claim (T-I6 / KBR-98), the content claim (T-W9's slice and the acceptance
scenarios), or anything about a specific binary version (§8.6 records that
the pin is a hand-maintained pairing with a floating action tag, and that
"a job asserting a *specific* CLI version rests on a human having noticed").

**How many requests one turn makes.** §6.4.2's sentence says *"its
request"*, singular. Claude Code in ``-p`` mode makes **more than one**
POST per turn: a session-title-generation call before the user reply
(observed on Claude Code 2.1.276; the binary's own stderr names the
two ``query_source`` values, ``generate_session_title`` and the default).
This test asserts connectivity with ``len(captures) >= 1`` and pins the
*user* turn by content, not by index — pinning to exactly one would fail
the moment Claude Code adds another pre-flight call, and that is the
wrong signal for a connectivity smoke.

**Hermetic by construction.** Four redirects, each closing a different
leak:

* The recorder upstream returns a canned reply, the bridge fixture
  carries a fixed dummy key (``tests/harness/bridge.py`` :data:`_KEY`),
  and every loop is loopback. Nothing here talks to a provider.
* ``HOME`` points at a fresh ``tmp_path`` directory. Claude Code keeps a
  fifth file, ``~/.claude.json``, that it writes *outside* the
  ``CLAUDE_CONFIG_DIR`` it honours (see
  ``anthropics/claude-code#25762``); ``HOME`` is the coarse redirect that
  covers it.
* ``CLAUDE_CONFIG_DIR`` is the fine redirect, belt to ``HOME``'s braces
  for "all settings, session history, and plugins"
  (https://code.claude.com/docs/en/env-vars).
* ``CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`` kills the binary's
  statsig / sentry / update checks, which would otherwise make
  "no network beyond loopback" false for the child. The repo already uses
  this flag for its own hermetic runs
  (``tests/integration/test_tmux_disconnect.py``,
  ``scripts/capture_corpus_t_c1.py``).

**The binary path follows kitty's own fallback chain.** §8.6 records that
``~/.local/bin/claude`` is deliberately not on ``PATH`` and that kitty's
fallback chain is the rung that finds it. This test uses
:func:`kitty.launchers.discovery.discover_binary` so the two would drift
together, not independently. ``discover_binary`` checks ``PATH`` first
and falls back to the platform locations; on a dev machine where
``claude`` resolves through ``PATH`` to a kitty wrapper, the wrapper's
spawn config overrides ``ANTHROPIC_BASE_URL`` at process level — the
captures would then be empty and the test would fail loudly, naming the
path it tried. The connectivity claim still holds; the *raw-binary*
assertion is the wrapper-overrides-``HOME``-and-``CLAUDE_CONFIG_DIR``
case, which the three redirects together cover.

**Missing binary is a failure, not a skip.** TEST_SUITE.md §8 line 4524
(*"Skips are failures in a gating job"*) applies: a green tick that
means "we did not test this" is worse than a red one.
``KITTY_AGENT_SMOKE_BINARY`` exists so a developer can point the test
at a specific binary, and so the falsification case in
:class:`TestMissingBinaryFalsification` can simulate the absence
deterministically — poisoning ``PATH`` cannot, because
``discover_binary`` falls back to ``~/.local/bin``, where the real
binary sits.

**``--dangerously-skip-permissions`` is required, not optional.** In
``-p`` non-interactive mode the binary has no TTY, so a permission
prompt would fail rather than hang. Skipping the permission step makes
the ``-p`` turn complete in any case where the binary would otherwise
need a human — and the smoke's claim is connectivity, not permissions;
a permissions-shaped test belongs elsewhere.

**Layer.** No ``pytestmark``; the path default from
``tests/layers.py::_PATH_DEFAULTS`` (row ``tests/agent_smoke/``) carries
the ``agent_smoke`` marker, following the T-J1 rule that a new
directory adds a row rather than a marker to each of its files. The
first merge exercises the test under ``pytest -m agent_smoke`` and in
the ``agent_live`` test runner; the per-PR gate that *positively*
selects ``agent_smoke`` is plan task T-K10 (KBR-119, currently To Do).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import pytest
from harness.bridge import BridgeFixture, transport
from harness.contract import WireFormat

from kitty.launchers.discovery import discover_binary

#: The one non-interactive prompt the smoke drives. Chosen to be answerable
#: from a canned response alone — no tool call, no long context — so a slow
#: or chatty binary still finishes inside the timeout.
_PROMPT = "Reply with the single word: ready"

#: Ceiling on one non-interactive turn, including the binary's own startup.
#: Generous for a contended CI runner; bounded so a hung binary fails this
#: test, not the job.
_TURN_TIMEOUT_SECONDS = 60.0


def _claude_binary(env: Mapping[str, str] | None = None) -> Path:
    """Return the Claude Code binary, or fail the test naming the lookup.

    Args:
        env: The environment to read, defaulting to :data:`os.environ`.
            ``KITTY_AGENT_SMOKE_BINARY``, when set to an existing file
            path, overrides the lookup entirely — that is the seam the
            falsification case below drives, and the only way to
            simulate a missing binary deterministically (poisoning
            ``PATH`` cannot, because :func:`discover_binary` falls back
            to ``~/.local/bin``, where the real binary sits).

    Returns:
        The resolved path to the ``claude`` executable.

    Raises:
        pytest.fail: When the binary cannot be located, naming the
            fallback chain that was searched, so a CI log reader can
            tell "install failed" from "wrong binary".
    """
    environment = os.environ if env is None else env
    override = environment.get("KITTY_AGENT_SMOKE_BINARY")
    if override:
        path = Path(override)
        if path.is_file():
            return path
        pytest.fail(
            f"KITTY_AGENT_SMOKE_BINARY names {override!r}, which is not a "
            f"file. The override exists so the missing-binary path can "
            f"be falsified deterministically; point it at the real "
            f"binary or unset it."
        )
    binary = discover_binary("claude")
    if binary is None:
        pytest.fail(
            "the Claude Code CLI was not found on PATH or in the platform "
            "fallback directories (~/.local/bin, ~/.nvm/versions/node/*/bin, "
            "~/.npm-global/bin, /usr/local/bin). The agent_smoke gate "
            "requires a real pinned binary — install it, or set "
            "KITTY_AGENT_SMOKE_BINARY to its absolute path."
        )
    return binary  # type: ignore[no-any-return]


def _hermetic_env(home: Path, config_dir: Path, base_url: str) -> dict[str, str]:
    """Build the child environment for one hermetic turn against ``base_url``.

    Three redirects, each closing a different leak — see the module
    docstring's "Hermetic by construction" section for the rationale of
    each. The fourth redirect (``ANTHROPIC_BASE_URL``) is the bridge
    loopback itself, which the smoke is *about* rather than a leak to
    close.

    Args:
        home: The fresh directory to hand ``HOME``, so ``~/.claude.json``
            writes land in the temp tree.
        config_dir: The fresh directory to hand ``CLAUDE_CONFIG_DIR``,
            so settings / history / plugins land in the temp tree.
        base_url: The bridge's loopback origin, from
            :attr:`~harness.bridge.BridgeFixture.base_url`.

    Returns:
        An environment suitable for :func:`asyncio.create_subprocess_exec`.
        Inherited on purpose — the binary needs ``PATH`` for its node
        runtime — with the five overrides layered on top and a dummy
        key standing in for a credential.
    """
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = base_url
    # A fixed dummy value, matching the bridge fixture's own `_KEY` shape:
    # the bridge does not validate the inbound key, and the recorder is
    # the only reader of what the bridge sends upstream.
    env["ANTHROPIC_API_KEY"] = "kitty-agent-smoke-not-a-credential"
    env["HOME"] = str(home)
    env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return env


async def _run_one_turn(binary: Path, env: dict[str, str]) -> tuple[int, str, str]:
    """Spawn the binary for one non-interactive turn and collect the output.

    Args:
        binary: The resolved Claude Code executable.
        env: The child environment, from :func:`_hermetic_env`.

    Returns:
        ``(returncode, stdout, stderr)`` after the process exits. The
        returncode is the actual exit status when ``communicate`` has
        completed; ``-1`` if ``communicate`` somehow returned without
        setting it (defensive — not exercised in practice).

    Raises:
        pytest.fail: When the process outlives
            :data:`_TURN_TIMEOUT_SECONDS`. The process is killed first
            so a hung binary does not survive the test to wedge the
            runner.
    """
    proc = await asyncio.create_subprocess_exec(
        str(binary),
        "-p",
        _PROMPT,
        "--dangerously-skip-permissions",
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TURN_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
            await proc.wait()
        pytest.fail(
            f"claude did not finish one non-interactive turn within "
            f"{_TURN_TIMEOUT_SECONDS}s; killed. The claim under test is "
            f"connectivity, so a timeout here means the bridge was "
            f"never reached or the reply never came back."
        )
    returncode = proc.returncode if proc.returncode is not None else -1
    return returncode, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


def _body_has_user_message(raw_body: bytes) -> bool:
    """Judge whether a captured request body carries at least one user turn.

    Args:
        raw_body: The captured request body bytes.

    Returns:
        ``True`` when the body parses as a JSON object whose ``messages``
        list holds at least one entry with ``role == "user"``; ``False``
        for a body that does not parse or does not carry one.
        Deliberately lenient on everything else: the smoke's claim is
        connectivity, not the Messages schema, and a schema claim
        belongs to the reader tests (T-A1).
    """
    try:
        body = json.loads(raw_body)
    except (ValueError, TypeError):
        return False
    if not isinstance(body, dict):
        return False
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    return any(isinstance(m, dict) and m.get("role") == "user" for m in messages)


class TestAgentStartupSmoke:
    """The §6.4.2 startup smoke — one claim, three assertions, all on evidence."""

    async def test_a_pinned_claude_code_runs_one_turn_and_reaches_the_bridge(self, tmp_path: Path) -> None:
        """Launch one real ``claude -p`` turn and assert it reached the bridge.

        The three assertions, in order, are the parts of the §6.4.2
        sentence: *the binary starts and exits cleanly* (exit 0),
        *it resolves the bridge URL* (a capture exists at all), and
        *its request arrives* (at least one capture is a
        ``POST /v1/messages`` carrying a user turn). A test that
        checked only the capture could not tell "the binary works"
        from "the binary hung and pytest timed out around it".

        Args:
            tmp_path: Pytest's per-test temp directory, hosting both
                ``HOME`` and ``CLAUDE_CONFIG_DIR`` so the run is
                hermetic.
        """
        # Both directories must exist before the binary writes into
        # them; some CLIs do not create their own state directory on
        # first run.
        home = tmp_path / "home"
        home.mkdir(parents=True, exist_ok=True)
        config_dir = tmp_path / "claude-config"
        config_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the binary before starting the fixture, so a missing
        # binary fails in one line without a half-started bridge to
        # unwind.
        binary = _claude_binary()

        async with BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)) as fixture:
            env = _hermetic_env(home, config_dir, fixture.base_url)
            returncode, stdout, stderr = await _run_one_turn(binary, env)
            captures = list(fixture.captures)

        # Evidence on the exit, before evidence on the capture: a
        # nonzero exit with zero captures reads as "the binary refused
        # to start", while a nonzero exit with a capture reads as "the
        # bridge answered something the binary rejected". Ordering the
        # assertions this way keeps both diagnoses visible in the
        # failure message.
        diagnostic = f"exit={returncode} stdout={stdout[:400]!r} stderr={stderr[:400]!r}"
        assert returncode == 0, f"claude did not exit cleanly: {diagnostic}"
        assert captures, (
            f"no request reached the recorder — connectivity claim "
            f"unproven: {diagnostic}"
        )

        # Pin the *user* turn by content, not by index: Claude Code in
        # -p mode may make more than one POST (e.g. a
        # session-title-generation call before the reply), and
        # asserting on a count would tie the smoke to today's binary
        # shape.
        user_captures = [
            c for c in captures
            if c.path == "/v1/messages"
            and _body_has_user_message(c.body)
        ]
        assert user_captures, (
            f"the bridge saw {len(captures)} request(s) but none carried a "
            f"user-role message on /v1/messages: {diagnostic}"
        )


class TestMissingBinaryFalsification:
    """The §8 falsification for the missing-binary rule (the KBR-132 shape).

    TEST_SUITE.md §8 line 4537 names the pattern: a detector is a
    detector only if *restoring the defect turns it red*. The defect
    here is :func:`_claude_binary` failing (not skipping) when no
    binary is found. :func:`kitty.launchers.discovery.discover_binary`
    falls back to ``~/.local/bin``, where the real binary sits, so
    poisoning ``PATH`` cannot simulate the absence — which is why
    :func:`_claude_binary` takes the ``KITTY_AGENT_SMOKE_BINARY``
    override: it is the seam that makes this case automatable, and the
    only one that does.
    """

    def test_a_missing_binary_fails_rather_than_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Point the override at a nonexistent path; ``pytest.fail`` must fire.

        Args:
            tmp_path: Pytest's per-test temp directory, supplying a path
                that is guaranteed not to exist.
            monkeypatch: Pytest's monkeypatch fixture, isolating the env
                override to this test.
        """
        monkeypatch.setenv("KITTY_AGENT_SMOKE_BINARY", str(tmp_path / "no-such-claude"))
        with pytest.raises(pytest.fail.Exception, match="not a file"):
            _claude_binary()

    def test_an_existing_override_path_is_honoured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Point the override at a real file; it must be returned untouched.

        Args:
            tmp_path: Pytest's per-test temp directory, supplying a real
                (if not actually executable) file path.
            monkeypatch: Pytest's monkeypatch fixture, isolating the env
                override to this test.
        """
        fake_binary = tmp_path / "a-real-file"
        fake_binary.touch()
        monkeypatch.setenv("KITTY_AGENT_SMOKE_BINARY", str(fake_binary))
        assert _claude_binary() == fake_binary
