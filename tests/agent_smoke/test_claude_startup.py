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
directory adds a row rather than a marker to each of its files. Until
plan task T-K10 (KBR-119, currently To Do) activates the per-PR gate,
**no runner in the tree selects this category** — the tests run only
when a developer explicitly invokes ``pytest -m agent_smoke``. That
debt is registered, not silent: ``PENDING_ACTIVATION_LAYERS["agent_smoke"]``
names T-K10, and the bidirectional checks at
``tests/layers.py::unaccounted_layers`` and ``stale_pending_layers``
hold the registry to the tree.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
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


def _claude_binary() -> Path:
    """Return the Claude Code binary, or fail the test naming the lookup.

    ``KITTY_AGENT_SMOKE_BINARY``, when set in the environment to an
    existing file path, overrides the lookup entirely — that is the seam
    the falsification case below drives, and the only way to simulate a
    missing binary deterministically (poisoning ``PATH`` cannot, because
    :func:`discover_binary` falls back to ``~/.local/bin``, where the
    real binary sits).

    Returns:
        The resolved path to the ``claude`` executable.

    Raises:
        pytest.fail: When the binary cannot be located, naming the
            fallback chain that was searched, so a CI log reader can
            tell "install failed" from "wrong binary".
    """
    override = os.environ.get("KITTY_AGENT_SMOKE_BINARY")
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
    binary = _resolve_default_binary()
    if binary is None:
        pytest.fail(
            "the Claude Code CLI was not found on PATH or in the platform "
            "fallback directories (~/.local/bin, ~/.nvm/versions/node/*/bin, "
            "~/.npm-global/bin, /usr/local/bin). The agent_smoke gate "
            "requires a real pinned binary — install it, or set "
            "KITTY_AGENT_SMOKE_BINARY to its absolute path."
        )
    return binary  # type: ignore[no-any-return]


def _resolve_default_binary() -> Path | None:
    """Return the production binary lookup, as a mockable seam.

    Returns:
        Whatever :func:`kitty.launchers.discovery.discover_binary` finds
        for ``claude``, or ``None`` when nothing is found. Its own
        function rather than an inline call so the
        real-missing-binary falsification can monkeypatch it: poisoning
        ``PATH`` cannot reach the fallback chain, so the absence of a
        binary is only simulatable at this boundary.

    Raises:
        Nothing: :func:`discover_binary` returns ``None`` rather than
            raising.
    """
    return discover_binary("claude")  # type: ignore[no-any-return]


def _hermetic_env(home: Path, config_dir: Path, base_url: str) -> dict[str, str]:
    """Build the child environment for one hermetic turn against ``base_url``.

    The child inherits the machine's basics (``PATH`` for its node
    runtime, locale, temp dirs) but **not** the invoking shell's own
    Anthropic / Claude configuration. The filter strips the same families
    the tmux E2E strips (``tests/integration/test_tmux_disconnect.py:140``):
    an exported ``ANTHROPIC_AUTH_TOKEN`` would hand the child a real
    credential, ``CLAUDE_CODE_USE_BEDROCK`` / ``CLAUDE_CODE_USE_VERTEX``
    would redirect it off the bridge onto a cloud backend, and the
    ``CLAUDECODE`` marker would make the binary treat this run as a
    nested session. On top of the filtered base, four redirects close
    the remaining leaks — see the module docstring's
    "Hermetic by construction" section for the rationale of each.

    Args:
        home: The fresh directory to hand ``HOME``, so ``~/.claude.json``
            writes land in the temp tree.
        config_dir: The fresh directory to hand ``CLAUDE_CONFIG_DIR``,
            so settings / history / plugins land in the temp tree.
        base_url: The bridge's loopback origin, from
            :attr:`~harness.bridge.BridgeFixture.base_url`.

    Returns:
        An environment suitable for :func:`asyncio.create_subprocess_exec`,
        with the five overrides layered on top and a dummy key standing
        in for a credential.
    """
    # The filter is prefix-based on purpose: Claude Code's documented
    # surface grows faster than any explicit list would track, and a new
    # `ANTHROPIC_*` or `CLAUDE_CODE_*` variable arriving in the shell
    # must not silently redirect the child.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "CLAUDECODE"))
    }
    env["ANTHROPIC_BASE_URL"] = base_url
    # A fixed dummy value, matching the bridge fixture's own `_KEY` shape:
    # the bridge does not validate the inbound key, and the recorder is
    # the only reader of what the bridge sends upstream.
    env["ANTHROPIC_API_KEY"] = "kitty-agent-smoke-not-a-credential"
    env["HOME"] = str(home)
    env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return env


async def _run_one_turn(binary: Path, env: dict[str, str], cwd: Path) -> tuple[int, str, str]:
    """Spawn the binary for one non-interactive turn and collect the output.

    Args:
        binary: The resolved Claude Code executable.
        env: The child environment, from :func:`_hermetic_env`.
        cwd: The directory to run the binary in. Deliberately **not** the
            checkout: Claude Code loads the project-level ``CLAUDE.md``
            from the cwd, and a future root-level ``.claude/`` directory
            with hooks would execute them unprompted under
            ``--dangerously-skip-permissions``.

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
        cwd=str(cwd),
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
            returncode, stdout, stderr = await _run_one_turn(binary, env, home)
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
    binary is found. Two cases, because two branches can carry the
    defect:

    * the **override branch** — ``KITTY_AGENT_SMOKE_BINARY`` names a
      path that does not exist;
    * the **default branch** — :func:`_resolve_default_binary` returns
      ``None``, which is what a machine with no installed binary
      actually reaches.

    Neither is drivable by poisoning ``PATH``: ``discover_binary``
    falls back to ``~/.local/bin``, where the real binary sits. The
    override branch is drivable through the env var; the default
    branch only through monkeypatching :func:`_resolve_default_binary`
    at its own boundary. A refactor that turns *either* ``pytest.fail``
    into ``pytest.skip`` — the §8 rule's exact forbidden shape — turns
    its case red.
    """

    def test_a_missing_override_fails_rather_than_skips(
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

    def test_a_truly_missing_binary_fails_rather_than_skips(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Drive the default branch to ``None``; ``pytest.fail`` must fire.

        This is the branch a machine with genuinely no installed binary
        reaches. Monkeypatching :func:`_resolve_default_binary` is the
        only deterministic way to reach it — on this box (and any CI
        runner the install step provisions) a real ``claude`` exists, so
        unpatched execution would never get past the lookup.

        Args:
            monkeypatch: Pytest's monkeypatch fixture, replacing the
                lookup for the duration of this test.
        """
        monkeypatch.setattr(
            "agent_smoke.test_claude_startup._resolve_default_binary",
            lambda: None,
        )
        with pytest.raises(pytest.fail.Exception, match="was not found"):
            _claude_binary()


class TestTimeoutKill:
    """The timeout-kill branch of :func:`_run_one_turn`, driven end to end.

    A hung binary must be killed before the test fails, or the hung
    child survives the test to wedge the runner and leak processes into
    every later test. No test here exercises that branch through the
    real binary — a hung ``claude`` would itself hang the smoke — so
    the branch is driven with a stub executable and a shrunk timeout.
    The stub records its own PID before sleeping; the assertion reads
    that PID and proves the process is reaped, which is the guarantee
    the branch exists for.
    """

    @pytest.mark.skipif(sys.platform == "win32", reason="drives a /bin/sh script, which the POSIX paths assume")
    def test_a_hung_binary_is_killed_before_the_test_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Let a stub sleep past a shrunk timeout; the child must be reaped.

        Args:
            tmp_path: Pytest's per-test temp directory, hosting the
                stub, its PID file, and the ``cwd`` the helper runs it
                in.
            monkeypatch: Pytest's monkeypatch fixture, shrinking
                :data:`_TURN_TIMEOUT_SECONDS` for this test only.
        """
        monkeypatch.setattr("agent_smoke.test_claude_startup._TURN_TIMEOUT_SECONDS", 1.0)

        # The stub ignores every argument the helper passes (its -p, the
        # prompt, the permissions flag) and sleeps long enough that a
        # correct timeout always fires first. It writes its own PID
        # before sleeping, so the assertion can check that specific
        # process — not "some sleep somewhere" — is gone.
        pid_file = tmp_path / "stub-pid"
        stub = tmp_path / "sleepy-stub"
        stub.write_text(f"#!/bin/sh\necho $$ > '{pid_file}'\nexec sleep 30\n")
        stub.chmod(0o755)

        with pytest.raises(pytest.fail.Exception, match="did not finish"):
            asyncio.run(_run_one_turn(stub, dict(os.environ), tmp_path))

        # `os.kill(pid, 0)` sends no signal; it raises ProcessLookupError
        # exactly when the process no longer exists. That is the
        # strongest statement available of "the child was reaped".
        pid = int(pid_file.read_text().strip())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
