"""Agent settings precedence — three runs, three winners (TEST_SUITE.md §6.4.2).

Plan task **T-I6** (KBR-98).
`.requirements/20260922T100000Z_agent_settings_precedence/REQUIREMENTS.md`.

The claim under test is a fact about *Claude Code's* behaviour, not
kitty's: that ``--settings <file>`` outranks ``~/.claude/settings.json``
outranks the process environment for ``ANTHROPIC_BASE_URL`` (and
therefore for the bridge URL Claude Code is pointed at). The launcher
in :mod:`kitty.launchers.claude` depends on this — it writes a
per-session ``--settings`` file precisely because that scope wins —
and the KBR-1 investigation put the precedence order under suspicion.

**The three-sentinel design.** Three runs, each binding three real
:class:`~tests.harness.bridge.BridgeFixture`s on three independent
loopback ports — one recorder, one sentinel A (the env's
``ANTHROPIC_BASE_URL``), one sentinel B (the global settings file's
``env.ANTHROPIC_BASE_URL``). Each run asserts that exactly one of the
three received the user-role ``POST /v1/messages``, and the other two
captured nothing:

| Run      | ``--settings``  | ``~/.claude/settings.json`` | env | Winner    |
|----------|-----------------|------------------------------|-----|-----------|
| Main     | present → recorder | present → sentinel B       | A   | recorder  |
| Control 1| absent          | present → sentinel B         | A   | sentinel B|
| Control 2| absent          | absent                       | A   | sentinel A|

**Falsification framing (§6.3 validation rule, applied at L4).** A
sentinel that is never hit cannot prove it can be hit, so each
sentinel serves as the winner in at least one run — without the
controls, a typo in A's URL reads as a pass because A is *supposed*
to be silent in Main and Control 1, and a broken A is silent too.
A wrong-precedence binary fails Main with one of the sentinels as
the winner, and the controls then narrow the diagnosis. The diagnostic
message names the fixture that captured — the destination that
captured is named in the failure.

**L4 rationale.** A precedence order is a fact about Claude Code, not
kitty code; the only observable surface is the real binary's
destination choice, and only the real binary's destination choice is
what T-I6's launcher design depends on. A lower-layer test would have
to mock the very thing under suspicion.

**Session-title assumption.** Claude Code in ``-p`` mode makes more
than one POST per turn (T-I5's step file records the discovery: a
session-title-generation call precedes the user reply). The test
asserts the user turn goes to the winner; the session-title call's
destination is assumed to follow the same precedence chain, so the
winner's ``captures`` is non-empty on the same turn. The test does
not separately assert the title call's destination — it is a Claude
Code implementation detail that today's binary happens to share.

**Inheritance.** Reuses T-I5's :func:`_claude_binary`,
:func:`_hermetic_env`, :func:`_body_has_user_message`, and
:func:`_run_one_turn` (extended with the keyword-only ``extra_args``
parameter this task added). The bridge fixture is shared with
:mod:`harness.bridge`. The ``agent_smoke`` marker is inherited from
the ``tests/agent_smoke/`` path default (``tests/layers.py``); no
``pytestmark`` on this file, per the T-J1 rule. The canonical
hermeticity posture lives in T-I5's module docstring
(``tests/agent_smoke/test_claude_startup.py``) — four redirects, one
cwd pin, one env-key filter — and is not restated here.

**Layer.** No runner in the tree selects ``agent_smoke`` until T-K10
lands (KBR-119, To Do). Until then, the tests run only when a
developer explicitly invokes ``pytest -m agent_smoke``. That debt is
registered, not silent: ``PENDING_ACTIVATION_LAYERS["agent_smoke"]``
names T-K10.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from harness.bridge import BridgeFixture, transport
from harness.contract import WireFormat

from agent_smoke.test_claude_startup import (
    _body_has_user_message,
    _claude_binary,
    _hermetic_env,
    _run_one_turn,
)


def _write_session_settings_file(path: Path, base_url: str) -> None:
    """Write the ``--settings`` file Claude Code reads as the highest-priority scope.

    The file is exactly ``{"env": {"ANTHROPIC_BASE_URL": <base_url>}}``.
    A wider file would couple this task to the launcher's key set and
    break the next time the launcher grows a new key — the precedence
    claim is about one key, not the launcher's key list.

    Args:
        path: The file location Claude Code will be pointed at via
            ``--settings <path>``.
        base_url: The bridge URL the session settings should target —
            i.e. the recorder in the Main run.
    """
    path.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": base_url}}), encoding="utf-8")


def _write_global_settings_file(path: Path, base_url: str) -> None:
    """Write the ``~/.claude/settings.json`` (under ``CLAUDE_CONFIG_DIR``) file.

    Same shape as the session file — ``{"env": {"ANTHROPIC_BASE_URL":
    <base_url>}}`` — but lives under ``$CLAUDE_CONFIG_DIR/settings.json``
    so Claude Code reads it as the user-scope source. Control 2 omits
    this file entirely; that absence is itself the negative case.

    Args:
        path: The file location, ``$CLAUDE_CONFIG_DIR/settings.json``.
        base_url: The bridge URL the global settings should target —
            i.e. sentinel B in Main, the winner in Control 1.
    """
    path.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": base_url}}), encoding="utf-8")


def _loser_labels() -> tuple[str, str]:
    """Return the canonical ``(loser-A, loser-B)`` labels for diagnostics.

    Returns:
        Two human-readable names that name the destination, not the
        position in a tuple, so a failure on a different fixture than
        the one named by position reads with the right destination.
    """
    return ("sentinel A (env)", "sentinel B (global settings)")


async def _run_precedence_turn(
    *,
    tmp_path: Path,
    binary: Path,
    winner: BridgeFixture,
    winner_label: str,
    losers: Sequence[tuple[str, BridgeFixture]],
    env_url: str,
    extra_args: Sequence[str] = (),
    global_settings_url: str | None = None,
) -> None:
    """Run one turn; assert the named winner captures, every loser stays silent.

    Args:
        tmp_path: The pytest-supplied temp directory, hosting both
            ``HOME`` and ``CLAUDE_CONFIG_DIR``.
        binary: The resolved Claude Code executable.
        winner: The fixture that should receive the user-turn POST.
        winner_label: Human-readable name of the winning destination,
            used in assertion messages.
        losers: ``(label, fixture)`` pairs for every destination
            expected to capture nothing. Labels name the destination
            (e.g. ``"sentinel A (env)"``), not the position in a tuple.
        env_url: The ``ANTHROPIC_BASE_URL`` value in the child env.
            For Main and both controls this is sentinel A's URL —
            the env is one of three sources, and the others win.
        extra_args: Additional CLI args appended to ``claude`` after
            ``--dangerously-skip-permissions``. Main passes
            ``("--settings", <path>)``; both controls pass an empty
            tuple.
        global_settings_url: When set, writes
            ``$CLAUDE_CONFIG_DIR/settings.json`` pointing at this URL.
            ``None`` leaves the file absent (Control 2's condition).

    Raises:
        AssertionError: When the winner captured nothing, or any
            loser captured anything.
    """
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir(parents=True, exist_ok=True)

    if global_settings_url is not None:
        _write_global_settings_file(config_dir / "settings.json", global_settings_url)

    env = _hermetic_env(home, config_dir, env_url)
    returncode, stdout, stderr = await _run_one_turn(binary, env, home, extra_args=tuple(extra_args))
    diagnostic = f"exit={returncode} stdout={stdout[:400]!r} stderr={stderr[:400]!r}"

    winner_captures = list(winner.captures)
    user_captures = [
        c for c in winner_captures
        if c.path == "/v1/messages" and _body_has_user_message(c.body)
    ]
    assert user_captures, (
        f"the winner ({winner_label}) captured {len(winner_captures)} "
        f"request(s) but none carried a user-role message on /v1/messages; "
        f"loser captures: {[(label, len(list(f.captures))) for label, f in losers]}: {diagnostic}"
    )
    for label, fixture in losers:
        loser_captures = list(fixture.captures)
        assert not loser_captures, (
            f"{label} captured {len(loser_captures)} request(s) but "
            f"should have been silent under the precedence Claude Code "
            f"documents: {diagnostic}"
        )


class TestSettingsPrecedence:
    """The §6.4.2 precedence claim — three runs, three winners."""

    async def test_main_run(self, tmp_path: Path) -> None:
        """Session settings (``--settings``) outrank global settings and env.

        With all three sources pointing at three different fixtures,
        the recorder (the ``--settings`` destination) must receive the
        user-turn POST; sentinel A (env) and sentinel B (global
        settings) must capture nothing.
        """
        binary = _claude_binary()
        # Create ``home/`` before any helper writes into it — the
        # session settings file lands at ``home/session-settings.json``
        # and the writer does not mkdir.
        home = tmp_path / "home"
        home.mkdir(parents=True, exist_ok=True)
        async with (
            BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)) as recorder,
            BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)) as sentinel_a,
            BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)) as sentinel_b,
        ):
            session_settings = home / "session-settings.json"
            _write_session_settings_file(session_settings, recorder.base_url)
            await _run_precedence_turn(
                tmp_path=tmp_path,
                binary=binary,
                winner=recorder,
                winner_label="the recorder (--settings)",
                losers=zip(_loser_labels(), (sentinel_a, sentinel_b), strict=True),
                env_url=sentinel_a.base_url,
                extra_args=("--settings", str(session_settings)),
                global_settings_url=sentinel_b.base_url,
            )

    async def test_control_1_global_settings_win(self, tmp_path: Path) -> None:
        """Without ``--settings``, the global settings file outranks env.

        ``--settings`` is absent; the global settings file points at
        sentinel B (the winner); the env points at sentinel A. Recorder
        and sentinel A must capture nothing.
        """
        binary = _claude_binary()
        async with (
            BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)) as recorder,
            BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)) as sentinel_a,
            BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)) as sentinel_b,
        ):
            await _run_precedence_turn(
                tmp_path=tmp_path,
                binary=binary,
                winner=sentinel_b,
                winner_label="sentinel B (global settings)",
                losers=zip(
                    ("the recorder (no source points at it)", "sentinel A (env)"),
                    (recorder, sentinel_a),
                    strict=True,
                ),
                env_url=sentinel_a.base_url,
                extra_args=(),
                global_settings_url=sentinel_b.base_url,
            )

    async def test_control_2_env_wins(self, tmp_path: Path) -> None:
        """Without ``--settings`` and without a global settings file, env wins.

        Sentinel A (the env destination) must receive the user-turn
        POST; recorder (no source points at it) and sentinel B (no
        file points at it) must capture nothing. The global-settings
        file is **absent** — not empty — so the absence is the
        negative case proving the file is the only path between the
        global scope and Claude Code.
        """
        binary = _claude_binary()
        async with (
            BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)) as recorder,
            BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)) as sentinel_a,
            BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES)) as sentinel_b,
        ):
            await _run_precedence_turn(
                tmp_path=tmp_path,
                binary=binary,
                winner=sentinel_a,
                winner_label="sentinel A (env)",
                losers=zip(
                    ("the recorder (no source points at it)", "sentinel B (no file points at it)"),
                    (recorder, sentinel_b),
                    strict=True,
                ),
                env_url=sentinel_a.base_url,
                extra_args=(),
                global_settings_url=None,
            )
