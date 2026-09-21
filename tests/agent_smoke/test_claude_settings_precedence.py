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
the winner, and the controls then narrow the diagnosis.

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
``pytestmark`` on this file, per the T-J1 rule.

**Hermeticity.** T-I5's posture, unchanged: four redirects (``HOME``,
``CLAUDE_CONFIG_DIR``, ``CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1``,
``ANTHROPIC_BASE_URL``), one ``cwd`` pin to the temp tree, one env-key
filter stripping the ``ANTHROPIC_*`` / ``CLAUDE_CODE_*`` / ``CLAUDECODE``
/ proxy / node-runtime families. The ``ANTHROPIC_BASE_URL`` redirect
carries sentinel A's URL — the T-I5 helper is parameterised, so the
same function reaches a different destination for the precedence
runs.

**Layer.** No runner in the tree selects ``agent_smoke`` until T-K10
lands (KBR-119, To Do). Until then, the tests run only when a
developer explicitly invokes ``pytest -m agent_smoke``. That debt is
registered, not silent: ``PENDING_ACTIVATION_LAYERS["agent_smoke"]``
names T-K10.
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.bridge import BridgeFixture, transport
from harness.contract import WireFormat

from agent_smoke.test_claude_startup import (
    _body_has_user_message,
    _claude_binary,
    _hermetic_env,
    _run_one_turn,
)

#: The one prompt each run drives. Same shape as T-I5's :data:`_PROMPT` —
#: answerable from the canned recorder reply, no tool call, no long
#: context, so a slow or chatty binary still finishes inside the smoke
#: timeout.
_PROMPT = "Reply with the single word: ready"


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
            i.e. sentinel B in Main and Control 1.
    """
    path.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": base_url}}), encoding="utf-8")


async def _three_fixtures_up() -> tuple[BridgeFixture, BridgeFixture, BridgeFixture]:
    """Bring up the recorder, sentinel A and sentinel B on independent ports.

    Returns:
        A three-tuple ``(recorder, sentinel_a, sentinel_b)`` of started
        :class:`BridgeFixture`s. All three are bound to independent
        ephemeral loopback ports before the function returns — so a
        ``claude`` connect attempt inside the next turn never sees a
        half-up destination. The caller owns the teardown: the test
        methods' ``finally`` blocks stop all three, in the same order
        they were started, so a half-up start inside this function
        leaves the earlier fixtures running for the ``finally`` to
        release.
    """
    recorder = BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES))
    sentinel_a = BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES))
    sentinel_b = BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES))
    # Sequential starts, not concurrent: a start failure in the second
    # or third fixture would leave the first bound if the starts raced.
    await recorder.start()
    await sentinel_a.start()
    await sentinel_b.start()
    return recorder, sentinel_a, sentinel_b


async def _run_turn_and_assert_winner(
    *,
    tmp_path: Path,
    binary: Path,
    recorder: BridgeFixture,
    sentinel_a: BridgeFixture,
    sentinel_b: BridgeFixture,
    extra_args: tuple[str, ...] = (),
    write_global_settings: bool = True,
) -> None:
    """Run one turn and assert recorder wins, sentinels stay silent.

    Args:
        tmp_path: The pytest-supplied temp directory, hosting both
            ``HOME`` and ``CLAUDE_CONFIG_DIR``.
        binary: The resolved Claude Code executable.
        recorder: The fixture that should receive the user-turn POST.
        sentinel_a: The fixture wired to ``ANTHROPIC_BASE_URL`` in the
            child env. Expected to capture nothing.
        sentinel_b: The fixture wired to ``env.ANTHROPIC_BASE_URL`` in
            ``$CLAUDE_CONFIG_DIR/settings.json``. Expected to capture
            nothing.
        extra_args: Additional CLI args appended to ``claude`` after
            ``--dangerously-skip-permissions``. Main passes
            ``("--settings", <path>)``; controls pass an empty tuple.
        write_global_settings: When ``True``, writes
            ``$CLAUDE_CONFIG_DIR/settings.json`` pointing at sentinel B.
            When ``False``, the file is absent (Control 2's condition).

    Raises:
        AssertionError: When the winner captured nothing or either
            sentinel captured anything.
    """
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir(parents=True, exist_ok=True)

    if write_global_settings:
        _write_global_settings_file(config_dir / "settings.json", sentinel_b.base_url)

    env = _hermetic_env(home, config_dir, sentinel_a.base_url)
    returncode, stdout, stderr = await _run_one_turn(binary, env, home, extra_args=extra_args)
    diagnostic = f"exit={returncode} stdout={stdout[:400]!r} stderr={stderr[:400]!r}"

    recorder_captures = list(recorder.captures)
    sentinel_a_captures = list(sentinel_a.captures)
    sentinel_b_captures = list(sentinel_b.captures)

    user_captures = [
        c for c in recorder_captures
        if c.path == "/v1/messages" and _body_has_user_message(c.body)
    ]
    assert user_captures, (
        f"the recorder captured {len(recorder_captures)} request(s) but "
        f"none carried a user-role message on /v1/messages; "
        f"A captured {len(sentinel_a_captures)}, B captured "
        f"{len(sentinel_b_captures)}: {diagnostic}"
    )
    assert not sentinel_a_captures, (
        f"sentinel A (env) captured {len(sentinel_a_captures)} request(s) "
        f"but should have been silent under the precedence Claude Code "
        f"documents: {diagnostic}"
    )
    assert not sentinel_b_captures, (
        f"sentinel B (global settings) captured "
        f"{len(sentinel_b_captures)} request(s) but should have been "
        f"silent under the precedence Claude Code documents: {diagnostic}"
    )


async def _run_turn_and_assert_silent_loser(
    *,
    tmp_path: Path,
    binary: Path,
    winner: BridgeFixture,
    losers: tuple[BridgeFixture, ...],
    env_url: str,
    extra_args: tuple[str, ...] = (),
    write_global_settings: bool = True,
    winner_label: str,
) -> None:
    """Run one turn and assert the named winner receives the user turn.

    Args:
        tmp_path: The pytest-supplied temp directory.
        binary: The resolved Claude Code executable.
        winner: The fixture that should receive the user-turn POST.
        losers: The fixtures expected to capture nothing.
        env_url: The ``ANTHROPIC_BASE_URL`` value in the child env.
            For Control 2 this is sentinel A's URL (the env is the
            winner); for Control 1 it is also sentinel A's URL (and
            sentinel B wins via the global settings file).
        extra_args: Additional CLI args for ``claude``. Both control
            runs pass an empty tuple — ``--settings`` is absent on
            purpose.
        write_global_settings: Whether to write the
            ``$CLAUDE_CONFIG_DIR/settings.json`` file. Control 2
            passes ``False``.
        winner_label: Human-readable name of the winning destination,
            used in assertion messages only.

    Raises:
        AssertionError: When the winner captured nothing, any loser
            captured anything, or ``--settings`` was passed when it
            should have been absent.
    """
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir(parents=True, exist_ok=True)

    if write_global_settings:
        # The global settings file's destination is whichever fixture is
        # the winner in a control run (sentinel B in Control 1; absent
        # in Control 2). The caller-supplied winner is that fixture.
        _write_global_settings_file(config_dir / "settings.json", winner.base_url)

    env = _hermetic_env(home, config_dir, env_url)
    returncode, stdout, stderr = await _run_one_turn(binary, env, home, extra_args=extra_args)
    diagnostic = f"exit={returncode} stdout={stdout[:400]!r} stderr={stderr[:400]!r}"

    winner_captures = list(winner.captures)
    user_captures = [
        c for c in winner_captures
        if c.path == "/v1/messages" and _body_has_user_message(c.body)
    ]
    assert user_captures, (
        f"the winner ({winner_label}) captured {len(winner_captures)} "
        f"request(s) but none carried a user-role message on /v1/messages: {diagnostic}"
    )
    for label, fixture in zip(("loser A", "loser B"), losers, strict=True):
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
        # Create the temp tree before any helper writes into it —
        # ``home/`` is where the session settings file lands, and the
        # helper that writes it does not mkdir.
        home = tmp_path / "home"
        home.mkdir(parents=True, exist_ok=True)
        recorder, sentinel_a, sentinel_b = await _three_fixtures_up()
        try:
            session_settings = home / "session-settings.json"
            _write_session_settings_file(session_settings, recorder.base_url)
            await _run_turn_and_assert_winner(
                tmp_path=tmp_path,
                binary=binary,
                recorder=recorder,
                sentinel_a=sentinel_a,
                sentinel_b=sentinel_b,
                extra_args=("--settings", str(session_settings)),
                write_global_settings=True,
            )
        finally:
            for fixture in (recorder, sentinel_a, sentinel_b):
                await fixture.stop()

    async def test_control_1_global_settings_win(self, tmp_path: Path) -> None:
        """Without ``--settings``, the global settings file outranks env.

        Same two losing fixtures as Main (recorder pointed at by env,
        sentinel A pointed at by env), but ``--settings`` is absent
        and sentinel B (the global-settings destination) wins.
        """
        binary = _claude_binary()
        recorder, sentinel_a, sentinel_b = await _three_fixtures_up()
        try:
            # In Control 1 the global-settings file is the only writer,
            # so its destination (sentinel B) is the winner. The other
            # two fixtures (recorder, sentinel A) are losers.
            await _run_turn_and_assert_silent_loser(
                tmp_path=tmp_path,
                binary=binary,
                winner=sentinel_b,
                losers=(recorder, sentinel_a),
                env_url=sentinel_a.base_url,
                extra_args=(),
                write_global_settings=True,
                winner_label="sentinel B (global settings)",
            )
        finally:
            for fixture in (recorder, sentinel_a, sentinel_b):
                await fixture.stop()

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
        recorder, sentinel_a, sentinel_b = await _three_fixtures_up()
        try:
            await _run_turn_and_assert_silent_loser(
                tmp_path=tmp_path,
                binary=binary,
                winner=sentinel_a,
                losers=(recorder, sentinel_b),
                env_url=sentinel_a.base_url,
                extra_args=(),
                write_global_settings=False,
                winner_label="sentinel A (env)",
            )
        finally:
            for fixture in (recorder, sentinel_a, sentinel_b):
                await fixture.stop()
