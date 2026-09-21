"""Agent-boundary hermetic tests — TEST_SUITE.md §6.4.2's startup smoke.

Plan **T-I5** (KBR-97) and **T-I6** (KBR-98).

This directory holds the tests that drive a **real pinned agent binary**
against the bridge fixture with no provider, no credentials and no
network beyond loopback. Four test classes ship with the package
(eight test functions in total across T-I5 and T-I6):
:class:`~tests.agent_smoke.test_claude_startup.TestAgentStartupSmoke`
(the connectivity claim), :class:`TestMissingBinaryFalsification`
(the §8 missing-binary rule, KBR-132 shape, on both the override
branch and the default-lookup branch), and :class:`TestTimeoutKill`
(the safety-critical branch that reaps a hung child) — those three
are T-I5. T-I6's
:class:`~tests.agent_smoke.test_claude_settings_precedence.TestSettingsPrecedence`
adds the three-run settings-precedence claim (Main + Control 1 +
Control 2, every destination demonstrated live); it inherits the
bridge fixture from :mod:`harness.bridge` and reuses T-I5's
:func:`~tests.agent_smoke.test_claude_startup._claude_binary`,
:func:`~tests.agent_smoke.test_claude_startup._hermetic_env`,
:func:`~tests.agent_smoke.test_claude_startup._body_has_user_message`,
and the ``--settings``-aware extension of
:func:`~tests.agent_smoke.test_claude_startup._run_one_turn`. The
placement of T-I6 in this package is settled by T-I5's step file
(`.system_design/steps/t_i5_agent_startup_smoke.md`), which states
T-I6's three-sentinel design inherits T-I5's bridge fixture and test
harness. The live nightly scenarios (T-K11, T-I14) still belong
elsewhere and inherit the bridge fixture from :mod:`harness.bridge`
rather than this package.

**Layer wiring.** ``tests/layers.py::_PATH_DEFAULTS`` carries the
``("tests/agent_smoke/", "agent_smoke")`` row, so no file here needs a
``pytestmark`` — the same mechanism ``tests/acceptance/`` and
``tests/integration/`` use. ``agent_smoke`` is in
:attr:`layers.RESOURCE_DEPENDENT_LAYERS`, so the default ``pytest``
invocation excludes it, and in ``PENDING_ACTIVATION_LAYERS`` keyed to
plan task T-K10 until its CI job is activated.

**Why a package and not loose modules.** ``tests/`` has no ``__init__.py``,
so pytest inserts ``tests/`` onto ``sys.path`` and these modules import
as ``agent_smoke.<name>`` when needed from a sibling test module — and
mypy resolves sibling-package imports like ``harness.bridge`` from a
directory that carries this file. Adding a ``tests/__init__.py`` would
break the bare-``pytest`` import path that the whole suite relies on;
see :mod:`harness`'s own docstring for the full reasoning.
"""
