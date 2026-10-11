"""The hypothesis profile for runs under mutmut, and the switch that loads it.

mutmut 3.8.0 runs stats collection and the clean tests as two in-process
``pytest.main()`` calls (``ForkServerRunner`` -> ``PytestRunner.run_tests`` ->
``pytest.main``). The second run reuses the first run's imported test modules
from ``sys.modules``, so a ``@given``-decorated *method* runs a second time
bound to a new instance, and ``HealthCheck.differing_executors`` fires: the
check keeps the first run's bound self in a per-function ``thread_local``
(hypothesis ``core.py``, the ``prev_self`` comparison). Free ``@given``
functions never fire (both runs see ``selfy=None``); the first
method-decorated property test in the selection is where the clean phase
dies. KBR-272 hit exactly that on the first CI-hosted mutmut run
(``TestIdentityBelowBudget.test_cc_bodies_below_budget_are_unchanged``,
CI runs 35399984387 and 35402669830, reproduced locally).

The condition is manufactured by mutmut's execution model, not by the tests,
which is the case hypothesis's own message names as safe to suppress. The
suppression is therefore scoped by environment, not by editing the property
tests: mutmut exports ``MUTMUT_DEPENDENCY_DEPTH`` before its first pytest
invocation and nothing else sets it, so the profile loads there and nowhere
else. ``tests/conftest.py`` calls :func:`apply` at import time, after the
``kitty-bridge-ci`` block and before any test module's ``@settings(...)``
snapshots the default profile.

The profile is built as a *child of the currently active* default -- the
``kitty-bridge-ci`` profile on CI, hypothesis's defaults on a developer
machine -- so a mutmut run keeps that profile's derandomisation, deadline
and database behaviour. An explicit ``suppress_health_check`` replaces the
parent's list rather than extending it, so the parent's suppressed checks
are unioned back in.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from hypothesis import HealthCheck, settings

#: The one variable mutmut exports into every pytest invocation it spawns.
#: The repo reads it as the "this run is mutmut's" signal; a rename upstream
#: would silently stop the profile from loading, which
#: ``tests/test_hypothesis_mutmut_profile.py`` pins.
MUTMUT_ENV_VAR = "MUTMUT_DEPENDENCY_DEPTH"

#: The profile's registered name, loaded by ``tests/conftest.py``.
PROFILE_NAME = "mutmut"


def mutmut_detected(env: Mapping[str, str] | None = None) -> bool:
    """Return whether this run is a mutmut-spawned pytest invocation.

    Args:
        env: The environment to inspect; ``os.environ`` when ``None``.

    Returns:
        ``True`` when mutmut's dependency-depth variable is present. Presence
        rather than a value check: mutmut writes a plain integer, and a
        future format change must fail open (suppression active) rather than
        fail the mutation run.
    """
    environment = os.environ if env is None else env
    return MUTMUT_ENV_VAR in environment


def build_profile(parent: settings | None = None) -> settings:
    """Build the mutmut hypothesis profile as a child of ``parent``.

    Args:
        parent: The settings whose behaviour the mutmut run should keep --
            ``settings.default`` when ``None``, i.e. whatever profile
            ``tests/conftest.py`` loaded before this one (``kitty-bridge-ci``
            on CI, hypothesis's defaults locally).

    Returns:
        A settings object suppressing exactly ``differing_executors`` *plus*
        whatever the parent already suppressed. Every unsuppressed parent
        health check remains active; the union is pinned by the registry
        test.
    """
    base = settings.default if parent is None else parent
    # sorted() only to make the suppression list deterministic; HealthCheck
    # members do not order among themselves, so the sort key is the name.
    merged = sorted(
        set(base.suppress_health_check) | {HealthCheck.differing_executors},
        key=lambda check: check.name,
    )
    return settings(base, suppress_health_check=merged)


def apply(env: Mapping[str, str] | None = None) -> None:
    """Load the mutmut profile when running under mutmut.

    Args:
        env: The environment to inspect; ``os.environ`` when ``None``.

    Must run before test modules are imported (``tests/conftest.py`` does),
    because a ``@settings(...)`` decorator snapshots the default profile's
    values at decoration time -- a later ``load_profile`` would not reach
    tests already decorated. It must also run *after* the other profile
    blocks, so this one parents from the environment they built.
    """
    if mutmut_detected(env):
        settings.register_profile(PROFILE_NAME, build_profile())
        settings.load_profile(PROFILE_NAME)
