"""The mutmut hypothesis profile: suppression scoped to mutmut runs only.

mutmut 3.8.0 runs stats collection and the clean tests as two in-process
``pytest.main()`` calls. The second run reuses the first run's imported test
modules from ``sys.modules``, so every ``@given``-decorated *method* runs a
second time bound to a new instance, and hypothesis's
``HealthCheck.differing_executors`` fires: its per-function ``thread_local``
holds the first run's bound self. Free ``@given`` functions never fire (both
runs see ``selfy=None``); the first method-decorated property test in the
selection is where the clean phase dies (KBR-272, CI run 35402669830).

The condition is an artifact of mutmut's execution model, not a defect in
the tests -- the case hypothesis's own message names as safe to suppress.
This module pins the suppression to be exactly that narrow: it activates
only under mutmut's environment and suppresses only the one health check.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings
from hypothesis_mutmut_profile import (
    MUTMUT_ENV_VAR,
    PROFILE_NAME,
    build_profile,
    mutmut_detected,
)

UNSET_ENV: dict[str, str] = {}
MUTMUT_ENV: dict[str, str] = {MUTMUT_ENV_VAR: "2"}


def test_mutmut_detected_reads_only_its_own_variable() -> None:
    """The env predicate keys on mutmut's variable and nothing else.

    A gate run, a developer run, or a run under any other tool's environment
    must not pick the profile up: the suppression exists for mutmut's
    in-process double ``pytest.main``, which no other runner performs.
    """
    assert mutmut_detected(UNSET_ENV) is False
    assert mutmut_detected({"PATH": "/usr/bin"}) is False
    assert mutmut_detected(MUTMUT_ENV) is True
    assert mutmut_detected({MUTMUT_ENV_VAR: ""}) is True


def test_the_profile_suppresses_only_the_one_health_check() -> None:
    """Narrowness is the property: the profile adds exactly one suppression.

    The assertion is relative to the ambient default, not an exact set:
    whatever profile this session loaded (hypothesis's defaults on a
    developer machine, ``kitty-bridge-ci`` under ``CI``, the mutmut profile
    itself under mutmut) is the parent ``build_profile()`` layers from, and
    its suppressions legitimately flow through. What must hold in every
    environment is that the profile suppresses the ambient set plus
    ``differing_executors`` and adds nothing beyond that. The first draft of
    this test asserted an exact set and passed locally while failing every
    CI leg -- the ambient default there was a loaded profile, and the union
    it produced was the profile working, not a defect.
    """
    parent = settings()  # child of the ambient default, whatever it is
    profile = build_profile(parent=parent)
    required = set(parent.suppress_health_check) | {HealthCheck.differing_executors}
    assert required <= set(profile.suppress_health_check)
    added = set(profile.suppress_health_check) - set(parent.suppress_health_check)
    assert added <= {HealthCheck.differing_executors}


def test_the_profile_layers_over_the_active_default() -> None:
    """A parent's own suppressions survive the union, and the parent's
    behaviour carries.

    On CI the active default is ``kitty-bridge-ci``, which suppresses
    ``too_slow`` and derandomises; the mutmut profile must keep those, and
    an explicit ``suppress_health_check`` replacing the parent's list is
    exactly the bug this guards against.
    """
    parent = settings(suppress_health_check=[HealthCheck.too_slow], derandomize=True)
    layered = build_profile(parent=parent)
    assert set(layered.suppress_health_check) == {
        HealthCheck.too_slow,
        HealthCheck.differing_executors,
    }
    assert layered.derandomize is True


def test_the_profile_name_and_env_var_are_stable_strings() -> None:
    """The two identifiers the rest of the repo may cite stay pinned.

    ``tests/conftest.py`` loads this profile by name and the baseline doc
    records the variable; a silent rename of either would desynchronize
    three places that were never imported from one another.
    """
    assert MUTMUT_ENV_VAR == "MUTMUT_DEPENDENCY_DEPTH"
    assert PROFILE_NAME == "mutmut"
