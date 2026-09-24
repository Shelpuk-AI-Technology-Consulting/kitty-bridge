"""Mutmut test-selection equals the §8.2 socket-binding registry.

`.system_design/TEST_SUITE.md` §8.2 enumerates seventeen modules that bind real
sockets or spawn real processes and so are ``l1`` by path default today.
KBR-290 removes them from **mutmut's** test selection (the Fast job's
``pytest -m "l1 or l2"`` selection is unchanged) so the nightly mutation run
never depends on loopback socket timing. The shape is a one-line
``--ignore <path>`` row per module in :mod:`tests.socket_binding_l1_modules`,
the registry KBR-272's timeout-mark guard also reads.

This guard is the machine-readable pin that holds ``pyproject.toml``'s
``[tool.mutmut] pytest_add_cli_args_test_selection`` against the registry, in
both directions and on the full selection list (not only the ignore set):

* **Forward** -- every registry module has a matching ``--ignore`` row, in the
  registry's order. A missing row fails the guard naming the absent path.
* **Reverse** -- pyproject carries no ``--ignore`` row that is not in the
  registry. A phantom row fails the guard naming the unexpected path.
* **Fixed prefix** -- the selection still starts with ``-m l1 --ignore
  tests/test_internal_keys_not_sent_upstream.py``, so a future hand-edit
  cannot silently drop ``-m l1`` and re-widen mutmut to the whole suite.

**Layer.** L2 -- the subject is a structural artifact (a config-style
selection) that must agree with the registry, both edited by hand. The two
halves of the ``l1 or l2`` job.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from socket_binding_l1_modules import SOCKET_BINDING_L1_MODULES

pytestmark = pytest.mark.l2

REPO_ROOT = Path(__file__).resolve().parent.parent

# tomllib landed in 3.11; the 3.10 leg of the test matrix needs ``tomli``
# (declared as a conditional dev extra in pyproject.toml).
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


# The fixed prefix every mutmut invocation in this repo carries. A change to
# either element is a substantive config change; the assertion below catches
# silent drift in either direction. ``tests/test_internal_keys_not_sent_upstream.py``
# stays ignored because mutmut's clean-run context trips it (see
# ``.system_design/TEST_SUITE.md`` §6.1).
_MUTMUT_SELECTION_PREFIX: tuple[str, ...] = (
    "-m",
    "l1",
    "--ignore",
    "tests/test_internal_keys_not_sent_upstream.py",
)


def _expected_selection() -> list[str]:
    """Build the expected ``pytest_add_cli_args_test_selection`` list.

    Returns:
        The fixed prefix followed by one ``--ignore <path>`` pair per
        registry module, in the registry's order.
    """
    expected: list[str] = []
    expected.extend(_MUTMUT_SELECTION_PREFIX)
    for path in SOCKET_BINDING_L1_MODULES:
        expected.append("--ignore")
        expected.append(path)
    return expected


def _read_pyproject_selection() -> list[str]:
    """Read ``[tool.mutmut] pytest_add_cli_args_test_selection`` from pyproject.toml.

    Returns:
        The selection list exactly as the file carries it, order preserved.

    Raises:
        KeyError: When the ``[tool.mutmut]`` table or the selection key is
            missing -- a structural config change the guard must report.
        tomllib.TOMLDecodeError: When ``pyproject.toml`` is not valid TOML.
    """
    pyproject = REPO_ROOT / "pyproject.toml"
    with pyproject.open("rb") as fp:
        data = tomllib.load(fp)
    return list(data["tool"]["mutmut"]["pytest_add_cli_args_test_selection"])


def test_mutmut_test_selection_equals_registry() -> None:
    """pyproject's mutmut selection equals the registry-derived expected list, in order.

    Asserts equality in both directions: a missing ``--ignore`` row, an extra
    ``--ignore`` row, a missing ``-m l1``, or an out-of-order entry each fail
    with the offending component named in the failure message.
    """
    expected = _expected_selection()
    actual = _read_pyproject_selection()

    if actual != expected:
        # Surface the first divergence position with both sides.
        for index, (exp, act) in enumerate(zip(expected, actual, strict=False)):
            if exp != act:
                pytest.fail(
                    f"pyproject.toml's [tool.mutmut] pytest_add_cli_args_test_selection "
                    f"diverges at index {index}: expected {exp!r}, got {act!r}; "
                    f"full expected: {expected}, full actual: {actual}"
                )
        # Lengths differ -- name both sides' surplus so a missing ignore and a
        # phantom ignore each report what they hold, not what the other lacks.
        missing_from_actual = expected[len(actual):]
        extra_in_actual = actual[len(expected):]
        pytest.fail(
            f"pyproject.toml's [tool.mutmut] pytest_add_cli_args_test_selection "
            f"length differs: expected {len(expected)} entries, got {len(actual)}; "
            f"missing from pyproject: {missing_from_actual}; "
            f"extra in pyproject: {extra_in_actual}"
        )


def test_every_registry_module_resolves_against_live_source() -> None:
    """Each registry entry points at an existing file on disk.

    Catches a fat-fingered path that pyproject would happily mirror: the
    equality assertion above could pass for two equal-but-bogus sides, this
    precondition stops that drift.
    """
    for path in SOCKET_BINDING_L1_MODULES:
        assert path.startswith("tests/"), f"registry entry {path!r} must be under tests/"
        assert (REPO_ROOT / path).is_file(), f"registry entry {path!r} does not exist on disk"


def test_registry_has_exactly_seventeen_entries() -> None:
    """The count is fixed by §8.2's enumeration of seventeen modules."""
    assert len(SOCKET_BINDING_L1_MODULES) == 17, (
        f"registry has {len(SOCKET_BINDING_L1_MODULES)} entries; §8.2 enumerates seventeen"
    )
