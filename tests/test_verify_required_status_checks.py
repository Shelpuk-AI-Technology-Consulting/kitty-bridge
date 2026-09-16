"""KBR-152 — the required-status-checks verifier is not rot.

``scripts/verify_required_status_checks.py`` reads the live branch-rules
JSON GitHub returns for ``main`` and answers one question: is
``ci-required`` the required status check, and the only one? The script
is the offline witness for a live repository setting that the ticket's
own acceptance procedure (a throwaway pull request that must refuse to
merge) only exercises once.

Twelve cases pin the contract:

* the pre-change shape — three rules, no ``required_status_checks``
  row — reads as **not** enforced (the defect KBR-152 filed);
* the post-change shape — the fourth row naming exactly
  ``ci-required`` — reads as enforced;
* every partial shape GitHub could return reads as **not** enforced:
  an empty check list, missing ``parameters``, a wrong context, the
  right context beside a second one, two ``required_status_checks``
  rows, a non-dict parameters value, a non-dict check entry, a rule
  that is not a dict at all;
* ``main``'s three exit codes — 0 (enforced), 1 (not enforced), 2
  (the ``gh`` call itself failed).

**Layer.** L2 — the subject is a script outside ``src/kitty`` that
consumes a data format GitHub defines and edits, held against inline
snapshots of that format. The same shape as
``tests/test_aggregate_mutation_baseline.py``, for an API JSON instead
of mutmut's ``.meta``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.l2


_REPO_ROOT = Path(__file__).resolve().parent.parent

# Load the verifier by file path rather than package name: `scripts/` is
# not a package and is deliberately NOT added to `sys.path` — one file,
# not a whole directory of script entry points. The module must be
# registered in ``sys.modules`` before ``exec_module`` — Python 3.13's
# ``@dataclass`` resolves the class's namespace through ``sys.modules``
# while decorating (the reason ``test_aggregate_mutation_baseline.py``
# gives for the same two lines).
_spec = importlib.util.spec_from_file_location(
    "verify_required_status_checks",
    _REPO_ROOT / "scripts" / "verify_required_status_checks.py",
)
assert _spec is not None and _spec.loader is not None
verifier = importlib.util.module_from_spec(_spec)
sys.modules["verify_required_status_checks"] = verifier
_spec.loader.exec_module(verifier)
del _spec


#: The three rule types `main` carried before KBR-152, as the live API
#: reported them. Synthesised here rather than read from the snapshot in
#: `.requirements/`: that folder is gitignored, and a test that only ran
#: where a private file exists would pass on every runner but the ones
#: it is for.
_PRE_CHANGE_RULES = [
    {"type": "deletion", "ruleset_id": 21038306},
    {"type": "non_fast_forward", "ruleset_id": 21038306},
    {
        "type": "pull_request",
        "ruleset_id": 21038306,
        "parameters": {"required_approving_review_count": 0},
    },
]


def _with_required_check(contexts: list[str]) -> list[dict]:
    """Return the pre-change rules with one ``required_status_checks`` row added.

    Args:
        contexts: The ``context`` values the row's check list carries.

    Returns:
        A rule list shaped as ``GET /repos/{owner}/{repo}/rules/branches/main``
        would return it after adding such a row.
    """
    return _PRE_CHANGE_RULES + [
        {
            "type": "required_status_checks",
            "ruleset_id": 21038306,
            "parameters": {
                "required_status_checks": [{"context": c} for c in contexts],
                "strict_required_status_checks_policy": False,
            },
        }
    ]


class TestCiRequiredIsEnforced:
    """The decision over the branch-rules list, every shape GitHub can send."""

    def test_the_pre_change_state_reads_as_not_enforced(self) -> None:
        """The defect KBR-152 filed: no ``required_status_checks`` row at all."""
        assert verifier.ci_required_is_enforced(_PRE_CHANGE_RULES) is False

    def test_the_post_change_state_reads_as_enforced(self) -> None:
        """Exactly one row naming exactly ``ci-required`` is the fix."""
        assert verifier.ci_required_is_enforced(_with_required_check(["ci-required"])) is True

    def test_an_empty_check_list_reads_as_not_enforced(self) -> None:
        """A row of the right type that requires nothing requires nothing."""
        assert verifier.ci_required_is_enforced(_with_required_check([])) is False

    def test_a_wrong_context_reads_as_not_enforced(self) -> None:
        """Any other single context is somebody else's gate, not ours."""
        assert verifier.ci_required_is_enforced(_with_required_check(["other-check"])) is False

    def test_ci_required_beside_a_second_context_reads_as_not_enforced(self) -> None:
        """The ticket names ``ci-required`` *and nothing else*.

        A second context re-creates the drift the aggregate exists to
        prevent: adding or renaming a matrix job would then mean editing
        branch protection. The verifier must call that state out rather
        than bless it because the wanted context is in the list.
        """
        assert verifier.ci_required_is_enforced(
            _with_required_check(["ci-required", "other-check"])
        ) is False

    def test_two_rows_of_the_type_read_as_not_enforced(self) -> None:
        """The effective-rules list is GitHub's, not ours — ambiguity fails safe.

        Two rows of one type should not reach this function (the endpoint
        returns the *effective* rules, already merged across rulesets),
        so a second row means the input is not the shape the verifier
        was written for. Reading it as "enforced" would trust whichever
        row the loop happened to land on.
        """
        rules = _with_required_check(["ci-required"]) * 2
        # ``* 2`` duplicates the whole row, so the pair is identical.
        # Two rows of one type should not reach this function — a guard
        # on the count would only confirm the input shape we are already
        # refusing, not the verdict.
        assert verifier.ci_required_is_enforced(rules) is False

    def test_missing_parameters_reads_as_not_enforced(self) -> None:
        """A row without ``parameters`` requires nothing."""
        rules = _PRE_CHANGE_RULES + [{"type": "required_status_checks"}]
        assert verifier.ci_required_is_enforced(rules) is False

    def test_non_dict_parameters_reads_as_not_enforced(self) -> None:
        """A row whose ``parameters`` is not a mapping is an unreadable shape."""
        rules = _PRE_CHANGE_RULES + [
            {"type": "required_status_checks", "parameters": ["ci-required"]}
        ]
        assert verifier.ci_required_is_enforced(rules) is False

    def test_non_dict_check_entries_read_as_not_enforced(self) -> None:
        """A check list whose entries are bare strings is not the API's shape."""
        rules = _PRE_CHANGE_RULES + [
            {
                "type": "required_status_checks",
                "ruleset_id": 21038306,
                "parameters": {"required_status_checks": ["ci-required"]},
            }
        ]
        assert verifier.ci_required_is_enforced(rules) is False

    def test_a_rule_that_is_not_a_dict_is_skipped_without_crashing(self) -> None:
        """Malformed input reads as not enforced, not as an exception.

        The verifier runs against a live API. Whatever GitHub sends back,
        ``main`` must print a verdict and set an exit code; a JSON body
        that is not a list of dicts is a verdict of "not enforced".
        """
        assert verifier.ci_required_is_enforced(["not-a-dict", None, 7]) is False


class TestMainExitCodes:
    """``main``'s contract: 0 enforced, 1 not enforced, 2 the read itself failed."""

    def _run_main_with_rules(self, monkeypatch: pytest.MonkeyPatch, rules: list | object) -> int:
        """Drive ``main`` with a stubbed ``gh`` read and return its exit code.

        Args:
            monkeypatch: pytest's monkeypatch fixture.
            rules: The value the stubbed ``_gh_branch_rules`` returns —
                a rule list, or an exception to raise.

        Returns:
            ``main``'s exit code.
        """

        def fake_read() -> object:
            if isinstance(rules, Exception):
                raise rules
            return rules

        monkeypatch.setattr(verifier, "_gh_branch_rules", fake_read)
        return verifier.main()

    def test_exit_zero_when_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._run_main_with_rules(
            monkeypatch, _with_required_check(["ci-required"])
        ) == 0

    def test_exit_one_when_not_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._run_main_with_rules(monkeypatch, _PRE_CHANGE_RULES) == 1

    def test_exit_two_when_the_gh_call_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed read is not a verdict — exit 2, never 0 or 1.

        Conflating "could not read" with "not enforced" would let an
        outage of the API masquerade as a findings-free gate.
        """
        assert self._run_main_with_rules(monkeypatch, RuntimeError("gh not found")) == 2

    def test_main_parses_the_json_the_runner_shows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The shell-out layer, not just the decision: JSON in, verdict out.

        ``_gh_branch_rules`` is stubbed at the JSON-string boundary in the
        other cases; this one stubs the process boundary instead, so the
        parse the script actually performs is on the record too.
        """
        calls: list[list[str]] = []

        def fake_run(argv: list[str], **_: object) -> object:
            calls.append(argv)

            class Completed:
                returncode = 0
                stdout = json.dumps(_with_required_check(["ci-required"]))
                stderr = ""

            return Completed()

        monkeypatch.setattr(verifier.subprocess, "run", fake_run)
        assert verifier.main() == 0
        assert calls and "rules/branches" in calls[0][2]
