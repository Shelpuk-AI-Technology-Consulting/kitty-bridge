#!/usr/bin/env python3
"""Verify that ``ci-required`` is the required status check on ``main``.

KBR-152's fix is a live repository setting — a ``required_status_checks``
rule on ruleset ``21038306`` naming the ``ci-required`` aggregate and
nothing else — not a file in this repository, so nothing in the suite can
see it. The ticket's own acceptance procedure (a throwaway pull request
with a deliberately failing job that must refuse to merge) exercises the
setting once and then deletes its evidence. This script is the cheap,
re-runnable witness: one ``gh api`` call, one verdict, one exit code.

Run it after any manual edit to the repository's rulesets, and whenever
there is reason to doubt the gate:

    gh auth status        # the caller needs a token that can read the repo
    .venv/bin/python scripts/verify_required_status_checks.py

Exit codes: 0 — ``ci-required`` is the required status check and the only
one; 1 — it is not (or the rules list carries any other shape this
verifier cannot read as enforced); 2 — the ``gh`` read itself failed,
which is not a verdict about the setting and must never be read as one.
"""

from __future__ import annotations

import json
import subprocess
import sys

#: The single status check the branch protection must require. The
#: aggregate, not the individual jobs: naming ``test / Python 3.x``
#: directly would re-create the drift the aggregate exists to prevent —
#: adding or renaming a matrix job would then mean editing branch
#: protection.
REQUIRED_CHECK_CONTEXT = "ci-required"

#: The endpoint that reports the *effective* rules on a branch — merged
#: across every ruleset that targets it, so an org-sourced ruleset cannot
#: quietly carry the requirement this one lacks (or the reverse).
_BRANCH_RULES_ENDPOINT = "repos/{owner}/{repo}/rules/branches/main"

_REPOSITORY = "Shelpuk-AI-Technology-Consulting/kitty-bridge"


def ci_required_is_enforced(rules: list) -> bool:
    """Decide whether the branch rules require ``ci-required`` and nothing else.

    Args:
        rules: The parsed body of
            ``GET /repos/{owner}/{repo}/rules/branches/main`` — the
            *effective* rules on the branch, one mapping per rule.

    Returns:
        ``True`` only when exactly one rule of type
        ``required_status_checks`` is present, its ``parameters`` are a
        mapping, and its check list is exactly one entry naming
        :data:`REQUIRED_CHECK_CONTEXT`. Every other shape — including
        shapes a future GitHub revision might introduce — reads as
        ``False``: a verifier that guessed at a shape it did not
        recognise would bless a half-configured gate.
    """
    # "Exactly one" rather than "at least one": the endpoint returns the
    # rules already merged across rulesets, so a second row of the type is
    # not a stricter gate — it is an input this verifier was not written
    # for, and trusting either row would be a guess.
    matching = [
        rule
        for rule in rules
        if isinstance(rule, dict) and rule.get("type") == "required_status_checks"
    ]
    if len(matching) != 1:
        return False

    # Each level unreadable reads as absent, the same fail-safe the review
    # suite's ``_matrix_values`` applies to a matrix it cannot parse.
    parameters = matching[0].get("parameters")
    if not isinstance(parameters, dict):
        return False
    checks = parameters.get("required_status_checks")
    if not isinstance(checks, list) or len(checks) != 1:
        return False

    only = checks[0]
    return isinstance(only, dict) and only.get("context") == REQUIRED_CHECK_CONTEXT


def _gh_branch_rules() -> list:
    """Read the effective branch rules for :data:`_REPOSITORY` via ``gh``.

    Returns:
        The parsed rule list.

    Raises:
        subprocess.CalledProcessError: ``gh`` is missing, unauthenticated,
            or the API rejects the request.
        json.JSONDecodeError: The response is not the JSON shape the
            endpoint documents.
    """
    endpoint = _BRANCH_RULES_ENDPOINT.format(
        owner=_REPOSITORY.split("/")[0], repo=_REPOSITORY.split("/")[1]
    )
    result = subprocess.run(
        ["gh", "api", endpoint], capture_output=True, text=True, check=True
    )
    parsed = json.loads(result.stdout)
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON list from {endpoint}, got {type(parsed).__name__}")
    return parsed


def main() -> int:
    """Read the live rules and print the verdict.

    Returns:
        0 when :func:`ci_required_is_enforced` holds on the live rules,
        1 when it does not, 2 when the read itself failed — a state the
        caller must never read as a verdict about the setting.
    """
    # The whole call is wrapped so that an outage, a missing token and a
    # stale endpoint all land on exit 2 rather than a traceback: the
    # script's one job is a verdict, and "could not read" is not one.
    try:
        rules = _gh_branch_rules()
    except Exception as error:  # noqa: BLE001 - see block comment above
        print(f"could not read the branch rules: {error}", file=sys.stderr)
        return 2

    if ci_required_is_enforced(rules):
        print(f"enforced: {REQUIRED_CHECK_CONTEXT!r} is the required status check")
        return 0

    print(
        f"NOT enforced: no rule requires exactly {REQUIRED_CHECK_CONTEXT!r}. "
        "A pull request with a red gate can be merged — see KBR-152.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
