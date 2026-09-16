"""KBR-152 acceptance canary — deliberately fails, on a deliberately throwaway branch.

This module exists to prove that a pull request whose ``test / Python 3.x``
leg is red cannot be merged now that ``ci-required`` is a required status
check on ``main`` (KBR-152). It lives on
``throwaway/kbr-152-required-check-fails`` only and is deleted with that
branch once the refusal has been observed.
"""


def test_kbr152_acceptance_canary_deliberately_fails() -> None:
    """Fail on purpose: KBR-152's acceptance needs a red required check."""
    assert False, "deliberate failure — KBR-152 acceptance canary"
