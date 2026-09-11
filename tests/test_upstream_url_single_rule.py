"""KBR-143 — the set of modules that resolve a base URL, pinned.

Three sites composed the upstream URL independently, by the same wrong two lines.
Fixing one and missing another would leave the bug alive on whichever path was
missed, so this file pins the population: a module that resolves a base URL has to
reach for the one composition helper, and a *new* such module fails here until
somebody registers it and gives it a behavioural case of its own.

Two artifacts that must agree, both readable statically — the `TEST_SUITE.md` §6.2
case — so the file is L2 and is named in ``tests/test_layer_selection.py``'s
allowlist.  The behavioural half lives at L1, in
``tests/providers/test_upstream_url_composition.py``, where mutation testing judges
it (§2.1).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.l2

_SRC = Path(__file__).resolve().parents[1] / "src" / "kitty"

# A *call on an adapter* -- `provider.build_base_url(...)`. The leading dot is what
# makes this a call rather than a mention: without it the pattern also matches the
# comment in `providers/base.py` that names the method in prose, which would demand
# the helper from a module that merely talks about it.
_CALLS_BUILD_BASE_URL = re.compile(r"\.build_base_url\s*\(")

# The modules allowed to resolve a base URL, each with a behavioural case at L1.
# `providers/ollama_cloud.py` is deliberately absent: it owns its transport and reads
# `provider_config["base_url"]` directly, so this scan cannot see it. That is the
# reason the L1 file asserts all three sites rather than trusting this guard alone.
_EXPECTED_CONSUMERS = {"server.py", "validation.py"}


def _modules_resolving_a_base_url() -> set[str]:
    """Return the name of every module under ``src/kitty`` that calls ``build_base_url``.

    Returns:
        The module file names.  Modules that only *define* the method are excluded by
        the leading dot, and so is the one that names it in a comment.
    """
    return {
        path.name
        for path in sorted(_SRC.rglob("*.py"))
        if _CALLS_BUILD_BASE_URL.search(path.read_text(encoding="utf-8"))
    }


def test_the_scan_finds_its_known_positives():
    """The scan must find the call sites it exists to check, or it proves nothing.

    Per `TEST_SUITE.md` §6.2, a structural guard asserts its own scan works.  A regex
    that matched nothing would make both assertions below vacuously true.
    """
    assert _modules_resolving_a_base_url() >= _EXPECTED_CONSUMERS


def test_no_new_module_resolves_a_base_url_unnoticed():
    """The population is pinned in both directions.

    The reverse direction is the one that earns its keep: a fourth site resolving a
    base URL is how this bug would come back, and it fails here until it is added to
    ``_EXPECTED_CONSUMERS`` — which is the moment to give it a behavioural case.
    """
    assert _modules_resolving_a_base_url() == _EXPECTED_CONSUMERS


@pytest.mark.parametrize("module_name", sorted(_EXPECTED_CONSUMERS))
def test_every_consumer_reaches_the_shared_helper(module_name: str):
    """A module that resolves a base URL must not join it to a path by itself.

    A containment check, which is weak on its own in a file of several thousand lines
    — it cannot tell one call site from another.  What makes it sound is the pairing:
    this says each consumer reaches the helper, and the L1 file says each consumer
    produces what the helper produces.

    Args:
        module_name: The file name of the consumer under test.
    """
    path = next(p for p in _SRC.rglob("*.py") if p.name == module_name)

    assert "compose_upstream_url" in path.read_text(encoding="utf-8")
