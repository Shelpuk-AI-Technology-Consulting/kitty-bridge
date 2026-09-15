"""T-H1 (KBR-88) — the mutation scope registry agrees with the source.

``.system_design/TEST_SUITE.md`` §6.1 names the code mutation testing
covers as a table of function wildcards. The machine-readable form of
that table is :mod:`tests.mutmut_scope`; this file is the guard that
the table is not rot.

Three kinds of deliberate defect the guard must catch — each is exercised
against an injected target, not the real registry, so a guard that is
broken in any of three specific ways fails its own self-check:

* a module path that no longer imports,
* a class name that has been removed or renamed,
* a method name that has been removed or renamed.

The §6.1 table has been wrong before — KBR-134 and KBR-151 each had to
add scope after the fact — and the §6.1 doc itself records an earlier
"excluded ``server.py``" correction. The guard is the mechanism that
stops the next drift from being silent.

**Layer.** L2 — the subject is a structural artifact (a config-like
registry) that must agree with the source it scopes, both edited by
hand. The two halves of the ``l1 or l2`` job.
"""

from __future__ import annotations

import fnmatch

import pytest
from mutmut_scope import (
    CLS,
    TARGET_GROUPS,
    Target,
    all_targets,
    mangled_patterns,
    patterns_for,
    resolve_target,
)

pytestmark = pytest.mark.l2

# Known-positive anchor: a real mangled mutant key for a target already
# in the registry, from a mutmut run that did generate it. The whole-module
# pattern ``kitty.validation.*`` must match it — this proves the *module
# path* part of the pattern derivation, not the *mangler prefix* (fnmatch's
# ``*`` absorbs the prefix, so a mangler drift would still match). The
# specific-function pin below catches mangler drift; the two assertions
# together cover what one alone cannot.
# The key was sampled from mutants/src/kitty/validation.py.meta on
# 2026-09-15 (T-H1 trial run); it is the first entry in that file.
KNOWN_VALIDATION_MUTANT_KEY = (
    "kitty.validation.x__unusable_url_result__mutmut_1"
)

# Specific-function pin (catches mangler-prefix drift). The registry's
# ``openai_subscription`` group carries ``_convert_content_types`` as a
# module-level function; its derived pattern is therefore
# ``kitty.providers.openai_subscription.x__convert_content_types__mutmut_*``.
# Character-level equality to the derived pattern pins the mangler rule
# exactly (any change to ``mangled_patterns``'s top-level branch would
# change this string). The negative control below rejects the unmangled
# name, proving the pattern distinguishes mangled from unmangled — a
# mutmut that dropped the ``x_`` prefix would emit
# ``kitty.providers.openai_subscription._convert_content_types__mutmut_N``,
# which the pattern must NOT fnmatch.
EXPECTED_SPECIFIC_PATTERN = (
    "kitty.providers.openai_subscription.x__convert_content_types__mutmut_*"
)
UNMANGLED_ORIGINAL_NAME = (
    "kitty.providers.openai_subscription._convert_content_types"
)


def test_every_registry_target_resolves_against_live_source() -> None:
    """Each (module, class, method) row points at a real symbol."""
    for _group, target in all_targets():
        resolve_target(target)  # AssertionError names the failing row


def test_whole_module_pattern_matches_a_real_mutant_key() -> None:
    """A whole-module pattern in the registry fnmatches a real mutant key.

    Proves only the *module path* part of the pattern derivation. The
    ``supporting`` group's ``kitty.validation`` entry produces the
    pattern ``kitty.validation.*``; fnmatch's ``*`` absorbs the
    ``x_`` / ``xǁ`` mangler prefix, so a mutmut that dropped the
    prefix would still match. The mangler-prefix pin lives in the next
    test; both are needed because neither one alone covers what the
    previous draft claimed this test did.
    """
    pattern = patterns_for("supporting")
    assert any(
        fnmatch.fnmatch(KNOWN_VALIDATION_MUTANT_KEY, p) for p in pattern
    ), (
        f"pattern set for `supporting` group does not fnmatch the known "
        f"validation mutant key {KNOWN_VALIDATION_MUTANT_KEY!r}; "
        f"got {pattern!r}"
    )


def test_specific_function_pattern_is_pinned_and_rejects_unmangled() -> None:
    """A specific-function pattern is character-equal to the derivation
    AND rejects the unmangled original name.

    The derivation in :func:`mutmut_scope.mangled_patterns` for a
    top-level function appends ``x_`` to the function name. A change
    to that rule (a future mutmut that drops the prefix, or a typo in
    the derivation) changes this exact string; the character-equality
    assertion catches it. The negative control then proves the pattern
    is doing useful work — it would be worthless if it matched the
    unmangled name, since mutmut never emits unmangled names.
    """
    derived = patterns_for("openai_subscription")
    assert EXPECTED_SPECIFIC_PATTERN in derived, (
        f"derived pattern set {derived!r} does not contain the expected "
        f"specific-function pattern {EXPECTED_SPECIFIC_PATTERN!r}; "
        f"the mangling derivation has drifted from its known shape"
    )
    assert not any(
        fnmatch.fnmatch(UNMANGLED_ORIGINAL_NAME, p) for p in derived
    ), (
        f"a derived pattern fnmatches the unmangled name "
        f"{UNMANGLED_ORIGINAL_NAME!r}; the pattern is not pinning the "
        f"mangler prefix and would match anything"
    )


def test_every_pattern_anchors_on_a_mutant_suffix() -> None:
    """Specific-function patterns must end with ``__mutmut_*``.

    Whole-module patterns (``module.*``) are exempted: a real mutant key
    matches them because fnmatch's ``*`` already spans the mangler prefix
    and the class qualifier, and the test concern ("would match the
    original name") is moot for whole-module patterns — matching the
    original name matches the same module's mutants anyway.

    The constraint applies to specific-function and specific-method
    patterns: those must anchor on ``__mutmut_*`` because without it the
    pattern would match only the original function name, which mutmut
    never emits as a key (it always runs through ``make_mutant_key``),
    making the pattern match nothing and silently reporting zero mutants
    for that target.
    """
    for _group, target in all_targets():
        if target.cls is None and target.function_or_method is None:
            continue  # whole module — exempt
        for pattern in mangled_patterns(target):
            assert pattern.endswith("__mutmut_*"), (
                f"pattern {pattern!r} for {target!r} does not anchor on "
                f"the mutmut suffix — it would match the original name "
                f"rather than a real mutant key"
            )


def test_provider_hook_patterns_anchor_on_method_name() -> None:
    """Cross-class entries must include the method name as a fixed segment.

    A wildcard-only pattern (``*__mutmut_*``) would match every mutant in
    every group, defeating the purpose of the registry. The hook entries
    must carry the method name in the pattern so they match only methods
    of that name.
    """
    hook_targets = TARGET_GROUPS["provider_hooks"]
    for target in hook_targets:
        patterns = mangled_patterns(target)
        assert len(patterns) == 1, (
            f"cross-class target {target!r} should produce one pattern, "
            f"got {patterns!r}"
        )
        method = target.function_or_method
        assert f"{CLS}{method}__mutmut_" in patterns[0], (
            f"pattern {patterns[0]!r} for {target!r} does not include the "
            f"fixed method name segment; it would match unrelated methods"
        )


def test_each_target_group_has_at_least_one_pattern() -> None:
    """No target group is empty — otherwise the recorded ``mutmut run``
    with that group's patterns would filter every mutant out."""
    for group in TARGET_GROUPS:
        assert patterns_for(group), f"target group {group!r} produces no patterns"


# ── Falsification cases ─────────────────────────────────────────────────
#
# Each of the following tests injects a deliberately-wrong target into a
# private copy of the resolve function and asserts the guard catches the
# defect. The injection is constructed against the real symbols, so a
# guard that is broken in any of these three specific ways fails on this
# very file rather than only when the registry goes wrong.


def test_guard_catches_wrong_module_path() -> None:
    """A module that does not exist must trigger an ImportError."""
    bogus = Target("kitty.does_not_exist_anywhere", None, None)
    with pytest.raises(ModuleNotFoundError):
        resolve_target(bogus)


def test_guard_catches_wrong_class_name() -> None:
    """A typo in the class column must trigger an AssertionError."""
    bogus = Target("kitty.bridge.server", "BridgeServerr", "_compact_messages")
    with pytest.raises(AssertionError, match="class does not exist"):
        resolve_target(bogus)


def test_guard_catches_wrong_method_name() -> None:
    """A typo in the method column must trigger an AssertionError."""
    bogus = Target(
        "kitty.bridge.server", "BridgeServer", "_compact_messages_typo"
    )
    with pytest.raises(AssertionError, match="method does not exist"):
        resolve_target(bogus)
