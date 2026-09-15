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
import re
import sys
from pathlib import Path

import pytest
from mutmut_scope import (
    CLS,
    DEFERRED_GROUPS,
    TARGET_GROUPS,
    Target,
    all_targets,
    mangled_patterns,
    patterns_for,
    resolve_target,
)

pytestmark = pytest.mark.l2

# tomllib landed in 3.11; the 3.10 leg of the test matrix needs
# ``tomli`` (declared as a conditional dev extra in pyproject.toml).
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

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


def _only_mutate_globs() -> list[str]:
    """Return ``[tool.mutmut] only_mutate`` from ``pyproject.toml``.

    The config is the file-level half of the scope; the registry in this
    package is the symbol-level half. Reading the config here (rather
    than duplicating its list) means the guard below fails the moment
    either half drifts from the other.

    Returns:
        The list of fnmatch globs exactly as written in the config.
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        config = tomllib.load(f)
    return list(config["tool"]["mutmut"]["only_mutate"])


def test_every_registry_file_is_covered_by_only_mutate() -> None:
    """Each registry-named module's file is matched by some ``only_mutate`` glob.

    mutmut generates mutants per *file*; a registry row whose file is
    not in ``only_mutate`` silently produces zero mutants, which reads
    in the recorded baseline as an empty group rather than an error.
    This is the forward direction of the config ⇄ registry agreement.
    Groups in :data:`DEFERRED_GROUPS` are exempt — their exclusion from
    ``only_mutate`` is deliberate (see ``pyproject.toml``'s comment for
    ``server.py``), and the exemption is what this test enforces in the
    other direction: when a deferred group's blocker is fixed, removing
    it from ``DEFERRED_GROUPS`` re-enables this check for it.
    """
    globs = _only_mutate_globs()
    for group, target in all_targets():
        if group in DEFERRED_GROUPS:
            continue  # deliberate exclusion — see pyproject.toml
        if target.module == "*":
            continue  # cross-module: covered via the adapter file set below
        rel = (
            target.module.split(".", 1)[1]
            if target.module.startswith("kitty.")
            else target.module
        )
        path = f"src/kitty/{rel.replace('.', '/')}"
        candidates = [f"{path}.py"]
        # Whole-package scopes (``kitty.profiles``) are directories; any
        # file inside them satisfies the glob.
        pkg_dir = Path(__file__).resolve().parent.parent / path
        if pkg_dir.is_dir():
            candidates.append(f"{path}/*")
        assert any(
            fnmatch.fnmatch(candidate, g) for candidate in candidates for g in globs
        ), (
            f"registry row {target!r} (group {group!r}) names {path}, but no "
            f"`only_mutate` glob in pyproject.toml matches it — mutmut "
            f"would generate zero mutants for this target"
        )


def test_only_mutate_excludes_every_unscoped_provider_file() -> None:
    """Every ``src/kitty/providers/*.py`` file the registry does not name
    matches no ``only_mutate`` glob.

    Derives the in-scope set from the registry + source rather than
    hardcoding it. A provider file is IN scope when either:

    * a registry row names its module directly (e.g. ``base.py`` via
      KBR-134's ``_strip_endpoint_suffix``, ``model_context.py`` via
      the whole-module KBR-151 row, ``openai_subscription.py`` via its
      four P13–P17 rows), or
    * the file carries any of the three §6.1 hooks
      (``translate_to_upstream`` / ``normalize_request`` /
      ``build_upstream_headers``), matched by the cross-module rows.

    Everything else in ``providers/`` — ``__init__.py``,
    ``registry.py``, ``model_context_sync.py``, ``google_aistudio.py``,
    ``novita.py`` today — is correctly excluded, and a future
    hand-add of its path to ``only_mutate`` would silently widen the
    scope unless this test catches it. The forward direction (every
    registry file is covered) lives in
    ``test_every_registry_file_is_covered_by_only_mutate``; the two
    together pin the scope agreement.
    """
    providers_dir = (
        Path(__file__).resolve().parent.parent / "src" / "kitty" / "providers"
    )
    # Modules the registry names under providers/ (excluding the
    # cross-module wildcard rows, which are handled via the hook scan).
    # Stored with slashes (path form) since that's what we compare
    # against below.
    registry_named = {
        target.module.split(".", 1)[1].replace(".", "/")
        for _group, target in all_targets()
        if target.module.startswith("kitty.providers.")
    }
    hook_re = re.compile(
        r"^    def (translate_to_upstream|normalize_request|build_upstream_headers)\(",
        re.MULTILINE,
    )
    unscoped: list[str] = []
    for path in sorted(providers_dir.glob("*.py")):
        module_rel = f"providers/{path.stem}"  # e.g. "providers/model_context"
        if module_rel in registry_named or any(
            module_rel.startswith(r + "/") for r in registry_named
        ):
            continue  # registry names this module directly
        if hook_re.search(path.read_text()):
            continue  # carries a cross-module hook
        rel = f"src/kitty/providers/{path.name}"
        unscoped.append(rel)
    assert unscoped, (
        "no unscoped providers/*.py files found — either every provider "
        "is now registry-named (verify §6.1 is current) or the test's "
        "registry-naming logic has drifted"
    )
    globs = _only_mutate_globs()
    for rel in unscoped:
        assert not any(
            fnmatch.fnmatch(rel, g) for g in globs
        ), (
            f"{rel} is not named by any registry row and carries none of "
            f"the three §6.1 hooks but is matched by an `only_mutate` "
            f"glob — the glob list has been widened to include "
            f"unscoped providers"
        )


def test_every_only_mutate_entry_has_a_registry_row() -> None:
    """No ``only_mutate`` glob covers a file that no registry row names.

    Closes the reverse direction of the scope agreement — the
    forward test (``test_every_registry_file_is_covered_by_only_mutate``)
    requires every registry file to be glob-covered; this one
    requires every glob-covered file to have a registry row. Without
    it, a hand-add of a stray glob would pass the forward test (no
    registry file would lose coverage) while silently widening the
    scope to a file §6.1 does not name.
    """
    globs = _only_mutate_globs()
    repo_root = Path(__file__).resolve().parent.parent
    registry_files = set()
    cross_module_hooks: set[str] = set()
    for _group, target in all_targets():
        if target.module == "*":
            # Cross-module rows (``Target("*", "*", hook)``) make every
            # provider module that carries the hook implicitly in scope
            # — collect the hook names here so the reverse test below
            # accounts for them too.
            if target.function_or_method:
                cross_module_hooks.add(target.function_or_method)
            continue
        rel = (
            target.module.split(".", 1)[1]
            if target.module.startswith("kitty.")
            else target.module
        )
        path = repo_root / "src" / "kitty" / rel.replace(".", "/")
        for candidate in (path.with_suffix(".py"), path):  # file or dir
            if candidate.exists():
                # Store repo-relative so the comparison below matches
                # the walked paths' spelling.
                registry_files.add(str(candidate.relative_to(repo_root)))
                break

    # Add hook-carrying provider files to the in-scope set, since
    # cross-module rows make them implicitly in scope.
    if cross_module_hooks:
        hook_re = re.compile(
            r"^    def ("
            + "|".join(re.escape(h) for h in sorted(cross_module_hooks))
            + r")\(",
            re.MULTILINE,
        )
        providers_dir = repo_root / "src" / "kitty" / "providers"
        if providers_dir.is_dir():
            for path in providers_dir.glob("*.py"):
                if hook_re.search(path.read_text()):
                    registry_files.add(str(path.relative_to(repo_root)))

    # Walk the repo and assert every matched .py file has a registry
    # entry. Walk, rather than enumerate, because glob coverage can
    # span whole directories (`kitty.profiles/*`).
    for path in sorted(repo_root.glob("src/kitty/**/*.py")):
        rel = str(path.relative_to(repo_root))
        if any(fnmatch.fnmatch(rel, g) for g in globs):
            assert rel in registry_files or any(
                rel.startswith(f + "/") for f in registry_files
            ), (
                f"{rel} is matched by an `only_mutate` glob but no "
                f"registry row names it — either the registry is "
                f"missing an entry, or the glob list has grown wider "
                f"than the registry's scope"
            )


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
