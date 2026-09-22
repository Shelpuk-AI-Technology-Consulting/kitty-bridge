"""Mutation scope registry — T-H1 (KBR-88).

TEST_SUITE.md §6.1 names the code mutation testing covers as a table of
function wildcards (one per target group). That table is prose and so is
the first thing to drift from the code. This module is the
machine-readable form of the table: every entry names a
``(module, class_or_None, function_or_method_or_None)`` tuple whose
existence in the live source a guard test (``tests/test_mutmut_scope.py``)
asserts, and the mutmut fnmatch patterns the recorded ``mutmut run``
invocation carries are derived from these entries by :func:`patterns_for`.

The narrowing cannot live in ``pyproject.toml``'s ``source_paths``: mutmut
treats those as filesystem paths and silently drops non-path entries
(``walk_all_files`` only follows directories and files). Function-level
narrowing lives here.

Mutant's name format (mutmut's ``make_mutant_key``):

* top-level function ``f`` in module ``m`` → ``m.x_<f>``
* method ``m`` of class ``C`` in module ``mod`` → ``mod.xǁCǁm``

The trailing ``__mutmut_<line>`` segment is appended at mutation time, so
each pattern ends with ``__mutmut_*`` to anchor on real mutant keys rather
than the original function name (which would match nothing — the
mangler's ``x_``/``xǁ`` prefix is what makes the name *mutable*).
"""

from __future__ import annotations

import importlib
from typing import NamedTuple

# mutmut's CLASS_NAME_SEPARATOR (mutmut/utils/format_utils.py). U+01C1,
# Latin "lateral click". Doubled to keep the surrounding mangled name
# visually distinct from a real dotted path.
CLS = "ǁ"


class Target(NamedTuple):
    """A scope entry under §6.1.

    Attributes:
        module: Dotted module path (``"kitty.bridge.server"``), or ``"*"``
            when the entry is cross-module (the provider-adapter hook
            case — see ``provider_hooks`` below).
        cls: The containing class name, or ``None`` for top-level functions
            / whole-module scope. ``"*"`` means "any class within the
            module(s)" and is used with cross-module entries.
        function_or_method: The function or method name. ``None`` means the
            whole module (or, when ``cls`` is set, the whole class) is in
            scope — used for groups where §6.1 names a module rather than
            a list of methods.
    """

    module: str
    cls: str | None
    function_or_method: str | None


TARGET_GROUPS: dict[str, list[Target]] = {
    # Translation — the I1 core.
    "translators_and_engine": [
        Target("kitty.bridge.messages", None, None),
        Target("kitty.bridge.responses", None, None),
        Target("kitty.bridge.gemini", None, None),
        Target("kitty.bridge.engine", None, None),
    ],
    # Compaction, pairing, normalisation. All BridgeServer methods.
    # _get_max_context_chars is a dispatcher since KBR-151 — see the
    # model_context group for the matcher itself. _tool_result_content_size
    # is module-level (KBR-223): the shared M3/M4 size extractor, registered
    # here so the pragma scheme's only-unmarked-defs rule keeps holding.
    "compaction_and_pairing": [
        Target("kitty.bridge.server", "BridgeServer", "_compact_messages"),
        Target("kitty.bridge.server", "BridgeServer", "_compact_with_tighter_budget"),
        Target("kitty.bridge.server", "BridgeServer", "_validate_tool_call_pairing"),
        Target("kitty.bridge.server", "BridgeServer", "_truncate_oversized_tool_results"),
        Target("kitty.bridge.server", "BridgeServer", "_apply_compaction"),
        Target("kitty.bridge.server", "BridgeServer", "_normalize_model"),
        Target("kitty.bridge.server", "BridgeServer", "_get_max_context_chars"),
        Target("kitty.bridge.server", None, "_tool_result_content_size"),
    ],
    # Provider adapter hooks — every adapter implements these three, plus
    # the ProviderAdapter._strip_endpoint_suffix from KBR-134.
    "provider_hooks": [
        Target("*", "*", "translate_to_upstream"),
        Target("*", "*", "normalize_request"),
        Target("*", "*", "build_upstream_headers"),
        Target("kitty.providers.base", "ProviderAdapter", "_strip_endpoint_suffix"),
    ],
    # Where the compaction budget is actually resolved — KBR-151 added this.
    # _get_max_context_chars dispatches to get_model_context_tokens in here,
    # which calls _resolve_catalog → _match_catalog → _colliding_keys.
    "model_context": [
        Target("kitty.providers.model_context", None, None),
    ],
    # P13–P17 and the F1 user-agent.
    "openai_subscription": [
        Target(
            "kitty.providers.openai_subscription",
            "OpenAISubscriptionAdapter",
            "_cc_to_responses",
        ),
        Target(
            "kitty.providers.openai_subscription",
            "OpenAISubscriptionAdapter",
            "_prepare_responses_body",
        ),
        # _convert_content_types is module-level, unlike the other three:
        # it has no ``self`` and predates the adapter class it serves.
        Target("kitty.providers.openai_subscription", None, "_convert_content_types"),
        Target(
            "kitty.providers.openai_subscription",
            "OpenAISubscriptionAdapter",
            "_build_user_agent",
        ),
    ],
    # Register row P18 — the Converse body's modelId/stream pops. KBR-89
    # (T-H2) extracted them out of the bedrock transport's network methods,
    # where mutmut could not reach them, into this pure builder. Not a
    # provider hook: the sibling group above is the pattern (one adapter's
    # body builders in their own row), so the bedrock builder gets the same
    # shape rather than diluting the hooks' cross-adapter claim.
    "bedrock_transport": [
        Target(
            "kitty.providers.bedrock",
            "BedrockAdapter",
            "_bedrock_body",
        ),
    ],
    # Register row P19 — the /api/chat body's stream overwrite. KBR-90
    # (T-H5) extracted it out of the ollama_cloud transport's network
    # methods, where mutmut could not reach it, into this pure builder.
    # Same per-adapter shape as bedrock_transport — a sibling, not a
    # provider_hooks member (that group's description claims "every
    # adapter implements these three", and an ollama-only body builder
    # would violate the claim).
    "ollama_transport": [
        Target(
            "kitty.providers.ollama_cloud",
            "OllamaCloudAdapter",
            "_ollama_body",
        ),
    ],
    # Egress containment — I3.
    "egress": [
        Target("kitty.egress", None, None),
        Target("kitty.egress_guard", None, None),
    ],
    # The CC content predicates (KBR-285). The streaming hold's release
    # predicate and the non-streaming detector's Chat Completions arm carry
    # a deliberate byte-for-byte mirror (KBR-277); mutating both side-by-side
    # is the divergence guard the mirror's docstrings cite. Like
    # compaction_and_pairing, these live in server.py under the pragma
    # scheme — only their pragmas are absent.
    "content_classifiers": [
        Target("kitty.bridge.server", None, "_cc_chunk_carries_content"),
        Target("kitty.bridge.server", "BridgeServer", "_is_empty_cc_response"),
    ],
    # Supporting correctness: tool auditing, profile schema/resolver,
    # response-time URL validation.
    "supporting": [
        Target("kitty.bridge.tool_audit", None, None),
        Target("kitty.profiles", None, None),
        Target("kitty.validation", None, None),
    ],
}


# Groups whose baseline is recorded as **pending**, not measured. A group
# lands here when the file(s) it covers are not in the current
# ``[tool.mutmut] only_mutate`` list (see `pyproject.toml` for the
# reason), so no mutants are generated for its targets and the
# aggregator cannot compute a score. Removing an entry from this set is
# the migration step for any follow-up ticket that lifts the
# corresponding `only_mutate` exclusion.
DEFERRED_GROUPS: frozenset[str] = frozenset()


def mangled_patterns(target: Target) -> list[str]:
    """Return the mutmut fnmatch pattern(s) covering ``target``'s scope.

    Each returned pattern ends with ``__mutmut_*`` so it matches only real
    mutant keys (the mangler has rewritten the original name), never the
    original function name — matching the original would match nothing,
    since the mangled name is what mutmut actually tests.
    """
    if target.cls == "*" or target.module == "*":
        # Cross-class scope. The function name is the only anchor; the
        # rest of the mangled key is wild. Anchored on the trailing
        # function name to keep it from accidentally matching unrelated
        # methods (e.g. ``translate_to_upstream_thing``).
        return [f"*x{CLS}*{CLS}{target.function_or_method}__mutmut_*"]

    if target.cls is None and target.function_or_method is None:
        # Whole module — fnmatch's ``*`` spans the ``x_``/``xǁ``
        # mangling prefix and any class qualifier, so a single pattern
        # covers top-level functions and class methods alike.
        return [f"{target.module}.*"]

    if target.cls is None:
        # Specific top-level function. The mangler prefixes ``x_`` to the
        # function name; if the original started with ``_`` the pattern
        # carries a double underscore, which is correct (the mangled key
        # also has it).
        return [f"{target.module}.x_{target.function_or_method}__mutmut_*"]

    # Specific method on a specific class.
    return [
        f"{target.module}.x{CLS}{target.cls}{CLS}{target.function_or_method}__mutmut_*"
    ]


def patterns_for(group: str) -> list[str]:
    """Return the union of mutmut fnmatch patterns covering a target group.

    Pass these on the ``mutmut run`` command line to scope the run to one
    group. The recorded invocation in
    ``.system_design/MUTATION_BASELINE.md`` uses this function (or its
    per-group equivalent) to build its positional arguments.
    """
    return [p for t in TARGET_GROUPS[group] for p in mangled_patterns(t)]


def all_targets() -> list[tuple[str, Target]]:
    """Flatten ``(group, target)`` pairs in declaration order."""
    return [(g, t) for g, ts in TARGET_GROUPS.items() for t in ts]


def resolve_target(target: Target) -> None:
    """Import the module and assert the class/method exists.

    Raises:
        AssertionError: If the module cannot be imported, the class does
            not exist, or the method does not exist on the class. The
            message names the failing component so a guard failure points
            at the offending registry row.
        ImportError: Re-raised if the module path is syntactically valid
            but the import fails for reasons unrelated to the registry
            (a real bug in the source being looked up).
    """
    if target.module == "*":
        return  # cross-module: no specific symbol to resolve
    mod = importlib.import_module(target.module)
    if target.cls is None:
        return  # whole module — import succeeded
    cls_obj = getattr(mod, target.cls, None)
    assert cls_obj is not None, (
        f"{target.module}.{target.cls}: class does not exist "
        f"(registry row {target!r} points at a removed class)"
    )
    if target.function_or_method is None:
        return  # whole class
    assert hasattr(cls_obj, target.function_or_method), (
        f"{target.module}.{target.cls}.{target.function_or_method}: "
        f"method does not exist (registry row {target!r} points at a "
        f"removed or renamed method)"
    )


__all__ = [
    "CLS",
    "DEFERRED_GROUPS",
    "Target",
    "TARGET_GROUPS",
    "all_targets",
    "mangled_patterns",
    "patterns_for",
    "resolve_target",
]
