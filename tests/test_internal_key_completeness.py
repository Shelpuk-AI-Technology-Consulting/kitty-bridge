"""Structural guard: every internal metadata key kitty writes is registered.

Kitty's translators and adapters hand each other metadata on underscore-prefixed
keys of the Chat Completions request dict.  ``ProviderAdapter._INTERNAL_KEYS``
is the frozenset ``translate_to_upstream`` strips, and it is the only thing
keeping those keys off the provider's wire.

KBR-6 is what happens when the set falls behind the code: ``_effort`` and
``_thinking_adaptive`` were added to ``MessagesTranslator.translate_request``
and never added to the set, so sixteen providers received them on every
request — an intermediary signature in a field no vendor API defines.

The complementary check — that each ``translate_to_upstream`` override strips
the set — is necessary but **not** sufficient, and is exactly the test that
existed and passed while the defect was live: every override strips the set
correctly.  The set itself was wrong.  This file checks the set.

See ``.system_design/TEST_SUITE.md`` §6.2.3, gap G15.
"""

from __future__ import annotations

import ast
import collections
from pathlib import Path

import pytest
from internal_key_scan import SRC, KeyWrite, scan_source, scan_tree

from kitty.providers.base import ProviderAdapter

# L2: the subject of this file is an artifact outside `src/kitty` Python code,
# or a structural scan of source text -- two things edited separately that must
# agree. It gates pull requests exactly as before, in the `l1 or l2` job; the
# marker records which half of that expression it answers to, and keeps a
# source-text scan out of the L1 set that mutation testing will judge.
pytestmark = pytest.mark.l2

#: Which file is expected to mint which internal keys.
#:
#: A *key set* per file, not a per-file count.  ``server.py`` holds roughly
#: twenty near-identical ``cc_request["_resolved_key"]`` writes across the retry
#: and failover branches; an exact-count registry would fire on any refactor
#: that adds or merges a branch, get bumped reflexively, and die.  A key set is
#: stable under that duplication and still fails when a file starts minting a
#: new key or when the visitor goes blind on a file.
_EXPECTED_KEYS: dict[str, set[str]] = {
    "bridge/messages/translator.py": {
        "_effort",
        "_reasoning_effort",
        "_thinking_adaptive",
        "_thinking_enabled",
        # KBR-178: Chat Completions has no `top_k`, so the inbound value rides
        # an internal key to the Anthropic-family adapters that accept it.
        "_top_k",
    },
    "bridge/responses/translator.py": {"_reasoning_effort", "_thinking_enabled"},
    "bridge/server.py": {
        "_native_messages_request",
        "_original_body",
        "_provider_config",
        "_resolved_key",
        # KBR-178: `_convert_native_to_cc_format` is a second Messages -> CC
        # converter and mints the same internal key the translator does.
        "_top_k",
    },
    "providers/kimi.py": {"_thinking_enabled"},
}


def _keys_by_file(writes: list[KeyWrite]) -> dict[str, set[str]]:
    """Group scan results into the key set minted by each file.

    Args:
        writes: Scan results.

    Returns:
        Mapping of relative path to the set of keys written in that file.
    """
    grouped: dict[str, set[str]] = collections.defaultdict(set)
    for write in writes:
        grouped[write.path].add(write.key)
    return dict(grouped)


class TestEveryInternalKeyIsRegistered:
    """R1: no internal key may be written without joining ``_INTERNAL_KEYS``."""

    def test_the_scanned_tree_is_the_imported_package(self):
        """The scan reads files; the expectation is imported. They must agree.

        ``SRC`` is derived from this file's location, but ``_INTERNAL_KEYS``
        comes from whatever ``kitty`` is importable. In a git worktree sharing
        the main checkout's editable install those are two different trees, and
        the guard then compares one tree's writes against the other tree's set —
        which can produce a false pass as easily as a false failure.
        """
        import kitty

        assert Path(kitty.__file__).resolve().parent == SRC, (
            f"scanning {SRC} but importing {Path(kitty.__file__).resolve().parent}. "
            "Run pytest with PYTHONPATH=src from this worktree."
        )

    def test_every_internal_key_written_is_registered(self):
        writes, _excluded = scan_tree()
        known = set(ProviderAdapter._INTERNAL_KEYS)

        offenders = [str(write) for write in writes if write.key not in known]

        assert not offenders, (
            "Internal metadata key(s) written into a request dict but missing from "
            "ProviderAdapter._INTERNAL_KEYS, so they are forwarded to the provider:\n  "
            + "\n  ".join(offenders)
            + "\n\nAdd them to _INTERNAL_KEYS in src/kitty/providers/base.py. If the key is "
            "not part of a request body, do NOT add it to the set — add a justified "
            "exclusion to tests/internal_key_scan.py instead, per TEST_SUITE.md §6.2.3."
        )


class TestTheScanCannotRotIntoANoOp:
    """R3: a guard that stops seeing anything must fail, not pass."""

    def test_the_scan_actually_finds_something(self):
        """A broken visitor returning nothing would pass every other check here."""
        writes, _excluded = scan_tree()

        assert len(writes) >= 30, f"the scan found only {len(writes)} write(s); it has gone blind"

    def test_key_set_per_file_is_unchanged(self):
        """A file that starts or stops minting a key must be reviewed."""
        writes, _excluded = scan_tree()
        actual = _keys_by_file(writes)

        changed = {path: keys for path, keys in actual.items() if keys != _EXPECTED_KEYS.get(path)}
        detail = [
            f"{path}: expected {sorted(_EXPECTED_KEYS.get(path, set()))}, found {sorted(keys)}"
            for path, keys in sorted(changed.items())
        ]

        assert not detail, (
            "The set of internal keys minted per file changed — review each for upstream "
            "exposure, then update _EXPECTED_KEYS in this file: " + "; ".join(detail)
        )

    def test_registry_has_no_stale_entries(self):
        """A stale entry would excuse a file that no longer mints anything."""
        writes, _excluded = scan_tree()
        stale = sorted(set(_EXPECTED_KEYS) - set(_keys_by_file(writes)))

        assert not stale, f"_EXPECTED_KEYS names files that mint no internal key any more: {stale}"


#: One synthetic module per way of writing a key, with the ``KeyWrite.form``
#: each must report.  A visitor method that silently stopped working — or that
#: started mislabelling what it found — would pass every check above.
#:
#: Keyed by case name, valued ``(source, expected_form)``.
_SYNTHETIC_WRITES: dict[str, tuple[str, str]] = {
    "subscript-assign": ('def f(cc):\n    cc["_leaked"] = 1\n', "subscript-assign"),
    "subscript-augassign": ('def f(cc):\n    cc["_leaked"] += 1\n', "subscript-assign"),
    "subscript-annassign": ('def f(cc):\n    cc["_leaked"]: int = 1\n', "subscript-assign"),
    "subscript-in-tuple": ('def f(cc):\n    cc["_leaked"], x = 1, 2\n', "subscript-assign"),
    "dict-literal": ('def f():\n    return {"_leaked": 1}\n', "dict-literal"),
    "dict-literal-via-update": ('def f(cc):\n    cc.update({"_leaked": 1})\n', "dict-literal"),
    "dict-literal-via-unpack": ('def f(cc):\n    return {**cc, "_leaked": 1}\n', "dict-literal"),
    "setdefault": ('def f(cc):\n    cc.setdefault("_leaked", 1)\n', "setdefault"),
    "update-kwarg": ("def f(cc):\n    cc.update(_leaked=1)\n", "update-kwarg"),
    "dict-kwarg": ("def f(cc):\n    return dict(cc, _leaked=1)\n", "dict-kwarg"),
}


class TestTheScanIsFalsifiable:
    """R3d: each covered form is proven to be detected, not assumed."""

    @pytest.mark.parametrize("case", sorted(_SYNTHETIC_WRITES))
    def test_scan_flags_a_synthetic_key(self, case: str):
        source, expected_form = _SYNTHETIC_WRITES[case]
        writes, _excluded = scan_source(source)

        assert [w.key for w in writes] == ["_leaked"], f"the scan missed a {case} write"
        assert writes[0].form == expected_form, (
            f"{case} was reported as {writes[0].form!r}, not {expected_form!r}; "
            "the assertion message names the form, so a mislabel misdirects the reader"
        )

    def test_scan_flags_a_write_to_an_unusual_target_name(self):
        """The scan must not filter by target name.

        Restricting it to targets called ``cc_request`` or ``body`` would pass
        every other test in this file while missing a real leak.
        """
        writes, _excluded = scan_source('def f(payload):\n    payload["_leaked"] = 1\n')

        assert [w.key for w in writes] == ["_leaked"]

    def test_scan_flags_a_write_inside_a_nested_function(self):
        """Scope tracking must not create a blind spot."""
        source = 'def outer(cc):\n    def inner():\n        cc["_leaked"] = 1\n    return inner\n'
        writes, _excluded = scan_source(source)

        assert [w.key for w in writes] == ["_leaked"]

    def test_registered_keys_are_still_reported_by_the_scan(self):
        """The scan reports every internal key, not only unregistered ones.

        Filtering inside the visitor would make ``_EXPECTED_KEYS`` unusable and
        would hide a key that later left ``_INTERNAL_KEYS``.
        """
        writes, _excluded = scan_source('def f(cc):\n    cc["_resolved_key"] = 1\n')

        assert [w.key for w in writes] == ["_resolved_key"]


class TestTheRequestObjectExclusion:
    """R4: an aiohttp request object is not a request body — nor a free pass."""

    def test_web_request_exclusion_still_matches(self):
        """An exclusion that matches nothing is a guard that has gone blind."""
        _writes, excluded = scan_tree()

        assert excluded, (
            "The web.Request exclusion no longer matches anything. Either the "
            "request-scoped writes in BridgeServer._auth_middleware moved, or the "
            "annotation changed — confirm the exclusion is still needed before removing it."
        )

    def test_the_excluded_keys_are_the_known_request_scoped_ones(self):
        """Pin what is excluded, so the hatch cannot quietly widen."""
        _writes, excluded = scan_tree()

        assert {write.key for write in excluded} == {"_key_id", "_profile_name", "_mapped_profile"}

    def test_excluded_keys_are_not_in_internal_keys(self):
        """They are request-scoped storage, not body keys.

        Adding them to ``_INTERNAL_KEYS`` is the tempting way to silence the
        scan without the exclusion.  It would make the set describe something
        it does not govern, and would then excuse a genuine leak if one of
        those names were ever written into a real request body.
        """
        _writes, excluded = scan_tree()
        known = set(ProviderAdapter._INTERNAL_KEYS)

        assert not {write.key for write in excluded} & known

    def test_annotation_excludes_but_a_bare_local_does_not(self):
        """The exclusion is keyed on the annotation, never on the name.

        Both functions below write to a variable called ``request``. Only the
        annotated one is request-scoped storage; a name-based rule would wave
        the other one through.
        """
        annotated = 'def f(request: web.Request):\n    request["_leaked"] = 1\n'
        bare = 'def f(request):\n    request["_leaked"] = 1\n'

        annotated_writes, annotated_excluded = scan_source(annotated)
        bare_writes, bare_excluded = scan_source(bare)

        assert annotated_writes == [] and [w.key for w in annotated_excluded] == ["_leaked"]
        assert [w.key for w in bare_writes] == ["_leaked"] and bare_excluded == []

    def test_the_exclusion_does_not_leak_out_of_its_scope(self):
        """A later function reusing the name must not inherit the exclusion."""
        source = (
            'def handler(request: web.Request):\n    request["_ok"] = 1\n\n'
            'def helper(request):\n    request["_leaked"] = 1\n'
        )
        writes, excluded = scan_source(source)

        assert [w.key for w in writes] == ["_leaked"]
        assert [w.key for w in excluded] == ["_ok"]

    #: Ways to make the excluded name stop meaning "the aiohttp request object".
    #:
    #: The exclusion is seeded from an annotation but applied to a *name*, so it
    #: has to retire the moment the name is rebound. Without these, the rule
    #: decays into the name-based one R4 forbids: annotate a parameter
    #: ``web.Request``, reassign it to a body dict, and every write to it is
    #: waved through.
    #:
    #: These are cases, not a specification. The scan does not enumerate binding
    #: statements — four review rounds showed that list cannot be completed —
    #: it asks whether the scope rebinds the name at all. So this dict is
    #: evidence the rule holds across the shapes Python offers, and a new form
    #: appearing in some future Python needs a case here but not a code change.
    _SHADOWING = {
        "reassigned-to-a-dict": (
            "def f(request: web.Request):\n    request = {}\n    request[\"_leaked\"] = 1\n"
        ),
        "reassigned-to-the-parsed-body": (
            "async def f(request: web.Request):\n"
            "    request = await request.json()\n"
            '    request["_leaked"] = 1\n'
        ),
        "redeclared-as-a-nested-parameter": (
            "def outer(request: web.Request):\n"
            "    def inner(request):\n"
            '        request["_leaked"] = 1\n'
            "    return inner\n"
        ),
        "rebound-by-a-for-target": (
            "def f(request: web.Request, items):\n"
            "    for request in items:\n"
            '        request["_leaked"] = 1\n'
        ),
        "rebound-by-a-with-target": (
            "def f(request: web.Request, ctx):\n"
            "    with ctx as request:\n"
            '        request["_leaked"] = 1\n'
        ),
        # The four below are the async and walrus forms. The handlers this
        # exclusion protects are all coroutines, so these are the likely
        # shapes here, not the exotic ones.
        "rebound-by-an-annotated-target": (
            "async def f(request: web.Request):\n"
            "    request: dict = await request.json()\n"
            '    request["_leaked"] = 1\n'
        ),
        "rebound-by-an-async-for-target": (
            "async def f(request: web.Request, items):\n"
            "    async for request in items:\n"
            '        request["_leaked"] = 1\n'
        ),
        "rebound-by-an-async-with-target": (
            "async def f(request: web.Request, ctx):\n"
            "    async with ctx as request:\n"
            '        request["_leaked"] = 1\n'
        ),
        "rebound-by-a-walrus": (
            "async def f(request: web.Request):\n"
            "    if (request := await request.json()):\n"
            '        request["_leaked"] = 1\n'
        ),
        # Tuple and starred targets bind every name inside them. The recursion
        # must stop at Subscript, though — see `test_a_subscript_target_is_not_
        # treated_as_a_rebind`, which is the case that would break the guard
        # rather than merely widen it.
        "rebound-by-a-tuple-for-target": (
            "def f(request: web.Request, pairs):\n"
            "    for request, item in pairs:\n"
            '        request["_leaked"] = 1\n'
        ),
        "rebound-by-a-tuple-with-target": (
            "def f(request: web.Request, ctx):\n"
            "    with ctx as (request, other):\n"
            '        request["_leaked"] = 1\n'
        ),
        "rebound-by-a-starred-target": (
            "def f(request: web.Request, items):\n"
            "    first, *request = items\n"
            '    request["_leaked"] = 1\n'
        ),
        "rebound-by-an-augmented-assignment": (
            "def f(request: web.Request, extra):\n"
            "    request += extra\n"
            '    request["_leaked"] = 1\n'
        ),
        "rebound-by-an-except-clause": (
            "def f(request: web.Request):\n"
            "    try:\n"
            "        pass\n"
            "    except ValueError as request:\n"
            '        request["_leaked"] = 1\n'
        ),
        "rebound-by-an-import-alias": (
            "def f(request: web.Request):\n"
            "    import json as request\n"
            '    request["_leaked"] = 1\n'
        ),
        "rebound-by-a-from-import-alias": (
            "def f(request: web.Request):\n"
            "    from json import loads as request\n"
            '    request["_leaked"] = 1\n'
        ),
        # The forms below are the reason the rule is decided per scope rather
        # than per statement. Each is a binding form Python has that the earlier
        # statement-by-statement version did not enumerate; under the scope rule
        # none of them needed its own handler.
        "rebound-by-a-match-capture": (
            "def f(request: web.Request, data):\n"
            "    match data:\n"
            "        case request:\n"
            '            request["_leaked"] = 1\n'
        ),
        "rebound-by-a-match-mapping-rest": (
            "def f(request: web.Request, data):\n"
            "    match data:\n"
            '        case {"a": 1, **request}:\n'
            '            request["_leaked"] = 1\n'
        ),
        "rebound-by-a-class-statement": (
            "def f(request: web.Request):\n"
            "    class request:\n"
            "        pass\n"
            '    request["_leaked"] = 1\n'
        ),
        "rebound-by-a-comprehension-target": (
            "def f(request: web.Request, xs):\n"
            "    ys = [request for request in xs]\n"
            '    request["_leaked"] = 1\n'
        ),
        "rebound-by-a-nested-def": (
            "def f(request: web.Request):\n"
            "    def request():\n"
            "        pass\n"
            '    request["_leaked"] = 1\n'
        ),
    }

    @pytest.mark.parametrize("case", sorted(_SHADOWING))
    def test_a_rebound_name_loses_the_exclusion(self, case: str):
        writes, excluded = scan_source(self._SHADOWING[case])

        assert [w.key for w in writes] == ["_leaked"], (
            f"{case}: the name no longer holds the aiohttp request object, so the "
            "exclusion must not still apply to it"
        )
        assert excluded == []

    def test_a_subscript_target_is_not_treated_as_a_rebind(self):
        """The counterweight to the rebinding rule, and the sharper risk.

        `cc["_x"] = 1` writes *through* `cc`; it does not rebind it. If the
        walk that retires a rebound name recursed into `Subscript` targets it
        would retire the exclusion on the very statement the exclusion exists
        to suppress — and the three request-scoped writes in `_auth_middleware`
        would start being reported as leaks.

        Two consecutive writes, because the first would disarm the exclusion
        for the second if the recursion were wrong.
        """
        source = (
            "def f(request: web.Request):\n"
            '    request["_first"] = 1\n'
            '    request["_second"] = 2\n'
        )
        writes, excluded = scan_source(source)

        assert writes == []
        assert [w.key for w in excluded] == ["_first", "_second"]

    def test_an_attribute_target_is_not_treated_as_a_rebind(self):
        """`request.state = x` rebinds an attribute, not the name."""
        source = (
            "def f(request: web.Request):\n"
            "    request.state = {}\n"
            '    request["_leaked"] = 1\n'
        )
        writes, excluded = scan_source(source)

        assert writes == []
        assert [w.key for w in excluded] == ["_leaked"]

    def test_a_lambda_parameter_does_not_inherit_the_exclusion(self):
        source = 'def f(request: web.Request):\n    return lambda request: request.setdefault("_leaked", 1)\n'
        writes, _excluded = scan_source(source)

        assert [w.key for w in writes] == ["_leaked"]


class TestTheScanParsesRatherThanGreps:
    """R3: the check is structural, so prose cannot trip it or hide from it."""

    def test_a_key_named_only_in_a_docstring_is_not_reported(self):
        """A grep-based guard would fire on the design document's own examples."""
        writes, _excluded = scan_source('def f():\n    """Mentions cc["_leaked"] in prose."""\n')

        assert writes == []

    def test_every_scanned_file_parses(self):
        """A file that fails to parse must not be skipped in silence."""
        from internal_key_scan import SCANNED_PACKAGES, SRC

        for package in SCANNED_PACKAGES:
            for file in sorted((SRC / package).rglob("*.py")):
                ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
