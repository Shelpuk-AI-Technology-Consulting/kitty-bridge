"""Property tests for ``describe_tool_input_anomaly`` (KBR-74, T-F5).

`.system_design/TEST_SUITE.md` §6.1 property row 7: the tool-input anomaly
detector **never reports an anomaly for input that validates against the
declared schema**. False positives here are worse than misses — the
detector's whole value is that a warning means something (kitty-bridge#33),
and the example-based suite (``tests/bridge/test_tool_use_audit.py``) pins
only the cases the incident's author happened to think of.

Five properties, plus a §1.4 falsification control:

- **P1 — no false positive on the resolvable subset.** For every schema in
  the small JSON-Schema subset the detector actually resolves
  (``type: object``, non-empty root ``properties``, non-empty root
  ``required``) and every input that validates against that schema, the
  detector returns ``None``. Three arms:
  - *P1a* — broad coverage under ``additionalProperties: true / absent``,
    extras optional.
  - *P1b* — the targeted discriminator: the input always carries at least
    one undeclared permitted key. Under ``additionalProperties: true /
    absent`` every valid input has ``missing=[]``, so the only mutation P1
    can catch is "drop the ``not missing`` precondition and fire on
    ``unexpected`` alone" — and that mutation is visible only when the
    input carries an undeclared permitted key. Without this arm the
    property green-lights it.
  - *P1c* — ``additionalProperties: false``, no extras by definition.
- **P2 — the composition-keyword stand-down is unconditional.** For every
  keyword the detector stands down on, an input that *would* have been
  reported under the resolvable reading still returns ``None``. The
  violating input is what proves the stand-down — not the input's
  validity — returns ``None``.
- **P4 — the both-conditions rule stays a both-conditions rule.** An
  input missing a required property without carrying an undeclared key
  stays silent. This carries kitty-bridge#33's discipline — the rule
  "must not later be relaxed into an or" (``tool_audit.py`` docstring)
  — into this file, so a future edit that fires on missing alone fails
  here and not only in the example suite.
- **P5 — generator self-conformance.** Every drawn pair passes a
  hand-rolled validator scoped to exactly the subset FR-1 promises, so a
  generator drift that emits invalid "valid" pairs is caught before the
  property runs. Same posture as ``tests/harness/test_transcripts.py``
  judging the substrate's own conformance: the validator checks the
  *output* against the *spec*; the generator *constructs* to the spec.
  Different code paths, shared understanding.
- **P3 — falsification control (plan §1.4 harness rule).** The P1
  expression must be live: a detector regressed to the pre-precision
  "report on any undeclared root key" shape makes P1's exact assertion
  fail against a valid input that carries one extra key. The *miss*
  class (an always-``None`` detector) is already pinned by the example
  suite's positive findings — per L1 rule A.10, an assertion another
  test already covers is not duplicated here.

**Scope decision — hand-rolled generator, not ``hypothesis-jsonschema``**
(decision recorded in ``TEST_SUITE.md`` §6.1, ticket KBR-74's "Added
scope"). The detector resolves a small, fixed subset, so a generator over
exactly that subset produces inputs whose validity is provable by
construction; ``hypothesis-jsonschema`` generates against the full schema
language including the composition keywords the detector declines, and
that extra coverage is wasted at the price of a new dev dependency. The
substrate's own docstring carves this task out (T-F1 ships local
strategies per property family; T-F4's egress strategies did the same).

**Validity by composition, not by re-derivation.** Every (schema, input)
pair is drawn together in :func:`_valid_pair` — one ``@st.composite`` draw
fixes the property names, their types, the required set and the
``additionalProperties`` posture, then builds the input from those same
names. The P5 validator is the *audit* on that construction, kept out of
the property bodies: inlining it would re-derive the generator inside the
property and mask a shared bug.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kitty.bridge import tool_audit

# Imported, not hard-coded: a composition keyword added to the source list
# must join this property automatically, or the stand-down narrows silently.
_COMPOSITION_KEYWORDS = tool_audit._COMPOSITION_KEYWORDS

# ── Local name/value strategies ─────────────────────────────────────────────
#
# Mirrors ``tests/harness/transcripts.py`` posture: private helpers, short
# ASCII alphabets so hypothesis's shrinking output stays legible. Kept local
# rather than imported — the substrate's helpers target message and tool-name
# shapes, and importing its privates would couple unrelated test work.


_SHORT_ASCII = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        max_codepoint=0x7E,
    ),
    min_size=1,
    max_size=8,
)

_LEAF_TYPES = ("string", "integer", "number", "boolean", "null")

# Draft-07 subtyping rules enforced at the value level: booleans are not
# integers or numbers in JSON Schema (Python's ``bool`` is an ``int``
# subclass, which is exactly the trap), and NaN/Infinity are not valid
# JSON at all — the same JSON-strict rule the T-F1 substrate enforces via
# ``allow_nan=False``.
_INTEGER_VALUE = st.integers(min_value=-(10**6), max_value=10**6)
_NUMBER_VALUE = st.one_of(
    _INTEGER_VALUE,
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
)
_ANY_SCALAR_VALUE = st.one_of(
    _SHORT_ASCII,
    _INTEGER_VALUE,
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.booleans(),
    st.none(),
)


def _leaf_value(leaf_type: str) -> st.SearchStrategy[Any]:
    """Return a JSON value matching one declared leaf type.

    Args:
        leaf_type: One of ``_LEAF_TYPES``.

    Returns:
        A Hypothesis strategy whose draws are all valid for ``leaf_type``
        under draft-07.
    """
    if leaf_type == "string":
        return _SHORT_ASCII
    if leaf_type == "integer":
        return _INTEGER_VALUE
    if leaf_type == "number":
        return _NUMBER_VALUE
    if leaf_type == "boolean":
        return st.booleans()
    if leaf_type == "null":
        return st.none()
    raise ValueError(f"not a leaf type: {leaf_type!r}")


def _leaf_subschema() -> st.SearchStrategy[dict]:
    """Return a JSON-Schema subschema declaring one leaf type."""
    return st.builds(lambda t: {"type": t}, t=st.sampled_from(_LEAF_TYPES))


def _array_subschema() -> st.SearchStrategy[dict]:
    """Return a JSON-Schema subschema for an array of one leaf type."""
    return st.builds(
        lambda t: {"type": "array", "items": {"type": t}},
        t=st.sampled_from(_LEAF_TYPES),
    )


@st.composite
def _nested_object_subschema(draw: st.DrawFn) -> dict:
    """Draw a one-level object subschema: leaf properties, non-empty required.

    Args:
        draw: The composite draw function.

    Returns:
        ``{"type": "object", "properties": {...}, "required": [...]}`` with
        ``required`` a non-empty subset of the property names.
    """
    names = draw(st.lists(_SHORT_ASCII, min_size=1, max_size=3, unique=True))
    required_count = draw(st.integers(min_value=1, max_value=len(names)))
    return {
        "type": "object",
        "properties": {n: {"type": "string"} for n in names},
        "required": names[:required_count],
    }


def _subschema() -> st.SearchStrategy[dict]:
    """Return any property subschema the generator emits.

    Returns:
        Leaf, array-of-leaf, or one-level nested object — the variety of
        property types Claude Code actually declares, without the
        composition keywords the detector stands down on.
    """
    return st.one_of(_leaf_subschema(), _array_subschema(), _nested_object_subschema())


def _value_for(subschema: dict) -> st.SearchStrategy[Any]:
    """Return a JSON value validating ``subschema`` per draft-07.

    Args:
        subschema: A subschema drawn from :func:`_subschema`.

    Returns:
        A strategy whose draws validate against ``subschema``.
    """
    declared = subschema.get("type")
    if declared == "array":
        return st.lists(_leaf_value(subschema["items"]["type"]), max_size=3)
    if declared == "object":
        # Values only need the nested required keys (all leaf strings by
        # construction); extra nested keys would also be valid but are not
        # drawn — the detector never descends, so this keeps the generator
        # minimal without weakening any claim it makes.
        return st.fixed_dictionaries({name: _SHORT_ASCII for name in subschema["required"]})
    return _leaf_value(declared)


@st.composite
def _valid_pair(draw: st.DrawFn, *, strict: bool = False, require_extra: bool = False) -> tuple[str, dict, dict]:
    """Draw one ``(tool_name, schema, tool_input)`` triple that validates.

    The single source of P1's input space. ``strict`` selects the
    ``additionalProperties: false`` posture (no undeclared keys in the
    input); otherwise the posture is ``true`` or absent (the JSON-Schema
    default), and the input may carry undeclared keys. ``require_extra``
    forces at least one — the P1b discriminator arm.

    Args:
        draw: The composite draw function.
        strict: When True, the schema carries ``additionalProperties: false``
            and the input's keys are all declared.
        require_extra: When True, the input always carries at least one
            undeclared key; only legal when ``strict`` is False.

    Returns:
        ``(tool_name, schema, tool_input)`` where ``tool_input`` validates
        against ``schema`` per draft-07 semantics.
    """
    if strict and require_extra:
        raise ValueError("require_extra is meaningless under additionalProperties: false")

    # Names: one unique pool, partitioned into declared properties,
    # "required but absent from properties" names (legal draft-07 while
    # additionalProperties permits them — the value is then unconstrained),
    # and extra keys.
    n_declared = draw(st.integers(min_value=1, max_value=5))
    n_required_only = 0 if strict else draw(st.integers(min_value=0, max_value=2))
    min_extras = 1 if require_extra else 0
    n_extras = 0 if strict else draw(st.integers(min_value=min_extras, max_value=3))
    names = draw(
        st.lists(
            _SHORT_ASCII,
            min_size=n_declared + n_required_only + n_extras,
            max_size=n_declared + n_required_only + n_extras,
            unique=True,
        )
    )
    declared = names[:n_declared]
    required_only = names[n_declared : n_declared + n_required_only]
    extras = names[n_declared + n_required_only :]

    # Declared properties: one subschema each; required is a non-empty
    # subset of the declared names (clamped draw, no filter — filters
    # shrink badly) plus the required-only names.
    subschemas = draw(st.lists(_subschema(), min_size=n_declared, max_size=n_declared))
    properties = dict(zip(declared, subschemas, strict=True))
    required_count = draw(st.integers(min_value=1, max_value=n_declared))
    required = declared[:required_count] + required_only

    # Schema: the resolvable subset, with the additionalProperties posture
    # the arm asked for. Absent means the JSON-Schema default (permitted),
    # so "true" and "absent" are distinct draws.
    schema: dict = {"type": "object", "properties": properties, "required": required}
    if strict:
        schema["additionalProperties"] = False
    elif draw(st.booleans()):
        schema["additionalProperties"] = True

    # Input: every required name present (declared names draw a value of
    # their declared type; required-only names are unconstrained), then the
    # remaining declared names, then the extras.
    tool_input: dict = {}
    for key in required:
        if key in properties:
            tool_input[key] = draw(_value_for(properties[key]))
        else:
            tool_input[key] = draw(_ANY_SCALAR_VALUE)
    for key in declared:
        if key not in tool_input:
            tool_input[key] = draw(_value_for(properties[key]))
    for key in extras:
        tool_input[key] = draw(_ANY_SCALAR_VALUE)

    return draw(_SHORT_ASCII), schema, tool_input


# ── P5's hand-rolled validator ──────────────────────────────────────────────
#
# Scoped to exactly what FR-1 promises. It shares no code with the
# generator, so a generator bug cannot hide behind its own assumptions.
# Deliberately NOT checked (named so the scope is auditable): subschema
# value structure beyond the type tag for array items and nested-object
# values (the generator emits only shapes the detector cannot see into),
# recursive ``$ref`` resolution, and every composition keyword — those are
# outside FR-1's subset by design.


def _value_matches_declared_type(value: Any, subschema: Any) -> bool:
    """Return True when ``value`` matches ``subschema``'s declared type tag.

    Args:
        value: The JSON value to judge.
        subschema: The subschema declaring the type (its ``type`` key is
            read; unknown or missing types are not judged and return True).

    Returns:
        True when the value is valid for the declared type under draft-07,
        including the subtyping rules (booleans are not integers/numbers;
        NaN/Infinity are not numbers).
    """
    declared = subschema.get("type") if isinstance(subschema, dict) else None
    if declared == "string":
        return isinstance(value, str)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return not (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "null":
        return value is None
    if declared == "array":
        return isinstance(value, list)
    if declared == "object":
        # Structure: the nested required keys must be present. The nested
        # properties' own value types are outside this validator's scope.
        required = subschema.get("required")
        if not isinstance(value, dict):
            return False
        if isinstance(required, list):
            return all(key in value for key in required)
        return True
    return True


def _pair_is_valid(schema: Any, tool_input: Any) -> bool:
    """Return True when ``tool_input`` validates against ``schema``.

    Args:
        schema: The drawn schema (expected: FR-1's resolvable subset).
        tool_input: The drawn input.

    Returns:
        True when the pair satisfies draft-07 semantics for the subset:
        object-typed schema, non-empty ``properties``, non-empty string
        ``required`` all present in the input, every present declared key's
        value matching its declared type, and undeclared keys present only
        when ``additionalProperties`` is not ``false``.
    """
    if not isinstance(schema, dict) or not isinstance(tool_input, dict):
        return False
    if schema.get("type") != "object":
        return False
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return False
    required = schema.get("required")
    if not isinstance(required, list) or not required:
        return False
    strict = schema.get("additionalProperties") is False

    # Draft-07: a required name must be present in the input, whether or
    # not a property constrains its value.
    for key in required:
        if not isinstance(key, str) or key not in tool_input:
            return False

    for key, value in tool_input.items():
        if key in properties:
            if not _value_matches_declared_type(value, properties[key]):
                return False
        elif strict:
            # additionalProperties: false — no undeclared key may appear.
            return False
    return True


# ── P1: no false positive on the resolvable subset ──────────────────────────


class TestNoFalsePositive:
    """P1: an input that validates against the declared schema is silent."""

    @given(_valid_pair())
    @settings(max_examples=200)
    def test_valid_input_is_silent(self, pair: tuple[str, dict, dict]) -> None:
        """P1a — broad arm: any valid pair under ``additionalProperties: true/absent``."""
        name, schema, tool_input = pair
        assert tool_audit.describe_tool_input_anomaly(name, tool_input, schema) is None

    @given(_valid_pair(require_extra=True))
    @settings(max_examples=200)
    def test_valid_input_with_undeclared_key_is_silent(self, pair: tuple[str, dict, dict]) -> None:
        """P1b — discriminator arm: valid input carrying an undeclared permitted key.

        This is the only arm that can catch "drop the ``not missing``
        precondition and fire on ``unexpected`` alone": the mutation leaves
        ``missing=[]`` on every valid input, so only an undeclared key in
        the input makes ``unexpected`` non-empty and exposes it.
        """
        name, schema, tool_input = pair
        assert tool_audit.describe_tool_input_anomaly(name, tool_input, schema) is None

    @given(_valid_pair(strict=True))
    @settings(max_examples=200)
    def test_valid_input_under_strict_schema_is_silent(self, pair: tuple[str, dict, dict]) -> None:
        """P1c — strict arm: valid input under ``additionalProperties: false``."""
        name, schema, tool_input = pair
        assert tool_audit.describe_tool_input_anomaly(name, tool_input, schema) is None

    def test_required_name_absent_from_properties_is_silent(self) -> None:
        """The required-only interplay, stated legibly.

        A required name with no matching property is legal draft-07 while
        ``additionalProperties`` permits extras. The detector sees it as
        "unexpected" — but it is also *present*, so ``missing`` stays empty
        and the both-conditions rule keeps the detector silent. The
        generator covers this structurally (``n_required_only``); this
        example documents the rule for the reader.
        """
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a", "ghost"],
            "additionalProperties": True,
        }
        assert tool_audit.describe_tool_input_anomaly("T", {"a": "ok", "ghost": 1}, schema) is None


# ── P2: the composition-keyword stand-down is unconditional ─────────────────


class TestCompositionKeywordStandDown:
    """P2: any composition keyword forces ``None``, even for a violating input.

    The stand-down fires before the resolve-and-compare half runs, so the
    input drawn here deliberately *would* have been reported under the
    resolvable reading (an unexpected root key and a missing required
    property at once). That is what proves the ``None`` comes from the
    stand-down and not from the input happening to be valid.
    """

    @pytest.mark.parametrize("keyword", _COMPOSITION_KEYWORDS)
    def test_stand_down_holds_for_a_violating_input(self, keyword: str) -> None:
        """The detector returns ``None`` despite an obviously malformed input."""
        schema = {
            "type": "object",
            "properties": {"findings": {"type": "array"}},
            "required": ["findings"],
            keyword: [{"type": "object"}],
        }
        assert tool_audit.describe_tool_input_anomaly("T", {"oops": "text"}, schema) is None


# ── P4: the both-conditions rule is not relaxed into an "or" ────────────────


class TestBothConditionsRuleStaysBoth:
    """P4: missing-a-required-property alone never fires (kitty-bridge#33).

    ``tool_audit.py``'s docstring pins the discipline: "a root ``required``
    name absent from the input is proof the input violates the declared
    schema … and why it must not later be relaxed into an 'or'". An edit
    that fires on ``missing`` alone would make every ordinary incomplete
    payload a warning — the noise the detector exists to avoid. These
    examples carry that discipline into the property file so the no-false-
    positive property cannot be satisfied by such an edit.
    """

    @pytest.mark.parametrize(
        ("required", "tool_input"),
        [
            (["a"], {}),
            (["a", "b"], {"a": "ok"}),
            (["a", "ghost"], {"a": "ok"}),
            # The single-key shape that is *not* the nesting fingerprint:
            # one root key, but its value does not contain the missing
            # required names, so neither rule may fire.
            (["a", "b"], {"a": {}}),
        ],
    )
    def test_missing_alone_is_silent(self, required: list[str], tool_input: dict) -> None:
        """The detector stays silent while a required property is absent."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": required,
        }
        assert tool_audit.describe_tool_input_anomaly("T", tool_input, schema) is None


# ── P5: generator self-conformance ──────────────────────────────────────────


class TestGeneratorSelfConformance:
    """P5: every drawn pair validates under the hand-rolled FR-1 validator.

    Runs before (and independent of) the properties, so a generator drift
    that emits invalid "valid" pairs fails here — where the message names
    the validator — instead of silently weakening P1.
    """

    @given(_valid_pair())
    @settings(max_examples=200)
    def test_broad_pairs_validate(self, pair: tuple[str, dict, dict]) -> None:
        """The broad arm's pairs satisfy FR-1's subset semantics."""
        _, schema, tool_input = pair
        assert _pair_is_valid(schema, tool_input)

    @given(_valid_pair(require_extra=True))
    @settings(max_examples=100)
    def test_discriminator_pairs_validate_and_carry_an_extra(self, pair: tuple[str, dict, dict]) -> None:
        """The discriminator arm's pairs validate *and* really carry an undeclared key.

        The second assertion is what makes P1b honest: if the generator
        stopped emitting extras, P1b would silently degenerate into P1a.
        """
        _, schema, tool_input = pair
        assert _pair_is_valid(schema, tool_input)
        assert any(key not in schema["properties"] for key in tool_input)

    @given(_valid_pair(strict=True))
    @settings(max_examples=100)
    def test_strict_pairs_validate(self, pair: tuple[str, dict, dict]) -> None:
        """The strict arm's pairs satisfy FR-1 under ``additionalProperties: false``."""
        _, schema, tool_input = pair
        assert _pair_is_valid(schema, tool_input)


# ── P3: falsification control (plan §1.4) ──────────────────────────────────
#
# The pre-precision detector: report whenever any root key is undeclared.
# If the production detector ever regressed to this shape, P1b's inputs
# (valid, carrying an undeclared key) would each produce a warning — and
# this control proves P1's assertion expression actually fails under that
# regression, i.e. the property is live rather than vacuous.

_NAIVE_DESCRIPTION = "naive reporter: undeclared root key(s) present"


def _naive_undeclared_key_reporter(name: str, tool_input: object, schema: dict | None) -> str | None:
    """Report on any root key the schema's ``properties`` does not declare.

    Args:
        name: The tool's name (unused; present to mirror the production signature).
        tool_input: The returned input object.
        schema: The declared schema, or ``None`` when undeclared (never reported).

    Returns:
        A description when any root key is undeclared, else ``None``.
    """
    if not isinstance(schema, dict) or not isinstance(tool_input, dict):
        return None
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None
    if not any(key not in properties for key in tool_input):
        return None
    return _NAIVE_DESCRIPTION


class TestFalsificationControl:
    """P3: P1's assertion expression is live against a report-happy regression."""

    def test_p1_expression_fails_under_a_report_happy_regression(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Monkeypatching the naive reporter makes P1's exact assertion raise.

        Args:
            monkeypatch: Pytest's patching fixture.
        """
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": True,
        }
        # Valid per draft-07 (the extra key is permitted), and exactly the
        # shape P1b generates — the naive reporter fires on it.
        valid_input = {"a": "ok", "extra": 1}
        assert _naive_undeclared_key_reporter("regressed", valid_input, schema) is not None

        monkeypatch.setattr(tool_audit, "describe_tool_input_anomaly", _naive_undeclared_key_reporter)
        with pytest.raises(AssertionError):
            # P1's assertion expression, verbatim — evaluated against the
            # regressed detector it must be able to catch.
            assert tool_audit.describe_tool_input_anomaly("regressed", valid_input, schema) is None
