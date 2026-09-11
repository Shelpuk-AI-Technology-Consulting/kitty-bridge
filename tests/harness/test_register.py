"""The register's schema and its trigger vocabulary.

``.system_design/TEST_SUITE.md`` §3.2 · plan task **T-W3** (KBR-26).

Everything here is a property of the data alone — it reads no file and resolves
no symbol, so it belongs at L1 by §2.2's allocation rule.  The guards that
compare the data against the design document and against the source tree read
real artifacts and live in :mod:`tests.harness.test_register_agreement` at L2.

The schema is small, and every assertion below exists because a specific way of
filling it in would be wrong in a way no later test could see.  A row escaping
the path vocabulary without a reason, a trigger member nothing uses, a row
declared conditional whose trigger is ``ALWAYS`` — each would ship green today
and surface months later as an oracle that claims too much or a corpus entry
nobody knows to write.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from harness import contract as c
from harness import register as r
from harness.test_contract import _KITTY_IMPORT

#: The path roots a register row may start at. ``residual`` is deliberately
#: **absent**: §3.3.1a makes it never a legal anchor, because a bare collection
#: claims its members and a non-empty residual fails the run before matching, so
#: such a row could only ever claim a delta the oracle was meant to stop at.
_LEGAL_ROOTS = ("envelope.", "conversation.", "headers[", "reply.", "route.")

#: The bare-collection anchors §3.3.1a defines, which claim their members and so
#: need the same "does this match anything" proof a wildcard does.
_COLLECTION_ANCHORS = (
    c.CONVERSATION_SYSTEM,
    c.CONVERSATION_TURNS,
    c.CONVERSATION_TOOLS,
    c.CONVERSATION_SAMPLING,
)

#: Every register path whose *pattern* form differs from its concrete form,
#: paired with a concrete path built from the same helper. Module-level so the
#: parametrised test and the coverage check below read one list rather than two
#: that agree by luck.
#:
#: **Only those.** A path with no wildcard that claims no members -- ``envelope.
#: model``, ``headers[user-agent]``, ``conversation.sampling[max_tokens]``,
#: ``reply.parts[0]`` -- *is* its own concrete form, so asserting it matches
#: itself proves nothing: `_segment_matches` returns ``True`` on string equality
#: for any bracket-balanced string. Those paths are covered by
#: :meth:`TestThePathsEachRowTouches.test_every_path_has_balanced_brackets` and
#: by :data:`_LEGAL_ROOTS`, which is the honest division of labour. An earlier
#: draft listed ``reply.parts[0]`` against itself here and was exactly that
#: tautology.
_SHAPES: tuple[tuple[str, str], ...] = (
    (c.part_path(c.WILDCARD, c.WILDCARD), c.part_path(2, 0)),
    (c.tool_path(c.WILDCARD, "strict"), c.tool_path("get_weather", "strict")),
    (c.CONVERSATION_SAMPLING, c.sampling_path("temperature")),
    (c.CONVERSATION_SYSTEM, c.system_path(1)),
    (c.CONVERSATION_TURNS, c.turn_path(0, "role")),
)


class TestTheRowsThemselves:
    """§3.2 publishes 42 live rows; the data must be those rows and no others."""

    def test_the_register_holds_every_live_row(self) -> None:
        """15 bridge-level rows less the withdrawn M13, plus 28 provider-level."""
        assert len(r.REGISTER) == 42

    def test_the_register_is_a_tuple_and_not_a_list(self) -> None:
        """`mypy` does not run over `tests/`, so the annotation is not enforcement.

        A mutable `REGISTER` would let one test's edit leak into the next.
        """
        assert isinstance(r.REGISTER, tuple)

    def test_row_ids_are_unique(self) -> None:
        """Every document and ticket refers to a row by id, so a duplicate is ambiguous."""
        ids = [row.id for row in r.REGISTER]

        assert len(set(ids)) == len(ids)

    def test_a_row_cannot_be_edited_in_place(self) -> None:
        """The register is a specification.

        A test able to edit a row could make its own failure disappear; the
        falsification cases build a modified copy instead.
        """
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.REGISTER[0].conditional = True  # type: ignore[misc]

    def test_every_row_is_fully_populated(self) -> None:
        """An empty cell is a row that claims nothing while appearing to."""
        for row in r.REGISTER:
            assert row.site, f"{row.id} names no site"
            assert row.paths, f"{row.id} names no path"
            assert row.design_ref.startswith("§"), f"{row.id} does not point back at the design"

    def test_every_site_is_addressed_as_a_file_and_a_qualified_name(self) -> None:
        """A bare method name resolves in twelve provider modules at once.

        ``build_upstream_headers`` alone is defined in twelve of them, so an
        unqualified site would go on resolving no matter which class was renamed
        — the failure :mod:`tests.harness.test_register_agreement` exists to
        catch, made impossible here by the address form.
        """
        for row in r.REGISTER:
            for site in row.site:
                path, separator, qualified = site.partition(":")
                assert separator, f"{row.id}: site {site!r} names no file"
                assert path.startswith("kitty/") and path.endswith(".py"), f"{row.id}: {site!r}"
                assert qualified, f"{row.id}: site {site!r} names no symbol"


class TestTheTriggerVocabulary:
    """T-W6 indexes the corpus by these names, so the vocabulary is load-bearing."""

    def test_every_row_names_a_trigger_from_the_closed_vocabulary(self) -> None:
        """A free-form string would let two authors spell one condition two ways."""
        for row in r.REGISTER:
            assert isinstance(row.trigger, r.Trigger), f"{row.id} carries {row.trigger!r}"

    def test_every_trigger_in_the_vocabulary_is_used(self) -> None:
        """Drift in the other direction: a member nothing names is a stale condition."""
        used = {row.trigger for row in r.REGISTER}

        assert set(r.Trigger) - used == set()

    def test_a_row_that_always_fires_cannot_also_be_conditional(self) -> None:
        """The two fields must agree, or the row is permanently unsatisfiable.

        §3.3.2 assertion 2 would demand a corpus entry in which an
        ``ALWAYS`` row's mutation is absent. Nobody can write one. This is the
        defect class that reached §3.2.2's own unconditional list — M14, P20 and
        P21 sat outside it while their trigger cells read ``Always``.
        """
        for row in r.REGISTER:
            if row.trigger is r.Trigger.ALWAYS:
                assert not row.conditional, f"{row.id} always fires but is declared conditional"

    def test_rows_sharing_a_trigger_agree_on_whether_it_is_conditional(self) -> None:
        """One condition cannot be a request property for one row and a route property for another."""
        by_trigger: dict[r.Trigger, set[bool]] = {}
        for row in r.REGISTER:
            by_trigger.setdefault(row.trigger, set()).add(row.conditional)

        disagreements = {trigger.name for trigger, flags in by_trigger.items() if len(flags) > 1}

        assert disagreements == set()

    def test_no_two_rows_are_indistinguishable(self) -> None:
        """Guards against a degenerate vocabulary.

        A two-member enum (``ALWAYS``, ``CONDITIONAL``) would satisfy every other
        assertion here and destroy T-W6's index. Sharing is not forbidden
        outright — a row is allowed to be told apart by any of the three axes:

        * P3 and P4 share ``REASONING_EFFORT_PRESENT`` and differ by site;
        * P2a and P2b share a site and a path and differ by trigger, which
          §3.2.2 requires ("the oracle must not treat one as covering the other");
        * P9a and P9b share a site and a trigger and differ by path — one sets
          the User-Agent on three adapters, the other swaps MiMo's auth scheme.

        What must never happen is two rows alike on all three, because then no
        test could show that one fired and the other did not.
        """
        seen: set[tuple[r.Trigger, tuple[str, ...], tuple[str, ...]]] = set()
        for row in r.REGISTER:
            key = (row.trigger, row.site, row.paths)
            assert key not in seen, f"{row.id} is indistinguishable from an earlier row"
            seen.add(key)

    def test_the_two_truncation_rows_do_not_share_a_trigger(self) -> None:
        """§3.2.2 insists M3 and M4 are distinct, and the distinction is the trigger.

        M3 is the pre-processing pass and M4 the one inside compaction. (§3.2.1
        calls M3 "unconditional" meaning the *step* always runs; both rows are
        `conditional=True` here, since neither mutates without an oversized
        result.) One shared trigger would make M4's complement case wrong
        — it would be satisfied by a request that simply carries a small tool
        result, which says nothing about compaction.
        """
        rows = {row.id: row for row in r.REGISTER}

        assert rows["M3"].trigger is not rows["M4"].trigger


class TestThePathsEachRowTouches:
    """KBR-26 acceptance criterion 1: a path, or an escape with a reason."""

    def test_every_row_carries_a_path_or_an_escape_with_a_reason(self) -> None:
        """Asserted over the data, so a newly added row cannot slip past it."""
        assert [problem for row in r.REGISTER for problem in r.row_shape_problems(row)] == []

    def test_a_row_escaping_without_a_reason_is_rejected(self) -> None:
        """The falsification case for the rule above.

        §3.3.1a: an escape with an empty cell "would leave those rows silently
        unfalsifiable". A rule that never rejects anything is the same defect one
        level up.
        """
        silent_escape = dataclasses.replace(
            next(row for row in r.REGISTER if row.id == "P16"),
            not_projectable_reason=None,
        )

        assert any("without a reason" in problem for problem in r.row_shape_problems(silent_escape))

    def test_a_row_mixing_the_escape_with_real_paths_is_rejected(self) -> None:
        """The other malformed shape: a row that claims both to model its effect and not to."""
        both = dataclasses.replace(
            next(row for row in r.REGISTER if row.id == "M1"),
            paths=(c.ENVELOPE_MODEL, c.NOT_PROJECTABLE),
        )

        assert any("mixes the escape" in problem for problem in r.row_shape_problems(both))

    def test_every_path_has_balanced_brackets(self) -> None:
        """`path_matches` raises on an unbalanced bracket rather than mis-splitting.

        Named for what it proves and no more: ``path_matches(p, p)`` is ``True``
        for *any* bracket-balanced string, so this is a syntax check. That a
        pattern names something real is the next test's job.
        """
        for row in r.REGISTER:
            if not row.is_projectable:
                continue
            for path in row.paths:
                assert c.path_matches(path, path), f"{row.id}: {path!r} does not even match itself"

    @pytest.mark.parametrize(("pattern", "concrete"), _SHAPES)
    def test_each_pattern_shape_the_register_uses_names_a_concrete_path(self, pattern: str, concrete: str) -> None:
        """AC R3: a pattern must match a concrete path built from the same helper.

        A row whose pattern matches nothing claims nothing, and §3.3.2 assertion
        1 then reports a *false* I1 breach on a mutation that **is** registered.
        Every wildcard and bare-collection shape the register actually uses is
        listed here, so a change to the matcher that broke one would be caught.
        """
        assert pattern in {path for row in r.REGISTER for path in row.paths}, f"{pattern} is unused by the register"
        assert c.path_matches(pattern, concrete)

    def test_every_register_path_that_is_a_pattern_is_listed_in_the_shapes(self) -> None:
        """Otherwise a newly introduced pattern joins the register unproven.

        :data:`_SHAPES` is hand-written, so nothing would force
        ``turn_path(WILDCARD, "role")`` into it if a future row used one. A
        pattern that matches no concrete path claims nothing, and §3.3.2
        assertion 1 then reports a false I1 breach on a *registered* mutation —
        the direction §3.3.1a calls unrecoverable.

        Named for what it checks: a *pattern*, meaning a path whose form differs
        from the concrete path it names — one carrying a wildcard, or a bare
        collection that claims its members. A plain path such as
        ``headers[user-agent]`` is its own concrete form, so it is out of scope
        here by construction rather than by oversight; :data:`_SHAPES` records
        why.
        """
        covered = {pattern for pattern, _ in _SHAPES}
        used = {path for row in r.REGISTER for path in row.paths if c.WILDCARD in path or path in _COLLECTION_ANCHORS}

        assert used - covered == set(), (
            f"used by the register but never shown to match anything: {sorted(used - covered)}"
        )

    def test_every_path_starts_at_a_root_the_vocabulary_defines(self) -> None:
        """§3.3.1a closes the set of roots. A typo would build an unreachable path."""
        for row in r.REGISTER:
            if not row.is_projectable:
                continue
            for path in row.paths:
                assert path.startswith(_LEGAL_ROOTS), f"{row.id}: {path!r} names no root in the vocabulary"

    def test_every_legal_root_is_one_the_register_actually_uses(self) -> None:
        """The other half, and the half that caught a contradiction.

        ``residual`` was on this list while
        :meth:`test_no_row_is_anchored_at_the_residual` forbade it and §3.3.1a
        called it never legal — two assertions in one class stating opposite
        rules, with nothing to say which was the rule. A permitted-roots list
        nothing checks drifts into describing a vocabulary rather than this
        register's use of it.
        """
        unused = [
            root
            for root in _LEGAL_ROOTS
            if not any(path.startswith(root) for row in r.REGISTER if row.is_projectable for path in row.paths)
        ]

        assert unused == [], f"declared legal but used by no row: {unused}"

    def test_no_row_is_anchored_at_the_residual(self) -> None:
        """§3.3.1a: `residual` is never a legal anchor.

        A bare collection claims its members, and a non-empty residual fails the
        run *before* register matching, so such a row could only ever claim a
        delta the oracle was supposed to stop at — including the injected
        ``x-kitty-trace`` field that is one of §3.3.1's five mandatory
        falsification cases. P1 is the row that wanted it.
        """
        for row in r.REGISTER:
            for path in row.paths:
                assert not path.startswith("residual"), f"{row.id} is anchored at the residual"

    def test_p15_is_anchored_at_strict_and_not_at_the_whole_tool(self) -> None:
        """The regression case §3.3.1a names by hand.

        A pattern is a prefix, so ``conversation.tools[*]`` would claim a
        *deleted tool description* — one of §3.3.1's five oracle falsification
        cases — and the matcher cannot detect that by construction.
        """
        p15 = next(row for row in r.REGISTER if row.id == "P15")

        assert p15.paths == ("conversation.tools[*].strict",)
        assert not c.path_matches(p15.paths[0], "conversation.tools[get_weather].description")

    def test_the_escape_is_used_only_where_the_design_expects_it(self) -> None:
        """§3.3.1a names the whole-body translations and P16; P1 and M15 were added with a reason.

        Pinned so that widening the escape is a deliberate edit here, not a
        quiet way to make a hard row go away.  M15 (KBR-144) is the seventh: the
        two spellings of a Responses ``input`` are one request, so the rewrite is
        invisible to a wire-independent projection for P16's reason.
        """
        escaped = {row.id for row in r.REGISTER if not row.is_projectable}

        assert escaped == {"M2", "M9", "M15", "P1", "P11", "P12", "P16"}


class TestTheModuleStandsAlone:
    """§3.3.1's independent-oracle rule, applied to the specification itself.

    The pattern is **imported from** :mod:`tests.harness.test_contract` rather
    than restated. A second regex written from memory is how one guard ends up
    catching nine spellings and its twin catching three, with nothing to say
    which is which — and this guard's only value is that it cannot rot.
    """

    def test_the_register_imports_nothing_from_kitty(self) -> None:
        """A specification that reads its subject is satisfied by whatever the subject does.

        :func:`~harness.register.defined_symbols` reads ``src/kitty`` as text,
        through :mod:`ast`, which is why the source scan does not breach this.

        Read from source rather than by inspecting imports, so an import inside a
        function body is caught as well as a module-level one.
        """
        source = Path(r.__file__).read_text(encoding="utf-8")

        # A guard that passes on an empty read is indistinguishable from one that
        # cannot fail (house rule, `tests/test_egress_coverage.py`).
        assert len(source) > 1000, "read no meaningful source; the guard would pass vacuously"

        offending = [line.strip() for line in source.splitlines() if _KITTY_IMPORT.search(line)]

        assert offending == [], f"register.py must not import kitty: {offending}"

    def test_the_import_guard_fires_on_every_form_it_claims_to_catch(self) -> None:
        """The positive control. Without it a dead pattern reads as a clean bill of health."""
        forms = [
            "from kitty.bridge import server",
            "import kitty",
            "from  kitty import server",
            "import  kitty.bridge",
            "from src.kitty import server",
            'mod = importlib.import_module("kitty.bridge.server")',
            '__import__("kitty")',
            'mod = importlib.import_module("src.kitty.bridge.server")',
            '__import__("src.kitty")',
        ]

        undetected = [form for form in forms if not _KITTY_IMPORT.search(form)]

        assert undetected == [], f"the guard would miss these: {undetected}"

    def test_the_import_guard_does_not_fire_on_innocent_text(self) -> None:
        """The negative control: a pattern matching everything would also pass above.

        The last two lines are real content of ``register.py`` — it discusses
        importing kitty at length and reads ``src/kitty`` through :mod:`ast`.
        """
        innocent = [
            "# kitty-bridge is the product under test",
            "from harness import contract as c",
            "**It imports nothing from ``src/kitty``, and must not.**",
            'for path in sorted(src_root.rglob("*.py")):',
        ]

        assert [line for line in innocent if _KITTY_IMPORT.search(line)] == []
