"""L1 contract tests for the eval task-set framework (KBR-111, plan task T-K2).

Plan task **T-K2 — Independently authored task set** delivers the
loader, the manifest, the human-authorship invariant, and the authoring
protocol — not the tasks themselves. ``TEST_SUITE.md`` §6.4.3 line 2422
is the rule these tests pin: *A model-generated test that the model's
own code passes establishes nothing — the same misunderstanding can be
present in both.*

These tests prove the framework's contracts at L1:

- a person-authored task loads cleanly (AC-1.1);
- a model-authored task is refused with a message that names the rule
  and the remediation (AC-1.2, AC-1.3, AC-1.5) — and ``person("")`` is
  refused at construction (AC-1.4);
- the empty set is refused (AC-5.1);
- the id-equals-stem rule is structural, the discovery is structural,
  and the structural refusals name the file (AC-4.1 – AC-4.4);
- the revision is deterministic and sensitive to a file's bytes
  (AC-2.1, AC-2.2), and the loader's snapshot is one atomic value
  (AC-2.3);
- the framework imports nothing from ``src/kitty`` (AC-3.1);
- the README names the gate's refusal messages (AC-7.1) so it cannot
  silently drift from what the loader actually emits.

**Layer.** This file defaults to ``l1`` by path (``tests/harness/`` is
not in ``tests.layers._PATH_DEFAULTS``); no explicit marker is needed.
The composition proof lives in
``tests/harness/test_eval_tasks_composition.py`` (eval-marked).

**New file, LF.** Per ``kitty-bridge-tests-conftest-crlf``.
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from typing import Any

import pytest

import harness.eval_tasks as eval_tasks
from harness.eval_harness import TaskSpec, Verdict
from harness.eval_tasks import (
    EvalTask,
    EvalTaskAuthor,
    TaskSet,
    load_task_set,
)
from harness.test_contract import _KITTY_IMPORT

# ── REQ 1 — The authorship invariant ────────────────────────────────────────


def test_eval_task_author_person_refuses_an_empty_name() -> None:
    """AC-1.4 — empty and whitespace-only names are refused at construction.

    A person authorship without a name is not a traceable authorship,
    and accepting one would let a contributor slip a model-style
    authorship past the gate by passing the empty string.
    """
    with pytest.raises(ValueError):
        EvalTaskAuthor.person("")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        EvalTaskAuthor.person("   ")  # type: ignore[arg-type]


def test_eval_task_author_constructor_refuses_empty_name() -> None:
    """AC-1.4 (structural) — hand-rolled construction also refuses empty names.

    The ``.person()`` factory refuses empty input at its boundary,
    but a contributor who hand-constructs
    ``EvalTaskAuthor(kind="person", name="")`` would otherwise slip
    past the structural rule. ``__post_init__`` closes that path:
    every constructor call goes through it, regardless of which
    factory the caller used. Without this, an empty-name person
    authorship reaches the loader's gate (which inspects ``kind``
    only) and silently loads — defeating AC-1.4's intent.
    """
    with pytest.raises(ValueError):
        EvalTaskAuthor(kind="person", name="")
    with pytest.raises(ValueError):
        EvalTaskAuthor(kind="person", name="   ")
    # A non-empty name on a person authorship still constructs.
    ada = EvalTaskAuthor(kind="person", name="ada")
    assert ada.kind == "person"
    assert ada.name == "ada"
    # MODEL is exempt from the empty-name rule because its name is a
    # refusal-message label, not an author's identity. A model-shaped
    # authorship with an empty name still constructs — and the loader
    # refuses it on the ``kind`` check at the gate.
    assert EvalTaskAuthor.MODEL.kind == "model"
    assert EvalTaskAuthor.MODEL.name  # non-empty in practice


def test_eval_task_author_person_accepts_a_real_name() -> None:
    """AC-1.1 (construction half) — ``person("ada")`` is a valid authorship."""
    author = EvalTaskAuthor.person("ada")
    assert author.kind == "person"
    assert author.name == "ada"


def test_eval_task_author_model_is_the_rejected_sentinel() -> None:
    """AC-1.2 (sentinel shape) — ``MODEL`` is the named rejected sentinel.

    Pinning the exact shape (not identity) catches a future regression
    that lets ``MODEL`` drift — e.g. an authorship named "model" but
    with ``kind="human"``. A tautological ``MODEL is MODEL`` cannot
    fail; comparing to the literal canonical shape can.
    """
    assert EvalTaskAuthor.MODEL.kind == "model"
    assert EvalTaskAuthor(kind="model", name="model-generated") == EvalTaskAuthor.MODEL


def _accept_anything(_reply: Any) -> Verdict:
    """A trivial PASS check, used to keep AC-1.3 content-independent."""
    return Verdict.PASS


def _write_task_module(root: Path, name: str, body: str) -> Path:
    """Write ``<root>/<name>.py`` with ``body`` and return the path."""
    path = root / f"{name}.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_load_task_set_accepts_a_person_authored_task(tmp_path: Path) -> None:
    """AC-1.1 — a person-authored task loads and is a ``TaskSpec``."""
    _write_task_module(
        tmp_path,
        "demo",
        """
        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor

        TASK = EvalTask(
            id="demo",
            prompt="hello",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=EvalTaskAuthor.person("ada"),
        )
        """,
    )

    loaded = load_task_set(root=tmp_path)

    assert isinstance(loaded, TaskSet)
    assert len(loaded.tasks) == 1
    task = loaded.tasks[0]
    assert isinstance(task, TaskSpec)
    assert task.id == "demo"
    assert task.prompt == "hello"
    assert task.acceptance_check(object()) is Verdict.PASS


def test_load_task_set_refuses_a_model_authored_task_naming_the_rule(
    tmp_path: Path,
) -> None:
    """AC-1.2 — the gate fires before ``run_eval`` runs and cites §6.4.3.

    This is also the §1.4 falsification case: a loader defect that
    returns the model-authored task instead of refusing is detected by
    ``pytest.raises(ValueError)`` with a message naming the rule, the
    id, and the remediation.
    """
    _write_task_module(
        tmp_path,
        "demo",
        """
        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor

        TASK = EvalTask(
            id="demo",
            prompt="hello",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=EvalTaskAuthor.MODEL,
        )
        """,
    )

    with pytest.raises(ValueError) as exc_info:
        load_task_set(root=tmp_path)

    message = str(exc_info.value)
    # The three pieces an operator needs to fix it: the offending id,
    # the rule, and the remediation.
    assert "demo" in message
    assert "§6.4.3" in message
    assert "EvalTaskAuthor.person(" in message


def test_load_task_set_refuses_a_model_authored_task_with_a_trivial_check(
    tmp_path: Path,
) -> None:
    """AC-1.3 — the gate is structural, not content-dependent.

    Same as AC-1.2, but the acceptance check is the trivial
    ``non-empty reply is PASS`` lambda. A gate that examined content
    would behave differently here; a structural gate refuses
    identically. The assertion compares both refusal messages modulo
    the id, so a mutation that smuggles check-content into the
    message fails this test.
    """
    _write_task_module(
        tmp_path,
        "demo",
        """
        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor

        TASK = EvalTask(
            id="demo",
            prompt="hello",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=EvalTaskAuthor.MODEL,
        )
        """,
    )
    with pytest.raises(ValueError) as exc_info:
        load_task_set(root=tmp_path)
    demo_message = str(exc_info.value)

    # Wipe and reload with the trivial AC-1.3 acceptance check.
    (tmp_path / "demo.py").unlink()
    _write_task_module(
        tmp_path,
        "trivial",
        """
        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor

        TASK = EvalTask(
            id="trivial",
            prompt="hi",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=EvalTaskAuthor.MODEL,
        )
        """,
    )
    with pytest.raises(ValueError) as exc_info:
        load_task_set(root=tmp_path)
    trivial_message = str(exc_info.value)

    # Identical modulo the id — pins structural equivalence. Both
    # lambdas above literally ``lambda _reply: Verdict.PASS``, matching
    # the AC text exactly.
    sentinel = "<ID>"
    assert demo_message.replace("'demo'", sentinel) == trivial_message.replace(
        "'trivial'", sentinel
    )


def test_load_task_set_refuses_a_forged_kind_only_person_passes(
    tmp_path: Path,
) -> None:
    """AC-1.5 — only ``kind == "person"`` passes; forgery is fail-closed.

    Two forged shapes, both refused: the AC's literal
    ``EvalTaskAuthor(kind="human", name="x")`` (a genuine EvalTaskAuthor
    instance with a non-person kind) and an unrelated dataclass
    (whatever shape a future contributor might reach for). The loader
    dispatches on ``kind``, so both fail the same gate.
    """
    # The AC's literal forged shape: a genuine EvalTaskAuthor whose
    # kind is not "person". The dataclass constructor does not
    # validate ``kind`` — the gate does.
    _write_task_module(
        tmp_path,
        "forged",
        """
        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor

        TASK = EvalTask(
            id="forged",
            prompt="hi",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=EvalTaskAuthor(kind="human", name="x"),
        )
        """,
    )

    with pytest.raises(ValueError) as exc_info:
        load_task_set(root=tmp_path)
    assert "forged" in str(exc_info.value)

    # The unrelated-dataclass shape: the gate is duck-typed on
    # ``.kind``, so a foreign object with a non-person kind is
    # refused the same way.
    (tmp_path / "forged.py").unlink()
    _write_task_module(
        tmp_path,
        "notperson",
        """
        from dataclasses import dataclass

        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor


        @dataclass(frozen=True)
        class NotPerson:
            kind: str = "human"
            name: str = "ada"


        TASK = EvalTask(
            id="notperson",
            prompt="hi",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=NotPerson(),  # type: ignore[arg-type]
        )
        """,
    )

    with pytest.raises(ValueError) as exc_info:
        load_task_set(root=tmp_path)
    assert "notperson" in str(exc_info.value)


# ── REQ 4 — Task identity is the filename stem ───────────────────────────────


def test_load_task_set_refuses_a_task_whose_id_differs_from_its_stem(
    tmp_path: Path,
) -> None:
    """AC-4.1 — ``module.TASK.id`` must equal the filename stem."""
    _write_task_module(
        tmp_path,
        "foo",
        """
        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor

        TASK = EvalTask(
            id="bar",
            prompt="hi",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=EvalTaskAuthor.person("ada"),
        )
        """,
    )

    with pytest.raises(ValueError) as exc_info:
        load_task_set(root=tmp_path)
    message = str(exc_info.value)
    assert "foo" in message
    assert "bar" in message


@pytest.mark.parametrize(
    ("stem", "reason"),
    [
        ("Foo", "uppercase first letter — not a snake_case identifier"),
        ("bad-name", "hyphen — not importable as a Python identifier"),
        ("1abc", "leading digit — not importable as a Python identifier"),
    ],
)
def test_load_task_set_refuses_a_module_with_an_invalid_stem(
    tmp_path: Path, stem: str, reason: str
) -> None:
    """AC-4.2 — stems must match ``[a-z][a-z0-9_]*``.

    Parametrised over the three realistic ways a contributor produces
    an invalid stem (capitalised, hyphenated, digit-leading). Each
    shape is refused before the module is even imported — a mutator
    that drops the stem regex entirely fails every case here, and the
    refusal message names the file so the operator finds it.
    """
    task_body = f"""
        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor

        TASK = EvalTask(
            id={stem!r},
            prompt="hi",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=EvalTaskAuthor.person("ada"),
        )
        """
    (tmp_path / f"{stem}.py").write_text(textwrap.dedent(task_body), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_task_set(root=tmp_path)
    message = str(exc_info.value)
    assert f"{stem}.py" in message, f"refusal must name the file ({reason}): {message}"
    assert "stem" in message.lower()


def test_load_task_set_refuses_a_module_without_a_task(tmp_path: Path) -> None:
    """AC-4.3 (missing half) — a module with no ``TASK`` is refused."""
    _write_task_module(
        tmp_path,
        "empty",
        """
        # Intentionally declares nothing.
        sentinel = 1
        """,
    )

    with pytest.raises(ValueError) as exc_info:
        load_task_set(root=tmp_path)
    assert "empty" in str(exc_info.value)


def test_load_task_set_refuses_a_module_whose_task_is_not_an_eval_task(
    tmp_path: Path,
) -> None:
    """AC-4.3 (wrong-type half) — ``TASK`` must be an ``EvalTask``."""
    _write_task_module(
        tmp_path,
        "wrong",
        """
        from harness.eval_tasks import EvalTaskAuthor

        TASK = EvalTaskAuthor.person("ada")  # type: ignore[assignment]
        """,
    )

    with pytest.raises(ValueError) as exc_info:
        load_task_set(root=tmp_path)
    assert "wrong" in str(exc_info.value)


def test_load_task_set_refuses_a_module_that_fails_to_import(
    tmp_path: Path,
) -> None:
    """AC-4.4 — a syntax error in a task module is refused, named, and chained."""
    (tmp_path / "broken.py").write_text("def TASK(\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_task_set(root=tmp_path)
    assert "broken" in str(exc_info.value)


# ── REQ 5 — An empty task set is refused ────────────────────────────────────


def test_load_task_set_refuses_an_empty_registry(tmp_path: Path) -> None:
    """AC-5.1 — a root with no task modules is refused, pointing at the README."""
    # Create one unrelated file to make sure the loader is filtering,
    # not "any non-empty dir is a task set".
    (tmp_path / "not_a_task.txt").write_text("noise", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_task_set(root=tmp_path)
    message = str(exc_info.value)
    assert "empty" in message.lower()
    assert "README_eval_tasks.md" in message


def test_load_task_set_skips_underscore_prefixed_modules(tmp_path: Path) -> None:
    """The underscore convention marks "not a task" — the loader honours it.

    The proof needs a sibling task: with only ``_draft.py`` the test
    passes either way — the underscore filter produces an empty
    registry (refused), but a broken filter would let ``_draft``
    reach the stem check, fail on the leading ``_``, and refuse
    identically. A second valid task lets the assertion be
    ``[task.id for task in loaded.tasks] == ["kept"]`` — proving
    the filter **silently skipped** the draft, not refused it.
    Without the convention a contributor could leave a half-written
    task module called ``_wip.py`` and have it silently load.
    """
    _write_task_module(
        tmp_path,
        "_draft",
        """
        # Underscore-prefixed — the loader must skip this.
        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor

        TASK = EvalTask(
            id="_draft",
            prompt="hi",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=EvalTaskAuthor.MODEL,
        )
        """,
    )
    _write_task_module(
        tmp_path,
        "kept",
        """
        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor

        TASK = EvalTask(
            id="kept",
            prompt="hi",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=EvalTaskAuthor.person("ada"),
        )
        """,
    )

    loaded = load_task_set(root=tmp_path)
    # The draft is silently skipped; the kept one loads. A broken
    # filter would let _draft reach the stem check (leading ``_``
    # fails ``^[a-z][a-z0-9_]*$``) and refuse the entire load —
    # ``loaded.tasks`` would be empty, not ``["kept"]``.
    assert [task.id for task in loaded.tasks] == ["kept"]


# ── REQ 2 — Determinism, atomicity, content-addressed revision ──────────────


def test_load_task_set_is_deterministic_across_calls(tmp_path: Path) -> None:
    """AC-2.1 — two calls over the same root produce equal ``TaskSet`` values."""
    _write_task_module(
        tmp_path,
        "first",
        """
        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor

        TASK = EvalTask(
            id="first",
            prompt="hi",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=EvalTaskAuthor.person("ada"),
        )
        """,
    )
    _write_task_module(
        tmp_path,
        "second",
        """
        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor

        TASK = EvalTask(
            id="second",
            prompt="there",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=EvalTaskAuthor.person("lin"),
        )
        """,
    )

    first = load_task_set(root=tmp_path)
    second = load_task_set(root=tmp_path)

    # Determinism is over the observables that matter: the tasks'
    # identity (id, prompt, author) in a stable order, and the
    # byte-derived revision. Dataclass equality is deliberately not
    # asserted — each load re-executes the module source, so the
    # acceptance checks are fresh function objects with identical
    # behaviour, and no loader can preserve callable identity across
    # two exec_module calls.
    assert [(t.id, t.prompt, t.authored_by) for t in first.tasks] == [
        (t.id, t.prompt, t.authored_by) for t in second.tasks
    ]
    # The revision is the same string both times — no wall-clock, no
    # accidental nondeterminism.
    assert first.revision == second.revision


def test_revision_changes_when_a_task_file_changes(tmp_path: Path) -> None:
    """AC-2.2 — the revision is a function of the files' bytes."""
    path = _write_task_module(
        tmp_path,
        "demo",
        """
        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor

        TASK = EvalTask(
            id="demo",
            prompt="hi",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=EvalTaskAuthor.person("ada"),
        )
        """,
    )

    before = load_task_set(root=tmp_path)

    # Edit the prompt (same id, same author, same check — different bytes).
    path.write_text(
        textwrap.dedent(
            """
            from harness.eval_harness import Verdict
            from harness.eval_tasks import EvalTask, EvalTaskAuthor

            TASK = EvalTask(
                id="demo",
                prompt="good morning",
                acceptance_check=lambda _reply: Verdict.PASS,
                authored_by=EvalTaskAuthor.person("ada"),
            )
            """
        ),
        encoding="utf-8",
    )

    after = load_task_set(root=tmp_path)
    assert before.revision != after.revision


def test_revision_is_a_sha256_hex_digest(tmp_path: Path) -> None:
    """AC-2.2 — the revision is a 64-char lowercase hex SHA-256.

    Pinning the format keeps it diff-able across runs and stops a
    future revision-from-mtime regression slipping in (mtime is
    non-deterministic across CI legs and contributor machines).
    """
    _write_task_module(
        tmp_path,
        "demo",
        """
        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor

        TASK = EvalTask(
            id="demo",
            prompt="hi",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=EvalTaskAuthor.person("ada"),
        )
        """,
    )

    loaded = load_task_set(root=tmp_path)
    assert len(loaded.revision) == 64
    int(loaded.revision, 16)  # raises ValueError if not lowercase hex
    # Cross-check against a hand-computed digest: the same shape the
    # loader must produce, not just a hex-looking string.
    expected = hashlib.sha256(
        b"demo.py\x00" + (tmp_path / "demo.py").read_bytes()
    ).hexdigest()
    assert loaded.revision == expected


# ── REQ 3 — The framework does not depend on src/kitty ───────────────────────


def test_eval_tasks_imports_nothing_from_kitty() -> None:
    """AC-3.1 — the framework's source carries no kitty import.

    A library that did would prove self-consistency, not fidelity
    (§3.3.1). The regex is imported from the canonical definition in
    :mod:`harness.test_contract`, not redefined — the failures library
    guard (``tests/harness/test_failures.py``) and T-K1's own guard
    (``tests/harness/test_eval_harness.py``) use the same constant,
    so a future form added there extends all three guards together,
    and the canonical's own positive/negative control tests pin the
    pattern against silent drift.
    """
    source = eval_tasks.__file__
    assert source is not None, "eval_tasks module has no source path"

    body = Path(source).read_text(encoding="utf-8")
    # A non-trivial body length — an empty file would pass vacuously,
    # the same trap the failures and T-K1 guards avoid.
    assert len(body) > 500, "read no meaningful source; the guard would pass vacuously"

    offending = [line.strip() for line in body.splitlines() if _KITTY_IMPORT.search(line)]
    assert offending == [], f"eval_tasks must not import kitty: {offending}"


# ── AC-2.3 / REQ 6 — the snapshot is one value, ready for run_eval ──────────


def test_task_set_tasks_and_revision_travel_together(tmp_path: Path) -> None:
    """AC-2.3 — the caller cannot mix a task list with a foreign revision.

    The runner consumes ``RunConfig(dataset_revision=...)`` and
    ``tasks: Sequence[TaskSpec]`` from two arguments; the loader's
    ``TaskSet`` carries both in one snapshot so a caller that reads
    them from a single ``load_task_set`` call cannot accidentally mix
    them across two reads. A future loader backed by a changing
    registry could not violate the snapshot; today's committed
    registry is read-only, and the test pins the shape.
    """
    _write_task_module(
        tmp_path,
        "demo",
        """
        from harness.eval_harness import Verdict
        from harness.eval_tasks import EvalTask, EvalTaskAuthor

        TASK = EvalTask(
            id="demo",
            prompt="hi",
            acceptance_check=lambda _reply: Verdict.PASS,
            authored_by=EvalTaskAuthor.person("ada"),
        )
        """,
    )

    loaded = load_task_set(root=tmp_path)

    # `tasks` is a tuple (immutable) and `revision` is a str (hashable).
    assert isinstance(loaded.tasks, tuple)
    assert isinstance(loaded.revision, str)
    # The runner consumes ``Sequence[TaskSpec]``; every entry must pass
    # the isinstance check so the harness's own type contract is upheld.
    for task in loaded.tasks:
        assert isinstance(task, TaskSpec)


# ── AC-7.1 — The README documents the gate ─────────────────────────────────


def test_readme_names_the_gate_messages_and_works_example() -> None:
    """AC-7.1 — the README names the refusal messages and shows a worked example.

    A protocol doc that omits the gate's exact strings lets a future
    author invent different ones and silently diverge. The test scans
    the README for literal fragments of refusal messages the loader
    actually emits, so a README that strips the actual refusal blocks
    but keeps generic phrasing fails here.
    """
    readme = Path(eval_tasks.__file__).resolve().parent.parent / "README_eval_tasks.md"
    body = readme.read_text(encoding="utf-8")

    # Worked example fragments — the README must show what an author writes.
    assert "EvalTaskAuthor.person(" in body, "README must show the remediation"
    assert "§6.4.3" in body, "README must cite the rule"
    assert "TASK = EvalTask(" in body, "README must show the worked example"

    # Literal fragments of the loader's actual refusal messages — the
    # scan catches a README that paraphrases the gate but stops naming
    # the strings the loader emits.
    assert "is model-authored" in body, (
        "README must quote the authorship refusal so a future author "
        "recognises it in CI"
    )
    assert "the eval task set is empty" in body, (
        "README must quote the empty-set refusal so a future author "
        "recognises it in CI"
    )
    assert "declares TASK.id=" in body, (
        "README must quote the id-equals-stem refusal so a future "
        "author recognises it in CI"
    )
    assert "invalid stem" in body, (
        "README must quote the invalid-stem refusal so a future "
        "author recognises it in CI"
    )


# ── helpers ──────────────────────────────────────────────────────────────────


def test_eval_task_is_a_task_spec() -> None:
    """``EvalTask(TaskSpec)`` keeps the harness's type contract honest."""
    author = EvalTaskAuthor.person("ada")
    task = EvalTask(
        id="x",
        prompt="hi",
        acceptance_check=_accept_anything,
        authored_by=author,
    )
    assert isinstance(task, TaskSpec)
    # The dataclass inheritance carries the base fields through.
    assert task.id == "x"
    assert task.prompt == "hi"
    assert task.authored_by is author
