"""The eval task-set framework (KBR-111, plan task **T-K2**).

`TEST_SUITE.md` §6.4.3 line 2422 names the rule this module enforces:
*"A model-generated test that the model's own code passes establishes
nothing — the same misunderstanding can be present in both."* The
framework's reason for existing is to make that rule testable: the
loader refuses any task whose ``authored_by.kind`` is not ``"person"``,
naming the offending id and citing the rule. The team authors the
acceptance tests; the gate accepts nothing else.

**What this module owns**

- :class:`EvalTaskAuthor` — a frozen value type carrying who wrote a
  task. ``EvalTaskAuthor.MODEL`` is a named rejected sentinel;
  ``EvalTaskAuthor.person(name)`` is the only constructor the gate
  accepts.
- :class:`EvalTask` — a frozen dataclass subclassing
  :class:`harness.eval_harness.TaskSpec` that adds one field,
  ``authored_by``. Every ``EvalTask`` is a ``TaskSpec`` and is a valid
  argument to :func:`harness.eval_harness.run_eval`.
- :class:`TaskSet` — a frozen dataclass carrying the loaded tasks and
  their content-addressed revision in a single snapshot. The caller
  cannot mix a task list from one registry state with a revision from
  another.
- :func:`load_task_set` — the loader. Walks ``<root>/*.py`` (excluding
  ``__init__.py`` and ``_``-prefixed modules), imports each, and
  harvests exactly one task per module from a declared ``TASK``
  constant. The discovery is structural: there is no ``register_task()``
  to forget, so the "module constant is the only way a task enters the
  set" claim is a shape, not a convention.

**What this module does NOT own**

- Real eval tasks (the team's follow-on work);
- statistics, decision rule, missing-data ceiling (T-K3, blocked on Q4
  and Q13);
- the on-disk run record (T-K1 already owns that).

**Authoring protocol.** ``tests/harness/README_eval_tasks.md`` is the
end-to-end guide a human follows to add a task. The file is the
single source of truth for the protocol; this docstring is the why.

**Import rule.** This module imports nothing from ``src/kitty`` —
mirroring T-K1's rule for ``eval_harness``. The guard is
``test_eval_tasks.py::test_eval_tasks_imports_nothing_from_kitty``,
which uses the canonical ``_KITTY_IMPORT`` regex from
:mod:`harness.test_contract`.

**New file, LF.** Per ``kitty-bridge-tests-conftest-crlf``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

from harness.eval_harness import TaskSpec

__all__ = [
    "EvalTask",
    "EvalTaskAuthor",
    "TaskSet",
    "load_task_set",
]

_STEM_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class EvalTaskAuthor:
    """Who wrote a task — the loader accepts only ``kind == "person"``.

    The gate is fail-closed by shape: any value whose ``kind`` is not
    ``"person"`` is refused. A hand-forged
    ``EvalTaskAuthor(kind="human", ...)`` and ``MODEL`` both fail the
    gate the same way; there is no "almost MODEL" the loader quietly
    accepts. The discipline keeps a contributor from smuggling in a
    model-shaped authorship under a renamed kind.

    Attributes:
        kind: ``"person"`` for an authored-by-a-human task, ``"model"``
            for the rejected sentinel. Any other value is treated as
            model-shaped and refused.
        name: The author's display name. Required for ``kind == "person"``;
            the :meth:`person` constructor refuses an empty or
            whitespace-only value. For ``MODEL``, a short label that
            the loader names in its refusal messages so a maintainer
            reading the log sees what was found.
    """

    kind: str
    name: str

    @staticmethod
    def person(name: str) -> EvalTaskAuthor:
        """Return a person authorship; refuse an empty or whitespace-only name.

        Args:
            name: The author's display name. Trimmed; the trimmed value
                is recorded.

        Returns:
            A person authorship carrying ``name``.

        Raises:
            ValueError: When ``name`` is empty or whitespace-only. A
                person authorship without a name is not a traceable
                authorship, and accepting one would let a contributor
                bypass the gate by passing ``""``.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "EvalTaskAuthor.person() requires a non-empty name; the author "
                "is recorded for traceability and the empty string is reserved "
                "for the MODEL sentinel (TEST_SUITE.md §6.4.3: independently "
                "authored acceptance tests)"
            )
        return EvalTaskAuthor(kind="person", name=name.strip())


#: The named rejected sentinel — there to be refused. Carrying the
#: literal string ``"model-generated"`` as its name lets the loader's
#: refusal message name what was found, so an operator reading a CI
#: log can see at a glance that an authorship slot was left as the
#: rejected shape rather than replaced with a real ``person(...)``.
EvalTaskAuthor.MODEL = EvalTaskAuthor(
    kind="model",
    name="model-generated",
)


@dataclass(frozen=True)
class EvalTask(TaskSpec):
    """One task with an authorship attached — a ``TaskSpec`` plus who wrote it.

    The harness consumes the base ``TaskSpec`` fields (``id``,
    ``prompt``, ``acceptance_check``); ``authored_by`` is the field the
    gate reads. Subclassing ``TaskSpec`` (a frozen dataclass) lets
    ``EvalTask`` instances pass ``isinstance(TaskSpec)`` and feed
    directly into ``run_eval`` without an adapter.

    Attributes:
        id: The task's stable identifier; the loader enforces
            ``id == filename stem``.
        prompt: What the arm is asked.
        acceptance_check: The callable judging a reply and returning a
            :class:`Verdict`. Per-task by design (§6.4.3: what counts
            as a refusal is evidence about *this* task's context).
        authored_by: Who wrote the acceptance check. The loader
            refuses any task whose ``authored_by.kind != "person"``.
    """

    authored_by: EvalTaskAuthor


@dataclass(frozen=True)
class TaskSet:
    """The loaded registry: every task plus its content-addressed revision.

    One snapshot, one revision — the caller cannot mix a task list
    from one registry state with a revision from another. The runner
    consumes ``TaskSet.tasks`` as ``Sequence[TaskSpec]`` and pins
    ``TaskSet.revision`` into ``RunConfig.dataset_revision`` per
    §6.4.3 ("dataset revision ... pinned and recorded with each run").

    Attributes:
        tasks: The loaded tasks, ordered as the loader discovered them
            (sorted by filename stem).
        revision: A SHA-256 hex digest over the sorted task-file
            bytes. Stable across processes and machines — no
            mtime, no clock, no random.
    """

    tasks: tuple[TaskSpec, ...]
    revision: str


def load_task_set(root: Path | None = None) -> TaskSet:
    """Load the task set under ``root`` and return one ``TaskSet`` snapshot.

    The loader walks ``<root>/*.py`` (excluding ``__init__.py`` and
    files whose stem starts with ``_``), imports each module in
    sorted order, and harvests exactly one task per module from a
    module-level ``TASK = EvalTask(...)`` declaration. The structural
    shape — one TASK per file, id-equals-stem — is what makes
    "the module constant is the only way a task enters the set" a
    property of the loader rather than a convention contributors
    must remember.

    Args:
        root: The directory to load. ``None`` means the package's
            own directory (``tests/harness/eval_tasks/``). Tests
            pass a temp directory so they can write task modules
            hermetically without touching the shipped registry.

    Returns:
        A :class:`TaskSet` whose ``tasks`` carry every loaded task
        and whose ``revision`` is a SHA-256 hex digest over the
        sorted task-file bytes.

    Raises:
        ValueError: When ``root`` carries no task modules (the empty
            case is refused — a run with zero tasks has nothing to
            measure and no denominator to hold); when a module
            declares no ``TASK``; when ``TASK`` is not an
            :class:`EvalTask`; when ``TASK.id`` does not equal the
            filename stem; when ``authored_by.kind != "person"``; or
            when a module fails to import. Each message names the
            file and (where relevant) the offending id and the
            remediation.
    """
    if root is None:
        root = Path(__file__).resolve().parent

    # Sorted by stem for a stable ordering: two machines that load
    # the same directory produce the same ``tasks`` tuple order, so
    # the revision is reproducible across hosts.
    task_paths = sorted(
        path
        for path in root.glob("*.py")
        if path.name != "__init__.py" and not path.stem.startswith("_")
    )

    if not task_paths:
        raise ValueError(
            "the eval task set is empty (TEST_SUITE.md §6.4.3 requires a fixed "
            "task set; add at least one person-authored task — see "
            "tests/harness/README_eval_tasks.md)"
        )

    tasks: list[TaskSpec] = []
    revision_input = bytearray()

    for path in task_paths:
        stem = path.stem
        if not _STEM_PATTERN.fullmatch(stem):
            raise ValueError(
                f"eval task module {path.name!r} has an invalid stem; stems must "
                f"match {_STEM_PATTERN.pattern!r} (lowercase snake_case, "
                "importable as a Python identifier)"
            )

        # One read per file, feeding both the revision and the exec:
        # the bytes the revision was computed over are the bytes the
        # module was compiled from, so a file changed between two
        # reads could not mix a task list from one registry state with
        # a revision from another (REQ 2's atomicity claim).
        source = path.read_bytes()

        # Per-file prefix in the revision: the filename, a NUL that
        # cannot occur in a UTF-8 source file, then the bytes. The
        # NUL separator stops two files with the same content from
        # hashing to the same digest (the corpus's body_sha256 uses
        # the same shape — see tests/corpus/README.md).
        revision_input.extend(path.name.encode("utf-8"))
        revision_input.append(0)
        revision_input.extend(source)

        task = _harvest_task(path, stem, source)
        if task.authored_by.kind != "person":
            # The §1.4 falsification case (AC-1.2 / AC-1.3): the gate
            # refuses the entire load the moment it sees a non-person
            # authorship, naming the offending id and the rule. A
            # loader that silently dropped the offending task would
            # pass every other AC but fail this one — the diagnosis
            # is "the loader forgot to refuse" rather than "the
            # acceptance check was wrong."
            raise ValueError(
                f"eval task {task.id!r} is model-authored (authored_by="
                f"{task.authored_by.name!r}); TEST_SUITE.md §6.4.3 requires "
                "independently authored acceptance tests — set authored_by="
                'EvalTaskAuthor.person(name="<your name>") before loading'
            )
        tasks.append(task)

    revision = hashlib.sha256(bytes(revision_input)).hexdigest()
    return TaskSet(tasks=tuple(tasks), revision=revision)


def _harvest_task(path: Path, stem: str, source: bytes) -> EvalTask:
    """Import ``path`` and return the module's declared ``TASK``.

    Args:
        path: The task file on disk.
        stem: The filename stem (``path.name`` without ``.py``),
            pre-validated against :data:`_STEM_PATTERN`.
        source: The file's bytes — the **same** bytes the revision
            was computed over. Compiling from these (rather than a
            second read through ``importlib``) closes the TOCTOU
            window between the hash and the exec (REQ 2's
            atomicity claim).

    Returns:
        The module's ``TASK`` attribute, typed as :class:`EvalTask`.

    Raises:
        ValueError: When the module fails to import (the original
            exception is chained and the message names the file);
            when the module declares no ``TASK``; when ``TASK`` is
            not an :class:`EvalTask`; or when ``TASK.id`` does not
            equal ``stem``. Each branch names the file so an
            operator reading a CI log finds the offender immediately.
    """
    # Synthetic module name avoids colliding with anything already
    # imported; ``harness.eval_tasks.<stem>`` would be the natural
    # name, but a test that points the loader at a tmp_path outside
    # the package tree cannot import relative to it. The synthetic
    # name keeps the loader working at any root.
    spec = importlib.util.spec_from_file_location(f"_eval_task_{stem}", path)
    if spec is None:
        raise ValueError(
            f"eval task module {path.name!r}: importlib refused to build a spec "
            "for this file"
        )

    module = importlib.util.module_from_spec(spec)
    try:
        # Compile from ``source`` (the same bytes the revision was
        # computed over) and exec into the freshly-created module's
        # namespace. We do **not** call ``spec.loader.exec_module``
        # because that would re-read the file — defeating the
        # single-read design above.
        code = compile(source, str(path), "exec")
        exec(code, module.__dict__)  # noqa: S102 — controlled input, sandboxed by spec
    except Exception as exc:
        # ``Exception``, not ``BaseException`` — a Ctrl-C mid-load
        # or a module that calls ``sys.exit()`` must propagate so the
        # operator can stop a long nightly, mirroring
        # ``harness.eval_harness.run_eval``'s own per-trial guard.
        raise ValueError(
            f"eval task module {path.name!r} failed to import; the loader does "
            "not catch and rewrite the error, so the original traceback is "
            f"chained. Fix the module: {type(exc).__name__}: {exc}"
        ) from exc

    if not hasattr(module, "TASK"):
        raise ValueError(
            f"eval task module {path.name!r} declares no TASK; every task module "
            "must declare a single TASK = EvalTask(...) at module scope"
        )
    task = module.TASK
    if not isinstance(task, EvalTask):
        raise ValueError(
            f"eval task module {path.name!r} declares TASK of type "
            f"{type(task).__name__}; the loader accepts only EvalTask"
        )
    if task.id != stem:
        raise ValueError(
            f"eval task module {path.name!r} declares TASK.id={task.id!r}; the id "
            f"must equal the filename stem {stem!r}"
        )
    return task
