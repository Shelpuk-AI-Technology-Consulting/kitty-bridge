# Eval task-set authoring protocol (KBR-111, plan task T-K2)

The eval task set lives under `tests/harness/eval_tasks/`. This document
is the only thing a human adding a task has to read end-to-end; the
loader enforces every rule below.

## What a task is

A task is a Python module declaring exactly one module-level
`TASK = EvalTask(...)` constant:

```python
# tests/harness/eval_tasks/<id>.py
from harness.eval_harness import Verdict
from harness.eval_tasks import EvalTask, EvalTaskAuthor


def _is_a_real_answer(reply: object) -> Verdict:
    """The acceptance check: 'right' for this task."""
    if not isinstance(reply, str):
        return Verdict.FAIL
    # The actual judgement: a person writes this. The framework does
    # not know what 'right' means for your task — only you do.
    return Verdict.PASS if reply.startswith("42") else Verdict.FAIL


TASK = EvalTask(
    id="<id>",
    prompt="<the prompt the model sees>",
    acceptance_check=_is_a_real_answer,
    authored_by=EvalTaskAuthor.person(name="<your name>"),
)
```

The framework reads four fields:

- `id` — the task's stable identifier. **Must equal the filename
  stem** (the loader checks this; see *The id rule* below).
- `prompt` — what the arm is asked.
- `acceptance_check` — a callable taking the model's reply and
  returning one of `Verdict.PASS`, `Verdict.FAIL`, `Verdict.REFUSED`.
- `authored_by` — who wrote the acceptance check. **Must be
  `EvalTaskAuthor.person(name="<your name>")`**. The loader refuses
  any other value.

## Why the `authored_by` field exists

`TEST_SUITE.md` §6.4.3 line 2422 names the rule:

> *A model-generated test that the model's own code passes establishes
> nothing — the same misunderstanding can be present in both.*

The framework's reason for existing is to make that rule **testable**.
The loader refuses any task whose `authored_by.kind` is not `"person"`,
so a model-authored acceptance check cannot reach `run_eval`. The rule
is structural: a task with the trivial `non-empty reply is PASS`
lambda is refused identically to one with a sophisticated check.

`EvalTaskAuthor.person("")` and `EvalTaskAuthor.person("   ")` both
raise `ValueError` at construction. A person authorship without a name
is not a traceable authorship, and the empty-name rule is enforced
structurally — `EvalTaskAuthor.__post_init__` refuses it for any
`kind="person"` instance, so a hand-rolled
`EvalTaskAuthor(kind="person", name="")` is refused the same way as
the public factory.

## The id rule

`TASK.id` must equal the filename stem. Two tasks cannot share an id,
because two files cannot share a stem in a directory — the constraint
is structural, not a separate check. Renaming the file requires
renaming `TASK.id`; renaming `TASK.id` requires renaming the file.

Stems must match `[a-z][a-z0-9_]*` — lowercase snake_case, importable
as a Python identifier. `Foo.py`, `bad-name.py`, `1.py` are all
refused.

## How to test your task locally

Two layers:

```sh
# L1 contract tests — run by `pytest` by default.
.venv/bin/python -m pytest tests/harness/test_eval_tasks.py -q

# Eval slice — runs the loaded task end-to-end through both arms of
# the bridge. Excluded from the default `pytest` run per
# `pyproject.toml` `addopts`; opt in explicitly.
.venv/bin/python -m pytest -m eval tests/harness/test_eval_tasks_composition.py -q
```

The composition test writes a one-task registry to a temp directory,
calls `load_task_set(root=...)`, and drives the resulting
`TaskSet.tasks` through `run_eval` with the `BridgeFixture` pair T-K1's
vertical slice uses. If your acceptance check misclassifies the
scripted upstream's reply, this is the test that catches it.

## The gate's refusal messages

These are the refusals an author is most likely to meet. Every
refusal the loader raises names the offending file (and, where
relevant, the task id and the remediation), so a CI log always
tells you which module to fix.

### Model-authored task

```
eval task '<id>' is model-authored (authored_by='model-generated');
TEST_SUITE.md §6.4.3 requires independently authored acceptance tests —
set authored_by=EvalTaskAuthor.person(name="<your name>") before loading
```

**Fix.** Replace `EvalTaskAuthor.MODEL` (or any non-person authorship)
with `EvalTaskAuthor.person(name="<your name>")`. The name is recorded
in the run record and shown in CI logs.

### Empty registry

```
the eval task set is empty (TEST_SUITE.md §6.4.3 requires a fixed
task set; add at least one person-authored task — see
tests/harness/README_eval_tasks.md)
```

**Fix.** Add at least one task module under `tests/harness/eval_tasks/`.
The framework refuses an empty set rather than silently running zero
trials — a run with zero trials has nothing to measure and no
denominator to hold.

### Id does not equal stem

```
eval task module '<file>.py' declares TASK.id='<id>'; the id must
equal the filename stem '<stem>'
```

**Fix.** Rename the file and `TASK.id` so they agree. The structural
rule is the loader's, not yours — both must change.

### Invalid stem

```
eval task module '<file>.py' has an invalid stem; stems must match
'^[a-z][a-z0-9_]*$' (lowercase snake_case, importable as a Python
identifier)
```

**Fix.** Rename the file (and `TASK.id`) to lowercase snake_case —
`Foo.py`, `bad-name.py`, and `1abc.py` are all refused.

### Module declares no `TASK`

```
eval task module '<file>.py' declares no TASK; every task module must
declare a single TASK = EvalTask(...) at module scope
```

**Fix.** Add the `TASK = EvalTask(...)` line. A file with no `TASK`
is silently ignored and the loader treats the registry as empty;
this refusal turns the silent miss into a loud one.

### `TASK` is not an `EvalTask`

```
eval task module '<file>.py' declares TASK of type '<type>';
the loader accepts only EvalTask
```

**Fix.** Make `TASK` an `EvalTask` instance. Any other value — an
`EvalTaskAuthor`, a plain `TaskSpec`, a function — is refused.

### Module fails to import

```
eval task module '<file>.py' failed to import; the loader does not
catch and rewrite the error, so the original traceback is chained.
Fix the module: <ExceptionType>: <message>
```

**Fix.** Open the file; the chained traceback is the original error
(a syntax error, a missing import, a bad constructor argument). The
loader refuses rather than skipping the module — a broken task never
silently drops out of the set.

## Files a task module may reference

- `harness.eval_harness.Verdict` — `PASS`, `FAIL`, `REFUSED`.
- `harness.eval_tasks.EvalTask`, `EvalTaskAuthor` — the loader's
  types.
- Anything in `tests/harness/` — fixtures, builders, helpers. The
  harness package is independent of `src/kitty` and stays that way;
  a task module that imports from `src/kitty` will be caught by
  the `test_eval_tasks_imports_nothing_from_kitty` guard at L1.

## What stays out of a task module

- `@pytest.fixture`, `@pytest.mark.*`, or any pytest machinery — the
  loader does not run pytest against task modules; it imports them.
- `register_task(...)` — there is no such function. The structural
  shape (one `TASK =` per file) is what makes the load deterministic.
- `EvalTaskAuthor.MODEL` in a file that ships — the loader refuses the
  whole load when it sees one. Use `MODEL` only in tests that prove
  the gate fires.

## Worked example — full task file

```python
# tests/harness/eval_tasks/trivia_capital_of_france.py

from harness.eval_harness import Verdict
from harness.eval_tasks import EvalTask, EvalTaskAuthor


def _capital_of_france(reply: object) -> Verdict:
    """The real judgement — written by a person.

    Accepts any reply that names Paris; the wording and casing are the
    caller's. A refusal-shaped error from the model classifies
    ``REFUSED`` rather than ``FAIL`` so §6.4.3's failure taxonomy
    can report 'refused' separately from 'wrong'.
    """
    if not isinstance(reply, str):
        return Verdict.FAIL
    cleaned = reply.strip().rstrip(".").lower()
    return Verdict.PASS if "paris" in cleaned else Verdict.FAIL


TASK = EvalTask(
    id="trivia_capital_of_france",
    prompt="What is the capital of France?",
    acceptance_check=_capital_of_france,
    authored_by=EvalTaskAuthor.person(name="Ada Lovelace"),
)
```

This file lives at `tests/harness/eval_tasks/trivia_capital_of_france.py`
because the stem equals the id.

## What the runner records

When a run executes, the on-disk record (`RunRecord.config`) carries
`dataset_revision`: a SHA-256 hex digest over the sorted task-file
bytes. Two runs of the same task set produce the same revision; an
edit to any task file (even whitespace) changes it. The revision is the
durable evidence of what was pinned (§6.4.3: "Model id, provider,
dataset revision, temperature and all sampling settings pinned and
recorded with each run").

The task list travels in the run record too: `RunRecord.trials[*].task_id`
keys every trial by the task that drove it, so an operator who needs
to know "what was in this run?" reconstructs the set from the trials'
`task_id`s and the loader's sorted-file discovery.

## Cross-references

- `TEST_SUITE.md` §6.4.3 — the spec this protocol implements.
- `TEST_SUITE_IMPLEMENTATION_PLAN.md` §13 — T-K2 row in the plan.
- `tests/harness/eval_tasks/__init__.py` — the loader's source.
- `tests/harness/test_eval_tasks.py` — the framework's L1 contract
  tests, including the import-rule guard.
- `tests/harness/test_eval_tasks_composition.py` — the eval-marked
  end-to-end composition test.
- `tests/harness/eval_harness.py` — T-K1's harness skeleton (the
  runner this protocol's tasks feed into).