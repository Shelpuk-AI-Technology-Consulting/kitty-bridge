"""End-to-end composition proof: ``load_task_set`` → ``run_eval`` (KBR-111, T-K2).

The plan's §1.2 pattern — a single driven request proving every
contract composes, not just exists. This slice exercises the whole
T-K2 framework against a *loaded* task, not an in-process one: a
person-authored task module is written to a temp registry,
``load_task_set`` discovers it, and the resulting :class:`TaskSet`
feeds :func:`harness.eval_harness.run_eval` through the
:class:`~harness.bridge.BridgeFixture` + scripted recorder pair T-K1's
vertical slice uses. File discovery, the stem check, the authorship
gate, and the runner — one driven path.

**Why the acceptance check is illustrative.** The task the slice
writes is a person-authored demonstration of the framework, not a
real eval task: the check accepts any reply, because the slice's job
is to prove composition (a loaded task reaches the runner), not to
exercise task semantics. The authoring protocol at
``tests/harness/README_eval_tasks.md`` carries the real guidance on
what an acceptance check should judge.

**Layer.** Module-level ``pytest.mark.eval``, mirroring T-K1's
vertical-slice marker. The ``eval`` layer is in
:data:`tests.layers.PENDING_ACTIVATION_LAYERS` against plan task
**T-K12**; the default ``pytest`` run excludes ``eval``-marked tests
via ``pyproject.toml`` ``addopts``. This slice runs locally with
``pytest -m eval tests/harness/test_eval_tasks_composition.py``.

**New file, LF.** Per ``kitty-bridge-tests-conftest-crlf``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import aiohttp
import pytest

from harness.bridge import (
    MODEL,
    AiohttpTransport,
    BridgeFixture,
    InboundProtocol,
    inbound_path,
    minimal_inbound_body,
)
from harness.contract import WireFormat
from harness.eval_harness import (
    ArmSpec,
    ModelReply,
    RunConfig,
    TaskSpec,
    TrialCategory,
    run_eval,
)
from harness.eval_tasks import TaskSet, load_task_set
from harness.failures import success

pytestmark = pytest.mark.eval

# The demo task module the slice writes. The authorship is real
# person-shaped authorship ("ada") because the slice also proves the
# gate lets person-authored tasks through — a MODEL-authored module
# would be refused before run_eval ran a single trial.
_DEMO_TASK_MODULE = '''
    from harness.eval_harness import Verdict
    from harness.eval_tasks import EvalTask, EvalTaskAuthor


    def _accept(reply: object) -> Verdict:
        """Illustrative check — the README carries the real guidance."""
        return Verdict.PASS if reply else Verdict.FAIL


    TASK = EvalTask(
        id="composition_demo",
        prompt="hello",
        acceptance_check=_accept,
        authored_by=EvalTaskAuthor.person(name="ada"),
    )
    '''


async def test_loaded_task_set_drives_run_eval_through_both_arms(
    tmp_path: Path,
) -> None:
    """A loaded, person-authored task reaches ``run_eval`` and classifies SUCCESS.

    Both arms reach the same scripted upstream (``success`` for
    Anthropic Messages), so the run record proves the pairing
    end-to-end. The revision the loader computes is pinned into the
    run config — AC-2.3's atomicity proof — and the record
    round-trips through JSON.
    """
    (tmp_path / "composition_demo.py").write_text(
        textwrap.dedent(_DEMO_TASK_MODULE),
        encoding="utf-8",
    )

    # The load: one person-authored task, one revision.
    taskset: TaskSet = load_task_set(root=tmp_path)
    assert len(taskset.tasks) == 1
    assert taskset.tasks[0].id == "composition_demo"

    # The runner: the loaded tasks feed the two-arm eval.
    transport = AiohttpTransport(
        format=WireFormat.ANTHROPIC_MESSAGES,
        responder=success(WireFormat.ANTHROPIC_MESSAGES),
    )
    async with BridgeFixture(transport) as bridge:
        upstream_url = f"{transport.recorder.base_url}/v1/messages"

        async def kitty_arm(task: TaskSpec, sample_index: int) -> ModelReply:
            status, text = await bridge.post(
                inbound_path(InboundProtocol.MESSAGES),
                minimal_inbound_body(InboundProtocol.MESSAGES, task.prompt),
            )
            assert status == 200, f"kitty arm: bridge returned {status}: {text[:200]}"
            return ModelReply(reply=text)

        async def direct_arm(task: TaskSpec, sample_index: int) -> ModelReply:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    upstream_url,
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": task.prompt}],
                        "max_tokens": 16,
                    },
                ) as response,
            ):
                text = await response.text()
            assert response.status == 200, f"direct arm: upstream returned {response.status}: {text[:200]}"
            return ModelReply(reply=text)

        config = RunConfig(
            model_id=MODEL,
            provider="anthropic",
            dataset_revision=taskset.revision,
            temperature=0.0,
            top_p=1.0,
            max_tokens=16,
            seed=0,
            n_samples=1,
            deadline_seconds=30.0,
            sampling_overrides={},
        )

        record = await run_eval(
            config,
            taskset.tasks,
            [ArmSpec(name="kitty", executor=kitty_arm), ArmSpec(name="direct", executor=direct_arm)],
        )

        # Scheduled: 1 sample × 1 task × 2 arms = 2; both classify SUCCESS.
        assert record.scheduled == 2
        assert len(record.trials) == 2
        assert all(trial.category is TrialCategory.SUCCESS for trial in record.trials)
        assert record.pass_rate_per_arm == {"kitty": 1.0, "direct": 1.0}

        # AC-2.3 — the revision the loader computed is what the run pinned.
        assert record.config["dataset_revision"] == taskset.revision

        # And the run record round-trips through JSON — the durability
        # the on-disk artifact claims.
        rebuilt = record.__class__.from_json(record.to_json())
        assert rebuilt == record


async def test_loaded_model_authored_task_set_never_reaches_run_eval(
    tmp_path: Path,
) -> None:
    """The gate fires before ``run_eval`` runs a single trial.

    A MODEL-authored task module is refused by ``load_task_set``, so
    the composition cannot even be built. The gate ordering is the
    AC-1.2 claim at composition scope: the refusal happens at the
    loader boundary, not inside the runner.
    """
    (tmp_path / "model_authored.py").write_text(
        textwrap.dedent(
            """
            from harness.eval_harness import Verdict
            from harness.eval_tasks import EvalTask, EvalTaskAuthor

            TASK = EvalTask(
                id="model_authored",
                prompt="hello",
                acceptance_check=lambda reply: Verdict.PASS if reply else Verdict.FAIL,
                authored_by=EvalTaskAuthor.MODEL,
            )
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_task_set(root=tmp_path)
    assert "model_authored" in str(exc_info.value)
    assert "§6.4.3" in str(exc_info.value)
