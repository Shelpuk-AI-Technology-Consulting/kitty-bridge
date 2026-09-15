"""End-to-end vertical slice for the eval harness (KBR-110, T-K1).

The plan's §1.2 pattern: a single driven request proving every
contract composes, not just exists. This slice exercises the eval
harness against a real :class:`~harness.bridge.BridgeFixture` and a
scripted :class:`~harness.recorder.RecordingUpstream`, with one task
driven through both arms. Both arms reach the **same** scripted
upstream behaviour (``harness.failures.success``), so the run record
proves the harness's two-arm shape, the runner's grid, the
:class:`~harness.eval_harness.classify` taxonomy, the
:class:`~harness.eval_harness.RunRecord` JSON, and the
:class:`~harness.eval_harness.RunConfig` digest compose rather than
merely coexist.

**Layer.** Module-level ``pytest.mark.eval``. The ``eval`` layer is
in :data:`tests.layers.PENDING_ACTIVATION_LAYERS` against plan task
**T-K12**; the default ``pytest`` run excludes ``eval``-marked tests
via :attr:`pyproject.toml [tool.pytest.ini_options] addopts`, and
T-K12's nightly is what will gate this. The slice runs locally with
``pytest -m eval tests/harness/test_eval_harness_vertical_slice.py``.

**New file, LF.** Per ``kitty-bridge-tests-conftest-crlf``.
"""

from __future__ import annotations

import aiohttp
import pytest

from harness.bridge import (
    BridgeFixture,
    InboundProtocol,
    MODEL,
    AiohttpTransport,
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
    Verdict,
    run_eval,
)
from harness.failures import success

pytestmark = pytest.mark.eval


async def test_eval_harness_drives_one_task_through_both_arms_end_to_end() -> None:
    """The two-arm eval harness drives one task through the bridge and direct.

    Both arms reach the same scripted upstream (``success`` for
    Anthropic Messages), so the run record's per-arm tallies agree —
    the harness's headline claim is that *paired* runs can compare
    arms, and the slice proves the pairing end-to-end.

    The acceptance check is deliberately trivial (``non-empty reply
    is PASS``) — T-K2 owns per-task acceptance; the slice's job is to
    prove the contracts compose, not to exercise the task set.
    """
    transport = AiohttpTransport(format=WireFormat.ANTHROPIC_MESSAGES, responder=success(WireFormat.ANTHROPIC_MESSAGES))
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
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    upstream_url,
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": task.prompt}],
                        "max_tokens": 16,
                    },
                ) as response:
                    text = await response.text()
            assert response.status == 200, f"direct arm: upstream returned {response.status}: {text[:200]}"
            return ModelReply(reply=text)

        def accept(reply: object) -> Verdict:
            return Verdict.PASS if reply else Verdict.FAIL

        task = TaskSpec(id="slice-task", prompt="hello", acceptance_check=accept)
        config = RunConfig(
            model_id=MODEL,
            provider="anthropic",
            dataset_revision="2026-09-15",
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
            [task],
            [ArmSpec(name="kitty", executor=kitty_arm), ArmSpec(name="direct", executor=direct_arm)],
        )

        # Scheduled: 1 sample × 1 task × 2 arms = 2.
        assert record.scheduled == 2
        assert len(record.trials) == 2
        # Both arms reached the same scripted upstream; both classify SUCCESS.
        assert all(trial.category is TrialCategory.SUCCESS for trial in record.trials)
        assert record.pass_rate_per_arm == {"kitty": 1.0, "direct": 1.0}

        # The config digest carries every pinned setting.
        assert record.config["model_id"] == MODEL
        assert record.config["provider"] == "anthropic"
        assert record.config["n_samples"] == 1

        # And the run record round-trips through JSON — the durability
        # the on-disk artifact claims.
        rebuilt = record.__class__.from_json(record.to_json())
        assert rebuilt == record
