"""L3 isolation test for ``_backend_context`` — TEST_SUITE.md §6.3.1.

Plan task **T-I10** (Jira **KBR-102**). Pinned here end-to-end: concurrent
HTTP requests through a real balancing ``BridgeServer`` each carry their
own per-request ``X-Kitty-Backend`` and ``X-Kitty-Model`` response headers,
matching the upstream capture body the request became.

Complements the L1 mechanism tests at
``tests/bridge/test_concurrent_backend_selection.py``: those prove the
ContextVar mechanism in isolation by calling ``_select_backend()`` and
reading the ContextVar directly. That is necessary and not sufficient —
§6.3.1 names the **boundary** (concurrent *requests*), and a regression
that made a request handler read an instance field, or set the ContextVar
outside the per-request task context, would pass every L1 test and corrupt
real traffic. Only this module exercises the bridge handlers.

The bridge boundary claim:

    Concurrent requests never observe each other's backend selection.

proves isolation end-to-end when, for each of N concurrent requests:

  * the upstream capture's ``"model"`` JSON field (set by the profile
    after ``_normalize_model`` ran) names member ``M_i``;
  * the response headers ``X-Kitty-Backend`` (``self._backends[idx][2].name``)
    and ``X-Kitty-Model`` (``ctx["model"]``) name the same member ``M_i``.

If the ContextVar leaked, two requests could share a context copy and one
response would name the peer's member; the per-request join below would
catch it because the header model would differ from that request's capture
body model.

Route sampling rationale (``multi_round_review_sweep_rule`` (a)): this module
samples ``/v1/messages`` (non-streaming and streaming), the same route T-I8
sampled. The ContextVar is a module-level singleton (``server.py:1934``)
shared by all four handlers, so the property is universal by construction;
the sample is for observability, not for coverage.

Header access choice (G1): the module reads response headers via a private
``aiohttp.ClientSession`` in the test, posting to ``fixture.base_url``
directly. ``BridgeFixture``'s context manager still owns the lifecycle
(``async with BridgeFixture(...)``), the recorder hand-off
(``fixture.captures``), and transport start/stop — the only thing bypassed
is the ``BridgeFixture.post`` helper, whose return type ``tuple[int, str]``
discards headers (``tests/harness/bridge.py:859``). This is the surgical
choice: it does **not** extend ``BridgeFixture`` (which would be T-W8-scope,
needing its own step), and it does **not** drop the fixture's lifecycle.
The helper is local to this module.

Process-wide ``random.choices`` note (F3): inside these tests,
``pin_backend_order`` patches ``random.choices`` process-wide for the test's
duration (``tests/harness/bridge.py:686``). Do not call ``random.choices``
for any other purpose here.

Marker convention (sweep rule / §3.3.1): every request body carries a
distinct ``marker()`` string; the upstream capture body carries it verbatim;
the join pairs capture ↔ request by marker rather than by arrival order,
which is non-deterministic under concurrency.

Per-test ContextVar isolation across the suite is enforced by the autouse
``_reset_backend_context`` fixture in ``tests/conftest.py:92-104``; this
module must not defeat it.

Layer note (accepted debt): ``l3`` has no CI job yet — T-K6 owed, see
``tests/layers.py::PENDING_ACTIVATION_LAYERS``. This file runs only via
bare local ``pytest`` until the Subsystem job ships; same as the two
existing L3 modules in ``tests/bridge/``.

Marker: ``pytestmark = pytest.mark.l3``. The ``tests/bridge/`` path default
is ``l1``; this file overrides, like ``test_cross_attempt_content_l3.py``.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any

import aiohttp
import pytest
from harness.bridge import (
    DEFAULT_TIMEOUT,
    BridgeFixture,
    marker,
    pin_backend_order,
    transport,
)
from harness.contract import WireFormat

pytestmark = pytest.mark.l3


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INBOUND_PATH = "/v1/messages"

#: ``backend_models=["m0", "m1"]`` → ``f"harness{i}"`` for backend index
#: ``i`` at ``tests/harness/bridge.py:763``. ``X-Kitty-Backend`` carries the
#: profile name; the capture body carries the profile's model. The two
#: halves of the join live on different sides of this mapping.
_MODEL_TO_PROFILE_NAME: dict[str, str] = {"m0": "harness0", "m1": "harness1"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_pool_profile_names(fixture: BridgeFixture) -> None:
    """Assert the balancing pool's profile names match ``_MODEL_TO_PROFILE_NAME``.

    ``backend_models=["m0", "m1"]`` produces profile names ``harness0`` /
    ``harness1`` via ``tests/harness/bridge.py:763``
    (``backend_for(..., name=f"harness{i}", model=m, ...)``). Asserted here
    rather than trusted, so a future harness rename fails AT THIS LINE with
    a message naming the convention, instead of at every header check below
    whose diagnostics would then name the wrong profile.

    Args:
        fixture: A started bridge fixture whose server carries the pool.
    """
    names = [backend[2].name for backend in fixture.server._backends]
    assert names == ["harness0", "harness1"], (
        f"the harness profile-name convention changed: got {names!r}; "
        f"_MODEL_TO_PROFILE_NAME must be updated with it or every "
        f"X-Kitty-Backend diagnostic below names the wrong profile"
    )


async def _post_returning_headers(
    fixture: BridgeFixture, body: dict[str, Any]
) -> tuple[int, Any, str]:
    """POST a Messages body and return ``(status, headers, text)``.

    Bypasses :meth:`~harness.bridge.BridgeFixture.post` because that helper
    returns ``tuple[int, str]`` and discards headers; the §6.3.1 oracle
    lives in the response headers. Uses ``fixture.base_url`` so the
    fixture still owns the bridge lifecycle.

    Args:
        fixture: A started bridge fixture.
        body: The Messages-API request body.

    Returns:
        The HTTP status, a case-insensitive headers mapping (aiohttp's
        ``CIMultiDict``), and the response text.
    """
    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session, session.post(
        f"{fixture.base_url}{_INBOUND_PATH}", json=body
    ) as response:
        return response.status, response.headers, await response.text()


# ---------------------------------------------------------------------------
# §6.3.1 — _backend_context isolation at the bridge boundary
# ---------------------------------------------------------------------------


class TestBackendContextIsolationBridgeBoundary:
    """§6.3.1 — concurrent requests never observe each other's backend selection.

    Each test fires concurrent HTTP requests through a real balancing
    ``BridgeServer`` whose pool has two members (``m0``/``m1``) round-robined
    by :func:`~harness.bridge.pin_backend_order`. Per request the test joins
    the response headers (read from the ContextVar at middleware / streaming
    time) to the upstream capture's ``"model"`` JSON field (set by the
    profile after ``_normalize_model`` re-ran for that member). If a peer's
    selection leaked into a request's response via a shared ContextVar, the
    per-request join would fail on the request whose header was overwritten.
    """

    @pytest.mark.asyncio
    async def test_concurrent_requests_attribute_to_their_own_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two concurrent requests → each response names its own member.

        Round-robin over a 2-member pool for N=2 selections gives one
        selection per member; the ``X-Kitty-Backend`` multiset is therefore
        ``{harness0: 1, harness1: 1}`` and the ``X-Kitty-Model`` multiset
        is ``{m0: 1, m1: 1}``. A regression that read a shared ContextVar
        (or the instance field) would yield ``{harness0: 2}`` or the
        inverse — and the per-request join catches it either way.

        Args:
            monkeypatch: Reverts :func:`pin_backend_order` on tear-down.
        """
        pin_backend_order(monkeypatch)
        markers = [marker() for _ in range(2)]
        bodies = [
            {
                "model": "harness-model",
                "max_tokens": 16,
                "stream": False,
                "messages": [{"role": "user", "content": m}],
            }
            for m in markers
        ]

        async with BridgeFixture(
            transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES),
            backend_models=["m0", "m1"],
        ) as fixture:
            _assert_pool_profile_names(fixture)
            results = await asyncio.gather(
                *(_post_returning_headers(fixture, b) for b in bodies)
            )
            captures = list(fixture.captures)

        # §3.3.1: an assertion over an empty list passes vacuously. Assert
        # the capture count FIRST so this is the failing line under the R6
        # falsification recipe (patched _select_backend → zero captures),
        # before any per-response pre-flight runs.
        assert len(captures) == len(markers), (
            f"expected one upstream capture per concurrent request; got "
            f"{len(captures)} captures for {len(markers)} requests"
        )

        # Each marker lands in exactly one capture (no retry ⇒ no duplicate).
        marker_to_model: dict[str, str] = {}
        for capture in captures:
            body = json.loads(capture.body.decode("utf-8"))
            for m in markers:
                if m in body["messages"][0]["content"]:
                    assert m not in marker_to_model, (
                        f"marker {m!r} landed in two captures; a retry "
                        f"fired and the isolation claim is no longer tested"
                    )
                    marker_to_model[m] = body["model"]
        assert set(marker_to_model) == set(markers), (
            f"every request must reach the upstream; missing markers: "
            f"{set(markers) - set(marker_to_model)}"
        )

        # Per-request join: each response's headers must name the same
        # member the corresponding capture did. ``asyncio.gather`` preserves
        # task order, so ``results[i]`` corresponds to ``markers[i]``
        # regardless of completion order.
        for (status, headers, _text), m in zip(results, markers, strict=True):
            assert status == 200, (
                f"concurrent request with marker {m!r} failed: status "
                f"{status}; the recorder accepts the default Messages "
                f"reply, so a non-200 indicates the bridge translation broke"
            )
            assert headers.get("X-Kitty-Backend"), (
                f"non-streaming response for marker {m!r} carries no "
                f"X-Kitty-Backend header"
            )
            served_model = marker_to_model[m]
            expected_backend_header = _MODEL_TO_PROFILE_NAME[served_model]
            assert headers.get("X-Kitty-Model") == served_model, (
                f"response for marker {m!r} names model "
                f"{headers.get('X-Kitty-Model')!r} but the upstream capture "
                f"for that request was served by member {served_model!r}. "
                f"The ContextVar leaked: a peer's selection reached this "
                f"response"
            )
            assert headers.get("X-Kitty-Backend") == expected_backend_header, (
                f"response for marker {m!r} names backend "
                f"{headers.get('X-Kitty-Backend')!r} but the upstream "
                f"capture for that request was served by profile "
                f"{expected_backend_header!r}"
            )

        # Multiset shape: round-robin over 2 members × 2 selections ⇒ one
        # of each member. Holds regardless of completion order because the
        # round-robin counter is process-wide and atomic per call.
        served_backends = sorted(
            _MODEL_TO_PROFILE_NAME[m] for m in marker_to_model.values()
        )
        assert served_backends == ["harness0", "harness1"], (
            f"round-robin over a 2-member pool for 2 selections must give "
            f"one of each member; got {Counter(served_backends)}"
        )
        served_models = sorted(marker_to_model.values())
        assert served_models == ["m0", "m1"], (
            f"capture-model multiset must be {{m0:1, m1:1}}; got "
            f"{Counter(served_models)}"
        )

    @pytest.mark.asyncio
    async def test_streaming_concurrent_requests_attribute_to_their_own_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two concurrent ``stream=True`` requests → each response names its own member.

        Pins the second stamping path — ``server.py:5471`` (inside
        ``_stream_messages``, which begins at ``server.py:5394``) — that
        the non-streaming test cannot reach. Headers go out with
        ``prepare()``; ``await response.text()`` drains the SSE body.

        Args:
            monkeypatch: Reverts :func:`pin_backend_order` on tear-down.
        """
        pin_backend_order(monkeypatch)
        markers = [marker() for _ in range(2)]
        bodies = [
            {
                "model": "harness-model",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": m}],
            }
            for m in markers
        ]

        async with BridgeFixture(
            transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES),
            backend_models=["m0", "m1"],
        ) as fixture:
            _assert_pool_profile_names(fixture)
            results = await asyncio.gather(
                *(_post_returning_headers(fixture, b) for b in bodies)
            )
            captures = list(fixture.captures)

        # Non-vacuous guard — same ordering as the non-streaming case: assert
        # the capture count FIRST so this is the failing line under the R6
        # falsification recipe before any per-response pre-flight runs.
        assert len(captures) == len(markers), (
            f"expected one upstream capture per concurrent streaming "
            f"request; got {len(captures)} captures for {len(markers)} "
            f"requests"
        )

        marker_to_model: dict[str, str] = {}
        for capture in captures:
            body = json.loads(capture.body.decode("utf-8"))
            for m in markers:
                if m in body["messages"][0]["content"]:
                    assert m not in marker_to_model, (
                        f"marker {m!r} landed in two captures; a retry "
                        f"fired and the isolation claim is no longer tested"
                    )
                    marker_to_model[m] = body["model"]
        assert set(marker_to_model) == set(markers)

        for (status, headers, _text), m in zip(results, markers, strict=True):
            assert status == 200, (
                f"concurrent streaming request with marker {m!r} failed: "
                f"status {status}; the recorder accepts the default "
                f"Messages SSE reply, so a non-200 indicates the bridge "
                f"translation broke"
            )
            served_model = marker_to_model[m]
            assert headers.get("X-Kitty-Backend"), (
                f"streaming response for marker {m!r} carries no "
                f"X-Kitty-Backend header; the streaming stamp site at "
                f"server.py:5471 must run before prepare()"
            )
            assert (
                headers.get("X-Kitty-Backend") == _MODEL_TO_PROFILE_NAME[served_model]
            ), (
                f"streaming response for marker {m!r} names backend "
                f"{headers.get('X-Kitty-Backend')!r} but the capture shows "
                f"{served_model!r}. The streaming stamp read a peer's "
                f"ContextVar"
            )
            assert headers.get("X-Kitty-Model") == served_model

        served_backends = sorted(
            _MODEL_TO_PROFILE_NAME[m] for m in marker_to_model.values()
        )
        assert served_backends == ["harness0", "harness1"]

    @pytest.mark.asyncio
    async def test_staggered_starts_pin_per_task_selection_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Staggered starts → A selects first (harness0), B second (harness1).

        Pins the **per-task ordering** claim — task[0]'s response names
        ``harness0``, task[1]'s names ``harness1`` — using a condition-wait
        that deterministically puts A's ``_select_backend`` before B's
        task is created. A's upstream capture arriving proves A already
        ran ``_select_backend`` (the ContextVar write happens before the
        upstream POST), so by the time B's task exists A has consumed
        round-robin index 0.

        The isolation claim is owned by the two concurrent tests above,
        which exercise real interleaving. This case adds the orthogonal
        observation that the per-task list is deterministic when started in
        a known order, which is what a future reader of ``/stats`` or the
        session summary is implicitly relying on.

        Args:
            monkeypatch: Reverts :func:`pin_backend_order` on tear-down.
        """
        pin_backend_order(monkeypatch)
        markers = [marker() for _ in range(2)]
        body_a = {
            "model": "harness-model",
            "max_tokens": 16,
            "stream": False,
            "messages": [{"role": "user", "content": markers[0]}],
        }
        body_b = {**body_a, "messages": [{"role": "user", "content": markers[1]}]}

        async with BridgeFixture(
            transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES),
            backend_models=["m0", "m1"],
        ) as fixture:
            _assert_pool_profile_names(fixture)
            # Start A first.
            task_a = asyncio.create_task(_post_returning_headers(fixture, body_a))
            # Wait on A's upstream request to arrive before creating B. A's
            # capture arriving means A already ran ``_select_backend`` (the
            # write happens before the upstream POST).
            deadline = asyncio.get_running_loop().time() + 5.0
            while len(fixture.captures) < 1:
                if asyncio.get_running_loop().time() > deadline:
                    task_a.cancel()
                    raise AssertionError(
                        "task A's upstream capture did not arrive within "
                        "5s; the recorder could be unreachable"
                    )
                await asyncio.sleep(0.01)
            # Now create B — its ``_select_backend`` is guaranteed to run
            # while A's response is still in flight.
            task_b = asyncio.create_task(_post_returning_headers(fixture, body_b))
            results = await asyncio.gather(task_a, task_b)
            captures = list(fixture.captures)

        assert len(captures) == len(markers)

        marker_to_model: dict[str, str] = {}
        for capture in captures:
            body = json.loads(capture.body.decode("utf-8"))
            for m in markers:
                if m in body["messages"][0]["content"]:
                    assert m not in marker_to_model
                    marker_to_model[m] = body["model"]
        assert set(marker_to_model) == set(markers)

        # Per-task list: task[0] (A) must select first → harness0; task[1]
        # (B) selects second → harness1. ``asyncio.gather`` preserves task
        # order.
        for (status, headers, _text), m, expected_backend, expected_model in zip(
            results,
            markers,
            ["harness0", "harness1"],
            ["m0", "m1"],
            strict=True,
        ):
            assert status == 200
            assert headers.get("X-Kitty-Backend") == expected_backend, (
                f"task's response names "
                f"{headers.get('X-Kitty-Backend')!r}; the staggered start "
                f"pins A=harness0, B=harness1. A peer's selection leaked "
                f"into A's response — the instance-field overwrite during "
                f"B's ``_select_backend`` was read by A's "
                f"``_attribution_headers`` instead of A's ContextVar"
            )
            assert headers.get("X-Kitty-Model") == expected_model
            # Sanity: the capture body for this marker names the same model
            # the header did — the join holds for the staggered case too.
            assert marker_to_model[m] == expected_model, (
                f"capture body model for marker {m!r} is "
                f"{marker_to_model[m]!r}; the staggered start pins "
                f"A=m0, B=m1"
            )
