"""Schemathesis conformance run against a bridge started in bridge mode.

``.system_design/TEST_SUITE.md`` §6.2.1 (L2) · Jira **KBR-82** (T-G6).

§6.2.1 requires the bridge's published OpenAPI document be driven against a
running bridge with ``schemathesis``, and the three named checks
(``not_a_server_error``, ``response_schema_conformance``,
``status_code_conformance``) must all be green. The conformance run is the
artifact that catches the kind of malformed-body 500 KBR-159 measured
manually — without it, an unguarded ``.get`` somewhere in the translator
would only surface when an agent happened to send that shape.

The test is **deterministic under CI** (the existing
``tests/conftest.py`` profile derandomises hypothesis) and **bounded to a
small example count** so the Fast gate stays under budget; the deeper
nightly fuzz is T-K7's job. Each generated case is driven through
``case.call_and_validate`` inside ``asyncio.to_thread`` because the call is
sync (uses ``requests``) and the bridge's aiohttp loop must stay free to
process the request — without ``to_thread`` the test deadlocks against the
bridge. ``base_url=`` is passed per call (verified: 4.x's
``SchemathesisConfig`` has no ``base_url`` field; the per-case kwarg is the
supported wiring).

The bridge is started in bridge mode (``adapter=None``) so all five POST
routes plus three GETs are registered — the "must target bridge mode"
warning §6.2.1 names is the failure mode a single-protocol bridge would
produce otherwise.

Each POST route's ingress normaliser rejects ``messages``/``contents`` as a
non-list with a 400 in the dialect's envelope (this task). The
conformance run therefore sends malformed bodies *and* well-formed ones,
and verifies both paths return statuses the schema documents — no 500.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import schemathesis
import yaml
from harness.bridge import BridgeFixture, transport
from harness.contract import WireFormat
from hypothesis import HealthCheck, Phase, settings
from schemathesis.checks import not_a_server_error
from schemathesis.config import GenerationConfig, ProjectConfig, ProjectsConfig, SchemathesisConfig
from schemathesis.specs.openapi.checks import (
    response_schema_conformance,
    status_code_conformance,
)

pytestmark = pytest.mark.l2

#: The schema file the conformance run reads. Kept in the repo at a stable
#: location so the test is deterministic across CI runs.
SCHEMA_PATH = Path(__file__).parent.parent / "openapi" / "kitty-bridge.yaml"

#: The three checks §6.2.1 names. Import split is real on 4.x: the no-5xx
#: check lives in ``schemathesis.checks``; the OpenAPI-specific conformance
#: checks live in ``schemathesis.specs.openapi.checks``.
THE_THREE_CHECKS: list = [not_a_server_error, status_code_conformance, response_schema_conformance]

#: Hypothesis example budget per generated case. The L2 gate is a regression
#: check, not a fuzzer; the deeper nightly fuzz is T-K7 (not in scope).
MAX_EXAMPLES: int = 5

#: Per-case request timeout in seconds. The bridge's normalisers refuse the
#: malformed shapes the conformance run fuzzer generates, but a few
#: near-valid shapes can still trigger the L3 retry ladder; ten seconds
#: is well below the Fast-gate budget while still catching genuine hangs.
CASE_TIMEOUT_SECONDS: float = 10.0


def _load_schema() -> schemathesis.schemas.BaseSchema:
    """Load the published OpenAPI document as a schemathesis schema.

    Returns:
        A schemathesis OpenAPI schema ready to drive conformance against.
        The schema is loaded with ``generation.max_examples = 5`` so the
        Fast gate stays under budget; T-K7's nightly job is where deeper
        fuzzing lives. ``no_shrink=True`` keeps a single failing case from
        generating dozens of shrinking attempts — the L2 gate is a
        regression check, not a fuzzer.
    """
    with open(SCHEMA_PATH) as f:
        schema_dict = yaml.safe_load(f)
    config = SchemathesisConfig(
        projects=ProjectsConfig(
            default=ProjectConfig(
                generation=GenerationConfig(max_examples=MAX_EXAMPLES, no_shrink=True)
            ),
        )
    )
    return schemathesis.openapi.from_dict(schema_dict, config=config)


@pytest.fixture()
async def cc_bridge() -> BridgeFixture:
    """A real bridge in bridge mode, pointed at a recording Chat-Completions upstream.

    Bridge mode is what registers all five POST routes plus three GETs;
    the conformance run needs every route registered to actually exercise
    it.
    """
    async with BridgeFixture(transport("aiohttp", WireFormat.CHAT_COMPLETIONS)) as fixture:
        yield fixture


@_load_schema().parametrize()
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[Phase.generate, Phase.target],
)
async def test_api_conformance(case: schemathesis.Case, cc_bridge: BridgeFixture) -> None:
    """Every generated case must satisfy the three named checks.

    Args:
        case: A schemathesis-generated request case.
        cc_bridge: The started bridge fixture. The bridge's ``base_url`` is
            the target the case is run against. One bridge serves all the
            generated inputs — that is the point of the run.
    """
    await asyncio.to_thread(
        case.call_and_validate,
        base_url=cc_bridge.base_url,
        checks=THE_THREE_CHECKS,
        timeout=CASE_TIMEOUT_SECONDS,
    )
