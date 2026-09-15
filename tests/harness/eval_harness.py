"""The two-arm eval harness skeleton (KBR-110, plan task **T-K1**).

`TEST_SUITE.md` §6.4.3 specifies the paired-delta eval methodology and
`TEST_SUITE_IMPLEMENTATION_PLAN.md` §13 scopes what the skeleton owns:
pinning, two arms, the failure taxonomy, and the denominator. The
statistics and the decision rule are plan task **T-K3** and are
deliberately out of scope here.

The harness is **pure except for the injected ``clock`` keyword
argument** — it does no IO, launches no subprocess, touches no real
provider. The arm executor is the seam that does; whoever wires the
nightly (plan task T-K12) supplies it. Keeping the skeleton pure is
what lets its load-bearing rules — pinning, taxonomy, the denominator —
be judged at L1 and handed a deliberate defect (§1.4) without starting
a server.

**Import rule.** This module imports nothing from ``src/kitty``. The
independence precedent and its rationale are in
``tests/harness/failures.py``; the structural guard is
``test_the_eval_harness_imports_nothing_from_kitty`` in the file
beside this one. ``bridge.py`` is the one module in this package
permitted to import the product; this is not it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TypeAlias

__all__ = [
    "RunConfig",
    "JSONValue",
]

#: The shape ``sampling_overrides`` accepts: any JSON-serialisable scalar
#: or container. Named here so the field annotation and the constructor's
#: serialisability check describe one type, not two.
JSONValue: TypeAlias = "str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]"


@dataclass(frozen=True)
class RunConfig:
    """The settings that define one eval run, pinned at construction.

    §6.4.3: "Model id, provider, dataset revision, temperature and all
    sampling settings pinned and recorded with each run. An unpinned
    model makes the series meaningless." The dataclass makes that
    refusal mechanical: there is no way to construct a ``RunConfig``
    with a required field left as ``None``, and the error names the
    offending field so a maintainer reading a CI log knows which one
    the operator missed.

    ``sampling_overrides`` is the open extension surface for
    provider-specific knobs the fixed fields do not enumerate. It is
    pinned with everything else: its keys are recorded in sorted order
    so the digest is stable across dict construction orders, and its
    values must survive a JSON round-trip because the recorded digest
    is the only durable evidence of what was pinned.
    """

    #: The model identifier exactly as the provider publishes it.
    model_id: str
    #: The provider serving ``model_id``; a model id alone is ambiguous.
    provider: str
    #: Revision stamp of the task set the run used.
    dataset_revision: str
    #: Sampling temperature (§6.4.3 names it explicitly).
    temperature: float
    #: Nucleus-sampling ceiling, the standard companion to temperature.
    top_p: float
    #: Upper bound on reply length; matters to refusal/drop diagnoses.
    max_tokens: int
    #: Determinism hook for the model's own sampling.
    seed: int
    #: Samples per task per arm; §6.4.3 fixes N in advance. Must be >= 1.
    n_samples: int
    #: Wall-clock bound on a single trial, in seconds.
    deadline_seconds: float
    #: Provider-specific sampling knobs, pinned like the fixed fields.
    sampling_overrides: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        """Refuse any unset required field and validate the boundaries.

        Raises:
            ValueError: When any required field is ``None`` (the message
                names the field), when ``n_samples < 1``, or when
                ``sampling_overrides`` carries a value that cannot be
                serialised to JSON.
            TypeError: When ``sampling_overrides`` is not a ``Mapping``.
        """
        # Every required field unset is refused, naming the field. The
        # message is the only diagnostic a CI log gets, so the field name
        # is load-bearing and asserted by the falsification case (F1).
        for field in fields(self):
            if getattr(self, field.name) is None:
                raise ValueError(
                    f"RunConfig.{field.name} is unset; every field that defines "
                    "a run must be pinned (TEST_SUITE.md §6.4.3: an unpinned "
                    "model makes the series meaningless)"
                )

        # A zero-sample run has nothing to measure and no denominator to
        # hold; refusing it here keeps the boundary error at the site the
        # operator controls rather than deep in the runner.
        if self.n_samples < 1:
            raise ValueError(
                f"RunConfig.n_samples must be >= 1; got {self.n_samples}"
            )

        # The overrides map is pinned with the rest, which means its
        # contents have to reach the record intact — so a value that
        # cannot survive a JSON round-trip is refused here, at
        # construction, rather than silently dropped at write time.
        # Snapshot to a plain dict so any Mapping whose contents are
        # serialisable is accepted (a MappingProxyType or UserDict carries
        # JSON-clean bytes but the stdlib's JSON encoder refuses the
        # container itself, which would silently lose a perfectly valid
        # pin set).
        if not isinstance(self.sampling_overrides, Mapping):
            raise TypeError(
                f"RunConfig.sampling_overrides must be a Mapping; got "
                f"{type(self.sampling_overrides).__name__}"
            )
        try:
            json.dumps(dict(self.sampling_overrides))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"RunConfig.sampling_overrides must be JSON-serialisable; "
                f"its recorded digest is the durable evidence of what was "
                f"pinned. {exc}"
            ) from exc
