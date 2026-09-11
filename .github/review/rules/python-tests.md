# Rule: tests (`tests/**`, `.github/review/tests/**`)

Two suites live under this rule. `tests/` is the repository's own suite, run by
`tests.yml` on Python 3.10–3.13 with
`pytest -m "l1 or l2" --require-category=l1 --require-category=l2`.
`.github/review/tests/` is the review system's suite, run by `ci.yml` on a
**bare interpreter with no installed dependencies** — see `rules/ci.md` for why,
and never add a third-party import there.

## The one question worth asking about every test

**Would this test fail if the behaviour it names were broken?**

A test that passes either way is not a test — it is a line-coverage contribution
and a maintenance cost. Look for these shapes and raise them:

- **Asserting on a mock.** The test drives a mock and then asserts the mock was
  called. Nothing about the real code was proved.
- **Asserting a substring where the point is the whole value.** A spawn-config
  test that checks `ANTHROPIC_BASE_URL` is present will not notice that
  `ANTHROPIC_AUTH_TOKEN` stopped being cleared. Assert the exact map.
- **Asserting "no exception".** Unless the behaviour under test *is* "does not
  raise", this passes on almost any implementation.
- **A test whose assertion was written from the implementation** rather than from
  the requirement — it will keep passing through the bug it was meant to catch,
  because it encodes the bug.
- **A negative that never had a positive.** A test that asserts a key is absent
  from a log needs a sibling proving the log is produced at all, or it passes
  when logging silently stops.

## Determinism

A flaky test is worse than no test: it trains people to ignore red. Anything
ambient must come through a seam the test controls — the clock, randomness, the
filesystem, the network, the environment, and the user's real config directory.

- **`sleep()` as a synchronisation primitive is the leading cause of flakiness.**
  Wait on a condition with a timeout instead. Raise every `sleep` in a new test.
- **Real network access in `tests/` is a defect.** `aioresponses` is the declared
  dev dependency for this; a new test that reaches an upstream provider is a
  finding whatever it proves.
- **Tests must not depend on each other or on execution order**, and must not
  write to the developer's real `~/.config/kitty`. A test that touches the real
  config directory is critical: it destroys a user's profiles when run locally.

## The layer the test belongs at

Prove each behaviour at the lowest layer that can prove it.

| The claim | Where it belongs | Marker |
|---|---|---|
| A translator maps this field to that one | unit, in `tests/bridge/` | `l1` |
| A launcher builds this exact env map | unit, in `tests/` | `l1` |
| A profile written by an old version still loads | unit, over a committed fixture | `l1` |
| Agent and bridge agree on the wire shape | protocol/contract test | `l2` |
| The bridge survives an upstream disconnect mid-stream | subsystem, against a real local server | `l3` |
| `kitty claude` end to end | `tests/integration/` | `agent_live` |

**Every test carries exactly one layer marker, and a CI job is a marker
expression over them.** The marker is assigned from the file's path by
`tests/conftest.py`, so a new test in an existing directory needs nothing; a
file declares its own layer only where that default is wrong, and
`tests/test_layer_markers.py` fails the suite if any test carries none or two.

Two things to raise when reviewing a change here:

- **A marker added by hand needs a reason in the diff.** The default is right
  for almost everything; an explicit one says "this file is not what its
  directory suggests", which is a claim worth reading.
- 🔴 **Moving a test to a layer no job runs removes it from CI, silently.**
  Only `l1` and `l2` are gated today. `PENDING_ACTIVATION_LAYERS` in
  `tests/layers.py` lists the rest with the task that activates each, and a
  test fails if a populated layer is neither run nor listed — but a *file*
  moved into a listed layer is not caught by that. Treat such a move as a
  coverage deletion unless the job that runs it lands in the same change.

Two findings follow from this table:

- **A behaviour proved only at the top.** Slow, flaky, and uninformative when it
  fails. Say which lower layer would have proved it.
- **A behaviour proved only at the bottom when the wiring is the risk.** A
  translator unit-tested in both directions still tells you nothing about whether
  the server routes to it.

## Invariant guards

Several tests exist specifically to hold something nothing else enforces:

`test_egress_fail_closed.py`, `test_egress_coverage.py`,
`test_egress_https_proxy.py`, `test_exit_code_mapping.py`,
`test_provider_list_sync.py`, `test_model_context_packaged_catalog.py`,
`test_pypi_packaging.py`, `test_github_actions.py`, `test_entry_point_refresh.py`.

**A change that weakens one of these to make a diff pass is a critical finding**,
and it is worth stating explicitly in the finding what invariant was being held.
Deleting such a test, loosening its assertion, or adding a skip to it all count.
An `xfail` or `skip` added anywhere without a stated reason and a linked issue
is a warning at minimum. A `ratchet` carries both by construction, so its
trigger is different: a `ratchet` added to one of the invariant guards named
above, or one whose registry row does not land in the same diff.

**`ratchet` is the third amnesty form and has its own rule.** It exempts one
named assertion, from a row in `tests/exemptions.py` carrying that assertion,
its expected failure condition and its Jira key
(`.system_design/TEST_SUITE.md` §8.3). Two things to raise:

- 🔴 **More than one assertion inside a `ratchet` block.** The block must hold
  one assertion and the statements that build its subject. A second assertion is
  never evaluated once the first fails, so a failure it would have reported is
  invisible — the blanket amnesty §8 rejects, in miniature. Nothing detects
  this; review is the enforcement.
- **A `ratchet` call passing `_registry=`.** That is the mechanism's private
  test seam. From a guard it is a local amnesty that no one can count by reading
  the registry, which defeats the point of having one.

## Coverage of the change itself

Production code with no corresponding test is a finding. Ask, for each behaviour
the diff adds:

- is there a test that names it, and would it fail without the change;
- does it cover the empty, `None`, malformed and very-large input;
- does it cover the **failure** path, which in this repository is usually the
  interesting one — a refused launch, a fail-closed egress, a dangling
  credential reference, a stream that ends early.

## Style

Test code is code. Same Google-style docstrings, same surgical diffs. A test
module's name should say what unit it covers, and a test's name should read as
the specification sentence it proves.

## Severity

- **Critical** — an invariant guard weakened or deleted; a test that writes to
  the real user config directory.
- **Warning** — new behaviour with no test; a test that would pass either way; a
  `sleep`-based wait; real network access; an unexplained skip; a behaviour
  proved at the wrong layer.
- **Suggestion** — naming, duplication, a fixture that would remove setup noise.
