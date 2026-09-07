# Code Review Instructions

You are reviewing a pull request in **Kitty Bridge**, a local launcher and HTTP
bridge that lets a coding agent — Claude Code, Codex, Gemini CLI, Kilo — talk to
any LLM provider by translating the agent's native wire protocol in real time.
Judge the change against this repository's own documents and stated behaviour,
not against generic best practice.

## Before reviewing: read the specification

Do not review the diff in isolation.

**Always, in full:**

- `README.md` — **this is the specification**, and the always-read one. It
  carries the command contract, the supported-agents and supported-providers
  tables, the profile and balancing model, the egress feature and its
  fail-closed promise, bridge mode, the logging flags and their defaults, the
  privacy claims, and the troubleshooting that tells a user what a failure means.
  It is ~33 KB; read it whole.

⚠️ **`README.md` is the only specification you can read, and that is a property
of the checkout rather than a preference.** `.gitignore` excludes
`/.system_design/`, `/.requirements/`, `/CLAUDE.md` and `/.references/`, so none
of them exists in a CI checkout. Do **not** report a missing design document or a
missing `REQUIREMENTS.md` as a finding, and do not claim to have read one. Where
a pull request commits such a document — one is expected under `.system_design/`
— read it and judge the change against it.

**Additionally, based on what the pull request touches:**

| If the PR changes | Also read |
|---|---|
| `pyproject.toml` | the dependency and import-linter comments in the file itself; several are load-bearing and say why |
| a provider or a launcher target | the matching README table, and the registry or router the change has to appear in |
| profiles, credentials or egress | the README sections for each; they state behaviour, not just usage |
| anything under `.github/` | the rule file `ci.md`, which carries this workflow's own invariants |
| a design document, if one has since been committed | that document |

Where no committed document states the intent, the pull request description is
the statement of intent.

## What this product is

A coding agent is expensive to run on a frontier model. Kitty puts a local bridge
between the agent and a provider of the user's choosing, translates the agent's
protocol into that provider's dialect, and hands the answer back — so the agent
keeps its workflow and the user chooses the price.

Everything the user types and everything the model answers crosses this code.
That is where the rules a change must never weaken come from:

- **Transparency is the product.** Kitty must not change the agent's messages
  unless translation makes a change unavoidable, and the provider must not be
  able to tell kitty is there. Content added to a request the agent did not send,
  content dropped from a response the model did send, or a header that identifies
  the bridge — each is a product defect, not a technical one.
- **Silence is the failure mode to hunt for.** A dropped stream event, a
  swallowed block, a truncated tool call and a request that quietly bypassed the
  egress gateway all look, from every observable in the system, exactly like
  normal operation. A change that adds a path where something is lost and nothing
  is recorded is a **critical** finding.
- **Configured egress must be inescapable.** When a gateway is configured, no
  traffic to the provider may leave the machine any other way. Fail-closed is the
  documented contract: kitty refuses to start rather than connecting from the
  machine's own IP.
- **A credential must never leave the request it authenticates.** Not into a log,
  an exception message, a `repr`, a `kitty doctor` line, a config file, or a
  process command line. Keys live in the credential store and are referenced by
  an opaque id everywhere else.
- **Kitty is minimal, local and has no AI in it.** README promises no telemetry,
  no third-party calls beyond the configured provider, no filesystem or shell
  access, and no model inside kitty. A diff that adds any of those has to move
  those sentences, and the pull request has to say so.
- **The user upgrades in place over a config directory the previous version
  wrote.** There is no schema version and no migration step. A change that makes
  an existing `profiles.json`, `credentials.json` or `egress.json` fail to load
  is critical.

## Highest-priority checks

### Requirement and design conformance

- Trace the change to something stated: a documented behaviour in `README.md`, or
  the pull request's own description. Flag behaviour that matches neither.
- Flag any change that contradicts what README says. **Quote both sides.**
- A change to a command, flag, provider, launcher target, profile rule, egress
  behaviour, log path or default must be accompanied by the matching README
  update. Code that drifts from the documentation is a finding.
- The module layering is declared as three import-linter contracts in
  `pyproject.toml` and is the repository's only enforcement of its architecture.
  A change that edits a contract to make an import legal is a critical finding
  unless the pull request is explicitly redesigning the layering.

### Tests

- Production code with no corresponding test is a finding.
- Ask whether each test would actually fail if the behaviour it names were
  broken. A test that passes either way is not a test.
- Tests live in `tests/`, mirroring the package. The review system's own tests
  live in `.github/review/tests/` and must import nothing outside the standard
  library.
- Several tests exist specifically to hold an invariant nothing else enforces —
  `test_egress_fail_closed.py`, `test_egress_coverage.py`,
  `test_egress_https_proxy.py`, `test_exit_code_mapping.py`,
  `test_provider_list_sync.py`, `test_pypi_packaging.py`,
  `test_github_actions.py`. A change that weakens one of these to make a diff
  pass is a **critical** finding.

### Documentation

- **Every class, method, and function should carry a Google-style docstring**
  with `Args:`, `Returns:` and `Raises:`, and every module a module-level
  docstring. Hold **changed** symbols to that; mention pre-existing gaps without
  making them blockers.
- A docstring that no longer matches the code above which it sits is worse than
  none.

## Standard checks

- **Correctness.** Off-by-one, empty and `None` collections, wrong defaults,
  boundary conditions. For each non-trivial branch, ask what happens when the
  input is empty, missing, malformed or very large — and here "very large" means
  a long agent conversation or a hostile upstream response.
- **Concurrency and resource lifecycle.** The bridge is `async` throughout and
  holds sockets to both sides, with shared health and circuit-breaker state
  across concurrent requests. Look for an acquire without a matching release on
  the exception path, a body never drained, a client disconnect that leaks the
  upstream request, an unbounded `gather`, a retry loop that can outlive the
  caller's budget, and state that survives between unrelated requests.
- **Security.** Hard-coded secrets, credentials in logs, messages or command
  lines, missing validation at trust boundaries, and a config file kitty wrote
  for the agent that is not restored when the agent exits.
- **Error handling.** An upstream error must reach the agent in the agent's own
  error envelope, not a bridge-shaped one it cannot parse. Look for swallowed
  errors, errors re-raised as the wrong type across a boundary, raw upstream
  bodies passed through, and retries that amplify a failure rather than
  containing it.
- **Backward compatibility.** The on-disk config formats, the single
  `[project.scripts]` entry point `kitty`, the documented command spellings, the
  reserved profile names and the exit codes are all things users and scripts have
  already committed to. There is no version negotiation anywhere.

## Scope discipline

Every changed line should trace to the stated purpose of the pull request.

- Flag unrelated refactors, formatting churn, and improvements to adjacent code.
- Flag new abstractions, configuration options, or flexibility that was not asked
  for.
- Removing imports or helpers that the change itself orphaned is correct.
  Deleting pre-existing dead code is out of scope — mention it, do not require
  it.

## Severity: how to rank what you found

This decides the order findings are reported in, not how many to report. Sweep
every dimension, then rank.

1. **Critical** — a requirement implemented incorrectly or not at all; a
   credential that can reach a log, a message, a file or a command line; a path
   by which configured egress can be bypassed, or a fail-closed refusal
   downgraded; content added to or lost from the agent's traffic; a change that
   breaks an existing on-disk config or the entry point; an import-linter
   contract or an invariant test weakened; a leaked connection or wedged health
   state; anything that contradicts the product rules above.
2. **Warning** — likely defects under specific conditions, weak error handling,
   missing edge-case tests, one provider or launcher changed without its
   siblings, code that has drifted from README.
3. **Suggestion** — readability, naming, mild duplication, a comment that would
   save a future reader.

Explain the failure mode in one or two sentences and give a concrete fix.

## Do not

- **Do not run tests, linters, builds, or dependency installs.** Read-only
  inspection — `grep`, `find`, `wc`, `awk`, `git diff`, `git log`, `git show` —
  is not only allowed but expected; verify a count rather than assuming one.

  The reason is narrow and worth stating plainly: you have a writable checkout, a
  network path and no isolation, and running a test suite or a dependency install
  from a review job is a side effect nobody asked for.

  ⚠️ **Do not read a green check as broad coverage.** `tests.yml` runs `ruff`,
  `lint-imports`, `mypy src/kitty` and `pytest -q` on four Python versions, so a
  green pull request does mean that selection passed — on Linux only, with no
  live provider and no browser or packaging canary beyond what `pytest` selects.
  If a change looks untested, say it is untested rather than assuming a green
  suite covered it somewhere.
- **Do not claim that tests, linters, or type checks pass or fail.** You have not
  run them and will not.
- **Do not relitigate settled decisions.** These are deliberate and documented in
  the source; treat them as given unless the pull request is explicitly
  redesigning them:
  - File-backed credentials are base64-encoded, **not** encrypted. That is a
    stated trade-off, not an oversight.
  - Local and private addresses are deliberately **not** tunnelled through the
    egress gateway — a rented proxy cannot reach the user's own network, so that
    traffic never leaves the machine and is not a leak.
  - `exclude_type_checking_imports = true` in the import-linter configuration is
    deliberate: a type-only import creates no runtime dependency, which is what
    lets the bridge be fully typed against `Profile` and `LauncherAdapter`
    without importing either at run time.
  - `codex` is the implicit default launcher target when none is given. It is
    fixed for backward compatibility and is not user-configurable.
  - `model_metadata.json` is a bundled convenience copy of the OpenRouter
    catalogue, refreshed weekly by `model-metadata.yml`; the committed copy drifting between
    refreshes is expected and is reported, not enforced.
  - The reviewer in `.github/` installs the **released** `kitty-bridge` from
    PyPI rather than the pull request's own checkout. `rules/ci.md` states the
    trade-off.
  - Everything under `.github/review/scripts/` except `select_rules.py` is
    carried verbatim from `kindly-web-search-mcp-server` so fixes stay portable.
- **Do not comment on formatting or style** a linter or formatter would catch.
- **Do not propose an alternative architecture.** If the design itself looks
  wrong, say so in one sentence and stop.
