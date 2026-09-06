# Rule: launcher adapters (`launchers/**`, `cli/launcher.py`)

A launcher adapter answers one question per agent: given a profile, a bridge port
and a resolved credential, what environment variables, what cleared variables and
what CLI flags does the child process start with? `cli/launcher.py` is the
orchestrator that consumes that answer and spawns the child.

This is the contract with the agent, and it is the second most
security-relevant surface in the repository after the credential store. Get it
wrong and the agent talks to the provider directly, or with the wrong credential,
or with the user's real key visible in a process listing.

## Invariants

- **Kitty must be the only writer of the endpoint and credential the child
  reads.** For each agent, the adapter both *sets* the variables pointing at the
  local bridge and *clears* the ones that would override them
  (`ANTHROPIC_BEDROCK_BASE_URL` and its siblings are the existing example). A
  variable that the adapter sets but does not know how to be overridden by is a
  path around the bridge. Adding a set without checking for its overriders is a
  **critical** finding.
- **A credential must never reach a command line.** Process arguments are
  world-readable on most systems. Anything secret goes in the environment or a
  file, never in `cli_args`. Quote the line.
- **A settings or config file kitty writes for the child must be removed or
  restored when the child exits** — README promises *"When the agent exits, kitty
  restores the agent's config files."* A path that can leave the user's real
  config replaced, or a temp file containing a token behind, is critical. Check
  the exception path and the signal path, not just the happy one.
- **The child inherits the real terminal.** `stdin`, `stdout` and `stderr` are
  passed through; kitty does not interpose, does not read stdin while the agent
  runs, and touches neither `TERM` nor `COLORTERM`. A change that wraps a stream,
  allocates a pty, or emits a terminal escape sequence is a finding. A *suspected*
  regression here is expensive out of all proportion to the diff: an open bug
  report against this behaviour has so far cost a full static audit of the launch
  path plus a four-run bisection protocol, and has still not named a cause.
- **User arguments pass through unchanged.** The adapter prepends its own flags;
  it does not filter, reorder or rewrite what the user typed. A diff that starts
  interpreting the agent's own arguments is a finding.

## Adding an agent

A new launcher target lands complete only when all of these move together, and a
reviewer should check each: the adapter itself; the target name registered in
`cli/router.py`; the name added to the reserved-profile-name list (or a user can
create a profile that shadows it); binary discovery and a `kitty doctor` check;
and README's supported-agents table. A missing reserved name is the one that
produces a confusing routing bug months later.

## Binary discovery

`discovery.py` decides which binary runs. Two failure shapes worth checking:

- resolving through `PATH` where a stale or shadowing entry can win — an absolute
  path is what makes this deterministic;
- a discovery failure reported as something other than "the agent is not
  installed", which sends the user looking at their profile.

## Layering

`pyproject.toml` forbids `kitty.launchers` from importing `kitty.cli`. The
orchestrator lives in `cli/` and depends downward on the adapters, not the other
way round. A new import inverting that is a finding.

## Tests

Each adapter's spawn config is pure data, so it is cheaply testable and there is
no excuse for an untested change. The useful assertion is on the **exact** env
map and flag list, not on a substring: a test that checks `ANTHROPIC_BASE_URL` is
present will not notice that `ANTHROPIC_AUTH_TOKEN` stopped being cleared.

## Severity

- **Critical** — a path by which the child can reach the provider without the
  bridge; a credential in a command line; a config file not restored on any exit
  path; a cleared-variable list that no longer covers its overriders.
- **Warning** — a new target missing its router entry, reserved name, doctor
  check or README row; a discovery change with no stated reason; a spawn-config
  change tested only by substring.
- **Suggestion** — naming, duplication between adapters, docstrings.
