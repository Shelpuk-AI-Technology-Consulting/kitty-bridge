# Rule: the command surface and TUI (`cli/**`, `tui/**`)

The router, the built-in commands, and the interactive menus those commands open.
The TUI is in this rule rather than its own because it has no behaviour of its
own — it is how the commands ask their questions.

`cli/launcher.py` is covered by `rules/launchers.md` as well as this one. Its
spawn behaviour is judged there.

## The routing contract

README documents what `kitty <word> ...` means, and the order matters:

1. a built-in command (`setup`, `profile`, `doctor`, `bridge`, `egress`,
   `cleanup`, `auth`),
2. a launcher target (`claude`, `codex`, `gemini`, `kilo`),
3. a profile name,
4. otherwise fail fast, listing profiles and targets.

Two consequences a reviewer has to hold:

- **Every name in tiers 1 and 2 must be a reserved profile name.** A command or
  target added without joining the reserved list lets a user create a profile
  that can never be selected — and the failure appears months later as "my
  profile stopped working". Check `rules/credentials.md` for the other half.
- **A change to the order changes what an existing user's typed command does.**
  Flag a reorder the pull request does not explain.

## Failing fast

The stated contract is that kitty checks the profile and resolves the credential
**before** launching, so a misconfiguration produces a clear message rather than
a cryptic failure inside the agent. `--no-validate` is the documented escape for
offline use.

- A path that launches the agent on an unvalidated profile bypasses that.
- A message that names the symptom but not the remediation is a finding in this
  repository specifically: every failure here is something the user can fix, and
  the message is the only place they will learn how.

## Interactive versus non-interactive

`kitty setup` and `kitty profile` are interactive and **must fail with a
deterministic error on a non-TTY**, not hang waiting for input. Anything that can
block on stdin without a TTY check is a finding — it wedges CI and scripted
installs with no output.

The auto-launch path (routing reaches a target or profile, no profiles exist, so
`setup` runs and the original command is retried) is the one place this is easy
to get wrong.

## Output

- **A secret must never be printed.** Keys, tokens, and the egress password are
  masked wherever kitty prints them — including `doctor`, `egress show`, and any
  new table or menu. A new render path that prints a config object wholesale is a
  **critical** finding; check what the object's `repr` contains.
- Errors go to stderr, results to stdout. In bridge mode stdout carries protocol
  output; a stray `print` there corrupts it.
- Exit codes are a contract for scripted use (`tests/test_exit_code_mapping.py`).
  A change that collapses two distinct failures onto one code, or that returns 0
  on a failure, is a finding.

## Layering

`kitty.cli` sits at the top and may import downward. `kitty.tui` is a leaf and
must not import `cli`, `launchers` or `bridge`. A menu that reaches into the
launcher to do its work has the dependency backwards.

## Tests

The TUI is testable — the existing suite drives the menus and the wizard without
a terminal. "It is interactive, so it cannot be tested" is not accepted here;
`tests/tui/` is the proof. A new prompt, menu branch or wizard step with no test
is a finding.

## Severity

- **Critical** — a secret printed; a launch on an unvalidated profile; a
  non-TTY path that can block.
- **Warning** — a command or target missing from the reserved names; a routing
  order change with no reason; an exit code collapsed; a new menu branch with no
  test; a message with no remediation.
- **Suggestion** — wording, layout, duplication between menus.
