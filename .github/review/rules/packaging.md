# Rule: packaging, typing and repository hygiene (`pyproject.toml`, `.gitignore`)

`pyproject.toml` is not bookkeeping here. It carries five things a change can
break silently, and `.gitignore` decides what the CI checkout — and therefore
this reviewer — can see at all.

## The five things `pyproject.toml` carries

### 1. The import-linter contracts

Three `[[tool.importlinter.contracts]]` blocks are the **only** enforcement of
the module layering:

- `kitty.profiles`, `kitty.credentials`, `kitty.providers` and `kitty.tui` are
  leaves and must not import `kitty.cli`, `kitty.launchers` or `kitty.bridge`;
- `kitty.bridge` must not import `kitty.cli`, `kitty.launchers`,
  `kitty.profiles`, `kitty.credentials` or `kitty.tui`;
- `kitty.launchers` must not import `kitty.cli`.

`exclude_type_checking_imports = true` is deliberate and documented in the file:
a type-only import creates no runtime dependency, which is what lets the bridge
be fully typed against `Profile` and `LauncherAdapter` without importing either
at run time. Do not raise it as a loophole.

**Removing a module from a `source_modules` list, or a module from a
`forbidden_modules` list, is a critical finding** unless the pull request is
explicitly redesigning the layering and says so. It is the exact shape of "make
the check pass" — and the check going green afterwards proves nothing, because
the contract it was enforcing is gone.

### 2. The dependency bounds

`aiohttp>=3.11,<3.14` is the only upper bound and it is load-bearing: the bridge
is built on that client's streaming API. A bound lifted or widened changes what
every user's next `pip install --upgrade kitty-bridge` resolves — and this
repository's own PR reviewer installs the released package on every run, so a bad
resolve degrades the reviewer too. A widening with no note about what was tested
against is a **warning** at minimum.

### 3. `requires-python` and the classifiers

`>=3.10`, with per-version classifiers. **The `tests.yml` matrix must track
both.** A floor raised in one place and not the others produces a package that
installs on a version nothing tests, or a matrix that burns minutes on a version
the package refuses. Check all three moved together.

### 4. The entry point

`[project.scripts]` declares exactly one: `kitty = "kitty.cli.main:main"`. Users
have typed `kitty` into shells, scripts, CI and README. Renaming it, or moving
the module behind it, is a **critical** backward-compatibility break. This rule
pulls in `rules/cli.md` for that reason.

### 5. The tool configuration

`ruff` (line length 120, the seven selected rule families), `mypy` (with its
`ignore_missing_imports` override list), and `pytest` (`asyncio_mode = "auto"`,
`testpaths`). Removing a `ruff` family, adding a module to the mypy override
list, or narrowing `testpaths` all reduce what CI checks while leaving it green.
Each needs a stated reason.

`mypy src/kitty` is a **gate, not a report** — the comment in `tests.yml` records
that it found four user-visible defects the test suite could not, each on a path
tests do not execute. A change that makes it advisory is a critical finding.

## `.gitignore`

🔴 It excludes `/.system_design/`, `/.requirements/`, `/CLAUDE.md` and
`/.references/`. **None of those reaches a CI checkout**, which means none of
them reaches the automated reviewer either — `README.md` is the only committed
specification it can read.

So a line added or removed here silently widens or narrows every future review,
and nothing goes red. Two specific checks:

- **A line added** that excludes something the reviewer or the test suite reads
  is a finding. Say what stops being visible.
- **A line removed** that starts committing a directory is a *good* change if it
  is one the reviewer should read — and it needs `select_rules.py` checked, since
  a newly committed directory that no pattern matches is reviewed with **zero
  rule files loaded**.

Also check nothing secret can be committed: a credential store, a debug log or a
`.env` reaching the tree is critical.

## Packaging correctness

`tests/test_pypi_packaging.py` is the existing guard. The wheel is built from
`packages = ["src/kitty"]`, so a data file added outside a package directory does
not ship — `providers/model_metadata.json` is inside one, which is why it works.
A new data file added anywhere else is a finding: it passes every local test,
because a source checkout has it, and is absent from the installed package.

## Severity

- **Critical** — a layering contract weakened; the entry point renamed; mypy or a
  lint gate made advisory; a secret path removed from `.gitignore`.
- **Warning** — a dependency bound widened with no note; `requires-python`,
  classifiers and the CI matrix out of step; a data file that will not ship; a
  `.gitignore` change that narrows what the reviewer can read.
- **Suggestion** — ordering, comments, a bound that could be tightened.
