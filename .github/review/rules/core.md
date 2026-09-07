# Rule: shared types and validation (`types.py`, `validation.py`, package entry points)

These four files are imported by every layer. `types.py` carries the dataclasses
the bridge, the providers, the launchers and the CLI all pass between them;
`validation.py` carries the predicates each of them validates its inputs with;
`__init__.py` decides the package's importable surface and `__main__.py` is what
`python -m kitty` runs.

So a change here is a change to **every** component, which is why the selector
fans this rule out to all six consumers rather than scoping it to a directory.
Review it that way: for each field or predicate the diff touches, ask who reads
it and what happens to them.

## What to check hardest

- **A field added to a shared type is a field every producer must now set.**
  Look for the construction sites the diff did not touch. A dataclass field with
  a default hides this — the object constructs fine and carries the wrong value.
- **A field renamed or retyped is a wire-format change** wherever that type is
  serialised. Trace it to the profile store and the credential store, both of
  which round-trip through JSON on the user's disk. A user's existing
  `~/.config/kitty/profiles.json` must still load.
- **A validation predicate loosened is a check removed everywhere at once.**
  `validation.py` is the single definition of what a legal profile name, model
  id or URL looks like. Widening one of them widens what reaches the provider
  adapters and the launcher env builders, both of which trust it.
- **The import layering.** `pyproject.toml` declares three import-linter
  contracts, and these modules sit below all of them. A new import added to
  `types.py` or `validation.py` that reaches up into `bridge`, `cli`, `launchers`
  or `providers` inverts the layering. `lint-imports` catches it; say so rather
  than assuming the author ran it.

## Backward compatibility

`kitty` is installed from PyPI and upgraded in place over a config directory the
previous version wrote. There is no migration step and no schema version on the
profile file. So:

- A change that makes an existing `profiles.json`, `credentials.json` or
  `egress.json` fail to load is a **critical** finding, whatever else it does
  right.
- The `[project.scripts]` entry point is `kitty`. Users have typed it into
  shells, scripts and CI. Renaming it, or renaming the module path behind it, is
  a critical finding.

## Severity

- **Critical** — a shared type or predicate change that breaks an existing
  on-disk config, inverts the import layering, or silently changes what a
  consumer receives.
- **Warning** — a new field or predicate with no test, or with consumers the
  change did not update.
- **Suggestion** — naming, docstring drift, a comment that would save the next
  reader tracing the consumers by hand.
