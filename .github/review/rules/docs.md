# Rule: documentation (`README.md`, `assets/**`, and the design documents)

## README.md is the user-facing specification

It is not documentation *about* this repository — it is the closest committed
thing it has to a **user-facing** contract, and it is one of the two
specifications an automated review can read; the design documents under
`.system_design/` are the internal one (see `rules/packaging.md` on
`.gitignore`). It carries:

- the command contract and the routing order,
- the supported-agents and supported-providers tables,
- the profile model, the balancing and backup semantics, and the reserved names,
- the egress feature, its resolution order and its fail-closed promise,
- bridge mode, the logging flags and their default paths,
- the privacy claims — *"Just a bridge"*, *"No AI inside Kitty"*, *"Does Kitty
  record your prompts? No."*,
- the troubleshooting and FAQ that tell a user what a given failure means.

**A change to behaviour that does not move README is drift, and drift is the
finding.** Quote both sides — the README sentence and the diff line — so the
author can see they disagree without going to look.

The pairs that go stale most often, and are worth checking by name whenever the
diff touches them:

| If the diff changes… | README must move |
|---|---|
| a provider adapter or the registry | the supported-providers table |
| a launcher target | the supported-agents table |
| a command, flag or the router | the commands section and the routing description |
| profile lifecycle or balancing rules | *Profiles* and *Balanced Profiles* |
| egress resolution, coverage or fail-closed behaviour | *Static Egress*, including the bullet list |
| a log path or a default | *Logging* |
| an error message a user will search for | *Troubleshooting* / *FAQ* |
| the directory layout | *Project structure* |

## The privacy claims are load-bearing

Four sentences in README are promises, not marketing, and a change that
contradicts one is a **critical** finding whatever else it does:

- *"Kitty runs on your machine. It does not send your prompts, code, or files to
  any third-party service beyond the backend LLM provider you explicitly
  configure."*
- *"It does not get filesystem access, shell access, or any other extra
  capabilities."*
- *"Kitty does not use an LLM, embeddings, or any other AI system internally."*
- *"Kitty does not send data to third parties, store conversations, or collect
  telemetry."*

If a diff adds a network call, a write, a telemetry hook or a model call, the
question is not whether it is useful — it is which of those four sentences now
has to change, and whether the pull request says so.

## Docstrings

Every class, method and function carries a **Google-style** docstring with
`Args:`, `Returns:` and `Raises:`, and every module a module-level docstring.
Hold **changed** symbols to that; mention pre-existing gaps without making them
blockers.

A docstring that no longer matches the code beneath it is worse than none —
raise it at the same severity as a wrong comment, because a reader trusts it.

## Design documents

`.system_design/` **is tracked and reaches a CI checkout.** `.requirements/` is
still in `.gitignore`, so per-task requirement documents do not: never report a
missing `REQUIREMENTS.md` as a finding, and never claim to have read one.

Read the design document covering the area the change touches and judge the
change against it. Two failures are findings, and the second is the one that gets
missed: code that contradicts a stable design, **and** a design left stale by a
change that invalidates it. A design document that still describes behaviour the
diff just replaced is a defect with the same shape as a wrong docstring, and it
is worse in one respect — the next author reads it as the specification. These
files are large; read the relevant section, not the whole thing.

## Assets

`assets/**` is what README renders. A replaced or removed image that leaves a
broken reference is a documentation defect and nothing else selects it.

## Severity

- **Critical** — a change that contradicts one of the four privacy claims.
- **Warning** — behaviour changed without the matching README section; a
  docstring that now describes something else; a broken asset reference; a
  documented default that the code no longer uses.
- **Suggestion** — wording, a missing cross-reference, an example that could be
  clearer.
