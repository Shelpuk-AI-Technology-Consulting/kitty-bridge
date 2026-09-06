# Rule: provider adapters (`providers/**`, `scripts/**`)

Twenty-odd adapters, one contract: take the bridge's request, build this
provider's dialect of it, authenticate, send it, and map what comes back —
including the errors — into the shared shape the bridge translates from. The
registry is what keeps the set consistent; `model_metadata.json` is the bundled
catalogue and `scripts/script_update_model_metadata_table.py` is what regenerates
it.

They are one rule rather than twenty because a change to one adapter is almost
always a question about whether the other nineteen need the same change.

## The contract each adapter owes

- **Register.** An adapter that exists but is not reachable through the registry
  is dead code that looks alive. A new provider that does not appear in the
  registry, the setup wizard's provider list, and README's provider table is
  incompletely landed — all three, and the README one is the most often missed.
- **Authenticate the way that provider actually requires**, and nowhere else. A
  key belongs in exactly the request it authenticates.
- **Map errors, do not pass them through raw.** An upstream body reaches the
  agent through the bridge's error envelope. An adapter that lets a provider's
  raw error text through is both a transparency problem and a leak risk — those
  bodies quote request fields back.
- **Say what the provider actually supports.** Capability flags (streaming,
  tools, reasoning effort, context length) are read by the bridge to decide what
  to send. A flag set optimistically produces a runtime failure the user reads as
  kitty being broken.

## Credentials — the sharpest check

**A provider key must never leave the request it authenticates.** Not into a log,
an exception message, a `repr`, a `kitty doctor` line, an error returned to the
agent, or a URL. Two shapes to look for specifically:

- a key interpolated into a **path or query string** rather than a header, which
  puts it in every intermediary's access log;
- an exception raised with the request object, or a `raise ... from e` that
  carries a body containing the echoed key.

Treat either as **critical** and quote the line.

## Transparency to upstream

See `rules/bridge.md` for the full statement. The adapter half of it: the request
this code builds is what the provider sees, so **anything identifying kitty that
the agent would not itself have sent is a finding here**. Existing attribution
headers are the deliberate exception; a new one, or a widened one, is not.

## The catalogue

`model_metadata.json` is a convenience copy of the OpenRouter catalogue,
refreshed weekly by `ci.yml`. Two things follow:

- **A hand-edit to that file is a finding**, because the next scheduled refresh
  overwrites it. The fix belongs in the generator.
- **The generator is not a throwaway script.** It writes a file the package ships
  and every profile resolves against. A change to its output shape needs the
  consumers checked — `model_context.py` and `model_context_sync.py` read it.

A change to the generator with no corresponding test is a finding;
`tests/test_model_context_packaged_catalog.py` and `test_provider_list_sync.py`
are the guards that already exist here.

## Layering

`pyproject.toml` forbids `kitty.providers` from importing `kitty.cli`,
`kitty.launchers` or `kitty.bridge`. Providers are a leaf. A new import reaching
up is a finding.

## Consistency across the set

When a diff changes one adapter, ask explicitly whether the others need it, and
say so in the finding. This is the single most valuable thing a reviewer does in
this directory, because the tests are per-adapter and a per-adapter test suite
cannot notice that nineteen adapters were left behind.

Use `other_instances` for this: **one** finding naming the pattern, with the
other call sites listed, not nineteen findings.

## Severity

- **Critical** — a key that can reach a log, message, URL or command line; an
  identifying field added to an upstream request; a raw upstream body reaching
  the agent.
- **Warning** — an adapter changed without its siblings; a capability flag that
  overstates support; a registry, wizard or README entry not updated; a
  hand-edited catalogue.
- **Suggestion** — duplication between adapters that a shared base would remove,
  naming, docstrings.
