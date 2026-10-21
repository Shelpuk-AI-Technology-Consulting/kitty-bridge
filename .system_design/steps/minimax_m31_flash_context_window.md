---
id: minimax_m31_flash_context_window
depends_on: []
---

# Pin MiniMax M3.1 Flash's 1M context window

Adds two rows to the pin catalog
(`src/kitty/providers/model_context_overrides.json`):
`MiniMax-M3.1-Flash: 1000000` and `minimax-m3.1-flash:free: 1000000`, plus
the pinning-test coverage. Design of record: `SYSTEM_DESIGN.md` §17.

## Why

`MiniMax-M3.1-Flash` is absent from both catalogs, so every launch path
resolved `DEFAULT_CONTEXT_TOKENS` (200,000) and the bridge compacted roughly
five times too early on a 1M-token model. Verified 2026-10-21: MiniMax's API
overview and the OpenRouter API both report 1,000,000; OpenRouter serves the
model only as `minimax/minimax-m3.1-flash:free`, which the suffix matcher
cannot reach (`:free` is not a `/`-delimited prefix) and the packaged
metadata snapshot predates the model. Owner chose the two-key pin over a
metadata refresh (2026-10-21); see §17.3 D2/D3.

## Implementation notes

Landed 2026-10-21. TDD: the pinning tests were written first and failed 17
cases for the right reason (catalog lacked the keys; the resolver fell
through to `provider_config` / the 200k default), then the two catalog rows
turned them green — 75/75 in `tests/test_model_context_packaged_catalog.py`,
96/96 across `test_model_context.py` + `test_model_context_sync.py`. The
empirical acceptance probe confirmed the four requirement-3
`(provider, model)` pairs resolve 1,000,000 against the real packaged
catalog (200,000 before the change). `minimax_token` joined
`PROVIDER_VARIANTS` so the coding-plan pair is literally asserted (the
existing `minimax_coding_plan` label is log-only); both OpenRouter
spellings ride the suffix-spelling table. Requirements review
(system-design-reviewer) surfaced no blockers; its six findings — stale
step-3 wording, the untested `minimax_token` pair, the remote-cache
isolation caveat, gate ordering, the provider-dialect comment, the rollout
sentence — are all applied to the requirements doc and tests.
