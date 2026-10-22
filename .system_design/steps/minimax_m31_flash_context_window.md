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

PR #301 review round 1 (CI bot, 2026-10-22, 3 findings, all addressed):
(1) §17.4's "EXPECTED_CATALOG equality with the shipped file" had no
implementing test — `TestShippedCatalogMatchesExpectations` now asserts the
shipped file equals `EXPECTED_CATALOG` in both directions (proven to fail on
a dropped `:free` pin before being restored); (2) §17.2's collision-rejection
sentence now scopes the wholesale rejection to the synced cache, with the
packaged catalog held by CI instead; (3) §17.1's catalog summary no longer
calls every key cloud-native. Merge with main (PR #299) resolved the
SYSTEM_DESIGN.md tail conflict; the context-window path was untouched.

PR #301 review round 2 (CI bot, 2026-10-22, 3 findings, all addressed,
docs-only): (1) §17.5's "refreshed manually" was false — `model-metadata.yml`
refreshes the table weekly via cron and opens a PR; §17.5 now says so;
(2) the PROVIDER_VARIANTS comment now names which rows are registered
provider types (`zai_coding`, `minimax_token`, `custom_anthropic`) and which
are descriptive service names (`qwen_cloud`, `minimax_coding_plan`,
`deepseek`); (3) two stale module-docstring statements corrected —
`z-ai/glm-5.3-flash` has since appeared in the metadata table, and the two
M3.1 pins now lack metadata counterparts, which makes the shipped data
itself distinguish catalog-over-metadata for them.
