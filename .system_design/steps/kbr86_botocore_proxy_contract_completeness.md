---
id: kbr86_botocore_proxy_contract_completeness
depends_on: []
---

# KBR-86 — botocore proxy contract completeness (T-G11 remainder)

Ticket: [KBR-86](https://shelpuk.atlassian.net/browse/KBR-86) (In Progress;
T-G11). Design: `TEST_SUITE.md` §6.2.4 (botocore row), §5.5 (ambient-proxy
paragraph). Requirements:
`.requirements/20260921T115359Z_kbr86_botocore_proxy_contract_completeness/REQUIREMENTS.md`.
Predecessor: **KBR-64** (PR #204) landed the contract file, the
`botocore>=1.34` declaration and the §6.2.4 row; KBR-85 rounds 9/11 surfaced
the completeness gaps this step closes.

## What the task does

Closes the three completeness gaps the KBR-86 comment of 2026-09-18 enumerates
(mirroring what the curl_cffi twin settled over eleven review rounds), plus the
documentation ask from the 2026-09-12 comment. Two scope questions were put to
the product owner and decided 2026-09-21: **client caching is out** (the
`_get_boto3_client` per-request build stays; the contract file documents the
fact instead) and **the version floor stays `>=1.34`** (the 2026-09-15
"floor should match 1.43.93" suggestion is superseded by KBR-64's reviewed
rationale: bumps route through the lockfile and the contract test is the gate).

The change:

1. `tests/harness/test_botocore_transport_contract.py`:
   `test_config_proxies_overrides_matching_no_proxy` becomes parametrized over
   `["NO_PROXY", "no_proxy"]` (the both-set form only ever proved the
   lowercase-read path — urllib's lowercase preference shadows the uppercase
   variable); new non-matching-`NO_PROXY` control (mirror of the curl twin's
   same-named probe); new class `TestTheAmbientProxySourceWhenTheMappingIsAbsent`
   with the per-variable falsification parametrized over
   `["HTTPS_PROXY", "https_proxy"]` (mapping patched away, dead ambient, drive
   must die against the dead proxy — non-200, zero harness-proxy attempts, and
   the failure text naming `127.0.0.1` as the vacuous-pass discriminator) and
   the scheme-scoping negative parametrized over
   `["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"]` (botocore reads
   only the scheme key, so these must not steer an `https://` request even
   with the mapping absent). File collects 12 cases (nine new-or-changed,
   three unchanged).
2. Measured-version cites 1.43.93 → 1.43.94 at all three harness sites
   (`test_botocore_transport_contract.py`, `botocore_containment.py`,
   `botocore.py`); the green baseline run on the resolved version
   re-establishes the measurement. The KBR-78 header-test cite
   (`test_upstream_identity_consistency.py`) and the recorder's
   `SCHEMA_VERSION = "botocore-1.43.93"` wire-versioned literal are deliberately
   untouched (different contract / not a cite).
3. `TEST_SUITE.md` companion sweep: §6.2.4 row, §5.5's botocore clause, and the
   G10 row's botocore half all name the new pins.

## Key decisions

- **Two falsification regimes, not one four-var probe.** The curl twin's
  four-variable "is read" parametrization works because curl consults the
  catch-all; botocore does not (`ProxyConfiguration.proxy_url_for` looks up
  only the scheme key — verified against installed botocore 1.43.94 source and
  a `urllib.request.getproxies()` probe). Pinning botocore's actual behaviour
  means a positive falsification for the two scheme casings and a scoping
  negative for the other four — the botocore-true mirror of the twin's
  "is read" + scoping-negative pair, not a copy of its parametrize list.
- **Text discriminator on the "is read" probe.** `status != 200` alone would
  pass vacuously on any unrelated drive failure; the failure body carries the
  botocore error message (`Failed to connect to proxy URL: "https://127.0.0.1:1"`),
  so the probe asserts `127.0.0.1` appears in `result.text` — the curl twin's
  error-substring pattern.
- **Red-conditions demonstrated before green.** Every new probe family was
  watched failing for the right reason via temporary mutations (mapping
  patched away for the precedence probes; `setenv` dropped for the "is read"
  probe; variable swapped to `HTTPS_PROXY` for the scoping negative), then the
  mutations were reverted and the file re-run green.

## Implementation notes

- 2026-09-21: landed as described; 12/12 green locally; `ruff check`,
  `ruff format`, and the CI mypy gate (`mypy src/kitty`) pass. Requirements
  reviewed by the system-design reviewer before implementation (nine findings
  applied, including the vacuous-pass discriminator, the `-k` selector fix in
  AC1, and the count arithmetic); diff reviewed by the code reviewer.
