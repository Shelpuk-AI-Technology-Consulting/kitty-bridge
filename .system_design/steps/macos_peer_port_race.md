---
id: macos_peer_port_race
depends_on: []
---

# The silent-connection test waits for the peer port to be attributed

## Planned delivery

§7.2.3's contract is that the recorder's connection record is created at
accept time with `peer_port = -1` and filled in by `connection_made` on
handshake success. Its "every port it opens is filled in before the check
runs" is scoped to the conformance driver's probes: a probe's reply cannot
arrive before the server-side handshake completes, so the reply is its
synchronisation — and the silent connection this step fixes sends none, so
its check must wait explicitly. One check did not wait:
`test_curl_recorder.py`'s silent-connection test read the log the moment
its client-side handshake returned. The server-side handshake — the one
that fills the port — can complete a loop iteration later, so on the macOS
leg the record still carried the placeholder and the assertion read
`assert -1 == <port>`.

This step closes that gap the way the file's other async-appearing
assertions already do: wait on the condition — the record carrying the
reported port — with the bounded 2-second timeout `_wait_until` provides,
before reading the log.

Requirements and the test plan live in
`.requirements/20261102T072014Z_macos_peer_port_race/REQUIREMENTS.md`.

## Decisions and why

- **Fix the check, not the recorder.** A record that keeps `-1` is honest
  evidence — the connection existed, no peer ever identified itself over
  TLS — so nothing else may fill it in; filling it from `connection_lost`
  would destroy that property. The recorder already behaves as §7.2.3
  describes; only the check's timing was wrong.
- **Wait, do not sleep.** A fixed sleep trades a race for a slow test and
  stays timing-dependent; `_wait_until` waits on the condition with a
  bounded timeout — the shape every other async-appearing assertion in the
  file uses.
- **Assert the wait's result.** The condition *is* the assertion, so a port
  that is never attributed must fail here with the record list in the
  message, rather than surfacing later in an unrelated check.
- **Scope: the check that was wrong.** The sibling `test_per_session_check`
  hit the same race and was fixed by commit `235fdff` (2026-09-16); this
  was the second site, left untouched by that fix.

## Implementation notes (2026-11-02)

- `tests/harness/test_curl_recorder.py` — one test
  (`TestSocketLevelConnectionLogging::test_a_silent_connection_is_logged_with_its_peer_port`)
  now waits on the peer port being attributed before it reads the log. The
  load-bearing assertions are unchanged: one record, the reported peer port,
  zero requests.
- The recorder (`tests/harness/curl_recorder.py`) is unchanged.
- Verified by a throwaway harness that deferred the peer-port fill by 0, 50
  and 200 ms: the pre-fix assertion reproduces the CI's failure shape at
  50 and 200 ms (`assert -1 == <port>`), and the shipped test passes at all
  three.
- The fix unblocks the v1.21.0 release: the tag's `Publish to PyPI` run
  (41359296734) failed on the macOS leg and skipped the publish job.
