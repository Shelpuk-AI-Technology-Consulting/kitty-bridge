---
id: redacted_thinking_unsupported_upstream
depends_on: []
---

# Recover when an upstream rejects `redacted_thinking`

## Planned delivery

Implements `SYSTEM_DESIGN.md` §18: detect the upstream's `unknown variant` rejection, strip only
`redacted_thinking`, retry the same backend ahead of M9, and remember the backend. Testable through
`tests/bridge/test_redacted_thinking_unsupported.py`.

## Implementation notes

What landed (2026-11-04):

1. `_is_redacted_thinking_unsupported_error` and `_strip_redacted_thinking_blocks`, joined by
   `BridgeServer._recover_unsupported_redacted_thinking`, called from `_stream_messages` ahead of M9 and
   from `_make_upstream_request` ahead of M17.
2. `_redacted_thinking_strip_backends`, consulted by `_upstream_body_for`, so a later turn sends no
   rejected request.
3. Register row M34 and trigger `REDACTED_THINKING_UNSUPPORTED`, with the count pins and §4.3 C3 lists.

Root cause, found by reproducing it rather than from the error text: the "expected" list names `tool_use`,
so M9 also matched. Its rebuild restored the same block through the KBR-228 carriage, so the retry failed
identically. `_is_tool_use_format_error` stays broad on purpose (`SYSTEM_DESIGN.md` §18.2).

Design review moved the change from an M17 amendment to its own row, because M17's trigger cannot describe
a remembered backend, and scoped the carrier repair to the stream, the only place it runs. Known limits are
in §18.3.
