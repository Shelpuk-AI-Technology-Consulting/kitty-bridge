---
id: kbr158_delete_dead_profile_base_url
depends_on: []
---

# KBR-158 — Delete the dead `Profile.base_url` field; reject top-level `base_url`

## What

Drop `Profile.base_url` and the `HttpsUrl` class it used. Add a targeted
`@model_validator(mode='before')` that rejects a top-level `base_url` key whose
value is **not `None`**, with a message naming `provider_config["base_url"]`.
Rewrite the five `tests/test_profile_schema.py` cases that exercise the dead
field. Record the decision in `.system_design/SYSTEM_DESIGN.md` (new §10) and
update the stale references (`TEST_SUITE.md §7.5.2`,
`tests/harness/bridge.py` `profile_for` docstring,
`src/kitty/launchers/base.py` `build_spawn_config` docstring).

## Why

The dead field validates differently from the live channel (`provider_config`
untyped dict — the wizards' output) and is silently ignored by pydantic's
default `extra='ignore'`. A reader of the schema would reasonably conclude
HTTPS-only base URLs and pydantic normalisation; neither is true of the path
that actually runs. A hand-edited profile carrying top-level `base_url` was
silently routed to the provider's default endpoint — the wrong way to be
wrong. Replacing silent ignore with a pointed rejection is the smallest diff
that meets the ticket's own acceptance clause ("never silently ignored").

**Value-aware, not key-presence.** `store._serialize_entry` dumps with
`model_dump(mode="json")` and no `exclude_none`, so every `profiles.json` on
disk carries `"base_url": null`. A key-presence check would raise on load for
every existing profile, and `_deserialize_entry`'s broad `except Exception`
(F44 precedent) would silently drop the lot — the never-silently-ignored
failure mode aimed at every user at once. A serialized `null` means "not set"
(the field was `Optional` with default `None`), so the check is
`data.get("base_url") is not None`; only a typed value — the user who actually
set a URL — gets the pointed message. The rejection fires where a `Profile`
is constructed in code; a hand-edited *store file* with a non-null top-level
`base_url` still goes through the existing invalid-entry path (warning log +
skipped entry). Surfacing per-entry validation messages from the store is out
of scope. The validator guards with `isinstance(data, dict)` because
`mode='before'` also receives model instances (via
`Profile.model_validate(some_profile)`), on which the key test would silently
fall through.

## Why not the other options

- **Make authoritative + migrate** inverts the channel
  `.system_design/TEST_SUITE.md §7.5.2` already names authoritative, requires
  relaxing `HttpsUrl` for `http://` local endpoints, touches all five reader
  sites plus both wizards, and needs a profile migration. Large diff, real
  compatibility work, for a contract the wizards can validate at write time.
- **Type `provider_config` instead** is a per-provider discriminated-union
  design (Vertex reads `project_id/location`; Azure reads `resource`/
  `api-version`; etc.) — a much larger project than a Low-priority bug
  ticket should carry, and it duplicates validation the wizards + per-adapter
  checks already perform.

## Verification

- `grep -rn "base_url" src/kitty/profiles/schema.py` → only the
  `provider_config` dict field.
- `python -c "from kitty.profiles.schema import Profile; assert 'base_url' not in Profile.model_fields"` → exit 0.
- `pytest tests/test_profile_schema.py` → exit 0.
- `pytest` (unit scope) → exit 0; no test outside `test_profile_schema.py`
  regresses.
- `grep -n "HttpsUrl" .system_design/TEST_SUITE.md` and
  `tests/harness/bridge.py` → no stale "loopback recorder serving http://"
  reference to the deleted field/class.

## Implementation notes

Pydantic 2.13.5 is the resolved version. `@model_validator(mode='before')`
receives the raw input dict before per-field validation, so it sees
undeclared keys; raising `ValueError` is wrapped by pydantic into a
`ValidationError` whose loc is the model. The message names
`provider_config["base_url"]` directly — the loc does not need to enumerate
the field because the message says where the value should go.
