"""L1 unit tests for the setup wizard's credential reuse scan (KBR-87).

``kitty.cli.profile_cmd._find_reusable_auth_ref`` decides which stored key the
setup wizard may re-use. Under the KBR-87 corruption contract a stored value can
raise :class:`~kitty.credentials.store.CredentialError` — the scan must skip a
damaged profile and keep the wizard reachable, because re-entry *is* the
recovery.
"""

from __future__ import annotations

import json
import uuid

from kitty.cli.profile_cmd import _find_reusable_auth_ref
from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
from kitty.profiles.schema import Profile
from kitty.profiles.store import ProfileStore


def _profile(name: str, auth_ref: str) -> Profile:
    return Profile(name=name, provider="zai_regular", model="gpt-4o", auth_ref=auth_ref)


def test_scan_skips_a_corrupt_profile_and_returns_the_valid_one(tmp_path) -> None:
    """A damaged profile does not crash the scan; a healthy sibling is returned."""
    good_ref, bad_ref = str(uuid.uuid4()), str(uuid.uuid4())
    store = ProfileStore(path=tmp_path / "profiles.json")
    store.save(_profile("good", good_ref))
    store.save(_profile("bad", bad_ref))
    (tmp_path / "credentials.json").write_text(
        json.dumps({good_ref: "c3RvcmVkLWtleQ==", bad_ref: "@@@ not base64 @@@"}), encoding="utf-8"
    )
    cred_store = CredentialStore(backends=[FileBackend(path=tmp_path / "credentials.json")])

    assert _find_reusable_auth_ref(store, cred_store, "zai_regular") == good_ref


def test_scan_returns_none_when_every_profile_is_corrupt(tmp_path) -> None:
    """All-damaged means nothing reusable — the wizard prompts fresh, no crash."""
    bad_ref = str(uuid.uuid4())
    store = ProfileStore(path=tmp_path / "profiles.json")
    store.save(_profile("bad", bad_ref))
    (tmp_path / "credentials.json").write_text(
        json.dumps({bad_ref: "@@@ not base64 @@@"}), encoding="utf-8"
    )
    cred_store = CredentialStore(backends=[FileBackend(path=tmp_path / "credentials.json")])

    assert _find_reusable_auth_ref(store, cred_store, "zai_regular") is None
