"""Tests for credentials/store.py, keyring_backend.py, file_backend.py."""

import base64
import json
import uuid
from unittest.mock import MagicMock

import pytest

from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialBackend, CredentialError, CredentialNotFoundError, CredentialStore
from kitty.profiles.schema import Profile

VALID_UUID = str(uuid.uuid4())


def _make_profile(name: str = "test", auth_ref: str = VALID_UUID) -> Profile:
    return Profile(name=name, provider="zai_regular", model="gpt-4o", auth_ref=auth_ref)


class TestCredentialBackendInterface:
    def test_cannot_be_instantiated(self):
        from kitty.credentials.store import CredentialBackend

        with pytest.raises(TypeError):
            CredentialBackend()  # type: ignore[abstract]


class TestCredentialStoreFallback:
    def test_tries_backends_in_order(self):
        primary = MagicMock(spec=CredentialBackend)
        secondary = MagicMock(spec=CredentialBackend)
        primary.get.return_value = None
        secondary.get.return_value = "secret-from-secondary"

        store = CredentialStore(backends=[primary, secondary])
        result = store.get("ref")
        assert result == "secret-from-secondary"
        primary.get.assert_called_once_with("ref")
        secondary.get.assert_called_once_with("ref")

    def test_returns_from_first_backend(self):
        primary = MagicMock(spec=CredentialBackend)
        secondary = MagicMock(spec=CredentialBackend)
        primary.get.return_value = "secret-from-primary"

        store = CredentialStore(backends=[primary, secondary])
        result = store.get("ref")
        assert result == "secret-from-primary"
        secondary.get.assert_not_called()

    def test_set_writes_to_specified_backend(self):
        backend = MagicMock(spec=CredentialBackend)
        store = CredentialStore(backends=[backend])
        store.set("ref", "value")
        backend.set.assert_called_once_with("ref", "value")

    def test_set_writes_to_backend_by_index(self):
        backend0 = MagicMock(spec=CredentialBackend)
        backend1 = MagicMock(spec=CredentialBackend)
        store = CredentialStore(backends=[backend0, backend1])
        store.set("ref", "value", backend_index=1)
        backend0.set.assert_not_called()
        backend1.set.assert_called_once_with("ref", "value")

    def test_delete_deletes_from_all_backends(self):
        backend0 = MagicMock(spec=CredentialBackend)
        backend1 = MagicMock(spec=CredentialBackend)
        store = CredentialStore(backends=[backend0, backend1])
        store.delete("ref")
        backend0.delete.assert_called_once_with("ref")
        backend1.delete.assert_called_once_with("ref")

    def test_resolve_raises_on_missing_auth_ref(self, tmp_path):
        backend = MagicMock(spec=CredentialBackend)
        backend.get.return_value = None
        store = CredentialStore(backends=[backend])
        profile = _make_profile()
        with pytest.raises(CredentialNotFoundError, match=VALID_UUID):
            store.resolve(profile)

    def test_resolve_returns_key(self, tmp_path):
        backend = MagicMock(spec=CredentialBackend)
        backend.get.return_value = "my-api-key"
        store = CredentialStore(backends=[backend])
        profile = _make_profile()
        assert store.resolve(profile) == "my-api-key"


class TestFileBackend:
    def test_crud_round_trip(self, tmp_path):
        backend = FileBackend(path=tmp_path / "creds.json")
        backend.set("ref1", "my-secret-key")
        assert backend.get("ref1") == "my-secret-key"
        backend.delete("ref1")
        assert backend.get("ref1") is None

    def test_handles_missing_file(self, tmp_path):
        backend = FileBackend(path=tmp_path / "nonexistent.json")
        assert backend.get("any-ref") is None

    def test_handles_corrupt_json(self, tmp_path):
        path = tmp_path / "creds.json"
        path.write_text("{invalid")
        backend = FileBackend(path=path)
        assert backend.get("any-ref") is None

    def test_absent_ref_returns_none_when_other_refs_exist(self, tmp_path):
        """An absent ref reads as ``None`` even beside stored ones — the "no credential" half."""
        backend = FileBackend(path=tmp_path / "creds.json")
        backend.set("ref1", "my-secret-key")
        assert backend.get("missing-ref") is None

    def test_explicit_json_null_reads_as_absent(self, tmp_path):
        """A hand-written ``null`` value is the absent spelling, not corruption.

        ``set()`` never writes null and ``data.get(ref)`` returns ``None`` for it,
        so it takes the absent branch by construction. Pinned so a future refactor
        of the absent-check cannot silently reclassify it (KBR-87 review round 1).
        """
        path = tmp_path / "creds.json"
        path.write_text(json.dumps({"ref1": None}), encoding="utf-8")
        backend = FileBackend(path=path)
        assert backend.get("ref1") is None

    @pytest.mark.parametrize(
        "raw",
        [
            "@@@ not base64 @@@",  # not valid base64 (validate=True rejects the alphabet)
            "mötley-key",  # a non-ASCII string — rejected before any alphabet check
            base64.b64encode(b"\xff\xfe\xfa").decode("ascii"),  # valid base64, invalid UTF-8
            123,  # a hand-edited non-string value
        ],
    )
    def test_corrupt_stored_value_raises_credential_error(self, tmp_path, raw):
        """A present-but-undecodable value raises ``CredentialError`` naming the ref.

        The KBR-87 contract (SYSTEM_DESIGN.md §11.2): ``None`` means absent;
        ``CredentialError`` means present-but-undecodable. Before the contract,
        every one of these shapes silently read as ``None`` and surfaced to the
        user as "no API key for profile X" — store damage disguised as a missing
        key (KBR-154 diagnostic family).
        """
        path = tmp_path / "creds.json"
        path.write_text(json.dumps({"ref1": raw}), encoding="utf-8")
        backend = FileBackend(path=path)

        with pytest.raises(CredentialError, match="ref1"):
            backend.get("ref1")

    def test_validate_true_rejects_what_validate_false_would_silently_decode(self, tmp_path):
        """The load-bearing arm of the ``validate=True`` choice.

        Under ``validate=False``, ``"ab@=="`` strips the non-alphabet ``@`` to the
        length-valid ``"ab=="`` and silently decodes to ``"i"`` — corruption read
        back as a plausible single-character credential. Under ``validate=True``
        (the production choice) it rejects outright. This is the falsification
        the other corruption arms cannot provide: those raise under both modes
        (length, non-ASCII, UTF-8, and type errors are validate-independent).
        """
        path = tmp_path / "creds.json"
        path.write_text(json.dumps({"ref1": "ab@=="}), encoding="utf-8")
        backend = FileBackend(path=path)

        with pytest.raises(CredentialError, match="ref1"):
            backend.get("ref1")

        # The negative control for the falsification itself: what validate=True
        # rejects is exactly what validate=False would have accepted as a value.
        assert base64.b64decode("ab@==", validate=False) == b"i"

    def test_corrupt_ref_does_not_disturb_other_refs(self, tmp_path):
        """Corruption is per-ref: a damaged ref1 leaves a valid ref2 readable."""
        path = tmp_path / "creds.json"
        path.write_text(
            json.dumps({"ref1": "@@@ not base64 @@@", "ref2": base64.b64encode(b"fine").decode("ascii")}),
            encoding="utf-8",
        )
        backend = FileBackend(path=path)

        assert backend.get("ref2") == "fine"
        with pytest.raises(CredentialError, match="ref1"):
            backend.get("ref1")


class TestKeyringBackend:
    def test_delegates_to_keyring(self, monkeypatch):
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = "stored-key"
        monkeypatch.setattr("kitty.credentials.keyring_backend.keyring", mock_keyring)

        from kitty.credentials.keyring_backend import KeyringBackend

        backend = KeyringBackend()
        assert backend.get("ref") == "stored-key"
        mock_keyring.get_password.assert_called_once_with("kitty", "ref")

    def test_set_calls_keyring(self, monkeypatch):
        mock_keyring = MagicMock()
        monkeypatch.setattr("kitty.credentials.keyring_backend.keyring", mock_keyring)

        from kitty.credentials.keyring_backend import KeyringBackend

        backend = KeyringBackend()
        backend.set("ref", "value")
        mock_keyring.set_password.assert_called_once_with("kitty", "ref", "value")

    def test_delete_calls_keyring(self, monkeypatch):
        mock_keyring = MagicMock()
        monkeypatch.setattr("kitty.credentials.keyring_backend.keyring", mock_keyring)

        from kitty.credentials.keyring_backend import KeyringBackend

        backend = KeyringBackend()
        backend.delete("ref")
        mock_keyring.delete_password.assert_called_once_with("kitty", "ref")
