"""Tests for credentials/store.py, keyring_backend.py, file_backend.py."""

import uuid
from unittest.mock import MagicMock

import pytest

from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialBackend, CredentialNotFoundError, CredentialStore
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
