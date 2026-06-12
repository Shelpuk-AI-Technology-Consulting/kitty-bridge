"""Stage 7 tests — credential backend fixes (F37, F39).

F37: Corrupt credentials.json should be backed up before being reset.
F39: keyring_backend.set() must catch exceptions and raise CredentialError.
"""

from __future__ import annotations

import logging

import pytest

from kitty.credentials.file_backend import FileBackend

# ── F37: Corrupt credentials.json recovery ──────────────────────────────


class TestCorruptCredentialsRecovery:
    """F37: When credentials.json is corrupt, back it up before returning {}."""

    def test_corrupt_json_creates_backup_file(self, tmp_path, caplog):
        """A corrupt credentials.json should be renamed with a timestamp suffix."""
        creds_path = tmp_path / "credentials.json"
        creds_path.write_text("this is not json {{{", encoding="utf-8")

        with caplog.at_level(logging.CRITICAL, logger="kitty.credentials.file_backend"):
            backend = FileBackend(creds_path)
        # Trigger a read — _read_raw is lazy, called by get()
        result = backend.get("any-ref")
        assert result is None  # corrupt file returns empty

        # Backup file should exist
        backup_files = list(tmp_path.glob("credentials.json.corrupt.*"))
        assert len(backup_files) == 1, f"Expected exactly 1 backup, got {backup_files}"

        # The backup should contain the original corrupt content
        backup_content = backup_files[0].read_text(encoding="utf-8")
        assert backup_content == "this is not json {{{"

        # The original file should be empty (newly created)
        assert creds_path.read_text(encoding="utf-8").strip() == "{}"

        # A CRITICAL log should be emitted
        assert any(r.levelno >= logging.CRITICAL for r in caplog.records), (
            "Expected CRITICAL log for corrupt credentials file"
        )

    def test_corrupt_then_set_preserves_original(self, tmp_path):
        """After corruption is detected, the backup still has the original content
        even if the new backend overwrites credentials.json."""
        creds_path = tmp_path / "credentials.json"
        creds_path.write_text("corrupt{{{", encoding="utf-8")

        backend = FileBackend(creds_path)
        # Trigger a read — _read_raw is lazy
        _ = backend.get("any-ref")
        # backup file is created
        backup_files = list(tmp_path.glob("credentials.json.corrupt.*"))
        assert len(backup_files) == 1

        # Writing a new value should not affect the backup
        backend.set("new-ref", "new-value")
        backup_content = backup_files[0].read_text(encoding="utf-8")
        assert backup_content == "corrupt{{{"


# ── F39: keyring_backend set() error handling ────────────────────────────


class TestKeyringBackendSetErrorHandling:
    """F39: keyring.set_password() must catch exceptions and raise CredentialError."""

    def test_keyring_set_raises_credential_error_on_failure(self, monkeypatch):
        import keyring

        from kitty.credentials import CredentialError
        from kitty.credentials.keyring_backend import KeyringBackend

        def boom(service: str, username: str, password: str) -> None:
            raise OSError("no D-Bus session")

        monkeypatch.setattr(keyring, "set_password", boom)

        backend = KeyringBackend()
        with pytest.raises(CredentialError, match="keyring"):
            backend.set("my-ref", "secret-value")
