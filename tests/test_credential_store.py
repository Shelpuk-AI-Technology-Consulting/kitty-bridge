"""Tests for credentials/store.py, keyring_backend.py, file_backend.py."""

import base64
import json
import logging
import re
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


# ── KBR-291: file-level corruption shapes the F37 path did not cover ────
#
# Two residuals SYSTEM_DESIGN.md §11.1 recorded as "not fixed" after
# KBR-87 closed the per-ref half. Both raise at the same boundary
# (`FileBackend.get`) per-ref corruption already raises from, so the
# KBR-87 receiver map (§11.2) produces the existing clean message at
# every call site — a damaged file is reported as damage, not as absence
# (the F37 message, misleading) and not as a traceback (the shape (b)
# crash).


class TestFileLevelCorruption:
    """KBR-291: `FileBackend.get` surfaces file-level damage as `CredentialError`.

    Two residuals SYSTEM_DESIGN.md §11.1 recorded as "not fixed" after
    KBR-87 closed the per-ref half: the file's bytes are not valid UTF-8
    (shape b — previously raised `UnicodeDecodeError` uncaught on every
    launch), and its JSON parses to a non-dict (shape a — previously
    silently returned `{}` so the next `set` destroyed the original).
    Both now raise at the same boundary (`FileBackend.get`) per-ref
    corruption already raises from, so the KBR-87 receiver map (§11.2)
    produces the existing clean message at every call site.
    """

    def test_invalid_utf8_bytes_raise_credential_error(self, tmp_path, caplog):
        """Shape (b): bytes that are not valid UTF-8 raise `CredentialError`.

        The bytes are constructed literally (KBR-154 AC3, verbatim) — a
        byte sequence no host locale would decode, so the test does not
        depend on `LANG`/`LC_ALL`. The CRITICAL log line names both the
        file path and the backup path (AC7); the exception message
        matches the file path so the per-ref template ("Credential for
        ref …") cannot be silently reused (AC8).
        """
        path = tmp_path / "creds.json"
        path.write_bytes(b"\xff\xfe\xfd")  # invalid UTF-8 start bytes
        backend = FileBackend(path=path)

        with (
            caplog.at_level(logging.CRITICAL, logger="kitty.credentials.file_backend"),
            pytest.raises(CredentialError, match=re.escape(str(path))),
        ):
            backend.get("any-ref")

        critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert critical, "Expected CRITICAL log for non-UTF-8 credentials file"
        message = critical[-1].getMessage()
        assert str(path) in message, f"CRITICAL log missing path: {message!r}"
        backups = list(tmp_path.glob("creds.json.corrupt.*"))
        assert backups, "Expected a backup file at credentials.json.corrupt.*"
        assert str(backups[0]) in message, f"CRITICAL log missing backup path: {message!r}"

    def test_invalid_utf8_backup_preserves_the_original_bytes(self, tmp_path):
        """Shape (b): the backup carries the original (non-decodable) bytes."""
        path = tmp_path / "creds.json"
        original = b"\xff\xfe\xfd not utf-8"
        path.write_bytes(original)
        backend = FileBackend(path=path)

        with pytest.raises(CredentialError):
            backend.get("any-ref")

        backups = list(tmp_path.glob("creds.json.corrupt.*"))
        assert len(backups) == 1, f"Expected exactly 1 backup, got {backups}"
        assert backups[0].read_bytes() == original

    def test_invalid_utf8_then_set_does_not_lose_the_backup(self, tmp_path):
        """Shape (b): after the damage event, `set` starts fresh and the
        backup stays untouched — the recovered record."""
        path = tmp_path / "creds.json"
        original = b"\xff\xfe\xfd"
        path.write_bytes(original)
        backend = FileBackend(path=path)

        with pytest.raises(CredentialError):
            backend.get("any-ref")

        backend.set("new-ref", "new-value")
        backups = list(tmp_path.glob("creds.json.corrupt.*"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == original
        assert backend.get("new-ref") == "new-value"

    @pytest.mark.parametrize(
        "payload",
        [
            b"\xff\xfe\xfd",  # shape (b) — invalid UTF-8
            b'["not", "a", "dict"]',  # shape (a) — non-dict JSON
        ],
    )
    def test_write_path_forgives_without_a_preceding_get(self, tmp_path, caplog, payload):
        """D6 pin: `set` swallows the file-level raise via
        ``_read_raw_for_write`` so the recovery command stays reachable.

        Critically, no ``backend.get(...)`` runs first — without D6, this
        test would fail with ``CredentialError`` on the ``set`` itself,
        re-creating the KBR-154 diagnostic family this ticket exists to
        eliminate. The preceding-``get`` variants in
        ``test_invalid_utf8_then_set_does_not_lose_the_backup`` and
        ``test_valid_json_non_dict_then_set_preserves_the_backup`` pass
        even if ``_read_raw_for_write`` were removed, because the first
        ``get`` already resets the file to ``{}``; this test is the one
        that fails without D6.
        """
        path = tmp_path / "creds.json"
        path.write_bytes(payload)
        backend = FileBackend(path=path)

        with caplog.at_level(logging.CRITICAL, logger="kitty.credentials.file_backend"):
            backend.set("new-ref", "new-value")  # must not raise

        # The CRITICAL log + backup fired from the write path's first
        # ``_read_raw`` before the raise was swallowed.
        backups = list(tmp_path.glob("creds.json.corrupt.*"))
        assert len(backups) == 1, "Write path must back up the corrupt file"
        assert backups[0].read_bytes() == payload
        assert any(r.levelno >= logging.CRITICAL for r in caplog.records), (
            "Write path must log CRITICAL even though it swallowed the raise"
        )
        # The write succeeded — the new ref reads back.
        assert backend.get("new-ref") == "new-value"

    @pytest.mark.parametrize(
        "payload",
        [
            b'["not", "a", "dict"]',  # top-level list
            b'"just a string"',  # top-level string
            b"42",  # top-level number
            b"true",  # top-level bool
            b"null",  # top-level null
        ],
    )
    def test_valid_json_non_dict_raises_credential_error(self, tmp_path, caplog, payload):
        """Shape (a): JSON that parses to a non-dict raises `CredentialError`.

        Before KBR-291 this returned `{}` silently — no backup, no log
        — and the next `set` overwrote the file with no trace of the
        original. The CRITICAL log names both the file path and the
        backup path (AC7); the exception message matches the file path
        so the per-ref template ("Credential for ref …") cannot be
        silently reused (AC8).
        """
        path = tmp_path / "creds.json"
        path.write_bytes(payload)
        backend = FileBackend(path=path)

        with (
            caplog.at_level(logging.CRITICAL, logger="kitty.credentials.file_backend"),
            pytest.raises(CredentialError, match=re.escape(str(path))),
        ):
            backend.get("any-ref")

        critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert critical, "Expected CRITICAL log for non-dict JSON credentials file"
        message = critical[-1].getMessage()
        assert str(path) in message, f"CRITICAL log missing path: {message!r}"
        backups = list(tmp_path.glob("creds.json.corrupt.*"))
        assert backups, "Expected a backup file at credentials.json.corrupt.*"
        assert str(backups[0]) in message, f"CRITICAL log missing backup path: {message!r}"

    def test_valid_json_non_dict_backup_preserves_the_original(self, tmp_path):
        """Shape (a): the backup carries the original non-dict payload."""
        path = tmp_path / "creds.json"
        payload = b'["not", "a", "dict"]'
        path.write_bytes(payload)
        backend = FileBackend(path=path)

        with pytest.raises(CredentialError):
            backend.get("any-ref")

        backups = list(tmp_path.glob("creds.json.corrupt.*"))
        assert len(backups) == 1, f"Expected exactly 1 backup, got {backups}"
        assert backups[0].read_bytes() == payload

    def test_valid_json_non_dict_then_set_preserves_the_backup(self, tmp_path):
        """Shape (a): after the damage event, `set` starts fresh and the
        backup stays untouched — the recovered record.

        This is the acceptance-criterion-2 arm: without the backup, the
        next `set` would have silently overwritten the original (the
        §11.1 recorded residual).
        """
        path = tmp_path / "creds.json"
        payload = b'["not", "a", "dict"]'
        path.write_bytes(payload)
        backend = FileBackend(path=path)

        with pytest.raises(CredentialError):
            backend.get("any-ref")

        backend.set("new-ref", "new-value")
        backups = list(tmp_path.glob("creds.json.corrupt.*"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == payload
        assert backend.get("new-ref") == "new-value"

    def test_error_chains_from_the_underlying_cause(self, tmp_path):
        """Shape (b) chains (`raise ... from`) so `__cause__` preserves
        the underlying `UnicodeDecodeError` for any consumer that
        introspects it (AC5a).
        """
        path = tmp_path / "creds.json"
        path.write_bytes(b"\xff\xfe\xfd")
        backend = FileBackend(path=path)

        with pytest.raises(CredentialError) as excinfo:
            backend.get("any-ref")
        assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)

    def test_error_does_not_chain_for_shape_a(self, tmp_path):
        """Shape (a) does **not** chain (AC5b): the JSON parsed cleanly,
        so there is no underlying exception to preserve. `__cause__` is
        `None`; a regression that adds `from json.JSONDecodeError` here
        would attach an unrelated exception to a non-error condition.
        """
        path = tmp_path / "creds.json"
        path.write_bytes(b'["not", "a", "dict"]')
        backend = FileBackend(path=path)

        with pytest.raises(CredentialError) as excinfo:
            backend.get("any-ref")
        assert excinfo.value.__cause__ is None

    @pytest.mark.parametrize(
        "payload",
        [
            b"\xff\xfe\xfd",  # shape (b)
            b'["not", "a", "dict"]',  # shape (a)
        ],
    )
    def test_delete_does_not_raise_after_file_level_damage(self, tmp_path, payload):
        """The write path forgives file-level damage (D6) — `delete`
        succeeds so the recovery command (`kitty setup`) does not crash
        on its own write. The CRITICAL log + backup still fire from
        `_read_raw`; nothing is silent.
        """
        path = tmp_path / "creds.json"
        path.write_bytes(payload)
        backend = FileBackend(path=path)

        # The damage event itself surfaces to `get` (read path) —
        # verify before testing the write-path forgiveness.
        with pytest.raises(CredentialError):
            backend.get("any-ref")

        backend.delete("any-ref")  # must not raise
        backups = list(tmp_path.glob("creds.json.corrupt.*"))
        assert len(backups) == 1, "Backup was not preserved across the delete"

    def test_f37_invalid_json_path_is_unchanged(self, tmp_path):
        """F37 regression pin (acceptance criterion 3): invalid JSON keeps
        the F37-original behaviour — `get` returns `None`, the file is
        reset to `{}`, no raise. The new shapes are additive, not a
        regression on the JSON path."""
        path = tmp_path / "creds.json"
        path.write_text("this is not json {{{", encoding="utf-8")
        backend = FileBackend(path=path)

        assert backend.get("any-ref") is None
        assert path.read_text(encoding="utf-8").strip() == "{}"

    @pytest.mark.parametrize(
        "payload",
        [
            b"\xff\xfe\xfd",  # shape (b) — invalid UTF-8
            b'["not", "a", "dict"]',  # shape (a) — non-dict JSON
        ],
    )
    def test_write_path_propagates_when_backup_could_not_be_made(self, tmp_path, monkeypatch, payload):
        """Pin the read-only-mount fallback: when ``os.replace`` fails
        inside ``_read_raw``, the file remains at ``self._path`` and
        the user was promised the original was preserved at the
        backup path. Letting the write path swallow the raise would
        overwrite the still-damaged original with no backup anywhere
        — silent credential loss with a false promise.

        The fix in ``_read_raw_for_write`` re-raises when
        ``self._path`` still exists after the damage event; this test
        pins that contract by simulating ``os.replace`` failure.
        """
        path = tmp_path / "creds.json"
        path.write_bytes(payload)
        backend = FileBackend(path=path)

        # Simulate a read-only filesystem: os.replace raises OSError
        # because the rename onto the backup path fails. The
        # credentials file stays at self._path (still damaged).
        def _raise_oserror(src: object, dst: object) -> None:
            raise OSError("simulated read-only mount")

        monkeypatch.setattr("kitty.credentials.file_backend.os.replace", _raise_oserror)

        # `get` still raises (the read-path signal is intact).
        with pytest.raises(CredentialError, match="could not be backed up"):
            backend.get("any-ref")

        # The damaged file is still at the path — the message claims
        # it was preserved at a backup that does not exist.
        assert path.exists(), "Expected damaged file to remain at path when backup fails"

        # `set` MUST propagate the error too — the original would be
        # silently overwritten without a backup otherwise. Without the
        # `_read_raw_for_write` self._path.exists() guard, this would
        # succeed and silently destroy the damaged file's bytes.
        with pytest.raises(CredentialError):
            backend.set("new-ref", "new-value")

        # The damaged file is still there — the user can recover by
        # hand once write access is restored.
        assert path.exists()
        assert path.read_bytes() == payload

    @pytest.mark.parametrize(
        "payload",
        [
            b'["not", "a", "dict"]',  # shape (a) — non-dict JSON
            b"\xff\xfe\xfd",  # shape (b) — invalid UTF-8
            b"not json at all {{{",  # F37 — invalid JSON (deliberate unchanged path)
        ],
    )
    def test_success_branch_message_names_the_backup(self, tmp_path, payload):
        """The success-branch message claims 'the original is preserved
        at {backup}' — verify it names the real backup path and that the
        backup actually exists at that path (no false promise).

        The F37 case here hits the success branch (backup succeeds), so
        ``get`` still returns ``None`` (acceptance criterion 3 — F37 is
        unchanged in the normal case).
        """
        path = tmp_path / "creds.json"
        path.write_bytes(payload)
        backend = FileBackend(path=path)

        if payload == b"not json at all {{{":
            # F37's success branch returns None, not raise.
            assert backend.get("any-ref") is None
        else:
            with pytest.raises(CredentialError) as excinfo:
                backend.get("any-ref")
            message = str(excinfo.value)
            backups = list(tmp_path.glob("creds.json.corrupt.*"))
            assert backups, "Expected backup to exist"
            assert str(backups[0]) in message, f"Success-branch message must name the backup path: {message!r}"
            assert "could not be backed up" not in message

    def test_f37_backup_failure_now_raises_instead_of_silent_reset(self, tmp_path, monkeypatch, caplog):
        """F37 regression-plus-fix: invalid JSON with a FAILED backup
        must raise `CredentialError` (honest message), not silently
        write `{}` over the damaged original.

        The acceptance criterion 3 contract — F37 behaves exactly as
        today — holds for the normal (backup-succeeds) case, which
        `test_f37_invalid_json_path_is_unchanged` and the stage-7 suite
        pin. This test pins the round-3 fix for the failure branch:
        when the rename fails, `_write_raw({})` would overwrite the
        still-damaged original with no backup anywhere; the raise is
        the only honest signal.
        """
        path = tmp_path / "creds.json"
        original = b"not json at all {{{"
        path.write_bytes(original)
        backend = FileBackend(path=path)

        def _raise_oserror(src: object, dst: object) -> None:
            raise OSError("simulated read-only mount")

        monkeypatch.setattr("kitty.credentials.file_backend.os.replace", _raise_oserror)

        with (
            caplog.at_level(logging.CRITICAL, logger="kitty.credentials.file_backend"),
            pytest.raises(CredentialError, match="could not be backed up"),
        ):
            backend.get("any-ref")

        # The damaged original survived — no silent overwrite.
        assert path.read_bytes() == original
        assert any(r.levelno >= logging.CRITICAL for r in caplog.records), (
            "Expected CRITICAL log even in the backup-failed branch"
        )
