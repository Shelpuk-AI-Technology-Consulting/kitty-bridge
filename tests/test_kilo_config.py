"""Tests for KiloAdapter prepare_launch / cleanup_launch config file patching."""

import json
import logging
import pathlib
import sys
import uuid

import pytest

from kitty.launchers import kilo as kilo_mod
from kitty.launchers.kilo import KiloAdapter
from kitty.profiles.schema import Profile


def _make_profile(**overrides):
    defaults = {
        "name": "test",
        "provider": "openrouter",
        "model": "test-model",
        "auth_ref": str(uuid.uuid4()),
    }
    defaults.update(overrides)
    return Profile(**defaults)


@pytest.fixture()
def adapter():
    return KiloAdapter()


@pytest.fixture()
def config_dir(tmp_path):
    """Return a temporary config directory path."""
    d = tmp_path / "kilo-config"
    d.mkdir()
    return d


@pytest.fixture()
def config_path(config_dir):
    return config_dir / "kilo.json"


@pytest.fixture(autouse=True)
def backup_path_redirect(tmp_path, monkeypatch):
    """Redirect the crash-recovery backup path to the test's temp dir.

    Autouse so no test in this file can touch the developer's real
    ``~/.config/kitty/kilo-config-backup.json``; the trio resolves the module
    attribute lazily at call time, so patching it here redirects every writer
    and reader.
    """
    monkeypatch.setattr(kilo_mod, "_DEFAULT_BACKUP_PATH", tmp_path / "kitty-backups" / "kilo-config-backup.json")


class TestPrepareLaunch:
    def test_creates_config_when_none_exists(self, adapter, config_path):
        adapter.build_spawn_config(_make_profile(), 18080, "my-api-key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        assert original is None  # No previous file
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert "kitty" in data["provider"]

    def test_config_contains_provider_with_base_url(self, adapter, config_path):
        adapter.build_spawn_config(_make_profile(), 18080, "my-api-key")
        adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        data = json.loads(config_path.read_text())
        kitty = data["provider"]["kitty"]
        assert kitty["options"]["baseURL"] == "http://127.0.0.1:18080/v1"

    def test_config_contains_provider_with_api_key(self, adapter, config_path):
        adapter.build_spawn_config(_make_profile(), 18080, "my-api-key")
        adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        data = json.loads(config_path.read_text())
        assert data["provider"]["kitty"]["options"]["apiKey"] == "my-api-key"

    def test_config_contains_model(self, adapter, config_path):
        adapter.build_spawn_config(_make_profile(model="gpt-4o"), 18080, "key")
        adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        data = json.loads(config_path.read_text())
        models = data["provider"]["kitty"]["models"]
        # Model key is the bare model name (no provider prefix)
        assert "gpt-4o" in models
        assert models["gpt-4o"]["id"] == "gpt-4o"
        # Active model uses provider/model format
        assert data["model"] == "kitty/gpt-4o"

    def test_patches_existing_config_preserving_other_providers(self, adapter, config_path):
        # Pre-existing config with another provider
        existing = {"provider": {"openai": {"options": {"apiKey": "existing-key"}}}}
        config_path.write_text(json.dumps(existing))

        adapter.build_spawn_config(_make_profile(), 18080, "my-key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        data = json.loads(config_path.read_text())
        assert "openai" in data["provider"]  # Preserved
        assert "kitty" in data["provider"]  # Added
        assert original is not None  # Saved original

    def test_returns_original_content(self, adapter, config_path):
        existing = {"provider": {"openai": {"options": {"apiKey": "sk-123"}}}}
        config_path.write_text(json.dumps(existing))

        adapter.build_spawn_config(_make_profile(), 18080, "key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        assert json.loads(original) == existing

    def test_handles_malformed_json(self, adapter, config_path):
        config_path.write_text("not valid json {{{")

        adapter.build_spawn_config(_make_profile(), 18080, "key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        # Should still create valid config, original saved as-is
        assert original == "not valid json {{{"
        data = json.loads(config_path.read_text())
        assert "kitty" in data["provider"]

    def test_config_has_npm_field(self, adapter, config_path):
        adapter.build_spawn_config(_make_profile(), 18080, "key")
        adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        data = json.loads(config_path.read_text())
        assert data["provider"]["kitty"]["npm"] == "@ai-sdk/openai-compatible"


class TestCleanupLaunch:
    def test_restores_original_config(self, adapter, config_path):
        existing = {"provider": {"openai": {"options": {"apiKey": "original"}}}}
        config_path.write_text(json.dumps(existing))

        adapter.build_spawn_config(_make_profile(), 18080, "key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        # Config now has kitty provider
        assert "kitty" in json.loads(config_path.read_text())["provider"]

        # Cleanup restores original
        adapter.cleanup_launch(original, settings_path=config_path)
        assert json.loads(config_path.read_text()) == existing

    def test_removes_temporary_config_when_no_original(self, adapter, config_path):
        adapter.build_spawn_config(_make_profile(), 18080, "key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        assert config_path.exists()
        adapter.cleanup_launch(original, settings_path=config_path)
        assert not config_path.exists()

    def test_cleanup_with_none_is_noop(self, adapter, config_path):
        # If prepare_launch was never called or returned None
        adapter.cleanup_launch(None, settings_path=config_path)
        assert not config_path.exists()


# ── KBR-262 — byte-exact kilo config restore across platforms ────────────────


class TestPrepareLaunchPreservesBytes:
    """`prepare_launch` must return the user's config bytes verbatim (KBR-262).

    The pre-fix reader used :meth:`pathlib.Path.read_text` (universal newlines),
    which silently strips ``\\r\\n`` on every platform — breaking the byte-identity
    contract that ``cleanup_launch``'s restore relies on for any CRLF config the
    user already has on disk.
    """

    def test_returns_original_preserving_crlf_bytes(self, adapter: KiloAdapter, config_path: pathlib.Path) -> None:
        """A CRLF kilo config is returned from ``prepare_launch`` verbatim (KBR-262).

        The config is staged with raw CRLF bytes (``Path.write_bytes``) rather
        than via ``json.dumps`` because ``write_text`` would itself normalise
        the bytes away from the underlying defect under test.

        Args:
            adapter: The Kilo adapter under test.
            config_path: The kilo config path fixture (per-test temp dir).
        """
        crlf_bytes = (
            b'{\r\n  "provider": {"openai": {"apiKey": "sk-original"}},\r\n'
            b'  "model": "opus",\r\n'
            b'  "comment": "user kept CRLF line endings"\r\n'
            b"}\r\n"
        )

        config_path.write_bytes(crlf_bytes)

        adapter.build_spawn_config(_make_profile(), 18080, "key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        assert original is not None, "prepare_launch saved no original for an existing CRLF config"
        assert original.encode("utf-8") == crlf_bytes, (
            f"prepare_launch returned bytes {original.encode('utf-8')!r}; "
            f"expected {crlf_bytes!r} — universal-newlines read defect"
        )


class TestCleanupLaunchPreservesBytes:
    """`cleanup_launch` must write the original config byte-exactly (KBR-262).

    The pre-fix writer left :meth:`pathlib.Path.write_text`'s open call at the
    default ``newline=None``, which makes CPython translate ``\\n`` to
    ``os.linesep`` on write. On Windows that is ``\\r\\n`` — silently corrupting
    any LF original into CRLF.
    """

    def test_calls_write_text_with_newline_empty(
        self,
        adapter: KiloAdapter,
        config_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`cleanup_launch` opens the destination with ``newline=""`` (KBR-262).

        The write-side defect is invisible behaviourally on POSIX because
        ``os.linesep == "\\n"`` there (``LF`` translation is identity). The
        only POSIX-runnable regression guard is the open call's kwargs —
        behavioural coverage lives in the Windows-only tests below.

        The current kilo.json is staged with kitty markers so the KBR-268
        ownership matrix reaches the owned-write branch.

        Args:
            adapter: The Kilo adapter under test.
            config_path: The kilo config path fixture (per-test temp dir).
            monkeypatch: Pytest's monkeypatch fixture (auto-restores).
        """
        config_path.write_text(
            json.dumps(
                {
                    "provider": {
                        "kitty": {"options": {"baseURL": "http://127.0.0.1:9/v1"}},
                    },
                    "model": "kitty/gpt-4o",
                }
            )
        )

        captured: list[dict[str, object]] = []
        real_open = pathlib.Path.open

        def recording_open(self: pathlib.Path, *args: object, **kwargs: object) -> object:
            captured.append(kwargs)
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "open", recording_open)

        original = '{"model": "opus"}\r\n{"foo": 1}\r\n'
        adapter.cleanup_launch(original, settings_path=config_path)

        write_opens = [k for k in captured if k.get("mode") == "w"]
        assert len(write_opens) == 1, (
            f"expected exactly one write-mode Path.open call from "
            f"cleanup_launch, got {len(write_opens)} in {captured!r}"
        )
        assert write_opens[0].get("newline") == "", (
            f"cleanup_launch opened the restore write without newline=''; write open kwargs were {write_opens[0]!r}"
        )

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason=(
            "write-side CRLF translation only fires on Windows; CI covers it on the Fast gate's Windows leg (KBR-164)"
        ),
    )
    @pytest.mark.parametrize(
        "original",
        [
            '{"model": "opus"}\n{"foo": 1}\n',
            '{"model": "opus"}\r\n{"foo": 1}\r\n',
        ],
        ids=["lf", "crlf"],
    )
    def test_writes_byte_exactly_on_windows(
        self,
        adapter: KiloAdapter,
        config_path: pathlib.Path,
        original: str,
    ) -> None:
        """On Windows, ``cleanup_launch`` writes the original byte-exactly (KBR-262).

        CI-verified on the Fast gate's Windows leg. Parametrised over LF and
        CRLF originals so each fact (LF must stay LF; CRLF must stay CRLF)
        lives in its own test ID — a future single-leg regression cannot hide
        behind a green other-leg pass.

        Current kilo.json staged with kitty markers so the KBR-268 ownership
        matrix reaches the owned-write branch.

        Args:
            adapter: The Kilo adapter under test.
            config_path: The kilo config path fixture (per-test temp dir).
            original: The text the user originally had on disk — parametrised.
        """
        config_path.write_text(
            json.dumps(
                {
                    "provider": {
                        "kitty": {"options": {"baseURL": "http://127.0.0.1:9/v1"}},
                    },
                    "model": "kitty/gpt-4o",
                }
            )
        )
        adapter.cleanup_launch(original, settings_path=config_path)

        assert config_path.read_bytes() == original.encode("utf-8"), (
            f"cleanup_launch wrote {config_path.read_bytes()!r}; expected {original.encode('utf-8')!r}"
        )


# ── KBR-268 — Kilo crash-recovery backup trio + detector ─────────────────────


class TestKiloConfigBackupHelpers:
    """The save/load/delete trio mirrors ``claude.py`` and must round-trip the
    user's config bytes byte-exactly, including CRLF (KBR-262 contract).
    """

    def test_save_then_load_round_trips_bytes(self, tmp_path) -> None:
        """A save/load pair returns the original bytes verbatim."""
        backup_path = tmp_path / "backup.json"
        content = '{"model":"opus"}\r\n{"provider": {"x": 1}}\r\n'

        kilo_mod.save_kilo_config_backup(content, backup_path=backup_path)

        assert backup_path.read_bytes() == content.encode("utf-8")
        assert kilo_mod.load_kilo_config_backup(backup_path=backup_path) == content

    def test_load_returns_none_when_missing(self, tmp_path) -> None:
        """A missing backup file loads as ``None`` so cleanup can no-op."""
        assert kilo_mod.load_kilo_config_backup(backup_path=tmp_path / "absent.json") is None

    def test_delete_is_idempotent(self, tmp_path) -> None:
        """Deleting an absent backup does not raise (unlink(missing_ok=True))."""
        backup_path = tmp_path / "absent.json"

        kilo_mod.delete_kilo_config_backup(backup_path=backup_path)  # no raise

    def test_save_failure_is_best_effort(self, tmp_path, caplog) -> None:
        """A failed backup write logs a warning and does not raise.

        The backup path's parent is a file, so ``mkdir(parents=True)`` fails;
        the trio catches and warns so ``prepare_launch`` can still complete.
        """
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        backup_path = blocker / "backup.json"  # parent is a file → mkdir raises

        with caplog.at_level(logging.WARNING, logger="kitty.launchers.kilo"):
            kilo_mod.save_kilo_config_backup("payload", backup_path=backup_path)

        assert any("save_kilo_config_backup" in rec.message for rec in caplog.records), (
            f"expected a save warning, got: {[r.message for r in caplog.records]}"
        )
        assert not backup_path.exists()


class TestKiloKittyValuesPresent:
    """Detector used by ``prepare_launch`` (skip-polluted) and by
    ``run_kilo_cleanup`` (restore-on-marker) — gates the backup contract.
    """

    def test_provider_kitty_loopback_base_url_returns_true(self) -> None:
        """A ``provider.kitty`` block with a loopback baseURL is a kitty marker."""
        config = {
            "provider": {
                "kitty": {"options": {"baseURL": "http://127.0.0.1:18080/v1"}},
            },
        }
        assert kilo_mod._kilo_kitty_values_present(config) is True

    def test_provider_kitty_localhost_hostname_returns_true(self) -> None:
        """``localhost`` is loopback too — same detector path."""
        config = {
            "provider": {
                "kitty": {"options": {"baseURL": "http://localhost:8080/v1"}},
            },
        }
        assert kilo_mod._kilo_kitty_values_present(config) is True

    def test_model_kitty_prefix_returns_true(self) -> None:
        """A ``model: kitty/<x>`` value alone is a kitty marker (hand-trimmed crash)."""
        config = {"model": "kitty/gpt-4o"}
        assert kilo_mod._kilo_kitty_values_present(config) is True

    def test_provider_kitty_remote_base_url_only_returns_false(self) -> None:
        """A remote-URL ``provider.kitty`` alone is not a kitty marker — a user
        who named their own provider ``kitty`` must not have their config
        auto-restored (Claude parity: loopback URL is the unambiguous signal).
        """
        config = {
            "provider": {
                "kitty": {"options": {"baseURL": "https://api.openai.com/v1"}},
            },
        }
        assert kilo_mod._kilo_kitty_values_present(config) is False

    def test_no_markers_returns_false(self) -> None:
        """An unrelated config returns ``False``."""
        config = {
            "provider": {"openai": {"options": {"apiKey": "sk-original"}}},
            "model": "opus",
        }
        assert kilo_mod._kilo_kitty_values_present(config) is False

    def test_non_dict_returns_false(self) -> None:
        """A JSON array, scalar, or ``None`` is never a kitty marker."""
        for value in ([], None, "string", 42):
            assert kilo_mod._kilo_kitty_values_present(value) is False

    def test_empty_dict_returns_false(self) -> None:
        """An empty config object returns ``False`` (no provider, no model)."""
        assert kilo_mod._kilo_kitty_values_present({}) is False


class TestPrepareLaunchBackup:
    """``prepare_launch`` writes the backup **only** when the captured original
    is clean — the clean-capture rule that prevents the crash-then-relaunch
    clobbering of the true backup (KBR-268 BLOCKER 1).
    """

    def test_writes_backup_byte_identical_for_clean_config(
        self, adapter: KiloAdapter, config_path: pathlib.Path
    ) -> None:
        """A clean existing config gets a byte-exact backup before patching (AC-R1a)."""
        existing = {"provider": {"openai": {"options": {"apiKey": "sk-original"}}}}
        config_path.write_text(json.dumps(existing))

        adapter.build_spawn_config(_make_profile(), 18080, "key")
        adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        backup_path = kilo_mod._DEFAULT_BACKUP_PATH
        assert backup_path.exists(), "prepare_launch did not write the backup"
        assert backup_path.read_text(encoding="utf-8") == json.dumps(existing)

    def test_prepare_launch_succeeds_when_backup_write_raises(
        self,
        adapter: KiloAdapter,
        config_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failing backup write never blocks the launch (AC-R1a, end to end).

        The seam is ``kilo._atomic_write_text`` (kilo's own module binding —
        the trio's OSError swallow then runs for real), so this probe drives
        ``prepare_launch`` → trio → failing write and asserts the launch
        still completes with the config patched.

        Args:
            adapter: The Kilo adapter under test.
            config_path: The kilo config path fixture (per-test temp dir).
            monkeypatch: Pytest's monkeypatch fixture (auto-restores).
        """

        def raising_write(path: pathlib.Path, content: str) -> None:
            raise OSError("simulated backup write failure")

        monkeypatch.setattr(kilo_mod, "_atomic_write_text", raising_write)

        existing = {"provider": {"openai": {"options": {"apiKey": "sk-original"}}}}
        config_path.write_text(json.dumps(existing))

        adapter.build_spawn_config(_make_profile(), 18080, "key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        assert original is not None, "prepare_launch failed on a backup write error"
        assert "kitty" in json.loads(config_path.read_text())["provider"], (
            "prepare_launch did not patch the config despite the backup failure"
        )
        assert not kilo_mod._DEFAULT_BACKUP_PATH.exists(), (
            "a failed backup write must not leave a partial backup behind"
        )

    def test_skips_backup_and_warns_when_original_has_markers(
        self,
        adapter: KiloAdapter,
        config_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A marker-bearing captured original does NOT overwrite a pre-existing
        backup (AC-R1b). The true backup survives; ``prepare_launch`` warns.
        """
        # Pre-existing clean backup (the "true original" we must preserve).
        backup_path = kilo_mod._DEFAULT_BACKUP_PATH
        true_original = '{"provider":{"openai":{"options":{"apiKey":"true-original"}}}}'
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(true_original)

        # Current kilo.json carries crashed-session markers.
        patched = {
            "provider": {
                "kitty": {"options": {"baseURL": "http://127.0.0.1:9/v1"}},
            },
            "model": "kitty/gpt-4o",
        }
        config_path.write_text(json.dumps(patched))

        with caplog.at_level(logging.WARNING, logger="kitty.launchers.kilo"):
            adapter.build_spawn_config(_make_profile(), 18080, "key")
            adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        assert backup_path.read_text(encoding="utf-8") == true_original, (
            "prepare_launch overwrote the true backup with polluted content"
        )
        assert any("kitty cleanup" in rec.message.lower() for rec in caplog.records), (
            f"expected a kitty-cleanup warning, got: {[r.message for r in caplog.records]}"
        )

    def test_markers_present_without_backup_writes_no_backup(
        self,
        adapter: KiloAdapter,
        config_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Marker-bearing original with no prior backup still writes no backup
        (AC-R1c) — backing up polluted content would make a later cleanup
        ``restore`` the damage.
        """
        patched = {
            "provider": {
                "kitty": {"options": {"baseURL": "http://127.0.0.1:9/v1"}},
            },
            "model": "kitty/gpt-4o",
        }
        config_path.write_text(json.dumps(patched))

        with caplog.at_level(logging.WARNING, logger="kitty.launchers.kilo"):
            adapter.build_spawn_config(_make_profile(), 18080, "key")
            adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        assert not kilo_mod._DEFAULT_BACKUP_PATH.exists()
        assert any("kitty cleanup" in rec.message.lower() for rec in caplog.records)

    def test_no_backup_when_config_did_not_exist(self, adapter: KiloAdapter, config_path: pathlib.Path) -> None:
        """From-scratch sessions write no backup (AC-R2)."""
        assert not config_path.exists()

        adapter.build_spawn_config(_make_profile(), 18080, "key")
        adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        assert not kilo_mod._DEFAULT_BACKUP_PATH.exists()


class TestCleanupLaunchOwnership:
    """``cleanup_launch`` restores + deletes the backup **only** when this
    session still owns the file (current markers present). Otherwise the file
    and the backup are left alone — KBR-268 BLOCKER 2 parity with Claude.
    """

    def _stage_and_patch(self, adapter: KiloAdapter, config_path: pathlib.Path) -> str:
        """Helper: stage a clean config, run prepare, return its captured original."""
        existing = {"provider": {"openai": {"options": {"apiKey": "original"}}}}
        config_path.write_text(json.dumps(existing))
        adapter.build_spawn_config(_make_profile(), 18080, "key")
        return adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

    def test_restores_and_deletes_backup_when_current_has_markers(
        self, adapter: KiloAdapter, config_path: pathlib.Path
    ) -> None:
        """Owned exit: clean original + marker-bearing current → restore + delete (AC-R3a)."""
        original = self._stage_and_patch(adapter, config_path)
        assert kilo_mod._DEFAULT_BACKUP_PATH.exists()

        adapter.cleanup_launch(original, settings_path=config_path)

        assert json.loads(config_path.read_text()) == {"provider": {"openai": {"options": {"apiKey": "original"}}}}, (
            "cleanup_launch did not restore the captured original"
        )
        assert not kilo_mod._DEFAULT_BACKUP_PATH.exists()

    def test_polluted_original_leaves_file_and_backup(
        self,
        adapter: KiloAdapter,
        config_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Captured original carries markers → polluted snapshot (concurrent
        session's patch). File and backup are both left alone (AC-R3b).
        """
        existing = {"provider": {"openai": {"options": {"apiKey": "original"}}}}
        config_path.write_text(json.dumps(existing))
        adapter.build_spawn_config(_make_profile(), 18080, "key")
        adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)

        # Another session's patch (markers) becomes this session's "original".
        polluted = (
            '{"provider":{"kitty":{"options":{"baseURL":"http://127.0.0.1:9/v1"}},'
            '"openai":{"options":{"apiKey":"A-session"}}},'
            '"model":"kitty/gpt-4o"}'
        )
        assert json.loads(polluted)["model"].startswith("kitty/")

        current_bytes_before = config_path.read_bytes()
        backup_bytes_before = kilo_mod._DEFAULT_BACKUP_PATH.read_bytes()

        with caplog.at_level(logging.WARNING, logger="kitty.launchers.kilo"):
            adapter.cleanup_launch(polluted, settings_path=config_path)

        assert config_path.read_bytes() == current_bytes_before, (
            "cleanup_launch overwrote the current file with a polluted snapshot"
        )
        assert kilo_mod._DEFAULT_BACKUP_PATH.read_bytes() == backup_bytes_before, (
            "cleanup_launch deleted the backup while restoring a polluted snapshot"
        )
        assert any("captured original carries kitty markers" in rec.message for rec in caplog.records), (
            f"expected the polluted-snapshot warning, got: {[r.message for r in caplog.records]}"
        )

    def test_clean_original_vs_clean_current_leaves_both(
        self,
        adapter: KiloAdapter,
        config_path: pathlib.Path,
    ) -> None:
        """Clean original + clean current (user hand-edit) → leave both alone (AC-R3c)."""
        original = self._stage_and_patch(adapter, config_path)
        backup_bytes_before = kilo_mod._DEFAULT_BACKUP_PATH.read_bytes()

        # User hand-edits during the session: the current file no longer carries markers.
        config_path.write_text(json.dumps({"provider": {"ollama": {"x": 1}}}))

        adapter.cleanup_launch(original, settings_path=config_path)

        assert json.loads(config_path.read_text()) == {"provider": {"ollama": {"x": 1}}}
        assert kilo_mod._DEFAULT_BACKUP_PATH.read_bytes() == backup_bytes_before

    def test_clean_original_vs_missing_current_leaves_both(
        self, adapter: KiloAdapter, config_path: pathlib.Path
    ) -> None:
        """Clean original + current file deleted (e.g. user removed kilo) → leave both alone (AC-R3d)."""
        original = self._stage_and_patch(adapter, config_path)
        backup_bytes_before = kilo_mod._DEFAULT_BACKUP_PATH.read_bytes()

        config_path.unlink()

        adapter.cleanup_launch(original, settings_path=config_path)

        assert not config_path.exists(), "cleanup_launch resurrected kilo.json despite a missing current file"
        assert kilo_mod._DEFAULT_BACKUP_PATH.read_bytes() == backup_bytes_before

    def test_clean_original_vs_unreadable_current_leaves_both(
        self, adapter: KiloAdapter, config_path: pathlib.Path
    ) -> None:
        """Clean original + unparseable current (ValueError) → leave both alone (AC-R3d)."""
        original = self._stage_and_patch(adapter, config_path)
        backup_bytes_before = kilo_mod._DEFAULT_BACKUP_PATH.read_bytes()

        config_path.write_text("not valid json {{{")

        adapter.cleanup_launch(original, settings_path=config_path)

        assert config_path.read_text() == "not valid json {{{"
        assert kilo_mod._DEFAULT_BACKUP_PATH.read_bytes() == backup_bytes_before

    def test_none_original_does_not_touch_backup(self, adapter: KiloAdapter, config_path: pathlib.Path) -> None:
        """``original=None`` (from-scratch session) never deletes a backup (AC-R3e part 1)."""
        # Stage a backup as if some other session wrote it; then run cleanup
        # with None original — the file we created is deleted, the backup is not.
        backup_bytes = b'{"provider":{"openai":{}}}'
        kilo_mod._DEFAULT_BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
        kilo_mod._DEFAULT_BACKUP_PATH.write_bytes(backup_bytes)
        config_path.write_text(json.dumps({"provider": {"kitty": {}}}))

        adapter.cleanup_launch(None, settings_path=config_path)

        assert not config_path.exists()
        assert kilo_mod._DEFAULT_BACKUP_PATH.read_bytes() == backup_bytes

    def test_failed_restore_keeps_backup(
        self,
        adapter: KiloAdapter,
        config_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A restore write failure does not delete the backup (AC-R3e part 2)."""
        original = self._stage_and_patch(adapter, config_path)

        real_write_text = pathlib.Path.write_text

        def failing_write_text(self: pathlib.Path, *args: object, **kwargs: object) -> None:
            if self == config_path:
                raise OSError("simulated restore write failure")
            return real_write_text(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "write_text", failing_write_text)

        with pytest.raises(OSError, match="simulated restore write failure"):
            adapter.cleanup_launch(original, settings_path=config_path)

        assert kilo_mod._DEFAULT_BACKUP_PATH.exists(), "backup was deleted despite the restore write failure"


class TestCleanupLaunchPreservesBytesWithBackup:
    """The KBR-262 byte-identity contract extends to the restore leg that now
    also deletes the backup: a CRLF user's config survives a round trip.
    """

    def test_writes_byte_exactly_and_removes_backup(self, adapter: KiloAdapter, config_path: pathlib.Path) -> None:
        """A CRLF clean original is restored byte-exactly and the backup is removed (AC-R11)."""
        crlf_bytes = (
            b'{\r\n  "provider": {"openai": {"apiKey": "sk-original"}},\r\n'
            b'  "model": "opus",\r\n'
            b'  "comment": "user kept CRLF line endings"\r\n'
            b"}\r\n"
        )
        config_path.write_bytes(crlf_bytes)

        adapter.build_spawn_config(_make_profile(), 18080, "key")
        original = adapter.prepare_launch({"KILO_PROVIDER": "kitty"}, settings_path=config_path)
        assert original is not None

        adapter.cleanup_launch(original, settings_path=config_path)

        assert config_path.read_bytes() == crlf_bytes, "cleanup_launch did not preserve CRLF after writing the original"
        assert not kilo_mod._DEFAULT_BACKUP_PATH.exists()
