"""Tests for KiloAdapter prepare_launch / cleanup_launch config file patching."""

import json
import pathlib
import sys
import uuid

import pytest

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

    def test_returns_original_preserving_crlf_bytes(
        self, adapter: KiloAdapter, config_path: pathlib.Path
    ) -> None:
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
            b'}\r\n'
        )

        config_path.write_bytes(crlf_bytes)

        adapter.build_spawn_config(_make_profile(), 18080, "key")
        original = adapter.prepare_launch(
            {"KILO_PROVIDER": "kitty"}, settings_path=config_path
        )

        assert original is not None, (
            "prepare_launch saved no original for an existing CRLF config"
        )
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

        Args:
            adapter: The Kilo adapter under test.
            config_path: The kilo config path fixture (per-test temp dir).
            monkeypatch: Pytest's monkeypatch fixture (auto-restores).
        """
        captured: list[dict[str, object]] = []
        real_open = pathlib.Path.open

        def recording_open(
            self: pathlib.Path, *args: object, **kwargs: object
        ) -> object:
            captured.append(kwargs)
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "open", recording_open)

        original = '{"model": "opus"}\r\n{"foo": 1}\r\n'
        adapter.cleanup_launch(original, settings_path=config_path)

        assert len(captured) == 1, (
            f"expected exactly one Path.open call from cleanup_launch, "
            f"got {len(captured)}"
        )
        assert captured[0].get("newline") == "", (
            f"cleanup_launch opened without newline=''; "
            f"pathlib.Path.open kwargs were {captured[0]!r}"
        )

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason=(
            "write-side CRLF translation only fires on Windows; "
            "CI covers it on the Fast gate's Windows leg (KBR-164)"
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

        Args:
            adapter: The Kilo adapter under test.
            config_path: The kilo config path fixture (per-test temp dir).
            original: The text the user originally had on disk — parametrised.
        """
        adapter.cleanup_launch(original, settings_path=config_path)

        assert config_path.read_bytes() == original.encode("utf-8"), (
            f"cleanup_launch wrote {config_path.read_bytes()!r}; "
            f"expected {original.encode('utf-8')!r}"
        )
