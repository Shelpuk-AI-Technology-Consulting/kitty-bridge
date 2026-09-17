"""Tests for profiles/schema.py — Profile data model, validation, reserved names.

KBR-158: the dead ``Profile.base_url`` field is gone. The base URL belongs in
``provider_config["base_url"]``. A top-level ``base_url`` with a non-None
value is rejected with a pointed message; ``None`` and absent both construct
(legacy ``profiles.json`` files carry ``"base_url": null`` and must keep
loading).
"""

import uuid

import pytest
from pydantic import ValidationError

from kitty.profiles.schema import RESERVED_NAMES, Profile


class TestProfileValidConstruction:
    def test_valid_profile_all_fields(self):
        profile = Profile(
            name="my-profile",
            provider="zai_regular",
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
            provider_config={
                "base_url": "https://example.com/v1",
                "temperature": 0.7,
            },
            is_default=True,
        )
        assert profile.name == "my-profile"
        assert profile.provider == "zai_regular"
        assert profile.model == "gpt-4o"
        assert profile.provider_config == {
            "base_url": "https://example.com/v1",
            "temperature": 0.7,
        }
        assert profile.is_default is True

    def test_valid_profile_minimal_fields(self):
        profile = Profile(
            name="minimal",
            provider="novita",
            model="gpt-4o-mini",
            auth_ref=str(uuid.uuid4()),
        )
        assert profile.name == "minimal"
        assert profile.provider_config == {}
        assert profile.is_default is False


class TestProfileNameValidation:
    def test_accepts_valid_lowercase_name(self):
        Profile(
            name="valid-name_123",
            provider="zai_regular",
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
        )

    def test_accepts_single_char_name(self):
        Profile(
            name="a",
            provider="zai_regular",
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
        )

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError, match="name"):
            Profile(
                name="",
                provider="zai_regular",
                model="gpt-4o",
                auth_ref=str(uuid.uuid4()),
            )

    def test_rejects_uppercase_name(self):
        with pytest.raises(ValidationError, match="name"):
            Profile(
                name="MyProfile",
                provider="zai_regular",
                model="gpt-4o",
                auth_ref=str(uuid.uuid4()),
            )

    def test_rejects_reserved_names(self):
        for name in RESERVED_NAMES:
            with pytest.raises(ValidationError, match="reserved"):
                Profile(
                    name=name,
                    provider="zai_regular",
                    model="gpt-4o",
                    auth_ref=str(uuid.uuid4()),
                )

    def test_name_preserved_when_valid(self):
        profile = Profile(
            name="already-lowercase",
            provider="zai_regular",
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
        )
        assert profile.name == "already-lowercase"


class TestProfileProviderValidation:
    @pytest.mark.parametrize(
        "provider",
        [
            "zai_regular",
            "zai_coding",
            "minimax",
            "novita",
            "ollama",
            "openai",
            "openrouter",
            "anthropic",
            "bedrock",
            "azure",
            "vertex",
            "fireworks",
        ],
    )
    def test_accepts_valid_providers(self, provider):
        Profile(
            name="test",
            provider=provider,
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
        )

    def test_rejects_invalid_provider(self):
        with pytest.raises(ValidationError, match="provider"):
            Profile(
                name="test",
                provider="invalid_provider",
                model="gpt-4o",
                auth_ref=str(uuid.uuid4()),
            )


class TestProfileBaseUrlChannel:
    """KBR-158: the base-URL channel is ``provider_config['base_url']``."""

    def test_provider_config_base_url_is_verbatim_http(self):
        """Local endpoints (vLLM, LM Studio, Ollama) use ``http://``; the
        schema layer must not normalise or reject them."""
        profile = Profile(
            name="local",
            provider="ollama",
            model="llama3",
            auth_ref=str(uuid.uuid4()),
            provider_config={"base_url": "http://localhost:11434"},
        )
        assert profile.provider_config["base_url"] == "http://localhost:11434"

    def test_provider_config_base_url_is_verbatim_https(self):
        profile = Profile(
            name="remote",
            provider="custom_openai",
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
            provider_config={"base_url": "https://api.example.com/v1"},
        )
        assert profile.provider_config["base_url"] == "https://api.example.com/v1"

    def test_base_url_is_not_a_declared_field(self):
        """Field absence is the pin: no resolver can read what does not exist."""
        assert "base_url" not in Profile.model_fields


class TestProfileRejectsTopLevelBaseUrl:
    """KBR-158: a top-level ``base_url`` with a non-None value is rejected."""

    _POINTER = r'provider_config\["base_url"\]'

    def test_rejects_https_url_kwarg(self):
        with pytest.raises(ValidationError, match=self._POINTER):
            Profile(
                name="test",
                provider="zai_regular",
                model="gpt-4o",
                auth_ref=str(uuid.uuid4()),
                base_url="https://api.example.com/v1",
            )

    def test_rejects_http_url_kwarg(self):
        with pytest.raises(ValidationError, match=self._POINTER):
            Profile(
                name="test",
                provider="zai_regular",
                model="gpt-4o",
                auth_ref=str(uuid.uuid4()),
                base_url="http://api.example.com/v1",
            )

    def test_rejects_non_url_kwarg(self):
        with pytest.raises(ValidationError, match=self._POINTER):
            Profile(
                name="test",
                provider="zai_regular",
                model="gpt-4o",
                auth_ref=str(uuid.uuid4()),
                base_url="not-a-url",
            )

    def test_rejects_via_model_validate(self):
        with pytest.raises(ValidationError, match=self._POINTER):
            Profile.model_validate(
                {
                    "name": "test",
                    "provider": "zai_regular",
                    "model": "gpt-4o",
                    "auth_ref": str(uuid.uuid4()),
                    "base_url": "https://api.example.com/v1",
                }
            )

    def test_allows_explicit_none_kwarg(self):
        """``None`` is the old optional-field "not set" shape and must keep
        constructing."""
        profile = Profile(
            name="test",
            provider="zai_regular",
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
            base_url=None,
        )
        assert profile.name == "test"

    def test_legacy_serialized_null_loads(self):
        """Migration safety: today's ``profiles.json`` files carry
        ``"base_url": null`` from the deleted optional field and must keep
        loading after the change."""
        profile = Profile.model_validate(
            {
                "name": "test",
                "provider": "zai_regular",
                "model": "gpt-4o",
                "auth_ref": str(uuid.uuid4()),
                "base_url": None,
            }
        )
        assert profile.name == "test"


class TestProfileOtherExtraKeysIgnored:
    """KBR-158: only ``base_url`` is rejected. Other extra keys keep today's
    silently-ignored behaviour (a separate ticket if ever needed)."""

    def test_unrelated_extra_key_loads(self):
        profile = Profile.model_validate(
            {
                "name": "test",
                "provider": "zai_regular",
                "model": "gpt-4o",
                "auth_ref": str(uuid.uuid4()),
                "unknown_future_field": "anything",
            }
        )
        assert profile.name == "test"


class TestProfileRoundTrip:
    """KBR-158: round-tripping a fresh profile produces no ``base_url`` key."""

    def test_round_trip_no_base_url_in_dump(self):
        profile = Profile(
            name="test",
            provider="zai_regular",
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
            provider_config={"base_url": "https://api.example.com/v1"},
        )
        dumped = profile.model_dump(mode="json")
        assert "base_url" not in dumped
        restored = Profile.model_validate(dumped)
        assert restored == profile


class TestProfileAuthRefValidation:
    def test_accepts_valid_uuidv4(self):
        Profile(
            name="test",
            provider="zai_regular",
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
        )

    def test_rejects_invalid_uuid(self):
        with pytest.raises(ValidationError, match="auth_ref"):
            Profile(
                name="test",
                provider="zai_regular",
                model="gpt-4o",
                auth_ref="not-a-uuid",
            )

    def test_rejects_empty_auth_ref(self):
        with pytest.raises(ValidationError, match="auth_ref"):
            Profile(
                name="test",
                provider="zai_regular",
                model="gpt-4o",
                auth_ref="",
            )


class TestProfileDefaults:
    def test_is_default_defaults_to_false(self):
        profile = Profile(
            name="test",
            provider="zai_regular",
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
        )
        assert profile.is_default is False

    def test_provider_config_defaults_to_empty_dict(self):
        profile = Profile(
            name="test",
            provider="zai_regular",
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
        )
        assert profile.provider_config == {}

    def test_backup_defaults_to_false(self):
        """R1: a profile is a primary pool member unless explicitly marked backup."""
        profile = Profile(
            name="test",
            provider="zai_regular",
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
        )
        assert profile.backup is False


class TestProfileBackupFlag:
    """R1: the reserve-tier flag used by balancing profiles."""

    def _make(self, **overrides) -> Profile:
        defaults = {
            "name": "test",
            "provider": "zai_regular",
            "model": "gpt-4o",
            "auth_ref": str(uuid.uuid4()),
        }
        defaults.update(overrides)
        return Profile(**defaults)

    def test_accepts_explicit_true(self):
        assert self._make(backup=True).backup is True

    def test_model_copy_toggles_backup(self):
        """The edit flow mutates frozen profiles via model_copy."""
        profile = self._make(backup=False)
        assert profile.model_copy(update={"backup": True}).backup is True

    def test_model_dump_includes_backup(self):
        """R2: the store serialises via model_dump, so the field must appear."""
        assert self._make(backup=True).model_dump()["backup"] is True


class TestReservedNames:
    def test_contains_expected_names(self):
        expected = {"setup", "doctor", "codex", "claude", "gemini", "kilo", "profile", "profiles", "help", "default"}
        assert expected == RESERVED_NAMES

    def test_is_frozenset(self):
        assert isinstance(RESERVED_NAMES, frozenset)


class TestProfileFrozen:
    def test_frozen_model_rejects_mutation(self):
        profile = Profile(
            name="test",
            provider="zai_regular",
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
        )
        with pytest.raises(ValidationError):
            profile.name = "changed"  # type: ignore[misc]


class TestAuthRefVersionEnforcement:
    def test_rejects_non_v4_uuid(self):
        """UUIDv1 should be rejected even though it's a valid UUID."""
        v1 = uuid.uuid1()
        with pytest.raises(ValidationError, match="version"):
            Profile(
                name="test",
                provider="zai_regular",
                model="gpt-4o",
                auth_ref=str(v1),
            )
