"""Tests for profiles/schema.py — Profile data model, validation, reserved names."""

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
            base_url="https://example.com/v1",
            provider_config={"temperature": 0.7},
            is_default=True,
        )
        assert profile.name == "my-profile"
        assert profile.provider == "zai_regular"
        assert profile.model == "gpt-4o"
        assert str(profile.base_url) == "https://example.com/v1"
        assert profile.provider_config == {"temperature": 0.7}
        assert profile.is_default is True

    def test_valid_profile_minimal_fields(self):
        profile = Profile(
            name="minimal",
            provider="novita",
            model="gpt-4o-mini",
            auth_ref=str(uuid.uuid4()),
        )
        assert profile.name == "minimal"
        assert profile.base_url is None
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


class TestProfileBaseUrlValidation:
    def test_accepts_valid_https_url(self):
        profile = Profile(
            name="test",
            provider="zai_regular",
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
            base_url="https://api.example.com/v1",
        )
        assert str(profile.base_url) == "https://api.example.com/v1"

    def test_rejects_http_url(self):
        with pytest.raises(ValidationError, match="base_url"):
            Profile(
                name="test",
                provider="zai_regular",
                model="gpt-4o",
                auth_ref=str(uuid.uuid4()),
                base_url="http://api.example.com/v1",
            )

    def test_rejects_non_url(self):
        with pytest.raises(ValidationError, match="base_url"):
            Profile(
                name="test",
                provider="zai_regular",
                model="gpt-4o",
                auth_ref=str(uuid.uuid4()),
                base_url="not-a-url",
            )

    def test_allows_none(self):
        profile = Profile(
            name="test",
            provider="zai_regular",
            model="gpt-4o",
            auth_ref=str(uuid.uuid4()),
            base_url=None,
        )
        assert profile.base_url is None


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
