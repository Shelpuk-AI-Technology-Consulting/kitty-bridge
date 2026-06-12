"""Tests for BalancingProfile schema — validation, frozen, reserved names."""

import pytest
from pydantic import ValidationError

from kitty.profiles.schema import RESERVED_NAMES, BalancingProfile


class TestBalancingProfileValidConstruction:
    def test_valid_balancing_profile(self):
        bp = BalancingProfile(name="my-balancer", members=["alpha", "beta"])
        assert bp.name == "my-balancer"
        assert bp.members == ["alpha", "beta"]
        assert bp.is_default is False

    def test_valid_with_more_than_two_members(self):
        bp = BalancingProfile(name="multi", members=["a", "b", "c", "d"])
        assert len(bp.members) == 4

    def test_is_default_can_be_set(self):
        bp = BalancingProfile(name="bal", members=["a", "b"], is_default=True)
        assert bp.is_default is True


class TestBalancingProfileNameValidation:
    def test_accepts_valid_lowercase_name(self):
        BalancingProfile(name="valid-name_123", members=["a", "b"])

    def test_accepts_single_char_name(self):
        BalancingProfile(name="a", members=["x", "y"])

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError, match="name"):
            BalancingProfile(name="", members=["a", "b"])

    def test_rejects_uppercase_name(self):
        with pytest.raises(ValidationError, match="name"):
            BalancingProfile(name="MyBalancer", members=["a", "b"])

    def test_rejects_reserved_names(self):
        for name in RESERVED_NAMES:
            with pytest.raises(ValidationError, match="reserved"):
                BalancingProfile(name=name, members=["a", "b"])


class TestBalancingProfileMembersValidation:
    def test_rejects_empty_members(self):
        with pytest.raises(ValidationError, match="at least 2"):
            BalancingProfile(name="bal", members=[])

    def test_rejects_single_member(self):
        with pytest.raises(ValidationError, match="at least 2"):
            BalancingProfile(name="bal", members=["only-one"])

    def test_rejects_duplicate_members(self):
        with pytest.raises(ValidationError, match="duplicate"):
            BalancingProfile(name="bal", members=["alpha", "alpha"])

    def test_rejects_self_reference(self):
        with pytest.raises(ValidationError, match="self-reference"):
            BalancingProfile(name="bal", members=["bal", "other"])

    def test_accepts_exactly_two_members(self):
        bp = BalancingProfile(name="bal", members=["alpha", "beta"])
        assert len(bp.members) == 2


class TestBalancingProfileFrozen:
    def test_frozen_model_rejects_mutation(self):
        bp = BalancingProfile(name="bal", members=["a", "b"])
        with pytest.raises(ValidationError):
            bp.name = "changed"  # type: ignore[misc]

    def test_members_list_is_frozen(self):
        bp = BalancingProfile(name="bal", members=["a", "b"])
        with pytest.raises(ValidationError):
            bp.members = ["x"]  # type: ignore[misc]


class TestBalancingProfileSerialization:
    def test_round_trip_json(self):
        bp = BalancingProfile(name="bal", members=["alpha", "beta"], is_default=True)
        data = bp.model_dump(mode="json")
        restored = BalancingProfile.model_validate(data)
        assert restored == bp

    def test_model_dump_includes_all_fields(self):
        bp = BalancingProfile(name="bal", members=["a", "b"])
        data = bp.model_dump(mode="json")
        assert "name" in data
        assert "members" in data
        assert "is_default" in data


class TestBalancingProfileMemberExistenceValidation:
    """F36: balancing profile members must exist in the profile store."""

    def test_validate_member_existence_rejects_missing_member(self):
        bp = BalancingProfile(name="balanced", members=["primary", "missing"])

        def exists(name: str) -> bool:
            return name == "primary"

        with pytest.raises(ValueError, match="missing member"):
            bp.validate_member_existence(exists)

    def test_validate_member_existence_accepts_all_existing_members(self):
        bp = BalancingProfile(name="balanced", members=["primary", "secondary"])

        def exists(name: str) -> bool:
            return name in {"primary", "secondary"}

        assert bp.validate_member_existence(exists) is bp
