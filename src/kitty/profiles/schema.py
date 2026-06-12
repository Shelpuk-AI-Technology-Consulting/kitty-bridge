"""Profile data model, validation rules, and reserved names."""

from __future__ import annotations

import re
import typing
import uuid
from typing import Literal

from pydantic import BaseModel, UrlConstraints, field_validator, model_validator
from pydantic import HttpUrl as _HttpUrl

RESERVED_NAMES: frozenset[str] = frozenset(
    {"setup", "doctor", "codex", "claude", "gemini", "kilo", "profile", "profiles", "help", "default"}
)

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

_PROVIDER_TYPES = Literal[
    "zai_regular",
    "zai_coding",
    "minimax",
    "minimax_token",
    "novita",
    "ollama",
    "openai",
    "openai_subscription",
    "openrouter",
    "anthropic",
    "bedrock",
    "azure",
    "vertex",
    "fireworks",
    "google_aistudio",
    "opencode_go",
    "custom_anthropic",
    "custom_openai",
    "kimi",
    "mimo",
    "byteplus",
    "ollama_cloud",
]

PROVIDER_LIST: list[str] = list(typing.get_args(_PROVIDER_TYPES))

PROVIDER_LABELS: dict[str, str] = {
    "anthropic": "Anthropic",
    "azure": "MS Azure",
    "bedrock": "AWS Bedrock",
    "byteplus": "BytePlus",
    "custom_anthropic": "Custom Anthropic-Compatible",
    "custom_openai": "Custom OpenAI-Compatible",
    "fireworks": "Fireworks FirePass",
    "google_aistudio": "Google AI Studio",
    "kimi": "Kimi Code",
    "minimax": "MiniMax",
    "minimax_token": "MiniMax Token Plan",
    "mimo": "Xiaomi MiMo",
    "novita": "Novita AI",
    "ollama": "Ollama",
    "ollama_cloud": "Ollama Cloud",
    "opencode_go": "OpenCode Go",
    "openai": "OpenAI",
    "openai_subscription": "OpenAI ChatGPT Plan",
    "openrouter": "OpenRouter",
    "vertex": "Google Cloud Vertex",
    "zai_coding": "Z.AI Coding Plan",
    "zai_regular": "Z.AI",
}

PROVIDER_SECTIONS: list[tuple[str, list[str]]] = [
    (
        "-- Regular API Key --",
        [
            "bedrock",
            "anthropic",
            "byteplus",
            "google_aistudio",
            "vertex",
            "azure",
            "minimax",
            "openai",
            "openrouter",
            "zai_regular",
        ],
    ),
    (
        "-- Coding Plans / Subscriptions --",
        [
            "fireworks",
            "kimi",
            "minimax_token",
            "novita",
            "ollama_cloud",
            "openai_subscription",
            "opencode_go",
            "mimo",
            "zai_coding",
        ],
    ),
    (
        "-- Local LLMs --",
        [
            "ollama",
        ],
    ),
    (
        "-- Generic --",
        [
            "custom_anthropic",
            "custom_openai",
        ],
    ),
]


class HttpsUrl(_HttpUrl):
    """HTTP URL constrained to HTTPS scheme only."""

    _constraints = UrlConstraints(max_length=2083, allowed_schemes=["https"])


def _validate_profile_name(v: str) -> str:
    """Shared name validation for Profile and BalancingProfile."""
    if not _NAME_PATTERN.match(v):
        raise ValueError(f"name must match {_NAME_PATTERN.pattern!r} (lowercase, 1-32 chars)")
    if v in RESERVED_NAMES:
        raise ValueError(f"name {v!r} is reserved")
    return v


class Profile(BaseModel):
    """A launcher-target-agnostic profile binding a provider, model, and API key reference.

    Profiles are shared across all launcher targets (Codex, Claude Code, etc.).
    """

    model_config = {"frozen": True}

    name: str
    provider: _PROVIDER_TYPES
    model: str
    auth_ref: str
    base_url: HttpsUrl | None = None
    provider_config: dict = {}
    is_default: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return _validate_profile_name(v)

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model must not be empty or whitespace-only")
        return v.strip()

    @field_validator("auth_ref")
    @classmethod
    def validate_auth_ref(cls, v: str) -> str:
        parsed = uuid.UUID(v)
        if parsed.version != 4:
            raise ValueError(f"auth_ref must be a UUIDv4, got version {parsed.version}")
        return v


class BalancingProfile(BaseModel):
    """A profile that randomly distributes LLM calls across regular profiles.

    Balancing profiles cannot be nested — members must all be regular profiles.
    """

    model_config = {"frozen": True}

    name: str
    members: list[str]
    is_default: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return _validate_profile_name(v)

    @model_validator(mode="after")
    def validate_members(self) -> BalancingProfile:
        if len(self.members) < 2:
            raise ValueError("balancing profile must have at least 2 members")
        if len(self.members) != len(set(self.members)):
            raise ValueError("balancing profile members must not contain duplicates")
        if self.name in self.members:
            raise ValueError(f"balancing profile {self.name!r} must not self-reference in members")
        return self

    def validate_member_existence(self, member_exists: callable) -> BalancingProfile:
        """F36: Verify all members reference existing profiles.

        Args:
            member_exists: Callable(name) -> bool; returns True if a profile
                with that name exists.

        Raises:
            ValueError: If any member does not exist.

        Returns:
            self (for chaining).
        """
        missing = [m for m in self.members if not member_exists(m)]
        if missing:
            raise ValueError(f"balancing profile {self.name!r} references missing member(s): {', '.join(missing)}")
        return self


BackendConfig = Profile | BalancingProfile

__all__ = ["PROVIDER_LIST", "RESERVED_NAMES", "Profile", "BalancingProfile", "BackendConfig"]
