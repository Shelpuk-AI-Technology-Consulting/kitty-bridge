"""Profile data model, validation rules, and reserved names."""

from __future__ import annotations

import re
import uuid
from typing import Literal

from pydantic import BaseModel, UrlConstraints, field_validator
from pydantic import HttpUrl as _HttpUrl

RESERVED_NAMES: frozenset[str] = frozenset(
    {"setup", "doctor", "codex", "claude", "gemini", "kilo", "profile", "profiles", "help", "default"}
)

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

_PROVIDER_TYPES = Literal["zai_regular", "zai_coding", "minimax", "novita", "ollama", "openai", "openrouter", "anthropic", "bedrock", "azure", "vertex", "fireworks"]


class HttpsUrl(_HttpUrl):
    """HTTP URL constrained to HTTPS scheme only."""

    _constraints = UrlConstraints(max_length=2083, allowed_schemes=["https"])


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
        if not _NAME_PATTERN.match(v):
            raise ValueError(f"name must match {_NAME_PATTERN.pattern!r} (lowercase, 1-32 chars)")
        if v in RESERVED_NAMES:
            raise ValueError(f"name {v!r} is reserved")
        return v

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


__all__ = ["RESERVED_NAMES", "Profile"]
