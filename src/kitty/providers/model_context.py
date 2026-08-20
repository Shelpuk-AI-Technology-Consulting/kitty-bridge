"""Model context window lookup from generated metadata table."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from platformdirs import user_cache_dir

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_TOKENS = 200_000

TOKENS_TO_CHARS_FACTOR = 4


def tokens_to_chars(tokens: int) -> int:
    """Convert a token count to an estimated character count."""
    return tokens * TOKENS_TO_CHARS_FACTOR


_METADATA_PATH = Path(__file__).parent / "model_metadata.json"

# Highest-priority source of model context lengths: a kitty-local, git-tracked
# catalog of known models whose context window must not be derived from the
# OpenRouter metadata (e.g. because the model is missing from it, or to pin a
# stable value that does not drift when model_metadata.json is refreshed). The
# OpenRouter refresh script writes only model_metadata.json, never this file.
_OVERRIDES_PATH = Path(__file__).parent / "model_context_overrides.json"

# Cached copy of the overrides catalog synced at runtime from the kitty-bridge
# GitHub repo (see model_context_sync). When present and valid it is a newer
# revision of the packaged file and replaces it wholesale. Single definition
# shared by the reader (_load_overrides) and the sync writer.
REMOTE_OVERRIDES_CACHE_PATH = Path(user_cache_dir("kitty")) / "model_context_overrides.json"


@lru_cache(maxsize=1)
def _load_metadata() -> dict[str, dict]:
    """Load model_metadata.json and return {lowercase_id: model_dict}."""
    try:
        raw = _METADATA_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("model_metadata.json not found at %s", _METADATA_PATH)
        return {}
    try:
        models = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("model_metadata.json is not valid JSON")
        return {}
    if not isinstance(models, list):
        logger.warning("model_metadata.json expected array, got %s", type(models).__name__)
        return {}

    valid_models = []
    for m in models:
        if not isinstance(m, dict) or "id" not in m:
            continue
        context_length = _coerce_context_tokens(m.get("context_length"))
        if context_length is None:
            continue
        valid_models.append({**m, "context_length": context_length})

    dropped = len(models) - len(valid_models)
    if dropped:
        logger.warning("model_metadata.json dropped %d invalid records", dropped)
    return {m["id"].lower(): m for m in valid_models}


def _coerce_context_tokens(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)  # type: ignore[call-overload]  # value comes from untyped JSON
        return result if result > 0 else None
    except (TypeError, ValueError):
        return None


def _parse_overrides(raw: str, source: str) -> dict[str, int] | None:
    """Parse overrides catalog JSON text into a validated mapping.

    Invalid entries (non-string / blank keys, non-positive / boolean /
    non-int values) are dropped with a warning, mirroring ``_load_metadata``.

    Args:
        raw: Raw JSON text of the catalog.
        source: Human-readable source description for log messages.

    Returns:
        A ``{lowercase_model: tokens}`` mapping (possibly empty), or ``None``
        when ``raw`` is not valid JSON or its root is not an object.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Model context overrides from %s are not valid JSON", source)
        return None
    if not isinstance(data, dict):
        logger.warning("Model context overrides from %s expected an object, got %s", source, type(data).__name__)
        return None

    overrides: dict[str, int] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key.strip():
            logger.warning("Model context overrides from %s dropped invalid key %r", source, key)
            continue
        tokens = _coerce_context_tokens(value)
        if tokens is None:
            logger.warning("Model context overrides from %s dropped invalid entry for %s", source, key)
            continue
        overrides[key.lower()] = tokens
    return overrides


@lru_cache(maxsize=1)
def _load_overrides() -> dict[str, int]:
    """Load the overrides catalog, preferring the remote-synced cache.

    Sources, in order:

    1. The remote-synced cache at ``REMOTE_OVERRIDES_CACHE_PATH`` — when it
       parses to a JSON object it **replaces the packaged catalog wholesale**
       (it is a newer revision of the same file; entries removed upstream
       must not linger via merging).
    2. The packaged ``model_context_overrides.json``.

    A missing or malformed source falls through to the next one; a missing
    packaged file yields an empty dict so the caller falls through to the
    lower-priority resolution layers. Invalid entries are dropped with a
    warning, mirroring ``_load_metadata``.

    Returns:
        A ``{lowercase_model: tokens}`` dict.
    """
    # A valid cached copy is a newer revision of the packaged catalog.
    try:
        raw = REMOTE_OVERRIDES_CACHE_PATH.read_text(encoding="utf-8")
    except OSError:
        raw = None
    if raw is not None:
        parsed = _parse_overrides(raw, source=str(REMOTE_OVERRIDES_CACHE_PATH))
        if parsed is not None:
            return parsed

    # Fall back to the catalog packaged with the wheel.
    try:
        raw = _OVERRIDES_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    parsed = _parse_overrides(raw, source="model_context_overrides.json")
    return {} if parsed is None else parsed


def _lookup_override(model: str) -> int | None:
    """Return the context length from the local overrides file, or None.

    Matching is case-insensitive and reuses the single-direction suffix
    convention of the metadata lookup (the query may carry a vendor prefix
    that a later normalize step strips; the override key is bare):

    1. Exact: ``model.lower()`` is an override key.
    2. Suffix: the query ``model.lower()`` ends with ``"/" + key`` for exactly
       one override key (e.g. ``"z-ai/glm-5.2"`` matches the key ``"glm-5.2"``).
       More than one suffix match is ambiguous: log a warning and return None
       so the caller falls through, matching the metadata ambiguity behavior.

    Args:
        model: The raw model name as seen by the resolver (may be
            vendor-prefixed).

    Returns:
        The override context length in tokens, or ``None`` when no unambiguous
        override matches.
    """
    overrides = _load_overrides()
    if not overrides:
        return None

    model_lower = model.lower()
    # 1. Exact match.
    if model_lower in overrides:
        return overrides[model_lower]

    # 2. Single-direction suffix match: "z-ai/glm-5.2" matches key "glm-5.2".
    suffix_matches = [k for k in overrides if model_lower.endswith("/" + k)]
    if len(suffix_matches) == 1:
        return overrides[suffix_matches[0]]
    if len(suffix_matches) > 1:
        logger.warning(
            "Ambiguous context override for %s: %d matches (%s)",
            model_lower,
            len(suffix_matches),
            suffix_matches,
        )
    return None


def get_model_context_tokens(
    provider: str,
    model: str,
    provider_config: dict | None = None,
) -> int:
    """Return the context window size in tokens for the given model.

    Lookup priority:
    1. Overrides catalog — the remote-synced cache when valid, else the
       packaged model_context_overrides.json — highest priority.
    2. provider_config["context_window"] — per-profile manual override.
    3. Exact match on model name in metadata table.
    4. Suffix match: bare model name (e.g. "gpt-4o") matches metadata
       entries with a provider prefix (e.g. "openai/gpt-4o").
    5. DEFAULT_CONTEXT_TOKENS fallback.

    WARNING: a packaged entry in the local overrides file silently trumps a
    per-profile ``provider_config["context_window"]``. To make a profile's
    context_window take effect for a model, omit that model from the overrides
    file.
    """
    override = _lookup_override(model)
    if override is not None:
        return override

    if provider_config and "context_window" in provider_config:
        override = _coerce_context_tokens(provider_config["context_window"])
        if override is not None:
            return override
        logger.warning("Invalid provider_config context_window for %s/%s", provider, model)

    metadata = _load_metadata()
    model_lower = model.lower()

    # 3. Exact match (model as-is, e.g. "openai/gpt-4o" or a bare name)
    if model_lower in metadata:
        value = _coerce_context_tokens(metadata[model_lower].get("context_length"))
        if value is not None:
            return value
        logger.warning("Invalid context_length in metadata for %s", model_lower)

    # 4. Suffix match: "gpt-4o" matches "openai/gpt-4o", "gpt-4o-mini", etc.
    suffix = "/" + model_lower
    matches = [entry for mid, entry in metadata.items() if mid.endswith(suffix)]
    if len(matches) == 1:
        value = _coerce_context_tokens(matches[0].get("context_length"))
        if value is not None:
            return value
        logger.warning("Invalid context_length in metadata suffix match for %s", model_lower)
    elif len(matches) > 1:
        match_ids = [mid for mid in metadata if mid.endswith(suffix)]
        logger.warning("Ambiguous context metadata for %s: %d matches (%s)", model_lower, len(matches), match_ids)

    return DEFAULT_CONTEXT_TOKENS


def get_balancing_min_context_tokens(
    backends: list[tuple[str, str, dict | None]],
) -> int:
    """Return the smallest context window across a list of balancing backends.

    Each backend is a (provider, model, provider_config) tuple.
    Returns DEFAULT_CONTEXT_TOKENS if the list is empty.
    """
    if not backends:
        return DEFAULT_CONTEXT_TOKENS
    return min(get_model_context_tokens(provider, model, config) for provider, model, config in backends)
