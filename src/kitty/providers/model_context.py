"""Model context window lookup from generated metadata table."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_TOKENS = 128_000

_METADATA_PATH = Path(__file__).parent / "model_metadata.json"


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
        result = int(value)
        return result if result > 0 else None
    except (TypeError, ValueError):
        return None


def get_model_context_tokens(
    provider: str,
    model: str,
    provider_config: dict | None = None,
) -> int:
    """Return the context window size in tokens for the given model.

    Lookup priority:
    1. provider_config["context_window"] — manual override
    2. Exact match in metadata table
    3. DEFAULT_CONTEXT_TOKENS fallback
    """
    if provider_config and "context_window" in provider_config:
        override = _coerce_context_tokens(provider_config["context_window"])
        if override is not None:
            return override
        logger.warning("Invalid provider_config context_window for %s/%s", provider, model)

    metadata = _load_metadata()
    model_lower = model.lower()

    if model_lower in metadata:
        value = _coerce_context_tokens(metadata[model_lower].get("context_length"))
        if value is not None:
            return value
        logger.warning("Invalid context_length in metadata for %s", model_lower)

    if provider != "openrouter":
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
