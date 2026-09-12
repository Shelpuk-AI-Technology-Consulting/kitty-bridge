"""Model context window lookup from generated metadata table."""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Iterable
from functools import cache, lru_cache
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


def _match_catalog(query: str, keys: Collection[str]) -> tuple[str | None, bool]:
    """Match a model name against a set of catalog keys.

    Tries, in order: an exact match; the query as a ``"/"``-delimited suffix of
    a key (``gpt-4o`` finds ``openai/gpt-4o``); a key as a ``"/"``-delimited
    suffix of the query (``z-ai/glm-5.2`` finds ``glm-5.2``). The second rule is
    the metadata table's, the third the overrides catalog's; both catalogs get
    both so that a name resolves the same window whichever dialect spells it.

    Args:
        query: The model name to look up, already lowercased.
        keys: The catalog's keys, already lowercased. The caller passes the
            cached catalog mapping itself, so no copy is built per request.

    Returns:
        A ``(key, ambiguous)`` pair. ``key`` is the match, or ``None`` when
        nothing matched. ``ambiguous`` reports that a step matched more than
        one key — which must not collapse into "no match", because a miss
        falls through to the tail retry while an ambiguous match stops the
        lookup for this catalog entirely (see :func:`_resolve_catalog`).
    """
    if query in keys:
        return query, False

    # Each step is decisive: a step that matches several keys cannot be
    # narrowed by trying the next one, it can only be reported.
    for candidates in (
        [k for k in keys if k.endswith("/" + query)],
        [k for k in keys if query.endswith("/" + k)],
    ):
        if len(candidates) == 1:
            return candidates[0], False
        if len(candidates) > 1:
            logger.warning(
                "Ambiguous context entry for %s: %d matches (%s)",
                query,
                len(candidates),
                sorted(candidates),
            )
            return None, True
    return None, False


def _resolve_catalog(model: str, keys: Collection[str]) -> str | None:
    """Resolve a model name against a catalog, retrying once without its prefix.

    A profile may name a model in its provider's dialect (``azure/gpt-4o``)
    while the catalog is keyed in another (``openai/gpt-4o``). When the name as
    given matches nothing, it is retried with its leading segment removed —
    once, at the first separator. Stripping further would reduce a Fireworks
    full path such as ``accounts/fireworks/routers/kimi`` to a bare name and
    could match an unrelated model (KBR-151).

    An ambiguous match is terminal: retrying the tail after the matcher has
    said it cannot identify the entry would log "ambiguous" and answer anyway.

    Args:
        model: The model name as configured, in any case.
        keys: The catalog's keys, already lowercased.

    Returns:
        The matching key, or ``None`` when the catalog cannot resolve the name.
    """
    query = model.lower()
    hit, ambiguous = _match_catalog(query, keys)
    if ambiguous:
        return None
    if hit is None:
        _, sep, tail = query.partition("/")
        if sep and tail:
            hit, ambiguous = _match_catalog(tail, keys)
            if ambiguous:
                return None
    return hit


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


def _colliding_keys(keys: Iterable[str]) -> list[str]:
    """Return catalog keys that share a final segment with another key.

    Final-segment uniqueness is what makes :func:`_match_catalog` unambiguous,
    and it is the weakest condition that does. Verified exhaustively over every
    one-, two- and three-segment key pair: zero ambiguous lookups survive it,
    where the looser "unique after the first separator" admits 640.

    The two rules coincide only while every key has at most two segments, which
    both shipped catalogs satisfy today. They part company on a key such as
    ``a/b/x``, whose tail is ``b/x`` but whose final segment is ``x`` — it
    collides with a key ``x`` that the looser rule would wave through, and the
    query ``x`` would then match both.

    Args:
        keys: A catalog's keys, already lowercased.

    Returns:
        The colliding keys, sorted, or an empty list when every final segment
        is unique.
    """
    by_segment: dict[str, list[str]] = {}
    for key in keys:
        by_segment.setdefault(key.rsplit("/", 1)[-1], []).append(key)
    return sorted(k for group in by_segment.values() if len(group) > 1 for k in group)


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
        # Two keys sharing a tail make every lookup for that tail ambiguous, and
        # an ambiguous overrides catalog hands the decision to the metadata
        # table it exists to overrule. Reject such a revision wholesale rather
        # than load it, as with a body that is not a JSON object.
        collisions = _colliding_keys(parsed) if parsed is not None else []
        if collisions:
            logger.warning(
                "Model context overrides from %s hold keys that differ only by prefix (%s); "
                "keeping the packaged catalog",
                REMOTE_OVERRIDES_CACHE_PATH,
                collisions,
            )
            parsed = None
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
    """Return the context length from the overrides catalog, or None.

    Uses the shared :func:`_resolve_catalog` matcher, so a pinned model
    resolves however the query and the key are spelled — bare or vendor-
    prefixed, in either dialect. Before KBR-151 this carried a prefixed query
    up to a bare key but not the reverse, so an overrides catalog holding a
    prefixed key was silently out-ranked by the lower-priority metadata table.

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

    key = _resolve_catalog(model, overrides)
    return overrides[key] if key is not None else None


@cache
def _log_default_fallback(provider: str, model: str) -> None:
    """Report once that a model's context window could not be resolved.

    Deduplicated per ``(provider, model)`` because the compaction budget is
    recomputed on every request, so an unconditional line would be one per
    turn. ``INFO`` rather than ``WARNING``: an unknown model is routine and
    correct for a profile pointing at a local or private model, and a warning
    that fires routinely is one that gets filtered.

    ``functools.cache`` rather than a bounded LRU: the keys come from profiles,
    never from the wire, and a bounded cache would re-emit after eviction and
    so would not deliver the once-per-model guarantee this exists for.

    Args:
        provider: The provider type, for identifying which profile is affected.
        model: The model name **as configured** — never the prefix-stripped
            tail, which is a string the operator never wrote.
    """
    logger.info(
        "No context window found for %s/%s; assuming %d tokens. "
        "Set context_window in the profile's provider_config to override.",
        provider,
        model,
        DEFAULT_CONTEXT_TOKENS,
    )


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
    3. Metadata table.

    Both catalogs are searched by :func:`_resolve_catalog`, so a model resolves
    the same window however it is spelled: bare, or carrying any single vendor
    or provider prefix (KBR-151).

    WARNING: an entry in the overrides catalog silently trumps a per-profile
    ``provider_config["context_window"]``. To make a profile's context_window
    take effect, omit that model from the overrides file — noting that since
    KBR-151 one key captures **every** prefixed spelling of its model, so
    ``openai/gpt-4o``, ``gpt-4o`` and ``azure/gpt-4o`` are one entry to omit,
    not three. A key's own prefix does not scope it to that vendor.
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
    matched_id = _resolve_catalog(model, metadata)
    if matched_id is not None:
        value = _coerce_context_tokens(metadata[matched_id].get("context_length"))
        if value is not None:
            return value
        logger.warning("Invalid context_length in metadata for %s", matched_id)

    # Nothing knows this model. Say so once per model rather than per request:
    # the budget is recomputed on every turn, and silence here is what let
    # KBR-151 size a conversation from a window the model does not have.
    _log_default_fallback(provider, model)
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
