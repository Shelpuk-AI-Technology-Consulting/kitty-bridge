"""Tests pinning the packaged model_context_overrides.json catalog.

Unlike ``test_model_context.py`` (which patches the overrides file with
synthetic data), these tests deliberately read the *real* packaged catalog
and metadata table. They exist so that drift between the shipped catalog and
the verified upstream context windows of the models we run is caught at CI
time.

Verified sources (2026-08, via Kindly Web Search):

- ``qwen3.8-max``: OpenRouter ``qwen/qwen3.8-max`` ("1,000,000-token context
  window"), Alibaba Cloud Model Studio pricing ("0<Token≤1M").
- ``MiniMax-M3``: MiniMax official blog ("1M context"), OpenRouter
  ("1,048,576-token context window").
- ``glm-5.3``: z.ai GLM-5.3 blog ("1M-token context window"), OpenRouter
  ``z-ai/glm-5.3`` ("1,048,576 token context window").
- ``glm-5.3-flash``: Z.AI developer docs
  (https://docs.z.ai/guides/vlm/glm-5.3-flash) — "Context Length: 1M", and
  "Text parameters are consistent with GLM-5.3, with support for a 1M-token
  context window". Absent from model_metadata.json, so without this entry it
  falls back to DEFAULT_CONTEXT_TOKENS (200k) — a five-fold under-estimate.
- ``glm-5.2``: OpenRouter ``z-ai/glm-5.2`` = 1,048,576.
- ``deepseek-v4-flash``: OpenRouter ``deepseek/deepseek-v4-flash`` FAQ
  ("1,048,576 token context window") for the canonical API name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# L2: the subject of this file is an artifact outside `src/kitty` Python code,
# or a structural scan of source text -- two things edited separately that must
# agree. It gates pull requests exactly as before, in the `l1 or l2` job; the
# marker records which half of that expression it answers to, and keeps a
# source-text scan out of the L1 set that mutation testing will judge.
pytestmark = pytest.mark.l2

EXPECTED_CATALOG: dict[str, int] = {
    "qwen3.8-max": 1_000_000,
    "MiniMax-M3": 1_048_576,
    "glm-5.3": 1_048_576,
    "glm-5.3-flash": 1_048_576,
    "glm-5.2": 1_048_576,
    "deepseek-v4-flash": 1_048_576,
}

# The catalog is provider-agnostic: the same cloud-native model name is used
# across Qwen Cloud, MiniMax Coding Plan, Z.AI coding plan, DeepSeek cloud,
# and custom Anthropic-compatible endpoints.
PROVIDER_VARIANTS = (
    "qwen_cloud",
    "minimax_coding_plan",
    "zai_coding",
    "deepseek",
    "custom_anthropic",
)


@pytest.fixture(autouse=True)
def _isolate_module_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Clear the module-level lru_caches and isolate the remote cache path.

    The tests read the real packaged catalog and metadata files, so those
    paths are not patched. The remote-synced cache path is pointed at a
    nonexistent tmp path so a real synced cache on this machine cannot leak
    into the packaged-catalog assertions.
    """
    import kitty.providers.model_context as mc

    monkeypatch.setattr(
        mc,
        "REMOTE_OVERRIDES_CACHE_PATH",
        tmp_path / "remote-cache" / "model_context_overrides.json",
    )
    mc._load_metadata.cache_clear()
    mc._load_overrides.cache_clear()
    yield
    mc._load_metadata.cache_clear()
    mc._load_overrides.cache_clear()


class TestPackagedCatalogEntries:
    """Every pinned model resolves to its verified window for any provider."""

    @pytest.mark.parametrize("provider", PROVIDER_VARIANTS)
    @pytest.mark.parametrize("model,expected", sorted(EXPECTED_CATALOG.items()))
    def test_model_resolves_to_verified_window(self, provider, model, expected):
        from kitty.providers.model_context import get_model_context_tokens

        assert get_model_context_tokens(provider, model, None) == expected


class TestPackagedCatalogSuffixSpelling:
    """Vendor-prefixed spellings resolve via the suffix rule to the same value."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("qwen/qwen3.8-max", 1_000_000),
            ("minimax/MiniMax-M3", 1_048_576),
            ("z-ai/glm-5.3", 1_048_576),
            ("z-ai/glm-5.3-flash", 1_048_576),
            ("z-ai/glm-5.2", 1_048_576),
            ("deepseek/deepseek-v4-flash", 1_048_576),
        ],
    )
    def test_vendor_prefixed_model_matches_catalog(self, model, expected):
        from kitty.providers.model_context import get_model_context_tokens

        assert get_model_context_tokens("openrouter", model, None) == expected


class TestPackagedCatalogPriority:
    """The packaged catalog outranks provider_config and the default.

    Catalog-vs-metadata precedence cannot be distinguished with the shipped
    catalog because every pinned value currently equals its metadata value;
    that precedence is pinned with synthetic data in ``test_model_context.py``
    (``TestLocalOverrides.test_override_beats_metadata_exact_and_suffix``).
    """

    @pytest.mark.parametrize("model,expected", sorted(EXPECTED_CATALOG.items()))
    def test_catalog_beats_provider_config(self, model, expected):
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(
            provider="qwen_cloud",
            model=model,
            provider_config={"context_window": 12345},
        )
        assert result == expected

    def test_unknown_model_still_falls_back_to_default(self):
        from kitty.providers.model_context import DEFAULT_CONTEXT_TOKENS, get_model_context_tokens

        result = get_model_context_tokens("openai", "totally-unknown-model-xyz", None)
        assert result == DEFAULT_CONTEXT_TOKENS


def _metadata_window(catalog_id: str) -> int:
    """Return the packaged metadata table's context length for an exact id.

    Expectations are read from the catalog rather than hardcoded: the table is
    regenerated from OpenRouter, so a pinned literal would turn a routine
    refresh into a failure that reads as "the KBR-151 fix broke".

    Args:
        catalog_id: An exact, lowercase id in ``model_metadata.json``.

    Returns:
        The context length in tokens.
    """
    import json

    from kitty.providers.model_context import _METADATA_PATH

    models = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
    return next(m["context_length"] for m in models if m["id"].lower() == catalog_id)


class TestProviderPrefixedProfileModels:
    """KBR-151: a profile model carrying a provider prefix resolves normally.

    ``OpenCodeGoAdapter`` and ``AzureOpenAIAdapter`` both define
    ``normalize_model_name`` precisely because users write their models that
    way, and OpenCode Go's own documentation names its models with the prefix.
    Before the fix these fell through to ``DEFAULT_CONTEXT_TOKENS``: on Azure
    that is 200,000 assumed against 128,000 real, so the bridge does not
    compact when it should and the upstream rejects the oversized request.
    """

    @pytest.mark.parametrize(
        ("provider", "prefixed", "bare"),
        [
            ("opencode_go", "opencode/minimax-m2.5", "minimax-m2.5"),
            ("opencode_go", "opencode/glm-5", "glm-5"),
            ("azure", "azure/gpt-4o", "gpt-4o"),
            ("azure", "Azure/GPT-4o", "gpt-4o"),
        ],
    )
    def test_prefixed_and_bare_resolve_alike(self, provider, prefixed, bare):
        from kitty.providers.model_context import DEFAULT_CONTEXT_TOKENS, get_model_context_tokens

        resolved = get_model_context_tokens(provider, prefixed, None)
        assert resolved == get_model_context_tokens(provider, bare, None)
        # Equality alone would pass with both sides sitting on the default,
        # which is the bug. The point is that a real window was found.
        assert resolved != DEFAULT_CONTEXT_TOKENS

    def test_balancing_minimum_sees_the_real_azure_window(self):
        """The pool's budget is its smallest member's, prefix or not.

        ``get_balancing_min_context_tokens`` takes a ``min()``, so a member
        that silently resolved the 200,000 default hid the Azure member's real
        128,000 and let the whole pool over-send.
        """
        from kitty.providers.model_context import get_balancing_min_context_tokens

        backends = [
            ("azure", "azure/gpt-4o", None),
            ("opencode_go", "opencode/minimax-m2.5", None),
        ]
        assert get_balancing_min_context_tokens(backends) == _metadata_window("openai/gpt-4o")


class TestNormalizeModelNameIsNotTheRightTransform:
    """KBR-151 rejected routing this lookup through ``normalize_model_name``.

    That method translates a model name into the **provider's** dialect; the
    catalogs are keyed in OpenRouter's. Measured over all 23 providers, doing
    so would have fixed 5,336 lookups and broken 754 — 395 on ``anthropic``,
    whose ``normalize_model_name`` replaces dots with hyphens, and 359 on
    ``vertex``, whose version prepends ``google/``.

    These two tests pin the windows that change would have destroyed, so it
    cannot be reintroduced silently.
    """

    def test_dotted_anthropic_model_keeps_its_window(self):
        """``claude-sonnet-4-5`` is absent from the catalog; the dotted id is not."""
        from kitty.providers.model_context import DEFAULT_CONTEXT_TOKENS, get_model_context_tokens

        resolved = get_model_context_tokens("anthropic", "claude-sonnet-4.5", None)
        assert resolved == _metadata_window("anthropic/claude-sonnet-4.5")
        assert resolved != DEFAULT_CONTEXT_TOKENS

    def test_bare_vertex_model_keeps_its_window(self):
        """Prepending ``google/`` would miss the exact id and the suffix rule."""
        from kitty.providers.model_context import DEFAULT_CONTEXT_TOKENS, get_model_context_tokens

        resolved = get_model_context_tokens("vertex", "gemini-2.5-pro", None)
        assert resolved == _metadata_window("google/gemini-2.5-pro")
        assert resolved != DEFAULT_CONTEXT_TOKENS


class TestCatalogTailsAreUnique:
    """Tail-uniqueness is what makes the KBR-151 matcher unambiguous.

    Two keys sharing a final segment both match one query, so every lookup for
    that name would go ambiguous and fall to the default. The metadata table is
    regenerated from OpenRouter, so a collision would arrive through a data
    refresh with nothing else red.
    """

    def test_no_two_metadata_ids_share_a_final_segment(self):
        import json

        from kitty.providers.model_context import _METADATA_PATH, _colliding_keys

        models = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
        ids = [m["id"].lower() for m in models if isinstance(m, dict) and "id" in m]
        assert _colliding_keys(ids) == [], (
            "two catalog ids share a final segment; every lookup for that model name "
            "will now resolve DEFAULT_CONTEXT_TOKENS"
        )

    def test_no_two_override_keys_share_a_final_segment(self):
        import json

        from kitty.providers.model_context import _OVERRIDES_PATH, _colliding_keys

        keys = [k.lower() for k in json.loads(_OVERRIDES_PATH.read_text(encoding="utf-8"))]
        assert _colliding_keys(keys) == []
