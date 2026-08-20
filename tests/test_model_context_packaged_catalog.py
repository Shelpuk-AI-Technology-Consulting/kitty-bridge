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
- ``glm-5.2``: OpenRouter ``z-ai/glm-5.2`` = 1,048,576.
- ``deepseek-v4-flash``: OpenRouter ``deepseek/deepseek-v4-flash`` FAQ
  ("1,048,576 token context window") for the canonical API name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

EXPECTED_CATALOG: dict[str, int] = {
    "qwen3.8-max": 1_000_000,
    "MiniMax-M3": 1_048_576,
    "glm-5.3": 1_048_576,
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
