"""Provider-level tests: error bodies ride ``ProviderError.upstream_body``.

Live diagnosis 2026-10-29 (pool ``my``): the OpenAI subscription adapter's
``map_error`` folded only the upstream ``message`` field, so an empty one
rendered bare and the structured body was dropped. The carriage is now a
shared base helper, and the custom-transport siblings that fold ``message``
the same way (Bedrock, Ollama Cloud) carry it too — a later ``map_error``
edit that drops the attach fails here instead of shipping the silent
incident again.

**Layer.** Collects as ``l1`` alongside the other provider unit tests.
"""

from __future__ import annotations

from kitty.providers.bedrock import BedrockAdapter
from kitty.providers.ollama_cloud import OllamaCloudAdapter
from kitty.providers.openai_subscription import OpenAISubscriptionAdapter


class TestBedrockBodyCarriage:
    """The Bedrock adapter carries the bounded body (review 2026-10-29)."""

    def test_map_error_attaches_bounded_body(self) -> None:
        """A 400 body rides the error as ``upstream_body``."""
        err = BedrockAdapter().map_error(400, {"message": ""})
        assert err.upstream_body == '{"message": ""}'

    def test_map_error_with_empty_body_carries_none(self) -> None:
        """An empty body carries no diagnostic."""
        assert BedrockAdapter().map_error(500, {}).upstream_body is None


class TestOllamaCloudBodyCarriage:
    """The Ollama Cloud adapter carries the bounded body (review 2026-10-29)."""

    def test_map_error_attaches_bounded_body(self) -> None:
        """A 400 body rides the error as ``upstream_body``."""
        err = OllamaCloudAdapter().map_error(400, {"error": {"message": ""}})
        assert err.upstream_body == '{"error": {"message": ""}}'

    def test_map_error_with_empty_body_carries_none(self) -> None:
        """An empty body carries no diagnostic."""
        assert OllamaCloudAdapter().map_error(500, {}).upstream_body is None

    def test_map_error_with_non_dict_body_carries_none(self) -> None:
        """The non-dict branch still maps a usable error, without a body."""
        err = OllamaCloudAdapter().map_error(502, "bad gateway")
        assert "bad gateway" in str(err)
        assert err.upstream_body is None


class TestOpenAISubscriptionBodyCarriage:
    """The OpenAI subscription adapter keeps its carriage through the helper."""

    def test_map_error_attaches_bounded_body(self) -> None:
        """A 400 body rides the error as ``upstream_body``."""
        err = OpenAISubscriptionAdapter().map_error(400, {"error": {"type": "invalid_request", "message": ""}})
        assert err.upstream_body == '{"error": {"type": "invalid_request", "message": ""}}'

    def test_map_error_truncates_to_500_chars(self) -> None:
        """The carriage is bounded so no consumer can log it unbounded."""
        err = OpenAISubscriptionAdapter().map_error(400, {"detail": "x" * 5000})
        assert err.upstream_body is not None
        assert len(err.upstream_body) == 500
