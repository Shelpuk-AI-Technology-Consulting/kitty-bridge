"""KBR-322: the subscription wizards' suggested default models.

The three interactive wizards (``kitty setup``, ``kitty profile`` and
``kitty auth openai``) each used to declare their own copy of the
``openai_subscription`` default, and all three kept offering ``gpt-5.3-codex``
after OpenAI retired it from the ChatGPT sign-in path. Accepting it produced a
profile whose first request failed with model-not-found.

These tests hold the single declaration in place: no wizard may carry its own
literal again, and the declared default may not be an id OpenAI has retired
from that path. The behavioural pins (an empty answer yields the default, and
the prompt names it) live with each wizard's own tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kitty.profiles.schema import PROVIDER_LIST, SUBSCRIPTION_DEFAULT_MODELS

_CLI = Path(__file__).resolve().parents[1] / "src" / "kitty" / "cli"
_WIZARDS = ("setup_cmd.py", "profile_cmd.py", "auth_cmd.py")

# Ids OpenAI has retired from Codex-with-ChatGPT-sign-in (learn.chatgpt.com/docs/models).
# Extend this when the next retirement lands, in the same change that moves the default.
_RETIRED_FROM_CHATGPT_SIGN_IN = frozenset(
    {"gpt-5.3-codex", "gpt-5.3-codex-spark", "gpt-5.4", "gpt-5.4-mini", "gpt-5.5"}
)


def test_openai_subscription_default_is_currently_served() -> None:
    assert SUBSCRIPTION_DEFAULT_MODELS["openai_subscription"] == "gpt-6-sol"
    assert SUBSCRIPTION_DEFAULT_MODELS["openai_subscription"] not in _RETIRED_FROM_CHATGPT_SIGN_IN


def test_anthropic_subscription_default_is_unchanged() -> None:
    # Out of scope for KBR-322: previous-generation but still served.
    assert SUBSCRIPTION_DEFAULT_MODELS["anthropic_subscription"] == "claude-sonnet-4-6"


def test_defaults_are_declared_only_for_real_subscription_providers() -> None:
    assert set(SUBSCRIPTION_DEFAULT_MODELS) == {"openai_subscription", "anthropic_subscription"}
    assert set(SUBSCRIPTION_DEFAULT_MODELS) <= set(PROVIDER_LIST)


@pytest.mark.parametrize("wizard", _WIZARDS)
def test_wizard_reads_the_shared_default(wizard: str) -> None:
    source = (_CLI / wizard).read_text(encoding="utf-8")
    assert "SUBSCRIPTION_DEFAULT_MODELS" in source
    for model in SUBSCRIPTION_DEFAULT_MODELS.values():
        assert f'"{model}"' not in source, f"{wizard} hard-codes {model!r} instead of reading the shared default"
    for retired in _RETIRED_FROM_CHATGPT_SIGN_IN:
        assert retired not in source, f"{wizard} still names the retired {retired!r}"
