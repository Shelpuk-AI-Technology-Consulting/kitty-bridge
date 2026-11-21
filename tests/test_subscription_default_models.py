"""KBR-322: the subscription wizards' suggested default models.

The three interactive wizards (``kitty setup``, ``kitty profile`` and
``kitty auth openai``) each used to declare their own copy of the
``openai_subscription`` default, and all three kept offering ``gpt-5.3-codex``
after OpenAI retired it from the ChatGPT sign-in path. Accepting it produced a
profile whose first request failed with model-not-found.

These tests hold the single declaration in place: defaults are declared for
exactly the two subscription providers; no wizard may carry its own literal
again; the declared default may not be an id OpenAI has retired from that path;
README must name the declared defaults; and the subscription adapter's fallback
model must equal the wizards' default and name no retired id. The behavioural
pins (an empty answer yields the default, and the prompt names it) live with
each wizard's own tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kitty.profiles.schema import PROVIDER_LIST, SUBSCRIPTION_DEFAULT_MODELS

_ROOT = Path(__file__).resolve().parents[1]
_CLI = _ROOT / "src" / "kitty" / "cli"
_README = _ROOT / "README.md"
_WIZARDS = ("setup_cmd.py", "profile_cmd.py", "auth_cmd.py")

# Ids OpenAI has retired from Codex-with-ChatGPT-sign-in (learn.chatgpt.com/docs/models).
# Extend this when the next retirement lands, in the same change that moves the default.
_RETIRED_FROM_CHATGPT_SIGN_IN = frozenset(
    {"gpt-5.3-codex", "gpt-5.3-codex-spark", "gpt-5.4", "gpt-5.4-mini", "gpt-5.5"}
)


def test_openai_subscription_default_is_currently_served() -> None:
    """The declared OpenAI default is not an id retired from the ChatGPT sign-in path."""
    assert SUBSCRIPTION_DEFAULT_MODELS["openai_subscription"] == "gpt-6-sol"
    assert SUBSCRIPTION_DEFAULT_MODELS["openai_subscription"] not in _RETIRED_FROM_CHATGPT_SIGN_IN


def test_anthropic_subscription_default_is_unchanged() -> None:
    """The Anthropic default stays ``claude-sonnet-4-6``."""
    # Out of scope for KBR-322: previous-generation but still served.
    assert SUBSCRIPTION_DEFAULT_MODELS["anthropic_subscription"] == "claude-sonnet-4-6"


def test_defaults_are_declared_only_for_real_subscription_providers() -> None:
    """Defaults exist for exactly the two subscription providers, both real provider ids."""
    assert set(SUBSCRIPTION_DEFAULT_MODELS) == {"openai_subscription", "anthropic_subscription"}
    assert set(SUBSCRIPTION_DEFAULT_MODELS) <= set(PROVIDER_LIST)


@pytest.mark.parametrize("wizard", _WIZARDS)
def test_wizard_reads_the_shared_default(wizard: str) -> None:
    """Each wizard reads the shared declaration and names no default id of its own."""
    source = (_CLI / wizard).read_text(encoding="utf-8")
    assert "SUBSCRIPTION_DEFAULT_MODELS" in source
    for model in SUBSCRIPTION_DEFAULT_MODELS.values():
        # Any appearance of the id, in either quote style or inside a longer
        # string, means the wizard carries its own copy again.
        assert model not in source, f"{wizard} hard-codes {model!r} instead of reading the shared default"
    for retired in _RETIRED_FROM_CHATGPT_SIGN_IN:
        assert retired not in source, f"{wizard} still names the retired {retired!r}"


def test_readme_names_the_declared_defaults() -> None:
    """README's wizard examples and model-id paragraph state the declared defaults, not stale ones."""
    readme = _README.read_text(encoding="utf-8")
    openai_default = SUBSCRIPTION_DEFAULT_MODELS["openai_subscription"]
    assert f"Model name (default: {openai_default})" in readme, "the Astra example's prompt must show the default"
    assert f"The wizards now suggest `{openai_default}`" in readme
    assert f"? Model: {openai_default}" in readme, "the setup example must show the declared default"
    assert (
        f"the wizard's suggested default, `{SUBSCRIPTION_DEFAULT_MODELS['anthropic_subscription']}`" in readme
    ), "README's Claude-default sentence must name the declared default"
    for line in readme.splitlines():
        is_model_prompt = "Model" in line and "(default: " in line
        assert not is_model_prompt or any(m in line for m in SUBSCRIPTION_DEFAULT_MODELS.values()), (
            f"README shows a wizard default the code does not declare: {line.strip()!r}"
        )


def test_subscription_adapter_falls_back_to_the_shared_default() -> None:
    """The ChatGPT-subscription adapter's fallback model is the wizards' default, never a retired id."""
    from kitty.providers import openai_subscription

    assert SUBSCRIPTION_DEFAULT_MODELS["openai_subscription"] == openai_subscription._FALLBACK_MODEL
    source = Path(openai_subscription.__file__).read_text(encoding="utf-8")
    for retired in _RETIRED_FROM_CHATGPT_SIGN_IN:
        assert retired not in source, f"openai_subscription.py names the retired {retired!r}"
