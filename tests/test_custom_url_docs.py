"""KBR-134 — the documentation and the prompts must not teach the broken URL form.

The reporter did exactly what Mistral's own documentation shows: he pasted the
full endpoint.  Normalisation now rescues that, but the places where a user reads
or types the value are what stop the mistake being made at all, and they are
edited separately from the code that consumes it.  This is the §6.2 case — two
artifacts that must agree, both readable statically.

KBR-9 extends this to ``custom_anthropic`` — the README never documented the
provider at all, and the same rule (base URL ends at the API root, paste-the-full
form is rescued) applies.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kitty.providers.custom_anthropic import CustomAnthropicAdapter
from kitty.providers.custom_openai import CustomOpenAIAdapter

pytestmark = pytest.mark.l2

_ROOT = Path(__file__).resolve().parents[1]
_README = _ROOT / "README.md"
_PROMPT_FILES = (
    _ROOT / "src" / "kitty" / "cli" / "setup_cmd.py",
    _ROOT / "src" / "kitty" / "cli" / "profile_cmd.py",
)

# The "Common endpoints" table rows: | Service | `URL` |
_TABLE_ROW = re.compile(r"^\|\s*[^|]+\|\s*`(https?://[^`]+)`\s*\|", re.MULTILINE)


def _documented_base_urls() -> list[str]:
    """Return every base URL the README's common-endpoints table lists.

    Returns:
        The URLs, in document order.
    """
    section = _README.read_text(encoding="utf-8").split("**Common endpoints:**", 1)
    assert len(section) == 2, "README no longer has a '**Common endpoints:**' table"
    return _TABLE_ROW.findall(section[1].split("\n\n## ", 1)[0])


def test_scan_finds_the_known_positives():
    """The scan must find the rows it exists to check, or it is a no-op.

    Per `TEST_SUITE.md` §6.2, a structural guard asserts its own scan works.
    Without this, a README reshuffle that broke the regex would leave every
    assertion below vacuously true.
    """
    urls = _documented_base_urls()

    assert len(urls) >= 5, f"expected the endpoints table to have several rows, found {urls}"
    assert "https://api.deepseek.com/v1" in urls


def test_mistral_is_documented():
    """The service that produced KBR-134 is listed, at its API root."""
    assert "https://api.mistral.ai/v1" in _documented_base_urls()


def test_no_documented_url_teaches_the_broken_form():
    """Every listed URL is already an API root — normalising it changes nothing.

    This is the check that matters: a future row pasted as a full endpoint would
    be an instruction to reproduce the reported bug.
    """
    adapter = CustomOpenAIAdapter()

    for url in _documented_base_urls():
        assert adapter.build_base_url({"base_url": url}) == url, url


def test_readme_states_the_rule():
    """The section says where the base URL ends, not only what one looks like."""
    text = _README.read_text(encoding="utf-8")

    assert "The base URL ends at the API root" in text


def test_readme_documents_the_azure_endpoint_it_now_accepts():
    """KBR-143 — the Azure form works, so the README has to say so.

    The ticket's commercial case is that a customer following Microsoft's
    documentation could not be served at all.  A fix nobody is told about leaves that
    customer exactly where they were, so the documented endpoint is part of the
    deliverable rather than a footnote.
    """
    text = _README.read_text(encoding="utf-8")

    assert "openai.azure.com" in text, "README does not show the Azure endpoint form"
    assert "api-version" in text, "README does not show that the query is kept"


def test_the_documented_azure_endpoint_actually_composes_back_to_itself():
    """The documented URL is checked against the code, not just spell-checked.

    A README example that has drifted from the behaviour is worse than none: it is an
    instruction to reproduce a bug.  This reads the URL out of the README and runs it
    through normalisation and composition.
    """
    adapter = CustomOpenAIAdapter()
    match = re.search(r"^(https://<resource>\.openai\.azure\.com\S+)$", _README.read_text(encoding="utf-8"), re.M)
    assert match, "the Azure endpoint example is no longer on a line of its own"

    documented = match.group(1)
    composed = adapter.compose_upstream_url(adapter.build_base_url({"base_url": documented}), adapter.upstream_path)

    assert composed == documented


@pytest.mark.parametrize("path", _PROMPT_FILES, ids=lambda p: p.name)
def test_wizard_prompt_names_the_api_root(path: Path):
    """Both places a user types the value say what the value is.

    Args:
        path: The CLI module carrying the prompt.
    """
    text = path.read_text(encoding="utf-8")

    assert "API base URL" in text, f"{path.name} no longer prompts for a base URL"
    assert "the API root" in text, f"{path.name} prompt does not name the API root"


# ── KBR-9: the custom_anthropic provider is documented at all ───────────────


def _custom_anthropic_section_text() -> str:
    """Return the body of the README's Custom Anthropic-Compatible Provider section.

    Returns:
        The section body, from its heading to the next heading of any level.

    Raises:
        AssertionError: When the section is missing — the scan refusing to
            pass on a README that dropped the provider documentation.
    """
    match = re.search(
        r"### Custom Anthropic-Compatible Provider\s*\n(.*?)(?=\n### |\n## |\Z)",
        _README.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert match, "README no longer has a '### Custom Anthropic-Compatible Provider' section"
    return match.group(1)


def test_custom_anthropic_section_heading_exists():
    """The scan must find the section it exists to check, or it is a no-op.

    Per `TEST_SUITE.md` §6.2, a structural guard asserts its own scan works.
    Without this, a README reshuffle that removed the section would leave the
    assertions below vacuously true.
    """
    assert _custom_anthropic_section_text().strip(), "the section body is empty"


def test_custom_anthropic_provider_table_row_is_in_the_generic_block():
    """The Generic provider table lists the provider, scoped to that block.

    Scoped, because ``custom_anthropic`` already appears in unrelated README
    prose (the troubleshooting sections); a substring test over the whole
    document would pass for the wrong reason if the row were ever moved.
    """
    text = _README.read_text(encoding="utf-8")
    generic_block = text.split("**Generic:**", 1)[1].split("\n\n## ", 1)[0]

    assert re.search(
        r"\|\s*\*\*Custom Anthropic-Compatible\*\*\s*\|\s*`custom_anthropic`\s*\|",
        generic_block,
    ), "the Generic provider table does not list the custom_anthropic row"


def test_custom_anthropic_section_states_the_api_root_rule():
    """The section says where the base URL ends and shows the example root."""
    section = _custom_anthropic_section_text()

    assert "The base URL ends at the API root" in section
    assert "https://api.anthropic.com" in section
    assert "/v1/messages" in section, "the section no longer names the appended path"


def test_the_documented_custom_anthropic_url_composes_back_to_itself():
    """The documented base URL is checked against the code, not just spell-checked.

    Mirrors ``test_the_documented_azure_endpoint_actually_composes_back_to_itself``:
    a README example that has drifted from the behaviour is an instruction to
    reproduce a bug.  Normalisation + composition must reproduce the full
    upstream endpoint the README implies.
    """
    adapter = CustomAnthropicAdapter()
    documented = "https://api.anthropic.com"

    composed = adapter.compose_upstream_url(
        adapter.build_base_url({"base_url": documented}), adapter.upstream_path
    )

    assert composed == "https://api.anthropic.com/v1/messages"


def test_pasting_the_full_custom_anthropic_endpoint_is_rescued():
    """KBR-134's suffix strip is pinned, because the README now promises it.

    Without this assertion, a future refactor that removes
    ``_strip_endpoint_suffix`` from this adapter would leave the README's
    paste-the-full-endpoint note a fiction — documentation advertising
    behaviour that 404s.
    """
    adapter = CustomAnthropicAdapter()

    assert (
        adapter.build_base_url({"base_url": "https://api.anthropic.com/v1/messages"})
        == "https://api.anthropic.com"
    )
