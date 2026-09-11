"""KBR-134 — the documentation and the prompts must not teach the broken URL form.

The reporter did exactly what Mistral's own documentation shows: he pasted the
full endpoint.  Normalisation now rescues that, but the places where a user reads
or types the value are what stop the mistake being made at all, and they are
edited separately from the code that consumes it.  This is the §6.2 case — two
artifacts that must agree, both readable statically.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

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
