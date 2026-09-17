"""KBR-157 — every base-URL-reading adapter composes a pasted endpoint to itself.

Six adapters read ``provider_config["base_url"]``: ``custom_openai`` and
``custom_anthropic`` (KBR-134) and ``azure`` (KBR-153) already strip a redundant
trailing endpoint, but ``ollama``, ``minimax`` and ``ollama_cloud`` do not —
so a user who pastes the full endpoint their provider's documentation shows
gets the doubled path KBR-134 was filed about.

This file is L1 by the path default in ``tests/layers.py``: the rule is a
pure function of a base URL string, which is the lowest layer that can prove
it (``TEST_SUITE.md`` §2.2).  The structural scan is the shape KBR-143's
``tests/test_upstream_url_single_rule.py`` uses — a textual/AST discovery of
the module set, pinned against an expected list — but it lives in the same
L1 file here because its only consumer is the parametrised behavioural
case below: a separate L2 file would have to duplicate the adapter registry
to stay in lockstep with the parametrise.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from kitty.providers.azure import AzureOpenAIAdapter
from kitty.providers.base import ProviderAdapter
from kitty.providers.custom_anthropic import CustomAnthropicAdapter
from kitty.providers.custom_openai import CustomOpenAIAdapter
from kitty.providers.minimax import MiniMaxAdapter
from kitty.providers.ollama import OllamaAdapter
from kitty.providers.ollama_cloud import OllamaCloudAdapter

_SRC = Path(__file__).resolve().parents[2] / "src" / "kitty" / "providers"


# The set the code is required to keep in sync — the adapter this ticket is
# closing the gap on, the two KBR-134 covered, and the KBR-153 sixth.  A
# seventh adapter reading `base_url` fails the pin below until it is added
# here AND gets its own behavioural case (the round-trip parametrise).
_EXPECTED_BASE_URL_READERS = frozenset(
    {
        "azure.py",
        "custom_anthropic.py",
        "custom_openai.py",
        "minimax.py",
        "ollama.py",
        "ollama_cloud.py",
    }
)


def _is_provider_config_or_default(node: ast.expr) -> bool:
    """Return whether ``node`` is ``provider_config`` or ``provider_config or {}``.

    Three spellings of the same intent appear in the codebase:
    ``provider_config["base_url"]``, ``provider_config.get("base_url")`` and
    ``(provider_config or {}).get("base_url")``.  The third is wrapped in a
    ``BoolOp(Or)`` over a ``Name`` and a ``Dict`` literal, so the helper
    accepts either the bare name or that wrapped form.

    Args:
        node: The AST expression node to inspect.

    Returns:
        True if the node refers to ``provider_config`` (optionally with a
        truthy-defaulting ``or {}`` wrapper).
    """
    if isinstance(node, ast.Name) and node.id == "provider_config":
        return True
    return (
        isinstance(node, ast.BoolOp)
        and isinstance(node.op, ast.Or)
        and len(node.values) == 2
        and isinstance(node.values[0], ast.Name)
        and node.values[0].id == "provider_config"
        and isinstance(node.values[1], ast.Dict)
        and not node.values[1].keys
        and not node.values[1].values
    )


def _reads_base_url_from_provider_config(tree: ast.AST) -> bool:
    """Return whether ``tree`` reads ``base_url`` from ``provider_config`` as code.

    A textual scan would match a docstring mention (``providers/base.py``'s
    ``requires_custom_url`` docstring names the key in prose), so the scan is
    AST-based: a match is a ``Subscript`` whose value resolves to
    ``provider_config`` (with an optional ``or {}`` wrapper) and whose slice
    is the constant ``"base_url"``, or a ``Call`` whose function is the
    ``.get`` of the same expression with the constant ``"base_url"`` as its
    first positional argument.  Docstrings and comments are not in the AST
    as subscript or call nodes, so neither is matched.

    The match is name-based: a future adapter that aliases
    ``cfg = provider_config; cfg["base_url"]`` would not be caught, because
    the scan only matches the parameter name directly.  Today's adapters all
    use the parameter name, so the blind spot is hypothetical — it is recorded
    here rather than widened so the scan stays precise.

    Args:
        tree: The parsed module AST to walk.

    Returns:
        True if any matching read appears in the tree.
    """
    for node in ast.walk(tree):
        # provider_config["base_url"]
        if (
            isinstance(node, ast.Subscript)
            and _is_provider_config_or_default(node.value)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "base_url"
        ):
            return True
        # provider_config.get("base_url")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_provider_config_or_default(node.func.value)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "base_url"
        ):
            return True
    return False


def _modules_reading_base_url() -> set[str]:
    """Return adapter file names under ``src/kitty/providers/`` that read ``base_url``.

    The scan walks the AST of each module for actual code reads (subscript or
    ``.get``), so a docstring mention of the key is not a match.  The same
    containment shape KBR-143's ``tests/test_upstream_url_single_rule.py``
    uses — a seventh adapter reading ``base_url`` fails the structural pin
    until it is registered here.

    Returns:
        The set of module file names.
    """
    found: set[str] = set()
    for path in sorted(_SRC.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _reads_base_url_from_provider_config(tree):
            found.add(path.name)
    return found


# ── Structural pin ────────────────────────────────────────────────────────


def test_the_scan_finds_its_known_positives():
    """The structural scan must find the readers it exists to check, or it proves nothing.

    Per ``TEST_SUITE.md`` §6.2, a structural guard asserts its own scan works.
    A regex that matched nothing would make the reverse-direction assertion
    below vacuously true.
    """
    assert _modules_reading_base_url() >= _EXPECTED_BASE_URL_READERS


def test_no_new_module_reads_base_url_unnoticed():
    """The set of base-URL readers is pinned in both directions.

    A seventh adapter joining the set is how the silent-gap defect would come
    back, and it fails here until it is added to ``_EXPECTED_BASE_URL_READERS``
    — which is the moment to give it a behavioural case (the round-trip
    parametrise below).  Same containment shape
    ``tests/test_upstream_url_single_rule.py::test_no_new_module_resolves_a_base_url_unnoticed``
    uses for ``build_base_url`` consumers.

    The last assertion ties the pin to the parametrise: a new reader added to
    the expected set without an ``_ADAPTERS`` row would leave the suite green
    with no behavioural case, so the two cannot drift apart silently.
    """
    assert _modules_reading_base_url() == _EXPECTED_BASE_URL_READERS
    assert {row[0] for row in _ADAPTERS} == {name.removesuffix(".py") for name in _EXPECTED_BASE_URL_READERS}


# ── Behavioural round-trip ────────────────────────────────────────────────


def _compose(adapter: ProviderAdapter, provider_config: dict, model: str) -> str:
    """Compose the upstream URL through the adapter's own resolution.

    Adapters with custom transport own their composition end-to-end and do not
    expose ``build_base_url`` as a public composition site; the rest compose
    through ``build_base_url`` plus the shared ``compose_upstream_url`` helper.
    The branch is the only asymmetry between the two paths, and it is the
    asymmetry the ticket's decision 2 records.

    Args:
        adapter: The provider adapter under test.
        provider_config: The profile's provider configuration, carrying a
            ``base_url`` that the test has set to the full documented endpoint.
        model: The model the bridge would route the request to.  Most adapters
            ignore it; ``azure`` encodes it as the deployment segment of the
            path (register row P20).

    Returns:
        The address the adapter would request.
    """
    if adapter.use_custom_transport:
        # The seam is the test's subject: ``_build_url`` is ``OllamaCloudAdapter``'s
        # sole composition site, and the parametrise exercises it directly.
        return adapter._build_url(provider_config)
    base = adapter.build_base_url(provider_config)
    path = adapter.get_upstream_path(adapter.normalize_model_name(model))
    return adapter.compose_upstream_url(base, path)


# Per-adapter fixtures: the adapter class, the documented endpoint a user
# would paste, the corresponding API root the round-trip test asserts was
# returned by `build_base_url`, and the model whose path the bridge composes.
# ``azure`` is the one row whose model is part of the path; the others pass
# ``some-model`` for symmetry and the helper ignores it for them.
_ADAPTERS: tuple[tuple[str, type[ProviderAdapter], str, str, str], ...] = (
    (
        "custom_openai",
        CustomOpenAIAdapter,
        "https://gw.example/v1/chat/completions",
        "https://gw.example/v1",
        "some-model",
    ),
    (
        "custom_anthropic",
        CustomAnthropicAdapter,
        "https://gw.example/v1/messages",
        "https://gw.example/v1",
        "some-model",
    ),
    (
        "ollama",
        OllamaAdapter,
        "http://ollama.example/v1/chat/completions",
        "http://ollama.example/v1",
        "some-model",
    ),
    (
        "minimax",
        MiniMaxAdapter,
        "https://api.minimax.io/v1/chat/completions",
        "https://api.minimax.io/v1",
        "some-model",
    ),
    (
        "ollama_cloud",
        OllamaCloudAdapter,
        "https://ollama.example/api/chat",
        "https://ollama.example",
        "some-model",
    ),
    (
        "azure",
        AzureOpenAIAdapter,
        # Microsoft's documented endpoint, with the api-version KBR-153 pins.
        "https://res.openai.azure.com/openai/deployments/d/chat/completions?api-version=2024-10-21",
        "https://res.openai.azure.com",
        "d",
    ),
)


@pytest.mark.parametrize(
    ("name", "_adapter_cls", "documented_endpoint", "_api_root", "model"),
    _ADAPTERS,
    ids=[row[0] for row in _ADAPTERS],
)
def test_pasted_documented_endpoint_composes_to_itself(
    name: str,
    _adapter_cls: type[ProviderAdapter],
    documented_endpoint: str,
    _api_root: str,
    model: str,
):
    """A base URL pasted as the full documented endpoint composes to that endpoint.

    The reported defect: ``base_url + upstream_path`` doubled the path and the
    upstream returned a 404.  This is the acceptance criterion KBR-157 names —
    "a base URL pasted as the full documented endpoint composes to that endpoint
    rather than to a doubled path" — proved at the lowest layer that can carry
    it.

    Args:
        name: The adapter's registry key, surfaced as the parametrise id.
        _adapter_cls: The adapter class to instantiate.
        documented_endpoint: The full endpoint URL the user pastes.
        _api_root: The API root that ``build_base_url`` is expected to return
            after stripping; unused by the assertion but documented for the
            left-alone case below.
        model: The model the bridge would route the request to.
    """
    adapter = _adapter_cls()
    cfg = {"base_url": documented_endpoint}

    assert _compose(adapter, cfg, model) == documented_endpoint


@pytest.mark.parametrize(
    ("name", "_adapter_cls", "api_root", "model"),
    [(row[0], row[1], row[3], row[4]) for row in _ADAPTERS],
    ids=[row[0] for row in _ADAPTERS],
)
def test_an_api_root_is_returned_unchanged(
    name: str,
    _adapter_cls: type[ProviderAdapter],
    api_root: str,
    model: str,
):
    """An unsuffixed base URL is returned unchanged, so the round-trip is not vacuous.

    The strip is "remove one trailing copy of the endpoint path if present";
    the right-untouched case is what makes that contract falsifiable — a
    "strip everything" mutation would still pass the round-trip on the
    documented endpoint, but would break here.

    Args:
        name: The adapter's registry key, surfaced as the parametrise id.
        _adapter_cls: The adapter class to instantiate.
        api_root: The base URL the adapter should leave alone.
        model: The model the bridge would route the request to.
    """
    adapter = _adapter_cls()
    cfg = {"base_url": api_root}

    if adapter.use_custom_transport:
        # See ``_compose`` for why we reach into ``_build_url`` here.
        composed = adapter._build_url(cfg)
        # The composed URL is the root plus the endpoint path, exactly once.
        assert composed == adapter.compose_upstream_url(api_root, adapter.get_upstream_path(model))
        return

    assert adapter.build_base_url(cfg) == api_root


# ── MiniMaxAdapter-specific gates ─────────────────────────────────────────


class TestMiniMaxStripOrdering:
    """The strip must run after the region switch and the scheme check.

    The ticket's decision 1 is recorded in ``.requirements/.../REQUIREMENTS.md``:
    moving the strip in front of either gate would let a region switch eat a
    pasted endpoint that was meant for the global URL, or turn a malformed
    scheme into a ``KeyError`` from ``urlsplit`` inside the helper.  These
    cases pin the order.
    """

    def setup_method(self) -> None:
        self.adapter = MiniMaxAdapter()

    def test_region_wins_over_pasted_base_url(self):
        """``region == "cn"`` still wins over a pasted global base URL.

        A user who toggles the region switch and forgets to clear the base
        URL must not have the strip discard the region-switched value.
        """
        assert (
            self.adapter.build_base_url(
                {"region": "cn", "base_url": "https://api.minimax.io/v1/chat/completions"}
            )
            == "https://api.minimaxi.com/v1"
        )

    def test_malformed_scheme_still_raises_before_strip(self):
        """The strip never sees a malformed scheme.

        A ``ftp://`` URL has to be rejected with the adapter's own message
        shape, not by an internal ``urlsplit`` error inside the helper.
        """
        with pytest.raises(ValueError, match="http"):
            self.adapter.build_base_url({"base_url": "ftp://api.minimax.io/v1/chat/completions"})

    def test_cn_url_form_is_a_valid_paste_target(self):
        """A pasted CN endpoint strips to the CN URL root, not the global one."""
        assert (
            self.adapter.build_base_url({"base_url": "https://api.minimaxi.com/v1/chat/completions"})
            == "https://api.minimaxi.com/v1"
        )
