"""Structural guards on the published OpenAPI document.

``.system_design/TEST_SUITE.md`` §6.2.1 (L2) · Jira **KBR-82** (T-G6).

``openapi/kitty-bridge.yaml`` is the artifact agents integrate against and
the fuzz target the schemathesis conformance run drives. Its biggest failure
mode is not being ill-formed (schemathesis loads it or refuses it), it is
**silently drifting away from the routes the bridge actually registers** —
the same shape of drift F2 (KBR-9) showed the README is subject to, and
which produced the KBR-9 endpoint-table guard. These tests pin the schema's
route set, its OpenAPI version, its self-description, and the 400 envelopes
KBR-82's added scope says must be documented — so a route dropped from the
schema, a dialect losing its error envelope, or a 500 slipping into the
contract is a red test, not a silently-unvalidated path.

Per §6.2, every guard here asserts its own scan finds a known positive:
the expected route set, titles, and reason values are named explicitly
below, so an empty or vacuous match fails loudly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from schemathesis.openapi import from_dict
from test_route_registration_matrix import _normalise_aiohttp_path

pytestmark = pytest.mark.l2

#: The schema file this module guards.
SCHEMA_PATH = Path(__file__).parent.parent / "openapi" / "kitty-bridge.yaml"

#: The eight routes `_register_routes` registers when `self._adapter is None`
#: (bridge mode), in OpenAPI path-template syntax — aiohttp regex converters
#: (`{model:.*}`) are normalised to `{model}` before comparison, mirroring
#: the KBR-9 endpoint-table guard's character-exact-match precedent.
BRIDGE_MODE_PATHS: frozenset[str] = frozenset(
    {
        "/healthz",
        "/stats",
        "/v1/models",
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1beta/models/{model}:generateContent",
        "/v1beta/models/{model}:streamGenerateContent",
    }
)

#: The method+path pairs the conformance fuzzer must reach. Used to pin the
#: schema's HTTP verbs so a dropped `post:` block does not reduce a route to
#: a documented-but-undrivable path.
BRIDGE_MODE_OPERATIONS: frozenset[str] = frozenset(
    {
        "GET /healthz",
        "GET /stats",
        "GET /v1/models",
        "POST /v1/chat/completions",
        "POST /v1/messages",
        "POST /v1/responses",
        "POST /v1beta/models/{model}:generateContent",
        "POST /v1beta/models/{model}:streamGenerateContent",
    }
)


def _schema_dict() -> dict:
    """Load and parse the published OpenAPI document.

    Returns:
        The parsed YAML document as a dict.

    Raises:
        FileNotFoundError: When the schema file is missing.
        yaml.YAMLError: When the file is not valid YAML.
    """
    with open(SCHEMA_PATH) as f:
        return yaml.safe_load(f)


class TestSchemaIsOpenAPI31:
    """The schema parses, is 3.1, and self-describes correctly."""

    def test_the_schema_declares_openapi_3_1(self) -> None:
        """Pin the version, because 3.0 and 3.1 differ in JSON-Schema dialect."""
        schema = _schema_dict()
        assert schema["openapi"] == "3.1.0"

    def test_the_schema_self_describes(self) -> None:
        """A structural assertion, not the unverifiable "no warnings" clause."""
        schema = _schema_dict()
        assert schema["info"]["title"] == "Kitty Bridge"
        assert "version" in schema["info"]

    def test_schemathesis_loads_the_schema(self) -> None:
        """The fuzz target loads cleanly — the conformance run's precondition."""
        schema = from_dict(_schema_dict())

        operations = set()
        for result in schema.get_all_operations():
            operation = result.ok()
            operations.add(f"{operation.method.upper()} {operation.path}")
        assert operations == BRIDGE_MODE_OPERATIONS


class TestSchemaPathsAgreeWithBridgeModeRoutes:
    """The schema's `paths` is exactly the bridge-mode route set."""

    def test_paths_match_bridge_mode_routes(self) -> None:
        """Both directions of agreement, after aiohttp-converter normalisation."""
        schema = _schema_dict()
        schema_paths = {_normalise_aiohttp_path(path) for path in schema["paths"]}

        missing = BRIDGE_MODE_PATHS - schema_paths
        extra = schema_paths - BRIDGE_MODE_PATHS
        assert not missing, f"routes registered by the bridge but missing from the schema: {sorted(missing)}"
        assert not extra, f"paths documented in the schema but not registered by the bridge: {sorted(extra)}"


class TestSchemaDocumentsThe400Envelopes:
    """The 400 envelopes each POST dialect deliberately returns are documented."""

    def test_every_post_route_documents_400(self) -> None:
        """A POST without a documented 400 is a POST whose error path is untested."""
        schema = _schema_dict()
        for path, methods in schema["paths"].items():
            post = methods.get("post")
            if post is None:
                continue
            assert "400" in post["responses"], f"POST {path} does not document a 400 response"

    def test_responses_error_documents_invalid_input_reason(self) -> None:
        """KBR-144's `reason: "invalid_input"` must stay in the contract."""
        schema = _schema_dict()
        envelope = schema["components"]["schemas"]["ResponsesError"]
        reason = envelope["properties"]["error"]["properties"]["reason"]
        assert "invalid_input" in reason.get("enum", []), "ResponsesError envelope lost the invalid_input reason"

    def test_responses_error_documents_compaction_failed_reason(self) -> None:
        """KBR-86's `reason: "compaction_failed"` must stay in the contract."""
        schema = _schema_dict()
        envelope = schema["components"]["schemas"]["ResponsesError"]
        reason = envelope["properties"]["error"]["properties"]["reason"]
        assert "compaction_failed" in reason.get("enum", []), (
            "ResponsesError envelope lost the compaction_failed reason"
        )

    def test_responses_error_reason_is_optional(self) -> None:
        """The JSON-/size-failure 400s carry no `reason`; the schema must admit that shape."""
        schema = _schema_dict()
        envelope = schema["components"]["schemas"]["ResponsesError"]
        inner = envelope["properties"]["error"]
        assert "reason" not in inner.get("required", []), (
            "ResponsesError made `reason` required — that would reject the "
            "JSON-/size-failure 400s the handler deliberately returns"
        )

    def test_gemini_error_documents_invalid_argument_status(self) -> None:
        """The Gemini dialect's 400 carries `status: "INVALID_ARGUMENT"`."""
        schema = _schema_dict()
        envelope = schema["components"]["schemas"]["GeminiError"]
        status = envelope["properties"]["error"]["properties"]["status"]
        assert "INVALID_ARGUMENT" in status.get("enum", [])


class TestSchemaDoesNotDocument500:
    """`not_a_server_error` is the contract; 500 must never enter it."""

    @pytest.mark.parametrize("path", sorted(BRIDGE_MODE_PATHS))
    def test_no_post_route_documents_500(self, path: str) -> None:
        """A documented 500 makes `status_code_conformance` accept the failure mode.

        Args:
            path: One of the bridge-mode routes.
        """
        schema = _schema_dict()
        methods = schema["paths"].get(path, {})
        for method in ("post", "get"):
            operation = methods.get(method)
            if operation is None:
                continue
            assert "500" not in operation["responses"], (
                f"{method.upper()} {path} documents a 500 — documenting it makes the "
                "conformance run accept the exact failure mode not_a_server_error exists to catch"
            )


class TestGeminiDocuments404ForUnroutableModel:
    """Schemathesis 4.x does not yet honour JSON-Schema ``pattern`` for path parameters.

    A small number of fuzzed model values (newlines, encoded bytes the
    aiohttp ``{model:.*}`` converter rejects) still produce 404s. The
    Gemini routes document 404 as the honest answer for those — every
    other route documents 200/400/401/502/503/504 only, so the schema
    stays a tight contract.
    """

    @pytest.mark.parametrize(
        "path",
        [
            "/v1beta/models/{model}:generateContent",
            "/v1beta/models/{model}:streamGenerateContent",
        ],
    )
    def test_gemini_documents_404(self, path: str) -> None:
        schema = _schema_dict()
        assert "404" in schema["paths"][path]["post"]["responses"], (
            f"{path} must document 404 — see the ``pattern`` note in the "
            "parameter schema"
        )

    @pytest.mark.parametrize(
        "path",
        sorted(BRIDGE_MODE_PATHS - {
            "/v1beta/models/{model}:generateContent",
            "/v1beta/models/{model}:streamGenerateContent",
        }),
    )
    def test_no_other_route_documents_404(self, path: str) -> None:
        """404 is not a documented response for any route other than the two Gemini ones.

        Args:
            path: One of the bridge-mode routes other than the two Gemini
                ones.
        """
        schema = _schema_dict()
        methods = schema["paths"].get(path, {})
        for method in ("post", "get"):
            operation = methods.get(method)
            if operation is None:
                continue
            assert "404" not in operation["responses"], (
                f"{method.upper()} {path} documents a 404 — only the Gemini "
                "routes do, because their ``{model:.*}`` aiohttp path "
                "converter rejects values the pattern can't bound"
            )
