"""Azure OpenAI provider adapter — CC-compatible with deployment-based endpoints."""

from __future__ import annotations

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["AzureOpenAIAdapter"]

_API_VERSION = "2024-10-21"


class AzureOpenAIAdapter(ProviderAdapter):
    """Azure OpenAI Service adapter.

    Azure OpenAI uses the same Chat Completions request/response format as
    OpenAI, but differs in:

    - **Endpoint URL**: includes the deployment-id in the path
      (``/openai/deployments/{deployment-id}/chat/completions?api-version=...``)
    - **Auth header**: ``api-key: KEY`` instead of ``Authorization: Bearer``
    - **Model selection**: via deployment-id in the URL, not ``model`` in the body

    Supports two auth modes:
    - **API key**: stored in Kitty's credential store, sent as ``api-key`` header
    - **Microsoft Entra ID token**: obtained via ``az account get-access-token``,
      stored as ``"Bearer <token>"`` in credential store, sent as ``Authorization`` header
    """

    @property
    def provider_type(self) -> str:
        return "azure"

    @property
    def default_base_url(self) -> str:
        return "https://{resource}.openai.azure.com"

    @property
    def upstream_path(self) -> str:
        """Default path — actual path is built dynamically per deployment."""
        return self.get_upstream_path("model")

    def get_upstream_path(self, model: str) -> str:
        """Build the upstream path for a specific deployment.

        Args:
            model: Azure OpenAI deployment name (e.g. ``"my-gpt4o"``).

        Returns:
            Path with api-version query parameter.
        """
        deployment_id = model or "deployment"
        return f"/openai/deployments/{deployment_id}/chat/completions?api-version={_API_VERSION}"

    # ── Auth ─────────────────────────────────────────────────────────────

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Build auth headers for Azure OpenAI.

        Detects Entra ID tokens (prefixed with ``"Bearer "`` or ``"bearer "``)
        and uses the ``Authorization`` header with proper casing.
        Otherwise uses the ``api-key`` header.
        """
        if self.is_entra_token(api_key):
            # Normalize to "Bearer <token>" regardless of stored casing
            token = api_key.split(" ", 1)[1] if " " in api_key else api_key
            return {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        return {
            "api-key": api_key,
            "Content-Type": "application/json",
        }

    def is_entra_token(self, key: str) -> bool:
        """Check if the credential is a Microsoft Entra ID bearer token."""
        return key.lower().startswith("bearer ")

    # ── Request translation ──────────────────────────────────────────────

    def translate_to_upstream(self, cc_request: dict) -> dict:
        """Translate CC request for Azure OpenAI.

        Removes ``model`` from body (deployment-id is in the URL path)
        and strips internal metadata fields.
        """
        # Shallow copy to avoid mutating the original
        result = {
            k: v
            for k, v in cc_request.items()
            if k
            not in (
                "model",
                "_resolved_key",
                "_provider_config",
            )
        }
        return result

    def translate_from_upstream(self, raw_response: dict) -> dict:
        """Azure OpenAI returns CC-compatible responses — passthrough."""
        return raw_response

    # ── Override upstream URL construction ───────────────────────────────

    def normalize_request(self, cc_request: dict) -> None:
        """No normalization needed for Azure OpenAI (CC-compatible)."""

    def normalize_model_name(self, model: str) -> str:
        """Azure uses deployment names — passthrough."""
        return model

    # ── Standard ProviderAdapter methods ─────────────────────────────────

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        request: dict = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
        }
        if "tools" in kwargs and kwargs["tools"]:
            request["tools"] = kwargs["tools"]
        for key in ("temperature", "top_p", "max_tokens"):
            if key in kwargs and kwargs[key] is not None:
                request[key] = kwargs[key]
        return request

    def parse_response(self, response_data: dict) -> dict:
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        result: dict = {
            "content": message.get("content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": response_data.get("usage", {}),
        }
        if "tool_calls" in message:
            result["tool_calls"] = message["tool_calls"]
        return result

    def map_error(self, status_code: int, body: dict) -> Exception:
        if not isinstance(body, dict):
            return ProviderError(f"Azure OpenAI error {status_code}: {body}")
        error_msg = body.get("error", body)
        msg = error_msg.get("message", str(error_msg)) if isinstance(error_msg, dict) else str(error_msg)
        return ProviderError(f"Azure OpenAI error {status_code}: {msg}")
