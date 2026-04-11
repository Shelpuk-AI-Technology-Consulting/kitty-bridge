"""Google Cloud Vertex AI provider adapter — CC-compatible with project/location endpoints."""

from __future__ import annotations

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["VertexAIAdapter"]

_DEFAULT_LOCATION = "us-central1"
_API_VERSION = "v1"


class VertexAIAdapter(ProviderAdapter):
    """Google Cloud Vertex AI adapter.

    Vertex AI exposes an OpenAI-compatible Chat Completions API at:
        ``https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{LOCATION}/endpoints/openapi/chat/completions``

    For the global endpoint:
        ``https://aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/global/endpoints/openapi/chat/completions``

    Auth uses a Google Cloud OAuth2 access token sent as ``Authorization: Bearer``.
    The token is obtained via:
    - ``gcloud auth application-default print-access-token``
    - ``google.auth.default()`` (ADC)
    - A service account key

    The request/response format is standard Chat Completions — passthrough.

    Provider config requires:
    - ``project_id``: GCP project ID
    - ``location``: GCP region (default: ``us-central1``)
    """

    @property
    def provider_type(self) -> str:
        return "vertex"

    @property
    def default_base_url(self) -> str:
        """Placeholder base URL — actual URL is built from provider_config."""
        return f"https://{_DEFAULT_LOCATION}-aiplatform.googleapis.com"

    @property
    def upstream_path(self) -> str:
        """Default path — includes the openapi endpoint and chat completions."""
        return self.get_upstream_path("")

    def get_upstream_path(self, model: str) -> str:
        """Build the upstream path for Vertex AI.

        The path always includes ``/endpoints/openapi/chat/completions``.
        The model is specified in the request body, not the URL.
        """
        return "/endpoints/openapi/chat/completions"

    # ── URL construction ─────────────────────────────────────────────────

    def build_base_url(self, provider_config: dict) -> str:
        """Build the base URL from provider_config.

        Args:
            provider_config: Must contain ``project_id``. Optional ``location``
                (default: ``us-central1``).

        Returns:
            Full base URL including the API version and project/location path.

        Raises:
            ProviderError: If ``project_id`` is missing or empty.
        """
        project_id = provider_config.get("project_id", "")
        if not project_id:
            raise ProviderError("Vertex AI requires 'project_id' in provider_config")
        location = provider_config.get("location", "") or _DEFAULT_LOCATION

        host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"

        return f"https://{host}/{_API_VERSION}/projects/{project_id}/locations/{location}"

    # ── Auth ─────────────────────────────────────────────────────────────

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Build auth headers for Vertex AI.

        Uses ``Authorization: Bearer`` with a Google Cloud OAuth2 access token.
        """
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def is_valid_access_token(self, token: str) -> bool:
        """Check if the token is a non-empty string."""
        if not isinstance(token, str):
            return False
        return bool(token.strip())

    # ── Request translation ──────────────────────────────────────────────

    def translate_to_upstream(self, cc_request: dict) -> dict:
        """Translate CC request for Vertex AI.

        Strips internal metadata fields.  The request body is otherwise
        standard Chat Completions format — passthrough.
        """
        return {
            k: v
            for k, v in cc_request.items()
            if k
            not in (
                "_resolved_key",
                "_provider_config",
            )
        }

    def translate_from_upstream(self, raw_response: dict) -> dict:
        """Vertex AI returns CC-compatible responses — passthrough."""
        return raw_response

    # ── Model name normalization ─────────────────────────────────────────

    def normalize_model_name(self, model: str) -> str:
        """Normalize model name for Vertex AI.

        Gemini models on Vertex AI use the ``google/{MODEL_ID}`` format.
        If the model already has a publisher prefix (e.g. ``meta/``, ``google/``),
        it is kept as-is.  Otherwise, ``google/`` is prepended.
        """
        if not model:
            return model
        if "/" in model:
            return model
        return f"google/{model}"

    def normalize_request(self, cc_request: dict) -> None:
        """No normalization needed for Vertex AI (CC-compatible)."""

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
            "usage": response_data.get("usage") or {},
        }
        if "tool_calls" in message:
            result["tool_calls"] = message["tool_calls"]
        return result

    def map_error(self, status_code: int, body: dict) -> Exception:
        if not isinstance(body, dict):
            return ProviderError(f"Vertex AI error {status_code}: {body}")
        error_msg = body.get("error", body)
        msg = error_msg.get("message", str(error_msg)) if isinstance(error_msg, dict) else str(error_msg)
        return ProviderError(f"Vertex AI error {status_code}: {msg}")
