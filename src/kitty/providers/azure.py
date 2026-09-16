"""Azure OpenAI provider adapter — CC-compatible with deployment-based endpoints."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["AzureOpenAIAdapter"]

_API_VERSION = "2024-10-21"

# Where a pasted Azure endpoint stops being a resource and starts being a
# deployment.  The deployment segment after this marker is dynamic — it is the
# request's model, per register row P20 — so a URL carrying it is right for
# exactly one request and is cut back to the resource root on configuration.
_DEPLOYMENT_MARKER = "/openai/deployments/"


class AzureOpenAIAdapter(ProviderAdapter):
    """Azure OpenAI Service adapter.

    Azure OpenAI uses the same Chat Completions request/response format as
    OpenAI, but differs in:

    - **Endpoint URL**: includes the deployment-id in the path
      (``/openai/deployments/{deployment-id}/chat/completions?api-version=...``)
    - **Auth header**: ``api-key: KEY`` instead of ``Authorization: Bearer``
    - **Model selection**: via deployment-id in the URL, not ``model`` in the body

    The deployment id is the request's normalized model (register row P20):
    a profile whose model is ``gpt-4o`` addresses the deployment *named*
    ``gpt-4o``, so a user is expected to type a deployment name as the model.
    The base URL is the resource root — required, see :meth:`build_base_url`.

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
        # A template, not an address: nothing substitutes ``{resource}``, and
        # :meth:`build_base_url` refuses to return it (KBR-153).  It stays only
        # to satisfy the abstract property contract.
        return "https://{resource}.openai.azure.com"

    @property
    def upstream_path(self) -> str:
        """Default path — actual path is built dynamically per deployment."""
        return self.get_upstream_path("model")

    @property
    def requires_custom_url(self) -> bool:
        """Whether the setup wizards prompt for this provider's base URL.

        True since KBR-153: without it no wizard ever wrote
        ``provider_config["base_url"]``, and the placeholder in
        :attr:`default_base_url` was left for DNS to fail on.
        """
        return True

    def build_base_url(self, provider_config: dict | None) -> str:
        """Return the Azure resource root from the profile's configured base URL.

        The endpoint Azure documents is
        ``https://<resource>.openai.azure.com/openai/deployments/<deployment>/
        chat/completions?api-version=<v>``, and the ``<resource>`` in
        :attr:`default_base_url` is a template nothing substitutes — so a base
        URL is required, and a pasted full endpoint is cut back to the
        resource root: the deployment segment is the request's model (P20),
        which :meth:`get_upstream_path` appends at request time.

        Args:
            provider_config: The profile's provider configuration.

        Returns:
            The base URL the deployment path is appended to.

        Raises:
            ValueError: When no base URL is configured, or the configured
                value is not an ``http(s)://`` URL — the same contract
                ``custom_openai.build_base_url`` states, and the shape
                ``kitty.validation.validate_api_key`` turns into the
                launch-time message naming what is missing.
        """
        # An absent value and an empty one are the same fault — a profile that
        # never named its resource — and get the missing-URL message rather
        # than the malformed-URL one quoting an empty string.
        url = (provider_config or {}).get("base_url")
        if not url:
            raise ValueError(
                "The azure provider requires a base_url entry in provider_config — the "
                "resource root, e.g. https://my-resource.openai.azure.com. Re-run "
                "`kitty setup` to enter it."
            )
        if not url.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid base_url in provider_config: {url!r}. Must be a non-empty "
                "http:// or https:// URL."
            )
        return self._cut_deployment_segment(url)

    @staticmethod
    def _cut_deployment_segment(url: str) -> str:
        """Drop everything from the first deployment marker onward.

        The marker is searched in the **parsed path**, not the raw string, so a
        query or fragment that merely contains it is left alone, and a
        percent-encoded path segment is not decoded into one.  The first
        occurrence wins: a doubled pasted path still resolves to the resource
        root, matching what composition would request for the unwritten URL.

        An unparseable URL is returned untouched rather than allowed to raise:
        :meth:`build_base_url` runs inside ``kitty.validation.validate_api_key``
        outside its own ``try`` — the same reason
        ``ProviderAdapter._strip_endpoint_suffix`` behaves this way.

        Args:
            url: The validated base URL.

        Returns:
            The URL with its path truncated at the marker, or ``url``
            unchanged when the marker is absent or the URL cannot be parsed.
        """
        # `urlsplit` rejects a malformed IPv6 literal such as "https://[::1/v1".
        try:
            parts = urlsplit(url)
        except ValueError:
            return url
        marker = parts.path.find(_DEPLOYMENT_MARKER)
        if marker == -1:
            return url
        return urlunsplit(parts._replace(path=parts.path[:marker]))

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
        strip = self._INTERNAL_KEYS | {"model"}
        result = {k: v for k, v in cc_request.items() if k not in strip}
        if "messages" in result:
            result["messages"] = self._strip_internal_message_keys(result["messages"])
        return result

    def translate_from_upstream(self, raw_response: dict) -> dict:
        """Azure OpenAI returns CC-compatible responses — passthrough."""
        return raw_response

    # ── Override upstream URL construction ───────────────────────────────

    def normalize_request(self, cc_request: dict) -> None:
        """No normalization needed for Azure OpenAI (CC-compatible)."""

    def normalize_model_name(self, model: str) -> str:
        """Strip provider prefix if present (e.g. 'azure/my-gpt4o')."""
        if "/" in model:
            return model.split("/", 1)[1] or model
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
        choices = response_data.get("choices") or [{}]
        choice = choices[0] if choices else {}
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
