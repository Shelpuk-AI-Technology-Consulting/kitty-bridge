"""Bridge HTTP server — protocol-aware proxy between coding agents and upstream providers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import ssl
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from aiohttp import web

from kitty.bridge.gemini.translator import GeminiTranslator
from kitty.bridge.messages.events import format_error_event as messages_format_error
from kitty.bridge.messages.translator import MessagesTranslator
from kitty.bridge.responses.events import (
    format_error_event as responses_format_error,
)
from kitty.bridge.responses.translator import ResponsesTranslator
from kitty.launchers.base import LauncherAdapter
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

__all__ = ["BridgeServer"]

logger = logging.getLogger(__name__)

# Debug log file for tracing bridge requests/responses
_DEBUG_LOG_DIR = Path.home() / ".cache" / "kitty"
_DEBUG_LOG_PATH = _DEBUG_LOG_DIR / "bridge.log"

_MAX_RETRIES = 3
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_BACKOFF_BASE = 1.0
_EMPTY_RETRY_DELAYS = [5.0, 15.0]  # delays between retries for empty responses (non-balancing)

# Error codes and patterns that indicate rate limiting or quota exhaustion.
# These trigger the circuit breaker even on non-retryable HTTP statuses (e.g. 400).
_RATE_LIMIT_CODES = {"1310"}
_RATE_LIMIT_PATTERNS = ("limit exhaust", "rate limit", "quota exceeded", "usage limit", "exhausted")
_CLIENT_MAX_SIZE = 16 * 1024 * 1024  # 16 MiB
_STREAM_READ_TIMEOUT = 120  # seconds — upstream must respond with first byte within this
_MAX_REQUEST_CHARS = 4_000_000  # ~1.2M estimated tokens; requests larger than this are rejected


def _is_retryable_exception(exc: Exception) -> bool:
    """Return True for transient network exceptions that should be retried."""
    if isinstance(exc, (asyncio.TimeoutError, ConnectionResetError, BrokenPipeError, aiohttp.ClientConnectionError)):
        return True
    if isinstance(exc, OSError):
        return exc.errno in {32, 104, 110, 111, 113}
    return False


def _truncate_for_log(text: str, limit: int = 2000) -> str:
    """Truncate long strings for logs while preserving the total original size."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text)} chars total)"


class UpstreamError(Exception):
    """Raised when the upstream provider returns a non-retryable error or retries are exhausted."""

    def __init__(self, status: int, body: dict) -> None:
        self.status = status
        self.body = body
        super().__init__(f"Upstream error {status}: {body}")


class BridgeServer:
    """HTTP bridge that translates between agent protocols and upstream Chat Completions."""

    def __init__(
        self,
        adapter: LauncherAdapter,
        provider: ProviderAdapter,
        resolved_key: str,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        model: str | None = None,
        debug: bool = False,
        provider_config: dict | None = None,
        backends: list[tuple[ProviderAdapter, str, Profile]] | None = None,
        access_log_path: str | None = None,
        profile_name: str = "default",
        keys_file: str | None = None,
        tls_cert: str | None = None,
        tls_key: str | None = None,
        state_file: str | None = None,
        backend_cooldown: int = 300,
        logging_enabled: bool = False,
        _usage_log_path: Path | None = None,
    ) -> None:
        # TLS validation: both or neither
        if tls_cert and not tls_key:
            raise ValueError("tls_cert provided without tls_key — both are required for TLS")
        if tls_key and not tls_cert:
            raise ValueError("tls_key provided without tls_cert — both are required for TLS")

        self._adapter = adapter
        self._provider = provider
        self._resolved_key = resolved_key
        self._host = host
        self._port = port
        self._model = model
        self._debug = debug
        self._provider_config = provider_config or {}
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._session: aiohttp.ClientSession | None = None
        self._thinking_warned = False
        self._log_path: Path | None = None

        # Access logging
        self._access_log_path = access_log_path
        self._access_log_file = None
        self._profile_name = profile_name

        # State file
        self._state_file = state_file

        # API key authentication
        self._keys_entries: dict[str, str | None] = {}  # key -> profile_name or None
        if keys_file:
            from kitty.bridge.keys import parse_keys_file

            entries = parse_keys_file(keys_file)
            self._keys_entries = {e.key: e.profile for e in entries}

        # TLS
        self._tls_cert = tls_cert
        self._tls_key = tls_key

        # Balancing mode: list of (provider, resolved_key, profile) tuples
        self._backends = backends

        self._backend_cooldown = backend_cooldown

        # Backend health tracking (parallel to _backends)
        self._backend_health: list[dict] = []
        if backends:
            self._backend_health = [
                {
                    "healthy": True,
                    "failed_at": None,
                    "cooldown": backend_cooldown,
                    "stream_error_count": 0,
                }
                for _ in backends
            ]

        # Active backend for current request (set by _select_backend)
        self._active_provider = provider
        self._active_key = resolved_key
        self._active_model = model
        self._active_provider_config = provider_config or {}
        self._current_backend_idx: int = -1

        # Usage logging
        self._logging_enabled = logging_enabled
        self._usage_log_path = _usage_log_path or (_DEBUG_LOG_DIR / "usage.log")

    def _get_next_backend(self) -> tuple[ProviderAdapter, str, str | None, dict]:
        """Select a healthy backend at random with equal probability.

        Skips backends that are in cooldown (unhealthy). If all backends
        are unhealthy, returns a random one anyway (let it fail naturally).

        Returns (provider, resolved_key, model, provider_config, backend_index).
        backend_index is -1 for non-balancing mode.
        """
        if self._backends:
            n = len(self._backends)

            # Check which backends are currently healthy (or cooldown expired)
            healthy_indices = []
            for idx in range(n):
                health = self._backend_health[idx]
                if health["healthy"]:
                    healthy_indices.append(idx)
                elif health["failed_at"] is not None:
                    elapsed = time.monotonic() - health["failed_at"]
                    if elapsed >= health["cooldown"]:
                        logger.info(
                            "Backend %s cooldown expired (%.0fs), retrying",
                            self._backends[idx][2].name,
                            elapsed,
                        )
                        health["healthy"] = True
                        health["failed_at"] = None
                        health["stream_error_count"] = 0
                        healthy_indices.append(idx)

            if healthy_indices:
                idx = random.choice(healthy_indices)
                backend = self._backends[idx]
                provider, key, profile = backend
                return provider, key, profile.model, profile.provider_config, idx  # type: ignore[union-attr]

            # All backends unhealthy — return a random one anyway
            logger.warning("All %d backends unhealthy, attempting request anyway", n)
            idx = random.randint(0, n - 1)
            backend = self._backends[idx]
            provider, key, profile = backend
            return provider, key, profile.model, profile.provider_config, idx  # type: ignore[union-attr]

        return self._provider, self._resolved_key, self._model, self._provider_config or {}, -1

    def _mark_backend_unhealthy(self, index: int, *, cooldown: int | None = None) -> None:
        """Mark a backend as unhealthy and log the event."""
        if not self._backends or index >= len(self._backend_health):
            return
        health = self._backend_health[index]
        health["healthy"] = False
        health["failed_at"] = time.monotonic()

        if cooldown is not None:
            health["cooldown"] = cooldown
            # If this is a stream error, increment the error count
            if cooldown < self._backend_cooldown:
                health["stream_error_count"] = health.get("stream_error_count", 0) + 1
        else:
            health["cooldown"] = self._backend_cooldown
            health["stream_error_count"] = 0  # Reset on persistent error

        profile_name = self._backends[index][2].name
        cooldown_val = health["cooldown"]
        logger.info(
            "Backend %s marked unhealthy for %ds after upstream error",
            profile_name,
            cooldown_val,
        )

    def _any_healthy_backend(self) -> bool:
        """Check if there's at least one healthy backend remaining."""
        if not self._backends:
            return False
        for _idx, health in enumerate(self._backend_health):
            if health["healthy"]:
                return True
            # Check if cooldown has expired
            if health["failed_at"] is not None:
                elapsed = time.monotonic() - health["failed_at"]
                if elapsed >= health["cooldown"]:
                    return True
        return False

    def _get_stream_error_cooldown(self, backend_idx: int) -> int:
        """Return cooldown for a transient stream error on the given backend.

        Uses exponential backoff starting at 30s, doubling on repeated failures.
        """
        if not self._backends or backend_idx < 0:
            return 30
        health = self._backend_health[backend_idx]
        # First failure: start at 30s regardless of initial cooldown
        if health["healthy"] or health.get("stream_error_count", 0) == 0:
            return 30
        count = health.get("stream_error_count", 0)
        return min(30 * (2**count), self._backend_cooldown)

    @staticmethod
    def _is_upstream_stream_error(chunk: dict) -> bool:
        """Return True if a streaming chunk from the upstream contains an error."""
        # Chat Completions error in SSE data
        if chunk.get("error") is not None:
            return True
        # OpenAI-style error wrapper
        if chunk.get("type") == "error":
            return True
        # Some providers nest it in choices
        choices = chunk.get("choices", [])
        if choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta", {})
            if delta.get("type") == "error" or delta.get("error") is not None:
                return True
        return False

    @staticmethod
    def _is_empty_cc_response(cc_response: dict) -> bool:
        """Return True if a Chat Completions response has no content and no tool calls.

        Used to detect empty upstream responses (HTTP 200 but no meaningful output).
        """
        choices = cc_response.get("choices", [])
        if not choices:
            return True
        message = choices[0].get("message", {})
        content = message.get("content")
        tool_calls = message.get("tool_calls", [])
        has_text = isinstance(content, str) and content.strip()
        return not has_text and not tool_calls

    @staticmethod
    def _chunk_has_finish_reason(chunk: dict) -> bool:
        """Return True if a streaming chunk contains a finish_reason."""
        choices = chunk.get("choices", [])
        if choices and isinstance(choices[0], dict):
            return choices[0].get("finish_reason") is not None
        return False

    def _select_backend(self) -> None:
        """Select next backend and set active fields for the current request."""
        provider, key, model, provider_config, backend_idx = self._get_next_backend()
        self._active_provider = provider
        self._active_key = key
        self._active_model = model
        self._active_provider_config = provider_config
        self._current_backend_idx = backend_idx

    @property
    def port(self) -> int:
        return self._port

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    # ── Debug Logging ─────────────────────────────────────────────────────

    def _setup_debug_logging(self) -> Path | None:
        """Configure file-based debug logging if debug mode is enabled. Returns log path or None."""
        if not self._debug:
            return None

        _DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)
        bridge_logger = logging.getLogger("kitty.bridge")
        bridge_logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers on repeated calls
        has_bridge_handler = any(
            isinstance(h, logging.FileHandler) and getattr(h, "_kitty_bridge_log", False)
            for h in bridge_logger.handlers
        )
        if not has_bridge_handler:
            fh = logging.FileHandler(_DEBUG_LOG_PATH, mode="a", encoding="utf-8")
            fh._kitty_bridge_log = True  # type: ignore[attr-defined]
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s │ %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            bridge_logger.addHandler(fh)

        return _DEBUG_LOG_PATH

    def _log_usage(self, usage: dict | None) -> None:
        """Append a JSONL usage entry to the usage log file.

        Fire-and-forget: failures are logged but never raised.
        """
        if not self._logging_enabled:
            return
        try:
            usage = usage or {}
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "profile": self._profile_name,
                "provider": self._active_provider.provider_type,
                "model": self._active_model or "",
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            self._usage_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._usage_log_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception as exc:
            logger.debug("Failed to write usage log: %s", exc)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _should_warn_no_tls(self) -> bool:
        """Return True if binding to non-localhost without TLS."""
        if self._tls_cert and self._tls_key:
            return False
        host = self._host
        return host not in ("127.0.0.1", "localhost", "::1")

    async def start_async(self) -> int:
        """Create the aiohttp app, register routes, start listening. Returns bound port."""
        self._log_path = self._setup_debug_logging()

        # Open access log file if configured
        if self._access_log_path:
            log_path = Path(self._access_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._access_log_file = log_path.open("a", encoding="utf-8")

        # TLS warning
        if self._should_warn_no_tls():
            import sys

            print(
                f"WARNING: Binding to {self._host} without TLS. API keys and responses will be sent in plain text.",
                file=sys.stderr,
            )

        self._app = web.Application(
            client_max_size=_CLIENT_MAX_SIZE, middlewares=[self._auth_middleware, self._access_log_middleware]
        )
        self._register_routes(self._app)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        # Build SSL context if TLS is configured
        ssl_context = None
        if self._tls_cert and self._tls_key:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            ssl_context.load_cert_chain(self._tls_cert, self._tls_key)

        site = web.TCPSite(self._runner, self._host, self._port, ssl_context=ssl_context)
        await site.start()
        for site_obj in self._runner.sites:
            if isinstance(site_obj, web.TCPSite):
                addrs = site_obj._server.sockets  # type: ignore[union-attr]
                if addrs:
                    self._port = addrs[0].getsockname()[1]
                    break
        logger.info("Bridge server started on %s:%d", self._host, self._port)
        logger.info("Debug log: %s", self._log_path)

        # Write state file if configured
        if self._state_file:
            import os

            from kitty.bridge.state import BridgeState, write_state

            state = BridgeState(
                pid=os.getpid(),
                host=self._host,
                port=self._port,
                profile=self._profile_name,
                started_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                tls=bool(self._tls_cert and self._tls_key),
            )
            write_state(self._state_file, state)

        return self._port

    def start(self) -> int:
        """Synchronous wrapper around start_async."""
        return asyncio.get_event_loop().run_until_complete(self.start_async())

    async def stop_async(self) -> None:
        """Gracefully stop the server and close the HTTP client session."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        self._app = None
        if self._access_log_file is not None:
            self._access_log_file.close()
            self._access_log_file = None
        # Remove state file
        if self._state_file:
            from kitty.bridge.state import remove_state

            remove_state(self._state_file)
        logger.info("Bridge server stopped")

    def stop(self) -> None:
        """Synchronous wrapper around stop_async."""
        asyncio.get_event_loop().run_until_complete(self.stop_async())

    # ── Route registration ────────────────────────────────────────────────

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_get("/healthz", self._handle_healthz)

        # Bridge mode (no adapter): register ALL protocol endpoints
        if self._adapter is None:
            app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
            app.router.add_post("/v1/messages", self._handle_messages)
            app.router.add_post("/v1/responses", self._handle_responses)
            app.router.add_post(
                "/v1beta/models/{model:.*}:generateContent",
                self._handle_gemini,
            )
            app.router.add_post(
                "/v1beta/models/{model:.*}:streamGenerateContent",
                self._handle_gemini,
            )
            app.router.add_get("/v1/models", self._handle_models)
            return

        # Agent launch mode: register only the matching protocol
        protocol = self._adapter.bridge_protocol
        if protocol == BridgeProtocol.RESPONSES_API:
            app.router.add_post("/v1/responses", self._handle_responses)
        elif protocol == BridgeProtocol.MESSAGES_API:
            app.router.add_post("/v1/messages", self._handle_messages)
        elif protocol == BridgeProtocol.GEMINI_API:
            app.router.add_post(
                "/v1beta/models/{model:.*}:generateContent",
                self._handle_gemini,
            )
            app.router.add_post(
                "/v1beta/models/{model:.*}:streamGenerateContent",
                self._handle_gemini,
            )
        elif protocol == BridgeProtocol.CHAT_COMPLETIONS_API:
            app.router.add_post("/v1/chat/completions", self._handle_chat_completions)

    # ── Auth middleware ────────────────────────────────────────────────────

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler: object) -> web.StreamResponse:
        # If no keys configured, allow all
        if not self._keys_entries:
            return await handler(request)  # type: ignore[misc]

        auth_header = request.headers.get("Authorization", "")
        token = auth_header[7:] if auth_header.startswith("Bearer ") else None

        if not token or token not in self._keys_entries:
            return web.json_response({"error": "Unauthorized"}, status=401)

        # Store key info for access logging
        key_hash = hashlib.sha256(token.encode()).hexdigest()[:8]
        request["_key_id"] = key_hash

        # If key maps to a profile, use that profile name for logging
        mapped_profile = self._keys_entries[token]
        request["_profile_name"] = mapped_profile or self._profile_name
        if mapped_profile is not None:
            request["_mapped_profile"] = mapped_profile

        return await handler(request)  # type: ignore[misc]

    # ── Access log middleware ─────────────────────────────────────────────

    @web.middleware
    async def _access_log_middleware(self, request: web.Request, handler: object) -> web.StreamResponse:
        start = time.monotonic()
        response: web.StreamResponse | None = None
        try:
            response = await handler(request)  # type: ignore[misc]
        except web.HTTPException:
            raise
        except Exception as exc:
            logger.exception("Handler error: %s", exc)
            response = web.json_response({"error": "Internal server error"}, status=500)
        finally:
            if self._access_log_file is not None:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                self._write_access_log(request, response, elapsed_ms)

        return response  # type: ignore[return-value]

    def _write_access_log(self, request: web.Request, response: web.StreamResponse | None, elapsed_ms: int) -> None:
        if self._access_log_file is None:
            return

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        client_ip = request.remote or "-"
        # key_id: will be populated by auth middleware later; for now '-'
        key_id = request.get("_key_id", "-")
        profile = request.get("_profile_name", self._profile_name)
        method = request.method
        path = request.path
        status = response.status if response is not None else 500

        bytes_in = request.content_length if request.content_length else "-"
        bytes_out = (
            response.content_length
            if (response is not None and hasattr(response, "content_length") and response.content_length)
            else "-"
        )

        line = (
            f"{now}\t{client_ip}\t{key_id}\t{profile}\t{method}\t{path}\t"
            f"{status}\t{bytes_in}\t{bytes_out}\t{elapsed_ms}\n"
        )
        self._access_log_file.write(line)
        self._access_log_file.flush()

    # ── Health check ──────────────────────────────────────────────────────

    async def _handle_healthz(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _handle_models(self, request: web.Request) -> web.Response:
        """Return OpenAI-compatible model list."""
        import time

        models = []
        if self._backends:
            for _provider, _key, profile in self._backends:
                models.append(profile.model)
        elif self._model:
            models.append(self._model)

        now = int(time.time())
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": m, "object": "model", "created": now, "owned_by": "kitty-bridge"} for m in models],
            }
        )

    # ── Responses API handler ─────────────────────────────────────────────

    async def _handle_responses(self, request: web.Request) -> web.StreamResponse:
        self._select_backend()
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"code": "invalid_request", "message": "Invalid JSON body"}},
                status=400,
            )

        logger.debug("═══ RESPONSES API REQUEST ═══")
        logger.debug("Request headers: %s", dict(request.headers))
        logger.debug("Request body: %s", json.dumps(body, indent=2, ensure_ascii=False))

        translator = ResponsesTranslator()
        cc_request = translator.translate_request(body)
        self._normalize_model(cc_request)
        self._active_provider.normalize_request(cc_request)

        logger.debug("Translated CC request: %s", json.dumps(cc_request, indent=2, ensure_ascii=False))

        # P4: Context size guardrail
        size_error = self._check_request_size(cc_request)
        if size_error is not None:
            return size_error

        if cc_request.get("stream"):
            return await self._stream_responses(request, body, translator, cc_request)

        try:
            cc_response = await self._request_with_retry(cc_request)
        except UpstreamError as exc:
            error_msg = self._translate_upstream_error(exc.status, exc.body)
            return web.json_response(
                {"error": {"code": "upstream_error", "message": error_msg}},
                status=exc.status,
            )
        except Exception as exc:
            return web.json_response(
                {"error": {"code": "internal_error", "message": str(exc)}},
                status=500,
            )

        result = translator.translate_response(cc_response)
        self._log_usage(cc_response.get("usage"))
        return web.json_response(result)

    async def _stream_responses(
        self,
        request: web.Request,
        body: dict,
        translator: ResponsesTranslator,
        cc_request: dict,
    ) -> web.StreamResponse:
        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        model = cc_request.get("model", body.get("model", ""))
        logger.debug("═══ STREAM RESPONSES START ═══ response_id=%s model=%s", response_id, model)
        sr = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await sr.prepare(request)

        # Emit response.created and response.in_progress via translator
        start_events = translator.translate_stream_start(response_id, model)
        # We buffer start events and only write them if the response is non-empty.
        # This prevents the client from seeing a partial message lifecycle
        # when we failover due to an empty response.
        for event in start_events:
            logger.debug("Buffered start event: %s", event.split("\n", 1)[0] if "\n" in event else event[:120])

        upstream_status = None
        terminal_status = "completed"
        last_usage: dict | None = None
        try:
            session = await self._get_session()
            url = self._build_upstream_url()
            headers = self._build_upstream_headers()
            logger.debug("Upstream POST → %s", url)

            upstream_body = self._active_provider.translate_to_upstream(cc_request)
            stream_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=_STREAM_READ_TIMEOUT)

            # Retry loop with backend failover for balancing mode
            n_backends = len(self._backends) if self._backends else 1
            max_attempts = (_MAX_RETRIES + 1) * n_backends
            for attempt in range(max_attempts):
                async with session.post(url, json=upstream_body, headers=headers, timeout=stream_timeout) as upstream:
                    upstream_status = upstream.status
                    logger.debug("Upstream response status: %d", upstream.status)
                    logger.debug("Upstream response headers: %s", dict(upstream.headers))

                    if upstream.status not in (200, 201):
                        error_body = await upstream.text()

                        # In balancing mode: mark unhealthy, try next backend
                        if self._backends and self._current_backend_idx >= 0:
                            self._mark_backend_unhealthy(self._current_backend_idx)
                            if self._any_healthy_backend() and attempt < max_attempts - 1:
                                self._select_backend()
                                self._normalize_model(cc_request)
                                self._active_provider.normalize_request(cc_request)
                                url = self._build_upstream_url()
                                headers = self._build_upstream_headers()
                                upstream_body = self._active_provider.translate_to_upstream(cc_request)
                                logger.info(
                                    "Responses stream failover: backend attempt %d/%d (status %d), switching backend",
                                    attempt + 1,
                                    max_attempts,
                                    upstream.status,
                                )
                                continue
                        elif self._should_retry_stream(upstream.status, error_body) and attempt < max_attempts - 1:
                            delay = _BACKOFF_BASE * (2 ** (attempt % (_MAX_RETRIES + 1)))
                            logger.warning(
                                "Upstream %d, retrying in %.1fs (%d/%d)",
                                upstream.status,
                                delay,
                                attempt + 1,
                                max_attempts,
                            )
                            await asyncio.sleep(delay)
                            continue

                        # No healthy backends left or non-balancing mode — surface error to agent
                        logger.error("Upstream error %d: %s", upstream.status, error_body)
                        terminal_status = "incomplete"
                        error_msg = self._translate_upstream_error(upstream.status, error_body)
                        error_event = responses_format_error(
                            {"code": "upstream_error", "message": error_msg},
                            seq=translator._next_seq(),
                        )
                        await sr.write(error_event.encode())
                        break

                    # Success path — stream the response
                    line_buffer = ""
                    chunk_count = 0
                    done = False
                    stream_error = False
                    finish_events: list[str] = []  # buffered finish events
                    async for chunk_bytes in upstream.content:
                        chunk_count += 1
                        raw = chunk_bytes.decode("utf-8", errors="replace")
                        logger.debug("Upstream chunk #%d (%d bytes): %s", chunk_count, len(raw), raw[:500])
                        if done:
                            break
                        line_buffer += raw
                        while "\n" in line_buffer:
                            line, line_buffer = line_buffer.split("\n", 1)
                            line = line.rstrip("\r")
                            if not line:
                                continue
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str.strip() == "[DONE]":
                                    logger.debug("Upstream [DONE] sentinel received")
                                    done = True
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                except json.JSONDecodeError:
                                    logger.warning("Failed to parse upstream SSE data: %s", data_str[:200])
                                    continue
                                # In balancing mode: detect in-stream errors and failover
                                if self._is_upstream_stream_error(chunk):
                                    logger.warning("Upstream sent error in stream chunk: %s", json.dumps(chunk)[:500])
                                    if self._backends and self._current_backend_idx >= 0:
                                        cooldown = self._get_stream_error_cooldown(self._current_backend_idx)
                                        self._mark_backend_unhealthy(self._current_backend_idx, cooldown=cooldown)
                                        if self._any_healthy_backend():
                                            stream_error = True
                                            done = True
                                            break
                                    # No healthy backends — skip the error chunk, let terminal error handle it
                                    stream_error = True
                                    done = True
                                    break
                                events = translator.translate_stream_chunk(response_id, chunk)
                                # Buffer finish events to detect empty responses before writing
                                if self._chunk_has_finish_reason(chunk):
                                    last_usage = chunk.get("usage")
                                    finish_events.extend(events)
                                else:
                                    for event in events:
                                        logger.debug("SSE → %s", event.split("\n", 1)[0][:120])
                                        await sr.write(event.encode())

                    logger.debug(
                        "Upstream stream ended. chunks=%d done=%s remaining_buffer=%d chars",
                        chunk_count,
                        done,
                        len(line_buffer),
                    )

                    # Flush remaining buffer (last chunk without trailing \n)
                    if not done and line_buffer.strip():
                        line = line_buffer.strip()
                        logger.debug("Flushing remaining buffer: %s", line[:500])
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() != "[DONE]":
                                try:
                                    chunk = json.loads(data_str)
                                    events = translator.translate_stream_chunk(response_id, chunk)
                                    if self._chunk_has_finish_reason(chunk):
                                        last_usage = chunk.get("usage")
                                        finish_events.extend(events)
                                    else:
                                        for event in events:
                                            logger.debug("SSE (flush) → %s", event.split("\n", 1)[0][:120])
                                            await sr.write(event.encode())
                                except json.JSONDecodeError:
                                    logger.warning("Failed to parse flushed SSE data: %s", data_str[:200])

                    # Handle in-stream error failover
                    if stream_error:
                        if attempt < max_attempts - 1:
                            self._select_backend()
                            self._normalize_model(cc_request)
                            self._active_provider.normalize_request(cc_request)
                            url = self._build_upstream_url()
                            headers = self._build_upstream_headers()
                            upstream_body = self._active_provider.translate_to_upstream(cc_request)
                            logger.info(
                                "Responses stream failover: in-stream error, backend attempt %d/%d, switching backend",
                                attempt + 1,
                                max_attempts,
                            )
                            continue
                        # No more backends — emit error event and mark incomplete
                        terminal_status = "incomplete"
                        error_event = responses_format_error(
                            {"code": "upstream_error", "message": "All upstream providers returned errors"},
                            seq=translator._next_seq(),
                        )
                        try:
                            await sr.write(error_event.encode())
                        except (ConnectionResetError, BrokenPipeError, OSError):
                            logger.debug(
                                "Client disconnected before error could be sent for %s",
                                response_id,
                            )

                    # Check for empty response
                    if translator.response_was_empty and finish_events:
                        translator.reset()
                        finish_events.clear()
                        if self._backends and self._current_backend_idx >= 0:
                            self._mark_backend_unhealthy(self._current_backend_idx)
                            if self._any_healthy_backend() and attempt < max_attempts - 1:
                                self._select_backend()
                                self._normalize_model(cc_request)
                                self._active_provider.normalize_request(cc_request)
                                url = self._build_upstream_url()
                                headers = self._build_upstream_headers()
                                upstream_body = self._active_provider.translate_to_upstream(cc_request)
                                logger.info(
                                    "Responses stream empty response: attempt %d/%d, switching backend",
                                    attempt + 1,
                                    max_attempts,
                                )
                                continue
                            logger.warning(
                                "All backends returned empty response for %s, emitting fallback",
                                response_id,
                            )
                        elif attempt < max_attempts - 1:
                            delay = _BACKOFF_BASE * (2 ** (attempt % (_MAX_RETRIES + 1)))
                            logger.warning(
                                "Responses stream empty response: retrying in %.1fs (%d/%d)",
                                delay,
                                attempt + 1,
                                max_attempts,
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.warning(
                                "Responses stream empty response after %d attempts, emitting fallback",
                                max_attempts,
                            )

                    # Write buffered finish events to client
                    for event in finish_events:
                        logger.debug("SSE (finish) → %s", event.split("\n", 1)[0][:120])
                        await sr.write(event.encode())
                    self._log_usage(last_usage)
                    break  # Exit retry loop
        except asyncio.TimeoutError:
            terminal_status = "incomplete"
            logger.error("Upstream POST timed out after %ds for %s", _STREAM_READ_TIMEOUT, response_id)
            error_event = responses_format_error(
                {
                    "code": "timeout",
                    "message": (
                        f"Upstream provider timed out ({_STREAM_READ_TIMEOUT}s). Try /clear to reduce context size."
                    ),
                },
                seq=translator._next_seq(),
            )
            try:
                await sr.write(error_event.encode())
            except (ConnectionResetError, BrokenPipeError, OSError):
                logger.debug("Client disconnected before timeout error could be sent for %s", response_id)
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("Client disconnected during Responses API streaming for %s", response_id)
        except Exception as exc:
            terminal_status = "incomplete"
            logger.exception("Exception in _stream_responses: %s", exc)
            error_event = responses_format_error(
                {"code": "internal_error", "message": str(exc)},
                seq=translator._next_seq(),
            )
            try:
                await sr.write(error_event.encode())
            except (ConnectionResetError, BrokenPipeError, OSError):
                logger.debug("Client disconnected before error could be sent for %s", response_id)

        # Ensure response.completed lifecycle is always sent before EOF
        try:
            synthesize_events = translator.synthesize_completed_events(
                response_id,
                model,
                status=terminal_status,
            )
            logger.debug("synthesize_completed_events produced %d events", len(synthesize_events))
            for event in synthesize_events:
                logger.debug("SSE (synthesize) → %s", event.split("\n", 1)[0][:120])
                await sr.write(event.encode())
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("Client disconnected before completion events for %s", response_id)

        logger.info("Responses stream completed for %s (upstream_status=%s)", response_id, upstream_status)
        # Client (e.g. Codex CLI) may disconnect before write_eof() completes — benign race.
        try:
            await sr.write_eof()
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("Client disconnected before stream EOF for %s", response_id)
        return sr

    # ── Messages API handler ──────────────────────────────────────────────

    async def _handle_messages(self, request: web.Request) -> web.StreamResponse:
        self._select_backend()
        try:
            body = await request.json()
        except web.HTTPRequestEntityTooLarge:
            logger.warning("Messages API request body exceeded %d bytes", _CLIENT_MAX_SIZE)
            return web.json_response(
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "Request body too large"},
                },
                status=400,
            )
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "Invalid JSON body"}},
                status=400,
            )

        logger.debug("═══ MESSAGES API REQUEST ═══")
        logger.debug("Request body: %s", json.dumps(body, indent=2, ensure_ascii=False))

        translator = MessagesTranslator(thinking_warned=self._thinking_warned)
        cc_request = translator.translate_request(body)
        self._normalize_model(cc_request)
        self._active_provider.normalize_request(cc_request)
        self._thinking_warned = translator.thinking_warned

        logger.debug("Translated CC request: %s", json.dumps(cc_request, indent=2, ensure_ascii=False))

        # P4: Context size guardrail
        size_error = self._check_request_size(cc_request)
        if size_error is not None:
            return size_error

        if cc_request.get("stream"):
            return await self._stream_messages(request, body, translator, cc_request)

        try:
            cc_response = await self._request_with_retry(cc_request)
        except UpstreamError as exc:
            error_msg = self._translate_upstream_error(exc.status, exc.body)
            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": error_msg}},
                status=exc.status,
            )
        except Exception as exc:
            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
                status=500,
            )

        result = translator.translate_response(cc_response)
        self._log_usage(cc_response.get("usage"))
        return web.json_response(result)

    async def _stream_messages(
        self,
        request: web.Request,
        body: dict,
        translator: MessagesTranslator,
        cc_request: dict,
    ) -> web.StreamResponse:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        model = cc_request.get("model", body.get("model", ""))

        logger.debug("═══ STREAM MESSAGES START ═══ message_id=%s model=%s", message_id, model)

        sr = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await sr.prepare(request)

        last_usage: dict | None = None
        try:
            session = await self._get_session()
            url = self._build_upstream_url()
            headers = self._build_upstream_headers()
            upstream_body = self._active_provider.translate_to_upstream(cc_request)
            logger.debug("Upstream POST → %s", url)

            stream_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=_STREAM_READ_TIMEOUT)

            # Retry loop for retryable upstream errors with backend failover
            n_backends = len(self._backends) if self._backends else 1
            max_attempts = (_MAX_RETRIES + 1) * n_backends
            for attempt in range(max_attempts):
                try:
                    async with session.post(
                        url,
                        json=upstream_body,
                        headers=headers,
                        timeout=stream_timeout,
                    ) as upstream:
                        logger.debug("Upstream response status: %d", upstream.status)

                        if upstream.status not in (200, 201):
                            error_body = await upstream.text()

                            retryable = self._should_retry_stream(upstream.status, error_body)
                            # In balancing mode: mark unhealthy and try next backend for ANY error
                            if self._backends and self._current_backend_idx >= 0:
                                self._mark_backend_unhealthy(self._current_backend_idx)
                                if self._any_healthy_backend() and attempt < max_attempts - 1:
                                    self._select_backend()
                                    self._normalize_model(cc_request)
                                    self._active_provider.normalize_request(cc_request)
                                    url = self._build_upstream_url()
                                    headers = self._build_upstream_headers()
                                    upstream_body = self._active_provider.translate_to_upstream(cc_request)
                                    logger.info(
                                        "Messages stream failover: attempt %d/%d (status %d), switching backend",
                                        attempt + 1,
                                        max_attempts,
                                        upstream.status,
                                    )
                                    continue
                                # No healthy backends left — fall through to surface error
                            elif retryable and attempt < max_attempts - 1:
                                delay = _BACKOFF_BASE * (2 ** (attempt % (_MAX_RETRIES + 1)))
                                await asyncio.sleep(delay)
                                continue

                            # All backends exhausted or non-balancing mode — surface error to agent
                            logger.error("Upstream error %d: %s", upstream.status, error_body)
                            error_msg = self._translate_upstream_error(upstream.status, error_body)
                            error_event = messages_format_error(
                                {"type": "error", "error": {"type": "api_error", "message": error_msg}},
                            )
                            await sr.write(error_event.encode())
                            break

                        # Success path — stream the response
                        line_buffer = ""
                        done = False
                        stream_error = False
                        chunk_count = 0
                        finish_events: list[str] = []  # buffered finish events
                        async for chunk_bytes in upstream.content:
                            if done:
                                break
                            chunk_count += 1
                            line_buffer += chunk_bytes.decode("utf-8", errors="replace")
                            while "\n" in line_buffer:
                                line, line_buffer = line_buffer.split("\n", 1)
                                line = line.rstrip("\r")
                                if not line:
                                    continue
                                if line.startswith("data: "):
                                    data_str = line[6:]
                                    if data_str.strip() == "[DONE]":
                                        done = True
                                        break
                                    try:
                                        chunk = json.loads(data_str)
                                    except json.JSONDecodeError:
                                        logger.warning("Failed to parse upstream SSE data: %s", data_str[:200])
                                        continue
                                    # In balancing mode: detect in-stream errors and failover
                                    if self._is_upstream_stream_error(chunk):
                                        logger.warning(
                                            "Upstream sent error in stream chunk: %s", json.dumps(chunk)[:500]
                                        )
                                        if self._backends and self._current_backend_idx >= 0:
                                            cooldown = self._get_stream_error_cooldown(self._current_backend_idx)
                                            self._mark_backend_unhealthy(self._current_backend_idx, cooldown=cooldown)
                                            if self._any_healthy_backend():
                                                stream_error = True
                                                done = True
                                                break
                                        # No healthy backends — skip the error chunk and let terminal error handle it
                                        stream_error = True
                                        done = True
                                        break
                                    events = translator.translate_stream_chunk(message_id, model, chunk)
                                    # Buffer finish events to detect empty responses before writing
                                    if self._chunk_has_finish_reason(chunk):
                                        last_usage = chunk.get("usage")
                                        finish_events.extend(events)
                                    else:
                                        for event in events:
                                            await sr.write(event.encode())

                        logger.debug("Upstream stream ended. chunks=%d done=%s", chunk_count, done)

                        # Flush remaining buffer (last chunk without trailing \n)
                        if not done and line_buffer.strip():
                            line = line_buffer.strip()
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str.strip() != "[DONE]":
                                    try:
                                        chunk = json.loads(data_str)
                                        events = translator.translate_stream_chunk(message_id, model, chunk)
                                        if self._chunk_has_finish_reason(chunk):
                                            last_usage = chunk.get("usage")
                                            finish_events.extend(events)
                                        else:
                                            for event in events:
                                                await sr.write(event.encode())
                                    except json.JSONDecodeError:
                                        logger.warning("Failed to parse flushed SSE data: %s", data_str[:200])

                        # Handle in-stream error failover
                        if stream_error:
                            if attempt < max_attempts - 1:
                                self._select_backend()
                                self._normalize_model(cc_request)
                                self._active_provider.normalize_request(cc_request)
                                url = self._build_upstream_url()
                                headers = self._build_upstream_headers()
                                upstream_body = self._active_provider.translate_to_upstream(cc_request)
                                logger.info(
                                    "Messages stream in-stream error: attempt %d/%d, switching backend",
                                    attempt + 1,
                                    max_attempts,
                                )
                                continue
                            # All backends failed — emit clean error event
                            error_event = messages_format_error(
                                {
                                    "type": "error",
                                    "error": {
                                        "type": "api_error",
                                        "message": "All upstream providers returned errors",
                                    },
                                },
                            )
                            try:
                                await sr.write(error_event.encode())
                            except (
                                ConnectionResetError,
                                BrokenPipeError,
                                OSError,
                            ):
                                logger.debug(
                                    "Client disconnected before error could be sent for %s",
                                    message_id,
                                )
                            break

                        # Check for empty response: if the translator detected an empty stream
                        # (finish_reason but no content), buffer the fallback events and retry
                        # instead of sending them to the client.
                        if translator.response_was_empty and finish_events:
                            retried = False
                            if self._backends and self._current_backend_idx >= 0:
                                self._mark_backend_unhealthy(self._current_backend_idx)
                                if self._any_healthy_backend() and attempt < max_attempts - 1:
                                    translator.reset()
                                    finish_events.clear()
                                    self._select_backend()
                                    self._normalize_model(cc_request)
                                    self._active_provider.normalize_request(cc_request)
                                    url = self._build_upstream_url()
                                    headers = self._build_upstream_headers()
                                    upstream_body = self._active_provider.translate_to_upstream(cc_request)
                                    logger.info(
                                        "Messages stream empty response: attempt %d/%d, switching backend",
                                        attempt + 1,
                                        max_attempts,
                                    )
                                    retried = True
                                else:
                                    logger.warning(
                                        "All backends returned empty response for %s, emitting fallback",
                                        message_id,
                                    )
                            elif attempt < max_attempts - 1:
                                # Non-balancing: retry with backoff
                                translator.reset()
                                finish_events.clear()
                                delay = _BACKOFF_BASE * (2 ** (attempt % (_MAX_RETRIES + 1)))
                                logger.warning(
                                    "Messages stream empty response: retrying in %.1fs (%d/%d)",
                                    delay,
                                    attempt + 1,
                                    max_attempts,
                                )
                                await asyncio.sleep(delay)
                                retried = True
                            else:
                                logger.warning(
                                    "Messages stream empty response after %d attempts, emitting fallback",
                                    max_attempts,
                                )
                            if retried:
                                continue

                        # Write buffered finish events to client
                        for event in finish_events:
                            await sr.write(event.encode())
                        self._log_usage(last_usage)
                        break  # Exit retry loop
                except Exception as exc:
                    if _is_retryable_exception(exc):
                        if attempt < max_attempts - 1:
                            # In balancing mode: mark unhealthy, try next backend
                            if self._backends and self._current_backend_idx >= 0:
                                self._mark_backend_unhealthy(self._current_backend_idx)
                                if self._any_healthy_backend():
                                    self._select_backend()
                                    self._normalize_model(cc_request)
                                    self._active_provider.normalize_request(cc_request)
                                    url = self._build_upstream_url()
                                    headers = self._build_upstream_headers()
                                    upstream_body = self._active_provider.translate_to_upstream(cc_request)
                                    logger.info(
                                        "Streaming failover: backend attempt %d/%d failed (%s), switching backend",
                                        attempt + 1,
                                        max_attempts,
                                        type(exc).__name__,
                                    )
                                    continue
                                # No healthy backends — fall through to surface error
                            else:
                                delay = _BACKOFF_BASE * (2 ** (attempt % (_MAX_RETRIES + 1)))
                                logger.warning(
                                    "Upstream request failed (%s), retrying in %.1fs (%d/%d)",
                                    type(exc).__name__,
                                    delay,
                                    attempt + 1,
                                    max_attempts - 1,
                                )
                                await asyncio.sleep(delay)
                                continue

                        if isinstance(exc, asyncio.TimeoutError):
                            logger.error("Upstream POST timed out after %ds for %s", _STREAM_READ_TIMEOUT, message_id)
                            error_msg = (
                                f"Upstream provider timed out ({_STREAM_READ_TIMEOUT}s). "
                                "Try /clear to reduce context size."
                            )
                        else:
                            logger.error(
                                "Upstream POST failed after %d attempts for %s: %s",
                                max_attempts,
                                message_id,
                                exc,
                            )
                            error_msg = str(exc) or "Upstream provider request failed"

                        error_event = messages_format_error(
                            {"type": "error", "error": {"type": "api_error", "message": error_msg}},
                        )
                        try:
                            await sr.write(error_event.encode())
                        except (ConnectionResetError, BrokenPipeError, OSError):
                            logger.debug("Client disconnected before error could be sent for %s", message_id)
                        break

                    raise

        except asyncio.TimeoutError:
            logger.error("Upstream POST timed out after %ds for %s", _STREAM_READ_TIMEOUT, message_id)
            error_event = messages_format_error(
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": (
                            f"Upstream provider timed out ({_STREAM_READ_TIMEOUT}s). Try /clear to reduce context size."
                        ),
                    },
                },
            )
            try:
                await sr.write(error_event.encode())
            except (ConnectionResetError, BrokenPipeError, OSError):
                logger.debug("Client disconnected before timeout error could be sent for %s", message_id)
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("Client disconnected during Messages API streaming for %s", message_id)
        except Exception as exc:
            logger.exception("Exception in _stream_messages: %s", exc)
            error_event = messages_format_error(
                {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
            )
            try:
                await sr.write(error_event.encode())
            except (ConnectionResetError, BrokenPipeError, OSError):
                logger.debug("Client disconnected before error could be sent for %s", message_id)

        logger.info("Messages stream completed for %s", message_id)
        # Client may disconnect before write_eof() completes — benign race.
        try:
            await sr.write_eof()
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("Client disconnected before stream EOF for %s", message_id)
        return sr

    async def _handle_gemini(self, request: web.Request) -> web.StreamResponse:
        """Handle Gemini generateContent / streamGenerateContent requests."""
        self._select_backend()
        model_from_path = request.match_info["model"]

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"code": 400, "message": "Invalid JSON body"}},
                status=400,
            )

        logger.debug("═══ GEMINI API REQUEST ═══ model=%s", model_from_path)
        logger.debug("Request body: %s", json.dumps(body, indent=2, ensure_ascii=False))

        translator = GeminiTranslator()
        cc_request = translator.translate_request(body)
        # Inject model from URL path so _normalize_model can override it
        cc_request["model"] = model_from_path
        self._normalize_model(cc_request)
        self._active_provider.normalize_request(cc_request)

        logger.debug("Translated CC request: %s", json.dumps(cc_request, indent=2, ensure_ascii=False))

        # P4: Context size guardrail
        size_error = self._check_request_size(cc_request)
        if size_error is not None:
            return size_error

        # Gemini CLI streams via streamGenerateContent; generateContent is non-streaming.
        # The translator defaults stream=True (Gemini's interactive mode always streams),
        # but non-streaming requests must use stream=False so _make_upstream_request
        # gets a single JSON response instead of an SSE stream.
        is_stream = "streamGenerateContent" in request.path
        if not is_stream:
            cc_request["stream"] = False

        if is_stream:
            return await self._stream_gemini(request, translator, cc_request)

        try:
            cc_response = await self._request_with_retry(cc_request)
        except UpstreamError as exc:
            error_msg = self._translate_upstream_error(exc.status, exc.body)
            return web.json_response(
                {"error": {"code": exc.status, "message": error_msg}},
                status=exc.status,
            )
        except Exception as exc:
            return web.json_response(
                {"error": {"code": 500, "message": str(exc)}},
                status=500,
            )

        result = translator.translate_response(cc_response)
        self._log_usage(cc_response.get("usage"))
        return web.json_response(result)

    async def _stream_gemini(
        self,
        request: web.Request,
        translator: GeminiTranslator,
        cc_request: dict,
    ) -> web.StreamResponse:
        """Stream Gemini generateContent response via SSE."""
        logger.debug("═══ STREAM GEMINI START ═══")

        sr = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await sr.prepare(request)

        last_usage: dict | None = None
        try:
            session = await self._get_session()
            url = self._build_upstream_url()
            headers = self._build_upstream_headers()
            upstream_body = self._active_provider.translate_to_upstream(cc_request)
            logger.debug("Upstream POST → %s", url)

            stream_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=_STREAM_READ_TIMEOUT)

            # Retry loop with backend failover for balancing mode
            n_backends = len(self._backends) if self._backends else 1
            max_attempts = (_MAX_RETRIES + 1) * n_backends
            for attempt in range(max_attempts):
                async with session.post(url, json=upstream_body, headers=headers, timeout=stream_timeout) as upstream:
                    logger.debug("Upstream response status: %d", upstream.status)

                    if upstream.status not in (200, 201):
                        error_body = await upstream.text()
                        logger.error("Upstream error %d: %s", upstream.status, error_body)

                        # In balancing mode: mark unhealthy, try next backend
                        if self._backends and self._current_backend_idx >= 0:
                            self._mark_backend_unhealthy(self._current_backend_idx)
                            if self._any_healthy_backend() and attempt < max_attempts - 1:
                                self._select_backend()
                                self._normalize_model(cc_request)
                                self._active_provider.normalize_request(cc_request)
                                url = self._build_upstream_url()
                                headers = self._build_upstream_headers()
                                upstream_body = self._active_provider.translate_to_upstream(cc_request)
                                logger.info(
                                    "Gemini stream failover: backend attempt %d/%d (status %d), switching backend",
                                    attempt + 1,
                                    max_attempts,
                                    upstream.status,
                                )
                                continue
                        elif self._should_retry_stream(upstream.status, error_body) and attempt < max_attempts - 1:
                            delay = _BACKOFF_BASE * (2 ** (attempt % (_MAX_RETRIES + 1)))
                            logger.warning(
                                "Upstream %d, retrying in %.1fs (%d/%d)",
                                upstream.status,
                                delay,
                                attempt + 1,
                                max_attempts,
                            )
                            await asyncio.sleep(delay)
                            continue

                        # No healthy backends left or non-balancing mode — surface error to agent
                        error_msg = self._translate_upstream_error(upstream.status, error_body)
                        error_sse = (
                            f"data: {json.dumps({'error': {'code': upstream.status, 'message': error_msg}})}\n\n"
                        )
                        try:
                            await sr.write(error_sse.encode())
                        except (ConnectionResetError, BrokenPipeError, OSError):
                            logger.debug("Client disconnected before upstream error could be sent")
                        break

                    # Success path — stream the response
                    line_buffer = ""
                    done = False
                    stream_error = False
                    finish_events: list[str] = []  # buffered finish events
                    async for chunk_bytes in upstream.content:
                        if done:
                            break
                        line_buffer += chunk_bytes.decode("utf-8", errors="replace")
                        while "\n" in line_buffer:
                            line, line_buffer = line_buffer.split("\n", 1)
                            line = line.rstrip("\r")
                            if not line:
                                continue
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str.strip() == "[DONE]":
                                    done = True
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                except json.JSONDecodeError:
                                    logger.warning("Failed to parse upstream SSE data: %s", data_str[:200])
                                    continue
                                # In balancing mode: detect in-stream errors and failover
                                if self._is_upstream_stream_error(chunk):
                                    logger.warning("Upstream sent error in stream chunk: %s", json.dumps(chunk)[:500])
                                    if self._backends and self._current_backend_idx >= 0:
                                        cooldown = self._get_stream_error_cooldown(self._current_backend_idx)
                                        self._mark_backend_unhealthy(self._current_backend_idx, cooldown=cooldown)
                                        if self._any_healthy_backend():
                                            stream_error = True
                                            done = True
                                            break
                                    # No healthy backends — skip the error chunk, let terminal error handle it
                                    stream_error = True
                                    done = True
                                    break
                                events = translator.translate_stream_chunk(chunk)
                                # Buffer finish events to detect empty responses before writing
                                if self._chunk_has_finish_reason(chunk):
                                    last_usage = chunk.get("usage")
                                    finish_events.extend(events)
                                else:
                                    for event in events:
                                        await sr.write(event.encode())

                    # Flush remaining buffer
                    if not done and line_buffer.strip():
                        line = line_buffer.strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() != "[DONE]":
                                try:
                                    chunk = json.loads(data_str)
                                    events = translator.translate_stream_chunk(chunk)
                                    if self._chunk_has_finish_reason(chunk):
                                        last_usage = chunk.get("usage")
                                        finish_events.extend(events)
                                    else:
                                        for event in events:
                                            await sr.write(event.encode())
                                except json.JSONDecodeError:
                                    logger.warning("Failed to parse flushed SSE data: %s", data_str[:200])

                    # Handle in-stream error failover
                    if stream_error:
                        if attempt < max_attempts - 1:
                            self._select_backend()
                            self._normalize_model(cc_request)
                            self._active_provider.normalize_request(cc_request)
                            url = self._build_upstream_url()
                            headers = self._build_upstream_headers()
                            upstream_body = self._active_provider.translate_to_upstream(cc_request)
                            logger.info(
                                "Gemini stream failover: in-stream error, backend attempt %d/%d, switching backend",
                                attempt + 1,
                                max_attempts,
                            )
                            continue
                        # No more backends — error already logged, stream continues to end
                        done = True

                    # Check for empty response
                    if translator.response_was_empty and finish_events:
                        translator.reset()
                        finish_events.clear()
                        if self._backends and self._current_backend_idx >= 0:
                            self._mark_backend_unhealthy(self._current_backend_idx)
                            if self._any_healthy_backend() and attempt < max_attempts - 1:
                                self._select_backend()
                                self._normalize_model(cc_request)
                                self._active_provider.normalize_request(cc_request)
                                url = self._build_upstream_url()
                                headers = self._build_upstream_headers()
                                upstream_body = self._active_provider.translate_to_upstream(cc_request)
                                logger.info(
                                    "Gemini stream empty response: attempt %d/%d, switching backend",
                                    attempt + 1,
                                    max_attempts,
                                )
                                continue
                            logger.warning(
                                "All backends returned empty response, emitting fallback",
                            )
                        elif attempt < max_attempts - 1:
                            delay = _BACKOFF_BASE * (2 ** (attempt % (_MAX_RETRIES + 1)))
                            logger.warning(
                                "Gemini stream empty response: retrying in %.1fs (%d/%d)",
                                delay,
                                attempt + 1,
                                max_attempts,
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.warning(
                                "Gemini stream empty response after %d attempts, emitting fallback",
                                max_attempts,
                            )

                    # Write buffered finish events
                    for event in finish_events:
                        await sr.write(event.encode())
                    self._log_usage(last_usage)
                    break  # Exit retry loop

        except asyncio.TimeoutError:
            logger.error("Upstream POST timed out after %ds for Gemini stream", _STREAM_READ_TIMEOUT)
            error_payload = {
                "error": {
                    "code": 504,
                    "message": f"Upstream provider timed out ({_STREAM_READ_TIMEOUT}s)",
                }
            }
            error_sse = f"data: {json.dumps(error_payload)}\n\n"
            try:
                await sr.write(error_sse.encode())
            except (ConnectionResetError, BrokenPipeError, OSError):
                logger.debug("Client disconnected before timeout error could be sent")
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("Client disconnected during Gemini streaming")
        except Exception as exc:
            logger.exception("Exception in _stream_gemini: %s", exc)
            error_sse = f"data: {json.dumps({'error': {'code': 500, 'message': str(exc)}})}\n\n"
            try:
                await sr.write(error_sse.encode())
            except (ConnectionResetError, BrokenPipeError, OSError):
                logger.debug("Client disconnected before error could be sent")

        try:
            await sr.write_eof()
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("Client disconnected before stream EOF")
        return sr

    # ── Chat Completions pass-through handler ─────────────────────────────

    async def _request_with_retry(self, cc_request: dict) -> dict:
        """Make an upstream request with automatic retry on errors and empty responses.

        For non-balancing mode (no backends): retries empty responses up to
        len(_EMPTY_RETRY_DELAYS)+1 times with backoff.
        For balancing mode: retries up to N times (where N = number of backends),
        marking failed backends as unhealthy.  Also retries empty responses
        across backends.
        """
        if self._backends:
            return await self._request_with_retry_balancing(cc_request)
        return await self._request_with_retry_single(cc_request)

    async def _request_with_retry_single(self, cc_request: dict) -> dict:
        """Non-balancing retry: retry empty responses with backoff."""
        max_attempts = len(_EMPTY_RETRY_DELAYS) + 1
        for attempt in range(max_attempts):
            cc_response = await self._make_upstream_request(cc_request)
            if not self._is_empty_cc_response(cc_response):
                return cc_response
            if attempt < max_attempts - 1:
                delay = _EMPTY_RETRY_DELAYS[attempt]
                logger.warning(
                    "Empty upstream response, retrying in %.1fs (%d/%d)",
                    delay,
                    attempt + 1,
                    max_attempts,
                )
                await asyncio.sleep(delay)
        # All retries exhausted — return the empty response
        # (translator will add fallback text)
        logger.warning("Empty upstream response after %d attempts, returning fallback", max_attempts)
        return cc_response

    async def _request_with_retry_balancing(self, cc_request: dict) -> dict:
        """Balancing retry: failover across backends on errors or empty responses."""
        n_backends = len(self._backends)
        last_exc: UpstreamError | Exception | None = None
        last_response: dict | None = None

        for attempt in range(n_backends):
            if attempt > 0:
                self._select_backend()
                self._normalize_model(cc_request)
                self._active_provider.normalize_request(cc_request)

            try:
                cc_response = await self._make_upstream_request(cc_request)
                if not self._is_empty_cc_response(cc_response):
                    return cc_response
                # Empty response — mark backend unhealthy and try next
                last_response = cc_response
                idx = self._current_backend_idx
                if idx >= 0:
                    self._mark_backend_unhealthy(idx)
                logger.info(
                    "Backend attempt %d/%d returned empty response, marking unhealthy",
                    attempt + 1,
                    n_backends,
                )
            except UpstreamError as exc:
                last_exc = exc
                idx = self._current_backend_idx
                if idx >= 0:
                    self._mark_backend_unhealthy(idx)
                logger.info(
                    "Backend attempt %d/%d failed (status %d), retrying",
                    attempt + 1,
                    n_backends,
                    exc.status,
                )
            except Exception as exc:
                last_exc = exc
                idx = self._current_backend_idx
                if idx >= 0:
                    self._mark_backend_unhealthy(idx)
                logger.info(
                    "Backend attempt %d/%d failed (%s), retrying",
                    attempt + 1,
                    n_backends,
                    exc,
                )

        # All attempts exhausted — return last response (translator adds fallback) or propagate error
        if last_exc is not None:
            if isinstance(last_exc, UpstreamError):
                raise last_exc
            raise last_exc
        # All backends returned empty — return the empty response
        if last_response is not None:
            return last_response
        raise RuntimeError("All backends exhausted with no response")

    async def _handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        """Handle Chat Completions pass-through requests.

        No translation is needed — the agent sends CC format and the upstream
        also expects CC format.  We only apply model normalization and provider
        normalization.
        """
        self._select_backend()
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
                status=400,
            )

        logger.debug("═══ CHAT COMPLETIONS PASS-THROUGH REQUEST ═══")
        logger.debug("Request body: %s", json.dumps(body, indent=2, ensure_ascii=False))

        cc_request = body
        self._normalize_model(cc_request)
        self._active_provider.normalize_request(cc_request)

        logger.debug("Normalized CC request: %s", json.dumps(cc_request, indent=2, ensure_ascii=False))

        # P4: Context size guardrail
        size_error = self._check_request_size(cc_request)
        if size_error is not None:
            return size_error

        if cc_request.get("stream"):
            return await self._stream_chat_completions(request, cc_request)

        try:
            cc_response = await self._request_with_retry(cc_request)
        except UpstreamError as exc:
            error_msg = self._translate_upstream_error(exc.status, exc.body)
            return web.json_response(
                {"error": {"message": error_msg, "type": "upstream_error"}},
                status=exc.status,
            )
        except Exception as exc:
            return web.json_response(
                {"error": {"message": str(exc), "type": "internal_error"}},
                status=500,
            )

        self._log_usage(cc_response.get("usage"))
        return web.json_response(cc_response)

    async def _stream_chat_completions(
        self,
        request: web.Request,
        cc_request: dict,
    ) -> web.StreamResponse:
        """Stream Chat Completions response via SSE pass-through."""
        logger.debug("═══ STREAM CHAT COMPLETIONS PASS-THROUGH START ═══")

        sr = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await sr.prepare(request)

        upstream_status = None
        last_usage: dict | None = None
        try:
            session = await self._get_session()
            url = self._build_upstream_url()
            headers = self._build_upstream_headers()
            upstream_body = self._active_provider.translate_to_upstream(cc_request)
            logger.debug("Upstream POST → %s", url)

            stream_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=_STREAM_READ_TIMEOUT)

            # Retry loop with backend failover for balancing mode
            n_backends = len(self._backends) if self._backends else 1
            max_attempts = (_MAX_RETRIES + 1) * n_backends
            for attempt in range(max_attempts):
                async with session.post(url, json=upstream_body, headers=headers, timeout=stream_timeout) as upstream:
                    upstream_status = upstream.status
                    logger.debug("Upstream response status: %d", upstream.status)

                    if upstream.status not in (200, 201):
                        error_body = await upstream.text()
                        logger.error("Upstream error %d: %s", upstream.status, error_body)

                        # In balancing mode: mark unhealthy, try next backend
                        if self._backends and self._current_backend_idx >= 0:
                            self._mark_backend_unhealthy(self._current_backend_idx)
                            if self._any_healthy_backend() and attempt < max_attempts - 1:
                                self._select_backend()
                                self._normalize_model(cc_request)
                                self._active_provider.normalize_request(cc_request)
                                url = self._build_upstream_url()
                                headers = self._build_upstream_headers()
                                upstream_body = self._active_provider.translate_to_upstream(cc_request)
                                logger.info(
                                    "CC stream failover: attempt %d/%d (status %d), switching backend",
                                    attempt + 1,
                                    max_attempts,
                                    upstream.status,
                                )
                                continue
                        elif self._should_retry_stream(upstream.status, error_body) and attempt < max_attempts - 1:
                            delay = _BACKOFF_BASE * (2 ** (attempt % (_MAX_RETRIES + 1)))
                            logger.warning(
                                "Upstream %d, retrying in %.1fs (%d/%d)",
                                upstream.status,
                                delay,
                                attempt + 1,
                                max_attempts,
                            )
                            await asyncio.sleep(delay)
                            continue

                        # No healthy backends left or non-balancing mode — surface error to agent
                        error_msg = self._translate_upstream_error(upstream.status, error_body)
                        error_sse = (
                            f"data: {json.dumps({'error': {'message': error_msg, 'type': 'upstream_error'}})}\n\n"
                        )
                        try:
                            await sr.write(error_sse.encode())
                        except (ConnectionResetError, BrokenPipeError, OSError):
                            logger.debug("Client disconnected before upstream error could be sent")
                        break

                    # Success path — pass-through stream
                    stream_error = False
                    has_content = False
                    async for chunk_bytes in upstream.content:
                        try:
                            # Detect in-stream errors (both balancing and non-balancing)
                            raw = chunk_bytes.decode("utf-8", errors="replace")
                            if '"error"' in raw or '"type":"error"' in raw:
                                # Try to parse and check if it's actually an error
                                for line in raw.split("\n"):
                                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                                        data_str = line[6:]
                                        try:
                                            chunk = json.loads(data_str)
                                            if self._is_upstream_stream_error(chunk):
                                                logger.warning("Upstream sent error in stream chunk: %s", raw[:500])
                                                if self._backends and self._current_backend_idx >= 0:
                                                    cooldown = self._get_stream_error_cooldown(
                                                        self._current_backend_idx
                                                    )
                                                    self._mark_backend_unhealthy(
                                                        self._current_backend_idx, cooldown=cooldown
                                                    )
                                                stream_error = True
                                                break
                                        except json.JSONDecodeError:
                                            pass
                                if stream_error:
                                    break

                            # Extract usage for logging
                            for line in raw.split("\n"):
                                if line.startswith("data: "):
                                    data_str = line[6:]
                                    if data_str.strip() != "[DONE]":
                                        try:
                                            chunk = json.loads(data_str)
                                            if "usage" in chunk:
                                                last_usage = chunk["usage"]
                                        except json.JSONDecodeError:
                                            pass

                            for translated in self._active_provider.translate_upstream_stream_event(chunk_bytes):
                                has_content = True
                                await sr.write(translated)
                        except (ConnectionResetError, BrokenPipeError, OSError):
                            logger.debug("Client disconnected during streaming")
                            break

                    # Handle in-stream error failover
                    if stream_error:
                        if not self._backends or not self._any_healthy_backend():
                            # All backends failed or non-balancing — emit clean error and stop
                            error_payload = {
                                "error": {
                                    "message": "All upstream providers returned errors",
                                    "type": "upstream_error",
                                }
                            }
                            error_sse = f"data: {json.dumps(error_payload)}\n\n"
                            try:
                                await sr.write(error_sse.encode())
                            except (ConnectionResetError, BrokenPipeError, OSError):
                                logger.debug("Client disconnected before upstream error could be sent")
                            break
                        if attempt < max_attempts - 1:
                            self._select_backend()
                            self._normalize_model(cc_request)
                            self._active_provider.normalize_request(cc_request)
                            url = self._build_upstream_url()
                            headers = self._build_upstream_headers()
                            upstream_body = self._active_provider.translate_to_upstream(cc_request)
                            logger.info(
                                "CC stream in-stream error: attempt %d/%d, switching backend",
                                attempt + 1,
                                max_attempts,
                            )
                            continue
                        # Exhausted all attempts — emit error and stop
                        error_payload = {
                            "error": {
                                "message": "All upstream providers returned errors",
                                "type": "upstream_error",
                            }
                        }
                        error_sse = f"data: {json.dumps(error_payload)}\n\n"
                        try:
                            await sr.write(error_sse.encode())
                        except (ConnectionResetError, BrokenPipeError, OSError):
                            logger.debug("Client disconnected before upstream error could be sent")
                        break

                    # Check for empty response (pass-through: no content bytes written)
                    if not has_content:
                        if self._backends and self._current_backend_idx >= 0:
                            self._mark_backend_unhealthy(self._current_backend_idx)
                            if self._any_healthy_backend() and attempt < max_attempts - 1:
                                self._select_backend()
                                self._normalize_model(cc_request)
                                self._active_provider.normalize_request(cc_request)
                                url = self._build_upstream_url()
                                headers = self._build_upstream_headers()
                                upstream_body = self._active_provider.translate_to_upstream(cc_request)
                                logger.info(
                                    "CC stream empty response: attempt %d/%d, switching backend",
                                    attempt + 1,
                                    max_attempts,
                                )
                                continue
                        elif attempt < max_attempts - 1:
                            delay = _BACKOFF_BASE * (2 ** (attempt % (_MAX_RETRIES + 1)))
                            logger.warning(
                                "CC stream empty response: retrying in %.1fs (%d/%d)",
                                delay,
                                attempt + 1,
                                max_attempts,
                            )
                            await asyncio.sleep(delay)
                            continue
                    self._log_usage(last_usage)
                    break  # Exit retry loop

        except asyncio.TimeoutError:
            logger.error("Upstream POST timed out after %ds for Chat Completions stream", _STREAM_READ_TIMEOUT)
            error_payload = {
                "error": {
                    "message": f"Upstream provider timed out ({_STREAM_READ_TIMEOUT}s)",
                    "type": "timeout_error",
                }
            }
            error_sse = f"data: {json.dumps(error_payload)}\n\n"
            try:
                await sr.write(error_sse.encode())
            except (ConnectionResetError, BrokenPipeError, OSError):
                logger.debug("Client disconnected before timeout error could be sent")
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("Client disconnected during Chat Completions streaming")
        except Exception as exc:
            logger.exception("Exception in _stream_chat_completions: %s", exc)
            error_sse = f"data: {json.dumps({'error': {'message': str(exc), 'type': 'internal_error'}})}\n\n"
            try:
                await sr.write(error_sse.encode())
            except (ConnectionResetError, BrokenPipeError, OSError):
                logger.debug("Client disconnected before error could be sent")

        logger.info("Chat Completions stream completed (upstream_status=%s)", upstream_status)
        try:
            await sr.write_eof()
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("Client disconnected before stream EOF")
        return sr

    # ── Upstream HTTP ─────────────────────────────────────────────────────

    def _normalize_model(self, cc_request: dict) -> None:
        """Override the model name with the profile model, then normalize.

        When the bridge is given a profile model, it replaces whatever model
        the agent sent with the profile model.  This ensures the upstream
        provider always receives the correct model name, regardless of which
        model the agent (e.g. Claude Code) selected internally.
        """
        if self._active_model is not None:
            original = cc_request.get("model")
            cc_request["model"] = self._active_model
            if original != self._active_model:
                logger.debug("Overrode model: %s -> %s", original, self._active_model)

        model = cc_request.get("model")
        if model:
            normalized = self._active_provider.normalize_model_name(model)
            if normalized != model:
                logger.debug("Normalized model: %s -> %s", model, normalized)
            cc_request["model"] = normalized

    def _check_request_size(self, cc_request: dict) -> web.Response | None:
        """Return a 400 error if the translated request exceeds the safe size limit.

        Returns None if the request is within limits, or a json_response to return
        immediately if it's too large.
        """
        request_size = len(json.dumps(cc_request, ensure_ascii=False))
        if request_size > _MAX_REQUEST_CHARS:
            logger.warning(
                "Request body size %d chars exceeds safe limit (%d) — rejecting",
                request_size,
                _MAX_REQUEST_CHARS,
            )
            return web.json_response(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": (
                            f"Request too large ({request_size / 1024:.0f} KB). "
                            "The conversation context has grown beyond what the upstream provider can handle. "
                            "Use /clear to reset the conversation context."
                        ),
                    },
                },
                status=400,
            )
        return None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # LLM calls can be very long (large context, extended thinking).
            # Remove the total timeout so streaming responses are never cut short.
            # Keep a connect timeout to fail fast on network issues.
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    @staticmethod
    def _is_rate_limit_error(status: int, body: object) -> bool:
        """Return True if the upstream error indicates rate limiting or quota exhaustion.

        Detects known error codes (e.g. 1310) and message patterns regardless
        of HTTP status code.  Returns False for context-too-large or auth errors.
        """
        if status == 401 or status == 403:
            return False
        code, message = BridgeServer._extract_error_fields(body)
        if code in _RATE_LIMIT_CODES:
            return True
        searchable = f"{code} {message}".lower()
        return any(p in searchable for p in _RATE_LIMIT_PATTERNS)

    @staticmethod
    def _should_retry_stream(status: int, error_body: str) -> bool:
        """Return True if a streaming error should trigger a retry / backend switch."""
        if status in _RETRYABLE_STATUSES:
            return True
        return BridgeServer._is_rate_limit_error(status, error_body)

    @staticmethod
    def _extract_error_fields(body: object) -> tuple[str, str]:
        """Extract (code, message) from an upstream error body.

        Handles both dict bodies (from non-streaming path) and raw JSON
        strings (from streaming path). Also handles double-nested errors
        where the `message` field itself contains a JSON string with an
        inner `error` object (e.g. Minimax).
        """
        error_obj: dict | None = None
        if isinstance(body, dict):
            error_obj = body.get("error")
        elif isinstance(body, str):
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    error_obj = parsed.get("error")
            except json.JSONDecodeError:
                pass

        code = ""
        message = ""
        if isinstance(error_obj, dict):
            if error_obj.get("code") is not None:
                code = str(error_obj.get("code"))
            if error_obj.get("message") is not None:
                message = str(error_obj.get("message"))

        # If no code was found at the top level, try parsing nested JSON
        # from the message field (e.g. Minimax wraps errors this way).
        if not code and isinstance(message, str) and message.startswith("{"):
            try:
                nested = json.loads(message)
                if isinstance(nested, dict):
                    inner_error = nested.get("error")
                    if isinstance(inner_error, dict):
                        if inner_error.get("code") is not None:
                            code = str(inner_error.get("code"))
                        # Append inner message for searchability
                        inner_msg = inner_error.get("message")
                        if inner_msg:
                            message = f"{message} {inner_msg}"
            except json.JSONDecodeError:
                pass

        return code, message

    @staticmethod
    def _translate_upstream_error(status: int, body: object) -> str:
        """Translate an upstream HTTP error into a user-friendly message.

        For auth errors (401/403), returns a clear message indicating the
        API key issue.
        """
        if isinstance(body, dict):
            details = json.dumps(body, ensure_ascii=False)
        elif body is None:
            details = ""
        else:
            details = str(body)

        if status in (401, 403):
            return (
                f"Upstream authentication failed (HTTP {status}): "
                "API key is invalid, expired, or lacks permission. "
                "Update your API key with 'kitty setup'."
            )

        code, error_message = BridgeServer._extract_error_fields(body)

        # Z.AI code 1261: prompt exceeds model context window
        # Minimax code 2013: context window exceeds limit
        searchable = f"{details}\n{error_message}".lower()
        is_context_too_large = (
            code == "1261"
            or code == "2013"
            or "exceeds max length" in searchable
            or "prompt exceeds" in searchable
            or "context length" in searchable
            or "context window exceeds" in searchable
            or "exceeds context" in searchable
            or "maximum context" in searchable
        )
        if is_context_too_large:
            return (
                "The conversation context has grown too large for the upstream model's context window. "
                "Use /clear to reset the conversation context."
            )

        if status == 500:
            is_provider_network_failure = code == "1234" or "network failure" in searchable
            if is_provider_network_failure:
                prefix = "Upstream provider temporary network/internal failure (HTTP 500). Please retry shortly."
            else:
                prefix = "Upstream provider temporary internal failure (HTTP 500). Please retry shortly."
            return f"{prefix} Details: {details}" if details else prefix

        return details

    def _build_upstream_url(self) -> str:
        base = self._active_provider.build_base_url(self._active_provider_config).rstrip("/")
        model = self._active_model or ""
        path = self._active_provider.get_upstream_path(model)
        return f"{base}{path}"

    def _build_upstream_headers(self) -> dict[str, str]:
        provider = self._active_provider
        # Providers that route to different endpoints per model may need
        # model-aware header construction (e.g. OpenCode Go).
        if hasattr(provider, "build_upstream_headers_for_model"):
            model = self._active_model or ""
            return provider.build_upstream_headers_for_model(self._active_key, model)
        return provider.build_upstream_headers(self._active_key)

    async def _make_upstream_request(self, cc_request: dict) -> dict:
        """Send a non-streaming request upstream.

        For providers with ``use_custom_transport=True``, delegates to the
        provider's ``make_request()`` method (e.g. boto3 for Bedrock).
        Otherwise uses aiohttp with retry/backoff.

        Returns the upstream response dict (in CC format) on success.
        Raises UpstreamError(status, body) on non-retryable or exhausted failures.
        """
        if self._active_provider.use_custom_transport:
            cc_request["_resolved_key"] = self._active_key
            cc_request["_provider_config"] = self._active_provider_config
            return await self._active_provider.make_request(cc_request)

        session = await self._get_session()
        url = self._build_upstream_url()
        headers = self._build_upstream_headers()
        upstream_body = self._active_provider.translate_to_upstream(cc_request)

        request_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=_STREAM_READ_TIMEOUT)

        last_status = 0
        last_body: dict = {}
        for attempt in range(_MAX_RETRIES + 1):
            async with session.post(url, json=upstream_body, headers=headers, timeout=request_timeout) as resp:
                last_status = resp.status
                try:
                    last_body = await resp.json()
                except Exception:
                    last_body = {}

                if last_status < 400:
                    return self._active_provider.translate_from_upstream(last_body)

                if last_status in _RETRYABLE_STATUSES and attempt < _MAX_RETRIES:
                    delay = _BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "Upstream %d, retrying in %.1fs (%d/%d)",
                        last_status,
                        delay,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    await asyncio.sleep(delay)
                    continue

                raise UpstreamError(last_status, last_body)

        raise UpstreamError(last_status, last_body)
