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
from kitty.bridge.messages.events import (
    format_content_block_delta_event,
    format_content_block_start_event,
    format_content_block_stop_event,
    format_message_delta_event,
    format_message_start_event,
    format_message_stop_event,
)
from kitty.bridge.messages.events import (
    format_error_event as messages_format_error,
)
from kitty.bridge.messages.translator import MessagesTranslator
from kitty.bridge.responses.events import (
    format_error_event as responses_format_error,
)
from kitty.bridge.responses.translator import ResponsesTranslator
from kitty.cloudflare import is_cloudflare_block
from kitty.launchers.base import LauncherAdapter
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter, ProviderError
from kitty.providers.model_context import (
    get_balancing_min_context_tokens,
    get_model_context_tokens,
    tokens_to_chars,
)
from kitty.types import BridgeProtocol

__all__ = ["BridgeServer"]

logger = logging.getLogger(__name__)

# Debug log file for tracing bridge requests/responses
_DEBUG_LOG_DIR = Path.home() / ".cache" / "kitty"
_DEBUG_LOG_PATH = _DEBUG_LOG_DIR / "bridge.log"

_MAX_RETRIES = 3
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

# Provider error codes that indicate a permanent failure for this request.
# Even when returned with a 5xx status, these should not be retried.
_NON_RETRYABLE_ERROR_CODES = frozenset(
    {
        "1211",  # Z.AI: Unknown Model
        "1261",  # Z.AI: Prompt exceeds max length (context too large)
        "2013",  # Minimax: Context window too large OR tool-call validation
        # failure (e.g. "tool call result does not follow tool call").
        # The 2013 code has two distinct root causes:
        #   1. Client-side broken pairing — the request body has a
        #      tool_result whose tool_use_id has no matching tool_use.
        #      The bridge now repairs this proactively: see
        #      _apply_compaction -> _validate_tool_call_pairing.
        #   2. Upstream streaming corruption — the upstream sent an
        #      empty SSE body for a tool result, leaving a partial
        #      tool message in the conversation. This is much rarer;
        #      when it happens, the user must /clear to reset state.
        # Both indicate the request will not succeed against the
        # same backend with the same conversation state; in a
        # balancing profile the bridge still fails over to the next
        # backend, in a single-backend profile the user must /clear.
    }
)
_AUTH_FAILURE_STATUSES = {401}  # 403 handled as Cloudflare
_AUTH_COOLDOWN = 86400  # 24h — effectively permanent per session
_BACKOFF_BASE = 1.0
_EMPTY_RETRY_DELAYS = [5.0, 15.0]  # delays between retries for empty responses (non-balancing)
_EMPTY_FINAL_DELAYS = [20.0, 40.0]  # final delays before emitting empty-response fallback

# Error codes and patterns that indicate rate limiting or quota exhaustion.
# These trigger the circuit breaker even on non-retryable HTTP statuses (e.g. 400).
_RATE_LIMIT_CODES = {"1310"}
_RATE_LIMIT_PATTERNS = ("limit exhaust", "rate limit", "quota exceeded", "usage limit", "exhausted")
_CLIENT_MAX_SIZE = 16 * 1024 * 1024  # 16 MiB
_STREAM_READ_TIMEOUT = 120  # seconds — upstream must respond with first byte within this
_SINGLE_BACKEND_COOLDOWN_CAP = 30  # seconds — cap cooldown for single-backend profiles
_CLOUDFLARE_FIRST_HIT_COOLDOWN = 15  # seconds — short cooldown for first Cloudflare block
_ALL_UNHEALTHY_FAST_FAIL_THRESHOLD = 60  # seconds — fast-fail if soonest retry exceeds this


class AllBackendsUnhealthyError(Exception):
    """Raised when all backends are unhealthy and the soonest retry exceeds the fast-fail threshold."""

    def __init__(self, backends: list[dict], retry_after: int) -> None:
        self.backends = backends
        self.retry_after = retry_after
        super().__init__(f"All {len(backends)} backends unhealthy; retry_after={retry_after}s")


_MAX_REQUEST_CHARS = 4_000_000  # ~1.2M estimated tokens; requests larger than this are rejected
_COMPACTION_CHAR_THRESHOLD = int(_MAX_REQUEST_CHARS * 0.7)  # Trigger compaction at 70% of max size
_COMPACTION_TAIL_COUNT = 20  # Minimum number of recent messages to preserve
_COMPACTION_GUARANTEED_MESSAGES_MAX = int(_MAX_REQUEST_CHARS * 0.9)  # Guaranteed budget for compacted messages
_TOOL_RESULT_TRUNCATION_LIMIT = 50_000  # Max chars for a tool result before truncation


def _is_retryable_exception(exc: Exception) -> bool:
    """Return True for transient network exceptions that should be retried."""
    if isinstance(exc, (asyncio.TimeoutError, ConnectionResetError, BrokenPipeError, aiohttp.ClientConnectionError)):
        return True
    if isinstance(exc, OSError):
        return exc.errno in {32, 104, 110, 111, 113}
    return False


def _is_transport_error(exc: Exception) -> bool:
    """Return True for connection-reset / transport errors (not timeouts)."""
    if isinstance(exc, (ConnectionResetError, BrokenPipeError, aiohttp.ClientConnectionError)):
        return True
    if isinstance(exc, OSError):
        return exc.errno in {32, 104}  # EPIPE, ECONNRESET
    # ProviderError with "connection failed" from custom-transport providers
    exc_msg = str(exc).lower()
    return bool("connection failed" in exc_msg or "connection reset" in exc_msg)


def _truncate_for_log(text: str, limit: int = 2000) -> str:
    """Truncate long strings for logs while preserving the total original size."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text)} chars total)"


def _log_cloudflare_block(status: int, body: str) -> None:
    """Log a Cloudflare block without exposing HTML at warning/error level."""
    logger.warning("Upstream Cloudflare block %d", status)
    logger.debug("Upstream Cloudflare response body: %s", _truncate_for_log(body))


class UpstreamError(Exception):
    """Raised when the upstream provider returns a non-retryable error or retries are exhausted."""

    def __init__(self, status: int, body: object) -> None:
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
        debug: bool | str = False,
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
        self._family_cooldown: dict[str, dict] = {}

        # Backend health tracking (parallel to _backends)
        self._backend_health: list[dict] = []
        if backends:
            self._backend_health = [
                {
                    "healthy": True,
                    "failed_at": None,
                    "cooldown": backend_cooldown,
                    "stream_error_count": 0,
                    "failure_count": 0,
                    "transport_error_count": 0,
                    "cloudflare_error_count": 0,
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

    def _get_next_backend(self, *, require_streaming: bool = False) -> tuple[ProviderAdapter, str, str | None, dict]:
        """Select a healthy backend at random with equal probability.

        Skips backends that are in cooldown (unhealthy). If all backends
        are unhealthy, returns a random one anyway (let it fail naturally).

        When require_streaming is True, excludes backends whose provider does
        not implement a custom transport stream_request() override.

        Returns (provider, resolved_key, model, provider_config, backend_index).
        backend_index is -1 for non-balancing mode.
        """
        if self._backends:
            n = len(self._backends)

            # Check which backends are currently healthy (or cooldown expired)
            healthy_indices = []
            for idx in range(n):
                health = self._backend_health[idx]
                provider = self._backends[idx][0]
                if require_streaming and type(provider).stream_request is ProviderAdapter.stream_request:
                    continue
                if require_streaming and not provider.use_custom_transport:
                    continue
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
                # Weight selection inversely by failure_count so repeatedly
                # failing backends are deprioritised among healthy peers.
                weights = []
                for i in healthy_indices:
                    w = 1.0 / (self._backend_health[i].get("failure_count", 0) + 1)
                    weights.append(w)
                idx = random.choices(healthy_indices, weights=weights, k=1)[0]
                backend = self._backends[idx]
                provider, key, profile = backend
                return provider, key, profile.model, profile.provider_config, idx  # type: ignore[union-attr]

            if require_streaming:
                logger.warning("All %d backends unhealthy or non-stream-capable", n)
            else:
                logger.warning("All %d backends unhealthy", n)
            candidates = []
            backend_status = []
            now = time.monotonic()
            for idx in range(n):
                provider = self._backends[idx][0]
                if require_streaming and type(provider).stream_request is ProviderAdapter.stream_request:
                    continue
                if require_streaming and not provider.use_custom_transport:
                    continue
                health = self._backend_health[idx]
                failed_at = health["failed_at"]
                remaining = 0
                if failed_at is not None:
                    elapsed = now - failed_at
                    remaining = max(0, int(health["cooldown"] - elapsed))
                backend_status.append(
                    {
                        "name": self._backends[idx][2].name,
                        "healthy": False,
                        "remaining_cooldown": remaining,
                    }
                )
                candidates.append((remaining, idx))
            if candidates:
                retry_after = min(remaining for remaining, _idx in candidates)
                if retry_after > _ALL_UNHEALTHY_FAST_FAIL_THRESHOLD:
                    raise AllBackendsUnhealthyError(backend_status, retry_after)
                near_retry_indices = [
                    idx for remaining, idx in candidates if remaining <= _ALL_UNHEALTHY_FAST_FAIL_THRESHOLD
                ]
                idx = random.choice(near_retry_indices)
                logger.warning(
                    "All backends unhealthy, attempting near-expiry backend %s (retry_after=%ds)",
                    self._backends[idx][2].name,
                    retry_after,
                )
            else:
                idx = random.randint(0, n - 1)
            backend = self._backends[idx]
            provider, key, profile = backend
            return provider, key, profile.model, profile.provider_config, idx  # type: ignore[union-attr]

        return self._provider, self._resolved_key, self._model, self._provider_config or {}, -1

    def _get_backend_family(self, index: int) -> str:
        if not self._backends or index < 0 or index >= len(self._backends):
            return "default"
        provider = self._backends[index][0]
        provider_type = getattr(provider, "provider_type", None)
        if callable(provider_type):
            return str(provider_type())
        if provider_type is not None:
            return str(provider_type)
        return type(provider).__name__

    def _mark_backend_unhealthy(self, index: int, *, cooldown: int | None = None, failure_kind: str = "hard") -> None:
        """Mark a backend as unhealthy and log the event.

        failure_kind: "hard" (default), "stream", "transport", "rate_limit", "cloudflare", or "auth".
        - "hard" resets stream_error_count and transport_error_count.
        - "stream" increments stream_error_count, resets transport_error_count.
        - "transport" increments transport_error_count, resets stream_error_count.
        - "rate_limit" marks the backend unhealthy (429 response).
        - "cloudflare" marks the backend unhealthy (403 Cloudflare block).
        - "auth" sets a session-persistent cooldown (24h) for invalid credentials.

        For backward compatibility, a short cooldown passed without an explicit
        failure_kind is treated as a stream error.
        """
        if not self._backends or index >= len(self._backend_health):
            return
        health = self._backend_health[index]
        health["healthy"] = False
        health["failed_at"] = time.monotonic()
        health["failure_count"] = health.get("failure_count", 0) + 1

        if failure_kind == "cloudflare":
            health["cloudflare_error_count"] = health.get("cloudflare_error_count", 0) + 1
            if health["cloudflare_error_count"] == 1:
                health["cooldown"] = _CLOUDFLARE_FIRST_HIT_COOLDOWN
            elif cooldown is not None:
                health["cooldown"] = cooldown
            else:
                health["cooldown"] = self._backend_cooldown
                if len(self._backends) == 1 and health["cooldown"] > _SINGLE_BACKEND_COOLDOWN_CAP:
                    health["cooldown"] = _SINGLE_BACKEND_COOLDOWN_CAP
                family = self._get_backend_family(index)
                family_health = self._family_cooldown.setdefault(
                    family,
                    {"abuse_count": 0, "failed_at": time.monotonic(), "cooldown": self._backend_cooldown},
                )
                family_health["abuse_count"] = family_health.get("abuse_count", 0) + 1
                family_health["failed_at"] = time.monotonic()
        elif failure_kind == "auth":
            health["cooldown"] = _AUTH_COOLDOWN
        elif cooldown is not None:
            health["cooldown"] = cooldown
        else:
            health["cooldown"] = self._backend_cooldown
            # Single-backend profiles get a shorter cooldown on hard failures — no failover alternative
            if len(self._backends) == 1 and health["cooldown"] > _SINGLE_BACKEND_COOLDOWN_CAP:
                logger.debug(
                    "Capping single-backend cooldown from %ds to %ds",
                    health["cooldown"],
                    _SINGLE_BACKEND_COOLDOWN_CAP,
                )
                health["cooldown"] = _SINGLE_BACKEND_COOLDOWN_CAP

        if failure_kind == "transport":
            health["transport_error_count"] = health.get("transport_error_count", 0) + 1
            health["stream_error_count"] = 0
        elif failure_kind == "stream" or (cooldown is not None and cooldown <= self._backend_cooldown):
            health["stream_error_count"] = health.get("stream_error_count", 0) + 1
            health["transport_error_count"] = 0
        else:
            health["stream_error_count"] = 0
            health["transport_error_count"] = 0

        profile_name = self._backends[index][2].name
        cooldown_val = health["cooldown"]
        logger.info(
            "Backend %s marked unhealthy for %ds after %s error",
            profile_name,
            cooldown_val,
            failure_kind,
        )

    def _any_healthy_backend(self, *, require_streaming: bool = False) -> bool:
        """Check if there's at least one healthy backend remaining."""
        if not self._backends:
            return False
        for idx, health in enumerate(self._backend_health):
            provider = self._backends[idx][0]
            if require_streaming and type(provider).stream_request is ProviderAdapter.stream_request:
                continue
            if require_streaming and not provider.use_custom_transport:
                continue
            if health["healthy"]:
                return True
            # Check if cooldown has expired
            if health["failed_at"] is not None:
                elapsed = time.monotonic() - health["failed_at"]
                if elapsed >= health["cooldown"]:
                    return True
        return False

    def _mark_backend_healthy(self, index: int) -> None:
        """Reset a backend to healthy state after a successful request.

        Resets ``healthy``, ``failed_at``, ``stream_error_count``, and
        ``transport_error_count``.  Does **not** reset ``failure_count``
        (cumulative lifetime statistic).
        """
        if not self._backends or index >= len(self._backend_health) or index < 0:
            return
        health = self._backend_health[index]
        health["healthy"] = True
        health["failed_at"] = None
        health["stream_error_count"] = 0
        health["transport_error_count"] = 0
        health["cloudflare_error_count"] = 0

    def _decide_cloudflare_action(
        self,
        *,
        attempt: int,
        max_attempts: int,
        cf_retried: set[int],
    ) -> str:
        idx = self._current_backend_idx
        if idx not in cf_retried:
            cf_retried.add(idx)
            return "retry_same"
        self._mark_backend_unhealthy(idx, failure_kind="cloudflare")
        if self._any_healthy_backend() and attempt < max_attempts - 1:
            return "failover"
        return "surface"

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

    def _get_transport_error_cooldown(self, backend_idx: int) -> int:
        """Return cooldown for a transport/connection-reset failure.

        Uses the configured backend_cooldown as base and escalates by 50%
        for each consecutive transport failure, capped at 2x the base.
        """
        if not self._backends or backend_idx < 0:
            return self._backend_cooldown
        health = self._backend_health[backend_idx]
        count = health.get("transport_error_count", 0)
        base = self._backend_cooldown
        cooldown = int(base * (1.5**count))
        return min(cooldown, base * 2)

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
    def _error_response(data: dict, *, status: int = 400) -> web.Response:
        return web.json_response(data, status=status, headers={"Connection": "close"})

    def _empty_response_context(self, upstream_error: str | None = None) -> dict:
        """Build diagnostic context for empty-response fallback text.

        When ``upstream_error`` is provided, the actual upstream error message
        is used in the fallback instead of the generic empty-response text.
        """
        ctx: dict = {}
        if upstream_error:
            ctx["upstream_error"] = upstream_error
        if self._backends and 0 <= self._current_backend_idx < len(self._backends):
            provider = self._backends[self._current_backend_idx][0]
            ctx["provider"] = provider.provider_type
            ctx["model"] = self._model
            total = 1 + len(_EMPTY_RETRY_DELAYS) + len(_EMPTY_FINAL_DELAYS)
            ctx["attempts"] = total
        return ctx

    @staticmethod
    def _is_empty_cc_response(cc_response: dict) -> bool:
        """Return True if a Chat Completions response has no content and no tool calls.

        Used to detect empty upstream responses (HTTP 200 but no meaningful output).
        """
        if cc_response.get("type") == "message":
            content_blocks = cc_response.get("content", [])
            return not any(
                isinstance(block, dict)
                and block.get("type") in {"text", "tool_use"}
                and (block.get("text") or block.get("id"))
                for block in content_blocks
            )
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

    def _select_backend(self, *, require_streaming: bool = False) -> None:
        """Select next backend and set active fields for the current request.

        When require_streaming is True, only selects backends whose provider
        overrides stream_request() from the base ProviderAdapter.
        """
        provider, key, model, provider_config, backend_idx = self._get_next_backend(
            require_streaming=require_streaming,
        )
        self._active_provider = provider
        self._active_key = key
        self._active_model = model
        self._active_provider_config = provider_config
        self._current_backend_idx = backend_idx

    def _log_backend_selection(self) -> None:
        """Log diagnostic info about the currently selected backend."""
        idx = self._current_backend_idx
        provider = self._active_provider
        if self._backends and idx >= 0:
            profile_name = self._backends[idx][2].name
            healthy = self._backend_health[idx]["healthy"]
            logger.debug(
                "Selected backend: profile=%s provider=%s model=%s healthy=%s idx=%d",
                profile_name,
                getattr(provider, "provider_type", type(provider).__name__),
                self._active_model,
                healthy,
                idx,
            )
        else:
            logger.debug(
                "Selected backend: provider=%s model=%s (single-backend)",
                getattr(provider, "provider_type", type(provider).__name__),
                self._active_model,
            )

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

        # Resolve effective log path: custom string path or default
        log_path = Path(self._debug) if isinstance(self._debug, str) else _DEBUG_LOG_PATH

        log_path.parent.mkdir(parents=True, exist_ok=True)
        bridge_logger = logging.getLogger("kitty.bridge")
        bridge_logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers on repeated calls
        has_bridge_handler = any(
            isinstance(h, logging.FileHandler) and getattr(h, "_kitty_bridge_log", False)
            for h in bridge_logger.handlers
        )
        if not has_bridge_handler:
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh._kitty_bridge_log = True  # type: ignore[attr-defined]
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s │ %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            bridge_logger.addHandler(fh)

        return log_path

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
            return self._error_response({"error": "Unauthorized"}, status=401)

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
        if not self._backends:
            return web.json_response({"status": "ok"})
        now = time.monotonic()
        backends = []
        any_healthy = False
        for idx, (_provider, _key, profile) in enumerate(self._backends):
            health = self._backend_health[idx]
            remaining = 0
            if not health["healthy"] and health["failed_at"] is not None:
                remaining = max(0, int(health["cooldown"] - (now - health["failed_at"])))
            if health["healthy"] or remaining == 0:
                any_healthy = True
            backends.append(
                {
                    "name": profile.name,
                    "healthy": health["healthy"],
                    "remaining_cooldown": remaining,
                    "failure_count": health.get("failure_count", 0),
                }
            )
        return web.json_response({"status": "ok" if any_healthy else "degraded", "backends": backends})

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
        except web.HTTPRequestEntityTooLarge:
            logger.warning("Responses API request body exceeded %d bytes", _CLIENT_MAX_SIZE)
            return self._error_response(
                {"error": {"code": "invalid_request", "message": "Request body too large"}},
            )
        except (json.JSONDecodeError, Exception):
            return self._error_response(
                {"error": {"code": "invalid_request", "message": "Invalid JSON body"}},
            )

        logger.debug("═══ RESPONSES API REQUEST ═══")
        logger.debug("Request headers: %s", dict(request.headers))
        logger.debug("Request body: %s", json.dumps(body, indent=2, ensure_ascii=False))

        translator = ResponsesTranslator()
        cc_request = translator.translate_request(body)
        self._normalize_model(cc_request)
        self._active_provider.normalize_request(cc_request)

        logger.debug("Translated CC request: %s", json.dumps(cc_request, indent=2, ensure_ascii=False))

        # Compact messages before size check
        self._apply_compaction(cc_request)

        # P4: Context size guardrail
        size_error = self._check_request_size(cc_request)
        if size_error is not None:
            return size_error

        if cc_request.get("stream"):
            return await self._stream_responses(request, body, translator, cc_request)

        # For custom-transport providers, attach the original Responses API body
        # so the provider can forward it directly without CC round-trip translation.
        if self._active_provider.use_custom_transport:
            cc_request["_original_body"] = body
            self._log_backend_selection()

        try:
            cc_response = await self._request_with_retry(cc_request)
        except UpstreamError as exc:
            error_msg = self._translate_upstream_error(exc.status, exc.body)
            return web.json_response(
                {"error": {"code": "upstream_error", "message": error_msg}},
                status=exc.status,
            )
        except Exception as exc:
            error_msg = self._custom_transport_error_message(exc)
            return web.json_response(
                {"error": {"code": "internal_error", "message": error_msg}},
                status=500,
            )

        result = translator.translate_response(cc_response, context=self._empty_response_context())
        self._log_usage(cc_response.get("usage"))
        if self._backends and self._current_backend_idx >= 0:
            self._mark_backend_healthy(self._current_backend_idx)
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

        # Custom-transport providers handle their own HTTP requests.
        # For Responses API, pass the original body so the provider can
        # forward it directly without CC round-trip translation.
        if self._active_provider.use_custom_transport:
            cc_request["_resolved_key"] = self._active_key
            cc_request["_provider_config"] = self._active_provider_config
            cc_request["_original_body"] = body
            self._log_backend_selection()

            n_backends = len(self._backends) if self._backends else 1
            _bytes_written = False

            async def _tracked_write(data: bytes) -> None:
                nonlocal _bytes_written
                _bytes_written = True
                await sr.write(data)

            for attempt in range(n_backends):
                _bytes_written = False
                try:
                    await self._active_provider.stream_request(cc_request, _tracked_write)
                except Exception as exc:
                    logger.warning("Custom-transport stream failed: %s", exc)
                    if self._backends and self._current_backend_idx >= 0:
                        kind = self._provider_error_failure_kind(exc)
                        self._mark_backend_unhealthy(
                            self._current_backend_idx,
                            failure_kind=kind,
                            cooldown=self._retry_after_from_exc(exc),
                        )
                        if self._any_healthy_backend(require_streaming=True) and attempt < n_backends - 1:
                            try:
                                self._select_backend(require_streaming=True)
                            except AllBackendsUnhealthyError as all_unhealthy:
                                logger.warning(
                                    "All streaming backends unhealthy (fast-fail), retry_after=%ds",
                                    all_unhealthy.retry_after,
                                )
                                break
                            self._normalize_model(cc_request)
                            self._active_provider.normalize_request(cc_request)
                            cc_request["_resolved_key"] = self._active_key
                            cc_request["_provider_config"] = self._active_provider_config
                            logger.info(
                                "Custom-transport failover: attempt %d/%d (%s), switching backend",
                                attempt + 1,
                                n_backends,
                                exc,
                            )
                            continue
                        # No custom-transport backend healthy — try cross-mode failover to standard backend.
                        # Only safe when no bytes were emitted to sr yet.
                        if not _bytes_written and self._any_healthy_backend() and attempt < n_backends - 1:
                            try:
                                self._select_backend()
                            except AllBackendsUnhealthyError:
                                logger.warning("Cross-mode failover: all backends unhealthy")
                                break
                            self._normalize_model(cc_request)
                            self._active_provider.normalize_request(cc_request)
                            cc_request.pop("_resolved_key", None)
                            cc_request.pop("_provider_config", None)
                            cc_request.pop("_original_body", None)
                            logger.info(
                                "Cross-mode failover: attempt %d/%d (%s), switching to standard backend",
                                attempt + 1,
                                n_backends,
                                exc,
                            )
                            break
                    # All backends exhausted or single-backend mode — surface error
                    error_msg = self._custom_transport_error_message(exc)
                    error_status, _ = self._map_provider_error(exc)
                    error_code = "rate_limit_exhausted" if error_status == 429 else "upstream_error"
                    error_event = responses_format_error(
                        {"code": error_code, "message": error_msg},
                        seq=translator._next_seq(),
                    )
                    try:
                        await sr.write(error_event.encode())
                    except (ConnectionResetError, BrokenPipeError, OSError):
                        logger.debug("Client disconnected before error event")
                    break
                upstream_status = 200
                break

            cc_request.pop("_resolved_key", None)
            cc_request.pop("_provider_config", None)
            cc_request.pop("_original_body", None)
            # If cross-mode failover switched to a non-custom-transport provider,
            # fall through to the standard streaming path below.
            if not self._active_provider.use_custom_transport:
                logger.info(
                    "Cross-mode failover: entering standard streaming path with %s",
                    type(self._active_provider).__name__,
                )
            else:
                try:
                    await sr.write_eof()
                except (ConnectionResetError, BrokenPipeError, OSError):
                    logger.debug("Client disconnected before stream EOF")
                return sr

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
            _original_max_attempts = (_MAX_RETRIES + 1) * n_backends
            max_attempts = _original_max_attempts + len(_EMPTY_FINAL_DELAYS)
            for attempt in range(max_attempts):
                if attempt >= _original_max_attempts:
                    delay = _EMPTY_FINAL_DELAYS[attempt - _original_max_attempts]
                    logger.warning(
                        "Empty upstream response: final retry in %.1fs (%d/%d)",
                        delay,
                        attempt + 1,
                        max_attempts,
                    )
                    await asyncio.sleep(delay)
                async with session.post(url, json=upstream_body, headers=headers, timeout=stream_timeout) as upstream:
                    upstream_status = upstream.status
                    logger.debug("Upstream response status: %d", upstream.status)
                    logger.debug("Upstream response headers: %s", dict(upstream.headers))

                    if upstream.status not in (200, 201):
                        error_body = await upstream.text()

                        # Cloudflare challenge — never retryable, abort immediately
                        if self._is_cloudflare_block(upstream.status, error_body):
                            _log_cloudflare_block(upstream.status, error_body)
                            error_msg = self._translate_upstream_error(upstream.status, error_body)
                            error_event = responses_format_error(
                                {"code": "upstream_error", "message": error_msg},
                                seq=translator._next_seq(),
                            )
                            await sr.write(error_event.encode())
                            if self._backends and self._current_backend_idx >= 0:
                                self._mark_backend_unhealthy(self._current_backend_idx, failure_kind="cloudflare")
                            break

                        # In balancing mode: mark unhealthy, try next backend
                        if self._backends and self._current_backend_idx >= 0:
                            kind = "auth" if upstream.status in _AUTH_FAILURE_STATUSES else "hard"
                            self._mark_backend_unhealthy(self._current_backend_idx, failure_kind=kind)
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
                            logger.debug(
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
                    events_emitted = False
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
                                        events_emitted = True

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
                                            events_emitted = True
                                except json.JSONDecodeError:
                                    logger.warning("Failed to parse flushed SSE data: %s", data_str[:200])

                    # Handle in-stream error failover
                    if stream_error:
                        if events_emitted:
                            logger.warning("Responses stream error after client events emitted; not retrying")
                        elif attempt < max_attempts - 1:
                            translator.reset()
                            finish_events.clear()
                            self._select_backend()
                            self._normalize_model(cc_request)
                            self._active_provider.normalize_request(cc_request)
                            url = self._build_upstream_url()
                            headers = self._build_upstream_headers()
                            upstream_body = self._active_provider.translate_to_upstream(cc_request)
                            logger.info(
                                "Responses stream failover: in-stream error (no output yet), "
                                "backend attempt %d/%d, switching backend",
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
                    # Mark backend healthy on clean stream completion
                    if self._backends and self._current_backend_idx >= 0:
                        self._mark_backend_healthy(self._current_backend_idx)
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
            return self._error_response(
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "Request body too large"},
                },
            )
        except (json.JSONDecodeError, Exception):
            return self._error_response(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "Invalid JSON body"}},
            )

        logger.debug("═══ MESSAGES API REQUEST ═══")
        logger.debug("Request body: %s", json.dumps(body, indent=2, ensure_ascii=False))

        translator = MessagesTranslator(thinking_warned=self._thinking_warned)
        if self._active_provider.use_native_messages:
            cc_request = dict(body)
            cc_request["_native_messages_request"] = True
            self._normalize_model(cc_request)
            self._active_provider.normalize_request(cc_request)
            logger.debug("Native Messages request: %s", json.dumps(cc_request, indent=2, ensure_ascii=False))
        else:
            cc_request = translator.translate_request(body)
            self._normalize_model(cc_request)
            self._active_provider.normalize_request(cc_request)
            self._thinking_warned = translator.thinking_warned
            logger.debug("Translated CC request: %s", json.dumps(cc_request, indent=2, ensure_ascii=False))

        # Compact messages before size check
        self._apply_compaction(cc_request)

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
            error_msg = self._custom_transport_error_message(exc)
            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": error_msg}},
                status=500,
            )

        if self._active_provider.use_native_messages:
            result = cc_response
        else:
            result = translator.translate_response(cc_response, context=self._empty_response_context())
        self._log_usage(cc_response.get("usage"))
        if self._backends and self._current_backend_idx >= 0:
            self._mark_backend_healthy(self._current_backend_idx)
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

        # Defer StreamResponse preparation until content is ready to stream.
        # If all upstream attempts fail before any content is emitted, we return
        # a plain HTTP error response (with proper status code) instead of a
        # 200-status SSE stream containing only an error event.  This lets
        # Claude Code's compaction handler properly detect API errors.
        sr: web.StreamResponse | None = None
        _last_error_status: int = 502  # default error status for pre-stream failures

        async def _ensure_prepared() -> web.StreamResponse:
            """Lazily prepare the SSE StreamResponse on first content write."""
            nonlocal sr
            if sr is None:
                sr = web.StreamResponse(
                    status=200,
                    headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
                )
                await sr.prepare(request)
            return sr

        def _make_error_response(error_data: dict, status: int) -> web.Response:
            """Build a Messages API error response for pre-stream failures."""
            return web.json_response(error_data, status=status)

        last_usage: dict | None = None
        stream_ok = False  # Set True only on clean completion

        # Custom-transport providers (e.g. openai_subscription) return
        # Responses API SSE.  We must collect the raw stream, parse it into
        # a Chat Completions response, translate to Messages API format, and
        # emit proper SSE events to the client.
        if self._active_provider.use_custom_transport:
            cc_request["_resolved_key"] = self._active_key
            cc_request["_provider_config"] = self._active_provider_config
            self._log_backend_selection()

            n_backends = len(self._backends) if self._backends else 1
            max_attempts = n_backends
            for attempt in range(max_attempts):
                raw_chunks: list[bytes] = []

                async def _collect(chunk: bytes, _raw_chunks: list[bytes] = raw_chunks) -> None:
                    _raw_chunks.append(chunk)

                try:
                    await self._active_provider.stream_request(cc_request, _collect)
                except Exception as exc:
                    logger.warning("Custom-transport stream failed: %s", exc)
                    # In balancing mode: mark unhealthy and failover
                    if self._backends and self._current_backend_idx >= 0:
                        kind = self._provider_error_failure_kind(exc)
                        self._mark_backend_unhealthy(
                            self._current_backend_idx,
                            failure_kind=kind,
                        )
                        if self._any_healthy_backend(require_streaming=True) and attempt < max_attempts - 1:
                            try:
                                self._select_backend(require_streaming=True)
                            except AllBackendsUnhealthyError as all_unhealthy:
                                logger.warning(
                                    "All streaming backends unhealthy (fast-fail), retry_after=%ds",
                                    all_unhealthy.retry_after,
                                )
                                break
                            self._normalize_model(cc_request)
                            self._active_provider.normalize_request(cc_request)
                            cc_request["_resolved_key"] = self._active_key
                            cc_request["_provider_config"] = self._active_provider_config
                            logger.info(
                                "Custom-transport failover: attempt %d/%d (%s), switching backend",
                                attempt + 1,
                                max_attempts,
                                exc,
                            )
                            continue
                        # No custom-transport backend healthy — try cross-mode failover to standard backend
                        if self._any_healthy_backend() and attempt < max_attempts - 1:
                            try:
                                self._select_backend()
                            except AllBackendsUnhealthyError:
                                logger.warning("Cross-mode failover: all backends unhealthy")
                                break
                            self._normalize_model(cc_request)
                            self._active_provider.normalize_request(cc_request)
                            cc_request.pop("_resolved_key", None)
                            cc_request.pop("_provider_config", None)
                            cc_request.pop("_original_body", None)
                            logger.info(
                                "Cross-mode failover: attempt %d/%d (%s), switching to standard backend",
                                attempt + 1,
                                max_attempts,
                                exc,
                            )
                            break
                    # All backends exhausted or single-backend mode — surface error
                    error_msg = self._custom_transport_error_message(exc)
                    error_status, error_type = self._map_provider_error(exc)
                    error_data = {"type": "error", "error": {"type": error_type, "message": error_msg}}
                    if sr is None:
                        return _make_error_response(error_data, status=error_status)
                    try:
                        await sr.write(messages_format_error(error_data).encode())
                    except (ConnectionResetError, BrokenPipeError, OSError):
                        logger.debug("Client disconnected before error event")
                    break
                else:
                    # Success — parse and emit SSE events
                    raw_bytes = b"".join(raw_chunks)
                    logger.debug(
                        "Custom-transport collected %d chunks, %d bytes raw SSE",
                        len(raw_chunks),
                        len(raw_bytes),
                    )

                    if hasattr(self._active_provider, "parse_stream_to_cc_response"):
                        cc_response = self._active_provider.parse_stream_to_cc_response(raw_bytes)
                    else:
                        from kitty.providers.openai_subscription import OpenAISubscriptionAdapter

                        cc_response = OpenAISubscriptionAdapter._parse_sse_to_response(raw_bytes)
                    logger.debug(
                        "Parsed CC response: %s",
                        json.dumps(cc_response, ensure_ascii=False)[:2000],
                    )
                    result = translator.translate_response(cc_response, context=self._empty_response_context())
                    logger.debug(
                        "Translated Messages API result: %s",
                        json.dumps(result, ensure_ascii=False)[:2000],
                    )

                    msg_id = result.get("id", message_id)
                    model = result.get("model", "")
                    usage = result.get("usage", {})

                    # Emit Messages API SSE events — client may disconnect mid-stream
                    try:
                        s = await _ensure_prepared()
                        await s.write(
                            format_message_start_event(
                                {
                                    "id": msg_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [],
                                    "model": model,
                                    "stop_reason": None,
                                    "stop_sequence": None,
                                    "usage": usage,
                                }
                            ).encode()
                        )

                        for idx, block in enumerate(result.get("content", [])):
                            btype = block.get("type")
                            if btype == "thinking":
                                await s.write(
                                    format_content_block_start_event(idx, {"type": "thinking", "thinking": ""}).encode()
                                )
                                await s.write(
                                    format_content_block_delta_event(
                                        idx, {"type": "thinking_delta", "thinking": block.get("thinking", "")}
                                    ).encode()
                                )
                                await s.write(format_content_block_stop_event(idx).encode())
                            elif btype == "text":
                                await s.write(
                                    format_content_block_start_event(idx, {"type": "text", "text": ""}).encode()
                                )
                                await s.write(
                                    format_content_block_delta_event(
                                        idx, {"type": "text_delta", "text": block.get("text", "")}
                                    ).encode()
                                )
                                await s.write(format_content_block_stop_event(idx).encode())
                            elif btype == "tool_use":
                                partial = json.dumps(block.get("input", {}), ensure_ascii=False)
                                await s.write(
                                    format_content_block_start_event(
                                        idx,
                                        {
                                            "type": "tool_use",
                                            "id": block.get("id", ""),
                                            "name": block.get("name", ""),
                                            "input": {},
                                        },
                                    ).encode()
                                )
                                await s.write(
                                    format_content_block_delta_event(
                                        idx, {"type": "input_json_delta", "partial_json": partial}
                                    ).encode()
                                )
                                await s.write(format_content_block_stop_event(idx).encode())

                        stop_reason = result.get("stop_reason", "end_turn")
                        await s.write(
                            format_message_delta_event({"stop_reason": stop_reason, "stop_sequence": None}, {}).encode()
                        )
                        await s.write(format_message_stop_event().encode())
                    except (ConnectionResetError, BrokenPipeError, OSError):
                        logger.debug("Client disconnected during custom-transport emit for %s", message_id)
                    self._log_usage(cc_response.get("usage"))
                    break

            cc_request.pop("_resolved_key", None)
            cc_request.pop("_provider_config", None)
            # If cross-mode failover switched to a non-custom-transport provider,
            # fall through to the standard streaming path below.
            if not self._active_provider.use_custom_transport:
                logger.info(
                    "Cross-mode failover: entering standard streaming path with %s",
                    type(self._active_provider).__name__,
                )
            else:
                if sr is not None:
                    try:
                        await sr.write_eof()
                    except (ConnectionResetError, BrokenPipeError, OSError):
                        logger.debug("Client disconnected before stream EOF")
                return sr

        try:
            session = await self._get_session()
            url = self._build_upstream_url()
            headers = self._build_upstream_headers()
            upstream_body = self._active_provider.translate_to_upstream(cc_request)
            logger.debug("Upstream POST → %s", url)

            stream_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=_STREAM_READ_TIMEOUT)

            # Retry loop for retryable upstream errors with backend failover
            n_backends = len(self._backends) if self._backends else 1
            _original_max_attempts = (_MAX_RETRIES + 1) * n_backends
            max_attempts = _original_max_attempts + len(_EMPTY_FINAL_DELAYS)
            for attempt in range(max_attempts):
                if attempt >= _original_max_attempts:
                    delay = _EMPTY_FINAL_DELAYS[attempt - _original_max_attempts]
                    logger.warning(
                        "Empty upstream response: final retry in %.1fs (%d/%d)",
                        delay,
                        attempt + 1,
                        max_attempts,
                    )
                    await asyncio.sleep(delay)
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

                            # Cloudflare challenge before output: fail over if a healthy backend remains.
                            if self._is_cloudflare_block(upstream.status, error_body):
                                _log_cloudflare_block(upstream.status, error_body)
                                if self._backends and self._current_backend_idx >= 0:
                                    self._mark_backend_unhealthy(self._current_backend_idx, failure_kind="cloudflare")
                                    if self._any_healthy_backend() and attempt < max_attempts - 1:
                                        self._select_backend()
                                        self._normalize_model(cc_request)
                                        self._active_provider.normalize_request(cc_request)
                                        url = self._build_upstream_url()
                                        headers = self._build_upstream_headers()
                                        upstream_body = self._active_provider.translate_to_upstream(cc_request)
                                        logger.info(
                                            "Messages stream Cloudflare failover: attempt %d/%d, switching backend",
                                            attempt + 1,
                                            max_attempts,
                                        )
                                        continue
                                error_msg = self._translate_upstream_error(upstream.status, error_body)
                                error_data = {"type": "error", "error": {"type": "api_error", "message": error_msg}}
                                if sr is None:
                                    _last_error_status = upstream.status
                                    return _make_error_response(error_data, status=upstream.status)
                                await sr.write(messages_format_error(error_data).encode())
                                break

                            retryable = self._should_retry_stream(upstream.status, error_body)
                            # In balancing mode: mark unhealthy and try next backend for ANY error
                            if self._backends and self._current_backend_idx >= 0:
                                kind = "auth" if upstream.status in _AUTH_FAILURE_STATUSES else "hard"
                                self._mark_backend_unhealthy(self._current_backend_idx, failure_kind=kind)
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
                            error_data = {"type": "error", "error": {"type": "api_error", "message": error_msg}}
                            if sr is None:
                                _last_error_status = upstream.status
                                return _make_error_response(error_data, status=upstream.status)
                            await sr.write(messages_format_error(error_data).encode())
                            break

                        # Success path — stream the response
                        if self._active_provider.use_native_messages:
                            # Native Messages: forward raw SSE bytes to client
                            async for chunk_bytes in upstream.content:
                                s = await _ensure_prepared()
                                await s.write(chunk_bytes)
                                events_emitted = True
                            stream_ok = True
                            break

                        line_buffer = ""
                        done = False
                        stream_error = False
                        events_emitted = False
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
                                            s = await _ensure_prepared()
                                            await s.write(event.encode())
                                            events_emitted = True

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
                                                s = await _ensure_prepared()
                                                await s.write(event.encode())
                                    except json.JSONDecodeError:
                                        logger.warning("Failed to parse flushed SSE data: %s", data_str[:200])

                        # Handle in-stream error failover
                        if stream_error:
                            if events_emitted:
                                logger.warning(
                                    "Messages stream error after client events emitted; finalizing partial response"
                                )
                                try:
                                    s = await _ensure_prepared()
                                    for event in translator.finalize_interrupted_stream():
                                        await s.write(event.encode())
                                except (
                                    ConnectionResetError,
                                    BrokenPipeError,
                                    OSError,
                                ):
                                    logger.debug(
                                        "Client disconnected before interrupted stream finalization for %s",
                                        message_id,
                                    )
                                break
                            elif attempt < max_attempts - 1:
                                translator.reset()
                                finish_events.clear()
                                self._select_backend()
                                self._normalize_model(cc_request)
                                self._active_provider.normalize_request(cc_request)
                                url = self._build_upstream_url()
                                headers = self._build_upstream_headers()
                                upstream_body = self._active_provider.translate_to_upstream(cc_request)
                                logger.info(
                                    "Messages stream in-stream error (no output yet): attempt %d/%d, switching backend",
                                    attempt + 1,
                                    max_attempts,
                                )
                                continue
                            # All backends failed — emit clean error event
                            error_data = {
                                "type": "error",
                                "error": {
                                    "type": "api_error",
                                    "message": "All upstream providers returned errors",
                                },
                            }
                            if sr is None:
                                return _make_error_response(error_data, status=502)
                            try:
                                await sr.write(messages_format_error(error_data).encode())
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
                                elif attempt < max_attempts - 1:
                                    final_idx = attempt - _original_max_attempts
                                    if 0 <= final_idx < len(_EMPTY_FINAL_DELAYS):
                                        delay = _EMPTY_FINAL_DELAYS[final_idx]
                                        logger.warning(
                                            "Messages stream empty response: final retry in %.1fs (%d/%d)",
                                            delay,
                                            attempt + 1,
                                            max_attempts,
                                        )
                                        await asyncio.sleep(delay)
                                        translator.reset()
                                        finish_events.clear()
                                        self._select_backend()
                                        self._normalize_model(cc_request)
                                        self._active_provider.normalize_request(cc_request)
                                        url = self._build_upstream_url()
                                        headers = self._build_upstream_headers()
                                        upstream_body = self._active_provider.translate_to_upstream(cc_request)
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
                        s = await _ensure_prepared()
                        for event in finish_events:
                            await s.write(event.encode())
                        self._log_usage(last_usage)
                        stream_ok = True
                        break  # Exit retry loop
                except Exception as exc:
                    if _is_retryable_exception(exc):
                        if attempt < max_attempts - 1:
                            # In balancing mode: mark unhealthy, try next backend
                            if self._backends and self._current_backend_idx >= 0:
                                is_transport = _is_transport_error(exc)
                                cooldown = (
                                    self._get_transport_error_cooldown(self._current_backend_idx)
                                    if is_transport
                                    else self._backend_cooldown
                                )
                                kind = "transport" if is_transport else "hard"
                                self._mark_backend_unhealthy(
                                    self._current_backend_idx,
                                    cooldown=cooldown,
                                    failure_kind=kind,
                                )
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
                                logger.debug(
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

                        error_data = {"type": "error", "error": {"type": "api_error", "message": error_msg}}
                        if sr is None:
                            return _make_error_response(error_data, status=_last_error_status)
                        try:
                            await sr.write(messages_format_error(error_data).encode())
                        except (ConnectionResetError, BrokenPipeError, OSError):
                            logger.debug("Client disconnected before error could be sent for %s", message_id)
                        break

                    raise

        except asyncio.TimeoutError:
            logger.error("Upstream POST timed out after %ds for %s", _STREAM_READ_TIMEOUT, message_id)
            error_data = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": (
                        f"Upstream provider timed out ({_STREAM_READ_TIMEOUT}s). Try /clear to reduce context size."
                    ),
                },
            }
            if sr is None:
                return _make_error_response(error_data, status=504)
            try:
                await sr.write(messages_format_error(error_data).encode())
            except (ConnectionResetError, BrokenPipeError, OSError):
                logger.debug("Client disconnected before timeout error could be sent for %s", message_id)
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("Client disconnected during Messages API streaming for %s", message_id)
        except Exception as exc:
            logger.exception("Exception in _stream_messages: %s", exc)
            error_data = {"type": "error", "error": {"type": "api_error", "message": str(exc)}}
            if sr is None:
                return _make_error_response(error_data, status=502)
            try:
                await sr.write(messages_format_error(error_data).encode())
            except (ConnectionResetError, BrokenPipeError, OSError):
                logger.debug("Client disconnected before error could be sent for %s", message_id)

        logger.info("Messages stream completed for %s", message_id)
        # Mark backend healthy on success so cooldown resets for next request
        if stream_ok and self._backends and self._current_backend_idx >= 0:
            self._mark_backend_healthy(self._current_backend_idx)
        # Client may disconnect before write_eof() completes — benign race.
        if sr is not None:
            try:
                await sr.write_eof()
            except (ConnectionResetError, BrokenPipeError, OSError):
                logger.debug("Client disconnected before stream EOF for %s", message_id)
            return sr
        # Fallback: if sr was never prepared (should not happen), return a generic error.
        return _make_error_response(
            {"type": "error", "error": {"type": "api_error", "message": "No upstream response"}},
            status=502,
        )

    async def _handle_gemini(self, request: web.Request) -> web.StreamResponse:
        """Handle Gemini generateContent / streamGenerateContent requests."""
        self._select_backend()
        model_from_path = request.match_info["model"]

        try:
            body = await request.json()
        except web.HTTPRequestEntityTooLarge:
            logger.warning("Gemini request body exceeded %d bytes", _CLIENT_MAX_SIZE)
            return self._error_response(
                {"error": {"code": 400, "message": "Request body too large"}},
            )
        except (json.JSONDecodeError, Exception):
            return self._error_response(
                {"error": {"code": 400, "message": "Invalid JSON body"}},
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

        # Compact messages before size check
        self._apply_compaction(cc_request)

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
            error_msg = self._custom_transport_error_message(exc)
            return web.json_response(
                {"error": {"code": 500, "message": error_msg}},
                status=500,
            )

        result = translator.translate_response(cc_response, context=self._empty_response_context())
        self._log_usage(cc_response.get("usage"))
        if self._backends and self._current_backend_idx >= 0:
            self._mark_backend_healthy(self._current_backend_idx)
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
        stream_ok = False  # Set True only on clean completion

        # Custom-transport providers handle their own HTTP requests.
        if self._active_provider.use_custom_transport:
            cc_request["_resolved_key"] = self._active_key
            cc_request["_provider_config"] = self._active_provider_config
            self._log_backend_selection()

            n_backends = len(self._backends) if self._backends else 1
            _bytes_written = False

            async def _tracked_write(data: bytes) -> None:
                nonlocal _bytes_written
                _bytes_written = True
                await sr.write(data)

            for attempt in range(n_backends):
                _bytes_written = False
                try:
                    await self._active_provider.stream_request(cc_request, _tracked_write)
                except Exception as exc:
                    logger.warning("Custom-transport stream failed: %s", exc)
                    if self._backends and self._current_backend_idx >= 0:
                        kind = self._provider_error_failure_kind(exc)
                        self._mark_backend_unhealthy(
                            self._current_backend_idx,
                            failure_kind=kind,
                            cooldown=self._retry_after_from_exc(exc),
                        )
                        if self._any_healthy_backend(require_streaming=True) and attempt < n_backends - 1:
                            try:
                                self._select_backend(require_streaming=True)
                            except AllBackendsUnhealthyError as all_unhealthy:
                                logger.warning(
                                    "All streaming backends unhealthy (fast-fail), retry_after=%ds",
                                    all_unhealthy.retry_after,
                                )
                                break
                            self._normalize_model(cc_request)
                            self._active_provider.normalize_request(cc_request)
                            cc_request["_resolved_key"] = self._active_key
                            cc_request["_provider_config"] = self._active_provider_config
                            logger.info(
                                "Custom-transport failover: attempt %d/%d (%s), switching backend",
                                attempt + 1,
                                n_backends,
                                exc,
                            )
                            continue
                        # No custom-transport backend healthy — try cross-mode failover to standard backend.
                        # Only safe when no bytes were emitted to sr yet.
                        if not _bytes_written and self._any_healthy_backend() and attempt < n_backends - 1:
                            try:
                                self._select_backend()
                            except AllBackendsUnhealthyError:
                                logger.warning("Cross-mode failover: all backends unhealthy")
                                break
                            self._normalize_model(cc_request)
                            self._active_provider.normalize_request(cc_request)
                            cc_request.pop("_resolved_key", None)
                            cc_request.pop("_provider_config", None)
                            logger.info(
                                "Cross-mode failover: attempt %d/%d (%s), switching to standard backend",
                                attempt + 1,
                                n_backends,
                                exc,
                            )
                            break
                    # All backends exhausted or single-backend mode — surface error
                    error_msg = self._custom_transport_error_message(exc)
                    error_status, _ = self._map_provider_error(exc)
                    error_code = error_status if error_status != 502 else "stream_error"
                    error_sse = f"data: {json.dumps({'error': {'code': error_code, 'message': error_msg}})}\n\n"
                    try:
                        await sr.write(error_sse.encode())
                    except (ConnectionResetError, BrokenPipeError, OSError):
                        logger.debug("Client disconnected before error event")
                    break
                break

            cc_request.pop("_resolved_key", None)
            cc_request.pop("_provider_config", None)
            # If cross-mode failover switched to a non-custom-transport provider,
            # fall through to the standard streaming path below.
            if not self._active_provider.use_custom_transport:
                logger.info(
                    "Cross-mode failover: entering standard streaming path with %s",
                    type(self._active_provider).__name__,
                )
            else:
                try:
                    await sr.write_eof()
                except (ConnectionResetError, BrokenPipeError, OSError):
                    logger.debug("Client disconnected before stream EOF")
                return sr

        try:
            session = await self._get_session()
            url = self._build_upstream_url()
            headers = self._build_upstream_headers()
            upstream_body = self._active_provider.translate_to_upstream(cc_request)
            logger.debug("Upstream POST → %s", url)

            stream_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=_STREAM_READ_TIMEOUT)

            # Retry loop with backend failover for balancing mode
            n_backends = len(self._backends) if self._backends else 1
            _original_max_attempts = (_MAX_RETRIES + 1) * n_backends
            max_attempts = _original_max_attempts + len(_EMPTY_FINAL_DELAYS)
            for attempt in range(max_attempts):
                if attempt >= _original_max_attempts:
                    delay = _EMPTY_FINAL_DELAYS[attempt - _original_max_attempts]
                    logger.warning(
                        "Empty upstream response: final retry in %.1fs (%d/%d)",
                        delay,
                        attempt + 1,
                        max_attempts,
                    )
                    await asyncio.sleep(delay)
                async with session.post(url, json=upstream_body, headers=headers, timeout=stream_timeout) as upstream:
                    logger.debug("Upstream response status: %d", upstream.status)

                    if upstream.status not in (200, 201):
                        error_body = await upstream.text()

                        # Cloudflare challenge — never retryable, abort immediately
                        if self._is_cloudflare_block(upstream.status, error_body):
                            _log_cloudflare_block(upstream.status, error_body)
                            error_msg = self._translate_upstream_error(upstream.status, error_body)
                            error_sse = (
                                f"data: {json.dumps({'error': {'code': 'upstream_error', 'message': error_msg}})}\n\n"
                            )
                            try:
                                await sr.write(error_sse.encode())
                            except (ConnectionResetError, BrokenPipeError, OSError):
                                logger.debug("Client disconnected before Cloudflare error event")
                            if self._backends and self._current_backend_idx >= 0:
                                self._mark_backend_unhealthy(self._current_backend_idx, failure_kind="cloudflare")
                            break

                        logger.error("Upstream error %d: %s", upstream.status, error_body)

                        # In balancing mode: mark unhealthy, try next backend
                        if self._backends and self._current_backend_idx >= 0:
                            kind = "auth" if upstream.status in _AUTH_FAILURE_STATUSES else "hard"
                            self._mark_backend_unhealthy(self._current_backend_idx, failure_kind=kind)
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
                            logger.debug(
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
                    events_emitted = False
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
                                        events_emitted = True

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
                                            events_emitted = True
                                except json.JSONDecodeError:
                                    logger.warning("Failed to parse flushed SSE data: %s", data_str[:200])

                    # Handle in-stream error failover
                    if stream_error:
                        if events_emitted:
                            logger.warning("Gemini stream error after client events emitted; not retrying")
                            break
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
                    stream_ok = True
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
        # Mark backend healthy on success so cooldown resets for next request
        if stream_ok and self._backends and self._current_backend_idx >= 0:
            self._mark_backend_healthy(self._current_backend_idx)
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
        max_attempts = len(_EMPTY_RETRY_DELAYS) + len(_EMPTY_FINAL_DELAYS) + 1
        for attempt in range(max_attempts):
            cc_response = await self._make_upstream_request(cc_request)
            if not self._is_empty_cc_response(cc_response):
                return cc_response
            if attempt < len(_EMPTY_RETRY_DELAYS):
                delay = _EMPTY_RETRY_DELAYS[attempt]
                logger.warning(
                    "Empty upstream response, retrying in %.1fs (%d/%d)",
                    delay,
                    attempt + 1,
                    max_attempts,
                )
                await asyncio.sleep(delay)
            elif attempt < len(_EMPTY_RETRY_DELAYS) + len(_EMPTY_FINAL_DELAYS):
                delay = _EMPTY_FINAL_DELAYS[attempt - len(_EMPTY_RETRY_DELAYS)]
                logger.warning(
                    "Empty upstream response: final retry in %.1fs (%d/%d)",
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
                try:
                    self._select_backend()
                except AllBackendsUnhealthyError:
                    logger.warning("All backends unhealthy (fast-fail), aborting retry loop")
                    break
                self._normalize_model(cc_request)
                self._active_provider.normalize_request(cc_request)

            try:
                cc_response = await self._make_upstream_request(cc_request, retry_rate_limit=False)
                if not self._is_empty_cc_response(cc_response):
                    return cc_response
                # Empty response — try next backend
                last_response = cc_response
                logger.info(
                    "Backend attempt %d/%d returned empty response, trying next",
                    attempt + 1,
                    n_backends,
                )
            except UpstreamError as exc:
                last_exc = exc
                idx = self._current_backend_idx
                if idx >= 0:
                    if exc.status == 429:
                        failure_kind = "rate_limit"
                    elif isinstance(exc.body, str) and self._is_cloudflare_block(exc.status, exc.body):
                        failure_kind = "cloudflare"
                    elif exc.status in _AUTH_FAILURE_STATUSES:
                        failure_kind = "auth"
                    else:
                        failure_kind = "hard"
                    self._mark_backend_unhealthy(idx, failure_kind=failure_kind)
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
                    failure_kind = self._provider_error_failure_kind(exc)
                    self._mark_backend_unhealthy(
                        idx,
                        failure_kind=failure_kind,
                        cooldown=self._retry_after_from_exc(exc),
                    )
                logger.info(
                    "Backend attempt %d/%d failed (%s), retrying",
                    attempt + 1,
                    n_backends,
                    exc,
                )

        # Final empty-response retries before fallback
        for final_idx, delay in enumerate(_EMPTY_FINAL_DELAYS, start=1):
            logger.warning(
                "Backend empty response: final retry %d/%d in %.1fs",
                final_idx,
                len(_EMPTY_FINAL_DELAYS),
                delay,
            )
            await asyncio.sleep(delay)
            try:
                self._select_backend()
            except AllBackendsUnhealthyError:
                logger.warning("All backends unhealthy (fast-fail), aborting final retry loop")
                break
            self._normalize_model(cc_request)
            self._active_provider.normalize_request(cc_request)
            try:
                cc_response = await self._make_upstream_request(cc_request, retry_rate_limit=False)
            except UpstreamError as exc:
                last_exc = exc
                idx = self._current_backend_idx
                if idx >= 0:
                    if exc.status == 429:
                        failure_kind = "rate_limit"
                    elif isinstance(exc.body, str) and self._is_cloudflare_block(exc.status, exc.body):
                        failure_kind = "cloudflare"
                    elif exc.status in _AUTH_FAILURE_STATUSES:
                        failure_kind = "auth"
                    else:
                        failure_kind = "hard"
                    self._mark_backend_unhealthy(idx, failure_kind=failure_kind)
                continue
            except Exception as exc:
                last_exc = exc
                idx = self._current_backend_idx
                if idx >= 0:
                    failure_kind = self._provider_error_failure_kind(exc)
                    self._mark_backend_unhealthy(
                        idx,
                        failure_kind=failure_kind,
                        cooldown=self._retry_after_from_exc(exc),
                    )
                continue

            if not self._is_empty_cc_response(cc_response):
                return cc_response
            last_response = cc_response

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
        except web.HTTPRequestEntityTooLarge:
            logger.warning("Chat Completions request body exceeded %d bytes", _CLIENT_MAX_SIZE)
            return self._error_response(
                {"error": {"message": "Request body too large", "type": "invalid_request_error"}},
            )
        except (json.JSONDecodeError, Exception):
            return self._error_response(
                {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
            )

        logger.debug("═══ CHAT COMPLETIONS PASS-THROUGH REQUEST ═══")
        logger.debug("Request body: %s", json.dumps(body, indent=2, ensure_ascii=False))

        cc_request = body
        self._normalize_model(cc_request)
        self._active_provider.normalize_request(cc_request)

        logger.debug("Normalized CC request: %s", json.dumps(cc_request, indent=2, ensure_ascii=False))

        # Compact messages before size check
        self._apply_compaction(cc_request)

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
            error_msg = self._custom_transport_error_message(exc)
            return web.json_response(
                {"error": {"message": error_msg, "type": "internal_error"}},
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

        # Custom-transport providers return Responses API SSE but CC clients
        # expect Chat Completions SSE.  Collect, parse, and re-emit.
        if self._active_provider.use_custom_transport:
            cc_request["_resolved_key"] = self._active_key
            cc_request["_provider_config"] = self._active_provider_config
            self._log_backend_selection()

            n_backends = len(self._backends) if self._backends else 1
            for attempt in range(n_backends):
                raw_chunks: list[bytes] = []

                async def _collect(chunk: bytes, _raw_chunks: list[bytes] = raw_chunks) -> None:
                    _raw_chunks.append(chunk)

                try:
                    await self._active_provider.stream_request(cc_request, _collect)
                except Exception as exc:
                    logger.warning("Custom-transport stream failed: %s", exc)
                    if self._backends and self._current_backend_idx >= 0:
                        kind = self._provider_error_failure_kind(exc)
                        self._mark_backend_unhealthy(
                            self._current_backend_idx,
                            failure_kind=kind,
                            cooldown=self._retry_after_from_exc(exc),
                        )
                        if self._any_healthy_backend(require_streaming=True) and attempt < n_backends - 1:
                            try:
                                self._select_backend(require_streaming=True)
                            except AllBackendsUnhealthyError as all_unhealthy:
                                logger.warning(
                                    "All streaming backends unhealthy (fast-fail), retry_after=%ds",
                                    all_unhealthy.retry_after,
                                )
                                break
                            self._normalize_model(cc_request)
                            self._active_provider.normalize_request(cc_request)
                            cc_request["_resolved_key"] = self._active_key
                            cc_request["_provider_config"] = self._active_provider_config
                            logger.info(
                                "Custom-transport failover: attempt %d/%d (%s), switching backend",
                                attempt + 1,
                                n_backends,
                                exc,
                            )
                            continue
                        # No custom-transport backend healthy — try cross-mode failover to standard backend
                        if self._any_healthy_backend() and attempt < n_backends - 1:
                            try:
                                self._select_backend()
                            except AllBackendsUnhealthyError:
                                logger.warning("Cross-mode failover: all backends unhealthy")
                                break
                            self._normalize_model(cc_request)
                            self._active_provider.normalize_request(cc_request)
                            cc_request.pop("_resolved_key", None)
                            cc_request.pop("_provider_config", None)
                            cc_request.pop("_original_body", None)
                            logger.info(
                                "Cross-mode failover: attempt %d/%d (%s), switching to standard backend",
                                attempt + 1,
                                n_backends,
                                exc,
                            )
                            break
                    error_status, error_type = self._map_provider_error(exc)
                    upstream_status = error_status
                    error_payload = {
                        "error": {"message": str(exc), "type": error_type},
                    }
                    try:
                        await sr.write(f"data: {json.dumps(error_payload)}\n\n".encode())
                        await sr.write(b"data: [DONE]\n\n")
                    except (ConnectionResetError, BrokenPipeError, OSError):
                        logger.debug("Client disconnected before error event")
                    break
                else:
                    raw_bytes = b"".join(raw_chunks)
                    if hasattr(self._active_provider, "parse_stream_to_cc_response"):
                        cc_response = self._active_provider.parse_stream_to_cc_response(raw_bytes)
                    else:
                        from kitty.providers.openai_subscription import OpenAISubscriptionAdapter

                        cc_response = OpenAISubscriptionAdapter._parse_sse_to_response(raw_bytes)

                    # Re-emit as Chat Completions SSE stream
                    response_id = cc_response.get("id", "chatcmpl-sub")
                    created = cc_response.get("created", 0)
                    model = cc_response.get("model", "")

                    def _cc_chunk(
                        delta: dict,
                        fin: str | None = None,
                        usage: dict | None = None,
                        _response_id: str = response_id,
                        _created: int = created,
                        _model: str = model,
                    ) -> bytes:
                        payload: dict = {
                            "id": _response_id,
                            "object": "chat.completion.chunk",
                            "created": _created,
                            "model": _model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": fin}],
                        }
                        if usage is not None:
                            payload["usage"] = usage
                        return f"data: {json.dumps(payload)}\n\n".encode()

                    choice = (cc_response.get("choices") or [{}])[0]
                    msg = choice.get("message", {})
                    finish_reason = choice.get("finish_reason", "stop")

                    # First chunk: role + content/tool_calls start
                    first_delta: dict = {"role": "assistant", "content": None}
                    if msg.get("content"):
                        first_delta["content"] = ""
                    if msg.get("tool_calls"):
                        first_delta["tool_calls"] = [
                            {
                                "index": i,
                                "id": tc.get("id", f"call_{i}"),
                                "type": "function",
                                "function": {"name": tc["function"]["name"], "arguments": ""},
                            }
                            for i, tc in enumerate(msg["tool_calls"])
                        ]
                    try:
                        await sr.write(_cc_chunk(first_delta))

                        # Content delta
                        if msg.get("content"):
                            await sr.write(_cc_chunk({"content": msg["content"]}))

                        # Tool call argument deltas
                        for i, tc in enumerate(msg.get("tool_calls", [])):
                            args = tc.get("function", {}).get("arguments", "")
                            if args:
                                await sr.write(
                                    _cc_chunk(
                                        {"tool_calls": [{"index": i, "function": {"arguments": args}}]},
                                    )
                                )

                        # Finish
                        await sr.write(
                            _cc_chunk({}, fin=finish_reason, usage=cc_response.get("usage")),
                        )
                        await sr.write(b"data: [DONE]\n\n")
                    except (ConnectionResetError, BrokenPipeError, OSError):
                        logger.debug("Client disconnected during custom-transport emit")
                    self._log_usage(cc_response.get("usage"))
                    break

            cc_request.pop("_resolved_key", None)
            cc_request.pop("_provider_config", None)
            # If cross-mode failover switched to a non-custom-transport provider,
            # fall through to the standard streaming path below.
            if not self._active_provider.use_custom_transport:
                logger.info(
                    "Cross-mode failover: entering standard streaming path with %s",
                    type(self._active_provider).__name__,
                )
            else:
                try:
                    await sr.write_eof()
                except (ConnectionResetError, BrokenPipeError, OSError):
                    logger.debug("Client disconnected before stream EOF")
                return sr

        try:
            session = await self._get_session()
            url = self._build_upstream_url()
            headers = self._build_upstream_headers()
            upstream_body = self._active_provider.translate_to_upstream(cc_request)
            logger.debug("Upstream POST → %s", url)

            stream_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=_STREAM_READ_TIMEOUT)

            # Retry loop with backend failover for balancing mode
            n_backends = len(self._backends) if self._backends else 1
            _original_max_attempts = (_MAX_RETRIES + 1) * n_backends
            max_attempts = _original_max_attempts + len(_EMPTY_FINAL_DELAYS)
            for attempt in range(max_attempts):
                if attempt >= _original_max_attempts:
                    delay = _EMPTY_FINAL_DELAYS[attempt - _original_max_attempts]
                    logger.warning(
                        "Empty upstream response: final retry in %.1fs (%d/%d)",
                        delay,
                        attempt + 1,
                        max_attempts,
                    )
                    await asyncio.sleep(delay)
                # Reset stream state for each retry attempt
                line_buffer = ""
                done = False
                stream_error = False
                has_content = False
                chunk_count = 0
                async with session.post(url, json=upstream_body, headers=headers, timeout=stream_timeout) as upstream:
                    upstream_status = upstream.status
                    logger.debug("Upstream response status: %d", upstream.status)

                    if upstream.status not in (200, 201):
                        error_body = await upstream.text()

                        # Cloudflare challenge — never retryable, abort immediately
                        if self._is_cloudflare_block(upstream.status, error_body):
                            _log_cloudflare_block(upstream.status, error_body)
                            error_msg = self._translate_upstream_error(upstream.status, error_body)
                            error_sse = (
                                f"data: {json.dumps({'error': {'message': error_msg, 'type': 'upstream_error'}})}\n\n"
                            )
                            try:
                                await sr.write(error_sse.encode())
                            except (ConnectionResetError, BrokenPipeError, OSError):
                                logger.debug("Client disconnected before Cloudflare error event")
                            if self._backends and self._current_backend_idx >= 0:
                                self._mark_backend_unhealthy(self._current_backend_idx, failure_kind="cloudflare")
                            break

                        logger.error("Upstream error %d: %s", upstream.status, error_body)

                        # In balancing mode: mark unhealthy, try next backend
                        if self._backends and self._current_backend_idx >= 0:
                            kind = "auth" if upstream.status in _AUTH_FAILURE_STATUSES else "hard"
                            self._mark_backend_unhealthy(self._current_backend_idx, failure_kind=kind)
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
                            logger.debug(
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

                    # Success path — line-buffered SSE pass-through stream
                    async for chunk_bytes in upstream.content:
                        try:
                            if done:
                                break
                            chunk_count += 1
                            line_buffer += chunk_bytes.decode("utf-8", errors="replace")
                            while "\n" in line_buffer:
                                line, line_buffer = line_buffer.split("\n", 1)
                                line = line.rstrip("\r")
                                if not line:
                                    continue
                                if not line.startswith("data: "):
                                    # Non-data SSE fields (event:, id:, retry:) not used by CC providers
                                    continue
                                data_str = line[6:]
                                if data_str.strip() == "[DONE]":
                                    raw_line_bytes = f"{line}\n\n".encode()
                                    for translated in self._active_provider.translate_upstream_stream_event(
                                        raw_line_bytes
                                    ):
                                        has_content = True
                                        await sr.write(translated)
                                    done = True
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                except json.JSONDecodeError:
                                    # Non-JSON data line — pass through
                                    raw_line_bytes = f"{line}\n\n".encode()
                                    for translated in self._active_provider.translate_upstream_stream_event(
                                        raw_line_bytes
                                    ):
                                        has_content = True
                                        await sr.write(translated)
                                    continue
                                # Detect in-stream errors
                                if self._is_upstream_stream_error(chunk):
                                    logger.warning("Upstream sent error in stream chunk: %s", data_str[:500])
                                    if self._backends and self._current_backend_idx >= 0:
                                        cooldown = self._get_stream_error_cooldown(self._current_backend_idx)
                                        self._mark_backend_unhealthy(self._current_backend_idx, cooldown=cooldown)
                                    stream_error = True
                                    done = True
                                    break
                                # Extract usage for logging
                                if "usage" in chunk and chunk["usage"] is not None:
                                    last_usage = chunk["usage"]
                                # Forward non-error chunk
                                raw_line_bytes = f"{line}\n\n".encode()
                                for translated in self._active_provider.translate_upstream_stream_event(raw_line_bytes):
                                    has_content = True
                                    await sr.write(translated)
                        except (ConnectionResetError, BrokenPipeError, OSError):
                            logger.debug("Client disconnected during streaming")
                            break

                    logger.debug("Upstream stream ended. chunks=%d done=%s", chunk_count, done)

                    # Flush remaining buffer (last chunk without trailing \n)
                    if not done and line_buffer.strip():
                        line = line_buffer.strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                raw_line_bytes = f"{line}\n\n".encode()
                                for translated in self._active_provider.translate_upstream_stream_event(raw_line_bytes):
                                    has_content = True
                                    await sr.write(translated)
                            else:
                                try:
                                    chunk = json.loads(data_str)
                                    if not self._is_upstream_stream_error(chunk):
                                        if "usage" in chunk and chunk["usage"] is not None:
                                            last_usage = chunk["usage"]
                                        raw_line_bytes = f"{line}\n\n".encode()
                                        for translated in self._active_provider.translate_upstream_stream_event(
                                            raw_line_bytes
                                        ):
                                            has_content = True
                                            await sr.write(translated)
                                except json.JSONDecodeError:
                                    raw_line_bytes = f"{line}\n\n".encode()
                                    for translated in self._active_provider.translate_upstream_stream_event(
                                        raw_line_bytes
                                    ):
                                        has_content = True
                                        await sr.write(translated)
                        line_buffer = ""  # Buffer consumed

                    # Handle in-stream error failover
                    if stream_error:
                        if has_content:
                            # Partial content already sent — retrying would corrupt the stream
                            logger.warning("CC stream error after content emitted; not retrying")
                            error_json = json.dumps(
                                {
                                    "error": {
                                        "message": "Upstream stream error during response",
                                        "type": "upstream_error",
                                    }
                                }
                            )
                            error_sse = f"data: {error_json}\n\n"
                            try:
                                await sr.write(error_sse.encode())
                            except (ConnectionResetError, BrokenPipeError, OSError):
                                logger.debug("Client disconnected before error could be sent")
                            break
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
                    # Mark backend healthy on clean stream completion
                    if self._backends and self._current_backend_idx >= 0:
                        self._mark_backend_healthy(self._current_backend_idx)
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

    def _validate_tool_call_pairing(self, messages: list[dict]) -> list[dict]:
        """Drop tool messages whose ``tool_call_id`` has no matching ``tool_use.id``.

        The upstream rejects requests with code 2013 ("tool call result does
        not follow tool call") when a ``tool`` message references a
        ``tool_call_id`` that no preceding ``assistant(tool_calls)`` message
        declares. This can happen when an upstream SSE stream returns an
        empty/malformed body for a tool result, leaving a partial ``tool``
        message in the conversation. The atomic-block grouping in
        ``_compact_messages`` keeps pairs together, but a corrupt input
        can still contain orphans.

        This helper drops orphan ``tool`` messages in place and logs a
        WARNING for each. Non-tool messages (system, user, assistant)
        are never modified or reordered.

        Args:
            messages: Conversation messages to validate.

        Returns:
            Cleaned messages list with orphans removed.
        """
        seen_tool_use_ids: set[str] = set()
        cleaned: list[dict] = []
        dropped: list[str] = []

        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        seen_tool_use_ids.add(tc["id"])
                cleaned.append(msg)
                continue

            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id and tool_call_id in seen_tool_use_ids:
                    cleaned.append(msg)
                else:
                    dropped.append(str(tool_call_id))
                continue

            # system, user, and any other role pass through
            cleaned.append(msg)

        if dropped:
            logger.warning(
                "Tool-call pairing: dropped %d orphan tool result(s) with "
                "tool_call_id(s) not matching any preceding tool_use: %s",
                len(dropped),
                dropped,
            )

        return cleaned

    def _compact_messages(self, messages: list[dict], max_messages_chars: int | None = None) -> list[dict]:
        """Compact message history to prevent upstream context window overflow.

        Applies three strategies in order:
        1. Tool result truncation: Replace large tool outputs with a size notice.
        2. Head+Tail pruning: Keep system prompt + initial messages (head)
           and the most recent messages (tail), dropping the middle.
           Tool-call/result pairs are kept atomic (never split).
        3. Guaranteed-fit fallback: If the result still exceeds the safe budget,
           iteratively drop oldest blocks (head first, then tail front) until
           it fits. System message and at least one tail block are always preserved.

        Returns the (possibly compacted) messages list.
        """
        if not messages:
            return messages

        original_size = len(json.dumps(messages, ensure_ascii=False))
        compaction_threshold = _COMPACTION_CHAR_THRESHOLD if max_messages_chars is None else max_messages_chars
        guaranteed_max = _COMPACTION_GUARANTEED_MESSAGES_MAX if max_messages_chars is None else max_messages_chars
        if original_size <= compaction_threshold:
            return messages

        # ── Step 1: Truncate large tool results ──────────────────────────
        compacted = []
        for msg in messages:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                content_len = len(msg["content"])
                if content_len > _TOOL_RESULT_TRUNCATION_LIMIT:
                    compacted.append(
                        {
                            **msg,
                            "content": f"[Tool output truncated — original size: {content_len:,} chars]",
                        }
                    )
                else:
                    compacted.append(msg)
            else:
                compacted.append(msg)

        size_after_truncation = len(json.dumps(compacted, ensure_ascii=False))
        if size_after_truncation <= compaction_threshold:
            logger.info(
                "Context compaction: tool result truncation reduced size from %d to %d chars (%d messages)",
                original_size,
                size_after_truncation,
                len(compacted),
            )
            return compacted

        # ── Step 2: Group messages into atomic blocks ────────────────────
        # A block is a single message, or an assistant(tool_calls) + all
        # immediately following tool(result) messages.
        blocks: list[tuple[list[dict], int]] = []  # (messages, total_chars)
        i = 0
        while i < len(compacted):
            block_msgs = [compacted[i]]
            if compacted[i].get("role") == "assistant" and compacted[i].get("tool_calls"):
                # Include all following tool results as one atomic block
                j = i + 1
                while j < len(compacted) and compacted[j].get("role") == "tool":
                    block_msgs.append(compacted[j])
                    j += 1
            block_size = sum(len(json.dumps(m, ensure_ascii=False)) for m in block_msgs)
            blocks.append((block_msgs, block_size))
            i += len(block_msgs)

        # ── Step 3: Identify system blocks (always preserved) ────────────
        system_blocks: list[tuple[list[dict], int]] = []
        remaining_blocks: list[tuple[list[dict], int]] = []
        for block in blocks:
            if not system_blocks and block[0][0].get("role") == "system":
                system_blocks.append(block)
            else:
                remaining_blocks.append(block)

        system_messages = [m for msgs, _ in system_blocks for m in msgs]

        # ── Step 4: Head+Tail pruning on remaining blocks ────────────────
        system_size = sum(s for _, s in system_blocks)
        head_budget = max(0, int(compaction_threshold * 0.2) - system_size)
        tail_budget = compaction_threshold * 0.8

        # Build head: initial context after system messages
        head_blocks: list[tuple[list[dict], int]] = []
        head_size = 0
        for block_msgs, block_size in remaining_blocks:
            if head_size + block_size <= head_budget:
                head_blocks.append((block_msgs, block_size))
                head_size += block_size
            else:
                break

        # Build tail: most recent blocks (respecting min count)
        head_block_ids = {id(msgs) for msgs, _ in head_blocks}
        tail_blocks: list[tuple[list[dict], int]] = []
        tail_size = 0
        min_tail = max(_COMPACTION_TAIL_COUNT, 1)
        for block_msgs, block_size in reversed(remaining_blocks):
            if id(block_msgs) in head_block_ids:
                break
            if len(tail_blocks) >= min_tail and tail_size + block_size > tail_budget:
                break
            tail_blocks.insert(0, (block_msgs, block_size))  # prepend
            tail_size += block_size

        result = list(system_messages)
        for block_msgs, _ in head_blocks:
            result.extend(block_msgs)
        for block_msgs, _ in tail_blocks:
            result.extend(block_msgs)

        result_size = len(json.dumps(result, ensure_ascii=False))

        # ── Step 5: Guaranteed-fit fallback ──────────────────────────────
        # If still too large, iteratively drop oldest blocks until under budget.
        # Drop head blocks first, then oldest tail blocks (keeping at least 1).
        head_dropped = 0
        tail_dropped = 0
        while result_size > guaranteed_max:
            if head_blocks:
                _, popped_size = head_blocks.pop(0)
                head_dropped += 1
                result_size -= popped_size
            elif len(tail_blocks) > 1:
                _, popped_size = tail_blocks.pop(0)  # drop oldest tail
                tail_dropped += 1
                result_size -= popped_size
            else:
                # Cannot shrink further without losing system or the last tail block
                break

        # Rebuild result from surviving blocks
        result = list(system_messages)
        for block_msgs, _ in head_blocks:
            result.extend(block_msgs)
        for block_msgs, _ in tail_blocks:
            result.extend(block_msgs)

        if head_dropped or tail_dropped:
            budget_met = result_size <= guaranteed_max
            logger.info(
                "Context compaction: guaranteed-fit fallback dropped %d head and %d tail blocks, "
                "reducing messages from %d to %d and size from %d to %d chars (budget %s)",
                head_dropped,
                tail_dropped,
                len(compacted),
                len(result),
                original_size,
                result_size,
                "met" if budget_met else "NOT met — cannot shrink further",
            )
        else:
            logger.info(
                "Context compaction: head+tail pruning reduced messages from %d to %d and size from %d to %d chars",
                len(compacted),
                len(result),
                original_size,
                result_size,
            )

        # Post-condition: every tool result must reference a kept tool_use.
        # Atomic-block grouping keeps pairs together, but a corrupt input
        # can still contain orphans (e.g. an upstream-empty SSE response
        # that the bridge faithfully recorded as a complete tool message).
        result = self._validate_tool_call_pairing(result)
        return result

    def _get_max_context_chars(self) -> int:
        """Compute the max request size in chars based on model context window.

        For balancing profiles, uses the smallest context across all backends.
        For single backends, uses the model's context window.
        Falls back to _MAX_REQUEST_CHARS when no model info is available.
        The result is capped at _MAX_REQUEST_CHARS (absolute safety limit).
        """
        if self._backends:
            backend_tuples = [(b[0].provider_type, b[2].model, b[2].provider_config) for b in self._backends]
            context_tokens = get_balancing_min_context_tokens(backend_tuples)
            return min(tokens_to_chars(context_tokens), _MAX_REQUEST_CHARS)

        if self._active_model:
            provider_type = self._active_provider.provider_type
            context_tokens = get_model_context_tokens(
                provider_type,
                self._active_model,
                self._active_provider_config,
            )
            return min(tokens_to_chars(context_tokens), _MAX_REQUEST_CHARS)

        return _MAX_REQUEST_CHARS

    def _apply_compaction(self, cc_request: dict) -> None:
        """Compact request messages in place before applying request-size guardrails.

        If the translated request contains a ``messages`` list, this method
        compacts it using ``_compact_messages()``. The compaction budget is
        reduced to account for non-message payload (tools, model, metadata)
        so that the full request fits within ``_MAX_REQUEST_CHARS``.

        Requests without messages are left unchanged.

        Safety net for ``use_native_messages`` providers: the compaction
        grouping logic in ``_compact_messages`` is Chat-Completions-format
        aware — it groups ``assistant(tool_calls) + tool(result)`` as an
        atomic block. When the provider forwards the raw Anthropic Messages
        body, ``tool_use`` is a content block on the assistant message and
        the following message is ``user`` with a ``tool_result`` block; the
        grouping misses the pairing and the pruner can drop the
        ``assistant(tool_use)`` while keeping the following
        ``user(tool_result)``. MiniMax then returns ``invalid params, tool
        call result does not follow tool call (2013)``. We log a warning if
        the provider is in native passthrough mode and the request exceeds
        the compaction threshold so operators see a signal if a future
        provider re-introduces this combination.
        """
        if "messages" not in cc_request:
            return

        # Safety-net warning: if the active provider is in native passthrough
        # mode and the request is large enough to risk compaction, log it.
        if self._active_provider.use_native_messages:
            request_size = len(json.dumps(cc_request, ensure_ascii=False))
            if request_size > _COMPACTION_CHAR_THRESHOLD:
                logger.warning(
                    "Native passthrough provider (%s) sending large request (%d chars) "
                    "above compaction threshold (%d). Compaction grouping is "
                    "Chat-Completions-format aware and may orphan tool_result blocks "
                    "whose tool_use was in the head. Set native_messages=False in the "
                    "profile or pre-compact the request to avoid this.",
                    type(self._active_provider).__name__,
                    request_size,
                    _COMPACTION_CHAR_THRESHOLD,
                )

        # Compute the size of the non-message payload (tools, model, etc.)
        # so that messages + overhead stays within _MAX_REQUEST_CHARS.
        non_msg = {k: v for k, v in cc_request.items() if k != "messages"}
        overhead = len(json.dumps(non_msg, ensure_ascii=False))
        # Budget for messages = max request size - overhead (with safety margin)
        max_chars = self._get_max_context_chars()
        messages_budget = max(0, max_chars - overhead - 10_000)  # 10K margin for JSON punctuation

        cc_request["messages"] = self._compact_messages(
            cc_request["messages"],
            max_messages_chars=messages_budget,
        )

        # Final safety net: drop any orphan tool_result blocks whose
        # tool_call_id has no matching tool_use. Compaction's atomic-block
        # grouping keeps well-formed pairs together, but a corrupt input
        # (e.g. an upstream-empty SSE response recorded as a partial tool
        # message) can still slip through. Without this pass the next
        # request would trigger upstream code 2013 ("tool call result
        # does not follow tool call").
        cc_request["messages"] = self._validate_tool_call_pairing(cc_request["messages"])

    def _check_request_size(self, cc_request: dict) -> web.Response | None:
        """Return a 400 error if the translated request exceeds the safe size limit.

        Returns None if the request is within limits, or a json_response to return
        immediately if it's too large.
        """
        request_size = len(json.dumps(cc_request, ensure_ascii=False))
        max_chars = self._get_max_context_chars()
        if request_size > max_chars:
            logger.warning(
                "Request body size %d chars exceeds safe limit (%d) — rejecting",
                request_size,
                max_chars,
            )
            return self._error_response(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": (
                            f"Request too large ({request_size / 1024:.0f} KB). "
                            "The conversation context has grown beyond what the upstream provider can handle, "
                            "even after automatic compaction. "
                            "Use /clear to reset the conversation context."
                        ),
                    },
                },
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

    _is_cloudflare_block = staticmethod(is_cloudflare_block)

    @staticmethod
    def _should_retry_stream(status: int, error_body: str) -> bool:
        """Return True if a streaming error should trigger a retry / backend switch."""
        if BridgeServer._is_cloudflare_block(status, error_body):
            return False
        if BridgeServer._is_non_retryable_error_code(status, error_body):
            return False
        if status in _RETRYABLE_STATUSES:
            return True
        return BridgeServer._is_rate_limit_error(status, error_body)

    @staticmethod
    def _is_non_retryable_error_code(status: int, body: object) -> bool:
        """Return True if the upstream error body contains a non-retryable error code.

        Some providers return permanent-failure error codes with 5xx HTTP status
        (e.g. Z.AI returns code 1211 "Unknown Model" on HTTP 500).  These should
        not be retried because the same request will always fail.

        Handles both structured JSON bodies (``{"error": {"code": "1211"}}``) and
        plain-text bracket-format errors (``[1211][Unknown Model][request_id]``)
        returned by some Anthropic-protocol endpoints.
        """
        code, _message = BridgeServer._extract_error_fields(body)
        if code and code in _NON_RETRYABLE_ERROR_CODES:
            return True
        # Z.AI Anthropic endpoint returns errors as plain text in the format
        # ``[code][message][request_id]``, which _extract_error_fields cannot parse.
        if isinstance(body, str):
            for non_retryable_code in _NON_RETRYABLE_ERROR_CODES:
                if f"[{non_retryable_code}]" in body:
                    return True
        return False

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

        if status == 403 and isinstance(body, str) and BridgeServer._is_cloudflare_block(status, body):
            return (
                "Cloudflare bot detection blocked the upstream request. "
                "This may be transient — a retry after a short delay may resolve it. "
                "This is not an API key problem."
            )

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

        # Minimax code 2013 reused for tool-call validation:
        # "tool call result does not follow tool call". This is a client-side
        # conversation-state error — the request body has a tool_result
        # block whose tool_use_id has no matching tool_use. The conversation
        # needs to be reset before the request can succeed against the
        # same backend.
        is_tool_call_validation = "tool call result" in searchable or "does not follow tool call" in searchable
        if is_tool_call_validation:
            return (
                "The conversation history has a broken tool_use/tool_result pairing "
                "(upstream returned: tool call result does not follow tool call). "
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

    @staticmethod
    def _map_provider_error(exc: Exception) -> tuple[int, str]:
        """Map a custom-transport exception to (http_status, error_type).

        Uses ProviderError.http_status when available; falls back to 502
        for unknown exceptions.  Returns an Anthropic-compatible error type
        string for the Messages API.
        """
        status = exc.http_status if isinstance(exc, ProviderError) and exc.http_status else 502

        if status == 429:
            return status, "rate_limit_error"
        if status in (401, 403):
            return status, "authentication_error"
        return status, "api_error"

    @staticmethod
    def _provider_error_failure_kind(exc: Exception) -> str:
        """Determine the correct failure_kind for _mark_backend_unhealthy.

        Uses ProviderError.http_status to pick the right classification so
        balancing profiles get proper cooldown and failover behavior.
        """
        if isinstance(exc, ProviderError):
            if exc.is_cloudflare:
                return "cloudflare"
            if exc.http_status == 429:
                return "rate_limit"
            if exc.http_status in (401, 403):
                return "auth"
        if _is_transport_error(exc):
            return "transport"
        return "hard"

    @staticmethod
    def _retry_after_from_exc(exc: Exception) -> int | None:
        """Extract a retry-after cooldown from a ProviderError, if present."""
        if isinstance(exc, ProviderError):
            return getattr(exc, "retry_after", None)
        return None

    def _custom_transport_error_message(self, exc: Exception) -> str:
        """Build an actionable error message for custom-transport failures.

        For auth failures, includes the backend profile name and re-login
        command. Non-auth errors pass through unchanged.
        """
        msg = str(exc)
        if self._provider_error_failure_kind(exc) != "auth":
            return msg
        profile_name = self._profile_name
        if self._backends and 0 <= self._current_backend_idx < len(self._backends):
            profile_name = self._backends[self._current_backend_idx][2].name
        provider = self._active_provider
        auth_command = getattr(provider, "auth_command", None) or "kitty auth openai"
        return (
            f"Authentication failed for profile '{profile_name}'. Please re-login with: {auth_command}. Details: {msg}"
        )

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

    async def _make_upstream_request(self, cc_request: dict, *, retry_rate_limit: bool = True) -> dict:
        """Send a non-streaming request upstream.

        For providers with ``use_custom_transport=True``, delegates to the
        provider's ``make_request()`` method (e.g. boto3 for Bedrock).
        Otherwise uses aiohttp with retry/backoff.

        Args:
            cc_request: The request payload in CC format.
            retry_rate_limit: When False, 429 is not retried on this backend
                (the caller handles failover instead).

        Returns the upstream response dict (in CC format) on success.
        Raises UpstreamError(status, body) on non-retryable or exhausted failures.
        """
        if self._active_provider.use_custom_transport:
            cc_request["_resolved_key"] = self._active_key
            cc_request["_provider_config"] = self._active_provider_config
            self._log_backend_selection()
            return await self._active_provider.make_request(cc_request)

        session = await self._get_session()
        url = self._build_upstream_url()
        headers = self._build_upstream_headers()
        upstream_body = self._active_provider.translate_to_upstream(cc_request)

        request_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=_STREAM_READ_TIMEOUT)

        last_status = 0
        last_body: object = {}
        for attempt in range(_MAX_RETRIES + 1):
            async with session.post(url, json=upstream_body, headers=headers, timeout=request_timeout) as resp:
                last_status = resp.status
                try:
                    last_body = await resp.json()
                except Exception:
                    last_body = await resp.text()

                if last_status < 400:
                    if (
                        self._active_provider.use_native_messages
                        and isinstance(last_body, dict)
                        and last_body.get("type") == "message"
                    ):
                        return last_body
                    return self._active_provider.translate_from_upstream(last_body)

                # In balancing mode (retry_rate_limit=False), raise 429 immediately
                # so the caller can fail over to another backend.
                if last_status == 429 and not retry_rate_limit:
                    raise UpstreamError(last_status, last_body)

                if self._is_non_retryable_error_code(last_status, last_body):
                    raise UpstreamError(last_status, last_body)

                if last_status in _RETRYABLE_STATUSES and attempt < _MAX_RETRIES:
                    delay = _BACKOFF_BASE * (2**attempt)
                    logger.debug(
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
