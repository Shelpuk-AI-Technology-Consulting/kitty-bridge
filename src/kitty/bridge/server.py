"""Bridge HTTP server — protocol-aware proxy between coding agents and upstream providers."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
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
_CLIENT_MAX_SIZE = 16 * 1024 * 1024  # 16 MiB


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
    ) -> None:
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
            isinstance(h, logging.FileHandler)
            and getattr(h, "_kitty_bridge_log", False)
            for h in bridge_logger.handlers
        )
        if not has_bridge_handler:
            fh = logging.FileHandler(_DEBUG_LOG_PATH, mode="a", encoding="utf-8")
            fh._kitty_bridge_log = True  # type: ignore[attr-defined]
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s │ %(message)s",
                datefmt="%H:%M:%S",
            ))
            bridge_logger.addHandler(fh)

        return _DEBUG_LOG_PATH

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start_async(self) -> int:
        """Create the aiohttp app, register routes, start listening. Returns bound port."""
        self._log_path = self._setup_debug_logging()
        self._app = web.Application(client_max_size=_CLIENT_MAX_SIZE)
        self._register_routes(self._app)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        for site_obj in self._runner.sites:
            if isinstance(site_obj, web.TCPSite):
                addrs = site_obj._server.sockets  # type: ignore[union-attr]
                if addrs:
                    self._port = addrs[0].getsockname()[1]
                    break
        logger.info("Bridge server started on %s:%d", self._host, self._port)
        logger.info("Debug log: %s", self._log_path)
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
        logger.info("Bridge server stopped")

    def stop(self) -> None:
        """Synchronous wrapper around stop_async."""
        asyncio.get_event_loop().run_until_complete(self.stop_async())

    # ── Route registration ────────────────────────────────────────────────

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_get("/healthz", self._handle_healthz)
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

    # ── Health check ──────────────────────────────────────────────────────

    async def _handle_healthz(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    # ── Responses API handler ─────────────────────────────────────────────

    async def _handle_responses(self, request: web.Request) -> web.StreamResponse:
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
        self._provider.normalize_request(cc_request)

        logger.debug("Translated CC request: %s", json.dumps(cc_request, indent=2, ensure_ascii=False))

        if cc_request.get("stream"):
            return await self._stream_responses(request, body, translator, cc_request)

        try:
            cc_response = await self._make_upstream_request(cc_request)
        except UpstreamError as exc:
            return web.json_response(
                {"error": {"code": "upstream_error", "message": str(exc)}},
                status=exc.status,
            )
        except Exception as exc:
            return web.json_response(
                {"error": {"code": "internal_error", "message": str(exc)}},
                status=500,
            )

        result = translator.translate_response(cc_response)
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
        for event in translator.translate_stream_start(response_id, model):
            logger.debug("SSE → %s", event.split("\n", 1)[0] if "\n" in event else event[:120])
            await sr.write(event.encode())

        upstream_status = None
        terminal_status = "completed"
        try:
            session = await self._get_session()
            url = self._build_upstream_url()
            headers = self._build_upstream_headers()
            logger.debug("Upstream POST → %s", url)

            upstream_body = self._provider.translate_to_upstream(cc_request)
            async with session.post(url, json=upstream_body, headers=headers) as upstream:
                upstream_status = upstream.status
                logger.debug("Upstream response status: %d", upstream.status)
                logger.debug("Upstream response headers: %s", dict(upstream.headers))

                if upstream.status not in (200, 201):
                    terminal_status = "incomplete"
                    error_body = await upstream.text()
                    logger.error("Upstream error %d: %s", upstream.status, error_body)
                    error_event = responses_format_error(
                        {"code": "upstream_error", "message": error_body},
                        seq=translator._next_seq(),
                    )
                    await sr.write(error_event.encode())
                else:
                    line_buffer = ""
                    chunk_count = 0
                    done = False
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
                                events = translator.translate_stream_chunk(response_id, chunk)
                                for event in events:
                                    logger.debug("SSE → %s", event.split("\n", 1)[0][:120])
                                    await sr.write(event.encode())

                    logger.debug("Upstream stream ended. chunks=%d done=%s remaining_buffer=%d chars",
                                 chunk_count, done, len(line_buffer))

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
                                    for event in events:
                                        logger.debug("SSE (flush) → %s", event.split("\n", 1)[0][:120])
                                        await sr.write(event.encode())
                                except json.JSONDecodeError:
                                    logger.warning("Failed to parse flushed SSE data: %s", data_str[:200])

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
        self._provider.normalize_request(cc_request)
        self._thinking_warned = translator.thinking_warned

        logger.debug("Translated CC request: %s", json.dumps(cc_request, indent=2, ensure_ascii=False))

        if cc_request.get("stream"):
            return await self._stream_messages(request, body, translator, cc_request)

        try:
            cc_response = await self._make_upstream_request(cc_request)
        except UpstreamError as exc:
            error_msg = self._translate_upstream_error(exc.status, str(exc))
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

        sr = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await sr.prepare(request)

        try:
            session = await self._get_session()
            url = self._build_upstream_url()
            headers = self._build_upstream_headers()
            upstream_body = self._provider.translate_to_upstream(cc_request)

            async with session.post(url, json=upstream_body, headers=headers) as upstream:
                if upstream.status not in (200, 201):
                    error_body = await upstream.text()
                    error_msg = self._translate_upstream_error(upstream.status, error_body)
                    error_event = messages_format_error(
                        {"type": "error", "error": {"type": "api_error", "message": error_msg}},
                    )
                    await sr.write(error_event.encode())
                else:
                    line_buffer = ""
                    done = False
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
                                events = translator.translate_stream_chunk(message_id, model, chunk)
                                for event in events:
                                    await sr.write(event.encode())

                    # Flush remaining buffer (last chunk without trailing \n)
                    if not done and line_buffer.strip():
                        line = line_buffer.strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() != "[DONE]":
                                try:
                                    chunk = json.loads(data_str)
                                    events = translator.translate_stream_chunk(message_id, model, chunk)
                                    for event in events:
                                        await sr.write(event.encode())
                                except json.JSONDecodeError:
                                    logger.warning("Failed to parse flushed SSE data: %s", data_str[:200])

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

        # Client may disconnect before write_eof() completes — benign race.
        try:
            await sr.write_eof()
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("Client disconnected before stream EOF for %s", message_id)
        return sr

    async def _handle_gemini(self, request: web.Request) -> web.StreamResponse:
        """Handle Gemini generateContent / streamGenerateContent requests."""
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
        self._provider.normalize_request(cc_request)

        logger.debug("Translated CC request: %s", json.dumps(cc_request, indent=2, ensure_ascii=False))

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
            cc_response = await self._make_upstream_request(cc_request)
        except UpstreamError as exc:
            return web.json_response(
                {"error": {"code": exc.status, "message": str(exc)}},
                status=exc.status,
            )
        except Exception as exc:
            return web.json_response(
                {"error": {"code": 500, "message": str(exc)}},
                status=500,
            )

        result = translator.translate_response(cc_response)
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

        try:
            session = await self._get_session()
            url = self._build_upstream_url()
            headers = self._build_upstream_headers()
            logger.debug("Upstream POST → %s", url)
            upstream_body = self._provider.translate_to_upstream(cc_request)

            async with session.post(url, json=upstream_body, headers=headers) as upstream:
                if upstream.status not in (200, 201):
                    error_body = await upstream.text()
                    logger.error("Upstream error %d: %s", upstream.status, error_body)
                    error_sse = f"data: {json.dumps({'error': {'code': upstream.status, 'message': error_body}})}\n\n"
                    try:
                        await sr.write(error_sse.encode())
                    except (ConnectionResetError, BrokenPipeError, OSError):
                        logger.debug("Client disconnected before upstream error could be sent")
                else:
                    line_buffer = ""
                    done = False
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
                                events = translator.translate_stream_chunk(chunk)
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
                                    for event in events:
                                        await sr.write(event.encode())
                                except json.JSONDecodeError:
                                    logger.warning("Failed to parse flushed SSE data: %s", data_str[:200])

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

    async def _handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        """Handle Chat Completions pass-through requests.

        No translation is needed — the agent sends CC format and the upstream
        also expects CC format.  We only apply model normalization and provider
        normalization.
        """
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
        self._provider.normalize_request(cc_request)

        logger.debug("Normalized CC request: %s", json.dumps(cc_request, indent=2, ensure_ascii=False))

        if cc_request.get("stream"):
            return await self._stream_chat_completions(request, cc_request)

        try:
            cc_response = await self._make_upstream_request(cc_request)
        except UpstreamError as exc:
            return web.json_response(
                {"error": {"message": str(exc), "type": "upstream_error"}},
                status=exc.status,
            )
        except Exception as exc:
            return web.json_response(
                {"error": {"message": str(exc), "type": "internal_error"}},
                status=500,
            )

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
        try:
            session = await self._get_session()
            url = self._build_upstream_url()
            headers = self._build_upstream_headers()
            logger.debug("Upstream POST → %s", url)
            upstream_body = self._provider.translate_to_upstream(cc_request)

            async with session.post(url, json=upstream_body, headers=headers) as upstream:
                upstream_status = upstream.status
                logger.debug("Upstream response status: %d", upstream.status)

                if upstream.status not in (200, 201):
                    error_body = await upstream.text()
                    logger.error("Upstream error %d: %s", upstream.status, error_body)
                    error_sse = f"data: {json.dumps({'error': {'message': error_body, 'type': 'upstream_error'}})}\n\n"
                    try:
                        await sr.write(error_sse.encode())
                    except (ConnectionResetError, BrokenPipeError, OSError):
                        logger.debug("Client disconnected before upstream error could be sent")
                else:
                    async for chunk_bytes in upstream.content:
                        try:
                            for translated in self._provider.translate_upstream_stream_event(chunk_bytes):
                                await sr.write(translated)
                        except (ConnectionResetError, BrokenPipeError, OSError):
                            logger.debug("Client disconnected during streaming")
                            break

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
        if self._model is not None:
            original = cc_request.get("model")
            cc_request["model"] = self._model
            if original != self._model:
                logger.debug("Overrode model: %s -> %s", original, self._model)

        model = cc_request.get("model")
        if model:
            normalized = self._provider.normalize_model_name(model)
            if normalized != model:
                logger.debug("Normalized model: %s -> %s", model, normalized)
            cc_request["model"] = normalized

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # LLM calls can be very long (large context, extended thinking).
            # Remove the total timeout so streaming responses are never cut short.
            # Keep a connect timeout to fail fast on network issues.
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session


    @staticmethod
    def _translate_upstream_error(status: int, body: str) -> str:
        """Translate an upstream HTTP error into a user-friendly message.

        For auth errors (401/403), returns a clear message indicating the
        API key issue. For other errors, returns the raw body.
        """
        if status in (401, 403):
            return (
                f"Upstream authentication failed (HTTP {status}): "
                "API key is invalid, expired, or lacks permission. "
                "Update your API key with 'kitty setup'."
            )
        return body

    def _build_upstream_url(self) -> str:
        base = self._provider.build_base_url(self._provider_config).rstrip("/")
        model = self._model or ""
        path = self._provider.get_upstream_path(model)
        return f"{base}{path}"

    def _build_upstream_headers(self) -> dict[str, str]:
        return self._provider.build_upstream_headers(self._resolved_key)

    async def _make_upstream_request(self, cc_request: dict) -> dict:
        """Send a non-streaming request upstream.

        For providers with ``use_custom_transport=True``, delegates to the
        provider's ``make_request()`` method (e.g. boto3 for Bedrock).
        Otherwise uses aiohttp with retry/backoff.

        Returns the upstream response dict (in CC format) on success.
        Raises UpstreamError(status, body) on non-retryable or exhausted failures.
        """
        if self._provider.use_custom_transport:
            cc_request["_resolved_key"] = self._resolved_key
            cc_request["_provider_config"] = self._provider_config
            return await self._provider.make_request(cc_request)

        session = await self._get_session()
        url = self._build_upstream_url()
        headers = self._build_upstream_headers()
        upstream_body = self._provider.translate_to_upstream(cc_request)

        last_status = 0
        last_body: dict = {}
        for attempt in range(_MAX_RETRIES + 1):
            async with session.post(url, json=upstream_body, headers=headers) as resp:
                last_status = resp.status
                try:
                    last_body = await resp.json()
                except Exception:
                    last_body = {}

                if last_status < 400:
                    return self._provider.translate_from_upstream(last_body)

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
