"""The provider-session aiohttp recording upstream.

`.system_design/TEST_SUITE.md` §5.5, §7.2 · plan task **T-B1** (KBR-40).

§7.2 gives each of the bridge's five client configurations its own recorder. The
primary one (T-W4, :mod:`harness.recorder`) observes the sessions
:meth:`BridgeServer._session_for` owns, which is every default-transport
adapter. This one observes the two product paths that build an **aiohttp session
of their own** and therefore never appear there:

* ``ollama_cloud``, which posts Ollama's native ``/api/chat`` and resolves
  ``provider_config["base_url"]`` inside its own request path rather than
  through ``build_base_url``;
* the OpenAI **login** OAuth token leg (:mod:`kitty.auth.openai_oauth`), which
  runs at startup, before anything else has been proven (§5.5).

**Why this subclasses T-W4's recorder rather than being a second server.**
§7.2.1 catalogues six ways an aiohttp recorder can look correct and lie —
``raw_headers`` decoded latin-1, ``rel_url.raw_path`` rather than the decoded
``path``, the ``Host`` header read directly rather than ``request.host``,
``client_max_size=0`` supplied through a ``request_factory``,
``auto_decompress=False``, and connection logging through
``web.Server.connection_made`` keyed on the handler object. A second
implementation of that server would be a second chance to get each of them
wrong, and §7.3 states the principle from the proxy's side: *"two proxy
implementations is how two harnesses come to disagree about what 'tunnelled'
means."* What actually differs between the two recorders is **vocabulary** — the
format served, the suffix that selects a reply, and what a minimal success looks
like — so this subclass overrides those three things and nothing else.

**It imports nothing from ``src/kitty``, and must not**, for the reason
:mod:`harness.recorder` states: §3.3.1's independent-oracle rule. The product
import this delivery needs lives in :mod:`harness.provider_aiohttp`, which binds
an adapter; nothing here asks kitty how to read a request.
"""

from __future__ import annotations

import json
from typing import Any

from harness.contract import CapturedRequest, WireFormat

# ``_wants_stream`` is private and deliberately imported anyway: whether a body
# asked for a stream is one decision, and Ollama spells it the same way the other
# formats do. A copy here would be a second source of truth, and the alternative
# — exporting it — means editing ``recorder.py``, which five tickets consume.
from harness.recorder import RecordingUpstream, Reply, _wants_stream

__all__ = [
    "ProviderRecordingUpstream",
    "OLLAMA_CHAT_SUFFIX",
    "OAUTH_TOKEN_SUFFIX",
    "format_for_provider_path",
    "is_oauth_token_path",
    "ollama_success_body",
    "ollama_success_stream",
    "oauth_token_body",
]

#: The suffix that selects an Ollama ``/api/chat`` reply. A **suffix** for the
#: reason §7.2 gives: a recorder impersonates a provider, and providers dispatch
#: on the URL. ``ollama_cloud`` posts to the bare path today, but a redirected
#: base URL carrying a path prefix would still have to match.
OLLAMA_CHAT_SUFFIX = "/api/chat"

#: The suffix that selects an OAuth token reply. Both of the login leg's two
#: POSTs — the authorization-code exchange and the id_token exchange — go to this
#: one endpoint, so one suffix covers the whole leg.
OAUTH_TOKEN_SUFFIX = "/oauth/token"

#: The one wire format this recorder serves. §7.2 assigns it the Ollama
#: ``/api/chat`` shape; OpenAI Responses is T-B2's and Bedrock Converse is T-B3's.
_SERVED_FORMAT = WireFormat.OLLAMA_CHAT

#: The model and text every minimal success carries. Non-empty deliberately:
#: every emptiness judgement in ``server.py`` keys on content being present, and
#: an empty reply costs the 80-second retry ladder (§7.2.1).
_REPLY_MODEL = "recorder-model"
_REPLY_TEXT = "ok"


def format_for_provider_path(path: str) -> WireFormat | None:
    """Return the wire format a request at ``path`` selects, or ``None``.

    The counterpart of :func:`harness.recorder.format_for_path` for this
    recorder's own suffix table, and pure for the same reason: it answers the
    question and records nothing, so an assertion can use it without mutating
    the evidence it judges.

    Args:
        path: The request's raw path.

    Returns:
        :attr:`~harness.contract.WireFormat.OLLAMA_CHAT` when ``path`` names the
        Ollama chat endpoint, and ``None`` when no suffix matches — ``None``
        rather than a default, because "no rule applies" and "the fallback
        applies" are different facts and only the caller knows which it wants.
    """
    if path.endswith(OLLAMA_CHAT_SUFFIX):
        return _SERVED_FORMAT
    return None


def is_oauth_token_path(path: str) -> bool:
    """Return whether ``path`` addresses the OAuth token endpoint.

    Separate from :func:`format_for_provider_path` because a token exchange has
    no :class:`~harness.contract.WireFormat` and must not acquire one: §3.3.1
    pairs every format with a wire reader, and no projection reads a
    form-encoded token grant. §7.5 left that choice to this task; the decision
    and its alternatives are recorded in §7.2.

    Args:
        path: The request's raw path.

    Returns:
        Whether the OAuth token reply applies.
    """
    return path.endswith(OAUTH_TOKEN_SUFFIX)


def ollama_success_body() -> dict[str, Any]:
    """Return the smallest non-streaming ``/api/chat`` success.

    "Minimal" means the fewest keys that still read as a success to the bridge,
    not a faithful sample of a provider's response — the same rule
    :func:`harness.recorder.minimal_success_body` states. The keys are the ones
    Ollama's own API reference documents for a non-streaming chat response and
    the ones ``OllamaCloudAdapter.translate_from_upstream`` reads.

    Returns:
        The response body, ready to serialize.
    """
    return {
        "model": _REPLY_MODEL,
        "message": {"role": "assistant", "content": _REPLY_TEXT},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 1,
        "eval_count": 1,
    }


def ollama_success_stream() -> tuple[bytes, ...]:
    """Return the smallest streamed ``/api/chat`` success.

    **NDJSON, not SSE.** Ollama streams a series of JSON objects separated by
    newlines, and ``OllamaCloudAdapter.stream_request`` splits on ``"\\n"`` and
    parses each line; an SSE ``data:`` frame would parse as nothing, the adapter
    would emit no content, and the bridge would judge the response empty — the
    80-second ladder rather than a loud failure.

    The final object carries ``done`` and ``done_reason`` and **no** message:
    that is the shape the adapter's loop breaks on to emit its finish reason.

    Returns:
        The NDJSON lines, in order, each already newline-terminated.
    """
    return (
        json.dumps(
            {
                "model": _REPLY_MODEL,
                "message": {"role": "assistant", "content": _REPLY_TEXT},
                "done": False,
            }
        ).encode()
        + b"\n",
        json.dumps(
            {
                "model": _REPLY_MODEL,
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 1,
            }
        ).encode()
        + b"\n",
    )


def oauth_token_body() -> dict[str, Any]:
    """Return a token response the OAuth login leg can read.

    Carries every field the leg requires: ``_exchange_code_for_tokens`` rejects
    a response missing ``access_token`` or ``id_token``, and
    ``_exchange_id_token_for_api_key`` rejects one missing ``openai_api_key``.
    Both are served from one body because both POSTs address one endpoint.

    The ``id_token`` is a syntactically valid unsigned JWT with a trivial payload
    — the leg base64-decodes the payload without verifying the signature — and
    is not a credential of any kind.

    Returns:
        The token response, ready to serialize.
    """
    return {
        "access_token": "recorder-access-token",
        "refresh_token": "recorder-refresh-token",
        "id_token": "eyJhbGciOiJub25lIn0.eyJzdWIiOiJyZWNvcmRlciJ9.",
        "expires_in": 3600,
        "openai_api_key": "recorder-api-key",
    }


class ProviderRecordingUpstream(RecordingUpstream):
    """T-W4's recorder, speaking the provider-session vocabulary.

    Everything the conformance contract judges — what is captured, in what
    order, on which connection, from which peer port — is inherited unchanged,
    so this recorder is judged by the same fourteen checks and cannot drift from
    the primary one on any of them. Three things are overridden: the format it
    accepts, the suffix table it dispatches on, and what it replies with.

    A plain subclass rather than a ``@dataclass`` one, because it adds no
    fields; ``test_bridge_falsification._BlindRecorder`` is the same shape.
    """

    def __post_init__(self) -> None:
        """Reject a format this recorder does not serve.

        The base implementation is **replaced, not extended**: it is validation
        and nothing else today, and it would reject every format this recorder
        serves. If T-W4 ever adds setup there, this override has to call it —
        which is the one reason to re-read this docstring.

        Raises:
            ValueError: When ``default_format`` is not
                :attr:`~harness.contract.WireFormat.OLLAMA_CHAT`. The base class
                would reject it too, but for the wrong reason and with the wrong
                message — it names the primary recorder's two formats, neither of
                which this one serves.
        """
        if self.default_format is not _SERVED_FORMAT:
            raise ValueError(
                f"the provider-session recorder serves {_SERVED_FORMAT.value}, not {self.default_format.value}"
            )

    def _format_for(self, path: str) -> WireFormat:
        """Return the wire format to answer a request at ``path`` in.

        The recording half of the lookup, overridden rather than extended
        because the two recorders' suffix tables are disjoint: neither serves a
        format the other does, so a shared table would only ever answer a
        request one of them could not reply to.

        Args:
            path: The request's raw path.

        Returns:
            The matched format, or :attr:`default_format` — recording the miss,
            which :meth:`assert_all_paths_matched` turns into a failure at
            teardown.
        """
        matched = format_for_provider_path(path)
        if matched is not None:
            return matched

        self.unmatched.append(path)
        return self.default_format

    async def _default_responder(self, captured: CapturedRequest, response: Reply) -> None:
        """Reply with the minimal success for the request's endpoint.

        Args:
            captured: The recorded request.
            response: The unprepared response to write through.
        """
        # The OAuth leg is answered before any format lookup: it has no
        # `WireFormat` and must not be recorded as an unmatched path, which is
        # what consulting the format table would do.
        if is_oauth_token_path(captured.path):
            await self._reply_json(response, oauth_token_body())
            return

        # Called for its recording side effect as much as its answer: every
        # reply this recorder sends is Ollama-shaped, but a path that selected
        # none has to be reported at teardown.
        self._format_for(captured.path)

        if _wants_stream(captured.body):
            await response.begin(200, {"Content-Type": "application/x-ndjson"})
            for chunk in ollama_success_stream():
                await response.write(chunk)
            await response.write_eof()
            return

        await self._reply_json(response, ollama_success_body())

    @staticmethod
    async def _reply_json(response: Reply, payload: dict[str, Any]) -> None:
        """Send one complete JSON body.

        Args:
            response: The unprepared response.
            payload: The body to serialize.
        """
        encoded = json.dumps(payload).encode()
        # Set the length before preparing: aiohttp would otherwise chunk the
        # reply, and a chunked body is a difference from a real provider that
        # nothing in the harness would explain to the next reader.
        response.content_length = len(encoded)
        await response.begin(200, {"Content-Type": "application/json"})
        await response.write(encoded)
        await response.write_eof()
