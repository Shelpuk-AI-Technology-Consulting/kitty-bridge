"""A string ``input`` on ``/v1/responses`` is the same request as the array form.

``.system_design/TEST_SUITE.md`` §1 (I1 Message Fidelity) · §3.2.3 (serialization
paths) · §6.2.1 (``not_a_server_error``) · Jira **KBR-144**.

OpenAI's ``CreateResponse`` schema defines ``input`` as ``oneOf`` a **string**
(*"a text input to the model, equivalent to a text input with the ``user``
role"*) or an array of input items.  Kitty accepted only the array: the string
was iterated character by character and the endpoint answered 500.

Two upstream bodies are built from one inbound Responses request (§3.2.3), and
each gets its own equality here, because the defect landed differently on each:

* the **default transport** built a Chat Completions body from the translator,
  and never got that far — it raised;
* the **Responses-origin custom transport** (``openai_subscription``) built its
  body from the *raw* inbound body and did **not** raise.  It shipped
  ``input: ["h", "i"]``.  A well-formed, non-empty, silently wrong request.

That second case is why the assertions below are equalities between the two
forms rather than status checks.  A test that asserted "200, and ``input`` is a
list" passes on the defect.
"""

from __future__ import annotations

import copy
import json

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.responses.translator import (
    InvalidResponsesRequest,
    ResponsesTranslator,
    normalize_responses_request,
)
from kitty.bridge.server import BridgeServer
from kitty.providers.base import ProviderAdapter

# ── Fixtures of shape, not of value ──────────────────────────────────────────

_TEXT = "hi"
_MODEL = "test-model"


def _array_input(text: str = _TEXT) -> list[dict]:
    """Return the array spelling of a single user text input.

    Args:
        text: The user's text.

    Returns:
        The single-item ``input`` list the string form is equivalent to.
    """
    return [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}]


def _string_body(**extra: object) -> dict:
    """Return a Responses request whose ``input`` is a bare string.

    Args:
        **extra: Additional top-level fields, e.g. ``stream``.

    Returns:
        The request body.
    """
    return {"model": _MODEL, "input": _TEXT, **extra}


def _array_body(**extra: object) -> dict:
    """Return the equivalent Responses request whose ``input`` is an array.

    Args:
        **extra: Additional top-level fields, e.g. ``stream``.

    Returns:
        The request body.
    """
    return {"model": _MODEL, "input": _array_input(), **extra}


class _StubProvider(ProviderAdapter):
    """A default-transport provider that names an upstream we can intercept."""

    @property
    def provider_type(self) -> str:
        """Return the adapter's registry key.

        Returns:
            A name no real provider uses.
        """
        return "stub"

    @property
    def default_base_url(self) -> str:
        """Return the upstream base URL.

        Returns:
            A URL ``aioresponses`` intercepts; nothing leaves the process.
        """
        return "https://upstream.invalid/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        """Build a Chat Completions request.

        Args:
            model: The model name.
            messages: The Chat Completions messages.
            **kwargs: Remaining request parameters.

        Returns:
            The request body.
        """
        return {"model": model, "messages": messages, "stream": kwargs.get("stream", False)}

    def parse_response(self, response_data: dict) -> dict:
        """Return the upstream response unchanged.

        Args:
            response_data: The upstream body.

        Returns:
            The same body.
        """
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        """Map an upstream error onto an exception.

        Args:
            status_code: The upstream HTTP status.
            body: The upstream error body.

        Returns:
            The exception to raise.
        """
        return Exception(f"upstream {status_code}: {body}")


class _RecordingCustomTransport(_StubProvider):
    """A custom-transport provider that records the body it is handed.

    §3.2.3: on this path the shipped bytes are built *inside* the transport from
    ``cc_request["_original_body"]``, so that is the only honest capture point.
    """

    def __init__(self) -> None:
        """Initialise the capture slot."""
        self.original_bodies: list[dict] = []

    @property
    def use_custom_transport(self) -> bool:
        """Declare that this adapter owns its own HTTP.

        Returns:
            Always ``True``.
        """
        return True

    async def make_request(self, cc_request: dict) -> dict:
        """Record the Responses body and answer with a minimal success.

        Args:
            cc_request: The bridge's request, carrying ``_original_body``.

        Returns:
            A minimal Chat Completions response.
        """
        self.original_bodies.append(cc_request["_original_body"])
        return {
            "id": "chatcmpl-1",
            "model": _MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }

    async def stream_request(self, cc_request: dict, write) -> None:  # noqa: ANN001
        """Record the Responses body and write one complete SSE reply.

        Args:
            cc_request: The bridge's request, carrying ``_original_body``.
            write: The bridge's byte sink for the downstream stream.
        """
        self.original_bodies.append(cc_request["_original_body"])
        await write(b'data: {"type": "response.completed"}\n\n')


def _bridge(provider: ProviderAdapter | None = None, *, model: str | None = _MODEL) -> BridgeServer:
    """Build a bridge-mode server, which is the only mode registering all routes.

    Args:
        provider: The active provider; a default-transport stub when omitted.
        model: The profile's model, or ``None`` for a profile that sets none —
            the branch of ``_normalize_model`` that leaves the client's value.

    Returns:
        An unstarted :class:`~kitty.bridge.server.BridgeServer`.
    """
    return BridgeServer(None, provider or _StubProvider(), "test-key", model=model)  # type: ignore[arg-type]


async def _post(port: int, body: object) -> tuple[int, object]:
    """POST a body to ``/v1/responses`` and read the reply.

    Args:
        port: The bridge's ephemeral port.
        body: The JSON body to send.

    Returns:
        The status code and the decoded reply, JSON where the reply is JSON and
        the raw text otherwise — a streamed reply is SSE, not JSON.
    """
    async with aiohttp.ClientSession() as session, session.post(
        f"http://127.0.0.1:{port}/v1/responses",
        data=json.dumps(body),
        headers={"content-type": "application/json"},
    ) as resp:
        raw = await resp.text()
        try:
            return resp.status, json.loads(raw)
        except json.JSONDecodeError:
            return resp.status, raw


_CC_REPLY = {
    "id": "chatcmpl-1",
    "model": _MODEL,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
}

_CC_CHUNK = {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}
_CC_STREAM = b"data: " + json.dumps(_CC_CHUNK).encode() + b"\n\ndata: [DONE]\n\n"


# ── R1, R5, R6 — the normaliser itself ───────────────────────────────────────


class TestTheNormaliser:
    """The pure function, asserted without a server or an event loop."""

    def test_a_string_input_becomes_one_user_text_message(self) -> None:
        """The spec's equivalence, spelled out rather than implied by an equality."""
        assert normalize_responses_request(_string_body())["input"] == _array_input()

    def test_an_array_input_is_returned_unchanged(self) -> None:
        """The normaliser must not start rewriting well-formed input."""
        assert normalize_responses_request(_array_body()) == _array_body()

    def test_a_body_with_no_input_is_returned_unchanged(self) -> None:
        """``input`` is optional; its absence is not a shape error."""
        assert normalize_responses_request({"model": _MODEL}) == {"model": _MODEL}

    @pytest.mark.parametrize("body", [_string_body(), _array_body()], ids=["string", "array"])
    def test_the_caller_s_body_is_not_mutated(self, body: dict) -> None:
        """The handler logs the body it received; normalising in place would relabel that log.

        Both branches, so the claim stays true of the normaliser rather than of
        the one branch that rewrites anything today.
        """
        before = copy.deepcopy(body)
        normalize_responses_request(body)
        assert body == before

    def test_an_empty_string_is_a_string_like_any_other(self) -> None:
        """``""`` is in the ``oneOf`` string branch, so it gets no special case.

        This changes behaviour that was previously accidental: on the
        subscription path ``_prepare_responses_body``'s ``if
        original_body.get("input")`` is falsy for ``""``, so the field was
        dropped from the upstream body entirely.  A turn whose text is empty and
        no turn at all are different requests; the client asked for the first.
        """
        assert normalize_responses_request({"model": _MODEL, "input": ""})["input"] == _array_input("")

    def test_an_empty_array_is_left_alone(self) -> None:
        """The array branch's degenerate case is already in the target form."""
        assert normalize_responses_request({"model": _MODEL, "input": []})["input"] == []

    @pytest.mark.parametrize(
        "body",
        [_string_body(), _array_body(), {"model": _MODEL}, {"model": _MODEL, "input": ""}, {"input": []}],
    )
    def test_normalisation_is_idempotent(self, body: dict) -> None:
        """Called at the handler and again in the translator, so this is load-bearing."""
        once = normalize_responses_request(body)
        assert normalize_responses_request(once) == once

    @pytest.mark.parametrize("bad", [123, None, {"role": "user"}, True])
    def test_an_input_that_is_neither_string_nor_array_is_refused(self, bad: object) -> None:
        """``InputParam`` is ``oneOf`` string or array, and is not nullable."""
        with pytest.raises(InvalidResponsesRequest, match="input"):
            normalize_responses_request({"model": _MODEL, "input": bad})

    @pytest.mark.parametrize("bad", [[1], ["text"], [None], [_array_input()[0], 2]])
    def test_an_array_item_that_is_not_an_object_is_refused(self, bad: list) -> None:
        """Every member of the array form is an ``InputItem`` object."""
        with pytest.raises(InvalidResponsesRequest, match=r"input\[\d\]"):
            normalize_responses_request({"model": _MODEL, "input": bad})

    @pytest.mark.parametrize("bad", [["a", "list"], "a string", 5, None])
    def test_a_body_that_is_not_a_json_object_is_refused(self, bad: object) -> None:
        """Valid JSON is not the same claim as a valid request."""
        with pytest.raises(InvalidResponsesRequest, match="object"):
            normalize_responses_request(bad)

    def test_the_message_names_the_field_and_carries_no_exception_text(self) -> None:
        """A 400 exists to tell the client what to change."""
        with pytest.raises(InvalidResponsesRequest) as excinfo:
            normalize_responses_request({"model": _MODEL, "input": 123})
        message = str(excinfo.value)
        assert "input" in message
        assert "int" in message or "number" in message
        assert "Traceback" not in message


# ── R2, R4, R6 — the translator ──────────────────────────────────────────────


class TestTheTwoFormsAreOneRequest:
    """The equivalence, proven at the lowest layer that can prove it."""

    def test_both_forms_translate_to_the_same_chat_completions_body(self) -> None:
        """One equality, so the two forms cannot be edited apart."""
        assert ResponsesTranslator().translate_request(_string_body()) == ResponsesTranslator().translate_request(
            _array_body()
        )

    def test_the_string_form_reaches_the_upstream_as_the_user_s_text(self) -> None:
        """Pins the value, so the equality above cannot be satisfied by two wrong bodies."""
        assert ResponsesTranslator().translate_request(_string_body())["messages"] == [
            {"role": "user", "content": _TEXT}
        ]

    def test_a_body_without_a_model_is_translated_rather_than_refused(self) -> None:
        """``CreateResponse`` declares no required fields, and register row M1 overrides ``model`` anyway."""
        assert ResponsesTranslator().translate_request({"input": _TEXT})["messages"] == [
            {"role": "user", "content": _TEXT}
        ]


# ── R2 — the default transport, end to end ───────────────────────────────────


class TestTheUpstreamBodyOnTheDefaultTransport:
    """What the twenty default-transport adapters actually send."""

    async def _upstream_body_for(self, body: dict, *, stream: bool) -> dict:
        """Drive one request through a real bridge and return the captured upstream body.

        Args:
            body: The inbound Responses request.
            stream: Whether to exercise the streaming handler.

        Returns:
            The Chat Completions body the bridge POSTed upstream.
        """
        captured: list[dict] = []
        server = _bridge()
        port = await server.start_async()
        try:
            # `passthrough` keeps the client's own POST to the bridge off the mock.
            with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:

                def _record(url: object, **kwargs: object) -> None:
                    captured.append(kwargs["json"])  # type: ignore[arg-type]

                if stream:
                    mocked.post(
                        "https://upstream.invalid/v1/chat/completions",
                        status=200,
                        body=_CC_STREAM,
                        headers={"Content-Type": "text/event-stream"},
                        callback=_record,
                    )
                else:
                    mocked.post(
                        "https://upstream.invalid/v1/chat/completions",
                        status=200,
                        payload=_CC_REPLY,
                        callback=_record,
                    )
                status, payload = await _post(port, body)
            assert status == 200, f"expected the request to be served, got {status}"
            # R1 claims a *stream*, not merely a 200: `_post` hands back raw text
            # when the reply is not JSON, which is what an SSE reply looks like.
            if stream:
                assert "data:" in payload, f"expected an SSE reply, got {payload!r}"
            assert len(captured) == 1, f"expected exactly one upstream attempt, got {len(captured)}"
            return captured[0]
        finally:
            await server.stop_async()

    @pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
    async def test_both_forms_send_the_same_upstream_body(self, stream: bool) -> None:
        """R2 — the string form must be indistinguishable upstream."""
        from_string = await self._upstream_body_for(_string_body(stream=stream), stream=stream)
        from_array = await self._upstream_body_for(_array_body(stream=stream), stream=stream)
        assert from_string == from_array


# ── R3 — the Responses-origin custom transport ───────────────────────────────


class TestTheUpstreamBodyOnTheCustomTransport:
    """The path that did not raise, and shipped the defect (§3.2.3, row ``curl_cffi``)."""

    async def _original_body_for(self, body: dict, *, stream: bool) -> dict:
        """Drive one request and return the Responses body handed to the transport.

        Args:
            body: The inbound Responses request.
            stream: Whether to exercise the streaming handler.

        Returns:
            ``cc_request["_original_body"]`` as the transport received it.
        """
        provider = _RecordingCustomTransport()
        server = _bridge(provider)
        port = await server.start_async()
        try:
            status, _ = await _post(port, body)
            assert status == 200, f"expected the request to be served, got {status}"
            assert len(provider.original_bodies) == 1
            return provider.original_bodies[0]
        finally:
            await server.stop_async()

    @pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
    async def test_both_forms_hand_the_transport_the_same_body(self, stream: bool) -> None:
        """R3 — the equality that closes the silent-corruption defect."""
        from_string = await self._original_body_for(_string_body(stream=stream), stream=stream)
        from_array = await self._original_body_for(_array_body(stream=stream), stream=stream)
        assert from_string == from_array

    @pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
    async def test_the_user_s_text_is_not_shredded_into_characters(self, stream: bool) -> None:
        """The falsification control.

        The defect produced ``input: ["h", "i"]`` — a list, non-empty, and a
        200.  An assertion that checked only the status or only the type passes
        on it, so the shredded body is named explicitly.
        """
        captured = await self._original_body_for(_string_body(stream=stream), stream=stream)
        assert captured["input"] != list(_TEXT)
        assert captured["input"] == _array_input()

    async def test_the_body_that_ships_is_the_same_for_both_forms(self) -> None:
        """Composed through the real builder, so the claim is about the shipped bytes.

        §3.2.3 puts this path's serialization boundary inside the transport:
        ``_prepare_responses_body`` is what turns ``_original_body`` into the
        request ``openai_subscription`` sends.
        """
        from kitty.providers.openai_subscription import OpenAISubscriptionAdapter

        from_string = await self._original_body_for(_string_body(), stream=False)
        from_array = await self._original_body_for(_array_body(), stream=False)
        assert OpenAISubscriptionAdapter._prepare_responses_body(
            from_string
        ) == OpenAISubscriptionAdapter._prepare_responses_body(from_array)


# ── R5 — the endpoint answers 400, never 500 ─────────────────────────────────


class TestTheInputFamilyNeverReturnsAServerError:
    """§6.2.1's ``not_a_server_error``, for the shapes this change claims.

    Scoped deliberately: ``tools``, a ``reasoning`` item's ``summary`` and a
    ``function_call_output`` without ``call_id`` still reach the catch-all.
    Those are filed separately and belong to §6.2.1's ``schemathesis`` job,
    which fuzzes the published schema instead of guessing which shapes a human
    will try.
    """

    @pytest.mark.parametrize(
        "body",
        [
            {"model": _MODEL, "input": 123},
            {"model": _MODEL, "input": [1, 2]},
            {"model": _MODEL, "input": None},
            ["not", "an", "object"],
            "not an object either",
        ],
    )
    async def test_a_malformed_body_is_a_client_error(self, body: object) -> None:
        """A 400 in the endpoint's existing envelope, not a 500."""
        server = _bridge()
        port = await server.start_async()
        try:
            status, payload = await _post(port, body)
        finally:
            await server.stop_async()
        assert status == 400, f"expected 400 for {body!r}, got {status}: {payload!r}"
        error = payload["error"]  # type: ignore[index]
        assert error["code"] == "invalid_request"
        # Without the marker this is byte-identical to the "Invalid JSON body" 400
        # one branch above, so the assertion would pass on a normaliser that never
        # ran. `_compaction_failed_response` documents the same reasoning.
        assert error["reason"] == "invalid_input", f"answered by a different 400: {error!r}"
        assert "input" in error["message"] or "object" in error["message"]
        assert "internal error" not in error["message"].lower()

    @pytest.mark.parametrize("profile_model", [_MODEL, None], ids=["profile_sets_a_model", "profile_sets_none"])
    async def test_a_body_without_a_model_is_served(self, profile_model: str | None) -> None:
        """R4 — ``model`` is optional in ``CreateResponse``; refusing it would be kitty's error.

        Both branches of ``_normalize_model`` are exercised.  What the second one
        sends is stated rather than asserted away: with no profile model and none
        from the client there is nothing to fill in, so ``model: ""`` goes
        upstream and the provider answers in its own dialect.  That is a 400 from
        the provider, not a 500 from kitty, which is the claim §6.2.1 makes.
        """
        captured: list[dict] = []
        server = _bridge(model=profile_model)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
                mocked.post(
                    "https://upstream.invalid/v1/chat/completions",
                    status=200,
                    payload=_CC_REPLY,
                    callback=lambda url, **kwargs: captured.append(kwargs["json"]),
                )
                status, _ = await _post(port, {"input": _TEXT})
        finally:
            await server.stop_async()
        assert status == 200
        assert len(captured) == 1, f"expected exactly one upstream attempt, got {len(captured)}"
        assert captured[0]["model"] == (profile_model or "")


# ── R6 — the obligation on the future wire reader ────────────────────────────


class TestNoSerializationBoundarySeesAStringInput:
    """Normalisation happens before the body forks, so no §3.2.3 boundary sees a string.

    Register row **M15** (``.system_design/TEST_SUITE.md`` §3.2.1) records this
    rewrite and takes §3.3.1a's ``not projectable`` escape, on the argument that
    the two spellings are one request.  That argument holds only while the
    rewrite lands *before* the body forks — otherwise one §3.2.3 boundary would
    ship a string and the other an array, and M15 would be describing two
    different upstream bodies.

    This is the narrow, decidable half of that claim, and all it is.  The other
    half — that the future OpenAI-Responses reader (plan task **T-A3**) projects
    both forms into the identical ``Request`` — cannot be tested before the
    reader exists, and is **not** tested here: it is carried by M15's
    ``not_projectable_reason``, which the L2 register guards keep alive, and by
    the T-A3 row of the implementation plan.
    """

    async def test_the_custom_transport_boundary_never_sees_a_string(self) -> None:
        """The ``curl_cffi`` row of §3.2.3."""
        provider = _RecordingCustomTransport()
        server = _bridge(provider)
        port = await server.start_async()
        try:
            await _post(port, _string_body())
        finally:
            await server.stop_async()
        assert isinstance(provider.original_bodies[0]["input"], list)

    def test_the_default_transport_boundary_never_sees_a_string(self) -> None:
        """The ``Bridge aiohttp`` row: a CC body has ``messages``, and no ``input`` at all."""
        translated = ResponsesTranslator().translate_request(_string_body())
        assert "input" not in translated
        assert isinstance(translated["messages"], list)
