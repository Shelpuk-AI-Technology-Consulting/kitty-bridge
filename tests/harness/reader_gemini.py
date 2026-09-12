"""The Gemini wire projection — one of the six independent readers.

`.system_design/TEST_SUITE.md` §3.3.1, §3.3.1a, §3.3.1b, §3.3.5, §7.4.1 · plan
task **T-A4** (KBR-36).

Projects a captured Gemini ``generateContent`` / ``streamGenerateContent``
request into :class:`harness.contract.Request`, so the fidelity oracle can
compare what an agent sent against what kitty actually put on the wire.  Gemini
is an **inbound-only** format in kitty: ``BridgeServer`` serves
``/v1beta/models/{model}:generateContent`` for the Gemini CLI and translates to
Chat Completions upstream, so this reader reads the *agent* side of the
comparison.

**Written against the published schema, never against kitty's output.**  Every
key table below was read from the Gemini API discovery document,
``https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta``,
``revision`` :data:`SCHEMA_VERSION` — not from a kitty request and not from
memory.  §3.3.1's independent-oracle rule: a reader validated against kitty's
output inherits kitty's bugs, and the whole of I1 then proves only that kitty
agrees with itself.

**It imports nothing from ``src/kitty``, and must not.**
``test_reader_gemini.py`` asserts the absence structurally.

**Three things this format forces that no earlier reader met.**  Each is
recorded in §7.4.1 so T-A5 and T-A6 inherit the answer rather than inventing one:

1. **The route is an input** (§3.3.5).  :attr:`~harness.contract.Envelope.model`
   comes from the path segment and :attr:`~harness.contract.Envelope.stream`
   from the operation suffix.  Nothing in the body shows either.
2. **Control fields nest.**  Sampling lives under ``generationConfig`` and the
   tool choice under ``toolConfig.functionCallingConfig``, while
   :func:`harness.contract.extra_path` *raises* on a dotted key.  So a nested
   control field maps to ``envelope.extra[<leaf published key>]``.
3. **Every field has two legal spellings.**  Gemini's JSON is ProtoJSON, whose
   parsers "accept both the lowerCamelCase name … and the original proto field
   name" (``protobuf.dev/programming-guides/json/``), and Google's own published
   examples mix the two freely — ``system_instruction`` and
   ``function_declarations`` beside ``maxOutputTokens`` and ``stopSequences``.
   A reader that knew one spelling would residualise the other and fail the run
   on Google's own published example.

**What totality means here, and where it stops.**  Every one of the nine
published top-level body keys is classified into the envelope, the conversation, or
the residual, and :func:`harness.contract.verify_total` fails the run on
anything left over.  Two boundaries are deliberate and named rather than
implied: ``consumed`` covers *top-level* keys only (§3.3.1's stated boundary),
and the **query string is not read at all** — ``?alt=sse`` and ``?key=`` are
route surface, and ``verify_total`` compares ``consumed | residual`` against the
*body*, so a query key in either account would be reported as a claim on a key
the body does not have.  Asserting the route is T-D2's (KBR-52).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from harness import contract as c

#: The published schema version every table in this module was derived from —
#: the discovery document's ``revision``, which is how Google versions it.
#: Spelled ``SCHEMA_VERSION`` to match ``reader_responses.py``, so T-D8 does not
#: have to learn six names for one idea.  Recorded so a future divergence is
#: traceable rather than mysterious: when Google adds a top-level key or a
#: ``Part`` member, the schema-agreement tests go red and this string says which
#: revision the reader last agreed with.
SCHEMA_VERSION = "20260910"

# --------------------------------------------------------------------------
# The route — §3.3.5
# --------------------------------------------------------------------------

#: The published REST path shape, ``v1beta/{+model}:generateContent``.  ``model``
#: is greedy up to the operation's colon because the segment may itself contain
#: a slash, which is how ``models/gemini-2.5-pro`` reaches the route.
_ROUTE = re.compile(r"^/(?P<version>v[^/]+)/models/(?P<model>[^:]+):(?P<operation>[A-Za-z]+)$")

#: The two published generate operations, and the ``stream`` each implies.
#: ``countTokens`` and ``embedContent`` are published methods on the same
#: resource, so accepting "any operation" would project a token count as a
#: conversation.
_OPERATIONS: Mapping[str, bool] = {"generateContent": False, "streamGenerateContent": True}

#: The prefix the published path form puts before a model name.  A profile that
#: names ``models/gemini-2.5-pro`` and one that names ``gemini-2.5-pro`` reach
#: the same destination, so they must project the same model.
_MODEL_PREFIX = "models/"

# --------------------------------------------------------------------------
# Top-level key classification — `GenerateContentRequest`, 9 REST body keys
# --------------------------------------------------------------------------

#: Keys carrying semantic content rather than control.
_CONVERSATION_KEYS = frozenset({"contents", "systemInstruction", "tools", "toolConfig"})

#: Top-level control fields that map to ``envelope.extra`` under their own
#: published key.
_TOP_LEVEL_EXTRA_KEYS = frozenset({"serviceTier", "safetySettings", "cachedContent"})

#: All nine, for the totality check and for the test that asserts this module's
#: tables still agree with the published schema.
#:
#: **Nine, not the discovery document's ten.**  That document carries the proto
#: message, whose ``model`` field the REST surface exposes as the *path
#: parameter* — the endpoint is ``/v1beta/{model=models/*}:generateContent`` and
#: the reference's own "Request body" table lists exactly these nine, omitting
#: ``model``.  So a body-level ``model`` is an unrecognised key and residualises
#: like any other.  An earlier draft made it a second carrier to reconcile
#: against the path, which put an *assertion* inside a reader: a comparison whose
#: only outcome is a residual can never be claimed by a register row (§3.3.1a
#: makes ``residual`` an illegal anchor), so it could only ever kill the run —
#: and finding disagreements is §3.3.2's job, over a route T-D2 already owns.
PUBLISHED_TOP_LEVEL_KEYS = _CONVERSATION_KEYS | _TOP_LEVEL_EXTRA_KEYS | frozenset({"generationConfig", "store"})

# --------------------------------------------------------------------------
# `GenerationConfig`, 25 keys
# --------------------------------------------------------------------------

#: A JSON number, which may arrive as an ``int``.  ``bool`` is excluded by
#: :func:`_typed_leaf` rather than here, because it is a subclass of ``int``.
_NUMBER = (int, float)

#: The eleven ``generationConfig`` members with a canonical Chat Completions
#: spelling (§3.3.1b), each with the type the schema publishes for it.
#:
#: ⚠️ The last two are a **name collision**, and mapping either onto its own
#: spelling would be wrong.  Gemini's ``logprobs`` is an *integer* — "the number
#: of top logprobs … to return at each decoding step" — which Chat Completions
#: calls ``top_logprobs``; Gemini's ``responseLogprobs`` is the *boolean* Chat
#: Completions calls ``logprobs``.  A reader that carried ``logprobs`` across
#: unchanged would show a cross-format delta on every request that asks for them.
_SAMPLING_KEYS: Mapping[str, tuple[str, tuple[type, ...]]] = {
    "temperature": ("temperature", _NUMBER),
    "topP": ("top_p", _NUMBER),
    "topK": ("top_k", (int,)),
    "maxOutputTokens": ("max_tokens", (int,)),
    "presencePenalty": ("presence_penalty", _NUMBER),
    "frequencyPenalty": ("frequency_penalty", _NUMBER),
    "seed": ("seed", (int,)),
    "candidateCount": ("n", (int,)),
    "stopSequences": ("stop", (list,)),
    "responseLogprobs": ("logprobs", (bool,)),
    "logprobs": ("top_logprobs", (int,)),
}

#: Every other published ``generationConfig`` member.  These are recognised
#: declared control fields of the format, so §3.3.1b maps them to
#: ``envelope.extra`` rather than the residual — that section names
#: ``generationConfig.responseSchema`` as exactly this case.
#:
#: None of ``responseMimeType``, ``responseSchema``, ``responseJsonSchema``,
#: ``_responseJsonSchema`` or ``responseFormat`` maps onto the canonical
#: ``response_format``: ``responseFormat`` is per-modality output configuration
#: and the others are schema constraints, so folding any of them onto the Chat
#: Completions union would put a claim into the projection that the wire does
#: not make.
_GENERATION_EXTRA_KEYS = frozenset(
    {
        "_responseJsonSchema",
        "audioTranscriptionConfig",
        "enableAffectiveDialog",
        "enableEnhancedCivicAnswers",
        "imageConfig",
        "mediaResolution",
        "responseFormat",
        "responseJsonSchema",
        "responseMimeType",
        "responseModalities",
        "responseSchema",
        "speechConfig",
        "thinkingConfig",
        "translationConfig",
    }
)

#: All 25.
PUBLISHED_GENERATION_CONFIG_KEYS = frozenset(_SAMPLING_KEYS) | _GENERATION_EXTRA_KEYS

# --------------------------------------------------------------------------
# `ToolConfig` and `Tool`
# --------------------------------------------------------------------------

#: ``ToolConfig`` members other than ``functionCallingConfig``, which becomes
#: ``tool_choice``.  Flattened to their leaf key for the same reason as
#: ``generationConfig``'s.
_TOOL_CONFIG_EXTRA_KEYS = frozenset({"retrievalConfig", "includeServerSideToolInvocations"})

#: All three.
PUBLISHED_TOOL_CONFIG_KEYS = _TOOL_CONFIG_EXTRA_KEYS | frozenset({"functionCallingConfig"})

#: ``FunctionCallingConfig``'s two published members.
PUBLISHED_FUNCTION_CALLING_CONFIG_KEYS = frozenset({"mode", "allowedFunctionNames"})

#: The three published modes with a canonical value.  ``MODE_UNSPECIFIED`` and
#: ``VALIDATED`` have none and residualise instead — :class:`Envelope` enforces
#: :data:`~harness.contract.TOOL_CHOICE_VALUES`, so inventing one would raise.
#:
#: Matched **case-insensitively**: the published enumeration spells them in
#: upper case, and Google's own published ``function_calling.sh`` example sends
#: ``"mode": "auto"``.
_TOOL_CHOICE_MODES: Mapping[str, str] = {"AUTO": "auto", "ANY": "any", "NONE": "none"}

#: ``Tool``'s eight built-in capability toggles.  These configure server-side
#: capabilities rather than declaring a function the agent wrote, so they are
#: control and map to ``envelope.extra`` — :class:`~harness.contract.ToolDecl`
#: requires a name, and inventing ``ToolDecl("googleSearch")`` would put a
#: vendor spelling into a form whose purpose is wire independence.
_BUILT_IN_TOOL_KEYS = frozenset(
    {
        "codeExecution",
        "computerUse",
        "fileSearch",
        "googleMaps",
        "googleSearch",
        "googleSearchRetrieval",
        "mcpServers",
        "urlContext",
    }
)

#: All nine.
PUBLISHED_TOOL_KEYS = _BUILT_IN_TOOL_KEYS | frozenset({"functionDeclarations"})

#: ``FunctionDeclaration``'s seven published members.  ``behavior``, ``response``
#: and ``responseJsonSchema`` have no slot in :class:`~harness.contract.ToolDecl`
#: and residualise; ``parametersJsonSchema`` is the published mutually-exclusive
#: alternative to ``parameters`` and fills the same slot.
PUBLISHED_FUNCTION_DECLARATION_KEYS = frozenset(
    {
        "behavior",
        "description",
        "name",
        "parameters",
        "parametersJsonSchema",
        "response",
        "responseJsonSchema",
    }
)

# --------------------------------------------------------------------------
# `Content` and `Part`
# --------------------------------------------------------------------------

#: ``Content``'s two published members.
PUBLISHED_CONTENT_KEYS = frozenset({"parts", "role"})

#: The published producer values, onto §3.3.1b's closed role set.  An absent or
#: empty role is ``user``: the schema says the field "can be left blank or unset"
#: for a single-turn query, and request content with no producer is the client's.
_ROLES: Mapping[str, str] = {"user": "user", "model": "assistant"}

#: One producer value the published schema does **not** list, recognised on
#: §7.4.1's first evidence rule: the client demonstrably sends it, so rejecting
#: it would fail the run on real traffic, "which is a harness defect and not a
#: finding".  The evidence is ``src/kitty/bridge/gemini/translator.py``'s
#: ``_ROLE_MAP``, which reads ``role: "function"`` straight off the inbound body
#: — the same shape as that section's worked example, Anthropic's ``effort``.
#: It is the spelling older Google SDKs used for a function-response turn, and
#: ``user`` is where §3.3.1b already puts a tool result, so nothing is invented.
#:
#: Kept in its own table rather than folded into :data:`_ROLES`, so that a
#: reader of this module can see one deliberate exception to the published
#: schema rather than a fourth undocumented producer value.
_EVIDENCED_ROLES: Mapping[str, str] = {"function": "user"}

#: ``Part`` members the grammar models semantically.
_MODELLED_PART_KEYS = frozenset({"text", "inlineData", "fileData", "functionCall", "functionResponse"})

#: ``Part`` members with no slot in the grammar, which therefore project as
#: :class:`~harness.contract.Opaque`.  All four name a concept no other format
#: has, so KBR-35's stated exception applies and the wire name in snake_case is
#: the canonical ``kind``; no cross-vendor alias table is needed here.  **T-A5
#: does need one** — Converse writes ``searchResult`` where Anthropic writes
#: ``search_result`` — so this module is not the precedent for that case.
_OPAQUE_PART_KEYS: Mapping[str, str] = {
    "executableCode": "executable_code",
    "codeExecutionResult": "code_execution_result",
    "toolCall": "tool_call",
    "toolResponse": "tool_response",
}

#: ``Part`` members that modify another member rather than being content of
#: their own.  The grammar has no slot for any of them, so they residualise at
#: their own path — ``thoughtSignature`` only when the part is not a thought,
#: since :class:`~harness.contract.Thinking` carries it when it is.
_PART_MODIFIER_KEYS = frozenset(
    {
        "audioTranscription",
        "mediaProcessing",
        "mediaResolution",
        "partMetadata",
        "thought",
        "thoughtSignature",
        "videoMetadata",
    }
)

#: All sixteen.
PUBLISHED_PART_KEYS = _MODELLED_PART_KEYS | frozenset(_OPAQUE_PART_KEYS) | _PART_MODIFIER_KEYS

#: ``FunctionCall``'s three published members.
PUBLISHED_FUNCTION_CALL_KEYS = frozenset({"id", "name", "args"})

#: ``FunctionResponse``'s six published members.  ``scheduling`` and
#: ``willContinue`` govern NON_BLOCKING call scheduling, which the grammar does
#: not model, so they residualise.
PUBLISHED_FUNCTION_RESPONSE_KEYS = frozenset({"id", "name", "response", "parts", "scheduling", "willContinue"})

#: ``Blob``'s and ``FileData``'s published members.  ``displayName`` names the
#: blob to the model and has no slot, so it residualises.
PUBLISHED_BLOB_KEYS = frozenset({"data", "mimeType", "displayName"})
PUBLISHED_FILE_DATA_KEYS = frozenset({"fileUri", "mimeType", "displayName"})


class GeminiProjection:
    """Reads a Gemini ``generateContent`` request into :class:`~harness.contract.Request`.

    Implements :class:`~harness.contract.Projection` for
    :attr:`~harness.contract.WireFormat.GEMINI`.

    Attributes:
        wire_format: Always :attr:`~harness.contract.WireFormat.GEMINI`.
    """

    wire_format = c.WireFormat.GEMINI

    def read_request(self, captured: c.CapturedRequest) -> c.Request:
        """Project a captured Gemini request.

        Args:
            captured: The request as observed on the wire. **Both the path and
                the body are read**: the model and the operation live in the
                URL (§3.3.5), so a body-only reader could not project a Gemini
                request at all. The query is deliberately not read — see the
                module docstring.

        Returns:
            The wire-independent projection, total over the body.

        Raises:
            UnreadableBodyError: When the path is not a published generate
                route, or the body is not a readable Gemini request. The
                catch-all below is the actual guarantee: the contract names
                three failure shapes, and an escaping ``KeyError`` would be an
                undefined fourth on the one path T-D1 uses to tell an unreadable
                body from an I1 breach. ``IndexError`` is deliberately **not** in
                the tuple: no sequence in this module is indexed positionally
                except under a guard, so catching it would be handling an
                impossible case.
        """
        model, stream = _read_route(captured.path)
        body = _parse_body(captured.body)

        # `ValueError` is deliberately NOT caught: from inside a reader it means
        # the reader mis-routed a field, which is a reader bug and must surface.
        try:
            return _project(body, model, stream)
        except (KeyError, TypeError, AttributeError) as exc:
            raise c.UnreadableBodyError(f"unreadable Gemini body: {exc!r}") from exc


def _read_route(path: str) -> tuple[str, bool]:
    """Read the model and the streaming flag out of the request path.

    §3.3.5: "Gemini carries the model and the operation in the path … which is
    why M10 lifts the model into the body in the first place."

    ``stream`` comes from the **operation**, never from ``?alt=sse``: ``alt``
    selects SSE framing over JSON-array framing for a method that streams either
    way, so reading it instead would project a streaming request as
    non-streaming and let P17's register row claim a delta nobody caused.

    Args:
        path: :attr:`~harness.contract.CapturedRequest.path`, including the
            ``:generateContent`` operation.

    Returns:
        The model name, with any ``models/`` prefix stripped, and whether the
        operation streams.

    Raises:
        UnreadableBodyError: When the path is not one of the two published
            generate routes. Structural, not residualisable: there is no partial
            projection to salvage when the model cannot be read at all.
    """
    matched = _ROUTE.match(path)
    if matched is None:
        raise c.UnreadableBodyError(
            f"path {path!r} is not a published Gemini generate route (/{{version}}/models/{{model}}:generateContent)"
        )

    operation = matched.group("operation")
    if operation not in _OPERATIONS:
        raise c.UnreadableBodyError(
            f"operation {operation!r} is not one of {sorted(_OPERATIONS)}; "
            "countTokens and embedContent are published on the same resource and are not requests"
        )

    model = matched.group("model")
    return model.removeprefix(_MODEL_PREFIX), _OPERATIONS[operation]


def _parse_body(raw: bytes) -> Mapping[str, Any]:
    """Decode the request body into a JSON object.

    Args:
        raw: The raw body bytes.

    Returns:
        The parsed body.

    Raises:
        UnreadableBodyError: When the bytes are not JSON, or are JSON that is
            not an object. A JSON array is valid JSON and an invalid request.
    """
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise c.UnreadableBodyError(f"body is not valid JSON: {exc}") from exc

    if not isinstance(body, dict):
        raise c.UnreadableBodyError(f"body must be a JSON object, got {type(body).__name__}")

    return body


# --------------------------------------------------------------------------
# ProtoJSON's two spellings
# --------------------------------------------------------------------------


def _camel(key: str) -> str:
    """Convert a proto ``snake_case`` field name to its lowerCamelCase JSON name.

    Args:
        key: A wire key.

    Returns:
        The lowerCamelCase form. A key with no underscore is returned unchanged.
    """
    head, *rest = key.split("_")
    return head + "".join(word[:1].upper() + word[1:] for word in rest)


def _resolve(key: str, published: frozenset[str]) -> str:
    """Return the published name a wire key stands for, or the key itself.

    ProtoJSON parsers "accept both the lowerCamelCase name … and the original
    proto field name", and Google's published examples use both — so a reader
    that knew only the schema's spelling would residualise ``system_instruction``
    and fail the run on Google's own ``system_instruction.sh``.

    The published name is tried **first**, which is what keeps
    ``_responseJsonSchema`` resolving to itself rather than to the
    ``ResponseJsonSchema`` that naive conversion of its leading underscore
    would produce.

    Args:
        key: The wire key as the client spelled it.
        published: The published names of the object being read.

    Returns:
        The published name when one matches, otherwise ``key`` unchanged, which
        leaves it unrecognised and so bound for the residual.
    """
    if key in published:
        return key

    camel = _camel(key)
    return camel if camel in published else key


def _aliased(
    source: Mapping[str, Any], published: frozenset[str], prefix: str, residual: dict[str, Any]
) -> dict[str, tuple[str, Any]]:
    """Return a published-name view of an object, keeping each key's wire spelling.

    The wire spelling has to survive because a residual key is "the body's own
    path" (§7.4.1) and ``consumed`` is compared against the body's own keys by
    :func:`~harness.contract.verify_total`; the published name is what the
    reader dispatches on.

    Args:
        source: The object being read.
        published: The published names of that object.
        prefix: The object's path from the body root, ``""`` at the top level.
        residual: The residual mapping, extended in place with any key that
            collides with one already seen.

    Returns:
        A mapping of published name to ``(wire key, value)``, in wire order.

    Note:
        When two wire keys resolve to the same published name, the **published**
        spelling is read and the other residualises at its own wire path — a
        body cannot mean two things, and resolving by position would make the
        projection depend on the order a serialiser emitted them in.
    """
    view: dict[str, tuple[str, Any]] = {}

    for wire_key, value in source.items():
        name = _resolve(wire_key, published)
        if name not in view:
            view[name] = (wire_key, value)
            continue

        # The **published** spelling wins wherever it appears, so the projection
        # does not depend on the order a serialiser emitted two aliases in.
        # §7.4.1 designs key order out of the projection elsewhere for the same
        # reason — "canonical JSON rather than the raw wire slice, because a
        # translator that reorders keys must not change the digest".
        held_key, held_value = view[name]
        if wire_key == name and held_key != name:
            view[name] = (wire_key, value)
            residual[_join(prefix, held_key)] = held_value
        else:
            residual[_join(prefix, wire_key)] = value

    return view


def _join(prefix: str, key: str) -> str:
    """Return a residual key for ``key`` inside the object at ``prefix``.

    Args:
        prefix: The object's path from the body root, ``""`` at the top level.
        key: The wire key.

    Returns:
        The dotted path, or the bare key at the top level — where the bare form
        is required, because :func:`~harness.contract.verify_total` compares the
        residual's keys against the body's own.
    """
    return f"{prefix}.{key}" if prefix else key


def _residualise(
    view: Mapping[str, tuple[str, Any]],
    mapped: frozenset[str] | set[str],
    prefix: str,
    residual: dict[str, Any],
) -> None:
    """Record every key of an aliased view the reader did not map.

    §3.3.1's "unknown fields fail closed", applied at depth:
    :func:`~harness.contract.verify_total` sees top-level keys only, so this is
    what closes the gap beneath them.

    Args:
        view: The aliased view, from :func:`_aliased`.
        mapped: The published names the caller accounted for.
        prefix: The object's path from the body root.
        residual: The residual mapping, extended in place.
    """
    for name, (wire_key, value) in view.items():
        if name not in mapped:
            residual[_join(prefix, wire_key)] = value


def _typed_leaf(
    view: Mapping[str, tuple[str, Any]],
    name: str,
    expected: tuple[type, ...],
    prefix: str,
    residual: dict[str, Any],
    default: Any = None,
) -> Any:
    """Return an optional leaf, residualising it when the wire carried the wrong type.

    §7.4.1's wrongly-typed-leaf rule, applied wherever the grammar has an absent
    value to fall back to. Without it these fields *fail open*: the contract
    validates only roles, sampling keys and ``tool_choice``, so a dict in a field
    declared ``str | None`` is carried silently and the residual stays empty.
    T-A1 measured eight of nine leaves failing open when the rule was applied
    field by field rather than through one helper, which is why every optional
    leaf in this module goes through this one.

    Args:
        view: The aliased view of the object being read.
        name: The leaf's published name.
        expected: The types the schema publishes for it.
        prefix: The object's path from the body root.
        residual: The residual mapping, extended in place.
        default: The grammar's absent value for this field.

    Returns:
        The leaf, or ``default`` when the wire value was the wrong type.
    """
    if name not in view:
        return default

    wire_key, value = view[name]
    if value is None:
        # ProtoJSON reads `null` as the field's default, so an explicit null is
        # the absent value rather than a type error.
        return default

    # `bool` is a subclass of `int`, so an unguarded `isinstance` would carry
    # `"topK": true` through as the integer 1 — a value the agent never sent.
    wrong = not isinstance(value, expected) or (isinstance(value, bool) and bool not in expected)
    if wrong:
        residual[_join(prefix, wire_key)] = value
        return default

    return value


def _members(value: Any) -> tuple[Any, ...] | None:
    """Return a repeated field's members, accepting the published singleton form.

    Google's published ``system_instruction.sh`` and ``function_calling.sh``
    examples send ``contents`` and ``parts`` as **single objects** where the
    schema declares repeated fields, and they are the examples this reader is
    validated against. Accepting only a list would fail the run on two of
    Google's own samples.

    Args:
        value: The field's wire value.

    Returns:
        The members in order, or ``None`` when the value is neither a list nor
        an object and so cannot be read as either.
    """
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, list):
        return tuple(value)
    return None


# --------------------------------------------------------------------------
# The top-level walk
# --------------------------------------------------------------------------


def _project(body: Mapping[str, Any], model: str, stream: bool) -> c.Request:
    """Classify every top-level key into the envelope, the conversation or the residual.

    Args:
        body: The parsed request body.
        model: The model, read from the path.
        stream: Whether the path's operation streams.

    Returns:
        The projection.

    Raises:
        UnreadableBodyError: Propagated from the per-field readers.
    """
    residual: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    view = _aliased(body, PUBLISHED_TOP_LEVEL_KEYS, "", residual)

    # Every other control field the format defines, keyed by its **published**
    # name. §3.3.1a says "keyed by the wire key"; ProtoJSON gives every field two
    # legal wire keys, and a register row can name only one, so the published
    # lowerCamelCase name is the address and the snake_case original resolves
    # onto it.
    for name in _TOP_LEVEL_EXTRA_KEYS:
        if name in view:
            extra[name] = view[name][1]

    sampling, generation_extra = _read_generation_config(view, residual)
    extra.update(generation_extra)
    extra.update(_read_tool_config(view, residual))

    tools, tool_extra = _read_tools(view, residual)
    extra.update(tool_extra)

    conversation = c.Conversation(
        system=_read_system_instruction(view, residual),
        turns=_read_contents(view, residual),
        tools=tools,
        sampling=sampling,
    )

    envelope = c.Envelope(
        model=model,
        stream=stream,
        store=_typed_leaf(view, "store", (bool,), "", residual),
        extra=extra,
    )

    _residualise(view, PUBLISHED_TOP_LEVEL_KEYS, "", residual)

    return c.Request(
        envelope=envelope,
        conversation=conversation,
        residual=residual,
        consumed=frozenset(wire_key for name, (wire_key, _) in view.items() if name in PUBLISHED_TOP_LEVEL_KEYS),
        source=body,
    )


# --------------------------------------------------------------------------
# Control fields
# --------------------------------------------------------------------------


def _read_generation_config(
    view: Mapping[str, tuple[str, Any]], residual: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split ``generationConfig`` into canonical sampling and format-specific control.

    §3.3.1b: sampling normalises onto a **closed** set of fifteen canonical
    keys, and "a key outside the set that the reader recognises as a declared
    control field of that format maps to ``envelope.extra[<wire key>]`` — only
    an *unrecognised* key residualises." That section names
    ``generationConfig.responseSchema`` as exactly this case.

    Args:
        view: The aliased top-level view.
        residual: The residual mapping, extended in place.

    Returns:
        The canonical sampling mapping and the extra entries, each keyed by the
        **leaf** published name — ``envelope.extra[responseSchema]``, never
        ``envelope.extra[generationConfig.responseSchema]``, because
        :func:`~harness.contract.extra_path` raises on a dotted key.
    """
    if "generationConfig" not in view:
        return {}, {}

    wire_key, value = view["generationConfig"]
    if not isinstance(value, Mapping):
        residual[wire_key] = value
        return {}, {}

    nested = _aliased(value, PUBLISHED_GENERATION_CONFIG_KEYS, wire_key, residual)

    sampling: dict[str, Any] = {}
    for name, (canonical, expected) in _SAMPLING_KEYS.items():
        read = _typed_leaf(nested, name, expected, wire_key, residual)
        if read is not None:
            sampling[canonical] = read

    extra = {name: nested[name][1] for name in _GENERATION_EXTRA_KEYS if name in nested}

    _residualise(nested, PUBLISHED_GENERATION_CONFIG_KEYS, wire_key, residual)
    return sampling, extra


def _read_tool_config(view: Mapping[str, tuple[str, Any]], residual: dict[str, Any]) -> dict[str, Any]:
    """Read ``toolConfig`` into ``tool_choice`` and the rest of the envelope's extra.

    Args:
        view: The aliased top-level view.
        residual: The residual mapping, extended in place.

    Returns:
        The extra entries, ``tool_choice`` among them when a mode maps.
    """
    if "toolConfig" not in view:
        return {}

    wire_key, value = view["toolConfig"]
    if not isinstance(value, Mapping):
        residual[wire_key] = value
        return {}

    nested = _aliased(value, PUBLISHED_TOOL_CONFIG_KEYS, wire_key, residual)
    extra = {name: nested[name][1] for name in _TOOL_CONFIG_EXTRA_KEYS if name in nested}

    choice = _read_tool_choice(nested, wire_key, residual)
    if choice is not None:
        extra[c.TOOL_CHOICE_KEY] = choice

    _residualise(nested, PUBLISHED_TOOL_CONFIG_KEYS, wire_key, residual)
    return extra


def _read_tool_choice(nested: Mapping[str, tuple[str, Any]], prefix: str, residual: dict[str, Any]) -> str | None:
    """Normalise ``functionCallingConfig`` onto the canonical tool-choice vocabulary.

    §3.3.1b: four wire keys name one concept, so ``tool_choice`` is the single
    deliberate exception to keying ``extra`` by the wire key.

    Args:
        nested: The aliased ``toolConfig`` view.
        prefix: ``toolConfig``'s path from the body root.
        residual: The residual mapping, extended in place.

    Returns:
        One of :data:`~harness.contract.TOOL_CHOICE_VALUES`, a ``tool:<name>``
        selection, or ``None`` when the config names no mode the vocabulary
        carries.
    """
    if "functionCallingConfig" not in nested:
        return None

    wire_key, value = nested["functionCallingConfig"]
    path = _join(prefix, wire_key)
    if not isinstance(value, Mapping):
        residual[path] = value
        return None

    config = _aliased(value, PUBLISHED_FUNCTION_CALLING_CONFIG_KEYS, path, residual)
    mode = _typed_leaf(config, "mode", (str,), path, residual)

    # Case-insensitively: the published enumeration is upper case, and Google's
    # own `function_calling.sh` example sends `"mode": "auto"`.
    choice = _TOOL_CHOICE_MODES.get(mode.upper()) if isinstance(mode, str) else None
    if choice is None and "mode" in config:
        # MODE_UNSPECIFIED and VALIDATED have no canonical value, and `Envelope`
        # enforces the closed set, so inventing one would raise a `ValueError`
        # that T-D1 would read as a reader bug.
        #
        # `setdefault` with the **wire** value, not `mode`: a wrongly-typed mode
        # has already been residualised correctly by `_typed_leaf`, and `mode` is
        # by then the default it fell back to. Overwriting would tell a
        # maintainer the client sent `null` when it sent an object.
        residual.setdefault(_join(path, config["mode"][0]), config["mode"][1])

    allowed = _typed_leaf(config, "allowedFunctionNames", (list,), path, residual)
    if allowed is not None:
        if choice == "any" and len(allowed) == 1 and isinstance(allowed[0], str):
            # "must call a function, and only this one" is exactly `tool:<name>`.
            choice = f"tool:{allowed[0]}"
        else:
            # A restriction to several names has no canonical form; the mode
            # still projects, so the residual names only what was lost.
            residual[_join(path, config["allowedFunctionNames"][0])] = allowed

    _residualise(config, PUBLISHED_FUNCTION_CALLING_CONFIG_KEYS, path, residual)
    return choice


def _read_tools(
    view: Mapping[str, tuple[str, Any]], residual: dict[str, Any]
) -> tuple[tuple[c.ToolDecl, ...], dict[str, Any]]:
    """Read the declared tools and the built-in capability toggles beside them.

    Args:
        view: The aliased top-level view.
        residual: The residual mapping, extended in place.

    Returns:
        The function declarations in order, and the built-in toggles as extra
        entries.

    Raises:
        UnreadableBodyError: When ``tools`` is not a list, or an entry is not an
            object.
    """
    if "tools" not in view:
        return (), {}

    wire_key, value = view["tools"]
    entries = _members(value)
    if entries is None:
        raise c.UnreadableBodyError(f"tools must be a list, got {type(value).__name__}")

    declared: list[c.ToolDecl] = []
    extra: dict[str, Any] = {}

    for index, entry in enumerate(entries):
        path = f"{wire_key}[{index}]"
        if not isinstance(entry, Mapping):
            raise c.UnreadableBodyError(f"{path} must be an object, got {type(entry).__name__}")

        tool = _aliased(entry, PUBLISHED_TOOL_KEYS, path, residual)
        for name in _BUILT_IN_TOOL_KEYS:
            if name not in tool:
                continue
            if name in extra:
                # A second entry re-declaring one has no second address; the
                # first is the one `envelope.extra[<key>]` names.
                residual[_join(path, tool[name][0])] = tool[name][1]
                continue
            extra[name] = tool[name][1]

        declared.extend(_read_function_declarations(tool, path, residual))
        _residualise(tool, PUBLISHED_TOOL_KEYS, path, residual)

    return tuple(declared), extra


def _read_function_declarations(
    tool: Mapping[str, tuple[str, Any]], prefix: str, residual: dict[str, Any]
) -> list[c.ToolDecl]:
    """Read one ``Tool`` entry's function declarations.

    Args:
        tool: The aliased ``Tool`` view.
        prefix: The entry's path from the body root.
        residual: The residual mapping, extended in place.

    Returns:
        The declarations, in order.

    Raises:
        UnreadableBodyError: When ``functionDeclarations`` is not a list or an
            entry is not an object. An entry with no usable name residualises
            instead — see :func:`_read_required_name`.
    """
    if "functionDeclarations" not in tool:
        return []

    wire_key, value = tool["functionDeclarations"]
    path = _join(prefix, wire_key)
    entries = _members(value)
    if entries is None:
        raise c.UnreadableBodyError(f"{path} must be a list, got {type(value).__name__}")

    declared: list[c.ToolDecl] = []
    for index, entry in enumerate(entries):
        item = f"{path}[{index}]"
        if not isinstance(entry, Mapping):
            raise c.UnreadableBodyError(f"{item} must be an object, got {type(entry).__name__}")

        declaration = _aliased(entry, PUBLISHED_FUNCTION_DECLARATION_KEYS, item, residual)
        name = _read_required_name(declaration, item, residual)
        schema, mapped = _read_declaration_schema(declaration, item, residual)
        declared.append(
            c.ToolDecl(
                name=name,
                description=_typed_leaf(declaration, "description", (str,), item, residual),
                schema=schema,
                # Absent, not False: Gemini defines no `strict`, and P15's
                # presence and absence must stay distinguishable.
                strict=None,
            )
        )
        _residualise(declaration, mapped, item, residual)

    return declared


def _read_required_name(view: Mapping[str, tuple[str, Any]], path: str, residual: dict[str, Any]) -> str:
    """Return a required ``name``, residualising it when the wire carried none.

    §3.3.1b settles this and the answer is **not** the wrongly-typed-leaf rule's
    usual one: "an absent ``name`` *does* residualise … ``ToolUse.name`` is a
    ``str`` with no such value: ``""`` claims a tool *named* empty-string, and a
    call nobody can name cannot be paired or addressed." So absent and ``null``
    residualise too, not only a wrong type — and the part is still **projected**,
    because dropping or raising on it would blind the oracle to the rest of a
    request it could otherwise diff. T-A3 takes the same branch.

    Args:
        view: The aliased view of the object carrying the name.
        path: That object's path from the body root.
        residual: The residual mapping, extended in place.

    Returns:
        The name, or ``""`` when the wire carried none — which the residual then
        names, so the run fails with the field identified.
    """
    # `_typed_leaf` has already residualised a wrongly-typed value; what is left
    # is absent or explicitly null, which this records in its place.
    name: str | None = _typed_leaf(view, "name", (str,), path, residual)
    if name is None:
        residual.setdefault(
            _join(path, view["name"][0] if "name" in view else "name"),
            view["name"][1] if "name" in view else None,
        )
        return ""

    return name


def _read_declaration_schema(
    declaration: Mapping[str, tuple[str, Any]], path: str, residual: dict[str, Any]
) -> tuple[Mapping[str, Any] | None, set[str]]:
    """Read a declaration's parameter schema from whichever key carries it.

    ``parameters`` (OpenAPI) and ``parametersJsonSchema`` (JSON Schema) are
    published as mutually exclusive alternatives filling one slot. When a
    declaration carries both it is outside the schema, so ``parameters`` — the
    older and far commoner spelling — wins and the other residualises.

    Args:
        declaration: The aliased ``FunctionDeclaration`` view.
        path: The declaration's path from the body root.
        residual: The residual mapping, extended in place.

    Returns:
        The schema or ``None``, and the published names this consumed, so the
        caller's :func:`_residualise` does not also claim the loser.
    """
    mapped = {"name", "description"}

    # The first of the two that is present wins, so `parameters` beats
    # `parametersJsonSchema` on a declaration carrying both — which is outside
    # the published schema. The loser is left out of `mapped`, so the caller's
    # `_residualise` puts it in the residual rather than dropping it.
    for name in ("parameters", "parametersJsonSchema"):
        if name not in declaration:
            continue
        # A wrongly-typed schema residualises rather than coercing: `dict()` on
        # a list of two-character strings *fabricates* a schema the agent never
        # sent, and on a plain string raises a bare `ValueError`.
        return _typed_leaf(declaration, name, (dict,), path, residual), mapped | {name}

    return None, mapped


# --------------------------------------------------------------------------
# The conversation
# --------------------------------------------------------------------------


def _read_system_instruction(view: Mapping[str, tuple[str, Any]], residual: dict[str, Any]) -> tuple[c.Text, ...]:
    """Lift ``systemInstruction`` into :attr:`~harness.contract.Conversation.system`.

    §3.3.1b: system instructions lift here, never into a turn, from whichever of
    the four carriers the format uses — Gemini's is this dedicated field.

    Args:
        view: The aliased top-level view.
        residual: The residual mapping, extended in place.

    Returns:
        One entry per text part, in order.

    Raises:
        UnreadableBodyError: When the field is not an object.
    """
    if "systemInstruction" not in view:
        return ()

    wire_key, value = view["systemInstruction"]
    if not isinstance(value, Mapping):
        raise c.UnreadableBodyError(f"{wire_key} must be a Content object, got {type(value).__name__}")

    content = _aliased(value, PUBLISHED_CONTENT_KEYS, wire_key, residual)
    parts = _members(content["parts"][1]) if "parts" in content else ()
    if parts is None:
        raise c.UnreadableBodyError(f"{wire_key}.parts must be a list or an object")

    system: list[c.Text] = []
    for index, part in enumerate(parts):
        path = f"{_join(wire_key, content['parts'][0])}[{index}]"
        if not isinstance(part, Mapping):
            raise c.UnreadableBodyError(f"{path} must be an object, got {type(part).__name__}")

        # The schema says system instructions are "text only", so anything else
        # in one has no slot in `Conversation.system` and residualises whole.
        member = _aliased(part, PUBLISHED_PART_KEYS, path, residual)
        text = _typed_leaf(member, "text", (str,), path, residual)
        if text is not None:
            system.append(c.Text(text))
        _residualise(member, {"text"}, path, residual)

    # `role` is meaningless on a system instruction and the grammar has no slot
    # for it, so it residualises like any other key the grammar cannot carry.
    _residualise(content, {"parts"}, wire_key, residual)
    return tuple(system)


def _read_contents(view: Mapping[str, tuple[str, Any]], residual: dict[str, Any]) -> tuple[c.Turn, ...]:
    """Read ``contents`` into normalised turns.

    Args:
        view: The aliased top-level view.
        residual: The residual mapping, extended in place.

    Returns:
        The turns, normalised per §3.3.1b's ordered pipeline.

    Raises:
        UnreadableBodyError: When ``contents`` is neither a list nor an object,
            a member is not an object, or a member's role is outside the
            published ``user``/``model``.
    """
    if "contents" not in view:
        # Absent is not an error: the oracle catches a vanished conversation as
        # a `conversation.turns` delta, which is a better diagnosis than an
        # unreadable body — and a reader that rejected an incomplete body could
        # not project the very mutation M5, M6 and M7 produce.
        return ()

    wire_key, value = view["contents"]
    members = _members(value)
    if members is None:
        raise c.UnreadableBodyError(f"{wire_key} must be a list or an object")

    turns: list[c.Turn] = []
    for index, member in enumerate(members):
        path = f"{wire_key}[{index}]"
        if not isinstance(member, Mapping):
            raise c.UnreadableBodyError(f"{path} must be an object, got {type(member).__name__}")

        content = _aliased(member, PUBLISHED_CONTENT_KEYS, path, residual)
        role = _read_role(content, path)
        parts = _read_parts(content, path, residual)
        turns.append(c.Turn(role=role, parts=_clause_three(role, parts)))
        _residualise(content, PUBLISHED_CONTENT_KEYS, path, residual)

    return _normalise_turns(turns)


def _read_role(content: Mapping[str, tuple[str, Any]], path: str) -> str:
    """Map a ``Content``'s producer onto §3.3.1b's closed role set.

    Args:
        content: The aliased ``Content`` view.
        path: The content's path from the body root.

    Returns:
        ``user`` or ``assistant``.

    Raises:
        UnreadableBodyError: When the role is present and is neither a published
            value nor the one evidenced value :data:`_EVIDENCED_ROLES` names.
            Structural, not residualisable: §7.4.1 makes "a role outside
            ``user``/``assistant``" its own example of a failure with no partial
            projection to salvage, because the turn cannot be built at all.
    """
    if "role" not in content:
        return "user"

    role = content["role"][1]
    if role in (None, ""):
        # The schema says the field "can be left blank or unset", and request
        # content with no stated producer is the client's.
        return "user"

    if isinstance(role, str) and role in _ROLES:
        return _ROLES[role]

    if isinstance(role, str) and role in _EVIDENCED_ROLES:
        return _EVIDENCED_ROLES[role]

    raise c.UnreadableBodyError(
        f"{path}.role must be one of {sorted(set(_ROLES) | set(_EVIDENCED_ROLES))}, got {role!r}; "
        "Gemini has no system role — a system prompt is the top-level systemInstruction"
    )


def _read_parts(content: Mapping[str, tuple[str, Any]], prefix: str, residual: dict[str, Any]) -> tuple[c.Part, ...]:
    """Read one ``Content``'s parts, in order.

    Args:
        content: The aliased ``Content`` view.
        prefix: The content's path from the body root.
        residual: The residual mapping, extended in place.

    Returns:
        The parts, in wire order.

    Raises:
        UnreadableBodyError: When ``parts`` is neither a list nor an object, or
            a member is not an object.
    """
    if "parts" not in content:
        return ()

    wire_key, value = content["parts"]
    path = _join(prefix, wire_key)
    members = _members(value)
    if members is None:
        raise c.UnreadableBodyError(f"{path} must be a list or an object")

    parts: list[c.Part] = []
    for index, member in enumerate(members):
        item = f"{path}[{index}]"
        if not isinstance(member, Mapping):
            raise c.UnreadableBodyError(f"{item} must be an object, got {type(member).__name__}")
        parts.append(_read_part(member, item, residual))

    return tuple(parts)


def _read_part(part: Mapping[str, Any], path: str, residual: dict[str, Any]) -> c.Part:
    """Read one ``Part`` into a projection part.

    ``Part`` is a union whose members are distinguished by **field name**, not
    by a ``type`` discriminator, which is why nothing here reads a ``type`` and
    why the opaque digest excludes nothing.

    Args:
        part: The part object.
        path: The part's path from the body root.
        residual: The residual mapping, extended in place.

    Returns:
        The projected part.

    Raises:
        UnreadableBodyError: When the part carries none of the published data
            members — §7.4.1's "a content block with no type": there is no
            partial part to salvage.
    """
    view = _aliased(part, PUBLISHED_PART_KEYS, path, residual)

    for name, kind in _OPAQUE_PART_KEYS.items():
        if name in view:
            return _read_opaque(view, name, kind, path, residual)

    if "inlineData" in view:
        return _read_inline_data(view, path, residual)

    if "fileData" in view:
        return _read_file_data(view, path, residual)

    if "functionCall" in view:
        return _read_function_call(view, path, residual)

    if "functionResponse" in view:
        return _read_function_response(view, path, residual)

    # Last, because `thought` is a part-level flag rather than a data member: a
    # `thought` beside a `functionCall` marks that call, and only a part with no
    # data member at all is itself a thought. Gemini returns signature-only
    # thought parts — `{"thought": true, "thoughtSignature": …}` with no text —
    # and §3.3.1 settles what they project as: "An empty block is a part with an
    # empty string, never nothing."
    if "text" in view or view.get("thought", ("", None))[1] is True:
        return _read_text(view, path, residual)

    raise c.UnreadableBodyError(f"{path} carries none of the published Part members {sorted(PUBLISHED_PART_KEYS)}")


def _read_text(view: Mapping[str, tuple[str, Any]], path: str, residual: dict[str, Any]) -> c.Text | c.Thinking:
    """Read a text part, as thinking when the part is marked as a thought.

    Args:
        view: The aliased ``Part`` view.
        path: The part's path from the body root.
        residual: The residual mapping, extended in place.

    Returns:
        :class:`~harness.contract.Thinking` when ``thought`` is true, carrying
        ``thoughtSignature`` — what M8's carrier repair manipulates — otherwise
        :class:`~harness.contract.Text`.

    Raises:
        UnreadableBodyError: When ``text`` is present and is not a string.
    """
    # A wrongly-typed `text` is **structural** and raises, where every other
    # optional leaf in this module residualises. §7.4.1's exception: `Text` and
    # `Thinking` *are* their value, so the grammar has no absent value to carry
    # and `Text("")` would fabricate an empty part the agent never sent — and an
    # empty part is meaningful here, since P5e and P8 both inject one. T-A1 takes
    # the same branch; T-A3 residualises the whole entry and returns no part,
    # which shifts every later part's index (§7.4.2 rule 7).
    if "text" in view and not isinstance(view["text"][1], str):
        raise c.UnreadableBodyError(f"{path} text must be a string, got {type(view['text'][1]).__name__}")

    text = _typed_leaf(view, "text", (str,), path, residual, default="")
    thought = _typed_leaf(view, "thought", (bool,), path, residual)

    if thought:
        signature = _typed_leaf(view, "thoughtSignature", (str,), path, residual)
        _residualise(view, {"text", "thought", "thoughtSignature"}, path, residual)
        return c.Thinking(text=text, signature=signature)

    # An empty block is a part with an empty string, never nothing (§3.3.1).
    _residualise(view, {"text", "thought"}, path, residual)
    return c.Text(text)


def _read_inline_data(view: Mapping[str, tuple[str, Any]], path: str, residual: dict[str, Any]) -> c.Image:
    """Read an inline ``Blob``, identifying it by digest rather than carrying bytes.

    ``inlineData`` is the format's only bytes carrier, whatever the media type,
    so audio and PDF payloads project the same way an image does and
    ``media_type`` records which it was.

    Args:
        view: The aliased ``Part`` view.
        path: The part's path from the body root.
        residual: The residual mapping, extended in place.

    Returns:
        The image part.

    Raises:
        UnreadableBodyError: When the blob is not an object. A payload that does
            not decode residualises instead — see the comment below.
    """
    wire_key, value = view["inlineData"]
    item = _join(path, wire_key)
    if not isinstance(value, Mapping):
        raise c.UnreadableBodyError(f"{item} must be an object, got {type(value).__name__}")

    blob = _aliased(value, PUBLISHED_BLOB_KEYS, item, residual)
    raw = _typed_leaf(blob, "data", (str,), item, residual, default="")
    try:
        decoded: bytes | None = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        # Residualised, not raised on: `Image.digest` is `str | None`, so the
        # grammar *has* an absent value here, and §7.4.1 is explicit that
        # "raising is the other wrong answer: it blinds the oracle to everything
        # else in a request it could otherwise diff". The case is real rather
        # than hypothetical — `validate=True` rejects every RFC 2045 line break,
        # and Google's own image sample passes `-w0` to `base64(1)` precisely
        # because its default output is wrapped. Nor is the part dropped: the
        # index would shift and invent a delta on every later part.
        residual[_join(item, blob["data"][0] if "data" in blob else "data")] = raw
        decoded = None

    _residualise(blob, {"data", "mimeType"}, item, residual)
    _residualise(view, {"inlineData"}, path, residual)
    # The media type is excluded from the digest and carried separately, so a
    # changed media type is its own delta rather than an unexplained change.
    return c.Image(
        digest=c.image_digest(decoded) if decoded is not None else None,
        media_type=_typed_leaf(blob, "mimeType", (str,), item, residual),
    )


def _read_file_data(view: Mapping[str, tuple[str, Any]], path: str, residual: dict[str, Any]) -> c.Image:
    """Read a ``FileData`` reference, which carries a URI rather than bytes.

    §3.3.1: "Gemini's ``fileData.fileUri`` has no bytes: ``digest`` is then
    absent and ``ref`` holds the URI."

    Args:
        view: The aliased ``Part`` view.
        path: The part's path from the body root.
        residual: The residual mapping, extended in place.

    Returns:
        The image part, by reference.

    Raises:
        UnreadableBodyError: When the file data is not an object.
    """
    wire_key, value = view["fileData"]
    item = _join(path, wire_key)
    if not isinstance(value, Mapping):
        raise c.UnreadableBodyError(f"{item} must be an object, got {type(value).__name__}")

    data = _aliased(value, PUBLISHED_FILE_DATA_KEYS, item, residual)
    projected = c.Image(
        ref=_typed_leaf(data, "fileUri", (str,), item, residual),
        media_type=_typed_leaf(data, "mimeType", (str,), item, residual),
    )
    _residualise(data, {"fileUri", "mimeType"}, item, residual)
    _residualise(view, {"fileData"}, path, residual)
    return projected


def _read_function_call(view: Mapping[str, tuple[str, Any]], path: str, residual: dict[str, Any]) -> c.ToolUse:
    """Read a ``functionCall`` into a tool use.

    **Gemini ``v1beta`` publishes an optional ``id``** on ``FunctionCall``, and
    the client echoes it back on the matching ``FunctionResponse``. An earlier
    note in ``contract.py`` said Gemini carried none, from the Cloud /
    Agent-Platform reference; the Developer API surface this reader reads
    differs, and KBR-36's own comment asked for it to be confirmed rather than
    assumed. Optional either way, so pairing still falls back to §3.3.1's
    name-and-position rule when no id is sent.

    ``args`` is a JSON **object** on the wire, so nothing here decodes a string:
    KBR-174's shared ``arguments`` helper is for the two formats that carry it
    as a string and is not a dependency of this reader.

    Args:
        view: The aliased ``Part`` view.
        path: The part's path from the body root.
        residual: The residual mapping, extended in place.

    Returns:
        The tool use.

    Raises:
        UnreadableBodyError: When the call is not an object. A call with no
            usable name residualises instead — see :func:`_read_required_name`.
    """
    wire_key, value = view["functionCall"]
    item = _join(path, wire_key)
    if not isinstance(value, Mapping):
        raise c.UnreadableBodyError(f"{item} must be an object, got {type(value).__name__}")

    call = _aliased(value, PUBLISHED_FUNCTION_CALL_KEYS, item, residual)
    name = _read_required_name(call, item, residual)
    projected = c.ToolUse(
        name=name,
        arguments=_typed_leaf(call, "args", (dict,), item, residual, default={}),
        id=_typed_leaf(call, "id", (str,), item, residual),
    )
    _residualise(call, PUBLISHED_FUNCTION_CALL_KEYS, item, residual)
    _residualise(view, {"functionCall"}, path, residual)
    return projected


def _read_function_response(view: Mapping[str, tuple[str, Any]], path: str, residual: dict[str, Any]) -> c.ToolResult:
    """Read a ``functionResponse`` into a tool result.

    ``response`` is a bare struct, which is why
    :class:`~harness.contract.Json` exists — §3.3.1: "Gemini's
    ``functionResponse.response`` is a bare struct", so a text-only result type
    would discard the usual payload entirely.

    ``is_error`` is always ``False``: Gemini publishes no error flag on a
    function response, and inventing one would make an absent flag and a
    reported success indistinguishable.

    Args:
        view: The aliased ``Part`` view.
        path: The part's path from the body root.
        residual: The residual mapping, extended in place.

    Returns:
        The tool result.

    Raises:
        UnreadableBodyError: When the response is not an object.
    """
    wire_key, value = view["functionResponse"]
    item = _join(path, wire_key)
    if not isinstance(value, Mapping):
        raise c.UnreadableBodyError(f"{item} must be an object, got {type(value).__name__}")

    answer = _aliased(value, PUBLISHED_FUNCTION_RESPONSE_KEYS, item, residual)
    content: list[c.Text | c.Image | c.Json | c.Opaque] = []

    payload = _typed_leaf(answer, "response", (dict,), item, residual)
    if payload is not None:
        content.append(c.Json(payload))

    content.extend(_read_response_parts(answer, item, residual))

    projected = c.ToolResult(
        content=content,
        tool_use_id=_typed_leaf(answer, "id", (str,), item, residual),
        is_error=False,
    )
    # `name` pairs the result with its call where no id was sent, and the
    # grammar's pairing rule is by name and position, so it is accounted for
    # rather than residualised.
    _residualise(answer, {"response", "parts", "id", "name"}, item, residual)
    _residualise(view, {"functionResponse"}, path, residual)
    return projected


def _read_response_parts(answer: Mapping[str, tuple[str, Any]], prefix: str, residual: dict[str, Any]) -> list[c.Image]:
    """Read a function response's inline media parts.

    Args:
        answer: The aliased ``FunctionResponse`` view.
        prefix: The response's path from the body root.
        residual: The residual mapping, extended in place.

    Returns:
        One image per ``inlineData`` member, in order.

    Raises:
        UnreadableBodyError: When ``parts`` is neither a list nor an object, or
            a member is not an object.
    """
    if "parts" not in answer:
        return []

    wire_key, value = answer["parts"]
    path = _join(prefix, wire_key)
    members = _members(value)
    if members is None:
        raise c.UnreadableBodyError(f"{path} must be a list or an object")

    images: list[c.Image] = []
    for index, member in enumerate(members):
        item = f"{path}[{index}]"
        if not isinstance(member, Mapping):
            raise c.UnreadableBodyError(f"{item} must be an object, got {type(member).__name__}")

        part = _aliased(member, frozenset({"inlineData"}), item, residual)
        if "inlineData" in part:
            images.append(_read_inline_data(part, item, residual))
        else:
            _residualise(part, set(), item, residual)

    return images


def _read_opaque(
    view: Mapping[str, tuple[str, Any]], name: str, kind: str, path: str, residual: dict[str, Any]
) -> c.Opaque:
    """Read a part the grammar does not model, keeping it detectable.

    §3.3.1 put ``digest`` on :class:`~harness.contract.Opaque` precisely so that
    unmodelled content stays detectable: a bare ``Opaque("executable_code")``
    would make two different programs project identically, and a swapped one
    would produce no delta at all.

    Args:
        view: The aliased ``Part`` view.
        name: The published ``Part`` member carrying the payload.
        kind: The canonical snake_case name for it.
        path: The part's path from the body root.
        residual: The residual mapping, extended in place.

    Returns:
        The opaque part, carrying a digest of its payload.
    """
    payload = view[name][1]
    _residualise(view, {name}, path, residual)
    return c.Opaque(kind=kind, digest=_payload_digest(payload))


def _payload_digest(payload: Any) -> str:
    """Return the digest of an unmodelled part's payload.

    §7.4.1's recipe, exactly. Over **canonical** JSON rather than the raw wire
    slice, so a translator that reorders keys does not change the digest.
    ``ensure_ascii`` is pinned alongside ``sort_keys`` and ``separators``
    because its default is ``True`` while the surrounding prose says UTF-8: an
    author who passed ``False`` would get a different digest for the same part,
    visible only on non-ASCII content, which is the cross-reader disagreement
    §7.4.1 exists to prevent. In T-A1 all three wrong spellings survived
    mutation testing until one digest was pinned to an external literal.

    Nothing is excluded. §7.4.1 excludes ``type`` because it is already
    :attr:`~harness.contract.Opaque.kind` and ``cache_control`` because it
    residualises; a Gemini ``Part`` is discriminated by **field name** and
    defines no ``cache_control``, so neither exclusion has anything to remove.

    Args:
        payload: The unmodelled member's value.

    Returns:
        Lowercase hex SHA-256 of the canonical payload.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# §3.3.1b's merge pipeline
# --------------------------------------------------------------------------


def _clause_three(role: str, parts: Sequence[c.Part]) -> tuple[c.Part, ...]:
    """Order one ``Content``'s parts, results first, for a ``user`` content.

    §3.3.1b's third clause, applied where §7.4.1 says it is **not** idle: "it
    governs the formats that carry text and results inside **one message**,
    where the run has no natural boundary". Gemini is that case — a single
    ``parts`` array may hold text and ``functionResponse`` members together, so
    there is no message boundary to delimit a run and clause 1 cannot do the
    work by splitting.

    **Scoped to one content, and to ``user``.** Applying it to a *merged* turn
    projects ``functionResponse -> user(text) -> functionResponse`` as
    ``[ToolResult, ToolResult, Text]``, hoisting a result ahead of text the
    agent sent *before* it. Per content, that sequence keeps its three separate
    groups and concatenates to ``[ToolResult, Text, ToolResult]``, which is what
    §3.3.1b requires.

    Args:
        role: The content's projected role.
        parts: The content's parts, in wire order.

    Returns:
        The parts, results first for a ``user`` content, stable within each
        group so a reader never sorts.
    """
    if role != "user":
        return tuple(parts)

    results = [part for part in parts if isinstance(part, c.ToolResult)]
    others = [part for part in parts if not isinstance(part, c.ToolResult)]
    return tuple(results) + tuple(others)


def _normalise_turns(turns: Sequence[c.Turn]) -> tuple[c.Turn, ...]:
    """Merge consecutive same-role turns, preserving the order of their groups.

    The fourth and last clause of §3.3.1b's ordered pipeline. Clause 3 has
    already run, per content, in :func:`_clause_three`; this one only
    concatenates, and **never re-sorts what it concatenates**.

    **A re-sort here would be a defect, not a simplification.** It hoists a
    result ahead of text the agent sent *before* it — moving history the bridge
    did not move — and because paths are index-based that invented delta lands
    on every part of the turn and every turn after it.

    **What the merge still hides**, recorded because §3.3.1 requires a
    projection's blind spots to be on the record: a mutation whose only effect
    is to split or join two consecutive same-role turns produces no delta, which
    weakens §3.3.2 assertion 2 for M5, M6 and M7. Accepted, because without the
    merge this reader and the Chat Completions reader disagree about turn
    boundaries on the standard ``assistant(tool_calls) -> tool -> tool``
    exchange, and that disagreement reports a false delta on every subsequent
    turn.

    Args:
        turns: The turns, one per content, in wire order.

    Returns:
        The normalised turns.
    """
    merged: list[c.Turn] = []
    for turn in turns:
        if merged and merged[-1].role == turn.role:
            previous = merged.pop()
            merged.append(c.Turn(role=turn.role, parts=tuple(previous.parts) + tuple(turn.parts)))
        else:
            merged.append(turn)

    return tuple(merged)
