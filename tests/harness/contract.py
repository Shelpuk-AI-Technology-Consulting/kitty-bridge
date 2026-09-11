"""The input contract — everything a wire projection reads and a recorder produces.

`.system_design/TEST_SUITE.md` §3.3.1 · plan task **T-W2** (KBR-25).

This module is the single owner of the vocabulary the fidelity oracle is
written in.  Eleven tasks depend on it: the register (T-W3), the recorder (T-W4),
the vertical slice (T-W9), the six wire readers (T-A1–T-A6), the reply
projection (T-A7) and the oracle core (T-D1).

**It imports nothing from ``src/kitty``, and must not.**  §3.3.1's
independent-oracle rule: "The oracle must not be written in terms of the code
under test."  A projection that asked kitty how to read a body would inherit
kitty's bugs and the whole of I1 would prove only self-consistency.
``test_contract.py`` asserts the absence structurally.

**What lives here and what does not.**  This module defines *shapes and rules*.
It reads no bodies — the six readers do that, each against its format's
published examples.  It captures nothing — the recorders do that.  It asserts
nothing about kitty — the oracle does that.

**The totality rule is the load-bearing part.**  §3.3.1 requires every key in a
body to be classified into the envelope, the conversation, or the residual, and
makes a non-empty residual fail the run.  :func:`verify_total` enforces it, and
the reason it needs :attr:`Request.consumed` rather than just checking the
residual is worth stating: *a reader that **drops** an unknown key produces an
empty residual and would sail through.*  Totality is decidable only against the
source body, so a reader must declare what it claimed to handle.  That is
exactly the falsification case plan §1.4 requires this harness to ship with.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

# --------------------------------------------------------------------------
# Wire formats
# --------------------------------------------------------------------------


class WireFormat(Enum):
    """The six wire formats §3.3.1 names, and no others.

    Closed deliberately.  §7.4 notes that "a **boolean** declaration cannot
    select among the six projections"; a bare ``str`` has the opposite failure,
    where six reader authors spell one format three ways and the oracle's
    format-keyed lookup silently misses.
    """

    ANTHROPIC_MESSAGES = "anthropic_messages"
    CHAT_COMPLETIONS = "chat_completions"
    OPENAI_RESPONSES = "openai_responses"
    GEMINI = "gemini"
    BEDROCK_CONVERSE = "bedrock_converse"
    OLLAMA_CHAT = "ollama_chat"


# --------------------------------------------------------------------------
# Immutability helpers
# --------------------------------------------------------------------------


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an unmodifiable view over a **shallow** copy of ``value``.

    ``frozen=True`` blocks rebinding a field but not mutation *through* it, so a
    plain ``dict`` field would leave the oracle able to alter what it compares.

    **The freeze is deliberately shallow.**  Recursively freezing (dict to
    ``MappingProxyType``, list to tuple) would close the last level, but it
    changes *equality*: ``arguments == {"a": [1, 2]}`` becomes false once the
    list is a tuple.  The six readers are specified to test against each
    format's published examples, which are dict and list literals — so every one
    of them would have to compare against a bespoke frozen form instead.  Not
    mutating the JSON leaves is therefore a convention the oracle keeps, not a
    guarantee this type enforces.

    Args:
        value: The mapping to freeze, or ``None`` for an empty one.

    Returns:
        A read-only mapping proxy over a shallow copy.
    """
    return MappingProxyType(dict(value or {}))


def _frozen_field() -> Any:
    """Return a dataclass field defaulting to an empty frozen mapping.

    Returns:
        A ``dataclasses.field`` with a frozen-mapping default factory.
    """
    return field(default_factory=lambda: _freeze_mapping(None))


# --------------------------------------------------------------------------
# Part variants
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Text:
    """A run of plain text.

    Carries no content-type tag.  P16 rewrites Responses' ``input_text`` to
    ``output_text``, but that tag is redundant with :attr:`Turn.role` on that
    path, so carrying it would put one vendor's spelling into a form whose
    purpose is wire independence.  P16 is unconditional and therefore exempt
    from §3.3.2 assertion 2; its guards are the Responses reader's own L1 test
    and T-G4.

    Attributes:
        text: The text content, empty string included.
    """

    text: str

    # Every projection type sets this, so unhashability is total rather than
    # data-dependent — see the module note on multiset matching. It survives
    # `@dataclass` only because none of these classes defines `__eq__` in its
    # own body; adding one would silently restore a working `__hash__`.
    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ToolUse:
    """A tool call the assistant made.

    Attributes:
        id: The call id, or ``None`` where the format carries none. Gemini's
            ``functionCall`` is ``{name, args}`` with no id, so a required id
            would force T-A4 to synthesise one and show a delta on every tool
            turn.
        name: The tool's name.
        arguments: The parsed arguments. Chat Completions encodes these as a
            JSON *string* and Messages as an object; normalising here stops a
            spurious delta on every cross-format comparison.
    """

    name: str
    arguments: Mapping[str, Any] = _frozen_field()
    id: str | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Freeze the arguments mapping in place."""
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))


@dataclass(frozen=True)
class Json:
    """A structured tool result.

    Bedrock Converse's ``toolResult.content`` union has ``json`` as its common
    case and Gemini's ``functionResponse.response`` is a bare struct, so a
    text-only result type would discard the usual payload entirely.

    Attributes:
        value: The parsed JSON value. Typed ``Any`` rather than ``Mapping``
            because a Chat Completions tool message may carry a JSON *array*,
            while Converse's ``json`` and Gemini's ``response`` are objects.
            Not frozen — see :func:`_freeze_mapping` on why nesting is held by
            convention rather than enforced.
    """

    value: Any = None

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Opaque:
    """Content the grammar does not model, kept detectable rather than dropped.

    Covers Converse's ``document``/``video``/``searchResult`` and Anthropic's
    ``document``/``search_result``.  Unlike a content-type tag on :class:`Text`,
    this names content the grammar *cannot* express — without it the content
    vanishes from both sides identically and the "no unclaimed delta" assertion
    passes over a real loss.

    Attributes:
        kind: A canonical snake_case name, never the wire's spelling. Anthropic
            writes ``search_result`` and Converse ``searchResult`` for one
            thing; the wire spellings would be a permanent unclaimed delta.
        digest: A content digest, as for :class:`Image`.
    """

    kind: str
    digest: str | None = None

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Image:
    """An image, identified by digest rather than carried as bytes.

    Attributes:
        digest: Lowercase hex SHA-256 of the *decoded* image bytes, or ``None``
            when the format carries a reference instead. ``media_type`` is
            deliberately **not** part of the digest, so a changed media type is
            its own delta rather than an unexplained digest change.
        media_type: The declared media type, when the format states one.
        ref: The URI, for Gemini's ``fileData.fileUri`` which carries no bytes.
    """

    digest: str | None = None
    media_type: str | None = None
    ref: str | None = None

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Thinking:
    """An extended-thinking block.

    An *empty* block is a part with an empty string, never nothing: P5e injects
    ``{"type":"thinking","thinking":""}`` and P8 an empty ``reasoning_content``.
    P8's trigger is conditional and *inferred*, so §3.3.2 assertion 2 needs its
    absence to be observable.

    Attributes:
        text: The thinking text, empty string included.
        signature: Anthropic's ``signature`` or Gemini's ``thoughtSignature`` —
            what M8's carrier repair manipulates.
    """

    text: str
    signature: str | None = None

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ToolResult:
    """The result of a tool call, as a part inside a ``user`` turn.

    Attributes:
        tool_use_id: The id of the call this answers, or ``None`` where the
            format carries none. When absent, pairing is by tool name and the
            k-th unanswered call of that name in the most recent assistant turn.
        content: Ordered content. Not recursive: no format nests a tool call
            inside a tool result.
        is_error: Whether the tool reported failure.
    """

    content: Sequence[Text | Image | Json | Opaque] = ()
    tool_use_id: str | None = None
    is_error: bool = False

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Freeze the content sequence in place."""
        object.__setattr__(
            self, "content", _checked_members(self.content, RESULT_PART_TYPES, "ToolResult.content")
        )


#: Every :data:`Part` variant, as a tuple for ``isinstance`` and for the guard
#: that asserts the union has not silently grown.
PART_TYPES = (Text, ToolUse, ToolResult, Thinking, Image, Json, Opaque)

#: What a :class:`ToolResult` may carry.  Deliberately narrower than
#: :data:`PART_TYPES` and deliberately not recursive: no wire format nests a
#: tool call inside a tool result.
RESULT_PART_TYPES = (Text, Image, Json, Opaque)

Part = Text | ToolUse | ToolResult | Thinking | Image | Json | Opaque


# --------------------------------------------------------------------------
# Captures
# --------------------------------------------------------------------------

#: What a redacted value is replaced by in a ``repr``. Masking to *nothing*
#: would hide a missing-credential bug as effectively as a leak hides a present
#: one, so the mask is visible.
REDACTION_MASK = "<redacted>"

#: Header names whose values never appear in a ``repr``, lowercased for
#: case-insensitive matching. Seven entries, in two groups.
#:
#: **Five the recorders (§7.2) will actually see**, one per carrier:
#: ``x-api-key`` from Anthropic, ``api-key`` from Azure and P9b's MiMo,
#: ``x-goog-api-key`` from Gemini, ``authorization`` from Vertex's OAuth leg,
#: and ``proxy-authorization`` from the CONNECT legs (§5.2.1).
#:
#: **Two precautionary**: ``cookie`` and ``set-cookie``. No adapter authenticates
#: by cookie today, so nothing exercises them — they are here because a session
#: cookie is a credential and the cost of listing one nobody sends is nil, while
#: the cost of omitting one somebody starts sending is a leak into every CI log.
#: :attr:`CapturedReply` is the likelier carrier of the two.
REDACTED_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "api-key", "x-goog-api-key", "cookie", "set-cookie"}
)

#: Query-string keys whose values never appear in a ``repr``. Gemini carries its
#: credential in the URL, which :attr:`CapturedRequest.query` preserves verbatim.
REDACTED_QUERY_KEYS = frozenset({"key", "api_key", "access_token"})


def _checked_members(values: Any, allowed: tuple[type, ...], field_name: str) -> tuple[Any, ...]:
    """Return ``values`` as a tuple, rejecting anything outside ``allowed``.

    The contract declares several closed sets — the :data:`Part` union, the
    parts a :class:`ToolResult` may carry, the types a :class:`Conversation`
    holds. *Closed* has meant *enforced* everywhere else in this module
    (:data:`ROLES`, :data:`SAMPLING_KEYS`, :data:`STOP_REASONS`,
    :data:`TOOL_CHOICE_VALUES`), and a declared-but-unchecked union is the same
    defect: a rule that reads like a guarantee and guarantees nothing.

    Args:
        values: The sequence given for the field.
        allowed: The types a member may be.
        field_name: The field's name, for the error message.

    Returns:
        The members, order untouched.

    Raises:
        TypeError: When a member is outside ``allowed``.
    """
    # Materialise once: reading a one-shot iterable twice would leave the store
    # empty and silent, which is the defect round 9 closed on headers.
    members = tuple(values)

    offenders = sorted({type(m).__name__ for m in members if not isinstance(m, allowed)})
    if offenders:
        names = ", ".join(t.__name__ for t in allowed)
        raise TypeError(f"{field_name} accepts only {names}; got {', '.join(offenders)}")

    return members


def _normalised_headers(headers: Any) -> tuple[tuple[str, str], ...]:
    """Return ``headers`` as validated name/value pairs.

    Shared by :class:`CapturedRequest` and :class:`CapturedReply`. It is a
    function rather than a copy in each ``__post_init__`` because the two copies
    it replaces both carried the same defect: a bug found in one was, silently,
    a bug in the other.

    Args:
        headers: The value given for a capture's ``headers`` field.

    Returns:
        The header pairs, order and casing untouched.

    Raises:
        TypeError: When ``headers`` is a mapping, when any entry is a string or
            bytes, or when any entry is not a name/value pair.
    """
    if isinstance(headers, Mapping):
        raise TypeError("headers must be a sequence of (name, value) pairs, not a mapping")

    # Materialise once. Iterating twice would consume a generator on the first
    # pass and leave the second seeing nothing -- storing no headers at all,
    # silently, from a guard whose whole purpose is to fail loudly.
    entries = tuple(headers)

    # A string is a sequence too, so `tuple("ab")` is a *valid-looking* 2-tuple
    # of characters. Rejecting the type is the only check that catches it; a
    # length check cannot.
    if any(isinstance(entry, str | bytes) for entry in entries):
        raise TypeError("each header must be a (name, value) pair, not a string")

    pairs = tuple(tuple(entry) for entry in entries)
    if any(len(pair) != 2 for pair in pairs):
        raise TypeError("each header must be a (name, value) pair")

    # A `bytes` name survives a length check and then misses the credential
    # mask outright: `b"authorization"` is not in a set of `str`, so the value
    # is rendered in full. Checking the length without the element types is how
    # a leak walks through a guard that looks like it covers this.
    if any(not isinstance(part, str) for pair in pairs for part in pair):
        raise TypeError("header names and values must both be str")

    return pairs


def _redact_headers(headers: Sequence[tuple[str, str]]) -> str:
    """Render header pairs with credential values masked.

    Args:
        headers: Ordered header pairs, original casing preserved.

    Returns:
        A display string safe for a log or an assertion diff.
    """
    shown = [(n, REDACTION_MASK if n.lower() in REDACTED_HEADERS else v) for n, v in headers]
    return repr(tuple(shown))


def _redact_query(query: str) -> str:
    """Render a raw query string with credential values masked.

    Splits on ``&`` and ``=`` without a URL parser, because the raw string is
    what was on the wire and re-encoding it would misrepresent the capture.

    Args:
        query: The raw query string as observed.

    Returns:
        A display string safe for a log or an assertion diff.
    """
    if not query:
        return repr(query)

    parts = []
    for pair in query.split("&"):
        name, sep, value = pair.partition("=")
        parts.append(f"{name}{sep}{REDACTION_MASK}" if sep and name.lower() in REDACTED_QUERY_KEYS else pair)
    return repr("&".join(parts))


@dataclass(frozen=True)
class CapturedRequest:
    """A complete request as observed on the wire.

    Bodies alone cannot prove correct routing (§3.3.5): on Azure the deployment
    id lives in the path and P6 removes ``model`` from the body, so two requests
    to two different deployments have byte-identical bodies.  The oracle
    therefore takes the whole request.

    The first seven fields are the contract T-W2 owns.  :attr:`arrival` and
    :attr:`peer_port` are **T-W4's to populate** and §5.2.1's to consume — they
    live here only so that T-W4, T-B1–T-B3 and T-E2 share one type instead of a
    subclass; no fidelity assertion reads them.

    Attributes:
        method: HTTP method.
        scheme: URL scheme.
        host: URL host.
        path: URL path, including Gemini's ``:generateContent`` operation.
        query: The raw query string, unparsed and unreordered.
        headers: Ordered pairs with original casing and duplicates preserved,
            because §4.3 C1 asserts on the exact header set.
        body: Raw body bytes, undecoded.
        arrival: Arrival timestamp. T-W4's.
        peer_port: Peer port of the accepted connection, the join key against
            the proxy's tunnel log (§5.2.1). T-W4's.
    """

    method: str
    scheme: str
    host: str
    path: str
    query: str
    headers: Sequence[tuple[str, str]] = ()
    body: bytes = b""
    arrival: float | None = None
    peer_port: int | None = None

    def __post_init__(self) -> None:
        """Freeze the header sequence in place.

        Raises:
            TypeError: When ``headers`` is a mapping, or any entry is not a
                name/value pair. R1.2 makes headers a sequence of pairs
                precisely so duplicates and casing survive — but iterating a
                mapping yields its *keys*, so a mapping would be silently
                shredded into character tuples and the header evidence §4.3 C1
                asserts on would be gone. Failing loudly is the point.
        """
        object.__setattr__(self, "headers", _normalised_headers(self.headers))

    def __repr__(self) -> str:
        """Render the capture with credentials masked.

        A dataclass ``repr`` reaches every pytest assertion diff and every CI
        log.  §7.1 already flags this trap for the corpus and §5.4 mandates the
        analogous property for ``EgressConfig``; a capture carries the same
        class of secret.

        Returns:
            A display string with no credential values in it.
        """
        return (
            f"CapturedRequest(method={self.method!r}, scheme={self.scheme!r}, host={self.host!r}, "
            f"path={self.path!r}, query={_redact_query(self.query)}, headers={_redact_headers(self.headers)}, "
            f"body={self.body!r}, arrival={self.arrival!r}, peer_port={self.peer_port!r})"
        )


@dataclass(frozen=True)
class CapturedReply:
    """A complete reply as observed on the wire, for :class:`ReplyProjection`.

    Attributes:
        status: HTTP status code.
        headers: Ordered pairs with original casing preserved.
        body: Raw body bytes. A streaming reply is reassembled before it
            reaches a reply projection; reassembly is T-A7's boundary.
    """

    status: int
    headers: Sequence[tuple[str, str]] = ()
    body: bytes = b""

    def __post_init__(self) -> None:
        """Freeze the header sequence in place.

        Raises:
            TypeError: When ``headers`` is a mapping, or any entry is not a
                name/value pair. R1.2 makes headers a sequence of pairs
                precisely so duplicates and casing survive — but iterating a
                mapping yields its *keys*, so a mapping would be silently
                shredded into character tuples and the header evidence §4.3 C1
                asserts on would be gone. Failing loudly is the point.
        """
        object.__setattr__(self, "headers", _normalised_headers(self.headers))

    def __repr__(self) -> str:
        """Render the reply with credentials masked.

        Returns:
            A display string with no credential values in it.
        """
        return (
            f"CapturedReply(status={self.status!r}, headers={_redact_headers(self.headers)}, body={self.body!r})"
        )


def image_digest(raw: bytes) -> str:
    """Return the canonical digest of decoded image bytes.

    Pinned so that six independently written readers agree.  Anthropic sends
    base64 plus a media type, Chat Completions a data URL, Converse raw bytes
    and Gemini either inline data or a URI — without one definition the Messages
    reader and the Chat Completions reader would produce different digests for
    the same image and §7.1's image corpus entry would fail on every run.

    Args:
        raw: The decoded image bytes. The media type is deliberately excluded,
            so a changed media type shows as its own delta.

    Returns:
        Lowercase hex SHA-256 of ``raw``.
    """
    return hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------
# Conversation
# --------------------------------------------------------------------------

#: The only two roles a projected turn may carry.  ``system``, ``developer`` and
#: Responses' ``instructions`` all lift into :attr:`Conversation.system` instead
#: of becoming turns (R8.2); Gemini's ``model`` maps to ``assistant``.  Closed,
#: because an open vocabulary would let Gemini show an unclaimed delta on every
#: assistant turn.
ROLES = frozenset({"user", "assistant"})

#: Canonical sampling parameter names, in the Chat Completions spelling.  The
#: first fourteen are exactly what P13 drops; ``top_k`` is carried by Gemini and
#: Converse and has no Chat Completions spelling.  Responses'
#: ``max_output_tokens`` normalises onto ``max_tokens`` (R8.5), but
#: ``max_completion_tokens`` stays distinct — P13 drops it in its own right.
SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_completion_tokens",
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "top_logprobs",
        "response_format",
        "stop",
        "n",
        "stream_options",
        "seed",
        "logit_bias",
    }
)

#: Canonical stop reasons for the response direction.  ``other`` is the escape:
#: Gemini adds ``SAFETY`` and ``RECITATION``, Anthropic has added values over
#: time, and Ollama reports ``done_reason: load``.  Without it a legitimate
#: safety-blocked reply would fail the run instead of projecting — the mistake
#: R9.3 avoids on the request side.  An unmapped value projects as ``other``,
#: with the wire's own string kept in :attr:`Reply.stop_reason_raw`.
#:
#: **Not in the residual.**  An earlier draft put it there, which defeated the
#: escape it was meant to be: :func:`verify_total` fails the run on *any*
#: non-empty residual, so a reader following that rule literally would have
#: failed every safety-blocked Gemini reply.  A value mapped to ``other`` has
#: been seen and classified — it is accounted for, not unaccounted — so the
#: residual was the wrong home for it on the contract's own terms.
STOP_REASONS = frozenset({"end_turn", "max_tokens", "stop_sequence", "tool_use", "error", "other"})


@dataclass(frozen=True)
class Turn:
    """One conversational turn.

    Attributes:
        role: Either ``user`` or ``assistant`` (:data:`ROLES`).
        parts: Ordered content. A tool result is a :class:`ToolResult` part
            inside a ``user`` turn, which is Anthropic Messages' shape; Chat
            Completions' separate ``role: "tool"`` messages merge into one turn.
    """

    role: str
    parts: Sequence[Part] = ()

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Validate the role and freeze the parts sequence.

        Raises:
            ValueError: When ``role`` is outside :data:`ROLES`. Construction is
                not parsing — T-D1 builds expected conversations by hand — so
                this is a ``ValueError`` and not :class:`UnreadableBodyError`.
            TypeError: When a member of ``parts`` is outside :data:`PART_TYPES`.
        """
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {sorted(ROLES)}, got {self.role!r}")
        object.__setattr__(self, "parts", _checked_members(self.parts, PART_TYPES, "Turn.parts"))


@dataclass(frozen=True)
class ToolDecl:
    """A tool the agent declared.

    Attributes:
        name: The tool's name. Paths address tools by name, not index (R7.3).
        description: The description, or ``None`` when absent. One oracle
            falsification case deletes it (§3.3.1).
        schema: The parameter schema, or ``None`` when absent.
        strict: P15 strips this on the Responses-origin path. ``None`` means
            absent, which must stay distinct from ``False`` or that row's
            presence and absence would be indistinguishable.
    """

    name: str
    description: str | None = None
    schema: Mapping[str, Any] | None = None
    strict: bool | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Freeze the schema mapping when one is present."""
        if self.schema is not None:
            object.__setattr__(self, "schema", _freeze_mapping(self.schema))


@dataclass(frozen=True)
class Conversation:
    """The semantic content of a request, independent of any wire format.

    Attributes:
        system: Ordered system text, lifted here from whichever of the four
            carriers the format uses (R8.2).
        turns: Ordered turns.
        tools: Ordered tool declarations.
        sampling: Sampling parameters, keyed by :data:`SAMPLING_KEYS`.
    """

    system: Sequence[Text] = ()
    turns: Sequence[Turn] = ()
    tools: Sequence[ToolDecl] = ()
    sampling: Mapping[str, Any] = _frozen_field()

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Freeze every sequence and validate the sampling keys.

        Raises:
            TypeError: When ``sampling`` is not a mapping.
            ValueError: When a sampling key is outside :data:`SAMPLING_KEYS`.
                The set is closed (§3.3.1b), and enforcing it here is what
                stops a reader dropping a format's own control field into
                ``sampling`` — Gemini's ``generationConfig`` members are the
                likely accident. Validated the way :class:`Turn` validates its
                role, rather than left as an unenforced reader obligation.
        """
        object.__setattr__(self, "system", _checked_members(self.system, (Text,), "Conversation.system"))
        object.__setattr__(self, "turns", _checked_members(self.turns, (Turn,), "Conversation.turns"))
        object.__setattr__(self, "tools", _checked_members(self.tools, (ToolDecl,), "Conversation.tools"))

        if not isinstance(self.sampling, Mapping):
            raise TypeError("sampling must be a mapping of canonical key to value")

        unknown = sorted(set(self.sampling) - SAMPLING_KEYS)
        if unknown:
            raise ValueError(
                f"sampling keys must be canonical (§3.3.1b); {unknown} are not. "
                "A recognised control field of the format belongs in envelope.extra; "
                "an unrecognised key belongs in the residual."
            )
        object.__setattr__(self, "sampling", _freeze_mapping(self.sampling))


@dataclass(frozen=True)
class Envelope:
    """Routing and control fields.

    Attributes:
        model: M1 replaces this; P6 removes it from the body and P20 moves it
            into the URL; Converse's ``modelId`` normalises onto it (R8.4).
        stream: P17 injects it, M11 forces it false, P18 and P19 rewrite it.
        store: P17 injects it.
        extra: Every other control field the format defines, **keyed by the wire
            key**, so a register row can name ``envelope.extra[thinking]``. The
            single exception is ``tool_choice``, which unifies four wire keys
            because they name one concept (R8.6).
    """

    model: str | None = None
    stream: bool | None = None
    store: bool | None = None
    extra: Mapping[str, Any] = _frozen_field()

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Validate any tool choice and freeze the extra mapping.

        ``extra`` is otherwise open by design — it holds whatever control
        fields a format defines, keyed by the wire key. ``tool_choice`` is the
        one entry with a *canonical* value (R8.6), so it is the one entry worth
        checking; leaving it unchecked would make :data:`TOOL_CHOICE_VALUES` a
        comment rather than a rule.

        Raises:
            ValueError: When ``extra["tool_choice"]`` is outside
                :data:`TOOL_CHOICE_VALUES` and is not a ``tool:<name>``
                selection.
        """
        choice = (self.extra or {}).get(TOOL_CHOICE_KEY)
        if choice is not None and not (
            choice in TOOL_CHOICE_VALUES or (isinstance(choice, str) and choice.startswith("tool:"))
        ):
            raise ValueError(
                f"tool_choice must be one of {sorted(TOOL_CHOICE_VALUES)} or 'tool:<name>', got {choice!r}"
            )

        object.__setattr__(self, "extra", _freeze_mapping(self.extra))


# --------------------------------------------------------------------------
# Projections
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    """A request projected into the wire-independent form.

    Attributes:
        envelope: Routing and control.
        conversation: Semantic content.
        residual: Paths the reader could not classify, mapped to their values.
            A non-empty residual **fails the run** (§3.3.1).
        consumed: Top-level body keys the reader mapped into the envelope or the
            conversation. Without this a dropped key is undetectable — see
            :func:`verify_total`.
        source: The mapping the reader parsed. Carried so that
            :func:`verify_total` needs no second parse, which could disagree
            with the reader's about duplicate keys.
    """

    envelope: Envelope
    conversation: Conversation
    residual: Mapping[str, Any] = _frozen_field()
    consumed: frozenset[str] = frozenset()
    source: Mapping[str, Any] = _frozen_field()

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Freeze the residual and source mappings."""
        object.__setattr__(self, "residual", _freeze_mapping(self.residual))
        object.__setattr__(self, "source", _freeze_mapping(self.source))


@dataclass(frozen=True)
class Reply:
    """A reply projected into the wire-independent form, for T-A7.

    Attributes:
        parts: Ordered content of the assistant's reply.
        stop_reason: One of :data:`STOP_REASONS`.
        stop_reason_raw: The wire's own value when ``stop_reason`` is ``other``,
            and ``None`` otherwise. Keeps a Gemini ``SAFETY`` distinguishable
            from a ``RECITATION`` without failing the run, which putting it in
            the residual would have done.
        usage: Token counts. **Carried but excluded from the fidelity diff** —
            usage is provider-reported and never agent-supplied, so a difference
            carries no I1 information.
        residual: As :attr:`Request.residual`.
        consumed: As :attr:`Request.consumed`.
        source: As :attr:`Request.source`.
    """

    parts: Sequence[Part] = ()
    stop_reason: str | None = None
    stop_reason_raw: str | None = None
    usage: Mapping[str, Any] = _frozen_field()
    residual: Mapping[str, Any] = _frozen_field()
    consumed: frozenset[str] = frozenset()
    source: Mapping[str, Any] = _frozen_field()

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Validate the stop reason and freeze every sequence and mapping.

        Raises:
            ValueError: When ``stop_reason`` is outside :data:`STOP_REASONS`, or
                when it does not pair correctly with ``stop_reason_raw``. Closed
                means enforced, the same posture :class:`Turn` takes on roles and
                :class:`Conversation` on sampling keys — a vocabulary declared
                closed but checked nowhere is a comment, not a rule, and the same
                applies to an invariant written only in a docstring.
        """
        if self.stop_reason is not None and self.stop_reason not in STOP_REASONS:
            raise ValueError(
                f"stop_reason must be one of {sorted(STOP_REASONS)}, got {self.stop_reason!r}; "
                "an unmapped wire value projects as 'other' with the original in stop_reason_raw"
            )

        # The pairing is the escape's whole value. `other` without the wire's
        # string discards what T-D10 needs to tell a SAFETY block from a
        # RECITATION; a raw value beside a canonical reason means the reader
        # mapped it and kept a stale original.
        if self.stop_reason == "other" and self.stop_reason_raw is None:
            raise ValueError("stop_reason 'other' must carry the wire's own value in stop_reason_raw")
        if self.stop_reason != "other" and self.stop_reason_raw is not None:
            raise ValueError(
                f"stop_reason_raw is only for 'other', but stop_reason is {self.stop_reason!r}"
            )

        object.__setattr__(self, "parts", _checked_members(self.parts, PART_TYPES, "Reply.parts"))
        object.__setattr__(self, "usage", _freeze_mapping(self.usage))
        object.__setattr__(self, "residual", _freeze_mapping(self.residual))
        object.__setattr__(self, "source", _freeze_mapping(self.source))


# --------------------------------------------------------------------------
# The totality rule
# --------------------------------------------------------------------------


class UnreadableBodyError(Exception):
    """A body a reader could not read at all.

    Named here so six independently written readers raise one type. T-D1 must
    distinguish "this body was unreadable" — T-C6 contributes a malformed entry
    to the corpus — from "this is an I1 breach", and cannot do that against six
    different exception types or a bare ``Exception``.

    **Three failure shapes, deliberately distinct**, so a caller can tell them
    apart:

    - ``UnreadableBodyError`` — *the body* is wrong. The reader met something
      it cannot read, such as malformed JSON or a role no format defines.
    - ``ValueError`` — *the caller* is wrong. A projection type was constructed
      with a bad role or a non-canonical sampling key. Raised from inside a
      reader, it means the reader mis-routed a field: a reader bug, not a
      transport one and not an I1 breach.
    - :class:`ProjectionTotalityError` — the reader *read* the body but did not
      account for all of it.
    """


class ProjectionTotalityError(AssertionError):
    """A reader failed to account for the body it read.

    Derives from ``AssertionError`` because it reports a failed check rather
    than a broken program: it reads as a test failure, not as a crash.
    """


class DroppedFieldsError(ProjectionTotalityError):
    """A body key the reader neither mapped nor residualised.

    The dangerous case, and the one a residual-only rule cannot see.
    """


class ResidualFieldsError(ProjectionTotalityError):
    """The reader could not classify something, and said so.

    §3.3.1: a non-empty residual fails the run — it is not reported as a diff
    and it is not ignored, because an unaccounted field is precisely where an
    unregistered mutation hides.
    """


def verify_total(projected: Request | Reply) -> None:
    """Assert that a reader accounted for every key in the body it read.

    Two distinct failures, reported in order of danger:

    1. a **dropped** key — in neither ``consumed`` nor ``residual``. The reader
       silently ignored it. This is what :attr:`Request.consumed` exists to
       catch: a dropping reader leaves the residual *empty*, so a
       "residual must be empty" rule would pass it.
    2. a **residual** — the reader could not classify it and said so.

    Args:
        projected: A :class:`Request` or a :class:`Reply`. Both carry
            ``source``, ``consumed`` and ``residual``.

    Raises:
        DroppedFieldsError: When a top-level key of ``projected.source`` appears
            in neither account. Any key claimed in ``consumed`` but absent from
            the body is named in the same message, so a typo'd claim is not
            mis-diagnosed as a drop the reader caused.
        ResidualFieldsError: When ``residual`` is non-empty.
    """
    # A key claimed in both accounts counts as a residual, so it is excluded
    # from the drop check here and caught below (R2.8).
    accounted = set(projected.consumed) | set(projected.residual)
    dropped = sorted(set(projected.source) - accounted)

    if dropped:
        claimed_absent = sorted(set(projected.consumed) - set(projected.source))
        detail = f"; claimed but absent: {claimed_absent}" if claimed_absent else ""
        raise DroppedFieldsError(
            f"reader dropped {dropped} — every body key must map to the envelope, "
            f"the conversation, or the residual{detail}"
        )

    if projected.residual:
        raise ResidualFieldsError(
            f"reader could not classify {sorted(projected.residual)} — a non-empty residual fails the run"
        )


# --------------------------------------------------------------------------
# The projection protocols
# --------------------------------------------------------------------------


@runtime_checkable
class Projection(Protocol):
    """Reads one wire format's request into the common form.

    Implemented by T-A1–T-A6, one reader per format, each written against that
    format's published examples and importing nothing from ``src/kitty``.

    ``isinstance`` works against this protocol; ``issubclass`` raises, because
    of the ``wire_format`` data member. ``isinstance`` also checks member
    *presence* only, never signatures.
    """

    wire_format: WireFormat

    def read_request(self, captured: CapturedRequest) -> Request:
        """Project a captured request.

        Takes the **whole** capture, not a body: Gemini carries the model and
        the operation in the URL path (§3.3.5), so a body-only reader could not
        project a Gemini request at all.

        Args:
            captured: The request as observed on the wire.

        Returns:
            The wire-independent projection.

        Raises:
            UnreadableBodyError: When the body cannot be read.
        """
        ...


@runtime_checkable
class ReplyProjection(Protocol):
    """Reads one wire format's reply into the common form.

    Separate from :class:`Projection` because §3.3.1 makes response translation
    "a different claim [that] gets a different test", and because the plan
    splits the work as six request tasks against one reply task (T-A7) with its
    own comparison task (T-D10).
    """

    wire_format: WireFormat

    def read_reply(self, captured: CapturedReply) -> Reply:
        """Project a captured reply.

        Args:
            captured: The reply as observed on the wire, already reassembled
                from SSE if it was streamed — reassembly is T-A7's boundary.

        Returns:
            The wire-independent projection.

        Raises:
            UnreadableBodyError: When the body cannot be read.
        """
        ...


# --------------------------------------------------------------------------
# The path vocabulary
# --------------------------------------------------------------------------

#: Anchors for the fields the register addresses by name (§3.3.1). Spelled once
#: here so T-W3's rows and T-D1's deltas cannot drift apart on spelling.
ENVELOPE_MODEL = "envelope.model"
ENVELOPE_STREAM = "envelope.stream"
ENVELOPE_STORE = "envelope.store"

#: Bare-collection anchors, for rows that change a collection as a whole: M5 and
#: M13 rewrite the turns, P5b joins the system blocks, and §3.3.1 pins P13 and
#: P14 to ``conversation.sampling`` rather than to one key.
CONVERSATION_SYSTEM = "conversation.system"
CONVERSATION_TURNS = "conversation.turns"
CONVERSATION_TOOLS = "conversation.tools"
CONVERSATION_SAMPLING = "conversation.sampling"

#: Response-direction anchors, for M12 and T-D10.
REPLY_STOP_REASON = "reply.stop_reason"
REPLY_USAGE = "reply.usage"

#: Route anchors (§3.3.5). M14, P20 and P21 change where a request goes, which
#: the body cannot show.
ROUTE_METHOD = "route.method"
ROUTE_SCHEME = "route.scheme"
ROUTE_HOST = "route.host"
ROUTE_PATH = "route.path"
ROUTE_QUERY = "route.query"

#: The route components a path may name, and the fields of
#: :class:`CapturedRequest` they correspond to.
ROUTE_COMPONENTS = frozenset({"method", "scheme", "host", "path", "query"})

#: The canonical `tool_choice` values.  Four formats spell one concept four ways
#: — Chat Completions and Messages `tool_choice`, Converse's
#: `toolConfig.toolChoice`, Gemini's `functionCallingConfig.mode` — so a
#: Messages-to-CC comparison would otherwise show an unclaimed delta on every
#: request that declares tools.  A specific tool is named `tool:<name>`.
TOOL_CHOICE_VALUES = frozenset({"auto", "any", "none"})

#: The key `tool_choice` normalises onto.  The single deliberate exception to
#: keying :attr:`Envelope.extra` by the wire key, because the four wire keys
#: name one concept and Gemini's is `toolConfig`.
TOOL_CHOICE_KEY = "tool_choice"

#: The value a register row carries when the projection deliberately does not
#: model its effect. It **requires a reason**. P16 uses it (the content-type tag
#: is redundant with the turn's role), as do the whole-body protocol
#: translations M2, M9, P11 and P12, which change everything and so name nothing
#: usefully. An empty cell would make those rows silently unfalsifiable.
NOT_PROJECTABLE = "not projectable"

#: The wildcard segment. §3.3.1 writes register fields with an empty index —
#: "P15 is ``conversation.tools[].strict``" — meaning every tool.
WILDCARD = "*"


def extra_path(key: str) -> str:
    """Return the path naming a format-specific control field.

    Args:
        key: The wire key, e.g. ``thinking`` for P2a.

    Returns:
        A path of the form ``envelope.extra[<key>]``.
    """
    return f"envelope.extra[{key}]"


def sampling_path(key: str) -> str:
    """Return the path naming one sampling parameter.

    Args:
        key: A member of :data:`SAMPLING_KEYS`.

    Returns:
        A path of the form ``conversation.sampling[<key>]``.
    """
    return f"conversation.sampling[{key}]"


def system_path(index: int) -> str:
    """Return the path naming one system text part.

    Args:
        index: Position in :attr:`Conversation.system`.

    Returns:
        A path of the form ``conversation.system[<i>]``.
    """
    return f"conversation.system[{index}]"


def turn_path(index: int, field_name: str | None = None) -> str:
    """Return the path naming one turn, or a field of it.

    Args:
        index: Position in :attr:`Conversation.turns`.
        field_name: An optional field, e.g. ``role``.

    Returns:
        A path of the form ``conversation.turns[<i>]``, with ``.<field>``
        appended when one is given.
    """
    base = f"conversation.turns[{index}]"
    return f"{base}.{field_name}" if field_name else base


def part_path(turn_index: int, part_index: int) -> str:
    """Return the path naming one part of one turn.

    §3.3.4 requires a failure to name the exact turn and part.

    Args:
        turn_index: Position in :attr:`Conversation.turns`.
        part_index: Position in that turn's parts.

    Returns:
        A path of the form ``conversation.turns[<i>].parts[<j>]``.
    """
    return f"conversation.turns[{turn_index}].parts[{part_index}]"


def tool_path(name: str, field_name: str | None = None) -> str:
    """Return the path naming one tool declaration, or a field of it.

    Tools are addressed **by name, not index**: translators reorder and filter
    declarations, so a positional path would report a delta whenever the list
    order changed while nothing about the declaration did.

    Args:
        name: The tool's name.
        field_name: An optional field, e.g. ``strict`` for P15.

    Returns:
        A path of the form ``conversation.tools[<name>]``, with ``.<field>``
        appended when one is given.
    """
    base = f"conversation.tools[{name}]"
    return f"{base}.{field_name}" if field_name else base


def header_path(name: str) -> str:
    """Return the path naming one request header.

    P9a, P9b and P9c change headers rather than the body, and §4.3 C1 asserts on
    the exact header set.

    Args:
        name: The header name; lowercased for addressing, though the capture
            itself preserves the original casing.

    Returns:
        A path of the form ``headers[<name>]``.
    """
    return f"headers[{name.lower()}]"


def residual_path(key: str) -> str:
    """Return the path naming one unclassified value.

    Args:
        key: The path into the body, which may itself contain dots — bracket
            contents are literal and are never re-parsed.

    Returns:
        A path of the form ``residual[<key>]``.
    """
    return f"residual[{key}]"


def reply_part_path(index: int) -> str:
    """Return the path naming one part of a reply.

    Args:
        index: Position in :attr:`Reply.parts`.

    Returns:
        A path of the form ``reply.parts[<i>]``.
    """
    return f"reply.parts[{index}]"


def reply_usage_path(key: str) -> str:
    """Return the path naming one usage counter.

    Usage is carried but **excluded from the fidelity diff** — it is
    provider-reported and never agent-supplied. The path form exists so T-D10
    can name a counter without hand-assembling a string.

    Args:
        key: The usage key, e.g. ``input_tokens``.

    Returns:
        A path of the form ``reply.usage[<key>]``.
    """
    return f"reply.usage[{key}]"


def route_path(component: str) -> str:
    """Return the path naming one route component.

    §3.3.5: M14 replaces the destination entirely, P20 encodes the model as an
    Azure deployment id in the path, and P21 puts the Vertex project and
    location in the base URL. None of that is visible in the body.

    Args:
        component: One of ``method``, ``scheme``, ``host``, ``path``, ``query``.

    Returns:
        A path of the form ``route.<component>``.

    Raises:
        ValueError: When ``component`` is not a route component.
    """
    if component not in ROUTE_COMPONENTS:
        raise ValueError(f"route component must be one of {sorted(ROUTE_COMPONENTS)}, got {component!r}")
    return f"route.{component}"


def _segments(path: str) -> list[str]:
    """Split a path into segments, treating bracket contents as literal.

    A key may contain dots — ``residual[generationConfig.topK]`` — so a naive
    ``split(".")`` would shatter it. Brackets are tracked by depth and their
    contents are never re-parsed.

    Args:
        path: A concrete path or a pattern.

    Returns:
        The path's segments, in order.

    Raises:
        ValueError: When brackets are unbalanced. R7.2d calls such a path "not
            addressable"; raising makes that *visible*. Letting the depth drift
            would silently glue the rest of the path into one segment, and
            :func:`path_matches` would then return a confident wrong answer.
    """
    segments: list[str] = []
    current: list[str] = []
    depth = 0

    for char in path:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced ']' in path {path!r}")
        elif char == "." and depth == 0:
            segments.append("".join(current))
            current = []
            continue
        current.append(char)

    if depth != 0:
        raise ValueError(f"unclosed '[' in path {path!r}")

    segments.append("".join(current))
    return segments


def _segment_matches(pattern: str, concrete: str) -> bool:
    """Return whether one pattern segment names one concrete segment.

    Args:
        pattern: A segment which may carry the ``[*]`` wildcard.
        concrete: The corresponding concrete segment.

    Returns:
        ``True`` when the pattern names the segment.
    """
    if pattern == concrete:
        return True

    prefix, sep, rest = pattern.partition("[")
    if not sep or not rest.endswith("]"):
        return False

    # `[]` is accepted as a wildcard because §3.3.1 spelled register fields that
    # way ("P15 is conversation.tools[].strict") before this vocabulary existed.
    # A row carried over in the old notation must not silently match nothing.
    if rest[:-1] not in (WILDCARD, ""):
        return False

    c_prefix, c_sep, c_rest = concrete.partition("[")
    return bool(c_sep) and c_prefix == prefix and c_rest.endswith("]")


def path_matches(pattern: str, concrete: str) -> bool:
    """Return whether a register row's pattern names a concrete delta path.

    §3.3.2 assertion 1 — "every difference must map to a register row" — is
    literally this match. Defining it here, rather than letting T-W3 write the
    patterns and T-D1 write the matcher, is what stops the two agreeing only by
    luck.

    **A pattern is a prefix.** It names the location it points at and everything
    beneath it, at any depth. So P13's anchor ``conversation.sampling`` claims a
    delta on any key under it, and a row anchored at
    ``conversation.turns[*].parts[*]`` claims
    ``conversation.turns[2].parts[0].signature`` — which M8's carrier repair
    produces. Restricting the prefix rule to bracket-free patterns would make
    that last case unclaimed, and under §3.3.2 assertion 1 an unclaimed delta
    fails the run: a false I1 breach manufactured by the matcher itself.

    Over-claiming in the other direction costs nothing, because a register row
    is anchored at the coarsest node it affects and every path beneath that node
    is, by construction, part of what the row changed.

    Args:
        pattern: A path which may carry ``[*]`` wildcards.
        concrete: A concrete path, as a delta reports it.

    Returns:
        ``True`` when the pattern names the path.

    Raises:
        ValueError: When either path has unbalanced brackets.
    """
    pattern_segments = _segments(pattern)
    concrete_segments = _segments(concrete)

    # A pattern may be shorter than the path it claims, never longer.
    if len(pattern_segments) > len(concrete_segments):
        return False

    for index, pattern_segment in enumerate(pattern_segments):
        concrete_segment = concrete_segments[index]

        # A bare collection name claims a bracketed member of itself, so
        # `sampling` claims `sampling[temperature]`.
        if "[" not in pattern_segment and concrete_segment.startswith(f"{pattern_segment}["):
            continue
        if not _segment_matches(pattern_segment, concrete_segment):
            return False

    return True

