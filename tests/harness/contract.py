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
published examples.  :func:`decode_arguments` is the edge case that proves
the line rather than crossing it: it reads one *value* whose decoding six
readers must agree on, and knows nothing of where in a body it was found.
It captures nothing — the recorders do that.  It asserts nothing about kitty —
the oracle does that.

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
import json
import re
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


def _freeze_optional(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Return a frozen view over ``value``, preserving its absence.

    The counterpart of :func:`_freeze_mapping` for a field whose *absence* is
    meaningful.  ``cache_control`` is the case: ``None`` is "no cache breakpoint"
    and ``{}`` is a malformed one, and collapsing the two would stop **M16**
    distinguishing a stripped breakpoint from a block that never carried one.

    Args:
        value: The mapping to freeze, or ``None`` when the field is absent.

    Returns:
        A read-only mapping proxy over a shallow copy, or ``None``.
    """
    return None if value is None else _freeze_mapping(value)


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
        cache_control: The cache breakpoint the agent set on this block, as the
            wire mapping — ``{"type": "ephemeral"}``, or the extended form
            carrying a ``ttl``. ``None`` means no breakpoint. Carried whole
            rather than as a boolean because a one-hour write and a five-minute
            one are different prices, so a flattened form would hide a silently
            downgraded lifetime. **M16** claims its removal.
        video_metadata: Gemini's ``Part.videoMetadata`` — a modifier on the
            part rather than content of its own, carried whole as the wire
            mapping (``{"fps": …, "startOffset": …}``) so a changed value is
            a delta rather than a silent equivalence (KBR-194). ``None`` when
            absent. Lives on :class:`Text` and :class:`Image`, the two parts
            Gemini attaches video to; other part types still residualise it.
    """

    text: str
    cache_control: Mapping[str, Any] | None = None
    video_metadata: Mapping[str, Any] | None = None

    # Every projection type sets this, so unhashability is total rather than
    # data-dependent — see the module note on multiset matching. It survives
    # `@dataclass` only because none of these classes defines `__eq__` in its
    # own body; adding one would silently restore a working `__hash__`.
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Freeze the cache breakpoint and the video metadata in place."""
        object.__setattr__(self, "cache_control", _freeze_optional(self.cache_control))
        object.__setattr__(self, "video_metadata", _freeze_optional(self.video_metadata))


@dataclass(frozen=True)
class ToolUse:
    """A tool call the assistant made.

    Attributes:
        id: The call id, or ``None`` where the format carries none — a
            required id would force such a reader to synthesise one and show a
            delta on every tool turn. **Not Gemini**, contrary to this
            docstring's original wording: `v1beta`'s ``FunctionCall`` publishes
            an optional ``id`` and ``FunctionResponse`` an optional ``id`` the
            client populates to match it (discovery document, revision
            ``20260910``; corrected by T-A4/KBR-36, which was asked to confirm
            rather than assume it). Optional either way, so the decision this
            sentence justifies is unchanged.
        name: The tool's name.
        arguments: The parsed arguments. Chat Completions encodes these as a
            JSON *string* and Messages as an object; normalising here stops a
            spurious delta on every cross-format comparison.
        cache_control: The cache breakpoint the agent set on this block, as the
            wire mapping — ``{"type": "ephemeral"}``, or the extended form
            carrying a ``ttl``. ``None`` means no breakpoint. Carried whole
            rather than as a boolean because a one-hour write and a five-minute
            one are different prices, so a flattened form would hide a silently
            downgraded lifetime. **M16** claims its removal.
        signature: The vendor's thinking signature on this call — Gemini
            attaches a ``thoughtSignature`` to the ``functionCall`` part
            itself (not only to thought parts) and requires clients to echo it
            back verbatim on the next turn, returning 4xx when omitted
            (KBR-194). The analogue of :attr:`Thinking.signature`, which
            carries the same wire field on a thought part.
    """

    name: str
    arguments: Mapping[str, Any] = _frozen_field()
    id: str | None = None
    cache_control: Mapping[str, Any] | None = None
    signature: str | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Freeze the arguments mapping and the cache breakpoint in place."""
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))
        object.__setattr__(self, "cache_control", _freeze_optional(self.cache_control))


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
        digest: A content digest.  Two recipes, and §7.4.1 states which applies:
            :func:`opaque_digest` for a block the grammar cannot model,
            :func:`text_digest` for content whose identity is a run of text.
            ``cache_control`` is excluded from **both**, so a stripped breakpoint
            on an unmodelled block shows at its own path rather than as an
            unexplained digest change no register row could name.
        cache_control: The cache breakpoint the agent set on this block, as the
            wire mapping — ``{"type": "ephemeral"}``, or the extended form
            carrying a ``ttl``. ``None`` means no breakpoint. Carried whole
            rather than as a boolean because a one-hour write and a five-minute
            one are different prices, so a flattened form would hide a silently
            downgraded lifetime. **M16** claims its removal.
    """

    kind: str
    digest: str | None = None
    cache_control: Mapping[str, Any] | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Validate the kind against the canonical vocabulary, and freeze the breakpoint.

        Checked rather than merely declared, for the reason this module applies
        to every other closed vocabulary: *a vocabulary declared closed but
        checked nowhere is a comment, not a rule*.  The docstring above has
        required a canonical name since T-W2 and named none, so the two readers
        written against it arrived at ``file`` and ``document`` for one concept
        (KBR-174).

        Raises:
            TypeError: When ``kind`` is not a ``str`` — a wrong *type*, which is
                this module's ``TypeError`` case throughout.
            ValueError: When ``kind`` is a wire spelling :data:`OPAQUE_ALIASES`
                reconciles, or is not snake_case.  Construction is not parsing —
                T-D1 builds expected conversations by hand — so this is a
                ``ValueError`` and not :class:`UnreadableBodyError`, exactly as
                for :attr:`Turn.role`.  A reader meeting such a type on the wire
                translates it (see :func:`opaque_kind`); a reader *constructing*
                one has a bug.
        """
        if not isinstance(self.kind, str):
            raise TypeError(f"Opaque.kind must be a str, got {type(self.kind).__name__}")

        canonical = OPAQUE_ALIASES.get(self.kind)
        if canonical is not None:
            raise ValueError(
                f"Opaque.kind {self.kind!r} is a wire spelling; the canonical name is {canonical!r} "
                "(§7.4.1). Readers translate through opaque_kind()."
            )

        if not _SNAKE_CASE.fullmatch(self.kind):
            raise ValueError(
                f"Opaque.kind must be canonical snake_case, got {self.kind!r} (§7.4.1). "
                "A camelCase wire type needs a canonical name in OPAQUE_ALIASES first."
            )

        object.__setattr__(self, "cache_control", _freeze_optional(self.cache_control))


@dataclass(frozen=True)
class Image:
    """An image, identified by digest rather than carried as bytes.

    **The pairing is enforced, both ways.** ``digest`` is ``None`` iff ``ref``
    is not ``None``: neither set projects every image identically (KBR-179's
    blindness for ``Opaque``); both set lets two readers populate the pair
    differently for one image and report a phantom delta on content neither
    altered. Closed means enforced — the same posture :class:`Turn` takes on
    roles and :class:`Conversation` on sampling keys, and the analogue of
    :class:`Reply`'s "``stop_reason_raw`` is only for ``'other'``".

    Attributes:
        digest: Lowercase hex SHA-256 of the *decoded* image bytes, or of
            the *raw encoded* bytes when the wire payload cannot be decoded
            (see :func:`image_digest` and §7.4 rule 7 row 3 — KBR-192).
            ``media_type`` is deliberately **not** part of the digest, so a
            changed media type is its own delta rather than an unexplained
            digest change.
        media_type: The declared media type, when the format states one.
        ref: The URI, for Gemini's ``fileData.fileUri`` which carries no bytes.
        display_name: Gemini's ``Blob.displayName`` / ``FileData.displayName``
            — the name of the blob/file to the model for ``REFERENCE_ONLY``
            verbalisation. ``None`` when absent (KBR-194).
        video_metadata: Gemini's ``Part.videoMetadata`` on an
            ``inlineData`` or ``fileData`` part — a modifier rather than
            content, carried whole as the wire mapping. ``None`` when
            absent. Real traffic puts it on Text and Image parts only.
        cache_control: The cache breakpoint the agent set on this block, as the
            wire mapping — ``{"type": "ephemeral"}``, or the extended form
            carrying a ``ttl``. ``None`` means no breakpoint. Carried whole
            rather than as a boolean because a one-hour write and a five-minute
            one are different prices, so a flattened form would hide a silently
            downgraded lifetime. **M16** claims its removal.
    """

    digest: str | None = None
    media_type: str | None = None
    ref: str | None = None
    display_name: str | None = None
    video_metadata: Mapping[str, Any] | None = None
    cache_control: Mapping[str, Any] | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Enforce the digest/ref XOR and freeze the cache breakpoint and the video metadata.

        Raises:
            ValueError: When ``digest`` and ``ref`` are both ``None`` or both
                set. A vocabulary or pairing declared but checked nowhere is a
                comment, not a rule — the same posture this module takes on
                every other closed invariant.
        """
        # Both-None projects every image identically (KBR-179's blindness for
        # Opaque); both-set lets two readers populate the pair differently and
        # report a phantom delta on content neither altered.
        if (self.digest is None) == (self.ref is None):
            raise ValueError(
                "Image.digest and Image.ref must be set together — exactly one "
                "of them must be None. Both-None is the blindness KBR-179 names "
                "for Opaque; both-set lets two readers populate the pair "
                f"differently for one image (got digest={self.digest!r}, ref={self.ref!r})."
            )

        object.__setattr__(self, "cache_control", _freeze_optional(self.cache_control))
        object.__setattr__(self, "video_metadata", _freeze_optional(self.video_metadata))


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
        scheduling: Gemini's ``functionResponse.scheduling`` value — the
            NON_BLOCKING calling toggle's response-side value, ``"SILENT"`` or
            ``"INTERRUPT"`` in the published enum, ``None`` for the default
            (KBR-194). The declaration's :attr:`ToolDecl.behavior` carries
            the same feature on the call side; ``willContinue`` rides on
            Gemini's wire shape unchanged because no current route populates
            it.
        cache_control: The cache breakpoint the agent set on this block, as the
            wire mapping — ``{"type": "ephemeral"}``, or the extended form
            carrying a ``ttl``. ``None`` means no breakpoint. Carried whole
            rather than as a boolean because a one-hour write and a five-minute
            one are different prices, so a flattened form would hide a silently
            downgraded lifetime. **M16** claims its removal.
    """

    content: Sequence[Text | Image | Json | Opaque] = ()
    tool_use_id: str | None = None
    is_error: bool = False
    scheduling: str | None = None
    cache_control: Mapping[str, Any] | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Freeze the content sequence and the cache breakpoint in place."""
        object.__setattr__(
            self, "content", _checked_members(self.content, RESULT_PART_TYPES, "ToolResult.content")
        )
        object.__setattr__(self, "cache_control", _freeze_optional(self.cache_control))


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
    """Return the canonical digest of image bytes.

    Pinned so that six independently written readers agree.  Anthropic sends
    base64 plus a media type, Chat Completions a data URL, Converse raw bytes
    and Gemini either inline data or a URI — without one definition the Messages
    reader and the Chat Completions reader would produce different digests for
    the same image and §7.1's image corpus entry would fail on every run.

    Two recipes are carried by the one function, distinguished by what the
    caller passes rather than by the algorithm:

    * **Decoded image bytes** — the canonical case, when the base64 payload
      decoded cleanly.
    * **Raw encoded bytes** — when the payload cannot be decoded, the reader
      digests the wire's own bytes (e.g. ``raw.encode("utf-8")`` of the wrapped
      base64 string) so the part keeps its identity and its position
      (§7.4 rule 7 row 3; KBR-192). Two differently-wrapped blobs of one payload
      digest differently — the compromise is pinned by
      ``TestImageDigestRecipe``.

    Args:
        raw: The image bytes to digest — decoded for the canonical case, raw
            encoded bytes when the payload cannot be decoded. The media type is
            deliberately excluded, so a changed media type shows as its own
            delta.

    Returns:
        Lowercase hex SHA-256 of ``raw``.
    """
    return hashlib.sha256(raw).hexdigest()


def text_digest(text: str) -> str:
    """Return the canonical digest of content identified by a run of text.

    The second of :attr:`Opaque.digest`'s two recipes.  A refusal is the case
    that needs it: Responses carries one as a typed content part, Chat
    Completions as a **bare string** on the message
    (``ChatCompletionResponseMessage.refusal`` is ``anyOf[string, null]``), so
    there is no block for :func:`opaque_digest` to hash.  Pinning one recipe for
    both would force the Chat Completions reader to invent a block wrapper, and
    the wrapper's shape would be a new thing six readers could disagree about.

    The digest is what makes a *rewritten* refusal visible rather than only a
    removed one — :attr:`Opaque.kind` alone would project every refusal alike.

    Args:
        text: The text whose identity the digest carries.

    Returns:
        Lowercase hex SHA-256 of ``text`` encoded as UTF-8.  The encoding is
        pinned because ``latin-1`` raises and ``cp1252`` silently differs on the
        same content.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def opaque_digest(block: Mapping[str, Any]) -> str:
    """Return the canonical digest of an unmodelled block's payload.

    The first of :attr:`Opaque.digest`'s two recipes, for a block type the
    grammar does not model.  Pinned here for the reason :func:`image_digest` is:
    six independently written readers must produce one digest for one block, and
    prose did not achieve it — in T-A1 **all three** wrong spellings of this
    recipe survived mutation testing, because every assertion compared two
    projections against each other and a recipe change that stays internally
    consistent is invisible to that.

    Three details are load-bearing:

    * **Canonical JSON, not the raw wire slice** — a translator that reorders
      keys must not change the digest, which is the whole reason this is not
      ``sha256(raw_block_bytes)``.
    * ``ensure_ascii`` **pinned** — its default is ``True`` while the surrounding
      prose says UTF-8, so an author "fixing" it to ``False`` gets a different
      digest for the same block, visible only on non-ASCII content.
    * ``cache_control`` **excluded** — otherwise one field behaves two ways: a
      diagnosed residual on a modelled block, an unexplained digest change on an
      unmodelled one (KBR-167).

    Args:
        block: The block, whose ``type`` is excluded because it is already
            :attr:`Opaque.kind`, and whose ``cache_control`` is excluded because
            it residualises under its own path instead.

    Returns:
        Lowercase hex SHA-256 of the canonical payload.
    """
    payload = {key: value for key, value in block.items() if key not in ("type", "cache_control")}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


#: What a canonical :attr:`Opaque.kind` may look like.  Four rules turn on this
#: predicate, so it is spelled once rather than left to each author's intuition:
#: lowercase segments of letters and digits, joined by single underscores, never
#: leading with a digit.  Accepts every kind the shipped readers emit; rejects
#: every camelCase member of Converse's ``ContentBlock`` union.
_SNAKE_CASE = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)*")

#: Wire spellings that name a concept another format already names differently,
#: mapped to the canonical :attr:`Opaque.kind`.  **Enforced**, not merely
#: declared: :class:`Opaque` rejects any key of this table, so a reader cannot
#: quietly project a rival name.
#:
#: An attached file is the concept that forced the table early.  §7.4.1 deferred
#: it to T-A5 "with the first reader that needs one", but T-A1 and T-A3 landed
#: first and named one thing two ways — ``document`` and ``file`` — which is the
#: permanent unclaimed delta the deferral was meant to avoid.  ``document`` wins
#: because Anthropic Messages *and* Bedrock Converse both spell it that way on
#: the wire, so exactly one reader moved (KBR-174).
#:
#: **The canonical kinds in use today, per format — NOT a closed set**, and not
#: a checked one either.  It cannot be closed: a format-unique type keeps its own
#: wire spelling by the stated exception.  So this list is *documentation* for
#: the next reader's author, and it is the one thing here that is not enforced —
#: only the Responses row has a test that derives its members from the reader's
#: own frozensets rather than restating them.  Treat it as a starting point to
#: check against the readers, never as an authority.
#:
#: * shared across formats — ``document``, ``search_result``
#: * Anthropic Messages — ``redacted_thinking``, ``server_tool_use``
#: * OpenAI Responses — ``refusal``, plus its 27 format-unique item types
#:   (``web_search_call``, ``mcp_call``, ``apply_patch_call`` …)
OPAQUE_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        # OpenAI Responses' attachment, and the name T-A3 first shipped for it.
        "input_file": "document",
        "file": "document",
        # Bedrock Converse, whose union is camelCase throughout (T-A5).  Of the
        # nine camelCase ContentBlock members, three project to first-class
        # parts (`toolUse` → ToolUse, `toolResult` → ToolResult,
        # `reasoningContent` → Thinking) and three are snake_case already
        # (`document`, `video`, `audio`); the Tool-union toggles (`cachePoint`,
        # `systemTool` on a `Tool` entry) never become Opaque at all — they are
        # envelope extras (§7.4.2 rule 5).  These six reconcile the spelling:
        "searchResult": "search_result",
        "cachePoint": "cache_point",
        "guardContent": "guard_content",
        "citationsContent": "citations_content",
        "toolAddition": "tool_addition",
        "toolRemoval": "tool_removal",
        # `redactedContent` is the nested key inside a Converse
        # `reasoningContent` block, not a union member: it is the redacted
        # branch of the same concept Anthropic spells `redacted_thinking`.
        "redactedContent": "redacted_thinking",
    }
)


def opaque_kind(wire_type: str) -> str:
    """Return the canonical :attr:`Opaque.kind` for a wire block type.

    A format-unique type passes through unchanged: it names a concept no other
    format has, so there is no second spelling to reconcile and the wire's own
    word is already canonical.  That is the deliberate exception to
    :attr:`Opaque.kind`'s "never the wire's spelling".

    **A non-snake_case type raises rather than being converted.**  Converting
    would mean shipping a camelCase splitter with no caller, no corpus and no
    test — and a splitter whose edge cases nobody has exercised is a second
    source of drift, not a cure for one.  Raising puts the next author at this
    module at the moment they meet the first such type, with a real corpus in
    hand.  That is **nine** types for Bedrock Converse, not one: its
    ``ContentBlock`` union carries ``toolUse``, ``toolResult``, ``guardContent``,
    ``cachePoint``, ``reasoningContent``, ``citationsContent``, ``searchResult``,
    ``toolAddition`` and ``toolRemoval``.  Of those nine, only six need an
    alias — the other three project to first-class parts via the reader's
    dispatch and never reach :func:`opaque_kind`.  See :data:`OPAQUE_ALIASES`'s
    inline comment for the split.

    Args:
        wire_type: The block or item type as the format spells it.

    Returns:
        The canonical name — the alias when :data:`OPAQUE_ALIASES` reconciles the
        spelling, and ``wire_type`` itself otherwise.

    Raises:
        TypeError: When ``wire_type`` is not a ``str`` — this module's split
            between a wrong *type* and a wrong *value*, matching
            :class:`Opaque`'s own check so the two cannot disagree.
        ValueError: When no alias exists and ``wire_type`` is not snake_case, so
            no canonical name can be derived without a decision.  A reader
            meeting this on the wire translates it into
            :class:`UnreadableBodyError`: the body is not projectable, but the
            reader is not at fault.
    """
    # Checked before the lookup, not after: an unhashable value raises
    # `TypeError: unhashable type` from `.get()` itself, which a reader's
    # `except ValueError` cannot catch, so it would escape as a raw crash.
    if not isinstance(wire_type, str):
        raise TypeError(f"opaque_kind() wire_type must be a str, got {type(wire_type).__name__}")

    canonical = OPAQUE_ALIASES.get(wire_type)
    if canonical is not None:
        return canonical

    # Not an alias and not snake_case means nobody has decided what this is
    # called; guessing here is how two readers end up with two names.
    if not _SNAKE_CASE.fullmatch(wire_type):
        raise ValueError(
            f"no canonical Opaque.kind for wire type {wire_type!r} (§7.4.1): it is neither snake_case "
            "nor listed in OPAQUE_ALIASES. Add it to OPAQUE_ALIASES with its canonical name."
        )

    return wire_type


def decode_arguments(raw: Any, path: str, residual: dict[str, Any]) -> Mapping[str, Any]:
    """Decode a tool call's JSON-string arguments, failing closed into the residual.

    Chat Completions and Responses both encode arguments as a JSON *string*
    while :attr:`ToolUse.arguments` is a mapping, so both readers decode and
    §3.3.1b requires them to decode alike.  Pinned here (KBR-174) because the two
    sit on opposite sides of register rows P13–P17: a disagreement on the
    malformed path would surface as an unclaimed delta at
    ``conversation.turns[*].parts[*]`` that no row claims, which §3.3.1a calls
    the unrecoverable direction.

    **Only an absent or blank value is silently empty.**  An absent ``arguments``
    honestly means *no arguments*, which is why it does not residualise the way
    an absent tool ``name`` does even though the schema requires both.  The test
    generalises to every required field: *can the projection represent the
    absence losslessly?*  ``{}`` is a true statement about a call — seen and
    classified, the same ground on which :data:`STOP_REASONS` gives ``other`` its
    escape instead of the residual.  ``""`` for a name is not: it claims a tool
    *named* empty-string, and a call nobody can name cannot be paired with its
    result or addressed by a register row.

    The blank form is real corpus traffic, not a hypothesis — kitty's own
    ``openai_subscription.py`` writes ``func.get("arguments", "")``.

    **An already-decoded object is rejected, not accepted.**  Taking it would
    make a bridge that emitted the object form where the schema demands a string
    invisible to the oracle, which is the wire-format breach the readers exist to
    see.

    **Never raises on a value the wire can carry.**  This module defines a
    reader-raised ``ValueError`` as "the reader mis-routed a field — a reader
    bug", so failing closed into the residual keeps the diagnosis honest.  The
    qualifier is literal: ``json.loads`` still raises ``RecursionError`` on
    pathologically nested input, which every body-level parse in the harness
    shares and no corpus produces.

    Args:
        raw: The wire value, of any type.
        path: The residual key for this argument, which is **format-specific and
            stays the caller's** — ``input[<i>].arguments`` for Responses,
            ``messages[<i>].tool_calls[<j>].function.arguments`` for Chat
            Completions.
        residual: The reader's accumulator of unclassifiable values, mutated
            here.  Taken as a parameter rather than reported through a return
            flag so the fail-closed step cannot be skipped: ``mypy`` covers
            ``src/kitty`` only, so a caller that ignored such a flag would be
            caught by nothing.

    Returns:
        The decoded arguments, empty whenever the value did not carry a JSON
        object.
    """
    # Absent is a true empty, and the only non-string that is.
    if raw is None:
        return {}

    # Everything else that is not a string is a shape the schema forbids,
    # including the decoded object form.
    if not isinstance(raw, str):
        residual[path] = raw
        return {}

    if not raw.strip():
        return {}

    try:
        decoded = json.loads(raw)
    # `ValueError` alone: `JSONDecodeError` subclasses it, and naming both reads
    # as though two distinct cases existed.
    except ValueError:
        residual[path] = raw
        return {}

    # The raw string, never `decoded`: a residual key set is diffed across all
    # six readers (T-D8), so storing two different renderings of one unreadable
    # value would report a delta neither reader caused.
    if not isinstance(decoded, dict):
        residual[path] = raw
        return {}

    return decoded


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
        behavior: Gemini's published ``behavior`` — the NON_BLOCKING calling
            toggle's per-declaration value, ``"NON_BLOCKING"`` to defer or
            ``None`` for the default BLOCKING behaviour (KBR-194). Lives at
            declaration scope; ``functionResponse.scheduling`` rides
            :class:`ToolResult` separately for the response half.
        cache_control: The cache breakpoint the agent set on this block, as the
            wire mapping — ``{"type": "ephemeral"}``, or the extended form
            carrying a ``ttl``. ``None`` means no breakpoint. Carried whole
            rather than as a boolean because a one-hour write and a five-minute
            one are different prices, so a flattened form would hide a silently
            downgraded lifetime. **M16** claims its removal.
        type: The tool's discriminator, as the wire spells it — ``"custom"`` on
            an ordinary client tool, a dated vendor spelling such as
            ``web_search_20250305`` on a server tool (Anthropic Messages), or
            ``None`` where the format defines none. Carried rather than
            residualised because a mutation on it changes what a forced call is
            asking for (KBR-214 D10) and because G35's register row anchors its
            conditional on it (KBR-205). Addressed at
            ``conversation.tools[<name>].type``, by name, like every other
            tool leaf.
    """

    name: str
    description: str | None = None
    schema: Mapping[str, Any] | None = None
    strict: bool | None = None
    behavior: str | None = None
    type: str | None = None
    cache_control: Mapping[str, Any] | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Freeze the schema mapping and the cache breakpoint in place."""
        if self.schema is not None:
            object.__setattr__(self, "schema", _freeze_mapping(self.schema))
        object.__setattr__(self, "cache_control", _freeze_optional(self.cache_control))


@dataclass(frozen=True)
class Conversation:
    """The semantic content of a request, independent of any wire format.

    Attributes:
        system: Ordered system text, lifted here from whichever of the four
            carriers the format uses (R8.2).
        system_role: The role the source ``Content`` published on a system
            instruction — Gemini's ``systemInstruction.role`` today, others
            do not publish one. ``None`` when absent, which makes a dropped
            role visible to the oracle as a positive delta at
            ``conversation.system_role`` rather than a silent equivalence on
            ``"user"`` (KBR-194).
        turns: Ordered turns.
        tools: Ordered tool declarations.
        sampling: Sampling parameters, keyed by :data:`SAMPLING_KEYS`.
    """

    system: Sequence[Text] = ()
    system_role: str | None = None
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
        """Validate the extra keys, any tool choice, then freeze the extra mapping.

        ``extra`` is keyed by the wire key and compared whole (§3.3.1a) — no
        ``.`` allowed in a key, with one canonical-value exception: ``tool_choice``.
        ``tool_choice`` is the one entry with a *canonical* value (R8.6), so it
        is the one entry worth checking at the value level; leaving it unchecked
        would make :data:`TOOL_CHOICE_VALUES` a comment rather than a rule.
        The key-shape check is enforced here rather than only on the path builder
        so a reader cannot emit a nested key by accident; the harness-internal
        nature of this construction makes the raise the reader-bug posture
        ``contract`` already names. The guard short-circuits on the first dotted
        key — an asymmetry with :meth:`Conversation.__post_init__`, which lists
        every unknown sampling key in one message — kept because
        :func:`extra_path` raises on the same single key and this guard mirrors it.

        Raises:
            ValueError: When an ``extra`` key contains ``.`` (KBR-191), or when
                ``extra["tool_choice"]`` is outside :data:`TOOL_CHOICE_VALUES`
                and is not a ``tool:<name>`` selection.
        """
        for key in (self.extra or {}):
            if "." in key:
                raise ValueError(
                    f"envelope.extra is keyed by wire key and compared whole (§3.3.1a); "
                    f"{key!r} names a nested value; nesting belongs in the residual (§3.3.1a)"
                )

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

#: Bare-collection anchors, for rows that change a collection as a whole: M5, M6
#: and M7 rewrite the turns, P5b joins the system blocks, and §3.3.1 pins P13 and
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

#: The canonical address for the parallel-tool-use knob, fixed in §3.3.1b
#: (KBR-205, closing G36). Anthropic's nested, inverted flag
#: (`tool_choice.<shape>.disable_parallel_tool_use`) and Chat Completions'
#: top-level `parallel_tool_calls` (a boolean for "should the model emit
#: multiple tool calls?") name one concept with two spellings and opposite
#: polarity; both wires meet at this address in the Chat Completions
#: spelling and polarity. The reader writes the entry only when the wire
#: carries a non-default value — KBR-214 forwards only
#: `disable_parallel_tool_use: true` as `parallel_tool_calls: false`, and
#: the reader mirrors that. An absent value and an explicit default (`true`
#: on Anthropic, the implicit CC default) are one request on both wires;
#: writing both would invent a second field some providers reject.
PARALLEL_TOOL_CALLS_KEY = "parallel_tool_calls"

#: The value a register row carries when the projection deliberately does not
#: model its effect. It **requires a reason**. P16 uses it (the content-type tag
#: is redundant with the turn's role), as do the whole-body protocol
#: translations M2, M9, P11 and P12, which change everything and so name nothing
#: usefully. An empty cell would make those rows silently unfalsifiable.
NOT_PROJECTABLE = "not projectable"

#: The wildcard segment. §3.3.1 writes register fields with an empty index —
#: "P15 is ``conversation.tools[].strict``" — meaning every tool.
WILDCARD = "*"

#: §3.3.1's reader-side declared-ignored mechanism (KBR-205). Keys are the
#: exact wire spellings of ``(block type, field name)`` — no wildcards, no
#: prefixes, no fuzzy match, so a re-spelled sibling still residualises and
#: fails the run (the unregistered-mutation guard). Values are the
#: ``(expected type, reason)`` pair:
#:
#: - The **expected type** is the vendor's published type for the field's
#:   value. A reader consumes a field only when its wire value is an instance
#:   of that type — §7.4.1's wrongly-typed-leaf rule, applied at the
#:   registry layer so the same discipline five readers cannot quietly drift
#:   on. A wrong-typed value residualises at its own path with the field
#:   named, and the run fails closed.
#: - The **reason** is required and non-empty (enforced by
#:   :func:`ignored_field_problems`), and cites the vendor evidence — §3.3.1's
#:   independent-oracle rule. It is the audit trail a future reader consults
#:   to answer "why does this field disappear?" without re-reading the
#:   ticket.
#:
#: **Built once, imported by every reader.** Seven authors answering this
#: separately is the coordination failure §7.4.1 exists to prevent; a reader
#: that wants to declare a field ignored but its siblings do not proposes a
#: new entry here, and carries the falsification case (§1.4) with it.
#: :func:`is_ignored` and :func:`ignored_fields_for` are the only ways to
#: consult the registry; readers must go through them so the spelling cannot
#: drift between T-A1 and T-A6.
#:
#: **What belongs here, what does not.** A field whose mutation changes what
#: the agent asked for — `tool_choice.disable_parallel_tool_use`,
#: `tools[i].type` — is **mapped** onto a shared canonical address, not
#: ignored, so the mutation remains visible to the oracle. A field whose
#: strip would hide a cost the user bears — `cache_control`'s cache
#: breakpoint — gets a **slot** in the grammar and a register row, not an
#: ignore. This registry is for the third class: descriptive, vendor-defined
#: fields the agent neither reads nor writes, whose loss changes no
#: instruction the request carried. The trade-off, recorded so a future
#: reader can decide whether to widen the mechanism: **value changes inside
#: an ignored field are invisible to the oracle**, the same shape §7.4.1's
#: merge rule names in its "what the merge hides" caveat. The ticket owner's
#: reasoning was that vendor-runtime-set or vendor-metadata fields are not
#: fidelity signals the agent depends on; a future case where that ceases to
#: hold belongs in §11, not in this registry.
IGNORED_BLOCK_FIELDS: Mapping[tuple[str, str], tuple[type, str]] = MappingProxyType(
    {
        # Anthropic Messages — SDK types retrieved 2026-09-14
        # (`anthropic-sdk-python` `src/anthropic/types/`). Each reason cites
        # the source file so a reviewer can audit without re-fetching.
        (
            "text",
            "citations",
        ): (
            list,
            "Anthropic TextBlockParam.citations: Optional[Iterable[TextCitationParam]] "
            "(anthropic-sdk-python src/anthropic/types/text_block_param.py) — citation list "
            "Anthropic emits on a response text block; replayed into a request when an agent "
            "feeds a prior response back as input. Vendor-shaped metadata, not a control the "
            "agent sets; carrying it on a Part would put a vendor spelling into the wire-"
            "independent form, and dropping it loses no instruction the request carried.",
        ),
        (
            "image",
            "transformations",
        ): (
            dict,
            "Anthropic ImageBlockParam.transformations: Optional[ImageTransformationsParam] "
            "(anthropic-sdk-python src/anthropic/types/image_block_param.py) — server-side "
            "preprocessing config ('downsize' vs 'error' on an oversized image). Server-side; "
            "the model observes the result, not the field. Stripped on translation; preserved "
            "on the native passthrough. The ticket author's reasoning: not a fidelity signal "
            "the agent depends on.",
        ),
        (
            "tool_use",
            "caller",
        ): (
            dict,
            "Anthropic ToolUseBlockParam.caller: Caller union "
            "(anthropic-sdk-python src/anthropic/types/tool_use_block_param.py) — the "
            "invocation context (direct, server-side, programmatic). Vendor-recorded metadata; "
            "Anthropic's runtime sets it on a tool_use emitted in an earlier assistant turn, "
            "and Claude Code replays it as part of the conversation history. The bridge does "
            "not own this field and does not transform it; treating it as consumed preserves "
            "totality on real Claude Code bodies.",
        ),
        (
            "tool_use",
            "toolset_name",
        ): (
            str,
            "Anthropic ToolUseBlockParam.toolset_name: Optional[str] "
            "(anthropic-sdk-python src/anthropic/types/tool_use_block_param.py) — names the "
            "toolset family a toolset-member tool_use belongs to ('computer', 'browser'). "
            "Paired with the toolset declaration itself; an unmodelled block today.",
        ),
        (
            "tool_result",
            "toolset_name",
        ): (
            str,
            "Anthropic ToolResultBlockParam.toolset_name: Optional[str] "
            "(anthropic-sdk-python src/anthropic/types/tool_result_block_param.py) — the "
            "member result echoes the paired tool_use's toolset_name. A dispatch key for "
            "the agent's own handler, not a fidelity signal the bridge shapes.",
        ),
    }
)


def ignored_field_problems(
    registry: Mapping[tuple[str, str], tuple[type, str]],
) -> tuple[str, ...]:
    """Report every way one entry of the declared-ignored registry is malformed.

    Pure, and separate from the loop that applies it, so a deliberately bad
    entry can be handed to it — §1.4 requires the falsification case to run in
    the suite, and a rule enforced only inside a ``for`` over the shipped
    registry cannot be given one. The same posture :func:`row_shape_problems`
    takes on register rows.

    Args:
        registry: A candidate registry of ``(block wire type, field wire key)``
            to ``(expected type, reason)``. The block key is the wire
            discriminator (Anthropic ``"text"``, ``"image"``, ``"tool_use"``,
            ``"tool_result"``); the field key is the wire spelling of the
            field on that block. Both keys must be non-empty strings; the
            expected type must be a real type; the reason must be a non-empty
            string after stripping whitespace.

    Returns:
        One message per problem, empty when the registry is well formed.
    """
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for (block_kind, field_name), entry in registry.items():
        # Tuple key check — guards against an entry built with a non-string
        # block or field. A non-string key would silently mismatch nothing
        # when a reader's helper consults the registry, which is exactly the
        # silent escape §3.3.1's declared-ignored mechanism is meant to
        # forbid.
        key_problem = False
        if not isinstance(block_kind, str) or not block_kind:
            problems.append(
                f"registry entry {block_kind!r}:{field_name!r}: block key must be a non-empty string"
            )
            key_problem = True
        if not isinstance(field_name, str) or not field_name:
            problems.append(
                f"registry entry {block_kind!r}:{field_name!r}: field key must be a non-empty string"
            )
            key_problem = True
        if not key_problem and (block_kind, field_name) in seen:
            problems.append(
                f"registry entry ({block_kind!r}, {field_name!r}): duplicate entry; "
                "the audit trail is one entry per wire spelling"
            )
        seen.add((block_kind, field_name))

        # The value is the (expected type, reason) pair. A non-tuple, a
        # non-type, or a None reason are all rejected at this layer so a
        # future reader cannot quietly build a malformed registry and have
        # it pass.
        if not isinstance(entry, tuple) or len(entry) != 2:
            problems.append(
                f"registry entry ({block_kind!r}, {field_name!r}): value must be "
                f"(expected_type, reason); got {entry!r}"
            )
            continue

        expected_type, reason = entry
        if not isinstance(expected_type, type):
            problems.append(
                f"registry entry ({block_kind!r}, {field_name!r}): expected type must "
                f"be a type, got {expected_type!r}"
            )

        # The reason is the audit trail. None, empty or whitespace-only is
        # rejected — the same posture :func:`row_shape_problems` takes on
        # `not_projectable_reason` (§3.3.1a: "an empty cell would leave
        # those rows silently unfalsifiable").
        if reason is None:
            problems.append(
                f"registry entry ({block_kind!r}, {field_name!r}): reason is required "
                "(§3.3.1a); an entry without one would be silently unfalsifiable"
            )
        elif not isinstance(reason, str):
            problems.append(
                f"registry entry ({block_kind!r}, {field_name!r}): reason must be a string, "
                f"got {type(reason).__name__}"
            )
        elif not reason.strip():
            problems.append(
                f"registry entry ({block_kind!r}, {field_name!r}): reason must be non-empty (§3.3.1a)"
            )

    return tuple(problems)


def is_ignored(block_kind: str, field_name: str) -> bool:
    """Return whether a block field is in the declared-ignored registry.

    The single spelling readers consult: :func:`consumed_ignored_fields`
    uses it for the per-block lookup, and a future reader that wants to know
    whether *its* format has a declared-ignored field for a given block
    kind asks here rather than indexing :data:`IGNORED_BLOCK_FIELDS`
    directly. The two-step spelling — registry → helper → reader — is what
    keeps seven authors from quietly disagreeing on the rule.

    Args:
        block_kind: The wire discriminator of the block carrying the field.
        field_name: The wire spelling of the field on that block.

    Returns:
        ``True`` when an exact-match entry exists for the pair, ``False``
        otherwise. Re-spelled siblings are deliberately absent — a typo'd
        ``"CitationS"`` is not the same entry as ``"citations"``, and the
        guard it fails to satisfy (§3.3.1) is exactly the one that protects
        against unregistered mutations.
    """
    return (block_kind, field_name) in IGNORED_BLOCK_FIELDS


def ignored_fields_for(block_kind: str) -> Mapping[str, tuple[type, str]]:
    """Return the declared-ignored fields for one block wire type.

    Reads :data:`IGNORED_BLOCK_FIELDS` and filters by the given block kind.
    Used by readers to extend their per-block mapped-keys set with the
    declared-ignored fields for that kind, then call
    :func:`consumed_ignored_fields` to consume them in one place.

    Args:
        block_kind: The wire discriminator of the block (Anthropic
            ``"text"``, ``"image"``, ``"tool_use"``, ``"tool_result"``).

    Returns:
        A frozen mapping from field wire name to its
        ``(expected_type, reason)``. Empty when the block kind has no
        declared-ignored entries today — the right answer for a reader
        whose format has no field the registry knows about.
    """
    entries: dict[str, tuple[type, str]] = {}
    for (kind, field_name), entry in IGNORED_BLOCK_FIELDS.items():
        if kind == block_kind:
            entries[field_name] = entry
    return _freeze_mapping(entries)


def consumed_ignored_fields(
    block: Mapping[str, Any],
    block_kind: str,
    path: str,
    residual: dict[str, Any],
) -> set[str]:
    """Consume the block's declared-ignored fields whose values are correctly typed.

    Consults :data:`IGNORED_BLOCK_FIELDS` for the block's wire kind via
    :func:`ignored_fields_for`. A field whose wire value is an instance of
    the registry's expected type is consumed — the reader's
    :func:`~tests.harness.reader_anthropic_messages._residualise` pass will
    exclude it, and ``verify_total`` will see it as accounted for. The
    value is dropped on the floor: declared-ignored fields do not get a
    slot on any :class:`Part`, and the projection does not carry them.

    A field whose value carries the **wrong** type residualises at its own
    path, §7.4.1's wrongly-typed-leaf rule: the registry declares the
    *field* ignorable, not the *value* well-formed. A reader that mistyped
    a sibling (``citations: 7`` instead of an iterable) still fails the
    run, with the field named.

    A field whose value is absent (``None`` or missing) is silently
    consumed, the same posture :func:`_read_cache_control` takes on
    ``cache_control: null``. The bridge does not invent an absence the body
    did not declare.

    Args:
        block: The block being read.
        block_kind: The block's wire discriminator (Anthropic ``"text"``,
            ``"image"``, ``"tool_use"``, ``"tool_result"``).
        path: The block's path from the body root, used to build residual
            keys for wrong-typed values.
        residual: The residual mapping, extended in place when a value
            carries the wrong type.

    Returns:
        The set of field names consumed — the caller's
        :func:`_residualise` pass excludes these from the mapped-keys set so
        they do not residualise as if unknown.
    """
    consumed: set[str] = set()
    for field_name, (expected_type, _reason) in ignored_fields_for(block_kind).items():
        if field_name not in block:
            continue
        value = block[field_name]
        if value is None:
            # Absent by the cache_control precedent. The block carried the
            # key but with `null`; treat as no-op.
            consumed.add(field_name)
            continue
        if isinstance(value, expected_type):
            consumed.add(field_name)
        else:
            residual[f"{path}.{field_name}"] = value
    return consumed


# Falsification of the unrecoverable half of §3.3.1a, raised at module import
# the same way `row_shape_problems` is invoked against `REGISTER`. A bad
# shipped entry must fail the gate, not slip through review.
assert not ignored_field_problems(IGNORED_BLOCK_FIELDS), (
    "IGNORED_BLOCK_FIELDS has malformed entries — see ignored_field_problems"
)


def _index(value: int | str) -> str:
    """Render a collection index for a path, concrete or wildcard.

    ``mypy`` runs on ``src/kitty`` only (plan §1.3), so the ``int | str``
    annotation on the builders below is documentation rather than enforcement.
    A typo such as ``part_path(0, "oops")`` would otherwise build a path that
    looks concrete, matches nothing, and makes a register row claim nothing.

    Args:
        value: A position, or :data:`WILDCARD`.

    Returns:
        The index in its string form.

    Raises:
        ValueError: When ``value`` is neither an ``int`` nor :data:`WILDCARD`.
            ``True`` and ``False`` are rejected despite ``bool`` being a subclass
            of ``int``.
    """
    # `bool` first, because it is a subclass of `int`: without that clause
    # `part_path(True, 0)` builds "conversation.turns[True].parts[0]", which
    # reads as concrete and matches nothing -- the exact failure this validator
    # exists to prevent, and the one an `isinstance(value, int)` test waves
    # through. `None` and `1.5` are the other half.
    if isinstance(value, bool) or (value != WILDCARD and not isinstance(value, int)):
        raise ValueError(f"a path index must be an int or {WILDCARD!r}, got {value!r}")
    return str(value)


def extra_path(key: str) -> str:
    """Return the path naming a format-specific control field.

    §3.3.1a: ``extra`` is diffed **one wire key at a time** and the value under a
    key is compared whole, so ``envelope.extra[thinking.budget_tokens]`` is not a
    path this vocabulary defines.  Enforced rather than merely stated, the way
    :class:`Conversation` enforces the closed sampling set: six readers written
    by six authors cannot quietly disagree about whether a nested value has an
    address of its own.  The trade is that a vendor shipping a dotted wire key
    would have to be a design decision rather than a silent match failure.
    ``residual_path`` still accepts dots, and its docstring says why.

    Args:
        key: The wire key, e.g. ``thinking`` for P2a.

    Returns:
        A path of the form ``envelope.extra[<key>]``.

    Raises:
        ValueError: When ``key`` contains a dot.
    """
    if "." in key:
        raise ValueError(
            f"envelope.extra is keyed by wire key and compared whole (§3.3.1a); {key!r} names a nested value"
        )
    return f"envelope.extra[{key}]"


def sampling_path(key: str) -> str:
    """Return the path naming one sampling parameter.

    Args:
        key: A member of :data:`SAMPLING_KEYS`.

    Returns:
        A path of the form ``conversation.sampling[<key>]``.
    """
    return f"conversation.sampling[{key}]"


def system_path(index: int | str, field_name: str | None = None) -> str:
    """Return the path naming one system text part, or a field of it.

    Args:
        index: Position in :attr:`Conversation.system`, or :data:`WILDCARD`
            when a register row names every one of them.
        field_name: An optional field, e.g. ``cache_control`` for M16.

    Returns:
        A path of the form ``conversation.system[<i>]``, with ``.<field>``
        appended when one is given.
    """
    base = f"conversation.system[{_index(index)}]"
    return f"{base}.{field_name}" if field_name else base


def turn_path(index: int | str, field_name: str | None = None) -> str:
    """Return the path naming one turn, or a field of it.

    Args:
        index: Position in :attr:`Conversation.turns`, or :data:`WILDCARD`
            when a register row names every one of them.
        field_name: An optional field, e.g. ``role``.

    Returns:
        A path of the form ``conversation.turns[<i>]``, with ``.<field>``
        appended when one is given.
    """
    base = f"conversation.turns[{_index(index)}]"
    return f"{base}.{field_name}" if field_name else base


def part_path(turn_index: int | str, part_index: int | str, field_name: str | None = None) -> str:
    """Return the path naming one part of one turn, or a field of it.

    §3.3.4 requires a failure to name the exact turn and part.

    Either index accepts :data:`WILDCARD`.  A register row writes a *pattern*
    over every turn and part — M3, M4, M8, P5e and P8 all do — where a delta
    writes concrete indices, and both must come from this one builder or the
    spelling drifts between T-W3 and T-D1.

    ``field_name`` exists for **M16**, and naming the field matters for the
    reason §3.3.1a gives for P15: the bare part path would also claim a *deleted
    part*, which is one of §3.3.1's five oracle falsification cases.

    Args:
        turn_index: Position in :attr:`Conversation.turns`, or :data:`WILDCARD`.
        part_index: Position in that turn's parts, or :data:`WILDCARD`.
        field_name: An optional field of the part, e.g. ``cache_control``.

    Returns:
        A path of the form ``conversation.turns[<i>].parts[<j>]``, with
        ``.<field>`` appended when one is given.
    """
    base = f"conversation.turns[{_index(turn_index)}].parts[{_index(part_index)}]"
    return f"{base}.{field_name}" if field_name else base


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

    The P9 header rows change headers rather than the body, and §4.3 C1 asserts on
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


def residual_key(prefix: str, key: str | None = None, index: int | None = None) -> str:
    """Return the residual key naming one unclassified value inside a body.

    §7.4.1 fixes two rules and this builder is the one shared spelling of
    both (KBR-193).  Rule 1: a wholly-unclassified **top-level** key is
    keyed by its bare name — ``x-kitty-trace``, never
    ``residual[x-kitty-trace]`` — because :func:`verify_total` compares the
    residual's keys against the body's own top-level keys, and a wrapped
    form would miss ``source`` and raise :class:`DroppedFieldsError` naming
    the wrong defect.  Rule 2: a nested key is keyed by its path from the
    body root with **array positions as indices** — ``tools[0].type``,
    ``messages[2].content[0].x_vendor_marker``.  It deliberately does
    **not** inherit §3.3.1a's by-name tool addressing: that convention
    exists because translators reorder declarations, a property of a
    *comparison*, while a residual key is never matched against a register
    pattern.

    This is the *mapping* builder.  :func:`residual_path` renders the
    *delta path* the oracle reports — the two are deliberately distinct,
    and §7.4.1 says so in as many words.

    Args:
        prefix: The path of the containing object from the body root,
            ``""`` at the body root.  May already contain bracketed array
            indices from earlier :func:`residual_key` calls.
        key: The wire key naming the value inside ``prefix``'s object.
            ``None`` when the value *is* the object at ``prefix`` (or at
            ``prefix[index]``) and no further field name follows.
        index: The array position to append to ``prefix`` in ``[index]``
            form, **before** ``key`` is appended.  ``None`` when the value
            does not live in an array.

    Returns:
        The key under which the value is stored in a request's
        ``residual`` mapping.

    Raises:
        ValueError: When the arguments cannot spell a body path — an empty
            ``prefix`` with an ``index`` (``"[0].field"`` is not a path),
            an explicitly-passed empty ``key`` (``"prefix."`` is a trailing
            dot), or an ``index`` that is not an ``int``.  The three
            non-int cases are worth naming separately: ``bool`` is
            rejected despite being a subclass of ``int`` (matching
            :func:`_index`), a ``float`` would silently build
            ``prefix[1.5]``, and *any* string — including the
            :data:`WILDCARD` sentinel — is rejected *unlike*
            :func:`_index`, because §7.4.1 fixes residual keys as array
            positions, never patterns.

    Examples:
        >>> residual_key("tool_choice")
        'tool_choice'
        >>> residual_key("", "tool_choice")
        'tool_choice'
        >>> residual_key("messages[2].content", "x_marker")
        'messages[2].content.x_marker'
        >>> residual_key("tools", "input_schema", index=0)
        'tools[0].input_schema'
        >>> residual_key("messages[2].content", "x_marker", index=0)
        'messages[2].content[0].x_marker'
    """
    # Validate before rendering: mypy runs on `src/kitty` only (plan §1.3),
    # so the annotation cannot catch a wrong argument type — the readers'
    # callers sit in `tests/harness`, outside its gate.
    if index is not None:
        if index == WILDCARD:
            raise ValueError(
                "a residual key is a body path, never a pattern: "
                f"index={index!r} is not an array position (§7.4.1 rule 2)"
            )
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"a residual index must be an int, got {index!r}")
        if not prefix:
            raise ValueError(
                f"an index needs a non-empty prefix to attach to, got prefix={prefix!r}"
            )
    if key == "":
        raise ValueError("an explicitly-passed key must be non-empty — 'prefix.' is a trailing dot")

    # Render in §7.4.1's own order: the index attaches to the prefix's tail,
    # then the key joins with a dot — or the bare prefix/key stands alone,
    # which is rule 1's whole point.
    rendered_prefix = f"{prefix}[{index}]" if index is not None else prefix
    if key is None:
        return rendered_prefix
    if not rendered_prefix:
        return key
    return f"{rendered_prefix}.{key}"


def reply_part_path(index: int | str) -> str:
    """Return the path naming one part of a reply.

    Args:
        index: Position in :attr:`Reply.parts`, or :data:`WILDCARD` for a row
            naming every part. M12 does **not** use the wildcard — it is
            anchored at index 0, because both translators substitute one text
            part into a reply that was empty. The wildcard is here for T-D10,
            whose reply diff reports concrete positions the register may need to
            claim in bulk.

    Returns:
        A path of the form ``reply.parts[<i>]``.
    """
    return f"reply.parts[{_index(index)}]"


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

    Over-claiming in the other direction is *recoverable*, not free, and only
    because §3.3.1a requires a row to be anchored at the **narrowest** path
    covering its effect — so every path beneath that node is, by construction,
    part of what the row changed. A coarser anchor silently claims what it must
    not, which is why T-W3 carries the discipline and T-D3 the falsification case.

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

