"""The golden Claude Code transcript corpus — format, scrubber and loader.

``.system_design/TEST_SUITE.md`` §7.1 · plan task **T-W6** (KBR-29).

The corpus holds real ``POST /v1/messages`` requests captured from Claude Code
sessions.  They are the **inbound** half of every fidelity comparison: §3.3's
oracle projects what the agent sent and what reached the provider, and asserts
the two agree except for the mutations on the register.  ``tests/corpus/README.md``
carries the capture procedure, the refresh owner and cadence, and the scrub
policy; this module is the format's only reader and writer.

**It imports nothing from ``src/kitty``, and must not** — the rule
:mod:`harness.contract` and :mod:`harness.register` are both written under.  A
corpus that asked kitty how to read a body would inherit kitty's bugs, and the
whole of I1 would prove only self-consistency.

**The scrubber and the detector are one pass, not two.**  :func:`scrub` and
:func:`findings` both call :func:`_scan`, which walks the pattern table once and
returns the rewritten text *and* what it rewrote.  Two separate implementations
would be two things to keep in step, and the failure mode of their drifting is
the one that matters: a detector that stops seeing what the scrubber stopped
removing reports every fixture clean.  Making them the same pass makes
``findings(scrub(x)) == ()`` true by construction rather than by vigilance.

**A shaped secret is the only kind either can find.**  Every pattern is anchored
on a known prefix or on an assignment to a known name.  The tempting addition —
a general high-entropy rule — was rejected on measurement, not taste: a captured
Claude Code body is full of long opaque strings that are *not* secrets (thinking
block signatures, ``toolu_`` identifiers, base64 images), and a rule that flagged
them would either be switched off or, worse, would rewrite them.  §3.3.3 requires
the text ``Please explain how kitty-bridge works`` to survive byte-identically;
a scrubber that mangles legitimate content breaks I1 in the act of defending the
repository.  The residue is handled by the review step, which the README makes
mandatory rather than advisory.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from harness.contract import REDACTED_HEADERS, REDACTED_QUERY_KEYS, CapturedRequest, WireFormat
from harness.register import Trigger

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class CorpusEntryError(AssertionError):
    """Raised when a file under the corpus root cannot be read as an entry.

    An ``AssertionError`` for the reason :class:`~harness.register.RegisterMarkdownError`
    is one: every failure mode here is *silent* if it yields an empty or partial
    entry instead.  A manifest key the loader quietly ignored is a trigger
    declaration nobody made, and §3.3.2's second assertion would then run over a
    complement that was never claimed.
    """


class UnscrubbedCorpusError(AssertionError):
    """Raised when a committed entry still carries a credential or an identifier.

    Separate from :class:`CorpusEntryError` because the two failures need
    different responses from a maintainer: a malformed manifest is a typo, and
    this is a secret that has reached a public repository.
    """


# --------------------------------------------------------------------------
# The scrubber's pattern table
# --------------------------------------------------------------------------

#: The placeholder a redacted run is replaced by, parameterised by class name.
#:
#: Class-named rather than a single opaque mask, because a reviewer reading a
#: diff needs to know *what* was removed to judge whether the entry is still
#: worth committing.  It contains no character JSON escapes, so substituting it
#: into a body leaves the body parseable.
REDACTION = "<redacted:{name}>"

#: Credential and identifier shapes, applied **in order**.
#:
#: Order is load-bearing twice.  ``anthropic_key`` precedes ``openai_key``
#: because ``sk-ant-…`` satisfies both and the more specific class is the more
#: useful thing to write into the fixture.  Every shaped class precedes
#: ``assigned_secret`` because an already-redacted value no longer looks like a
#: long opaque run, so the general rule stops reporting a second finding for a
#: secret the specific rule already removed.
#:
#: The prefixed shapes are gitleaks' (``config/gitleaks.toml``, fetched
#: 2026-09-12) **loosened at the tail**: gitleaks pins Anthropic keys at exactly
#: 93 characters plus ``AA`` because a detector's job is precision on today's
#: format.  A scrubber's job is the opposite — an ``sk-ant-api04-`` minted next
#: year must not walk through the net because its length changed — so the tail
#: here is ``{20,}`` and the prefix carries the specificity.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # `\b` is load-bearing, not tidiness. Unanchored, `sk-...` matches inside
    # ordinary words: `disk-usage-monitoring-service.py` becomes
    # `di<redacted:openai_key>.py`. File paths are the commonest payload in a
    # Claude Code body, so an unanchored rule mangles legitimate content at
    # scale -- breaking I1 in the act of defending the repository, which is
    # exactly what this table is supposed not to do. Every real position a key
    # occupies -- line start, after `"`, after `=`, after a space -- carries a
    # word boundary, so nothing true is lost.
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai_key", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_-]{20,}")),
    ("gcp_key", re.compile(r"AIza[A-Za-z0-9_-]{35}")),
    ("aws_key_id", re.compile(r"\b(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_\w{82})\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    # A PEM block is the one credential shape that belongs to no provider, and
    # a developer tree is full of them. The header alone is enough: a body
    # containing it is a body carrying a key, whatever follows.
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
    # `Bearer` inside a BODY -- a curl command in a prompt, an HAR file, a log
    # the agent read. The header vocabulary in `contract` covers the header;
    # nothing covered this.
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    (
        "assigned_secret",
        # The optional `\\?["']` is not decoration: a corpus body is JSON, so a
        # `.env` or a settings file the agent read arrives with its quotes
        # escaped -- `\"api_key\": \"…\"`. Measured on a real body shape; without
        # it this rule finds shell assignments and misses every quoted one.
        re.compile(
            r"(?:api[_-]?key|apikey|secret|token|password|passwd)"
            r"(?:\\?[\"'])?\s*[:=]\s*(?:\\?[\"'])?([A-Za-z0-9+/_=-]{20,})",
            re.IGNORECASE,
        ),
    ),
    # The lookahead excludes the retina asset shape -- `logo@2x.png`, `icon@3X.jpg`
    # -- which is exactly the shape of an address and common in a web tree.
    #
    # It targets that SHAPE and not a list of extensions, which was the first
    # attempt and was wrong in the dangerous direction: `.py` is Paraguay's ccTLD
    # and `.md` is Moldova's, so excluding them silently stopped scrubbing
    # `admin@empresa.py`. Trading a cosmetic false positive for a real miss is a
    # bad trade for a credential scrubber.
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@(?![0-9]+[xX]\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    # Replaces the username segment only, which is why the username is a capture
    # group and the prefix is not. The path itself is evidence -- which file the
    # agent read is part of the transcript -- and deleting it would cost more
    # than it protects.
    #
    # Three prefixes, not two: `\Users\` is here because the README promises it
    # and a lookbehind-based pattern silently did not deliver it. A Windows
    # capture leaked its username straight through a scrubber that looked
    # complete. `\\{1,2}` covers both the raw path and the JSON-escaped form a
    # body actually carries.
    #
    # `<`, `>` and NUL are excluded from the segment for reasons measured rather
    # than guessed. Without `<>` this pattern matches its OWN placeholder --
    # `/home/<redacted:home_path>/` -- so `scrub` looked idempotent (the
    # replacement equals what it replaced) while `findings` went on reporting a
    # secret that was no longer there; a lint that fires on scrubbed output is a
    # lint that gets switched off. Without NUL it would consume an allow-listed
    # literal's parking sentinel and destroy it.
    # `test_no_pattern_matches_a_placeholder` guards the table against the next
    # instance of the first; `TestKnownNonSecrets` against the second.
    ("home_path", re.compile(r"(?:/home/|/Users/|\\{1,2}Users\\{1,2})([^/\s\"'\\<>\x00]+)")),
)

#: The class names, for the README agreement guard.
PATTERN_NAMES: tuple[str, ...] = tuple(name for name, _ in PATTERNS)

#: The class a literal from ``extra`` is redacted under.
LITERAL_CLASS = "literal"

#: The class a credential header's value is redacted under.
CREDENTIAL_HEADER_CLASS = "credential_header"

#: The class a credential query parameter's value is redacted under.
CREDENTIAL_QUERY_CLASS = "credential_query"

#: Every class :data:`REDACTION` is ever formatted with.
#:
#: Enumerated here rather than in the test that needs it, because that test --
#: "no pattern matches a placeholder" -- is the guard behind both the
#: idempotence and the fixed-point property, and a guard that checks a
#: hand-copied subset of the classes silently stops covering the next one added.
REDACTION_CLASSES: tuple[str, ...] = (
    *PATTERN_NAMES,
    LITERAL_CLASS,
    CREDENTIAL_HEADER_CLASS,
    CREDENTIAL_QUERY_CLASS,
)


@dataclass(frozen=True)
class Finding:
    """One credential or identifier a scan found.

    **It never carries the matched value, not even a prefix.**  A finding is
    rendered into an assertion message, which reaches every CI log — and a CI
    log on a public repository is as public as the repository.  A contained
    authoring mistake would become a published one, and once a credential is in
    a public log the remedy is rotation, not a better diff.
    :class:`~harness.contract.CapturedRequest` masks its ``repr`` for the same
    reason.  The offset is what makes the finding actionable instead: it points
    a reviewer at the byte to look at in a file they already have.

    Attributes:
        name: The pattern class, e.g. ``anthropic_key``.
        where: Where it was found — ``body``, ``header:<name>`` or
            ``query:<key>``.
        offset: The byte offset of the match within that location.
    """

    name: str
    where: str
    offset: int


#: The prefix an allow-listed literal is parked under while the table runs.
#:
#: **Chosen against the text, not fixed.**  A fixed ``\x00``-delimited sentinel
#: is safe only while no body contains ``\x00`` — and T-C6 commits a
#: *malformed* body, which is by definition not constrained to being JSON.
#: Measured: with a fixed sentinel, a body carrying a literal NUL had the
#: allow-listed literal **injected** at a position it never occupied, because
#: unparking rewrote content that merely looked like a sentinel.  Deriving the
#: prefix from the text costs one substring search and removes the collision
#: entirely.
_SENTINEL_BASE = "\x00corpus"


def _sentinel_prefix(text: str) -> str:
    """Return a parking prefix that does not occur in ``text``.

    Args:
        text: The text about to be scanned.

    Returns:
        A prefix absent from ``text``, so no sentinel built on it can collide
        with content.
    """
    prefix = _SENTINEL_BASE
    while prefix in text:
        prefix += "0"
    return prefix


#: The shortest operator-supplied literal accepted, in either direction.
#:
#: ``extra`` is an unbounded substring replacement, so a three-character
#: username shreds a body everywhere those three characters occur — inside
#: words, inside base64, inside a tool name — and the damage is invisible to a
#: fixed false-positive control because the literal is operator-supplied.  An
#: operator whose username is shorter widens the literal instead: ``/home/tas/``
#: rather than ``tas``.
MIN_LITERAL = 4


def _park(text: str, allow: Sequence[str], prefix: str) -> str:
    """Replace each allow-listed literal with a sentinel the table cannot match.

    Args:
        text: The text to scan.
        allow: Literals a reviewer has declared are not secrets.
        prefix: A parking prefix absent from ``text``.

    Returns:
        The text with each literal parked.
    """
    for index, literal in enumerate(allow):
        text = text.replace(literal, f"{prefix}{index}\x00")
    return text


def _unpark(text: str, allow: Sequence[str], prefix: str) -> str:
    """Restore every parked literal.

    Args:
        text: The scanned text.
        allow: The same literals :func:`_park` was given.
        prefix: The same prefix :func:`_park` was given.

    Returns:
        The text with each literal restored.
    """
    for index, literal in enumerate(allow):
        text = text.replace(f"{prefix}{index}\x00", literal)
    return text


def _checked_literals(extra: Sequence[str], allow: Sequence[str]) -> None:
    """Fail unless every operator-supplied literal is long enough to be safe.

    Both ``extra`` and ``allow`` are unbounded substring operations, and both are
    dangerous when short — in opposite directions. A short ``extra`` shreds the
    body wherever those characters occur. A short ``allow`` is worse and was
    measured: clearing ``"key"`` parks the substring inside ``api_key``, so the
    ``assigned_secret`` rule no longer matches the name, and **the lint goes
    silently blind** on the very body a reviewer was vouching for. The
    stale-exemption check cannot catch it, because ``"key"`` is still present.

    Args:
        extra: Literals to redact.
        allow: Literals a reviewer has cleared.

    Raises:
        ValueError: When any literal is shorter than :data:`MIN_LITERAL`.
    """
    for field_name, literals in (("extra", extra), ("allow", allow)):
        short = [literal for literal in literals if len(literal) < MIN_LITERAL]
        if short:
            raise ValueError(
                f"{field_name} literals must be at least {MIN_LITERAL} characters; {short} would "
                "match inside ordinary words. Widen the literal instead — '/home/tas/', not 'tas'."
            )


#: The character :func:`_find` masks a claimed span with.
#:
#: A mask must do two things: match no pattern, and present a **word boundary**
#: at its edges, because that boundary is what lets the next round see a secret
#: that was flush against the one just claimed. A control character does both,
#: and `\x01` is the conventional choice.
#:
#: **A body already containing it is harmless, and that was measured rather than
#: assumed.** An earlier version searched for a control character free in the
#: body and raised when none was — defensive code for a failure that cannot
#: happen, and one that would have refused to scan a perfectly scannable body.
#: :data:`_find`'s ``claimed`` bytearray is the source of truth for what has been
#: claimed; the character only breaks word boundaries in the text the patterns
#: see, and a pre-existing `\x01` already breaks one. Verified on a body salted
#: with `\x01`: same findings, same offsets, same fixed point.
_CLAIM = "\x01"


def _masked(text: str, claimed: bytearray) -> str:
    """Return ``text`` with every claimed position replaced by ``char``.

    Length is preserved, which is the whole point: every offset recorded against
    the masked text is also an offset into the original.

    Args:
        text: The original text.
        claimed: One flag per character, non-zero where claimed.

    Returns:
        The masked text.
    """
    return "".join(_CLAIM if flag else original for original, flag in zip(text, claimed, strict=True))


def _find(text: str, extra: Sequence[str], allow: Sequence[str]) -> tuple[Finding, ...]:
    """Report every secret in ``text``, with true byte offsets into ``text``.

    **Non-mutating in length, and iterated to a fixed point.** Two properties
    that look incidental and are not:

    *Length-preserving masking* is what makes an offset usable. Deriving
    findings from the rewriting pass measured each position against text that
    earlier patterns had already shortened — six bytes out on a two-secret body,
    and worse on every additional one. Since R12 forbids reporting the matched
    value, the offset is the only actionable datum a reviewer gets.

    *The fixed point* is what makes the scan complete. Every credential pattern
    is anchored on a word boundary, and two secrets can sit flush against each
    other with no separator — an ``AIza…`` key whose last character is a letter,
    immediately followed by ``sk-proj-…``. On the first pass the second secret
    has no boundary in front of it and no pattern matches; masking the first one
    *creates* that boundary, and a single-pass scan has already moved on. One
    round of masking, then another pass, until nothing new is claimed.
    Termination is guaranteed because a claimed span is never released and a
    mask matches no pattern.

    Args:
        text: The text to inspect.
        extra: Literals the operator named.
        allow: Literals a reviewer has cleared — claimed before any pattern
            runs, so nothing inside one is ever reported.

    Returns:
        One finding per secret, in the order they were claimed.
    """
    found: list[Finding] = []
    claimed = bytearray(len(text))

    # Cleared literals are claimed first, so a match inside one is excluded by
    # the same precedence rule that stops one secret being reported twice.
    for literal in allow:
        for match in re.finditer(re.escape(literal), text):
            claimed[match.start() : match.end()] = b"\x01" * (match.end() - match.start())

    for literal in extra:
        position = text.find(literal)
        if position != -1:
            found.append(Finding(LITERAL_CLASS, "body", _byte_offset(text, position)))

    while True:
        progressed = False
        # Built once per round rather than per match: rebuilding it for each of
        # several thousand home paths in a multi-megabyte transcript is the
        # quadratic cost this loop exists to avoid.
        masked = _masked(text, claimed)

        for name, pattern in PATTERNS:
            for match in pattern.finditer(masked):
                start, end = match.span()
                if any(claimed[start:end]):
                    continue
                # Overlap is tested on the WHOLE match — that is what stops one
                # secret being reported twice — but only the span :func:`_rewrite`
                # actually replaces is claimed. The distinction is the difference
                # between the two halves agreeing and not: `assigned_secret`
                # rewrites its value and leaves `SECRET="` in place, so claiming
                # the whole match here would manufacture a word boundary the
                # rewriter never creates, and this half would report a secret the
                # other half leaves in the file.
                position, claim_end = match.span(1) if match.groups() else (start, end)
                found.append(Finding(name, "body", _byte_offset(text, position)))
                claimed[position:claim_end] = b"\x01" * (claim_end - position)
                progressed = True

        if not progressed:
            return tuple(found)


def _byte_offset(text: str, position: int) -> int:
    """Convert a character index into a byte offset.

    The committed artifact is a file of bytes, and a reviewer locating a finding
    is looking at bytes. A character index is the same number only while the
    body is ASCII, and a Claude Code transcript rarely is.

    Args:
        text: The text the index refers to.
        position: The character index.

    Returns:
        The corresponding UTF-8 byte offset.
    """
    return len(text[:position].encode("utf-8"))


def _rewrite(text: str, extra: Sequence[str], allow: Sequence[str]) -> str:
    """Replace every secret in ``text`` with a class-named placeholder.

    Args:
        text: The text to rewrite.
        extra: Literals the operator named.
        allow: Literals a reviewer has cleared, parked for the duration.

    Returns:
        The rewritten text.
    """
    prefix = _sentinel_prefix(text)
    text = _park(text, allow, prefix)

    for literal in extra:
        text = text.replace(literal, REDACTION.format(name=LITERAL_CLASS))

    # Iterated to a fixed point for the reason :func:`_find` is, and it must be
    # the same reason or the two would disagree: a placeholder's closing `>`
    # creates the word boundary a flush-adjacent secret was missing, and the
    # pattern that needed it has already had its turn. Terminates because a
    # placeholder matches nothing — `test_no_pattern_matches_a_placeholder`.
    while True:
        before = text

        for name, pattern in PATTERNS:
            replacement = REDACTION.format(name=name)

            def _sub(match: re.Match[str], _repl: str = replacement) -> str:
                # Replacing the whole match would delete the `api_key =` that
                # makes the entry legible, so a rule that captures its value
                # rewrites only the group.
                if match.groups():
                    start, end = match.span(1)
                    return match.group(0)[: start - match.start()] + _repl + match.group(0)[end - match.start() :]
                return _repl

            text = pattern.sub(_sub, text)

        if text == before:
            return _unpark(text, allow, prefix)


def _scan(
    text: str, extra: Sequence[str] = (), allow: Sequence[str] = ()
) -> tuple[str, tuple[Finding, ...]]:
    """Rewrite every secret in ``text`` and report what was rewritten.

    The seam both :func:`scrub` and :func:`findings` are built on. They read one
    pattern table through one function, so the failure that matters — a detector
    that stops seeing what the scrubber stopped removing, and reports every
    fixture clean — cannot arise from the two drifting apart.

    The two halves run different *algorithms* over that one table —
    :func:`_find` claims spans in length-preserving masks so its offsets stay
    true, :func:`_rewrite` substitutes placeholders — so agreement is a property
    to be kept, not one to be assumed. Both iterate to a fixed point and both
    claim exactly the span the rewriter replaces, which is what keeps them in
    step; measured over 4,000 adversarial bodies, dropping either of those made
    them disagree on 216 and 23 of them respectively.

    **What is guaranteed, and what is not.** The guarantee is the one the lint
    rests on: after :func:`scrub`, :func:`findings` reports nothing and a second
    scrub changes nothing — asserted over every ordered pair of known secret
    shapes at three separations, including none at all. The *count* of findings
    can still exceed the number of placeholders when one redaction subsumes a
    neighbour: a bearer token's value alphabet includes ``/`` and ``.``, so a
    token written flush against a path swallows it, and the path is reported but
    not separately replaced. That direction is safe — the region is redacted
    either way, and the failure message names more than it needs to rather than
    less. The dangerous direction, under-reporting, is what these two rules
    close.

    Args:
        text: The text to scan.
        extra: Literal strings the operator named — a username, a hostname, an
            organisation — that no pattern can recognise.
        allow: Literals a reviewer has declared are **not** secrets. The corpus
            needs this because the captures are taken while working on this
            repository, so a tool result quotes this tree — and
            ``tests/test_integration.py`` alone contains
            ``api_key = "sk-test-integration-key-12345"``.

    Returns:
        The rewritten text, and one :class:`Finding` per secret.

    Raises:
        ValueError: When any ``extra`` or ``allow`` literal is shorter than
            :data:`MIN_LITERAL`.
    """
    _checked_literals(extra, allow)

    return _rewrite(text, extra, allow), _find(text, extra, allow)


# --------------------------------------------------------------------------
# Scrubbing a whole capture
# --------------------------------------------------------------------------

#: Header whose value is recomputed rather than scrubbed, because scrubbing
#: changes the body's length.
_CONTENT_LENGTH = "content-length"


#: Headers whose presence means the captured bytes are not the entity body.
#:
#: The corpus stores the **entity body** — what the agent meant to send — not
#: the wire octets. A ``content-encoding`` capture hands the scrubber compressed
#: bytes, in which it finds nothing and reports clean: a false clean that no
#: plaintext falsification case can ever detect, which is the worst failure this
#: module has. ``transfer-encoding: chunked`` is the other half: aiohttp's
#: ``request.read()`` de-chunks, so the stored body would not correspond to its
#: own headers. Both are refused at write time rather than handled.
REFUSED_ENCODINGS = frozenset({"content-encoding", "transfer-encoding"})


class EncodedCaptureError(AssertionError):
    """Raised when a capture carries an encoding the corpus format cannot store."""


def scrub(
    captured: CapturedRequest, extra: Sequence[str] = (), allow: Sequence[str] = ()
) -> CapturedRequest:
    """Return ``captured`` with every credential and identifier replaced.

    Idempotent: a placeholder matches no pattern, so a second application is a
    no-op.  A fixed point of :func:`findings` by construction — both read
    :func:`_scan`.

    Args:
        captured: The request as observed.
        extra: Literal strings to redact, as :func:`_scan` takes them.
        allow: Literals a reviewer has declared are not secrets.

    Returns:
        A new :class:`~harness.contract.CapturedRequest`.  Header order, casing
        and duplicates are preserved; only values change.
    """
    body_text, _ = _scan(captured.body.decode("utf-8", errors="surrogateescape"), extra, allow)
    body = body_text.encode("utf-8", errors="surrogateescape")

    headers: list[tuple[str, str]] = []
    for name, value in captured.headers:
        lowered = name.lower()
        if lowered in REDACTED_HEADERS:
            # Whole-value replacement, not a scan: a credential header's value
            # IS the secret, whatever shape it happens to have.
            headers.append((name, REDACTION.format(name=CREDENTIAL_HEADER_CLASS)))
        elif lowered == _CONTENT_LENGTH:
            # Scrubbing changes the body's length, and a capture whose declared
            # length disagrees with its body is not a capture of anything that
            # could have been sent.
            headers.append((name, str(len(body))))
        else:
            headers.append((name, _scan(value, extra, allow)[0]))

    return CapturedRequest(
        method=captured.method,
        scheme=captured.scheme,
        host=captured.host,
        path=captured.path,
        query=_scrub_query(captured.query, extra, allow),
        headers=headers,
        body=body,
        arrival=captured.arrival,
        peer_port=captured.peer_port,
    )


def _scrub_query(query: str, extra: Sequence[str], allow: Sequence[str] = ()) -> str:
    """Return ``query`` with credential parameter values replaced.

    Splits on ``&`` and ``=`` without a URL parser, for the reason
    :func:`harness.contract._redact_query` does: the raw string is what was on
    the wire, and re-encoding it would misrepresent the capture.

    Args:
        query: The raw query string.
        extra: Literal strings to redact.
        allow: Literals a reviewer has declared are not secrets.

    Returns:
        The rewritten query string.
    """
    if not query:
        return query

    parts = []
    for pair in query.split("&"):
        name, sep, value = pair.partition("=")
        if sep and name.lower() in REDACTED_QUERY_KEYS:
            parts.append(f"{name}{sep}{REDACTION.format(name=CREDENTIAL_QUERY_CLASS)}")
        else:
            parts.append(_scan(pair, extra, allow)[0])
    return "&".join(parts)


def findings(captured: CapturedRequest, allow: Sequence[str] = ()) -> tuple[Finding, ...]:
    """Report every credential or identifier still present in ``captured``.

    Args:
        captured: The request to inspect.
        allow: Literals a reviewer has declared are not secrets.

    Returns:
        One finding per surviving secret, body findings first, then headers,
        then the query.  Empty when the capture is clean.
    """
    found = list(_scan(captured.body.decode("utf-8", errors="surrogateescape"), (), allow)[1])

    for name, value in captured.headers:
        if name.lower() in REDACTED_HEADERS and value != REDACTION.format(name=CREDENTIAL_HEADER_CLASS):
            found.append(Finding(CREDENTIAL_HEADER_CLASS, f"header:{name}", 0))
        elif name.lower() not in REDACTED_HEADERS:
            found.extend(Finding(f.name, f"header:{name}", f.offset) for f in _scan(value, (), allow)[1])

    for pair in captured.query.split("&") if captured.query else ():
        key, sep, value = pair.partition("=")
        if sep and key.lower() in REDACTED_QUERY_KEYS and value != REDACTION.format(name=CREDENTIAL_QUERY_CLASS):
            found.append(Finding(CREDENTIAL_QUERY_CLASS, f"query:{key}", 0))
        else:
            found.extend(Finding(f.name, f"query:{key}", f.offset) for f in _scan(pair, (), allow)[1])

    return tuple(found)


# --------------------------------------------------------------------------
# The entry
# --------------------------------------------------------------------------

#: Every manifest key.  Closed: an unrecognised key raises rather than being
#: ignored, because the realistic mistake is a misspelled `triggers_met`, and a
#: loader that ignored it would report an entry declaring nothing at all.
MANIFEST_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "description",
        "origin",
        "origin_note",
        "captured_from",
        "captured_at",
        "method",
        "scheme",
        "host",
        "path",
        "query",
        "headers",
        "body_file",
        "body_sha256",
        "wire_format",
        "known_non_secrets",
        "triggers_met",
        "triggers_absent",
    }
)

CAPTURED = "captured"
SYNTHETIC = "synthetic"

#: What an entry id may be.
#:
#: An id is a **file name component and nothing else**. It is interpolated into
#: two paths, so `../escaped` makes :func:`write_entry` write outside the corpus
#: directory entirely — measured, not theorised. The load side already refused
#: such an entry (the manifest's id must equal the filename stem, and a filename
#: cannot contain a separator), which is exactly what made the gap easy to miss:
#: the asymmetry looked like a check that existed.
#:
#: Deliberately narrower than "no separators". A leading dot makes a file the
#: corpus glob does not see, and a leading dash is an option to every command a
#: maintainer will run over these files.
ENTRY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def _checked_id(entry_id: str) -> str:
    """Return ``entry_id`` if it is a legal entry id.

    Shared by :func:`load_entry` and :func:`write_entry` so the two sides cannot
    disagree about what an id is — the disagreement that let the writer escape
    the directory the reader polices.

    Args:
        entry_id: The candidate id.

    Returns:
        The id, unchanged.

    Raises:
        CorpusEntryError: When it is not a bare name of letters, digits,
            underscores and dashes.
    """
    if not ENTRY_ID.match(entry_id):
        raise CorpusEntryError(
            f"{entry_id!r} is not a legal entry id: an id is a file name component, so it must "
            "start with a letter or digit and hold only letters, digits, underscores and dashes"
        )
    return entry_id

#: Triggers a corpus entry may not declare in either direction.
#:
#: :mod:`harness.register` states the model as a binary — "a trigger is a *route*
#: property or a *request* property, and only the second can be varied by a
#: corpus entry" — but its own docstring names a **third** kind twenty lines
#: earlier: "M6 fires on *an upstream 400*, M8 on *a rejected thinking
#: round-trip*, M9 on *an upstream tool-use format error*, M12 on *an empty
#: upstream response*."  Those are properties of the **upstream response**,
#: arranged by a scripted recorder.  An entry that claimed one would be claiming
#: something it is not the thing that decides — precisely the over-declaration
#: §9.2's gap **G21** warns about, where "assertion 1 claims every delta and the
#: oracle passes over a broken bridge", with the aggravation that "the same
#: author writes the entry and its trigger index, so the mechanism has no second
#: reader".  This is that second reader.
#:
#: :attr:`~harness.register.Trigger.ALWAYS` is here because the register calls it
#: "the absence of a condition, not a condition": every request meets it, so
#: declaring it met is noise and declaring it absent is false.
#:
#: **This set is the subset that is *provable* from text already in the
#: repository, not a classification of the whole vocabulary.**  Classifying all
#: 25 triggers is `KBR-186`, filed rather than guessed, for the reason T-W3 gave
#: for deferring trigger predicates: data nothing in this change could prove
#: wrong is what plan §1.4 forbids.  Until it lands, T-D8 cannot read corpus
#: coverage for M6, M8, M9 and M12 — they are discharged by a scripted-recorder
#: test, not by an entry.
NOT_CORPUS_DECIDABLE: frozenset[Trigger] = frozenset(
    {
        Trigger.ALWAYS,
        Trigger.UPSTREAM_REJECTED_OVERSIZED_ON_BALANCING,
        Trigger.THINKING_ROUNDTRIP_REJECTED,
        Trigger.NATIVE_TOOL_USE_FORMAT_ERROR,
        Trigger.UPSTREAM_EMPTY_RESPONSE,
    }
)


@dataclass(frozen=True)
class CorpusEntry:
    """One captured request, as the corpus commits it.

    Frozen for :class:`~harness.register.MutationRow`'s reason: the corpus is
    evidence, and a test able to edit an entry in place could make its own
    failure disappear.

    Attributes:
        id: The entry's name; equals the manifest's filename stem.
        description: One line naming what the entry exercises.
        origin: ``captured`` or ``synthetic``.
        origin_note: Why a synthetic entry could not be captured; empty for a
            captured one.
        captured_from: The Claude Code version a captured entry came from; empty
            for a synthetic one.  Required so the refresh cadence — re-capture
            on a version bump — is actionable.
        captured_at: The ISO date of capture; empty for a synthetic entry.
        request: The request itself, as :mod:`harness.contract` types it.
        wire_format: The format the **inbound** body is written in, or ``None``
            for a body no reader can classify.  §7.4's oracle takes an
            ``inbound_format`` and the corpus supplies the inbound half;
            §3.3.4's "select by the shape observed on the wire" rule is stated
            for the *upstream* capture, and inbound is not always Anthropic
            Messages — P14–P16 exist because a Responses-origin body arrives
            inbound.  ``None`` is how T-C6's malformed entry is distinguishable
            from an I1 breach, which §7.4 names as a requirement.
        known_non_secrets: ``(literal, reason)`` pairs a reviewer has declared
            are not secrets.
        triggers_met: Register triggers this entry establishes as an inbound
            request, declared against the **committed** artifact.
        triggers_absent: Register triggers it provably does not establish.
    """

    id: str
    description: str
    origin: str
    origin_note: str
    captured_from: str
    captured_at: str
    request: CapturedRequest
    wire_format: WireFormat | None
    known_non_secrets: tuple[tuple[str, str], ...]
    triggers_met: frozenset[Trigger]
    triggers_absent: frozenset[Trigger]


def _require(manifest: Mapping[str, object], entry_id: str) -> None:
    """Fail unless ``manifest`` carries exactly the manifest keys.

    Args:
        manifest: The parsed manifest.
        entry_id: The entry's id, for the message.

    Raises:
        CorpusEntryError: When a key is missing or unrecognised.
    """
    present = set(manifest)

    unknown = sorted(present - MANIFEST_KEYS)
    if unknown:
        raise CorpusEntryError(f"{entry_id}: unknown manifest key(s) {unknown}; a typo declares nothing")

    missing = sorted(MANIFEST_KEYS - present)
    if missing:
        raise CorpusEntryError(f"{entry_id}: missing manifest key(s) {missing}")


def _triggers(names: object, field_name: str, entry_id: str) -> frozenset[Trigger]:
    """Return ``names`` as triggers.

    Args:
        names: The manifest's value for the field.
        field_name: The field, for the message.
        entry_id: The entry's id, for the message.

    Returns:
        The triggers named.

    Raises:
        CorpusEntryError: When the value is not a list of known trigger names.
    """
    if not isinstance(names, list):
        raise CorpusEntryError(f"{entry_id}: {field_name} must be a list of trigger names")

    known = {t.value: t for t in Trigger}
    resolved = set()
    for name in names:
        if name not in known:
            raise CorpusEntryError(f"{entry_id}: {field_name} names unknown trigger {name!r}")
        if known[name] in NOT_CORPUS_DECIDABLE:
            raise CorpusEntryError(
                f"{entry_id}: {field_name} names {name!r}, which an inbound request cannot "
                "arrange — it is decided by the upstream response or by nothing at all. "
                "Declare it at the test that scripts the recorder, not in the manifest."
            )
        resolved.add(known[name])
    return frozenset(resolved)


def load_entry(manifest_path: Path) -> CorpusEntry:
    """Read one entry from its manifest and sidecar body.

    Args:
        manifest_path: The ``<id>.json`` file.

    Returns:
        The entry, with ``request.body`` holding the sidecar's exact bytes.

    Raises:
        CorpusEntryError: When the manifest is malformed, contradictory, or
            names a body file that does not exist.
    """
    stem = _checked_id(manifest_path.stem)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusEntryError(f"{stem}: manifest is not JSON: {exc}") from exc

    if not isinstance(manifest, dict):
        raise CorpusEntryError(f"{stem}: manifest must be a JSON object")

    _require(manifest, stem)

    if manifest["id"] != stem:
        raise CorpusEntryError(f"{stem}: manifest id is {manifest['id']!r}; it must equal the filename stem")

    # Origin decides which provenance fields are required. A synthetic entry
    # without its reason is the one plan §6 explicitly asks to be recorded
    # beside the fixture; a captured entry without a version makes the refresh
    # cadence unactionable.
    origin = manifest["origin"]
    if origin not in (CAPTURED, SYNTHETIC):
        raise CorpusEntryError(f"{stem}: origin must be {CAPTURED!r} or {SYNTHETIC!r}, not {origin!r}")
    if origin == SYNTHETIC and not manifest["origin_note"]:
        raise CorpusEntryError(f"{stem}: a synthetic entry needs an origin_note saying why it is not captured")
    if origin == CAPTURED and not manifest["captured_from"]:
        raise CorpusEntryError(f"{stem}: a captured entry needs captured_from — the Claude Code version")
    if origin == CAPTURED and not manifest["captured_at"]:
        raise CorpusEntryError(f"{stem}: a captured entry needs captured_at")

    met = _triggers(manifest["triggers_met"], "triggers_met", stem)
    absent = _triggers(manifest["triggers_absent"], "triggers_absent", stem)
    both = sorted(t.value for t in met & absent)
    if both:
        raise CorpusEntryError(f"{stem}: trigger(s) {both} declared both met and absent")

    # The name is fixed rather than followed. `write_entry` always writes
    # `<id>.body`, so a manifest naming anything else describes a file that a
    # rewrite would silently orphan -- and a value with a path separator would
    # escape the corpus directory entirely, raising `ValueError` from
    # `with_name` instead of the `CorpusEntryError` every rejection promises.
    expected_body = f"{stem}.body"
    if manifest["body_file"] != expected_body:
        raise CorpusEntryError(
            f"{stem}: body_file is {manifest['body_file']!r}; it must be {expected_body!r}"
        )

    body_path = manifest_path.with_name(expected_body)
    if not body_path.is_file():
        raise CorpusEntryError(f"{stem}: body_file {expected_body!r} does not exist")

    # The digest is what makes "byte-exact" enforceable rather than aspirational.
    # This repository has no `.gitattributes` and already carries mixed CRLF/LF,
    # so a contributor with `core.autocrlf=true` rewrites LF to CRLF inside a
    # `.body` on checkout and commits it back. CI is `ubuntu-latest` only, so the
    # gate would consume the corrupted bytes rather than notice them -- and the
    # damage lands on the byte-level key-order assertion and on every size-derived
    # trigger. `.gitattributes` prevents it; this catches it if prevention fails.
    body = body_path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != manifest["body_sha256"]:
        raise CorpusEntryError(
            f"{stem}: body_sha256 is {manifest['body_sha256']!r} but the body hashes to {digest!r}. "
            "The body has been modified since it was written — check for line-ending translation."
        )

    non_secrets = manifest["known_non_secrets"]
    if not isinstance(non_secrets, list) or any(
        not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(x, str) for x in pair)
        for pair in non_secrets
    ):
        raise CorpusEntryError(f"{stem}: known_non_secrets must be a list of [literal, reason] pairs")
    if any(not reason for _, reason in non_secrets):
        raise CorpusEntryError(f"{stem}: every known_non_secrets entry needs a reason")
    # The floor `_scan` enforces, applied where the entry is authored so the
    # failure names the entry. A cleared literal shorter than this parks a
    # substring of a longer name -- clearing `"key"` breaks `api_key` apart and
    # the credential lint goes silently blind on that entry.
    short = sorted({literal for literal, _ in non_secrets if len(literal) < MIN_LITERAL})
    if short:
        raise CorpusEntryError(
            f"{stem}: known_non_secrets literal(s) {short} are shorter than {MIN_LITERAL} "
            "characters; such a literal matches inside ordinary names and would disable the lint"
        )

    fmt_name = manifest["wire_format"]
    formats = {f.value: f for f in WireFormat}
    if fmt_name is not None and fmt_name not in formats:
        raise CorpusEntryError(f"{stem}: wire_format {fmt_name!r} is not a WireFormat")

    headers = manifest["headers"]
    if not isinstance(headers, list):
        raise CorpusEntryError(f"{stem}: headers must be a list of [name, value] pairs")

    try:
        request = CapturedRequest(
            method=manifest["method"],
            scheme=manifest["scheme"],
            host=manifest["host"],
            path=manifest["path"],
            query=manifest["query"],
            headers=[tuple(pair) for pair in headers],  # type: ignore[misc]
            body=body,
        )
    except TypeError as exc:
        raise CorpusEntryError(f"{stem}: {exc}") from exc

    return CorpusEntry(
        id=stem,
        description=manifest["description"],
        origin=origin,
        origin_note=manifest["origin_note"],
        captured_from=manifest["captured_from"],
        captured_at=manifest["captured_at"],
        request=request,
        wire_format=None if fmt_name is None else formats[fmt_name],
        known_non_secrets=tuple((literal, reason) for literal, reason in non_secrets),
        triggers_met=met,
        triggers_absent=absent,
    )


def load_corpus(root: Path) -> tuple[CorpusEntry, ...]:
    """Read every entry under ``root``.

    Args:
        root: The corpus directory, normally ``tests/corpus``.

    Returns:
        The entries, ordered by id so a failure message is stable across runs.

    Raises:
        CorpusEntryError: When any entry is malformed.
    """
    return tuple(load_entry(path) for path in sorted(root.glob("*.json")))


def write_entry(root: Path, entry: CorpusEntry, *, extra: Sequence[str] = ()) -> Path:
    """Write ``entry`` to ``root`` as a manifest and a sidecar body, scrubbed.

    Scrubbing here rather than at the call site is deliberate: the capture
    procedure has six steps, and the one a tired operator skips must not be the
    one that keeps a key out of a public repository.

    Args:
        root: The corpus directory.
        entry: The entry to write.
        extra: Literal strings to redact, as :func:`scrub` takes them.

    Returns:
        The manifest's path.

    Raises:
        CorpusEntryError: When ``entry.id`` is not a bare file name component.
        EncodedCaptureError: When the capture carries a ``content-encoding`` or
            ``transfer-encoding`` header.
    """
    _checked_id(entry.id)

    encoded = sorted({n for n, _ in entry.request.headers if n.lower() in REFUSED_ENCODINGS})
    if encoded:
        raise EncodedCaptureError(
            f"{entry.id}: capture carries {encoded}. The corpus stores entity bodies, not wire "
            "octets: a compressed body is one the scrubber reads as noise and reports clean."
        )

    scrubbed = scrub(entry.request, extra, [literal for literal, _ in entry.known_non_secrets])
    body_file = f"{entry.id}.body"

    manifest = {
        "id": entry.id,
        "description": entry.description,
        "origin": entry.origin,
        "origin_note": entry.origin_note,
        "captured_from": entry.captured_from,
        "captured_at": entry.captured_at,
        "method": scrubbed.method,
        "scheme": scrubbed.scheme,
        "host": scrubbed.host,
        "path": scrubbed.path,
        "query": scrubbed.query,
        "headers": [list(pair) for pair in scrubbed.headers],
        "body_file": body_file,
        "body_sha256": hashlib.sha256(scrubbed.body).hexdigest(),
        "wire_format": None if entry.wire_format is None else entry.wire_format.value,
        "known_non_secrets": [list(pair) for pair in entry.known_non_secrets],
        "triggers_met": sorted(t.value for t in entry.triggers_met),
        "triggers_absent": sorted(t.value for t in entry.triggers_absent),
    }

    root.mkdir(parents=True, exist_ok=True)
    (root / body_file).write_bytes(scrubbed.body)
    path = root / f"{entry.id}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# The trigger index
# --------------------------------------------------------------------------


def entries_meeting(entries: Iterable[CorpusEntry], trigger: Trigger) -> tuple[CorpusEntry, ...]:
    """Return the entries declaring ``trigger`` met.

    Args:
        entries: The corpus.
        trigger: The register trigger.

    Returns:
        The matching entries, in the order given.
    """
    return tuple(entry for entry in entries if trigger in entry.triggers_met)


def entries_without(entries: Iterable[CorpusEntry], trigger: Trigger) -> tuple[CorpusEntry, ...]:
    """Return the entries declaring ``trigger`` **explicitly** absent.

    Silence is not absence.  §3.3.2's second assertion is only as good as the
    complement it runs against, and an entry whose author never considered this
    trigger has claimed nothing about it — offering it as a complement would
    quantify the assertion over entries nobody vetted.

    Args:
        entries: The corpus.
        trigger: The register trigger.

    Returns:
        The matching entries, in the order given.
    """
    return tuple(entry for entry in entries if trigger in entry.triggers_absent)


# --------------------------------------------------------------------------
# The lint
# --------------------------------------------------------------------------


def captured_only(entries: Iterable[CorpusEntry]) -> tuple[CorpusEntry, ...]:
    """Return the entries that are evidence rather than construction.

    §7.1's rationale is that a captured body is evidence and a hand-written one
    is our belief about what Claude Code sends.  Plan §6 deliberately synthesises
    two Epic C entries, and this task ships a third as a worked example of the
    format — so every query that means "what has the agent actually been
    observed to send" has to exclude them.  A helper does that by construction;
    a convention does it until somebody forgets.

    Args:
        entries: The corpus.

    Returns:
        The captured entries, in the order given.
    """
    return tuple(entry for entry in entries if entry.origin == CAPTURED)


def assert_corpus_clean(entries: Sequence[CorpusEntry]) -> None:
    """Fail unless every entry is free of credentials and identifiers.

    Args:
        entries: The corpus, normally :func:`load_corpus`'s output.

    Raises:
        UnscrubbedCorpusError: When any entry has findings, when an entry's
            ``known_non_secrets`` names a literal the body no longer contains,
            **or** when ``entries`` is empty.

    The empty case is an error rather than a pass.  "No secrets found" is
    satisfied perfectly by having looked at nothing, and a lint in that state is
    indistinguishable from a healthy one —
    :func:`layers.assert_no_layer_violations` sets the precedent.  It also turns
    a second, quieter mistake into a loud one: :func:`load_corpus` takes a root,
    so a caller pointed at the wrong directory gets zero entries, and without
    this rule that misconfiguration would report the corpus clean forever.

    The stale-exemption check is §6.2.3's rule for the ``web.Request``
    exclusion, applied here: an escape hatch that no longer matches anything has
    stopped being an exemption and become a blanket one, and nothing else would
    ever say so.
    """
    if not entries:
        raise UnscrubbedCorpusError(
            "corpus-lint: the scrubber check was handed no entries. It cannot "
            "pass by having nothing to check; either the corpus is empty or "
            "something stopped loading it."
        )

    problems: list[str] = []
    for entry in entries:
        allow = [literal for literal, _ in entry.known_non_secrets]
        problems += [
            f"{entry.id}: {f.name} at {f.where} offset {f.offset}" for f in findings(entry.request, allow)
        ]
        body = entry.request.body.decode("utf-8", errors="surrogateescape")
        problems += [
            f"{entry.id}: known_non_secrets names {literal!r}, which the body no longer contains"
            for literal in allow
            if literal not in body
        ]

    if problems:
        raise UnscrubbedCorpusError(
            "corpus-lint: committed entries carry credentials, identifiers or stale "
            "exemptions:\n  " + "\n  ".join(problems)
        )
