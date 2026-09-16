"""Hypothesis strategies that generate egress-shaped addresses, hostnames and URLs.

`.system_design/TEST_SUITE.md` §5.3, §6.1 · plan **T-F4** (KBR-73).

The egress properties need generators for the input space `should_bypass`,
`parse_proxy_url` and the two redactors read: IP literals in the bypassed
ranges and their IPv4-mapped spellings, hostnames inside and outside the
`localhost` family, and proxy / upstream URLs whose credentials appear in
each of the encodings a user could write. This module provides them.

**Out of scope (the mirror image of the boundary `transcripts.py` records).**

Transcript-shaped strategies — whole requests, messages, tool definitions —
live in `transcripts.py` and are owned by T-F1. This module owns only the
address / hostname / URL shapes the egress properties need; pre-empting a
transcript strategy here would couple unrelated work to a substrate that
does not care about it, exactly as T-F1's docstring argues in the other
direction.

**It imports nothing from ``src/kitty``**, the same posture every harness
module takes (§7 independence rule): a property test whose generator shared
assumptions with the code under test would prove self-consistency rather
than the property. In particular the range tables below are written from
RFC 1918 / RFC 4291 / the loopback and link-local definitions, not read
from ``ipaddress`` at generation time — the address ranges are constants of
the protocols, and reading them through the same module the code under test
uses would let a stdlib change redefine both sides of the property at once.

**What "in range" means, precisely.** A generated address is drawn from the
*interior* of a bypassed range — never its network or broadcast address,
which boundary semantics make ambiguous — and the negative hostname
strategy draws from names that are neither `localhost` nor a `.localhost`
suffix nor resolvable lookalikes, because the property is stated *with*
those exclusions (§5.3: `should_bypass` matches the family by name before
it ever parses an address).
"""

from __future__ import annotations

from hypothesis import strategies as st

__all__ = [
    "BYPASSED_V4_RANGES",
    "hostnames_outside_localhost_family",
    "localhost_family_hostnames",
    "mapped_addresses",
    "private_addresses",
    "proxy_urls_with_credentials",
    "upstream_urls_with_credentials",
]

# ── Address ranges ─────────────────────────────────────────────────────────
#
# The bypassed ranges, as (first, last) dotted-quad strings of each block's
# interior — the first and last address of every block are excluded so the
# strategy never has to reason about network/broadcast semantics. Loopback is
# 127.0.0.0/8 (interior 127.0.0.1-127.255.255.254), the RFC 1918 blocks are
# 10/8, 172.16/12 and 192.168/16, and link-local is 169.254/16. These are the
# five disjuncts `should_bypass` must classify as bypassed.

#: Interior bounds of every IPv4 range ``should_bypass`` must treat as local.
BYPASSED_V4_RANGES: tuple[tuple[str, str], ...] = (
    ("127.0.0.1", "127.255.255.254"),  # loopback
    ("10.0.0.1", "10.255.255.254"),  # RFC 1918 10/8
    ("172.16.0.1", "172.31.255.254"),  # RFC 1918 172.16/12
    ("192.168.0.1", "192.168.255.254"),  # RFC 1918 192.168/16
    ("169.254.0.1", "169.254.255.254"),  # link-local
)


def _dotted_to_int(dotted: str) -> int:
    """Convert a dotted-quad IPv4 string to its 32-bit integer value.

    Args:
        dotted: An address in ``a.b.c.d`` form.

    Returns:
        The unsigned 32-bit integer the four octets encode.
    """
    a, b, c, d = (int(part) for part in dotted.split("."))
    return (a << 24) | (b << 16) | (c << 8) | d


def _int_to_dotted(value: int) -> str:
    """Convert a 32-bit integer to its dotted-quad IPv4 string.

    Args:
        value: An unsigned 32-bit integer.

    Returns:
        The address in ``a.b.c.d`` form.
    """
    return ".".join(str((value >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def private_addresses() -> st.SearchStrategy[str]:
    """Return a strategy drawing dotted-quad IPv4 addresses from the bypassed ranges.

    Samples a range index then a uniform offset within that range, so the
    full set of interior addresses is never materialised in memory. The
    ranges together cover ~33 million addresses, which a ``sampled_from``
    over the enumerated list would carry as a Python list of integers.

    Returns:
        A strategy whose examples are interior addresses of the loopback,
        RFC 1918 and link-local blocks, in dotted-quad form.
    """
    bounds = [(_dotted_to_int(first), _dotted_to_int(last)) for first, last in BYPASSED_V4_RANGES]
    return (
        st.sampled_from(bounds)
        .flatmap(lambda pair: st.integers(min_value=pair[0], max_value=pair[1]))
        .map(_int_to_dotted)
    )


def mapped_addresses() -> st.SearchStrategy[str]:
    """Return a strategy drawing IPv4-mapped IPv6 literals of the bypassed ranges.

    §6.1 names the mapped form explicitly: ``ipaddress.ip_address`` accepts
    `::ffff:a.b.c.d`, and a strategy emitting only the dotted form would
    leave ``should_bypass("::ffff:10.0.0.5")`` untested.

    Returns:
        A strategy whose examples are ``::ffff:a.b.c.d`` strings for
        addresses in the bypassed ranges.
    """
    return private_addresses().map(lambda dotted: f"::ffff:{dotted}")


# ── Hostnames ──────────────────────────────────────────────────────────────

#: Second-level labels for generated hostnames. Drawn from dictionary words
#: unlikely to collide with `localhost` by construction, not by filtering —
#: a filter would hide a generator that had started emitting the family it
#: was supposed to avoid.
_OUTSIDE_LABELS = ("upstream", "api", "gateway", "ingress", "relay", "edge")

#: Public-style suffixes for generated hostnames. `.invalid` is guaranteed
#: never to resolve publicly (RFC 2606), which is what keeps a generated
#: name from escaping the test environment even if something did resolve it.
_PUBLIC_SUFFIXES = ("invalid", "example", "test")

#: The exact two spellings of the `localhost` family `should_bypass` matches
#: by name, before any address parsing: the bare name and the suffix form.
_LOCALHOST_FAMILY = ("localhost", ".localhost")


def localhost_family_hostnames() -> st.SearchStrategy[str]:
    """Return a strategy drawing hostnames inside the `localhost` family.

    These are the inputs `should_bypass` classifies by *name* (§5.3). They
    are consumed by the example-level assertions around the family, not by
    the negative property — the negative property is stated *with* this
    family excluded, and this strategy is what makes the exclusion concrete.

    Returns:
        A strategy whose examples are `localhost` and `something.localhost`
        strings.
    """
    label = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=12).filter(
        lambda s: not s.startswith("-") and not s.endswith("-")
    )
    return st.sampled_from(list(_LOCALHOST_FAMILY)) | label.map(lambda s: f"{s}.localhost")


def hostnames_outside_localhost_family() -> st.SearchStrategy[str]:
    """Return a strategy drawing hostnames that are not in the `localhost` family.

    The §5.3 property keeps the containment harness honest: for every
    hostname that is neither an IP literal nor `localhost` nor a
    `.localhost` suffix, `should_bypass` must return False. The generator
    guarantees its side of the contract structurally — generated labels
    cannot spell `localhost`, and no example carries a `localhost` suffix —
    so a failure of the property is a failure of the code, not of the
    generator's filtering.

    Returns:
        A strategy whose examples are `label.suffix` strings, none of which
        is or ends with `localhost`.
    """
    label = st.sampled_from(_OUTSIDE_LABELS)
    prefix = st.sampled_from(_OUTSIDE_LABELS)
    suffix = st.sampled_from(_PUBLIC_SUFFIXES)
    return st.builds(lambda a, b, c: f"{a}.{b}.{c}", label, prefix, suffix)


# ── Proxy URLs ─────────────────────────────────────────────────────────────

#: Alphabetic strokes for generated credential strings. Kept alphanumeric so
#: a generated password can be embedded in a URL without further escaping
#: unless the form under test asks for encoding.
_CRED_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

#: Fixed prefix and suffix wrapped around every sentinel credential. The
#: ``SECRET_`` / ``_END`` markers are absent from any host, path, query,
#: or fragment the strategy generates, so a sentinel cannot collide with
#: any other field and the absence-of-secret assertion is sound
#: regardless of how long the inner body is.
_SENTINEL_PREFIX = "SECRET_"
_SENTINEL_SUFFIX = "_END"

#: The characters whose URL-encoding the round-trip property exercises. `@`
#: and `:` are the two that break a naive userinfo split; `/` is included
#: because a decoder that re-escapes rather than unescapes fails on it.
_ENCODED_STROKES = ("@", ":", "/")


def _sentinel_text() -> st.SearchStrategy[str]:
    """Return a strategy drawing distinctive, collision-free credentials.

    The body is short alphanumeric text wrapped in fixed ``SECRET_`` /
    ``_END`` markers so no host, path, query or fragment the other
    strategies generate can contain the string. Length 8-16 chars is
    enough to make the inner body distinctive within the markers —
    the markers are what keep it collision-free, not the length.

    Returns:
        A strategy whose examples are ``SECRET_<body>_END`` strings.
    """
    return st.text(alphabet=_CRED_ALPHABET, min_size=8, max_size=16).map(
        lambda body: f"{_SENTINEL_PREFIX}{body}{_SENTINEL_SUFFIX}"
    )


def _credential_text() -> st.SearchStrategy[str]:
    """Return a strategy drawing distinctive credential strings.

    Length 12-24 rather than shorter: §6.1's redaction guidance says the
    sweep should "generate a distinctive sentinel that cannot collide with
    any other field". The remaining substring assertions in the properties
    hold only if a generated credential cannot coincidentally appear inside
    a generated host or path, and a longer alphabet-driven string makes
    that collision vanishingly unlikely without banning it by fiat.

    Returns:
        A strategy whose examples are 12-24 character alphanumeric strings.
    """
    return st.text(alphabet=_CRED_ALPHABET, min_size=12, max_size=24)


def _encoded_credential_text() -> st.SearchStrategy[str]:
    """Return a strategy drawing credentials that contain `@`, `:` or `/`.

    Returns:
        A strategy whose examples pair the cleartext password with its
        percent-encoded URL form, so the round-trip property can assert the
        *decoded* value came back.
    """
    raw = _credential_text()
    stroke = st.sampled_from(_ENCODED_STROKES)
    return st.tuples(raw, stroke, raw).map(
        lambda parts: (f"{parts[0]}{parts[1]}{parts[2]}", f"{parts[0]}%{ord(parts[1]):02X}{parts[2]}")
    )


def proxy_urls_with_credentials() -> st.SearchStrategy[tuple[str, str, str]]:
    """Return a strategy drawing proxy URLs with credentials, in all three forms.

    §6.1: "test percent-encoded and URL-embedded forms as separate cases."
    The strategy produces the 4-tuple ``(proxy_origin, username,
    password, written_url)`` the round-trip property asserts against,
    alongside the URL in one of the three writings a user could put in the
    environment or a profile: cleartext userinfo, percent-encoded
    userinfo, and the bare URL (no credentials — the credential
    components are ``None``).

    Returns:
        A strategy whose examples are 4-tuples
        ``(proxy_origin, username, password, written_url)`` where
        ``written_url`` embeds the credentials in one of the three forms
        and ``proxy_origin`` is the scheme/host/port part the parser
        must preserve.
    """
    origin = st.builds(
        lambda scheme, label, port: f"{scheme}://{label}.proxy.invalid:{port}",
        st.sampled_from(("http", "https")),
        st.sampled_from(_OUTSIDE_LABELS),
        st.integers(min_value=1, max_value=65535),
    )
    username = _credential_text()

    cleartext = st.tuples(origin, username, _credential_text()).map(
        lambda t: (t[0], t[1], t[2], f"{t[0].partition('://')[0]}://{t[1]}:{t[2]}@{t[0].partition('://')[2]}")
    )
    encoded = st.tuples(origin, username, _encoded_credential_text()).map(
        lambda t: (
            t[0],
            t[1],
            t[2][0],
            f"{t[0].partition('://')[0]}://{t[1]}:{t[2][1]}@{t[0].partition('://')[2]}",
        )
    )
    bare = origin.map(lambda u: (u, None, None, u))

    return cleartext | encoded | bare


# ── Upstream URLs for the second redactor ──────────────────────────────────

#: Query parameter names the upstream-URL strategy draws from. `api-key` and
#: `sig` are the credential-bearing names gateways actually use; `debug` is
#: the valueless parameter KBR-143's rule leaves as written; `api-version`
#: is the routing parameter whose name merely *contains* nothing special but
#: whose value a redaction must still mask.
_QUERY_NAMES = ("api-key", "sig", "code", "debug", "api-version")


def upstream_urls_with_credentials() -> st.SearchStrategy[dict[str, str | None]]:
    """Return a strategy drawing upstream URLs exercising every redactor shape.

    Each example is a **record** — a dict mapping a shape name to the URL
    fragment exhibiting it — because the property is stated per component
    ("no component that can carry a credential is returned verbatim") and a
    single monolithic URL would make a failure hard to attribute. The keys
    name the shape; the values are the URL to redact. A shape whose value
    is ``None`` is absent from that example.

    Returns:
        A strategy of dicts with some of the keys ``userinfo``, ``query_value``,
        ``valueless_param``, ``overlapping_param``, ``unparseable``, ``no_authority``,
        ``fragment``, each mapping to a URL exercising that shape.
    """
    host = st.sampled_from(_OUTSIDE_LABELS).map(lambda label: f"{label}.upstream.invalid")
    password = _credential_text()
    user = _credential_text()
    query_name = st.sampled_from(_QUERY_NAMES)
    path = st.sampled_from(("/v1/messages", "/v1/chat/completions", "/v1/responses", ""))

    userinfo = st.builds(
        lambda scheme, u, p, h, pth: {
            "userinfo": f"{scheme}://{u}:{p}@{h}{pth}",
            "expected_userinfo": (u, p),
        },
        st.sampled_from(("https", "http")),
        user,
        password,
        host,
        path,
    )

    multi_at = st.builds(
        lambda scheme, u, p1, p2, h, pth: {
            "userinfo": f"{scheme}://{u}:{p1}@{p2}@{h}{pth}",
            "expected_userinfo": (u, f"{p1}@{p2}"),
        },
        st.sampled_from(("https", "http")),
        user,
        password,
        password,
        host,
        path,
    )

    query_value = st.builds(
        lambda scheme, h, pth, name, value: {
            "query_value": f"{scheme}://{h}{pth}?{name}={value}",
            "expected_query": (name, value),
        },
        st.sampled_from(("https",)),
        host,
        path,
        query_name.filter(lambda n: n != "debug"),
        _credential_text(),
    )

    valueless = st.builds(
        lambda scheme, h, pth: {
            "valueless_param": f"{scheme}://{h}{pth}?debug",
            "expected_valueless": "debug",
        },
        st.sampled_from(("https",)),
        host,
        path,
    )

    overlapping = st.builds(
        lambda scheme, h, pth, name: {
            "overlapping_param": f"{scheme}://{h}{pth}?{name}x=1&{name}=2",
            "expected_overlapping": name,
        },
        st.sampled_from(("https",)),
        host,
        path,
        query_name.filter(lambda n: len(n) >= 3),
    )

    # Malformed IPv6 brackets trip ``urlsplit``; the helper then redacts
    # textually (deliberately over-broad).  The URL carries a sentinel
    # secret so the property can assert ``secret not in redacted``
    # structurally — the sentinel markers (``SECRET_`` / ``_END``) are
    # absent from any host, path, query, or fragment the strategy emits,
    # so an absence check is collision-free.
    unparseable_query = _sentinel_text().map(
        lambda secret: {
            "unparseable": f"https://[::1/v1/messages?code={secret}",
            "expected_unparseable_secret": secret,
        }
    )
    unparseable_userinfo = _sentinel_text().map(
        lambda secret: {
            "unparseable": f"https://[{secret}@host/v1",
            "expected_unparseable_secret": secret,
        }
    )
    unparseable = unparseable_query | unparseable_userinfo

    # Same sentinel strategy for the no-authority shape: credentials live
    # in the path (out of reach of any netloc rule), and a sentinel
    # witness lets the property assert the password is gone without
    # depending on the helper's exact textual output.
    no_authority = st.builds(
        lambda scheme, u, p, h, pth, sentinel: {
            "no_authority": f"{scheme}:{u}:{p}@{h}{pth}",
            "expected_no_authority_password": sentinel,
        },
        st.sampled_from(("u", "x")),
        user,
        password,
        host,
        path,
        _sentinel_text(),
    )

    # Fragment is masked wholesale — the structural property asserts the
    # output's fragment is exactly ``****``, which is sufficient to pin
    # the secret's removal without a separate expected-* field.
    fragment = st.builds(
        lambda scheme, h, pth, secret: {"fragment": f"{scheme}://{h}{pth}#token={secret}"},
        st.sampled_from(("https",)),
        host,
        path,
        _credential_text(),
    )

    return st.one_of(userinfo, multi_at, query_value, valueless, overlapping, unparseable, no_authority, fragment)
