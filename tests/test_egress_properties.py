"""Property tests for the egress module (KBR-73 / T-F4).

`.system_design/TEST_SUITE.md` §5.3, §6.1 · plan **T-F4**.

Five properties exercise the egress module so the L1 selection has
property coverage to back its example-based tests:

* **P1** — every address in the bypassed ranges, and the IPv4-mapped
  form of each, is classified as bypassed by :func:`should_bypass`.
* **P2** — no hostname outside the ``localhost`` family is bypassed;
  stated **with** the family excluded, because ``should_bypass`` matches
  it by name before any address parsing (§5.3).
* **P3** — :func:`parse_proxy_url` round-trips credentials, including
  percent-encoded ``@`` and ``:``.
* **P4** — :meth:`EgressConfig.masked` redacts the password **structurally**
  (parse the masked output, assert the password component is exactly the
  mask) — never a substring test, which false-fails on password ``proxy``
  against host ``proxy.example`` (§6.1).
* **P5** — :meth:`ProviderAdapter.redact_url_for_display` does not return
  any credential-bearing component verbatim, per the shape the URL
  exhibits. A single substring invariant would false-fail on the two
  deliberate non-redactions (the valueless ``?debug`` parameter, the
  parameter name containing another), so the property is per-shape.

Two further L1 tests verify the DEBUG-log redaction helpers by attaching a
``StringIO`` handler to ``logging.getLogger("kitty.bridge")`` and asserting
the formatted record carries the redacted text. This closes the
helper→artifact loop that the structural guard cannot see without sockets.

The strategies are local to :mod:`harness.egress` (transcripts.py's mirror
boundary). They generate hard-coded boundary literals rather than
deriving ranges from ``ipaddress`` at runtime — the strategy must not share
a code path with the function under test (§7 independence rule).
"""

from __future__ import annotations

import logging
from io import StringIO
from urllib.parse import parse_qsl, urlsplit

import pytest
from harness import egress as egr
from hypothesis import given, settings

from kitty.egress import parse_proxy_url, should_bypass
from kitty.providers.base import ProviderAdapter

# L1: this file's subject is the egress module's pure logic plus the
# DEBUG-log redaction helpers it consumes. The marker is set explicitly
# rather than via the path default because the path-default rule applies
# to existing files; a new file declares its own layer.
pytestmark = pytest.mark.l1


# ── P1 — every bypassed address is classified as bypassed ────────────────

@given(egr.private_addresses())
@settings(max_examples=200)
def test_p1_every_dotted_quad_in_a_bypassed_range_is_bypassed(url_host: str) -> None:
    """Every interior address of a bypassed IPv4 range is bypassed.

    Args:
        url_host: The dotted-quad address, formatted into a ``http://`` URL.
    """
    assert should_bypass(f"http://{url_host}") is True


@given(egr.mapped_addresses())
@settings(max_examples=200)
def test_p1_every_ipv4_mapped_form_of_a_bypassed_range_is_bypassed(url_host: str) -> None:
    """The IPv4-mapped form of every bypassed range is still classified as bypassed.

    Args:
        url_host: The mapped literal (``::ffff:a.b.c.d``), formatted into a URL.
    """
    assert should_bypass(f"http://[{url_host}]/v1") is True


# ── P2 — no hostname outside the ``localhost`` family is bypassed ───────

@given(egr.hostnames_outside_localhost_family())
@settings(max_examples=200)
def test_p2_no_hostname_outside_the_localhost_family_is_bypassed(host: str) -> None:
    """No public-style hostname outside the ``localhost`` family is bypassed.

    Args:
        host: A hostname the generator structurally cannot place in the
            ``localhost`` family — labels drawn from ``(upstream, api,
            gateway, ...)``, suffixes drawn from ``(invalid, example,
            test)``.
    """
    assert should_bypass(f"https://{host}/v1") is False


# Stated with the family excluded, this property has a deliberate
# counterpart that anchors the boundary: every `localhost`-family hostname
# the strategy emits — the bare name and any `*.localhost` suffix — must
# be bypassed.  The test consumes the strategy directly rather than
# re-listing spellings by hand; case-insensitivity is pinned by the
# example tests in `tests/test_egress.py`.
@given(egr.localhost_family_hostnames().map(lambda host: f"http://{host}/v1"))
@settings(max_examples=20)
def test_p2_localhost_family_is_bypassed(url: str) -> None:
    """Every ``localhost``-family hostname the strategy emits is bypassed.

    Args:
        url: A URL whose hostname is ``localhost`` or a ``.localhost`` suffix.
    """
    assert should_bypass(url) is True


# ── P3 — parse_proxy_url round-trip with credentials ─────────────────────

@given(egr.proxy_urls_with_credentials())
@settings(max_examples=200)
def test_p3_parse_proxy_url_round_trips_credentials(proxy_tuple: tuple[str, str | None, str | None, str]) -> None:
    """``parse`` of the written URL yields the credentials it was given.

    Args:
        proxy_tuple: ``(proxy_origin, username, password, written_url)``.
            ``proxy_origin`` is the scheme/host/port part; ``written_url``
            embeds credentials in one of cleartext, percent-encoded, or
            bare forms (the bare form has both credentials ``None``).
    """
    origin, username, password, written = proxy_tuple

    cfg = parse_proxy_url(written)

    assert cfg.proxy_url == origin
    assert cfg.username == username
    assert cfg.password == password


# ── P4 — masked() redaction is structural, not substring-based ────────────

@given(egr.proxy_urls_with_credentials())
@settings(max_examples=200)
def test_p4_masked_password_component_is_exactly_the_mask(proxy_tuple: tuple[str, str | None, str | None, str]) -> None:
    """The password component of ``masked()`` is the mask, parsed out structurally.

    Args:
        proxy_tuple: ``(proxy_origin, username, password, written_url)``,
            same shape as P3's.
    """
    origin, username, password, written = proxy_tuple
    if username is None or password is None:
        # Bare-origin case — no masking happens, no password component to
        # parse. Covered by P3 already; nothing to assert here.
        return

    cfg = parse_proxy_url(written)
    masked = cfg.masked()

    parts = urlsplit(masked)
    assert "@" in parts.netloc, (
        f"masked output {masked!r} lost the userinfo separator; cannot "
        "parse the password component to assert it"
    )
    masked_userinfo, _, masked_host = parts.netloc.rpartition("@")
    masked_user, _, masked_password = masked_userinfo.partition(":")

    # Structural: parse the masked output and assert the password
    # component is exactly the mask. This is the §6.1-required form; a
    # substring check ("password not in masked") false-fails when the
    # password equals the username — the username legitimately survives —
    # or when the password is a substring of the host (password ``proxy``,
    # host ``proxy.example``).
    assert masked_user == username
    assert masked_password == "****"
    assert masked_host == urlsplit(origin).netloc, (
        f"masked output altered the host: {masked_host!r} != {urlsplit(origin).netloc!r}"
    )


# ── P5 — redact_url_for_display per-shape post-conditions ────────────────


def _assert_userinfo_redacted(redacted: str, original_url: str, password: str, username: str) -> None:
    """Assert the redacted URL carries no userinfo, by structure not substring.

    The structural form: the redacted netloc must equal the original
    netloc minus its userinfo (everything up to the last ``@``). Parsing
    both sides and comparing means a credential that happens to look like
    a substring of the host cannot false-fail the property — the same
    §6.1 warning P4's comment records ("password ``proxy``, host
    ``proxy.example``").

    Args:
        redacted: The output of ``redact_url_for_display``.
        original_url: The unredacted input, whose netloc supplies the
            host the output must be reduced to.
        password: The password that was in the input userinfo.
        username: The username that was in the input userinfo.
    """
    del password, username  # witnesses of what the userinfo carried; the structural check below is the assertion
    parts = urlsplit(redacted)
    original_netloc = urlsplit(original_url).netloc
    assert "@" in original_netloc, (
        f"strategy produced a userinfo URL without an @ in its netloc: {original_url!r}"
    )
    bare_host = original_netloc.rsplit("@", 1)[1]
    assert parts.netloc == bare_host, (
        f"redacted netloc {parts.netloc!r} != the bare host {bare_host!r} "
        f"the userinfo must be reduced to (input {original_url!r})"
    )


def _query_pairs(query: str) -> list[tuple[str, str | None]]:
    """Parse a query string into ``(name, value-or-None)`` pairs preserving order.

    Args:
        query: The raw query string (without the leading ``?``).

    Returns:
        The list of pairs in the order the source string names them. A
        valueless parameter yields ``(name, None)``.
    """
    # ``keep_blank_values`` keeps a ``?debug`` shape as ``("debug", "")``;
    # we map ``""`` to ``None`` so the test asserts the contract's intended
    # representation.
    return [(name, value if value else None) for name, value in parse_qsl(query, keep_blank_values=True)]


@given(egr.upstream_urls_with_credentials())
@settings(max_examples=200)
def test_p5_redact_url_for_display_per_shape(shapes: dict[str, str]) -> None:
    """One shape per example; each shape has its own post-condition.

    The strategy emits a single shape key per example (``st.one_of`` over
    disjoint strategies), so a failure attributes to one shape rather
    than collapsing the whole property.

    Args:
        shapes: Dict with one key naming the shape, the value being the
            URL to redact. The expected counterparts (``expected_userinfo``,
            ``expected_query`` etc.) ride along as supplementary keys.
    """
    for key, url in shapes.items():
        if key.startswith("expected_"):
            continue
        redacted = ProviderAdapter.redact_url_for_display(url)
        if key == "userinfo":
            # The strategy carries the (username, password) pair it built
            # the URL from — consume it directly rather than re-deriving
            # the split from the input, which is how the userinfo parse
            # could share a bug with the redactor under test.
            username, password = shapes["expected_userinfo"]
            _assert_userinfo_redacted(redacted, url, password, username)
        elif key == "query_value":
            expected_name = shapes["expected_query"][0]
            pairs = _query_pairs(urlsplit(redacted).query)
            masked = {name: value for name, value in pairs if value is not None}
            assert expected_name in masked, (
                f"expected parameter {expected_name!r} missing from "
                f"redacted URL {redacted!r}"
            )
            assert masked[expected_name] == "****", (
                f"value of {expected_name!r} not masked; got "
                f"{masked[expected_name]!r} in redacted URL {redacted!r}"
            )
        elif key == "valueless_param":
            assert shapes["expected_valueless"] in urlsplit(redacted).query, (
                f"valueless parameter {shapes['expected_valueless']!r} "
                f"was masked; expected verbatim survival in {redacted!r}"
            )
        elif key == "overlapping_param":
            name = shapes["expected_overlapping"]
            pairs = dict(_query_pairs(urlsplit(redacted).query))
            assert f"{name}x" in pairs, (
                f"parameter {name!r}x not preserved as a distinct name in "
                f"redacted URL {redacted!r}"
            )
            assert pairs[f"{name}x"] == "****"
            assert name in pairs, (
                f"parameter {name!r} not preserved as a distinct name in "
                f"redacted URL {redacted!r}"
            )
            assert pairs[name] == "****"
        elif key == "unparseable":
            # The contract: the sentinel secret must not survive. The
            # strategy's sentinel (``SECRET_<body>_END``) is structurally
            # absent from every other field the strategy generates, so
            # this absence check cannot false-fail on a coincidental
            # substring of the host — and it asserts the property ("no
            # credential verbatim") rather than the helper's private
            # delegate, which a regression could satisfy on both sides
            # at once.
            secret = shapes["expected_unparseable_secret"]
            assert secret not in redacted, (
                f"sentinel secret {secret!r} survived textual redaction; "
                f"got {redacted!r}"
            )
        elif key == "no_authority":
            # Same sentinel reasoning: the password lives in the path,
            # out of reach of any netloc rule, and the sentinel witness
            # lets the property assert its absence without depending on
            # the helper's exact textual output.
            password = shapes["expected_no_authority_password"]
            assert password not in redacted, (
                f"sentinel password {password!r} survived in no-authority "
                f"redacted URL {redacted!r}"
            )
        elif key == "fragment":
            # Structural: the fragment is replaced wholesale by the mask.
            assert urlsplit(redacted).fragment == "****", (
                f"fragment not masked wholesale; got {urlsplit(redacted).fragment!r}"
            )


# ── L1 behavioural: the DEBUG-log redaction helpers ─────────────────────
#
# These tests attach a StringIO handler to ``kitty.bridge`` and assert the
# formatted record carries the redacted text — closing the
# helper→artifact loop the structural guard cannot.


def _bridge_logger_capture() -> tuple[StringIO, int]:
    """Attach a fresh ``StringIO`` handler to ``kitty.bridge`` for one test.

    Returns:
        The buffer the handler writes formatted records into, plus the
        logger's prior level — the caller restores it on teardown so
        one test does not silently change the level another test sees.
    """
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("kitty.bridge")
    prior_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return buffer, prior_level


def _drain_handler(buffer: StringIO, prior_level: int) -> str:
    """Return the formatted record text, detach the handler, restore the level.

    Args:
        buffer: The buffer returned by :func:`_bridge_logger_capture`.
        prior_level: The level recorded before the test attached its handler.
    """
    logger = logging.getLogger("kitty.bridge")
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and handler.stream is buffer:
            logger.removeHandler(handler)
            handler.close()
            break
    logger.setLevel(prior_level)
    return buffer.getvalue()


@given(
    egr.upstream_urls_with_credentials()
    .filter(lambda d: "userinfo" in d)
    .map(lambda d: d["userinfo"])
)
@settings(max_examples=200)
def test_debug_url_helper_redacts_the_emitted_record(url: str) -> None:
    """``_debug_url`` produces a redacted record when the bridge logs through it.

    Args:
        url: An upstream URL whose userinfo the helper must strip, drawn from
            the strategy's userinfo-bearing shapes only — the other shapes
            (query, fragment, unparseable, no_authority) belong to a
            different code path and would collapse to the same fallback
            here, turning the property into a constant-input soak.
    """
    from kitty.bridge.server import BridgeServer  # noqa: PLC0415 — import here to surface the helper's home

    buffer, prior_level = _bridge_logger_capture()
    try:
        logging.getLogger("kitty.bridge").debug("Upstream POST → %s", BridgeServer._debug_url(url))
        rendered = buffer.getvalue()
    finally:
        _drain_handler(buffer, prior_level)

    parts = urlsplit(url)
    expected_userinfo = parts.netloc.partition("@")[0]
    assert expected_userinfo not in rendered, (
        f"userinfo {expected_userinfo!r} survived in the DEBUG record: {rendered!r}"
    )


def test_debug_headers_helper_redacts_the_emitted_record() -> None:
    """``_debug_headers`` produces a redacted record when the bridge logs through it.

    Closes the helper→artifact loop for the header redaction. The dict-
    level assertions prove the helper returns the right dict; this test
    proves a ``logger.debug("... headers: %s", _debug_headers(...))``
    call writes a record whose formatted text carries the mask in place
    of every credential-bearing header value.
    """
    from kitty.bridge.server import BridgeServer  # noqa: PLC0415

    headers = {
        "authorization": "Bearer super-secret-bridge-key",
        "x-goog-api-key": "AIza-fake-google-key",
        "content-type": "application/json",
        "anthropic-version": "2024-02-01",
    }

    buffer, prior_level = _bridge_logger_capture()
    try:
        logging.getLogger("kitty.bridge").debug(
            "Request headers: %s", BridgeServer._debug_headers(headers)
        )
        rendered = buffer.getvalue()
    finally:
        _drain_handler(buffer, prior_level)

    assert "super-secret-bridge-key" not in rendered, (
        f"authorization value survived in the DEBUG record: {rendered!r}"
    )
    assert "AIza-fake-google-key" not in rendered, (
        f"x-goog-api-key value survived in the DEBUG record: {rendered!r}"
    )
    assert "application/json" in rendered, (
        f"non-sensitive header content-type was masked; got {rendered!r}"
    )
    assert "2024-02-01" in rendered, (
        f"non-sensitive header anthropic-version was masked; got {rendered!r}"
    )


def test_debug_headers_helper_redacts_authorization_but_keeps_content_type() -> None:
    """``_debug_headers`` masks ``authorization`` and friends, keeps the rest verbatim.

    Asserts the substring-match rule directly: a header whose name contains
    ``auth`` is masked; one whose name does not is preserved.
    """
    from kitty.bridge.server import BridgeServer  # noqa: PLC0415

    headers = {
        "authorization": "Bearer super-secret-bridge-key",
        "content-type": "application/json",
        "x-goog-api-key": "AIza-fake-google-key",
        "x-custom-id": "12345",
        "anthropic-version": "2024-02-01",
    }

    redacted = BridgeServer._debug_headers(headers)

    assert redacted["authorization"] == "****"
    assert redacted["x-goog-api-key"] == "****"
    assert redacted["content-type"] == "application/json"
    assert redacted["x-custom-id"] == "12345"
    assert redacted["anthropic-version"] == "2024-02-01"
    assert "super-secret-bridge-key" not in str(redacted)
    assert "AIza-fake-google-key" not in str(redacted)


def test_debug_headers_helper_is_case_insensitive() -> None:
    """Header names are matched case-insensitively per RFC 9110."""
    from kitty.bridge.server import BridgeServer  # noqa: PLC0415

    redacted = BridgeServer._debug_headers({"Authorization": "Bearer s3cr3t"})

    assert redacted["Authorization"] == "****"
