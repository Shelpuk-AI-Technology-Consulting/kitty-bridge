"""The conformance checks' own falsification suite.

`.system_design/TEST_SUITE.md` §7.2 · plan task **T-W4** (KBR-27), rule §1.4.

Plan §1.4 exists because four review rounds produced four harnesses that would
have passed while proving nothing. The checks in :mod:`harness.recorder_conformance`
are the thing every Epic B recorder is judged by, so they are themselves given
deliberate defects here: each check is handed a :class:`~harness.contract.CapturedRequest`
that is wrong in exactly the way that check exists to notice, and must reject it.

These tests need no recorder and no socket. That is the point of the checks being
pure functions over data — a decision entangled with a live server could not be
given a falsification case at all, which is the reasoning ``tests/layers.py``
records for the same choice.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from harness.contract import CapturedRequest
from harness.recorder_conformance import (
    CHECK_NAMES,
    PER_EXCHANGE_CHECKS,
    PER_SESSION_CHECKS,
    Recording,
    SentRequest,
    _parse_sent,
    check_arrival_increases,
    check_body,
    check_connection_logged,
    check_header_casing,
    check_header_duplicates,
    check_header_order,
    check_host,
    check_method,
    check_path,
    check_peer_port,
    check_query,
    check_request_count,
    check_request_order,
    check_scheme,
    correlate,
)

# A probe rich enough that every per-exchange check has something to lose:
# mixed casing, a duplicate, an obs-text value, a percent-encoded path whose
# escapes differ in case, and a query that is neither sorted nor decodable
# without changing it.
_RAW = (
    b"POST /v1/mess%61ges/m1/a%2Fb%2fc?Beta=A%20B&key=k&beta=c HTTP/1.1\r\n"
    b"Host: Upstream.Example:443\r\n"
    b"X-Api-Key: secret\r\n"
    b"anthropic-version: 2023-06-01\r\n"
    b"ANTHROPIC-BETA: one\r\n"
    b"anthropic-beta: two\r\n"
    b"X-Weird: caf\xe9\r\n"
    b"X-Probe-Marker: m1\r\n"
    b"Content-Length: 7\r\n"
    b"\r\n"
    b'{"a":1}'
)

_SCHEME = "http"


@pytest.fixture
def sent() -> SentRequest:
    """Return the reference request the conforming capture is built from.

    Returns:
        A :class:`SentRequest` parsed from :data:`_RAW`, as the driver would
        produce it.
    """
    return _parse_sent(_RAW, source_port=54321, marker="m1")


@pytest.fixture
def captured(sent: SentRequest) -> CapturedRequest:
    """Return a capture that is correct in every respect.

    Every falsification case below mutates exactly one field of this, so a case
    that fails proves the check noticed *that* field and not some other
    difference.

    Args:
        sent: The reference request.

    Returns:
        A conforming :class:`CapturedRequest`.
    """
    return CapturedRequest(
        method=sent.method,
        scheme=_SCHEME,
        host=sent.header("Host") or "",
        path=sent.path,
        query=sent.query,
        headers=sent.headers,
        body=sent.body,
        arrival=100.0,
        peer_port=sent.source_port,
    )


def test_the_check_list_is_the_fourteen_the_design_specifies() -> None:
    """Guard the list itself, because every other assertion counts against it.

    R4.2 asserts a defect trips its *named* check and no other. That sentence is
    only meaningful against a written list, so a check silently dropped from the
    tuples would weaken the whole falsification suite without failing anything.
    """
    assert len(CHECK_NAMES) == 14, f"expected 14 checks, found {len(CHECK_NAMES)}: {CHECK_NAMES}"
    assert len(set(CHECK_NAMES)) == 14, f"duplicate check names: {CHECK_NAMES}"
    assert len(PER_EXCHANGE_CHECKS) == 10
    assert len(PER_SESSION_CHECKS) == 4


class TestEveryPerExchangeCheckPassesAConformingCapture:
    """The positive control for the ten per-exchange checks.

    Without it, a check that had stopped comparing anything would read as a
    clean bill of health forever — and every falsification case below would
    still pass, because they only require a check to *fail*.
    """

    @pytest.mark.parametrize("check", PER_EXCHANGE_CHECKS, ids=lambda c: c.__name__)
    def test_check_accepts_a_faithful_capture(
        self, check, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert a correct capture is not rejected.

        Args:
            check: One of the ten per-exchange checks.
            captured: A conforming capture.
            sent: The request that produced it.
        """
        check(captured, sent, scheme=_SCHEME)


class TestEachPerExchangeCheckRejectsItsOwnDefect:
    """Each check, handed the one defect it exists to notice."""

    def test_method_check_rejects_a_changed_method(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert a rewritten method is caught.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        with pytest.raises(AssertionError, match="method"):
            check_method(replace(captured, method="GET"), sent, scheme=_SCHEME)

    def test_scheme_check_rejects_a_scheme_the_recorder_does_not_serve(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert a capture claiming the wrong scheme is caught.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        with pytest.raises(AssertionError, match="scheme"):
            check_scheme(replace(captured, scheme="https"), sent, scheme=_SCHEME)

    def test_host_check_rejects_a_lower_cased_authority(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert the authority's casing is defended.

        ``request.url`` lower-cases the host, so this is the shape a recorder
        built on it produces.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        with pytest.raises(AssertionError, match="host"):
            check_host(replace(captured, host="upstream.example:443"), sent, scheme=_SCHEME)

    def test_host_check_rejects_an_invented_authority_when_none_was_sent(self) -> None:
        """Assert a fabricated host is caught on a request that sent no ``Host``.

        This is the defect aiohttp's ``request.host`` produces by itself: with no
        ``Host`` header it returns ``socket.getfqdn()``, the machine's own name.
        It is invisible on any probe that *does* send a ``Host``, which is why
        the falsification matrix aims this check at the no-``Host`` probe.
        """
        raw = b"POST /m2/nohost HTTP/1.0\r\nX-Probe-Marker: m2\r\nContent-Length: 1\r\n\r\nx"
        hostless = _parse_sent(raw, source_port=1, marker="m2")
        fabricated = CapturedRequest(
            method="POST", scheme=_SCHEME, host="build-runner-07", path="/m2/nohost",
            query="", headers=hostless.headers, body=b"x", arrival=1.0, peer_port=1,
        )
        with pytest.raises(AssertionError, match="invented"):
            check_host(fabricated, hostless, scheme=_SCHEME)

    def test_path_check_rejects_a_percent_decoded_path(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert a decoded path is caught.

        ``request.path`` decodes, so this is what a recorder built on it records.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        with pytest.raises(AssertionError, match="path"):
            check_path(replace(captured, path="/v1/messages/m1/a/b/c"), sent, scheme=_SCHEME)

    def test_path_check_rejects_case_folded_escapes(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert re-quoting that normalises ``%2f`` to ``%2F`` is caught.

        A decode check alone would not see this: the path is still encoded, and
        still decodes to the same string.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        folded = captured.path.replace("%2f", "%2F")
        assert folded != captured.path, "the probe must contain a lower-case escape to fold"
        with pytest.raises(AssertionError, match="path"):
            check_path(replace(captured, path=folded), sent, scheme=_SCHEME)

    def test_query_check_rejects_a_percent_decoded_query(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert a decoded query is caught.

        ``request.query_string`` decodes, turning ``%20`` into a space.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        with pytest.raises(AssertionError, match="query"):
            check_query(replace(captured, query="Beta=A B&key=k&beta=c"), sent, scheme=_SCHEME)

    def test_query_check_rejects_a_dropped_query(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert losing the query entirely is caught.

        On Azure the query carries the API version and the path carries the
        deployment, and P6 removes ``model`` from the body — so a lost query can
        make two different routes look identical.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        with pytest.raises(AssertionError, match="query"):
            check_query(replace(captured, query=""), sent, scheme=_SCHEME)

    def test_query_check_rejects_a_reordered_query(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert re-serialising the query in sorted order is caught.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        with pytest.raises(AssertionError, match="query"):
            check_query(replace(captured, query="Beta=A%20B&beta=c&key=k"), sent, scheme=_SCHEME)

    def test_body_check_rejects_a_re_encoded_body(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert a body that has been through a text round-trip is caught.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        with pytest.raises(AssertionError, match="body"):
            check_body(replace(captured, body=b'{"a": 1}'), sent, scheme=_SCHEME)

    def test_header_casing_check_rejects_lower_cased_names(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert normalised header casing is caught.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        lowered = tuple((name.lower(), value) for name, value in captured.headers)
        with pytest.raises(AssertionError, match="casing"):
            check_header_casing(replace(captured, headers=lowered), sent, scheme=_SCHEME)

    def test_header_order_check_rejects_sorted_headers(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert reordering is caught.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        ordered = tuple(sorted(captured.headers))
        assert [n for n, _ in ordered] != [n for n, _ in captured.headers], (
            "the probe's headers must not already be sorted, or this proves nothing"
        )
        with pytest.raises(AssertionError, match="order"):
            check_header_order(replace(captured, headers=ordered), sent, scheme=_SCHEME)

    def test_header_duplicate_check_rejects_a_collapsed_repeat(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert dropping a repeated header is caught.

        A mapping-shaped recorder loses exactly this: ``anthropic-beta`` is sent
        twice and a dict keeps one.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        seen: set[str] = set()
        collapsed = []
        for name, value in captured.headers:
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            collapsed.append((name, value))
        assert len(collapsed) < len(captured.headers), "the probe must contain a duplicate header"
        with pytest.raises(AssertionError, match="header values per name"):
            check_header_duplicates(replace(captured, headers=tuple(collapsed)), sent, scheme=_SCHEME)

    def test_peer_port_check_rejects_the_recorders_own_listening_port(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert reporting the listening port instead of the peer's is caught.

        §5.2.1 joins the proxy's tunnel log on this value, so the defect
        produces a join that succeeds against the wrong connection rather than
        one that fails.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        with pytest.raises(AssertionError, match="peer_port"):
            check_peer_port(replace(captured, peer_port=8080), sent, scheme=_SCHEME)


class TestEachPerSessionCheckRejectsItsOwnDefect:
    """The four whole-recording checks, each handed its own defect."""

    @staticmethod
    def _session(
        markers: tuple[str, ...], arrivals: tuple[float, ...], ports: tuple[int, ...]
    ) -> tuple[Recording, list[SentRequest]]:
        """Build a recording and the sends that should have produced it.

        Args:
            markers: The markers, in recorded order.
            arrivals: The arrival stamps, in recorded order.
            ports: The peer ports, in recorded order.

        Returns:
            The recording, the sent requests (always in marker order ``m1,
            m2, ...`` so a recording that differs is out of order), and the
            ports the driver opened.
        """
        requests = [
            CapturedRequest(
                method="POST", scheme=_SCHEME, host="h", path=f"/{marker}/x", query="",
                headers=(("Host", "h"), ("X-Probe-Marker", marker)), body=b"", arrival=arrival,
                peer_port=port,
            )
            for marker, arrival, port in zip(markers, arrivals, ports, strict=True)
        ]
        sent = [
            _parse_sent(
                f"POST /{marker}/x HTTP/1.1\r\nHost: h\r\nX-Probe-Marker: {marker}\r\n\r\n".encode(),
                source_port=port, marker=marker,
            )
            for marker, port in zip(sorted(markers), sorted(ports), strict=True)
        ]
        connections = [(i, port, 1) for i, port in enumerate(sorted(ports))]
        return Recording(requests=requests, connections=connections), sent, sorted(ports)

    def test_arrival_check_rejects_a_constant_clock(self) -> None:
        """Assert a recorder stamping the same time on everything is caught.

        This is the defect a ``>=`` comparison lets through, which is why the
        requirement is *strictly* increasing.
        """
        recording, sent, opened = self._session(("m1", "m2"), (7.0, 7.0), (101, 102))
        with pytest.raises(AssertionError, match="strictly increase"):
            check_arrival_increases(recording, sent, opened)

    def test_arrival_check_rejects_a_backwards_clock(self) -> None:
        """Assert a timestamp taken after the response, out of order, is caught."""
        recording, sent, opened = self._session(("m1", "m2"), (9.0, 8.0), (101, 102))
        with pytest.raises(AssertionError, match="strictly increase"):
            check_arrival_increases(recording, sent, opened)

    def test_arrival_check_rejects_a_missing_timestamp(self) -> None:
        """Assert a recorder that never populates ``arrival`` is caught.

        ``CapturedRequest.arrival`` defaults to ``None``, so a recorder that
        simply forgot the field would otherwise produce captures that look
        structurally valid.
        """
        recording, sent, opened = self._session(("m1", "m2"), (1.0, 2.0), (101, 102))
        recording.requests[0] = replace(recording.requests[0], arrival=None)
        with pytest.raises(AssertionError, match="float"):
            check_arrival_increases(recording, sent, opened)

    def test_order_check_rejects_a_reordered_recording(self) -> None:
        """Assert a recorder appending out of order is caught.

        Phrased over identity, not over ``arrival`` — so this defect and the
        clock defects above each fail exactly one check.
        """
        recording, sent, opened = self._session(("m2", "m1"), (1.0, 2.0), (102, 101))
        with pytest.raises(AssertionError, match="request order"):
            check_request_order(recording, sent, opened)

    def test_order_check_passes_a_constant_clock(self) -> None:
        """Assert the ordering check is *not* tripped by a clock defect.

        The negative half of R4.2: a defect must fail its named check and no
        other. If ordering were phrased over ``arrival``, a constant clock would
        fail here too and the falsification matrix would be ambiguous.
        """
        recording, sent, opened = self._session(("m1", "m2"), (7.0, 7.0), (101, 102))
        check_request_order(recording, sent, opened)

    def test_count_check_rejects_a_dropped_request(self) -> None:
        """Assert a recorder that silently loses a request is caught."""
        recording, sent, opened = self._session(("m1", "m2"), (1.0, 2.0), (101, 102))
        recording.requests.pop()
        with pytest.raises(AssertionError, match="no capture for"):
            check_request_count(recording, sent, opened)

    def test_connection_check_rejects_a_missing_connection_record(self) -> None:
        """Assert a recorder logging no connection for a request is caught."""
        recording, sent, opened = self._session(("m1", "m2"), (1.0, 2.0), (101, 102))
        recording.connections.pop()
        with pytest.raises(AssertionError, match="opened but never logged"):
            check_connection_logged(recording, sent, opened)

    def test_connection_check_rejects_logging_only_connections_that_sent_a_request(self) -> None:
        """Assert the silent-connection defect is caught.

        §5.2.1's whole negative is an upstream connection with no matching
        tunnel. A connection that carries no HTTP request is invisible in the
        request list, so a recorder that logs only request-bearing connections
        cannot support the containment claim — and would pass every other check
        in this module.
        """
        recording, sent, opened = self._session(("m1",), (1.0,), (101,))

        # A second connection was opened and sent nothing, and the recorder did
        # not log it. It produces no `SentRequest` — which is exactly why the
        # check takes the opened ports separately. Judged against `sent` alone
        # it could only ever ask about connections that *did* send something,
        # and that is the one thing a lazily-logging recorder gets right.
        silent_port = 999
        assert silent_port not in {p for _, p, _ in recording.connections}

        with pytest.raises(AssertionError, match="opened but never logged"):
            check_connection_logged(recording, sent, [*opened, silent_port])


class TestCorrelationIsByMarkerNotPosition:
    """The property that keeps one defect from tripping fourteen checks."""

    def test_correlate_finds_a_capture_out_of_position(self) -> None:
        """Assert a capture is found by its marker regardless of where it sits.

        This is what lets the per-exchange checks pass against a recorder whose
        only fault is ordering, so that :func:`check_request_order` is the single
        check that fails.
        """
        recording = Recording(
            requests=[
                CapturedRequest(method="POST", scheme="http", host="h", path="/m2/x", query="",
                                headers=(("X-Probe-Marker", "m2"),)),
                CapturedRequest(method="POST", scheme="http", host="h", path="/m1/x", query="",
                                headers=(("X-Probe-Marker", "m1"),)),
            ]
        )
        first = _parse_sent(b"POST /m1/x HTTP/1.1\r\nHost: h\r\nX-Probe-Marker: m1\r\n\r\n", 1, "m1")
        assert correlate(recording, first).path == "/m1/x"

    def test_correlate_refuses_an_ambiguous_marker(self) -> None:
        """Assert two captures carrying one marker is an error, not a silent pick.

        Returning the first match would let a recorder that duplicated a request
        pass every per-exchange check.
        """
        recording = Recording(
            requests=[
                CapturedRequest(method="POST", scheme="http", host="h", path="/m1/x", query="",
                                headers=(("X-Probe-Marker", "m1"),)),
                CapturedRequest(method="POST", scheme="http", host="h", path="/m1/x", query="",
                                headers=(("X-Probe-Marker", "m1"),)),
            ]
        )
        sent = _parse_sent(b"POST /m1/x HTTP/1.1\r\nHost: h\r\nX-Probe-Marker: m1\r\n\r\n", 1, "m1")
        with pytest.raises(AssertionError, match="exactly one capture"):
            correlate(recording, sent)


class TestTheHeaderChecksAreOrthogonal:
    """The three header checks must not overlap, or R4.2 is unreachable.

    R4.2 requires each deliberate defect to be rejected by its **named** check
    and by no other. The three header defects — lower-casing names, sorting
    them, dropping a duplicate — all disturb the same field, so the checks are
    defined to be blind to each other's territory: casing by containment, order
    as a subsequence of lower-cased names, duplicates by per-name value lists.

    Those definitions are easy to "tidy" back into equality comparisons by a
    later reader who sees three checks doing nearly the same thing. These tests
    are what makes that tidying fail.
    """

    @staticmethod
    def _lower_cased(captured: CapturedRequest) -> CapturedRequest:
        """Return the capture a name-lower-casing recorder would produce.

        Args:
            captured: A conforming capture.

        Returns:
            The same capture with every header name lower-cased.
        """
        return replace(captured, headers=tuple((n.lower(), v) for n, v in captured.headers))

    @staticmethod
    def _sorted(captured: CapturedRequest) -> CapturedRequest:
        """Return the capture a header-sorting recorder would produce.

        Args:
            captured: A conforming capture.

        Returns:
            The same capture with its headers sorted.
        """
        return replace(captured, headers=tuple(sorted(captured.headers)))

    @staticmethod
    def _deduplicated(captured: CapturedRequest) -> CapturedRequest:
        """Return the capture a mapping-shaped recorder would produce.

        Args:
            captured: A conforming capture.

        Returns:
            The same capture with repeated header names collapsed to the first.
        """
        seen: set[str] = set()
        kept = []
        for name, value in captured.headers:
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            kept.append((name, value))
        return replace(captured, headers=tuple(kept))

    def test_lower_casing_fails_only_the_casing_check(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert a re-casing recorder is not also reported as reordering or dropping.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        defective = self._lower_cased(captured)
        with pytest.raises(AssertionError, match="casing"):
            check_header_casing(defective, sent, scheme=_SCHEME)
        check_header_order(defective, sent, scheme=_SCHEME)
        check_header_duplicates(defective, sent, scheme=_SCHEME)

    def test_sorting_fails_only_the_order_check(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert a sorting recorder is not also reported as re-casing or dropping.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        defective = self._sorted(captured)
        assert [n for n, _ in defective.headers] != [n for n, _ in captured.headers], (
            "the probe's headers must not already be sorted, or this proves nothing"
        )
        with pytest.raises(AssertionError, match="order"):
            check_header_order(defective, sent, scheme=_SCHEME)
        check_header_casing(defective, sent, scheme=_SCHEME)
        check_header_duplicates(defective, sent, scheme=_SCHEME)

    def test_dropping_a_duplicate_fails_only_the_duplicate_check(
        self, captured: CapturedRequest, sent: SentRequest
    ) -> None:
        """Assert a mapping-shaped recorder is not also reported as re-casing or reordering.

        Args:
            captured: A conforming capture.
            sent: The request that produced it.
        """
        defective = self._deduplicated(captured)
        assert len(defective.headers) < len(captured.headers), (
            "the probe must contain a duplicate header, or this proves nothing"
        )
        with pytest.raises(AssertionError, match="header values per name"):
            check_header_duplicates(defective, sent, scheme=_SCHEME)
        check_header_casing(defective, sent, scheme=_SCHEME)
        check_header_order(defective, sent, scheme=_SCHEME)


class TestTheSessionChecksAreOrthogonal:
    """Ordering, counting and timing must not overlap, for the same reason."""

    @staticmethod
    def _session(markers, arrivals, ports):
        """Build a recording and the sends that should have produced it.

        Args:
            markers: The markers, in recorded order.
            arrivals: The arrival stamps, in recorded order.
            ports: The peer ports, in recorded order.

        Returns:
            The recording and the sent requests in marker order.
        """
        return TestEachPerSessionCheckRejectsItsOwnDefect._session(markers, arrivals, ports)

    def test_a_reordering_recorder_still_passes_the_count_and_clock_checks(self) -> None:
        """Assert an out-of-order recorder trips the ordering check alone."""
        recording, sent, opened = self._session(("m2", "m1"), (2.0, 1.0), (102, 101))
        with pytest.raises(AssertionError, match="request order"):
            check_request_order(recording, sent, opened)
        check_request_count(recording, sent, opened)
        # Read in SENT order the arrivals are 1.0 then 2.0, which is correct.
        # Walking the recorded list instead would report a clock defect here,
        # and the falsification matrix could not tell the two apart.
        check_arrival_increases(recording, sent, opened)

    def test_a_dropping_recorder_trips_the_count_check_alone(self) -> None:
        """Assert a recorder losing a request is not also reported as misordering.

        This is the defect nothing else catches: §4.3 C6's "``/healthz`` never
        causes an upstream request" rests on a recorder that cannot silently
        lose one.
        """
        recording, sent, opened = self._session(("m1", "m2"), (1.0, 2.0), (101, 102))
        recording.requests.pop()
        with pytest.raises(AssertionError, match="no capture for"):
            check_request_count(recording, sent, opened)
        check_request_order(recording, sent, opened)
        check_arrival_increases(recording, sent, opened)

    def test_a_constant_clock_trips_the_clock_check_alone(self) -> None:
        """Assert a broken clock is not also reported as misordering or dropping."""
        recording, sent, opened = self._session(("m1", "m2"), (7.0, 7.0), (101, 102))
        with pytest.raises(AssertionError, match="strictly increase"):
            check_arrival_increases(recording, sent, opened)
        check_request_order(recording, sent, opened)
        check_request_count(recording, sent, opened)


class TestTheAssertionsWithNoDefectOfTheirOwn:
    """Coverage for check branches that no falsification recorder reaches.

    A review pass mutation-tested every assertion in the conformance module and
    found three that could be deleted with the suite still green. One was a
    tautology and is gone. These are the other two: both guard real failure
    modes that no *recorder* defect happens to produce, so they need a
    hand-built recording instead — an assertion nothing exercises is
    indistinguishable from one that cannot fail.
    """

    def test_count_check_rejects_a_capture_nobody_sent(self) -> None:
        """Assert an invented capture is caught.

        The mirror of a dropped request: a recorder that duplicates one, or
        manufactures one from a health check, corrupts the evidence just as
        badly. §4.3 C6 — "``/healthz`` never causes an upstream request" —
        rests on this half.
        """
        recording, sent, opened = TestEachPerSessionCheckRejectsItsOwnDefect._session(
            ("m1",), (1.0,), (101,)
        )
        recording.requests.append(
            CapturedRequest(
                method="POST", scheme=_SCHEME, host="h", path="/ghost/x", query="",
                headers=(("X-Probe-Marker", "ghost"),), body=b"", arrival=2.0, peer_port=101,
            )
        )
        with pytest.raises(AssertionError, match="captures nobody sent"):
            check_request_count(recording, sent, opened)

    def test_connection_check_rejects_a_miscounted_connection(self) -> None:
        """Assert a connection log that miscounts its requests is caught.

        This is the other half of R1.10: a capture must be attributable to the
        connection that carried it. A log that records the connection but loses
        track of what rode on it cannot support §4.3 C5's distinct-connection
        count.
        """
        recording, sent, opened = TestEachPerSessionCheckRejectsItsOwnDefect._session(
            ("m1", "m2"), (1.0, 2.0), (101, 102)
        )
        connection_id, peer_port, _ = recording.connections[0]
        recording.connections[0] = (connection_id, peer_port, 0)

        with pytest.raises(AssertionError, match="accounts for 1 requests, 2 were sent"):
            check_connection_logged(recording, sent, opened)
