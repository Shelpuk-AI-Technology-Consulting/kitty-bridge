"""Dependency behaviour contract for the standard library's :mod:`ipaddress`.

``.system_design/TEST_SUITE.md`` §6.2.4 asks for a small, fast test per dependency
whose behaviour an invariant rests on, so that an upgrade fails here with a clear
message rather than in a user's terminal.  The interpreter is the least pinned
dependency this project has: ``requires-python = ">=3.10"`` bounds its *minor*
version and says nothing about the patch release, which is where ``ipaddress``
has actually moved.

There are two consumers.

:func:`kitty.egress.should_bypass` decides whether a destination is reached
directly or through the egress proxy, which is invariant **I3** (§5).  It reads
``is_loopback or is_private or is_link_local``, and the verdict that disjunction
reaches for the IPv4-mapped form of each range is pinned below.  **The terms are
not individually stable and the contract does not pretend they are:**
``ip_address("::ffff:169.254.1.1").is_link_local`` flips at the same four
release boundaries as ``is_unspecified`` (measured ``False`` on 3.10.13-15,
3.11.8-10, 3.12.4-6 and 3.13.0; ``True`` from 3.10.16, 3.11.11, 3.12.7 and
3.13.1).  The disjunction survives only because ``is_private`` delegates to the
mapped address on every supported release and IPv4's ``is_private`` already
covers ``169.254.0.0/16``.  Anyone who later splits that disjunction, narrows it
or logs per term re-opens the patch dependence in the containment direction.

:func:`kitty.bridge.manage._connect_target` maps a recorded bind address to an
address this machine can dial, so that ``bridge_reachable`` can settle
:attr:`~kitty.bridge.manage.ProcessLiveness.UNKNOWN`.  It asks ``ipaddress``
three things, each pinned below: whether a string is an address literal at all,
whether an IPv6 address is the IPv4-mapped form of some IPv4 address, and
whether an address is the unspecified one.

**What this module deliberately does not assert, and why.**
``ipaddress.ip_address("::ffff:0.0.0.0").is_unspecified`` is **not** pinned here.
Its value is a property of the interpreter's patch release rather than of the
address: CPython `gh-122792 <https://github.com/python/cpython/issues/122792>`_
made ``IPv6Address``'s ``is_*`` properties delegate to the mapped
``IPv4Address``, and the change was backported mid-branch — first appearing in
3.10.16, 3.11.11, 3.12.7 and 3.13.1.  Every release below those lines is a
supported configuration, so asserting ``True`` would be red on the 35 releases
that predate the backport, and asserting ``False`` red on every release after
them.

That is the rule this file establishes, and it is the opposite of the obvious
move.  Before KBR-146, ``_connect_target`` read that property and a contract
pinning it would have enforced the defect rather than caught it.  When a
dependency's behaviour is version-dependent, the fix is to stop depending on it
and pin the stable neighbour — here ``ipv4_mapped``, which reads the same on
every supported release.  Pinning the moving value only relocates the failure.

``ipv4_mapped`` itself, by contrast, was measured identical on 18 releases
spanning all four supported branches.  That measurement is what makes the
forced-property test in ``tests/bridge/test_bridge_management.py`` a
*reproduction* of a pre-backport interpreter rather than a resemblance to one.

**What this module cannot prove.**  ``.github/workflows/tests.yml`` names bare
minor versions and ``actions/setup-python`` resolves each to the newest patch, so
every assertion here is only ever evaluated on the **new** side of such a
boundary.  It catches forward drift; it cannot catch a value that differs on an
older patch a user is running, which is the shape KBR-146 had.  That half rests
on the forced-property test.  Recorded as gap **G25**, not closed.

Traces to `KBR-146 <https://shelpuk.atlassian.net/browse/KBR-146>`_.
"""

from __future__ import annotations

import ipaddress

import pytest

# L2: this module asserts what a separately upgraded artifact — the interpreter —
# does. It imports no kitty code; a contract that reads the consumer would be
# satisfied by whatever the consumer happens to do.
pytestmark = pytest.mark.l2


def test_the_ipv4_mapped_wildcard_reports_the_unspecified_ipv4_address():
    """Pin that ``::ffff:0.0.0.0`` unwraps to the IPv4 wildcard.

    This is the single fact that lets ``_connect_target`` recognise an
    IPv4-mapped wildcard bind without consulting the patch-dependent
    ``is_unspecified`` property of the IPv6 address itself.

    The **type** is pinned alongside the value because the consumer reaches
    ``ipv4_mapped`` through an ``isinstance(address, IPv6Address)`` gate.  Were
    that relationship to stop holding, the gate would yield ``None``, control
    would fall through to ``is_unspecified``, and KBR-146 would be reinstated
    verbatim with every other assertion in this module still green.
    """
    address = ipaddress.ip_address("::ffff:0.0.0.0")

    assert isinstance(address, ipaddress.IPv6Address)

    mapped = address.ipv4_mapped
    assert mapped == ipaddress.IPv4Address("0.0.0.0")
    assert mapped.is_unspecified is True


def test_a_non_wildcard_mapped_address_unwraps_to_a_non_wildcard():
    """Pin that unwrapping distinguishes a real mapped bind from a wildcard.

    Without this, ``_connect_target`` could rewrite every IPv4-mapped address to
    loopback and claim a bridge that is not there.  It is the negative half of
    the fact above, and the two must move together or not at all.
    """
    mapped = ipaddress.ip_address("::ffff:127.0.0.1").ipv4_mapped

    assert mapped == ipaddress.IPv4Address("127.0.0.1")
    assert mapped.is_unspecified is False


@pytest.mark.parametrize("host", ["::", "::0", "0:0:0:0:0:0:0:0", "::1"])
def test_a_native_ipv6_address_reports_no_mapped_address(host: str):
    """Pin that only the ``::ffff:`` form unwraps, so the IPv6 branch stays reachable.

    Args:
        host: A native IPv6 address literal — the three spellings of the
            wildcard that ``bridge.yaml`` passes through untouched, and loopback.

    If ``ipv4_mapped`` ever became non-``None`` for these, ``_connect_target``
    would answer with an IPv4 loopback for an IPv6-only bridge and the probe
    would find nothing listening.
    """
    assert ipaddress.ip_address(host).ipv4_mapped is None


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("0.0.0.0", True),
        ("::", True),
        ("::0", True),
        ("0:0:0:0:0:0:0:0", True),
        ("127.0.0.1", False),
        ("::1", False),
        ("192.0.2.1", False),
    ],
)
def test_unspecified_is_stable_for_addresses_that_are_not_ipv4_mapped(host: str, expected: bool):
    """Pin ``is_unspecified`` for every form ``_connect_target`` still asks it about.

    Args:
        host: An address literal that is not the IPv4-mapped form.
        expected: What ``is_unspecified`` must report for it.

    The list is bounded to the inputs the consumer actually reaches this
    property with, which is the rule the module docstring states: an address
    outside it (``255.255.255.255``, say) is not a coverage gap, because no
    consumer asks about it.

    gh-122792 moved this property only for IPv4-mapped addresses.  For
    everything else it has answered the same since 3.3, and ``_connect_target``
    reads it for exactly these inputs — so here it is safe to pin, and pinning
    it is what makes the exclusion of the mapped form a deliberate boundary
    rather than an omission.
    """
    assert ipaddress.ip_address(host).is_unspecified is expected


@pytest.mark.parametrize("host", ["localhost", "bridge.internal", "not an address"])
def test_a_hostname_is_rejected_as_an_address_literal(host: str):
    """Pin that a hostname raises rather than parsing into something plausible.

    Args:
        host: A string that is not an address literal.

    ``_connect_target`` treats ``ValueError`` as "this is a hostname, probe it
    exactly as recorded".  A future ``ipaddress`` that resolved names, or that
    raised a different exception, would turn that branch into an unhandled
    crash on a user-facing command.
    """
    with pytest.raises(ValueError):
        ipaddress.ip_address(host)


@pytest.mark.parametrize(
    "host",
    [
        "::ffff:127.0.0.1",
        "::ffff:10.0.0.1",
        "::ffff:172.16.0.1",
        "::ffff:192.168.1.1",
        "::ffff:169.254.1.1",
    ],
)
def test_the_mapped_form_of_every_bypassed_range_is_still_classified_as_bypassed(host: str):
    """Pin I3's bypass verdict for IPv4-mapped destinations.

    Args:
        host: The IPv4-mapped form of an address ``should_bypass`` must bypass —
            loopback, the three private ranges, and link-local.

    The **disjunction** is pinned, not its terms, because ``is_link_local`` is
    not independently stable across patch releases (see the module docstring)
    and ``is_private`` is what actually carries the mapped cases.  Pinning the
    terms would make this module red on every pre-backport release; pinning the
    verdict states the thing ``should_bypass`` needs and nothing more.
    """
    address = ipaddress.ip_address(host)

    assert (address.is_loopback or address.is_private or address.is_link_local) is True


def test_a_mapped_public_address_is_not_classified_as_bypassed():
    """Hold the other side of the bypass verdict, so the pin cannot pass vacuously.

    A contract that only asserts "bypassed" is satisfied by a stdlib that says
    yes to everything — which would route every destination around the egress
    proxy and breach I3 in the direction that matters.
    """
    address = ipaddress.ip_address("::ffff:8.8.8.8")

    assert (address.is_loopback or address.is_private or address.is_link_local) is False
