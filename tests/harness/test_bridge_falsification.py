"""Four deliberate defects the bridge fixture's conformance check must catch.

`.system_design/TEST_SUITE.md` §7.5.4 · plan task **T-W8** (KBR-31), plan §1.4 ·
`.requirements/20260911T233557Z_bridge_fixture_core/REQUIREMENTS.md` R6.

Plan §1.4: *"the first working version of every harness ships with at least one
falsification case — a deliberate defect it must detect, running in the suite."*
Four review rounds on the design produced four harnesses that would have passed
while proving nothing, and a bridge fixture's own version of that failure is
**silent**: a bridge pointed at the wrong live upstream answers 200, and the
recording is empty.

**One defect per assertion, and each passes the other three.** That is the only
argument that establishes the four assertions are not redundant — "each is
insufficient alone" is a different and weaker claim. The defects are:

=========================  ==========================  ==================================
Defect                     Assertion it breaks         What the other three see
=========================  ==========================  ==================================
:class:`_DecoyTransport`   exactly one capture         200, clean teardown, no marker to
                                                       look for because the list is empty
:class:`_BlindTransport`   the marker is in the body   one capture, 200, clean teardown
:class:`_RefusingTransport` the status is 200          one capture, marker present, clean
:class:`_MisdeclaredTransport` teardown is clean       one capture, marker present, 200
=========================  ==========================  ==================================

The last is why it is a transport and not a unit test on
``assert_teardown_clean``. A unit test proves that function raises; it does not
prove the conformance check **calls** it — and plan §1.4's own list includes "a
guard proving a function was *called* when the enforcement was the branch after
it".

**None of these is registered.** They are passed to the check as instances. A
shared registry holding four things that are wrong on purpose is a hazard, and
it would also make the registry-completeness meta-test assert over them.

**Layer.** No ``pytestmark``: the ``l1`` path default, for the reason
``test_bridge.py`` records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import pytest

from harness.bridge import (
    AiohttpTransport,
    Binding,
    MisdeclaredFormatError,
    assert_transport_reaches_its_recorder,
)
from harness.contract import CapturedRequest, WireFormat
from harness.recorder import RecordingUpstream, Reply

#: The format every defect below declares, so the one thing that differs between
#: them is the defect itself.
FORMAT = WireFormat.ANTHROPIC_MESSAGES


class _BlindRecorder(RecordingUpstream):
    """A recorder that counts a request and keeps none of its content."""

    def store(self, index: int, captured: CapturedRequest) -> None:
        """Store the capture with its body removed.

        ``store`` is T-W4's seam for exactly this: a defect that disturbs one
        thing and leaves the connection log and the ordering intact.

        Args:
            index: The reserved slot.
            captured: The capture, stored without its body.
        """
        super().store(index, replace(captured, body=b""))


@dataclass
class _DecoyTransport(AiohttpTransport):
    """Points the bridge at a *different* live recorder than the one it reports.

    §7.5.4's measured row 3, and the expensive failure: the request succeeds, the
    bridge is satisfied, the client gets 200, and this transport's capture list
    is empty. Every oracle built on the fixture would then be quantified over
    nothing and would pass for free.
    """

    name = "falsify-decoy"

    _decoy: RecordingUpstream = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build both recorders: the one reported, and the one actually reached."""
        super().__post_init__()
        self._decoy = RecordingUpstream(default_format=self.format)

    async def start(self) -> None:
        """Start both, so the decoy really answers rather than timing out."""
        await super().start()
        await self._decoy.start()

    async def stop(self) -> None:
        """Stop both, in a ``finally`` so neither leaks if the other raises."""
        try:
            await super().stop()
        finally:
            await self._decoy.stop()

    def bind(self) -> Binding:
        """Return a binding that reaches the decoy.

        Returns:
            The adapter this transport's format calls for, pointed elsewhere.
        """
        adapter, _config = super().bind()
        return adapter, {"base_url": self._decoy.base_url}


@dataclass
class _BlindTransport(AiohttpTransport):
    """Counts the request and records nothing about it.

    Plan §1.4's fourth named harness failure: "a projection that could not see
    the model name, in a product whose purpose is changing the model name".
    """

    name = "falsify-blind"

    def __post_init__(self) -> None:
        """Use the body-dropping recorder instead of the real one."""
        self._recorder = _BlindRecorder(default_format=self.format, responder=self.responder)


@dataclass
class _RefusingTransport(AiohttpTransport):
    """Receives the request correctly and refuses it.

    Measured at one capture, marker present, clean teardown, downstream 400, in
    0.00 s. An upstream **500** would have been the obvious choice and is wrong:
    measured at four captures over seven seconds, it trips the capture-count
    assertion too and so is not orthogonal.
    """

    name = "falsify-refusing"

    def __post_init__(self) -> None:
        """Install a responder that answers 400 with a well-formed error body."""
        self.responder = _refuse
        super().__post_init__()


@dataclass
class _MisdeclaredTransport(AiohttpTransport):
    """Declares one format and binds an adapter that posts to the other.

    The recorder dispatches its reply by path **suffix**, so the request is
    answered in the format the *path* named and the adapter parses it happily.
    Nothing is recorded as unmatched, the capture is complete and correct, and
    the declared format was simply never under test.
    """

    name = "falsify-misdeclared"

    def bind(self) -> Binding:
        """Return an adapter for the format this transport does **not** declare.

        Returns:
            A Chat Completions adapter, while :attr:`format` says Anthropic
            Messages.
        """
        from kitty.providers.custom_openai import CustomOpenAIAdapter

        _adapter, config = super().bind()
        return CustomOpenAIAdapter(), config


async def _refuse(captured: CapturedRequest, reply: Reply) -> None:
    """Answer 400 with a well-formed, non-empty error body.

    The body is load-bearing. An empty or unparseable one is not a loud failure
    but the 80-second empty-response ladder (§7.2.1), which would break the
    capture-count assertion as well and cost this case its orthogonality.

    Args:
        captured: The capture, unused.
        reply: The unprepared response.
    """
    await reply.begin(400, {"content-type": "application/json"})
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "no"}}
    await reply.write(json.dumps(body).encode())


async def _failure_message(transport: AiohttpTransport) -> str:
    """Run the conformance check against a defect and return why it failed.

    Args:
        transport: An unstarted defective transport.

    Returns:
        The failure message.

    Raises:
        Failed: When the check passed, which is the whole point of this module.
    """
    with pytest.raises(AssertionError) as excinfo:
        await assert_transport_reaches_its_recorder(transport)
    return str(excinfo.value)


class TestEachDefectIsCaught:
    """One case per conformance assertion."""

    async def test_a_binding_that_reaches_another_upstream_is_caught(self) -> None:
        """The decoy — the failure no status check can see.

        The inbound request returns 200 and the recording is empty, so this is
        the one defect that makes every downstream assertion pass vacuously.
        """
        message = await _failure_message(_DecoyTransport(FORMAT))

        assert "0 capture" in message
        assert "quantified over nothing" in message

    async def test_a_capture_that_cannot_see_the_content_is_caught(self) -> None:
        """The blind capture — right count, right status, no evidence."""
        message = await _failure_message(_BlindTransport(FORMAT))

        assert "marker" in message

    async def test_a_correct_request_that_still_fails_the_client_is_caught(self) -> None:
        """The refusal — the upstream got exactly the right request.

        A count and a marker both pass here. Only the inbound status says the
        client was not served.
        """
        message = await _failure_message(_RefusingTransport(FORMAT))

        assert "400" in message

    async def test_a_format_that_was_never_under_test_is_caught(self) -> None:
        """The mis-declared format — a complete, correct capture list.

        This is the opposite of the decoy, not a variant of it, and it is the
        only case that proves the conformance check *calls* the teardown check
        rather than merely that the teardown check works.
        """
        with pytest.raises(MisdeclaredFormatError) as excinfo:
            await assert_transport_reaches_its_recorder(_MisdeclaredTransport(FORMAT))

        assert "anthropic_messages" in str(excinfo.value)


class TestTheDefectsAreOrthogonal:
    """Each defect must break its own assertion and no other.

    Without this, a later tidy-up that collapsed two checks into one would leave
    every case above still green, and the suite would have quietly halved its
    evidence. ``recorder_conformance.py``'s orthogonality tests exist for the
    same reason and will look equally redundant to a reader in a hurry.
    """

    async def test_the_four_messages_are_mutually_distinguishable(self) -> None:
        """Pairwise, on the words that identify which assertion fired."""
        messages = {
            "decoy": await _failure_message(_DecoyTransport(FORMAT)),
            "blind": await _failure_message(_BlindTransport(FORMAT)),
            "refusing": await _failure_message(_RefusingTransport(FORMAT)),
            "misdeclared": await _failure_message(_MisdeclaredTransport(FORMAT)),
        }

        assert len(set(messages.values())) == 4, f"two defects report the same failure: {messages}"

        # The specific confusion worth naming: the decoy and the mis-declared
        # format differ by whether the capture list is empty or complete, so a
        # count-shaped message from the latter would mean the two had merged.
        assert "capture" in messages["decoy"] and "marker" not in messages["decoy"]
        assert "marker" in messages["blind"]
        assert "400" in messages["refusing"]
        assert "anthropic_messages" in messages["misdeclared"]

    async def test_the_three_survivable_defects_still_deliver_a_complete_capture(self) -> None:
        """The blind, refusing and mis-declared cases all reach their own recorder.

        Stated positively so the decoy's emptiness stays a property of the decoy
        alone. If a change made every defect produce an empty recording, all four
        cases above would still pass and all four would be testing one thing.
        """
        for defect, expected_body in (
            (_RefusingTransport(FORMAT), True),
            (_MisdeclaredTransport(FORMAT), True),
            (_BlindTransport(FORMAT), False),
        ):
            with pytest.raises(AssertionError):
                await assert_transport_reaches_its_recorder(defect)

            assert len(defect.captures) == 1, f"{defect.name} should still have reached its own recorder"
            assert bool(defect.captures[0].body) is expected_body

        decoy = _DecoyTransport(FORMAT)
        with pytest.raises(AssertionError):
            await assert_transport_reaches_its_recorder(decoy)
        assert list(decoy.captures) == []
