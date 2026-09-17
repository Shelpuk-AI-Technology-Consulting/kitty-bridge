"""The botocore transport, and how the bridge fixture reaches it.

`.system_design/TEST_SUITE.md` §7.5, §7.2.4 · plan task **T-B3** (KBR-42).

:mod:`harness.botocore_recorder` is the server; this module is how the product
is pointed at it. Unlike the three default-transport shapes, ``bedrock``
builds a boto3 client per request and botocore owns the HTTP stack, so the
product's own redirection channel (``provider_config["base_url"]`` through
``build_base_url``) is unreachable here (§7.5.2). The seam is
``provider_config["endpoint_url"]``, which ``BedrockAdapter._get_boto3_client``
passes as ``endpoint_url=`` to ``session.client(...)`` — and ``bind()`` is
what supplies it, which is why the extension interface is ``bind()`` and not
a base-URL helper (§7.5.2).

**Proxy precedence, measured.** (KBR-64's T-G11 probe; see
:mod:`harness.test_botocore_transport_contract` for the pinned contract.)
``BedrockAdapter._get_boto3_client`` reads the process-wide egress from
``kitty.egress.get_egress()`` and passes
``botocore.config.Config(proxies=egress.proxies_dict())`` to
``session.client``. Measured on the resolved botocore (1.43.93): the explicit
``Config(proxies=...)`` overrides ambient ``NO_PROXY`` matching the
destination and ambient ``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``ALL_PROXY`` in
both letter cases — the documented precedence AWS publishes
(https://docs.aws.amazon.com/boto3/latest/guide/configuration.html), pinned
as tests rather than assumed.

**Why the transport keeps one adapter, like T-B1's.** The adapter is
stateless (``_get_boto3_client`` builds a client per call and caches nothing,
KBR-190), so a fresh adapter per ``bind()`` would be legitimate — but every
balancing member sharing one adapter is the shape
:func:`harness.bridge.backend_for` already expects, and keeping one makes
``stop()`` trivially correct: there is nothing to release, so the recorder's
port is the only resource this transport owns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from harness.botocore_recorder import BedrockRecordingUpstream
from harness.bridge import Binding, register_transport
from harness.contract import CapturedRequest, WireFormat
from harness.recorder import ConnectionRecord, Responder
from kitty.providers.bedrock import BedrockAdapter

__all__ = [
    "HarnessBedrockAdapter",
    "BotocoreTransport",
]

#: The AWS region a harness profile names. ``get_region`` would fall back to
#: the same value, but naming it here keeps ``bind()``'s returned config
#: self-describing: a test reading it learns the whole provider_config the
#: bridge will carry, not the one key this transport added.
_HARNESS_REGION = "us-east-1"


class HarnessBedrockAdapter(BedrockAdapter):
    """A ``BedrockAdapter`` that resolves the harness key to a fake pair.

    ``BridgeServer`` carries the shared ``"harness-key"`` resolved key
    (:mod:`harness.bridge`` ``_KEY``), which ``parse_aws_credentials``
    rejects — it demands ``access_key:secret_key`` and the harness key has
    no colon. The product is not wrong: a real key would have the pair. The
    transport overrides the one method so the harness key resolves to the
    fake pair the recorder sees (which it never validates — the request
    arrives at a loopback aiohttp server, not at AWS).

    **This is not a product seam.** Unlike ``provider_config["endpoint_url"]``,
    which the product honours, this override exists for the fixture only and
    never runs in production: production profiles resolve a real key through
    the same ``parse_aws_credentials``, unchanged.

    **SSO branch is intentionally un-overridden.** ``is_sso_mode`` returns
    True for ``""`` and ``"sso"`` and routes through ``boto3.Session(
    profile_name=…, region_name=…)`` *before* ``parse_aws_credentials``
    runs. The harness key ``"harness-key"`` is the only value that
    exercises this override today — a future test that set the resolved
    key to ``""`` or ``"sso"`` would silently fall into the SSO branch and
    use whatever credentials the test machine has. Production profiles
    ship a real key, so the brittleness is contained to the harness.
    """

    def parse_aws_credentials(self, raw: str) -> tuple[str, ...]:
        """Resolve any harness key to the documented fake pair.

        Args:
            raw: The resolved key — the harness's shared ``"harness-key"``
                in this transport's case; a real key in production, which
                the base class handles and this override never sees.

        Returns:
            The fake pair, of the shape the boto3 session constructor takes.
        """
        return ("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")


@dataclass
class BotocoreTransport:
    """The transport for the ``bedrock`` adapter.

    Attributes:
        format: The upstream wire format served. Only
            :attr:`~harness.contract.WireFormat.BEDROCK_CONVERSE`; the
            recorder rejects anything else at construction.
        responder: What to reply with; the recorder's minimal success when
            omitted. A closure over mutable state is how a reply is scripted
            to change between attempts.
    """

    #: A class attribute, not a field: every instance answers to one registry key.
    name = "botocore"

    format: WireFormat
    responder: Responder | None = None
    _recorder: BedrockRecordingUpstream = field(init=False, repr=False)
    _adapter: BedrockAdapter | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Build the recorder eagerly, so an unserved format fails here.

        Raises:
            ValueError: When ``format`` is not one this recorder serves.
        """
        self._recorder = BedrockRecordingUpstream(default_format=self.format, responder=self.responder)

    @property
    def recorder(self) -> BedrockRecordingUpstream:
        """Return the underlying recorder.

        Returns:
            The :class:`~harness.botocore_recorder.BedrockRecordingUpstream`,
            for tests that need the parts of it the transport interface does
            not expose.
        """
        return self._recorder

    async def start(self) -> None:
        """Bind an ephemeral loopback port and begin recording."""
        await self._recorder.start()

    async def stop(self) -> None:
        """Release the recorder's port.

        Nothing else to release: ``_get_boto3_client`` builds a client per
        request and caches nothing (KBR-190), so unlike
        :class:`~harness.provider_aiohttp.ProviderAiohttpTransport` there is
        no adapter-owned session to close first. The recorder is stopped
        unconditionally: a transport that fails to close must not leave a
        port bound, because the next test's ephemeral port allocation is the
        only thing that would notice.
        """
        await self._recorder.stop()

    def bind(self) -> Binding:
        """Return the adapter and config that reach this recorder.

        Returns:
            The ``bedrock`` adapter, and the provider configuration naming
            the recorder's base URL under ``endpoint_url`` — the key
            ``BedrockAdapter._get_boto3_client`` consumes — plus ``region``,
            which is a pre-existing ``provider_config`` key
            (``get_region`` reads it), named so the returned config is
            self-describing.

        Raises:
            RuntimeError: When the transport has not been started, because
                the recorder has no port until then.
        """
        # Lazy init — ``__post_init__`` builds the recorder eagerly but
        # holds off on the adapter so a transport that is constructed but
        # never bound does not allocate an unused harness subclass. The
        # ``test_bind_shares_the_one_adapter_across_calls`` test pins the
        # identity across repeated binds, so the mutation is intentional.
        if self._adapter is None:
            self._adapter = HarnessBedrockAdapter()
        # ``base_url`` raises ``RuntimeError("recorder is not running; call
        # start() first")`` when the recorder has not been started — the
        # exact matcher ``test_binding_before_starting_raises`` relies on.
        return self._adapter, {"endpoint_url": self._recorder.base_url, "region": _HARNESS_REGION}

    @property
    def captures(self) -> Sequence[CapturedRequest]:
        """Return the completed captures, in arrival order.

        Returns:
            What the recorder holds.
        """
        return self._recorder.requests

    @property
    def connections(self) -> Sequence[ConnectionRecord]:
        """Return every accepted connection.

        Returns:
            One record per connection, including any that carried no request —
            §5.2.1's bypass shape, and the peer ports T-E4's tunnel join reads.
        """
        return self._recorder.connections

    def assert_teardown_clean(self) -> None:
        """Assert every request was answered in the format this transport declares.

        **One check, not two, and that is measured rather than assumed** —
        the same measurement T-B1 records: this recorder serves one format,
        so a wrong-path request takes the fallback, the adapter parses
        nothing out of the reply, and the conformance check fails loudly on
        captures plus a timeout. A second pass asserting the path suffix
        would be an assertion no defect could falsify, which is exactly what
        §7.5.4 found and removed when the same repeat appeared in T-W8.

        Unlike the aiohttp transports there is no *second* format to
        mis-declare against — the recorder's suffix table has two entries,
        but both select :data:`WireFormat.BEDROCK_CONVERSE`, so the
        :class:`~harness.bridge.MisdeclaredFormatError` arm of the primary
        transport's check has no case to catch here. The fallback report is
        the whole claim.

        Raises:
            UnmatchedPathError: When a request took the recorder's fallback.
        """
        self._recorder.assert_all_paths_matched()


register_transport(BotocoreTransport.name, BotocoreTransport)
