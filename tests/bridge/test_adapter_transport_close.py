"""Bridge teardown releases the transports its adapters own — KBR-190.

``.system_design/TEST_SUITE.md`` §5.5.  ``BridgeServer.stop_async`` closed the
two sessions the bridge builds and nothing else, so an adapter with
``use_custom_transport`` leaked its own connection pool on every start/stop
cycle inside a living process.

**Why these are L1 and construct a real server anyway.**  §2.2's own worked
example puts a test that must build a ``BridgeServer`` at L1 — the behaviour is
not a pure function, but nothing here touches a socket, a clock or a provider:
the adapters are stubs that count, and the bridge is never started.
"""

from __future__ import annotations

import uuid

import pytest

from kitty.bridge.server import BridgeServer, _backend_context
from kitty.profiles import Profile
from kitty.providers.base import ProviderAdapter


class _CountingAdapter(ProviderAdapter):
    """A minimal adapter that records every ``aclose`` it receives.

    Attributes:
        closes: How many times :meth:`aclose` has been awaited.
    """

    def __init__(self, *, fails: bool = False) -> None:
        """Create an adapter that counts closes, and optionally fails them.

        Args:
            fails: When True, :meth:`aclose` raises after counting — the
                misbehaving-provider case R7's containment is written for.
        """
        self.closes = 0
        self._fails = fails

    @property
    def provider_type(self) -> str:
        """Return the registry name this stub answers to."""
        return "counting"

    @property
    def default_base_url(self) -> str:
        """Return an address no test may reach."""
        return "https://example.invalid/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        """Return an empty request; nothing here serves traffic."""
        return {}

    def parse_response(self, response_data: dict) -> dict:
        """Return the response unchanged; nothing here serves traffic."""
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        """Return a generic error; nothing here serves traffic."""
        return Exception(f"stub {status_code}")

    async def aclose(self) -> None:
        """Count the close, and raise afterwards when built to fail.

        Raises:
            RuntimeError: When the adapter was constructed with ``fails=True``.
        """
        self.closes += 1
        if self._fails:
            raise RuntimeError("provider teardown failed")


def _profile() -> Profile:
    """Return a throwaway profile for a backend tuple.

    Returns:
        A profile the bridge reads only for ``backup`` and ``name``.
    """
    return Profile(name="counting", provider="openai", model="m", auth_ref=str(uuid.uuid4()))


class TestEveryAdapterIsClosedExactlyOnce:
    """R6 — the bridge closes what it holds, from its own backing fields."""

    async def test_the_single_mode_provider_is_closed(self):
        provider = _CountingAdapter()
        server = BridgeServer(None, provider, "test-key")

        await server.stop_async()

        assert provider.closes == 1

    async def test_every_distinct_backend_is_closed(self):
        first, second = _CountingAdapter(), _CountingAdapter()
        server = BridgeServer(
            None,
            first,
            "key-0",
            backends=[(first, "key-0", _profile()), (second, "key-1", _profile())],
        )

        await server.stop_async()

        assert (first.closes, second.closes) == (1, 1)

    async def test_an_adapter_reachable_twice_is_closed_once(self):
        """``bridge_runner`` passes ``provider=backends[0][0]`` — the normal shape."""
        shared = _CountingAdapter()
        server = BridgeServer(None, shared, "key-0", backends=[(shared, "key-0", _profile())])

        await server.stop_async()

        assert shared.closes == 1

    async def test_it_reads_the_backing_fields_not_the_request_context(self):
        """Teardown is not a request, so it must not read request-scoped state.

        ``_active_provider`` resolves through a ``ContextVar`` that
        ``_select_backend`` writes per request.  An implementation that closed
        ``self._active_provider`` would pass every case above — none of them
        issues a request, so the var is unset and the property falls back to the
        backing field.  Setting it here is the only way that defect dies.
        """
        held = _CountingAdapter()
        stranger = _CountingAdapter()
        server = BridgeServer(None, held, "test-key")
        token = _backend_context.set({"provider": stranger, "key": "k", "model": "m", "idx": 0})

        try:
            await server.stop_async()
        finally:
            _backend_context.reset(token)

        assert held.closes == 1
        assert stranger.closes == 0, "teardown read the request-scoped backend, not the bridge's own"


class TestOneAdapterCannotCostTheOthers:
    """R7 — a provider's teardown failure is contained, not propagated."""

    async def test_a_failing_close_does_not_stop_the_next_adapter(self):
        failing, healthy = _CountingAdapter(fails=True), _CountingAdapter()
        server = BridgeServer(
            None,
            failing,
            "key-0",
            backends=[(failing, "key-0", _profile()), (healthy, "key-1", _profile())],
        )

        await server.stop_async()

        assert failing.closes == 1
        assert healthy.closes == 1

    async def test_a_failing_close_is_logged(self, caplog):
        failing = _CountingAdapter(fails=True)
        server = BridgeServer(None, failing, "test-key")

        with caplog.at_level("WARNING", logger="kitty.bridge.server"):
            await server.stop_async()

        assert "provider teardown failed" in caplog.text

    async def test_the_session_summary_is_still_written(self, tmp_path):
        """The `finally` that guarantees the record must survive a bad adapter."""
        summary = tmp_path / "summary.json"
        server = BridgeServer(
            None,
            _CountingAdapter(fails=True),
            "test-key",
            session_summary_path=summary,
        )

        await server.stop_async()

        assert summary.exists()


class _RecordingSession:
    """An aiohttp-session stand-in that records when the bridge closes it.

    Attributes:
        closed: Whether :meth:`close` has run — the flag ``stop_async`` guards on.
    """

    def __init__(self, label: str, events: list[str]) -> None:
        """Record closes into a shared sequence under ``label``.

        Args:
            label: The name to append when this session is closed.
            events: The shared sequence every teardown step appends to.
        """
        self.closed = False
        self._label = label
        self._events = events

    async def close(self) -> None:
        """Append this session's label and mark it closed."""
        self._events.append(self._label)
        self.closed = True


class TestTeardownOrder:
    """R7 — the order is the safety argument, so it is pinned, not assumed."""

    async def test_the_whole_teardown_sequence_is_pinned(self, tmp_path, monkeypatch):
        """Every step is given a double, and the sequence is compared whole.

        A never-started bridge has ``None`` for its runner, both sessions and its
        log file, so a test that simply calls ``stop_async`` records a
        one-element sequence and an "adapters come last" assertion passes while
        proving nothing.  Each step is populated here, and the comparison is
        equality against the full list rather than containment.

        The two constraints it pins: the runner drains **first**, because that
        drain is what makes closing a provider's transport safe at all; and the
        adapters come **last**, so a provider that fails to close cannot cost the
        bridge its own sessions or leave a state file claiming a live bridge.
        """
        events: list[str] = []

        class _RecordingAdapter(_CountingAdapter):
            async def aclose(self) -> None:
                """Record the close in the shared sequence."""
                events.append("adapter")
                await super().aclose()

        class _RecordingRunner:
            async def cleanup(self) -> None:
                """Record the runner drain in the shared sequence."""
                events.append("runner")

        state_file = tmp_path / "bridge_state.json"
        # `remove_state` is a function-body import, so the double has to patch the
        # name in `kitty.bridge.state`, not one bound on `kitty.bridge.server`.
        monkeypatch.setattr(
            "kitty.bridge.state.remove_state",
            lambda _path: events.append("state"),
        )

        server = BridgeServer(None, _RecordingAdapter(), "test-key", state_file=str(state_file))
        server._runner = _RecordingRunner()  # type: ignore[assignment]
        server._session = _RecordingSession("session", events)  # type: ignore[assignment]
        server._proxy_session = _RecordingSession("proxy_session", events)  # type: ignore[assignment]

        await server.stop_async()

        assert events == ["runner", "session", "proxy_session", "state", "adapter"]

    async def test_a_failing_bridge_teardown_still_closes_the_adapters(self):
        """Containment runs both ways — this is the direction R7 nearly missed.

        Skipping the adapters because the bridge's own cleanup failed would leak
        exactly what this ticket exists to fix.  The failure still propagates:
        the bridge's own teardown has no containment decision behind it.
        """
        provider = _CountingAdapter()
        server = BridgeServer(None, provider, "test-key")

        class _FailingSession:
            closed = False

            async def close(self) -> None:
                """Fail the bridge's own session close."""
                raise RuntimeError("bridge session close failed")

        server._session = _FailingSession()  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="bridge session close failed"):
            await server.stop_async()

        assert provider.closes == 1

    async def test_a_failing_bridge_teardown_still_writes_the_summary(self, tmp_path):
        """The `finally` that guarantees the record, exercised by the path that reaches it."""
        summary = tmp_path / "summary.json"
        server = BridgeServer(None, _CountingAdapter(), "test-key", session_summary_path=summary)

        class _FailingSession:
            closed = False

            async def close(self) -> None:
                """Fail the bridge's own session close."""
                raise RuntimeError("bridge session close failed")

        server._session = _FailingSession()  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="bridge session close failed"):
            await server.stop_async()

        assert summary.exists()

    async def test_a_failed_drain_deliberately_skips_the_adapters(self):
        """The one direction containment must NOT run, and it is a decision.

        An undrained handler may still hold a live stream, and closing a
        transport under one is the single shape ``curl_cffi`` is known to have
        crashed on.  Leaking a pool beats crashing the process, so a drain
        failure is fatal to the adapter close rather than contained.
        """
        provider = _CountingAdapter()
        server = BridgeServer(None, provider, "test-key")

        class _FailingRunner:
            async def cleanup(self) -> None:
                """Fail the drain."""
                raise RuntimeError("drain failed")

        server._runner = _FailingRunner()  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="drain failed"):
            await server.stop_async()

        assert provider.closes == 0, "an undrained handler may hold a live stream"
