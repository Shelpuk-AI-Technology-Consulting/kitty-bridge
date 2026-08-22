"""Tests for the bridge's session attribution counters and ``/stats`` surface.

Issue #26: the bridge substitutes the backend's real model for the requested
one but exposes no machine-readable record of it, so consumers attribute cost
and failures to the alias.  These tests pin the accumulator that feeds all
three attribution surfaces (``GET /stats``, the shutdown summary file, and the
``X-Kitty-*`` headers).
"""

import contextvars
import json
import uuid

import aiohttp
import pytest

from kitty.bridge.server import AllBackendsUnhealthyError, BridgeServer
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter


class StubProvider(ProviderAdapter):
    """Minimal provider adapter standing in for a real upstream.

    Args:
        provider_type: Value reported by the :attr:`provider_type` property.
    """

    def __init__(self, provider_type: str = "stub") -> None:
        self._provider_type = provider_type

    @property
    def provider_type(self) -> str:
        """Return the provider identifier used in attribution records."""
        return self._provider_type

    @property
    def default_base_url(self) -> str:
        """Return the upstream base URL (never contacted in these tests)."""
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        """Return a Chat Completions payload for the given model and messages."""
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        """Return the upstream response unchanged."""
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        """Return a generic exception for an upstream error status."""
        return Exception(f"Error {status_code}")


def _make_backends(n: int, backup_indices: set[int] | None = None):
    """Build ``n`` balancing backend tuples with distinct providers and models.

    Args:
        n: Number of backends to build.
        backup_indices: Indices whose profile is a reserve-tier member.

    Returns:
        A list of ``(provider, resolved_key, profile)`` tuples.
    """
    backup_indices = backup_indices or set()
    backends = []
    for i in range(n):
        provider = StubProvider(provider_type=f"stub-{i}")
        profile = Profile(
            name=f"profile-{i}",
            provider="openai",
            model=f"real-model-{i}",
            auth_ref=str(uuid.uuid4()),
            backup=i in backup_indices,
        )
        backends.append((provider, f"key-{i}", profile))
    return backends


def _make_server(n: int = 2, backup_indices: set[int] | None = None, **kwargs) -> BridgeServer:
    """Build a balancing ``BridgeServer`` over ``n`` stub backends.

    Args:
        n: Number of balancing backends.
        backup_indices: Indices marked as reserve tier.
        **kwargs: Extra keyword arguments forwarded to ``BridgeServer``.

    Returns:
        An unstarted server — ``__init__`` performs no I/O.
    """
    backends = _make_backends(n, backup_indices)
    return BridgeServer(
        adapter=None,
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model=backends[0][2].model,
        backends=backends,
        profile_name="test-pool",
        **kwargs,
    )


def _make_single_server(**kwargs) -> BridgeServer:
    """Build a non-balancing ``BridgeServer`` with one configured model.

    Args:
        **kwargs: Extra keyword arguments forwarded to ``BridgeServer``.

    Returns:
        An unstarted single-backend server.
    """
    return BridgeServer(
        adapter=None,
        provider=StubProvider(provider_type="stub-single"),
        resolved_key="key",
        model="real-model-single",
        profile_name="solo",
        **kwargs,
    )


class TestCounterBaseline:
    """A server that has served nothing reports zeroes, not absent keys."""

    def test_fresh_server_reports_zero_counters(self) -> None:
        """A1.1: every counter starts at zero on an unused server."""
        stats = _make_server()._session_stats()

        assert stats["requests"] == 0
        assert stats["attempts"] == 0
        assert stats["failovers"] == 0
        assert stats["retries"] == 0
        assert stats["all_backends_unhealthy"] == 0
        assert stats["models_served"] == {}

    def test_fresh_server_lists_every_backend_with_zero_attempts(self) -> None:
        """A1.1: backends appear before they are used, so consumers see the pool."""
        stats = _make_server(3)._session_stats()

        assert [b["name"] for b in stats["backends"]] == ["profile-0", "profile-1", "profile-2"]
        assert all(b["attempts"] == 0 for b in stats["backends"])


class TestSelectionCounting:
    """``_select_backend`` is the single funnel every upstream attempt goes through."""

    def test_one_selection_counts_one_request_and_no_failover(self) -> None:
        """A1.2: the first selection in a request is the request, not a failover."""
        server = _make_server()

        contextvars.copy_context().run(server._select_backend)

        stats = server._session_stats()
        assert stats["requests"] == 1
        assert stats["attempts"] == 1
        assert stats["failovers"] == 0

    def test_switching_backend_within_one_request_counts_a_failover(self, monkeypatch) -> None:
        """A1.3: a failover is a switch to a *different* backend mid-request."""
        import kitty.bridge.server as server_mod

        server = _make_server(2)
        chosen = iter([[0], [1]])
        monkeypatch.setattr(server_mod.random, "choices", lambda *a, **kw: next(chosen))

        def _one_request() -> None:
            """Select backend 0, then fail over to backend 1."""
            server._select_backend()
            server._select_backend()

        contextvars.copy_context().run(_one_request)

        stats = server._session_stats()
        assert stats["requests"] == 1
        assert stats["attempts"] == 2
        assert stats["failovers"] == 1
        assert stats["retries"] == 0

    def test_reselecting_the_same_backend_is_a_retry_not_a_failover(self) -> None:
        """A1.3: a same-backend re-selection is not evidence a provider failed over.

        Several paths re-select after an empty-but-successful response, and a
        weighted draw can return the same member. Counting those as failovers
        would inflate the one number the issue was filed to obtain.
        """
        server = _make_server(1)

        def _one_request() -> None:
            """Select the only backend twice, as an empty-response retry does."""
            server._select_backend()
            server._select_backend()

        contextvars.copy_context().run(_one_request)

        stats = server._session_stats()
        assert stats["requests"] == 1
        assert stats["attempts"] == 2
        assert stats["failovers"] == 0
        assert stats["retries"] == 1

    def test_separate_requests_do_not_inflate_the_failover_count(self) -> None:
        """A7.2: a fresh request context starts at attempt 1, so failovers stay 0."""
        server = _make_server()

        for _ in range(3):
            contextvars.copy_context().run(server._select_backend)

        stats = server._session_stats()
        assert stats["requests"] == 3
        assert stats["attempts"] == 3
        assert stats["failovers"] == 0

    def test_attempts_are_attributed_to_the_backend_actually_selected(self) -> None:
        """A1.6: per-backend attempts follow the chosen index, not the first one."""
        server = _make_server(3)
        # Park backends 0 and 2 so selection can only land on backend 1.
        server._mark_backend_unhealthy(0)
        server._mark_backend_unhealthy(2)

        contextvars.copy_context().run(server._select_backend)

        by_name = {b["name"]: b for b in server._session_stats()["backends"]}
        assert by_name["profile-1"]["attempts"] == 1
        assert by_name["profile-0"]["attempts"] == 0
        assert by_name["profile-2"]["attempts"] == 0

    def test_model_attempts_are_keyed_by_the_real_model(self) -> None:
        """A1.6: attribution keys on the backend's model, never the requested alias."""
        server = _make_server(3)
        server._mark_backend_unhealthy(0)
        server._mark_backend_unhealthy(2)

        contextvars.copy_context().run(server._select_backend)

        assert server._session_stats()["models_served"]["real-model-1"]["attempts"] == 1


class TestUsageCounting:
    """Token totals belong to the model that produced them."""

    def test_completion_tokens_are_attributed_to_the_serving_model(self) -> None:
        """A1.4: a completed call records completions and both token counts."""
        server = _make_server(1)

        def _one_request() -> None:
            """Select the only backend, then report its usage."""
            server._select_backend()
            server._log_usage({"prompt_tokens": 10, "completion_tokens": 3})

        contextvars.copy_context().run(_one_request)

        served = server._session_stats()["models_served"]["real-model-0"]
        assert served["completions"] == 1
        assert served["input_tokens"] == 10
        assert served["output_tokens"] == 3

    def test_usage_is_counted_even_when_usage_logging_is_disabled(self) -> None:
        """A1.4: attribution must not depend on the opt-in ``--logging`` flag."""
        server = _make_server(1, logging_enabled=False)

        def _one_request() -> None:
            """Select a backend and report usage with file logging off."""
            server._select_backend()
            server._log_usage({"prompt_tokens": 7, "completion_tokens": 2})

        contextvars.copy_context().run(_one_request)

        assert server._session_stats()["models_served"]["real-model-0"]["input_tokens"] == 7

    def test_usage_log_file_still_requires_logging_enabled(self, tmp_path) -> None:
        """A6.3: counting usage must not start writing the usage log by itself."""
        usage_log = tmp_path / "usage.log"
        server = _make_server(1, logging_enabled=False, _usage_log_path=usage_log)

        def _one_request() -> None:
            """Select a backend and report usage with file logging off."""
            server._select_backend()
            server._log_usage({"prompt_tokens": 1, "completion_tokens": 1})

        contextvars.copy_context().run(_one_request)

        assert not usage_log.exists()

    def test_a_call_with_no_usage_record_counts_as_a_completion(self) -> None:
        """A1.4: a completion with no usage block still happened; tokens stay 0."""
        server = _make_server(1)

        def _one_request() -> None:
            """Select a backend and report a completion carrying no usage."""
            server._select_backend()
            server._log_usage(None)

        contextvars.copy_context().run(_one_request)

        served = server._session_stats()["models_served"]["real-model-0"]
        assert served["completions"] == 1
        assert served["input_tokens"] == 0
        assert served["output_tokens"] == 0


class TestAllBackendsUnhealthyCounting:
    """The "all backends unhealthy: never" flag the issue asks for by name."""

    def test_far_cooldown_raise_is_counted(self, monkeypatch) -> None:
        """A1.5: the fast-fail raise site increments the counter."""
        import kitty.bridge.server as server_mod

        server = _make_server(2)
        now = 1000.0
        for idx in range(2):
            health = server._backend_health[idx]
            health["healthy"] = False
            health["failed_at"] = now
            health["cooldown"] = 300
        monkeypatch.setattr(server_mod.time, "monotonic", lambda: now + 10)

        with pytest.raises(AllBackendsUnhealthyError):
            server._get_next_backend()

        assert server._session_stats()["all_backends_unhealthy"] == 1

    def test_no_stream_capable_backend_raise_is_counted(self) -> None:
        """A1.5: the require_streaming raise site increments the counter too."""
        server = _make_server(2)

        with pytest.raises(AllBackendsUnhealthyError):
            server._get_next_backend(require_streaming=True)

        assert server._session_stats()["all_backends_unhealthy"] == 1

    def test_a_healthy_session_reports_it_never_happened(self) -> None:
        """A1.5: a session that never exhausted its pool says so explicitly."""
        server = _make_server(2)

        contextvars.copy_context().run(server._select_backend)

        assert server._session_stats()["all_backends_unhealthy"] == 0


class TestBackendDescription:
    """Each backend entry names what a consumer needs to attribute a call."""

    def test_reserve_members_are_reported_as_the_backup_tier(self) -> None:
        """A2.3: tier distinguishes the reserve pool from the primary pool."""
        stats = _make_server(2, backup_indices={1})._session_stats()

        by_name = {b["name"]: b for b in stats["backends"]}
        assert by_name["profile-0"]["tier"] == "primary"
        assert by_name["profile-1"]["tier"] == "backup"

    def test_backend_entries_name_provider_and_real_model(self) -> None:
        """A2.2: provider and model come from the backend, not the request."""
        entry = _make_server(1)._session_stats()["backends"][0]

        assert entry["provider"] == "stub-0"
        assert entry["model"] == "real-model-0"

    def test_cooldown_events_track_lifetime_failures(self) -> None:
        """A2.4: cooldown_events is the lifetime failure count, not current health."""
        server = _make_server(1)
        server._mark_backend_unhealthy(0)
        server._mark_backend_healthy(0)

        entry = server._session_stats()["backends"][0]
        assert entry["cooldown_events"] == 1
        assert entry["healthy"] is True

    def test_single_backend_mode_reports_one_synthesised_entry(self) -> None:
        """A2.1: consumers parse one shape whether or not balancing is on."""
        stats = _make_single_server()._session_stats()

        assert stats["mode"] == "single"
        assert stats["backends"] == [
            {
                "name": "solo",
                "provider": "stub-single",
                "model": "real-model-single",
                "tier": "primary",
                "attempts": 0,
                "healthy": True,
                "remaining_cooldown": 0,
                "cooldown_events": 0,
            }
        ]

    def test_balancing_mode_is_labelled_as_such(self) -> None:
        """A2.2: mode tells the consumer whether failover was even possible."""
        assert _make_server(2)._session_stats()["mode"] == "balancing"

    def test_single_mode_can_never_report_a_failover(self) -> None:
        """A2.8: in single mode ``failovers`` is structurally 0 and carries no signal.

        A consumer must read ``mode`` first: zero failovers here is not
        evidence that nothing went wrong, only that there was nowhere to go.
        """
        server = _make_single_server()

        def _one_request() -> None:
            """Re-select the only backend, as a retry path does."""
            server._select_backend()
            server._select_backend()

        contextvars.copy_context().run(_one_request)

        stats = server._session_stats()
        assert stats["failovers"] == 0
        assert stats["retries"] == 1

    def test_single_backend_selection_is_counted(self) -> None:
        """A1.6: attribution works without a balancing pool."""
        server = _make_single_server()

        contextvars.copy_context().run(server._select_backend)

        stats = server._session_stats()
        assert stats["attempts"] == 1
        assert stats["backends"][0]["attempts"] == 1
        assert stats["models_served"]["real-model-single"]["attempts"] == 1


class TestDocumentContract:
    """The document is what two independent surfaces both serve."""

    def test_document_is_json_serialisable(self) -> None:
        """A2.6: the same object is served over HTTP and written to a file.

        One snapshot is round-tripped rather than compared against a second
        call — ``generated_at`` has second resolution and can tick between two.
        """
        server = _make_server(2, backup_indices={1})
        contextvars.copy_context().run(server._select_backend)
        snapshot = server._session_stats()

        assert json.loads(json.dumps(snapshot)) == snapshot

    def test_document_names_the_profile_and_a_generation_time(self) -> None:
        """A2.2: a summary read hours later must say what it describes and when."""
        stats = _make_server()._session_stats()

        assert stats["profile"] == "test-pool"
        assert stats["generated_at"].endswith("Z")

    def test_started_at_is_absent_before_the_server_starts(self) -> None:
        """A2.2: an unstarted server has no start time to report."""
        assert _make_server()._session_stats()["started_at"] is None


class TestSummaryPathResolution:
    """Where the shutdown summary goes is nominated, never guessed."""

    def test_no_nomination_means_no_summary(self, monkeypatch) -> None:
        """A4.4: the summary is strictly opt-in."""
        monkeypatch.delenv("KITTY_SESSION_SUMMARY", raising=False)

        assert _make_server()._session_summary_path is None

    def test_environment_variable_nominates_the_path(self, monkeypatch, tmp_path) -> None:
        """A4.3: service and embedded bridges have no CLI flag to pass."""
        monkeypatch.setenv("KITTY_SESSION_SUMMARY", str(tmp_path / "env.json"))

        assert _make_server()._session_summary_path == tmp_path / "env.json"

    def test_explicit_argument_wins_over_the_environment(self, monkeypatch, tmp_path) -> None:
        """A4.3: the CLI flag overrides an inherited environment variable."""
        monkeypatch.setenv("KITTY_SESSION_SUMMARY", str(tmp_path / "env.json"))

        server = _make_server(session_summary_path=tmp_path / "flag.json")

        assert server._session_summary_path == tmp_path / "flag.json"


class TestConcurrentRequests:
    """Interleaved requests each keep their own attempt ordinal."""

    @pytest.mark.asyncio
    async def test_two_interleaved_requests_each_count_one_request(self) -> None:
        """A7.1: concurrency must not turn one request's retry into another's."""
        import asyncio

        server = _make_server(2)
        gate = asyncio.Event()

        async def _request() -> None:
            """Select, yield to the peer request, then select again."""
            server._select_backend()
            gate.set()
            await asyncio.sleep(0)
            server._select_backend()

        await asyncio.gather(_request(), _request())

        stats = server._session_stats()
        assert stats["requests"] == 2
        assert stats["attempts"] == 4
        # Which of the two second selections switched backend is a random draw;
        # what must hold is that neither request's re-selection was mistaken
        # for the other request's first one.
        assert stats["failovers"] + stats["retries"] == 2


class TestStatsEndpoint:
    """``GET /stats`` is the live sibling of ``/healthz``."""

    @pytest.mark.asyncio
    async def test_stats_endpoint_serves_the_document(self) -> None:
        """A2.1: the endpoint exists and returns the session document."""
        server = _make_single_server()
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session, session.get(f"http://127.0.0.1:{port}/stats") as resp:
                assert resp.status == 200
                body = await resp.json()
        finally:
            await server.stop_async()

        assert body["mode"] == "single"
        assert len(body["backends"]) == 1

    @pytest.mark.asyncio
    async def test_stats_endpoint_lists_every_balancing_member(self) -> None:
        """A2.2: the pool is described member by member."""
        server = _make_server(3, backup_indices={2})
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session, session.get(f"http://127.0.0.1:{port}/stats") as resp:
                body = await resp.json()
        finally:
            await server.stop_async()

        assert [b["name"] for b in body["backends"]] == ["profile-0", "profile-1", "profile-2"]
        assert body["backends"][2]["tier"] == "backup"

    @pytest.mark.asyncio
    async def test_stats_agrees_with_healthz_about_backend_health(self, monkeypatch) -> None:
        """A2.4: two surfaces over one state must never disagree.

        The clock is frozen for both requests: ``remaining_cooldown`` is an
        ``int()`` of a live subtraction, so two unfrozen reads straddling a
        whole-second boundary differ by 1 and the test would flake.
        """
        import kitty.bridge.server as server_mod

        server = _make_server(2)
        server._mark_backend_unhealthy(0)
        monkeypatch.setattr(server_mod.time, "monotonic", lambda: server._backend_health[0]["failed_at"] + 5)
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/healthz") as resp:
                    health = await resp.json()
                async with session.get(f"http://127.0.0.1:{port}/stats") as resp:
                    stats = await resp.json()
        finally:
            await server.stop_async()

        health_by_name = {b["name"]: b for b in health["backends"]}
        for entry in stats["backends"]:
            peer = health_by_name[entry["name"]]
            assert entry["healthy"] == peer["healthy"]
            assert entry["remaining_cooldown"] == peer["remaining_cooldown"]
            # /stats renames two /healthz fields; the values must still agree.
            assert entry["cooldown_events"] == peer["failure_count"]
            assert (entry["tier"] == "backup") == peer["backup"]

    @pytest.mark.asyncio
    async def test_stats_is_open_when_no_keys_file_is_configured(self) -> None:
        """A2.7: ``kitty claude`` deliberately runs the bridge without a keys file.

        On that path ``/stats`` is readable by any local process, exactly like
        ``/healthz``. Asserted so the exposure is a recorded decision rather
        than an accident of middleware ordering.
        """
        server = _make_server(1)
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session, session.get(f"http://127.0.0.1:{port}/stats") as resp:
                assert resp.status == 200
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_stats_identifies_the_process_that_produced_it(self) -> None:
        """A2.9: two bridges nominating one summary path must be tellable apart."""
        import os

        server = _make_single_server()
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session, session.get(f"http://127.0.0.1:{port}/stats") as resp:
                body = await resp.json()
        finally:
            await server.stop_async()

        assert body["pid"] == os.getpid()
        assert body["port"] == port

    @pytest.mark.asyncio
    async def test_stats_requires_a_key_when_one_is_configured(self, tmp_path) -> None:
        """A2.5: the document names profiles, providers and models."""
        keys_file = tmp_path / "keys.txt"
        keys_file.write_text("sk-test-key\n", encoding="utf-8")
        server = _make_server(1, keys_file=str(keys_file))
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session, session.get(f"http://127.0.0.1:{port}/stats") as resp:
                assert resp.status == 401
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_stats_reports_the_start_time_once_running(self) -> None:
        """A2.2: a live bridge can say how long the session has been open."""
        server = _make_single_server()
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session, session.get(f"http://127.0.0.1:{port}/stats") as resp:
                body = await resp.json()
        finally:
            await server.stop_async()

        assert body["started_at"].endswith("Z")

    @pytest.mark.asyncio
    async def test_stats_carries_no_attribution_headers(self) -> None:
        """A5.3: no backend serves ``/stats``, so nothing may be attributed to one."""
        server = _make_single_server()
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session, session.get(f"http://127.0.0.1:{port}/stats") as resp:
                assert "X-Kitty-Backend" not in resp.headers
            async with aiohttp.ClientSession() as session, session.get(f"http://127.0.0.1:{port}/healthz") as resp:
                assert "X-Kitty-Backend" not in resp.headers
        finally:
            await server.stop_async()


class TestSessionSummaryFile:
    """The record that survives the bridge being torn down."""

    @pytest.mark.asyncio
    async def test_summary_is_written_on_shutdown(self, tmp_path) -> None:
        """A3.1: CI uploads a file, not a 42 MB debug log."""
        target = tmp_path / "summary.json"
        server = _make_single_server(session_summary_path=target)
        await server.start_async()
        await server.stop_async()

        assert json.loads(target.read_text(encoding="utf-8"))["mode"] == "single"

    @pytest.mark.asyncio
    async def test_summary_matches_the_live_document(self, tmp_path) -> None:
        """A3.1: the file and the endpoint are two readings of one accumulator."""
        target = tmp_path / "summary.json"
        server = _make_single_server(session_summary_path=target)
        port = await server.start_async()
        async with aiohttp.ClientSession() as session, session.get(f"http://127.0.0.1:{port}/stats") as resp:
            live = await resp.json()
        await server.stop_async()

        written = json.loads(target.read_text(encoding="utf-8"))
        live.pop("generated_at")
        written.pop("generated_at")
        assert written == live

    @pytest.mark.asyncio
    async def test_summary_reflects_counters_accumulated_before_shutdown(self, tmp_path) -> None:
        """A3.5: a summary that always reports zero would be worthless."""
        target = tmp_path / "summary.json"
        server = _make_single_server(session_summary_path=target)
        await server.start_async()
        contextvars.copy_context().run(server._select_backend)
        await server.stop_async()

        assert json.loads(target.read_text(encoding="utf-8"))["attempts"] == 1

    @pytest.mark.asyncio
    async def test_missing_parent_directories_are_created(self, tmp_path) -> None:
        """A3.2: CI nominates a path inside a directory it has not made yet."""
        target = tmp_path / "artifacts" / "run-1" / "summary.json"
        server = _make_single_server(session_summary_path=target)
        await server.start_async()
        await server.stop_async()

        assert target.exists()

    @pytest.mark.asyncio
    async def test_no_summary_is_written_when_none_is_nominated(self, monkeypatch) -> None:
        """A3.3: an un-nominated bridge writes nothing, anywhere.

        Asserting on an empty ``tmp_path`` would hold whatever the code did,
        since the server is never told about it — so the write itself is spied.
        """
        monkeypatch.delenv("KITTY_SESSION_SUMMARY", raising=False)
        server = _make_single_server()
        writes: list = []
        monkeypatch.setattr(
            type(server._usage_log_path),
            "write_text",
            lambda self, *a, **kw: writes.append(self),
        )
        await server.start_async()
        await server.stop_async()

        assert server._session_summary_path is None
        assert writes == []

    @pytest.mark.asyncio
    async def test_an_unwritable_path_does_not_break_shutdown(self, tmp_path, caplog) -> None:
        """A3.4: observability must never be able to fail the thing observed."""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        server = _make_single_server(session_summary_path=blocker / "summary.json")
        await server.start_async()

        with caplog.at_level("WARNING", logger="kitty.bridge.server"):
            await server.stop_async()

        assert server._runner is None
        assert any("Failed to write session summary" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_a_failing_teardown_still_leaves_the_record(self, tmp_path, monkeypatch) -> None:
        """A3.8: a graceful shutdown that trips must not cost CI its artifact."""
        target = tmp_path / "summary.json"
        server = _make_single_server(session_summary_path=target)
        await server.start_async()

        def _boom(path: str) -> None:
            """Fail the state-file removal, as a locked file would."""
            raise OSError("state file locked")

        monkeypatch.setattr("kitty.bridge.state.remove_state", _boom)
        server._state_file = str(tmp_path / "state.json")

        with pytest.raises(OSError, match="state file locked"):
            await server.stop_async()

        assert target.exists()
