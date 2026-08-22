"""Tests for CLI main entry point — argument passthrough."""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
import uuid

import pytest


class TestCLIParseKnownArgs:
    """Verify that argparse uses parse_known_args so unknown flags pass through."""

    @staticmethod
    def _make_parser() -> argparse.ArgumentParser:
        """Recreate the parser from kitty.cli.main."""
        parser = argparse.ArgumentParser(
            prog="kitty",
            description="Kitty Bridge — launch coding agents through a local API bridge.",
        )
        parser.add_argument("--version", "-v", action="version", version="kitty test")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument("command", nargs="*", help="Command to run")
        return parser

    def test_parse_args_rejects_unknown_flags(self) -> None:
        """parse_args() rejects unknown flags — this is the bug we're fixing."""
        parser = self._make_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["glm", "claude", "--dangerously-skip-permissions", "--resume", "foo"])

    def test_parse_known_args_passes_unknown_flags_through(self) -> None:
        """parse_known_args() should leave unknown flags in the remainder."""
        parser = self._make_parser()
        args, unknown = parser.parse_known_args(["glm", "claude", "--dangerously-skip-permissions", "--resume", "foo"])
        assert args.command == ["glm", "claude"]
        assert unknown == ["--dangerously-skip-permissions", "--resume", "foo"]

    def test_parse_known_args_debug_flag_still_works(self) -> None:
        """Known flags like --debug should still be parsed correctly."""
        parser = self._make_parser()
        args, unknown = parser.parse_known_args(["--debug", "claude", "--resume", "foo"])
        assert args.debug is True
        assert args.command == ["claude"]
        assert unknown == ["--resume", "foo"]

    def test_parse_known_args_no_unknown(self) -> None:
        """When there are no unknown flags, remainder should be empty."""
        parser = self._make_parser()
        args, unknown = parser.parse_known_args(["claude"])
        assert args.command == ["claude"]
        assert unknown == []

    def test_parse_known_args_empty(self) -> None:
        """Empty args should work."""
        parser = self._make_parser()
        args, unknown = parser.parse_known_args([])
        assert args.command == []
        assert unknown == []


class TestCLIIntegrationPassthrough:
    """Integration test: verify kitty passes agent flags through without error."""

    def test_unknown_flags_not_rejected(self) -> None:
        """Running kitty with agent-specific flags should not produce an argparse error."""
        result = subprocess.run(
            [sys.executable, "-m", "kitty", "--version"],
            capture_output=True,
            text=True,
        )
        # --version exits with 0 and prints version
        assert result.returncode == 0
        assert "kitty" in result.stdout

    def test_main_with_agent_flags_does_not_crash(self) -> None:
        """Running kitty main() with agent CLI flags should not raise SystemExit for argparse errors.

        We patch the router and launch_target to avoid needing a real profile/credential setup.
        """
        from unittest.mock import MagicMock, patch

        from kitty.cli.router import RouteResult
        from kitty.launchers.base import LauncherAdapter
        from kitty.profiles.schema import Profile

        profile = Profile(
            name="glm",
            provider="openrouter",
            model="claude-3",
            auth_ref=str(uuid.uuid4()),
            is_default=True,
        )
        mock_adapter = MagicMock(spec=LauncherAdapter)
        mock_result = RouteResult(
            adapter=mock_adapter,
            profile=profile,
            extra_args=["--dangerously-skip-permissions", "--resume", "tui-display-polish"],
        )

        with (
            patch(
                "sys.argv",
                ["kitty", "glm", "claude", "--dangerously-skip-permissions", "--resume", "tui-display-polish"],
            ),
            patch("kitty.cli.router.CLIRouter.route", return_value=mock_result),
            patch("kitty.cli.main._launch_target", return_value=0) as mock_launch,
        ):
            with pytest.raises(SystemExit) as exc_info:
                from kitty.cli.main import main

                main()

            # Should exit 0 from successful launch, NOT exit 2 from argparse error
            assert exc_info.value.code == 0
            # Verify extra_args were passed to launch_target
            launch_call = mock_launch.call_args
            assert launch_call[0][3] == ["--dangerously-skip-permissions", "--resume", "tui-display-polish"]


@contextlib.contextmanager
def _cli_run(argv: list[str], *, backends: list[object], egress: object, cleanup_exit: int = 0):
    """Drive ``kitty.cli.main.main`` with substituted stores and a fixed egress result.

    ``main`` imports its collaborators inside the function body, so patching
    them on their source modules takes effect at call time. The profile store is
    substituted as well, so a test never reads the developer's real
    ``profiles.json``.

    Args:
        argv: Full ``sys.argv`` for the run, including the program name.
        backends: What the substituted profile store reports; an empty list
            reproduces both the fresh-install and the corrupt-store state.
        egress: An ``Exception`` instance to raise from ``resolve_egress``, or
            the value it should return.
        cleanup_exit: Exit code the patched ``run_cleanup`` returns.

    Yields:
        The patched ``cleanup_cmd.run_cleanup`` mock.
    """
    from unittest.mock import MagicMock, patch

    # Import every module `main()` loads lazily BEFORE patching. Each of these
    # binds `resolve_egress` or `ProfileStore` with a top-level
    # `from ... import`, so a first import triggered by `main()` while the patch
    # is live would capture the mock permanently and leak into unrelated tests —
    # an order-dependent failure that only shows up in some run orders.
    import kitty.cli.auth_cmd  # noqa: F401
    import kitty.cli.cleanup_cmd  # noqa: F401
    import kitty.cli.doctor_cmd  # noqa: F401
    import kitty.cli.egress_cmd  # noqa: F401
    import kitty.cli.profile_cmd  # noqa: F401
    import kitty.cli.router  # noqa: F401
    import kitty.cli.setup_cmd  # noqa: F401
    import kitty.profiles.resolver  # noqa: F401

    store = MagicMock()
    store.get_all_backends.return_value = backends
    resolve = MagicMock(side_effect=egress) if isinstance(egress, Exception) else MagicMock(return_value=egress)
    run_cleanup = MagicMock(return_value=cleanup_exit)

    with (
        patch("sys.argv", argv),
        patch("kitty.profiles.store.ProfileStore", return_value=store),
        patch("kitty.egress_store.resolve_egress", resolve),
        patch("kitty.cli.cleanup_cmd.run_cleanup", run_cleanup),
    ):
        yield run_cleanup


class TestRecoveryCommandReachability:
    """Issue #27: ``kitty cleanup`` must survive the breakage it repairs.

    The router guard is covered at the unit level in ``test_cli_router.py``.
    These tests drive ``main()`` itself, because the second guard — egress
    resolution — sits in ``main`` and fires before routing happens at all.
    """

    def test_cleanup_runs_when_egress_cannot_resolve(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The trap in the issue: a poisoned gateway blocked its own repair tool."""
        from kitty.cli.main import main
        from kitty.egress import get_egress

        broken = ValueError("stored gateway no longer resolves")
        with (
            _cli_run(["kitty", "cleanup"], backends=[], egress=broken) as run,
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0
        run.assert_called_once_with()
        assert "stored gateway no longer resolves" not in capsys.readouterr().err
        # The exemption skips the exit, not the install: no stale route leaks in.
        assert get_egress() is None

    @pytest.mark.parametrize("command", ["cleanup", "egress"])
    def test_every_recovery_command_survives_a_broken_gateway(
        self,
        command: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Both members must clear the guard, or the shared set is only half used.

        Without ``egress`` here, narrowing ``main()``'s check back to
        ``== "cleanup"`` would keep the whole suite green.
        """
        from unittest.mock import patch

        from kitty.cli.main import main

        with (
            _cli_run(["kitty", command], backends=[], egress=ValueError("gateway gone")) as run,
            patch("kitty.cli.egress_cmd.run_egress_menu") as menu,
            contextlib.suppress(SystemExit),
        ):
            main()

        # Reaching the command at all is the claim; only `cleanup` exits.
        reached = run if command == "cleanup" else menu
        reached.assert_called_once()
        assert "gateway gone" not in capsys.readouterr().err

    def test_cleanup_exit_code_comes_from_run_cleanup(self) -> None:
        """A distinctive code proves the exit is cleanup's, not the guard's 1."""
        from kitty.cli.main import main

        with (
            _cli_run(["kitty", "cleanup"], backends=[], egress=ValueError("boom"), cleanup_exit=3),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 3

    def test_cleanup_runs_when_egress_resolves_normally(self) -> None:
        """The exemption must not skip installing a healthy egress route."""
        from unittest.mock import MagicMock

        from kitty.cli.main import main
        from kitty.egress import get_egress

        sentinel = MagicMock(name="egress-config")
        with (
            _cli_run(["kitty", "cleanup"], backends=[], egress=sentinel) as run,
            pytest.raises(SystemExit),
        ):
            main()

        run.assert_called_once_with()
        assert get_egress() is sentinel

    def test_non_recovery_command_still_fails_closed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Fail-closed is the point of the guard; only recovery commands are exempt."""
        from kitty.cli.main import main

        with (
            _cli_run(["kitty", "doctor"], backends=[object()], egress=ValueError("bad proxy URL")) as run,
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        assert "bad proxy URL" in capsys.readouterr().err
        run.assert_not_called()


class TestNonTTYExit:
    """Issue #27: off a TTY the CLI must diagnose, not raise a traceback."""

    @pytest.mark.parametrize("command", ["setup", "profile", "auth", "egress"])
    def test_interactive_command_exits_2_with_a_message(
        self,
        command: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Exit 2 is distinct from the generic 1, so a CI script can branch on it."""
        from unittest.mock import patch

        from kitty.cli.main import main

        with (
            _cli_run(["kitty", command], backends=[object()], egress=None),
            patch("sys.stdin.isatty", return_value=False),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 2
        # Rich soft-wraps at 80 columns, so collapse whitespace before matching.
        stderr = " ".join(capsys.readouterr().err.split())
        assert "interactive terminal" in stderr
        assert f"kitty {command}" in stderr
        assert "Traceback" not in stderr

    def test_hijacked_setup_explains_the_redirection(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The operator typed `doctor`; naming only `setup` is the old confusion."""
        from unittest.mock import patch

        from kitty.cli.main import main

        with (
            _cli_run(["kitty", "doctor"], backends=[], egress=None),
            patch("sys.stdin.isatty", return_value=False),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 2
        stderr = " ".join(capsys.readouterr().err.split())
        assert "No profiles are configured" in stderr
        assert "kitty cleanup" in stderr

    @pytest.mark.parametrize(
        ("argv", "backends"),
        [
            (["kitty", "setup"], []),
            (["kitty", "setup"], [object()]),
            (["kitty"], []),
            (["kitty"], [object()]),
        ],
        ids=["setup, fresh install", "setup, profiles present", "bare kitty, fresh install", "bare kitty, profiles"],
    )
    def test_no_redirection_note_when_nobody_was_redirected(
        self,
        argv: list[str],
        backends: list[object],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Only a *different* typed command means the router redirected anyone.

        The router raises ``needs_setup`` in three situations — the empty-store
        guard, an explicit ``kitty setup``, and bare ``kitty`` — so the flag on
        its own would claim a redirection in cases where none happened. The
        ``profiles present`` cases would additionally state something false.
        """
        from unittest.mock import patch

        from kitty.cli.main import main

        with (
            _cli_run(argv, backends=backends, egress=None),
            patch("sys.stdin.isatty", return_value=False),
            pytest.raises(SystemExit),
        ):
            main()

        stderr = " ".join(capsys.readouterr().err.split())
        assert "kitty setup requires an interactive terminal" in stderr
        assert "No profiles are configured" not in stderr

    def test_non_tty_error_is_still_raised_by_check_tty(self) -> None:
        """Only ``main``'s handling changes; the library contract is untouched."""
        from unittest.mock import patch

        from kitty.tui.prompts import NonTTYError, check_tty

        with patch("sys.stdin.isatty", return_value=False), pytest.raises(NonTTYError):
            check_tty()

    def test_other_exceptions_still_propagate(self) -> None:
        """The wrapper catches NonTTYError only — it must not swallow real bugs."""
        from unittest.mock import patch

        from kitty.cli.main import main

        with (
            _cli_run(["kitty", "setup"], backends=[object()], egress=None),
            patch("sys.stdin.isatty", return_value=True),
            patch("kitty.cli.setup_cmd.run_setup_wizard", side_effect=RuntimeError("wizard exploded")),
            pytest.raises(RuntimeError, match="wizard exploded"),
        ):
            main()


class TestSessionSummaryFlag:
    """The shutdown summary path is nominated on the command line (issue #26)."""

    def test_parser_accepts_the_session_summary_flag(self) -> None:
        """A4.1: the flag exists and yields a Path."""
        from pathlib import Path

        from kitty.cli.main import _build_parser

        args, _unknown = _build_parser().parse_known_args(["--session-summary", "out/run.json", "claude"])

        assert args.session_summary == Path("out/run.json")

    def test_session_summary_defaults_to_none(self) -> None:
        """A4.4: nothing is written unless the operator asks for it."""
        from kitty.cli.main import _build_parser

        args, _unknown = _build_parser().parse_known_args(["claude"])

        assert args.session_summary is None


# ── Bridge-mode CLI seams for the attribution surfaces (issue #26) ───────────


class _WiringSentinel(Exception):
    """Raised from a stubbed ``BridgeServer.__init__`` to stop the flow."""


def _profile_stub(name: str = "my-profile") -> object:
    """Build a minimal single-profile stand-in for bridge construction.

    Args:
        name: Profile name the attribution surfaces should report.

    Returns:
        A namespace carrying the attributes ``_run_bridge`` reads.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        name=name,
        provider="zai_regular",
        model="test-model",
        auth_ref="ref-1",
        provider_config={},
        backup=False,
    )


def _patch_bridge_prerequisites(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise everything ``_run_bridge*`` touches before constructing a server.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    monkeypatch.setattr("kitty.egress_guard.egress_block_reason", lambda *a, **k: None)
    monkeypatch.setattr("kitty.cli.main.egress_block_reason", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("kitty.providers.registry.get_provider", lambda *a, **k: object())
    monkeypatch.setattr("kitty.cli.main.get_provider", lambda *a, **k: object(), raising=False)


def _record_bridge_kwargs(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace ``BridgeServer`` with a stub that records its keyword arguments.

    Raising from ``__init__`` stops the flow before any event loop starts, so
    the assertion runs against the exact construction call under test.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        A list that receives one kwargs dict per construction attempt.
    """
    seen: list[dict] = []

    class _RecordingServer:
        """Stand-in for ``BridgeServer`` that records and aborts."""

        def __init__(self, *args, **kwargs):
            """Record the construction kwargs, then abort the flow."""
            seen.append(kwargs)
            raise _WiringSentinel

    monkeypatch.setattr("kitty.bridge.server.BridgeServer", _RecordingServer)
    return seen


def _balancing_stubs(monkeypatch: pytest.MonkeyPatch, member: object) -> None:
    """Stub the profile store and resolver a balancing bridge builds members from.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        member: The single member profile to resolve.
    """

    class FakeProfileStore:
        """Stand-in for ``ProfileStore`` — constructed inside the function."""

        def __init__(self, *args, **kwargs):
            """Accept and ignore the real store's arguments."""

    class FakeProfileResolver:
        """Stand-in resolver returning one balancing member."""

        def __init__(self, store):
            """Accept and ignore the store."""

        def resolve_balancing(self, name):
            """Return the single member profile."""
            return [member]

    monkeypatch.setattr("kitty.profiles.store.ProfileStore", FakeProfileStore)
    monkeypatch.setattr("kitty.profiles.resolver.ProfileResolver", FakeProfileResolver)


class TestBridgeModeAttributionWiring:
    """``kitty bridge`` must hand the bridge what the attribution surfaces need."""

    def test_run_bridge_passes_the_summary_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A4.2: the single-profile ``kitty bridge`` path nominates the summary."""
        from pathlib import Path
        from types import SimpleNamespace

        import kitty.cli.main as cli_main

        _patch_bridge_prerequisites(monkeypatch)
        seen = _record_bridge_kwargs(monkeypatch)
        cred_store = SimpleNamespace(get=lambda ref: "sk-test-key")

        with pytest.raises(_WiringSentinel):
            cli_main._run_bridge(
                _profile_stub(),
                cred_store,
                validate=False,
                session_summary_path=Path("out/run.json"),
            )

        assert seen[0]["session_summary_path"] == Path("out/run.json")

    def test_run_bridge_names_the_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A9.1: without this the bridge calls every session "default"."""
        from types import SimpleNamespace

        import kitty.cli.main as cli_main

        _patch_bridge_prerequisites(monkeypatch)
        seen = _record_bridge_kwargs(monkeypatch)
        cred_store = SimpleNamespace(get=lambda ref: "sk-test-key")

        with pytest.raises(_WiringSentinel):
            cli_main._run_bridge(_profile_stub(name="my-profile"), cred_store, validate=False)

        assert seen[0]["profile_name"] == "my-profile"

    def test_run_bridge_balancing_passes_the_summary_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A4.2: the balancing ``kitty bridge`` path nominates the summary too."""
        from pathlib import Path
        from types import SimpleNamespace

        import kitty.cli.main as cli_main

        _patch_bridge_prerequisites(monkeypatch)
        _balancing_stubs(monkeypatch, _profile_stub(name="member-1"))
        seen = _record_bridge_kwargs(monkeypatch)
        cred_store = SimpleNamespace(get=lambda ref: "sk-test-key")

        with pytest.raises(_WiringSentinel):
            cli_main._run_bridge_balancing(
                SimpleNamespace(name="ci-pool"),
                cred_store,
                validate=False,
                session_summary_path=Path("out/pool.json"),
            )

        assert seen[0]["session_summary_path"] == Path("out/pool.json")

    def test_run_bridge_balancing_names_the_pool_not_its_member(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A9.2: the session ran as a pool; naming one member misdescribes it."""
        from types import SimpleNamespace

        import kitty.cli.main as cli_main

        _patch_bridge_prerequisites(monkeypatch)
        _balancing_stubs(monkeypatch, _profile_stub(name="member-1"))
        seen = _record_bridge_kwargs(monkeypatch)
        cred_store = SimpleNamespace(get=lambda ref: "sk-test-key")

        with pytest.raises(_WiringSentinel):
            cli_main._run_bridge_balancing(SimpleNamespace(name="ci-pool"), cred_store, validate=False)

        assert seen[0]["profile_name"] == "ci-pool"


class TestLaunchTargetBalancingNaming:
    """The agent-launch balancing path names the pool, not its first member."""

    def test_balancing_launch_passes_the_pool_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A9.2 at the seam that implements it — ``_launch_target_balancing``."""
        from types import SimpleNamespace

        import kitty.cli.launcher as launcher_mod
        import kitty.cli.main as cli_main

        seen: list[dict] = []

        def _record_launch(**kwargs):
            """Record the launch kwargs instead of starting anything."""
            seen.append(kwargs)
            return 0

        monkeypatch.setattr(launcher_mod, "launch", _record_launch)
        monkeypatch.setattr("kitty.providers.registry.get_provider", lambda *a, **k: object())
        _balancing_stubs(monkeypatch, _profile_stub(name="member-1"))
        cred_store = SimpleNamespace(get=lambda ref: "sk-test-key")

        exit_code = cli_main._launch_target_balancing(
            object(),
            SimpleNamespace(name="ci-pool"),
            cred_store,
            [],
        )

        assert exit_code == 0
        assert seen[0]["profile_name"] == "ci-pool"


class TestBridgeModePanelAdvertisesStats:
    """An endpoint nobody is told about is an endpoint nobody uses."""

    @staticmethod
    def _capture_panel(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Stub the panel renderer to capture its body, then abort the flow.

        Args:
            monkeypatch: The pytest monkeypatch fixture.

        Returns:
            A list receiving the rendered panel body.
        """
        bodies: list[str] = []

        def _record_panel(title, body, *args, **kwargs):
            """Record the panel body, then stop before the server blocks."""
            bodies.append(body)
            raise _WiringSentinel

        monkeypatch.setattr("kitty.tui.display.print_panel", _record_panel)
        return bodies

    @staticmethod
    def _stub_server(monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace ``BridgeServer`` with one that starts and stops instantly.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """

        class _NoopServer:
            """Stand-in for ``BridgeServer`` that never binds a socket."""

            def __init__(self, *args, **kwargs):
                """Accept and ignore the real server's arguments."""

            async def start_async(self) -> int:
                """Report a fixed port without listening."""
                return 12345

            async def stop_async(self) -> None:
                """Do nothing — no resources were acquired."""

        monkeypatch.setattr("kitty.bridge.server.BridgeServer", _NoopServer)

    @staticmethod
    def _skip_catalog_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep the panel tests offline.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        import kitty.providers.model_context_sync as sync

        async def _skip_refresh(**kwargs):
            """Report success without touching the network."""
            return True

        monkeypatch.setattr(sync, "refresh_model_context_overrides", _skip_refresh)

    def test_single_profile_panel_lists_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A8.1: the single-profile bridge panel advertises ``GET /stats``."""
        from types import SimpleNamespace

        import kitty.cli.main as cli_main

        self._skip_catalog_refresh(monkeypatch)
        _patch_bridge_prerequisites(monkeypatch)
        self._stub_server(monkeypatch)
        bodies = self._capture_panel(monkeypatch)
        cred_store = SimpleNamespace(get=lambda ref: "sk-test-key")

        with pytest.raises(_WiringSentinel):
            cli_main._run_bridge(_profile_stub(), cred_store, validate=False)

        assert "/stats" in bodies[0]
        assert "/healthz" in bodies[0]

    def test_balancing_panel_lists_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A8.1: the balancing bridge panel advertises ``GET /stats`` too."""
        from types import SimpleNamespace

        import kitty.cli.main as cli_main

        self._skip_catalog_refresh(monkeypatch)
        _patch_bridge_prerequisites(monkeypatch)
        _balancing_stubs(monkeypatch, _profile_stub(name="member-1"))
        self._stub_server(monkeypatch)
        bodies = self._capture_panel(monkeypatch)
        cred_store = SimpleNamespace(get=lambda ref: "sk-test-key")

        with pytest.raises(_WiringSentinel):
            cli_main._run_bridge_balancing(SimpleNamespace(name="ci-pool"), cred_store, validate=False)

        assert "/stats" in bodies[0]
        assert "/healthz" in bodies[0]
