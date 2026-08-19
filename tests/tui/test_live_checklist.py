"""Tests for LiveChecklist — live-updating doctor display."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

from kitty.tui.display import CheckItem, LiveChecklist


class TestCheckItem:
    def test_pending_status(self) -> None:
        item = CheckItem("test")
        assert item.ok is None
        assert "⠿" in item.status_text

    def test_success_status(self) -> None:
        item = CheckItem("test")
        item.ok = True
        assert "✓" in item.status_text

    def test_failure_status(self) -> None:
        item = CheckItem("test")
        item.ok = False
        assert "✗" in item.status_text


class TestLiveChecklistNonTTY:
    """Tests for LiveChecklist fallback (non-TTY) mode."""

    def test_add_prints_pending(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            cl = LiveChecklist("test")
            cl.add("check one")
        assert "⠿" in buf.getvalue()
        assert "check one" in buf.getvalue()

    def test_resolve_success_prints_status(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            cl = LiveChecklist("test")
            item = cl.add("check one")
            cl.resolve(item, True, "detail msg")
        output = buf.getvalue()
        assert "✓" in output
        assert "detail msg" in output

    def test_resolve_failure_prints_error(self) -> None:
        buf_err = StringIO()
        with patch("sys.stderr", buf_err):
            cl = LiveChecklist("test")
            item = cl.add("check one")
            cl.resolve(item, False, "broken")
        assert "✗" in buf_err.getvalue()
        assert "broken" in buf_err.getvalue()

    def test_run_checks_returns_failure_count(self) -> None:
        with patch("sys.stdout", new=StringIO()), patch("sys.stderr", new=StringIO()):
            cl = LiveChecklist("test")
            failures = cl.run_checks(
                [
                    ("ok check", lambda: (True, "fine")),
                    ("bad check", lambda: (False, "broken")),
                ]
            )
        assert failures == 1

    def test_run_checks_catches_exceptions(self) -> None:
        with patch("sys.stdout", new=StringIO()), patch("sys.stderr", new=StringIO()):
            cl = LiveChecklist("test")

            def _boom() -> tuple[bool, str]:
                raise RuntimeError("kaboom")

            failures = cl.run_checks([("boom", _boom)])
        assert failures == 1

    def test_run_checks_all_pass_returns_zero(self) -> None:
        with patch("sys.stdout", new=StringIO()):
            cl = LiveChecklist("test")
            failures = cl.run_checks(
                [
                    ("a", lambda: (True, "")),
                    ("b", lambda: (True, "ok")),
                ]
            )
        assert failures == 0


class TestLiveChecklistTTY:
    """Tests for LiveChecklist live (TTY) mode — uses Rich.Live."""

    def _make_tty_checklist(self) -> LiveChecklist:
        """Create a LiveChecklist that thinks it's in a TTY."""
        with patch("sys.stdout.isatty", return_value=True):
            cl = LiveChecklist("test")
        return cl

    def test_run_checks_uses_live(self) -> None:
        with patch("kitty.tui.display.Live") as mock_live:
            mock_live.return_value.__enter__ = MagicMock(return_value=mock_live.return_value)
            mock_live.return_value.__exit__ = MagicMock(return_value=None)
            cl = self._make_tty_checklist()
            with patch("sys.stdout", new=StringIO()):
                cl.run_checks([("check a", lambda: (True, "ok"))])
            mock_live.assert_called_once()

    def test_run_checks_returns_correct_failure_count(self) -> None:
        with patch("kitty.tui.display.Live") as mock_live:
            mock_live.return_value.__enter__ = MagicMock(return_value=mock_live.return_value)
            mock_live.return_value.__exit__ = MagicMock(return_value=None)
            cl = self._make_tty_checklist()
            with patch("sys.stdout", new=StringIO()):
                failures = cl.run_checks(
                    [
                        ("a", lambda: (True, "")),
                        ("b", lambda: (False, "err")),
                        ("c", lambda: (True, "ok")),
                    ]
                )
        assert failures == 1

    def test_run_checks_exception_caught(self) -> None:
        with patch("kitty.tui.display.Live") as mock_live:
            mock_live.return_value.__enter__ = MagicMock(return_value=mock_live.return_value)
            mock_live.return_value.__exit__ = MagicMock(return_value=None)
            cl = self._make_tty_checklist()

            def _boom() -> tuple[bool, str]:
                raise ValueError("ouch")

            with patch("sys.stdout", new=StringIO()):
                failures = cl.run_checks([("boom", _boom)])
        assert failures == 1


class TestLiveChecklistTTYRendering:
    """What a person actually sees in a terminal.

    The tests above mock ``Live`` away, so they verify the return value but
    never the display — which is how `kitty doctor` came to show nothing but
    pending spinners for every check while still exiting with the right code.
    These render for real and assert on the output.
    """

    @staticmethod
    def _render(checks: list[tuple[str, object]]) -> tuple[str, int]:
        """Run checks through the TTY path and capture what was drawn.

        Args:
            checks: ``(label, callable)`` pairs, as ``run_checks`` takes.

        Returns:
            ``(rendered_output, failure_count)``.
        """
        from rich.console import Console

        buf = StringIO()
        console = Console(file=buf, width=90, force_terminal=False)
        with (
            patch("sys.stdout.isatty", return_value=True),
            patch("kitty.tui.display._console", console),
        ):
            checklist = LiveChecklist("kitty doctor")
            failures = checklist.run_checks(checks)  # type: ignore[arg-type]
        return buf.getvalue(), failures

    def test_resolved_checks_show_their_status_symbol(self) -> None:
        out, failures = self._render(
            [("Target claude", lambda: (True, "binary found")), ("Creds", lambda: (False, "no key"))]
        )

        assert "✓" in out, f"passing check never rendered its result:\n{out}"
        assert "✗" in out, f"failing check never rendered its result:\n{out}"
        assert failures == 1

    def test_detail_text_reaches_the_user(self) -> None:
        """A bare pass/fail is not a diagnosis — the reason has to be shown."""
        out, _ = self._render([("Target claude", lambda: (False, "binary not found on PATH"))])

        assert "binary not found on PATH" in out, f"the reason for the failure was never displayed:\n{out}"

    def test_no_check_is_left_pending(self) -> None:
        out, _ = self._render([("a", lambda: (True, "ok")), ("b", lambda: (True, "ok"))])

        assert "⠿" not in out, f"a finished check is still showing the pending spinner:\n{out}"

    def test_checks_sharing_a_label_both_resolve(self) -> None:
        """Two profiles can produce identically-labelled checks."""
        out, failures = self._render(
            [("Credentials", lambda: (True, "resolved-first")), ("Credentials", lambda: (False, "missing-second"))]
        )

        assert "resolved-first" in out
        assert "missing-second" in out
        assert failures == 1

    def test_a_raising_check_reports_its_exception(self) -> None:
        def _boom() -> tuple[bool, str]:
            raise ValueError("upstream unreachable")

        out, failures = self._render([("boom", _boom)])

        assert failures == 1
        assert "upstream unreachable" in out
