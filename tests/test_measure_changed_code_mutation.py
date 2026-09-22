"""Unit tests for ``scripts/measure_changed_code_mutation.py`` (T-H4, KBR-92).

The script maps a git rev-range onto the §6.1 mutation-scope registry
(``tests/mutmut_scope.py``) and derives the mutmut fnmatch patterns for exactly
the functions the range touches — the changed-code subset Q11 (TEST_SUITE.md
§11) asks to time against the fast gate's budget.

Layer: L1 by path default. The pure mapping is tested against synthetic
fixtures (milliseconds, no subprocess); the runner contract is tested through
an injected fake so no test launches mutmut. The KBR-285 tests are anchored on
a pinned snapshot of the analyzer's verified output
(``tests/data/kbr285_diff_snapshot.json``, generated once by a throwaway
`.scratch/` script and hand-verified against the actual diff) — the oracle is
the real diff, not a free-form list that could pass for the wrong reason. The
real scoped ``mutmut run`` is the measurement itself and is never a test.
"""

from __future__ import annotations

import importlib.util
import json
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "measure_changed_code_mutation.py"
_FIXTURE = _REPO_ROOT / "tests" / "data" / "kbr285_diff_snapshot.json"


def _load_script() -> ModuleType:
    """Import the script by filesystem path, the KBR-88 aggregator precedent.

    ``scripts/`` is not a package, so the spec loader binds the module under a
    stable name in ``sys.modules`` exactly the way
    ``tests/test_aggregate_mutation_baseline.py`` loads its script.
    """
    spec = importlib.util.spec_from_file_location(
        "measure_changed_code_mutation", _SCRIPT
    )
    if spec is None or spec.loader is None:  # pragma: no cover - load guard
        raise ImportError(f"cannot load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_changed_code_mutation"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mcm() -> ModuleType:
    """The script module, loaded once for the file."""
    if not _SCRIPT.exists():
        pytest.fail(f"script not implemented yet: {_SCRIPT}")
    return _load_script()


@pytest.fixture(scope="session")
def kbr285_snapshot() -> dict[str, Any]:
    """The pinned KBR-285 diff snapshot (the test oracle, not derivable)."""
    with open(_FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# R1 — diff → module mapping
# --------------------------------------------------------------------------


def test_module_for_maps_src_paths_to_dotted_modules(mcm: ModuleType) -> None:
    """Source files map to dotted modules; non-source paths map to None."""
    assert mcm.module_for("src/kitty/bridge/server.py") == "kitty.bridge.server"
    assert mcm.module_for("src/kitty/providers/openai.py") == "kitty.providers.openai"
    assert mcm.module_for("src/kitty/providers/__init__.py") == "kitty.providers"
    assert mcm.module_for("README.md") is None
    assert mcm.module_for("docs/notes.md") is None
    assert mcm.module_for("src/kitty/bridge/server.py.bak") is None


# --------------------------------------------------------------------------
# R2 — diff hunks → touched defs
# --------------------------------------------------------------------------


def test_enclosing_defs_finds_method_bodies_and_skips_untouched_siblings(
    mcm: ModuleType,
) -> None:
    """A change inside one method's body touches that method, not its sibling."""
    source = (
        "class C:\n"
        "    def a(self):\n"  # line 2
        "        return 1\n"  # line 3 — touched
        "\n"
        "    def b(self):\n"  # line 5
        "        return 2\n"  # line 6
    )
    assert mcm.enclosing_defs(source, {3}) == {("C", "a")}


def test_enclosing_defs_ignores_module_level_constant_changes(
    mcm: ModuleType,
) -> None:
    """A module-level constant edit touches no def (nothing mutable moved)."""
    source = (
        "X = 1\n"  # line 1 — touched
        "def f():\n"
        "    return X\n"
    )
    assert mcm.enclosing_defs(source, {1}) == set()


def test_enclosing_defs_catches_a_newly_added_def(mcm: ModuleType) -> None:
    """A wholly new def overlaps its own span."""
    source = (
        "def f():\n"  # line 1
        "    return 1\n"
        "def g():\n"  # line 3 — touched (new)
        "    return 2\n"  # line 4 — touched
    )
    assert mcm.enclosing_defs(source, {3, 4}) == {(None, "g")}


def test_enclosing_defs_attributes_nested_def_change_to_outer_method(
    mcm: ModuleType,
) -> None:
    """A change inside a nested def belongs to the enclosing class method.

    mutmut's mangler enumerates top-level functions and class methods only —
    nested defs mutate as part of their enclosing function — so the span
    mapping must never invent a nested-def target of its own.
    """
    source = (
        "class C:\n"
        "    def outer(self):\n"  # line 2
        "        def inner():\n"  # line 3
        "            return 1\n"  # line 4 — touched
        "        return inner()\n"
    )
    assert mcm.enclosing_defs(source, {4}) == {("C", "outer")}


def test_enclosing_defs_includes_decorator_lines_in_span(mcm: ModuleType) -> None:
    """A decorator-line change touches the decorated function."""
    source = (
        "class C:\n"
        "    @property\n"  # line 2 — touched
        "    def f(self):\n"
        "        return 1\n"
    )
    assert mcm.enclosing_defs(source, {2}) == {("C", "f")}


# --------------------------------------------------------------------------
# R3 — registry intersection (KBR-285 fixture) + function-level patterns
# --------------------------------------------------------------------------


def test_diff_meta_matches_kbr285_snapshot(
    mcm: ModuleType, kbr285_snapshot: dict[str, Any]
) -> None:
    """The git plumbing reproduces the pinned snapshot exactly."""
    diff = mcm.diff_meta_from_git_range(kbr285_snapshot["rev_range"], str(_REPO_ROOT))
    expected_modules = {info["module"] for info in kbr285_snapshot["files"].values()}
    assert set(diff.touched_modules) == expected_modules
    expected_defs = {
        info["module"]: frozenset(
            (cls, fn) for cls, fn in info["touched_defs"]
        )
        for info in kbr285_snapshot["files"].values()
    }
    assert dict(diff.touched_defs) == expected_defs


def test_kbr285_intersection_yields_ten_covered_defs(
    mcm: ModuleType, kbr285_snapshot: dict[str, Any]
) -> None:
    """11 touched defs, 10 registry-covered: _stream_chat_completions is out."""
    from mutmut_scope import TARGET_GROUPS

    diff = mcm.diff_meta_from_git_range(kbr285_snapshot["rev_range"], str(_REPO_ROOT))
    covered = mcm.targets_covered(diff, TARGET_GROUPS)
    assert len(covered) == 10
    covered_fns = {(t.cls, t.function_or_method) for _, t in covered}
    assert ("BridgeServer", "_stream_chat_completions") not in covered_fns


def test_kbr285_compaction_and_pairing_targets_untouched(
    mcm: ModuleType, kbr285_snapshot: dict[str, Any]
) -> None:
    """No compaction_and_pairing row is covered — server.py's hunks miss them."""
    from mutmut_scope import TARGET_GROUPS

    diff = mcm.diff_meta_from_git_range(kbr285_snapshot["rev_range"], str(_REPO_ROOT))
    covered = mcm.targets_covered(diff, TARGET_GROUPS)
    assert {group for group, _ in covered}.isdisjoint({"compaction_and_pairing"})


def test_kbr285_patterns_match_expected_fnmatch_strings(
    mcm: ModuleType, kbr285_snapshot: dict[str, Any]
) -> None:
    """Each covered def yields its exact function-level mangled pattern.

    The patterns carry the FILE module (``kitty.bridge.messages.translator``)
    rather than the PACKAGE module (``kitty.bridge.messages``) — both
    spellings fnmatch the same mutant keys under mutmut's ``module.*``
    semantics (``*`` spans dots), but the file spelling is the honest
    "this is the file we touched" surface.
    """
    from mutmut_scope import TARGET_GROUPS

    diff = mcm.diff_meta_from_git_range(kbr285_snapshot["rev_range"], str(_REPO_ROOT))
    covered = mcm.targets_covered(diff, TARGET_GROUPS)
    patterns = mcm.patterns_for_covered(covered)
    expected = {
        "kitty.bridge.messages.translator.xǁMessagesTranslatorǁtranslate_response__mutmut_*",
        "kitty.bridge.messages.translator.xǁMessagesTranslatorǁtranslate_stream_chunk__mutmut_*",
        "kitty.bridge.responses.translator.xǁResponsesTranslatorǁtranslate_response__mutmut_*",
        "kitty.bridge.responses.translator.xǁResponsesTranslatorǁtranslate_stream_chunk__mutmut_*",
        "kitty.bridge.responses.translator.x__extract_text_parts__mutmut_*",
        "kitty.bridge.gemini.translator.xǁGeminiTranslatorǁtranslate_response__mutmut_*",
        "kitty.bridge.gemini.translator.xǁGeminiTranslatorǁtranslate_stream_chunk__mutmut_*",
        "kitty.bridge.gemini.translator.x__extract_text_parts__mutmut_*",
        "kitty.bridge.server.x__cc_chunk_carries_content__mutmut_*",
        "kitty.bridge.server.xǁBridgeServerǁ_is_empty_cc_response__mutmut_*",
    }
    assert set(patterns) == expected


def test_provider_hooks_cross_module_row_covers_any_provider(
    mcm: ModuleType,
) -> None:
    """A cross-module row covers its hook name in any registered module."""
    from mutmut_scope import TARGET_GROUPS

    diff = mcm.DiffMeta(
        rev_range="synthetic",
        touched_modules=frozenset({"kitty.providers.openai"}),
        touched_defs={
            "kitty.providers.openai": frozenset(
                {(None, "translate_to_upstream"), (None, "normalize_request")}
            )
        },
    )
    registered = frozenset({"kitty.providers.openai"})
    covered = mcm.targets_covered(diff, TARGET_GROUPS, registered)
    names = {(t.module, t.function_or_method) for _, t in covered}
    assert ("kitty.providers.openai", "translate_to_upstream") in names
    assert ("kitty.providers.openai", "normalize_request") in names


def test_provider_hooks_ignores_unregistered_modules(mcm: ModuleType) -> None:
    """A provider hook in a non-registered module yields no provider_hooks target.

    The def may still be covered by some other group — e.g.
    ``translate_to_upstream`` in ``kitty.bridge.engine`` matches engine's
    own whole-module row in ``translators_and_engine``. What must NOT
    happen is a cross-module ``provider_hooks`` row claiming coverage in a
    module outside ``only_mutate``.
    """
    from mutmut_scope import TARGET_GROUPS

    diff = mcm.DiffMeta(
        rev_range="synthetic",
        touched_modules=frozenset({"kitty.bridge.engine"}),
        touched_defs={"kitty.bridge.engine": frozenset({(None, "translate_to_upstream")})},
    )
    registered = frozenset({"kitty.providers.openai"})
    covered = mcm.targets_covered(diff, TARGET_GROUPS, registered)
    provider_hooks_hits = [t for g, t in covered if g == "provider_hooks"]
    assert provider_hooks_hits == []


# --------------------------------------------------------------------------
# R4 — runner contract, wall-clock cap, empty-intersection refusal
# --------------------------------------------------------------------------


def test_scoped_patterns_empty_intersection_raises(mcm: ModuleType) -> None:
    """Zero covered defs must refuse — an empty pattern list means the full run."""
    diff = mcm.DiffMeta(
        rev_range="synthetic",
        touched_modules=frozenset({"kitty.bridge.server"}),
        touched_defs={
            "kitty.bridge.server": frozenset(
                {("BridgeServer", "_stream_chat_completions")}
            )
        },
    )
    from mutmut_scope import TARGET_GROUPS

    with pytest.raises(mcm.EmptyIntersectionError):
        mcm.scoped_patterns(diff, TARGET_GROUPS, frozenset())


def test_main_refuses_run_on_empty_intersection(
    mcm: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--run` with an out-of-scope-only diff exits non-zero without mutmut.

    The repo-loading seams (``_load_registry``, ``_load_only_mutate_modules``)
    are patched so the synthetic ``--repo`` does not have to host the real
    test artifacts.
    """
    from mutmut_scope import TARGET_GROUPS

    synthetic = mcm.DiffMeta(
        rev_range="synthetic",
        touched_modules=frozenset({"kitty.bridge.server"}),
        touched_defs={
            "kitty.bridge.server": frozenset(
                {("BridgeServer", "_stream_chat_completions")}
            )
        },
    )
    monkeypatch.setattr(mcm, "diff_meta_from_git_range", lambda *a, **k: synthetic)
    monkeypatch.setattr(mcm, "_load_registry", lambda repo: TARGET_GROUPS)
    monkeypatch.setattr(mcm, "_load_only_mutate_modules", lambda repo: frozenset())
    invoked: list[list[str]] = []
    monkeypatch.setattr(
        mcm, "run_mutmut", lambda command, **k: invoked.append(command) or {}
    )
    # main() returns an int; the `sys.exit(main())` guard converts at the
    # process boundary, so the refusal is asserted on the return value.
    assert mcm.main(
        [
            "synthetic",
            "--run",
            "--repo",
            str(tmp_path),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    ) != 0
    assert invoked == []


class _FakeProc:
    """Minimal proc handle satisfying the runner contract.

    ``reap_delay`` simulates the kernel not having reaped the child yet —
    after a SIGKILL, ``poll()`` returns ``None`` for that many real-time
    seconds before the fake surfaces the exit code. With ``reap_delay=0``
    the fake mimics the OLD runner behaviour (immediate read after
    SIGKILL); with ``reap_delay > 0`` it forces the runner's bounded
    reap loop to do real work before reading ``returncode``, so the
    over-budget test catches a regression to the immediate-read path.
    """

    def __init__(
        self,
        returncode: int | None = 0,
        hang: bool = False,
        *,
        dies_on_sigterm: bool = False,
        reap_delay: float = 0.0,
    ) -> None:
        self.returncode = returncode
        self._hang = hang
        self._dies_on_sigterm = dies_on_sigterm
        self._reap_delay = reap_delay
        self._sigkill_at: float | None = None
        self.signals: list[int] = []

    def poll(self) -> int | None:
        if not self._hang:
            return self.returncode
        if self._sigkill_at is not None:
            elapsed = time.monotonic() - self._sigkill_at
            if elapsed >= self._reap_delay:
                self.returncode = -signal.SIGKILL
                self._hang = False
                return self.returncode
        return None

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        if sig == signal.SIGTERM and self._dies_on_sigterm:
            self._hang = False
            self.returncode = -sig
        elif sig == signal.SIGKILL:
            self._sigkill_at = time.monotonic()


def test_runner_builds_the_positional_pattern_command_and_records_load(
    mcm: ModuleType, tmp_path: Path
) -> None:
    """Patterns ride positionally after `mutmut run`; load and clock recorded."""
    seen: list[list[str]] = []

    def fake_runner(command: list[str], **kwargs: Any) -> _FakeProc:
        seen.append(command)
        return _FakeProc(returncode=0)

    patterns = [
        "kitty.bridge.messages.xǁMessagesTranslatorǁtranslate_response__mutmut_*"
    ]
    summary = mcm.run_mutmut(
        mcm.build_command(patterns),
        repo_root=str(tmp_path),
        log_path=tmp_path / "logs" / "run.log",
        runner=fake_runner,
    )
    assert seen == [["mutmut", "run", *patterns]]
    assert summary["status"] == "ok"
    assert summary["exit_code"] == 0
    assert isinstance(summary["wall_clock_seconds"], float)
    assert summary["wall_clock_seconds"] >= 0.0
    assert isinstance(summary["load_avg_1m"], float)
    assert isinstance(summary["cpu_count"], int)


def test_runner_wall_clock_cap_sends_sigterm_and_classifies_over_budget(
    mcm: ModuleType, tmp_path: Path
) -> None:
    """A run outliving the budget is SIGTERMed, then SIGKILLed, not hung.

    ``reap_delay`` is positive so the fake mimics a real kernel that has
    not yet reaped the SIGKILL'd child — the bounded reap loop in
    ``run_mutmut`` must do real work to see ``returncode``. With the OLD
    immediate-read code this assertion would fail (``exit_code`` would be
    ``None``), so the test pins the reap-wait behaviour.
    """
    proc = _FakeProc(returncode=None, hang=True, reap_delay=0.2)

    def fake_runner(command: list[str], **kwargs: Any) -> _FakeProc:
        return proc

    summary = mcm.run_mutmut(
        mcm.build_command(["kitty.bridge.server.x__f__mutmut_*"]),
        repo_root=str(tmp_path),
        log_path=tmp_path / "logs" / "run.log",
        runner=fake_runner,
        budget_seconds=0.05,
        grace_seconds=0.5,
    )
    assert proc.signals[:1] == [signal.SIGTERM]
    assert signal.SIGKILL in proc.signals
    assert summary["status"] == "over_budget"
    # After the bounded reap wait the kernel has (per the fake) delivered
    # the signal, so the summary carries the real -SIGKILL marker, not null.
    assert summary["exit_code"] == -signal.SIGKILL
    assert summary["wall_clock_seconds"] >= 0.05


def test_runner_sigterm_within_grace_skips_sigkill(
    mcm: ModuleType, tmp_path: Path
) -> None:
    """A process that dies on SIGTERM during the grace window stays at SIGTERM.

    The script must NOT escalate to SIGKILL when SIGTERM already reaped the
    process — over-eager escalation would mask the real exit code (-SIGTERM)
    in the summary.
    """
    proc = _FakeProc(returncode=None, hang=True, dies_on_sigterm=True)

    def fake_runner(command: list[str], **kwargs: Any) -> _FakeProc:
        return proc

    summary = mcm.run_mutmut(
        mcm.build_command(["kitty.bridge.server.x__f__mutmut_*"]),
        repo_root=str(tmp_path),
        log_path=tmp_path / "logs" / "run.log",
        runner=fake_runner,
        budget_seconds=0.05,
        grace_seconds=0.2,
    )
    assert proc.signals == [signal.SIGTERM]
    assert signal.SIGKILL not in proc.signals
    assert summary["status"] == "over_budget"
    assert summary["exit_code"] == -signal.SIGTERM


def test_runner_exception_propagates_without_nameerror(
    mcm: ModuleType, tmp_path: Path
) -> None:
    """A runner that raises must surface its exception, not a `NameError`.

    Without defensive initialisation, the `finally` block's
    ``getattr(proc, "close_log", None)`` would dereference an unbound
    `proc` and raise `NameError`, replacing the runner's real exception
    with a confusing secondary failure. The runner's exception must
    propagate verbatim.
    """
    class _RunnerError(RuntimeError):
        pass

    def boom(command: list[str], **kwargs: Any) -> _FakeProc:
        raise _RunnerError("runner could not spawn")

    with pytest.raises(_RunnerError, match="runner could not spawn"):
        mcm.run_mutmut(
            mcm.build_command(["kitty.bridge.server.x__f__mutmut_*"]),
            repo_root=str(tmp_path),
            log_path=tmp_path / "logs" / "run.log",
            runner=boom,
        )


def test_runner_handles_platforms_without_os_getloadavg(
    mcm: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows lacks ``os.getloadavg``; the runner must report ``None`` and
    not raise.

    Pins the platform-portability contract: the load keys remain present
    in the JSON summary with ``None`` values so downstream consumers see a
    uniform schema.
    """
    # Simulate Windows: ``os.getloadavg`` raises ``AttributeError``.
    monkeypatch.delattr("os.getloadavg", raising=False)
    proc = _FakeProc(returncode=0)

    def fake_runner(command: list[str], **kwargs: Any) -> _FakeProc:
        return proc

    summary = mcm.run_mutmut(
        mcm.build_command(["kitty.bridge.server.x__f__mutmut_*"]),
        repo_root=str(tmp_path),
        log_path=tmp_path / "logs" / "run.log",
        runner=fake_runner,
    )
    assert summary["status"] == "ok"
    assert summary["load_avg_1m"] is None
    assert summary["load_avg_5m"] is None
    assert summary["load_avg_15m"] is None
    assert summary["cpu_count"] is not None  # cpu_count exists on Windows
