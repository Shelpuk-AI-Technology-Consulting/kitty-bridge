"""Pin the kitty CLI surface that the automated PR reviewer drives.

🔴 **Why this file exists, stated plainly, because nothing else in the repository
can catch what it catches.**

``.github/workflows/claude-code-review.yml`` runs Claude Code through Kitty
Bridge, and it installs that bridge with ``pip install --upgrade kitty-bridge``
-- the **released** package from PyPI, not this checkout. That is deliberate:
reviewing with the pull request's own bridge would let a broken change break its
own review and report the failure as a provider fault.

The consequence is that **the reviewer never exercises the code under review**.
So a pull request that changes the CLI surface the workflow depends on passes its
own review under the *old* bridge, merges, and breaks the reviewer for every
unrelated pull request at the next release -- at which point the failure notice
points at the three ``KITTY_*`` organisation settings, and all three are correct.

This module is the guard for that, and it works against the **working tree**
rather than against the installed package, which is the whole point. Each case
below pins one thing ``configure_kitty.py`` or the workflow assumes. If you are
changing something here, the workflow is a caller: change both together.

Hermetic by construction -- a temporary config directory, no network, no
provider, no spend. See ``.github/review/rules/ci.md`` for the invariants these
cases serve.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# L2: the subject of this file is an artifact outside `src/kitty` Python code,
# or a structural scan of source text -- two things edited separately that must
# agree. It gates pull requests exactly as before, in the `l1 or l2` job; the
# marker records which half of that expression it answers to, and keeps a
# source-text scan out of the L1 set that mutation testing will judge.
pytestmark = pytest.mark.l2

ROOT = Path(__file__).resolve().parent.parent
REVIEW_SCRIPTS = ROOT / ".github" / "review" / "scripts"
CONFIGURE_KITTY = REVIEW_SCRIPTS / "configure_kitty.py"
WORKFLOW = ROOT / ".github" / "workflows" / "claude-code-review.yml"

#: A gateway record in the shape kitty's own store writes. Deliberately not built
#: by importing :class:`~kitty.egress_store.EgressRecord` and serialising it: the
#: claim under test is that a document written by something *outside* this
#: package still loads, and generating the fixture from the code that reads it
#: would make the case pass by construction.
GATEWAY_DOCUMENT = {
    "version": 1,
    "egress": {
        "proxy_url": "http://proxy.example.invalid:12323",
        # No username, and that is not laziness. kitty refuses a record carrying
        # a username with no resolvable password -- "egress proxy username and
        # password must be provided together" -- which would make this fixture
        # exercise the refusal path while claiming to exercise the success one.
        # A password would mean an `auth_ref` and a matching credential store
        # entry, i.e. a second document, to prove something this case is not
        # about.
        "username": None,
        "auth_ref": None,
    },
}


def _kitty_env(config_dir: Path) -> dict[str, str]:
    """Return an environment that points kitty at an isolated config directory.

    kitty resolves its config directory through ``platformdirs``, which reads
    ``XDG_CONFIG_HOME`` on Linux and ``LOCALAPPDATA`` on Windows. Both are set so
    a developer's real ``~/.config/kitty`` is never touched by these cases —
    destroying somebody's profiles from a test run is the one failure this file
    must not have.

    ``KITTY_EGRESS_PROXY`` is cleared for the same reason the workflow binds it
    empty: it outranks the stored gateway, so a value in the developer's own
    environment would decide the outcome of every case below.

    🔴 **``PYTHONIOENCODING`` is not cosmetic, and leaving it out made a case
    here pass for the wrong reason.** kitty prints an ``ℹ`` when no gateway is
    configured. On a Windows console the child's default encoding is cp1252,
    which cannot encode it, so the process died with a ``UnicodeEncodeError``
    and exited 1 -- and `test_no_gateway_is_a_non_zero_exit` read that 1 as the
    refusal it was asserting. Measured by mutation: changing
    ``run_egress_show``'s ``return 1`` to ``return 0`` left every case green.
    Forcing UTF-8 makes the exit status the one the code chose.

    Args:
        config_dir: The directory to isolate kitty's configuration into.

    Returns:
        A complete environment mapping for a child process.
    """

    env = dict(os.environ)
    env["XDG_CONFIG_HOME"] = str(config_dir)
    env["LOCALAPPDATA"] = str(config_dir)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KITTY_EGRESS_PROXY", None)

    # 🔴 **The working tree first on the path, and this module is worthless
    # without it.** Its entire purpose is to catch a change to the CLI surface
    # *before* it is released -- so it must import THIS checkout, not whatever
    # `kitty` an editable install happens to resolve to. Measured: run from a git
    # worktree, `python -m kitty` resolved to the main checkout's `src/`, and a
    # mutation to `run_egress_show`'s refusal survived every case here because
    # the mutated file was never the one executed. A guard that silently tests a
    # different tree is worse than no guard.
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT / "src") + (os.pathsep + existing if existing else "")
    )
    return env


def _run_kitty(env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    """Invoke the working tree's kitty CLI as a child process.

    ``python -m kitty`` rather than the ``kitty`` console script, because the
    console script may resolve to an installed release rather than to this
    checkout — which is precisely the confusion this module exists to rule out.

    Args:
        env: The child environment, from :func:`_kitty_env`.
        *args: Command-line arguments after the module name.

    Returns:
        The finished process, with output captured.
    """

    return subprocess.run(
        [sys.executable, "-m", "kitty", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture()
def tmp_config_dir(tmp_path: Path) -> Path:
    """Alias of :func:`config_dir` for cases that only need isolation.

    Same directory, different name: a case that never inspects the directory
    reads better asking for "a temporary config dir" than for "the config dir",
    and the distinction stops the two uses drifting apart.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The directory kitty will resolve to under :func:`_kitty_env`.
    """

    target = tmp_path / "kittyconfig-alias"
    target.mkdir()
    return target


@pytest.fixture()
def config_dir(tmp_path: Path) -> Path:
    """Return an empty, isolated kitty config directory.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The directory kitty will resolve to under :func:`_kitty_env`.
    """

    target = tmp_path / "kittyconfig"
    target.mkdir()
    return target


def _resolved_config_dir(env: dict[str, str]) -> Path:
    """Ask a child process where kitty will actually look for its config.

    Derived rather than assumed. ``platformdirs`` appends different suffixes per
    platform — ``<XDG_CONFIG_HOME>/kitty`` on Linux, ``<LOCALAPPDATA>/kitty/kitty``
    on Windows — and hard-coding either would make every case below pass or fail
    for a reason that has nothing to do with the workflow.

    Args:
        env: The child environment, from :func:`_kitty_env`.

    Returns:
        The directory kitty reads ``egress.json`` from.
    """

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from platformdirs import user_config_dir; print(user_config_dir('kitty'))",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return Path(proc.stdout.strip())


class TestTheFlagsTheWrapperPasses:
    """The launcher `configure_kitty.py` writes is `kitty --no-validate --debug-file <log> claude "$@"`.

    Every token in that line is a promise about this CLI. A renamed or removed
    flag makes the wrapper fail at startup, which reaches the workflow as an
    empty execution record and is classified as `fatal -- no execution record`:
    a verdict that points at the provider and names nothing about the flag.
    """

    def test_the_wrapper_line_is_the_one_this_test_pins(self):
        """🔴 The control. Without it this class pins flags nobody passes.

        If the wrapper is rewritten to use different flags, every case below
        keeps passing while testing something the workflow no longer does. This
        reads the actual generated line and fails when it moves.
        """

        source = CONFIGURE_KITTY.read_text(encoding="utf-8")
        for token in ("--no-validate", "--debug-file", 'claude "$@"'):
            assert token in source, (
                f"the generated launcher no longer passes {token!r}; this module "
                "pins a command line that is no longer the one the reviewer runs"
            )

    @pytest.mark.parametrize("flag", ["--no-validate", "--debug-file", "--version"])
    def test_the_flag_is_still_accepted(self, flag: str, config_dir: Path):
        """Each flag the wrapper or the workflow passes is still recognised.

        Asserted through `--help`, which lists every option the parser defines,
        rather than by invoking the flag — invoking `--no-validate` would launch
        an agent.
        """

        proc = _run_kitty(_kitty_env(config_dir), "--help")
        assert proc.returncode == 0, proc.stderr
        assert flag in proc.stdout, (
            f"`kitty --help` no longer lists {flag!r}. The review workflow's "
            "launcher passes it, and an unrecognised flag makes kitty exit "
            "before Claude Code starts"
        )

    def test_the_claude_target_is_still_routable(self, tmp_config_dir: Path):
        """`claude` must remain a launcher target the router recognises.

        The wrapper's whole purpose is `kitty ... claude "$@"`. A rename would
        make the router treat `claude` as an unknown profile name and fail with a
        message about profiles — which is the least helpful place to look.

        Both halves, because they can move independently: the adapter's own
        `name`, and the key `main.py` registers it under, which is what the
        router actually matches the word `claude` against.
        """

        # 🔴 A child process under the pinned environment, NOT an in-process
        # import. `import kitty...` here resolves through the interpreter's own
        # `sys.path`, which — run from a git worktree against a shared editable
        # install — is a *different checkout*. That is precisely the wrong-tree
        # failure this module's `_kitty_env` exists to prevent, and this one line
        # escaped it: the module would have gone on claiming to test the working
        # tree while asking a different one.
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "from kitty.launchers.claude import ClaudeAdapter;"
                "print(ClaudeAdapter().name)",
            ],
            cwd=ROOT,
            env=_kitty_env(tmp_config_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert probe.returncode == 0, probe.stderr
        assert probe.stdout.strip() == "claude"

        registry = (ROOT / "src" / "kitty" / "cli" / "main.py").read_text(
            encoding="utf-8"
        )
        assert '"claude": ClaudeAdapter()' in registry, (
            "the router no longer registers `claude` as a launcher target, so "
            "the review workflow's launcher would route it as a profile name"
        )

    def test_version_reports_something_the_run_summary_can_show(self, config_dir: Path):
        """`kitty --version` is what the workflow stamps into the run summary.

        That stamp is the only thing separating "a bad release broke every
        review" from "this change broke its own review", so it has to keep
        printing a version rather than, say, opening the TUI.
        """

        proc = _run_kitty(_kitty_env(config_dir), "--version")
        assert proc.returncode == 0, proc.stderr
        assert "kitty" in proc.stdout.lower()
        assert any(char.isdigit() for char in proc.stdout), proc.stdout


class TestEgressShowExitCode:
    """🔴 `kitty egress show`'s exit code is the reviewer's containment gate.

    The workflow's `Verify kitty resolved the egress gateway` step reads nothing
    but that exit status — both streams are discarded, because they carry the
    proxy address, the username and the credential reference, and GitHub masks a
    secret's whole value rather than the JSON fields inside it.

    So this exit code is load-bearing in a way no other exit code in the package
    is: **if it ever returned 0 with no gateway resolved, the review would run
    from the runner's own IP and the check would go green.** That is the outcome
    the whole structure exists to prevent, and this is the only test that can
    catch a change to it.
    """

    def test_no_gateway_is_a_non_zero_exit(self, config_dir: Path):
        """The refusal. An empty config directory must not resolve a gateway."""

        env = _kitty_env(config_dir)
        proc = _run_kitty(env, "egress", "show")
        assert proc.returncode != 0, (
            "`kitty egress show` exited 0 with no gateway configured. The review "
            "workflow reads only this status, so this would let a review run "
            "unproxied and report green"
        )

    def test_a_configured_gateway_is_a_zero_exit(self, config_dir: Path):
        """The other direction, and it is not optional.

        A command that always exited non-zero would satisfy the case above while
        making every review fail as `fatal`. Both directions, or the gate is
        either useless or a permanent outage.
        """

        env = _kitty_env(config_dir)
        resolved = _resolved_config_dir(env)
        resolved.mkdir(parents=True, exist_ok=True)
        (resolved / "egress.json").write_text(
            json.dumps(GATEWAY_DOCUMENT), encoding="utf-8"
        )

        proc = _run_kitty(env, "egress", "show")
        assert proc.returncode == 0, (
            "`kitty egress show` did not exit 0 for a gateway written in the "
            f"store's own documented shape. stdout={proc.stdout!r} "
            f"stderr={proc.stderr!r}"
        )

    def test_a_disabling_document_does_not_resolve(self, config_dir: Path):
        """The shapes `configure_kitty.py` refuses, checked from kitty's side too.

        `configure_kitty.py` rejects a document with no envelope, an `egress` of
        `null`, or an empty `proxy_url` — before kitty ever sees it. This is the
        same claim from the other end: were the static check ever deleted as
        "redundant", these documents must still not resolve a gateway.

        ⚠️ An empty `proxy_url` is deliberately **absent** from this list, and
        that absence is the reason both checks exist. It *loads*: kitty reports
        healthy and exits 0, while the HTTP client ignores `proxy=""` and
        connects directly. Neither check subsumes the other.
        """

        env = _kitty_env(config_dir)
        resolved = _resolved_config_dir(env)
        resolved.mkdir(parents=True, exist_ok=True)

        for name, document in (
            ("no envelope", {"proxy_url": "http://proxy.example.invalid:12323"}),
            ("null gateway", {"version": 1, "egress": None}),
            ("unknown version", {"version": 999, "egress": GATEWAY_DOCUMENT["egress"]}),
        ):
            (resolved / "egress.json").write_text(
                json.dumps(document), encoding="utf-8"
            )
            proc = _run_kitty(env, "egress", "show")
            assert proc.returncode != 0, (
                f"a {name} document resolved a gateway, so a review would run "
                "unproxied with the check green"
            )


class TestTheConfigContract:
    """The three documents `configure_kitty.py` writes are the three kitty reads.

    It writes them into `--config-dir`, defaulting to `~/.config/kitty`, and
    kitty resolves its own directory through `platformdirs`. On the Linux runner
    those are the same path. If they ever diverge, the workflow writes three
    files nobody reads and the review runs against whatever the runner already
    had — with no error at either end.
    """

    FILES = ("profiles.json", "credentials.json", "egress.json")

    def test_the_writer_and_the_readers_name_the_same_files(self):
        """Both halves, by name.

        A rename on either side is silent: the writer keeps writing, the reader
        keeps finding nothing, and `kitty egress show` reports no gateway — which
        the workflow reports as a settings problem.
        """

        written = CONFIGURE_KITTY.read_text(encoding="utf-8")
        for name in self.FILES:
            assert name in written, f"configure_kitty.py no longer writes {name}"

        read_by = "\n".join(
            (ROOT / "src" / "kitty" / relative).read_text(encoding="utf-8")
            for relative in (
                Path("profiles") / "store.py",
                # `file_backend.py`, not `store.py`: the store is the interface
                # and the file backend is what names the file. Discovered by
                # this case failing, which is the shape a name-based pin is for.
                Path("credentials") / "file_backend.py",
                Path("egress_store.py"),
            )
        )
        for name in self.FILES:
            assert name in read_by, (
                f"no store reads {name}, which the review workflow writes"
            )

    @pytest.mark.skipif(
        os.name != "posix",
        reason="the runner is Linux; the XDG path agreement is a POSIX claim",
    )
    def test_the_default_write_path_is_where_kitty_looks(self, config_dir: Path):
        """`~/.config/kitty` on both sides, on the platform the workflow runs on.

        Skipped off POSIX rather than asserted loosely: `platformdirs` uses
        `%LOCALAPPDATA%` on Windows and `configure_kitty.py` hard-codes the XDG
        path, so the two genuinely differ there — and the workflow never runs
        there. A test that papered over that would be asserting something false.
        """

        env = _kitty_env(config_dir)
        assert _resolved_config_dir(env) == config_dir / "kitty"

        written = CONFIGURE_KITTY.read_text(encoding="utf-8")
        assert '".config" / "kitty"' in written, (
            "configure_kitty.py no longer defaults to the XDG kitty directory"
        )


class TestTheWorkflowStillDrivesThisSurface:
    """The other end of every pin above: the workflow still calls what is pinned.

    🔴 **Without this the whole module can go stale silently.** If the workflow
    stopped running `kitty egress show`, every case here would keep passing while
    guarding a command nothing invokes — a test suite that has quietly become
    documentation.
    """

    def test_the_workflow_invokes_the_commands_this_module_pins(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        for invocation in ("egress show", "--version"):
            assert invocation in text, (
                f"the review workflow no longer runs {invocation!r}; this module "
                "is pinning a surface nobody drives"
            )

    def test_the_workflow_still_runs_the_script_that_writes_the_wrapper(self):
        """🔴 `configure_kitty.py` is the source of every wrapper-flag pin here.

        `TestTheFlagsTheWrapperPasses` reads the launcher out of that script. If
        the workflow stopped running it — writing the launcher inline, or
        dropping the step — every one of those cases would keep passing against a
        file nothing executes. That is the same staleness this class exists to
        prevent, one level down, and the first version of this class missed it.
        """

        text = WORKFLOW.read_text(encoding="utf-8")
        assert "configure_kitty.py" in text, (
            "the review workflow no longer runs configure_kitty.py, so the "
            "wrapper this module pins is not the one the reviewer launches"
        )

    def test_the_workflow_still_installs_the_released_bridge(self):
        """The premise of this whole module, asserted rather than assumed.

        Every case here exists because the reviewer runs the *released* package
        and therefore cannot catch its own breakage. If the workflow ever
        installed this checkout instead, the reviewer would catch these itself
        and this module's justification would change — so the premise is pinned
        where it is relied on.
        """

        text = WORKFLOW.read_text(encoding="utf-8")
        assert "pip install --upgrade" in text
        assert "kitty-bridge" in text
        assert "pip install -e" not in text, (
            "the review workflow installs this checkout rather than the released "
            "package. That changes the premise this module is written on — see "
            "its docstring — and `.github/review/rules/ci.md` with it"
        )
