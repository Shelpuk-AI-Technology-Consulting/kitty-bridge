"""Tests for KBR-132: the certificate helper fails loudly, it never skips.

``.system_design/TEST_SUITE.md`` §8 forbids a resource-availability skip inside
a gating job. ``tls_certs`` alongside this module is where that rule is kept for the bridge
TLS tests, and this module is the check on it.

🔴 **The catch shape below is the point of these tests, not ceremony.**
``pytest.fail.Exception`` (``Failed``) and ``pytest.skip.Exception``
(``Skipped``) are *siblings*: both derive from ``OutcomeException``, which
derives from ``BaseException``. So ``pytest.raises(Failed)`` does **not** catch
a ``Skipped`` — if the defect were reintroduced, the ``Skipped`` would escape
the ``raises`` block, propagate out of the test body, and pytest would report
these tests as **skipped**, which passes the gate. The detector would have
reproduced the very defect it exists to detect. Every case therefore catches
``Skipped`` *first* and converts it into an assertion failure.

The two exception attributes are the most public form pytest offers; the classes
themselves live in the private ``_pytest.outcomes``. They are undocumented but
long-stable and widely used, which is a deliberate trade against importing a
private module, and is recorded here because `pyproject.toml` pins
``pytest>=8.0`` with no upper bound.
"""

from __future__ import annotations

import ssl
import subprocess
from pathlib import Path

import pytest

from .tls_certs import generate_self_signed_cert

# Distinctive enough that finding it in a failure message proves the helper
# passed openssl's own diagnostics through rather than inventing a summary.
_STDERR_SENTINEL = "problems making Certificate Request"


def _failure_message_from(tmp_path: Path) -> str:
    """Call the helper and return the message of the failure it must raise.

    Args:
        tmp_path: Directory handed to :func:`generate_self_signed_cert`.

    Returns:
        The text of the ``Failed`` exception the helper raised.

    Raises:
        AssertionError: When the helper skips, or returns successfully. Both are
            the KBR-132 defect: a skip removes the test from a gating job, and a
            success means the failure condition was not reproduced at all.
    """
    # `Skipped` is caught FIRST and re-raised as an ordinary assertion failure.
    # Letting it propagate would mark this test skipped -- green, silent, and
    # the exact outcome these tests exist to make impossible.
    try:
        generate_self_signed_cert(tmp_path)
    except pytest.skip.Exception as skipped:
        raise AssertionError(
            f"The helper skipped instead of failing ({skipped}). A skip inside a "
            "gating job is what KBR-132 fixed; TEST_SUITE.md section 8 forbids it."
        ) from skipped
    except pytest.fail.Exception as failed:
        return str(failed)

    raise AssertionError(
        "The helper returned successfully where it had to fail. The failure "
        "condition was not reproduced, so this test proves nothing."
    )


@pytest.fixture()
def no_openssl_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``PATH`` at an empty directory for the duration of one test.

    Args:
        tmp_path: The test's temporary directory, which also holds the empty
            directory ``PATH`` is redirected to.
        monkeypatch: Fixture restoring the real ``PATH`` afterwards.

    The binary is made genuinely absent rather than :func:`subprocess.run` being
    stubbed, because absence is the CI condition being guarded; a stub would
    prove only that the ``except`` branch is reachable.
    """
    empty = tmp_path / "no-binaries"
    empty.mkdir()

    monkeypatch.setenv("PATH", str(empty))


class TestMissingOpensslFailsTheRun:
    """The resource is absent: the run goes red, never green-by-omission."""

    @pytest.mark.usefixtures("no_openssl_on_path")
    def test_missing_openssl_fails_it_does_not_skip(self, tmp_path: Path) -> None:
        """PATH carrying no openssl produces a failure, not a skip (AC1).

        This is the falsification case for the whole change: restoring the
        ``pytest.skip`` in the helper turns this test red.

        The assertion names the *missing-binary* branch rather than just
        ``"openssl"``: all three of the helper's failure messages contain that
        word, so the looser assertion would pass on any failure at all and prove
        only that something went wrong somewhere.
        """
        message = _failure_message_from(tmp_path)

        assert "not found on PATH" in message

    @pytest.mark.usefixtures("no_openssl_on_path")
    def test_the_failure_message_says_why_it_did_not_skip(self, tmp_path: Path) -> None:
        """The message names the rule, so a CI log is self-explanatory (AC1b).

        Asserted rather than left to prose: a reader of a red CI job needs to
        know that reinstating the skip is the one fix that is not available.
        """
        message = _failure_message_from(tmp_path)

        assert "TEST_SUITE.md" in message


class TestOpensslMisbehaviourFailsTheRun:
    """The resource is present but unusable: still red, and still legible."""

    def test_openssl_error_fails_with_its_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-zero exit fails and carries openssl's own diagnostics (AC2).

        Stubbed rather than shimmed onto ``PATH``: a fake executable would make
        this case depend on the platform's script conventions, and what is under
        test is the helper's branch, not the shell's.
        """

        def _failing_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(args=["openssl"], returncode=1, stdout="", stderr=_STDERR_SENTINEL)

        monkeypatch.setattr(subprocess, "run", _failing_run)

        message = _failure_message_from(tmp_path)

        assert _STDERR_SENTINEL in message
        assert "TEST_SUITE.md" in message

    def test_openssl_timeout_fails_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wedged openssl fails the test rather than hanging the job (AC2a).

        The gate runs this on every pull request across four Python versions;
        the job's 30-minute ceiling is a backstop for the whole run and says
        nothing about which call stopped.

        🔴 **The recorded kwargs are what make this test mean anything.** A
        helper that catches ``TimeoutExpired`` but never passes ``timeout=`` to
        :func:`subprocess.run` satisfies "converts the exception to a failure"
        perfectly and still hangs forever on a real stuck process. That is one
        of the four harnesses the plan's §1.4 rule was written about — a guard
        proving a function was *called* when the enforcement is the argument it
        was called with. So the bound is asserted as **armed**, not handled.
        """
        seen: dict[str, object] = {}

        def _hanging_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
            seen.update(kwargs)
            raise subprocess.TimeoutExpired(cmd="openssl", timeout=60)

        monkeypatch.setattr(subprocess, "run", _hanging_run)

        message = _failure_message_from(tmp_path)

        assert seen.get("timeout"), "openssl is spawned with no timeout; nothing bounds a wedged process"
        assert seen.get("stdin") == subprocess.DEVNULL, (
            "openssl inherits stdin; an unexpected prompt would block on the terminal, "
            "not on the timeout above"
        )
        assert "did not finish" in message
        assert "TEST_SUITE.md" in message


class TestGeneratedMaterialIsUsable:
    """The helper still does its job -- failing loudly is not the whole contract."""

    def test_generated_cert_and_key_load_into_an_ssl_context(self, tmp_path: Path) -> None:
        """The pair is accepted by a real server-side SSL context (AC3).

        Needs a real ``openssl``, and fails loudly without one by the rule
        above. That is the intended behaviour of this suite, not an oversight.
        """
        cert, key = generate_self_signed_cert(tmp_path)

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(cert), str(key))
