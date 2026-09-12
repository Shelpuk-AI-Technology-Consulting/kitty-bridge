"""Throwaway TLS certificates for the bridge tests, generated loudly.

``.system_design/TEST_SUITE.md`` §8: **skips are failures in a gating job.** A
platform or interpreter skip is permitted; a *resource-availability* skip is
not, because a gating job that goes green after running nothing is the most
expensive kind of false confidence.

The bridge TLS tests need a certificate and a key, and generate them by spawning
``openssl``. That binary is a resource, so this module's whole contract is that
its absence or misbehaviour **fails** the run rather than quietly removing tests
from it (KBR-132). Every exit from :func:`generate_self_signed_cert` other than
success goes through :func:`pytest.fail`.

**A consequence worth stating plainly:** ``openssl`` is an environment
prerequisite of the Fast CI job, not something a test may probe for. Forbidding
the skip is the same statement as requiring the runner to provide the binary.

The sibling module ``tests/test_egress_https_proxy.py`` keeps its own generator:
it builds a CA and two signed leaves with different extensions, which is a
different function, and ``TEST_SUITE.md`` §8.2 names that module for relocation
under T-K6 (KBR-115).

**Import note.** Imported relatively -- ``from .tls_certs import ...`` -- which
is how ``tests/bridge/test_tool_use_auditor.py`` already reaches its sibling,
and which resolves through the package rather than through ``sys.path``. The
absolute spelling works today too, but only because ``tests/`` has no
``__init__.py`` while ``tests/bridge/`` does, so pytest's ``prepend`` import
mode puts ``tests/`` on ``sys.path``. That is a property of the import mode, not
of this package; ``tests/layers.py`` already rests on it once, and the relative
form avoids adding a second place that breaks if the mode ever changes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# Generous enough that a loaded CI runner never trips it, short enough that a
# genuinely wedged `openssl` is reported as a failure rather than eating the
# job's cap (60 minutes since KBR-164 widened it for the platform legs; the
# argument is unchanged by the number, which is why this reads "the cap" and
# names it only in passing). RSA-2048 keygen is milliseconds when it works.
_OPENSSL_TIMEOUT_SECONDS = 60

# Repeated in every failure message so a CI log says why the run did not simply
# skip. Whoever reads it should not have to find the design document first.
_WHY_NOT_A_SKIP = (
    "This fails rather than skipping: TEST_SUITE.md section 8 forbids a "
    "resource-availability skip inside a gating job, so openssl is a "
    "prerequisite of the environment, not something a test probes for (KBR-132)."
)


def generate_self_signed_cert(tmp_path: Path) -> tuple[Path, Path]:
    """Generate a throwaway self-signed certificate and key for a local server.

    Spawns ``openssl req -x509`` to write a PEM certificate and an unencrypted
    RSA key into ``tmp_path``. The certificate is valid for one day and carries
    ``CN=localhost``, which is all the bridge's TLS tests need — nothing here
    verifies a chain.

    Args:
        tmp_path: Directory receiving ``cert.pem`` and ``key.pem``. The caller's
            ``tmp_path`` fixture, so the files die with the test.

    Returns:
        The ``(certificate, key)`` paths, in that order.

    Raises:
        pytest.fail.Exception: When ``openssl`` is absent from ``PATH``, exits
            non-zero, or does not finish within
            :data:`_OPENSSL_TIMEOUT_SECONDS`. Never
            :func:`pytest.skip` — see this module's docstring.
    """
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"

    # `stdin` is closed so a prompt from an unexpected openssl build blocks on
    # nothing and surfaces as the timeout below rather than as a hung job.
    try:
        completed = subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(key_path),
                "-out",
                str(cert_path),
                "-days",
                "1",
                "-nodes",
                "-subj",
                "/CN=localhost",
            ],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=_OPENSSL_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        pytest.fail(f"openssl was not found on PATH, and this test needs it. {_WHY_NOT_A_SKIP}")
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"openssl did not finish within {_OPENSSL_TIMEOUT_SECONDS}s. {_WHY_NOT_A_SKIP}"
        )

    if completed.returncode != 0:
        pytest.fail(
            f"openssl req failed (exit {completed.returncode}):\n{completed.stderr}\n"
            f"{_WHY_NOT_A_SKIP}"
        )

    return cert_path, key_path
