"""Dependency contract: what ``keyring`` promises ``KeyringBackend``.

``.system_design/TEST_SUITE.md`` §6.2.4 · plan task **T-G12**
([KBR-87](https://shelpuk.atlassian.net/browse/KBR-87)).

``pyproject.toml`` declares ``keyring>=23.0`` (measured 25.7.0 on the
environment this contract landed on), and :class:`kitty.credentials.keyring_backend.
KeyringBackend` hands it every credential read and write blind: nothing in kitty
observes which backend keyring resolved. Backend resolution varies by platform **by
design** — macOS Keychain, Windows Credential Manager, Linux Secret Service, and a
``fail.Keyring`` fallback when nothing else classifies — so §6.2.4's "where no stable
neighbour exists" rule applies: pin the resolution **mechanics**, not the live native
service, and record the choice (SYSTEM_DESIGN.md §11.3).

Five facts are load-bearing, and each test names the keyring change that would turn
it red — the falsification channel the implementation plan's §1.4 harness rule asks
for:

1. **The module-level API delegates to the resolved backend.**
   ``keyring.set_password``/``get_password``/``delete_password`` all operate on
   ``keyring.get_keyring()``. This is the headline claim: because the delegation
   holds, pinning ``get_keyring()`` pins where credentials go. A release that cached
   the backend at import time, or that swapped the delegate for the public calls,
   fails the round-trip against a backend installed *after* import.
2. **``PYTHON_KEYRING_BACKEND`` selects the backend.** ``keyring.core.load_env()``
   (a public module function) returns an instance of the named class. The rename of
   the variable, or a change in the import-path grammar, turns this red. The private
   ``_detect_backend`` consults this first — recorded here as context, deliberately
   not pinned: kitty's code reads nothing of it.
3. **Resolution always lands on a ``keyring.backends.*`` class** — on every
   platform, including headless Linux where it is the ``fail.Keyring`` fallback. A
   release relocating the backends namespace fails here.
4. **Per-platform native class where the native service is reachable.** macOS
   Keychain; Windows Credential Manager. Both arms guard on actual availability and
   skip with a stated reason otherwise — the macOS arm is expected to skip on the
   macOS CI leg, because the bare ``keyring>=23.0`` dependency does not carry pyobjc;
   a permanently-skipping arm is not coverage, and the skip reason says so. The
   Windows arm genuinely asserts (pywin32-ctypes is a base dependency). On Linux the
   ``fail.Keyring`` fallback is asserted **only behind a SecretService-unavailable
   guard**: on a D-Bus-equipped developer box the chainer classifies, and an
   unconditional fallback assertion would be red for environmental, not contract,
   reasons — the mirrored form of the trap the ipaddress contract's docstring
   records.
5. **``keyring.errors.PasswordDeleteError`` subclasses ``keyring.errors.KeyringError``**
   — the exception family ``KeyringBackend.delete`` suppresses. A release renaming
   either class fails here with a message naming the surface, rather than as an
   ``AttributeError`` in a credential flow.

This module asserts what a separately upgraded artifact does; it imports no kitty
code, so a contract that read the consumer would be satisfied by whatever the
consumer happens to do (the ipaddress-contract posture, §6.2.4).
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterator

import keyring
import keyring.backend
import keyring.backends.fail  # noqa: F401  (re-exported through the module attribute)
import keyring.backends.null
import keyring.core
import keyring.errors
import pytest

pytestmark = pytest.mark.l2

# The production service name is "kitty" (KeyringBackend._SERVICE). The probe uses
# a distinct name: the memory backend is process-local, but a credential probe has
# no business sharing a namespace with real credentials even by accident.
_PROBE_SERVICE = "kitty-contract-probe"


class _MemoryKeyring(keyring.backend.KeyringBackend):
    """An in-memory backend implementing the full ``KeyringBackend`` API.

    It exists to make the delegation observable: installing it as the resolved
    backend and driving the public API proves the API operates on
    ``keyring.get_keyring()`` — the fact ``KeyringBackend``'s blind delegation
    rests on.

    ``priority`` is below every auto-selectable backend (``fail.Keyring`` sits at
    0, the chainer at -1) because keyring's discovery enumerates **every loaded
    subclass of ``KeyringBackend``** — measured: a test-defined class at priority
    1 wins ``init_backend()``'s ``max(by_priority)`` for the whole process the
    moment this module is imported. At -2 the class is invisible to auto-detection
    and reachable only through the explicit ``set_keyring`` install, which does
    not consult priority.
    """

    priority = -2

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        self.store.pop((service, username), None)


@pytest.fixture()
def memory_backend() -> Iterator[_MemoryKeyring]:
    """Install the in-memory backend and restore process resolution afterwards."""
    backend = _MemoryKeyring()
    keyring.set_keyring(backend)
    try:
        yield backend
    finally:
        # Re-detect through the public API: env and config are untouched here, so
        # init_backend() re-resolves whatever this platform resolves on a fresh
        # start — no private module state is poked.
        keyring.core.init_backend()


def test_the_module_level_api_delegates_to_the_resolved_backend(memory_backend: _MemoryKeyring) -> None:
    """``set``/``get``/``delete`` all operate on ``keyring.get_keyring()``.

    Falsification: a keyring release that resolved the backend once at import, or
    that routed the public calls anywhere other than ``get_keyring()``, fails the
    round-trip — the writes would land somewhere this backend never sees.
    """
    keyring.set_password(_PROBE_SERVICE, "agent", "secret")

    assert memory_backend.store == {(_PROBE_SERVICE, "agent"): "secret"}
    assert keyring.get_password(_PROBE_SERVICE, "agent") == "secret"

    keyring.delete_password(_PROBE_SERVICE, "agent")
    assert keyring.get_password(_PROBE_SERVICE, "agent") is None
    assert memory_backend.store == {}


def test_python_keyring_backend_env_var_selects_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """``PYTHON_KEYRING_BACKEND=<dotted path>`` yields an instance of that class.

    Falsification: a rename of the environment variable, or a change in the
    ``package.module.Class`` grammar, turns this red. This is the documented
    override users reach for when auto-detection picks wrong — the same knob
    ``KeyringBackend``'s users would need on a headless box.
    """
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")

    resolved = keyring.core.load_env()

    assert isinstance(resolved, keyring.backends.null.Keyring)


def test_resolution_always_lands_on_a_keyring_backends_class() -> None:
    """The resolved backend's class lives under ``keyring.backends.*``.

    Falsification: a release relocating the backend classes outside the namespace,
    or resolving to an object outside it, fails here — before any credential flow
    sees the new shape.
    """
    resolved = keyring.get_keyring()

    assert type(resolved).__module__.startswith("keyring.backends.")


def test_on_linux_without_secretservice_resolution_lands_on_the_fail_fallback() -> None:
    """Headless Linux (no SecretService) resolves to ``fail.Keyring``.

    Falsification: a release that changed the terminal fallback — or stopped
    falling back at all — turns this red on exactly the environments where kitty's
    F39 handler is what stands between the user and a crash. Guarded: on a
    D-Bus-equipped box the SecretService backend classifies and resolution is the
    chainer, so the assertion skips there rather than failing for environmental
    reasons.
    """
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux-only arm")
    if _secretservice_available():
        pytest.skip("SecretService reachable; resolution classifies to it, not the fallback")

    assert isinstance(keyring.get_keyring(), keyring.backends.fail.Keyring)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only native resolution")
def test_on_macos_resolution_lands_on_the_keychain_when_the_security_api_is_available() -> None:
    """macOS resolves to the Keychain backend whenever its Security API classifies.

    Falsification: a release rotating the macOS backend class (rename, move, or a
    priority collapse) turns this red on every machine where the Keychain actually
    works. The import is part of the probe: the bare ``keyring>=23.0`` dependency
    does not carry pyobjc, so on the macOS CI leg this arm skips with a stated
    reason instead of pretending coverage.
    """
    try:
        from keyring.backends import macOS

        priority = macOS.Keyring.priority
    except (ImportError, AttributeError, RuntimeError):
        pytest.skip("macOS Security framework (pyobjc) unavailable in this environment")
    if priority < 1:
        pytest.skip("macOS Keychain backend does not classify as recommended here")

    assert isinstance(keyring.get_keyring(), macOS.Keyring)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only native resolution")
def test_on_windows_resolution_lands_on_credential_manager_when_available() -> None:
    """Windows resolves to the Credential Manager backend when it classifies.

    Falsification: as the macOS arm — a rotated backend class or a priority
    collapse turns this red on the Windows Fast-gate leg, where pywin32-ctypes (a
    base dependency) makes the arm genuinely assert rather than skip.
    """
    try:
        from keyring.backends import Windows

        priority = Windows.WinvaultKeyring.priority
    except (ImportError, AttributeError, RuntimeError):
        pytest.skip("Windows credential APIs unavailable in this environment")
    if priority < 1:
        pytest.skip("Windows Credential Manager backend does not classify as recommended here")

    assert isinstance(keyring.get_keyring(), Windows.WinvaultKeyring)


def test_password_delete_error_is_in_the_documented_errors_family() -> None:
    """``PasswordDeleteError`` subclasses ``KeyringError``.

    Falsification: a release renaming either class fails here with a message naming
    the surface ``KeyringBackend.delete`` suppresses, rather than as an
    ``AttributeError`` raised from inside a credential flow.
    """
    assert issubclass(keyring.errors.PasswordDeleteError, keyring.errors.KeyringError)


def _secretservice_available() -> bool:
    """Report whether the SecretService backend classifies as available here.

    The availability probe is ``priority``, which the backend computes by touching
    the session bus; the failure mode is environment-specific (``RuntimeError`` for
    a missing bus, ``PermissionError`` on a hardened runner — measured on the
    environment this landed on), so the guard catches ``Exception``. Any failure
    means "not available", which is the only fact the guard needs.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        from keyring.backends import SecretService

        return bool(SecretService.Keyring.priority >= 1)
    except Exception:
        return False


def test_the_module_finds_known_positives() -> None:
    """Self-guard: the module cannot rot into a no-op.

    Deleting a behavioural test drops the count below the floor and goes red — the
    same shape the aiohttp twin uses, counting **sync** functions because none of
    this module's tests are coroutines (the aiohttp guard counts coroutine methods
    and would pass vacuously here). The guard excludes itself by name, so the
    floor counts behavioural tests only: without the exclusion, an off-by-one
    floor would let the deletion of any single test pass green.
    """
    guard_name = "test_the_module_finds_known_positives"
    behavioural: list[str] = []
    for name, obj in vars(sys.modules[__name__]).items():
        if name == guard_name:
            continue
        if inspect.isclass(obj) and obj.__module__ == __name__:
            behavioural.extend(n for n, m in inspect.getmembers(obj, inspect.isfunction) if n.startswith("test_"))
        elif inspect.isfunction(obj) and obj.__module__ == __name__ and name.startswith("test_"):
            behavioural.append(name)

    assert len(behavioural) >= 7, f"contract module lost its behavioural coverage; only {behavioural} remain"
