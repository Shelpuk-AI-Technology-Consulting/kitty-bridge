"""Tests for the impersonated Codex CLI identity leaf.

KBR-161.  :mod:`kitty.codex_identity` is the single source for the version and
user-agent that **both** legs of the OpenAI subscription provider present --
the API leg's ``_build_codex_headers`` and the OAuth leg's token POSTs.  KBR-8
established that one client stating two versions is a one-line detection rule
for bridge traffic; this module's whole purpose is that there is nowhere for a
second version to come from.

These are L1: the rendering of one pure function.  That the two *legs* agree is
a separate claim, proved in ``tests/test_oauth_leg_identity.py`` at L2.
"""

from __future__ import annotations

import platform
import re

from kitty import codex_identity

#: ``codex_cli_rs/<version> (<os_type> <os_version>; <arch>)`` -- the format
#: ``get_codex_user_agent()`` in ``codex-rs/login/src/auth/default_client.rs``
#: produces, which this impersonates.
_USER_AGENT = re.compile(r"^codex_cli_rs/(?P<version>[^\s]+) \((?P<os>[^;]+); (?P<arch>[^)]+)\)$")


class TestBuildCodexUserAgent:
    def test_renders_the_codex_cli_format(self) -> None:
        """The string parses as Codex CLI's user-agent grammar."""
        match = _USER_AGENT.match(codex_identity.build_codex_user_agent())

        assert match is not None, codex_identity.build_codex_user_agent()

    def test_states_the_module_version_and_the_running_platform(self) -> None:
        """Version comes from the constant; the rest from :mod:`platform`."""
        match = _USER_AGENT.match(codex_identity.build_codex_user_agent())

        assert match is not None
        assert match.group("version") == codex_identity.CODEX_CLI_VERSION
        assert match.group("os") == f"{platform.system()} {platform.release()}"
        assert match.group("arch") == platform.machine()

    def test_the_version_is_read_at_call_time(self, monkeypatch) -> None:
        """Patching the constant moves the user-agent.

        Read at call time rather than bound at import so that the L2 guard can
        patch one name and prove every consumer moves with it.  A module-level
        alias elsewhere would silently not move -- which is the KBR-8 defect.
        """
        monkeypatch.setattr(codex_identity, "CODEX_CLI_VERSION", "9.9.9-sentinel")

        assert codex_identity.build_codex_user_agent().startswith("codex_cli_rs/9.9.9-sentinel ")

    def test_the_version_is_not_kitty_s_release_train(self) -> None:
        """KBR-8: the impersonated version must not track kitty's version."""
        import kitty

        assert kitty.__version__ != codex_identity.CODEX_CLI_VERSION
