"""The impersonated Codex CLI identity, shared by every leg that presents it.

:class:`~kitty.providers.openai_subscription.OpenAISubscriptionAdapter` presents
a Codex CLI identity to OpenAI on **two** transports: the API leg, over
``curl_cffi``, and the OAuth token leg, which lives in :mod:`kitty.auth`. Both
must state the same version, so the version lives here rather than in either of
them.

It is a leaf for the same reason :mod:`kitty.io_encoding` is one: two layers
need one fact, and ``kitty.providers`` already imports ``kitty.auth``, so
housing it in either would either invert that direction or make the providers
layer depend on auth for something that is not a credential. The impersonated
identity is a **provider** fact, not a token fact. This module imports nothing
from ``kitty``, so any layer may depend on it; ``pyproject.toml``'s import
contracts pin that.

Read the version through the module (``codex_identity.CODEX_CLI_VERSION``), never
by binding it into a local alias. KBR-8 was a single request claiming to be two
different Codex CLI versions at once, and a module-level alias re-creates exactly
that: the alias binds at import and stops moving when the source does.
"""

from __future__ import annotations

import platform

__all__ = ["CODEX_CLI_VERSION", "build_codex_user_agent"]

#: The impersonated Codex CLI version, and the single source for **every**
#: version field kitty sends to OpenAI -- the API leg's ``version`` header, its
#: ``User-Agent``, and the four OAuth token POSTs' ``User-Agent`` (KBR-8,
#: KBR-161).  A client stating two different versions is an intermediary and
#: nothing else.
#:
#: The value is the real released Codex CLI version, not kitty's.  (Codex's own
#: checked-in reference workspace carries 0.0.0, a dev placeholder; its release
#: builds carry the real one, which is what this impersonates.)  Nothing rewrites
#: this at build time -- it is edited here when the impersonated version moves.
CODEX_CLI_VERSION = "0.128.0"


def build_codex_user_agent() -> str:
    """Build the ``User-Agent`` string the Codex CLI sends.

    Renders ``codex_cli_rs/<version> (<os_type> <os_version>; <arch>)``, matching
    ``get_codex_user_agent()`` in ``codex-rs/login/src/auth/default_client.rs``.

    The version is deliberately **not** ``kitty.__version__`` (KBR-8). Reading it
    from there made a single request claim to be two different Codex CLI versions
    at once -- something no genuine client does, and so a one-line detection rule
    for bridge traffic -- and tied the header to kitty's release train, changing
    it on every kitty release and nothing else. See
    ``.system_design/TEST_SUITE.md`` finding F1 and §4.3 C1.

    Returns:
        The impersonated Codex CLI user-agent, versioned from
        :data:`CODEX_CLI_VERSION`.
    """
    # Read the constant off the module rather than closing over it, so patching
    # `CODEX_CLI_VERSION` moves every consumer together -- the property the L2
    # single-source guard asserts.
    return (
        f"codex_cli_rs/{CODEX_CLI_VERSION} "
        f"({platform.system()} {platform.release()}; {platform.machine()})"
    )
