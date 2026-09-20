"""L1 unit tests for the launcher credential-resolution path (KBR-87).

``launch_async`` resolves the API key against the credential store as its first
step. Under the corruption contract (SYSTEM_DESIGN.md §11.2), a present-but-
undecodable stored value raises :class:`~kitty.credentials.store.CredentialError`,
and ``launch_async``'s ``except`` tuple was widened in KBR-87 to include it so
the same clean ``Error: …`` + exit fires — no raw traceback.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kitty.cli.launcher import launch_async


def _corrupt_cred_store() -> object:
    """A stand-in store whose ``resolve`` raises the corruption error."""
    from kitty.credentials.store import CredentialError

    def _raise(_profile: object) -> str:
        raise CredentialError(
            "Credential for ref 'ref-1' is corrupt (binascii.Error): "
            "restore it from a backup or re-enter the credential."
        )

    return SimpleNamespace(resolve=_raise)


async def test_corrupt_credential_returns_one_and_skips_the_bridge(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corrupt credential returns ``1`` with the clean corruption message.

    The handler runs before ``egress_block_reason`` (step 2) and the spawn (step
    5+), so patching the network-ish model-context refresh (step 0) keeps the
    test hermetic without affecting the path under test.
    """

    async def _noop() -> None:
        return None

    monkeypatch.setattr(
        "kitty.providers.model_context_sync.refresh_model_context_overrides", _noop
    )

    exit_code = await launch_async(
        adapter=SimpleNamespace(name="codex"),
        provider=SimpleNamespace(),
        profile=SimpleNamespace(name="my-profile", provider="zai_regular", model="gpt-4o", auth_ref="ref-1"),
        cred_store=_corrupt_cred_store(),
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "corrupt" in captured.out + captured.err
