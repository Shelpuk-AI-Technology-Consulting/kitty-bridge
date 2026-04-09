"""CodexAdapter — configures OpenAI Codex CLI to talk to the local bridge."""

from __future__ import annotations

from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.types import BridgeProtocol

__all__ = ["CodexAdapter"]


class CodexAdapter(LauncherAdapter):
    """Launcher adapter for OpenAI Codex CLI (https://github.com/openai/codex).

    Configures Codex to route requests through the local bridge using
    ``-c`` override flags for a custom provider named ``kitty``.
    """

    @property
    def name(self) -> str:
        return "codex"

    @property
    def binary_name(self) -> str:
        return "codex"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.RESPONSES_API

    def build_spawn_config(self, profile: Profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        del resolved_key  # Bridge handles upstream auth; key not passed to child process
        return SpawnConfig(
            cli_args=[
                "-c",
                "model_provider=kitty",
                "-c",
                f"model_providers.kitty.base_url=http://127.0.0.1:{bridge_port}/v1",
                "-c",
                "model_providers.kitty.wire_api=responses",
                "-c",
                "model_providers.kitty.supports_websockets=false",
                "-c",
                "model_providers.kitty.name=Kitty Bridge",
                "-c",
                f"model={profile.model}",
            ],
            env_overrides={},
            env_clear=[],
        )
