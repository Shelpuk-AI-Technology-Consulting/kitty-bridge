"""GeminiAdapter — configures Google Gemini CLI to talk to the local bridge."""

from __future__ import annotations

from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.types import BridgeProtocol

__all__ = ["GeminiAdapter"]


class GeminiAdapter(LauncherAdapter):
    """Launcher adapter for Google Gemini CLI (https://github.com/google-gemini/gemini-cli).

    Configures Gemini CLI to route requests through the local bridge using
    ``GOOGLE_GEMINI_BASE_URL`` and ``GEMINI_API_KEY`` environment variables
    supported by the ``@google/genai`` SDK.
    """

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def binary_name(self) -> str:
        return "gemini"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.GEMINI_API

    def build_spawn_config(self, profile: Profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        del profile  # Model selection is handled by Gemini CLI itself; not passed via env
        return SpawnConfig(
            cli_args=[],
            env_overrides={
                "GOOGLE_GEMINI_BASE_URL": f"http://127.0.0.1:{bridge_port}",
                "GEMINI_API_KEY": resolved_key,
            },
            env_clear=[],
        )
