"""End-to-end integration tests — launch real agents through Kitty bridge.

These tests start a real BridgeServer, spawn actual agent CLIs (Codex, Claude Code,
Gemini, Kilo), and verify they produce correct responses. They require:
- All agent CLIs installed (codex, claude, gemini, kilo)
- Kitty profiles configured (~/.config/kitty/profiles.json)
- Kindly MCP configured for each agent

Run: pytest tests/integration/ -v -s --timeout=180
Skip in normal runs: pytest tests/ --ignore=tests/integration
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from kitty.bridge.server import BridgeServer
from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
from kitty.launchers.base import LauncherAdapter
from kitty.launchers.claude import ClaudeAdapter
from kitty.launchers.codex import CodexAdapter
from kitty.launchers.discovery import discover_binary
from kitty.launchers.gemini import GeminiAdapter
from kitty.launchers.kilo import KiloAdapter
from kitty.profiles.schema import Profile
from kitty.profiles.store import ProfileStore
from kitty.providers.registry import get_provider

# ── Constants ──────────────────────────────────────────────────────────────

PROFILES_DIR = Path.home() / ".config" / "kitty"
PROFILES_FILE = PROFILES_DIR / "profiles.json"
CREDENTIALS_FILE = PROFILES_DIR / "credentials.json"

AGENT_NAMES = ["codex", "claude", "gemini", "kilo"]
PROFILE_NAMES = ["glm", "minimax", "gemma", "kimi"]

MATH_PROMPT = "What is 2+2? Return ONLY the number, nothing else."
MATH_TIMEOUT = 90
MATH_EXPECTED = "4"

WEB_SEARCH_PROMPT = (
    "Use the Kindly web search MCP tool (web_search) to search for "
    "'current weather in Venice Italy'. Return the weather information you find. "
    "Include the temperature if available."
)
WEB_SEARCH_TIMEOUT = 180

ADAPTERS: dict[str, LauncherAdapter] = {
    "codex": CodexAdapter(),
    "claude": ClaudeAdapter(),
    "gemini": GeminiAdapter(),
    "kilo": KiloAdapter(),
}

# Agent-specific CLI command builders
# Each returns (binary_name, [args_before_prompt], [args_after_prompt])
def _build_agent_cmd(agent_name: str, prompt: str) -> list[str]:
    """Build the full CLI command for an agent."""
    binary = str(_get_binary(agent_name))
    if agent_name == "codex":
        return [binary, "exec", "--full-auto", prompt]
    elif agent_name == "claude":
        return [binary, "-p", prompt, "--dangerously-skip-permissions"]
    elif agent_name == "gemini":
        return [binary, "-p", prompt, "-y"]
    elif agent_name == "kilo":
        return [binary, "run", "--auto", prompt]
    raise ValueError(f"Unknown agent: {agent_name}")


# ── Helpers ────────────────────────────────────────────────────────────────


def _resolve_key(profile: Profile) -> str:
    from kitty.credentials.store import CredentialNotFoundError
    cred_store = CredentialStore(backends=[FileBackend()])
    try:
        return cred_store.resolve(profile)
    except CredentialNotFoundError:
        pytest.skip(f"Credentials for profile '{profile.name}' not found")


def _get_profile(name: str) -> Profile:
    store = ProfileStore()
    profile = store.get(name)
    if profile is None:
        pytest.skip(f"Profile '{name}' not found in {PROFILES_FILE}")
    return profile


def _get_binary(name: str) -> Path:
    path = discover_binary(name)
    if path is None:
        pytest.skip(f"Agent CLI '{name}' not found on PATH")
    return path


# ── Bridge + Agent Runner ──────────────────────────────────────────────────


async def _run_agent_through_bridge(
    agent_name: str,
    profile: Profile,
    prompt: str,
    timeout: int,
) -> str:
    """Start bridge, spawn agent CLI, capture stdout, return output."""
    adapter = ADAPTERS[agent_name]
    provider = get_provider(profile.provider)
    resolved_key = _resolve_key(profile)

    # Start bridge server
    server = BridgeServer(adapter, provider, resolved_key, model=profile.model)
    port = await server.start_async()

    # Build spawn config (env vars for the agent)
    spawn_config = adapter.build_spawn_config(profile, port, resolved_key)

    # Prepare external config files (e.g. Claude settings.json, Kilo opencode.json)
    original_settings = None
    if hasattr(adapter, "prepare_launch"):
        original_settings = adapter.prepare_launch(spawn_config.env_overrides)

    # Build environment
    env = os.environ.copy()
    for key in spawn_config.env_clear:
        env.pop(key, None)
    env.update(spawn_config.env_overrides)

    # Build CLI command
    cmd = _build_agent_cmd(agent_name, prompt)
    # Prepend any adapter-specific CLI args (e.g. Codex -c flags)
    if spawn_config.cli_args:
        # Insert adapter CLI args after the binary name
        cmd = [cmd[0]] + spawn_config.cli_args + cmd[1:]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            pytest.fail(f"Agent {agent_name} timed out after {timeout}s")
    finally:
        # Cleanup external config files
        try:
            if hasattr(adapter, "cleanup_launch"):
                adapter.cleanup_launch(original_settings)
        except Exception:
            pass
        # Stop bridge server
        try:
            await server.stop_async()
        except Exception:
            pass

    output = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()

    if not output and stderr_text:
        # Some agents output to stderr (e.g. Gemini with -p flag)
        output = stderr_text

    return output


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestAgentMathE2E:
    """Test that each agent returns 4 when asked 2+2."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    @pytest.mark.parametrize("profile_name", PROFILE_NAMES)
    async def test_math_2plus2(self, agent_name, profile_name):
        profile = _get_profile(profile_name)
        output = await _run_agent_through_bridge(
            agent_name, profile, MATH_PROMPT, MATH_TIMEOUT,
        )
        assert MATH_EXPECTED in output, f"Expected '4' in output, got: {output[:500]}"


@pytest.mark.slow
class TestAgentWebSearchE2E:
    """Test that each agent can use Kindly MCP to search the web."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    @pytest.mark.parametrize("profile_name", PROFILE_NAMES)
    async def test_web_search_weather(self, agent_name, profile_name):
        profile = _get_profile(profile_name)
        output = await _run_agent_through_bridge(
            agent_name, profile, WEB_SEARCH_PROMPT, WEB_SEARCH_TIMEOUT,
        )
        # Verify the output contains weather-related keywords
        output_lower = output.lower()
        weather_keywords = ["venice", "weather", "temperature", "°c", "°f", "celsius", "fahrenheit"]
        found = [kw for kw in weather_keywords if kw in output_lower]
        assert found, (
            f"Expected weather keywords {weather_keywords} in output, "
            f"but none found. Output: {output[:1000]}"
        )
        # Verify that Kindly's page_content was used (not just a generic response)
        # The response should contain specific weather data, not just "I can't search"
        first_200 = output_lower[:200]
        assert not ("can't" in first_200 and "search" in first_200), (
            f"Agent appears to have failed to use web search. Output: {output[:500]}"
        )
