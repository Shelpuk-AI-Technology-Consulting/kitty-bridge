# Kitty Code

A thin launcher for coding agents (Codex, Claude Code) that routes requests through a local API bridge to any upstream Chat Completions provider.

## What it does

kitty sits between your coding agent and the upstream API:

```
Agent (codex / claude) → kitty bridge → Chat Completions → upstream provider
```

The bridge translates between the agent's wire protocol (Responses API or Messages API) and a standard Chat Completions API, so you can use any compatible provider without the agent needing native support.

## Install

```bash
pip install -e .
```

Requires Python 3.10+.

## Quick start

```bash
# First-time setup — interactive wizard creates a profile
kitty setup

# Launch Codex through the bridge
kitty codex

# Launch Claude Code through the bridge
kitty claude

# Use a specific profile
kitty my-profile codex
```

## Commands

| Command | Description |
|---------|-------------|
| `kitty setup` | Interactive wizard to create your first profile |
| `kitty profile` | Manage profiles (create, delete, set default, list) |
| `kitty doctor` | Diagnose installation issues |
| `kitty cleanup` | Remove stale bridge values from agent config files |
| `kitty codex` | Launch Codex CLI through the bridge |
| `kitty claude` | Launch Claude Code CLI through the bridge |
| `kitty <profile>` | Launch default target with a named profile |
| `kitty <profile> codex` | Launch Codex with a named profile |
| `kitty <profile> claude` | Launch Claude Code with a named profile |
| `kitty --version` | Print version |
| `kitty --help` | Print help |

## How it works

1. kitty resolves your profile (provider, model, API key)
2. Starts a local HTTP bridge on a random port
3. Configures the agent (Codex or Claude Code) to talk to the bridge
4. The bridge translates requests to Chat Completions format and forwards them to your provider
5. Responses are translated back to the agent's native format
6. When the agent exits, kitty restores the agent's config files to their original state. This cleanup is guaranteed to run even if kitty crashes (via `atexit` handler). If kitty is killed with `SIGKILL` (which cannot be intercepted), use `kitty cleanup` to manually restore the config.

## Supported providers

- ZAI (regular and coding)
- MiniMax
- Novita
- OpenAI
- OpenRouter

## Architecture

```
src/kitty/
├── bridge/          # HTTP bridge server + protocol translation
│   ├── server.py    # aiohttp-based bridge
│   ├── engine.py    # Shared translation primitives
│   ├── responses/   # Responses API translation
│   └── messages/    # Messages API translation
├── cli/             # Command-line interface
│   ├── main.py      # Entry point
│   ├── router.py    # Argument routing
│   ├── launcher.py  # Bridge + child process orchestration
│   ├── doctor_cmd.py
│   ├── setup_cmd.py
│   └── profile_cmd.py
├── credentials/     # API key storage (file + keyring backends)
├── launchers/       # Agent-specific adapters
│   ├── codex.py     # Codex CLI adapter (-c flags)
│   ├── claude.py    # Claude Code adapter (env vars)
│   └── discovery.py # Binary discovery (PATH + fallbacks)
├── profiles/        # Profile management
├── providers/       # Upstream provider adapters
├── tui/             # Terminal UI components
└── types.py         # Shared types (BridgeProtocol enum)
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Type check
mypy src/kitty
```

## Pre-flight validation

Before launching the bridge, kitty validates your API key by making a lightweight test request to the upstream provider. If the key is expired or invalid, kitty reports the error immediately and exits — no waiting for the agent to start only to fail.

```bash
# Skip validation (e.g. in air-gapped environments)
kitty --no-validate glm claude
```

## FAQ

### "API Error: Unable to connect to API (ConnectionRefused)"

This means the coding agent (Claude Code) is trying to connect to a bridge server that isn't running. Two common causes:

**Cause 1: Stale config from a previous crashed session.** If kitty was killed with `SIGKILL` or the machine lost power, the agent's config file may still contain `ANTHROPIC_BASE_URL=http://127.0.0.1:{dead_port}` from the previous session. Fix it with:

```bash
kitty cleanup
```

**Cause 2: Running the agent directly without kitty.** If you launch `claude` directly (not through `kitty`), it reads the stale config values that kitty injected during a previous session. Always launch through kitty, or run `kitty cleanup` first.

### "API Error: 401" or "token expired or incorrect"

Your API key has expired or been revoked. Update it with:

```bash
kitty setup
```

You can also check your key with `kitty doctor`.

### "kitty cleanup" — what does it do?

`kitty cleanup` scans the agent's config file (e.g. `~/.claude/settings.json`) for values that were injected by a previous kitty session but never cleaned up (because kitty crashed). It removes:

- `ANTHROPIC_BASE_URL` pointing to `127.0.0.1` or `localhost`
- `ANTHROPIC_MODEL` and `ANTHROPIC_DEFAULT_*_MODEL` set by kitty
- `ANTHROPIC_API_KEY` set by kitty

It **preserves** your own config values like `API_TIMEOUT_MS`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`, `ANTHROPIC_AUTH_TOKEN`, and non-localhost `ANTHROPIC_BASE_URL`.

### How does kitty ensure cleanup on crash?

kitty uses a three-layer cleanup strategy:

1. **`finally` block** — the normal path. After the agent exits, kitty restores the config in a `finally` block.
2. **`atexit` handler** — registered immediately after patching the config. Runs on `sys.exit()`, unhandled exceptions, and `SIGTERM` (which triggers Python's normal shutdown sequence).
3. **`kitty cleanup`** — manual fallback for the rare case where even `atexit` doesn't run (e.g. `SIGKILL`, kernel OOM kill).

### What about SIGKILL?

`SIGKILL` cannot be intercepted by any process. If kitty is killed with `kill -9`, neither the `finally` block nor the `atexit` handler will run. In this case, run `kitty cleanup` manually.

### Does kitty validate my API key before launching?

Yes. kitty sends a minimal test request to the upstream provider before starting the bridge. If the key is invalid/expired, kitty exits immediately with a clear error message. This saves you from waiting for the agent to start only to see a cryptic API error.

To skip validation (e.g. in air-gapped environments):

```bash
kitty --no-validate glm claude
```

## License

MIT
