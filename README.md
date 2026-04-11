# Kitty Bridge

A thin launcher for coding agents that routes requests through a local API bridge to any upstream Chat Completions provider.

## What it does

kitty sits between your coding agent and the upstream API:

```
Agent (codex / claude / gemini / kilo) → kitty bridge → Chat Completions → upstream provider
```

The bridge translates between the agent's wire protocol (Responses API, Messages API, or Gemini API) and a standard Chat Completions API, so you can use any compatible provider without the agent needing native support.

## Install

```bash
pip install -e .
```

Requires Python 3.10+.

## Quick start

```bash
# First-time setup — interactive wizard creates a profile
kitty setup

# Launch Codex CLI through the bridge
kitty codex

# Launch Claude Code through the bridge
kitty claude

# Launch Gemini CLI through the bridge
kitty gemini

# Launch Kilo Code through the bridge
kitty kilo

# Use a specific profile
kitty my-profile codex

# Start a standalone OpenAI-compatible API server (bridge mode)
kitty bridge
kitty my-profile bridge
```

## Commands

| Command | Description |
|---------|-------------|
| `kitty setup` | Interactive wizard to create your first profile |
| `kitty profile` | Manage profiles (create, delete, set default, list) |
| `kitty doctor` | Diagnose installation issues |
| `kitty cleanup` | Remove stale bridge values from agent config files |
| `kitty codex` | Launch Codex CLI through the bridge |
| `kitty claude` | Launch Claude Code through the bridge |
| `kitty gemini` | Launch Gemini CLI through the bridge |
| `kitty kilo` | Launch Kilo Code through the bridge |
| `kitty bridge` | Start a standalone OpenAI-compatible API server |
| `kitty <profile>` | Launch default agent with a named profile |
| `kitty <profile> codex` | Launch Codex with a named profile |
| `kitty <profile> claude` | Launch Claude Code with a named profile |
| `kitty <profile> gemini` | Launch Gemini CLI with a named profile |
| `kitty <profile> kilo` | Launch Kilo Code with a named profile |
| `kitty <profile> bridge` | Start bridge server with a named profile |
| `kitty --version` | Print version |
| `kitty --help` | Print help |

## Supported agents

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — Anthropic's coding agent (Messages API)
- [Codex CLI](https://github.com/openai/codex) — OpenAI's coding agent (Responses API)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli) — Google's coding agent (Gemini API)
- [Kilo Code](https://github.com/kilocode/kilo-code) — Open-source coding agent

## Supported providers

| Provider           | Type ID | Notes                                    |
|--------------------|---------|------------------------------------------|
| Z.AI (regular)     | `zai_regular` | Standard Z.AI API                        |
| Z.AI (coding plan) | `zai_coding` | Z.AI Coding Plan endpoint      |
| MiniMax            | `minimax` |                                          |
| Novita             | `novita` |                                          |
| Ollama             | `ollama` | Local LLM deployment (OpenAI-compatible) |
| OpenAI             | `openai` |                                          |
| OpenRouter         | `openrouter` | Multi-provider router                    |
| Fireworks          | `fireworks` | Fire Pass endpoint                       |
| Anthropic          | `anthropic` | Direct Anthropic Messages API            |
| AWS Bedrock        | `bedrock` | Uses boto3 SigV4 auth                    |
| Azure OpenAI       | `azure` | Azure-specific endpoint format           |
| Google Vertex AI   | `vertex` | Requires project/location config         |

## How it works

### Agent launch mode

1. kitty resolves your profile (provider, model, API key)
2. Starts a local HTTP bridge on a random port
3. Configures the agent to talk to the bridge
4. The bridge translates requests to Chat Completions format and forwards them to your provider
5. Responses are translated back to the agent's native format
6. When the agent exits, kitty restores the agent's config files to their original state

### Bridge mode

Start a standalone OpenAI-compatible API server without launching an agent:

```bash
# Use default profile
kitty bridge

# Use a specific profile
kitty my-profile bridge
```

The server exposes:
- `POST /v1/chat/completions` — Chat Completions API (streaming and non-streaming)
- `POST /v1/messages` — Anthropic Messages API (for Claude Code)
- `POST /v1/responses` — OpenAI Responses API (for Codex)
- `POST /v1/gemini/generateContent` — Gemini API (for Gemini CLI)
- `GET /healthz` — Health check

### Balancing profiles

Balancing profiles distribute LLM calls across multiple providers in round-robin order:

```bash
# Create a balancing profile via the profile menu
kitty profile
# → "Create balancing profile" → select 2+ member profiles

# Use a balancing profile
kitty my-balancer codex
kitty my-balancer bridge
```

Each request is routed to the next provider in the rotation. Useful for cost optimization, rate limit distribution, and resilience across multiple providers.

## Profiles

A **profile** binds a provider, model, and API key together. Profiles are stored in `~/.config/kitty/profiles.json`.

```bash
# Create profiles interactively
kitty setup
kitty profile

# Use a specific profile
kitty my-profile claude

# Set a default profile (used when no profile is named)
kitty profile  # → "Set default profile"
```

A **balancing profile** is a list of regular profiles. Requests are round-robined across members.

## Pre-flight validation

Before launching, kitty validates your API key by making a lightweight test request to the upstream provider. If the key is expired or invalid, kitty reports the error immediately and exits.

```bash
# Skip validation (e.g. in air-gapped environments)
kitty --no-validate my-profile claude
```

## Cleanup

kitty uses a three-layer cleanup strategy to restore agent config files:

1. **`finally` block** — normal path after agent exits
2. **`atexit` handler** — runs on `sys.exit()`, unhandled exceptions, and `SIGTERM`
3. **`kitty cleanup`** — manual fallback for `SIGKILL` or kernel OOM

```bash
# Manually restore stale agent config
kitty cleanup
```

## Architecture

```
src/kitty/
├── bridge/          # HTTP bridge server + protocol translation
│   ├── server.py    # aiohttp-based bridge with round-robin balancing
│   ├── engine.py    # Shared translation primitives
│   ├── responses/   # Responses API translation (Codex)
│   ├── messages/    # Messages API translation (Claude Code)
│   └── gemini/      # Gemini API translation
├── cli/             # Command-line interface
│   ├── main.py      # Entry point
│   ├── router.py    # Argument routing
│   ├── launcher.py  # Bridge + child process orchestration
│   ├── doctor_cmd.py
│   ├── setup_cmd.py
│   └── profile_cmd.py
├── credentials/     # API key storage (file backend)
├── launchers/       # Agent-specific adapters
│   ├── claude.py    # Claude Code adapter (env vars + settings.json)
│   ├── codex.py     # Codex CLI adapter
│   ├── gemini.py    # Gemini CLI adapter
│   ├── kilo.py      # Kilo Code adapter
│   └── discovery.py # Binary discovery (PATH + fallbacks)
├── profiles/        # Profile management
│   ├── schema.py    # Profile, BalancingProfile, BackendConfig
│   ├── store.py     # JSON persistence with type discriminator
│   └── resolver.py  # Profile lookup and default resolution
├── providers/       # Upstream provider adapters
├── tui/             # Terminal UI components
└── types.py         # Shared types
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

## FAQ

### "API Error: Unable to connect to API (ConnectionRefused)"

The coding agent is trying to connect to a bridge server that isn't running. Two common causes:

**Stale config from a crashed session.** If kitty was killed with `SIGKILL`, the agent's config file may still contain stale values. Fix:
```bash
kitty cleanup
```

**Running the agent directly without kitty.** Always launch through kitty, or run `kitty cleanup` first.

### "API Error: 401" or "token expired or incorrect"

Your API key has expired or been revoked:
```bash
kitty setup
```

### "Prompt exceeds max length" (Z.AI error 1261)

The conversation context has grown beyond the model's context window. Use `/clear` in the agent to reset the conversation.

### "pip's dependency resolver ... langchain/langchain-core requires langsmith<0.4, but you have langsmith 0.4.x"

`kitty-bridge` does not depend on `langsmith`. This warning means your current Python environment already has a `langchain 0.3.x` stack that is incompatible with the installed `langsmith` version.

If you are staying on `langchain 0.3.x`, repair the environment first:

```bash
python -m pip check
python -m pip install --upgrade "langsmith<0.4"
python -m pip check
```

Then install kitty again:

```bash
python -m pip install -e .
```

Using a fresh virtual environment per project avoids this class of cross-project dependency conflict.

## License

MIT
