# kitty-bridge

Use your favorite coding agent with any LLM provider.

> **Claude Code with MiniMax. Codex with GLM. Gemini CLI with OpenRouter. One command.**

```bash
pip install kitty-bridge
```

## Quick Start

**1. Install**

```bash
pip install kitty-bridge
```

Requires Python 3.10+.

**2. Set up a profile**

```bash
kitty setup
```

An interactive wizard walks you through picking a provider, a model, and entering your API key. Takes 30 seconds.

**3. Launch your agent**

```bash
kitty claude      # Claude Code → your provider
kitty codex       # Codex CLI → your provider
kitty gemini      # Gemini CLI → your provider
kitty kilo        # Kilo Code → your provider
```

That's it. Your coding agent now talks to the LLM you chose — not the one it was built for.

### Example: Use GLM with Claude Code

```bash
$ pip install kitty-bridge
$ kitty setup
  ? Provider: openai
  ? Model: openai/gpt-5.4-pro
  ? API key: ********

$ kitty claude
  ✓ Bridge running on port <random_port>
  ✓ Claude Code launched
  > Hello! How can I help you today?
```

### Example: Use Gemma 4 31B with Claude Code

```bash
$ pip install kitty-bridge
$ kitty setup
  ? Provider: Google AI Studio
  ? Model: gemma-4-31b-it
  ? API key: ********

$ kitty claude
  ✓ Bridge running on port <random_port>
  ✓ Claude Code launched
  > Hello! How can I help you today?
```

## Balanced Profiles

A **balanced profile** combines multiple providers into one. Each request is sent to a randomly chosen healthy provider. If one provider goes down, the others pick up the slack automatically.

**Why use it:**
- **Cost savings** — spread requests across cheaper providers
- **Rate limit resilience** — never hit a single provider's limit
- **Fault tolerance** — if one provider is down, the others keep working

**How to create one:**

```bash
kitty profile
# → "Create balancing profile" → select 2+ member profiles
```

**Example:** Combine MiniMax, Novita, and Z.AI into one balanced profile called `my-pool`, then use it with any agent:

```bash
kitty my-pool claude
kitty my-pool codex
```

When you run this, each request goes to a random healthy member. If MiniMax returns an error, kitty silently retries on Novita or Z.AI — your agent never sees the failure.

## Bridge Mode

Bridge mode starts a standalone OpenAI-compatible API server on your machine. Use it when you want to connect tools that speak the OpenAI API — IDEs, custom scripts, anything that accepts a base URL.

```bash
kitty bridge          # use default profile
kitty my-profile bridge   # use a specific profile
```

Point your tool at `http://localhost:<port>` and it just works.

**Available endpoints:**

| Endpoint | Protocol | Used by |
|----------|----------|---------|
| `POST /v1/chat/completions` | Chat Completions | General purpose |
| `POST /v1/messages` | Anthropic Messages | Claude Code |
| `POST /v1/responses` | OpenAI Responses | Codex |
| `POST /v1/gemini/generateContent` | Gemini | Gemini CLI |
| `GET /healthz` | Health check | Monitoring |

## Supported Agents

| Agent | Command | What it is |
|-------|---------|------------|
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | `kitty claude` | Anthropic's coding agent |
| [Codex CLI](https://github.com/openai/codex) | `kitty codex` | OpenAI's coding agent |
| [Gemini CLI](https://github.com/google-gemini/gemini-cli) | `kitty gemini` | Google's coding agent |
| [Kilo Code](https://github.com/Kilo-Org/kilocode) | `kitty kilo` | Open-source coding agent |

## Supported Providers

| Provider | Type ID | Notes |
|----------|---------|-------|
| OpenAI | `openai` | |
| Anthropic | `anthropic` | Direct Anthropic Messages API |
| OpenRouter | `openrouter` | Multi-provider router |
| MiniMax | `minimax` | |
| Novita | `novita` | |
| Z.AI (regular) | `zai_regular` | General-purpose endpoint |
| Z.AI (coding) | `zai_coding` | Coding-optimized endpoint |
| Fireworks | `fireworks` | |
| Google AI Studio | `google_aistudio` | Gemini models via OpenAI-compatible endpoint |
| Ollama | `ollama` | Local LLM deployment |
| OpenCode | `opencode_go` | |
| AWS Bedrock | `bedrock` | Uses boto3 SigV4 auth |
| Azure OpenAI | `azure` | Requires deployment name |
| Google Vertex AI | `vertex` | Requires project and location |

## Commands

| Command | Description |
|---------|-------------|
| `kitty setup` | Create your first profile (interactive wizard) |
| `kitty profile` | Manage profiles (create, delete, set default, list) |
| `kitty doctor` | Diagnose installation issues |
| `kitty cleanup` | Restore agent config files after a crash |
| `kitty bridge` | Start a standalone API server |
| `kitty claude` | Launch Claude Code with default profile |
| `kitty codex` | Launch Codex with default profile |
| `kitty gemini` | Launch Gemini CLI with default profile |
| `kitty kilo` | Launch Kilo Code with default profile |
| `kitty <profile> <agent>` | Launch an agent with a specific profile |
| `kitty <profile> bridge` | Start bridge with a specific profile |
| `kitty --no-validate <profile> <agent>` | Skip API key validation |
| `kitty --debug <profile> <agent>` | Enable debug logging to ~/.cache/kitty/bridge.log |
| `kitty --debug-file /path <profile> <agent>` | Write debug logs to a custom path |
| `kitty --logging <profile> <agent>` | Enable token usage logging to ~/.cache/kitty/usage.log |
| `kitty --log-file /path <profile> <agent>` | Write usage logs to a custom path (implies --logging) |
| `kitty --version` | Print version |
| `kitty --help` | Print help |

## Technical Details

### How it works

kitty sits between your coding agent and the upstream LLM provider:

```
Agent (Claude Code / Codex / Gemini / Kilo) → kitty bridge → upstream provider
```

When you run `kitty claude`:
1. kitty reads your profile (provider, model, API key)
2. Starts a local HTTP bridge on a random port
3. Configures the agent to send requests to the bridge instead of its default endpoint
4. The bridge translates each request to the provider's format and forwards it
5. Responses are translated back to the agent's native format
6. When the agent exits, kitty restores the agent's config files

### Profiles

A **profile** binds a provider, model, and API key together. Stored in `~/.config/kitty/profiles.json`.

```bash
kitty setup        # create a profile interactively
kitty profile      # manage existing profiles
kitty my-profile claude  # use a specific profile
```

Profile names must be 1-32 characters, lowercase letters, numbers, dashes, or underscores. Reserved words like `setup`, `claude`, `codex`, `gemini`, `kilo`, `profile`, `bridge` cannot be used as profile names.

### Pre-flight validation

Before launching, kitty validates your API key with a lightweight test request. If the key is invalid, you get a clear error immediately — not a cryptic failure inside the agent.

```bash
kitty --no-validate my-profile claude  # skip validation (e.g. air-gapped/offline environments)
```

### Logging

kitty has two independent logging streams, each with its own flag and optional custom path.

**Token usage logs** — records prompt/completion token counts per request:

```bash
# Default location: ~/.cache/kitty/usage.log
kitty --logging claude

# Custom location
kitty --log-file /tmp/my-usage.log claude
```

**Debug logs** — verbose tracing of requests, responses, and protocol translation:

```bash
# Default location: ~/.cache/kitty/bridge.log
kitty --debug claude

# Custom location
kitty --debug-file /tmp/my-debug.log claude
```

Both flags work in launch mode and bridge mode:

```bash
kitty --debug --log-file /tmp/usage.log my-profile bridge
kitty --debug-file /tmp/debug.log --logging my-profile codex
```

| Flag | What it logs | Default path | Custom path flag |
|------|-------------|--------------|-----------------|
| `--logging` | Token usage | `~/.cache/kitty/usage.log` | `--log-file PATH` |
| `--debug` | Request/response tracing | `~/.cache/kitty/bridge.log` | `--debug-file PATH` |

### Cleanup

kitty restores agent config files after the agent exits. Three layers of cleanup:

1. **Normal exit** — `finally` block
2. **Crash / `SIGTERM`** — `atexit` handler
3. **`SIGKILL` / kernel OOM** — run `kitty cleanup` manually

### Project structure

```
src/kitty/
├── bridge/          # HTTP bridge + protocol translation
├── cli/             # Command-line interface
├── credentials/     # API key storage
├── launchers/       # Agent-specific adapters
├── profiles/        # Profile management
├── providers/       # Upstream provider adapters
├── tui/             # Terminal UI components
└── types.py         # Shared types
```

## FAQ

### "API Error: Unable to connect to API (ConnectionRefused)"

The agent is trying to connect to a bridge that isn't running. Usually caused by a stale config from a previous crashed session:

```bash
kitty cleanup
```

### "API Error: 401" or "token expired or incorrect"

Your API key has expired or been revoked. Run setup again:

```bash
kitty setup
```

### "Prompt exceeds max length" (Z.AI error 1261)

The conversation has grown beyond the model's context window. Use `/clear` in the agent to reset.

### Can I use kitty with Cursor, Windsurf, or other IDEs?

Yes — start a bridge, then point your IDE's "OpenAI base URL" setting at `http://localhost:<port>`:

```bash
kitty bridge
# Then configure your IDE to use http://localhost:<port>/v1/chat/completions
```

### Can I run a local model?

Yes. Install [Ollama](https://ollama.ai), pull a model, then create a profile with provider `ollama`:

```bash
kitty setup
# Provider: ollama
# Base URL: http://localhost:11434/v1
# Model: llama3
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy src/kitty
```

## License

MIT
