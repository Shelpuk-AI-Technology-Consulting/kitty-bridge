<p align="center">
  <a href="https://pypi.org/project/kitty-bridge/">
    <img src="https://img.shields.io/pypi/v/kitty-bridge.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/kitty-bridge/">
    <img src="https://img.shields.io/pypi/pyversions/kitty-bridge.svg" alt="Python version">
  </a>
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
  <img src="https://img.shields.io/github/actions/workflow/status/Shelpuk-AI-Technology-Consulting/kitty-code/ci.yml?branch=main" alt="CI">
  <a href="https://pepy.tech/projects/kitty-bridge" target="_blank" rel="noopener noreferrer">
    <img src="https://static.pepy.tech/badge/kitty-bridge" alt="Total PyPI downloads">
  </a>
</p>

Use your favorite coding agent with any LLM provider.

> **Claude Code with MiniMax. Codex with GLM. Gemini CLI with OpenRouter. One command.**

<p align="left">
  <img src="https://raw.githubusercontent.com/Shelpuk-AI-Technology-Consulting/kitty-code/main/assets/logo.png" alt="Kitty Bridge" width="800">
</p>

## Why Kitty Bridge?

Frontier models are expensive. Claude Opus 4.7 costs $25 per 1M output tokens. GPT-5.5 runs $30–45 per 1M output tokens.
A single long coding session can burn through dollars in minutes. And if you hit your subscription rate limits mid-task,
you're stuck waiting.

Meanwhile, there are coding plans from providers like Z.AI, Novita, Fireworks, MiniMax, and others that deliver capable
coding models at a **fraction of the cost** — often 90% cheaper than the flagship models.

Kitty Bridge gives you three ways to save:

✅ **Switch to a cheaper provider** — Use Claude Code with MiniMax, Codex with GLM, or Gemini CLI with OpenRouter. One
command, instant savings.

✅ **Use your existing subscriptions** — Already paying for ChatGPT Plus or Pro? Use it through Kitty instead of buying
API credits separately.

✅ **Mix models with balanced profiles** — Combine a powerful model like GPT-5.5 with a smaller, cheaper one in a single
session. Kitty randomly distributes requests across both, giving you the reasoning power of a frontier model at roughly
**20–25% of the cost**.

```
Agent (Claude Code / Codex / Gemini / Kilo) → Kitty Bridge → any LLM provider
```

Kitty sits between your coding agent and the upstream provider, translating each agent's native protocol in real time.
Your agent keeps its workflow — you choose the model and the price.

Kitty is also intentionally minimal:

✅ **Local** — Kitty runs on your machine. It does not send your prompts, code, or files to any third-party service
beyond the backend LLM provider you explicitly configure.

✅ **Just a bridge** — Kitty only proxies and translates traffic between the coding agent and the backend model API. It
does not get filesystem access, shell access, or any other extra capabilities.

✅ **No AI inside Kitty** — Kitty does not use an LLM, embeddings, or any other AI system internally. It is a
deterministic local bridge with advanced routing, balancing, and compatibility features.

In short: Kitty is a **minimal, local, and safe bridge** with advanced functionality.

If you like what we're building, please ⭐ **star this repo** – it's a huge motivation for us to keep going!

## Before You Start

You need **two things** to use Kitty:

1. **A coding agent** installed on your
   machine — [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [Codex CLI](https://github.com/openai/codex), [Gemini CLI](https://github.com/google-gemini/gemini-cli),
   or [Kilo Code](https://github.com/Kilo-Org/kilocode)
2. **An account with an LLM provider** — either an API key (pay per token) or a subscription/coding plan

**API key vs. subscription?** Some providers offer regular API access where you pay per token. Others offer subscription
plans (e.g. your ChatGPT Plus subscription) that include usage quota. Kitty supports both — the setup wizard will guide
you based on which provider you pick.

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

> **New to this?** The easiest way to start is with your existing **ChatGPT Plus or Pro subscription** — select "OpenAI
> ChatGPT Plan" during setup. No API key needed; Kitty authenticates through your browser. Alternatively, sign up
> at [OpenRouter](https://openrouter.ai) for a free API key that works with many models.

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

### Example: Use your ChatGPT subscription with Claude Code

```bash
$ pip install kitty-bridge
$ kitty setup
  ? Provider: openai_subscription
  ? Model: gpt-5.4

  Opening browser for OpenAI authentication...

$ kitty claude
  ✓ Bridge running on port <random_port>
  ✓ Claude Code launched
  > Hello! How can I help you today?
```

No API key required — kitty authenticates with your ChatGPT Plus or Pro account through a browser-based OAuth flow. Each
profile gets its own independent session.

### Example: Use MiMo V2 Pro with Claude Code

```bash
$ pip install kitty-bridge
$ kitty setup
  ? Provider: Xiaomi MiMo
  ? Model: mimo-v2-pro
  ? API key: ********

$ kitty claude
  ✓ Bridge running on port <random_port>
  ✓ Claude Code launched
  > Hello! How can I help you today?
```

## Balanced Profiles

A **balanced profile** combines multiple providers into one. Each request is sent to a randomly chosen healthy provider.
If one provider goes down, the others pick up the slack automatically.

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

When you run this, each request goes to a random healthy member. If MiniMax returns an error, kitty silently retries on
Novita or Z.AI — your agent never sees the failure.

## Bridge Mode

Bridge mode starts a standalone OpenAI-compatible API server on your machine. Use it when you want to connect tools that
speak the OpenAI API — IDEs, custom scripts, anything that accepts a base URL.

```bash
kitty bridge          # use default profile
kitty my-profile bridge   # use a specific profile
```

Point your tool at `http://localhost:<port>` and it just works.

**Available endpoints:**

| Endpoint                          | Protocol           | Used by         |
|-----------------------------------|--------------------|-----------------|
| `POST /v1/chat/completions`       | Chat Completions   | General purpose |
| `POST /v1/messages`               | Anthropic Messages | Claude Code     |
| `POST /v1/responses`              | OpenAI Responses   | Codex           |
| `POST /v1/gemini/generateContent` | Gemini             | Gemini CLI      |
| `GET /healthz`                    | Health check       | Monitoring      |

## Supported Agents

| Agent                                                         | Command        | What it is               |
|---------------------------------------------------------------|----------------|--------------------------|
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | `kitty claude` | Anthropic's coding agent |
| [Codex CLI](https://github.com/openai/codex)                  | `kitty codex`  | OpenAI's coding agent    |
| [Gemini CLI](https://github.com/google-gemini/gemini-cli)     | `kitty gemini` | Google's coding agent    |
| [Kilo Code](https://github.com/Kilo-Org/kilocode)             | `kitty kilo`   | Open-source coding agent |

## Supported Providers

**Regular API Key** — sign up, get an API key, pay per token:

| Provider         | Type ID           | Notes                                                                                    |
|------------------|-------------------|------------------------------------------------------------------------------------------|
| Anthropic        | `anthropic`       | Direct API only (pay per token). Subscription plans (Claude Pro/Team) are not supported. |
| AWS Bedrock      | `bedrock`         | Uses boto3 SigV4 auth                                                                    |
| MS Azure         | `azure`           | Requires deployment name                                                                 |
| BytePlus         | `byteplus`        |                                                                                          |
| Google AI Studio | `google_aistudio` | Gemini models via OpenAI-compatible endpoint                                             |
| Google Vertex AI | `vertex`          | Requires project and location                                                            |
| MiniMax          | `minimax`         |                                                                                          |
| OpenAI           | `openai`          | Direct API (pay per token). For your ChatGPT subscription, use the plan below.           |
| OpenRouter       | `openrouter`      | Multi-provider router                                                                    |
| Z.AI             | `zai_regular`     | General-purpose endpoint                                                                 |

**Coding Plans / Subscriptions** — use your existing subscription or coding plan, no API key needed:

| Provider            | Type ID               | Notes                                             |
|---------------------|-----------------------|---------------------------------------------------|
| Fireworks FirePass  | `fireworks`           |                                                   |
| Kimi Code           | `kimi`                |                                                   |
| Novita AI           | `novita`              |                                                   |
| Ollama Cloud        | `ollama_cloud`        | Hosted models via ollama.com API                  |
| OpenAI ChatGPT Plan | `openai_subscription` | Uses your ChatGPT Plus/Pro subscription via OAuth |
| OpenCode Go         | `opencode_go`         |                                                   |
| Xiaomi MiMo         | `mimo`                |                                                   |
| Z.AI Coding Plan    | `zai_coding`          | Coding-optimized endpoint                         |

**Local LLMs:**

| Provider | Type ID  | Notes                |
|----------|----------|----------------------|
| Ollama   | `ollama` | Local LLM deployment |

**Generic:**

| Provider                     | Type ID         | Notes                                                          |
|------------------------------|-----------------|----------------------------------------------------------------|
| **Custom OpenAI-Compatible** | `custom_openai` | Any service with a `/v1/chat/completions` endpoint — see below |

### Custom OpenAI-Compatible Provider

Use the `custom_openai` provider to connect to **any** service that exposes an OpenAI-compatible Chat Completions API.
This works with DeepSeek, Together AI, Groq, vLLM, LM Studio, and any other service that accepts
`POST /v1/chat/completions` with Bearer auth and SSE streaming.

```bash
$ kitty setup
  ? Provider: Custom OpenAI-Compatible
  ? API base URL: https://api.deepseek.com/v1
  ? Model: deepseek-chat
  ? API key: ********

$ kitty claude
  ✓ Bridge running on port <random_port>
  ✓ Claude Code launched
```

**Common endpoints:**

| Service      | Base URL                                |
|--------------|-----------------------------------------|
| DeepSeek     | `https://api.deepseek.com/v1`           |
| Together AI  | `https://api.together.xyz/v1`           |
| Groq         | `https://api.groq.com/openai/v1`        |
| Fireworks    | `https://api.fireworks.ai/inference/v1` |
| vLLM (local) | `http://localhost:8000/v1`              |
| LM Studio    | `http://localhost:1234/v1`              |

Both HTTPS and HTTP (local) endpoints are supported.

## Commands

| Command                                      | Description                                               |
|----------------------------------------------|-----------------------------------------------------------|
| `kitty setup`                                | Create your first profile (interactive wizard)            |
| `kitty profile`                              | Manage profiles (create, edit, delete, set default, list) |
| `kitty doctor`                               | Diagnose installation and configuration issues            |
| `kitty cleanup`                              | Restore agent config files after a crash                  |
| `kitty bridge`                               | Start a standalone API server                             |
| `kitty claude`                               | Launch Claude Code with default profile                   |
| `kitty codex`                                | Launch Codex with default profile                         |
| `kitty gemini`                               | Launch Gemini CLI with default profile                    |
| `kitty kilo`                                 | Launch Kilo Code with default profile                     |
| `kitty <profile> <agent>`                    | Launch an agent with a specific profile                   |
| `kitty <profile> bridge`                     | Start bridge with a specific profile                      |
| `kitty --no-validate <profile> <agent>`      | Skip API key validation                                   |
| `kitty --debug <profile> <agent>`            | Enable debug logging to `~/.cache/kitty/bridge.log`       |
| `kitty --debug-file /path <profile> <agent>` | Write debug logs to a custom path                         |
| `kitty --logging <profile> <agent>`          | Enable token usage logging to `~/.cache/kitty/usage.log`  |
| `kitty --log-file /path <profile> <agent>`   | Write usage logs to a custom path (implies `--logging`)   |
| `kitty --version`                            | Print version                                             |
| `kitty --help`                               | Print help                                                |

### Updating

```bash
pip install --upgrade kitty-bridge
```

## Technical Details

### How it works

Kitty sits between your coding agent and the upstream LLM provider. The high-level flow is the same as shown above:

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

Profile names must be 1-32 characters, lowercase letters, numbers, dashes, or underscores. Reserved words like `setup`,
`claude`, `codex`, `gemini`, `kilo`, `profile`, `bridge` cannot be used as profile names.

**Things to know about profile management:**

- Deleting a regular profile automatically removes it from all balancing profiles. If a balancing profile drops below 2
  members, it is deleted entirely.
- Deleting the default profile automatically promotes the first remaining profile as the new default.
- Editing a profile's API key creates a new credential entry. Other profiles sharing the old key are not affected.

### Pre-flight validation

Before launching, kitty checks that your profile configuration is valid and that your credentials can be resolved. If
something is wrong, you get a clear error immediately — not a cryptic failure inside the agent.

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

| Flag        | What it logs             | Default path                | Custom path flag    |
|-------------|--------------------------|-----------------------------|---------------------|
| `--logging` | Token usage              | `~/.cache/kitty/usage.log`  | `--log-file PATH`   |
| `--debug`   | Request/response tracing | `~/.cache/kitty/bridge.log` | `--debug-file PATH` |

### Cleanup

kitty restores agent config files after the agent exits. Three layers of cleanup:

1. **Normal exit** — `finally` block
2. **Crash / `SIGTERM`** — `atexit` handler
3. **`SIGKILL` / kernel OOM** — run `kitty cleanup` manually

If your agent shows connection errors after a crash, run `kitty cleanup` to restore its configuration files.

### Troubleshooting

Run `kitty doctor` to check your installation. It verifies that:

- Agent binaries are installed and discoverable
- A default profile exists
- All profile credentials can be resolved

For deeper issues, use the logging flags:

```bash
kitty --debug my-profile claude          # trace requests/responses to ~/.cache/kitty/bridge.log
kitty --logging my-profile claude        # log token usage to ~/.cache/kitty/usage.log
kitty --debug --log-file /tmp/usage.log my-profile claude  # both, with custom paths
```

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

The agent is trying to connect to a bridge that isn't running. Usually caused by a stale config from a previous crashed
session:

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

Yes, but with caveats. Cursor uses a proprietary protocol that Kitty cannot integrate with automatically. However, you
can start Kitty in bridge mode and point your IDE's "OpenAI base URL" setting at the bridge endpoint:

```bash
kitty bridge
# Then configure your IDE to use http://localhost:<port>/v1/chat/completions
```

This is a manual, best-effort configuration. Some IDE-specific features may not work.

### Can I use my Anthropic (Claude Pro/Team) subscription with Kitty?

No. Anthropic's Terms of Service prohibit accessing their subscription APIs from third-party software. You **can** use
Kitty with Anthropic's API directly — sign up at [console.anthropic.com](https://console.anthropic.com), create an API
key, and use the `anthropic` provider. You will be billed per token, not through your subscription.

### What is the difference between "OpenAI" and "OpenAI ChatGPT Plan"?

- **OpenAI** — standard developer API. You create an API key at [platform.openai.com](https://platform.openai.com) and
  pay per token.
- **OpenAI ChatGPT Plan** — uses your existing ChatGPT Plus or Pro subscription through a browser-based OAuth login. No
  API key needed; you use your subscription's included quota.

### What is a "coding plan"?

Some providers offer subscription plans specifically designed for coding agents. Instead of a traditional API key with
per-token billing, these plans typically authenticate via OAuth or a dedicated session and include usage quotas.
Examples include Z.AI Coding Plan, Fireworks FirePass, Novita AI, and Kimi Code.

### Can I run a local model?

Yes. Install [Ollama](https://ollama.ai), pull a model, then create a profile with provider `ollama`:

```bash
kitty setup
# Provider: ollama
# Base URL: http://localhost:11434/v1
# Model: llama3
```

### What is the difference between Ollama and Ollama Cloud?

- **Ollama** — connects to a local or remote Ollama instance using its OpenAI-compatible endpoint. No API key needed
  for local deployments.
- **Ollama Cloud** — connects to [ollama.com](https://ollama.com) cloud models using an API key. No local GPU required.
  Some models require a paid subscription.

### Does Kitty record my prompts or send data anywhere?

No. Kitty runs entirely on your machine. All prompts and responses pass directly between your coding agent and the
upstream provider. Kitty does not send data to third parties, store conversations, or collect telemetry.

### Something is broken. How do I debug it?

1. Run `kitty doctor` to check your installation and credentials
2. Run `kitty cleanup` if you see connection errors after a crash
3. Use `kitty --debug <profile> <agent>` to trace all requests and responses to `~/.cache/kitty/bridge.log`

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy src/kitty
```

## License

MIT
