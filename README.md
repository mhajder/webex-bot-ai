<div align="center">

# Webex Bot - AI Assistant

**Bring any LLM and your MCP tools into Webex spaces, with thread-aware conversations**

[![PyPI](https://img.shields.io/pypi/v/webex-bot-ai?logo=pypi&logoColor=white)](https://pypi.org/project/webex-bot-ai/)
[![Python](https://img.shields.io/pypi/pyversions/webex-bot-ai?logo=python&logoColor=white)](https://pypi.org/project/webex-bot-ai/)
[![License](https://img.shields.io/github/license/mhajder/webex-bot-ai)](https://github.com/mhajder/webex-bot-ai/blob/main/LICENSE)

[![Pre-commit Checks](https://github.com/mhajder/webex-bot-ai/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/mhajder/webex-bot-ai/actions/workflows/pre-commit.yml)
[![Docker](https://img.shields.io/badge/ghcr.io-webex--bot--ai-2496ED?logo=docker&logoColor=white)](https://github.com/mhajder/webex-bot-ai/pkgs/container/webex-bot-ai)
[![published](https://static.production.devnetcloud.com/codeexchange/assets/images/devnet-published.svg)](https://developer.cisco.com/codeexchange/github/repo/mhajder/webex-bot-ai)

[Quick start](https://github.com/mhajder/webex-bot-ai#-quick-start) · [Configuration](https://github.com/mhajder/webex-bot-ai#-configuration) · [MCP integration](https://github.com/mhajder/webex-bot-ai#mcp-integration) · [Usage](https://github.com/mhajder/webex-bot-ai#-usage)

</div>

A conversational AI bot for Webex that:

- 🤖 Responds to user mentions with AI-generated answers
- 💬 Maintains conversation context within threads
- 🔧 Integrates with MCP (Model Context Protocol) servers for extended capabilities
- 🌐 Supports multiple LLM providers via LiteLLM (OpenAI, Google Gemini, Ollama, OpenRouter, Anthropic, etc.)

## ✨ Features

- **Thread-aware conversations**: The bot remembers context within Webex threads, allowing natural follow-up questions
- **Rolling thread summarization**: Background tasks summarize older messages as threads grow, preserving context indefinitely without hitting context limits
- **Room-wide history search**: Uses SQLite FTS5 keyword search to look up past messages, solutions, and context across Webex space history
- **Smart mention handling**: The bot recognizes its own name and doesn't confuse it with questions
- **Multiple LLM providers**: Use OpenAI, Google Gemini, Ollama (local/cloud), OpenRouter, Anthropic, or any LiteLLM-supported provider
- **MCP Integration**: Connect to multiple MCP servers via HTTP for extended tool capabilities
- **Access control**: Restrict bot to approved users, domains, or rooms
- **Clean code**: Follows Python best practices with ruff linting and formatting

## 🚀 Quick Start

### Prerequisites

- Python 3.11 to 3.14
- [UV](https://docs.astral.sh/uv/) package manager
- Webex bot token ([create one here](https://developer.webex.com/my-apps))
- API key for your LLM provider (e.g., OpenAI, Gemini)

### Installation

#### Option 1: Install from PyPI

Run the package directly from PyPI using UVX:

```bash
uvx webex-bot-ai
```

#### Option 2: Local Development Setup

For development or running from source:

1. Install dependencies:

```bash
git clone https://github.com/mhajder/webex-bot-ai.git
cd webex-bot-ai
uv sync
```

#### Option 3: Run with Docker (Persistent Storage)

Run in a container with database volume persistence:

```bash
docker run -d \
  --name webex-bot-ai \
  --env-file .env \
  -v webex_bot_data:/app/data \
  ghcr.io/mhajder/webex-bot-ai:latest
```

### Configuration

1. Configure environment:

```bash
cp .env.example .env
# Edit .env with your configuration
```

2. Set required variables in `.env`:

```env
WEBEX_ACCESS_TOKEN=your_webex_bot_token
OPENAI_API_KEY=your_openai_api_key
```

3. Run the bot:

```bash
webex-bot-ai
```

## 🔧 Configuration

### Bot Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `WEBEX_ACCESS_TOKEN` | Webex bot access token (required) | - |
| `BOT_NAME` | Bot name for mention handling | `Assistant` |
| `BOT_DISPLAY_NAME` | Display name in Webex | `AI Assistant` |

### LLM Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_MODEL` | LiteLLM model identifier | `gpt-4o-mini` |
| `LLM_TEMPERATURE` | Sampling temperature (0.0-2.0) | `0.7` |
| `LLM_MAX_TOKENS` | Maximum response tokens | `2048` |
| `LLM_API_BASE` | Custom API endpoint | - |
| `LLM_API_KEY` | API key passed to LiteLLM; when unset, the provider's own variable (e.g. `OPENAI_API_KEY`) is used | - |

### Conversation Settings & Rolling Summarization

| Variable | Description | Default |
|----------|-------------|---------|
| `CONVERSATION_ENABLE_PERSISTENCE` | Persist conversation threads to SQLite database | `true` |
| `CONVERSATION_ENABLE_SUMMARIZATION` | Enable rolling thread summarization for long threads | `true` |
| `CONVERSATION_SUMMARY_THRESHOLD` | Message count threshold to trigger rolling summarization | `50` |
| `CONVERSATION_KEEP_RECENT_MESSAGES` | Number of recent messages retained verbatim after summary | `20` |
| `CONVERSATION_SUMMARY_MODEL` | Optional distinct LLM model for thread summarization | Same as `LLM_MODEL` |
| `CONVERSATION_SUMMARY_API_KEY` | API key for the summary model | `LLM_API_KEY` if same provider as `LLM_MODEL`, else the provider's own env var |
| `CONVERSATION_SUMMARY_API_BASE` | API base URL for the summary model | `LLM_API_BASE` if same provider as `LLM_MODEL` |
| `CONVERSATION_MAX_HISTORY_MESSAGES` | Maximum in-memory history messages | `50` |
| `CONVERSATION_TIMEOUT_HOURS` | Hours before a thread is considered stale | `24` |
| `CONVERSATION_RETENTION_DAYS` | Days to keep inactive threads in the database (`0` = forever); cleanup runs at startup and daily | `30` |
| `CONVERSATION_DB_PATH` | SQLite database file path | `conversations.db` |

### Model Examples

```env
# OpenAI
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# Google Gemini
LLM_MODEL=gemini/gemini-2.5-flash
GEMINI_API_KEY=AIzaSy...

# Ollama (local)
LLM_MODEL=ollama_chat/gpt-oss:120b
LLM_API_BASE=http://localhost:11434

# OpenRouter
LLM_MODEL=openrouter/meta-llama/llama-3.1-70b-instruct
OPENROUTER_API_KEY=sk-or-...

# Anthropic
LLM_MODEL=claude-3-sonnet-20240229
ANTHROPIC_API_KEY=sk-ant-...
```

### Access Control

```env
# Restrict to specific users
WEBEX_APPROVED_USERS=user1@example.com,user2@example.com

# Restrict to specific email domains
WEBEX_APPROVED_DOMAINS=example.com

# Restrict to specific rooms
WEBEX_APPROVED_ROOMS=room_id_1,room_id_2
```

### MCP Integration

Connect to MCP HTTP transport servers for extended tool capabilities:

```env
MCP_ENABLED=true
MCP_REQUEST_TIMEOUT=30

# Single server
MCP_SERVERS=[{"name": "my-server", "url": "http://localhost:8000/mcp", "enabled": true}]

# Multiple servers with auth
MCP_SERVERS='[
  {"name": "tools-server", "url": "http://localhost:8000/mcp", "enabled": true},
  {"name": "secure-server", "url": "https://api.example.com/mcp", "auth_token": "your-token", "enabled": true}
]'
```

### Sentry Error Tracking (Optional)

Enable error tracking and performance monitoring with Sentry:

```bash
# Install with Sentry support
uv sync --extra sentry
```

Configure Sentry via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `SENTRY_DSN` | Sentry DSN (enables Sentry when set) | - |
| `SENTRY_TRACES_SAMPLE_RATE` | Trace sampling rate (0.0-1.0) | `0.1` |
| `SENTRY_SEND_DEFAULT_PII` | Include PII in events: user emails/names, LLM prompts and responses, INFO log breadcrumbs | `false` |
| `SENTRY_ENVIRONMENT` | Environment name (e.g., `production`) | - |
| `SENTRY_RELEASE` | Release/version identifier | Package version |
| `SENTRY_PROFILE_SESSION_SAMPLE_RATE` | Profile session sampling rate | `0.1` |
| `SENTRY_PROFILE_LIFECYCLE` | Profile lifecycle mode | `trace` |
| `SENTRY_ENABLE_LOGS` | Enable logging integration | `true` |

Example configuration:

```env
# Enable Sentry error tracking
SENTRY_DSN=https://your-key@o12345.ingest.us.sentry.io/6789
SENTRY_ENVIRONMENT=production
```

## 💡 Usage

1. **Start a conversation**: Mention the bot in a Webex space:
   ```
   @BotName What is AI?
   ```

2. **Follow-up in thread**: Reply in the same thread for context-aware responses:
   ```
   @BotName Tell me more.
   ```

3. **The bot maintains context** within the thread, so you can have natural conversations.

## 🔨 Development

```bash
uv sync --group dev
uv run prek install

# Lint, format and type-check
uv run ruff check .
uv run ruff format .
uv run ty check .
```

### Adding New Features

- **Commands**: Add new commands in `src/commands/`
- **MCP Tools**: Connect to MCP servers via configuration
- **LLM Providers**: Configure via `LLM_MODEL` using [LiteLLM syntax](https://docs.litellm.ai/docs/providers)

## 📦 Dependencies

- [webex_bot](https://github.com/fbradyirl/webex_bot) - Webex bot framework
- [litellm](https://github.com/BerriAI/litellm) - Universal LLM API
- [fastmcp](https://github.com/jlowin/fastmcp) - MCP client/server framework
- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) - Configuration management
- [sentry-sdk](https://github.com/getsentry/sentry-python) - Error tracking (optional)

## 🤝 Contributing

Contributions are welcome. See [CONTRIBUTING.md](https://github.com/mhajder/webex-bot-ai/blob/main/CONTRIBUTING.md) for the dev setup, checks and commit message format, and use the [issue templates](https://github.com/mhajder/webex-bot-ai/issues/new/choose) for bugs and feature requests.

Found a security problem? Please report it privately as described in [SECURITY.md](https://github.com/mhajder/webex-bot-ai/blob/main/SECURITY.md).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
