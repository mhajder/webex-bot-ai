## v0.3.0 (2026-07-26)

### Feat

- **conversation**: sync unmentioned Webex thread messages via Webex API
- **conversation**: implement multi-turn rolling thread summarization
- **memory**: add Webex room-wide chat history search using SQLite FTS5
- **config**: support GEMINI_API_KEY validation and update docs

### Fix

- **conversation**: resolve summarization deadlock, sql ordering, and tool argument parsing
- **llm**: suppress LiteLLM cache cost warnings for Ollama models
- **docker**: copy only .venv from builder
- **docker**: resolve litellm build failure on alpine

### Refactor

- extract helper methods and centralize sync DB operations

## v0.2.0 (2026-01-15)

### Feat

- add multi-arch Docker image build support

### Fix

- **sentry**: set user context from Webex activity for enhanced tracking
- **sentry**: add username parameter to set_user_context for enhanced user tracking
- **sentry**: suppress Pydantic serialization warnings in LiteLLM integration
- ensure new event loop when MCP is disabled
- **sentry**: enable log forwarding in Sentry init

### Refactor

- move MCP client to mcp_client package

## v0.1.0 (2026-01-01)

### Feat

- add webex-bot-ai
