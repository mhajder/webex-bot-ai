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
