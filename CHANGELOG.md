# Changelog

## [0.3.2](https://github.com/mhajder/webex-bot-ai/compare/v0.3.1...v0.3.2) (2026-11-12)


### 🧩 CI

* build release notes from CHANGELOG.md and fix the image tag ([c7c8842](https://github.com/mhajder/webex-bot-ai/commit/c7c8842aa27fb6ba0e8643f7a87268261f3b87d5))
* pin actions to commit SHAs at latest versions ([6ee0908](https://github.com/mhajder/webex-bot-ai/commit/6ee0908fbfcf24d8a6c6cf4622303bb4fa627968))
* release with release-please ([4d0936e](https://github.com/mhajder/webex-bot-ai/commit/4d0936eef77b98c4968fb677cf7a6e6b777ec3ec))
* replace softprops/action-gh-release with the gh CLI ([97ef054](https://github.com/mhajder/webex-bot-ai/commit/97ef054adb7b0433197bb66df693746bcbb6121d))


### 📦 Build

* **deps:** bump docker/login-action ([#12](https://github.com/mhajder/webex-bot-ai/issues/12)) ([5da0f10](https://github.com/mhajder/webex-bot-ai/commit/5da0f107a000ccdf3200b3dad64117ca21b5fba3))
* **deps:** bump ty to 0.0.98 ([53cb7b6](https://github.com/mhajder/webex-bot-ai/commit/53cb7b6117b34910b270375a130a14cea2967524))
* **deps:** raise dependency floors past known advisories and bump dev tools ([d954b5c](https://github.com/mhajder/webex-bot-ai/commit/d954b5ca69c579261c637c4de7027b8fb7d8e1f2))
* **docker:** pin base images by version and digest and update them with Dependabot ([e085633](https://github.com/mhajder/webex-bot-ai/commit/e0856339221b71a211f4a44a933e0a4e25241043))
* fix ty check hook and type check for all platforms ([adba272](https://github.com/mhajder/webex-bot-ai/commit/adba272d39788673a2ed56caeff3021ccca4bdec))

## v0.3.1 (2026-10-20)

### Fix

- **sentry**: stop sending PII and full traces by default
- **main**: use the Webex display name and schedule DB cleanup
- **ai**: fix tool calling, Webex history sync and prompt handling
- **mcp**: keep tool names intact and bound call timeouts
- **conversation**: make summarization, timeout and search work
- **config**: load .env before settings and parse settings leniently
- **docker**: run as numeric UID 10001 with root-owned app files
- satisfy ty with the updated litellm typings

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
