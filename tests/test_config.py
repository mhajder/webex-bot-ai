"""Tests for configuration settings and validation in src/config.py."""

import importlib.util
import sys
from pathlib import Path

import dotenv
import pytest
from pydantic import ValidationError

from src.config import (
    _MCP_CONFIG_ERRORS,
    BotSettings,
    ConversationSettings,
    LLMSettings,
    MCPServerConfig,
    MCPSettings,
    Settings,
)

CONFIG_ENV_VARS = (
    "BOT_NAME",
    "BOT_DISPLAY_NAME",
    "WEBEX_ACCESS_TOKEN",
    "BOT_APPROVED_USERS",
    "BOT_APPROVED_DOMAINS",
    "BOT_APPROVED_ROOMS",
    "WEBEX_APPROVED_USERS",
    "WEBEX_APPROVED_DOMAINS",
    "WEBEX_APPROVED_ROOMS",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_MAX_TOKENS",
    "LLM_API_BASE",
    "LLM_API_KEY",
    "LLM_SYSTEM_PROMPT",
    "MCP_ENABLED",
    "MCP_REQUEST_TIMEOUT",
    "MCP_SERVERS",
    "CONVERSATION_MAX_HISTORY_MESSAGES",
    "CONVERSATION_TIMEOUT_HOURS",
    "CONVERSATION_RETENTION_DAYS",
    "CONVERSATION_DB_PATH",
    "CONVERSATION_ENABLE_PERSISTENCE",
    "CONVERSATION_ENABLE_SUMMARIZATION",
    "CONVERSATION_SUMMARY_THRESHOLD",
    "CONVERSATION_KEEP_RECENT_MESSAGES",
    "CONVERSATION_SUMMARY_MODEL",
    "CONVERSATION_SUMMARY_API_KEY",
    "CONVERSATION_SUMMARY_API_BASE",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "AZURE_API_KEY",
    "COHERE_API_KEY",
    "GROQ_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all configuration environment variables (including any loaded from .env)."""
    for var in CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    _MCP_CONFIG_ERRORS.clear()


def test_bot_settings_defaults_and_env_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = BotSettings()
    assert bot.name == "Assistant"
    assert bot.display_name == "AI Assistant"
    assert bot.webex_access_token == ""
    assert bot.approved_users == []
    assert bot.approved_domains == []
    assert bot.approved_rooms == []

    monkeypatch.setenv("BOT_NAME", "NetBot")
    monkeypatch.setenv("BOT_DISPLAY_NAME", "Network Bot")
    monkeypatch.setenv("WEBEX_ACCESS_TOKEN", "webex-token")
    monkeypatch.setenv("WEBEX_APPROVED_USERS", " a@b.com , , c@d.com ")
    monkeypatch.setenv("WEBEX_APPROVED_DOMAINS", "example.com, example.org ")
    monkeypatch.setenv("WEBEX_APPROVED_ROOMS", "room1,room2, ")

    configured = BotSettings()
    assert configured.name == "NetBot"
    assert configured.display_name == "Network Bot"
    assert configured.webex_access_token == "webex-token"  # noqa: S105
    assert configured.approved_users == ["a@b.com", "c@d.com"]
    assert configured.approved_domains == ["example.com", "example.org"]
    assert configured.approved_rooms == ["room1", "room2"]


def test_bot_settings_explicit_lists_not_overwritten_by_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBEX_APPROVED_USERS", "env@example.com")
    monkeypatch.setenv("WEBEX_APPROVED_DOMAINS", "env.example.com")
    monkeypatch.setenv("WEBEX_APPROVED_ROOMS", "env-room")

    bot = BotSettings(
        approved_users=["explicit@example.com"],
        approved_domains=["explicit.com"],
        approved_rooms=["explicit-room"],
    )
    assert bot.approved_users == ["explicit@example.com"]
    assert bot.approved_domains == ["explicit.com"]
    assert bot.approved_rooms == ["explicit-room"]


def test_llm_settings_defaults_bounds_and_system_prompt() -> None:
    llm = LLMSettings()
    assert llm.model == "gpt-4o-mini"
    assert llm.temperature == 0.7
    assert llm.max_tokens == 2048
    assert llm.api_base is None
    assert llm.api_key is None

    prompt = llm.get_system_prompt("NetBot")
    assert "You are NetBot" in prompt
    assert 'Your name is "NetBot"' in prompt
    assert '{"name": "tool_name", "arguments":' in prompt

    with pytest.raises(ValidationError):
        LLMSettings(temperature=2.5)
    with pytest.raises(ValidationError):
        LLMSettings(max_tokens=0)


def test_mcp_settings_parsing_and_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_SERVERS", "")
    empty_mcp = MCPSettings()
    assert empty_mcp.servers == []
    assert empty_mcp.enabled_servers == []
    assert _MCP_CONFIG_ERRORS == []

    monkeypatch.setenv(
        "MCP_SERVERS",
        '[{"name": "s1", "url": "http://localhost:8000/mcp", "enabled": true}, '
        '{"name": "s2", "url": "http://localhost:8001/mcp", "enabled": false}]',
    )
    valid_mcp = MCPSettings()
    assert len(valid_mcp.servers) == 2
    assert [s.name for s in valid_mcp.enabled_servers] == ["s1"]
    assert _MCP_CONFIG_ERRORS == []


def test_mcp_settings_invalid_json_and_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_SERVERS", "{not-json")
    bad_json = MCPSettings()
    assert bad_json.servers == []
    assert any("not valid JSON" in err for err in _MCP_CONFIG_ERRORS)

    _MCP_CONFIG_ERRORS.clear()
    monkeypatch.setenv("MCP_SERVERS", '{"name": "dict-not-list"}')
    not_list = MCPSettings()
    assert not_list.servers == []
    assert any("must be a JSON list" in err for err in _MCP_CONFIG_ERRORS)

    _MCP_CONFIG_ERRORS.clear()
    monkeypatch.setenv(
        "MCP_SERVERS",
        '[{"name": "good", "url": "http://ok/mcp"}, {"name": "bad", "auth_token": "secret-token"}]',
    )
    partial = MCPSettings()
    assert len(partial.servers) == 1
    assert partial.servers[0].name == "good"
    assert len(_MCP_CONFIG_ERRORS) == 1
    assert "entry 1 skipped" in _MCP_CONFIG_ERRORS[0]
    assert "url" in _MCP_CONFIG_ERRORS[0]
    assert "secret-token" not in _MCP_CONFIG_ERRORS[0]


def test_mcp_server_config_defaults() -> None:
    cfg = MCPServerConfig(name="srv", url="http://localhost:9000/mcp")
    assert cfg.enabled is True
    assert cfg.headers == {}
    assert cfg.auth_token is None


def test_conversation_settings_defaults_and_timeout_seconds() -> None:
    conv = ConversationSettings(timeout_hours=2)
    assert conv.timeout_seconds == 7200
    assert conv.retention_days == 30
    assert conv.summary_threshold == 50
    assert conv.keep_recent_messages == 20

    with pytest.raises(ValidationError):
        ConversationSettings(summary_threshold=1)
    with pytest.raises(ValidationError):
        ConversationSettings(retention_days=-1)


def test_settings_validate_config_token_and_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_missing = Settings()
    errors = settings_missing.validate_config()
    assert "WEBEX_ACCESS_TOKEN is required" in errors
    assert any("API key required" in e for e in errors)

    monkeypatch.setenv("WEBEX_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("LLM_MODEL", "ollama_chat/llama3.2")
    assert Settings().validate_config() == []

    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_API_KEY", "custom-key")
    assert Settings().validate_config() == []
    monkeypatch.delenv("LLM_API_KEY")

    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-2.5-flash")
    assert len(Settings().validate_config()) == 1
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    assert Settings().validate_config() == []
    monkeypatch.delenv("GEMINI_API_KEY")

    monkeypatch.setenv("LLM_MODEL", "unknown-provider/my-model")
    assert len(Settings().validate_config()) == 1
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    assert Settings().validate_config() == []


def test_settings_config_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBEX_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-2.5-flash")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("MCP_SERVERS", "{bad")

    cfg = Settings()
    assert cfg.validate_config() == []
    # Duplicate MCP error should be deduplicated
    _MCP_CONFIG_ERRORS.append(_MCP_CONFIG_ERRORS[0])
    warnings = cfg.config_warnings()
    assert len(warnings) == 2
    assert any("not valid JSON" in w for w in warnings)
    assert any(
        "GEMINI_API_KEY" in w and "only OPENAI_API_KEY is set" in w for w in warnings
    )

    # Providing GEMINI_API_KEY or LLM_API_KEY clears the provider mismatch warning
    monkeypatch.delenv("MCP_SERVERS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    _MCP_CONFIG_ERRORS.clear()
    assert Settings().config_warnings() == []

    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("LLM_API_KEY", "proxy-key")
    assert Settings().config_warnings() == []


def test_config_module_reload_without_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: False)
    monkeypatch.setenv("WEBEX_ACCESS_TOKEN", "reloaded-token")
    config_file = Path(sys.modules["src.config"].__file__ or "")
    spec = importlib.util.spec_from_file_location("_isolated_src_config", config_file)
    assert spec is not None
    assert spec.loader is not None
    isolated_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated_mod)
    assert isolated_mod.settings.bot.webex_access_token == "reloaded-token"  # noqa: S105
