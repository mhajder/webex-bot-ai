"""Configuration module for the Webex AI Bot.

Uses pydantic-settings for type-safe configuration with environment variable support.
Supports multiple LLM providers via LiteLLM and multiple MCP servers.
"""

import json
import os
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    """Bot identity and Webex connection settings."""

    model_config = SettingsConfigDict(env_prefix="BOT_", extra="ignore")

    # Bot identity
    name: str = Field(default="Assistant", description="Bot name for mention handling")
    display_name: str = Field(
        default="AI Assistant", description="Display name shown in Webex"
    )

    # Webex settings
    webex_access_token: str = Field(
        default="",
        description="Webex bot access token",
        validation_alias="WEBEX_ACCESS_TOKEN",
    )

    # Access control (comma-separated in env)
    approved_users: list[str] = Field(
        default_factory=list, description="List of approved user emails"
    )
    approved_domains: list[str] = Field(
        default_factory=list, description="List of approved email domains"
    )
    approved_rooms: list[str] = Field(
        default_factory=list, description="List of approved room IDs"
    )

    def model_post_init(self, __context: Any) -> None:
        """Parse comma-separated lists from environment variables."""
        # Parse approved users
        if not self.approved_users:
            users_env = os.getenv("WEBEX_APPROVED_USERS", "")
            if users_env:
                self.approved_users = [
                    u.strip() for u in users_env.split(",") if u.strip()
                ]

        # Parse approved domains
        if not self.approved_domains:
            domains_env = os.getenv("WEBEX_APPROVED_DOMAINS", "")
            if domains_env:
                self.approved_domains = [
                    d.strip() for d in domains_env.split(",") if d.strip()
                ]

        # Parse approved rooms
        if not self.approved_rooms:
            rooms_env = os.getenv("WEBEX_APPROVED_ROOMS", "")
            if rooms_env:
                self.approved_rooms = [
                    r.strip() for r in rooms_env.split(",") if r.strip()
                ]


class LLMSettings(BaseSettings):
    """LLM/AI provider settings using LiteLLM."""

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    # Model configuration (LiteLLM format: provider/model)
    # Examples: "gpt-4o-mini", "ollama_chat/gpt-oss:120b", "openrouter/meta-llama/llama-3.1-8b"
    model: str = Field(
        default="gpt-4o-mini",
        description="LiteLLM model identifier (e.g., gpt-4o, ollama_chat/gpt-oss:120b)",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)

    # API settings (LiteLLM uses provider-specific env vars)
    api_base: str | None = Field(
        default=None, description="Custom API base URL for the provider"
    )
    api_key: str | None = Field(
        default=None,
        description="API key (if not using provider-specific env var)",
        validation_alias="OPENAI_API_KEY",
    )

    # System prompt template
    system_prompt: str = Field(
        default="""You are {bot_name}, an expert AI assistant in Webex designed to help users by providing accurate, data-driven responses through available tools and integrations.

CORE MISSION:
Your primary goal is to provide clear, concise, and accurate responses derived exclusively from data retrieved through available tools. When tools can provide information, use them instead of guessing.

INTERACTION WORKFLOW:
1. **Identify Intent:** Determine what specific information the user is seeking and understand the scope of their request.
2. **Leverage Tools Strategically:** Translate user requests into precise tool calls to fetch relevant data. If multiple steps are needed, use them logically.
3. **Process & Validate Data:** Parse responses from tools, extract key information, validate its relevance, and consolidate into a coherent summary.
4. **Present Clearly:** Format data using tables, bullet points, or clear paragraphs as appropriate. Provide context and explain what the data means.

CRITICAL TOOL CALLING RULES:
5. TOOL CALLING FORMAT: When you need to call a tool, respond ONLY with VALID JSON in this EXACT format:
   {{"name": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}

   MANDATORY REQUIREMENTS:
   - The JSON must have exactly TWO top-level keys: "name" and "arguments"
   - "name" must be a STRING with the tool name
   - "arguments" must be an OBJECT (dict) containing the parameters for that tool
   - If the tool takes no parameters, "arguments" should be an empty object: {{}}
   - Do NOT include extra keys or wrap the JSON in any other structure
   - Do NOT include any text before or after the JSON

6. THE EXACT WORKFLOW YOU MUST FOLLOW:
   STEP 1: User asks a question
   STEP 2: IF the question requires data, call ONE tool with JSON format above
   STEP 3: You receive tool results
   STEP 4: PROCESS the results and respond with plain text/markdown summary
   STEP 5: DO NOT call any more tools after step 3

9. AFTER RECEIVING TOOL RESULTS - CRITICAL RULE: When you receive tool results from a tool call, STOP calling tools immediately. Do NOT attempt to call 'final_response', 'summarize', or any other tool. Proceed directly to responding with plain text/markdown that explains the results to the user. Your response text IS the final answer - no additional tool calls are needed or permitted.

PROFESSIONAL GUIDELINES:
9. **Be Specific & Clarify:** If a query is ambiguous or lacks necessary parameters, ask proactively for clarification before making assumptions.
10. **Prioritize Accuracy:** When available, provide data from tools rather than relying on general knowledge. Clearly distinguish between tool-derived facts and contextual information.
11. **Provide Context:** Explain what data means, provide relevant context, and use appropriate units or formatting for numerical data.
12. **Suggest Next Steps:** For troubleshooting or analysis queries, suggest logical follow-up questions or related information the user might find useful.
13. **Handle Errors Gracefully:** If a tool call fails or returns no data, inform the user clearly. Explain potential reasons and suggest alternative approaches.
14. **State Limitations Clearly:** If you cannot fulfill a request, be honest about limitations and avoid fabricating data.
15. **Tool Selection Strategy:** ALWAYS prioritize read-only/GET tools over tools that modify, create, update, or delete data. Only use write/change tools when the user EXPLICITLY and CLEARLY requests to modify, create, update, or delete something. If a user query is ambiguous about modification (e.g., "update config"), ask for explicit confirmation before using any write tools.
16. **Maintain Read-Only Default:** Unless explicitly and unambiguously authorized by the user, assume your role is strictly to retrieve and present information, never to modify systems or data.
17. **No Code Generation:** Never generate scripts, code blocks, or programming code in your responses.
18. **Professional Tone:** Respond helpfully, professionally, and knowledgeably.

CONVERSATION MANAGEMENT:
19. Your name is "{bot_name}". When users mention you by name (e.g., "@{bot_name}"), they are addressing you - this is NOT a question about your name.
20. Focus on answering the user's actual question, ignoring the @mention of your name.
21. Remember the conversation history in this thread to provide contextual responses.
22. Format responses nicely for Webex using markdown when appropriate.

RESPONSE PROTOCOL:
23. FINAL RESPONSES: After receiving tool results or providing a regular response, ALWAYS respond with readable text/markdown, NEVER with JSON. Do NOT attempt any additional tool calls. Simply provide your answer as text.""",
        description="System prompt template with {bot_name} placeholder",
    )

    def get_system_prompt(self, bot_name: str) -> str:
        """Generate system prompt with bot name substituted."""
        return self.system_prompt.format(bot_name=bot_name)


class MCPServerConfig(BaseSettings):
    """Configuration for a single MCP server."""

    name: str = Field(description="Unique identifier for this MCP server")
    url: str = Field(description="HTTP URL for the MCP server")
    enabled: bool = Field(default=True)
    headers: dict[str, str] = Field(
        default_factory=dict, description="Custom headers for this server"
    )
    auth_token: str | None = Field(
        default=None, description="Bearer token for authentication"
    )


class MCPSettings(BaseSettings):
    """MCP (Model Context Protocol) settings for tool integration."""

    model_config = SettingsConfigDict(env_prefix="MCP_", extra="ignore")

    enabled: bool = Field(default=False, description="Enable MCP integration")
    request_timeout: int = Field(
        default=30, ge=1, description="Request timeout in seconds"
    )

    # Servers configuration (JSON string in env var)
    # Format: [{"name": "server1", "url": "http://...", "enabled": true}, ...]
    servers: list[MCPServerConfig] = Field(
        default_factory=list, description="List of MCP server configurations"
    )

    def model_post_init(self, __context: Any) -> None:
        """Parse MCP servers from environment variable."""
        if not self.servers:
            servers_json = os.getenv("MCP_SERVERS", "")
            if servers_json:
                try:
                    servers_data = json.loads(servers_json)
                    self.servers = [MCPServerConfig(**s) for s in servers_data]
                except json.JSONDecodeError:
                    pass

    @property
    def enabled_servers(self) -> list[MCPServerConfig]:
        """Get list of enabled MCP servers."""
        return [s for s in self.servers if s.enabled]


class ConversationSettings(BaseSettings):
    """Conversation history management settings."""

    model_config = SettingsConfigDict(env_prefix="CONVERSATION_", extra="ignore")

    max_history_messages: int = Field(
        default=50, ge=1, description="Maximum messages to keep per thread"
    )
    timeout_hours: int = Field(
        default=24, ge=1, description="Hours before conversation is considered stale"
    )
    db_path: str = Field(
        default="conversations.db", description="Path to SQLite database file"
    )
    enable_persistence: bool = Field(
        default=True, description="Whether to persist conversations to database"
    )

    @property
    def timeout_seconds(self) -> int:
        """Get timeout in seconds."""
        return self.timeout_hours * 3600


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""

    bot: BotSettings = Field(default_factory=BotSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.bot.webex_access_token:
            errors.append("WEBEX_ACCESS_TOKEN is required")

        # Check if at least one API key is configured for the model
        model = self.llm.model.lower()

        # Ollama doesn't need API key for local models
        if model.startswith("ollama"):
            return errors

        # Check for provider-specific API keys
        provider_keys = {
            "openai": "OPENAI_API_KEY",
            "gpt": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "azure": "AZURE_API_KEY",
            "cohere": "COHERE_API_KEY",
            "groq": "GROQ_API_KEY",
        }

        # Check which provider the model uses
        api_key_found = False

        # Check generic keys first
        if self.llm.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"):
            api_key_found = True
        else:
            # Check provider-specific keys
            for provider_prefix, env_key in provider_keys.items():
                if model.startswith(provider_prefix) and os.getenv(env_key):
                    api_key_found = True
                    break

            # If no provider prefix matched, check all provider-specific keys
            if not api_key_found:
                for env_key in provider_keys.values():
                    if os.getenv(env_key):
                        api_key_found = True
                        break

        if not api_key_found:
            errors.append(
                "API key required for the configured model. "
                "Set OPENAI_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY, "
                "COHERE_API_KEY, GROQ_API_KEY, AZURE_API_KEY, or LLM_API_KEY."
            )

        return errors


# Global settings instance
settings = Settings()
