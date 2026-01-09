"""Webex AI Bot - Main Entry Point.

A conversational AI bot for Webex that:
- Responds to user mentions with AI-generated answers
- Maintains conversation context within threads
- Integrates with MCP servers for extended capabilities
- Supports multiple LLM providers via LiteLLM
- Optional Sentry integration for error tracking
"""

import asyncio
import logging
import sys

from dotenv import load_dotenv
from webex_bot.webex_bot import WebexBot

from src.commands import AICommand
from src.config import settings
from src.conversation import ConversationManager
from src.mcp_client import MCPMultiClient
from src.sentry import capture_exception, init_sentry

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("webex_bot").setLevel(logging.WARNING)

log = logging.getLogger(__name__)


def validate_configuration() -> bool:
    """Validate that all required configuration is present."""
    errors = settings.validate_config()
    if errors:
        for error in errors:
            log.error("Configuration error: %s", error)
        return False
    return True


def create_bot() -> WebexBot:
    """Create and configure the Webex bot instance."""
    log.info("Creating bot with name: %s", settings.bot.display_name)

    bot_kwargs = {
        "teams_bot_token": settings.bot.webex_access_token,
        "bot_name": settings.bot.display_name,
        "include_demo_commands": False,
        "threads": True,
    }

    if settings.bot.approved_users:
        bot_kwargs["approved_users"] = settings.bot.approved_users
        log.info("Approved users: %s", settings.bot.approved_users)

    if settings.bot.approved_domains:
        bot_kwargs["approved_domains"] = settings.bot.approved_domains
        log.info("Approved domains: %s", settings.bot.approved_domains)

    if settings.bot.approved_rooms:
        bot_kwargs["approved_rooms"] = settings.bot.approved_rooms
        log.info("Approved rooms: %d room(s)", len(settings.bot.approved_rooms))

    return WebexBot(**bot_kwargs)


async def setup_mcp_servers() -> MCPMultiClient | None:
    """Set up MCP server connections if enabled.

    Returns:
        MCPMultiClient instance if any servers connected, None otherwise.
    """
    if not settings.mcp.enabled:
        log.info("MCP integration disabled")
        return None

    enabled_servers = settings.mcp.enabled_servers
    if not enabled_servers:
        log.info("MCP enabled but no servers configured")
        return None

    log.info("Initializing MCP connections to %d server(s)...", len(enabled_servers))

    mcp_client = MCPMultiClient(timeout=settings.mcp.request_timeout)

    connected_count = 0
    for server in enabled_servers:
        try:
            success = await mcp_client.connect(
                name=server.name,
                url=server.url,
                headers=server.headers,
                auth_token=server.auth_token,
            )
            if success:
                connected_count += 1
                log.info(
                    "Connected to MCP server '%s' with %d tools",
                    server.name,
                    len(mcp_client.get_server_tools(server.name)),
                )
        except Exception as e:
            log.exception("Failed to connect to MCP server '%s': %s", server.name, e)

    if connected_count == 0:
        log.warning("Failed to connect to any MCP servers")
        return None

    log.info(
        "MCP setup complete: %d/%d servers connected, %d total tools available",
        connected_count,
        len(enabled_servers),
        len(mcp_client.available_tools),
    )

    return mcp_client


def setup_commands(
    bot: WebexBot, mcp_client: MCPMultiClient | None = None
) -> ConversationManager:
    """Set up bot commands.

    Args:
        bot: The WebexBot instance.
        mcp_client: Optional MCPMultiClient for tool integration.

    Returns:
        The ConversationManager instance used by the bot.
    """
    conversation_manager = ConversationManager(
        max_messages=settings.conversation.max_history_messages,
        timeout_seconds=settings.conversation.timeout_seconds,
        db_path=settings.conversation.db_path,
        enable_persistence=settings.conversation.enable_persistence,
    )

    bot_name = settings.bot.name

    if bot_name.lower() == "assistant" and hasattr(bot, "teams_bot_email"):
        email_part = bot.teams_bot_email.split("@")[0].split("-")[0]
        bot_name = email_part.title()
        log.info("Extracted bot name from email: %s", bot_name)

    ai_command = AICommand(
        conversation_manager=conversation_manager,
        bot_name=bot_name,
        mcp_client=mcp_client,
    )

    bot.commands.clear()
    bot.add_command(ai_command)
    bot.help_command = ai_command

    log.info("AI command configured as primary handler")
    if mcp_client:
        log.info("MCP tools available: %d", len(mcp_client.available_tools))

    return conversation_manager


def main() -> None:
    """Main entry point for the bot."""
    # Initialize Sentry early (before any other setup)
    sentry_enabled = init_sentry()

    log.info("=" * 60)
    log.info("Starting Webex AI Bot")
    log.info("=" * 60)

    if sentry_enabled:
        log.info("Sentry error tracking: enabled")

    if not validate_configuration():
        log.error("Configuration validation failed. Exiting.")
        sys.exit(1)

    log.info("Bot name: %s", settings.bot.name)
    log.info("LLM model: %s", settings.llm.model)
    log.info("MCP enabled: %s", settings.mcp.enabled)

    mcp_client = None
    conversation_manager = None

    try:
        # For MCP setup, create an event loop but don't close it
        # webex_bot will need it for its websocket operations
        if settings.mcp.enabled:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            mcp_client = loop.run_until_complete(setup_mcp_servers())
            # Keep the loop open for webex_bot to use
        else:
            mcp_client = None

        bot = create_bot()

        # Setup commands and get the conversation manager instance
        conversation_manager = setup_commands(bot, mcp_client)

        # Initialize the conversation database for the actual manager
        try:
            # Use existing loop if MCP is enabled, otherwise create a new one
            if not settings.mcp.enabled:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(conversation_manager.initialize())
            log.info("Conversation database initialized and ready")
        except Exception as e:
            log.warning("Could not initialize conversation database: %s", e)

        log.info("Bot is starting... Press Ctrl+C to stop.")

        bot.run()

    except KeyboardInterrupt:
        log.info("Bot stopped by user")
    except Exception as e:
        log.exception("Bot error: %s", e)
        capture_exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
