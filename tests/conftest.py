"""Shared test fixtures for webex-bot-ai."""

import types
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from src.config import _MCP_CONFIG_ERRORS
from src.conversation.persistence import ConversationStore


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Return an isolated temporary SQLite database path."""
    return str(tmp_path / "test_conversations.db")


@pytest_asyncio.fixture
async def store(db_path: str) -> AsyncGenerator[ConversationStore, None]:
    """Create and initialize a temporary ConversationStore."""
    conv_store = ConversationStore(db_path)
    await conv_store.init_db()
    yield conv_store


@pytest.fixture(autouse=True)
def _isolate_mcp_config_errors() -> Any:
    """Ensure module-level MCP config warnings do not leak across tests."""
    saved = list(_MCP_CONFIG_ERRORS)
    _MCP_CONFIG_ERRORS.clear()
    yield
    _MCP_CONFIG_ERRORS.clear()
    _MCP_CONFIG_ERRORS.extend(saved)


def make_completion_response(
    content: str | None = "Hello!",
    tool_calls: list[Any] | None = None,
    *,
    empty_choices: bool = False,
) -> types.SimpleNamespace:
    """Build a LiteLLM-compatible completion response object."""
    if empty_choices:
        return types.SimpleNamespace(choices=[])
    message = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


def make_tool_call(
    call_id: str,
    name: str,
    arguments: str | dict[str, Any] | None,
) -> types.SimpleNamespace:
    """Build a LiteLLM-compatible tool call object."""
    return types.SimpleNamespace(
        id=call_id,
        function=types.SimpleNamespace(name=name, arguments=arguments),
    )
