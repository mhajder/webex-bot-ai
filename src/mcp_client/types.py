"""MCP tool and result dataclasses."""

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Represents a tool available from an MCP server."""

    name: str
    description: str
    input_schema: dict = field(default_factory=dict)
    server_name: str = ""

    def to_litellm_tool(self) -> dict:
        """Convert to LiteLLM/OpenAI function calling format.

        Handles tools with no parameters correctly by setting empty properties.
        """
        # Ensure input_schema has proper structure
        schema = self.input_schema
        if not schema:
            schema = {"type": "object", "properties": {}}

        # If schema doesn't have properties, add empty ones
        if "properties" not in schema:
            schema["properties"] = {}

        # If schema doesn't have required, add empty list
        if "required" not in schema:
            schema["required"] = []

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


@dataclass
class MCPToolResult:
    """Result from calling an MCP tool."""

    tool_name: str
    success: bool
    result: Any
    error: str | None = None
