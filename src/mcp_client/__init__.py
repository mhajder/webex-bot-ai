"""MCP (Model Context Protocol) client for connecting to multiple MCP servers."""

from src.mcp_client.client import MCPMultiClient
from src.mcp_client.types import MCPTool, MCPToolResult

__all__ = ["MCPMultiClient", "MCPTool", "MCPToolResult"]
