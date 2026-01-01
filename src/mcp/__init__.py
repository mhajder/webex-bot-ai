"""MCP (Model Context Protocol) client for connecting to multiple MCP servers."""

from src.mcp.client import MCPMultiClient
from src.mcp.types import MCPTool, MCPToolResult

__all__ = ["MCPMultiClient", "MCPTool", "MCPToolResult"]
