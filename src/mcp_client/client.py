"""MCP (Model Context Protocol) client for connecting to multiple MCP servers."""

import asyncio
import logging
from typing import Any

from src.mcp_client.types import MCPTool, MCPToolResult

log = logging.getLogger(__name__)


class MCPMultiClient:
    """Client for connecting to and managing multiple MCP servers.

    Uses FastMCP's StreamableHttpTransport for HTTP-based connections.
    Aggregates tools from all connected servers and routes tool calls
    to the appropriate server.
    """

    def __init__(self, timeout: int = 30):
        """Initialize the multi-server MCP client.

        Args:
            timeout: Default timeout in seconds for requests.
        """
        self.timeout = timeout
        self._servers: dict[str, dict[str, Any]] = {}
        self._tools: dict[str, MCPTool] = {}  # tool_name -> MCPTool
        self._tool_server_map: dict[str, str] = {}  # tool_name -> server_name
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to any MCP server."""
        return self._connected and len(self._servers) > 0

    @property
    def server_count(self) -> int:
        """Get the number of connected servers."""
        return len(self._servers)

    @property
    def available_tools(self) -> list[MCPTool]:
        """Get list of available tools from all connected servers."""
        return list(self._tools.values())

    async def connect(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        auth_token: str | None = None,
    ) -> bool:
        """Connect to an MCP server using HTTP transport.

        Args:
            name: Unique identifier for this server connection.
            url: HTTP URL for the MCP server endpoint.
            headers: Optional custom headers.
            auth_token: Optional bearer token for authentication.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            from fastmcp.client import Client, StreamableHttpTransport

            # Build headers with auth if provided
            request_headers = dict(headers) if headers else {}
            if auth_token:
                request_headers["Authorization"] = f"Bearer {auth_token}"

            log.info("Connecting to MCP server '%s' at %s", name, url)

            # Create transport and client
            transport = StreamableHttpTransport(url=url, headers=request_headers)

            # Store server info
            self._servers[name] = {
                "url": url,
                "transport": transport,
                "client": None,
                "tools": [],
            }

            # Connect and discover tools
            async with Client(transport=transport) as client:
                # List available tools
                tools_result = await client.list_tools()
                server_tools = []

                for tool in tools_result:
                    input_schema = (
                        tool.inputSchema if hasattr(tool, "inputSchema") else {}
                    )

                    # Log the schema for debugging
                    log.debug(
                        "Tool '%s' schema: %s",
                        tool.name,
                        input_schema,
                    )

                    # Validate schema structure
                    if input_schema and "properties" not in input_schema:
                        log.warning(
                            "Tool '%s' has incomplete schema (missing 'properties'). "
                            "Schema: %s",
                            tool.name,
                            input_schema,
                        )

                    mcp_tool = MCPTool(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=input_schema,
                        server_name=name,
                    )

                    tool_name = tool.name
                    if tool_name in self._tools:
                        tool_name = f"{name}_{tool.name}"
                        mcp_tool.name = tool_name

                    self._tools[tool_name] = mcp_tool
                    self._tool_server_map[tool_name] = name
                    server_tools.append(mcp_tool)

                self._servers[name]["tools"] = server_tools
                log.info(
                    "Connected to MCP server '%s'. Discovered %d tools.",
                    name,
                    len(server_tools),
                )

            self._connected = True
            return True

        except ImportError as e:
            log.error("FastMCP import error: %s. Run: uv add fastmcp", e)
            return False
        except Exception as e:
            log.exception(
                "Failed to connect to MCP server '%s' at %s: %s", name, url, e
            )
            return False

    async def disconnect(self, name: str | None = None) -> None:
        """Disconnect from MCP server(s).

        Args:
            name: Server name to disconnect from. If None, disconnects from all.
        """
        if name:
            if name in self._servers:
                tools_to_remove = [
                    t for t, s in self._tool_server_map.items() if s == name
                ]
                for tool_name in tools_to_remove:
                    self._tools.pop(tool_name, None)
                    self._tool_server_map.pop(tool_name, None)

                del self._servers[name]
                log.info("Disconnected from MCP server '%s'", name)
        else:
            self._servers.clear()
            self._tools.clear()
            self._tool_server_map.clear()
            self._connected = False
            log.info("Disconnected from all MCP servers")

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        """Call a tool on the appropriate MCP server.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            MCPToolResult with the outcome.
        """
        if tool_name not in self._tools:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found",
            )

        server_name = self._tool_server_map[tool_name]
        server_info = self._servers.get(server_name)

        if not server_info:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Server '{server_name}' not connected",
            )

        try:
            from fastmcp.client import Client

            log.info(
                "Calling MCP tool '%s' on server '%s' with args: %s",
                tool_name,
                server_name,
                arguments,
            )

            original_name = tool_name
            if tool_name.startswith(f"{server_name}_"):
                original_name = tool_name[len(f"{server_name}_") :]

            async with Client(transport=server_info["transport"]) as client:
                result = await asyncio.wait_for(
                    client.call_tool(original_name, arguments or {}),
                    timeout=self.timeout,
                )

                if hasattr(result, "content"):
                    content_parts = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            content_parts.append(item.text)
                    result_text = (
                        "\n".join(content_parts) if content_parts else str(result)
                    )
                else:
                    result_text = str(result)

                log.info("Tool '%s' executed successfully", tool_name)
                return MCPToolResult(
                    tool_name=tool_name,
                    success=True,
                    result=result_text,
                )

        except TimeoutError:
            error_msg = f"Tool call timed out after {self.timeout}s"
            log.error("Tool '%s' timeout: %s", tool_name, error_msg)
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=error_msg,
            )
        except Exception as e:
            log.exception("Tool '%s' error: %s", tool_name, e)
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
            )

    def get_tools_for_litellm(self) -> list[dict]:
        """Get all tools formatted for LiteLLM/OpenAI function calling.

        Returns:
            List of tool definitions in OpenAI format.
        """
        return [tool.to_litellm_tool() for tool in self._tools.values()]

    def get_server_tools(self, server_name: str) -> list[MCPTool]:
        """Get tools from a specific server.

        Args:
            server_name: Name of the server.

        Returns:
            List of MCPTool objects from that server.
        """
        return [t for t in self._tools.values() if t.server_name == server_name]
