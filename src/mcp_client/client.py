"""MCP (Model Context Protocol) client for connecting to multiple MCP servers."""

import asyncio
import logging
from typing import Any

from src.mcp_client.types import MCPTool, MCPToolResult

log = logging.getLogger(__name__)

# Names of the bot's built-in tools. MCP tools with these names are exposed
# under the server-prefixed name, so they neither duplicate a tool definition
# nor get shadowed by the built-in handler.
RESERVED_TOOL_NAMES = frozenset({"search_room_history"})


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
        # exposed tool_name -> name the server knows the tool by
        self._tool_original_names: dict[str, str] = {}
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

    def get_tool(self, tool_name: str) -> MCPTool | None:
        """Get a tool definition by its exposed name."""
        return self._tools.get(tool_name)

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

            # Connect and discover tools
            async with Client(transport=transport) as client:
                # List available tools
                tools_result = await client.list_tools()
                server_tools = []
                discovered: list[tuple[str, str, MCPTool]] = []

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
                    if tool_name in self._tools or tool_name in RESERVED_TOOL_NAMES:
                        tool_name = f"{name}_{tool.name}"
                        mcp_tool.name = tool_name

                    discovered.append((tool_name, tool.name, mcp_tool))
                    server_tools.append(mcp_tool)

                # Register the server only once discovery succeeded, so a failed
                # connection leaves no half-registered server behind.
                for tool_name, original_name, mcp_tool in discovered:
                    self._tools[tool_name] = mcp_tool
                    self._tool_server_map[tool_name] = name
                    self._tool_original_names[tool_name] = original_name
                self._servers[name] = {
                    "url": url,
                    "headers": request_headers,
                    "tools": server_tools,
                }
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
        """Disconnect from MCP server(s) and clean up resources.

        Args:
            name: Server name to disconnect from. If None, disconnects from all.
        """
        if name:
            if name in self._servers:
                # Each call opens and closes its own client session, so there is
                # no persistent transport to close here.
                self._servers.pop(name)
                tools_to_remove = [
                    t for t, s in self._tool_server_map.items() if s == name
                ]
                for tool_name in tools_to_remove:
                    self._tools.pop(tool_name, None)
                    self._tool_server_map.pop(tool_name, None)
                    self._tool_original_names.pop(tool_name, None)

                log.info("Disconnected from MCP server '%s'", name)
        else:
            server_names = list(self._servers.keys())
            for s_name in server_names:
                await self.disconnect(s_name)
            self._tools.clear()
            self._tool_server_map.clear()
            self._tool_original_names.clear()
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
            from fastmcp.client import Client, StreamableHttpTransport

            log.info(
                "Calling MCP tool '%s' on server '%s' with args: %s",
                tool_name,
                server_name,
                arguments,
            )

            original_name = self._tool_original_names.get(tool_name, tool_name)

            transport = StreamableHttpTransport(
                url=server_info["url"],
                headers=server_info.get("headers", {}),
            )

            # The timeout covers session setup as well as the call, so a server
            # that hangs while connecting can't pin the caller's worker thread.
            async with (
                asyncio.timeout(self.timeout),
                Client(transport=transport) as client,
            ):
                result = await client.call_tool(original_name, arguments or {})

                if hasattr(result, "content"):
                    content_parts = [
                        item.text for item in result.content if hasattr(item, "text")
                    ]
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
