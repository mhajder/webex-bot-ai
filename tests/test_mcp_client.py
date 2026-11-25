"""Tests for MCP client and tool definitions in src/mcp_client/."""

import logging
import sys
import types
from typing import Any

import pytest

from src.mcp_client.client import RESERVED_TOOL_NAMES, MCPMultiClient
from src.mcp_client.types import MCPTool, MCPToolResult


def test_mcp_tool_to_litellm_tool_full_and_empty_schemas() -> None:
    tool_full = MCPTool(
        name="get_devices",
        description="List devices",
        input_schema={
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
            "required": ["limit"],
        },
        server_name="nms",
    )
    assert tool_full.to_litellm_tool() == {
        "type": "function",
        "function": {
            "name": "get_devices",
            "description": "List devices",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": ["limit"],
            },
        },
    }

    original_partial = {"type": "object"}
    tool_partial = MCPTool(
        name="ping",
        description="Ping",
        input_schema=original_partial,
    )
    converted = tool_partial.to_litellm_tool()
    assert converted["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
    }
    assert original_partial == {"type": "object"}

    tool_empty = MCPTool(name="status", description="Status")
    assert tool_empty.to_litellm_tool()["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
    }


def test_mcp_tool_result_dataclass() -> None:
    res = MCPToolResult(tool_name="ping", success=True, result="ok")
    assert res.error is None


def _install_fake_fastmcp(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tools_by_url: dict[str, list[Any]] | None = None,
    call_handler: Any = None,
    connect_error_urls: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Install a fake fastmcp.client module into sys.modules."""
    transports: list[dict[str, Any]] = []

    class FakeTransport:
        def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
            self.url = url
            self.headers = headers or {}
            transports.append({"url": url, "headers": self.headers})

    class FakeClient:
        def __init__(self, transport: FakeTransport) -> None:
            self.transport = transport

        async def __aenter__(self) -> "FakeClient":
            if connect_error_urls and self.transport.url in connect_error_urls:
                raise ConnectionError(f"cannot reach {self.transport.url}")
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def list_tools(self) -> list[Any]:
            return (tools_by_url or {}).get(self.transport.url, [])

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            if call_handler is None:
                raise AssertionError("call_handler not configured")
            return await call_handler(self.transport.url, name, arguments)

    fake_client_mod = types.SimpleNamespace(
        Client=FakeClient,
        StreamableHttpTransport=FakeTransport,
    )
    monkeypatch.setitem(sys.modules, "fastmcp", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "fastmcp.client", fake_client_mod)
    return transports


@pytest.mark.asyncio
async def test_connect_discovers_tools_handles_collisions_and_reserved_names(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    reserved_name = next(iter(RESERVED_TOOL_NAMES))
    tools_by_url = {
        "http://s1/mcp": [
            types.SimpleNamespace(
                name="get_info",
                description="Info 1",
                inputSchema={
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
            ),
            types.SimpleNamespace(
                name=reserved_name,
                description="Shadowed built-in",
                inputSchema={"type": "object"},  # Missing properties -> warning
            ),
        ],
        "http://s2/mcp": [
            types.SimpleNamespace(
                name="get_info",
                description="Info 2",
                inputSchema={"type": "object", "properties": {}},
            ),
        ],
    }
    transports = _install_fake_fastmcp(monkeypatch, tools_by_url=tools_by_url)

    client = MCPMultiClient(timeout=15)
    assert client.is_connected is False
    assert client.server_count == 0
    assert client.available_tools == []
    assert client.get_tool("missing") is None

    with caplog.at_level(logging.WARNING):
        ok1 = await client.connect(
            name="srv1",
            url="http://s1/mcp",
            headers={"X-Custom": "1"},
            auth_token="tok-1",
        )
        ok2 = await client.connect(name="srv2", url="http://s2/mcp")

    assert ok1 is True
    assert ok2 is True
    assert client.is_connected is True
    assert client.server_count == 2
    assert "incomplete schema" in caplog.text

    assert transports[0]["headers"] == {
        "X-Custom": "1",
        "Authorization": "Bearer tok-1",
    }
    assert transports[1]["headers"] == {}

    tool_names = {t.name for t in client.available_tools}
    assert tool_names == {"get_info", f"srv1_{reserved_name}", "srv2_get_info"}
    assert client._tool_original_names["srv2_get_info"] == "get_info"
    assert client._tool_original_names[f"srv1_{reserved_name}"] == reserved_name
    assert len(client.get_server_tools("srv1")) == 2
    assert len(client.get_server_tools("srv2")) == 1
    assert len(client.get_tools_for_litellm()) == 3


@pytest.mark.asyncio
async def test_connect_failure_and_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_fastmcp(monkeypatch, connect_error_urls={"http://down/mcp"})
    client = MCPMultiClient()
    assert await client.connect("down", "http://down/mcp") is False
    assert client.is_connected is False
    assert client.server_count == 0
    assert client.available_tools == []

    monkeypatch.setitem(sys.modules, "fastmcp.client", None)
    assert await client.connect("no_fastmcp", "http://any/mcp") is False


@pytest.mark.asyncio
async def test_disconnect_single_and_all(monkeypatch: pytest.MonkeyPatch) -> None:
    tools_by_url = {
        "http://s1/mcp": [
            types.SimpleNamespace(
                name="t1", description="T1", inputSchema={"properties": {}}
            )
        ],
        "http://s2/mcp": [
            types.SimpleNamespace(
                name="t2", description="T2", inputSchema={"properties": {}}
            )
        ],
    }
    _install_fake_fastmcp(monkeypatch, tools_by_url=tools_by_url)
    client = MCPMultiClient()
    await client.connect("srv1", "http://s1/mcp")
    await client.connect("srv2", "http://s2/mcp")

    # Unknown server disconnect is a no-op
    await client.disconnect("unknown")
    assert client.server_count == 2

    await client.disconnect("srv1")
    assert client.server_count == 1
    assert client.get_tool("t1") is None
    assert client.get_tool("t2") is not None
    assert client.is_connected is True

    await client.disconnect()
    assert client.server_count == 0
    assert client.available_tools == []
    assert client.is_connected is False


@pytest.mark.asyncio
async def test_call_tool_success_prefix_mapping_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    async def handler(url: str, name: str, arguments: dict[str, Any]) -> Any:
        calls.append((url, name, arguments))
        if name == "text_tool":
            return types.SimpleNamespace(
                content=[
                    types.SimpleNamespace(text="line1"),
                    types.SimpleNamespace(other="ignored"),
                    types.SimpleNamespace(text="line2"),
                ]
            )
        if name == "raw_tool":
            return {"status": "ok"}
        if name == "empty_content_tool":
            return types.SimpleNamespace(content=[])
        if name == "timeout_tool":
            raise TimeoutError("slow")
        raise RuntimeError("backend failed")

    tools_by_url = {
        "http://s1/mcp": [
            types.SimpleNamespace(
                name="text_tool", description="", inputSchema={"properties": {}}
            ),
            types.SimpleNamespace(
                name="raw_tool", description="", inputSchema={"properties": {}}
            ),
            types.SimpleNamespace(
                name="empty_content_tool",
                description="",
                inputSchema={"properties": {}},
            ),
            types.SimpleNamespace(
                name="timeout_tool", description="", inputSchema={"properties": {}}
            ),
            types.SimpleNamespace(
                name="err_tool", description="", inputSchema={"properties": {}}
            ),
        ],
        "http://s2/mcp": [
            types.SimpleNamespace(
                name="text_tool", description="", inputSchema={"properties": {}}
            ),
        ],
    }
    _install_fake_fastmcp(monkeypatch, tools_by_url=tools_by_url, call_handler=handler)

    client = MCPMultiClient(timeout=5)
    await client.connect("srv1", "http://s1/mcp")
    await client.connect("srv2", "http://s2/mcp")

    # Unknown tool
    missing_res = await client.call_tool("does_not_exist")
    assert missing_res.success is False
    assert "not found" in (missing_res.error or "")

    # Prefixed tool maps back to original name on srv2
    prefixed_res = await client.call_tool("srv2_text_tool", {"k": "v"})
    assert prefixed_res.success is True
    assert prefixed_res.result == "line1\nline2"
    assert calls[-1] == ("http://s2/mcp", "text_tool", {"k": "v"})

    # Non-content object falls back to str(result)
    raw_res = await client.call_tool("raw_tool")
    assert raw_res.success is True
    assert raw_res.result == "{'status': 'ok'}"

    # Empty content list falls back to str(result)
    empty_res = await client.call_tool("empty_content_tool")
    assert empty_res.success is True
    assert "content=[]" in str(empty_res.result)

    # Timeout
    timeout_res = await client.call_tool("timeout_tool")
    assert timeout_res.success is False
    assert timeout_res.error == "Tool call timed out after 5s"

    # Exception
    err_res = await client.call_tool("err_tool")
    assert err_res.success is False
    assert err_res.error == "backend failed"

    # Simulate stale server map entry when server removed
    client._servers.pop("srv1")
    disconnected_res = await client.call_tool("raw_tool")
    assert disconnected_res.success is False
    assert disconnected_res.error == "Server 'srv1' not connected"
