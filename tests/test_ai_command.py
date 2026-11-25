"""Tests for AICommand, tool execution, and Webex thread sync in src/commands/ai.py."""

import asyncio
import concurrent.futures
import json
import time
import types
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from webex_bot.formatting import quote_info

from src.commands.ai import (
    _MAX_SYNC_WATERMARKS,
    _MAX_TOOL_ROUNDS,
    _ROOM_DENIED_RETRY_SECONDS,
    AICommand,
    _flatten_tool_messages,
    _run_async_in_thread,
    _webex_created_ts,
    _webex_text,
)
from src.config import settings
from src.conversation.manager import ConversationManager
from src.mcp_client.types import MCPTool, MCPToolResult
from tests.conftest import make_completion_response, make_tool_call


def _make_cmd(
    *,
    bot_name: str = "Max",
    display_name: str = "Max Bot",
    mcp_client: Any = None,
    bot: Any = None,
    db_path: str | None = None,
    enable_persistence: bool = False,
) -> AICommand:
    settings.bot.display_name = display_name
    manager = ConversationManager(
        max_messages=50,
        timeout_seconds=3600,
        db_path=db_path or "unused.db",
        enable_persistence=enable_persistence,
        enable_summarization=False,
    )
    return AICommand(
        conversation_manager=manager,
        bot_name=bot_name,
        mcp_client=mcp_client,
        bot=bot,
    )


def test_module_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    # _flatten_tool_messages merges consecutive same-role messages (except system)
    msgs = [
        {"role": "system", "content": "sys1"},
        {"role": "system", "content": "sys2"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "Let me check",
            "tool_calls": [
                {"function": {"name": "ping", "arguments": '{"host": "1.1.1.1"}'}}
            ],
        },
        {"role": "tool", "content": "pong"},
        {"role": "user", "content": "summarize"},
    ]
    flat = _flatten_tool_messages(msgs)
    assert flat == [
        {"role": "system", "content": "sys1"},
        {"role": "system", "content": "sys2"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": 'Let me check\nCalled tools: ping({"host": "1.1.1.1"})',
        },
        {"role": "user", "content": "Tool result:\npong\n\nsummarize"},
    ]

    dt = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert _webex_created_ts(types.SimpleNamespace(created=dt)) == dt.timestamp()
    assert (
        _webex_created_ts(types.SimpleNamespace(created="2026-01-01T12:00:00+00:00"))
        == dt.timestamp()
    )
    assert _webex_created_ts(types.SimpleNamespace(created="not-a-date")) == 0.0
    assert _webex_created_ts(types.SimpleNamespace()) == 0.0

    assert _webex_text({"text": "  hi  "}) == "hi"
    assert _webex_text({"markdown": "**md**"}) == "**md**"
    assert _webex_text({"html": "<p>h</p>"}) == "<p>h</p>"
    assert _webex_text(types.SimpleNamespace(text=" obj ")) == "obj"
    assert _webex_text(types.SimpleNamespace(html="<i>i</i>")) == "<i>i</i>"

    async def add(a: int, b: int) -> int:
        await asyncio.sleep(0)
        return a + b

    assert _run_async_in_thread(add(2, 3)) == 5
    monkeypatch.undo()


def test_clean_prompt_strips_html_and_mentions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.bot, "display_name", "AI Helper")
    cmd = _make_cmd(bot_name="Max", display_name="AI Helper")

    assert cmd._clean_prompt("") == ""
    assert (
        cmd._clean_prompt("<p><spark-mention>Max</spark-mention>, status?</p>")
        == "status?"
    )
    assert cmd._clean_prompt("@Max: check routers") == "check routers"
    assert cmd._clean_prompt("AI Helper, list alerts") == "list alerts"
    # Word boundary prevents stripping "Max" from "Maximum"
    assert cmd._clean_prompt("Maximum size?") == "Maximum size?"


def test_get_thread_id_variants() -> None:
    cmd = _make_cmd()

    # attachment_actions priorities
    assert (
        cmd._get_thread_id(
            None, types.SimpleNamespace(inputs={"thread_parent_id": "t-input"}, id="m1")
        )
        == "t-input"
    )
    assert (
        cmd._get_thread_id(None, types.SimpleNamespace(parentId="p1", id="m1")) == "p1"
    )
    assert (
        cmd._get_thread_id(None, types.SimpleNamespace(parent_id="p2", id="m1")) == "p2"
    )
    assert cmd._get_thread_id(None, types.SimpleNamespace(id="m1")) == "m1"

    # activity dict variants
    assert (
        cmd._get_thread_id({"parent": {"id": "p-reply", "type": "reply"}}) == "p-reply"
    )
    assert cmd._get_thread_id({"parentId": "p-dict"}) == "p-dict"
    assert cmd._get_thread_id({"object": {"id": "obj-1"}}) == "obj-1"
    assert cmd._get_thread_id({"target": {"id": "tgt-1"}}) == "tgt-1"

    # activity object variants
    assert cmd._get_thread_id(types.SimpleNamespace(parent={"id": "obj-p"})) == "obj-p"
    assert (
        cmd._get_thread_id(
            types.SimpleNamespace(parent=types.SimpleNamespace(id="obj-p2"))
        )
        == "obj-p2"
    )
    assert cmd._get_thread_id(types.SimpleNamespace(parentId="attr-p")) == "attr-p"
    assert cmd._get_thread_id(types.SimpleNamespace(parent_id="attr-p2")) == "attr-p2"
    assert cmd._get_thread_id({}) is None


def test_get_room_id_variants() -> None:
    cmd = _make_cmd()

    assert cmd._get_room_id(None, types.SimpleNamespace(roomId="r1")) == "r1"
    assert cmd._get_room_id(None, types.SimpleNamespace(room_id="r2")) == "r2"
    assert cmd._get_room_id({"roomId": "r3"}) == "r3"
    assert cmd._get_room_id({"room_id": "r4"}) == "r4"
    assert cmd._get_room_id({"target": {"roomId": "r5"}}) == "r5"
    assert cmd._get_room_id({"message": {"id": "r6"}}) == "r6"
    assert cmd._get_room_id(types.SimpleNamespace(roomId="r7")) == "r7"
    assert cmd._get_room_id(types.SimpleNamespace(room_id="r8")) == "r8"
    assert cmd._get_room_id(types.SimpleNamespace(target={"room_id": "r9"})) == "r9"
    assert (
        cmd._get_room_id(types.SimpleNamespace(data=types.SimpleNamespace(id="r10")))
        == "r10"
    )
    assert (
        cmd._get_room_id(types.SimpleNamespace(raw=types.SimpleNamespace(roomId="r11")))
        == "r11"
    )
    assert cmd._get_room_id({}) is None


def test_wdm_fetch_and_thread_root_fallback() -> None:
    session = MagicMock()
    session.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "id": "root-1",
            "text": "Max initial question",
            "personId": "user-1",
            "personEmail": "user@example.com",
        },
    )
    bot_me = types.SimpleNamespace(id="bot-id", emails=["bot@webex.bot"])
    fake_bot = types.SimpleNamespace(
        websocket_client=types.SimpleNamespace(session=session),
        teams=types.SimpleNamespace(
            people=types.SimpleNamespace(me=lambda: bot_me),
            messages=MagicMock(),
        ),
    )
    fake_bot.teams.messages.get.side_effect = RuntimeError("404 geo-routed")

    cmd = _make_cmd(bot=fake_bot)
    activity = {
        "parent": {"id": "root-1"},
        "target": {"id": "conv-1", "url": "https://wdm/conversations/conv-1"},
    }
    assert cmd._fetch_thread_root("root-1", activity) == ("user", "initial question")
    session.get.assert_called_once_with("https://wdm/messages/root-1")

    # REST success path returning bot-authored root
    fake_bot.teams.messages.get.side_effect = None
    fake_bot.teams.messages.get.return_value = types.SimpleNamespace(
        text="Bot root reply",
        personId="bot-id",
        personEmail="bot@webex.bot",
    )
    assert cmd._fetch_thread_root("root-1", activity) == ("assistant", "Bot root reply")

    # Non-dict activity returns None on WDM fallback
    assert cmd._fetch_parent_message_via_wdm("not-a-dict") is None


class _StatusError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_list_messages_caches_403_only_and_expires() -> None:
    err_403 = _StatusError("forbidden", 403)
    err_401 = _StatusError("unauthorized", 401)

    messages_api = MagicMock()
    messages_api.list.side_effect = err_401
    fake_bot = types.SimpleNamespace(teams=types.SimpleNamespace(messages=messages_api))
    cmd = _make_cmd(bot=fake_bot)

    # 401 must NOT blacklist the room
    assert cmd._list_messages("room-1", 10) == []
    assert "room-1" not in cmd._room_list_denied

    # 403 caches the denial
    messages_api.list.side_effect = err_403
    assert cmd._list_messages("room-1", 10) == []
    assert "room-1" in cmd._room_list_denied
    assert messages_api.list.call_count == 2

    # Subsequent call within cooldown skips API
    assert cmd._list_messages("room-1", 10) == []
    assert messages_api.list.call_count == 2

    # After cooldown expires, retries API
    cmd._room_list_denied["room-1"] = time.time() - _ROOM_DENIED_RETRY_SECONDS - 1
    messages_api.list.side_effect = None
    messages_api.list.return_value = [types.SimpleNamespace(id="m1")]
    assert len(cmd._list_messages("room-1", 10)) == 1
    assert "room-1" not in cmd._room_list_denied


def test_sync_webex_thread_history_and_room_context() -> None:
    now = time.time()
    bot_me = types.SimpleNamespace(id="bot-id", emails=["bot@webex.bot"])
    root_msg = types.SimpleNamespace(
        id="t1",
        text="Root question",
        personId="u1",
        personEmail="u1@example.com",
        created=now - 120,
    )
    replies = [
        types.SimpleNamespace(
            id="r-unmentioned",
            text="Extra detail without mention",
            personId="u2",
            personEmail="u2@example.com",
            created=now - 60,
            mentionedPeople=[],
            roomType="group",
        ),
        types.SimpleNamespace(
            id="r-bot",
            text="Bot message",
            personId="bot-id",
            personEmail="bot@webex.bot",
            created=now - 50,
            mentionedPeople=[],
            roomType="group",
        ),
        types.SimpleNamespace(
            id="r-future",
            text="Posted after current message",
            personId="u2",
            personEmail="u2@example.com",
            created=now + 10,
            mentionedPeople=[],
            roomType="group",
        ),
    ]
    # Convert numeric timestamps to ISO strings for _webex_created_ts
    for m in [root_msg, *replies]:
        m.created = datetime.fromtimestamp(m.created, tz=UTC).isoformat()

    messages_api = MagicMock()
    messages_api.get.return_value = root_msg
    messages_api.list.return_value = replies
    fake_bot = types.SimpleNamespace(
        teams=types.SimpleNamespace(
            people=types.SimpleNamespace(me=lambda: bot_me),
            messages=messages_api,
        )
    )
    cmd = _make_cmd(bot=fake_bot)

    # Cold start sync adds root, unmentioned reply, and bot reply (not future reply)
    cmd._sync_webex_thread_history(
        "t1", "room-1", current_msg_id="cur-1", current_created=now
    )
    history = cmd.conversation_manager.get_history("t1")
    assert [m.content for m in history] == [
        "Root question",
        "Extra detail without mention",
        "Bot message",
    ]

    # Warm sync skips bot and addressed messages, only picks up new unmentioned reply
    new_reply = types.SimpleNamespace(
        id="r-new-unmentioned",
        text="Another unmentioned reply",
        personId="u2",
        personEmail="u2@example.com",
        created=datetime.fromtimestamp(now - 10, tz=UTC).isoformat(),
        mentionedPeople=[],
        roomType="group",
    )
    mentioned_reply = types.SimpleNamespace(
        id="r-mentioned",
        text="Max follow up",
        personId="u2",
        personEmail="u2@example.com",
        created=datetime.fromtimestamp(now - 5, tz=UTC).isoformat(),
        mentionedPeople=["bot-id"],
        roomType="group",
    )
    messages_api.list.return_value = [new_reply, mentioned_reply]
    cmd._sync_webex_thread_history(
        "t1", "room-1", current_msg_id="cur-2", current_created=now
    )
    assert [m.content for m in cmd.conversation_manager.get_history("t1")][-1] == (
        "Another unmentioned reply"
    )

    # Direct room warm sync short-circuits without calling list()
    prev_calls = messages_api.list.call_count
    cmd._sync_webex_thread_history("t1", "room-1", direct=True)
    assert messages_api.list.call_count == prev_calls

    # Watermark pruning when >= _MAX_SYNC_WATERMARKS
    for i in range(_MAX_SYNC_WATERMARKS):
        cmd._sync_watermarks[f"old-{i}"] = 1.0
    cmd._sync_webex_thread_history("t1", "room-1", current_created=now)
    assert len(cmd._sync_watermarks) < _MAX_SYNC_WATERMARKS

    # Room context skips current_msg_id and expired messages
    expired = types.SimpleNamespace(
        id="old-room-msg",
        text="Old",
        personId="u1",
        created=datetime.fromtimestamp(now - 99999, tz=UTC).isoformat(),
    )
    recent = types.SimpleNamespace(
        id="recent-room-msg",
        text="Recent space chat",
        personId="u1",
        created=datetime.fromtimestamp(now - 30, tz=UTC).isoformat(),
    )
    current = types.SimpleNamespace(
        id="cur-top",
        text="Current",
        personId="u1",
        created=datetime.fromtimestamp(now, tz=UTC).isoformat(),
    )
    messages_api.list.return_value = [expired, recent, current]
    assert cmd._fetch_room_context("room-1", current_msg_id="cur-top") == [
        ("user", "Recent space chat")
    ]


@pytest.mark.asyncio
async def test_build_messages_with_room_context_and_fts_dedup(db_path: str) -> None:
    cmd = _make_cmd(db_path=db_path, enable_persistence=True)
    await cmd.conversation_manager.initialize()

    # Seed another thread in same room with searchable info
    assert cmd.conversation_manager.store is not None
    from src.conversation.types import Message

    await cmd.conversation_manager.store.save_message(
        "t-past",
        Message(role="assistant", content="Switch uplink uses VLAN 100", timestamp=1.0),
        room_id="room-1",
    )

    cmd.conversation_manager.add_user_message(
        "t-cur", "VLAN 100 status?", room_id="room-1"
    )
    msgs = cmd._build_messages(
        "VLAN 100 status?",
        thread_id="t-cur",
        room_id="room-1",
        room_context=[("user", "Earlier top-level mention")],
        prompt_saved=True,
    )
    assert msgs[0]["role"] == "system"
    assert any("[Recent Messages in this Webex Space]" in m["content"] for m in msgs)
    assert any(
        "[Relevant Past Discussion Context in this Webex Space]" in m["content"]
        and "Switch uplink uses VLAN 100" in m["content"]
        for m in msgs
    )
    assert msgs[-1] == {"role": "user", "content": "VLAN 100 status?"}


def test_get_mcp_tools_and_handle_tool_calls() -> None:
    tool_def = MCPTool(
        name="get_device",
        description="Get device",
        input_schema={
            "type": "object",
            "properties": {"hostname": {"type": "string"}},
        },
        server_name="nms",
    )
    open_tool = MCPTool(
        name="tool.raw",
        description="Raw",
        input_schema={"type": "object", "additionalProperties": True},
        server_name="nms",
    )
    mock_mcp = MagicMock()
    mock_mcp.available_tools = [tool_def, open_tool]
    mock_mcp.get_tools_for_litellm.return_value = [
        tool_def.to_litellm_tool(),
        open_tool.to_litellm_tool(),
    ]
    mock_mcp.get_tool.side_effect = lambda n: {
        "get_device": tool_def,
        "tool.raw": open_tool,
    }.get(n)

    async def fake_call_tool(name: str, args: dict[str, Any]) -> MCPToolResult:
        if name == "get_device":
            return MCPToolResult(
                tool_name=name, success=True, result=f"ok:{args['hostname']}"
            )
        return MCPToolResult(tool_name=name, success=False, result=None, error="failed")

    mock_mcp.call_tool.side_effect = fake_call_tool

    cmd = _make_cmd(mcp_client=mock_mcp, enable_persistence=True)
    tools = cmd._get_mcp_tools()
    assert {t["function"]["name"] for t in tools} == {
        "get_device",
        "tool.raw",
        "search_room_history",
    }

    # Patch room search sync
    cmd.conversation_manager.search_room_history_sync = MagicMock(  # type: ignore[method-assign]
        return_value=[types.SimpleNamespace(role="user", content="found bgp config")]
    )

    calls = [
        # Prefix stripped ("tool.get_device" -> "get_device") & unknown arg filtered
        make_tool_call("c1", "tool.get_device", '{"hostname": "sw1", "extra": "drop"}'),
        # Real tool literally named "tool.raw" keeps prefix & additionalProperties=True
        make_tool_call("c2", "tool.raw", {"any_key": 123}),
        # Built-in search_room_history with alternate arg key
        make_tool_call("c3", "search_room_history", '{"keywords": "bgp"}'),
        # Unknown tool
        make_tool_call("c4", "nonexistent", "{bad-json"),
        # Missing room_id for search_room_history
        make_tool_call("c5", "search_room_history", "null"),
    ]
    results = cmd._handle_tool_calls(calls, room_id="room-1")
    assert results["c1"] == "ok:sw1"
    assert results["c2"] == "Error: failed"
    assert "found bgp config" in results["c3"]
    assert results["c4"] == "Error: Tool 'nonexistent' does not exist"
    assert "was missing" in results["c5"]

    # Built-in search_room_history with no matches
    cmd.conversation_manager.search_room_history_sync.return_value = []
    res_empty = cmd._handle_tool_calls(
        [make_tool_call("c6", "search_room_history", {"query": "nothing"})],
        room_id="room-1",
    )
    assert "No historical messages matching 'nothing'" in res_empty["c6"]


def test_call_mcp_tool_timeout_and_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    cmd = _make_cmd(mcp_client=None)
    no_client = cmd._call_mcp_tool("x", {})
    assert no_client.success is False
    assert no_client.error == "MCP client not configured"

    mock_mcp = MagicMock()
    cmd.mcp_client = mock_mcp

    fake_future = MagicMock()
    fake_future.result.side_effect = concurrent.futures.TimeoutError()

    def _submit_and_close(_fn: Any, coro: Any) -> MagicMock:
        coro.close()
        return fake_future

    monkeypatch.setattr("src.commands.ai._executor.submit", _submit_and_close)
    timed_out = cmd._call_mcp_tool("slow", {})
    assert timed_out.success is False
    assert "timed out" in (timed_out.error or "")

    fake_future.result.side_effect = RuntimeError("executor crash")
    with (
        patch("src.commands.ai.set_tag") as mock_tag,
        patch("src.commands.ai.capture_exception") as mock_cap,
    ):
        errored = cmd._call_mcp_tool("boom", {})
    assert errored.success is False
    assert errored.error == "executor crash"
    mock_tag.assert_called_once_with("mcp_tool", "boom")
    mock_cap.assert_called_once()


def test_native_tool_calls_multi_round_and_limit() -> None:
    cmd = _make_cmd()
    cmd._handle_tool_calls = MagicMock(return_value={"tc1": "tool output"})  # type: ignore[method-assign]

    # No tool calls on round 1 -> returns None
    no_tools = make_completion_response("plain text", tool_calls=None).choices[0]
    assert cmd._handle_native_tool_calls(no_tools, []) is None

    # Multi-round: round 1 calls tool, round 2 returns text
    r1_choice = make_completion_response(
        None, tool_calls=[make_tool_call("tc1", "search_room_history", {"query": "x"})]
    ).choices[0]
    r2_resp = make_completion_response("Final answer from tool")
    with patch.object(cmd, "_call_llm", return_value=r2_resp):
        out = cmd._handle_native_tool_calls(
            r1_choice,
            [{"role": "user", "content": "hi"}],
            room_id="r1",
            tools=[{"t": 1}],
        )
    assert out == "Final answer from tool"

    # Hitting _MAX_TOOL_ROUNDS triggers _flatten_tool_messages and final nudge
    loop_resp = make_completion_response(
        None, tool_calls=[make_tool_call("tc1", "search_room_history", {"query": "x"})]
    )
    forced_resp = make_completion_response("Forced summary")
    side_effects = [loop_resp] * (_MAX_TOOL_ROUNDS - 1) + [forced_resp]
    with patch.object(cmd, "_call_llm", side_effect=side_effects) as mock_llm:
        forced = cmd._handle_native_tool_calls(
            loop_resp.choices[0], [{"role": "user", "content": "hi"}], room_id="r1"
        )
    assert forced == "Forced summary"
    # Final call has tools=None
    assert mock_llm.call_args.kwargs["tools"] is None


def test_parse_and_handle_json_tool_calls() -> None:
    assert AICommand._parse_json_tool_call(None) is None
    assert AICommand._parse_json_tool_call("Just text") is None
    assert AICommand._parse_json_tool_call('{"name": "only_name"}') is None
    assert AICommand._parse_json_tool_call('{"name": "t", "arguments": {"a": 1}}') == (
        "t",
        {"a": 1},
    )
    assert AICommand._parse_json_tool_call(
        '```json\n{"name": "t", "arguments": "{\\"x\\": 2}"}\n```'
    ) == ("t", {"x": 2})
    assert AICommand._parse_json_tool_call(
        '```\n{"name": "t", "arguments": "bad"}\n```'
    ) == ("t", {})
    assert AICommand._parse_json_tool_call('{"name": "t", "arguments": [1, 2]}') == (
        "t",
        {},
    )

    cmd = _make_cmd()
    cmd._handle_tool_calls = MagicMock(return_value={"json_call_1": "res1"})  # type: ignore[method-assign]

    assert cmd._handle_json_tool_calls("Normal answer", []) is None

    with patch.object(
        cmd, "_call_llm", return_value=make_completion_response("JSON tool answer")
    ):
        res = cmd._handle_json_tool_calls(
            '{"name": "my_tool", "arguments": {"k": "v"}}',
            [{"role": "user", "content": "q"}],
            room_id="r1",
        )
    assert res == "JSON tool answer"

    # Exhausting _MAX_TOOL_ROUNDS in JSON mode returns fallback message
    cmd._handle_tool_calls = MagicMock(return_value={})  # type: ignore[method-assign]
    always_tool = make_completion_response('{"name": "loop", "arguments": {}}')
    with patch.object(cmd, "_call_llm", return_value=always_tool):
        exhausted = cmd._handle_json_tool_calls(
            '{"name": "loop", "arguments": {}}',
            [{"role": "user", "content": "q"}],
        )
    assert "couldn't finish gathering" in (exhausted or "")


def test_execute_end_to_end_and_error_paths() -> None:
    cmd = _make_cmd(bot_name="Max")
    assert cmd.pre_execute("hi", None, None) is None
    assert "Please ask me a question" in cmd.execute("   ", None, None)[0]
    assert "I'm here to help" in cmd.execute("@Max", None, None)[0]

    # Top-level message fetches room context, calls LLM, saves user & assistant messages
    cmd._fetch_room_context = MagicMock(return_value=[("user", "earlier")])  # type: ignore[method-assign]
    attachment = types.SimpleNamespace(
        id="msg-1",
        roomId="room-1",
        roomType="group",
        created="2026-01-01T12:00:00+00:00",
    )
    activity = {
        "actor": {
            "entryUUID": "u-uuid",
            "emailAddress": "u@example.com",
            "displayName": "User",
        },
        "object": {"id": "msg-1"},
    }
    with (
        patch(
            "src.commands.ai.litellm.completion",
            return_value=make_completion_response("Answer 1"),
        ) as mock_litellm,
        patch("src.sentry.set_user_context") as mock_user_ctx,
    ):
        out = cmd.execute("Max, what is up?", attachment, activity)

    assert out == [quote_info("Answer 1")]
    mock_user_ctx.assert_called_once_with("u-uuid", "u@example.com", "User")
    mock_litellm.assert_called_once()
    cmd._fetch_room_context.assert_called_once_with("room-1", "msg-1")
    history = cmd.conversation_manager.get_history("msg-1")
    assert [(m.role, m.content) for m in history] == [
        ("user", "what is up?"),
        ("assistant", "Answer 1"),
    ]

    # Empty choices
    with patch(
        "src.commands.ai.litellm.completion",
        return_value=make_completion_response(empty_choices=True),
    ):
        empty_choices_out = cmd.execute("Max, test", attachment, activity)
    assert "couldn't generate a response" in empty_choices_out[0]

    # Empty response content
    with patch(
        "src.commands.ai.litellm.completion",
        return_value=make_completion_response("   "),
    ):
        empty_content_out = cmd.execute("Max, test", attachment, activity)
    assert "received an empty response" in empty_content_out[0]

    # LLM exception captures to Sentry and returns safe error
    with (
        patch(
            "src.commands.ai.litellm.completion", side_effect=RuntimeError("api fail")
        ),
        patch("src.commands.ai.set_tag") as mock_tag,
        patch("src.commands.ai.capture_exception") as mock_cap,
    ):
        err_out = cmd.execute("Max, test", attachment, activity)
    assert "encountered an error: RuntimeError" in err_out[0]
    mock_tag.assert_any_call("llm_model", cmd.model)
    mock_tag.assert_any_call("thread_id", "msg-1")
    mock_cap.assert_called_once()


def test_call_llm_passes_api_base_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    cmd = _make_cmd()
    monkeypatch.setattr(settings.llm, "api_base", "http://localhost:11434")
    monkeypatch.setattr(settings.llm, "api_key", "sk-test")

    with patch(
        "src.commands.ai.litellm.completion",
        return_value=make_completion_response("ok"),
    ) as mock_comp:
        cmd._call_llm([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

    kwargs = mock_comp.call_args.kwargs
    assert kwargs["api_base"] == "http://localhost:11434"
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["tools"] == [{"type": "function"}]


def test_additional_edge_cases_for_webex_tools_and_sentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # WDM session via teams._session, non-200 status, and exception
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=404)
    bot_with_teams_session = types.SimpleNamespace(
        teams=types.SimpleNamespace(
            _session=session,
            messages=MagicMock(get=MagicMock(side_effect=RuntimeError("no rest"))),
            people=types.SimpleNamespace(
                me=lambda: types.SimpleNamespace(id="b1", emails=[])
            ),
        )
    )
    cmd = _make_cmd(bot=bot_with_teams_session)
    activity = {
        "parent": {"id": "p1"},
        "target": {"id": "c1", "url": "https://wdm/conversations/c1"},
    }
    assert cmd._fetch_parent_message_via_wdm(activity) is None
    assert cmd._fetch_thread_root("p1", activity) is None

    # Empty WDM text returns None from _fetch_thread_root
    session.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"id": "p1", "text": "   "},
    )
    assert cmd._fetch_thread_root("p1", activity) is None

    # Exception during session.get
    session.get.side_effect = RuntimeError("network")
    assert cmd._fetch_parent_message_via_wdm(activity) is None
    assert cmd._fetch_parent_message_via_wdm({"parent": {}, "target": {}}) is None

    # Direct room message is always addressed to bot
    assert cmd._is_addressed_to_bot(types.SimpleNamespace(roomType="direct")) is True

    # No webex_bot -> sync and room context return early
    no_bot_cmd = _make_cmd(bot=None)
    no_bot_cmd._sync_webex_thread_history("t1", "r1")
    assert no_bot_cmd._fetch_room_context("r1") == []

    # _fetch_room_context handles empty list and exceptions
    cmd._list_messages = MagicMock(return_value=[])  # type: ignore[method-assign]
    assert cmd._fetch_room_context("r1") == []
    cmd._list_messages = MagicMock(return_value=[types.SimpleNamespace(id="m1")])  # type: ignore[method-assign]
    monkeypatch.setattr(
        "src.commands.ai._webex_created_ts",
        lambda _m: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    assert cmd._fetch_room_context("r1") == []
    monkeypatch.undo()

    # _set_sentry_context with object activity & exception safety
    obj_activity = types.SimpleNamespace(
        actor=types.SimpleNamespace(
            entryUUID="u-obj", emailAddress="o@example.com", displayName="Obj"
        )
    )
    with patch("src.sentry.set_user_context") as mock_user_ctx:
        cmd._set_sentry_context(obj_activity)
    mock_user_ctx.assert_called_once_with("u-obj", "o@example.com", "Obj")

    with patch("src.sentry.set_user_context", side_effect=RuntimeError("boom")):
        cmd._set_sentry_context(obj_activity)

    # Tool execution error paths in _handle_tool_calls
    cmd.conversation_manager.search_room_history_sync = MagicMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("fts fail")
    )
    cmd._call_mcp_tool = MagicMock(side_effect=json.JSONDecodeError("bad", "", 0))  # type: ignore[method-assign]
    err_results = cmd._handle_tool_calls(
        [
            make_tool_call("s1", "search_room_history", {"query": "x"}),
            make_tool_call("m1", "some_tool", {"a": 1}),
        ],
        room_id="r1",
    )
    assert "Error executing room search" in err_results["s1"]
    assert "Invalid JSON arguments" in err_results["m1"]

    cmd._call_mcp_tool = MagicMock(side_effect=ValueError("bad value"))  # type: ignore[method-assign]
    err_results2 = cmd._handle_tool_calls(
        [make_tool_call("m2", "some_tool", {})], room_id="r1"
    )
    assert err_results2["m2"] == "Error: ValueError: bad value"

    # Empty tool_results and empty choices in native/json loops
    cmd._handle_tool_calls = MagicMock(return_value={})  # type: ignore[method-assign]
    tc_choice = make_completion_response(
        "fallback", tool_calls=[make_tool_call("tc", "x", {})]
    ).choices[0]
    assert cmd._handle_native_tool_calls(tc_choice, []) is None

    cmd._handle_tool_calls = MagicMock(return_value={"tc": "ok"})  # type: ignore[method-assign]
    with patch.object(
        cmd, "_call_llm", return_value=make_completion_response(empty_choices=True)
    ):
        assert cmd._handle_native_tool_calls(tc_choice, []) == ""
        assert cmd._handle_json_tool_calls('{"name": "x", "arguments": {}}', []) == ""

    # execute() thread reply branch calls _sync_webex_thread_history
    reply_cmd = _make_cmd(bot_name="Max")
    reply_cmd._sync_webex_thread_history = MagicMock()  # type: ignore[method-assign]
    attachment = types.SimpleNamespace(
        id="reply-msg",
        parentId="root-thread",
        roomId="room-1",
        roomType="direct",
        created="2026-01-01T12:00:00+00:00",
    )
    with patch(
        "src.commands.ai.litellm.completion",
        return_value=make_completion_response("Reply answer"),
    ):
        out = reply_cmd.execute("Max follow up", attachment, {})
    assert out == [quote_info("Reply answer")]
    reply_cmd._sync_webex_thread_history.assert_called_once()
