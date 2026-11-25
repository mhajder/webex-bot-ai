"""Tests for application startup and wiring in src/main.py."""

import types
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src import main as main_mod
from src.commands.ai import AICommand
from src.config import MCPServerConfig, Settings
from src.mcp_client.client import MCPMultiClient


def test_validate_configuration_logs_warnings_and_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(Settings, "config_warnings", lambda _self: ["warn-1"])
    monkeypatch.setattr(Settings, "validate_config", lambda _self: ["err-1"])

    with caplog.at_level("WARNING"):
        assert main_mod.validate_configuration() is False
    assert "Configuration warning: warn-1" in caplog.text
    assert "Configuration error: err-1" in caplog.text

    monkeypatch.setattr(Settings, "config_warnings", lambda _self: [])
    monkeypatch.setattr(Settings, "validate_config", lambda _self: [])
    assert main_mod.validate_configuration() is True


def test_create_bot_passes_approved_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_mod.settings.bot, "webex_access_token", "tok-123")
    monkeypatch.setattr(main_mod.settings.bot, "display_name", "AI Helper")
    monkeypatch.setattr(main_mod.settings.bot, "approved_users", ["u@example.com"])
    monkeypatch.setattr(main_mod.settings.bot, "approved_domains", ["example.com"])
    monkeypatch.setattr(main_mod.settings.bot, "approved_rooms", ["r1", "r2"])

    with patch("src.main.WebexBot") as mock_bot_cls:
        main_mod.create_bot()

    mock_bot_cls.assert_called_once_with(
        teams_bot_token="tok-123",
        bot_name="AI Helper",
        include_demo_commands=False,
        threads=True,
        approved_users=["u@example.com"],
        approved_domains=["example.com"],
        approved_rooms=["r1", "r2"],
    )

    # Without optional access-control filters
    monkeypatch.setattr(main_mod.settings.bot, "approved_users", [])
    monkeypatch.setattr(main_mod.settings.bot, "approved_domains", [])
    monkeypatch.setattr(main_mod.settings.bot, "approved_rooms", [])
    with patch("src.main.WebexBot") as mock_bot_cls:
        main_mod.create_bot()

    mock_bot_cls.assert_called_once_with(
        teams_bot_token="tok-123",
        bot_name="AI Helper",
        include_demo_commands=False,
        threads=True,
    )


@pytest.mark.asyncio
async def test_setup_mcp_servers_disabled_empty_and_partial_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_mod.settings.mcp, "enabled", False)
    assert await main_mod.setup_mcp_servers() is None

    monkeypatch.setattr(main_mod.settings.mcp, "enabled", True)
    monkeypatch.setattr(main_mod.settings.mcp, "servers", [])
    assert await main_mod.setup_mcp_servers() is None

    servers = [
        MCPServerConfig(name="s1", url="http://s1/mcp", enabled=True),
        MCPServerConfig(name="s2", url="http://s2/mcp", enabled=True),
    ]
    monkeypatch.setattr(main_mod.settings.mcp, "servers", servers)

    # All fail -> None
    fake_fail = MagicMock()
    fake_fail.connect = AsyncMock(side_effect=[False, RuntimeError("boom")])
    with patch("src.main.MCPMultiClient", return_value=fake_fail):
        assert await main_mod.setup_mcp_servers() is None

    # One succeeds -> returns client
    fake_ok = MagicMock()
    fake_ok.connect = AsyncMock(side_effect=[True, False])
    fake_ok.get_server_tools.return_value = ["t1"]
    fake_ok.available_tools = ["t1"]
    with patch("src.main.MCPMultiClient", return_value=fake_ok):
        client = await main_mod.setup_mcp_servers()
    assert client is fake_ok


def test_setup_commands_resolves_display_name(
    monkeypatch: pytest.MonkeyPatch, db_path: str
) -> None:
    monkeypatch.setattr(main_mod.settings.conversation, "db_path", db_path)
    monkeypatch.setattr(main_mod.settings.conversation, "enable_persistence", False)
    monkeypatch.setattr(main_mod.settings.bot, "name", "Assistant")

    fake_bot = MagicMock()
    fake_bot.commands = {"old": MagicMock()}
    fake_bot.teams.people.me.return_value = types.SimpleNamespace(
        displayName="  Webex AI Bot  "
    )

    conv_mgr = main_mod.setup_commands(fake_bot, mcp_client=None)
    assert conv_mgr is not None
    assert fake_bot.commands == {}
    added_cmd = fake_bot.add_command.call_args.args[0]
    assert isinstance(added_cmd, AICommand)
    assert added_cmd.bot_name == "Webex AI Bot"
    assert fake_bot.help_command is added_cmd

    # When people.me() raises, falls back to configured BOT_NAME
    fake_bot.teams.people.me.side_effect = RuntimeError("api down")
    main_mod.setup_commands(fake_bot, mcp_client=None)
    fallback_cmd = fake_bot.add_command.call_args.args[0]
    assert fallback_cmd.bot_name == "Assistant"

    # Explicit BOT_NAME skips people.me() and logs MCP tool count when client passed
    monkeypatch.setattr(main_mod.settings.bot, "name", "ExplicitBot")
    fake_bot.teams.people.me.reset_mock()
    fake_mcp = cast(MCPMultiClient, types.SimpleNamespace(available_tools=["t1", "t2"]))
    main_mod.setup_commands(fake_bot, mcp_client=fake_mcp)
    fake_bot.teams.people.me.assert_not_called()
    assert fake_bot.add_command.call_args.args[0].bot_name == "ExplicitBot"


def test_main_validation_failure_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.main.init_sentry", lambda: False)
    monkeypatch.setattr("src.main.validate_configuration", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()
    assert exc_info.value.code == 1


def test_main_happy_path_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, db_path: str
) -> None:
    monkeypatch.setattr(main_mod.settings.mcp, "enabled", True)
    monkeypatch.setattr(main_mod.settings.conversation, "db_path", db_path)
    monkeypatch.setattr(main_mod.settings.conversation, "retention_days", 7)

    fake_mcp = MagicMock()
    fake_mcp.is_connected = True
    fake_mcp.disconnect = AsyncMock()

    fake_bot = MagicMock()
    fake_conv_mgr = MagicMock()
    fake_conv_mgr.initialize = AsyncMock()

    monkeypatch.setattr("src.main.init_sentry", lambda: True)
    monkeypatch.setattr("src.main.validate_configuration", lambda: True)
    monkeypatch.setattr("src.main.setup_mcp_servers", AsyncMock(return_value=fake_mcp))
    monkeypatch.setattr("src.main.create_bot", lambda: fake_bot)
    monkeypatch.setattr("src.main.setup_commands", lambda *_a: fake_conv_mgr)

    main_mod.main()

    fake_conv_mgr.initialize.assert_awaited_once()
    fake_conv_mgr.start_cleanup.assert_called_once_with(retention_days=7)
    fake_bot.run.assert_called_once()
    fake_mcp.disconnect.assert_awaited_once()


def test_main_keyboard_interrupt_and_unhandled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_mod.settings.mcp, "enabled", False)
    monkeypatch.setattr("src.main.init_sentry", lambda: False)
    monkeypatch.setattr("src.main.validate_configuration", lambda: True)

    fake_conv_mgr = MagicMock()
    fake_conv_mgr.initialize = AsyncMock(side_effect=RuntimeError("db init fail"))
    fake_bot = MagicMock()
    fake_bot.run.side_effect = KeyboardInterrupt()

    monkeypatch.setattr("src.main.create_bot", lambda: fake_bot)
    monkeypatch.setattr("src.main.setup_commands", lambda *_a: fake_conv_mgr)

    # KeyboardInterrupt exits cleanly without SystemExit(1)
    main_mod.main()

    # Unhandled exception captures to Sentry and exits with 1 (and logs disconnect failure)
    fake_mcp = MagicMock()
    fake_mcp.is_connected = True
    fake_mcp.disconnect = AsyncMock(side_effect=RuntimeError("disconnect fail"))
    monkeypatch.setattr(main_mod.settings.mcp, "enabled", True)
    monkeypatch.setattr("src.main.setup_mcp_servers", AsyncMock(return_value=fake_mcp))
    fake_bot.run.side_effect = RuntimeError("fatal bot crash")
    with (
        patch("src.main.capture_exception") as mock_cap,
        pytest.raises(SystemExit) as exc_info,
    ):
        main_mod.main()
    assert exc_info.value.code == 1
    mock_cap.assert_called_once()
    fake_mcp.disconnect.assert_awaited_once()
