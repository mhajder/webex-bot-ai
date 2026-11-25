"""Tests for ConversationManager and rolling summarization in src/conversation/manager.py."""

import asyncio
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import litellm
import pytest

from src.config import settings
from src.conversation.manager import (
    _SUMMARY_BACKOFF_BASE_SECONDS,
    _SUMMARY_BACKOFF_MAX_SECONDS,
    ConversationManager,
)
from src.conversation.types import Message


def test_from_settings_creates_configured_manager(
    db_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings.conversation, "max_history_messages", 12)
    monkeypatch.setattr(settings.conversation, "timeout_hours", 2)
    monkeypatch.setattr(settings.conversation, "db_path", db_path)
    monkeypatch.setattr(settings.conversation, "enable_persistence", False)
    monkeypatch.setattr(settings.conversation, "enable_summarization", False)
    monkeypatch.setattr(settings.conversation, "summary_threshold", 15)
    monkeypatch.setattr(settings.conversation, "keep_recent_messages", 5)
    monkeypatch.setattr(settings.conversation, "summary_model", "gpt-4o-mini")

    manager = ConversationManager.from_settings()
    assert manager.max_messages == 12
    assert manager.timeout_seconds == 7200
    assert manager.enable_persistence is False
    assert manager.enable_summarization is False
    assert manager.summary_threshold == 15
    assert manager.keep_recent_messages == 5
    assert manager.summary_model == "gpt-4o-mini"


def test_add_message_trimming_and_api_consolidation() -> None:
    manager = ConversationManager(
        max_messages=3,
        enable_persistence=False,
        enable_summarization=False,
    )
    # Empty thread ID is ignored
    manager.add_message("", "user", "ignored")
    assert manager.get_thread_count() == 0
    assert manager.has_history("") is False
    assert manager.get_history("") == []
    assert manager.get_thread_summary("") is None

    manager.add_user_message("t1", "u1")
    manager.add_user_message("t1", "u2")
    manager.add_assistant_message("t1", "a1")
    manager.add_assistant_message("t1", "a2")

    # Trimmed to max_messages=3 -> [u2, a1, a2]
    history = manager.get_history("t1")
    assert [m.content for m in history] == ["u2", "a1", "a2"]
    assert manager.has_history("t1") is True
    assert manager.get_thread_count() == 1
    assert manager.get_message_count("t1") == 3

    manager._summaries["t1"] = "Earlier summary"
    api_msgs = manager.get_messages_for_api("t1", system_prompt="Be helpful")
    assert api_msgs == [
        {"role": "system", "content": "Be helpful"},
        {
            "role": "system",
            "content": "Summary of previous conversation in this thread:\nEarlier summary",
        },
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a1\n\na2"},
    ]


def test_add_message_caps_at_twice_summary_threshold() -> None:
    manager = ConversationManager(
        enable_persistence=False,
        enable_summarization=True,
        summary_threshold=5,
        keep_recent_messages=2,
    )
    # Prevent background summarization from mutating history during test
    manager._summary_retry_at = time.time() + 9999
    for i in range(15):
        manager.add_user_message("t1", f"m{i}")

    assert manager.get_message_count("t1") == 10
    assert [m.content for m in manager.get_history("t1")][:2] == ["m5", "m6"]


@pytest.mark.asyncio
async def test_persistence_sync_and_async_reload_and_clear(db_path: str) -> None:
    manager = ConversationManager(
        max_messages=50,
        enable_persistence=True,
        enable_summarization=False,
        db_path=db_path,
    )
    await manager.initialize()

    # Inside running event loop -> spawns background persistence task
    manager.add_user_message("t1", "from async", room_id="r1")
    pending = manager._pending_tasks_on_current_loop()
    assert pending
    await asyncio.gather(*pending)

    # From a worker thread without a running loop -> sync persistence fallback
    done = threading.Event()

    def worker() -> None:
        manager.add_assistant_message("t1", "from thread", room_id="r1")
        done.set()

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)
    assert done.is_set()

    # Cold-start new manager reloads history and summary from SQLite
    assert manager.store is not None
    await manager.store.save_thread_summary("t1", "Persisted summary")

    reloaded = ConversationManager(
        max_messages=50,
        enable_persistence=True,
        enable_summarization=False,
        db_path=db_path,
    )
    assert reloaded.get_thread_summary("t1") == "Persisted summary"
    history = reloaded.get_history("t1")
    assert [m.content for m in history] == ["from async", "from thread"]

    hits = reloaded.search_room_history_sync("r1", "async")
    assert len(hits) == 1

    await reloaded.clear_thread_async("t1")
    assert reloaded.has_history("t1") is False
    assert reloaded.get_history("t1") == []

    # Sync clear_thread also removes from DB
    reloaded.add_user_message("t2", "temp", room_id="r1")
    await asyncio.gather(*reloaded._pending_tasks_on_current_loop())
    reloaded.clear_thread("t2")
    assert reloaded.store is not None
    assert await reloaded.store.load_thread("t2") == []


@pytest.mark.asyncio
async def test_timeout_expiration_in_memory_and_persisted(db_path: str) -> None:
    manager = ConversationManager(
        timeout_seconds=10,
        enable_persistence=True,
        enable_summarization=False,
        db_path=db_path,
    )
    await manager.initialize()
    manager.add_user_message("t-stale", "old question", room_id="r1")
    await asyncio.gather(*manager._pending_tasks_on_current_loop())

    # Simulate in-memory timeout
    manager._timestamps["t-stale"] = time.time() - 20
    assert manager.get_history("t-stale") == []
    assert manager.has_history("t-stale") is False

    # Simulate cold-start DB reload where stored message is older than timeout
    assert manager.store is not None
    await manager.store.save_message(
        "t-db-stale",
        Message(role="user", content="very old", timestamp=time.time() - 100),
        room_id="r1",
    )
    await manager.store.save_thread_summary("t-db-stale", "old summary")

    reloaded = ConversationManager(
        timeout_seconds=10,
        enable_persistence=True,
        enable_summarization=False,
        db_path=db_path,
    )
    assert reloaded.get_history("t-db-stale") == []
    assert reloaded.get_thread_summary("t-db-stale") is None


def test_cleanup_stale_threads_and_start_cleanup(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    manager = ConversationManager(timeout_seconds=172800, enable_persistence=False)
    manager.add_user_message("live", "fresh")
    manager.add_user_message("stale", "old")
    manager._timestamps["stale"] = time.time() - 200000
    manager._generations["orphan"] = 3

    removed = manager.cleanup_stale_threads()
    assert removed == 1
    assert manager.has_history("live") is True
    assert manager.has_history("stale") is False
    assert "orphan" not in manager._generations

    # start_cleanup clamps retention_days < ceil(timeout_seconds / 86400)
    started_threads: list[Any] = []

    class FakeThread:
        def __init__(self, target: Any, name: str, daemon: bool) -> None:
            started_threads.append((target, name, daemon))

        def start(self) -> None:
            pass

    monkeypatch.setattr("src.conversation.manager.threading.Thread", FakeThread)
    with caplog.at_level("WARNING"):
        manager.start_cleanup(retention_days=1, interval_seconds=60)
    assert "using 2 days" in caplog.text
    assert started_threads[0][1] == "conversation-cleanup"
    assert started_threads[0][2] is True


@pytest.mark.asyncio
async def test_summarization_sync_and_async_callables_and_persistence(
    db_path: str,
) -> None:
    prompts: list[str] = []

    async def async_summarizer(prompt: str) -> str:
        prompts.append(prompt)
        return "  Async summary of conversation  "

    manager = ConversationManager(
        enable_persistence=True,
        enable_summarization=True,
        summary_threshold=4,
        keep_recent_messages=2,
        db_path=db_path,
        summary_llm_callable=async_summarizer,
    )
    await manager.initialize()

    # Add 5 messages; keep_recent=2 -> 3 summarized, 2 retained
    for i in range(5):
        role = "user" if i % 2 == 0 else "assistant"
        manager._history["t1"].append(
            Message(role=role, content=f"msg-{i}", timestamp=float(100 + i))
        )
        manager._timestamps["t1"] = time.time()
        assert manager.store is not None
        await manager.store.save_message("t1", manager._history["t1"][-1], room_id="r1")

    # Also store an older DB message dropped from memory to verify merge behavior
    assert manager.store is not None
    await manager.store.save_message(
        "t1",
        Message(role="user", content="dropped-from-memory", timestamp=50.0),
        room_id="r1",
    )

    summary = await manager.summarize_thread_history("t1")
    assert summary == "Async summary of conversation"
    assert "dropped-from-memory" in prompts[0]
    assert [m.content for m in manager.get_history("t1")] == ["msg-3", "msg-4"]
    assert (
        await manager.store.get_thread_summary("t1") == "Async summary of conversation"
    )
    assert len(await manager.store.load_thread("t1")) == 2

    # Rolling second pass incorporates Existing Conversation Summary
    sync_prompts: list[str] = []
    manager._summary_llm_callable = lambda p: (
        sync_prompts.append(p),
        "Updated rolling summary",
    )[1]
    for i in range(5, 8):
        manager._history["t1"].append(
            Message(role="user", content=f"msg-{i}", timestamp=float(100 + i))
        )
    summary2 = await manager.summarize_thread_history("t1")
    assert summary2 == "Updated rolling summary"
    assert (
        "Existing Conversation Summary:\nAsync summary of conversation"
        in sync_prompts[0]
    )


@pytest.mark.asyncio
async def test_summarization_guards_backoff_and_generation_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ConversationManager(
        enable_persistence=False,
        enable_summarization=True,
        summary_threshold=4,
        keep_recent_messages=2,
        summary_llm_callable=lambda _p: "",
    )
    assert await manager.summarize_thread_history("") is None

    # Not enough messages to summarize
    manager.add_user_message("t1", "one")
    assert await manager.summarize_thread_history("t1") is None

    # Active summarization guard
    manager._active_summarizations.add("t1")
    assert await manager.summarize_thread_history("t1") is None
    manager._active_summarizations.clear()

    # Empty summary triggers backoff
    for i in range(4):
        manager._history["t1"].append(
            Message(role="user", content=f"m{i}", timestamp=float(i))
        )
    assert await manager.summarize_thread_history("t1") is None
    assert manager._summary_failures == 1
    assert manager._summary_retry_at >= time.time() + _SUMMARY_BACKOFF_BASE_SECONDS - 5

    # Backoff caps at _SUMMARY_BACKOFF_MAX_SECONDS
    manager._summary_failures = 10
    manager._record_summary_result(success=False)
    assert manager._summary_retry_at <= time.time() + _SUMMARY_BACKOFF_MAX_SECONDS + 1

    # trigger_summarization_task skips during backoff
    called: list[str] = []
    monkeypatch.setattr(
        manager, "summarize_thread_history", lambda tid: called.append(tid)
    )
    manager.trigger_summarization_task("t1")
    assert called == []

    # Exception in callable also triggers backoff
    manager2 = ConversationManager(
        enable_persistence=False,
        enable_summarization=True,
        summary_threshold=3,
        keep_recent_messages=1,
        summary_llm_callable=lambda _p: (_ for _ in ()).throw(RuntimeError("llm down")),
    )
    for i in range(3):
        manager2._history["t2"].append(
            Message(role="user", content=f"m{i}", timestamp=float(i))
        )
    assert await manager2.summarize_thread_history("t2") is None
    assert manager2._summary_failures == 1

    # Clearing thread while summary LLM call is running discards result
    cleared_during_call = False

    async def slow_summarizer(_p: str) -> str:
        nonlocal cleared_during_call
        manager3.clear_thread("t3")
        cleared_during_call = True
        return "Stale summary"

    manager3 = ConversationManager(
        enable_persistence=False,
        enable_summarization=True,
        summary_threshold=3,
        keep_recent_messages=1,
        summary_llm_callable=slow_summarizer,
    )
    for i in range(3):
        manager3._history["t3"].append(
            Message(role="user", content=f"m{i}", timestamp=float(i))
        )
    assert await manager3.summarize_thread_history("t3") is None
    assert cleared_during_call is True
    assert manager3.get_thread_summary("t3") is None


def test_summary_credentials_and_generate_summary_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.llm, "model", "ollama_chat/llama3.2")
    monkeypatch.setattr(settings.llm, "api_key", "main-key")
    monkeypatch.setattr(settings.llm, "api_base", "http://ollama:11434")
    monkeypatch.setattr(settings.conversation, "summary_api_key", None)
    monkeypatch.setattr(settings.conversation, "summary_api_base", None)

    # Same provider (ollama == ollama_chat) reuses main LLM credentials
    assert ConversationManager._summary_credentials("ollama/qwen2.5") == (
        "main-key",
        "http://ollama:11434",
    )
    # Different provider does NOT reuse main credentials
    assert ConversationManager._summary_credentials("anthropic/claude-3-5-sonnet") == (
        None,
        None,
    )

    # Explicit summary credentials win
    monkeypatch.setattr(settings.conversation, "summary_api_key", "sum-key")
    monkeypatch.setattr(
        settings.conversation, "summary_api_base", "https://sum.invalid"
    )
    assert ConversationManager._summary_credentials("anthropic/claude-3-5-sonnet") == (
        "sum-key",
        "https://sum.invalid",
    )

    manager = ConversationManager(
        enable_persistence=False, summary_model="ollama/qwen2.5"
    )
    fake_resp = MagicMock(spec=litellm.ModelResponse)
    fake_resp.choices = [MagicMock(message=MagicMock(content="  Concise summary  "))]

    with patch("litellm.completion", return_value=fake_resp) as mock_comp:
        result = manager._generate_summary_llm("Summarize this")

    assert result == "Concise summary"
    mock_comp.assert_called_once()
    kwargs = mock_comp.call_args.kwargs
    assert kwargs["model"] == "ollama/qwen2.5"
    assert kwargs["api_key"] == "sum-key"
    assert kwargs["api_base"] == "https://sum.invalid"

    with patch("litellm.completion", return_value={"not": "ModelResponse"}):
        assert manager._generate_summary_llm("Summarize this") == ""

    # Fallback when litellm.get_llm_provider raises
    with patch("litellm.get_llm_provider", side_effect=RuntimeError("unknown")):
        monkeypatch.setattr(settings.llm, "model", "custom/m1")
        monkeypatch.setattr(settings.conversation, "summary_api_key", None)
        monkeypatch.setattr(settings.conversation, "summary_api_base", None)
        assert ConversationManager._summary_credentials("custom/m2") == (
            "main-key",
            "http://ollama:11434",
        )


@pytest.mark.asyncio
async def test_manager_trigger_sync_thread_default_llm_and_db_errors(
    db_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Default _generate_summary_llm path inside summarize_thread_history
    manager = ConversationManager(
        enable_persistence=True,
        enable_summarization=True,
        summary_threshold=3,
        keep_recent_messages=1,
        db_path=db_path,
    )
    await manager.initialize()
    for i in range(3):
        manager._history["t-llm"].append(
            Message(role="user", content=f"msg {i}", timestamp=float(i))
        )
        manager._timestamps["t-llm"] = time.time()

    fake_resp = MagicMock(spec=litellm.ModelResponse)
    fake_resp.choices = [MagicMock(message=MagicMock(content="LLM summary"))]
    with patch("litellm.completion", return_value=fake_resp):
        assert await manager.summarize_thread_history("t-llm") == "LLM summary"

    # Active summarization skip and sync thread scheduling
    manager._active_summarizations.add("t-sync")
    manager.trigger_summarization_task("t-sync")
    manager._active_summarizations.clear()

    started: list[Any] = []

    class FakeThread:
        def __init__(
            self, target: Any, args: tuple[Any, ...], name: str, daemon: bool
        ) -> None:
            started.append((target, args, name, daemon))

        def start(self) -> None:
            target, args, *_ = started[-1]
            target(*args)

    done = threading.Event()

    def run_in_worker() -> None:
        with patch("src.conversation.manager.threading.Thread", FakeThread):
            manager.trigger_summarization_task("t-empty")
        done.set()

    t = threading.Thread(target=run_in_worker)
    t.start()
    t.join(timeout=5)
    assert done.is_set()
    assert len(started) == 1

    # DB error paths degrade gracefully
    broken_store = MagicMock()
    broken_store.get_thread_summary_sync.side_effect = RuntimeError("db err")
    broken_store.load_thread_sync.side_effect = RuntimeError("db err")
    broken_store.archive_thread_sync.side_effect = RuntimeError("db err")
    broken_store.delete_thread_sync.side_effect = RuntimeError("db err")
    broken_store.delete_thread = MagicMock(side_effect=RuntimeError("db err"))
    broken_store.save_message_sync.side_effect = RuntimeError("db err")

    err_mgr = ConversationManager(
        enable_persistence=True, enable_summarization=False, db_path=db_path
    )
    err_mgr.store = broken_store

    # Sync save failure in worker thread
    worker_done = threading.Event()
    threading.Thread(
        target=lambda: (err_mgr.add_user_message("t-err", "hi"), worker_done.set())
    ).start()
    worker_done.wait(timeout=5)

    assert err_mgr.get_thread_summary("t-err-2") is None
    assert err_mgr.get_history("t-err-2") == []
    err_mgr._timestamps["t-err"] = time.time() - 999999
    assert err_mgr._expire_if_stale("t-err") is True
    err_mgr.clear_thread("t-err")
    await err_mgr.clear_thread_async("t-err")
    await err_mgr.clear_thread_async("")

    # start_cleanup with retention_days=0 runs cleanup_stale_threads in thread
    class RunOnceSleep(BaseException):
        pass

    cleanup_calls: list[int] = []
    monkeypatch.setattr(
        err_mgr, "cleanup_stale_threads", lambda: (cleanup_calls.append(1), 0)[1]
    )
    monkeypatch.setattr(
        "src.conversation.manager.time.sleep",
        lambda _s: (_ for _ in ()).throw(RunOnceSleep()),
    )

    captured_target: list[Any] = []

    class ImmediateThread:
        def __init__(self, target: Any, name: str, daemon: bool) -> None:  # noqa: ARG002
            captured_target.append(target)

        def start(self) -> None:
            with pytest.raises(RunOnceSleep):
                captured_target[0]()

    monkeypatch.setattr("src.conversation.manager.threading.Thread", ImmediateThread)
    err_mgr.start_cleanup(retention_days=0)
    assert cleanup_calls == [1]
