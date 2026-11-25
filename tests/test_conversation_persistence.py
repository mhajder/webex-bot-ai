"""Tests for SQLite persistence and FTS5 search in src/conversation/."""

import contextlib
import sqlite3
import types
from unittest.mock import patch

import pytest

from src.conversation.persistence import ConversationStore, _build_fts_query
from src.conversation.types import Message


def test_message_to_api_format() -> None:
    msg = Message(role="user", content="hello", timestamp=123.0, metadata={"k": "v"})
    assert msg.to_api_format() == {"role": "user", "content": "hello"}


def test_build_fts_query_filters_stopwords_quotes_operators_and_caps_terms() -> None:
    assert _build_fts_query("") == ("", "")
    assert _build_fts_query("the and or is a") == ("", "")

    fts_query, like_term = _build_fts_query(
        "Check OR status for switch-01.core and NOT router-02 AND switch-01.core"
    )
    assert fts_query == '"check" OR "status" OR "switch-01.core" OR "router-02"'
    assert like_term == "switch-01.core"

    # Capped at 16 terms
    many_terms = " ".join(f"keyword{i}" for i in range(25))
    capped_fts, _ = _build_fts_query(many_terms)
    assert len(capped_fts.split(" OR ")) == 16


@pytest.mark.asyncio
async def test_init_db_migrates_legacy_schema_and_backfills_fts(db_path: str) -> None:
    # Create legacy schema without room_id, thread_summary, archived, or FTS5
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "CREATE TABLE conversations (thread_id TEXT PRIMARY KEY, created_at TIMESTAMP, updated_at TIMESTAMP)"
        )
        conn.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT
            )
            """
        )
        conn.execute("INSERT INTO conversations (thread_id) VALUES ('t-legacy')")
        conn.execute(
            "INSERT INTO messages (thread_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
            ("t-legacy", "user", "legacy switch configuration", 100.0, None),
        )
        conn.commit()

    store = ConversationStore(db_path)
    await store.init_db()
    # Idempotent second init
    await store.init_db()

    loaded = await store.load_thread("t-legacy")
    assert len(loaded) == 1
    assert loaded[0].content == "legacy switch configuration"

    # Backfilled FTS row should be searchable if room_id is set later or via FTS table
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
        assert {"room_id", "thread_summary"}.issubset(cols)
        msg_cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        assert {"room_id", "archived"}.issubset(msg_cols)
        fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        assert fts_count == 1


@pytest.mark.asyncio
async def test_async_crud_stats_and_cleanup(store: ConversationStore) -> None:
    m1 = Message(role="user", content="first", timestamp=10.0, metadata={"a": 1})
    m2 = Message(role="assistant", content="second", timestamp=20.0)
    m3 = Message(role="user", content="third", timestamp=30.0)

    await store.save_message("t1", m1, room_id="r1")
    # Saving without room_id preserves existing room_id via COALESCE
    await store.save_message("t1", m2, room_id=None)
    await store.save_message("t1", m3, room_id="r1")
    await store.save_message(
        "t2", Message(role="user", content="other", timestamp=15.0), room_id="r2"
    )

    assert len(await store.load_thread("t1")) == 3
    limited = await store.load_thread("t1", limit=2)
    assert [m.content for m in limited] == ["second", "third"]

    thread_ids = await store.get_all_thread_ids()
    assert set(thread_ids) == {"t1", "t2"}

    stats = await store.get_thread_stats("t1")
    assert stats is not None
    assert stats["message_count"] == 3
    assert await store.get_thread_stats("missing") is None

    # Age t2 beyond 30 days and run cleanup_old_threads
    with contextlib.closing(sqlite3.connect(store.db_path)) as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = datetime('now', '-40 days') WHERE thread_id = 't2'"
        )
        conn.commit()

    deleted = await store.cleanup_old_threads(days=30)
    assert deleted == 1
    assert await store.get_all_thread_ids() == ["t1"]
    assert await store.load_thread("t2") == []

    await store.delete_thread("t1")
    assert await store.get_all_thread_ids() == []
    assert await store.load_thread("t1") == []


@pytest.mark.asyncio
async def test_summaries_and_archiving(store: ConversationStore) -> None:
    await store.save_message(
        "t1", Message(role="user", content="msg1", timestamp=100.0), room_id="r1"
    )
    await store.save_message(
        "t1", Message(role="assistant", content="msg2", timestamp=200.0), room_id="r1"
    )
    await store.save_message(
        "t1", Message(role="user", content="msg3", timestamp=300.0), room_id="r1"
    )

    await store.save_thread_summary("t1", "Summary v1")
    assert await store.get_thread_summary("t1") == "Summary v1"
    assert store.get_thread_summary_sync("t1") == "Summary v1"

    store.save_thread_summary_sync("t1", "Summary v2")
    assert await store.get_thread_summary("t1") == "Summary v2"
    assert await store.get_thread_summary("missing") is None

    archived_count = await store.archive_messages_until("t1", until=200.0)
    assert archived_count == 2
    remaining = await store.load_thread("t1")
    assert [m.content for m in remaining] == ["msg3"]

    # inactive_since earlier than newest message (300.0) should NOT archive
    assert store.archive_thread_sync("t1", inactive_since=250.0) is False
    assert store.get_thread_summary_sync("t1") == "Summary v2"

    # inactive_since after newest message archives remaining message & clears summary
    assert store.archive_thread_sync("t1", inactive_since=350.0) is True
    assert await store.load_thread("t1") == []
    assert store.get_thread_summary_sync("t1") is None
    # Archiving again returns False (nothing left to archive)
    assert store.archive_thread_sync("t1") is False


@pytest.mark.asyncio
async def test_room_search_fts_and_like_fallback(store: ConversationStore) -> None:
    await store.save_message(
        "t1",
        Message(
            role="user", content="Configured BGP neighbor 10.0.0.1", timestamp=10.0
        ),
        room_id="room-a",
    )
    await store.save_message(
        "t2",
        Message(
            role="assistant", content="OSPF adjacency flapping on eth1", timestamp=20.0
        ),
        room_id="room-a",
    )
    await store.save_message(
        "t3",
        Message(role="user", content="BGP neighbor in another room", timestamp=30.0),
        room_id="room-b",
    )
    # Archived messages stay searchable in room history
    await store.archive_messages_until("t1", until=15.0)

    assert await store.search_room_history("", "BGP") == []
    assert await store.search_room_history("room-a", "   ") == []
    assert await store.search_room_history("room-a", "the and or") == []
    assert store.search_room_history_sync("room-a", "the and or") == []

    async_hits = await store.search_room_history("room-a", "BGP neighbor")
    assert len(async_hits) == 1
    assert async_hits[0].content == "Configured BGP neighbor 10.0.0.1"
    assert async_hits[0].metadata["thread_id"] == "t1"

    sync_hits = store.search_room_history_sync("room-a", "flapping eth1")
    assert len(sync_hits) == 1
    assert sync_hits[0].metadata["thread_id"] == "t2"

    # Force LIKE fallback by dropping the FTS5 table
    with contextlib.closing(sqlite3.connect(store.db_path)) as conn:
        conn.execute("DROP TRIGGER IF EXISTS messages_ai")
        conn.execute("DROP TRIGGER IF EXISTS messages_ad")
        conn.execute("DROP TABLE messages_fts")
        conn.commit()

    fallback_async = await store.search_room_history("room-a", "10.0.0.1")
    assert len(fallback_async) == 1
    assert fallback_async[0].content == "Configured BGP neighbor 10.0.0.1"

    fallback_sync = store.search_room_history_sync("room-a", "adjacency")
    assert len(fallback_sync) == 1
    assert fallback_sync[0].content == "OSPF adjacency flapping on eth1"


def test_sync_methods_and_uninitialized_db(
    db_path: str, store: ConversationStore
) -> None:
    msg = Message(role="user", content="sync msg", timestamp=50.0)
    store.save_message_sync("t-sync", msg, room_id="r-sync")
    store.save_message_sync(
        "t-sync",
        Message(role="assistant", content="reply", timestamp=60.0),
        room_id="r-sync",
    )

    loaded = store.load_thread_sync("t-sync")
    assert [m.content for m in loaded] == ["sync msg", "reply"]
    assert [m.content for m in store.load_thread_sync("t-sync", limit=1)] == ["reply"]

    store.delete_thread_sync("t-sync")
    assert store.load_thread_sync("t-sync") == []

    # Operations against a fresh uninitialized DB path degrade cleanly
    uninit = ConversationStore(f"{db_path}.missing")
    uninit.save_message_sync("t", msg)
    assert uninit.load_thread_sync("t") == []
    uninit.delete_thread_sync("t")
    assert uninit.archive_thread_sync("t") is False
    assert uninit.get_thread_summary_sync("t") is None
    assert uninit.search_room_history_sync("r", "query") == []


@pytest.mark.asyncio
async def test_store_error_paths(db_path: str) -> None:
    bad_store = ConversationStore(db_path)
    # Calling async methods before init_db raises OperationalError
    with pytest.raises(sqlite3.OperationalError):
        await bad_store.load_thread("t")
    with pytest.raises(sqlite3.OperationalError):
        await bad_store.save_message("t", Message(role="user", content="x"))
    with pytest.raises(sqlite3.OperationalError):
        await bad_store.delete_thread("t")
    with pytest.raises(sqlite3.OperationalError):
        await bad_store.get_all_thread_ids()
    with pytest.raises(sqlite3.OperationalError):
        await bad_store.get_thread_stats("t")
    with pytest.raises(sqlite3.OperationalError):
        await bad_store.cleanup_old_threads()

    # Search/summary/archive helper error paths return safe defaults
    assert await bad_store.search_room_history("r", "query") == []
    assert await bad_store.get_thread_summary("t") is None
    assert await bad_store.archive_messages_until("t", 10.0) == 0
    await bad_store.save_thread_summary("t", "summary")

    # Sync non-"no such table" OperationalError logs warning without raising
    fake_conn = types.SimpleNamespace(
        cursor=lambda: types.SimpleNamespace(
            execute=lambda *_a, **_kw: (_ for _ in ()).throw(
                sqlite3.OperationalError("database is locked")
            )
        ),
        close=lambda: None,
    )
    with patch("src.conversation.persistence.sqlite3.connect", return_value=fake_conn):
        bad_store.save_message_sync("t", Message(role="user", content="x"))
