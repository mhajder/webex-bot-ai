"""SQLite-based persistence layer for conversation history."""

import contextlib
import json
import logging
import re
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from src.conversation.types import Message

log = logging.getLogger(__name__)


def _now() -> str:
    """Current UTC time in one consistent text format for TIMESTAMP columns.

    Matches SQLite's ``datetime('now')`` layout (space separator) so string
    comparisons in cleanup queries work, and avoids the default sqlite3
    datetime adapter deprecated since Python 3.12.
    """
    return datetime.now(UTC).isoformat(sep=" ")


# Filler words dropped from search queries; matching them would rank nearly
# every message as relevant once terms are OR-ed
_STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "out",
        "over",
        "own",
        "please",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "thank",
        "thanks",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    ]
)

# Upper bound on OR-ed terms, so a pasted log doesn't build a huge query
_MAX_QUERY_TERMS = 16


def _build_fts_query(query: str) -> tuple[str, str]:
    """Build an FTS5 MATCH expression and a LIKE fallback term from user text.

    Keywords (filler words and single characters dropped) are OR-ed, so a
    message matches if it shares any keyword with the query; results are
    ordered by FTS5's bm25 rank, which favors messages matching more and
    rarer keywords. Each term is quoted as an FTS5 string so punctuation
    (``.``, ``-``) and operator words (AND/OR/NOT/NEAR) don't raise FTS5
    syntax errors. The LIKE fallback uses the longest keyword.
    """
    terms = list(
        dict.fromkeys(
            t
            for t in (t.strip(".-") for t in re.findall(r"[\w.-]+", query.lower()))
            if len(t) > 1 and t not in _STOPWORDS
        )
    )[:_MAX_QUERY_TERMS]
    if not terms:
        return "", ""
    fts_query = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
    return fts_query, max(terms, key=len)


def _row_to_message(row: Sequence[Any]) -> Message:
    """Convert a (role, content, timestamp, metadata, thread_id) search row."""
    return Message(
        role=row[0],
        content=row[1],
        timestamp=row[2],
        metadata={
            **(json.loads(row[3]) if row[3] else {}),
            "thread_id": row[4],
        },
    )


class ConversationStore:
    """Persists and retrieves conversation history to/from SQLite.

    Handles all database operations for storing and loading thread conversations,
    allowing conversation history to survive application restarts.
    """

    def __init__(self, db_path: str = "conversations.db"):
        """Initialize the store with database path.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path

    async def init_db(self) -> None:
        """Initialize database schema if it doesn't exist."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create conversations table
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        thread_id TEXT PRIMARY KEY,
                        room_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

                # Create messages table
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        thread_id TEXT NOT NULL,
                        room_id TEXT,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        metadata TEXT,
                        FOREIGN KEY (thread_id) REFERENCES conversations(thread_id)
                        ON DELETE CASCADE
                    )
                    """
                )

                # Auto-migrate columns if database existed before room_id or thread_summary
                with contextlib.suppress(Exception):
                    await db.execute(
                        "ALTER TABLE conversations ADD COLUMN room_id TEXT"
                    )

                with contextlib.suppress(Exception):
                    await db.execute(
                        "ALTER TABLE conversations ADD COLUMN thread_summary TEXT"
                    )

                with contextlib.suppress(Exception):
                    await db.execute("ALTER TABLE messages ADD COLUMN room_id TEXT")

                # Archived messages (summarized or timed out) stay in the table
                # and the FTS index for room search, but are not loaded back
                # into the thread context.
                with contextlib.suppress(Exception):
                    await db.execute(
                        "ALTER TABLE messages ADD COLUMN archived INTEGER NOT NULL DEFAULT 0"
                    )

                # Create indexes for faster queries
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_thread_id
                    ON messages(thread_id)
                    """
                )
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_room_id
                    ON messages(room_id)
                    """
                )

                # Create SQLite FTS5 Virtual Table for room-wide message search
                try:
                    await db.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                            content,
                            thread_id UNINDEXED,
                            room_id UNINDEXED,
                            role UNINDEXED,
                            tokenize='porter unicode61'
                        );
                        """
                    )

                    # Triggers to keep FTS table synchronized with messages
                    await db.execute(
                        """
                        CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                            INSERT INTO messages_fts(rowid, content, thread_id, room_id, role)
                            VALUES (new.id, new.content, new.thread_id, new.room_id, new.role);
                        END;
                        """
                    )
                    await db.execute(
                        """
                        CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                            DELETE FROM messages_fts WHERE rowid = old.id;
                        END;
                        """
                    )
                    # Backfill rows written before the FTS table/triggers existed
                    await db.execute(
                        """
                        INSERT INTO messages_fts(rowid, content, thread_id, room_id, role)
                        SELECT id, content, thread_id, room_id, role FROM messages
                        WHERE id NOT IN (SELECT rowid FROM messages_fts)
                        """
                    )
                except Exception as fts_err:
                    log.warning(
                        "FTS5 table initialization skipped/not supported: %s", fts_err
                    )

                await db.commit()
                log.info(
                    "Database initialized at %s with FTS5 search support", self.db_path
                )
        except Exception as e:
            log.exception("Failed to initialize database: %s", e)
            raise

    async def save_message(
        self, thread_id: str, message: Message, room_id: str | None = None
    ) -> None:
        """Save a message to the database.

        Args:
            thread_id: The thread identifier.
            message: The message to save.
            room_id: Optional Webex room identifier.
        """
        try:
            now = _now()
            async with aiosqlite.connect(self.db_path) as db:
                # Ensure conversation exists
                await db.execute(
                    """
                    INSERT INTO conversations (thread_id, room_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(thread_id) DO UPDATE SET
                        room_id=COALESCE(excluded.room_id, conversations.room_id),
                        updated_at=excluded.updated_at
                    """,
                    (thread_id, room_id, now, now),
                )

                # Save message
                metadata_json = json.dumps(message.metadata)
                await db.execute(
                    """
                    INSERT INTO messages (thread_id, room_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        room_id,
                        message.role,
                        message.content,
                        message.timestamp,
                        metadata_json,
                    ),
                )

                await db.commit()
                log.debug(
                    "Saved %s message to thread %s (room %s)",
                    message.role,
                    thread_id,
                    room_id,
                )
        except Exception as e:
            log.exception("Failed to save message for thread %s: %s", thread_id, e)
            raise

    async def load_thread(
        self, thread_id: str, limit: int | None = None
    ) -> list[Message]:
        """Load the non-archived messages for a thread from the database.

        Args:
            thread_id: The thread identifier.
            limit: Optional maximum number of most recent messages to load.

        Returns:
            List of Message objects ordered by timestamp.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if limit:
                    query = """
                        SELECT role, content, timestamp, metadata FROM (
                            SELECT role, content, timestamp, metadata
                            FROM messages
                            WHERE thread_id = ? AND archived = 0
                            ORDER BY timestamp DESC
                            LIMIT ?
                        ) ORDER BY timestamp ASC
                    """
                    params = (thread_id, limit)
                else:
                    query = """
                        SELECT role, content, timestamp, metadata
                        FROM messages
                        WHERE thread_id = ? AND archived = 0
                        ORDER BY timestamp ASC
                    """
                    params = (thread_id,)

                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()

                messages = []
                for role, content, timestamp, metadata_json in rows:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    message = Message(
                        role=role,
                        content=content,
                        timestamp=timestamp,
                        metadata=metadata,
                    )
                    messages.append(message)

                log.debug("Loaded %d messages from thread %s", len(messages), thread_id)
                return messages
        except Exception as e:
            log.exception("Failed to load messages for thread %s: %s", thread_id, e)
            raise

    async def delete_thread(self, thread_id: str) -> None:
        """Delete all messages and conversation record for a thread.

        Args:
            thread_id: The thread identifier.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "DELETE FROM messages WHERE thread_id = ?",
                    (thread_id,),
                )
                await db.execute(
                    "DELETE FROM conversations WHERE thread_id = ?",
                    (thread_id,),
                )
                await db.commit()
                log.debug("Deleted conversation for thread %s", thread_id)
        except Exception as e:
            log.exception("Failed to delete thread %s: %s", thread_id, e)
            raise

    async def get_all_thread_ids(self) -> list[str]:
        """Get all thread IDs from the database.

        Returns:
            List of thread identifiers.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT thread_id FROM conversations ORDER BY updated_at DESC"
                )
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            log.exception("Failed to get all thread IDs: %s", e)
            raise

    async def get_thread_stats(self, thread_id: str) -> dict | None:
        """Get statistics about a thread.

        Args:
            thread_id: The thread identifier.

        Returns:
            Dictionary with message_count and timestamps, or None if not found.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM messages WHERE thread_id = ?) as message_count,
                        (SELECT created_at FROM conversations WHERE thread_id = ?) as created_at,
                        (SELECT updated_at FROM conversations WHERE thread_id = ?) as updated_at
                    """,
                    (thread_id, thread_id, thread_id),
                )
                row = await cursor.fetchone()
                if row and row[0]:
                    return {
                        "message_count": row[0],
                        "created_at": row[1],
                        "updated_at": row[2],
                    }
                return None
        except Exception as e:
            log.exception("Failed to get stats for thread %s: %s", thread_id, e)
            raise

    async def cleanup_old_threads(self, days: int = 30) -> int:
        """Delete conversations older than specified days.

        Args:
            days: Number of days to keep.

        Returns:
            Number of threads deleted.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    DELETE FROM conversations
                    WHERE updated_at < datetime('now', '-' || ? || ' days')
                    """,
                    (days,),
                )
                deleted = cursor.rowcount
                # PRAGMA foreign_keys is off, so ON DELETE CASCADE never fires
                await db.execute(
                    """
                    DELETE FROM messages
                    WHERE thread_id NOT IN (SELECT thread_id FROM conversations)
                    """
                )
                await db.commit()
                if deleted > 0:
                    log.info("Cleaned up %d old conversation threads", deleted)
                return deleted
        except Exception as e:
            log.exception("Failed to cleanup old threads: %s", e)
            raise

    async def search_room_history(
        self, room_id: str, query: str, limit: int = 5
    ) -> list[Message]:
        """Search past messages in a Webex room using SQLite FTS5 with LIKE fallback.

        Args:
            room_id: Webex room/space identifier.
            query: Search query terms.
            limit: Maximum matching messages to return.

        Returns:
            List of matching Message objects with metadata.
        """
        if not room_id or not query or not query.strip():
            return []

        fts_query, like_term = _build_fts_query(query)
        if not fts_query:
            return []

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Try FTS5 search first
                try:
                    cursor = await db.execute(
                        """
                        SELECT m.role, m.content, m.timestamp, m.metadata, m.thread_id
                        FROM messages_fts f
                        JOIN messages m ON f.rowid = m.id
                        WHERE f.room_id = ? AND messages_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (room_id, fts_query, limit),
                    )
                    rows = list(await cursor.fetchall())
                    if rows:
                        return [_row_to_message(r) for r in rows]
                except Exception as fts_e:
                    log.debug("FTS5 match failed, falling back to LIKE: %s", fts_e)

                # Fallback to standard LIKE search
                cursor = await db.execute(
                    """
                    SELECT role, content, timestamp, metadata, thread_id
                    FROM messages
                    WHERE room_id = ? AND content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (room_id, f"%{like_term}%", limit),
                )
                rows = await cursor.fetchall()
                return [_row_to_message(r) for r in rows]
        except Exception as e:
            # The query is user text; keep it out of WARNING logs (sent to Sentry)
            log.warning("Failed room search in room %s: %s", room_id, e)
            return []

    def save_message_sync(
        self, thread_id: str, message: Message, room_id: str | None = None
    ) -> None:
        """Synchronously save a message to database (fallback when async is unavailable)."""
        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.cursor()
                now_iso = _now()
                cursor.execute(
                    """
                    INSERT INTO conversations (thread_id, room_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(thread_id) DO UPDATE SET
                        room_id=COALESCE(excluded.room_id, conversations.room_id),
                        updated_at=excluded.updated_at
                    """,
                    (thread_id, room_id, now_iso, now_iso),
                )
                metadata_json = json.dumps(message.metadata)
                cursor.execute(
                    """
                    INSERT INTO messages (thread_id, room_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        room_id,
                        message.role,
                        message.content,
                        message.timestamp,
                        metadata_json,
                    ),
                )
                conn.commit()
                log.debug(
                    "Sync-saved %s message to thread %s (room %s)",
                    message.role,
                    thread_id,
                    room_id,
                )
        except sqlite3.OperationalError as e:
            # Only a missing schema means "not initialized yet"; anything else
            # (e.g. "database is locked") loses the message and must be visible.
            if "no such table" in str(e):
                log.debug("Database not ready for sync save: %s", e)
            else:
                log.warning(
                    "Failed to sync-save message for thread %s: %s", thread_id, e
                )
        except Exception as e:
            log.exception("Failed to sync-save message for thread %s: %s", thread_id, e)

    def search_room_history_sync(
        self, room_id: str, query: str, limit: int = 5
    ) -> list[Message]:
        """Synchronously search past messages in a Webex room using SQLite FTS5 with LIKE fallback."""
        if not room_id or not query or not query.strip():
            return []

        fts_query, like_term = _build_fts_query(query)
        if not fts_query:
            return []

        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.cursor()

                # Try FTS5 search first
                try:
                    cursor.execute(
                        """
                        SELECT m.role, m.content, m.timestamp, m.metadata, m.thread_id
                        FROM messages_fts f
                        JOIN messages m ON f.rowid = m.id
                        WHERE f.room_id = ? AND messages_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (room_id, fts_query, limit),
                    )
                    rows = cursor.fetchall()
                    if rows:
                        return [_row_to_message(r) for r in rows]
                except sqlite3.OperationalError as fts_e:
                    log.debug("FTS search fallback to LIKE: %s", fts_e)

                # Fallback to LIKE search
                cursor.execute(
                    """
                    SELECT role, content, timestamp, metadata, thread_id
                    FROM messages
                    WHERE room_id = ? AND content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (room_id, f"%{like_term}%", limit),
                )
                return [_row_to_message(r) for r in cursor.fetchall()]
        except Exception as e:
            log.warning("Sync room search failed: %s", e)
            return []

    def load_thread_sync(
        self, thread_id: str, limit: int | None = None
    ) -> list[Message]:
        """Synchronously load thread messages from database."""
        messages = []
        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.cursor()
                if limit:
                    query = """
                        SELECT role, content, timestamp, metadata FROM (
                            SELECT role, content, timestamp, metadata
                            FROM messages
                            WHERE thread_id = ? AND archived = 0
                            ORDER BY timestamp DESC
                            LIMIT ?
                        ) ORDER BY timestamp ASC
                    """
                    params: tuple = (thread_id, limit)
                else:
                    query = """
                        SELECT role, content, timestamp, metadata
                        FROM messages
                        WHERE thread_id = ? AND archived = 0
                        ORDER BY timestamp ASC
                    """
                    params = (thread_id,)
                cursor.execute(query, params)
                rows = cursor.fetchall()

                for role, content, timestamp, metadata_json in rows:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    messages.append(
                        Message(
                            role=role,
                            content=content,
                            timestamp=timestamp,
                            metadata=metadata,
                        )
                    )
                return messages
        except sqlite3.OperationalError:
            return []

    def delete_thread_sync(self, thread_id: str) -> None:
        """Synchronously delete thread from database."""
        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
                cursor.execute(
                    "DELETE FROM conversations WHERE thread_id = ?", (thread_id,)
                )
                conn.commit()
                log.debug("Deleted thread %s from database", thread_id)
        except sqlite3.OperationalError:
            pass

    async def save_thread_summary(self, thread_id: str, summary: str) -> None:
        """Save or update the thread summary in database."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO conversations (thread_id, thread_summary, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(thread_id) DO UPDATE SET
                        thread_summary = excluded.thread_summary,
                        updated_at = excluded.updated_at
                    """,
                    (thread_id, summary, _now()),
                )
                await db.commit()
                log.debug("Saved thread summary for thread %s", thread_id)
        except Exception as e:
            log.exception("Failed to save summary for thread %s: %s", thread_id, e)

    def save_thread_summary_sync(self, thread_id: str, summary: str) -> None:
        """Synchronously save or update the thread summary in database."""
        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.cursor()
                now_iso = _now()
                cursor.execute(
                    """
                    INSERT INTO conversations (thread_id, thread_summary, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(thread_id) DO UPDATE SET
                        thread_summary = excluded.thread_summary,
                        updated_at = excluded.updated_at
                    """,
                    (thread_id, summary, now_iso),
                )
                conn.commit()
                log.debug("Sync-saved thread summary for thread %s", thread_id)
        except Exception as e:
            log.exception("Failed to sync-save summary for thread %s: %s", thread_id, e)

    async def get_thread_summary(self, thread_id: str) -> str | None:
        """Get the thread summary from database."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT thread_summary FROM conversations WHERE thread_id = ?",
                    (thread_id,),
                )
                row = await cursor.fetchone()
                return row[0] if row and row[0] else None
        except Exception as e:
            log.exception("Failed to get summary for thread %s: %s", thread_id, e)
            return None

    def get_thread_summary_sync(self, thread_id: str) -> str | None:
        """Synchronously get the thread summary from database."""
        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT thread_summary FROM conversations WHERE thread_id = ?",
                    (thread_id,),
                )
                row = cursor.fetchone()
                return row[0] if row and row[0] else None
        except Exception as e:
            log.exception("Failed to sync-get summary for thread %s: %s", thread_id, e)
            return None

    async def archive_messages_until(self, thread_id: str, until: float) -> int:
        """Archive a thread's messages with timestamp <= ``until`` (after summarization).

        Archived messages are no longer loaded into the thread context but
        remain searchable through room history search.

        Returns:
            Number of messages archived.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    UPDATE messages SET archived = 1
                    WHERE thread_id = ? AND archived = 0 AND timestamp <= ?
                    """,
                    (thread_id, until),
                )
                await db.commit()
                archived = cursor.rowcount
                log.debug("Archived %d old messages for thread %s", archived, thread_id)
                return archived
        except Exception as e:
            log.exception("Failed to archive messages for thread %s: %s", thread_id, e)
            return 0

    def archive_thread_sync(
        self, thread_id: str, inactive_since: float | None = None
    ) -> bool:
        """Archive all messages of a thread and drop its summary.

        Used when a conversation times out, so stale context is not reloaded
        from the database while the messages stay searchable.

        Args:
            thread_id: The thread identifier.
            inactive_since: If given, only archive when the thread's newest
                message is older than this epoch timestamp.

        Returns:
            True if any message or summary was archived (False when the
            thread was already archived, so callers don't log it repeatedly).
        """
        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                if inactive_since is not None:
                    row = conn.execute(
                        "SELECT MAX(timestamp) FROM messages WHERE thread_id = ?",
                        (thread_id,),
                    ).fetchone()
                    if not row or row[0] is None or row[0] >= inactive_since:
                        return False
                changed = conn.execute(
                    "UPDATE messages SET archived = 1 WHERE thread_id = ? AND archived = 0",
                    (thread_id,),
                ).rowcount
                changed += conn.execute(
                    """
                    UPDATE conversations SET thread_summary = NULL
                    WHERE thread_id = ? AND thread_summary IS NOT NULL
                    """,
                    (thread_id,),
                ).rowcount
                conn.commit()
                return changed > 0
        except sqlite3.OperationalError as e:
            log.debug("Database not ready to archive thread %s: %s", thread_id, e)
            return False
