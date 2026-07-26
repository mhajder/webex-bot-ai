"""SQLite-based persistence layer for conversation history."""

import contextlib
import json
import logging
import re
import sqlite3
from datetime import UTC, datetime

import aiosqlite

from src.conversation.types import Message

log = logging.getLogger(__name__)


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
                    (thread_id, room_id, datetime.now(UTC), datetime.now(UTC)),
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
        """Load all messages for a thread from the database.

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
                            WHERE thread_id = ?
                            ORDER BY timestamp DESC
                            LIMIT ?
                        ) ORDER BY timestamp ASC
                    """
                    params = (thread_id, limit)
                else:
                    query = """
                        SELECT role, content, timestamp, metadata
                        FROM messages
                        WHERE thread_id = ?
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
                await db.commit()
                deleted = cursor.rowcount
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

        clean_terms = re.findall(r"[\w.-]+", query)
        clean_query = " ".join(clean_terms) if clean_terms else query.strip()

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
                        (room_id, clean_query, limit),
                    )
                    rows = await cursor.fetchall()
                    if rows:
                        return [
                            Message(
                                role=r[0],
                                content=r[1],
                                timestamp=r[2],
                                metadata={
                                    **(json.loads(r[3]) if r[3] else {}),
                                    "thread_id": r[4],
                                },
                            )
                            for r in rows
                        ]
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
                    (room_id, f"%{clean_query}%", limit),
                )
                rows = await cursor.fetchall()
                return [
                    Message(
                        role=r[0],
                        content=r[1],
                        timestamp=r[2],
                        metadata={
                            **(json.loads(r[3]) if r[3] else {}),
                            "thread_id": r[4],
                        },
                    )
                    for r in rows
                ]
        except Exception as e:
            log.warning("Failed room search for '%s' in room %s: %s", query, room_id, e)
            return []

    def save_message_sync(
        self, thread_id: str, message: Message, room_id: str | None = None
    ) -> None:
        """Synchronously save a message to database (fallback when async is unavailable)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            now_iso = datetime.now(UTC).isoformat()
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
            conn.close()
            log.debug(
                "Sync-saved %s message to thread %s (room %s)",
                message.role,
                thread_id,
                room_id,
            )
        except sqlite3.OperationalError as e:
            log.debug("Database not ready for sync save: %s", e)
        except Exception as e:
            log.exception("Failed to sync-save message for thread %s: %s", thread_id, e)

    def search_room_history_sync(
        self, room_id: str, query: str, limit: int = 5
    ) -> list[Message]:
        """Synchronously search past messages in a Webex room using SQLite FTS5 with LIKE fallback."""
        if not room_id or not query or not query.strip():
            return []

        clean_terms = re.findall(r"[\w.-]+", query)
        clean_query = " ".join(clean_terms) if clean_terms else query.strip()

        try:
            conn = sqlite3.connect(self.db_path)
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
                    (room_id, clean_query, limit),
                )
                rows = cursor.fetchall()
                if rows:
                    conn.close()
                    return [
                        Message(
                            role=r[0],
                            content=r[1],
                            timestamp=r[2],
                            metadata={
                                **(json.loads(r[3]) if r[3] else {}),
                                "thread_id": r[4],
                            },
                        )
                        for r in rows
                    ]
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
                (room_id, f"%{clean_query}%", limit),
            )
            rows = cursor.fetchall()
            conn.close()

            return [
                Message(
                    role=r[0],
                    content=r[1],
                    timestamp=r[2],
                    metadata={
                        **(json.loads(r[3]) if r[3] else {}),
                        "thread_id": r[4],
                    },
                )
                for r in rows
            ]
        except Exception as e:
            log.warning("Sync room search failed: %s", e)
            return []

    def load_thread_sync(
        self, thread_id: str, limit: int | None = None
    ) -> list[Message]:
        """Synchronously load thread messages from database."""
        messages = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            if limit:
                query = """
                    SELECT role, content, timestamp, metadata FROM (
                        SELECT role, content, timestamp, metadata
                        FROM messages
                        WHERE thread_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ) ORDER BY timestamp ASC
                """
                params: tuple = (thread_id, limit)
            else:
                query = """
                    SELECT role, content, timestamp, metadata
                    FROM messages
                    WHERE thread_id = ?
                    ORDER BY timestamp ASC
                """
                params = (thread_id,)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

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
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            cursor.execute(
                "DELETE FROM conversations WHERE thread_id = ?", (thread_id,)
            )
            conn.commit()
            conn.close()
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
                    (thread_id, summary, datetime.now(UTC)),
                )
                await db.commit()
                log.debug("Saved thread summary for thread %s", thread_id)
        except Exception as e:
            log.exception("Failed to save summary for thread %s: %s", thread_id, e)

    def save_thread_summary_sync(self, thread_id: str, summary: str) -> None:
        """Synchronously save or update the thread summary in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            now_iso = datetime.now(UTC).isoformat()
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
            conn.close()
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
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT thread_summary FROM conversations WHERE thread_id = ?",
                (thread_id,),
            )
            row = cursor.fetchone()
            conn.close()
            return row[0] if row and row[0] else None
        except Exception as e:
            log.exception("Failed to sync-get summary for thread %s: %s", thread_id, e)
            return None

    async def trim_old_messages(self, thread_id: str, keep_count: int) -> int:
        """Trim messages for thread_id to keep only the most recent keep_count.

        Returns:
            Number of messages deleted.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    DELETE FROM messages
                    WHERE thread_id = ? AND id NOT IN (
                        SELECT id FROM (
                            SELECT id FROM messages
                            WHERE thread_id = ?
                            ORDER BY timestamp DESC, id DESC
                            LIMIT ?
                        )
                    )
                    """,
                    (thread_id, thread_id, keep_count),
                )
                await db.commit()
                deleted = cursor.rowcount
                log.debug("Trimmed %d old messages for thread %s", deleted, thread_id)
                return deleted
        except Exception as e:
            log.exception("Failed to trim messages for thread %s: %s", thread_id, e)
            return 0

    def trim_old_messages_sync(self, thread_id: str, keep_count: int) -> int:
        """Synchronously trim messages for thread_id to keep only the most recent keep_count."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM messages
                WHERE thread_id = ? AND id NOT IN (
                    SELECT id FROM (
                        SELECT id FROM messages
                        WHERE thread_id = ?
                        ORDER BY timestamp DESC, id DESC
                        LIMIT ?
                    )
                )
                """,
                (thread_id, thread_id, keep_count),
            )
            conn.commit()
            deleted = cursor.rowcount
            conn.close()
            log.debug("Sync-trimmed %d old messages for thread %s", deleted, thread_id)
            return deleted
        except Exception as e:
            log.exception(
                "Failed to sync-trim messages for thread %s: %s", thread_id, e
            )
            return 0
