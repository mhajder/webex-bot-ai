"""SQLite-based persistence layer for conversation history."""

import json
import logging
from datetime import datetime

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
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        metadata TEXT,
                        FOREIGN KEY (thread_id) REFERENCES conversations(thread_id)
                        ON DELETE CASCADE
                    )
                    """
                )

                # Create index for faster queries
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_thread_id
                    ON messages(thread_id)
                    """
                )

                await db.commit()
                log.info("Database initialized at %s", self.db_path)
        except Exception as e:
            log.exception("Failed to initialize database: %s", e)
            raise

    async def save_message(self, thread_id: str, message: Message) -> None:
        """Save a message to the database.

        Args:
            thread_id: The thread identifier.
            message: The message to save.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Ensure conversation exists
                await db.execute(
                    """
                    INSERT OR IGNORE INTO conversations (thread_id, created_at, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (thread_id, datetime.now(), datetime.now()),
                )

                # Save message
                metadata_json = json.dumps(message.metadata)
                await db.execute(
                    """
                    INSERT INTO messages (thread_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        message.role,
                        message.content,
                        message.timestamp,
                        metadata_json,
                    ),
                )

                # Update conversation's updated_at timestamp
                await db.execute(
                    """
                    UPDATE conversations SET updated_at = ? WHERE thread_id = ?
                    """,
                    (datetime.now(), thread_id),
                )

                await db.commit()
                log.debug("Saved %s message to thread %s", message.role, thread_id)
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
                query = """
                    SELECT role, content, timestamp, metadata
                    FROM messages
                    WHERE thread_id = ?
                    ORDER BY timestamp ASC
                """
                params = (thread_id,)

                if limit:
                    query += " LIMIT ?"
                    params = (thread_id, limit)

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
