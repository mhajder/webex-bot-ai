"""Conversation message and history manager for maintaining thread context."""

import asyncio
import logging
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from threading import Lock
from typing import Literal

from src.conversation.persistence import ConversationStore
from src.conversation.types import Message

log = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation history for thread-based context.

    Stores conversation history per thread ID, allowing the bot to maintain
    context across multiple messages in the same Webex thread.

    Thread-safe implementation using locks for concurrent access.
    """

    def __init__(
        self,
        max_messages: int = 50,
        timeout_seconds: int = 86400,  # 24 hours
        db_path: str = "conversations.db",
        enable_persistence: bool = True,
    ):
        """Initialize the conversation manager.

        Args:
            max_messages: Maximum number of messages to keep per thread.
            timeout_seconds: Time after which conversations are considered stale.
            db_path: Path to SQLite database file.
            enable_persistence: Whether to persist conversations to database.
        """
        self._history: dict[str, list[Message]] = defaultdict(list)
        self._timestamps: dict[str, float] = {}
        self._lock = Lock()
        self._persistence_tasks: set = set()  # Track background persistence tasks
        self.max_messages = max_messages
        self.timeout_seconds = timeout_seconds
        self.enable_persistence = enable_persistence
        self.store = ConversationStore(db_path) if enable_persistence else None

    async def initialize(self) -> None:
        """Initialize the database. Call this on startup.

        Must be called before using the manager if persistence is enabled.
        """
        if self.enable_persistence and self.store:
            await self.store.init_db()
            log.info("Conversation manager initialized with persistence enabled")

    def add_message(
        self,
        thread_id: str,
        role: Literal["user", "assistant", "system", "tool"],
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """Add a message to the conversation history.

        Args:
            thread_id: The thread identifier.
            role: Message role ("user", "assistant", "system", "tool").
            content: Message content.
            metadata: Optional metadata dictionary.
        """
        if not thread_id:
            log.warning("Cannot add message: thread_id is None")
            return

        with self._lock:
            message = Message(
                role=role,
                content=content,
                metadata=metadata or {},
            )

            self._history[thread_id].append(message)
            self._timestamps[thread_id] = time.time()

            # Trim history if it exceeds max messages
            if len(self._history[thread_id]) > self.max_messages:
                self._history[thread_id] = self._history[thread_id][
                    -self.max_messages :
                ]

            log.debug(
                "Added %s message to thread %s. Total messages: %d",
                role,
                thread_id,
                len(self._history[thread_id]),
            )

        # Persist to database asynchronously
        if self.enable_persistence and self.store:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Event loop is running, create task
                    task = asyncio.create_task(
                        self.store.save_message(thread_id, message)
                    )
                    self._persistence_tasks.add(task)
                    task.add_done_callback(self._persistence_tasks.discard)
                else:
                    # Event loop exists but not running, use sync save
                    self._save_message_sync(thread_id, message)
            except RuntimeError:
                # No event loop in current thread, fall back to sync save
                log.debug("Event loop not available, using sync persistence")
                try:
                    self._save_message_sync(thread_id, message)
                except Exception as sync_error:
                    log.exception(
                        "Failed to sync-persist message for thread %s: %s",
                        thread_id,
                        sync_error,
                    )
            except Exception as e:
                log.exception(
                    "Failed to create persistence task for thread %s: %s", thread_id, e
                )

    def add_user_message(self, thread_id: str, content: str) -> None:
        """Add a user message to the history."""
        self.add_message(thread_id, "user", content)

    def add_assistant_message(self, thread_id: str, content: str) -> None:
        """Add an assistant message to the history."""
        self.add_message(thread_id, "assistant", content)

    def get_history(self, thread_id: str) -> list[Message]:
        """Get the conversation history for a thread.

        Args:
            thread_id: The thread identifier.

        Returns:
            List of Message objects for the thread.
        """
        if not thread_id:
            return []

        with self._lock:
            # Check if conversation has timed out
            if thread_id in self._timestamps:
                age = time.time() - self._timestamps[thread_id]
                if age > self.timeout_seconds:
                    log.info(
                        "Thread %s has timed out after %.0fs. Clearing history.",
                        thread_id,
                        age,
                    )
                    self._clear_thread(thread_id)
                    return []

            # Return in-memory history if available
            if self._history.get(thread_id):
                return list(self._history[thread_id])

        # If no in-memory history and persistence is enabled, load from DB synchronously
        if self.enable_persistence and self.store:
            try:
                history = self._load_thread_sync(thread_id)
                if history:
                    with self._lock:
                        self._history[thread_id] = history
                        self._timestamps[thread_id] = time.time()
                    log.debug(
                        "Loaded %d messages for thread %s from database",
                        len(history),
                        thread_id,
                    )
                return list(history)
            except Exception as e:
                log.exception(
                    "Failed to load thread %s from database: %s", thread_id, e
                )
                return []

        return []

    def _load_thread_sync(self, thread_id: str) -> list[Message]:
        """Synchronously load thread messages from database.

        This method uses sqlite3 directly to avoid event loop issues
        when called from webex_bot's threading context.
        """
        import json

        messages = []
        try:
            conn = sqlite3.connect(self.store.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content, timestamp, metadata
                FROM messages
                WHERE thread_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (thread_id, self.max_messages),
            )
            rows = cursor.fetchall()
            conn.close()

            for role, content, timestamp, metadata_json in rows:
                metadata = json.loads(metadata_json) if metadata_json else {}
                message = Message(
                    role=role,
                    content=content,
                    timestamp=timestamp,
                    metadata=metadata,
                )
                messages.append(message)

            return messages
        except sqlite3.OperationalError:
            # Database doesn't exist yet
            return []

    def _save_message_sync(self, thread_id: str, message: Message) -> None:
        """Synchronously save a message to database.

        This is a fallback when async save is not available.
        """
        import json

        try:
            conn = sqlite3.connect(self.store.db_path)
            cursor = conn.cursor()

            # Ensure conversation exists
            cursor.execute(
                """
                INSERT OR IGNORE INTO conversations (thread_id, created_at, updated_at)
                VALUES (?, ?, ?)
                """,
                (thread_id, datetime.now().isoformat(), datetime.now().isoformat()),
            )

            # Save message
            metadata_json = json.dumps(message.metadata)
            cursor.execute(
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

            # Update conversation's updated_at
            cursor.execute(
                """
                UPDATE conversations SET updated_at = ? WHERE thread_id = ?
                """,
                (datetime.now().isoformat(), thread_id),
            )

            conn.commit()
            conn.close()
            log.debug("Sync-saved %s message to thread %s", message.role, thread_id)
        except sqlite3.OperationalError as e:
            log.debug("Database not ready for sync save: %s", e)
        except Exception as e:
            log.exception("Failed to sync-save message for thread %s: %s", thread_id, e)

    def get_messages_for_api(
        self,
        thread_id: str,
        system_prompt: str | None = None,
    ) -> list[dict]:
        """Get messages formatted for the LLM API.

        Args:
            thread_id: The thread identifier.
            system_prompt: Optional system prompt to prepend.

        Returns:
            List of message dicts with "role" and "content" keys.
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        for msg in self.get_history(thread_id):
            messages.append(msg.to_api_format())

        return messages

    def has_history(self, thread_id: str) -> bool:
        """Check if a thread has conversation history."""
        if not thread_id:
            return False

        with self._lock:
            return thread_id in self._history and len(self._history[thread_id]) > 0

    def get_thread_count(self) -> int:
        """Get the number of active conversation threads."""
        with self._lock:
            return len(self._history)

    def get_message_count(self, thread_id: str) -> int:
        """Get the number of messages in a thread."""
        with self._lock:
            return len(self._history.get(thread_id, []))

    def clear_thread(self, thread_id: str) -> None:
        """Clear the conversation history for a thread."""
        with self._lock:
            self._clear_thread(thread_id)

        # Also delete from database
        if self.enable_persistence and self.store:
            try:
                self._delete_thread_sync(thread_id)
            except Exception as e:
                log.exception(
                    "Failed to delete thread %s from database: %s", thread_id, e
                )

    def _delete_thread_sync(self, thread_id: str) -> None:
        """Synchronously delete thread from database."""
        try:
            conn = sqlite3.connect(self.store.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM conversations WHERE thread_id = ?", (thread_id,)
            )
            conn.commit()
            conn.close()
            log.debug("Deleted thread %s from database", thread_id)
        except sqlite3.OperationalError:
            # Database doesn't exist yet
            pass

    def _clear_thread(self, thread_id: str) -> None:
        """Internal method to clear thread (must be called with lock held)."""
        if thread_id in self._history:
            del self._history[thread_id]
        if thread_id in self._timestamps:
            del self._timestamps[thread_id]
        log.debug("Cleared history for thread %s", thread_id)

    def cleanup_stale_threads(self) -> int:
        """Remove all stale conversation threads.

        Returns:
            Number of threads removed.
        """
        removed = 0
        current_time = time.time()

        with self._lock:
            stale_threads = [
                thread_id
                for thread_id, timestamp in self._timestamps.items()
                if current_time - timestamp > self.timeout_seconds
            ]

            for thread_id in stale_threads:
                self._clear_thread(thread_id)
                removed += 1

        if removed > 0:
            log.info("Cleaned up %d stale conversation threads", removed)

        return removed
