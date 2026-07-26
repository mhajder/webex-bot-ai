"""Conversation message and history manager for maintaining thread context."""

import asyncio
import inspect
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from threading import Lock
from typing import Literal

from src.conversation.persistence import ConversationStore
from src.conversation.types import Message

log = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation history for thread-based context.

    Stores conversation history per thread ID, allowing the bot to maintain
    context across multiple messages in the same Webex thread. Supports
    rolling thread summarization for preserving long thread context.

    Thread-safe implementation using locks for concurrent access.
    """

    def __init__(
        self,
        max_messages: int = 50,
        timeout_seconds: int = 86400,  # 24 hours
        db_path: str = "conversations.db",
        enable_persistence: bool = True,
        enable_summarization: bool = True,
        summary_threshold: int = 50,
        keep_recent_messages: int = 20,
        summary_model: str | None = None,
        summary_llm_callable: Callable[[str], str] | None = None,
    ):
        """Initialize the conversation manager.

        Args:
            max_messages: Maximum number of messages to keep per thread.
            timeout_seconds: Time after which conversations are considered stale.
            db_path: Path to SQLite database file.
            enable_persistence: Whether to persist conversations to database.
            enable_summarization: Whether to enable rolling thread summarization.
            summary_threshold: Message count threshold to trigger rolling summarization.
            keep_recent_messages: Number of recent messages to keep after summarization.
            summary_model: Optional LLM model identifier for thread summarization.
            summary_llm_callable: Optional custom callable to generate summary (for testing).
        """
        self._history: dict[str, list[Message]] = defaultdict(list)
        self._summaries: dict[str, str] = {}
        self._timestamps: dict[str, float] = {}
        self._active_summarizations: set[str] = set()
        self._lock = Lock()
        self._persistence_tasks: set = set()  # Track background persistence tasks
        self.max_messages = max_messages
        self.timeout_seconds = timeout_seconds
        self.enable_persistence = enable_persistence
        self.enable_summarization = enable_summarization
        self.summary_threshold = summary_threshold
        self.keep_recent_messages = keep_recent_messages
        self.summary_model = summary_model
        self._summary_llm_callable = summary_llm_callable
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
        room_id: str | None = None,
    ) -> None:
        """Add a message to the conversation history.

        Args:
            thread_id: The thread identifier.
            role: Message role ("user", "assistant", "system", "tool").
            content: Message content.
            metadata: Optional metadata dictionary.
            room_id: Optional Webex room identifier.
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
            msg_count = len(self._history[thread_id])

            # Trim history if summarization is disabled or as a safety cap
            if not self.enable_summarization and msg_count > self.max_messages:
                self._history[thread_id] = self._history[thread_id][
                    -self.max_messages :
                ]
            elif self.enable_summarization and msg_count > self.summary_threshold * 2:
                # Safeguard against unbounded memory growth
                self._history[thread_id] = self._history[thread_id][
                    -(self.summary_threshold * 2) :
                ]

            log.debug(
                "Added %s message to thread %s (room %s). Total messages: %d",
                role,
                thread_id,
                room_id,
                len(self._history[thread_id]),
            )

        # Persist to database asynchronously
        if self.enable_persistence and self.store:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Event loop is running, create task
                    task = asyncio.create_task(
                        self.store.save_message(thread_id, message, room_id=room_id)
                    )
                    self._persistence_tasks.add(task)
                    task.add_done_callback(self._persistence_tasks.discard)
                else:
                    # Event loop exists but not running, use sync save
                    self.store.save_message_sync(thread_id, message, room_id=room_id)
            except RuntimeError:
                # No event loop in current thread, fall back to sync save
                log.debug("Event loop not available, using sync persistence")
                try:
                    self.store.save_message_sync(thread_id, message, room_id=room_id)
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

        # Trigger background rolling thread summarization if threshold met
        if self.enable_summarization and msg_count >= self.summary_threshold:
            self.trigger_summarization_task(thread_id)

    def trigger_summarization_task(self, thread_id: str) -> None:
        """Trigger async background summarization task if event loop is running."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = asyncio.create_task(self.summarize_thread_history(thread_id))
                self._persistence_tasks.add(task)
                task.add_done_callback(self._persistence_tasks.discard)
        except RuntimeError:
            log.debug("Event loop not available for background summarization")
        except Exception as e:
            log.exception("Failed to schedule background summarization task: %s", e)

    def _generate_summary_llm(self, prompt: str) -> str:
        """Call LLM synchronously to generate a thread summary."""
        import litellm

        from src.config import settings

        model = self.summary_model or settings.llm.model
        kwargs: dict = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI assistant that summarizes conversation threads. "
                        "Produce a clear, concise, rolling summary of the conversation so far. "
                        "Preserve key facts, user requests, technical details, decisions, and outcomes."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        if settings.llm.api_base:
            kwargs["api_base"] = settings.llm.api_base

        response = litellm.completion(**kwargs)
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""

    async def summarize_thread_history(self, thread_id: str) -> str | None:
        """Summarize older messages in a thread asynchronously.

        Takes older messages beyond keep_recent_messages, combines them with
        any existing thread_summary, calls LLM to generate an updated summary,
        persists the summary, and trims the older messages from history.

        Args:
            thread_id: The thread identifier.

        Returns:
            Updated thread summary string, or None if skipped/failed.
        """
        if not thread_id:
            return None

        with self._lock:
            if thread_id in self._active_summarizations:
                log.debug("Summarization already in progress for thread %s", thread_id)
                return None
            self._active_summarizations.add(thread_id)

            history = list(self._history.get(thread_id, []))
            num_to_summarize = len(history) - self.keep_recent_messages

        if num_to_summarize <= 0:
            with self._lock:
                self._active_summarizations.discard(thread_id)
            return None

        to_summarize = history[:num_to_summarize]
        existing_summary = self.get_thread_summary(thread_id)

        try:
            transcript = "\n".join(
                f"[{msg.role.upper()}]: {msg.content}" for msg in to_summarize
            )

            if existing_summary:
                prompt = (
                    f"Existing Conversation Summary:\n{existing_summary}\n\n"
                    f"New Messages to Incorporate into Summary:\n{transcript}\n\n"
                    "Please generate an updated rolling summary of the conversation history."
                )
            else:
                prompt = (
                    f"Conversation Messages to Summarize:\n{transcript}\n\n"
                    "Please generate a comprehensive summary of this conversation history."
                )

            log.info(
                "Starting rolling summarization for thread %s (%d messages to summarize)",
                thread_id,
                len(to_summarize),
            )

            if self._summary_llm_callable:
                if inspect.iscoroutinefunction(self._summary_llm_callable):
                    new_summary = await self._summary_llm_callable(prompt)
                else:
                    new_summary = await asyncio.to_thread(
                        self._summary_llm_callable, prompt
                    )
            else:
                new_summary = await asyncio.to_thread(
                    self._generate_summary_llm, prompt
                )

            if not new_summary or not new_summary.strip():
                log.warning("Received empty summary from LLM for thread %s", thread_id)
                return None

            new_summary = new_summary.strip()

            with self._lock:
                self._summaries[thread_id] = new_summary
                self._history[thread_id] = self._history[thread_id][num_to_summarize:]

            log.info(
                "Successfully summarized thread %s. New summary length: %d chars. "
                "%d active messages remaining in history.",
                thread_id,
                len(new_summary),
                len(self._history[thread_id]),
            )

            if self.enable_persistence and self.store:
                try:
                    if self._persistence_tasks:
                        await asyncio.gather(
                            *list(self._persistence_tasks), return_exceptions=True
                        )
                    await self.store.save_thread_summary(thread_id, new_summary)
                    await self.store.trim_old_messages(
                        thread_id, keep_count=self.keep_recent_messages
                    )
                except Exception as db_err:
                    log.exception(
                        "Failed to persist summary/trim messages for thread %s: %s",
                        thread_id,
                        db_err,
                    )

            return new_summary

        except Exception as e:
            log.exception("Error summarizing thread %s: %s", thread_id, e)
            return None
        finally:
            with self._lock:
                self._active_summarizations.discard(thread_id)

    def add_user_message(
        self, thread_id: str, content: str, room_id: str | None = None
    ) -> None:
        """Add a user message to the history."""
        self.add_message(thread_id, "user", content, room_id=room_id)

    def add_assistant_message(
        self, thread_id: str, content: str, room_id: str | None = None
    ) -> None:
        """Add an assistant message to the history."""
        self.add_message(thread_id, "assistant", content, room_id=room_id)

    def search_room_history_sync(
        self, room_id: str, query: str, limit: int = 5
    ) -> list[Message]:
        """Search room history synchronously using FTS5."""
        if not self.enable_persistence or not self.store:
            return []
        return self.store.search_room_history_sync(room_id, query, limit=limit)

    def get_thread_summary(self, thread_id: str) -> str | None:
        """Get the summary of older conversation history for a thread."""
        if not thread_id:
            return None

        with self._lock:
            if thread_id in self._summaries:
                return self._summaries[thread_id]

        if self.enable_persistence and self.store:
            try:
                summary = self.store.get_thread_summary_sync(thread_id)
                if summary:
                    with self._lock:
                        self._summaries[thread_id] = summary
                    return summary
            except Exception as e:
                log.exception("Failed to load thread summary for %s: %s", thread_id, e)

        return None

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
                history = self.store.load_thread_sync(
                    thread_id, limit=self.max_messages
                )
                summary = self.store.get_thread_summary_sync(thread_id)
                if history or summary:
                    with self._lock:
                        if history:
                            self._history[thread_id] = history
                        if summary:
                            self._summaries[thread_id] = summary
                        self._timestamps[thread_id] = time.time()
                    log.debug(
                        "Loaded %d messages and summary for thread %s from database",
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

        # Add thread summary if present
        summary = self.get_thread_summary(thread_id)
        if summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Summary of previous conversation in this thread:\n{summary}",
                }
            )

        # Add conversation history
        messages.extend(msg.to_api_format() for msg in self.get_history(thread_id))

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
                self.store.delete_thread_sync(thread_id)
            except Exception as e:
                log.exception(
                    "Failed to delete thread %s from database: %s", thread_id, e
                )

    async def clear_thread_async(self, thread_id: str) -> None:
        """Async method to clear thread and await pending tasks and DB deletion."""
        if not thread_id:
            return

        if self._persistence_tasks:
            await asyncio.gather(*list(self._persistence_tasks), return_exceptions=True)

        with self._lock:
            self._clear_thread(thread_id)

        if self.enable_persistence and self.store:
            try:
                await self.store.delete_thread(thread_id)
            except Exception as e:
                log.exception(
                    "Failed to delete thread %s from database: %s", thread_id, e
                )

    def _clear_thread(self, thread_id: str) -> None:
        """Internal method to clear thread (must be called with lock held)."""
        if thread_id in self._history:
            del self._history[thread_id]
        if thread_id in self._timestamps:
            del self._timestamps[thread_id]
        if thread_id in self._summaries:
            del self._summaries[thread_id]
        log.debug("Cleared history and summary for thread %s", thread_id)

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
