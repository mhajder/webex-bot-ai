"""Conversation message and history manager for maintaining thread context."""

import asyncio
import inspect
import logging
import math
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from threading import Lock
from typing import Literal

from src.conversation.persistence import ConversationStore
from src.conversation.types import Message

log = logging.getLogger(__name__)

# Backoff after failed summarizations: 60s, doubling, capped at 1 hour
_SUMMARY_BACKOFF_BASE_SECONDS = 60
_SUMMARY_BACKOFF_MAX_SECONDS = 3600


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
        summary_llm_callable: Callable[[str], str | Awaitable[str]] | None = None,
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
        # Bumped whenever a thread is cleared or times out, so a summary
        # started before that is recognized as stale when it finishes
        self._generations: defaultdict[str, int] = defaultdict(int)
        # Shared across threads: a failing summary model fails for every thread
        self._summary_failures = 0
        self._summary_retry_at = 0.0
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

    @classmethod
    def from_settings(cls) -> "ConversationManager":
        """Create a manager configured from the CONVERSATION_* settings."""
        from src.config import settings

        conv = settings.conversation
        return cls(
            max_messages=conv.max_history_messages,
            timeout_seconds=conv.timeout_seconds,
            db_path=conv.db_path,
            enable_persistence=conv.enable_persistence,
            enable_summarization=conv.enable_summarization,
            summary_threshold=conv.summary_threshold,
            keep_recent_messages=conv.keep_recent_messages,
            summary_model=conv.summary_model,
        )

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

        # Expire a timed-out thread and load persisted history/summary first,
        # so the new message isn't appended to stale context or to an empty
        # in-memory list that would hide the stored history.
        self.get_history(thread_id)

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

        # Persist to database asynchronously if a running loop is present in the current thread
        if self.enable_persistence and self.store:
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(
                    self.store.save_message(thread_id, message, room_id=room_id)
                )
                self._persistence_tasks.add(task)
                task.add_done_callback(self._persistence_tasks.discard)
            except RuntimeError:
                # No running event loop in current thread, fall back to sync save
                log.debug("No running event loop in thread, using sync persistence")
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
        # (nothing to summarize until there are more than keep_recent_messages)
        if (
            self.enable_summarization
            and msg_count >= self.summary_threshold
            and msg_count > self.keep_recent_messages
        ):
            self.trigger_summarization_task(thread_id)

    def trigger_summarization_task(self, thread_id: str) -> None:
        """Trigger background summarization.

        Uses the running event loop when there is one; otherwise (webex_bot runs
        commands in executor threads without a loop) runs it on a daemon thread
        with its own event loop.
        """
        with self._lock:
            if thread_id in self._active_summarizations:
                return
            if time.time() < self._summary_retry_at:
                log.debug(
                    "Skipping summarization of thread %s: backing off after failure",
                    thread_id,
                )
                return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            if loop is not None:
                task = loop.create_task(self.summarize_thread_history(thread_id))
                self._persistence_tasks.add(task)
                task.add_done_callback(self._persistence_tasks.discard)
            else:
                threading.Thread(
                    target=asyncio.run,
                    args=(self.summarize_thread_history(thread_id),),
                    # Webex IDs share a long base64 prefix; the tail is distinctive
                    name=f"summarize-{thread_id[-8:]}",
                    daemon=True,
                ).start()
        except Exception as e:
            log.exception("Failed to schedule background summarization task: %s", e)

    def _pending_tasks_on_current_loop(self) -> list[asyncio.Task]:
        """Unfinished persistence tasks on the running loop, excluding this task."""
        loop = asyncio.get_running_loop()
        current_task = asyncio.current_task()
        return [
            t
            for t in list(self._persistence_tasks)
            if t is not current_task and not t.done() and t.get_loop() is loop
        ]

    def _record_summary_result(self, success: bool) -> None:
        """Reset the failure backoff on success, or extend it on failure."""
        with self._lock:
            if success:
                self._summary_failures = 0
                self._summary_retry_at = 0.0
                return
            self._summary_failures += 1
            delay = min(
                _SUMMARY_BACKOFF_BASE_SECONDS * 2 ** (self._summary_failures - 1),
                _SUMMARY_BACKOFF_MAX_SECONDS,
            )
            self._summary_retry_at = time.time() + delay
        log.warning(
            "Summarization failed (%d in a row); retrying in %ds",
            self._summary_failures,
            delay,
        )

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
        api_key, api_base = self._summary_credentials(model)
        if api_base:
            kwargs["api_base"] = api_base
        if api_key:
            kwargs["api_key"] = api_key

        response = litellm.completion(**kwargs)
        if not isinstance(response, litellm.ModelResponse):
            return ""
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""

    @staticmethod
    def _summary_credentials(model: str) -> tuple[str | None, str | None]:
        """API key and base URL for the summary model.

        CONVERSATION_SUMMARY_API_KEY/_API_BASE win. Otherwise the main
        LLM_API_KEY/LLM_API_BASE are reused only when the summary model uses
        the same provider as LLM_MODEL; for another provider litellm falls
        back to that provider's own environment variables.
        """
        import litellm

        from src.config import settings

        conv = settings.conversation
        api_key, api_base = conv.summary_api_key, conv.summary_api_base

        def provider(name: str) -> str | None:
            try:
                found = litellm.get_llm_provider(name)[1]
            except Exception:
                found = name.split("/", 1)[0] if "/" in name else None
            # ollama/ and ollama_chat/ are two routes to the same server
            return "ollama" if found == "ollama_chat" else found

        if provider(model) == provider(settings.llm.model):
            api_key = api_key or settings.llm.api_key
            api_base = api_base or settings.llm.api_base
        return api_key, api_base

    async def summarize_thread_history(self, thread_id: str) -> str | None:
        """Summarize older messages in a thread asynchronously.

        Takes older messages beyond keep_recent_messages, combines them with
        any existing thread_summary, calls LLM to generate an updated summary,
        persists the summary, and removes the older messages from the active
        history (they are archived in the database, not deleted).

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
            generation = self._generations[thread_id]

        if num_to_summarize <= 0:
            with self._lock:
                self._active_summarizations.discard(thread_id)
            return None

        in_memory = history[:num_to_summarize]
        to_summarize = in_memory
        existing_summary = self.get_thread_summary(thread_id)

        try:
            if self.enable_persistence and self.store:
                # Memory is capped at 2x the threshold, so while summarization
                # is backing off the oldest messages are dropped from memory
                # unsummarized. They are still active in the database, so
                # summarize everything stored up to the same point; otherwise
                # archive_messages_until() would archive them unsummarized.
                # Merge rather than pick one list by length: memory may hold
                # messages not saved yet while the DB holds ones dropped from
                # memory, and both must be summarized.
                cutoff = in_memory[-1].timestamp
                stored = await asyncio.to_thread(self.store.load_thread_sync, thread_id)
                seen = {(m.timestamp, m.role, m.content) for m in in_memory}
                extra = [
                    m
                    for m in stored
                    if m.timestamp <= cutoff
                    and (m.timestamp, m.role, m.content) not in seen
                ]
                if extra:
                    to_summarize = sorted(
                        [*extra, *in_memory], key=lambda m: m.timestamp
                    )

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

            new_summary: str = ""
            if self._summary_llm_callable:
                if inspect.iscoroutinefunction(self._summary_llm_callable):
                    res = await self._summary_llm_callable(prompt)
                    new_summary = str(res) if res else ""
                else:
                    res = await asyncio.to_thread(self._summary_llm_callable, prompt)
                    new_summary = str(res) if res else ""
            else:
                new_summary = await asyncio.to_thread(
                    self._generate_summary_llm, prompt
                )

            if not new_summary or not new_summary.strip():
                log.warning("Received empty summary from LLM for thread %s", thread_id)
                self._record_summary_result(success=False)
                return None

            new_summary = new_summary.strip()
            self._record_summary_result(success=True)

            summarized_until = max(m.timestamp for m in to_summarize)

            # Remove the summarized messages by identity: the list may have been
            # capped or cleared while the LLM call was running, so slicing by
            # the original index could drop unsummarized messages.
            summarized_ids = {id(m) for m in in_memory}
            with self._lock:
                # The thread was cleared or timed out during the LLM call (it
                # may already have new messages): storing the summary would
                # resurrect the discarded context.
                if self._generations[thread_id] != generation:
                    log.info(
                        "Thread %s was cleared during summarization; "
                        "discarding summary",
                        thread_id,
                    )
                    return None
                self._summaries[thread_id] = new_summary
                if thread_id in self._history:
                    self._history[thread_id] = [
                        m
                        for m in self._history[thread_id]
                        if id(m) not in summarized_ids
                    ]

            log.info(
                "Successfully summarized thread %s. New summary length: %d chars. "
                "%d active messages remaining in history.",
                thread_id,
                len(new_summary),
                self.get_message_count(thread_id),
            )

            if self.enable_persistence and self.store:
                try:
                    pending_tasks = self._pending_tasks_on_current_loop()
                    if pending_tasks:
                        await asyncio.gather(*pending_tasks, return_exceptions=True)
                    with self._lock:
                        cleared = self._generations[thread_id] != generation
                    if cleared:
                        return None
                    await self.store.save_thread_summary(thread_id, new_summary)
                    # Archive rather than delete, so summarized messages stay
                    # available to room history search.
                    await self.store.archive_messages_until(
                        thread_id, until=summarized_until
                    )
                except Exception as db_err:
                    log.exception(
                        "Failed to persist summary/archive messages for thread %s: %s",
                        thread_id,
                        db_err,
                    )

            return new_summary

        except Exception as e:
            log.exception("Error summarizing thread %s: %s", thread_id, e)
            self._record_summary_result(success=False)
            return None
        finally:
            with self._lock:
                self._active_summarizations.discard(thread_id)

    def add_user_message(
        self,
        thread_id: str,
        content: str,
        room_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Add a user message to the history."""
        self.add_message(thread_id, "user", content, metadata=metadata, room_id=room_id)

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

        self._expire_if_stale(thread_id)

        with self._lock:
            if thread_id in self._summaries:
                return self._summaries[thread_id]
            # A thread active in memory was loaded with its summary (or is
            # new), so the in-memory state is authoritative: skip the DB.
            if thread_id in self._timestamps:
                return None

        if self.enable_persistence and self.store:
            try:
                self._expire_stored_if_stale(thread_id)
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

        if self._expire_if_stale(thread_id):
            return []

        with self._lock:
            # Return in-memory history if available
            if self._history.get(thread_id):
                return list(self._history[thread_id])

        # If no in-memory history and persistence is enabled, load from DB synchronously
        if self.enable_persistence and self.store:
            try:
                self._expire_stored_if_stale(thread_id)
                history = self.store.load_thread_sync(
                    thread_id, limit=self.max_messages
                )
                summary = self.store.get_thread_summary_sync(thread_id)
                if history or summary:
                    with self._lock:
                        # Another thread may have loaded this thread and added
                        # messages meanwhile; don't overwrite its state.
                        if self._history.get(thread_id):
                            return list(self._history[thread_id])
                        if history:
                            self._history[thread_id] = history
                        if summary:
                            self._summaries[thread_id] = summary
                        # Age from the last real activity, not the reload time,
                        # so the timeout matches the persisted thread.
                        self._timestamps[thread_id] = (
                            history[-1].timestamp if history else time.time()
                        )
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

    def _expire_if_stale(self, thread_id: str) -> bool:
        """Clear an in-memory thread that has timed out and archive it in the DB.

        Returns:
            True if the thread had timed out.
        """
        with self._lock:
            timestamp = self._timestamps.get(thread_id)
            age = time.time() - timestamp if timestamp is not None else 0.0
            if age <= self.timeout_seconds:
                return False
            log.info(
                "Thread %s has timed out after %.0fs. Clearing history.",
                thread_id,
                age,
            )
            self._clear_thread(thread_id)

        if self.enable_persistence and self.store:
            try:
                self.store.archive_thread_sync(thread_id)
            except Exception as e:
                log.exception("Failed to archive timed-out thread %s: %s", thread_id, e)
        return True

    def _expire_stored_if_stale(self, thread_id: str) -> None:
        """Archive a persisted thread whose newest message is older than the timeout.

        Covers threads that are not in memory (after a restart or
        cleanup_stale_threads), so stale history and summary are not reloaded.
        """
        if not self.store:
            return
        if self.store.archive_thread_sync(
            thread_id, inactive_since=time.time() - self.timeout_seconds
        ):
            log.info("Archived timed-out thread %s in the database", thread_id)

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

        # Load history first: it expires timed-out threads, which also drops
        # their summary.
        history = self.get_history(thread_id)

        # Add thread summary if present
        summary = self.get_thread_summary(thread_id)
        if summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Summary of previous conversation in this thread:\n{summary}",
                }
            )

        # Add conversation history, consolidating consecutive same-role messages
        raw_history = [msg.to_api_format() for msg in history]
        consolidated: list[dict] = []
        for msg in raw_history:
            if consolidated and consolidated[-1]["role"] == msg["role"]:
                consolidated[-1]["content"] += f"\n\n{msg['content']}"
            else:
                consolidated.append(dict(msg))

        messages.extend(consolidated)

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

        pending_tasks = self._pending_tasks_on_current_loop()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

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
        self._generations[thread_id] += 1
        if thread_id in self._history:
            del self._history[thread_id]
        if thread_id in self._timestamps:
            del self._timestamps[thread_id]
        if thread_id in self._summaries:
            del self._summaries[thread_id]
        log.debug("Cleared history and summary for thread %s", thread_id)

    def start_cleanup(self, retention_days: int, interval_seconds: int = 86400) -> None:
        """Run cleanup on a daemon thread now and then every ``interval_seconds``.

        Drops timed-out threads from memory and, with persistence, deletes
        threads (with their messages and search index rows) whose last
        activity is older than ``retention_days``. 0 keeps them forever.
        Retention is never shorter than the conversation timeout, so threads
        that are still live are not deleted.
        """
        if retention_days > 0:
            timeout_days = math.ceil(self.timeout_seconds / 86400)
            if retention_days < timeout_days:
                log.warning(
                    "Retention (%d days) is shorter than the conversation timeout; "
                    "using %d days",
                    retention_days,
                    timeout_days,
                )
                retention_days = timeout_days

        def run() -> None:
            while True:
                try:
                    self.cleanup_stale_threads()
                    if self.store and retention_days > 0:
                        asyncio.run(self.store.cleanup_old_threads(days=retention_days))
                except Exception as e:
                    log.exception("Conversation cleanup failed: %s", e)
                time.sleep(interval_seconds)

        threading.Thread(target=run, name="conversation-cleanup", daemon=True).start()
        log.info(
            "Conversation cleanup scheduled every %ds (retention: %s)",
            interval_seconds,
            f"{retention_days} days" if retention_days > 0 else "forever",
        )

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

            # Generations only matter for summaries in flight; drop the rest so
            # the dict doesn't grow with every thread the bot has ever seen.
            for thread_id in list(self._generations):
                if (
                    thread_id not in self._timestamps
                    and thread_id not in self._active_summarizations
                ):
                    del self._generations[thread_id]

        if removed > 0:
            log.info("Cleaned up %d stale conversation threads", removed)

        return removed
