"""AI command for handling user questions with LiteLLM and thread context."""

import asyncio
import concurrent.futures
import json
import logging
import re
import threading
import time
import types
from datetime import datetime
from itertools import islice
from typing import Any, Literal

import litellm
from webex_bot.formatting import quote_info
from webex_bot.models.command import Command

from src.config import settings
from src.conversation.manager import ConversationManager
from src.mcp_client.client import MCPMultiClient
from src.mcp_client.types import MCPTool, MCPToolResult
from src.sentry import capture_exception, set_tag

log = logging.getLogger(__name__)

# Thread pool executor for running async MCP operations
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="mcp"
)

# Default configuration for all Ollama models
# All Ollama models support function calling and use chat mode by default
_OLLAMA_MODEL_DEFAULTS = {
    "supports_function_calling": True,
    "mode": "chat",
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "cache_creation_input_token_cost": 0.0,
    "cache_read_input_token_cost": 0.0,
}

# Register Ollama models with LiteLLM
# This applies to all common Ollama model patterns
_model_registry = {
    "ollama": _OLLAMA_MODEL_DEFAULTS,
    "ollama_chat": _OLLAMA_MODEL_DEFAULTS,
}

try:
    litellm.register_model(model_cost=_model_registry)
    log.info(
        "Registered Ollama model defaults: supports_function_calling=True, mode=chat"
    )
except Exception as e:
    log.warning("Failed to register Ollama model defaults: %s", e)


WebexRole = Literal["user", "assistant"]

# How long a room that refused message listing (403) is skipped before retrying
_ROOM_DENIED_RETRY_SECONDS = 3600

# Maximum LLM round trips that may request tools before an answer is forced
_MAX_TOOL_ROUNDS = 5

# Number of striped locks serializing per-thread Webex syncs
_SYNC_LOCK_STRIPES = 64

# Sync watermark count above which expired watermarks are pruned
_MAX_SYNC_WATERMARKS = 1024

_FINAL_ANSWER_NUDGE = (
    "Based on the tool results above, please provide a clear and concise "
    "response to the user's original question. Do NOT call any tools. "
    "Just respond with text."
)


def _flatten_tool_messages(messages: list[dict]) -> list[dict]:
    """Rewrite tool calls and results as plain text messages.

    Lets a conversation that used tools be sent without tool definitions.
    Consecutive messages with the same role are merged, since some providers
    require alternating roles.
    """
    flat: list[dict] = []
    for msg in messages:
        if msg["role"] == "tool":
            role, content = "user", f"Tool result:\n{msg['content']}"
        elif msg["role"] == "assistant" and msg.get("tool_calls"):
            calls = ", ".join(
                f"{tc['function']['name']}({tc['function']['arguments']})"
                for tc in msg["tool_calls"]
            )
            role = "assistant"
            content = "\n".join(
                filter(None, [msg.get("content"), f"Called tools: {calls}"])
            )
        else:
            role, content = msg["role"], msg["content"]
        if flat and flat[-1]["role"] == role and role != "system":
            flat[-1]["content"] += f"\n\n{content}"
        else:
            flat.append({"role": role, "content": content})
    return flat


def _webex_created_ts(msg: Any) -> float:
    """Epoch timestamp of a Webex message's ``created`` field (0.0 if unknown)."""
    created = getattr(msg, "created", None)
    if isinstance(created, datetime):
        return created.timestamp()
    if isinstance(created, str):
        try:
            return datetime.fromisoformat(created).timestamp()
        except ValueError:
            pass
    return 0.0


def _webex_text(msg: Any) -> str:
    """Plain text of a Webex message (object or WDM dict), falling back to markdown/HTML."""
    if isinstance(msg, dict):
        return (msg.get("text") or msg.get("markdown") or msg.get("html") or "").strip()
    return (
        getattr(msg, "text", None)
        or getattr(msg, "markdown", None)
        or getattr(msg, "html", None)
        or ""
    ).strip()


def _run_async_in_thread(coro):
    """Run an async coroutine in a separate thread with its own event loop.

    This allows async operations to run without conflicting with
    the main event loop used by webex_bot.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class AICommand(Command):
    """Command that handles AI-powered responses with thread context.

    Features:
    - Maintains conversation history per thread
    - Properly handles bot name mentions without confusing the AI
    - Supports follow-up questions in the same thread
    - Integrates MCP tools for extended capabilities
    - Supports multiple LLM providers via LiteLLM
    """

    def __init__(
        self,
        conversation_manager: ConversationManager | None = None,
        bot_name: str | None = None,
        mcp_client: MCPMultiClient | None = None,
        bot: Any = None,
    ):
        """Initialize the AI command.

        Args:
            conversation_manager: Shared conversation manager instance.
            bot_name: Bot name for mention handling (defaults to config value).
            mcp_client: MCP client for tool integration (optional).
            bot: WebexBot instance (optional).
        """
        super().__init__(
            command_keyword="",
            help_message=(
                "Ask me anything! I can answer questions and "
                "remember our conversation in this thread."
            ),
        )

        self.webex_bot = bot
        self.bot_name = bot_name or settings.bot.name
        self.model = settings.llm.model
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens

        self.conversation_manager = (
            conversation_manager or ConversationManager.from_settings()
        )

        self.mcp_client = mcp_client

        # Newest Webex `created` time already synced per thread
        self._sync_watermarks: dict[str, float] = {}
        self._sync_watermarks_lock = threading.Lock()
        # Serializes syncs of the same thread across executor threads, so
        # concurrent mentions don't both add the same replies. A fixed set of
        # striped locks keeps memory bounded for a long-running bot.
        self._sync_locks = [threading.Lock() for _ in range(_SYNC_LOCK_STRIPES)]
        # Rooms where listing messages is forbidden (bots in group spaces),
        # mapped to when the 403 was received
        self._room_list_denied: dict[str, float] = {}

        self._build_mention_patterns()

        log.info(
            "AICommand initialized: bot_name='%s', model='%s', mcp_enabled=%s",
            self.bot_name,
            self.model,
            self.mcp_client is not None,
        )

    def _build_mention_patterns(self) -> None:
        """Build regex patterns to detect and clean bot mentions."""
        escaped_name = re.escape(self.bot_name)

        # (?!\w) keeps the name from matching the start of a longer word,
        # e.g. bot "Max" must not strip "Max" from "Maximum size?".
        self._mention_pattern = re.compile(
            rf"^\s*@?\s*{escaped_name}(?!\w)\s*[:,.\s]*",
            re.IGNORECASE,
        )

        self._alt_mention_patterns = []
        display_name = settings.bot.display_name
        if display_name and display_name.lower() != self.bot_name.lower():
            escaped_display = re.escape(display_name)
            alt_pattern = re.compile(
                rf"^\s*@?\s*{escaped_display}(?!\w)\s*[:,.\s]*",
                re.IGNORECASE,
            )
            self._alt_mention_patterns.append(alt_pattern)

    def _get_thread_id(
        self, activity: Any, attachment_actions: Any = None
    ) -> str | None:
        """Extract valid Webex thread ID (parent message ID or message ID).

        Prioritizes the webexpythonsdk Message object (attachment_actions)
        which contains actual Webex Base64 Message/Parent IDs rather than WebSocket event IDs.
        """
        # 1. Check attachment_actions (webexpythonsdk Message object)
        if attachment_actions:
            inputs = getattr(attachment_actions, "inputs", None)
            if isinstance(inputs, dict) and inputs.get("thread_parent_id"):
                return inputs["thread_parent_id"]

            parent_id = getattr(attachment_actions, "parentId", None) or getattr(
                attachment_actions, "parent_id", None
            )
            if parent_id and isinstance(parent_id, str):
                return parent_id

            msg_id = getattr(attachment_actions, "id", None)
            if msg_id and isinstance(msg_id, str):
                return msg_id

        # 2. Check activity dict/object
        if activity:
            if isinstance(activity, dict):
                if (
                    "parent" in activity
                    and isinstance(activity["parent"], dict)
                    and "id" in activity["parent"]
                    and activity["parent"].get("type") == "reply"
                ):
                    return activity["parent"]["id"]

                if activity.get("parentId"):
                    return activity["parentId"]

                if (
                    "object" in activity
                    and isinstance(activity["object"], dict)
                    and "id" in activity["object"]
                ):
                    return activity["object"]["id"]

                if (
                    "target" in activity
                    and isinstance(activity["target"], dict)
                    and "id" in activity["target"]
                ):
                    return activity["target"]["id"]
            else:
                if hasattr(activity, "parent") and activity.parent:
                    parent = activity.parent
                    if isinstance(parent, dict) and "id" in parent:
                        return parent["id"]
                    if hasattr(parent, "id") and parent.id:
                        return parent.id

                for attr in ("parentId", "parent_id"):
                    if hasattr(activity, attr):
                        value = getattr(activity, attr)
                        if value:
                            return value

        log.warning("Could not extract thread ID from activity/attachment_actions")
        return None

    def _get_room_id(self, activity: Any, attachment_actions: Any = None) -> str | None:
        """Extract Webex roomId from activity or attachment_actions payload with fallback."""
        if attachment_actions:
            for attr in ("roomId", "room_id"):
                if hasattr(attachment_actions, attr):
                    val = getattr(attachment_actions, attr)
                    if val:
                        return val

        if activity:
            if isinstance(activity, dict):
                if activity.get("roomId"):
                    return activity["roomId"]
                if activity.get("room_id"):
                    return activity["room_id"]

                for nested_key in ("target", "data", "raw", "message"):
                    nested = activity.get(nested_key)
                    if isinstance(nested, dict):
                        room_id = (
                            nested.get("roomId")
                            or nested.get("room_id")
                            or nested.get("id")
                        )
                        if room_id:
                            return room_id
            else:
                for attr in ("roomId", "room_id"):
                    if hasattr(activity, attr):
                        val = getattr(activity, attr)
                        if val:
                            return val

                for nested_attr in ("target", "data", "raw", "message"):
                    if hasattr(activity, nested_attr):
                        nested = getattr(activity, nested_attr)
                        if isinstance(nested, dict):
                            val = (
                                nested.get("roomId")
                                or nested.get("room_id")
                                or nested.get("id")
                            )
                            if val:
                                return val
                        elif hasattr(nested, "id") and nested.id:
                            return nested.id
                        elif hasattr(nested, "roomId") and nested.roomId:
                            return nested.roomId

        # A thread/message ID is not a room ID; returning one here would scope
        # room search and Webex room listing calls to the wrong identifier.
        return None

    def _clean_prompt(self, prompt: str) -> str:
        """Remove bot mentions and HTML tags from the prompt.

        This ensures the AI doesn't get confused by HTML tags or its own name
        appearing in messages.
        """
        if not prompt:
            return prompt

        # Remove HTML tags if present (e.g. <p>, <spark-mention>)
        if "<" in prompt and ">" in prompt:
            prompt = re.sub(r"<[^>]+>", " ", prompt)

        prompt = self._mention_pattern.sub("", prompt).strip()

        for pattern in self._alt_mention_patterns:
            prompt = pattern.sub("", prompt).strip()

        return prompt

    def _fetch_parent_message_via_wdm(self, activity: Any) -> dict[str, Any] | None:
        """Fetch parent message data directly via Webex WDM cluster endpoint.

        This bypasses Webex REST API 404/403 geo-routing restrictions for EU/US clusters
        by using the WDM cluster URL from the WebSocket activity payload.
        """
        if not isinstance(activity, dict):
            return None

        parent = activity.get("parent")
        target = activity.get("target")
        if not isinstance(parent, dict) or not isinstance(target, dict):
            return None

        parent_id = parent.get("id")
        conv_url = target.get("url")
        conv_target_id = target.get("id")

        if not parent_id or not conv_url or not conv_target_id:
            return None

        try:
            msg_url = conv_url.replace(
                f"conversations/{conv_target_id}", f"messages/{parent_id}"
            )
            session = None
            if self.webex_bot and hasattr(self.webex_bot, "websocket_client"):
                session = getattr(self.webex_bot.websocket_client, "session", None)

            if not session and self.webex_bot and hasattr(self.webex_bot, "teams"):
                session = getattr(self.webex_bot.teams, "_session", None)

            if not session:
                return None

            response = session.get(msg_url)
            if response.status_code == 200:
                data = response.json()
                log.info(
                    "Successfully fetched parent root message via WDM cluster: %s",
                    data.get("id"),
                )
                return data
            else:
                log.debug(
                    "WDM cluster request for parent %s returned HTTP %d",
                    parent_id,
                    response.status_code,
                )
        except Exception as e:
            log.debug("Failed to fetch parent message via WDM: %s", e)

        return None

    def _is_bot_author(self, person_id: str | None, person_email: str | None) -> bool:
        """Whether a Webex message was posted by this bot (identity cached)."""
        bot_me = getattr(self, "_bot_me", None)
        if not bot_me:
            bot_me = self.webex_bot.teams.people.me()
            self._bot_me = bot_me
        bot_emails = getattr(bot_me, "emails", None) or []
        bot_email = bot_emails[0] if bot_emails else ""
        bot_id = getattr(bot_me, "id", None)
        return (bot_id is not None and person_id == bot_id) or bool(
            bot_email and person_email == bot_email
        )

    def _is_addressed_to_bot(self, msg: Any) -> bool:
        """Whether the bot handles this message itself (mentioned, or a 1:1 room)."""
        if getattr(msg, "roomType", None) == "direct":
            return True
        bot_me = getattr(self, "_bot_me", None)
        bot_id = getattr(bot_me, "id", None)
        return bool(bot_id and bot_id in (getattr(msg, "mentionedPeople", None) or []))

    def _fetch_thread_root(
        self, thread_id: str, activity: Any
    ) -> tuple[WebexRole, str] | None:
        """Fetch a thread's root message via REST, falling back to the WDM cluster."""
        try:
            root = self.webex_bot.teams.messages.get(messageId=thread_id)
            text = _webex_text(root)
            person_id = getattr(root, "personId", None)
            person_email = getattr(root, "personEmail", None)
        except Exception as root_err:
            log.debug(
                "Could not fetch root thread message %s via REST, trying WDM: %s",
                thread_id,
                root_err,
            )
            wdm_parent = self._fetch_parent_message_via_wdm(activity)
            if not wdm_parent:
                return None
            text = _webex_text(wdm_parent)
            person_id = wdm_parent.get("personId")
            person_email = wdm_parent.get("personEmail")

        content = self._clean_prompt(text)
        if not content:
            return None
        role: WebexRole = (
            "assistant" if self._is_bot_author(person_id, person_email) else "user"
        )
        return role, content

    def _sync_webex_thread_history(
        self,
        thread_id: str,
        room_id: str,
        current_msg_id: str | None = None,
        activity: Any = None,
        current_created: float = 0.0,
        direct: bool = False,
    ) -> None:
        """Sync a thread from Webex, one sync per thread at a time."""
        with self._sync_locks[hash(thread_id) % len(self._sync_locks)]:
            self._sync_webex_thread_history_locked(
                thread_id,
                room_id,
                current_msg_id=current_msg_id,
                activity=activity,
                current_created=current_created,
                direct=direct,
            )

    def _sync_webex_thread_history_locked(
        self,
        thread_id: str,
        room_id: str,
        current_msg_id: str | None = None,
        activity: Any = None,
        current_created: float = 0.0,
        direct: bool = False,
    ) -> None:
        """Add thread replies posted without mentioning the bot to the history.

        Only replies newer than the last sync (and within the conversation
        timeout) are added, so nothing is synced twice. Replies posted after
        the current message (``current_created``) are left for the next sync,
        so they land after this turn's prompt and keep the history in order. The bot's own replies
        are recorded when they are generated, so they, and the thread root,
        are only taken from Webex on a cold start (no local history or
        summary, e.g. after a restart without persistence).
        """
        if not self.webex_bot or not hasattr(self.webex_bot, "teams"):
            return

        try:
            manager = self.conversation_manager
            history = manager.get_history(thread_id)
            cold = not history and not manager.get_thread_summary(thread_id)
            # In a direct room every message is addressed to the bot and stored
            # by its own execute() call, so only a cold start has anything new.
            if direct and not cold:
                return

            since = self._sync_watermarks.get(thread_id)
            if since is None:
                # After a restart: resume from the newest stored Webex time.
                # The local save time is later than replies posted while the
                # last answer was being generated, so it's only a fallback
                # for messages stored without one.
                since = max(
                    (m.metadata.get("webex_created") or 0.0 for m in history),
                    default=0.0,
                ) or (history[-1].timestamp if history else 0.0)
            floor = time.time() - manager.timeout_seconds
            since = max(since, floor)

            replies = self._list_messages(room_id, 30, parentId=thread_id)

            to_add: list[tuple[WebexRole, str, float | None]] = []
            if cold:
                root = self._fetch_thread_root(thread_id, activity)
                if root:
                    to_add.append((*root, None))

            newest = since
            for created, msg in sorted(
                ((_webex_created_ts(m), m) for m in replies), key=lambda p: p[0]
            ):
                if current_created and created > current_created:
                    break
                newest = max(newest, created)
                if created <= since or getattr(msg, "id", None) == current_msg_id:
                    continue
                is_bot = self._is_bot_author(
                    getattr(msg, "personId", None), getattr(msg, "personEmail", None)
                )
                # The bot's replies and messages addressed to it (mentions, or
                # anything in a direct room) are stored by their own execute()
                # call; syncing them too would store a concurrent mention twice.
                if not cold and (is_bot or self._is_addressed_to_bot(msg)):
                    continue
                content = self._clean_prompt(_webex_text(msg))
                if content:
                    to_add.append(("assistant" if is_bot else "user", content, created))
            with self._sync_watermarks_lock:
                # Watermarks older than the timeout floor behave like no
                # watermark, so drop them to keep the dict bounded (only once
                # it has grown, not on every sync).
                if len(self._sync_watermarks) >= _MAX_SYNC_WATERMARKS:
                    self._sync_watermarks = {
                        t: w for t, w in self._sync_watermarks.items() if w >= floor
                    }
                self._sync_watermarks[thread_id] = newest

            for role, content, created in to_add:
                manager.add_message(
                    thread_id=thread_id,
                    role=role,
                    content=content,
                    metadata={"webex_created": created} if created else None,
                    room_id=room_id,
                )
                log.info("Synced Webex message: [%s] %s", role, content[:50])

            if to_add:
                log.info(
                    "Synced %d Webex messages into thread %s", len(to_add), thread_id
                )

        except Exception as e:
            log.warning(
                "Failed to sync Webex thread history for thread %s: %s", thread_id, e
            )

    def _list_messages(self, room_id: str, limit: int, **params: Any) -> list[Any]:
        """List up to ``limit`` recent messages of a room ([] on failure).

        Rooms where listing is forbidden (bots in group spaces may only list
        messages that mention them) are skipped for _ROOM_DENIED_RETRY_SECONDS,
        then retried in case the bot has been given access since.
        """
        denied_at = self._room_list_denied.get(room_id)
        if denied_at is not None:
            if time.time() - denied_at < _ROOM_DENIED_RETRY_SECONDS:
                return []
            self._room_list_denied.pop(room_id, None)
        try:
            return list(
                islice(
                    self.webex_bot.teams.messages.list(
                        roomId=room_id, max=limit, **params
                    ),
                    limit,
                )
            )
        except Exception as list_err:
            # Only 403 is a per-room restriction; 401 is a token problem that
            # must not disable listing for this room.
            if getattr(list_err, "status_code", None) == 403:
                self._room_list_denied[room_id] = time.time()
            log.debug(
                "Could not list messages in room %s (%s): %s",
                room_id,
                params or "room",
                list_err,
            )
            return []

    def _fetch_room_context(
        self, room_id: str, current_msg_id: str | None = None
    ) -> list[tuple[WebexRole, str]]:
        """Recent room messages, used as one-off context for a top-level message.

        They are not stored in the thread history because they belong to
        other threads. Rooms where listing is forbidden (bots in group spaces
        may only list messages that mention them) are skipped afterwards.
        """
        if not self.webex_bot or not hasattr(self.webex_bot, "teams"):
            return []

        room_msgs = self._list_messages(room_id, 20)
        if not room_msgs:
            return []

        try:
            cutoff = time.time() - self.conversation_manager.timeout_seconds
            context: list[tuple[WebexRole, str]] = []
            for created, msg in sorted(
                ((_webex_created_ts(m), m) for m in room_msgs), key=lambda p: p[0]
            ):
                if getattr(msg, "id", None) == current_msg_id or created < cutoff:
                    continue
                content = self._clean_prompt(_webex_text(msg))
                if not content:
                    continue
                is_bot = self._is_bot_author(
                    getattr(msg, "personId", None), getattr(msg, "personEmail", None)
                )
                context.append(("assistant" if is_bot else "user", content))
            return context
        except Exception as e:
            log.warning("Failed to build room context for room %s: %s", room_id, e)
            return []

    def _build_messages(
        self,
        prompt: str,
        thread_id: str | None,
        room_id: str | None = None,
        room_context: list[tuple[WebexRole, str]] | None = None,
        prompt_saved: bool = False,
    ) -> list[dict[str, Any]]:
        """Build the messages list for the LLM API call.

        With ``prompt_saved`` the prompt is already the last message of the
        thread history; otherwise it is appended as a new user message.

        Includes:
        - System prompt with bot identity
        - Conversation history from this thread
        - Recent room messages (top-level messages only, not persisted)
        - Relevant past room context (via SQLite FTS5 search)
        - Current user message
        """
        system_prompt = settings.llm.get_system_prompt(self.bot_name)

        messages = self.conversation_manager.get_messages_for_api(
            thread_id=thread_id or "",
            system_prompt=system_prompt,
        )

        # Keep the current turn last, after the context blocks below
        if prompt_saved and messages and messages[-1]["role"] == "user":
            current_turn = messages.pop()
        else:
            current_turn = {"role": "user", "content": prompt}

        if room_context:
            messages.append(
                {
                    "role": "system",
                    "content": "[Recent Messages in this Webex Space]\n"
                    + "\n".join(
                        f"[{role}]: {content}" for role, content in room_context
                    ),
                }
            )

        # Augment with relevant room history via FTS5 if room_id is present
        if room_id and prompt:
            past_matches = self.conversation_manager.search_room_history_sync(
                room_id=room_id,
                query=prompt,
                limit=3,
            )

            # Filter out messages that are already in the current thread history
            existing_contents = {
                m.get("content") for m in messages if isinstance(m, dict)
            }
            existing_contents.update(content for _, content in room_context or [])
            # Consolidation merges consecutive same-role messages, so compare
            # against the individual thread messages as well.
            if thread_id:
                existing_contents.update(
                    m.content for m in self.conversation_manager.get_history(thread_id)
                )
            # The prompt is already persisted, so the search finds it too
            existing_contents.update((current_turn["content"], prompt))
            relevant_snippets = [
                f"[{msg.role}]: {msg.content}"
                for msg in past_matches
                if msg.content not in existing_contents
            ]

            if relevant_snippets:
                context_block = (
                    "[Relevant Past Discussion Context in this Webex Space]\n"
                    + "\n".join(relevant_snippets)
                )
                log.info(
                    "Augmenting prompt with %d historical room matches for room %s",
                    len(relevant_snippets),
                    room_id,
                )
                messages.append({"role": "system", "content": context_block})

        messages.append(current_turn)
        return messages

    def _get_mcp_tools(self) -> list[dict]:
        """Get available tools (MCP tools + built-in tools) formatted for LiteLLM.

        Works for all models:
        - OpenAI: Returns native tool_calls
        - Ollama: Returns JSON in content that we parse
        """
        tools = []
        if self.mcp_client and self.mcp_client.available_tools:
            tools.extend(self.mcp_client.get_tools_for_litellm())

        if self.conversation_manager and self.conversation_manager.enable_persistence:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "search_room_history",
                        "description": (
                            "Search historical messages and past discussions across all threads "
                            "in the current Webex space using keyword search. Use this tool when "
                            "a user explicitly asks to search chat history, look up past error "
                            "solutions, or recall previously posted configurations, URLs, or information."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Keywords or search terms to look up in space history.",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            )

        if tools:
            log.info("Including %d tools in API request", len(tools))
        return tools

    def _call_mcp_tool(self, tool_name: str, arguments: dict) -> MCPToolResult:
        """Call an MCP tool synchronously using a thread pool.

        This runs the async MCP call in a separate thread to avoid
        event loop conflicts with webex_bot.
        """
        client = self.mcp_client
        if not client:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error="MCP client not configured",
            )

        async def _async_call():
            return await client.call_tool(tool_name, arguments)

        try:
            future = _executor.submit(_run_async_in_thread, _async_call())
            result = future.result(timeout=settings.mcp.request_timeout + 5)
            return result
        except concurrent.futures.TimeoutError:
            log.error("MCP tool '%s' timed out", tool_name)
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool call timed out after {settings.mcp.request_timeout}s",
            )
        except Exception as e:
            log.exception("MCP tool '%s' error: %s", tool_name, e)
            set_tag("mcp_tool", tool_name)
            capture_exception(e)
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
            )

    def _find_tool(self, tool_name: str) -> MCPTool | None:
        """Return the MCP tool definition with this name, if any."""
        if not self.mcp_client:
            return None
        return self.mcp_client.get_tool(tool_name)

    def _tool_exists(self, tool_name: str) -> bool:
        """Check whether a tool name refers to a built-in or MCP tool."""
        return (
            tool_name == "search_room_history" or self._find_tool(tool_name) is not None
        )

    def _handle_tool_calls(
        self, tool_calls: list, room_id: str | None = None
    ) -> dict[str, str]:
        """Handle LLM function calls to MCP and built-in tools.

        Args:
            tool_calls: List of tool calls from the LLM response.
            room_id: Optional Webex room ID for room-scoped search tools.

        Returns:
            Dictionary mapping tool call IDs to their results.
        """
        results = {}

        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            # Strip common prefixes that models might add (e.g., "tool.", "tool_"),
            # but only when the prefixed name is not itself a real tool.
            if tool_name.startswith(("tool.", "tool_")) and not self._tool_exists(
                tool_name
            ):
                tool_name = tool_name[5:]

            raw_args = tool_call.function.arguments
            if isinstance(raw_args, dict):
                arguments = raw_args
            elif isinstance(raw_args, str) and raw_args.strip():
                try:
                    arguments = json.loads(raw_args)
                except Exception:
                    arguments = {}
            else:
                arguments = {}
            # e.g. "null" or a list: treat like no arguments, as JSON mode does
            if not isinstance(arguments, dict):
                arguments = {}

            # Built-in search_room_history tool handling
            if tool_name == "search_room_history":
                try:
                    query = (
                        arguments.get("query")
                        or arguments.get("q")
                        or arguments.get("search_term")
                        or arguments.get("keywords")
                        or arguments.get("text")
                        or ""
                    )
                    if room_id and query:
                        matches = self.conversation_manager.search_room_history_sync(
                            room_id=room_id, query=query, limit=5
                        )
                        if matches:
                            formatted = "\n".join(
                                f"[{m.role}]: {m.content}" for m in matches
                            )
                            results[tool_call.id] = (
                                f"Found {len(matches)} matching historical messages in this Webex space:\n{formatted}"
                            )
                        else:
                            results[tool_call.id] = (
                                f"No historical messages matching '{query}' were found in this Webex space."
                            )
                    else:
                        results[tool_call.id] = (
                            f"Search failed: room_id='{room_id}' or query='{query}' was missing."
                        )
                    log.info(
                        "Executed built-in tool 'search_room_history' for query '%s' in room %s",
                        query,
                        room_id,
                    )
                except Exception as e:
                    log.exception("Error executing search_room_history tool: %s", e)
                    results[tool_call.id] = f"Error executing room search: {e}"
                continue

            # Validate that the MCP tool exists
            tool_def = self._find_tool(tool_name)
            if self.mcp_client and tool_def is None:
                log.warning(
                    "Tool '%s' does not exist. Available tools: %s",
                    tool_name,
                    [t.name for t in self.mcp_client.available_tools],
                )
                results[tool_call.id] = f"Error: Tool '{tool_name}' does not exist"
                continue

            try:
                # Validate arguments against the tool schema. Only filter when
                # the schema declares its parameters and does not allow others;
                # otherwise every argument would be dropped.
                schema = (tool_def.input_schema or {}) if tool_def else {}
                properties = schema.get("properties") or {}
                if properties and not schema.get("additionalProperties"):
                    # Filter arguments to only include those defined in the schema
                    valid_arguments = {
                        k: v for k, v in arguments.items() if k in properties
                    }

                    if valid_arguments != arguments:
                        invalid_args = set(arguments.keys()) - set(properties.keys())
                        # Names only: argument values can carry user text, and
                        # WARNING logs reach Sentry even without PII enabled.
                        log.warning(
                            "Tool '%s' received invalid arguments: %s. "
                            "Only passing valid arguments: %s",
                            tool_name,
                            sorted(invalid_args),
                            sorted(valid_arguments),
                        )
                        arguments = valid_arguments

                log.info("Executing MCP tool '%s' with args: %s", tool_name, arguments)

                result = self._call_mcp_tool(tool_name, arguments)

                if result.success:
                    results[tool_call.id] = str(result.result)
                    log.info("Tool '%s' executed successfully", tool_name)
                else:
                    results[tool_call.id] = f"Error: {result.error}"
                    log.warning("Tool '%s' failed: %s", tool_name, result.error)

            except json.JSONDecodeError as e:
                log.error("Error parsing tool arguments for '%s': %s", tool_name, e)
                results[tool_call.id] = f"Error: Invalid JSON arguments - {e}"
            except Exception as e:
                log.exception("Error executing tool '%s': %s", tool_name, e)
                results[tool_call.id] = f"Error: {type(e).__name__}: {e}"

        return results

    def _call_llm(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> Any:
        """Make a synchronous LLM API call using LiteLLM.

        Args:
            messages: List of message dicts for the API.
            tools: Optional list of tool definitions.

        Returns:
            LiteLLM completion response.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if settings.llm.api_base:
            kwargs["api_base"] = settings.llm.api_base

        if settings.llm.api_key:
            kwargs["api_key"] = settings.llm.api_key

        if tools:
            kwargs["tools"] = tools

        return litellm.completion(**kwargs)

    def _set_sentry_context(self, activity: Any) -> None:
        """Extract user info from activity and set Sentry user context."""
        try:
            from src.sentry import set_user_context

            user_id = None
            email = None
            username = None

            if isinstance(activity, dict):
                actor = activity.get("actor", {})
                if isinstance(actor, dict):
                    user_id = actor.get("entryUUID") or actor.get("id")
                    email = actor.get("emailAddress")
                    username = actor.get("displayName")
            else:
                actor = getattr(activity, "actor", None)
                if actor:
                    user_id = getattr(actor, "entryUUID", None) or getattr(
                        actor, "id", None
                    )
                    email = getattr(actor, "emailAddress", None)
                    username = getattr(actor, "displayName", None)

            if user_id:
                set_user_context(user_id, email, username)
                log.debug(
                    "Sentry user context set: user_id=%s, email=%s", user_id, email
                )
        except Exception as e:
            log.exception("Error setting Sentry user context: %s", e)

    def _handle_native_tool_calls(
        self,
        choice: Any,
        messages: list[dict],
        room_id: str | None = None,
        tools: list[dict] | None = None,
    ) -> str | None:
        """Handle native function call-based tool execution from LLM.

        Runs tool calls and sends the results back until the model answers
        with text, allowing chained lookups for up to _MAX_TOOL_ROUNDS rounds.
        After the last round a text answer is forced (no tools offered).

        Returns updated response content string ("" if the model produced no
        answer), or None if no native tool calls occurred.
        """
        for round_num in range(1, _MAX_TOOL_ROUNDS + 1):
            tool_calls = getattr(choice.message, "tool_calls", None)
            if not tool_calls:
                return (choice.message.content or "") if round_num > 1 else None

            log.info(
                "LLM made %d function call(s) (round %d)", len(tool_calls), round_num
            )
            tool_results = self._handle_tool_calls(tool_calls, room_id=room_id)
            if not tool_results:
                return (choice.message.content or "") if round_num > 1 else None

            messages.append(
                {
                    "role": "assistant",
                    "content": choice.message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": (
                                    tc.function.arguments
                                    if isinstance(tc.function.arguments, str)
                                    else json.dumps(tc.function.arguments or {})
                                ),
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            messages.extend(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(
                        tool_results.get(tool_call.id, "Tool execution failed")
                    ),
                }
                for tool_call in tool_calls
            )

            if round_num == _MAX_TOOL_ROUNDS:
                break

            # The tool definitions must be resent: providers such as Anthropic
            # and Bedrock reject a history containing tool_use/tool_result
            # without them.
            log.info("Sending tool results back to LLM")
            completion = self._call_llm(messages, tools=tools or None)
            if not completion.choices:
                return ""
            choice = completion.choices[0]

        # Out of tool rounds: force a text answer from the results gathered so
        # far. Without tool definitions the model cannot call tools, and with
        # the tool exchange rewritten as plain text every provider accepts the
        # request (tool_choice="none" is not supported everywhere).
        log.info("Tool round limit reached; requesting a text answer")
        # The nudge goes through the flattening too, so it is merged into the
        # trailing tool-result user message instead of following it.
        final_messages = _flatten_tool_messages(
            [*messages, {"role": "user", "content": _FINAL_ANSWER_NUDGE}]
        )
        completion = self._call_llm(final_messages, tools=None)
        if not completion.choices:
            return ""
        return completion.choices[0].message.content or ""

    @staticmethod
    def _parse_json_tool_call(content: str | None) -> tuple[str, dict] | None:
        """Parse a JSON-mode tool call ``{"name": ..., "arguments": ...}``.

        Accepts the JSON bare or in a code fence. Returns (name, arguments),
        or None if the content is not a tool call.
        """
        if not content:
            return None

        json_str = content.strip()
        if "```json" in json_str:
            start = json_str.find("```json") + 7
            end = json_str.find("```", start)
            if end > start:
                json_str = json_str[start:end].strip()
        elif "```" in json_str:
            start = json_str.find("```") + 3
            end = json_str.find("```", start)
            if end > start:
                json_str = json_str[start:end].strip()

        try:
            json_data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return None
        if not (
            isinstance(json_data, dict)
            and "name" in json_data
            and "arguments" in json_data
        ):
            return None

        name = str(json_data["name"])
        arguments = json_data["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                log.warning(
                    "Failed to parse tool arguments as JSON for '%s': %s", name, e
                )
                arguments = {}
        if not isinstance(arguments, dict):
            log.warning(
                "Tool arguments for '%s' is not a dict or JSON string: %s",
                name,
                type(arguments),
            )
            arguments = {}
        return name, arguments

    def _handle_json_tool_calls(
        self,
        response_content: str | None,
        messages: list[dict],
        room_id: str | None = None,
    ) -> str | None:
        """Handle JSON-mode tool calls from non-native LLM providers (e.g. Ollama).

        Like the native path, chained calls are run for up to _MAX_TOOL_ROUNDS
        rounds, and the last round asks for a text answer. A tool call is
        never returned as the answer.

        Returns updated response content string, or None if no JSON tool call occurred.
        """
        parsed = self._parse_json_tool_call(response_content)
        if parsed is None:
            return None

        content = response_content or ""
        for round_num in range(1, _MAX_TOOL_ROUNDS + 1):
            name, arguments = parsed
            log.info("Detected JSON-mode tool call: %s (round %d)", name, round_num)
            tool_call = types.SimpleNamespace(
                id=f"json_call_{round_num}",
                function=types.SimpleNamespace(
                    name=name, arguments=json.dumps(arguments)
                ),
            )
            result = self._handle_tool_calls([tool_call], room_id=room_id).get(
                tool_call.id, "Tool execution failed"
            )

            nudge = (
                _FINAL_ANSWER_NUDGE
                if round_num == _MAX_TOOL_ROUNDS
                else (
                    "Use this result to answer the user's original question. "
                    "If you need more information, call another tool."
                )
            )
            # Non-empty: Anthropic/Bedrock reject empty assistant turns
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool '{name}' returned: {result}\n\n{nudge}",
                }
            )

            log.info("Sending tool results back to LLM")
            completion = self._call_llm(messages, tools=None)
            if not completion.choices:
                return ""
            content = completion.choices[0].message.content or ""
            parsed = self._parse_json_tool_call(content)
            if parsed is None:
                return content

        log.warning(
            "LLM still requested tools after %d JSON-mode rounds; giving up",
            _MAX_TOOL_ROUNDS,
        )
        return (
            "I couldn't finish gathering the information needed to answer. "
            "Please try a more specific question."
        )

    def execute(
        self,
        message: str,
        attachment_actions: Any,
        activity: Any,
    ) -> list[str]:
        """Execute the AI command to respond to user message.

        Args:
            message: The user's message with command keyword stripped.
            attachment_actions: Attachment actions object (unused).
            activity: The Webex activity object.

        Returns:
            List of response strings.
        """
        self._set_sentry_context(activity)

        if not message or not message.strip():
            log.warning("Received empty message")
            return ["Please ask me a question! Mention me with your query."]

        log.info("Raw message: '%s'", message[:100])
        prompt = self._clean_prompt(message.strip())

        if not prompt:
            log.warning("Message was empty after cleaning")
            return ["Please ask me a question! I'm here to help."]

        thread_id = self._get_thread_id(activity, attachment_actions)
        room_id = self._get_room_id(activity, attachment_actions)
        current_msg_id = (
            getattr(attachment_actions, "id", None)
            or getattr(activity, "id", None)
            or (activity.get("id") if isinstance(activity, dict) else None)
        )

        created = _webex_created_ts(attachment_actions)
        room_context: list[tuple[WebexRole, str]] = []
        if thread_id and room_id:
            if thread_id == current_msg_id:
                # New top-level message: no thread to sync yet, only the room
                room_context = self._fetch_room_context(room_id, current_msg_id)
            else:
                self._sync_webex_thread_history(
                    thread_id,
                    room_id,
                    current_msg_id=current_msg_id,
                    activity=activity,
                    current_created=created,
                    direct=getattr(attachment_actions, "roomType", None) == "direct",
                )

        if thread_id:
            # Save the prompt before the LLM call so it isn't lost if the call
            # fails (the sync watermark has already moved past it). Its Webex
            # time lets the next sync resume from here after a restart.
            self.conversation_manager.add_user_message(
                thread_id,
                prompt,
                room_id=room_id,
                metadata={"webex_created": created} if created else None,
            )

        log.info("=" * 60)
        log.info("PROCESSING MESSAGE")
        log.info("Thread ID: %s, Room ID: %s", thread_id, room_id)
        log.info(
            "Has history: %s",
            self.conversation_manager.has_history(thread_id) if thread_id else False,
        )
        log.info("Cleaned prompt: '%s'", prompt[:100])
        log.info("=" * 60)

        try:
            messages = self._build_messages(
                prompt,
                thread_id,
                room_id=room_id,
                room_context=room_context,
                prompt_saved=bool(thread_id),
            )
            tools = self._get_mcp_tools()

            log.info(
                "Sending to LLM: model=%s, temperature=%.1f, max_tokens=%d, tools=%d",
                self.model,
                self.temperature,
                self.max_tokens,
                len(tools),
            )

            completion = self._call_llm(messages, tools or None)

            if not completion.choices:
                log.error("LLM returned empty choices")
                return ["I'm sorry, I couldn't generate a response. Please try again."]

            choice = completion.choices[0]
            raw_content = choice.message.content

            # Attempt native or JSON tool call handling
            updated_content = self._handle_native_tool_calls(
                choice, messages, room_id=room_id, tools=tools
            )
            if updated_content is None:
                updated_content = self._handle_json_tool_calls(
                    raw_content, messages, room_id=room_id
                )

            response_content = (
                updated_content if updated_content is not None else raw_content
            )

            if not response_content or not response_content.strip():
                log.error("LLM returned empty content")
                return ["I'm sorry, I received an empty response. Please try again."]

            if thread_id:
                self.conversation_manager.add_assistant_message(
                    thread_id, response_content, room_id=room_id
                )
                log.info(
                    "Saved conversation to thread %s (room %s). Total: %d messages",
                    thread_id,
                    room_id,
                    self.conversation_manager.get_message_count(thread_id),
                )

            log.info("Response generated (%d chars)", len(response_content))
            return [quote_info(response_content)]

        except Exception as e:
            log.exception("Error calling LLM: %s", e)
            set_tag("llm_model", self.model)
            set_tag("thread_id", thread_id or "none")
            capture_exception(e)
            return [
                f"I'm sorry, I encountered an error: {type(e).__name__}. "
                "Please try again later."
            ]

    def pre_execute(
        self,
        message: str,  # noqa: ARG002
        attachment_actions: Any,  # noqa: ARG002
        activity: Any,  # noqa: ARG002
    ) -> str | None:
        """Optional pre-execution hook."""
        return None
