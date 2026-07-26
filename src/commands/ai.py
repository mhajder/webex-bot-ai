"""AI command for handling user questions with LiteLLM and thread context."""

import asyncio
import concurrent.futures
import json
import logging
import re
from datetime import datetime
from typing import Any

import litellm
from webex_bot.formatting import quote_info
from webex_bot.models.command import Command

from src.config import settings
from src.conversation.manager import ConversationManager
from src.mcp_client.client import MCPMultiClient
from src.mcp_client.types import MCPToolResult
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

        self.conversation_manager = conversation_manager or ConversationManager(
            max_messages=settings.conversation.max_history_messages,
            timeout_seconds=settings.conversation.timeout_seconds,
            db_path=settings.conversation.db_path,
            enable_persistence=settings.conversation.enable_persistence,
        )

        self.mcp_client = mcp_client

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

        self._mention_pattern = re.compile(
            rf"^\s*@?\s*{escaped_name}\s*[:,.\s]*",
            re.IGNORECASE,
        )

        self._alt_mention_patterns = []
        display_name = settings.bot.display_name
        if display_name and display_name.lower() != self.bot_name.lower():
            escaped_display = re.escape(display_name)
            alt_pattern = re.compile(
                rf"^\s*@?\s*{escaped_display}\s*[:,.\s]*",
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

        # Fallback to thread_id if room_id cannot be explicitly extracted
        return self._get_thread_id(activity, attachment_actions)

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

    def _sync_webex_thread_history(
        self,
        thread_id: str,
        room_id: str,
        current_msg_id: str | None = None,
        activity: Any = None,
    ) -> None:
        """Fetch pre-existing Webex thread and room messages via Webex API / WDM cluster.

        This catches up on unmentioned user messages posted in the space or thread
        before the bot was pinged.
        """
        if not self.webex_bot or not hasattr(self.webex_bot, "teams"):
            return

        try:
            teams = self.webex_bot.teams
            bot_me = getattr(self, "_bot_me", None)
            if not bot_me:
                bot_me = teams.people.me()
                self._bot_me = bot_me

            bot_person_id = getattr(bot_me, "id", None)
            bot_emails = getattr(bot_me, "emails", [])
            bot_email = bot_emails[0] if bot_emails else ""

            webex_messages = []
            existing_ids = set()

            # First try fetching parent message via direct WDM cluster endpoint
            if activity and isinstance(activity, dict) and "parent" in activity:
                wdm_parent = self._fetch_parent_message_via_wdm(activity)
                if wdm_parent:
                    raw_parent_text = (
                        wdm_parent.get("text")
                        or wdm_parent.get("markdown")
                        or wdm_parent.get("html")
                        or ""
                    )
                    if raw_parent_text.strip():
                        cleaned_parent = self._clean_prompt(raw_parent_text.strip())
                        if cleaned_parent:
                            parent_sender_id = wdm_parent.get("personId")
                            parent_sender_email = wdm_parent.get("personEmail")
                            parent_is_bot = parent_sender_id == bot_person_id or (
                                bot_email and parent_sender_email == bot_email
                            )
                            parent_role = "assistant" if parent_is_bot else "user"

                            self.conversation_manager.add_message(
                                thread_id=thread_id,
                                role=parent_role,
                                content=cleaned_parent,
                                room_id=room_id,
                            )
                            log.info(
                                "Synced parent root message via WDM: [%s] %s",
                                parent_role,
                                cleaned_parent[:50],
                            )

            # Fetch thread root message if thread_id is specified
            if thread_id:
                try:
                    root_msg = teams.messages.get(messageId=thread_id)
                    if root_msg and hasattr(root_msg, "id"):
                        webex_messages.append(root_msg)
                        existing_ids.add(root_msg.id)
                except Exception as root_err:
                    log.debug(
                        "Could not fetch root thread message %s (Webex API restriction/404): %s",
                        thread_id,
                        root_err,
                    )

            # Fetch thread replies if thread_id is specified
            if room_id and thread_id:
                try:
                    replies = list(
                        teams.messages.list(roomId=room_id, parentId=thread_id, max=30)
                    )
                    webex_messages.extend(
                        [
                            r
                            for r in replies
                            if hasattr(r, "id") and r.id not in existing_ids
                        ]
                    )
                    existing_ids.update(r.id for r in replies if hasattr(r, "id"))
                except Exception as list_err:
                    log.debug(
                        "Could not fetch thread replies for thread %s: %s",
                        thread_id,
                        list_err,
                    )

            # Also fetch recent messages in the room/space if permitted
            if room_id:
                try:
                    room_msgs = list(teams.messages.list(roomId=room_id, max=20))
                    webex_messages.extend(
                        [
                            m
                            for m in room_msgs
                            if hasattr(m, "id") and m.id not in existing_ids
                        ]
                    )
                    existing_ids.update(m.id for m in room_msgs if hasattr(m, "id"))
                except Exception as room_err:
                    log.debug(
                        "Could not fetch room messages for room %s (Webex API 403/404): %s",
                        room_id,
                        room_err,
                    )

            if not webex_messages:
                return

            def get_created_timestamp(msg: Any) -> float:
                created = getattr(msg, "created", None)
                if isinstance(created, datetime):
                    return created.timestamp()
                if isinstance(created, str):
                    try:
                        return datetime.fromisoformat(created).timestamp()
                    except ValueError:
                        pass
                return 0.0

            webex_messages.sort(key=get_created_timestamp)

            existing_history = self.conversation_manager.get_history(thread_id)
            existing_contents = {m.content.strip() for m in existing_history}
            synced_count = 0

            for msg in webex_messages:
                msg_id = getattr(msg, "id", None)
                if current_msg_id and msg_id == current_msg_id:
                    continue

                raw_text = (
                    getattr(msg, "text", None)
                    or getattr(msg, "markdown", None)
                    or getattr(msg, "html", None)
                    or ""
                )
                if not raw_text.strip():
                    continue

                msg_person_id = getattr(msg, "personId", None)
                msg_person_email = getattr(msg, "personEmail", None)

                is_bot = msg_person_id == bot_person_id or (
                    bot_email and msg_person_email == bot_email
                )
                role = "assistant" if is_bot else "user"
                cleaned_content = self._clean_prompt(raw_text.strip())

                if not cleaned_content:
                    continue

                if cleaned_content not in existing_contents:
                    self.conversation_manager.add_message(
                        thread_id=thread_id,
                        role=role,
                        content=cleaned_content,
                        room_id=room_id,
                    )
                    existing_contents.add(cleaned_content)
                    synced_count += 1
                    log.info(
                        "Synced unmentioned Webex message: [%s] %s",
                        role,
                        cleaned_content[:50],
                    )

            if synced_count > 0:
                log.info(
                    "Successfully synced %d unmentioned Webex messages for thread %s",
                    synced_count,
                    thread_id,
                )

        except Exception as e:
            log.warning(
                "Failed to sync Webex thread history for thread %s: %s", thread_id, e
            )

    def _build_messages(
        self,
        prompt: str,
        thread_id: str | None,
        room_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the messages list for the LLM API call.

        Includes:
        - System prompt with bot identity
        - Conversation history from this thread
        - Relevant past room context (via SQLite FTS5 search)
        - Current user message
        """
        system_prompt = settings.llm.get_system_prompt(self.bot_name)

        messages = self.conversation_manager.get_messages_for_api(
            thread_id=thread_id or "",
            system_prompt=system_prompt,
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

        if messages and messages[-1]["role"] == "user":
            if prompt not in messages[-1]["content"]:
                messages[-1]["content"] += f"\n\n{prompt}"
        else:
            messages.append({"role": "user", "content": prompt})

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

            # Strip common prefixes that models might add (e.g., "tool.", "tool_")
            if tool_name.startswith(("tool.", "tool_")):
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
            if self.mcp_client:
                tool_exists = any(
                    tool.name == tool_name for tool in self.mcp_client.available_tools
                )

                if not tool_exists:
                    log.warning(
                        "Tool '%s' does not exist. Available tools: %s",
                        tool_name,
                        [t.name for t in self.mcp_client.available_tools],
                    )
                    results[tool_call.id] = f"Error: Tool '{tool_name}' does not exist"
                    continue

            try:
                # Validate arguments against the tool schema
                if self.mcp_client:
                    # Get the tool definition to check its schema
                    tool_def = None
                    for tool in self.mcp_client.available_tools:
                        if tool.name == tool_name:
                            tool_def = tool
                            break

                    if tool_def:
                        # Get allowed parameters from schema
                        schema = tool_def.input_schema or {}
                        properties = schema.get("properties", {})

                        # Filter arguments to only include those defined in the schema
                        valid_arguments = {
                            k: v for k, v in arguments.items() if k in properties
                        }

                        if valid_arguments != arguments:
                            invalid_args = set(arguments.keys()) - set(
                                properties.keys()
                            )
                            log.warning(
                                "Tool '%s' received invalid arguments: %s. "
                                "Only passing valid arguments: %s",
                                tool_name,
                                invalid_args,
                                valid_arguments,
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
    ) -> str | None:
        """Handle native function call-based tool execution from LLM.

        Returns updated response content string, or None if no native tool calls occurred.
        """
        if not (hasattr(choice.message, "tool_calls") and choice.message.tool_calls):
            return None

        log.info("LLM made %d function call(s)", len(choice.message.tool_calls))
        tool_results = self._handle_tool_calls(
            choice.message.tool_calls, room_id=room_id
        )

        if not tool_results:
            return None

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
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ],
            }
        )

        for tool_call in choice.message.tool_calls:
            result = tool_results.get(tool_call.id, "Tool execution failed")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": (
                    "Based on the tool results above, please provide a clear and "
                    "concise response to the user's original question. Do NOT call "
                    "any tools. Just respond with text."
                ),
            }
        )

        log.info("Sending tool results back to LLM")
        completion = self._call_llm(messages, tools=None)
        if completion.choices:
            return completion.choices[0].message.content
        return None

    def _handle_json_tool_calls(
        self,
        response_content: str,
        messages: list[dict],
        room_id: str | None = None,
    ) -> str | None:
        """Handle JSON-mode tool calls from non-native LLM providers (e.g. Ollama).

        Returns updated response content string, or None if no JSON tool call occurred.
        """
        if not response_content:
            return None

        try:
            json_str = response_content.strip()

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

            json_data = json.loads(json_str)

            if (
                isinstance(json_data, dict)
                and "name" in json_data
                and "arguments" in json_data
            ):
                log.info("✅ Detected JSON-mode tool call: %s", json_data["name"])
                log.debug(
                    "Raw JSON-mode tool call data: %s",
                    json.dumps(json_data, indent=2),
                )

                tool_arguments = json_data["arguments"]

                if not isinstance(tool_arguments, dict):
                    try:
                        if isinstance(tool_arguments, str):
                            tool_arguments = json.loads(tool_arguments)
                        else:
                            log.warning(
                                "Tool arguments for '%s' is not a dict or JSON string: %s",
                                json_data["name"],
                                type(tool_arguments),
                            )
                            tool_arguments = {}
                    except json.JSONDecodeError as e:
                        log.warning(
                            "Failed to parse tool arguments as JSON for '%s': %s",
                            json_data["name"],
                            e,
                        )
                        tool_arguments = {}

                tool_call = type(
                    "ToolCall",
                    (),
                    {
                        "id": "json_call_0",
                        "function": type(
                            "Function",
                            (),
                            {
                                "name": json_data["name"],
                                "arguments": json.dumps(tool_arguments),
                            },
                        )(),
                    },
                )()

                tool_results = self._handle_tool_calls([tool_call], room_id=room_id)

                if tool_results:
                    messages.append({"role": "assistant", "content": ""})
                    result = tool_results.get("json_call_0", "Tool execution failed")
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Tool '{json_data['name']}' returned: {result}\n\n"
                                "Based on this tool result, please provide a clear and "
                                "concise response to the user's original question. "
                                "Do NOT call any tools. Just respond with text."
                            ),
                        }
                    )

                    log.info("Sending tool results back to LLM")
                    completion = self._call_llm(messages, tools=None)
                    if completion.choices:
                        return completion.choices[0].message.content
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass

        return None

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

        if (
            self.bot_name.lower() == "assistant"
            and self.webex_bot
            and hasattr(self.webex_bot, "teams_bot_email")
            and isinstance(self.webex_bot.teams_bot_email, str)
            and self.webex_bot.teams_bot_email
        ):
            email_part = self.webex_bot.teams_bot_email.split("@")[0].split("-")[0]
            self.bot_name = email_part.title()
            self._build_mention_patterns()
            log.info("Extracted bot name from email: %s", self.bot_name)

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

        if thread_id and room_id:
            self._sync_webex_thread_history(
                thread_id,
                room_id,
                current_msg_id=current_msg_id,
                activity=activity,
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
            messages = self._build_messages(prompt, thread_id, room_id=room_id)
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
                choice, messages, room_id=room_id
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
                self.conversation_manager.add_user_message(
                    thread_id, prompt, room_id=room_id
                )
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
