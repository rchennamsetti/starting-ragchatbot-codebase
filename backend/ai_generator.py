import anthropic
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of rounds in which tools are offered to Claude per query
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the course outline tool for course structure, syllabus, outline, lesson-list, and course-link questions
- Use the content search tool for questions about specific course content or detailed educational materials
- When using the course outline tool, return the course title, the course link, and the complete lesson list with each lesson number and lesson title
- You may use tools across up to 2 sequential rounds per query: call a tool, review its results, and — only if genuinely needed to answer the question — call a second tool informed by what you learned (e.g., look up a course outline to find a lesson title, then search course content using that title)
- Do not call tools speculatively or repeat an identical call — only make a second call if the first result indicates it's necessary
- Synthesize all gathered results into a single accurate, fact-based response
- If a search or outline lookup yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response, allowing Claude to make sequential tool calls
        across multiple API rounds, with conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        return self._run_tool_loop(messages, system_content, tools, tool_manager)

    def _run_tool_loop(self, messages: List[Dict[str, Any]], system_content: str,
                        tools: Optional[List], tool_manager) -> str:
        """
        Drive up to MAX_TOOL_ROUNDS rounds of tool-calling. Each round is a
        separate API call so Claude can reason about previous tool results
        before deciding whether to call another tool or answer directly.

        Args:
            messages: Conversation messages, mutated in place across rounds
            system_content: System prompt (with conversation history, if any)
            tools: Available tool definitions
            tool_manager: Manager to execute tools

        Returns:
            Final response text
        """
        round_num = 0
        force_final = False

        while True:
            round_num += 1
            offer_tools = bool(tools) and not force_final and round_num <= self.MAX_TOOL_ROUNDS

            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }
            if offer_tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            logger.debug(
                "Round %d: calling Claude model=%s (tools_offered=%s, message_count=%d)",
                round_num, self.model, offer_tools, len(messages)
            )

            response = self.client.messages.create(**api_params)

            logger.debug("Round %d: Claude stop_reason=%s", round_num, response.stop_reason)

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            # Nothing left to execute: no tool_use requested, tools weren't
            # offered this round, or there's no tool_manager to run them.
            if response.stop_reason != "tool_use" or not tool_use_blocks or not offer_tools:
                if response.stop_reason == "tool_use" and tool_use_blocks and not offer_tools:
                    logger.warning(
                        "Round %d: Claude returned tool_use though tools were not offered; ignoring",
                        round_num
                    )
                else:
                    logger.info("Round %d: terminating - no further tool use requested", round_num)
                return self._extract_text(response)

            if tool_manager is None:
                logger.warning("Round %d: tool_use requested but no tool_manager provided", round_num)
                return self._extract_text(response)

            # Add Claude's tool use turn to the conversation
            messages.append({"role": "assistant", "content": response.content})

            # Execute each requested tool call, tolerating individual failures
            tool_results = []
            any_error = False
            for block in tool_use_blocks:
                logger.info(
                    "Round %d: executing tool '%s' with input %s",
                    round_num, block.name, block.input
                )
                try:
                    tool_result = tool_manager.execute_tool(block.name, **block.input)
                    logger.debug("Round %d: tool '%s' result: %s", round_num, block.name, tool_result)
                except Exception as exc:
                    any_error = True
                    logger.warning(
                        "Round %d: tool '%s' raised an exception: %s",
                        round_num, block.name, exc, exc_info=True
                    )
                    tool_result = f"Tool '{block.name}' failed with error: {exc}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result
                })

            messages.append({"role": "user", "content": tool_results})

            if any_error:
                logger.info("Round %d: terminating tool rounds - tool execution error", round_num)
                force_final = True

    def _extract_text(self, response) -> str:
        """Return the first text block's content, or an empty string if none exists."""
        for block in response.content:
            if block.type == "text":
                return block.text
        logger.warning("No text block found in Claude response; returning empty string")
        return ""