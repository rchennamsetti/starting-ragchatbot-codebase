from unittest.mock import Mock

import pytest

from ai_generator import AIGenerator


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeToolUseBlock:
    def __init__(self, name, tool_input, block_id="toolu_1"):
        self.type = "tool_use"
        self.name = name
        self.input = tool_input
        self.id = block_id


class FakeResponse:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


@pytest.fixture
def generator():
    gen = AIGenerator(api_key="test-key", model="test-model")
    gen.client = Mock()
    return gen


def test_direct_answer_when_no_tool_needed(generator):
    generator.client.messages.create.return_value = FakeResponse(
        content=[FakeTextBlock("Paris is the capital of France.")],
        stop_reason="end_turn",
    )

    result = generator.generate_response("What is the capital of France?")

    assert result == "Paris is the capital of France."
    generator.client.messages.create.assert_called_once()


def test_tools_are_attached_when_provided(generator):
    generator.client.messages.create.return_value = FakeResponse(
        content=[FakeTextBlock("answer")], stop_reason="end_turn"
    )
    tools = [{"name": "search_course_content"}]

    generator.generate_response("some question", tools=tools)

    call_kwargs = generator.client.messages.create.call_args.kwargs
    assert call_kwargs["tools"] == tools
    assert call_kwargs["tool_choice"] == {"type": "auto"}


def test_tool_use_round_trip(generator):
    first_response = FakeResponse(
        content=[
            FakeToolUseBlock(
                "search_course_content",
                {"query": "lesson 2 content", "course_name": "Course A"},
            )
        ],
        stop_reason="tool_use",
    )
    final_response = FakeResponse(
        content=[FakeTextBlock("Lesson 2 covers vector search.")],
        stop_reason="end_turn",
    )
    generator.client.messages.create.side_effect = [first_response, final_response]

    tool_manager = Mock()
    tool_manager.execute_tool.return_value = "[Course A - Lesson 2]\nVector search content..."

    result = generator.generate_response(
        "What does lesson 2 cover?",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert result == "Lesson 2 covers vector search."
    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="lesson 2 content", course_name="Course A"
    )
    assert generator.client.messages.create.call_count == 2

    # Second call is still within MAX_TOOL_ROUNDS, so tools are offered again
    # (Claude just chooses not to use them here) and it must include the
    # tool result from round 1.
    final_call_kwargs = generator.client.messages.create.call_args_list[1].kwargs
    assert final_call_kwargs["tools"] == [{"name": "search_course_content"}]
    assert final_call_kwargs["tool_choice"] == {"type": "auto"}
    tool_result_message = final_call_kwargs["messages"][-1]
    assert tool_result_message["role"] == "user"
    assert tool_result_message["content"][0]["tool_use_id"] == "toolu_1"
    assert tool_result_message["content"][0]["content"] == (
        "[Course A - Lesson 2]\nVector search content..."
    )


def test_two_round_tool_use_outline_then_search(generator):
    """Claude looks up a course outline, then uses that info to search for content."""
    outline_response = FakeResponse(
        content=[
            FakeToolUseBlock(
                "get_course_outline", {"course_name": "Course X"}, block_id="toolu_1"
            )
        ],
        stop_reason="tool_use",
    )
    search_response = FakeResponse(
        content=[
            FakeToolUseBlock(
                "search_course_content",
                {"query": "vector search", "course_name": "Course Y"},
                block_id="toolu_2",
            )
        ],
        stop_reason="tool_use",
    )
    final_response = FakeResponse(
        content=[FakeTextBlock("Course Y covers the same topic as lesson 4.")],
        stop_reason="end_turn",
    )
    generator.client.messages.create.side_effect = [
        outline_response,
        search_response,
        final_response,
    ]

    tool_manager = Mock()
    tool_manager.execute_tool.side_effect = [
        "Course title: Course X\nLessons:\nLesson 4: Vector Search",
        "[Course Y - Lesson 1]\nVector search content...",
    ]

    result = generator.generate_response(
        "Find a course that covers the same topic as lesson 4 of Course X",
        tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert result == "Course Y covers the same topic as lesson 4."
    assert generator.client.messages.create.call_count == 3
    assert tool_manager.execute_tool.call_count == 2

    first_call, second_call = tool_manager.execute_tool.call_args_list
    assert first_call.args == ("get_course_outline",)
    assert first_call.kwargs == {"course_name": "Course X"}
    assert second_call.args == ("search_course_content",)
    assert second_call.kwargs == {"query": "vector search", "course_name": "Course Y"}

    # Final (3rd) call must not offer tools, and must carry both rounds'
    # tool_use/tool_result exchanges plus the original user message.
    final_call_kwargs = generator.client.messages.create.call_args_list[2].kwargs
    assert "tools" not in final_call_kwargs
    assert len(final_call_kwargs["messages"]) == 5


def test_round_limit_forces_final_answer_without_tools(generator):
    """If Claude keeps requesting tools past MAX_TOOL_ROUNDS, force a final
    tools-off call rather than looping indefinitely."""
    tool_use_response_1 = FakeResponse(
        content=[FakeToolUseBlock("search_course_content", {"query": "a"}, block_id="toolu_1")],
        stop_reason="tool_use",
    )
    tool_use_response_2 = FakeResponse(
        content=[FakeToolUseBlock("search_course_content", {"query": "b"}, block_id="toolu_2")],
        stop_reason="tool_use",
    )
    # Defensive: even if round 3's response somehow still looks like tool_use,
    # it must not be executed since tools weren't offered that round.
    round_3_response = FakeResponse(
        content=[FakeToolUseBlock("search_course_content", {"query": "c"}, block_id="toolu_3")],
        stop_reason="tool_use",
    )
    generator.client.messages.create.side_effect = [
        tool_use_response_1,
        tool_use_response_2,
        round_3_response,
    ]

    tool_manager = Mock()
    tool_manager.execute_tool.return_value = "some result"

    result = generator.generate_response(
        "multi-step question",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert generator.client.messages.create.call_count == 3
    assert tool_manager.execute_tool.call_count == 2

    third_call_kwargs = generator.client.messages.create.call_args_list[2].kwargs
    assert "tools" not in third_call_kwargs
    assert "tool_choice" not in third_call_kwargs

    # No text block on round 3's response, so the loop falls back to "".
    assert result == ""


def test_tool_execution_exception_is_handled_gracefully(generator):
    tool_use_response = FakeResponse(
        content=[FakeToolUseBlock("search_course_content", {"query": "a"}, block_id="toolu_1")],
        stop_reason="tool_use",
    )
    final_response = FakeResponse(
        content=[FakeTextBlock("Sorry, I couldn't complete that search.")],
        stop_reason="end_turn",
    )
    generator.client.messages.create.side_effect = [tool_use_response, final_response]

    tool_manager = Mock()
    tool_manager.execute_tool.side_effect = RuntimeError("vector store unavailable")

    result = generator.generate_response(
        "some question",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert result == "Sorry, I couldn't complete that search."
    assert generator.client.messages.create.call_count == 2
    tool_manager.execute_tool.assert_called_once()

    second_call_kwargs = generator.client.messages.create.call_args_list[1].kwargs
    assert "tools" not in second_call_kwargs
    tool_result_message = second_call_kwargs["messages"][-1]
    assert "vector store unavailable" in tool_result_message["content"][0]["content"]


def test_multiple_tool_calls_in_single_round(generator):
    response = FakeResponse(
        content=[
            FakeToolUseBlock("search_course_content", {"query": "a"}, block_id="toolu_1"),
            FakeToolUseBlock("get_course_outline", {"course_name": "b"}, block_id="toolu_2"),
        ],
        stop_reason="tool_use",
    )
    final_response = FakeResponse(
        content=[FakeTextBlock("combined answer")], stop_reason="end_turn"
    )
    generator.client.messages.create.side_effect = [response, final_response]

    tool_manager = Mock()
    tool_manager.execute_tool.side_effect = ["result A", "result B"]

    result = generator.generate_response(
        "question needing two tools",
        tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
        tool_manager=tool_manager,
    )

    assert result == "combined answer"
    assert tool_manager.execute_tool.call_count == 2

    second_call_kwargs = generator.client.messages.create.call_args_list[1].kwargs
    tool_result_entries = second_call_kwargs["messages"][-1]["content"]
    assert [entry["tool_use_id"] for entry in tool_result_entries] == ["toolu_1", "toolu_2"]
    assert [entry["content"] for entry in tool_result_entries] == ["result A", "result B"]


def test_mixed_text_and_tool_use_preserves_full_content(generator):
    response = FakeResponse(
        content=[
            FakeTextBlock("Let me check that..."),
            FakeToolUseBlock("search_course_content", {"query": "a"}, block_id="toolu_1"),
        ],
        stop_reason="tool_use",
    )
    final_response = FakeResponse(
        content=[FakeTextBlock("final answer")], stop_reason="end_turn"
    )
    generator.client.messages.create.side_effect = [response, final_response]

    tool_manager = Mock()
    tool_manager.execute_tool.return_value = "result"

    generator.generate_response(
        "question",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    second_call_kwargs = generator.client.messages.create.call_args_list[1].kwargs
    assistant_message = second_call_kwargs["messages"][1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == response.content


def test_conversation_history_is_added_to_system_prompt(generator):
    generator.client.messages.create.return_value = FakeResponse(
        content=[FakeTextBlock("answer")], stop_reason="end_turn"
    )

    generator.generate_response(
        "next question", conversation_history="User: hi\nAssistant: hello"
    )

    call_kwargs = generator.client.messages.create.call_args.kwargs
    assert "Previous conversation:" in call_kwargs["system"]
    assert "User: hi" in call_kwargs["system"]
