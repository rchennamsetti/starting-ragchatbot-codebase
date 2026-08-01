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

    # Second call must not offer tools again, and must include the tool result
    final_call_kwargs = generator.client.messages.create.call_args_list[1].kwargs
    assert "tools" not in final_call_kwargs
    tool_result_message = final_call_kwargs["messages"][-1]
    assert tool_result_message["role"] == "user"
    assert tool_result_message["content"][0]["tool_use_id"] == "toolu_1"
    assert tool_result_message["content"][0]["content"] == (
        "[Course A - Lesson 2]\nVector search content..."
    )


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
