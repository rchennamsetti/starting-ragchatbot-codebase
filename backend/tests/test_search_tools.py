from unittest.mock import Mock

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


def make_store(results: SearchResults) -> Mock:
    store = Mock()
    store.search.return_value = results
    return store


def test_execute_formats_results_with_lesson_headers():
    results = SearchResults(
        documents=["Vector search content...", "More content..."],
        metadata=[
            {"course_title": "Course A", "lesson_number": 2},
            {"course_title": "Course A", "lesson_number": 3},
        ],
        distances=[0.1, 0.2],
    )
    tool = CourseSearchTool(make_store(results))

    output = tool.execute(query="vector search")

    assert "[Course A - Lesson 2]\nVector search content..." in output
    assert "[Course A - Lesson 3]\nMore content..." in output
    assert tool.last_sources == ["Course A - Lesson 2", "Course A - Lesson 3"]


def test_execute_passes_filters_through_to_store():
    store = make_store(SearchResults(documents=[], metadata=[], distances=[]))
    tool = CourseSearchTool(store)

    tool.execute(query="topic", course_name="MCP", lesson_number=1)

    store.search.assert_called_once_with(
        query="topic", course_name="MCP", lesson_number=1
    )


def test_execute_reports_store_error():
    store = make_store(SearchResults.empty("No course found matching 'XYZ'"))
    tool = CourseSearchTool(store)

    assert tool.execute(query="anything", course_name="XYZ") == (
        "No course found matching 'XYZ'"
    )


def test_execute_empty_results_mentions_filters():
    store = make_store(SearchResults(documents=[], metadata=[], distances=[]))
    tool = CourseSearchTool(store)

    output = tool.execute(query="topic", course_name="Course A", lesson_number=5)

    assert "in course 'Course A'" in output
    assert "in lesson 5" in output


def test_tool_manager_execute_unknown_tool():
    manager = ToolManager()
    assert manager.execute_tool("does_not_exist") == "Tool 'does_not_exist' not found"


def test_tool_manager_sources_lifecycle():
    store = make_store(
        SearchResults(
            documents=["content"],
            metadata=[{"course_title": "Course A", "lesson_number": 1}],
            distances=[0.1],
        )
    )
    tool = CourseSearchTool(store)
    manager = ToolManager()
    manager.register_tool(tool)

    manager.execute_tool("search_course_content", query="anything")
    assert manager.get_last_sources() == ["Course A - Lesson 1"]

    manager.reset_sources()
    assert manager.get_last_sources() == []
