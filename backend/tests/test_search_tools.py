from unittest.mock import Mock

from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


def make_store(results: SearchResults) -> Mock:
    store = Mock()
    store.search.return_value = results
    store.get_lesson_link.return_value = None
    store.get_lesson_title.return_value = None
    store.get_course_link.return_value = None
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
    store = make_store(results)
    store.get_lesson_link.return_value = "https://example.com/lesson"
    store.get_lesson_title.return_value = "Vectors 101"
    tool = CourseSearchTool(store)

    output = tool.execute(query="vector search")

    assert "[Course A - Lesson 2]\nVector search content..." in output
    assert "[Course A - Lesson 3]\nMore content..." in output
    assert tool.last_sources == [
        {
            "course_title": "Course A",
            "lesson_number": 2,
            "lesson_title": "Vectors 101",
            "link": "https://example.com/lesson",
        },
        {
            "course_title": "Course A",
            "lesson_number": 3,
            "lesson_title": "Vectors 101",
            "link": "https://example.com/lesson",
        },
    ]


def test_execute_dedupes_repeated_course_lesson_chunks():
    results = SearchResults(
        documents=["First chunk...", "Second chunk..."],
        metadata=[
            {"course_title": "Course A", "lesson_number": 8},
            {"course_title": "Course A", "lesson_number": 8},
        ],
        distances=[0.1, 0.2],
    )
    store = make_store(results)
    store.get_lesson_link.return_value = "https://example.com/lesson-8"
    tool = CourseSearchTool(store)

    tool.execute(query="anything")

    assert tool.last_sources == [
        {
            "course_title": "Course A",
            "lesson_number": 8,
            "lesson_title": None,
            "link": "https://example.com/lesson-8",
        },
    ]


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


def test_course_outline_tool_returns_course_title_link_and_lessons():
    store = Mock()
    store.get_course_outline.return_value = {
        "title": "Course A",
        "course_link": "https://example.com/course-a",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "Intro"},
            {"lesson_number": 2, "lesson_title": "Vectors"},
        ],
    }
    tool = CourseOutlineTool(store)

    output = tool.execute(course_name="Course A")

    assert "Course title: Course A" in output
    assert "Course link: https://example.com/course-a" in output
    assert "Lesson 1: Intro" in output
    assert "Lesson 2: Vectors" in output


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
    assert manager.get_last_sources() == [
        {"course_title": "Course A", "lesson_number": 1, "lesson_title": None, "link": None}
    ]

    manager.reset_sources()
    assert manager.get_last_sources() == []


def test_tool_manager_accumulates_sources_across_multiple_tool_calls():
    """Sequential tool calling can invoke search_course_content more than
    once per query; every call's sources must be retained, not just the
    most recent one."""
    store = Mock()
    store.get_lesson_link.return_value = None
    store.get_lesson_title.return_value = None
    store.get_course_link.return_value = None
    store.search.side_effect = [
        SearchResults(
            documents=["content A"],
            metadata=[{"course_title": "Course A", "lesson_number": 1}],
            distances=[0.1],
        ),
        SearchResults(
            documents=["content B"],
            metadata=[{"course_title": "Course B", "lesson_number": 2}],
            distances=[0.1],
        ),
    ]
    tool = CourseSearchTool(store)
    manager = ToolManager()
    manager.register_tool(tool)

    manager.execute_tool("search_course_content", query="first round")
    manager.execute_tool("search_course_content", query="second round")

    assert manager.get_last_sources() == [
        {"course_title": "Course A", "lesson_number": 1, "lesson_title": None, "link": None},
        {"course_title": "Course B", "lesson_number": 2, "lesson_title": None, "link": None},
    ]


def test_tool_manager_dedupes_sources_across_repeated_calls():
    store = Mock()
    store.get_lesson_link.return_value = None
    store.get_lesson_title.return_value = None
    store.get_course_link.return_value = None
    same_result = SearchResults(
        documents=["content A"],
        metadata=[{"course_title": "Course A", "lesson_number": 1}],
        distances=[0.1],
    )
    store.search.side_effect = [same_result, same_result]
    tool = CourseSearchTool(store)
    manager = ToolManager()
    manager.register_tool(tool)

    manager.execute_tool("search_course_content", query="first round")
    manager.execute_tool("search_course_content", query="second round")

    assert manager.get_last_sources() == [
        {"course_title": "Course A", "lesson_number": 1, "lesson_title": None, "link": None},
    ]
