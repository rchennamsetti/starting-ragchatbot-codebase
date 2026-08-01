from vector_store import SearchResults, VectorStore

# _build_filter never touches `self`, so it can be exercised directly on the
# class without constructing a real VectorStore (which would spin up
# ChromaDB + the sentence-transformers embedding model).
build_filter = VectorStore._build_filter


def test_build_filter_no_filters():
    assert build_filter(None, None, None) is None


def test_build_filter_course_only():
    assert build_filter(None, "Course A", None) == {"course_title": "Course A"}


def test_build_filter_lesson_only():
    assert build_filter(None, None, 2) == {"lesson_number": 2}


def test_build_filter_course_and_lesson():
    assert build_filter(None, "Course A", 2) == {
        "$and": [{"course_title": "Course A"}, {"lesson_number": 2}]
    }


def test_search_results_from_chroma():
    chroma_results = {
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"course_title": "A"}, {"course_title": "B"}]],
        "distances": [[0.1, 0.2]],
    }

    results = SearchResults.from_chroma(chroma_results)

    assert results.documents == ["doc1", "doc2"]
    assert results.metadata == [{"course_title": "A"}, {"course_title": "B"}]
    assert results.distances == [0.1, 0.2]
    assert results.error is None


def test_search_results_from_chroma_handles_empty():
    chroma_results = {"documents": [], "metadatas": [], "distances": []}
    results = SearchResults.from_chroma(chroma_results)
    assert results.is_empty()


def test_search_results_empty_carries_error():
    results = SearchResults.empty("No course found matching 'XYZ'")
    assert results.is_empty()
    assert results.error == "No course found matching 'XYZ'"
