from document_processor import DocumentProcessor

SAMPLE_COURSE = """Course Title: Intro to Vector Search
Course Link: https://example.com/course
Course Instructor: Ada Lovelace

Lesson 0: Getting Started
Lesson Link: https://example.com/lesson0
Vector search finds similar items by comparing embeddings. Embeddings are dense numeric representations of meaning. This lesson introduces the core idea.

Lesson 1: Building an Index
This lesson covers how ChromaDB stores and indexes chunks. It also covers metadata filtering.
"""


def test_process_course_document_parses_metadata(tmp_path):
    file_path = tmp_path / "course.txt"
    file_path.write_text(SAMPLE_COURSE)
    processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

    course, chunks = processor.process_course_document(str(file_path))

    assert course.title == "Intro to Vector Search"
    assert course.course_link == "https://example.com/course"
    assert course.instructor == "Ada Lovelace"
    assert len(course.lessons) == 2
    assert course.lessons[0].lesson_number == 0
    assert course.lessons[0].title == "Getting Started"
    assert course.lessons[0].lesson_link == "https://example.com/lesson0"
    assert course.lessons[1].lesson_number == 1
    assert course.lessons[1].lesson_link is None


def test_process_course_document_chunks_belong_to_correct_lesson(tmp_path):
    file_path = tmp_path / "course.txt"
    file_path.write_text(SAMPLE_COURSE)
    processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

    _, chunks = processor.process_course_document(str(file_path))

    lesson0_chunks = [c for c in chunks if c.lesson_number == 0]
    lesson1_chunks = [c for c in chunks if c.lesson_number == 1]
    assert lesson0_chunks
    assert lesson1_chunks
    assert all(c.course_title == "Intro to Vector Search" for c in chunks)
    # chunk_index is a running counter across the whole document
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_first_chunk_of_a_lesson_is_prefixed_with_lesson_context(tmp_path):
    file_path = tmp_path / "course.txt"
    file_path.write_text(SAMPLE_COURSE)
    processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

    _, chunks = processor.process_course_document(str(file_path))

    lesson0_first = next(c for c in chunks if c.lesson_number == 0)
    assert lesson0_first.content.startswith("Lesson 0 content:")


def test_chunk_text_respects_chunk_size():
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=0)
    text = (
        "This is sentence one. This is sentence two. "
        "This is sentence three. This is sentence four."
    )

    chunks = processor.chunk_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 50 or chunk.count(" ") == 0 for chunk in chunks)


def test_chunk_text_creates_overlap_between_chunks():
    # Overlap works at sentence granularity: chunk_overlap must be large enough
    # to fit at least one whole trailing sentence for overlap to occur.
    processor = DocumentProcessor(chunk_size=30, chunk_overlap=15)
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."

    chunks = processor.chunk_text(text)

    assert len(chunks) >= 2
    # The tail sentence of one chunk should reappear at the start of the next.
    assert any(
        chunks[i].split(". ")[-1] in chunks[i + 1] for i in range(len(chunks) - 1)
    )


def test_no_lesson_markers_treats_whole_body_as_one_document(tmp_path):
    content = (
        "Course Title: No Lessons Course\n"
        "Course Link: https://example.com\n"
        "Course Instructor: Jane Doe\n\n"
        "Just a plain paragraph of content with no lesson markers at all."
    )
    file_path = tmp_path / "course.txt"
    file_path.write_text(content)
    processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

    course, chunks = processor.process_course_document(str(file_path))

    assert course.lessons == []
    assert len(chunks) == 1
    assert chunks[0].lesson_number is None
