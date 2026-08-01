def test_query_creates_session_when_missing(client, fake_rag_system):
    response = client.post("/api/query", json={"query": "What is lesson 1 about?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "This is a test answer."
    assert body["sources"] == ["Course A - Lesson 1"]
    assert body["session_id"] == "session_1"
    assert fake_rag_system.last_query_call == ("What is lesson 1 about?", "session_1")


def test_query_reuses_provided_session_id(client, fake_rag_system):
    response = client.post(
        "/api/query",
        json={"query": "Follow-up question", "session_id": "session_42"},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "session_42"
    assert fake_rag_system.last_query_call == ("Follow-up question", "session_42")


def test_query_missing_field_is_rejected(client):
    response = client.post("/api/query", json={})
    assert response.status_code == 422


def test_query_error_returns_500(client, fake_rag_system):
    def raise_error(query, session_id=None):
        raise RuntimeError("boom")

    fake_rag_system.query = raise_error

    response = client.post("/api/query", json={"query": "anything"})

    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_get_course_stats(client):
    response = client.get("/api/courses")

    assert response.status_code == 200
    assert response.json() == {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
