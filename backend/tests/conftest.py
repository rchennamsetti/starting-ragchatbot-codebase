import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# app.py mounts StaticFiles(directory="../frontend"), which is resolved
# relative to the process cwd — match how the app is normally run.
os.chdir(BACKEND_DIR)


@pytest.fixture
def fake_rag_system():
    """Stand-in for RAGSystem: real SessionManager, canned query/analytics."""
    from session_manager import SessionManager

    class FakeRAGSystem:
        def __init__(self):
            self.session_manager = SessionManager(max_history=2)
            self.last_query_call = None
            self.query_response = (
                "This is a test answer.",
                [{"course_title": "Course A", "lesson_number": 1, "link": None}],
            )
            self.analytics = {
                "total_courses": 2,
                "course_titles": ["Course A", "Course B"],
            }

        def query(self, query, session_id=None):
            self.last_query_call = (query, session_id)
            return self.query_response

        def get_course_analytics(self):
            return self.analytics

    return FakeRAGSystem()


@pytest.fixture
def client(fake_rag_system, monkeypatch):
    """TestClient for the FastAPI app with RAGSystem replaced before import,
    so no real Anthropic client or ChromaDB is ever constructed."""
    import rag_system as rag_system_module

    monkeypatch.setattr(rag_system_module, "RAGSystem", lambda config: fake_rag_system)

    sys.modules.pop("app", None)
    import app as app_module

    from fastapi.testclient import TestClient

    return TestClient(app_module.app)
