# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency management (Python 3.13+). There is no configured lint or build tooling — don't invent commands for these.

```bash
# Install dependencies (including dev/test deps)
uv sync

# Run the app (from repo root)
chmod +x run.sh && ./run.sh

# Equivalent manual start
cd backend && uv run uvicorn app:app --reload --port 8000

# Run tests (from repo root or backend/, both work)
uv run pytest

# Run a single test file / single test
uv run pytest backend/tests/test_api.py
uv run pytest backend/tests/test_api.py::test_query_creates_session_when_missing
```

App runs at `http://localhost:8000`, API docs at `http://localhost:8000/docs`. Requires `ANTHROPIC_API_KEY` in a `.env` file at the repo root (see `.env.example`).

### Tests (`backend/tests/`)

`conftest.py` replaces `rag_system.RAGSystem` with a lightweight fake **before** `app` is imported (`monkeypatch.setattr` on the `rag_system` module, then a fresh `import app`), so API tests never construct a real Anthropic client or ChromaDB instance — no API key or network access required. If you change how `app.py` wires up `rag_system`, check this fixture still applies. `test_vector_store.py` deliberately avoids constructing a real `VectorStore` (which would load the sentence-transformers embedding model) and instead calls the pure-logic methods (`_build_filter`, `SearchResults.*`) directly off the class.

## Architecture

This is a RAG chatbot that answers questions about course materials. It has two independent pipelines: document ingestion (runs once at startup) and query handling (runs per request), both orchestrated through `backend/rag_system.py`.

### Ingestion (startup, `app.py` → `RAGSystem.add_course_folder`)

On FastAPI startup, `docs/` is scanned for `.txt`/`.pdf`/`.docx` files. `document_processor.py` parses each file's expected format (`Course Title:` / `Course Link:` / `Course Instructor:` header, then `Lesson N: <title>` markers) and sentence-chunks each lesson's body (`CHUNK_SIZE`/`CHUNK_OVERLAP` in `config.py`). Files whose course title is already present in ChromaDB are skipped, so restarts don't re-embed — **course title is the unique identifier** used as the ChromaDB document ID.

Two ChromaDB collections are populated (`vector_store.py`):
- `course_catalog` — one row per course (title, instructor, link, lessons as JSON), used only to fuzzy-resolve a course name mentioned in a query.
- `course_content` — one row per chunk, embedded with `all-MiniLM-L6-v2`, this is what search actually queries.

### Query handling (per request)

`POST /api/query` → `RAGSystem.query()` → `AIGenerator.generate_response()`. **Claude itself decides whether to search** — the backend doesn't branch on query type. Claude is given the `search_course_content` tool (defined in `search_tools.py`) and a system prompt (`ai_generator.py`) instructing it to search only for course-specific questions, one search maximum per query, and to avoid meta-commentary like "based on the search results."

If Claude calls the tool, `ToolManager.execute_tool` runs `CourseSearchTool.execute`, which calls `VectorStore.search()` (optionally filtered by course name — resolved via `course_catalog` — and/or lesson number), formats matched chunks, and records them as "sources" for the UI. Results are fed back to Claude in a second API call (no tools this time) to produce the final answer. Sources are read once by `RAGSystem.query()` via `ToolManager.get_last_sources()` and then reset — they don't persist across queries.

Conversation history is per-session, capped at the last `MAX_HISTORY` exchanges (`config.py`), held in memory (`session_manager.py`, not persisted), and injected as plain text into the system prompt on each call.

### Key config (`backend/config.py`)

Chunk size/overlap, max search results, history length, embedding model, Claude model, and ChromaDB path are all centralized here — check this file before assuming a value.

### Frontend

Static HTML/CSS/vanilla JS (`frontend/`), served directly by FastAPI (`app.mount("/", StaticFiles(...))`). No build step. Talks to the backend only via `POST /api/query` and `GET /api/courses`.
