import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

from config import config
from rag_system import RAGSystem

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")

# Add trusted host middleware for proxy
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None

class Source(BaseModel):
    """A single source reference with optional clickable link"""
    course_title: str
    lesson_number: Optional[int] = None
    lesson_title: Optional[str] = None
    link: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[Source]
    session_id: str

class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]

class NewChatRequest(BaseModel):
    """Request model for starting a new chat session"""
    session_id: Optional[str] = None

class NewChatResponse(BaseModel):
    """Response model for starting a new chat session"""
    session_id: str

# API Endpoints

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()
            logger.debug("Created new session %s", session_id)

        logger.info("Handling query for session %s: %s", session_id, request.query)

        # Process query using RAG system
        answer, sources = rag_system.query(request.query, session_id)

        logger.debug("Query for session %s returned %d source(s)", session_id, len(sources))

        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
    except Exception as e:
        logger.error("Error handling query for session %s: %s", request.session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/new-chat", response_model=NewChatResponse)
async def new_chat(request: NewChatRequest):
    """Clear the previous session's history (if any) and start a fresh session"""
    try:
        if request.session_id:
            rag_system.session_manager.clear_session(request.session_id)
            logger.debug("Cleared session %s", request.session_id)
        session_id = rag_system.session_manager.create_session()
        logger.info("Started new chat session %s", session_id)
        return NewChatResponse(session_id=session_id)
    except Exception as e:
        logger.error("Error starting new chat: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    except Exception as e:
        logger.error("Error fetching course stats: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    docs_path = "../docs"
    if os.path.exists(docs_path):
        logger.info("Loading initial documents from %s...", docs_path)
        try:
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            logger.info("Loaded %d courses with %d chunks", courses, chunks)
        except Exception as e:
            logger.error("Error loading documents: %s", e, exc_info=True)
    else:
        logger.warning("Docs path %s does not exist; skipping initial document load", docs_path)

# Custom static file handler with no-cache headers for development
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import hashlib
import os
from pathlib import Path


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _asset_version(filename: str) -> str:
    """Short content hash used to cache-bust a frontend asset's URL.

    Computed per-request (not cached at startup) because `uvicorn --reload`
    only restarts on .py changes, not frontend/*.css or *.js edits.
    """
    return hashlib.md5((FRONTEND_DIR / filename).read_bytes()).hexdigest()[:8]


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve index.html with cache-busted asset URLs based on file content."""
    html = (FRONTEND_DIR / "index.html").read_text()
    html = html.replace("{{CSS_VERSION}}", _asset_version("style.css"))
    html = html.replace("{{JS_VERSION}}", _asset_version("script.js"))
    return HTMLResponse(html, headers=NO_CACHE_HEADERS)


# Serve static files for the frontend
app.mount("/", DevStaticFiles(directory="../frontend", html=True), name="static")