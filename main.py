"""
main.py - FastAPI Application Entry Point
AI-Powered Research Impact & Collaboration Intelligence System (ResearchIQ)
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from loguru import logger
import sys


# ── Logging Setup ─────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
    colorize=True,
)
# Ensure log directory exists before adding file sink
Path("./data/logs").mkdir(parents=True, exist_ok=True)
logger.add("./data/logs/app.log", rotation="10 MB", level="DEBUG", encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("=" * 55)
    logger.info("  🚀  ResearchIQ — Starting up...")
    logger.info("=" * 55)

    # ── Ensure data directories exist ────────────────────────
    from backend.config import settings
    settings.ensure_directories()
    if not settings.groq_key_configured():
        logger.warning(
            "Groq chat is not fully configured. Set a valid GROQ_API_KEY in .env "
            "to enable /api/chat/ask."
        )

    # ── Connect to MongoDB Atlas ──────────────────────────────
    try:
        from backend.database import DatabaseManager
        DatabaseManager.connect()
    except Exception as e:
        logger.error(f"❌ MongoDB startup failed: {e}")
        logger.warning("Some features will be unavailable until MongoDB connects.")

    # ── Initialize ChromaDB (vector store) ───────────────────
    try:
        from backend.vectordb.chroma_manager import ChromaManager
        chroma = ChromaManager()
        stats = chroma.get_collection_stats()
        logger.info(f"✅ ChromaDB ready — {stats['total_embeddings']} embeddings stored")
    except Exception as e:
        logger.warning(f"ChromaDB init warning (will retry on first use): {e}")

    logger.info("✅ ResearchIQ is ready!  →  http://localhost:8000")
    logger.info("=" * 55)

    yield  # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("Shutting down ResearchIQ...")
    try:
        from backend.database import DatabaseManager
        DatabaseManager.disconnect()
    except Exception:
        pass


# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="ResearchIQ — AI Research Intelligence System",
    description="RAG-powered research analytics platform for universities",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register API Routers ──────────────────────────────────────
from backend.auth.routes import router as auth_router
from backend.papers.routes import router as papers_router
from backend.rag.routes import router as rag_router
from backend.dashboard.routes import router as dashboard_router

app.include_router(auth_router)
app.include_router(papers_router)
app.include_router(rag_router)
app.include_router(dashboard_router)


# ── Health Check ──────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """System health check."""
    from backend.database import DatabaseManager
    db_status = "disconnected"
    try:
        DatabaseManager.get_db().command("ping")
        db_status = "connected"
    except Exception:
        pass

    return {
        "status": "healthy",
        "mongodb": db_status,
        "version": "1.0.0",
    }


# ── Serve Frontend SPA ────────────────────────────────────────
_frontend_dir = Path("./frontend")
_index_html = _frontend_dir / "index.html"

# Only mount static assets if the directory is non-empty
_static_dir = _frontend_dir / "static"
if _static_dir.exists() and any(_static_dir.iterdir()):
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", include_in_schema=False)
def serve_index():
    if _index_html.exists():
        return FileResponse(str(_index_html))
    return JSONResponse(
        {"message": "ResearchIQ API running. See /docs for API documentation."},
        status_code=200,
    )


@app.get("/{full_path:path}", include_in_schema=False)
def serve_spa(full_path: str):
    """
    Catch-all for SPA routing.
    API routes are matched first (registered before this handler).
    """
    # Never intercept API or system paths
    if full_path.startswith(("api/", "docs", "redoc", "health", "openapi")):
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    # Serve specific static file if it exists
    file_path = _frontend_dir / full_path
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))

    # Fall back to index.html for client-side routing
    if _index_html.exists():
        return FileResponse(str(_index_html))

    return JSONResponse({"detail": "Not Found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    from backend.config import settings
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level="info",
    )
