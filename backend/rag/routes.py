"""
backend/rag/routes.py - Chatbot & RAG API endpoints
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from loguru import logger

from backend.auth.auth_handler import get_current_user
from backend.rag.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/api/chat", tags=["Chatbot / RAG"])

# Module-level singleton — initialized on first request
_rag_instance: Optional[RAGPipeline] = None


def get_rag() -> RAGPipeline:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGPipeline()
    return _rag_instance


class ChatRequest(BaseModel):
    message: str
    department_filter: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list
    retrieved_count: int
    timestamp: str


SUGGESTED_QUESTIONS = [
    "Which departments produce the most research papers?",
    "What are the trending research domains in the last 3 years?",
    "Which authors collaborate most frequently?",
    "What are the most cited research keywords?",
    "Which papers are predicted to have the highest impact?",
    "Summarize the research themes in Computer Science.",
    "Who are the top researchers in Machine Learning?",
    "What papers focus on deep learning or neural networks?",
]


@router.post("/ask", response_model=ChatResponse)
def ask_question(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Ask a research question. Uses RAG to retrieve relevant papers and
    generate an answer via Groq LLM (openai/gpt-oss-120b).
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(request.message) > 1000:
        raise HTTPException(status_code=400, detail="Message too long (max 1000 characters)")

    logger.info(f"Chat query from [{current_user['name']}]: '{request.message[:80]}'")

    rag = get_rag()
    result = rag.generate_answer(
        query=request.message,
        department_filter=request.department_filter or None,
    )
    return ChatResponse(**result)


@router.get("/suggestions")
def get_suggestions(current_user: dict = Depends(get_current_user)):
    """Return suggested questions for the chatbot UI."""
    return {"suggestions": SUGGESTED_QUESTIONS}


@router.get("/search")
def semantic_search(
    query: str,
    n_results: int = 5,
    department: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """Pure semantic search — returns relevant chunks without LLM generation."""
    from backend.vectordb.chroma_manager import ChromaManager
    chroma = ChromaManager()
    results = chroma.search(query, n_results=n_results, department_filter=department)
    return {"query": query, "results": results, "count": len(results)}
