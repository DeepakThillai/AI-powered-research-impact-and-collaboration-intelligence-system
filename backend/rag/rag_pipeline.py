"""
backend/rag/rag_pipeline.py - RAG Pipeline using Groq LLM + local sentence-transformer embeddings
"""
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger

from backend.config import settings
from backend.vectordb.chroma_manager import ChromaManager
from backend.database import get_papers_collection


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline:
    1. Embed user query locally (sentence-transformers, no API)
    2. Retrieve top-K relevant chunks from ChromaDB (cosine similarity)
    3. Build a context prompt with retrieved paper details
    4. Send context + question to Groq LLM (openai/gpt-oss-120b)
    5. Return structured answer with source citations
    """

    SYSTEM_PROMPT = (
        "You are an AI research assistant for a university research intelligence system. "
        "You help faculty, students, and research heads understand research trends, "
        "collaboration patterns, and insights from published papers. "
        "When answering: be specific, cite paper titles and authors when relevant, "
        "highlight research trends, and be concise but comprehensive. "
        "If the context is insufficient, clearly say so."
    )

    def __init__(self):
        self.chroma = ChromaManager()
        self._init_groq()

    def _init_groq(self):
        """Initialize Groq client."""
        try:
            from groq import Groq
            self._client = Groq(api_key=settings.groq_api_key)
            logger.info(f"Groq LLM initialized: {settings.llm_model}")
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")
        except Exception as e:
            logger.error(f"Groq init failed: {e}")
            raise

    # Free-tier TPM budget for openai/gpt-oss-120b is 8000 tokens (input + output).
    # Reserve ~1500 for the answer, leaving ~6500 for system prompt + context.
    # Rough estimate: 1 token ≈ 4 chars.
    _MAX_CONTEXT_CHARS = 6500 * 4  # ~26 000 chars hard cap on the context string

    def retrieve(self, query: str, n_results: int = 4, department: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant paper chunks from ChromaDB."""
        results = self.chroma.search(query, n_results=n_results, department_filter=department)
        logger.debug(f"Retrieved {len(results)} chunks for: '{query[:80]}'")
        return results

    def _build_context(self, retrieved_chunks: List[Dict], query: str) -> str:
        """Build LLM context string from retrieved chunks (deduplicated by paper)."""
        if not retrieved_chunks:
            return "No relevant research papers found in the knowledge base."

        seen_papers: Dict[str, Dict] = {}
        for chunk in retrieved_chunks:
            pid = chunk.get("paper_id")
            if pid and pid not in seen_papers:
                seen_papers[pid] = chunk

        parts = [f"USER QUESTION: {query}\n\nRELEVANT RESEARCH PAPERS:\n"]
        budget = self._MAX_CONTEXT_CHARS - len(parts[0])

        for i, chunk in enumerate(seen_papers.values(), 1):
            meta = chunk.get("metadata", {})
            # Truncate excerpt to stay within budget — 200 chars per paper
            excerpt = chunk.get("text", "")[:200]
            entry = (
                f"[{i}] {meta.get('title', 'Unknown')} "
                f"({meta.get('publication_year', '?')})\n"
                f"Authors: {meta.get('authors', 'Unknown')}\n"
                f"Dept: {meta.get('department', '?')} | "
                f"Keywords: {meta.get('keywords', '')}\n"
                f"Excerpt: {excerpt}\n"
            )
            if len(entry) > budget:
                break
            parts.append(entry)
            budget -= len(entry)

        return "\n".join(parts)

    def _get_stats_context(self) -> str:
        """Fetch summary stats from MongoDB to enrich LLM context."""
        try:
            col = get_papers_collection()
            total = col.count_documents({})
            years = [y for y in col.distinct("publication_year") if y]
            year_range = f"{min(years)}–{max(years)}" if years else "N/A"
            # Keep stats brief — just total and year range to save tokens
            return f"{total} papers in DB, years {year_range}."
        except Exception:
            return ""

    def generate_answer(self, query: str, department_filter: Optional[str] = None) -> Dict:
        """
        Full RAG pipeline. Returns:
        { answer: str, sources: list, retrieved_count: int, timestamp: str }
        """
        retrieved = self.retrieve(query, n_results=4, department=department_filter)
        context = self._build_context(retrieved, query)
        stats_context = self._get_stats_context()

        full_prompt = (
            f"{context}\n\n"
            f"Stats: {stats_context}\n\n"
            f"Answer concisely: {query}"
        )

        try:
            completion = self._client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=1,
                max_completion_tokens=1500,
                top_p=1,
                reasoning_effort="medium",
                stream=False,
                stop=None,
            )
            answer = completion.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Groq LLM call failed: {e}")
            answer = (
                f"LLM error: {str(e)}\n\n"
                "Please verify your GROQ_API_KEY in the .env file and "
                "ensure the model name is correct."
            )

        # Deduplicated source list
        sources = []
        seen_ids: set = set()
        for chunk in retrieved:
            pid = chunk.get("paper_id")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                sources.append({
                    "paper_id": pid,
                    "title": chunk.get("title", ""),
                    "similarity": chunk.get("similarity_score", 0),
                })

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_count": len(retrieved),
            "timestamp": datetime.utcnow().isoformat(),
        }
