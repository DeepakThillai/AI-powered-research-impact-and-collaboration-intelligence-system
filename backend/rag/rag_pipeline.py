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
            logger.info(f"✅ Groq LLM initialized: {settings.llm_model}")
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")
        except Exception as e:
            logger.error(f"Groq init failed: {e}")
            raise

    def retrieve(self, query: str, n_results: int = 8, department: Optional[str] = None) -> List[Dict]:
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

        parts = [f"USER QUESTION: {query}\n\nRELEVANT RESEARCH PAPERS FROM KNOWLEDGE BASE:\n"]
        for i, chunk in enumerate(seen_papers.values(), 1):
            meta = chunk.get("metadata", {})
            parts.append(
                f"[Paper {i}]\n"
                f"Title: {meta.get('title', 'Unknown')}\n"
                f"Authors: {meta.get('authors', 'Unknown')}\n"
                f"Department: {meta.get('department', 'Unknown')}\n"
                f"Year: {meta.get('publication_year', 'Unknown')}\n"
                f"Venue: {meta.get('venue', 'Unknown')}\n"
                f"Keywords: {meta.get('keywords', '')}\n"
                f"Excerpt: {chunk.get('text', '')[:600]}\n"
                f"Relevance: {chunk.get('similarity_score', 0):.2%}\n"
            )
        return "\n".join(parts)

    def _get_stats_context(self) -> str:
        """Fetch summary stats from MongoDB to enrich LLM context."""
        try:
            col = get_papers_collection()
            total = col.count_documents({})
            pipeline = [{"$group": {"_id": "$department", "count": {"$sum": 1}}}]
            dept_counts = {
                r["_id"]: r["count"]
                for r in col.aggregate(pipeline)
                if r["_id"]
            }
            years = [y for y in col.distinct("publication_year") if y]
            year_range = f"{min(years)}–{max(years)}" if years else "N/A"
            return (
                f"Database summary: {total} papers | "
                f"Years: {year_range} | "
                f"Departments: {dept_counts}"
            )
        except Exception:
            return "Database statistics unavailable."

    def generate_answer(self, query: str, department_filter: Optional[str] = None) -> Dict:
        """
        Full RAG pipeline. Returns:
        { answer: str, sources: list, retrieved_count: int, timestamp: str }
        """
        retrieved = self.retrieve(query, n_results=8, department=department_filter)
        context = self._build_context(retrieved, query)
        stats_context = self._get_stats_context()

        full_prompt = (
            f"{context}\n\n"
            f"SYSTEM STATISTICS: {stats_context}\n\n"
            f"Please answer the question: {query}\n\n"
            "Provide a detailed, insightful answer citing specific papers, "
            "authors, and departments where relevant."
        )

        try:
            completion = self._client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.7,
                max_tokens=1500,   # NOTE: Groq uses max_tokens, not max_completion_tokens
                top_p=1,
                stream=False,
                stop=None,
            )
            answer = completion.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Groq LLM call failed: {e}")
            answer = (
                f"⚠️ LLM error: {str(e)}\n\n"
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
