"""
backend/vectordb/chroma_manager.py - ChromaDB Vector Store Operations
"""
from typing import List, Dict, Optional
from loguru import logger

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import settings
from backend.vectordb.embedding_client import EmbeddingClient


class ChromaManager:
    """Manages ChromaDB collections for research paper embeddings."""

    COLLECTION_NAME = "research_papers"
    _client = None
    _collection = None

    def __init__(self):
        self._init_client()

    def _init_client(self):
        """Initialize ChromaDB persistent client."""
        try:
            ChromaManager._client = chromadb.PersistentClient(
                path=settings.chroma_persist_path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            ChromaManager._collection = ChromaManager._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            count = ChromaManager._collection.count()
            logger.info(f"✅ ChromaDB ready. Collection '{self.COLLECTION_NAME}': {count} embeddings")
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            raise

    @property
    def collection(self):
        return ChromaManager._collection

    def add_paper(self, paper_doc: dict, chunks: List[str]) -> None:
        """Generate embeddings for paper chunks and store in ChromaDB."""
        if not chunks:
            logger.warning(f"No chunks to embed for paper {paper_doc.get('paper_id', '?')}")
            return

        embed_client = EmbeddingClient()
        paper_id = paper_doc["paper_id"]

        doc_texts = []
        doc_ids = []
        doc_metadatas = []

        # Summary chunk: title + abstract + keywords (highest retrieval priority)
        summary_text = (
            f"Title: {paper_doc.get('title', '')}\n"
            f"Abstract: {paper_doc.get('abstract', '')}\n"
            f"Keywords: {', '.join(paper_doc.get('keywords', []))}"
        )
        doc_texts.append(summary_text)
        doc_ids.append(f"{paper_id}_summary")
        doc_metadatas.append(self._build_metadata(paper_doc, "summary"))

        # Body chunks (up to 20)
        for i, chunk in enumerate(chunks[:20]):
            if not chunk.strip():
                continue
            doc_texts.append(chunk)
            doc_ids.append(f"{paper_id}_chunk_{i}")
            doc_metadatas.append(self._build_metadata(paper_doc, "body", chunk_index=i))

        try:
            embeddings = embed_client.embed_batch(doc_texts)
            self.collection.upsert(
                ids=doc_ids,
                embeddings=embeddings,
                documents=doc_texts,
                metadatas=doc_metadatas,
            )
            logger.info(f"✅ Stored {len(doc_ids)} embeddings for paper {paper_id[:8]}")
        except Exception as e:
            logger.error(f"Failed to store embeddings for {paper_id}: {e}")
            raise

    def _build_metadata(self, paper_doc: dict, chunk_type: str, chunk_index: int = 0) -> dict:
        """Build ChromaDB metadata dict (all values must be str/int/float/bool)."""
        return {
            "paper_id": str(paper_doc.get("paper_id", "")),
            "chunk_type": chunk_type,
            "chunk_index": str(chunk_index),
            "title": str(paper_doc.get("title", ""))[:500],
            "department": str(paper_doc.get("department", "")),
            "authors": ", ".join(paper_doc.get("authors", []))[:500],
            "publication_year": str(paper_doc.get("publication_year", "")),
            "venue": str(paper_doc.get("venue", ""))[:200],
            "keywords": ", ".join(paper_doc.get("keywords", []))[:500],
        }

    def search(self, query: str, n_results: int = 5, department_filter: Optional[str] = None) -> List[Dict]:
        """Semantic search: embed query and retrieve nearest neighbors."""
        total = self.collection.count()
        if total == 0:
            logger.warning("ChromaDB collection is empty — no papers embedded yet")
            return []

        embed_client = EmbeddingClient()
        query_embedding = embed_client.embed_single(query)

        # Clamp n_results to what's available
        effective_n = min(n_results, total)

        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": effective_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if department_filter:
            query_kwargs["where"] = {"department": {"$eq": department_filter}}

        try:
            results = self.collection.query(**query_kwargs)
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        formatted = []
        if results.get("documents") and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                formatted.append({
                    "text": doc,
                    "metadata": meta,
                    # cosine distance → similarity (ChromaDB returns 0=identical, 2=opposite)
                    "similarity_score": round(1 - dist, 4),
                    "paper_id": meta.get("paper_id"),
                    "title": meta.get("title"),
                })
        return formatted

    def delete_paper(self, paper_id: str) -> None:
        """Remove all embeddings for a paper."""
        try:
            results = self.collection.get(where={"paper_id": {"$eq": paper_id}})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} embeddings for paper {paper_id[:8]}")
        except Exception as e:
            logger.error(f"Failed to delete embeddings for {paper_id}: {e}")

    def get_collection_stats(self) -> dict:
        """Return collection statistics."""
        return {
            "total_embeddings": self.collection.count(),
            "collection_name": self.COLLECTION_NAME,
        }
