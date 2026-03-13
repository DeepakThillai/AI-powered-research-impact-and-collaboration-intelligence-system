"""
backend/vectordb/embedding_client.py
Local embeddings using sentence-transformers (all-MiniLM-L6-v2)
No API key required — runs entirely offline after first model download.
"""
import time
from typing import List
from loguru import logger

from backend.config import settings

# Module-level singleton so the model is loaded only once
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {settings.embedding_model} (first run downloads ~90 MB)")
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("✅ Embedding model loaded")
    return _model


class EmbeddingClient:
    """
    Local text embeddings using sentence-transformers.
    Model: all-MiniLM-L6-v2 → 384-dimensional vectors, very fast on CPU.
    Downloaded once (~90 MB) and cached locally by HuggingFace.
    """

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        text = str(text).strip()[:8000]
        model = _get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for a list of texts (batched for speed)."""
        model = _get_model()
        clean_texts = [str(t).strip()[:8000] for t in texts]
        embeddings = model.encode(clean_texts, batch_size=batch_size,
                                  convert_to_numpy=True, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Embed a search query (same model, same space)."""
        return self.embed_single(query)
