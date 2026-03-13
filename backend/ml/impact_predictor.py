"""
backend/ml/impact_predictor.py - Citation Impact Prediction using RandomForestRegressor
"""
from typing import Dict, List
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from loguru import logger

from backend.database import get_papers_collection


# Venue prestige lookup (heuristic)
HIGH_IMPACT_VENUES = {
    "neurips": 10, "nips": 10, "icml": 10, "iclr": 10,
    "cvpr": 9, "iccv": 9, "eccv": 8,
    "acl": 9, "emnlp": 8, "naacl": 8,
    "sigkdd": 9, "kdd": 9, "www": 8, "sigir": 8,
    "aaai": 8, "ijcai": 8,
    "nature": 10, "science": 10, "cell": 10, "lancet": 9,
    "ieee transactions": 8, "acm transactions": 8,
    "journal of machine learning": 9,
    "miccai": 8, "icra": 8, "iros": 7,
}

HIGH_IMPACT_KEYWORDS = {
    "deep learning", "neural network", "transformer", "large language model",
    "reinforcement learning", "attention mechanism", "bert", "gpt",
    "federated learning", "quantum computing", "drug discovery",
    "climate change", "cancer", "diffusion model", "generative ai",
}


class ImpactPredictor:
    """
    Predicts citation impact score (0–100) using RandomForestRegressor.
    Features are derived from paper metadata (no citation data needed).
    """

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        )
        self.is_fitted = False
        self._author_history: Dict[str, int] = {}
        self._try_fit()

    def _build_author_history(self) -> Dict[str, int]:
        """Count papers per author across entire collection (cached)."""
        col = get_papers_collection()
        counter: Counter = Counter()
        for p in col.find({}, {"authors": 1}):
            for author in p.get("authors", []):
                counter[author] += 1
        return dict(counter)

    def _extract_features(self, paper: dict) -> List[float]:
        """Extract 8 numerical features from paper metadata."""
        # 1. Venue prestige (0–10)
        venue = paper.get("venue", "").lower()
        venue_score = max(
            (score for key, score in HIGH_IMPACT_VENUES.items() if key in venue),
            default=0
        )

        # 2. Recency (0–1, normalized from year 2000)
        year = paper.get("publication_year") or 2020
        recency = max(0.0, (int(year) - 2000) / 25.0)

        # 3. Author count (normalized to 0–1)
        n_authors = min(len(paper.get("authors", [])) / 10.0, 1.0)

        # 4. High-impact keyword match (0–1)
        keywords = {k.lower() for k in paper.get("keywords", [])}
        abstract_words = set(paper.get("abstract", "").lower().split())
        all_terms = keywords | abstract_words
        kw_score = sum(1 for kw in HIGH_IMPACT_KEYWORDS if any(kw in t for t in all_terms))
        kw_feature = min(kw_score / 5.0, 1.0)

        # 5. Max author productivity (normalized)
        authors = paper.get("authors", [])
        max_hist = max((self._author_history.get(a, 0) for a in authors), default=0)
        author_feat = min(max_hist / 20.0, 1.0)

        # 6. Abstract length (normalized)
        abstract_len = len(paper.get("abstract", "").split())
        abstract_feat = min(abstract_len / 300.0, 1.0)

        # 7. Keyword count (normalized)
        kw_count = min(len(paper.get("keywords", [])) / 10.0, 1.0)

        # 8. Is conference? (1.0 = yes, 0.5 = journal/unknown)
        is_conf = 1.0 if any(
            w in venue for w in ["conference", "symposium", "workshop", "proceedings"]
        ) else 0.5

        return [venue_score, recency, n_authors, kw_feature,
                author_feat, abstract_feat, kw_count, is_conf]

    def _generate_synthetic_labels(self, papers: List[dict]) -> List[float]:
        """Generate heuristic impact labels for training (no real citation data)."""
        weights = [3.0, 1.5, 0.8, 2.5, 1.5, 0.5, 0.5, 1.0]
        scores = []
        for paper in papers:
            feats = self._extract_features(paper)
            raw = sum(f * w for f, w in zip(feats, weights)) * 10
            score = max(0.0, min(100.0, raw + float(np.random.normal(0, 3))))
            scores.append(score)
        return scores

    def _try_fit(self):
        """Train on existing papers if enough are available."""
        try:
            col = get_papers_collection()
            papers = list(col.find(
                {"processing_status": "completed"},
                {"title": 1, "abstract": 1, "keywords": 1, "venue": 1,
                 "authors": 1, "publication_year": 1, "department": 1}
            ))
            if len(papers) < 5:
                logger.info("Impact predictor: fewer than 5 papers — using heuristic fallback")
                return

            self._author_history = self._build_author_history()
            X = [self._extract_features(p) for p in papers]
            y = self._generate_synthetic_labels(papers)
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info(f"✅ Impact predictor trained on {len(papers)} papers")
        except Exception as e:
            logger.warning(f"Impact predictor training skipped: {e}")

    def predict_single(self, paper: dict) -> float:
        """Predict impact score (0–100) for a single paper."""
        if not self._author_history:
            self._author_history = self._build_author_history()

        feats = [self._extract_features(paper)]
        if self.is_fitted:
            score = float(self.model.predict(feats)[0])
        else:
            weights = [3.0, 1.5, 0.8, 2.5, 1.5, 0.5, 0.5, 1.0]
            score = sum(f * w for f, w in zip(feats[0], weights)) * 10

        return round(max(0.0, min(100.0, score)), 2)

    def get_top_impact_papers(self, top_n: int = 10) -> List[Dict]:
        """Return papers sorted by predicted impact score."""
        col = get_papers_collection()
        raw = list(col.find(
            {},
            {"paper_id": 1, "title": 1, "authors": 1, "venue": 1,
             "publication_year": 1, "department": 1, "predicted_impact_score": 1}
        ).sort("predicted_impact_score", -1).limit(top_n))

        results = []
        for p in raw:
            score = p.get("predicted_impact_score")
            if score is None:
                score = self.predict_single(p)
            results.append({
                "paper_id": str(p.get("paper_id", "")),
                "title": p.get("title", ""),
                "authors": p.get("authors", []),
                "venue": p.get("venue", ""),
                "publication_year": p.get("publication_year"),
                "department": p.get("department"),
                "predicted_impact_score": score,
            })

        return sorted(results, key=lambda x: x["predicted_impact_score"], reverse=True)
