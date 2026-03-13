"""
backend/ml/trend_detector.py - Research Trend Detection using TF-IDF + KMeans
"""
import re
from typing import List, Dict, Tuple
from collections import Counter
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from loguru import logger

from backend.database import get_papers_collection


class TrendDetector:
    """
    Identifies trending research topics using:
    - TF-IDF vectorization of paper abstracts + keywords
    - KMeans clustering to group similar papers
    - Frequency analysis of terms within clusters
    """

    def __init__(self, n_clusters: int = 8, max_features: int = 500):
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),  # unigrams + bigrams
            min_df=1,
            max_df=0.95,
        )
        self.kmeans = None
        self.feature_names = []
        self.cluster_labels = []
        self.cluster_topics = {}

    def _prepare_documents(self, papers: List[dict]) -> List[str]:
        """Combine title + abstract + keywords for each paper."""
        docs = []
        for paper in papers:
            parts = [
                paper.get("title", ""),
                paper.get("abstract", ""),
                " ".join(paper.get("keywords", [])),
            ]
            docs.append(" ".join(filter(None, parts)))
        return docs

    def fit(self, papers: List[dict]) -> Dict:
        """
        Fit TF-IDF + KMeans on the given papers.
        Returns cluster analysis results.
        """
        if len(papers) < 3:
            return {"error": "Need at least 3 papers for trend detection"}

        docs = self._prepare_documents(papers)

        # Adjust clusters if we have fewer papers than n_clusters
        actual_clusters = min(self.n_clusters, len(papers))

        try:
            # TF-IDF vectorization
            tfidf_matrix = self.vectorizer.fit_transform(docs)
            self.feature_names = self.vectorizer.get_feature_names_out()

            # Normalize for better clustering
            tfidf_normalized = normalize(tfidf_matrix)

            # KMeans clustering
            self.kmeans = KMeans(
                n_clusters=actual_clusters,
                random_state=42,
                n_init=10,
                max_iter=300,
            )
            self.cluster_labels = self.kmeans.fit_predict(tfidf_normalized)

            # Analyze clusters
            self.cluster_topics = self._extract_cluster_topics(
                papers, tfidf_matrix, actual_clusters
            )

            logger.info(f"Trend detection: {actual_clusters} clusters from {len(papers)} papers")
            return self.cluster_topics

        except Exception as e:
            logger.error(f"Trend detection failed: {e}")
            return {"error": str(e)}

    def _extract_cluster_topics(self, papers: List[dict], tfidf_matrix, n_clusters: int) -> Dict:
        """Extract top terms and paper counts for each cluster."""
        cluster_topics = {}
        cluster_centers = self.kmeans.cluster_centers_

        for cluster_id in range(n_clusters):
            # Papers in this cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_papers = [papers[i] for i, m in enumerate(cluster_mask) if m]

            # Top terms by centroid weight
            centroid = cluster_centers[cluster_id]
            top_indices = centroid.argsort()[-10:][::-1]
            top_terms = [self.feature_names[i] for i in top_indices if centroid[i] > 0]

            # Most common keywords in this cluster
            all_keywords = []
            for p in cluster_papers:
                all_keywords.extend(p.get("keywords", []))
            keyword_freq = Counter(all_keywords).most_common(5)

            # Year distribution
            years = [p.get("publication_year") for p in cluster_papers if p.get("publication_year")]
            year_dist = Counter(years)

            # Departments in this cluster
            depts = Counter(p.get("department", "Unknown") for p in cluster_papers)

            cluster_topics[f"cluster_{cluster_id}"] = {
                "cluster_id": cluster_id,
                "paper_count": len(cluster_papers),
                "top_terms": top_terms[:8],
                "topic_label": " + ".join(top_terms[:3]),  # Human-readable label
                "top_keywords": [k for k, _ in keyword_freq],
                "year_distribution": dict(sorted(year_dist.items())),
                "departments": dict(depts.most_common(3)),
                "paper_titles": [p["title"] for p in cluster_papers[:3]],
            }

        return cluster_topics

    def get_trending_keywords(self, top_n: int = 20) -> List[Dict]:
        """Get the most trending keywords across all papers."""
        col = get_papers_collection()
        all_keywords = []
        for paper in col.find({}, {"keywords": 1, "publication_year": 1}):
            year = paper.get("publication_year", 2020)
            weight = 1 + max(0, year - 2020) * 0.1  # Recent papers weighted higher
            for kw in paper.get("keywords", []):
                all_keywords.extend([kw.lower()] * int(weight * 10))

        freq = Counter(all_keywords)
        return [{"keyword": kw, "count": count} for kw, count in freq.most_common(top_n)]

    def run_full_analysis(self) -> Dict:
        """Run trend detection on all papers in the database."""
        col = get_papers_collection()
        papers = list(col.find(
            {"processing_status": "completed"},
            {"title": 1, "abstract": 1, "keywords": 1, "department": 1,
             "publication_year": 1, "authors": 1, "paper_id": 1}
        ))

        if not papers:
            return {
                "clusters": {},
                "trending_keywords": [],
                "total_papers_analyzed": 0,
                "message": "No processed papers found"
            }

        clusters = self.fit(papers)
        trending_kw = self.get_trending_keywords()

        return {
            "clusters": clusters,
            "trending_keywords": trending_kw,
            "total_papers_analyzed": len(papers),
            "analyzed_at": datetime.utcnow().isoformat(),
        }
