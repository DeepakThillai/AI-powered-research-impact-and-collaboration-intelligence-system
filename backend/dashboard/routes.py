"""
backend/dashboard/routes.py - Analytics Dashboard API
"""
from typing import Optional
from datetime import datetime
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from backend.auth.auth_handler import get_current_user, require_research_head
from backend.database import get_papers_collection, get_users_collection
from backend.ml.trend_detector import TrendDetector
from backend.ml.collaboration_network import CollaborationNetwork
from backend.ml.impact_predictor import ImpactPredictor

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


def _safe_dt(val):
    """Convert datetime to ISO string, leave other types unchanged."""
    if isinstance(val, datetime):
        return val.isoformat()
    return val


def _serialize(doc: dict) -> dict:
    """Recursively make a MongoDB doc JSON-safe."""
    result = {}
    for k, v in doc.items():
        if k == "_id":
            continue
        if isinstance(v, datetime):
            result[k] = v.isoformat()
        elif isinstance(v, dict):
            result[k] = _serialize(v)
        elif isinstance(v, list):
            result[k] = [
                _serialize(i) if isinstance(i, dict) else _safe_dt(i)
                for i in v
            ]
        else:
            result[k] = v
    return result


# ── Overview (all roles) ──────────────────────────────────────
@router.get("/overview")
def get_overview(current_user: dict = Depends(get_current_user)):
    """High-level stats visible to all authenticated users."""
    col = get_papers_collection()
    users_col = get_users_collection()

    total_papers = col.count_documents({})
    total_users = users_col.count_documents({})
    processed_papers = col.count_documents({"processing_status": "completed"})

    dept_pipeline = [
        {"$group": {"_id": "$department", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    dept_counts = [
        {"department": r["_id"], "count": r["count"]}
        for r in col.aggregate(dept_pipeline) if r["_id"]
    ]

    year_pipeline = [
        {"$group": {"_id": "$publication_year", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    year_counts = [
        {"year": r["_id"], "count": r["count"]}
        for r in col.aggregate(year_pipeline) if r["_id"]
    ]

    raw_recent = list(
        col.find({}, {"title": 1, "department": 1, "upload_date": 1,
                      "authors": 1, "venue": 1, "_id": 0})
        .sort("upload_date", -1)
        .limit(5)
    )
    recent = [_serialize(r) for r in raw_recent]

    return {
        "total_papers": total_papers,
        "processed_papers": processed_papers,
        "total_users": total_users,
        "department_counts": dept_counts,
        "year_wise_counts": year_counts,
        "recent_uploads": recent,
    }


# ── Trending Research Domains ─────────────────────────────────
@router.get("/trends")
def get_trends(current_user: dict = Depends(get_current_user)):
    """Research trend analysis using TF-IDF + KMeans clustering."""
    try:
        detector = TrendDetector()
        result = detector.run_full_analysis()
        return result
    except Exception as e:
        logger.error(f"Trend detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")


# ── Collaboration Network ─────────────────────────────────────
@router.get("/collaboration")
def get_collaboration(
    max_nodes: int = 50,
    current_user: dict = Depends(get_current_user),
):
    """Co-authorship network analysis and graph data."""
    try:
        network = CollaborationNetwork()
        network.build_network()
        stats = network.get_network_stats()
        graph_data = network.get_graph_for_visualization(max_nodes=max_nodes)
        return {"stats": stats, "graph": graph_data}
    except Exception as e:
        logger.error(f"Collaboration network error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Collaboration analysis failed: {str(e)}"
        )


# ── High Impact Papers ────────────────────────────────────────
@router.get("/impact")
def get_high_impact_papers(
    top_n: int = 10,
    current_user: dict = Depends(get_current_user),
):
    """Papers ranked by predicted citation impact score."""
    try:
        predictor = ImpactPredictor()
        papers = predictor.get_top_impact_papers(top_n=top_n)
        return {"top_papers": papers}
    except Exception as e:
        logger.error(f"Impact prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Impact prediction failed: {str(e)}"
        )


# ── Research Head Full Dashboard ──────────────────────────────
@router.get("/research-head")
def get_research_head_dashboard(
    current_user: dict = Depends(require_research_head),
):
    """Comprehensive dashboard — Research Head role only."""
    col = get_papers_collection()
    users_col = get_users_collection()

    total_papers = col.count_documents({})

    dept_pipeline = [
        {"$group": {"_id": "$department", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    dept_counts = {
        r["_id"]: r["count"]
        for r in col.aggregate(dept_pipeline) if r["_id"]
    }

    # Top authors by paper count
    author_counter: Counter = Counter()
    for paper in col.find({}, {"authors": 1}):
        for author in paper.get("authors", []):
            author_counter[author] += 1
    top_authors = [
        {"author": a, "papers": c}
        for a, c in author_counter.most_common(10)
    ]

    # Top publication venues
    venue_pipeline = [
        {"$group": {"_id": "$venue", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10},
    ]
    top_venues = [
        {"venue": r["_id"], "count": r["count"]}
        for r in col.aggregate(venue_pipeline) if r["_id"]
    ]

    # User activity (serialize datetimes)
    raw_users = list(
        users_col.find(
            {},
            {"name": 1, "department": 1, "role": 1,
             "papers_uploaded": 1, "last_login": 1, "_id": 0}
        ).limit(20)
    )
    user_activity = []
    for u in raw_users:
        user_activity.append({
            "name": u.get("name", ""),
            "department": u.get("department", ""),
            "role": u.get("role", ""),
            "papers_count": len(u.get("papers_uploaded", [])),
            "last_login": _safe_dt(u.get("last_login")),
        })

    # Trending keywords
    detector = TrendDetector()
    trending_keywords = detector.get_trending_keywords(top_n=15)

    # Top impact papers
    predictor = ImpactPredictor()
    top_impact_papers = predictor.get_top_impact_papers(top_n=5)

    return {
        "summary": {
            "total_papers": total_papers,
            "total_departments": len(dept_counts),
            "total_users": users_col.count_documents({}),
        },
        "department_counts": dept_counts,
        "top_authors": top_authors,
        "top_venues": top_venues,
        "trending_keywords": trending_keywords,
        "top_impact_papers": top_impact_papers,
        "user_activity": user_activity,
        "generated_at": datetime.utcnow().isoformat(),
    }


# ── My Papers Stats ───────────────────────────────────────────
@router.get("/my-papers")
def get_my_papers_stats(current_user: dict = Depends(get_current_user)):
    """Statistics for the current user's uploaded papers."""
    col = get_papers_collection()
    raw_papers = list(
        col.find(
            {"uploaded_by": current_user["user_id"]},
            {"_id": 0, "extracted_text": 0, "text_chunks": 0}
        ).sort("upload_date", -1)
    )
    papers = [_serialize(p) for p in raw_papers]
    return {"total": len(papers), "papers": papers}
