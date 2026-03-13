"""
backend/papers/routes.py - Paper Upload & Retrieval API
"""
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

from backend.auth.auth_handler import get_current_user
from backend.config import settings
from backend.database import get_papers_collection, get_users_collection
from backend.papers.pdf_processor import PDFProcessor

router = APIRouter(prefix="/api/papers", tags=["Papers"])

MAX_FILE_SIZE_MB = 50  # 50 MB limit


def _serialize_paper(paper: dict) -> dict:
    """Convert MongoDB document to JSON-safe dict."""
    result = {}
    for k, v in paper.items():
        if k == "_id":
            continue
        if isinstance(v, datetime):
            result[k] = v.isoformat()
        else:
            result[k] = v
    return result


# ── Upload Paper ──────────────────────────────────────────────
@router.post("/upload", status_code=201)
async def upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    abstract: str = Form(...),
    keywords: str = Form(...),
    department: str = Form(...),
    authors: str = Form(...),
    publication_year: int = Form(...),
    venue: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload a research paper PDF with metadata."""

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Validate year
    current_year = datetime.utcnow().year
    if not (1900 <= publication_year <= current_year + 1):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid publication year: {publication_year}"
        )

    # Validate required text fields
    if not title.strip():
        raise HTTPException(status_code=400, detail="Title is required")
    if not abstract.strip():
        raise HTTPException(status_code=400, detail="Abstract is required")

    # Generate paper ID and save file
    paper_id = str(uuid.uuid4())
    safe_title = "".join(c if c.isalnum() else "_" for c in title.strip()[:50])
    filename = f"{paper_id}_{safe_title}.pdf"
    file_path = Path(settings.papers_storage_path) / filename

    # Read file content and check size
    file_content = await file.read()
    file_size_mb = len(file_content) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size_mb:.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB"
        )

    # Save PDF to disk
    try:
        with open(file_path, "wb") as f:
            f.write(file_content)
        logger.info(f"Saved PDF: {filename} ({file_size_mb:.2f} MB)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Parse comma-separated lists
    keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
    authors_list = [a.strip() for a in authors.split(",") if a.strip()]

    if not authors_list:
        raise HTTPException(status_code=400, detail="At least one author is required")

    # Build metadata document
    paper_doc = {
        "paper_id": paper_id,
        "title": title.strip(),
        "abstract": abstract.strip(),
        "keywords": keywords_list,
        "department": department.strip(),
        "authors": authors_list,
        "publication_year": publication_year,
        "venue": venue.strip(),
        "uploaded_by": current_user["user_id"],
        "uploaded_by_name": current_user["name"],
        "file_path": str(file_path),
        "filename": filename,
        "file_size_mb": round(file_size_mb, 2),
        "page_count": 0,
        "upload_date": datetime.utcnow(),
        "processing_status": "pending",
        "extracted_text": "",
        "text_chunks": [],
        "sections": {},
        "embedding_stored": False,
        "predicted_impact_score": None,
        "processing_error": None,
    }

    # Insert into MongoDB
    col = get_papers_collection()
    col.insert_one(paper_doc)

    # Update user's paper list
    get_users_collection().update_one(
        {"user_id": current_user["user_id"]},
        {"$push": {"papers_uploaded": paper_id}}
    )

    # Trigger background processing
    background_tasks.add_task(process_paper_background, paper_id, str(file_path))

    return {
        "message": "Paper uploaded successfully. Processing started in background.",
        "paper_id": paper_id,
        "status": "pending"
    }


# ── Background Processing Pipeline ───────────────────────────
def process_paper_background(paper_id: str, file_path: str):
    """
    Full processing pipeline: extract text → chunk → embed → store in ChromaDB → predict impact.
    Runs as a FastAPI background task after upload.
    """
    col = get_papers_collection()
    col.update_one(
        {"paper_id": paper_id},
        {"$set": {"processing_status": "processing"}}
    )

    try:
        # Step 1: Extract text from PDF
        logger.info(f"[{paper_id[:8]}] Step 1: Extracting text from PDF...")
        cleaned_text, page_count = PDFProcessor.extract_text(file_path)
        sections = PDFProcessor.extract_sections(cleaned_text)
        chunks = PDFProcessor.chunk_text(cleaned_text)

        col.update_one({"paper_id": paper_id}, {"$set": {
            "extracted_text": cleaned_text[:50000],
            "text_chunks": chunks,
            "sections": sections,
            "page_count": page_count,
        }})
        logger.info(f"[{paper_id[:8]}] Text extracted: {page_count} pages, {len(chunks)} chunks")

        # Step 2: Generate embeddings and store in ChromaDB
        logger.info(f"[{paper_id[:8]}] Step 2: Generating embeddings...")
        # Re-fetch doc AFTER text update to pass complete doc to ChromaDB
        paper_doc = col.find_one({"paper_id": paper_id})
        if not paper_doc:
            raise ValueError(f"Paper {paper_id} not found in DB after text extraction")

        from backend.vectordb.chroma_manager import ChromaManager
        chroma = ChromaManager()
        chroma.add_paper(paper_doc, chunks)

        col.update_one({"paper_id": paper_id}, {"$set": {
            "processing_status": "completed",
            "embedding_stored": True,
        }})
        logger.info(f"[{paper_id[:8]}] Embeddings stored in ChromaDB")

        # Step 3: Predict impact score
        logger.info(f"[{paper_id[:8]}] Step 3: Predicting impact score...")
        from backend.ml.impact_predictor import ImpactPredictor
        predictor = ImpactPredictor()
        score = predictor.predict_single(paper_doc)
        col.update_one(
            {"paper_id": paper_id},
            {"$set": {"predicted_impact_score": score}}
        )
        logger.success(f"[{paper_id[:8]}] ✅ Processing complete. Impact score: {score:.2f}")

    except Exception as e:
        logger.error(f"[{paper_id[:8]}] ❌ Processing failed: {e}")
        col.update_one({"paper_id": paper_id}, {"$set": {
            "processing_status": "failed",
            "processing_error": str(e),
        }})


# ── List Papers ───────────────────────────────────────────────
@router.get("/list")
def list_papers(
    department: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 50,
    skip: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """List papers with optional filters. Returns JSON-safe documents."""
    col = get_papers_collection()
    query: dict = {}
    if department:
        query["department"] = department
    if year:
        query["publication_year"] = year

    raw_papers = list(
        col.find(query, {"_id": 0, "extracted_text": 0, "text_chunks": 0})
        .sort("upload_date", -1)
        .skip(skip)
        .limit(limit)
    )
    total = col.count_documents(query)
    papers = [_serialize_paper(p) for p in raw_papers]

    return {"total": total, "papers": papers}


@router.get("/my")
def get_my_papers(current_user: dict = Depends(get_current_user)):
    """Get papers uploaded by the current user."""
    col = get_papers_collection()
    raw_papers = list(
        col.find(
            {"uploaded_by": current_user["user_id"]},
            {"_id": 0, "extracted_text": 0, "text_chunks": 0}
        ).sort("upload_date", -1)
    )
    papers = [_serialize_paper(p) for p in raw_papers]
    return {"total": len(papers), "papers": papers}


@router.get("/{paper_id}/status")
def get_processing_status(paper_id: str, current_user: dict = Depends(get_current_user)):
    """Check processing status of an uploaded paper."""
    col = get_papers_collection()
    paper = col.find_one(
        {"paper_id": paper_id},
        {"processing_status": 1, "embedding_stored": 1, "processing_error": 1, "_id": 0}
    )
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


@router.get("/{paper_id}")
def get_paper(paper_id: str, current_user: dict = Depends(get_current_user)):
    """Get details of a specific paper."""
    col = get_papers_collection()
    paper = col.find_one(
        {"paper_id": paper_id},
        {"_id": 0, "extracted_text": 0, "text_chunks": 0}
    )
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return _serialize_paper(paper)
