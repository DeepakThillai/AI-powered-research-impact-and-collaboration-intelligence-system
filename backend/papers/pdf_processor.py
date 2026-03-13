"""
backend/papers/pdf_processor.py - PDF Text Extraction & Cleaning Pipeline
"""
import re
from pathlib import Path
from typing import Tuple, List, Dict

from loguru import logger

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    from pdfminer.layout import LAParams
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False


class PDFProcessor:
    """Extracts and cleans text from research paper PDFs."""

    @staticmethod
    def extract_text(file_path: str) -> Tuple[str, int]:
        """
        Extract text from PDF using PyPDF2 (primary) or pdfminer (fallback).
        Returns: (cleaned_text, page_count)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        raw_text = ""
        page_count = 0

        if HAS_PYPDF2:
            try:
                raw_text, page_count = PDFProcessor._extract_pypdf2(file_path)
                logger.debug(f"PyPDF2 extracted {page_count} pages from {path.name}")
            except Exception as e:
                logger.warning(f"PyPDF2 failed ({e}), trying pdfminer...")

        if not raw_text and HAS_PDFMINER:
            try:
                raw_text, page_count = PDFProcessor._extract_pdfminer(file_path)
                logger.debug(f"pdfminer extracted text from {path.name}")
            except Exception as e:
                logger.error(f"pdfminer also failed: {e}")

        if not raw_text:
            raise ValueError(
                "Could not extract text from PDF. "
                "The file may be scanned/image-based or password-protected."
            )

        cleaned = PDFProcessor.clean_text(raw_text)
        return cleaned, page_count

    @staticmethod
    def _extract_pypdf2(file_path: str) -> Tuple[str, int]:
        text_parts = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            page_count = len(reader.pages)
            for page in reader.pages:
                text = page.extract_text() or ""
                text_parts.append(text)
        return "\n".join(text_parts), page_count

    @staticmethod
    def _extract_pdfminer(file_path: str) -> Tuple[str, int]:
        laparams = LAParams(line_margin=0.5, char_margin=2.0)
        text = pdfminer_extract(file_path, laparams=laparams)
        page_count = text.count("\x0c") + 1
        return text, page_count

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Clean extracted PDF text."""
        if not raw_text:
            return ""
        text = raw_text
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = text.replace("\x0c", "\n")
        text = re.sub(r"^\s*[-\u2013]?\s*\d+\s*[-\u2013]?\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[^\x20-\x7E\n]", " ", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """Attempt to extract common paper sections."""
        sections = {}
        section_patterns = {
            "abstract": r"(?i)abstract[:\s]+(.*?)(?=\n\s*(?:introduction|keywords|1\.|I\.))",
            "introduction": r"(?i)(?:1\s*\.?\s*)?introduction[:\s]+(.*?)(?=\n\s*(?:2\s*\.|\brelated\b|\bmethodology\b|\bbackground\b))",
            "conclusion": r"(?i)(?:\d+\s*\.?\s*)?conclus(?:ion|ions)[:\s]+(.*?)(?=\n\s*(?:references|bibliography|acknowledgment)|\Z)",
        }
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()[:2000]
        return sections

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks for embedding."""
        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk: List[str] = []
        current_size = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_size + sentence_len > chunk_size and current_chunk:
                chunk_text_str = " ".join(current_chunk)
                chunks.append(chunk_text_str)
                # Keep overlap
                overlap_sentences: List[str] = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                current_chunk = overlap_sentences
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c for c in chunks if len(c.strip()) > 50]
