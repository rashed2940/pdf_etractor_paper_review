"""Paper ingestion orchestrator: PDF -> cleaned -> chunked -> embedded -> indexed.

Single entry point `ingest_pdf(path, paper_id=None)` used by both the CLI
and the background processing threads (Phase 5).

Handles (AGENTS.md §27):
- Encrypted / empty / corrupt PDFs -> PDFLoadError, surfaced to caller.
- Scanned PDFs -> optional OCR fallback (opt-in via OCR_ENABLED).
- Duplicate detection by SHA-256 (AGENTS.md §25).

Never silently fails: per-paper exceptions propagate to the caller so the
processing thread can mark the paper as failed in the status registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.logging import get_logger
from config.settings import get_settings
from embeddings.embedding_service import get_embedding_service
from ingestion.chunker import chunk_pages
from ingestion.metadata_extractor import extract_metadata
from ingestion.ocr_loader import OCRError, ocr_pdf
from ingestion.pdf_loader import (
    ExtractedDocument,
    PDFLoadError,
    compute_sha256,
    extract_pdf,
    is_scanned,
)
from ingestion.text_cleaner import clean_pages
from vectorstore.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Outcome of ingesting one PDF."""

    paper_id: str
    filename: str
    sha256: str
    page_count: int
    chunk_count: int
    metadata: dict[str, Any]
    is_duplicate: bool = False


def _build_paper_id(path: Path, sha256_hex: str) -> str:
    """Use the SHA-256 (first 12 hex chars) as the paper_id for stability."""
    return f"paper_{sha256_hex[:12]}"


def _extract_document(path: Path) -> ExtractedDocument:
    """Run text extraction, falling back to OCR for scanned PDFs."""
    try:
        doc = extract_pdf(path)
    except PDFLoadError as exc:
        s = get_settings()
        if "no extractable text" in str(exc).lower() and s.ocr_enabled:
            logger.warning(
                "Text extraction yielded no text; attempting OCR: %s", path
            )
            try:
                return ocr_pdf(path)
            except OCRError as ocr_exc:
                raise PDFLoadError(
                    f"PDF appears scanned and OCR failed for {path}: {ocr_exc}"
                ) from ocr_exc
        raise

    # If extracted but looks scanned, also try OCR when enabled.
    s = get_settings()
    if s.ocr_enabled and is_scanned(doc.pages, s.ocr_min_chars_per_page):
        logger.warning("Document appears scanned; attempting OCR: %s", path)
        try:
            return ocr_pdf(path)
        except OCRError as ocr_exc:
            logger.warning(
                "OCR fallback failed; using sparse text result: %s", ocr_exc
            )
            return doc

    return doc


def ingest_pdf(
    path: Path | str,
    *,
    vector_store: ChromaVectorStore | None = None,
    paper_id: str | None = None,
) -> IngestionResult:
    """Ingest a single PDF end-to-end.

    Args:
        path: Path to the PDF file.
        vector_store: Optional pre-built store (used in tests). A new one is
            created if omitted, which involves loading the embedding model.
        paper_id: Optional override; defaults to `paper_<sha256[:12]>`.

    Returns:
        IngestionResult with chunk_count > 0 on success.

    Raises:
        PDFLoadError: file unreadable / no text / OCR fallback failed.
    """
    path = Path(path)
    sha = compute_sha256(path)
    pid = paper_id or _build_paper_id(path, sha)

    # Duplicate detection (AGENTS.md §25).
    if vector_store is None:
        vector_store = ChromaVectorStore(get_embedding_service())
    existing = [p for p in vector_store.list_papers() if p.get("paper_id") == pid]
    if existing:
        logger.info("Skipping duplicate paper (sha256=%s): %s", sha[:12], path)
        return IngestionResult(
            paper_id=pid,
            filename=path.name,
            sha256=sha,
            page_count=0,
            chunk_count=0,
            metadata={},
            is_duplicate=True,
        )

    doc = _extract_document(path)

    # Clean each page (preserves page numbers).
    cleaned = clean_pages((p.page_number, p.text) for p in doc.pages)
    first_page_text = cleaned[0][1] if cleaned else ""
    metadata = extract_metadata(doc.metadata, first_page_text)
    metadata["sha256"] = sha
    metadata["page_count"] = doc.page_count

    # Chunk (section-aware with fallback).
    chunks = chunk_pages(pid, cleaned)
    if not chunks:
        raise PDFLoadError(
            f"Chunking produced 0 chunks for {path} (text too short?)"
        )

    added = vector_store.add_chunks(
        chunks,
        filename=path.name,
        title=metadata.get("title"),
    )

    logger.info(
        "Ingested paper_id=%s pages=%d chunks=%d title=%r",
        pid,
        doc.page_count,
        added,
        metadata.get("title"),
    )

    return IngestionResult(
        paper_id=pid,
        filename=path.name,
        sha256=sha,
        page_count=doc.page_count,
        chunk_count=added,
        metadata=metadata,
        is_duplicate=False,
    )


def ingest_directory(
    directory: Path | str,
    *,
    vector_store: ChromaVectorStore | None = None,
) -> list[IngestionResult]:
    """Ingest every *.pdf in `directory` (non-recursive).

    Per-paper exceptions are caught and logged so one bad PDF does not
    abort the whole batch (AGENTS.md §27).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    results: list[IngestionResult] = []
    for pdf in sorted(directory.glob("*.pdf")):
        try:
            results.append(ingest_pdf(pdf, vector_store=vector_store))
        except PDFLoadError as exc:
            logger.error("Failed to ingest %s: %s", pdf, exc)
            results.append(
                IngestionResult(
                    paper_id="",
                    filename=pdf.name,
                    sha256="",
                    page_count=0,
                    chunk_count=0,
                    metadata={"error": str(exc)},
                )
            )
    return results


__all__ = ["IngestionResult", "ingest_pdf", "ingest_directory"]
