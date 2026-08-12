"""PDF text extraction (text-based PDFs).

Primary path uses PyMuPDF (`fitz`). Designed so that OCR is a separate
adapter (AGENTS.md §28) and can be swapped in transparently.

Returns page-tracked text so the chunker can preserve page citations.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF (formerly `import fitz`)


@dataclass
class ExtractedPage:
    """One page of extracted text + page metadata."""

    page_number: int  # 1-based
    text: str


@dataclass
class ExtractedDocument:
    """All pages of a PDF + PyMuPDF document metadata."""

    metadata: dict[str, object]
    pages: list[ExtractedPage]
    sha256: str
    page_count: int


class PDFLoadError(Exception):
    """Raised when a PDF cannot be parsed (encrypted, corrupt, empty)."""


def compute_sha256(path: Path) -> str:
    """Stream the file and return its SHA-256 hex digest."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_scanned(
    pages: list[ExtractedPage], min_chars_per_page: int = 50
) -> bool:
    """Heuristic: a PDF is 'scanned' if most pages have very little text.

    Returns True when >= 80% of pages have < min_chars_per_page chars.
    """
    if not pages:
        return True
    sparse = sum(1 for p in pages if len(p.text.strip()) < min_chars_per_page)
    return sparse / len(pages) >= 0.8


def extract_pdf(
    path: Path | str,
    *,
    min_chars_per_page: int = 50,
    auto_ocr: bool | None = None,
) -> ExtractedDocument:
    """Open a PDF with PyMuPDF and extract text page-by-page.

    Args:
        path: Path to the PDF file.
        min_chars_per_page: Used by `is_scanned` heuristic downstream.
        auto_ocr: If True, automatically fall back to OCR when text
            density is low (overrides settings.ocr_enabled). If None,
            follows `settings.ocr_enabled`.

    Raises:
        PDFLoadError: file missing, encrypted, empty, or unreadable.
    """
    path = Path(path)
    if not path.exists():
        raise PDFLoadError(f"PDF not found: {path}")
    if path.stat().st_size == 0:
        raise PDFLoadError(f"PDF is empty (0 bytes): {path}")

    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # pragma: no cover - PyMuPDF raises generic errors
        raise PDFLoadError(f"PyMuPDF could not open {path}: {exc}") from exc

    try:
        if doc.is_encrypted:
            # AGENTS.md §27: handle encrypted PDFs gracefully.
            raise PDFLoadError(
                f"PDF is encrypted and cannot be read without a password: {path}"
            )

        if doc.page_count == 0:
            raise PDFLoadError(f"PDF has 0 pages: {path}")

        raw_meta = doc.metadata or {}
        pages: list[ExtractedPage] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            pages.append(ExtractedPage(page_number=i + 1, text=text))

    finally:
        doc.close()

    has_text = any(p.text.strip() for p in pages)
    if not has_text:
        raise PDFLoadError(f"PDF contains no extractable text (likely scanned): {path}")

    # --- Phase 6: auto-OCR when text density is low and OCR is enabled ---
    from config.settings import get_settings as _gs

    settings = _gs()
    should_ocr = auto_ocr if auto_ocr is not None else settings.ocr_enabled
    if should_ocr and is_scanned(pages, settings.ocr_min_chars_per_page):
        from ingestion.ocr_loader import OCRError, ocr_pdf

        logger_msg = (
            "PDF appears scanned (low text density); auto-OCR enabled — attempting OCR for %s",
            path,
        )
        try:
            return ocr_pdf(path)
        except OCRError as exc:
            # Surface a warning via log; fall back to the sparse text result
            # rather than failing entirely. This is best-effort OCR.
            from config.logging import get_logger as _gl

            _gl(__name__).warning(
                "Auto-OCR failed for %s: %s (returning sparse text result)",
                path,
                exc,
            )

    return ExtractedDocument(
        metadata=dict(raw_meta),
        pages=pages,
        sha256=compute_sha256(path),
        page_count=len(pages),
    )


__all__ = [
    "ExtractedPage",
    "ExtractedDocument",
    "PDFLoadError",
    "compute_sha256",
    "is_scanned",
    "extract_pdf",
]
