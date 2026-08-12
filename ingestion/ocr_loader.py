"""OCR fallback for scanned PDFs (AGENTS.md §28).

Initial implementation uses pytesseract over rasterized pages. Only
activated when:
- `OCR_ENABLED=true` in settings, AND
- the PDF was detected as scanned by `pdf_loader.is_scanned`.

The OCR adapter keeps the same return type (`ExtractedDocument`) so
the rest of the pipeline doesn't care which loader was used.
"""
from __future__ import annotations

from pathlib import Path

from config.logging import get_logger
from config.settings import get_settings
from ingestion.pdf_loader import (
    ExtractedDocument,
    ExtractedPage,
    PDFLoadError,
    compute_sha256,
)

logger = get_logger(__name__)


class OCRError(Exception):
    """Raised when OCR cannot run (binary missing, conversion fails, etc.)."""


def _ocr_enabled() -> bool:
    return bool(get_settings().ocr_enabled)


def ocr_pdf(path: Path | str) -> ExtractedDocument:
    """Run OCR over every page and return an ExtractedDocument.

    Raises:
        OCRError: OCR is disabled, or a dependency is missing, or Tesseract fails.
    """
    if not _ocr_enabled():
        raise OCRError(
            "OCR is disabled. Set OCR_ENABLED=true in .env to use the OCR fallback."
        )

    # Lazy imports: only pay the cost when OCR is actually requested.
    try:
        import pymupdf  # PyMuPDF for rasterization
        import pytesseract
    except ImportError as exc:  # pragma: no cover - optional dep
        raise OCRError(
            "OCR requires `pytesseract` and `pdf2image`. "
            "Install with `pip install pytesseract pdf2image`."
        ) from exc

    settings = get_settings()
    if settings.tesseract_cmd.strip():
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd.strip()

    path = Path(path)
    if not path.exists():
        raise OCRError(f"PDF not found: {path}")

    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # pragma: no cover
        raise OCRError(f"PyMuPDF could not open {path}: {exc}") from exc

    pages: list[ExtractedPage] = []
    raw_meta: dict[str, object] = {}
    try:
        if doc.page_count == 0:
            raise OCRError(f"PDF has 0 pages: {path}")
        raw_meta = dict(doc.metadata or {})

        for i in range(doc.page_count):
            page = doc.load_page(i)
            # 300 DPI is a good balance for academic text.
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            # pytesseract can read PNG bytes directly.
            text = pytesseract.image_to_string(img_bytes)
            pages.append(ExtractedPage(page_number=i + 1, text=text or ""))
    finally:
        doc.close()

    if not any(p.text.strip() for p in pages):
        raise OCRError(
            f"OCR produced no text (Tesseract binary installed and on PATH?): {path}"
        )

    logger.info(
        "OCR completed",
        extra={"path": str(path), "pages": len(pages)},
    )

    return ExtractedDocument(
        metadata=raw_meta,
        pages=pages,
        sha256=compute_sha256(path),
        page_count=len(pages),
    )


__all__ = ["OCRError", "ocr_pdf"]
