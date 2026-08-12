"""PDF metadata extraction.

Combines:
1. PyMuPDF document metadata (`doc.metadata`).
2. Regex-based heuristics on the first page (DOI, authors, title).

Never invents metadata (AGENTS.md §11.2, §24): if uncertain, returns `None`
or an empty list. Cross-checks regex hits against PDF metadata when both
are available.
"""
from __future__ import annotations

import re
from typing import Any

# PyMuPDF returns strings; the keys we care about.
_PDF_META_KEYS = {
    "title": "title",
    "author": "authors",
    "creationDate": "creation_date_raw",
    "modDate": "mod_date_raw",
    "subject": "subject",
    "keywords": "keywords",
}

# --- DOI regex (per Crossref recommended format) ---
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)

# --- arXiv id (a bonus identifier) ---
_ARXIV_RE = re.compile(r"\barXiv:(\d{4}\.\d{4,5}(v\d+)?)\b", re.IGNORECASE)

# --- Year on title page (1900-2099) ---
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")

# --- IEEE/ACM-style author blocks (very rough heuristic) ---
# Looks for a comma-separated name list near the top of page 1.
_AUTHOR_LINE_RE = re.compile(
    r"^([A-Z][a-zA-Z\-']+(?:\s+[A-Z]\.)?\s+[A-Z][a-zA-Z\-']+"
    r"(?:,\s*[A-Z][a-zA-Z\-']+(?:\s+[A-Z]\.)?\s+[A-Z][a-zA-Z\-']+)+)$"
)


def _coerce_authors(raw: Any) -> list[str]:
    """PyMuPDF returns authors as a single semicolon-separated string in
    many PDFs; some return a list. Normalize to a list of trimmed names."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(a).strip() for a in raw if str(a).strip()]
    text = str(raw)
    # Try semicolon, then newline, then " and ".
    for sep in (";", "\n", " and "):
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if parts:
                return parts
    return [text.strip()] if text.strip() else []


def _coerce_year(raw: Any) -> str | None:
    """Extract a 4-digit year (1950-2029) from a string or PDF date field."""
    if raw is None:
        return None
    text = str(raw)
    m = _YEAR_RE.search(text)
    return m.group(0) if m else None


def _extract_doi(text: str) -> str | None:
    m = _DOI_RE.search(text)
    return m.group(0) if m else None


def _extract_arxiv(text: str) -> str | None:
    m = _ARXIV_RE.search(text)
    return m.group(0) if m else None


def _guess_title(first_page_text: str, pdf_title: str | None) -> str | None:
    """Prefer PyMuPDF's title; otherwise take the longest non-empty line
    in the first ~30 lines as a weak title guess.

    Returns None when ambiguous — we don't fabricate.
    """
    if pdf_title and pdf_title.strip():
        return pdf_title.strip()
    if not first_page_text:
        return None
    lines = [ln.strip() for ln in first_page_text.splitlines() if ln.strip()]
    candidates = [ln for ln in lines[:30] if 8 <= len(ln) <= 200]
    if not candidates:
        return None
    # Pick the longest (a rough proxy for the actual title).
    return max(candidates, key=len)


def _guess_authors(
    first_page_text: str, pdf_authors: list[str]
) -> list[str]:
    """Prefer PDF metadata authors; otherwise fall back to a heuristic
    comma-separated author line detection. Returns [] when ambiguous."""
    if pdf_authors:
        return pdf_authors
    if not first_page_text:
        return []
    lines = [ln.strip() for ln in first_page_text.splitlines() if ln.strip()]
    for line in lines[:30]:
        # Skip the title-looking line.
        if len(line) > 200:
            continue
        if _AUTHOR_LINE_RE.match(line):
            return [a.strip() for a in line.split(",") if a.strip()]
    return []


def _guess_venue(first_page_text: str) -> str | None:
    """Try common venue markers like 'Proceedings of the ...', 'IEEE ...', etc."""
    if not first_page_text:
        return None
    patterns = [
        re.compile(r"(?:Proceedings of the|IEEE|ACM|USENIX|NDSS|S\&\?P|CCS)\s+.{3,120}"),
        re.compile(r"Published in\s+.{3,120}"),
        re.compile(r"Journal of\s+.{3,120}"),
    ]
    for pat in patterns:
        m = pat.search(first_page_text)
        if m:
            return m.group(0).strip()
    return None


def extract_metadata(
    pdf_meta: dict[str, Any],
    first_page_text: str,
) -> dict[str, Any]:
    """Combine PyMuPDF metadata with first-page heuristics.

    Args:
        pdf_meta: The dict from PyMuPDF `doc.metadata`. May be partial.
        first_page_text: Cleaned text of page 1 (no headers/footers).

    Returns:
        Dict with keys: title, authors, publication_date, venue, doi,
        arxiv_id, institution. Each value is the best available guess
        or None / [] if uncertain.
    """
    pdf_title = (pdf_meta.get("title") or "").strip() or None
    pdf_authors = _coerce_authors(pdf_meta.get("author"))

    # PyMuPDF dates come like "D:20240315123045+00'00'"; year is the first 4 digits.
    pdf_year = _coerce_year(pdf_meta.get("creationDate")) or _coerce_year(
        pdf_meta.get("modDate")
    )

    # First-page heuristics
    fp_year = _coerce_year(first_page_text)
    doi = _extract_doi(first_page_text)
    arxiv_id = _extract_arxiv(first_page_text)
    venue = _guess_venue(first_page_text)

    return {
        "title": _guess_title(first_page_text, pdf_title),
        "authors": _guess_authors(first_page_text, pdf_authors),
        "publication_date": pdf_year or fp_year,
        "venue": venue,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "institution": None,  # Not reliably detectable without heuristics we don't trust
    }


__all__ = ["extract_metadata"]
