"""PDF text cleaning utilities.

Applied AFTER extraction (PyMuPDF or OCR) and BEFORE chunking. Goals
(AGENTS.md §6):
- Fix hyphenated line breaks from PDF reflow ("exam-\nple" -> "example").
- Normalize whitespace and ligatures (e.g. "ﬁ" -> "fi").
- Strip repeating headers/footers (e.g. page numbers, running titles).

Pure functions; no I/O; safe to unit-test.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

# Common ligatures used in academic PDFs.
_LIGATURES: dict[str, str] = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "–": "-",  # en dash
    "—": "-",  # em dash
    " ": " ",  # non-breaking space
    " ": " ",  # thin space
    "": "",   # zero-width space
}

# Line-break hyphenation: a letter, hyphen, newline, letter -> join.
_HYPHEN_BREAK = re.compile(r"(?<=[A-Za-z])-\n(?=[A-Za-z])")

# Collapse multiple whitespace (including newlines) into a single space,
# but PRESERVE single newlines so section headers stay on their own line
# (needed by ingestion.chunker._detect_sections).
_MULTI_NEWLINE = re.compile(r"\n{2,}")
_MULTI_SPACE = re.compile(r"[ \t]+")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")  # trailing whitespace before newline


def normalize_unicode(text: str) -> str:
    """Replace common PDF ligatures and whitespace characters."""
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    return text


def fix_hyphenation(text: str) -> str:
    """Join words broken by a hyphen + newline across PDF lines."""
    return _HYPHEN_BREAK.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Collapse intra-line spaces; preserve single newlines; normalize
    paragraph breaks to exactly one blank line.

    Why preserve single newlines: section headers like "Abstract" or
    "1. Introduction" live on their own line in academic PDFs; collapsing
    them to spaces breaks downstream section-aware chunking.
    """
    # Strip trailing whitespace before each newline.
    text = _TRAILING_SPACE.sub("\n", text)
    # Collapse runs of spaces/tabs (within a line) to a single space.
    text = _MULTI_SPACE.sub(" ", text)
    # Normalize 3+ newlines to exactly 2 (one paragraph break).
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def strip_headers_footers(pages: list[str], min_repeats: int = 3) -> list[str]:
    """Remove lines that repeat across >= min_repeats pages.

    Typical case: "Page 5 of 12", journal name, author list as running header.
    Operates per-line (split on \\n) and treats a line as boilerplate if
    it appears on enough pages.
    """
    if len(pages) < min_repeats:
        return pages

    line_counts: Counter[str] = Counter()
    page_lines: list[list[str]] = []
    for page in pages:
        lines = [ln.strip() for ln in page.split("\n") if ln.strip()]
        page_lines.append(lines)
        line_counts.update(lines)

    boilerplate = {
        line for line, count in line_counts.items() if count >= min_repeats
    }

    cleaned: list[str] = []
    for lines in page_lines:
        kept = [ln for ln in lines if ln not in boilerplate]
        cleaned.append("\n".join(kept))
    return cleaned


def clean_text(
    text: str,
    *,
    fix_hyphens: bool = True,
    collapse_ws: bool = True,
    normalize: bool = True,
) -> str:
    """Apply the standard cleaning pipeline to a single string.

    Args:
        text: Raw extracted text.
        fix_hyphens: Repair hyphenated line breaks.
        collapse_ws: Collapse whitespace / preserve paragraphs.
        normalize: Replace ligatures and special whitespace.
    """
    if normalize:
        text = normalize_unicode(text)
    if fix_hyphens:
        text = fix_hyphenation(text)
    if collapse_ws:
        text = collapse_whitespace(text)
    return text


def clean_pages(
    pages: Iterable[tuple[int, str]],
    *,
    strip_repeats_min: int = 3,
) -> list[tuple[int, str]]:
    """Clean a sequence of (page_number, text) tuples.

    First strips cross-page headers/footers, then cleans each page.
    Returns a new list with same order.
    """
    pages_list = list(pages)
    raw_texts = [t for _, t in pages_list]
    stripped = strip_headers_footers(raw_texts, min_repeats=strip_repeats_min)
    out: list[tuple[int, str]] = []
    for (page_no, _), cleaned in zip(pages_list, stripped):
        out.append((page_no, clean_text(cleaned)))
    return out


__all__ = [
    "normalize_unicode",
    "fix_hyphenation",
    "collapse_whitespace",
    "strip_headers_footers",
    "clean_text",
    "clean_pages",
]
