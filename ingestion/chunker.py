"""Section-aware chunking with recursive fallback.

Preferred hierarchy (AGENTS.md §7):
    Abstract -> Introduction -> Background -> Related Work -> Methodology
    -> Architecture -> Experiments -> Results -> Discussion
    -> Limitations -> Conclusion

Strategy:
1. Detect sections by regex on the cleaned page text.
2. Split each section into chunks respecting chunk_size + chunk_overlap.
3. If section detection fails, fall back to `RecursiveCharacterTextSplitter`
    over the entire document.

Each chunk carries `paper_id`, `page`, `section`, `chunk_id` so the RAG
layer can cite evidence (AGENTS.md §24).
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from ingestion.text_cleaner import clean_text

# Section names to look for (case-insensitive). Order matters for tie-breaking.
_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Abstract", re.compile(r"^\s*(?:Abstract|ABSTRACT)\b", re.MULTILINE)),
    ("Introduction", re.compile(r"^\s*(?:\d+\.?\s*)?(?:Introduction|INTRODUCTION)\b", re.MULTILINE)),
    ("Background", re.compile(r"^\s*(?:\d+\.?\s*)?(?:Background|BACKGROUND)\b", re.MULTILINE)),
    ("Related Work", re.compile(r"^\s*(?:\d+\.?\s*)?(?:Related\s+Work|RELATED\s+WORK|Literature\s+Review)\b", re.MULTILINE)),
    ("Methodology", re.compile(r"^\s*(?:\d+\.?\s*)?(?:Methodology|METHODOLOGY|Methods|METHODS)\b", re.MULTILINE)),
    ("Architecture", re.compile(r"^\s*(?:\d+\.?\s*)?(?:Architecture|ARCHITECTURE|System\s+Design)\b", re.MULTILINE)),
    ("Experiments", re.compile(r"^\s*(?:\d+\.?\s*)?(?:Experiments?|EXPERIMENTS?|Experimental\s+(?:Setup|Study|Evaluation))\b", re.MULTILINE)),
    ("Results", re.compile(r"^\s*(?:\d+\.?\s*)?(?:Results|RESULTS)\b", re.MULTILINE)),
    ("Discussion", re.compile(r"^\s*(?:\d+\.?\s*)?(?:Discussion|DISCUSSION)\b", re.MULTILINE)),
    ("Limitations", re.compile(r"^\s*(?:\d+\.?\s*)?(?:Limitations|LIMITATIONS)\b", re.MULTILINE)),
    ("Conclusion", re.compile(r"^\s*(?:\d+\.?\s*)?(?:Conclusions?|CONCLUSIONS?|Summary\s+and\s+Conclusion)\b", re.MULTILINE)),
    ("References", re.compile(r"^\s*(?:References|Bibliography)\b", re.MULTILINE)),
]

# Sections that come BEFORE the first numbered section (Abstract, Introduction).
_EARLY_SECTIONS = {"Abstract", "Introduction"}


@dataclass
class TextChunk:
    """A single chunk with its citation metadata."""

    chunk_id: str
    paper_id: str
    text: str
    page: int | None
    section: str | None


def _detect_sections(text: str) -> list[tuple[str, int, int]]:
    """Find section headers in `text`.

    Returns a list of (section_name, start_offset, end_offset) for each
    detected section. Offsets are character positions into `text`.
    """
    matches: list[tuple[str, int]] = []
    for name, pat in _SECTION_PATTERNS:
        for m in pat.finditer(text):
            matches.append((name, m.start()))

    if not matches:
        return []

    # Sort by position, then keep only the first occurrence of each name
    # (sections like "Related Work" shouldn't match twice).
    matches.sort(key=lambda x: x[1])
    seen: set[str] = set()
    unique: list[tuple[str, int]] = []
    for name, start in matches:
        if name in seen:
            continue
        seen.add(name)
        unique.append((name, start))

    # Compute end offsets (next section's start).
    sections: list[tuple[str, int, int]] = []
    for i, (name, start) in enumerate(unique):
        end = unique[i + 1][1] if i + 1 < len(unique) else len(text)
        sections.append((name, start, end))
    return sections


def _split_into_subchunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split a single section's text into <= chunk_size-char pieces."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def chunk_pages(
    paper_id: str,
    pages: Iterable[tuple[int, str]],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    min_section_hits: int = 2,
) -> list[TextChunk]:
    """Chunk an entire document.

    Args:
        paper_id: Stable identifier used in every chunk's metadata.
        pages: Iterable of (page_number_1_based, cleaned_text).
        chunk_size: Override settings.chunk_size.
        chunk_overlap: Override settings.chunk_overlap.
        min_section_hits: If >= this many sections are detected, use the
            section-aware path; otherwise fall back to whole-doc splitting.

    Returns:
        List of TextChunk with paper_id, page, section, chunk_id set.
    """
    s = get_settings()
    cs = chunk_size or s.chunk_size
    co = chunk_overlap or s.chunk_overlap

    pages_list = [(p, clean_text(t)) for p, t in pages]
    full_text = "\n\n".join(t for _, t in pages_list)

    sections = _detect_sections(full_text)
    if len(sections) >= min_section_hits:
        return _chunk_by_sections(paper_id, pages_list, sections, cs, co)

    # Fallback: split the whole document, then look up which page each
    # chunk came from by walking through the pages.
    return _chunk_whole_doc(paper_id, pages_list, cs, co)


def _chunk_by_sections(
    paper_id: str,
    pages_list: list[tuple[int, str]],
    sections: list[tuple[str, int, int]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Build chunks respecting section boundaries."""
    chunks: list[TextChunk] = []
    # For each section, find which pages its text spans by char offset.
    # We approximate page boundaries as cumulative lengths joined with "\n\n".
    page_offsets: list[int] = [0]
    for _, t in pages_list:
        page_offsets.append(page_offsets[-1] + len(t) + 2)  # +2 for "\n\n"

    def offset_to_page(offset: int) -> int:
        for i in range(len(page_offsets) - 1):
            if page_offsets[i] <= offset < page_offsets[i + 1]:
                return pages_list[i][0]
        return pages_list[-1][0]

    full_text = "\n\n".join(t for _, t in pages_list)
    for section_name, start, end in sections:
        section_text = full_text[start:end]
        # Skip References section — usually not useful for RAG (and huge).
        if section_name == "References":
            continue
        for piece in _split_into_subchunks(section_text, chunk_size, chunk_overlap):
            # The piece's offset within section_text; translate to global.
            # We don't need exact page lookup here; approximate by section's
            # first page is good enough for citation purposes.
            page_no = offset_to_page(start)
            chunks.append(
                TextChunk(
                    chunk_id=f"{paper_id}_chunk_{uuid.uuid4().hex[:8]}",
                    paper_id=paper_id,
                    text=piece,
                    page=page_no,
                    section=section_name,
                )
            )
    return chunks


def _chunk_whole_doc(
    paper_id: str,
    pages_list: list[tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Fallback: chunk the entire document as one stream.

    Page attribution is approximate (the first page that contains the
    chunk's starting character).
    """
    full_text = "\n\n".join(t for _, t in pages_list)
    page_starts: list[int] = [0]
    for _, t in pages_list:
        page_starts.append(page_starts[-1] + len(t) + 2)

    pieces = _split_into_subchunks(full_text, chunk_size, chunk_overlap)
    chunks: list[TextChunk] = []
    cursor = 0
    for piece in pieces:
        # Find the page whose range contains the piece's start.
        page_no = pages_list[-1][0]
        for i in range(len(page_starts) - 1):
            if page_starts[i] <= cursor < page_starts[i + 1]:
                page_no = pages_list[i][0]
                break
        chunks.append(
            TextChunk(
                chunk_id=f"{paper_id}_chunk_{uuid.uuid4().hex[:8]}",
                paper_id=paper_id,
                text=piece,
                page=page_no,
                section=None,
            )
        )
        cursor += len(piece) - chunk_overlap
    return chunks


__all__ = ["TextChunk", "chunk_pages"]
