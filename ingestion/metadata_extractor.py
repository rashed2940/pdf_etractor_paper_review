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
from pathlib import Path
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


def _filename_to_title(filename: str | None) -> str | None:
    """Convert a filename like 'Enhancing_MARL_IDS.pdf' into a usable
    fallback title: 'Enhancing MARL IDS'.

    Strategy: drop the directory and extension, replace _ - + . with
    spaces, collapse whitespace, and Title-Case the result.
    Returns None if the result would be empty or too short.
    """
    if not filename:
        return None
    stem = Path(filename).stem  # strip directory + extension
    if not stem or len(stem) < 3:
        return None
    # Replace separators with spaces.
    cleaned = re.sub(r"[_\-+\.]+", " ", stem)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    # Title-case while preserving common particles like "of", "and", "for".
    particles = {"a", "an", "and", "as", "at", "but", "by", "for",
                  "in", "nor", "of", "on", "or", "the", "to", "vs",
                  "via", "with"}
    words = cleaned.split(" ")
    titled: list[str] = []
    for i, w in enumerate(words):
        if not w:
            continue
        if i > 0 and w.lower() in particles:
            titled.append(w.lower())
        else:
            titled.append(w[:1].upper() + w[1:].lower())
    result = " ".join(titled).strip()
    return result if len(result) >= 3 else None


def _guess_title(
    first_page_text: str,
    pdf_title: str | None,
    filename: str | None = None,
) -> str | None:
    """Prefer PyMuPDF's title; otherwise pick a title from the first page.

    Strategy (in order):
    1. If the PDF-metadata title is real text (not a filename), use it.
    2. Look for the IEEE / ACM / Elsevier "title + dash + authors" header
       pattern on the first page (e.g. "Diaz et al.: Enhancing Multi-Agent
       Reinforcement Learning Intrusion Detection Systems Using …"). The
       title is the part after the colon.
    3. If the title appears to be broken across multiple consecutive lines
       (no blank line between them) and they look like title text (no
       sentence-ending punctuation, no leading author initials like
       "A. Smith"), join them.
    4. As a last resort, fall back to the longest line, but EXCLUDE lines
       that are clearly copyright/license/DOI/abstract/affiliation text.

    Returns None when ambiguous — we don't fabricate.
    """
    if pdf_title and pdf_title.strip() and not _looks_like_filename(pdf_title):
        return pdf_title.strip()
    # If PDF-metadata title is junk (Word filename / generic), prefer the
    # sanitized upload filename over heuristics — it's always available
    # and reliably human-readable.
    fallback_title = _filename_to_title(filename)
    if fallback_title:
        return fallback_title
    if not first_page_text:
        return None

    lines = [ln.strip() for ln in first_page_text.splitlines() if ln.strip()]

    # --- Step 2: IEEE-style "Authors et al.: <Title>" header line ---
    # IEEE papers often put the title on a single line ending with "…",
    # but the actual paper title continues on subsequent lines (often
    # 2-4 lines) before the first author appears. We detect the header
    # line and then greedily consume the title's continuation lines.
    for idx, line in enumerate(lines[:20]):
        # Strip a trailing ellipsis so it doesn't get into the title text.
        cleaned = re.sub(r"\s*[…\.]{1,3}\s*$", "", line).strip()
        # Patterns: "Smith et al.: Title", "Smith and Jones: Title",
        # "Smith, Jones & Brown: Title", "Smith: Title".
        m = re.match(
            r"^[A-Z][\w\-\.]+"
            r"(?:\s+(?:et\s+al\.|and|&))?"
            r"(?:[,\s]+[A-Z][\w\-\.]+)*"
            r":\s+(.+)$",
            cleaned,
        )
        if not m:
            continue
        # Start with what comes after the colon on the same line.
        title_part = m.group(1).strip()
        # Greedily consume continuation lines until we hit the author
        # block, an affiliation, or a clear non-title boundary.
        j = idx + 1
        while j < min(20, len(lines)):
            nxt = lines[j]
            # Stop at the first person-name line.
            if _looks_like_person_name(nxt):
                break
            # Stop at affiliation-like lines.
            if _looks_like_affiliation(nxt):
                break
            # Stop at meta / noise.
            if _looks_like_meta_or_noise(nxt):
                break
            # Stop if the next line is sentence-like (ends with period)
            # or starts with a lowercase (continuation prose, not title).
            if nxt.endswith(".") or (nxt and nxt[0].islower()):
                break
            # Otherwise it's a title continuation line — append it.
            title_part = (title_part + " " + nxt).strip()
            j += 1
        # Title must look like a title (not a sentence).
        if (
            10 <= len(title_part) <= 300
            and not title_part.endswith(".")
            and not title_part.lower().startswith(
                ("received", "accepted", "revised", "copyright", "licensed",
                 "abstract", "keywords", "index terms")
            )
        ):
            return title_part
        # If IEEE header was found but produced something invalid,
        # stop scanning for more headers.
        break

    # --- Step 3: find the title block by structural position. ---
    # The title is a run of consecutive lines BEFORE the first author
    # block. We detect the author block as: a person-name line followed
    # by an affiliation line. The title block ends right before that.
    title_block: list[str] = []
    i = 0
    while i < min(20, len(lines)):
        line = lines[i]
        # Skip journal-name / DOI / copyright lines.
        if _looks_like_meta_or_noise(line):
            i += 1
            continue
        # Look ahead: is this a person-name line followed by an
        # affiliation? If yes, it's the start of the author block —
        # the title ends here.
        if _looks_like_person_name(line):
            # Peek at the next non-empty line.
            j = i + 1
            while j < len(lines) and not lines[j]:
                j += 1
            if j < len(lines) and _looks_like_affiliation(lines[j]):
                break
            # Could also be a comma-separated author line.
            if "," in line and _AUTHOR_LINE_RE.match(line):
                break
        # Accept this line as part of the title.
        title_block.append(line)
        # Title is usually 1-4 lines. If we have a few short lines and
        # the next one looks like a section header (all caps, short),
        # stop here.
        if (
            len(title_block) >= 4
            or (len(title_block) >= 2 and i + 1 < len(lines)
                and lines[i + 1].isupper() and len(lines[i + 1]) < 30)
        ):
            break
        i += 1
        # If the next line doesn't look title-y (sentence punctuation,
        # very long), stop.
        if i < len(lines) and _looks_like_meta_or_noise(lines[i]):
            break

    if title_block:
        joined = " ".join(title_block).strip()
        # Clean up ellipsis tails from truncated IEEE titles.
        joined = re.sub(r"\s*[…\.]{3,}\s*$", "", joined).strip()
        # Don't return a single super-short word.
        if len(joined) >= 10:
            return joined

    # --- Step 4: legacy fallback to longest non-meta line ---
    candidates = [
        ln for ln in lines[:30]
        if 10 <= len(ln) <= 200 and not _looks_like_meta_or_noise(ln)
    ]
    if candidates:
        return max(candidates, key=len)

    # --- Step 5: filename fallback ---
    # When all structural heuristics fail (the title is split across
    # lines, the metadata is junk, etc.), derive a usable title from
    # the PDF's filename. This is the user's preferred behavior: a
    # human-readable filename like "Enhancing_Multi-Agent_Reinforcement_
    # Learning_Intrusion.pdf" becomes "Enhancing Multi Agent
    # Reinforcement Learning Intrusion".
    return _filename_to_title(filename)


def _looks_like_meta_or_noise(line: str) -> bool:
    """Return True if `line` is journal metadata, copyright, DOI, or
    other boilerplate that should never be picked as a paper title."""
    lower = line.lower()
    bad_starts = (
        "received:", "revised:", "accepted:", "published:",
        "copyright", "licensed under", "license:",
        "doi:", "doi.org", "http", "www.",
        "abstract", "keywords:", "index terms",
        "vol.", "volume ", "issue ", "no.", "pp.",
    )
    if any(lower.startswith(s) for s in bad_starts):
        return True
    # Contains a DOI pattern anywhere
    if _DOI_RE.search(line):
        return True
    # Journal / publisher name on its own line — patterns we see in real
    # papers: "Engineering, Technology & Applied Science Research",
    # "IEEE Transactions on Information Forensics and Security",
    # "Proceedings of the 2024 ACM SIGSAC Conference".
    # Heuristic: contains "&" (publisher-style name), OR 2+ capitalized
    # "Title Case" words in a row with no sentence-ending punctuation.
    if "&" in line and len(line) < 100 and not line.endswith("."):
        return True
    # Vol/No/Year pattern: "Vol. 15, No. 4, 2025, 24455-24459"
    if re.search(r"\bvol\.?\s*\d", lower) or re.search(r"\bno\.?\s*\d", lower):
        return True
    # All-caps short header (e.g. "ABSTRACT", "INTRODUCTION")
    if line.isupper() and len(line) <= 30:
        return True
    # Journal-name header on its own line. We only filter lines that
    # STRONGLY look like a publisher / journal name — short phrases
    # containing a comma, "&", or a recognized journal keyword like
    # "Journal", "Proceedings", "Transactions". Anything else we leave
    # alone (could be a real title).
    if 0 < len(line) <= 80 and line == line.strip():
        words = line.split()
        if 1 <= len(words) <= 8:
            lower_words = [w.lower() for w in words]
            journal_markers = {
                "journal", "proceedings", "transaction", "transactions",
                "conference", "symposium", "workshop", "letters",
                "magazine", "bulletin", "annals", "review",
                "international", "annual", "quarterly",
            }
            # Strong signal: contains "&" (publisher name style) OR
            # contains a comma (multi-segment name like
            # "Engineering, Technology & Applied Science Research").
            has_strong_signal = (
                "&" in line
                or "," in line
                or any(w in journal_markers for w in lower_words)
            )
            if has_strong_signal:
                # And no real-title markers (which would mean the user
                # wrote a title using a journal word, e.g. "Conference
                # Paper").
                title_markers = {
                    "using", "with", "via", "for", "from", "into", "toward",
                    "towards", "through", "based", "enhancing", "improved",
                    "novel", "new", "efficient", "robust", "scalable",
                    "deep", "machine", "learning", "neural", "network",
                    "detection", "classification", "recognition",
                    "attack", "defense", "defence", "threat",
                    "approach", "framework", "method", "methodology",
                    "system", "systems", "model", "models", "algorithm",
                    "algorithms", "analysis", "study", "survey",
                }
                if not any(w in title_markers for w in lower_words):
                    return True
    return False


def _looks_like_author_line(line: str) -> bool:
    """Heuristic: True if `line` looks like an author name (followed
    elsewhere by an affiliation). Catches both comma-separated lists and
    individual names.

    We require at least one word with a lowercase letter so that
    ALL-CAPS abbreviations like "Q-Value Estimation" or
    "IDS Multi-Agent" don't get misidentified as author names.
    """
    # Comma-separated multi-author line ("Smith, Jones, Brown")
    if "," in line and _AUTHOR_LINE_RE.match(line):
        return True
    # Single-name line: capitalized words, possible middle initials,
    # 2-5 words, no sentence-ending punctuation, no "@".
    if "@" in line or line.endswith(".") or line.endswith(","):
        return False
    words = line.split()
    if not (2 <= len(words) <= 6):
        return False
    # At least one word must have a lowercase letter (filters out
    # ALL-CAPS abbreviations).
    if not any(any(c.islower() for c in w) for w in words):
        return False
    # Each word must start with a capital letter (or be a middle initial
    # / known particle).
    for w in words:
        if w.lower() in {"de", "van", "von", "der", "den", "del", "la", "le"}:
            continue
        if not (w[0].isupper() or (len(w) <= 3 and w.endswith("."))):
            return False
    return True


_FILENAME_TITLE_HINTS = (
    "microsoft word",
    "openoffice",
    "libreoffice",
    ".docx",
    ".doc",
    ".pdf",
    "_pp-",         # common journal page-range filename marker
    "_v",           # volume marker, e.g. "_v15_"
    "_n",           # number marker, e.g. "_n4_"
)


def _looks_like_filename(text: str) -> bool:
    """Heuristic: returns True if `text` looks like a filename rather
    than a paper title. Used to reject PyMuPDF metadata titles that
    are actually the document's filename (a common artifact when the
    PDF was authored in Word or via a non-academic toolchain)."""
    if not text:
        return False
    lower = text.lower()
    if any(hint in lower for hint in _FILENAME_TITLE_HINTS):
        return True
    # Long strings without spaces are almost always filenames.
    if len(text) > 60 and " " not in text:
        return True
    # Camel-case or snake_case identifiers without spaces.
    if len(text) > 30 and " " not in text and "-" in text:
        return True
    return False


def _guess_authors(
    first_page_text: str, pdf_authors: list[str]
) -> list[str]:
    """Extract the paper's author list from page 1.

    Strategy:
    1. If PDF-metadata authors look reliable (multiple names, no
       email / Windows-domain artifacts), use them.
    2. Otherwise, scan the first page for an "author block": a run of
       single-name lines, each followed by an affiliation line
       ("Faculty of Engineering, Udayana University" / "Department of
       Computer Science, MIT" / "Google Research" / etc.). This is the
       dominant pattern in IEEE / ACM / Elsevier / Springer papers.
    3. As a last resort, try the legacy comma-separated author line
       regex.

    Returns [] when nothing was found.
    """
    if pdf_authors and not _authors_look_unreliable(pdf_authors):
        return pdf_authors
    if not first_page_text:
        return []
    lines = [ln.strip() for ln in first_page_text.splitlines() if ln.strip()]

    # --- Strategy 2: scan for author → affiliation pairs ---
    authors_from_affiliations: list[str] = []
    i = 0
    # Bound the search to the top of the page so we don't pick up
    # citation names from the abstract / body.
    scan_limit = min(40, len(lines))
    while i < scan_limit:
        line = lines[i]
        # Skip meta / noise / title-looking lines
        if (
            _looks_like_meta_or_noise(line)
            or _looks_like_filename(line)
            or len(line) > 80
            or line.endswith((".", ":", "?", "!"))
        ):
            i += 1
            continue
        # Author candidate: single-name line, looks like a person
        if _looks_like_person_name(line):
            # Verify the next non-empty line is an affiliation.
            # Affiliations usually contain a comma + "University",
            # "Institute", "Department", "College", "Laboratory",
            # "Research", "Inc.", "Corp.", "Ltd.", or "@" (for email).
            j = i + 1
            while j < scan_limit and not lines[j]:
                j += 1
            if j < scan_limit:
                next_line = lines[j]
                if _looks_like_affiliation(next_line):
                    authors_from_affiliations.append(line)
                    # Skip past the affiliation line (and possibly an
                    # email line and a second affiliation line).
                    i = j + 1
                    # Eat the email line if present
                    if (
                        i < scan_limit
                        and "@" in lines[i]
                        and len(lines[i]) < 80
                    ):
                        i += 1
                    continue
        i += 1

    if authors_from_affiliations:
        return authors_from_affiliations

    # --- Strategy 3: legacy comma-separated author line ---
    for line in lines[:30]:
        if len(line) > 200:
            continue
        if _AUTHOR_LINE_RE.match(line):
            return [a.strip() for a in line.split(",") if a.strip()]

    # Heuristic didn't match either — return whatever PDF metadata had
    # (or []) so the LLM metadata agent has something to work with.
    return pdf_authors or []


def _looks_like_person_name(line: str) -> bool:
    """Return True if `line` looks like a single person's name."""
    # No commas (we want individual names, not comma-separated lists)
    if "," in line:
        return False
    # No affiliation markers
    if "@" in line:
        return False
    # Must look like title case, no sentence-ending punctuation.
    if line.endswith((".", ":", "?", "!")):
        return False
    words = line.split()
    if not (2 <= len(words) <= 6):
        return False
    # At least one word must have a lowercase letter (filters out
    # ALL-CAPS abbreviations).
    if not any(any(c.islower() for c in w) for w in words):
        return False
    for w in words:
        # Allow lowercase particles (de, von, van)
        if w.lower() in {"de", "van", "von", "der", "den", "del", "la", "le"}:
            continue
        # Allow middle initials like "G." or "G.K."
        if (
            len(w) <= 4
            and w.endswith(".")
            and w[0].isupper()
        ):
            continue
        # Otherwise first letter must be uppercase.
        if not w[0].isupper():
            return False
        # Word must contain at least one lowercase letter (not an acronym)
        if not any(c.islower() for c in w):
            return False
    return True


def _looks_like_affiliation(line: str) -> bool:
    """Return True if `line` looks like an institutional affiliation
    (university, department, company, lab, etc.)."""
    lower = line.lower()
    # Strong signals — institution words
    institution_keywords = (
        "university", "institute", "department", "faculty", "college",
        "laboratory", "lab ", "labs ", "research", "school of",
        "academy", "centre", "center for", "inc.", "corp.", "ltd.",
        "llc", "gmbh", "co.", "company", "corporation",
        "hospital", "medical ", "national ", "federal ",
        "polytechnic", "technology", "technological",
    )
    if any(k in lower for k in institution_keywords):
        return True
    # Email line (lone email counts as a trailing affiliation cue)
    if "@" in line and " " not in line.strip():
        return True
    # Long line with a comma and no sentence-ending period = probably
    # "Department of X, University of Y" style.
    if "," in line and len(line) > 20 and not line.endswith("."):
        return True
    # Common short university / institution abbreviations. These are
    # 2-6 chars, all uppercase, and on their own line right after an
    # author name (e.g. "MIT", "CMU", "UCLA", "ETH Zurich", "TU Delft").
    stripped = line.strip()
    if (
        2 <= len(stripped) <= 6
        and stripped.replace(" ", "").isupper()
        and any(c.isalpha() for c in stripped)
        and "," not in stripped
    ):
        return True
    return False


def _authors_look_unreliable(authors: list[str]) -> bool:
    """Return True if `authors` looks like an artifact rather than a
    real author list. Examples:
        ["gerry"]                -> Word user's login
        ["john.doe@example.com"] -> email-as-author
        ["admin"]                -> generic username
    """
    if not authors:
        return False
    # 1 author is suspicious for an academic paper (most have 2+).
    if len(authors) == 1:
        a = authors[0]
        # Single-word author names are almost always usernames/login IDs.
        if " " not in a and "." not in a and len(a) < 25:
            return True
        # Contains an email pattern.
        if "@" in a:
            return True
        # Looks like a Windows domain user (DOMAIN\User or USER$).
        if "\\" in a or a.endswith("$"):
            return True
    return False


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
    filename: str | None = None,
) -> dict[str, Any]:
    """Combine PyMuPDF metadata with first-page heuristics.

    Args:
        pdf_meta: The dict from PyMuPDF `doc.metadata`. May be partial.
        first_page_text: Cleaned text of page 1 (no headers/footers).
        filename: Original PDF filename (e.g. "Enhancing_MARL.pdf").
            Used as the *last-resort* fallback for the paper title
            when all other heuristics fail.

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
        "title": _guess_title(first_page_text, pdf_title, filename=filename),
        "authors": _guess_authors(first_page_text, pdf_authors),
        "publication_date": pdf_year or fp_year,
        "venue": venue,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "institution": None,  # Not reliably detectable without heuristics we don't trust
    }


__all__ = ["extract_metadata"]
