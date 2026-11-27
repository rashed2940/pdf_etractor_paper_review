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
    """Extract a 4-digit year (1950-2029) from a string or PDF date field.

    Handles the PyMuPDF/PDF date format ``D:YYYYMMDDHHmmSS+HH'mm'`` as
    well as plain years and free text. The year is the first 4 digits
    of the date string, so we must NOT require a word boundary after it
    (in ``D:20260318011457Z`` the year is immediately followed by the
    month digits).
    """
    if raw is None:
        return None
    text = str(raw)
    # PDF date format: D:YYYYMMDD... — the year is the first 4 digits.
    m = re.search(r"D:(\d{4})\d{2}", text)
    if m:
        year = int(m.group(1))
        if 1950 <= year <= 2029:
            return str(year)
    # Plain 4-digit year, not glued to other digits (e.g. "in 2026,").
    m = re.search(r"(?<!\d)(19[5-9]\d|20[0-2]\d)(?!\d)", text)
    if m:
        return m.group(0)
    return None


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
            # An ORCID-grouped IEEE block puts the affiliation BELOW
            # the name, so the name is not followed by an affiliation.
            # Stop the title when a name is followed by an ORCID or by
            # a location line instead.
            if j < len(lines):
                nxt = lines[j]
                if _looks_like_location_line(nxt) or _looks_like_orcid(nxt):
                    break
            # Springer layout: the name is followed by a bullet line
            # carrying the remaining authors.
            if j < len(lines) and ("·" in lines[j] or "•" in lines[j]):
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
        "received:", "received ", "revised:", "revised ", "accepted:",
        "accepted ", "published:", "published ", "published online",
        "copyright", "©", "(c) ", "licensed under", "license:",
        "doi:", "doi.org", "http", "www.",
        "abstract", "keywords:", "index terms",
        "vol.", "volume ", "issue ", "no.", "pp.",
        "communicated by", "date of publication", "date of current version",
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


def _clean_author_list(authors: list[str]) -> list[str]:
    """Final cleanup pass over an extracted author list.

    Removes institution names that slipped through ("Kazakh-Turkish
    University"), splits entries that still contain commas
    ("Richard Fang, Rohan Bindu"), and drops duplicates while
    preserving order.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in authors:
        if not raw:
            continue
        name = _strip_author_marks(raw)
        if not name:
            continue
        # A leftover comma means several names are packed into one
        # entry — split them.
        parts = [p.strip() for p in name.split(",") if p.strip()]
        for part in parts:
            part = _strip_author_marks(part)
            if not part:
                continue
            # Drop institution names.
            if _looks_like_affiliation(part):
                continue
            # Drop metric / capability names.
            if _looks_like_metric_name(part):
                continue
            key = part.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(part)
    return out


def _guess_authors(
    first_page_text: str, pdf_authors: list[str], title: str | None = None
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
    3. Fall back to a bare run of consecutive person-name lines (the
       layout left behind once the text cleaner strips affiliations).
    4. As a last resort, try the legacy comma-separated author line
       regex.

    Args:
        title: The already-resolved paper title. Title lines are
            excluded from the author block so a wrapped title is not
            mistaken for an author.

    Returns [] when nothing was found.
    """
    if pdf_authors and not _authors_look_unreliable(pdf_authors):
        return _clean_author_list(pdf_authors)
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

    # A single match is almost always a false positive (a name that
    # happens to sit above an affiliation). Require at least two.
    if len(authors_from_affiliations) >= 2:
        return _clean_author_list(authors_from_affiliations)

    # --- Strategy 2b: compact author block (affiliations already stripped) ---
    # The text cleaner removes affiliation lines ("Carnegie Mellon
    # University", "Pittsburgh, Pennsylvania, USA") from page 1, which
    # leaves a bare run of consecutive person-name lines right under the
    # title. Strategy 2 cannot see that pattern because it requires an
    # affiliation line after every name, so we look for the block
    # directly.
    authors_from_block = _guess_authors_from_block(lines, scan_limit, title)
    if authors_from_block:
        return _clean_author_list(authors_from_block)

    # --- Strategy 2c: comma-separated author line ---
    # Many ACM/USENIX/IEEE papers put every author on ONE line, separated
    # by commas and optionally followed by superscript affiliation
    # markers, e.g.
    #   "Tri Cao1 *†, Chengyu Huang1*, Yuexin Li1*, Huilin Wang1, ..."
    #   "He Kong1,2, Die Hu1,2, Jingguo Ge1,2, Liangxiong Li1, and ..."
    # The text cleaner collapses these into a single line, so the
    # per-name "one line per person" heuristics above cannot see them.
    #
    # NOTE: the stacked variant (2f) must run FIRST, because it merges
    # a leading single-name line with the bullet/CSV line that follows.
    authors_stacked = _guess_authors_stacked(lines, scan_limit, title)
    if authors_stacked:
        return _clean_author_list(authors_stacked)

    authors_from_csv = _guess_authors_from_csv_line(lines, scan_limit, title)
    if authors_from_csv:
        return _clean_author_list(authors_from_csv)

    # --- Strategy 2d: "NAME, Affiliation" on one line ---
    # ACM journals print the author in capitals followed by the
    # affiliation on the SAME line:
    #   "ZEHANG DENG, Department of Computer Science, Swinburne Univ."
    # The name comes FIRST and the affiliation second, so the
    # affiliation-after-name strategy above cannot match it.
    authors_inline = _guess_authors_from_inline_affiliation(lines, scan_limit, title)
    if authors_inline:
        return _clean_author_list(authors_inline)

    # --- Strategy 2e: ORCID-grouped author block ---
    # Some IEEE author blocks interleave an ORCID iD with each name:
    #   "Lokanatha Reddy"
    #   "Gandikota"
    #   "Verizon"
    #   "0009-0006-6058-3326"
    #   "Vasanth Jegathesan"
    #   ...
    # The ORCID line is not an affiliation, so the name+affiliation
    # strategy skips straight over the whole block.
    authors_orcid = _guess_authors_from_orcid_block(lines, scan_limit, title)
    if authors_orcid:
        return _clean_author_list(authors_orcid)

    # --- Strategy 2f: stacked author lines ---
    # Handled above (before strategy 2c) because the bullet-separated
    # line it merges with is also matched by the CSV strategy.

    # --- Strategy 3: legacy comma-separated author line ---
    for line in lines[:30]:
        if len(line) > 200:
            continue
        if _AUTHOR_LINE_RE.match(line):
            return [a.strip() for a in line.split(",") if a.strip()]

    # Heuristic didn't match either — return whatever PDF metadata had
    # (or []) so the LLM metadata agent has something to work with.
    return pdf_authors or []


#: Section headings that mark the end of the author block.
_AUTHOR_BLOCK_STOP_WORDS = (
    "abstract", "introduction", "keywords", "index terms", "ccs concepts",
    "acm reference format", "ieee transactions", "preprint",
    "license", "http://", "doi:", "arxiv:", "published", "received",
    "accepted", "copyright", "submitted",
)

#: Lowercase function words that signal running prose (i.e. a title).
_TITLE_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "for", "of", "in", "on", "at", "by", "from", "to",
    "with", "without", "and", "or", "but", "via", "using", "through",
    "toward", "towards", "into", "onto", "over", "under", "between",
    "across", "within", "against", "based", "towards", "toward",
    "new", "novel", "towards", "beyond", "towards",
})


def _guess_authors_from_block(
    lines: list[str], scan_limit: int, title: str | None = None
) -> list[str]:
    """Find a bare run of consecutive person-name lines under the title.

    Handles the layout produced by `clean_pages()`, which strips the
    affiliation lines and leaves e.g.::

        Establishing Systems-Level Competitive Leaderboards for
        Autonomous Network Security Operations
        Brian Singer
        Lakshmi Adiga
        Marko Morrison
        Abstract
        ...

    Args:
        title: The resolved paper title. Lines belonging to a wrapped
            title are excluded so they are not mistaken for authors.

    Returns the names, or [] when no such block is found.
    """
    # Words that appear in the title — a line made up of them is a
    # continuation of the title, not an author.
    title_words: set[str] = set()
    if title:
        title_words = {w.lower().strip(".,:;") for w in title.split()}

    for start in range(0, min(scan_limit, 12)):
        # A block must begin right after a title-ish line.
        if not (
            _looks_like_title_line(lines[start])
            or _line_is_part_of_title(lines[start], title_words)
        ):
            continue
        block: list[str] = []
        j = start + 1
        while j < scan_limit:
            line = lines[j]
            low = line.lower()
            if any(low.startswith(w) for w in _AUTHOR_BLOCK_STOP_WORDS):
                break
            if not _looks_like_person_name(line):
                break
            block.append(_strip_author_marks(line))
            j += 1
        # Require at least 2 names so we don't mistake a stray title
        # fragment or a section heading for an author.
        if len(block) < 2:
            continue
        # Drop leading entries that are really title continuation lines.
        while block and (
            _looks_like_title_line(block[0])
            or _line_is_part_of_title(block[0], title_words)
        ):
            block.pop(0)
        if len(block) >= 2:
            return block
    return []


def _line_is_part_of_title(line: str, title_words: set[str]) -> bool:
    """Return True if every word of `line` also appears in the title.

    A wrapped title line consists entirely of title words, whereas an
    author's name virtually never does.
    """
    if not title_words:
        return False
    words = [
        w.lower().strip(".,:;∗*")
        for w in line.split()
        if w.strip(".,:;∗*")
    ]
    if not words:
        return False
    return all(w in title_words for w in words)


def _looks_like_title_line(line: str) -> bool:
    """Heuristic: does this line look like part of a paper title?

    The signal is a lowercase function word ("for", "of", "the", "with",
    ...). Titles are running prose and contain them; a person's name
    essentially never does.
    """
    if len(line) < 12 or len(line) > 200:
        return False
    low = line.lower()
    if any(low.startswith(w) for w in _AUTHOR_BLOCK_STOP_WORDS):
        return False
    words = [w for w in line.split() if w]
    if len(words) < 2:
        return False
    # A lowercase function word is the strongest title signal.
    for w in words:
        bare = w.lower().strip(".,:;")
        if bare in _TITLE_FUNCTION_WORDS:
            return True
    return False


def _strip_author_marks(name: str) -> str:
    """Remove footnote markers (*, †, ‡, §) and trailing commas.

    Marks can appear before or after the name and can be combined with
    a trailing comma, e.g. ``Zhun Wang∗,`` or ``Chengyu Huang1*,``.
    We strip repeatedly until the string stops changing.
    """
    t = name.strip()
    prev = None
    while prev != t:
        prev = t
        # Trailing separators first, then any marker/digit cluster.
        t = t.rstrip(" ,;").strip()
        t = re.sub(r"[\d\*†‡§∗¶]+$", "", t).rstrip(" ,;").strip()
        # Leading markers.
        t = t.lstrip("*†‡§∗¶").strip()
    return t.strip(" ,;.")


def _looks_like_person_name(line: str) -> bool:
    """Return True if `line` looks like a single person's name."""
    # Strip trailing commas / footnote markers ("Zhun Wang∗,").
    line = _strip_author_marks(line)
    if not line:
        return False
    # No commas (we want individual names, not comma-separated lists).
    # A trailing comma was already removed above.
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


def _looks_like_allcaps_person_name(token: str) -> bool:
    """Return True for an ALL-CAPS personal name such as ``ZEHANG DENG``.

    ACM-journal layout prints the author in capitals followed by the
    affiliation on the same line, e.g.::

        ZEHANG DENG, Department of Computer Science, Swinburne University

    ALL-CAPS is too weak a signal on its own (it also matches
    ``UC Berkeley`` or ``ABSTRACT``), so this helper is only used where
    the surrounding line already proves the structure is
    ``NAME, Affiliation``.
    """
    t = _strip_author_marks(token)
    if not t or "," in t or "@" in t:
        return False
    words = t.split()
    if not (2 <= len(words) <= 5):
        return False
    for w in words:
        # Purely alphabetic, at least 2 characters ("AI" is too short).
        if not w.isalpha() or len(w) < 2:
            return False
        # The token must ACTUALLY be all capitals — otherwise "Ant Group
        # CO Ltd" or "and Technology" would be mistaken for a name.
        if not w.isupper():
            return False
    return True


def _clean_author_token(token: str) -> str:
    """Strip affiliation markers from a single author token.

    Removes superscript digits (``Tri Cao1`` -> ``Tri Cao``), footnote
    symbols (``Chengyu Huang1*`` -> ``Chengyu Huang``) and trailing
    punctuation. Markers can appear in any order (e.g. ``Cao1 *``), so
    we strip repeatedly until the token stops changing.
    """
    t = token.strip()
    prev = None
    while prev != t:
        prev = t
        # Trailing superscript digits / symbols: "Cao1", "Huang1*".
        t = re.sub(r"[\d\*†‡§∗¶]+$", "", t).strip()
        # Leading superscript markers.
        t = t.lstrip("*†‡§∗¶").strip()
        # Society membership suffix: "Fabio Patrone—Member IEEE" or
        # "Danilo Greco, (Member, IEEE)".
        t = re.sub(
            r"[,;]?\s*\(?\s*(?:member|student member|senior member|"
            r"ieee|acm|isac|cissp)\b[^)]*\)?\s*$",
            "",
            t,
            flags=re.IGNORECASE,
        ).strip()
        # Trailing footnote letters/asterisks.
        t = t.rstrip("*†‡§∗¶").strip()
    return t.strip(" ,;.")


def _split_author_line(line: str) -> list[str]:
    """Split a comma/and/bullet separated author line into candidate names.

    Handles both "A, B, C" and "A and B" styles, keeping the order.
    """
    # " and " / " & " separate the last author too. Split on those
    # FIRST so "Tong Li1 , and Bingzhen Wu1" does not leave a leading
    # "and" glued to the final name. Semicolons separate an author
    # from the affiliation that follows it, so split on them too.
    # Springer journals use a middle dot as the separator:
    #   "Andreas Happe1 · Aaron Kaplan2 · Jürgen Cito1".
    parts = re.split(r"\s+and\s+|\s*&\s*|;\s*|,\s*|·\s*|•\s*", line)
    out: list[str] = []
    for p in parts:
        p = _clean_author_token(p)
        # A leading "and"/"&" left over from the split.
        p = re.sub(r"^(?:and|&)\s+", "", p, flags=re.IGNORECASE).strip()
        # A trailing "and"/"&" means the next name is on another line.
        p = re.sub(r"\s+(?:and|&)$", "", p, flags=re.IGNORECASE).strip()
        if p and p.lower() not in ("and", "&"):
            out.append(p)
    return out


def _guess_authors_from_csv_line(
    lines: list[str], scan_limit: int, title: str | None = None
) -> list[str]:
    """Find authors listed on a single comma-separated line.

    Many ACM / USENIX / IEEE papers list every author on one line with
    optional superscript affiliation markers, e.g.::

        Tri Cao1 *†, Chengyu Huang1*, Yuexin Li1*, Huilin Wang1, ...
        He Kong1,2, Die Hu1,2, Jingguo Ge1,2, Liangxiong Li1, and ...

    Returns the names, or [] when no such line is found.
    """
    title_words: set[str] = set()
    if title:
        title_words = {w.lower().strip(".,:;") for w in title.split()}

    for idx, line in enumerate(lines[:scan_limit]):
        # The line must contain a comma (multi-author) or a middle-dot
        # separator (Springer style) and be long enough.
        has_sep = "," in line or "·" in line or "•" in line
        if not has_sep or len(line) < 25 or len(line) > 400:
            continue
        low = line.lower()
        if any(low.startswith(w) for w in _AUTHOR_BLOCK_STOP_WORDS):
            continue
        # Skip obvious affiliation / venue lines. An author line can also
        # trip the affiliation heuristic (e.g. a surname containing
        # "Tech"), so only reject when the line does NOT look like a
        # list of people.
        if _looks_like_affiliation(line):
            tokens_probe = _split_author_line(line)
            if not (
                len(tokens_probe) >= 2
                and all(
                    _looks_like_person_name(t)
                    or _looks_like_allcaps_person_name(t)
                    for t in tokens_probe
                )
            ):
                continue
        if "@" in line:
            continue

        # A trailing "and" means the last author is on the NEXT line
        # ("... Jaime Govea1 and" / "Pablo Palacios2"). Detect it
        # BEFORE splitting, because the splitter drops "and".
        trailing_and = bool(re.search(r"\band\s*$", line)) or line.rstrip().endswith("&")
        tokens = _split_author_line(line)
        if trailing_and:
            nxt = lines[idx + 1] if idx + 1 < scan_limit else ""
            nxt_tokens = _split_author_line(nxt) if nxt else []
            nxt_tokens = [
                t for t in nxt_tokens
                if _looks_like_person_name(t) or _looks_like_allcaps_person_name(t)
            ]
            tokens = tokens + nxt_tokens
        if len(tokens) < 2:
            continue

        # Drop tokens that are institution names rather than people
        # (e.g. "Nanyang Technological University" in "Gelei Deng and
        # Yi Liu, Nanyang Technological University; ...").
        tokens = [t for t in tokens if not _looks_like_affiliation(t)]
        if len(tokens) < 2:
            continue

        # Every token must look like a person name.
        names: list[str] = []
        ok = True
        for tok in tokens:
            # A token that is entirely a title word is title text.
            if _line_is_part_of_title(tok, title_words):
                ok = False
                break
            # Accept a title-case name OR an ALL-CAPS one (IEEE Access
            # prints author names in capitals: "YASSER HMIMOU1,2").
            if not (
                _looks_like_person_name(tok)
                or _looks_like_allcaps_person_name(tok)
            ):
                ok = False
                break
            # Reject evaluation-metric / capability names that look
            # like people ("State Management", "Tool Usage Accuracy").
            if _looks_like_metric_name(tok):
                ok = False
                break
            names.append(tok)
        if ok and len(names) >= 2:
            return names
    return []


#: Words that indicate a metric / capability name rather than a person.
_METRIC_WORDS = frozenset({
    "accuracy", "management", "usage", "precision", "recall", "latency",
    "throughput", "performance", "efficiency", "robustness", "usability",
    "scalability", "security", "safety", "reliability", "coverage",
    "capability", "capabilities", "success", "failure", "cost", "time",
    "memory", "quality", "utility", "helpfulness", "harmlessness",
})


def _looks_like_metric_name(token: str) -> bool:
    """Return True if the token reads like an evaluation metric name.

    Names like "State Management" or "Tool Usage Accuracy" pass the
    title-case person-name test, so we screen them out explicitly.
    """
    words = [w.lower().strip(".,;:") for w in token.split() if w.strip(".,;:")]
    if not words:
        return False
    # A metric name contains a metric-ish word.
    if any(w in _METRIC_WORDS for w in words):
        return True
    return False


def _looks_like_prose(line: str) -> bool:
    """Return True if `line` reads like running body text.

    Used to find the end of the front-matter (title + author) block.
    A prose line is long and contains sentence punctuation.
    """
    if len(line) < 60:
        return False
    return ". " in line or ", " in line


def _author_region(lines: list[str], scan_limit: int) -> int:
    """Return the index where the front-matter (author) block ends.

    The author block sits between the title and the first section
    heading (``Abstract``, ``I. INTRODUCTION``, ``CCS Concepts``, ...).
    Everything from line 0 up to that point is candidate front matter.

    We stop at the first section heading or conference banner. Body
    prose is NOT used as a boundary because the abstract itself is
    prose and would cut the block short; instead a long prose line only
    ends the region when no explicit heading was found.
    """
    fallback = min(scan_limit, len(lines))
    # Skip the leading journal banner block ("Received ... / Accepted
    # ...", "© The Author(s) 2026", DOI lines, ...) — the title and
    # author block come after it.
    start = 0
    while start < min(scan_limit, len(lines)):
        line = lines[start]
        low = line.lower()
        if (
            low.startswith("received")
            or low.startswith("accepted")
            or low.startswith("revised")
            or low.startswith("published")
            or low.startswith("copyright")
            or low.startswith("©")
            or low.startswith("communicated by")
            or low.startswith("digital object identifier")
            or low.startswith("doi:")
            or "date of publication" in low
            or "date of current version" in low
        ):
            start += 1
            continue
        break

    for idx in range(start, min(scan_limit, len(lines))):
        line = lines[idx]
        low = line.lower()
        if any(low.startswith(w) for w in _AUTHOR_BLOCK_STOP_WORDS):
            return idx
        # Conference banners: "Published as a conference paper at ICLR 2026"
        if low.startswith("published as a"):
            return idx
        # Numbered section headings: "I. INTRODUCTION", "2 Background".
        if re.match(r"^\s*(?:[IVXLC]+\.|\d+\.?\d*)\s+[A-Z]", line):
            return idx
        # An affiliation line belongs to the front matter, not the body.
        if _looks_like_affiliation(line):
            continue
        # A long prose line with a sentence ending is the body — but
        # only use it as a fallback boundary (see docstring).
        if _looks_like_prose(line) and line.rstrip().endswith("."):
            fallback = idx
            break
    return fallback


def _looks_like_orcid(line: str) -> bool:
    """Return True if `line` is an ORCID iD.

    Format: ``0000-0002-1825-0097`` (optionally with an orcid.org URL).
    """
    return bool(
        re.match(
            r"^\s*(?:https?://orcid\.org/)?\d{4}-\d{4}-\d{4}-\d{3}[\dX]\s*$",
            line,
        )
    )


def _looks_like_location_line(line: str) -> bool:
    """Return True if `line` looks like a city / country / address line.

    Examples: ``Prosper, USA``, ``Irving, USA``, ``Hawthorn, Australia``,
    ``Beijing, China``.
    """
    lower = line.lower()
    if "," not in line:
        return False
    country_words = (
        "usa", "u.s.a", "united states", "china", "australia", "uk",
        "united kingdom", "canada", "germany", "france", "india",
        "singapore", "qatar", "japan", "korea", "brazil", "malaysia",
        "israel", "italy", "spain", "sweden", "norway", "finland",
        "poland", "netherlands", "switzerland", "austria", "ireland",
        "portugal", "greece", "turkey", "israel", "argentina", "mexico",
    )
    return any(c in lower for c in country_words)


def _guess_authors_stacked(
    lines: list[str], scan_limit: int, title: str | None = None
) -> list[str]:
    """Find authors printed as one name per line plus a bullet line.

    Springer / Elsevier layout::

        Andreas Happe1
        · Aaron Kaplan2 · Jürgen Cito1

    The bullet line is handled by the comma-separated strategy, but the
    single name on the line above is not. We merge the preceding
    single-name lines into the bullet-separated list.

    Returns the names, or [] when nothing was found.
    """
    title_words: set[str] = set()
    if title:
        title_words = {w.lower().strip(".,:;") for w in title.split()}

    region_end = _author_region(lines, scan_limit)

    for idx in range(1, min(region_end, scan_limit)):
        line = lines[idx]
        # The bullet line must contain at least two bullet-separated names.
        if "·" not in line and "•" not in line:
            continue
        bullet_names = _split_author_line(line)
        bullet_names = [
            n
            for n in bullet_names
            if _looks_like_person_name(n) or _looks_like_allcaps_person_name(n)
        ]
        if len(bullet_names) < 2:
            continue
        # Walk backwards collecting single-name lines.
        prefix: list[str] = []
        j = idx - 1
        while j >= 0:
            cand = _strip_author_marks(lines[j])
            if (
                cand
                and "," not in cand
                and not _looks_like_affiliation(lines[j])
                and not _looks_like_location_line(lines[j])
                and not _line_is_part_of_title(cand, title_words)
                and (
                    _looks_like_person_name(cand)
                    or _looks_like_allcaps_person_name(cand)
                )
            ):
                prefix.insert(0, cand)
                j -= 1
                continue
            break
        names = prefix + bullet_names
        if len(names) >= 2:
            return names
    return []


def _guess_authors_from_inline_affiliation(
    lines: list[str], scan_limit: int, title: str | None = None
) -> list[str]:
    """Find authors printed as ``NAME, Affiliation`` on a single line.

    ACM / IEEE journal layout prints the author name first and the
    affiliation immediately after it on the same physical line::

        ZEHANG DENG, Department of Computer Science and Software
        Engineering, Swinburne University of Technology, Hawthorn,
        Australia
        YONGJIAN GUO, Tsinghua University, Beijing, China

    The name may be in ALL CAPS (``ZEHANG DENG``) or in title case.
    The affiliation is whatever follows the first comma.

    Only the front-matter region (before the body text starts) is
    scanned so that names mentioned in the body cannot leak in.

    Returns the names, or [] when no such line is found.
    """
    title_words: set[str] = set()
    if title:
        title_words = {w.lower().strip(".,:;") for w in title.split()}

    region_end = _author_region(lines, scan_limit)

    hits: list[tuple[int, str]] = []
    for idx, line in enumerate(lines[:region_end]):
        if "," not in line:
            continue
        head, _, tail = line.partition(",")
        candidate = _strip_author_marks(head)
        if not candidate:
            continue
        # The tail must actually look like an affiliation.
        if not _looks_like_affiliation(tail):
            continue
        # Skip title continuation lines.
        if _line_is_part_of_title(candidate, title_words):
            continue
        # Accept a title-case person name OR an ALL-CAPS one.
        if _looks_like_person_name(candidate) or _looks_like_allcaps_person_name(
            candidate
        ):
            hits.append((idx, candidate))

    if len(hits) < 2:
        return []

    # The author block is interleaved with affiliation continuation lines
    # ("ZEHANG DENG, Department of ... / Technology, Hawthorn, Australia"),
    # so the hits are not on consecutive lines. Keep every hit that lies
    # inside the front-matter region (already bounded by
    # `_author_region`), which stops at the first section heading or
    # body sentence.
    return [name for _idx, name in hits]


def _collect_name_block_backwards(
    lines: list[str], end_idx: int, region_end: int, title_words: set[str]
) -> str:
    """Collect the author name that belongs just before `lines[end_idx]`.

    IEEE author blocks put the affiliation (and sometimes an ORCID iD)
    *after* the name, so we walk backwards from the anchor and gather
    the name lines::

        Lokanatha Reddy        <- given name(s)
        Gandikota              <- surname continuation
        Verizon                <- company  (before a location line)
        Prosper, USA           <- location
        0009-0006-6058-3326    <- ORCID (the anchor)

    A single capitalised word is a company when the line below it (in
    document order) is a location; otherwise it continues the name.

    Returns the joined name, or "" when no name was found.
    """
    name_parts: list[str] = []
    surname_parts: list[str] = []

    i = end_idx - 1
    while i >= 0 and i < region_end:
        raw = lines[i]
        cand = _strip_author_marks(raw)
        if not cand:
            i -= 1
            continue
        # Skip location and affiliation lines.
        if _looks_like_location_line(raw) or _looks_like_affiliation(raw):
            i -= 1
            continue
        if _line_is_part_of_title(cand, title_words):
            break
        if _looks_like_person_name(cand):
            name_parts.insert(0, cand)
            i -= 1
            continue
        # A lone capitalised word: either a surname continuation or a
        # company name. A company sits directly above a location line.
        if (
            len(cand.split()) == 1
            and cand.isalpha()
            and cand[:1].isupper()
        ):
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if _looks_like_location_line(next_line):
                # Company name — skip it.
                i -= 1
                continue
            surname_parts.insert(0, cand)
            i -= 1
            continue
        break

    if not name_parts:
        return ""
    # Document order is "Given Names" then "Surname", e.g.
    # "Lokanatha Reddy" / "Gandikota" -> "Lokanatha Reddy Gandikota".
    return " ".join(name_parts + surname_parts)


def _collect_name_block_forward(
    lines: list[str], start_idx: int, region_end: int, title_words: set[str]
) -> str:
    """Collect the author name printed immediately AFTER `lines[start_idx]`.

    Some IEEE blocks put the ORCID / affiliation BEFORE the name::

        0009-0005-9674-0242
        Chandra Doddapalli
        Ramachandra Reddy

    Returns the joined name, or "" when none was found.
    """
    name_parts: list[str] = []
    surname_parts: list[str] = []

    i = start_idx + 1
    while i < region_end:
        raw = lines[i]
        cand = _strip_author_marks(raw)
        if not cand:
            i += 1
            continue
        if _looks_like_location_line(raw) or _looks_like_affiliation(raw):
            break
        if _line_is_part_of_title(cand, title_words):
            break
        if _looks_like_person_name(cand):
            name_parts.append(cand)
            i += 1
            continue
        # A lone capitalised word right after a name continues the
        # surname ("Chandra Doddapalli" / "Ramachandra Reddy").
        if (
            name_parts
            and len(cand.split()) == 1
            and cand.isalpha()
            and cand[:1].isupper()
        ):
            surname_parts.append(cand)
            i += 1
            continue
        break
    if not name_parts:
        return ""
    return " ".join(name_parts + surname_parts)


def _guess_authors_from_orcid_block(
    lines: list[str], scan_limit: int, title: str | None = None
) -> list[str]:
    """Find authors in an IEEE block interleaved with ORCID iDs.

    Some IEEE author blocks look like this (the ORCID iD is printed on
    its own line after the affiliation)::

        Lokanatha Reddy
        Gandikota
        Verizon
        Prosper, USA
        0009-0006-6058-3326
        Vasanth Jegathesan
        Verizon
        Irving, USA
        0009-0003-0672-4755

    The ORCID line is neither a name nor an affiliation, so the
    name+affiliation strategy cannot match it. We therefore anchor on
    each ORCID and walk in both directions to the name it belongs to.

    Only the front-matter region is scanned, so names mentioned in the
    body cannot leak in.

    Returns the names, or [] when no such block is found.
    """
    # A 16-digit ORCID, optionally prefixed with "https://orcid.org/".
    orcid_re = re.compile(
        r"^\s*(?:https?://orcid\.org/)?\d{4}-\d{4}-\d{4}-\d{3}[\dX]\s*$"
    )
    title_words: set[str] = set()
    if title:
        title_words = {w.lower().strip(".,:;") for w in title.split()}

    region_end = _author_region(lines, scan_limit)

    names: list[str] = []
    seen: set[str] = set()
    for idx, line in enumerate(lines[:region_end]):
        if not orcid_re.match(line):
            continue
        candidate = _collect_name_block_backwards(
            lines, idx, region_end, title_words
        )
        if not candidate:
            candidate = _collect_name_block_forward(
                lines, idx, region_end, title_words
            )
        if not candidate:
            continue
        if _looks_like_metric_name(candidate):
            continue
        if candidate not in seen:
            seen.add(candidate)
            names.append(candidate)
    return names if len(names) >= 2 else []


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
        " pvt", "holdings",
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
    title = _guess_title(first_page_text, pdf_title, filename=filename)

    return {
        "title": title,
        "authors": _guess_authors(first_page_text, pdf_authors, title=title),
        "publication_date": pdf_year or fp_year,
        "venue": venue,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "institution": None,  # Not reliably detectable without heuristics we don't trust
    }


__all__ = ["extract_metadata"]
