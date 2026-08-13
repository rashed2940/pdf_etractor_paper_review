"""End-to-end test against the actual ETASR PDF in the workspace.

This is a regression test for the title/author extraction bug where the
PDF metadata was reflecting the Word document filename and login user,
not the paper's actual title and authors.

If the PDF is missing the test is skipped (it lives under data/uploads/
which is gitignored).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.metadata_extractor import extract_metadata


PDF_CANDIDATES = [
    Path("data/uploads/Enhancing_Multi-Agent_Reinforcement_Learning_Intru.pdf"),
    Path("data/uploads/Enhancing_Multi-Agent_Reinforcement_Learning_Intru_49de7e.pdf"),
]


def _find_real_pdf() -> Path | None:
    for p in PDF_CANDIDATES:
        if p.exists():
            return p
    # Fallback: any ETASR PDF in data/uploads/
    uploads = Path("data/uploads")
    if uploads.exists():
        for f in uploads.glob("Enhancing_Multi-Agent*.pdf"):
            return f
    return None


@pytest.fixture(scope="module")
def real_pdf_meta():
    p = _find_real_pdf()
    if p is None:
        pytest.skip(
            "ETASR PDF not in data/uploads/ — this regression test "
            "needs the original file. Skipping."
        )
    try:
        import pymupdf
    except ImportError:
        pytest.skip("pymupdf not installed")
    doc = pymupdf.open(str(p))
    meta = dict(doc.metadata or {})
    first_page_text = doc[0].get_text()
    doc.close()
    return extract_metadata(meta, first_page_text)


class TestRealETASRPaper:
    def test_title_is_not_word_filename(self, real_pdf_meta):
        """PDF metadata title is 'Microsoft Word - ETASR_V15_N4_pp-24455-24459';
        the paper's real title is something else."""
        assert real_pdf_meta["title"] is not None
        title = real_pdf_meta["title"]
        assert "Microsoft Word" not in title
        assert "ETASR_V15" not in title
        assert ".docx" not in title
        # The real title contains keywords from the paper
        lower = title.lower()
        assert any(
            keyword in lower
            for keyword in ("intrusion detection", "multi-agent", "reinforcement",
                            "random forest", "mar")
        )

    def test_title_is_not_copyright_or_abstract(self, real_pdf_meta):
        title = real_pdf_meta["title"] or ""
        assert "Licensed under" not in title
        assert "Copyright" not in title
        assert "DOI" not in title
        assert "Received" not in title
        assert "ABSTRACT" not in title

    def test_authors_list_is_more_than_one(self, real_pdf_meta):
        """PDF metadata author is 'gerry' (Word login); the real paper
        has 6 authors."""
        authors = real_pdf_meta["authors"]
        assert len(authors) >= 3, f"Expected ≥3 authors, got {authors}"
        assert "gerry" not in authors
        # First author is Diaz (the corresponding author)
        assert any("Diaz" in a for a in authors)

    def test_publication_date_is_2025(self, real_pdf_meta):
        """Paper metadata says D:20250605013947 — should pick year 2025."""
        assert real_pdf_meta["publication_date"] == "2025"

    def test_doi_extracted(self, real_pdf_meta):
        """Paper has DOI 10.48084/etasr.11113 in body."""
        assert real_pdf_meta["doi"] is not None
        assert "etasr" in real_pdf_meta["doi"].lower()
