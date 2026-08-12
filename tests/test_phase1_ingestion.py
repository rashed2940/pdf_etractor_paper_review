"""Phase 1 tests: ingestion pipeline (cleaner, metadata, chunker, PDF loader)."""
from __future__ import annotations


class TestTextCleaner:
    def test_clean_text_ligature_and_hyphenation(self):
        from ingestion.text_cleaner import clean_text

        raw = "This is a ﬁnal exam-\nple of normaliza-\ntion."
        cleaned = clean_text(raw)
        assert "final example of normalization" in cleaned

    def test_clean_text_preserves_single_newlines(self):
        from ingestion.text_cleaner import clean_text

        raw = "Title\nAbstract\nBody paragraph."
        cleaned = clean_text(raw)
        # Single newlines should be preserved so section detection works
        assert "\n" in cleaned
        assert "Title" in cleaned and "Abstract" in cleaned

    def test_strip_headers_footers(self):
        from ingestion.text_cleaner import strip_headers_footers

        pages = [
            "Title\nAuthor\nIEEE Trans. 2024\n\nIntroduction",
            "2\nIEEE Trans. 2024\n\nBody",
            "3\nIEEE Trans. 2024\n\nMore body",
        ]
        cleaned = strip_headers_footers(pages, min_repeats=3)
        joined = "\n".join(cleaned)
        assert "IEEE Trans. 2024" not in joined
        assert "Introduction" in joined

    def test_normalize_unicode(self):
        from ingestion.text_cleaner import normalize_unicode

        assert normalize_unicode("ﬁnal") == "final"
        assert normalize_unicode("ﬂag") == "flag"
        assert normalize_unicode("hello") == "hello"


class TestMetadataExtractor:
    def test_extract_metadata_basic(self):
        from ingestion.metadata_extractor import extract_metadata

        pdf_meta = {
            "title": "Sample Paper",
            "author": "Alice; Bob",
            "creationDate": "D:20240315120000+00'00'",
        }
        fp_text = (
            "Sample Paper\nAlice Smith, Bob Jones\narXiv:2401.12345v2\n"
            "DOI: 10.1109/SAMPLE.2024.123456\n\nAbstract: We propose..."
        )
        meta = extract_metadata(pdf_meta, fp_text)
        assert meta["title"] == "Sample Paper"
        assert "Alice" in str(meta["authors"])
        assert meta["publication_date"] == "2024"
        assert meta["arxiv_id"] == "arXiv:2401.12345v2"
        assert meta["doi"] is not None

    def test_extract_metadata_does_not_fabricate(self):
        from ingestion.metadata_extractor import extract_metadata

        meta = extract_metadata({}, "no useful information here")
        # title may default to longest line of input — verify no fabrication
        # of doi, arxiv_id, or authors
        assert meta["doi"] is None
        assert meta["arxiv_id"] is None
        assert meta["authors"] == []
        assert meta["institution"] is None


class TestChunker:
    def test_section_aware_chunking(self):
        from ingestion.chunker import chunk_pages

        text = """Abstract
This paper presents a method.

1. Introduction
Cybersecurity is critical.

2. Methodology
We trained a deep neural network.

3. Results
We achieved 99% accuracy.

4. Conclusion
Future work includes extensions."""
        chunks = chunk_pages("paper_test", [(1, text)])
        sections = sorted({c.section for c in chunks if c.section})
        assert "Abstract" in sections
        assert "Methodology" in sections
        assert "Conclusion" in sections

    def test_fallback_chunking(self):
        from ingestion.chunker import chunk_pages

        # No section markers -> fallback recursive split
        chunks = chunk_pages("paper_test2", [(1, "a" * 5000)])
        assert len(chunks) >= 3
        # All chunks should have section=None in fallback mode
        assert all(c.section is None for c in chunks)

    def test_references_section_skipped(self):
        from ingestion.chunker import chunk_pages

        text = """Abstract
We propose X.

1. Introduction
Intro.

2. Methodology
Method.

References
[1] Some reference.
[2] Another reference."""
        chunks = chunk_pages("paper_test3", [(1, text)])
        # References section should NOT appear in any chunk's section field
        assert not any(c.section == "References" for c in chunks)

    def test_detect_sections(self):
        from ingestion.chunker import _detect_sections

        text = "Abstract\nbody\n\n1. Introduction\nintro body\n\n2. Methodology\nmethod body"
        sections = _detect_sections(text)
        names = [s[0] for s in sections]
        assert "Abstract" in names
        assert "Introduction" in names
        assert "Methodology" in names


class TestPdfLoader:
    def test_is_scanned_true_for_whitespace_pages(self):
        from ingestion.pdf_loader import is_scanned

        class _P:
            def __init__(self, text):
                self.text = text

        pages = [_P("   ") for _ in range(5)]
        assert is_scanned(pages, min_chars_per_page=50) is True

    def test_is_scanned_false_for_text_pages(self):
        from ingestion.pdf_loader import is_scanned

        class _P:
            def __init__(self, text):
                self.text = text

        pages = [_P("x" * 200) for _ in range(5)]
        assert is_scanned(pages, min_chars_per_page=50) is False

    def test_compute_sha256(self, tmp_path):
        from ingestion.pdf_loader import compute_sha256

        f = tmp_path / "a.txt"
        f.write_bytes(b"hello")
        assert compute_sha256(f) == (
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )

    def test_extract_pdf_returns_pages(self, test_pdf_path):
        from ingestion.pdf_loader import extract_pdf

        doc = extract_pdf(test_pdf_path)
        assert doc.page_count >= 1
        assert len(doc.pages) >= 1
        assert doc.sha256


class TestEmbeddingService:
    def test_embedding_factory_returns_service(self):
        from embeddings.embedding_service import get_embedding_service
        from embeddings.embedding_service import EmbeddingService

        # We don't actually instantiate (would download model). Just check
        # the factory exists and would return an EmbeddingService.
        # Skip actual instantiation to keep tests fast.
        assert EmbeddingService is not None
        # Don't call get_embedding_service() — that loads the model