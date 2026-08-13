"""Tests for the PDF-metadata-as-title and metadata-as-author
filtering heuristics.

Background: PDFs created from Word / non-academic toolchains often
have PyMuPDF metadata that is the *filename* (e.g. "Microsoft Word -
ETASR_V15_N4_pp-24455-24459.docx") rather than the paper's actual
title, and the author is the Word user's login name (e.g. "gerry").
We detect these cases and fall back to first-page heuristics.
"""
from __future__ import annotations

from ingestion.metadata_extractor import (
    _looks_like_filename,
    _authors_look_unreliable,
    _guess_title,
    _guess_authors,
)


class TestLooksLikeFilename:
    def test_microsoft_word_prefix(self):
        assert _looks_like_filename("Microsoft Word - ETASR_V15_N4_pp-24455-24459")

    def test_docx_extension(self):
        assert _looks_like_filename("mypaper.docx")

    def test_pdf_extension(self):
        assert _looks_like_filename("paper_v3.pdf")

    def test_page_range_marker(self):
        assert _looks_like_filename("ETASR_V15_N4_pp-24455-24459")

    def test_libreoffice_prefix(self):
        assert _looks_like_filename("LibreOffice Document - draft")

    def test_real_paper_title_accepted(self):
        assert not _looks_like_filename(
            "A hybrid MARL-based IDS for IoT networks"
        )

    def test_short_real_title_accepted(self):
        assert not _looks_like_filename("Deep Learning for Cybersecurity")

    def test_real_title_with_dash_accepted(self):
        assert not _looks_like_filename(
            "CNN-LSTM Based Intrusion Detection in 5G Networks"
        )

    def test_empty_string(self):
        assert not _looks_like_filename("")


class TestAuthorsLookUnreliable:
    def test_single_short_word_is_username(self):
        assert _authors_look_unreliable(["gerry"]) is True

    def test_single_admin_is_username(self):
        assert _authors_look_unreliable(["admin"]) is True

    def test_email_is_unreliable(self):
        assert _authors_look_unreliable(["john.doe@example.com"]) is True

    def test_windows_domain_user_unreliable(self):
        assert _authors_look_unreliable(["CORP\\jsmith"]) is True

    def test_real_two_author_list_accepted(self):
        assert _authors_look_unreliable(["Alice Smith", "Bob Jones"]) is False

    def test_real_three_author_list_accepted(self):
        assert _authors_look_unreliable(["A. B. Smith", "C. D. Jones"]) is False

    def test_empty_list_accepted(self):
        assert _authors_look_unreliable([]) is False


class TestGuessTitle:
    def test_pdf_metadata_title_used_when_real(self):
        text = (
            "Some other header line that is short\n"
            "A hybrid MARL-based IDS for IoT networks\n"
            "Alice Author, Bob Author\n"
        )
        result = _guess_title(text, "A hybrid MARL-based IDS for IoT networks")
        assert result == "A hybrid MARL-based IDS for IoT networks"

    def test_filename_title_rejected_falls_back_to_first_page(self):
        # New structural parser: the first non-meta, non-author line is
        # the title. The header line above the title is treated as
        # noise (not title material).
        text = (
            "Some header that is short\n"
            "Enhancing Multi-Agent Reinforcement Learning for Intrusion Detection\n"
            "Alice Author\n"
            "Faculty of Engineering, Some University\n"
            "alice@example.com\n"
            "Bob Author\n"
            "Some Other Affiliation\n"
        )
        result = _guess_title(text, "Microsoft Word - ETASR_V15_N4_pp-24455-24459")
        assert "Microsoft" not in result
        assert "Enhancing Multi-Agent Reinforcement Learning" in result
        assert "Alice" not in result  # authors shouldn't bleed into title

    def test_ieee_et_al_header_pattern(self):
        text = (
            "Journal Name\n"
            "Smith et al.: A New Approach to Intrusion Detection in IoT\n"
            "John Smith\n"
            "MIT\n"
        )
        result = _guess_title(text, None)
        assert result is not None
        assert "New Approach to Intrusion Detection" in result
        assert "Smith" not in result  # the author prefix is stripped

    def test_multiline_title_joined(self):
        # IEEE papers often break a long title across 3 lines.
        text = (
            "Journal header\n"
            "Enhancing Multi-Agent Reinforcement Learning\n"
            "Intrusion Detection Systems Using Random Forest\n"
            "Q-Value Estimation\n"
            "Ricky Diaz\n"
            "Faculty of Engineering, Udayana University\n"
        )
        # Without a filename, structural parser should still work.
        result = _guess_title(text, None)
        assert result is not None
        # The 3 title lines should be joined.
        assert "Enhancing Multi-Agent" in result
        assert "Random Forest" in result
        # Author name should not be in the title
        assert "Ricky" not in result

    def test_filename_used_when_no_metadata_and_structural_unreliable(self):
        # When the first-page is mostly meta/junk and the metadata is
        # also junk, the filename is used as the title.
        text = (
            "Journal name\n"
            "Vol. 99, No. 1, 2099\n"
            "www.example.com\n"
            "Received: tomorrow\n"
            "Licensed under a CC-BY 4.0 license\n"
            "ABSTRACT\n"
            "Some abstract text.\n"
        )
        result = _guess_title(
            text,
            "Microsoft Word - paper.docx",  # filename-like metadata
            filename="Enhancing_Multi-Agent_IDS.pdf",
        )
        assert result == "Enhancing Multi Agent Ids"

    def test_filename_unavailable_returns_none(self):
        # Last-resort: no metadata, no filename, no first page → None
        assert _guess_title("", None, filename=None) is None

    def test_copyright_line_not_picked_as_title(self):
        text = (
            "Journal name\n"
            "Short title here\n"
            "John Smith, Jane Doe\n"
            "Affiliation\n"
            "Licensed under a CC-BY 4.0 license | Copyright (c) by the authors | DOI: https://doi.org/10.48084/etasr.11113\n"
            "ABSTRACT\n"
        )
        result = _guess_title(text, None)
        assert result is not None
        assert "Licensed" not in result
        assert "Copyright" not in result
        assert "DOI" not in result
        assert "Short title here" in result

    def test_no_metadata_no_text_returns_none(self):
        assert _guess_title("", None) is None

    def test_only_filename_metadata_returns_none(self):
        # No first-page text AND no useful metadata title.
        assert _guess_title("", "Microsoft Word - something.docx") is None


class TestGuessAuthors:
    def test_real_authors_kept(self):
        text = "Title\nAlice Author, Bob Author\nAbstract..."
        result = _guess_authors(text, ["Alice Author", "Bob Author"])
        assert result == ["Alice Author", "Bob Author"]

    def test_username_replaced_with_affiliation_heuristic(self):
        # IEEE-style page 1: author name on one line, then affiliation,
        # then email. The next author follows the same pattern.
        text = (
            "Title here\n"
            "Ricky Aurelius Nurtanto Diaz\n"
            "Faculty of Engineering, Udayana University, Bali, Indonesia\n"
            "ricky@stikom-bali.ac.id\n"
            "I. Ketut Gede Darma Putra\n"
            "Department of Information Technology, Udayana University\n"
            "ikgdarmaputra@unud.ac.id\n"
            "Made Sudarma\n"
            "Department of Electrical Engineering, Udayana University\n"
            "msudarma@unud.ac.id\n"
        )
        # "gerry" is the Word username; replace with heuristic.
        result = _guess_authors(text, ["gerry"])
        assert "gerry" not in result
        assert "Ricky Aurelius Nurtanto Diaz" in result
        assert "I. Ketut Gede Darma Putra" in result
        assert "Made Sudarma" in result

    def test_email_replaced_with_heuristic(self):
        text = (
            "Title\n"
            "John Smith\n"
            "MIT\n"
            "jsmith@example.com\n"
            "Jane Doe\n"
            "Stanford University\n"
        )
        result = _guess_authors(text, ["john.doe@example.com"])
        assert "example.com" not in " ".join(result)
        assert result == ["John Smith", "Jane Doe"]

    def test_no_heuristic_match_returns_metadata_anyway(self):
        # No heuristic match but also no reliable metadata.
        text = "Some title but no author line\nMore body text."
        result = _guess_authors(text, ["gerry"])
        # We don't fabricate — return whatever we had, the LLM does better.
        assert result == ["gerry"]

    def test_empty_returns_empty(self):
        assert _guess_authors("", []) == []
