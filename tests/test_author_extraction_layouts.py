"""Tests for the author-extraction strategies added for the real-world
PDF layouts that previously produced empty author lists.

Covered layouts:
  * ACM journal   — "NAME, Affiliation" on one line (ALL CAPS)
  * IEEE 4-col   — name / affiliation / ORCID iD stack
  * Springer      — one name per line + middle-dot separated names
  * IEEE Access   — ALL CAPS comma separated names
  * Wrapped list  — "... Jaime Govea and" / "Pablo Palacios"

Also covers the front-matter chunk emitted by the section-aware
chunker, which is what lets the MetadataAgent see the author block at
all.
"""
from __future__ import annotations

from ingestion.metadata_extractor import (
    _guess_authors,
    _guess_authors_from_inline_affiliation,
    _guess_authors_from_orcid_block,
    _guess_authors_stacked,
    _looks_like_allcaps_person_name,
    _looks_like_orcid,
)


class TestAllCapsPersonName:
    def test_allcaps_name_accepted(self):
        assert _looks_like_allcaps_person_name("ZEHANG DENG")

    def test_title_case_name_rejected(self):
        # Title-case names are handled by _looks_like_person_name, and
        # must NOT be accepted by the all-caps helper (otherwise
        # "Ant Group" would look like a person).
        assert not _looks_like_allcaps_person_name("Ant Group")

    def test_institution_rejected(self):
        assert not _looks_like_allcaps_person_name("UC BERKELEY") is False
        assert not _looks_like_allcaps_person_name("ABSTRACT")

    def test_single_word_rejected(self):
        assert not _looks_like_allcaps_person_name("VERIZON")


class TestOrcidDetection:
    def test_plain_orcid(self):
        assert _looks_like_orcid("0009-0006-6058-3326")

    def test_orcid_with_url(self):
        assert _looks_like_orcid("https://orcid.org/0000-0002-1825-0097")

    def test_not_orcid(self):
        assert not _looks_like_orcid("Prosper, USA")


class TestInlineAffiliationLayout:
    """ACM style: NAME, Affiliation on a single line."""

    TEXT = (
        "AI Agents Under Threat: A Survey of Key Security\n"
        "Challenges and Future Pathways\n"
        "ZEHANG DENG, Department of Computer Science, Swinburne University\n"
        "YONGJIAN GUO, Tsinghua University, Beijing, China\n"
        "CHANGZHOU HAN, Department of Computer Science, Swinburne University\n"
        "Abstract\n"
        "The rapid growth of agents has created new risks.\n"
    )

    def test_extracts_all_names(self):
        lines = [l.strip() for l in self.TEXT.splitlines() if l.strip()]
        result = _guess_authors_from_inline_affiliation(lines, 40, None)
        assert result == [
            "ZEHANG DENG",
            "YONGJIAN GUO",
            "CHANGZHOU HAN",
        ]

    def test_body_names_do_not_leak_in(self):
        lines = [l.strip() for l in self.TEXT.splitlines() if l.strip()]
        result = _guess_authors_from_inline_affiliation(lines, 40, None)
        assert not any("Survey" in r for r in result)


class TestOrcidBlockLayout:
    """IEEE style: name / affiliation / location / ORCID stack."""

    TEXT = (
        "Agentic AI in Cybersecurity: Threat Modeling and\n"
        "Containment Strategies for Autonomous LLM\n"
        "Agents\n"
        "Lokanatha Reddy\n"
        "Gandikota\n"
        "Verizon\n"
        "Prosper, USA\n"
        "0009-0006-6058-3326\n"
        "Vasanth Jegathesan\n"
        "Verizon\n"
        "Irving, USA\n"
        "0009-0003-0672-4755\n"
        "Abstract-The rapid proliferation of autonomous agents.\n"
    )

    def test_extracts_names(self):
        lines = [l.strip() for l in self.TEXT.splitlines() if l.strip()]
        result = _guess_authors_from_orcid_block(lines, 40, None)
        assert "Vasanth Jegathesan" in result
        assert len(result) >= 2

    def test_company_not_mistaken_for_author(self):
        lines = [l.strip() for l in self.TEXT.splitlines() if l.strip()]
        result = _guess_authors_from_orcid_block(lines, 40, None)
        assert not any("Verizon" in r for r in result)


class TestStackedLayout:
    """Springer style: one name per line + middle-dot separated names."""

    TEXT = (
        "Received: 10 February 2025 / Accepted: 20 October 2025\n"
        "© The Author(s) 2026\n"
        "LLMs as Hackers: Autonomous Linux Privilege Escalation\n"
        "Attacks\n"
        "Andreas Happe1\n"
        "· Aaron Kaplan2 · Jürgen Cito1\n"
        "Empirical Software Engineering (2026) 31:70\n"
        "Abstract\n"
        "Penetration-testing is crucial for identifying vulnerabilities.\n"
    )

    def test_extracts_all_three_names(self):
        lines = [l.strip() for l in self.TEXT.splitlines() if l.strip()]
        result = _guess_authors_stacked(lines, 40, None)
        assert result == ["Andreas Happe", "Aaron Kaplan", "Jürgen Cito"]


class TestAllCapsCommaSeparated:
    """IEEE Access style: ALL CAPS names on one comma separated line."""

    TEXT = (
        "A Multi-Agent System for Cybersecurity Threat\n"
        "Detection and Correlation Using Large\n"
        "Language Models\n"
        "YASSER HMIMOU1,2, MOHAMED TABAA1,2, AZEDDINE KHIAT2\n"
        "1Multidisciplinary Laboratory of Research and Innovation, Morocco\n"
        "Abstract\n"
        "As cyber-attacks rapidly evolve, our architecture helps.\n"
    )

    def test_extracts_names(self):
        result = _guess_authors(self.TEXT, [], title=None)
        assert "YASSER HMIMOU" in result
        assert "MOHAMED TABAA" in result


class TestWrappedAuthorList:
    """The last author continues on the next line after a trailing 'and'."""

    TEXT = (
        "Autonomous federated defense\n"
        "for zero-day threats in IIoT:\n"
        "explainable agents with real-time\n"
        "edge inference\n"
        "William Villegas-Ch1*, Rommel Gutierrez1, Jaime Govea1 and\n"
        "Pablo Palacios2\n"
        "1Escuela de Ingenieria en Ciberseguridad, Quito, Ecuador\n"
        "Abstract\n"
        "The growing interconnection of industrial devices is a risk.\n"
    )

    def test_pulls_in_wrapped_last_author(self):
        result = _guess_authors(self.TEXT, [], title=None)
        assert "Pablo Palacios" in result
        assert "William Villegas-Ch" in result


class TestInstitutionCleanup:
    """Institution names must not survive as authors."""

    def test_institution_removed(self):
        text = (
            "Title line for the paper\n"
            "Kamalbek Berkimbayev\n"
            "Kazakh-Turkish University\n"
            "Gulnara Abitova\n"
            "Kazakh Agrotechnical Research\n"
            "Abstract\n"
            "Large language models are widely adopted in industry today.\n"
        )
        result = _guess_authors(text, [], title=None)
        assert not any("University" in r for r in result)
        assert not any("Research" in r for r in result)


class TestCommaPackedListSplit:
    """A leftover comma means several names are packed into one entry."""

    def test_comma_list_is_split(self):
        text = (
            "Some Paper Title About Agents\n"
            "Richard Fang, Rohan Bindu, Akul Gupta\n"
            "Abstract\n"
            "We evaluate agents on a large number of cyber security tasks.\n"
        )
        result = _guess_authors(text, [], title=None)
        assert "Richard Fang" in result
        assert not any("," in r for r in result)


class TestFrontMatterChunk:
    """The section-aware chunker must keep the title/author block."""

    def test_front_matter_chunk_is_emitted(self):
        from ingestion.chunker import FRONT_MATTER_SECTION, chunk_pages

        pages = [
            (
                1,
                "A Study of AI Agents for Cyber Defense\n"
                "Jane Doe, John Smith\n"
                "Abstract\n"
                "This paper studies agents.\n"
                "\n\nIntroduction\n"
                "Agents are everywhere now in modern security stacks.\n",
            ),
            (
                2,
                "Related Work\n"
                "Previous work studied agents for many years already.\n",
            ),
        ]
        chunks = chunk_pages("paper_test", pages, chunk_size=2000, chunk_overlap=100)
        front = [c for c in chunks if c.section == FRONT_MATTER_SECTION]
        assert front, "expected a Front Matter chunk"
        blob = " ".join(c.text for c in front)
        assert "Jane Doe" in blob
        assert "A Study of AI Agents" in blob

    def test_author_block_reaches_the_vector_metadata(self):
        """The front-matter chunk is what gives MetadataAgent the names."""
        from ingestion.chunker import chunk_pages

        pages = [
            (
                1,
                "A Study of AI Agents\n"
                "Jane Doe, John Smith\n"
                "Abstract\n"
                "Body text goes here about agents and security research.\n",
            )
        ]
        chunks = chunk_pages("paper_test", pages, chunk_size=2000, chunk_overlap=100)
        assert any("Jane Doe" in c.text for c in chunks)
