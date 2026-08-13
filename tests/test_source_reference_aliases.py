"""Tests for SourceReference alias-acceptance behavior.

Background: cloud LLMs (and some local ones) emit source citations
in a variety of shapes (`{"content": "..."}`, `{"excerpt": "..."}`,
`{"quote": "..."}`, etc.) even when our prompt specifies `text`.
We accept these aliases via a Pydantic model_validator so a single
LLM-emitted field-name mistake doesn't fail the whole agent output.

These tests pin down that behavior so future prompt changes don't
regress alias acceptance silently.
"""
from __future__ import annotations

from models.paper import SourceReference


class TestSourceReferenceAliases:
    def test_canonical_text_field(self):
        """The canonical `text` field works as before."""
        ref = SourceReference(paper_id="p1", page=3, text="hello")
        assert ref.text == "hello"
        assert ref.paper_id == "p1"
        assert ref.page == 3

    def test_content_alias(self):
        """LLM emits {"content": "..."} — we copy into `text`."""
        ref = SourceReference.model_validate(
            {"paper_id": "p1", "page": 3, "content": "hello via content"}
        )
        assert ref.text == "hello via content"
        assert ref.paper_id == "p1"
        assert ref.page == 3
        # The alias is excluded from the model dump.
        assert "content" not in ref.model_dump(exclude_none=False)

    def test_excerpt_alias(self):
        """LLM emits {"excerpt": "..."} — we copy into `text`."""
        ref = SourceReference.model_validate(
            {"paper_id": "p1", "excerpt": "excerpt text"}
        )
        assert ref.text == "excerpt text"

    def test_quote_alias(self):
        """LLM emits {"quote": "..."} — we copy into `text`."""
        ref = SourceReference.model_validate(
            {"paper_id": "p1", "quote": "quoted text"}
        )
        assert ref.text == "quoted text"

    def test_page_number_alias(self):
        """LLM emits {"page_number": N} — we copy into `page`."""
        ref = SourceReference.model_validate(
            {"paper_id": "p1", "page_number": 7, "text": "hi"}
        )
        assert ref.page == 7

    def test_text_takes_precedence_over_alias(self):
        """If both `text` and `content` are present, `text` wins."""
        ref = SourceReference.model_validate(
            {"paper_id": "p1", "text": "primary", "content": "alias"}
        )
        assert ref.text == "primary"

    def test_missing_paper_id_defaults_to_empty(self):
        """LLM occasionally omits paper_id. We default to "" so the
        agent result is still valid (the coordinator fills it in
        later from the AgentContext)."""
        ref = SourceReference.model_validate({"page": 3, "text": "hi"})
        assert ref.paper_id == ""
        assert ref.text == "hi"

    def test_unknown_keys_ignored(self):
        """Extra keys the LLM invents (e.g. "chunk_text", "snippet_id")
        are silently dropped so one bad source doesn't fail validation."""
        ref = SourceReference.model_validate(
            {
                "paper_id": "p1",
                "page": 3,
                "text": "hi",
                "chunk_text": "junk",
                "snippet_id": "abc",
                "irrelevant": 42,
            }
        )
        assert ref.text == "hi"
        # Unknown fields are dropped, not stored.
        assert not hasattr(ref, "chunk_text")
        assert not hasattr(ref, "snippet_id")
