"""MetadataAgent (AGENTS.md §11.2).

Extracts bibliographic metadata from a paper's content and PDF metadata.
Never invents missing fields — returns None / [] / "Not available" when
the agent cannot determine a value.
"""
from __future__ import annotations

from textwrap import dedent

from models.paper import AgentContext, MetadataResult
from agents.base import BaseAgent


class MetadataAgent(BaseAgent):
    """Extracts title, authors, date, venue, DOI, institution."""

    name = "metadata"
    llm_role = "metadata"
    output_model = MetadataResult

    def system_prompt(self) -> str:
        return dedent(
            """
            You are a scientific-paper metadata extractor.

            Given excerpts from a research paper, extract the following
            bibliographic fields. If a field cannot be confidently
            determined from the provided text, return null (or [] for
            authors).

            Schema (JSON):
            {
              "paper_id": "<the paper_id from context>",
              "title": "<string or null>",
              "authors": ["<name1>", "<name2>", ...],
              "publication_date": "<year as string, e.g. '2024' or null>",
              "venue": "<conference/journal name or null>",
              "doi": "<DOI string or null>",
              "institution": "<primary author affiliation or null>",
              "sources": []
            }

            Rules:
            - Cross-check the paper's first page for title, authors, venue.
            - DOIs look like "10.xxxx/yyyy".
            - Do not fabricate fields. If unsure, use null.
            - Preserve the spelling of author names exactly as written.
            """
        ).strip()

    def user_prompt(self, context: AgentContext) -> str:
        # Prefer any pre-extracted metadata; then the retrieved chunks.
        pre_meta = context.metadata or {}
        chunks_text = "\n\n---\n\n".join(
            f"[page {c.page} section {c.section or '-'}]\n{c.text}"
            for c in context.retrieved_chunks
        ) or "(no retrieved chunks)"
        pre_meta_block = (
            "\n".join(f"  {k}: {v}" for k, v in pre_meta.items() if v)
            or "  (no pre-extracted metadata)"
        )
        return (
            f"paper_id: {context.paper_id}\n\n"
            f"Pre-extracted metadata (from PDF metadata + heuristics):\n"
            f"{pre_meta_block}\n\n"
            f"Retrieved paper excerpts:\n{chunks_text}\n\n"
            f"Return the JSON object now."
        )


__all__ = ["MetadataAgent"]
