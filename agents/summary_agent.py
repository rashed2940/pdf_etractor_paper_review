"""SummaryAgent (AGENTS.md §14).

Produces a 100–250 word technical summary covering Problem / Method /
Results / Contribution / Limitations. Concise enough for the UI table.
"""
from __future__ import annotations

from textwrap import dedent

from models.paper import AgentContext, SummaryResult
from agents.base import BaseAgent


class SummaryAgent(BaseAgent):
    """Concise technical summarizer."""

    name = "summary"
    llm_role = "summarization"
    output_model = SummaryResult

    def system_prompt(self) -> str:
        return dedent(
            """
            You are a technical summarizer for cybersecurity papers.

            Write a concise 100–250 word summary of the paper that
            captures:
              1. Problem
              2. Method
              3. Results
              4. Contribution
              5. Limitations

            Schema (JSON):
            {
              "paper_id": "<paper_id>",
              "summary": "<the 100–250 word summary>",
              "problem": "<one sentence>",
              "method": "<one sentence>",
              "results": "<one sentence>",
              "contribution": "<one sentence>",
              "limitations": "<one sentence>",
              "sources": []
            }

            Rules:
            - Use only information from the provided excerpts.
            - Preserve technical terminology exactly (e.g. "CNN-LSTM",
              "CICIDS2017").
            - The `summary` field should be 100–250 words.
            - Use null for fields you cannot determine.
            """
        ).strip()

    def user_prompt(self, context: AgentContext) -> str:
        chunks_text = "\n\n---\n\n".join(
            f"[page {c.page} section {c.section or '-'}]\n{c.text}"
            for c in context.retrieved_chunks
        ) or "(no retrieved chunks)"
        return (
            f"paper_id: {context.paper_id}\n\n"
            f"Retrieved paper excerpts:\n{chunks_text}\n\n"
            f"Return the JSON object now."
        )


__all__ = ["SummaryAgent"]
