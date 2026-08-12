"""CybersecurityAgent (AGENTS.md §12).

Classifies a paper into one or more cybersecurity domains from
CYBERSECURITY_TAXONOMY (35 labels). Includes brief evidence-grounded
rationale.
"""
from __future__ import annotations

from textwrap import dedent

from models.paper import (
    AgentContext,
    CYBERSECURITY_TAXONOMY,
    CybersecurityResult,
)
from agents.base import BaseAgent


_TAXONOMY_LIST = "\n".join(f"- {label}" for label in CYBERSECURITY_TAXONOMY)


class CybersecurityAgent(BaseAgent):
    """Multi-label classification into the cybersecurity taxonomy."""

    name = "cybersecurity"
    llm_role = "cybersecurity"
    output_model = CybersecurityResult

    def system_prompt(self) -> str:
        return dedent(
            f"""
            You are a cybersecurity-domain classifier.

            Given excerpts from a research paper, classify it into one or
            more domains from the taxonomy below. Use multiple labels
            when the paper clearly spans multiple domains.

            Taxonomy:
            {_TAXONOMY_LIST}

            Schema (JSON):
            {{
              "paper_id": "<paper_id>",
              "areas": ["<Label1>", "<Label2>", ...],
              "rationale": "<1-3 sentences explaining the classification, citing evidence>",
              "sources": []
            }}

            Rules:
            - Only use labels from the taxonomy above. Use "Other" when
              nothing fits.
            - The rationale MUST reference specific phrases from the
              paper (e.g. "the paper proposes a CNN-LSTM for IoT intrusion
              detection").
            - Do not speculate beyond what the excerpts support.
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


__all__ = ["CybersecurityAgent"]
