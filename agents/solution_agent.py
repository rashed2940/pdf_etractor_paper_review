"""SolutionAgent (AGENTS.md §13).

Extracts what problem the paper solves, what solution it proposes,
what methodology/architecture/algorithms/datasets/experiments were used,
and what limitations the authors acknowledge.
"""
from __future__ import annotations

from textwrap import dedent

from models.paper import AgentContext, SolutionResult
from agents.base import BaseAgent


class SolutionAgent(BaseAgent):
    """Solution/Contribution extractor."""

    name = "solution"
    llm_role = "solution"
    output_model = SolutionResult

    def system_prompt(self) -> str:
        return dedent(
            """
            You are a scientific-paper solution/contribution analyzer.

            Given excerpts from a research paper, answer these questions
            strictly using the provided text:

            - problem_statement: What problem does the paper try to solve?
            - proposed_solution: What solution does it propose?
            - methodology: What is the overall methodology?
            - algorithms: Which algorithms or models are used? (list)
            - datasets: Which datasets are used? (list)
            - experiments: What experiments were performed?
            - key_results: What were the main quantitative/qualitative
              results? Cite numbers when available.
            - limitations: What limitations do the authors acknowledge?

            Schema (JSON):
            {
              "paper_id": "<paper_id>",
              "problem_statement": "<string or null>",
              "proposed_solution": "<string or null>",
              "methodology": "<string or null>",
              "algorithms": ["<name>", ...],
              "datasets": ["<name>", ...],
              "experiments": "<string or null>",
              "key_results": "<string or null>",
              "limitations": "<string or null>",
              "sources": []
            }

            Rules:
            - Use null for fields you cannot determine.
            - Do not invent numbers, datasets, or models.
            - Distinguish between what the paper PROPOSES and what it
              cites as prior work.
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


__all__ = ["SolutionAgent"]
