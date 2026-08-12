"""Analysis service (PLAN.md §3 Phase 4).

Thin orchestrator that exposes the two primary user-facing analytical
operations:
- `analyze_paper(paper_id)` — run all 4 agents against one paper; return PaperAnalysis.
- `compare_papers(question)` — multi-paper comparative QA (delegates to QAAgent).

These are the two functions the Flask UI will call in Phase 5.
"""
from __future__ import annotations

from typing import Any, Callable

from config.logging import get_logger
from models.paper import PaperAnalysis, QAResult
from agents.coordinator import Coordinator
from agents.qa_agent import QAAgent

logger = get_logger(__name__)


class AnalysisService:
    """High-level facade for paper analysis + multi-paper Q&A."""

    def __init__(
        self,
        *,
        coordinator: Coordinator | None = None,
        qa_agent: QAAgent | None = None,
        llm_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._coordinator = coordinator or Coordinator(llm_factory=llm_factory)
        self._qa_agent = qa_agent or QAAgent(llm_factory=llm_factory)
        logger.debug("AnalysisService initialized")

    # ----------------------------------------------------------- Single paper
    def analyze_paper(self, paper_id: str, *, top_k: int | None = None) -> PaperAnalysis:
        """Run all 4 analysis agents against a paper; return PaperAnalysis.

        Convenience wrapper around `Coordinator.analyze_paper`.
        """
        logger.info("analyze_paper(%s)", paper_id)
        return self._coordinator.analyze_paper(paper_id, top_k=top_k)

    # ----------------------------------------------------------- Multi-paper QA
    def compare_papers(self, question: str, *, top_k: int | None = None) -> QAResult:
        """Multi-paper comparative question.

        Delegates to QAAgent, which uses the RAG chain. The chain's
        `_is_comparative` heuristic routes to the comparative prompt
        when the question spans multiple papers.
        """
        logger.info("compare_papers(%r)", question[:100])
        return self._qa_agent.ask(question, top_k=top_k)

    # ----------------------------------------------------------- Bulk
    def analyze_all_papers(self, paper_ids: list[str]) -> list[PaperAnalysis]:
        """Run analyze_paper for every paper_id; returns per-paper results.

        Failures of individual papers don't stop the batch (each
        analyze_paper is exception-safe by design).
        """
        return [self.analyze_paper(pid) for pid in paper_ids]


__all__ = ["AnalysisService"]
