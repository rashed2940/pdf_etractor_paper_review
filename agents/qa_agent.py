"""QA Agent (AGENTS.md §15, §34).

A thin wrapper around `rag.chain.answer_question` that:
- Logs agent identity.
- Provides the agent name expected by Phase 4's coordinator registry.
- Exposes a uniform interface even though this agent isn't yet
  subclassing BaseAgent (Phase 3 introduces BaseAgent; QA will be
  refactored then if needed).

Multi-paper comparative questions are handled automatically by
`_is_comparative` inside the chain — no extra logic here.
"""
from __future__ import annotations

from typing import Any, Callable

from config.logging import get_logger
from models.paper import QAResult
from rag.chain import LLMFactory, answer_question

logger = get_logger(__name__)


class QAAgent:
    """Answers user questions about the uploaded paper library.

    Usage:
        agent = QAAgent()
        result = agent.ask("Which papers discuss LLM security?")
        print(result.answer, result.sources)
    """

    NAME = "qa_agent"
    DESCRIPTION = "Answers user questions about uploaded papers using RAG."

    def __init__(
        self,
        *,
        retriever: Any | None = None,
        llm_factory: LLMFactory | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm_factory = llm_factory
        logger.debug("QAAgent initialized")

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        paper_id: str | None = None,
        cybersecurity_area: str | None = None,
    ) -> QAResult:
        """Ask a question; return a QAResult with answer + citations."""
        logger.info("QAAgent.ask: %r", question[:100])
        return answer_question(
            question,
            retriever=self._retriever,
            llm_factory=self._llm_factory,
            top_k=top_k,
            paper_id=paper_id,
            cybersecurity_area=cybersecurity_area,
        )


__all__ = ["QAAgent"]
