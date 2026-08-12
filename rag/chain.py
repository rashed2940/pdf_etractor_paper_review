"""Retrieval-augmented QA chain (AGENTS.md §10).

Composes:
  Retriever  ->  Prompt builder  ->  LLM  ->  QAResult

The chain is intentionally a plain function (`answer_question`) rather
than a full LangChain chain object so it stays testable without spinning
up a real LLM. A `MockLLM` fixture in tests substitutes a deterministic
function for `make_llm()`.

The chain NEVER executes code from PDF content; the only thing it
forwards to the LLM is the assembled prompt text. The model is
instructed (via prompts.QA_SYSTEM_PROMPT) to treat documents as data.
"""
from __future__ import annotations

from typing import Any, Callable

from config.logging import get_logger
from config.settings import make_llm
from models.paper import QAResult, SourceReference
from rag.prompts import (
    COMPARATIVE_QA_SYSTEM_PROMPT,
    INJECTION_GUARD,
    QA_SYSTEM_PROMPT,
    REFUSAL_SENTINEL,
    build_user_prompt,
)
from rag.retriever import Retriever

logger = get_logger(__name__)

# Type alias for an injectable LLM (defaults to settings.make_llm()).
LLMFactory = Callable[[], Any]


def _is_refusal(text: str) -> bool:
    """Detect whether the LLM response is the standard refusal."""
    return REFUSAL_SENTINEL.lower() in text.lower()


def _build_messages(
    question: str,
    documents: list[dict],
    comparative: bool,
) -> list[dict[str, str]]:
    """Build the [system, user] message pair for the chat model."""
    system = (
        (COMPARATIVE_QA_SYSTEM_PROMPT if comparative else QA_SYSTEM_PROMPT)
        + "\n\n" + INJECTION_GUARD
    )
    user = build_user_prompt(question, documents)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _invoke_llm(llm: Any, messages: list[dict[str, str]]) -> str:
    """Call the LLM and return its textual content.

    Supports both LangChain chat models (have `.invoke(messages)`) and
    plain callables (used by tests).
    """
    if hasattr(llm, "invoke"):
        response = llm.invoke(messages)
        # LangChain returns BaseMessage with .content
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        return content
    return str(llm(messages))


def _is_comparative(question: str) -> bool:
    """Heuristic: question is comparative if it spans multiple papers."""
    q = question.lower()
    triggers = (
        "compare", "comparison", "differ", "difference", "differences",
        "across", "all papers", "between papers", "which paper",
        "common ", "best ", "strongest", "vs.", " versus ",
    )
    return any(t in q for t in triggers)


def answer_question(
    question: str,
    *,
    retriever: Retriever | None = None,
    llm_factory: LLMFactory | None = None,
    top_k: int | None = None,
    paper_id: str | None = None,
    cybersecurity_area: str | None = None,
) -> QAResult:
    """End-to-end RAG: retrieve, prompt, call LLM, parse, return QAResult.

    Args:
        question: User's natural-language question.
        retriever: Optional Retriever (default: builds one).
        llm_factory: Optional callable returning an LLM (default: settings.make_llm()).
        top_k: Override number of chunks to retrieve.
        paper_id: Restrict to a single paper.
        cybersecurity_area: Optional filter.

    Returns:
        QAResult with answer + sources + refused flag.
    """
    retriever = retriever or Retriever()
    results = retriever.retrieve(
        question,
        top_k=top_k,
        paper_id=paper_id,
        cybersecurity_area=cybersecurity_area,
    )

    documents = [
        {
            "paper_id": r.chunk.paper_id,
            "page": r.chunk.page,
            "section": r.chunk.section,
            "text": r.chunk.text,
            "score": r.chunk.score,
        }
        for r in results
    ]

    comparative = _is_comparative(question)
    messages = _build_messages(question, documents, comparative)

    llm_factory = llm_factory or make_llm
    llm = llm_factory()
    answer_text = _invoke_llm(llm, messages).strip()

    # Refusal handling (AGENTS.md §15)
    refused = _is_refusal(answer_text)

    # Build sources from retrieval (deduplicate by (paper_id, page)).
    sources: list[SourceReference] = []
    seen: set[tuple[str, int | None]] = set()
    for r in results:
        key = (r.source.paper_id, r.source.page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(r.source)

    logger.info(
        "QA: q=%r hits=%d refused=%s comparative=%s",
        question[:80],
        len(results),
        refused,
        comparative,
    )

    return QAResult(
        question=question,
        answer=answer_text,
        sources=sources,
        refused=refused,
    )


__all__ = ["answer_question", "LLMFactory"]
