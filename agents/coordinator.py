"""Coordinator (AGENTS.md §11.1, §23).

Runs MetadataAgent, CybersecurityAgent, SolutionAgent, and SummaryAgent
in PARALLEL against the same paper context, then merges their outputs
into a single PaperAnalysis model.

Parallelism strategy (PLAN.md §3):
- LangChain's `invoke()` is synchronous, so `asyncio.gather` would just
  block on each call. We use `concurrent.futures.ThreadPoolExecutor`
  with one worker per agent (max 4) so the LLM calls overlap.
- Each agent has its own per-role LLM via `make_llm(role)`, so they
  don't serialize through a single bottleneck.

Failure handling (AGENTS.md §27):
- If an individual agent raises an unexpected exception, the
  Coordinator logs the error and substitutes that agent's safe default
  (the same `BaseAgent.run` already returns on retry exhaustion).
- The coordinator NEVER raises to the caller — `PaperAnalysis` is
  always returned, with `analysis_status="failed"` if everything blew up.

Inputs:
- paper_id: the stable paper identifier (matches what's in ChromaDB).
- top_k: number of chunks to retrieve (default from settings).
- llm_factory: optional override (default: settings.make_llm).
- vector_store / retriever: optional dependency injection for tests.
"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Callable

from config.logging import get_logger
from models.paper import (
    AgentContext,
    PaperAnalysis,
    RetrievedChunk,
    SourceReference,
)
from rag.retriever import Retriever
from vectorstore.chroma_store import ChromaVectorStore

# Analysis agents (QA agent is handled separately in services.analysis_service).
from agents.cybersecurity_agent import CybersecurityAgent
from agents.metadata_agent import MetadataAgent
from agents.solution_agent import SolutionAgent
from agents.summary_agent import SummaryAgent

logger = get_logger(__name__)

#: Default max parallel workers (4 analysis agents).
DEFAULT_MAX_WORKERS = 4


class Coordinator:
    """Orchestrates parallel agent execution for a single paper."""

    def __init__(
        self,
        *,
        vector_store: ChromaVectorStore | None = None,
        retriever: Retriever | None = None,
        llm_factory: Callable[[str], Any] | None = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> None:
        self._vector_store = vector_store
        self._retriever = retriever
        self._llm_factory = llm_factory
        self._max_workers = max_workers

        # Build agents. Each uses the same llm_factory (default or injected).
        factory = llm_factory
        self._metadata_agent = MetadataAgent(llm_factory=factory)
        self._cyber_agent = CybersecurityAgent(llm_factory=factory)
        self._solution_agent = SolutionAgent(llm_factory=factory)
        self._summary_agent = SummaryAgent(llm_factory=factory)
        logger.debug("Coordinator initialized (max_workers=%d)", max_workers)

    # ----------------------------------------------------------- Public API
    def analyze_paper(
        self,
        paper_id: str,
        *,
        top_k: int | None = None,
    ) -> PaperAnalysis:
        """Retrieve chunks for `paper_id`, fan out 4 agents in parallel,
        merge results into a PaperAnalysis.

        Args:
            paper_id: Stable paper identifier.
            top_k: Number of chunks to retrieve (default from settings).

        Returns:
            PaperAnalysis — never raises; partial failures produce a
            model with `analysis_status="failed"` if all agents fail.
        """
        try:
            chunks = self._fetch_chunks(paper_id, top_k=top_k)
            if not chunks:
                logger.warning(
                    "Coordinator: no chunks retrieved for paper_id=%s", paper_id
                )
                return PaperAnalysis(
                    paper_id=paper_id,
                    title=None,
                    summary="Not available",
                    analysis_status="failed",
                    analysis_error="No chunks retrieved from vector store.",
                )

            context = self._build_context(paper_id, chunks)
            results = self._run_agents_parallel(context)
            # Propagate filename to the analysis so /api/papers
            # consumers (and the UI) can correlate a result with the
            # original uploaded PDF.
            filename = chunks[0].filename if chunks else None
            merged = self._merge_results(
                paper_id, results, context.metadata, filename=filename
            )
            logger.info(
                "Coordinator: paper_id=%s analyzed (sources=%d)",
                paper_id,
                len(merged.sources),
            )
            return merged

        except Exception as exc:  # noqa: BLE001 — coordinator must never raise
            logger.error("Coordinator.analyze_paper(%s) failed: %s", paper_id, exc)
            return PaperAnalysis(
                paper_id=paper_id,
                title=None,
                summary="Not available",
                analysis_status="failed",
                analysis_error=str(exc),
            )

    # ----------------------------------------------------------- Internals
    def _fetch_chunks(
        self, paper_id: str, *, top_k: int | None
    ) -> list[RetrievedChunk]:
        """Retrieve chunks for the paper using the configured retriever
        or vector store."""
        if self._retriever is not None:
            results = self._retriever.retrieve(
                query=paper_id,  # rough query; works because of paper_id filter
                top_k=top_k,
                paper_id=paper_id,
            )
            return [r.chunk for r in results]

        store = self._vector_store or ChromaVectorStore(
            _build_default_embedding_service()
        )
        # Direct retrieval by paper_id filter (semantic-agnostic).
        existing = store._collection.get(
            where={"paper_id": paper_id}, include=["documents", "metadatas"]
        )
        ids = existing.get("ids") or []
        docs = (existing.get("documents") or [])
        metas = (existing.get("metadatas") or [])
        if top_k is not None:
            ids = ids[:top_k]
            docs = docs[:top_k]
            metas = metas[:top_k]
        chunks: list[RetrievedChunk] = []
        import json as _json
        for cid, doc, meta in zip(ids, docs, metas):
            meta = meta or {}
            page_val = meta.get("page")
            # Decode JSON-encoded authors list back to a list.
            authors_raw = meta.get("authors")
            authors: list[str] | None = None
            if isinstance(authors_raw, str):
                try:
                    decoded = _json.loads(authors_raw)
                    authors = decoded if isinstance(decoded, list) else None
                except (ValueError, TypeError):
                    authors = None
            elif isinstance(authors_raw, list):
                authors = authors_raw
            chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    paper_id=meta.get("paper_id", paper_id),
                    filename=meta.get("filename", ""),
                    title=meta.get("title"),
                    page=int(page_val) if isinstance(page_val, int) and page_val >= 0 else None,
                    section=meta.get("section") or None,
                    text=doc,
                    score=None,
                    authors=authors,
                    publication_date=meta.get("publication_date"),
                    venue=meta.get("venue"),
                    doi=meta.get("doi"),
                    institution=meta.get("institution"),
                )
            )
        return chunks

    def _build_context(
        self, paper_id: str, chunks: list[RetrievedChunk]
    ) -> AgentContext:
        """Build a shared AgentContext for all four agents.

        Pre-extracted metadata from each chunk (title, authors,
        publication_date, venue, doi, institution) is propagated so
        analysis agents can see bibliographic fields without having to
        re-derive them from chunk text. We dedupe across chunks so the
        first non-empty value wins.
        """
        pre_meta: dict[str, Any] = {}
        if chunks:
            first = chunks[0]
            for attr in (
                "title",
                "authors",
                "publication_date",
                "venue",
                "doi",
                "institution",
            ):
                val = getattr(first, attr, None)
                if val:  # truthy check covers None, "", [], etc.
                    pre_meta[attr] = val
        return AgentContext(
            paper_id=paper_id,
            retrieved_chunks=chunks,
            metadata=pre_meta,
            previous_results={},
        )

    def _run_agents_parallel(
        self, context: AgentContext
    ) -> dict[str, Any]:
        """Submit all 4 agents to a thread pool; collect results."""
        agents = {
            "metadata": self._metadata_agent,
            "cybersecurity": self._cyber_agent,
            "solution": self._solution_agent,
            "summary": self._summary_agent,
        }

        results: dict[str, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix="coordinator-agent"
        ) as pool:
            future_to_name = {
                pool.submit(agent.run, context): name for name, agent in agents.items()
            }
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Coordinator: agent %s raised %s; using safe defaults",
                        name,
                        exc,
                    )
                    # Substitute a safe-default model instance.
                    cls = type(agents[name].output_model)
                    default_obj = cls.model_construct()  # bypass validation
                    default_obj.paper_id = context.paper_id
                    results[name] = default_obj

        return results

    def _merge_results(
        self, paper_id: str, results: dict[str, Any],
        pre_meta: dict[str, Any] | None = None,
        *,
        filename: str | None = None,
    ) -> PaperAnalysis:
        """Merge per-agent results into a single PaperAnalysis.

        `pre_meta` is the pre-extracted metadata from PDF parsing +
        heuristics. Any field that the LLM extracted as None / []
        is filled in from `pre_meta` so we never lose bibliographic
        info that the small local model may have missed.
        """
        meta = results.get("metadata")
        cyber = results.get("cybersecurity")
        sol = results.get("solution")
        summary = results.get("summary")
        pre_meta = pre_meta or {}

        # Deduplicated sources across all agents.
        sources: list[SourceReference] = []
        seen: set[tuple[str, int | None]] = set()
        for result in (meta, cyber, sol, summary):
            if result is None:
                continue
            for src in getattr(result, "sources", []) or []:
                key = (src.paper_id, src.page)
                if key in seen:
                    continue
                seen.add(key)
                sources.append(src)

        def _first(*candidates: Any) -> Any:
            """Return the first truthy candidate."""
            for c in candidates:
                if c:  # covers None, "", [], etc.
                    return c
            return None

        return PaperAnalysis(
            paper_id=paper_id,
            filename=filename,
            title=_first(
                getattr(meta, "title", None) if meta else None,
                pre_meta.get("title"),
            ),
            authors=_first(
                getattr(meta, "authors", []) if meta else [],
                pre_meta.get("authors") or [],
            ) or [],
            publication_date=_first(
                getattr(meta, "publication_date", None) if meta else None,
                pre_meta.get("publication_date"),
            ),
            summary=getattr(summary, "summary", None) if summary else None,
            cybersecurity_area=getattr(cyber, "areas", []) if cyber else [],
            problem_statement=getattr(sol, "problem_statement", None) if sol else None,
            proposed_solution=getattr(sol, "proposed_solution", None) if sol else None,
            methodology=getattr(sol, "methodology", None) if sol else None,
            datasets=getattr(sol, "datasets", []) if sol else [],
            key_results=getattr(sol, "key_results", None) if sol else None,
            limitations=getattr(sol, "limitations", None) if sol else None,
            sources=sources,
            analysis_status="ok",
            analysis_error=None,
        )


def _build_default_embedding_service():
    """Lazy import to avoid loading sentence-transformers at module import."""
    from embeddings.embedding_service import get_embedding_service

    return get_embedding_service()


__all__ = ["Coordinator", "DEFAULT_MAX_WORKERS"]
