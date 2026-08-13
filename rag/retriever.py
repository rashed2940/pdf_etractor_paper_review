"""Retrieval interface (AGENTS.md §10).

Thin wrapper around `vectorstore.chroma_store.ChromaVectorStore.query`
that:
- Maps our domain types (RetrievedChunk, SourceReference).
- Supports filtering by paper_id and by cybersecurity_area (post-filter
  on metadata if present; otherwise no-op for now since chunk-level
  cybersecurity_area is set in Phase 3).
- Exposes the configuration knobs from settings (top_k, score_threshold).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from config.logging import get_logger
from config.settings import get_settings
from models.paper import RetrievedChunk, SourceReference
from vectorstore.chroma_store import ChromaVectorStore, QueryHit

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """One retrieved chunk as the RAG layer sees it."""

    chunk: RetrievedChunk
    source: SourceReference


class Retriever:
    """Wraps the vector store and normalizes hits into our domain types."""

    def __init__(self, store: ChromaVectorStore | None = None) -> None:
        self._store = store

    def _ensure_store(self) -> ChromaVectorStore:
        if self._store is None:
            # Lazy import to avoid loading embeddings at import-time
            from embeddings.embedding_service import get_embedding_service

            self._store = ChromaVectorStore(get_embedding_service())
        return self._store

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        paper_id: str | None = None,
        cybersecurity_area: str | None = None,
    ) -> list[RetrievalResult]:
        """Semantic search over the vector store.

        Args:
            question: Natural-language question.
            top_k: Override settings.top_k.
            score_threshold: Override settings.score_threshold.
            paper_id: Restrict to a single paper.
            cybersecurity_area: Optional metadata filter (post-filter:
                applied after retrieval since chunk-level area is set
                in Phase 3 by the CybersecurityAgent and may not be on
                the chunk yet).

        Returns:
            List of RetrievalResult sorted by descending similarity.
        """
        s = get_settings()
        k = top_k or s.top_k
        threshold = score_threshold if score_threshold is not None else s.score_threshold

        store = self._ensure_store()
        hits: list[QueryHit] = store.query(
            question,
            top_k=k,
            score_threshold=threshold,
            paper_id=paper_id,
        )

        if cybersecurity_area:
            hits = self._post_filter_by_area(hits, cybersecurity_area)

        results: list[RetrievalResult] = []
        for hit in hits:
            chunk = RetrievedChunk(
                chunk_id=hit.chunk_id,
                paper_id=hit.paper_id,
                filename=hit.filename or "",
                title=hit.title,
                page=hit.page,
                section=hit.section,
                text=hit.text,
                score=hit.score,
                authors=hit.authors,
                publication_date=hit.publication_date,
                venue=hit.venue,
                doi=hit.doi,
                institution=hit.institution,
            )
            source = SourceReference(
                paper_id=hit.paper_id,
                page=hit.page,
                section=hit.section,
                chunk_id=hit.chunk_id,
                text=(hit.text[:300] + "…") if len(hit.text) > 300 else hit.text,
            )
            results.append(RetrievalResult(chunk=chunk, source=source))

        logger.debug(
            "Retriever: question=%r -> %d hits (k=%d threshold=%.2f)",
            question[:80],
            len(results),
            k,
            threshold,
        )
        return results

    @staticmethod
    def _post_filter_by_area(
        hits: Iterable[QueryHit], cybersecurity_area: str
    ) -> list[QueryHit]:
        """Best-effort post-filter on cybersecurity_area.

        Tries to interpret metadata hints that may be present on chunks
        in later phases. For now this only filters when the metadata
        is present; otherwise it's a no-op.
        """
        area_lc = cybersecurity_area.lower()
        kept: list[QueryHit] = []
        for h in hits:
            section_lc = (h.section or "").lower()
            if area_lc in section_lc:
                kept.append(h)
        return kept or list(hits)  # don't drop all if no chunks have area metadata


__all__ = ["RetrievalResult", "Retriever"]
