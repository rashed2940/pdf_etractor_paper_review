"""Hybrid search: combine semantic (ChromaDB) + BM25 lexical results.

Uses Reciprocal Rank Fusion (RRF) to merge the two ranked lists without
needing to normalize scores. RRF is robust to score-scale differences
between the two retrievers.

Off by default (HYBRID_SEARCH_ENABLED=false). When off, the Retriever
behaves exactly as before (semantic only).
"""
from __future__ import annotations

from config.logging import get_logger
from models.paper import RetrievedChunk
from rag.bm25 import BM25Index, is_hybrid_search_enabled
from rag.retriever import Retriever

logger = get_logger(__name__)

# RRF constant: standard k=60 from Cormack et al. (2009).
RRF_K = 60


class HybridRetriever:
    """Combines semantic + BM25 via Reciprocal Rank Fusion.

    Lifecycle:
    1. Construct once (lazy: doesn't build BM25 index until first query).
    2. Call `rebuild_bm25_index(chunks)` after each ingest (or rely on
       the auto-rebuild on first query if `auto_rebuild=True`).
    3. `retrieve(query, top_k)` returns merged results.

    Falls back gracefully to semantic-only when hybrid is disabled or
    when BM25 index is empty.
    """

    def __init__(
        self,
        retriever: Retriever,
        *,
        auto_rebuild: bool = True,
    ) -> None:
        self._retriever = retriever
        self._bm25 = BM25Index()
        self._auto_rebuild = auto_rebuild
        self._rebuilt_at_least_once = False

    @property
    def bm25(self) -> BM25Index:
        return self._bm25

    def rebuild_bm25_index(self, chunks: list[RetrievedChunk]) -> None:
        """Replace the BM25 index with `chunks`."""
        self._bm25.clear()
        self._bm25.add(chunks)
        self._rebuilt_at_least_once = True
        logger.info("BM25 index rebuilt with %d chunks", len(chunks))

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        paper_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieve: semantic + BM25 merged via RRF.

        Returns up to `top_k` chunks.
        """
        s = self._retriever._ensure_store()
        default_top_k = top_k or s._embedding.dimension and 5  # noqa
        # Use settings.top_k if not specified
        if top_k is None:
            from config.settings import get_settings as _gs

            top_k = _gs().top_k

        semantic_results = self._retriever.retrieve(
            query=query, top_k=top_k, paper_id=paper_id
        )
        semantic_chunks = [r.chunk for r in semantic_results]

        # If hybrid is off or BM25 is empty, return semantic results.
        if not is_hybrid_search_enabled() or not self._bm25._docs:
            # Optional auto-rebuild on first query if enabled.
            if (
                is_hybrid_search_enabled()
                and self._auto_rebuild
                and not self._rebuilt_at_least_once
            ):
                self._maybe_auto_rebuild(s)
            if not self._bm25._docs:
                return semantic_chunks

        bm25_chunks = self._bm25.search(query, top_k=top_k)

        # Merge via RRF.
        merged = self._rrf_merge(semantic_chunks, bm25_chunks, top_k=top_k)
        return merged

    def _maybe_auto_rebuild(self, store) -> None:
        """Build BM25 index from all chunks in the vector store."""
        try:
            all_items = store._collection.get(include=["documents", "metadatas"])
            ids = all_items.get("ids") or []
            docs = all_items.get("documents") or []
            metas = all_items.get("metadatas") or []
            chunks: list[RetrievedChunk] = []
            for cid, doc, meta in zip(ids, docs, metas):
                meta = meta or {}
                page_val = meta.get("page")
                chunks.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        paper_id=meta.get("paper_id", ""),
                        filename=meta.get("filename", ""),
                        title=meta.get("title"),
                        page=int(page_val)
                        if isinstance(page_val, int) and page_val >= 0
                        else None,
                        section=meta.get("section") or None,
                        text=doc,
                        score=None,
                    )
                )
            self.rebuild_bm25_index(chunks)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 auto-rebuild failed: %s", exc)

    @staticmethod
    def _rrf_merge(
        list_a: list[RetrievedChunk],
        list_b: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank_i))."""
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(list_a, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            chunk_map[cid] = chunk

        for rank, chunk in enumerate(list_b, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            chunk_map[cid] = chunk

        ranked_ids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        return [chunk_map[cid] for cid in ranked_ids[:top_k]]


__all__ = ["HybridRetriever", "RRF_K"]