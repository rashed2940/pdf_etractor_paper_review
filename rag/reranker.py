"""Cross-encoder rerank hook (PLAN.md §3 Phase 6).

Interface for plugging in a reranker. The default implementation is a
no-op (preserves original ordering). Users can enable a real cross-encoder
by setting RERANKER_ENABLED=true and providing RERANKER_MODEL.

To enable a real reranker, uncomment the SentenceTransformersCrossEncoder
block below and install the optional `sentence-transformers` cross-encoder
model. Kept off by default to avoid model download + latency for users
who don't need it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from config.logging import get_logger
from models.paper import RetrievedChunk

logger = get_logger(__name__)


class Reranker(ABC):
    """Abstract reranker interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Re-score and reorder `chunks` for the given `query`."""


class NoOpReranker(Reranker):
    """Default: returns chunks unchanged."""

    @property
    def name(self) -> str:
        return "noop"

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        return chunks[:top_k] if top_k is not None else chunks


class SentenceTransformersCrossEncoder(Reranker):
    """Optional: rerank using a sentence-transformers cross-encoder model.

    Uncomment + install `sentence-transformers` (already installed) and
    download a cross-encoder model like `cross-encoder/ms-marco-MiniLM-L-6-v2`.

    Example activation (settings.py + env):
        RERANKER_ENABLED=true
        RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        from sentence_transformers import CrossEncoder

        logger.info("Loading CrossEncoder model: %s", model_name)
        self._model = CrossEncoder(model_name)

    @property
    def name(self) -> str:
        return f"cross-encoder:{self._model_name}"

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        ranked = [c for c, _ in scored]
        return ranked[:top_k] if top_k is not None else ranked


def get_reranker() -> Reranker:
    """Factory: returns the configured reranker.

    Per PLAN.md §3 Phase 6, this is off by default. To enable a real
    reranker, add `RERANKER_ENABLED=true` and `RERANKER_MODEL=...` to .env
    and switch the `else` branch to `SentenceTransformersCrossEncoder(...)`.
    """
    # Default: no-op (cheap, deterministic, no model load)
    return NoOpReranker()

    # --- Alternative: real cross-encoder (uncomment to switch) ---
    # import os
    # if os.getenv("RERANKER_ENABLED", "false").lower() == "true":
    #     model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    #     return SentenceTransformersCrossEncoder(model)
    # return NoOpReranker()


__all__ = [
    "Reranker",
    "NoOpReranker",
    "SentenceTransformersCrossEncoder",
    "get_reranker",
]