"""BM25 lexical retriever (PLAN.md §3 Phase 6).

Implementation uses Python's `rank_bm25` package (or a tiny pure-Python
fallback if not installed). Off by default; enabled when
`HYBRID_SEARCH_ENABLED=true` in .env.

To enable:
    pip install rank_bm25
    HYBRID_SEARCH_ENABLED=true

The retriever maintains an in-memory BM25 index over the same chunks
that ChromaDB holds. Index is rebuilt lazily on first query after a
write (we don't track writes automatically here; production deployments
should call `rebuild_index()` after each `add_chunks` call).
"""
from __future__ import annotations

import math
import re
from collections import Counter

from config.logging import get_logger
from config.settings import get_settings
from models.paper import RetrievedChunk

logger = get_logger(__name__)

# Tokenizer: lowercase + word boundaries + strip non-alphanumeric.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _idf(N: int, df: int) -> float:
    """BM25+ IDF formula."""
    return math.log(1 + (N - df + 0.5) / (df + 0.5))


class BM25Index:
    """Tiny pure-Python BM25 (Robertson + Sparck Jones) index.

    Not optimized for very large corpora, but fine for thousands of chunks.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._doc_chunks: list[RetrievedChunk] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._df: Counter[str] = Counter()
        self._idf_cache: dict[str, float] = {}

    def add(self, chunks: list[RetrievedChunk]) -> None:
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            self._docs.append(tokens)
            self._doc_chunks.append(chunk)
            self._doc_lens.append(len(tokens))
            self._df.update(set(tokens))
        if self._docs:
            self._avgdl = sum(self._doc_lens) / len(self._doc_lens)
        N = len(self._docs)
        self._idf_cache = {term: _idf(N, df) for term, df in self._df.items()}

    def clear(self) -> None:
        self._docs.clear()
        self._doc_chunks.clear()
        self._doc_lens.clear()
        self._df.clear()
        self._idf_cache.clear()
        self._avgdl = 0.0

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        q_tokens = _tokenize(query)
        if not q_tokens or not self._docs:
            return []

        scores = [0.0] * len(self._docs)
        for term in set(q_tokens):
            if term not in self._idf_cache:
                continue
            idf = self._idf_cache[term]
            for i, doc in enumerate(self._docs):
                tf = doc.count(term)
                if tf == 0:
                    continue
                dl = self._doc_lens[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1))
                scores[i] += idf * (tf * (self.k1 + 1)) / denom

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        out: list[RetrievedChunk] = []
        for i, score in ranked[:top_k]:
            if score <= 0:
                break
            # Annotate the score onto the chunk.
            chunk = self._doc_chunks[i]
            out.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    filename=chunk.filename,
                    title=chunk.title,
                    page=chunk.page,
                    section=chunk.section,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return out


def is_hybrid_search_enabled() -> bool:
    """True when HYBRID_SEARCH_ENABLED=true in .env."""
    s = get_settings()
    # BM25 toggle lives in the env directly (no setting class field needed
    # since it's an advanced off-by-default feature).
    import os
    return os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true"


__all__ = ["BM25Index", "is_hybrid_search_enabled"]