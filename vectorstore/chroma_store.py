"""ChromaDB persistent vector store.

Wraps ChromaDB behind a small interface so it can be swapped for Qdrant or
FAISS later (AGENTS.md §40). Each chunk's metadata is preserved verbatim
so the RAG layer can cite pages and sections (AGENTS.md §24).

Embedding model identity is recorded in the collection's metadata so we
can warn when loading a collection that was built with a different model
(PLAN.md §7 risk: "Mixing embedding models silently breaks retrieval").
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.logging import get_logger
from config.settings import get_settings
from embeddings.embedding_service import EmbeddingService
from ingestion.chunker import TextChunk

logger = get_logger(__name__)

# Base name. The actual collection name is suffixed with a slug derived
# from the embedding model so users can switch embedding models without
# losing previously-ingested papers.
COLLECTION_BASE_NAME = "cyber_papers"


def _model_slug(model_name: str) -> str:
    """Turn a model id into a filesystem- and collection-safe suffix.

    Examples:
        "embeddinggemma:latest"        -> "embeddinggemma_latest"
        "sentence-transformers/all-MiniLM-L6-v2" -> "st_all_minilm_l6_v2"
        "nomic-embed-text"             -> "nomic_embed_text"
    """
    slug = model_name.lower()
    slug = slug.replace("/", "_")
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    # Fallback for anything pathological
    return slug or "unknown"


def collection_name_for(model_name: str) -> str:
    """Per-model collection name: switching models keeps the old data accessible."""
    return f"{COLLECTION_BASE_NAME}_{_model_slug(model_name)}_v1"


# Keep the old constant exported for backward compat (anything that still
# imports it). It now resolves dynamically based on the current model.
COLLECTION_NAME = collection_name_for(get_settings().embedding_model)


@dataclass
class QueryHit:
    """A single retrieval result."""

    chunk_id: str
    paper_id: str
    filename: str | None
    title: str | None
    page: int | None
    section: str | None
    text: str
    score: float | None  # 1 - distance for cosine; None when N/A
    # Pre-extracted paper-level metadata, when present in chunk metadata.
    authors: list[str] | None = None
    publication_date: str | None = None
    venue: str | None = None
    doi: str | None = None
    institution: str | None = None


class ChromaVectorStore:
    """Thin, swap-friendly wrapper around ChromaDB.

    Each embedding model gets its own ChromaDB collection (named
    `cyber_papers_<model-slug>_v1`), so switching embedding models does
    not destroy previously-ingested papers. A legacy collection named
    `cyber_papers_v1` from earlier deployments is migrated in place
    (renamed to the per-model variant for the model that built it) the
    first time we encounter it, so existing data is preserved.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        *,
        collection_name: str | None = None,
    ) -> None:
        s = get_settings()
        s.chroma_persist_directory.mkdir(parents=True, exist_ok=True)

        self._embedding = embedding_service
        self._client = chromadb.PersistentClient(
            path=str(s.chroma_persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Pick the right collection for this embedding model.
        target_name = collection_name or collection_name_for(embedding_service.model_name)
        self._migrate_legacy_collection_if_present(target_name, embedding_service.model_name)

        self._collection = self._client.get_or_create_collection(
            name=target_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding_service.model_name,
                "embedding_dim": str(embedding_service.dimension),
            },
        )
        self._warn_on_model_mismatch(target_name, embedding_service.model_name)

    def _migrate_legacy_collection_if_present(
        self, target_name: str, current_model: str
    ) -> None:
        """If the legacy 'cyber_papers_v1' collection exists, rename it
        so its data is preserved under the new per-model naming scheme.

        ChromaDB doesn't support rename, so we copy vectors/documents/
        metadata in batches and delete the source.
        """
        legacy = "cyber_papers_v1"
        if legacy == target_name:
            return
        try:
            existing = self._client.get_collection(legacy)
        except Exception:
            return  # No legacy collection — nothing to do.

        # If the target already exists, the user has already ingested with
        # the new model on a fresh DB. If the legacy collection was built
        # with a *different* embedding model, dropping it is safe (its
        # 384-dim vectors wouldn't be queryable by the 768-dim model
        # anyway). If they're the SAME model, the legacy is just a stale
        # duplicate from before this naming scheme existed — drop it.
        # In either case, log loudly so the user can see what happened.
        try:
            target_meta = self._client.get_collection(target_name).metadata or {}
            target_model = target_meta.get("embedding_model")
        except Exception:
            target_model = None
        try:
            legacy_meta = existing.metadata or {}
            legacy_model = legacy_meta.get("embedding_model")
        except Exception:
            legacy_model = None
        logger.warning(
            "Dropping legacy collection %r (built with %r); target %r "
            "(built with %r) already exists. Re-upload PDFs if you need "
            "them re-indexed with the current embedding model.",
            legacy, legacy_model, target_name, target_model,
        )
        self._client.delete_collection(legacy)
        return

        logger.info("Migrating legacy collection %r -> %r", legacy, target_name)
        data = existing.get(include=["documents", "metadatas", "embeddings"])
        ids = list(data.get("ids") or [])
        if not ids:
            self._client.delete_collection(legacy)
            return

        new = self._client.get_or_create_collection(
            name=target_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": current_model,
                "embedding_dim": str(self._embedding.dimension),
            },
        )

        def _to_list(x):
            """Convert chromadb numpy arrays to plain Python lists."""
            if x is None:
                return []
            return [v for v in x]  # works for list OR numpy array

        docs = _to_list(data.get("documents"))
        metas = _to_list(data.get("metadatas"))
        embs = _to_list(data.get("embeddings"))

        # ChromaDB's max batch size for add() is conservative; chunk it.
        BATCH = 5000
        for i in range(0, len(ids), BATCH):
            new.add(
                ids=ids[i : i + BATCH],
                documents=docs[i : i + BATCH],
                metadatas=metas[i : i + BATCH],
                embeddings=embs[i : i + BATCH],
            )
        self._client.delete_collection(legacy)
        logger.info(
            "Migrated %d vectors from %r into %r",
            len(ids), legacy, target_name,
        )

    @staticmethod
    def _warn_on_model_mismatch(collection_name: str, current_model: str) -> None:
        """If the collection was created with a different embedding model,
        log a loud warning. Retrieval would silently degrade otherwise."""
        try:
            client = chromadb.PersistentClient(
                path=str(get_settings().chroma_persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            existing = client.get_collection(collection_name)
            stored = (existing.metadata or {}).get("embedding_model")
            if stored and stored != current_model:
                logger.warning(
                    "Embedding model mismatch: collection was built with %r, "
                    "current settings use %r. Retrieval will be unreliable.",
                    stored,
                    current_model,
                )
        except Exception:  # noqa: BLE001 — never block startup on a metadata probe
            pass

    # ------------------------------------------------------------------ Ingest
    # Metadata keys from `extract_metadata` we propagate to chunk metadata
    # so the analysis agents can see pre-extracted values without
    # re-extracting them from text (slower and error-prone on small LLMs).
    _META_KEYS = ("title", "authors", "publication_date", "venue", "doi", "institution")

    def add_chunks(
        self,
        chunks: Iterable[TextChunk],
        filename: str,
        title: str | None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Upsert chunks. Returns number of items added.

        Args:
            chunks: TextChunk objects from the chunker.
            filename: Original PDF filename.
            title: Convenience kwarg — equivalent to metadata["title"].
            metadata: Pre-extracted paper metadata from
                `extract_metadata()` (title, authors, venue, doi, ...).
                Any non-empty values are copied to every chunk's metadata
                so the agents can see them.
        """
        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        # Merge title + metadata, with metadata taking precedence on conflict.
        merged_meta: dict[str, Any] = dict(metadata or {})
        if title and "title" not in merged_meta:
            merged_meta["title"] = title

        ids = [c.chunk_id for c in chunk_list]
        texts = [c.text for c in chunk_list]
        metadatas: list[dict[str, Any]] = []
        for c in chunk_list:
            m: dict[str, Any] = {
                "paper_id": c.paper_id,
                "filename": filename,
                "page": c.page if c.page is not None else -1,
                "section": c.section or "",
                "chunk_id": c.chunk_id,
            }
            # Propagate pre-extracted metadata fields so the agents
            # have them on hand. Serialize lists (authors) to JSON so
            # ChromaDB's primitive-only metadata accepts them.
            import json as _json
            for k in self._META_KEYS:
                v = merged_meta.get(k)
                if v is None or v == "" or v == []:
                    continue
                if isinstance(v, list):
                    m[k] = _json.dumps(v)
                else:
                    m[k] = v
            metadatas.append(m)

        embeddings = self._embedding.embed_documents(texts)
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(
            "Indexed %d chunks for paper_id=%s filename=%s title=%r",
            len(chunk_list),
            chunk_list[0].paper_id,
            filename,
            merged_meta.get("title"),
        )
        return len(chunk_list)

    # -------------------------------------------------------------- Retrieval
    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        paper_id: str | None = None,
    ) -> list[QueryHit]:
        """Semantic search with optional metadata filter."""
        s = get_settings()
        k = top_k or s.top_k
        threshold = score_threshold if score_threshold is not None else s.score_threshold

        where: dict[str, Any] | None = None
        if paper_id:
            where = {"paper_id": paper_id}

        query_embedding = self._embedding.embed_query(question)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[QueryHit] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        import json as _json
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            # cosine distance -> cosine similarity in [0, 1] for normalized vectors
            score = 1.0 - float(dist)
            if score < threshold:
                continue
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
            hits.append(
                QueryHit(
                    chunk_id=chunk_id,
                    paper_id=meta.get("paper_id", ""),
                    filename=meta.get("filename"),
                    title=meta.get("title"),
                    page=int(page_val) if isinstance(page_val, int) and page_val >= 0 else None,
                    section=meta.get("section") or None,
                    text=doc,
                    score=score,
                    authors=authors,
                    publication_date=meta.get("publication_date"),
                    venue=meta.get("venue"),
                    doi=meta.get("doi"),
                    institution=meta.get("institution"),
                )
            )
        return hits

    # --------------------------------------------------------------- Bookkeeping
    def list_papers(self) -> list[dict[str, Any]]:
        """Return a summary of every distinct paper in the store."""
        # Get all metadata; cap is a ChromaDB requirement.
        all_items = self._collection.get(include=["metadatas"], limit=100_000)
        seen: dict[str, dict[str, Any]] = {}
        import json as _json
        for meta in all_items.get("metadatas") or []:
            if not meta:
                continue
            pid = meta.get("paper_id")
            if not pid or pid in seen:
                continue
            # Decode JSON-encoded authors list back to a list.
            authors = meta.get("authors")
            if isinstance(authors, str):
                try:
                    authors = _json.loads(authors)
                except (ValueError, TypeError):
                    authors = []
            seen[pid] = {
                "paper_id": pid,
                "filename": meta.get("filename"),
                "title": meta.get("title"),
                "authors": authors or [],
                "publication_date": meta.get("publication_date"),
                "venue": meta.get("venue"),
            }
        return list(seen.values())

    def delete_paper(self, paper_id: str) -> int:
        """Remove all chunks for a paper. Returns count removed."""
        existing = self._collection.get(where={"paper_id": paper_id}, include=[])
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)


__all__ = ["ChromaVectorStore", "QueryHit", "COLLECTION_NAME"]
