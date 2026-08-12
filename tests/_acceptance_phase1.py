"""Phase 1 end-to-end acceptance test.

Run with:
    python tests/_acceptance_phase1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on sys.path when run from any cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.logging import setup_logging, get_logger  # noqa: E402
from config.settings import ensure_directories  # noqa: E402
from services.paper_service import ingest_pdf  # noqa: E402
from vectorstore.chroma_store import ChromaVectorStore  # noqa: E402
from embeddings.embedding_service import get_embedding_service  # noqa: E402


def main() -> int:
    setup_logging()
    log = get_logger("phase1_test")
    ensure_directories()

    pdf_path = ROOT / "data" / "_test_pdfs" / "sample_paper.pdf"
    if not pdf_path.exists():
        print(f"ERROR: Test PDF not found at {pdf_path}")
        return 1

    print(f"=== Ingesting: {pdf_path} ===")

    print("Loading embedding model (first run downloads the model)...")
    embedding = get_embedding_service()
    print(f"Embedding model: {embedding.model_name} (dim={embedding.dimension})")

    store = ChromaVectorStore(embedding)

    # Wipe test data so the run is reproducible.
    try:
        all_ids = store._collection.get(include=[])["ids"]
        if all_ids:
            store._collection.delete(ids=all_ids)
            print(f"Wiped {len(all_ids)} pre-existing chunks for reproducibility.")
    except Exception as e:  # noqa: BLE001
        print(f"(Wipe note: {e})")

    # === Ingest ===
    result = ingest_pdf(pdf_path, vector_store=store)
    print("\n=== Ingestion Result ===")
    print(f"  paper_id    : {result.paper_id}")
    print(f"  filename    : {result.filename}")
    print(f"  sha256      : {result.sha256[:16]}...")
    print(f"  page_count  : {result.page_count}")
    print(f"  chunk_count : {result.chunk_count}")
    print(f"  is_duplicate: {result.is_duplicate}")
    print(f"  metadata    : {result.metadata}")

    # === Acceptance: query returns relevant chunks with page numbers ===
    print("\n=== Query Test (semantic retrieval) ===")
    hits = store.query("intrusion detection neural network", top_k=3)
    for i, h in enumerate(hits, 1):
        print(
            f"  Hit {i}: paper={h.paper_id[:18]}... page={h.page} "
            f"section={h.section!r} score={h.score:.3f}"
        )
        print(f"         text: {h.text[:120]}...")
    assert len(hits) >= 1, "No hits returned"
    assert all(h.page is not None for h in hits), "Hits missing page numbers"

    # === Duplicate detection ===
    print("\n=== Duplicate Detection ===")
    result2 = ingest_pdf(pdf_path, vector_store=store)
    assert result2.is_duplicate, "Duplicate not detected"
    print(f"  is_duplicate correctly: {result2.is_duplicate}")

    print("\nPHASE 1 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
