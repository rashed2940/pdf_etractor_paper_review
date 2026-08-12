"""Background runner for PDF processing (PLAN.md §5, AGENTS.md §21).

Owns the threading.Thread per PDF that:
1. Updates `ProcessingStatusRegistry` as it progresses.
2. Calls `paper_service.ingest_pdf()` to extract + chunk + embed.
3. Calls `analysis_service.AnalysisService.analyze_paper()` for analysis.
4. Stores the final `PaperAnalysis` in `ResultsStore`.

Per-paper exceptions are caught and logged so one bad PDF doesn't
break the batch (AGENTS.md §27).
"""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

from config.logging import get_logger
from models.paper import PaperAnalysis
from services.analysis_service import AnalysisService
from services.paper_service import IngestionResult, ingest_pdf
from services.processing_status import get_registry
from services.results_store import get_results_store
from vectorstore.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


def _run_for_paper(
    upload_id: str,
    paper_id: str,
    path: Path,
    *,
    vector_store: ChromaVectorStore | None,
    analysis_service: AnalysisService | None,
) -> None:
    """Process one PDF end-to-end (ingest + analyze) with status updates.

    Designed to run inside a daemon Thread. Exceptions are caught and
    surfaced via the status registry; never propagated.
    """
    registry = get_registry()
    store = get_results_store()
    registry.update(upload_id, paper_id, state="extracting", progress=0.1, message="Reading PDF")

    try:
        # ---- Ingest phase ----
        registry.update(upload_id, paper_id, state="chunking", progress=0.3, message="Chunking + embedding")
        ingest_result: IngestionResult = ingest_pdf(path, vector_store=vector_store)
        if ingest_result.is_duplicate:
            # Chunks already exist for this paper. Re-run analysis so the
            # user gets a result row even if their first attempt failed.
            # The chip stays "Done" but with a clearer message.
            logger.info(
                "Duplicate detected for %s (sha256=%s); re-running analysis.",
                path.name, ingest_result.sha256[:12],
            )
            registry.update(
                upload_id, paper_id,
                state="analyzing", progress=0.6,
                message="Duplicate detected; re-running analysis",
            )
            real_paper_id = ingest_result.paper_id
            svc = analysis_service or AnalysisService()
            analysis = svc.analyze_paper(real_paper_id)
            store.put(real_paper_id, analysis)
            registry.mark_done(upload_id, paper_id)
            logger.info(
                "Background run OK (duplicate re-analyzed): paper_id=%s",
                real_paper_id,
            )
            return

        real_paper_id = ingest_result.paper_id

        # ---- Analysis phase ----
        registry.update(upload_id, paper_id, state="analyzing", progress=0.6, message="Running 4 analysis agents in parallel")
        svc = analysis_service or AnalysisService()
        analysis: PaperAnalysis = svc.analyze_paper(real_paper_id)

        # Persist result for the UI
        store.put(real_paper_id, analysis)

        registry.mark_done(upload_id, paper_id)
        logger.info(
            "Background run OK: paper_id=%s filename=%s",
            real_paper_id, path.name,
        )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=4)
        logger.error("Background run failed for %s: %s\n%s", path.name, exc, tb)
        registry.mark_failed(upload_id, paper_id, f"{exc}")


def start_background_processing(
    upload_id: str,
    paths_by_paper_id: dict[str, Path],
    *,
    vector_store: ChromaVectorStore | None = None,
    analysis_service: AnalysisService | None = None,
) -> list[threading.Thread]:
    """Spawn one daemon thread per PDF; return the live threads.

    Each thread targets `_run_for_paper`. Threads are daemon so the
    Flask process can exit cleanly even if processing is in flight.
    """
    threads: list[threading.Thread] = []
    for paper_id, path in paths_by_paper_id.items():
        t = threading.Thread(
            target=_run_for_paper,
            name=f"paper-{paper_id[:8]}",
            args=(upload_id, paper_id, path),
            kwargs={"vector_store": vector_store, "analysis_service": analysis_service},
            daemon=True,
        )
        t.start()
        threads.append(t)
    return threads


__all__ = ["start_background_processing", "_run_for_paper"]