"""Background runner for PDF processing (PLAN.md §5, AGENTS.md §21).

Owns the threading.Thread per PDF that:
1. Updates `ProcessingStatusRegistry` as it progresses.
2. Calls `paper_service.ingest_pdf()` to extract + chunk + embed.
3. Calls `analysis_service.AnalysisService.analyze_paper()` for analysis.
4. Stores the final `PaperAnalysis` in `ResultsStore`.

Per-paper exceptions are caught and logged so one bad PDF doesn't
break the batch (AGENTS.md §27).

Concurrency (why this file has a semaphore)
------------------------------------------
Each paper spawns 4 agent threads (Metadata / Cybersecurity / Solution /
Summary), and each agent makes an LLM call. Spawning one thread per
PDF therefore means `N_papers x 4` simultaneous LLM requests. Cloud
hosts (e.g. "minimax-m3:cloud") enforce a per-account concurrency
limit and answer the surplus with HTTP 429. Those 429s were burning
all of an agent's retries instantly and adding ~40% wall-clock
overhead (measured: 3,374 rate-limit hits over a 71-paper batch).

We therefore gate the *paper* level with a semaphore so at most
`PAPER_CONCURRENCY` papers are in flight at once. Tune it with the
`PAPER_CONCURRENCY` env var (default 3 => max 12 concurrent LLM calls).
"""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

from config.logging import get_logger
from config.settings import get_settings
from models.paper import PaperAnalysis
from services.analysis_service import AnalysisService
from services.paper_service import IngestionResult, ingest_pdf
from services.processing_status import get_registry
from services.results_store import get_results_store
from vectorstore.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Global paper-level concurrency gate
# ---------------------------------------------------------------------------
# Created lazily so the configured value is read at call time (and so
# tests that change settings are honoured). We use a plain Semaphore
# rather than a ThreadPoolExecutor because the existing design spawns
# a daemon thread per PDF and reports per-paper status chips; the
# semaphore just throttles how many run at once.
_paper_semaphore: threading.Semaphore | None = None
_paper_semaphore_lock = threading.Lock()


def _get_paper_semaphore() -> threading.Semaphore:
    """Return the process-wide paper concurrency semaphore."""
    global _paper_semaphore
    if _paper_semaphore is None:
        with _paper_semaphore_lock:
            if _paper_semaphore is None:
                limit = max(1, int(get_settings().paper_concurrency))
                _paper_semaphore = threading.Semaphore(limit)
                logger.info(
                    "Paper concurrency gate: max %d paper(s) in flight "
                    "(~%d concurrent LLM calls)",
                    limit,
                    limit * 4,
                )
    return _paper_semaphore


#: How long a paper may wait for a processing slot before giving up.
#: Generous enough for a large upload (many papers queued), but finite
#: so a stuck batch does not leave chips hanging forever.
_QUEUE_WAIT_TIMEOUT_SECONDS = 3600.0


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

    The whole body is gated by the process-wide paper semaphore so a
    bulk upload does not flood the LLM provider with `4 x N_papers`
    simultaneous requests (which triggers HTTP 429 rate limits).
    """
    registry = get_registry()
    store = get_results_store()
    sem = _get_paper_semaphore()

    # Acquire the gate before doing any work. The status chip says
    # "queued" so the UI shows why nothing is happening yet.
    acquired = sem.acquire(timeout=_QUEUE_WAIT_TIMEOUT_SECONDS)
    if not acquired:
        registry.update(
            upload_id, paper_id,
            state="failed", progress=0.0,
            message="Timed out waiting for a processing slot",
        )
        logger.error(
            "Paper %s gave up waiting for a processing slot (%ss)",
            paper_id, _QUEUE_WAIT_TIMEOUT_SECONDS,
        )
        return

    try:
        registry.update(upload_id, paper_id, state="extracting", progress=0.1, message="Reading PDF")

        # ---- Ingest phase ----
        registry.update(upload_id, paper_id, state="chunking", progress=0.3, message="Chunking + embedding")
        ingest_result: IngestionResult = ingest_pdf(path, vector_store=vector_store)
        if ingest_result.is_duplicate:
            # Chunks already exist for this paper. Re-run analysis so the
            # user gets a result row even if their first attempt failed.
            # The chip stays "Done" but with a clearer message.
            logger.info(
                "Duplicate detected for %s (sha256=%s); re-running analysis.",
                _safe_name(path), ingest_result.sha256[:12],
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
            real_paper_id,
            _safe_name(path),
        )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=4)
        logger.error(
            "Background run failed for %s: %s\n%s", _safe_name(path), exc, tb
        )
        registry.mark_failed(upload_id, paper_id, f"{exc}")
    finally:
        # Always release the slot, including the early `return` in the
        # duplicate-detection branch above.
        sem.release()


def _safe_name(path: Path | str) -> str:
    """Return the file name for logging, tolerating str paths."""
    try:
        return Path(path).name
    except Exception:  # noqa: BLE001
        return str(path)


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