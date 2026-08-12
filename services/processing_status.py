"""Thread-safe registry tracking per-paper processing status.

Used by the background processing threads spawned by the Flask upload
endpoint (PLAN.md §5). The UI polls /status/<upload_id> to render progress
chips. All mutations go through a single lock; reads are lock-free for
the common case by snapshotting a shallow copy.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PaperStatus:
    """Status of a single PDF being processed."""

    paper_id: str
    filename: str
    state: str = "queued"  # queued | extracting | chunking | embedding | indexing | analyzing | done | failed
    progress: float = 0.0  # 0.0 .. 1.0
    message: str = ""
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProcessingStatusRegistry:
    """In-memory, thread-safe registry of upload + paper statuses.

    An upload groups one or more papers. The UI polls by upload_id and
    receives a snapshot of every paper's current state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # upload_id -> { paper_id -> PaperStatus }
        self._uploads: dict[str, dict[str, PaperStatus]] = {}

    # -------- Upload lifecycle --------
    def create_upload(self, filenames: list[str]) -> str:
        """Create a new upload session; returns upload_id."""
        upload_id = uuid.uuid4().hex
        with self._lock:
            self._uploads[upload_id] = {
                fname: PaperStatus(paper_id=fname, filename=fname)
                for fname in filenames
            }
        return upload_id

    def get_upload(self, upload_id: str) -> dict[str, PaperStatus]:
        """Return a snapshot of all papers in this upload."""
        with self._lock:
            return dict(self._uploads.get(upload_id, {}))

    def list_uploads(self) -> list[str]:
        with self._lock:
            return list(self._uploads.keys())

    # -------- Per-paper updates --------
    def update(
        self,
        upload_id: str,
        paper_id: str,
        *,
        state: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            upload = self._uploads.get(upload_id)
            if not upload or paper_id not in upload:
                return
            s = upload[paper_id]
            if state is not None:
                s.state = state
            if progress is not None:
                s.progress = max(0.0, min(1.0, progress))
            if message is not None:
                s.message = message
            if error is not None:
                s.error = error
                s.state = "failed"
            s.updated_at = time.time()

    def mark_done(self, upload_id: str, paper_id: str) -> None:
        self.update(upload_id, paper_id, state="done", progress=1.0, message="Completed")

    def mark_failed(self, upload_id: str, paper_id: str, error: str) -> None:
        self.update(upload_id, paper_id, state="failed", progress=1.0, error=error)


# Singleton accessor; cheap because all state lives behind it.
_registry: ProcessingStatusRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ProcessingStatusRegistry:
    """Return the process-wide ProcessingStatusRegistry singleton."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ProcessingStatusRegistry()
    return _registry


__all__ = [
    "PaperStatus",
    "ProcessingStatusRegistry",
    "get_registry",
]
