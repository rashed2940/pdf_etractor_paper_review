"""Results store with optional SQLite persistence (PLAN.md §3 Phase 6).

Phase 5's ResultsStore was in-memory only. This version optionally
persists PaperAnalysis rows to SQLite so analyses survive Flask restarts.

The store is thread-safe; SQLite writes go through a single connection
protected by a lock (sufficient for our write rates).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from config.logging import get_logger
from config.settings import get_settings
from models.paper import PaperAnalysis, SourceReference

logger = get_logger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_analyses (
    paper_id TEXT PRIMARY KEY,
    filename TEXT,
    title TEXT,
    authors TEXT,           -- JSON array
    publication_date TEXT,
    summary TEXT,
    cybersecurity_area TEXT, -- JSON array
    problem_statement TEXT,
    proposed_solution TEXT,
    methodology TEXT,
    datasets TEXT,          -- JSON array
    key_results TEXT,
    limitations TEXT,
    sources TEXT,           -- JSON array
    analysis_status TEXT,
    analysis_error TEXT,
    created_at TEXT
);
"""


def _analysis_to_row(a: PaperAnalysis) -> dict[str, Any]:
    return {
        "paper_id": a.paper_id,
        "filename": a.filename,
        "title": a.title,
        "authors": json.dumps(a.authors or []),
        "publication_date": a.publication_date,
        "summary": a.summary,
        "cybersecurity_area": json.dumps(a.cybersecurity_area or []),
        "problem_statement": a.problem_statement,
        "proposed_solution": a.proposed_solution,
        "methodology": a.methodology,
        "datasets": json.dumps(a.datasets or []),
        "key_results": a.key_results,
        "limitations": a.limitations,
        "sources": json.dumps(
            [
                {
                    "paper_id": s.paper_id,
                    "page": s.page,
                    "section": s.section,
                    "chunk_id": s.chunk_id,
                    "text": s.text,
                }
                for s in (a.sources or [])
            ]
        ),
        "analysis_status": a.analysis_status,
        "analysis_error": a.analysis_error,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }


def _row_to_analysis(row: sqlite3.Row) -> PaperAnalysis:
    return PaperAnalysis(
        paper_id=row["paper_id"],
        filename=row["filename"],
        title=row["title"],
        authors=json.loads(row["authors"] or "[]"),
        publication_date=row["publication_date"],
        summary=row["summary"],
        cybersecurity_area=json.loads(row["cybersecurity_area"] or "[]"),
        problem_statement=row["problem_statement"],
        proposed_solution=row["proposed_solution"],
        methodology=row["methodology"],
        datasets=json.loads(row["datasets"] or "[]"),
        key_results=row["key_results"],
        limitations=row["limitations"],
        sources=[
            SourceReference(**s) for s in json.loads(row["sources"] or "[]")
        ],
        analysis_status=row["analysis_status"] or "pending",
        analysis_error=row["analysis_error"],
        created_at=row["created_at"],
    )


class ResultsStore:
    """Thread-safe in-memory cache backed by SQLite.

    The in-memory dict is the source of truth for reads (fast);
    writes are mirrored to SQLite so analyses survive restarts.
    On startup, `load_from_disk()` reads everything back.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, PaperAnalysis] = {}
        s = get_settings()
        self._db_path = db_path or s.sqlite_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        # Forward-only schema migrations: older DBs from before a
        # column was added are upgraded in place. SQLite has no
        # IF NOT EXISTS for ADD COLUMN in all versions, so we probe
        # pragma_table_info first and only ALTER when missing.
        self._migrate()
        self.load_from_disk()
        logger.info("ResultsStore initialized at %s", self._db_path)

    def _migrate(self) -> None:
        """Add any missing columns to an existing SQLite DB.

        Cheap forward-only migrations; never destructive.
        """
        existing = {
            row["name"]
            for row in self._conn.execute(
                "PRAGMA table_info(paper_analyses)"
            ).fetchall()
        }
        if "filename" not in existing:
            try:
                self._conn.execute(
                    "ALTER TABLE paper_analyses ADD COLUMN filename TEXT"
                )
                logger.info("Migrated paper_analyses: added 'filename' column")
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------- persistence
    def load_from_disk(self) -> int:
        """Reload all rows from SQLite into the in-memory cache."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM paper_analyses"
            ).fetchall()
            self._items = {_row_to_analysis(r).paper_id: _row_to_analysis(r) for r in rows}
            return len(self._items)

    # ------------------------------------------------- in-memory ops
    def put(self, paper_id: str, analysis: PaperAnalysis) -> None:
        with self._lock:
            self._items[paper_id] = analysis
            row = _analysis_to_row(analysis)
            self._conn.execute(
                """
                INSERT OR REPLACE INTO paper_analyses (
                    paper_id, filename, title, authors, publication_date, summary,
                    cybersecurity_area, problem_statement, proposed_solution,
                    methodology, datasets, key_results, limitations, sources,
                    analysis_status, analysis_error, created_at
                ) VALUES (
                    :paper_id, :filename, :title, :authors, :publication_date, :summary,
                    :cybersecurity_area, :problem_statement, :proposed_solution,
                    :methodology, :datasets, :key_results, :limitations, :sources,
                    :analysis_status, :analysis_error, :created_at
                )
                """,
                row,
            )

    def get(self, paper_id: str) -> PaperAnalysis | None:
        with self._lock:
            return self._items.get(paper_id)

    def all(self) -> list[PaperAnalysis]:
        with self._lock:
            return list(self._items.values())

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._conn.execute("DELETE FROM paper_analyses")

    def delete(self, paper_id: str) -> bool:
        """Remove a single paper's analysis row.

        Returns True if a row was deleted, False if the paper_id was
        unknown. Removes from both the in-memory cache and SQLite so
        the deletion survives a Flask restart.

        NOTE: this does NOT delete vector chunks or the original PDF
        file — those are owned by the vector store / upload directory
        and are cleaned up by the analysis service.
        """
        with self._lock:
            existed = paper_id in self._items
            self._items.pop(paper_id, None)
            cur = self._conn.execute(
                "DELETE FROM paper_analyses WHERE paper_id = ?", (paper_id,)
            )
            return existed or cur.rowcount > 0


_store: ResultsStore | None = None
_lock = threading.Lock()


def get_results_store() -> ResultsStore:
    """Return the process-wide ResultsStore singleton (with SQLite)."""
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = ResultsStore()
    return _store


__all__ = ["ResultsStore", "get_results_store"]