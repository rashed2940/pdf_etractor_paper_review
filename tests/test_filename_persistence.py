"""Regression test: PaperAnalysis.filename round-trips through the
SQLite ResultsStore so the UI can correlate fresh uploads with their
saved analysis row.

Background: the UI's "View results" link after upload depends on
matching the processing-registry's `paper_id`/filename against the
analysis row. The analysis row only knew `paper_id` and `title` until
this fix; `filename` was added to PaperAnalysis and persisted.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from models.paper import PaperAnalysis, SourceReference
from services.results_store import ResultsStore, _analysis_to_row, _row_to_analysis


def _make_analysis(paper_id: str, filename: str) -> PaperAnalysis:
    return PaperAnalysis(
        paper_id=paper_id,
        filename=filename,
        title="Test Paper",
        authors=["Alice", "Bob"],
        publication_date="2024",
        summary="A test summary that is long enough to pass the minimum "
        "word count threshold for the summary validator field. " * 5,
        cybersecurity_area=["LLM Security"],
        problem_statement="A test problem.",
        proposed_solution="A test solution.",
        methodology="A test method.",
        datasets=["CICIDS2017"],
        key_results="99% accuracy.",
        limitations="A test limitation.",
        sources=[
            SourceReference(paper_id=paper_id, page=3, text="A source excerpt.")
        ],
        analysis_status="ok",
    )


def test_paper_analysis_has_filename_field():
    a = PaperAnalysis(paper_id="p1", filename="foo.pdf")
    assert a.filename == "foo.pdf"


def test_filename_defaults_to_none():
    a = PaperAnalysis(paper_id="p1")
    assert a.filename is None


def test_analysis_to_row_includes_filename():
    a = _make_analysis("p1", "Enhancing_MARL_IDS.pdf")
    row = _analysis_to_row(a)
    assert row["filename"] == "Enhancing_MARL_IDS.pdf"


def test_row_to_analysis_includes_filename():
    a = _make_analysis("p1", "Enhancing_MARL_IDS.pdf")
    row = _analysis_to_row(a)
    # Simulate a sqlite3.Row
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    keys = ",".join(row.keys())
    placeholders = ",".join(f":{k}" for k in row.keys())
    conn.execute(
        f"CREATE TABLE t ({','.join(f'{k} TEXT' for k in row.keys())})"
    )
    conn.execute(f"INSERT INTO t VALUES ({placeholders})", row)
    fetched = conn.execute(f"SELECT {keys} FROM t").fetchone()
    rebuilt = _row_to_analysis(fetched)
    assert rebuilt.filename == "Enhancing_MARL_IDS.pdf"
    assert rebuilt.paper_id == "p1"
    assert rebuilt.title == "Test Paper"


def test_results_store_round_trips_filename(tmp_path: Path):
    """Put + get preserves filename."""
    db_path = tmp_path / "test.sqlite"
    store = ResultsStore(db_path=db_path)
    a = _make_analysis("p_round", "Round_Trip_Test.pdf")
    store.put("p_round", a)
    got = store.get("p_round")
    assert got is not None
    assert got.filename == "Round_Trip_Test.pdf"
    assert got.title == "Test Paper"
    assert got.authors == ["Alice", "Bob"]
    store.clear()


def test_results_store_migrates_existing_db_without_filename(tmp_path: Path):
    """Pre-existing SQLite DB without the filename column should be
    upgraded in place by _migrate()."""
    db_path = tmp_path / "legacy.sqlite"
    # Create a legacy DB without the filename column.
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(
        """
        CREATE TABLE paper_analyses (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            publication_date TEXT,
            summary TEXT,
            cybersecurity_area TEXT,
            problem_statement TEXT,
            proposed_solution TEXT,
            methodology TEXT,
            datasets TEXT,
            key_results TEXT,
            limitations TEXT,
            sources TEXT,
            analysis_status TEXT,
            analysis_error TEXT,
            created_at TEXT
        );
        """
    )
    legacy.execute(
        """
        INSERT INTO paper_analyses
            (paper_id, title, authors, publication_date, summary,
             cybersecurity_area, problem_statement, proposed_solution,
             methodology, datasets, key_results, limitations, sources,
             analysis_status, analysis_error, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "legacy1", "Legacy Title", "[]", "2023",
            "Legacy summary that is long enough to pass the minimum word count threshold. " * 3,
            "[]", None, None, None, "[]", None, None, "[]", "ok", None,
            "2024-01-01T00:00:00",
        ),
    )
    legacy.commit()
    legacy.close()

    # Now load via ResultsStore — should auto-migrate and read the row.
    store = ResultsStore(db_path=db_path)
    got = store.get("legacy1")
    assert got is not None
    assert got.paper_id == "legacy1"
    assert got.title == "Legacy Title"
    assert got.filename is None  # Old rows just have NULL
    store.clear()
