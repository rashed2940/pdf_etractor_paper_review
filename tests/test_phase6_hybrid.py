"""Phase 6 tests: BM25, RRF merge, reranker hook, SQLite persistence."""
from __future__ import annotations

from models.paper import RetrievedChunk


def _chunk(cid, text, paper_id="p1", page=1, section="Body"):
    return RetrievedChunk(
        chunk_id=cid, paper_id=paper_id, filename="f.pdf",
        title="T", page=page, section=section, text=text, score=None,
    )


class TestBM25:
    def test_basic_tokenize(self):
        from rag.bm25 import _tokenize

        tokens = _tokenize("Hello, World! 123 abc_xyz")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens
        assert "abc" in tokens
        assert "xyz" in tokens

    def test_search_finds_relevant_chunks(self):
        from rag.bm25 import BM25Index

        idx = BM25Index()
        idx.add([
            _chunk("c1", "We propose a CNN-LSTM for intrusion detection in IoT networks."),
            _chunk("c2", "We achieve 99 percent accuracy on the CICIDS2017 test set."),
            _chunk("c3", "This paper discusses LLM security threats and prompt injection."),
            _chunk("c4", "Network intrusion detection systems are essential for cybersecurity."),
        ])

        hits = idx.search("intrusion detection", top_k=3)
        assert len(hits) >= 1
        top_ids = {h.chunk_id for h in hits}
        assert "c1" in top_ids or "c4" in top_ids

        # More specific query
        hits = idx.search("LLM security prompt injection", top_k=2)
        assert hits[0].chunk_id == "c3"

    def test_search_empty_corpus(self):
        from rag.bm25 import BM25Index

        idx = BM25Index()
        assert idx.search("anything", top_k=5) == []

    def test_clear_resets_index(self):
        from rag.bm25 import BM25Index

        idx = BM25Index()
        idx.add([_chunk("c1", "hello world")])
        assert len(idx._docs) == 1
        idx.clear()
        assert len(idx._docs) == 0


class TestRRFMerge:
    def test_rrf_merges_two_lists(self):
        from rag.hybrid import HybridRetriever

        a = [_chunk("c1", "x"), _chunk("c2", "y")]
        b = [_chunk("c2", "y"), _chunk("c3", "z")]
        merged = HybridRetriever._rrf_merge(a, b, top_k=5)
        ids = [c.chunk_id for c in merged]
        assert set(ids) == {"c1", "c2", "c3"}

    def test_rrf_top_k_truncates(self):
        from rag.hybrid import HybridRetriever

        a = [_chunk(f"c{i}", f"text {i}") for i in range(5)]
        b = [_chunk(f"c{i}", f"text {i}") for i in range(5)]
        merged = HybridRetriever._rrf_merge(a, b, top_k=2)
        assert len(merged) == 2


class TestReranker:
    def test_default_is_noop(self):
        from rag.reranker import NoOpReranker, get_reranker

        r = get_reranker()
        assert isinstance(r, NoOpReranker)
        assert r.name == "noop"

    def test_noop_preserves_order(self):
        from rag.reranker import NoOpReranker

        chunks = [_chunk(f"c{i}", f"text {i}") for i in range(3)]
        r = NoOpReranker()
        out = r.rerank("query", chunks, top_k=2)
        assert len(out) == 2
        assert out[0].chunk_id == "c0"
        assert out[1].chunk_id == "c1"


class TestResultsStoreSQLite:
    def test_put_and_get(self, tmp_path):
        from services.results_store import ResultsStore
        from models.paper import PaperAnalysis

        db = tmp_path / "test.sqlite"
        store = ResultsStore(db_path=db)
        pa = PaperAnalysis(paper_id="p1", title="T", authors=["A"])
        store.put("p1", pa)
        got = store.get("p1")
        assert got is not None
        assert got.title == "T"

    def test_persistence_across_instances(self, tmp_path):
        from services.results_store import ResultsStore
        from models.paper import PaperAnalysis

        db = tmp_path / "test.sqlite"
        s1 = ResultsStore(db_path=db)
        s1.put("p1", PaperAnalysis(paper_id="p1", title="Persisted"))
        s1.put("p2", PaperAnalysis(paper_id="p2", summary="Summary"))
        # Close s1's connection so the second instance can take over the file
        try:
            s1._conn.close()
        except Exception:
            pass
        del s1

        s2 = ResultsStore(db_path=db)
        assert s2.count() == 2
        assert s2.get("p1").title == "Persisted"
        try:
            s2._conn.close()
        except Exception:
            pass

    def test_clear_removes_all(self, tmp_path):
        from services.results_store import ResultsStore
        from models.paper import PaperAnalysis

        db = tmp_path / "test.sqlite"
        store = ResultsStore(db_path=db)
        store.put("p1", PaperAnalysis(paper_id="p1"))
        store.put("p2", PaperAnalysis(paper_id="p2"))
        assert store.count() == 2
        store.clear()
        assert store.count() == 0

    def test_duplicate_reanalysis_runs_analysis(self, tmp_path, monkeypatch):
        """Regression test: when a PDF is re-uploaded, the runner must
        re-run analysis (not silently skip). Previously the duplicate
        branch returned immediately, leaving users with a 'Done' chip
        but no result in the database.
        """
        from pathlib import Path

        from models.paper import PaperAnalysis
        from services.processing_runner import _run_for_paper
        from services.processing_status import get_registry
        from services.results_store import get_results_store

        # Build a fake PDF path (the file doesn't need to exist on disk
        # because we'll mock ingest_pdf and AnalysisService).
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"%PDF-fake")

        # Track whether the analysis service was called.
        calls: list[str] = []

        class FakeAnalysisService:
            def analyze_paper(self, paper_id, **kwargs):
                calls.append(paper_id)
                return PaperAnalysis(
                    paper_id=paper_id,
                    title="Re-analyzed",
                    summary="Done in the duplicate branch.",
                )

        class FakeIngestionResult:
            def __init__(self, paper_id, is_dup):
                self.paper_id = paper_id
                self.is_duplicate = is_dup
                self.chunk_count = 5
                self.filename = "fake.pdf"
                self.sha256 = "deadbeef"

        def fake_ingest(path, vector_store=None, paper_id=None):
            return FakeIngestionResult("paper_dup", is_dup=True)

        monkeypatch.setattr(
            "services.processing_runner.ingest_pdf", fake_ingest
        )
        # Mock AnalysisService so we don't need a real LLM
        import services.processing_runner as pr

        original = pr.AnalysisService
        monkeypatch.setattr(pr, "AnalysisService", FakeAnalysisService)

        # Run the duplicate branch
        reg = get_registry()
        store = get_results_store()
        upload_id = reg.create_upload(["fake.pdf"])
        _run_for_paper(
            upload_id,
            "fake.pdf",
            fake_pdf,
            vector_store=None,
            analysis_service=FakeAnalysisService(),
        )

        # The analysis service MUST have been invoked for the duplicate.
        assert calls == ["paper_dup"], (
            f"Expected analysis to re-run on duplicate, got calls={calls}"
        )
        # The result MUST be persisted to the store.
        assert store.get("paper_dup") is not None
        assert store.get("paper_dup").title == "Re-analyzed"
        # Status must be 'done' (not 'failed').
        statuses = {
            s.state for s in reg.get_upload(upload_id).values()
        }
        assert statuses == {"done"}, f"Expected done, got {statuses}"

    def test_delete_removes_from_memory_and_sqlite(self, tmp_path):
        from services.results_store import ResultsStore
        from models.paper import PaperAnalysis

        db = tmp_path / "test.sqlite"
        store = ResultsStore(db_path=db)
        store.put("p1", PaperAnalysis(paper_id="p1", title="kept"))
        store.put("p2", PaperAnalysis(paper_id="p2", title="to-delete"))
        assert store.count() == 2

        deleted = store.delete("p2")
        assert deleted is True
        assert store.count() == 1
        assert store.get("p2") is None
        assert store.get("p1").title == "kept"

    def test_delete_unknown_returns_false(self, tmp_path):
        from services.results_store import ResultsStore

        store = ResultsStore(db_path=tmp_path / "test.sqlite")
        assert store.delete("nonexistent") is False

    def test_delete_persists_across_instances(self, tmp_path):
        from services.results_store import ResultsStore
        from models.paper import PaperAnalysis

        db = tmp_path / "test.sqlite"
        s1 = ResultsStore(db_path=db)
        s1.put("p1", PaperAnalysis(paper_id="p1"))
        s1.put("p2", PaperAnalysis(paper_id="p2"))
        s1.delete("p1")
        try:
            s1._conn.close()
        except Exception:
            pass
        del s1

        s2 = ResultsStore(db_path=db)
        assert s2.count() == 1
        assert s2.get("p1") is None
        assert s2.get("p2") is not None
        try:
            s2._conn.close()
        except Exception:
            pass

    def test_serialization_round_trip(self, tmp_path):
        from services.results_store import ResultsStore
        from models.paper import PaperAnalysis, SourceReference

        db = tmp_path / "test.sqlite"
        store = ResultsStore(db_path=db)
        pa = PaperAnalysis(
            paper_id="p1",
            title="Round Trip",
            authors=["Alice", "Bob"],
            cybersecurity_area=["LLM Security", "Intrusion Detection"],
            datasets=["CICIDS2017"],
            sources=[
                SourceReference(paper_id="p1", page=3, section="Methodology",
                                chunk_id="p1_c3", text="snippet"),
            ],
        )
        store.put("p1", pa)
        try:
            store._conn.close()
        except Exception:
            pass
        store2 = ResultsStore(db_path=db)
        got = store2.get("p1")
        assert got.title == "Round Trip"
        assert got.cybersecurity_area == ["LLM Security", "Intrusion Detection"]
        assert got.datasets == ["CICIDS2017"]
        assert len(got.sources) == 1
        assert got.sources[0].page == 3
        try:
            store2._conn.close()
        except Exception:
            pass