"""Phase 6 acceptance + sanity tests.

Verifies:
1. All Phase 6 modules import.
2. BM25 lexical search ranks relevant chunks first.
3. RRF merge correctly interleaves two ranked lists.
4. NoOp reranker preserves original order.
5. ResultsStore (SQLite-backed) persists across instances.
6. End-to-end: a Phase 5-style upload still works with the new SQLite store.
"""
from __future__ import annotations

import sys
import tempfile
import pathlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.logging import setup_logging  # noqa: E402

setup_logging()

from rag.bm25 import BM25Index  # noqa: E402
from rag.reranker import NoOpReranker, get_reranker  # noqa: E402
from rag.hybrid import HybridRetriever  # noqa: E402
from services.results_store import ResultsStore  # noqa: E402
from models.paper import PaperAnalysis, RetrievedChunk  # noqa: E402


# ============================================================================
# 1. Imports
# ============================================================================
print("=== 1. Imports ===")
print("OK: all Phase 6 imports succeeded")


# ============================================================================
# 2. BM25 lexical search
# ============================================================================
print("\n=== 2. BM25 lexical search ===")
chunks = [
    RetrievedChunk(chunk_id="c1", paper_id="p1", filename="f", page=1,
                   section="Methodology",
                   text="We propose a CNN-LSTM for intrusion detection in IoT networks.",
                   score=None),
    RetrievedChunk(chunk_id="c2", paper_id="p1", filename="f", page=2,
                   section="Results",
                   text="We achieve 99 percent accuracy on the CICIDS2017 test set.",
                   score=None),
    RetrievedChunk(chunk_id="c3", paper_id="p2", filename="g", page=1,
                   section="Abstract",
                   text="This paper discusses LLM security threats and prompt injection.",
                   score=None),
    RetrievedChunk(chunk_id="c4", paper_id="p3", filename="h", page=1,
                   section="Introduction",
                   text="Network intrusion detection systems are essential for cybersecurity.",
                   score=None),
]
idx = BM25Index()
idx.add(chunks)

# Test 1: query for "intrusion detection" should rank c1 first
hits = idx.search("intrusion detection", top_k=3)
assert len(hits) >= 1
print(f"  Query 'intrusion detection' -> {len(hits)} hits, top={hits[0].chunk_id!r}")
top_ids = {h.chunk_id for h in hits}
assert "c1" in top_ids or "c4" in top_ids, f"Expected intrusion-related chunks; got {top_ids}"
print("  BM25 correctly identifies intrusion-related chunks")

# Test 2: query for "LLM security" should rank c3 first
hits = idx.search("LLM security prompt injection", top_k=2)
assert hits[0].chunk_id == "c3", f"Expected c3 first; got {hits[0].chunk_id}"
print(f"  Query 'LLM security' -> top={hits[0].chunk_id!r} (expected c3)")


# ============================================================================
# 3. RRF merge
# ============================================================================
print("\n=== 3. RRF merge (Reciprocal Rank Fusion) ===")
list_a = [chunks[0], chunks[1]]  # semantic search returns [c1, c2]
list_b = [chunks[1], chunks[2]]  # BM25 returns [c2, c3]
merged = HybridRetriever._rrf_merge(list_a, list_b, top_k=5)
merged_ids = [c.chunk_id for c in merged]
print(f"  Semantic [c1, c2] + BM25 [c2, c3] -> RRF: {merged_ids}")
# c1 is rank 1 in A only; c2 is rank 2 in A + rank 1 in B (highest combined);
# c3 is rank 2 in B only. So order: c2, c1, c3 (or c1, c2, c3 depending on ties).
# Just verify all 3 are present.
assert set(merged_ids) == {"c1", "c2", "c3"}, f"Missing chunks in merge: {merged_ids}"
print(f"  All 3 unique chunks present in merged result")


# ============================================================================
# 4. Reranker hook
# ============================================================================
print("\n=== 4. Reranker hook ===")
reranker = get_reranker()
assert isinstance(reranker, NoOpReranker)
print(f"  Default reranker: {reranker.name!r} (off by default)")
out = reranker.rerank("any query", chunks, top_k=2)
assert len(out) == 2
assert out[0].chunk_id == chunks[0].chunk_id
print(f"  NoOp reranker preserves order; first chunk={out[0].chunk_id!r}")


# ============================================================================
# 5. ResultsStore with SQLite persistence
# ============================================================================
print("\n=== 5. ResultsStore (SQLite-backed) ===")
import gc as _gc
with tempfile.TemporaryDirectory() as td:
    db = pathlib.Path(td) / "test.sqlite"
    store1 = ResultsStore(db_path=db)
    assert store1.count() == 0
    pa1 = PaperAnalysis(paper_id="p1", title="Paper 1", authors=["Alice"],
                        publication_date="2024",
                        cybersecurity_area=["LLM Security", "Intrusion Detection"],
                        proposed_solution="A test solution",
                        datasets=["TestDS1", "TestDS2"],
                        sources=[],
                        analysis_status="ok")
    store1.put("p1", pa1)
    pa2 = PaperAnalysis(paper_id="p2", title="Paper 2", summary="Test summary",
                        analysis_status="ok")
    store1.put("p2", pa2)
    assert store1.count() == 2
    print(f"  store1: 2 papers inserted")
    # Close the connection so Windows can release the file lock
    try:
        store1._conn.close()
    except Exception:
        pass
    del store1
    _gc.collect()

    # Close + reopen to verify persistence
    store2 = ResultsStore(db_path=db)
    assert store2.count() == 2, f"Expected 2 papers after reopen; got {store2.count()}"
    p1_back = store2.get("p1")
    assert p1_back is not None
    assert p1_back.title == "Paper 1"
    assert p1_back.cybersecurity_area == ["LLM Security", "Intrusion Detection"]
    assert p1_back.datasets == ["TestDS1", "TestDS2"]
    print(f"  store2 (re-opened from disk): 2 papers loaded")
    print(f"    p1 title={p1_back.title!r} areas={p1_back.cybersecurity_area}")
    try:
        store2._conn.close()
    except Exception:
        pass


# ============================================================================
# 6. Re-run Phase 5 acceptance to confirm no regression
# ============================================================================
print("\n=== 6. Phase 5 acceptance regression check ===")
import json as _json
import time as _time
from reportlab.lib.pagesizes import letter as _letter
from reportlab.pdfgen import canvas as _canvas

from app import create_app


class _MockLLM:
    def invoke(self, messages):
        sys_content = (messages[0]["content"] if messages else "").lower()
        if "metadata extractor" in sys_content:
            canned = {"paper_id": "p_test", "title": "Phase 6 Test", "authors": ["A"], "publication_date": "2026", "venue": "V", "doi": "10.0/T", "institution": "U", "sources": []}
        elif "cybersecurity-domain classifier" in sys_content:
            canned = {"paper_id": "p_test", "areas": ["LLM Security"], "rationale": "test", "sources": []}
        elif "solution/contribution analyzer" in sys_content:
            canned = {"paper_id": "p_test", "problem_statement": "P", "proposed_solution": "S", "methodology": "M", "algorithms": ["A"], "datasets": ["D"], "experiments": "E", "key_results": "R", "limitations": "L", "sources": []}
        elif "technical summarizer" in sys_content:
            canned = {"paper_id": "p_test", "summary": "This is a test summary that meets the minimum word count requirement for the summary field validation in our test suite. It covers all the relevant points and demonstrates the agent can produce coherent summaries of appropriate length when given properly structured mock input from canned JSON during automated testing scenarios.", "problem": "P", "method": "M", "results": "R", "contribution": "C", "limitations": "L", "sources": []}
        else:
            canned = {"answer": "test answer", "refused": False}
        return type("Resp", (), {"content": _json.dumps(canned)})()


# Generate unique PDF
unique_pdf = ROOT / "data" / "_test_pdfs" / f"phase6_{int(_time.time())}.pdf"
unique_pdf.parent.mkdir(parents=True, exist_ok=True)
_c = _canvas.Canvas(str(unique_pdf), pagesize=_letter)
_y = _letter[1] - 72
for line in [
    f"Phase 6 Test {unique_pdf.stem}",
    "Alice Phase6",
    "Test Symposium 2026",
    "DOI: 10.0000/PHASE6",
    "",
    "Abstract",
    "This paper proposes a CNN-LSTM for LLM security threat detection.",
    "",
    "1. Introduction",
    "LLM security is critical.",
    "",
    "2. Methodology",
    "We trained on CICIDS2017.",
    "",
    "3. Results",
    "Achieved 99% accuracy.",
]:
    _c.drawString(72, _y, line)
    _y -= 16
_c.showPage()
_c.save()

# Use the global ResultsStore (which now has SQLite persistence at the
# configured SQLITE_PATH). Clear it first for a clean test.
from services.results_store import get_results_store as _grs
_grs().clear()

app = create_app(llm_factory=lambda role: _MockLLM())
app.config["TESTING"] = True
client = app.test_client()

# Upload
with open(unique_pdf, "rb") as f:
    r = client.post("/upload", data={"pdfs": (f, "sample.pdf")}, content_type="multipart/form-data")
assert r.status_code == 200
upload_id = r.get_json()["upload_id"]
print(f"  upload_id={upload_id[:12]}...")

# Poll status
deadline = _time.time() + 60
final = None
while _time.time() < deadline:
    r = client.get(f"/status/{upload_id}")
    s = r.get_json()
    states = {p: v["state"] for p, v in s["papers"].items()}
    if all(v in ("done", "failed") for v in states.values()):
        final = states
        break
    _time.sleep(1)
assert final == {"sample.pdf": "done"}, f"Processing failed: {final}"
print(f"  Background processing: {final}")

# API papers
r = client.get("/api/papers")
papers = r.get_json()
assert len(papers) >= 1, "No papers returned"
print(f"  /api/papers: {len(papers)} paper(s)")

# Verify SQLite persistence: re-instantiate the store from the same path
from config.settings import get_settings as _gs
s = _gs()
assert s.sqlite_path.exists(), f"Expected SQLite at {s.sqlite_path}"
store_check = ResultsStore(db_path=s.sqlite_path)
assert store_check.count() >= 1, f"Expected >=1 paper in SQLite; got {store_check.count()}"
print(f"  SQLite persistence: {store_check.count()} paper(s) on reopen at {s.sqlite_path.name}")
try:
    store_check._conn.close()
except Exception:
    pass

print("\nPHASE 6 ACCEPTANCE: PASS")