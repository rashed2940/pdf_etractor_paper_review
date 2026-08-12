"""Shared pytest fixtures (PLAN.md §3 Phase 7).

Centralizes the mock LLM, mock retriever, test PDF generation, Flask app
factory, and other fixtures used across all phase test modules.

All LLM-dependent tests use the MockLLM fixture here so no real Ollama
is needed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

# Ensure project root on sys.path so `import app`, `import agents.*`, etc. work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================================
# Mock LLM
# ============================================================================
class MockLLM:
    """Stand-in LLM that returns canned JSON based on the agent role.

    The canned responses are designed to validate each agent's specific
    Pydantic output model. The role is inferred by inspecting the system
    prompt for distinctive phrases.
    """

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []
        self.fail_first_n: int = 0  # number of garbage responses to emit first

    def invoke(self, messages):
        self.calls.append(list(messages))
        if self.fail_first_n > 0:
            self.fail_first_n -= 1
            return type("Resp", (), {"content": "this is not json"})()

        sys_content = (messages[0]["content"] if messages else "").lower()

        if "metadata extractor" in sys_content:
            canned = {
                "paper_id": "p_test",
                "title": "Test Paper",
                "authors": ["Alice", "Bob"],
                "publication_date": "2024",
                "venue": "Test Venue",
                "doi": "10.0/TEST",
                "institution": "Test Univ",
                "sources": [],
            }
        elif "cybersecurity-domain classifier" in sys_content:
            canned = {
                "paper_id": "p_test",
                "areas": ["Intrusion Detection", "IoT Security"],
                "rationale": "Test rationale.",
                "sources": [],
            }
        elif "solution/contribution analyzer" in sys_content:
            canned = {
                "paper_id": "p_test",
                "problem_statement": "Test problem.",
                "proposed_solution": "Test solution.",
                "methodology": "Test method.",
                "algorithms": ["CNN", "LSTM"],
                "datasets": ["CICIDS2017"],
                "experiments": "Test experiments.",
                "key_results": "99% accuracy.",
                "limitations": "Test limitation.",
                "sources": [],
            }
        elif "technical summarizer" in sys_content:
            canned = {
                "paper_id": "p_test",
                "summary": (
                    "This is a test summary that is intentionally long enough "
                    "to meet the minimum word count requirement for the "
                    "summary field validation in our test suite. It covers "
                    "all relevant points and demonstrates the agent can "
                    "produce coherent summaries of appropriate length when "
                    "given properly structured mock input from canned JSON "
                    "during automated testing scenarios. It also mentions "
                    "the limitations and future work to fully satisfy the "
                    "Schema expectations. Adding more text here to ensure "
                    "we meet the threshold of one hundred words consistently "
                    "across test runs and edge cases. We pad with extra "
                    "sentences about CNN-LSTM models on CICIDS2017 datasets "
                    "and intrusion detection in IoT networks for good measure."
                ),
                "problem": "P",
                "method": "M",
                "results": "R",
                "contribution": "C",
                "limitations": "L",
                "sources": [],
            }
        else:
            canned = {"answer": "Test answer.", "refused": False}

        return type("Resp", (), {"content": json.dumps(canned)})


@pytest.fixture
def mock_llm() -> MockLLM:
    """A fresh MockLLM per test."""
    return MockLLM()


@pytest.fixture
def mock_llm_factory(mock_llm):
    """A role-aware factory that returns the mock LLM."""
    return lambda role: mock_llm


@pytest.fixture
def mock_llm_factory_zero_arg(mock_llm):
    """A zero-arg factory for QAAgent-style consumers."""
    return lambda: mock_llm


# ============================================================================
# Agent context fixture
# ============================================================================
@pytest.fixture
def sample_agent_context():
    """A reusable AgentContext for testing agents in isolation."""
    from models.paper import AgentContext, RetrievedChunk

    return AgentContext(
        paper_id="paper_test",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="paper_test_c0",
                paper_id="paper_test",
                filename="sample.pdf",
                title="Test Paper",
                page=3,
                section="Methodology",
                text="We propose a CNN-LSTM trained on CICIDS2017.",
                score=0.91,
            ),
            RetrievedChunk(
                chunk_id="paper_test_c1",
                paper_id="paper_test",
                filename="sample.pdf",
                title="Test Paper",
                page=5,
                section="Results",
                text="We achieve 99% accuracy on the test set.",
                score=0.88,
            ),
        ],
        metadata={"title": "Deep Learning for Intrusion Detection"},
    )


# ============================================================================
# Test PDF generation
# ============================================================================
@pytest.fixture
def test_pdf_path(tmp_path):
    """Generate a fresh, unique PDF for tests; returns its path."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / f"phase_{int(time.time() * 1000)}.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    y = letter[1] - 72
    for line in [
        "Deep Learning for Intrusion Detection in IoT Networks",
        "Alice Test, Bob Test",
        "Proceedings of the 2026 Test Symposium",
        "DOI: 10.0000/TEST.2026.0001",
        "",
        "Abstract",
        "We propose a CNN-LSTM for intrusion detection in IoT networks.",
        "",
        "1. Introduction",
        "Intrusion detection is critical for cybersecurity in IoT.",
        "",
        "2. Methodology",
        "We trained on CICIDS2017 with PyTorch.",
        "",
        "3. Results",
        "Achieved 99% accuracy on the test set.",
    ]:
        c.drawString(72, y, line)
        y -= 16
    c.showPage()
    c.save()
    return pdf_path


# ============================================================================
# Flask app + client (mock LLM injected)
# ============================================================================
@pytest.fixture
def flask_app(tmp_path, monkeypatch):
    """Create a Flask app with mock LLM injected and isolated data dirs."""
    # Isolate data dirs so tests don't interfere with each other or production
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.setenv("UPLOAD_DIRECTORY", str(tmp_path / "uploads"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "app.sqlite"))

    # Clear any cached settings so the new env vars take effect
    from config import settings as settings_mod

    settings_mod.get_settings.cache_clear()

    from app import create_app

    app = create_app(llm_factory=lambda role: MockLLM())
    app.config["TESTING"] = True
    return app


@pytest.fixture
def flask_client(flask_app):
    """Flask test client."""
    return flask_app.test_client()


# ============================================================================
# Mock retriever (returns canned chunks regardless of query)
# ============================================================================
class MockRetriever:
    """Mock retriever returning canned chunks for any query."""

    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, **kwargs):
        from models.paper import RetrievedChunk, SourceReference

        # Return list of RetrievalResult-like objects
        class _Result:
            def __init__(self, chunk):
                self.chunk = chunk
                self.source = SourceReference(
                    paper_id=chunk.paper_id,
                    page=chunk.page,
                    section=chunk.section,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text[:300],
                )

        return [_Result(c) for c in self._chunks]


@pytest.fixture
def mock_retriever(sample_agent_context):
    """Mock retriever returning the sample_agent_context's chunks."""
    return MockRetriever(sample_agent_context.retrieved_chunks)