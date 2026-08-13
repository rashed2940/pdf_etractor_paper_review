"""Phase 4 tests: Coordinator with parallel execution."""
from __future__ import annotations

import json
import time

from models.paper import PaperAnalysis, SourceReference


class MockRetriever:
    """Returns canned chunks for any query."""

    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, **kwargs):
        class _Result:
            def __init__(self, chunk):
                self.chunk = chunk
                self.source = SourceReference(
                    paper_id=chunk.paper_id, page=chunk.page,
                    section=chunk.section, chunk_id=chunk.chunk_id,
                    text=chunk.text[:300],
                )

        return [_Result(c) for c in self._chunks]


def _make_role_aware_sleep_llm(sleep_seconds=0.2):
    """Build a role-aware LLM factory whose invoke sleeps `sleep_seconds`
    and returns canned JSON for each agent."""
    canned_by_phrase = {
        "metadata extractor": {
            "paper_id": "p_test", "title": "T", "authors": ["A"],
            "publication_date": None, "venue": None, "doi": None,
            "institution": None, "sources": [],
        },
        "cybersecurity-domain classifier": {
            "paper_id": "p_test", "areas": ["LLM Security"],
            "rationale": "test", "sources": [],
        },
        "solution/contribution analyzer": {
            "paper_id": "p_test", "problem_statement": "P",
            "proposed_solution": "S", "methodology": "M",
            "algorithms": ["A"], "datasets": ["D"], "experiments": "E",
            "key_results": "R", "limitations": "L", "sources": [],
        },
        "technical summarizer": {
            "paper_id": "p_test",
            "summary": (
                "test summary that is long enough to pass minimum word count check. "
                "It goes on with more words to make sure we hit at least one hundred "
                "words in the test summary field of our canned JSON response. Adding "
                "more text here to ensure we meet the threshold consistently across "
                "test runs and edge cases. We need at least one hundred words total "
                "in the summary."
            ),
            "problem": "P", "method": "M", "results": "R", "contribution": "C",
            "limitations": "L", "sources": [],
        },
    }

    class SleepLLM:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(list(messages))
            time.sleep(sleep_seconds)
            sys_content = (messages[0]["content"] if messages else "").lower()
            for phrase, canned in canned_by_phrase.items():
                if phrase in sys_content:
                    return type("R", (), {"content": json.dumps(canned)})()
            return type("R", (), {"content": "{}"})()

    return SleepLLM


class TestCoordinator:
    def test_analyze_paper_returns_paper_analysis(
        self, sample_agent_context, mock_llm_factory
    ):
        from agents.coordinator import Coordinator

        coord = Coordinator(
            llm_factory=mock_llm_factory,
            retriever=MockRetriever(sample_agent_context.retrieved_chunks),
        )
        result = coord.analyze_paper("paper_test")
        assert isinstance(result, PaperAnalysis)
        assert result.paper_id == "paper_test"
        assert result.title == "Test Paper"
        assert result.analysis_status == "ok"
        assert len(result.sources) >= 1

    def test_parallel_execution_speedup(self, sample_agent_context):
        """4 agents each sleep 0.2s should finish in ~0.2s, not ~0.8s."""
        from agents.coordinator import Coordinator

        SleepLLM = _make_role_aware_sleep_llm(sleep_seconds=0.2)
        sleep_instance = SleepLLM()  # one instance shared across agents
        coord = Coordinator(
            llm_factory=lambda r: sleep_instance,  # pass INSTANCE, not class
            retriever=MockRetriever(sample_agent_context.retrieved_chunks),
        )
        t0 = time.perf_counter()
        coord.analyze_paper("paper_test")
        elapsed = time.perf_counter() - t0
        # Sequential would be ~0.8s; parallel should be ~0.2s + overhead
        assert elapsed < 0.6, f"Execution appears sequential ({elapsed:.2f}s > 0.6s)"
        # All 4 agents invoked
        assert len(sleep_instance.calls) == 4, f"Expected 4 LLM calls; got {len(sleep_instance.calls)}"

    def test_metadata_filled_from_preextracted_when_llm_returns_null(self):
        """If MetadataAgent returns None/[] for a field but
        `_build_context` already pre-extracted the value, the final
        PaperAnalysis should use the pre-extracted value. This prevents
        the empty-title/authors bug when small LLMs mis-extract."""
        from agents.coordinator import Coordinator
        from models.paper import AgentContext, RetrievedChunk

        # LLM canned responses — metadata agent returns all-None for fields
        # that we DID pre-extract via PDF parsing.
        canned_by_phrase = {
            "metadata extractor": {
                "paper_id": "paper_test",
                "title": None,            # LLM missed it
                "authors": [],            # LLM missed them
                "publication_date": None, # LLM missed it
                "venue": None,
                "doi": None,
                "institution": None,
                "sources": [],
            },
            "cybersecurity-domain classifier": {
                "paper_id": "paper_test", "areas": ["LLM Security"],
                "rationale": "t", "sources": [],
            },
            "solution/contribution analyzer": {
                "paper_id": "paper_test", "problem_statement": "P",
                "proposed_solution": "S", "methodology": "M",
                "algorithms": ["A"], "datasets": ["D"], "experiments": "E",
                "key_results": "R", "limitations": "L", "sources": [],
            },
            "technical summarizer": {
                "paper_id": "paper_test",
                "summary": (
                    "test summary that is long enough to pass minimum word "
                    "count check. It goes on with more words to make sure we "
                    "hit at least one hundred words in the test summary field "
                    "of our canned JSON response. Adding more text here to "
                    "ensure we meet the threshold consistently across test "
                    "runs and edge cases. We need at least one hundred words "
                    "total in the summary to satisfy the schema validator."
                ),
                "problem": "P", "method": "M", "results": "R",
                "contribution": "C", "limitations": "L", "sources": [],
            },
        }

        class StaticLLM:
            def invoke(self, messages):
                sys_content = (messages[0]["content"] if messages else "").lower()
                for phrase, canned in canned_by_phrase.items():
                    if phrase in sys_content:
                        return type("R", (), {"content": json.dumps(canned)})()
                return type("R", (), {"content": "{}"})()

        # Build chunks that carry pre-extracted metadata (this is what
        # vector_store.add_chunks() stores and what Retriever returns).
        chunks = [
            RetrievedChunk(
                chunk_id="c0",
                paper_id="paper_test",
                filename="sample.pdf",
                title="Real Title From PDF",  # pre-extracted title
                page=1,
                section="Front Matter",
                text="Real Title From PDF\nAlice Real, Bob Real\n2025 IEEE X",
                score=0.9,
                authors=["Alice Real", "Bob Real"],         # pre-extracted
                publication_date="2025",                   # pre-extracted
                venue="IEEE X",
                doi="10.0/REAL",
                institution="Real Univ",
            ),
            RetrievedChunk(
                chunk_id="c1",
                paper_id="paper_test",
                filename="sample.pdf",
                title="Real Title From PDF",
                page=2,
                section="Methodology",
                text="We propose a novel architecture.",
                score=0.85,
                authors=["Alice Real", "Bob Real"],
                publication_date="2025",
                venue="IEEE X",
                doi="10.0/REAL",
                institution="Real Univ",
            ),
        ]
        # Coordinator path: _build_context reads `authors` from chunks[0].
        coord = Coordinator(
            llm_factory=lambda r: StaticLLM(),
            retriever=MockRetriever(chunks),
        )
        result = coord.analyze_paper("paper_test")
        # Pre-extracted values must override the LLM's null outputs.
        assert result.title == "Real Title From PDF"
        assert result.authors == ["Alice Real", "Bob Real"]
        assert result.publication_date == "2025"