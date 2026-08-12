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