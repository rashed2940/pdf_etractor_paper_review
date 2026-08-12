"""Phase 3 tests: BaseAgent + 4 specialist agents."""
from __future__ import annotations

import json

from models.paper import (
    AgentContext,
    CybersecurityResult,
    MetadataResult,
    RetrievedChunk,
    SolutionResult,
    SummaryResult,
)


def _ctx(paper_id="p_test", text="We propose a CNN-LSTM for intrusion detection."):
    return AgentContext(
        paper_id=paper_id,
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id=f"{paper_id}_c0",
                paper_id=paper_id,
                filename="test.pdf",
                title="Test",
                page=3,
                section="Methodology",
                text=text,
                score=0.9,
            )
        ],
    )


class TestBaseAgentHelpers:
    def test_extract_json_whole_string(self):
        from agents.base import _extract_json

        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_extract_json_fenced(self):
        from agents.base import _extract_json

        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_extract_json_embedded(self):
        from agents.base import _extract_json

        assert _extract_json('prose\n{"a": 1}\nmore') == {"a": 1}

    def test_extract_json_nested(self):
        from agents.base import _extract_json

        assert _extract_json('see ```json\n{"x": true}\n``` here') == {"x": True}

    def test_extract_json_returns_none_on_garbage(self):
        from agents.base import _extract_json

        assert _extract_json("no json here") is None

    def test_default_for_required_str_is_not_available(self):
        from agents.base import _default_for

        d = _default_for(MetadataResult)
        assert d.title is None
        assert d.authors == []
        assert d.paper_id == "Not available"


class TestMetadataAgent:
    def test_runs_with_canned_llm(self, sample_agent_context, mock_llm_factory):
        from agents.metadata_agent import MetadataAgent

        agent = MetadataAgent(llm_factory=mock_llm_factory)
        result = agent.run(sample_agent_context)
        assert isinstance(result, MetadataResult)
        assert result.title == "Test Paper"

    def test_system_prompt_has_injection_guard(self, sample_agent_context, mock_llm_factory):
        from agents.metadata_agent import MetadataAgent
        from rag.prompts import INJECTION_GUARD

        agent = MetadataAgent(llm_factory=mock_llm_factory)
        msgs = agent.build_messages(sample_agent_context)
        assert INJECTION_GUARD in msgs[0]["content"]


class TestCybersecurityAgent:
    def test_runs_with_canned_llm(self, sample_agent_context, mock_llm_factory):
        from agents.cybersecurity_agent import CybersecurityAgent
        from models.paper import CYBERSECURITY_TAXONOMY

        agent = CybersecurityAgent(llm_factory=mock_llm_factory)
        result = agent.run(sample_agent_context)
        assert isinstance(result, CybersecurityResult)
        assert "Intrusion Detection" in result.areas
        for a in result.areas:
            assert a in CYBERSECURITY_TAXONOMY


class TestSolutionAgent:
    def test_runs_with_canned_llm(self, sample_agent_context, mock_llm_factory):
        from agents.solution_agent import SolutionAgent

        agent = SolutionAgent(llm_factory=mock_llm_factory)
        result = agent.run(sample_agent_context)
        assert isinstance(result, SolutionResult)
        assert result.proposed_solution == "Test solution."
        assert result.datasets == ["CICIDS2017"]


class TestSummaryAgent:
    def test_runs_with_canned_llm(self, sample_agent_context, mock_llm_factory):
        from agents.summary_agent import SummaryAgent

        agent = SummaryAgent(llm_factory=mock_llm_factory)
        result = agent.run(sample_agent_context)
        assert isinstance(result, SummaryResult)
        # Mock returns a long summary — verify it's in the 100-250 range
        wc = len(result.summary.split())
        assert 100 <= wc <= 250, f"Summary word count {wc} outside [100, 250]"


class TestAgentRetryAndRepair:
    def test_garbage_then_good_succeeds(self, sample_agent_context):
        from agents.metadata_agent import MetadataAgent

        class BadThenGoodLLM:
            def __init__(self):
                self.calls = 0
                self._canned = json.dumps({
                    "paper_id": "p_test", "title": "Good Paper", "authors": [],
                    "publication_date": None, "venue": None, "doi": None,
                    "institution": None, "sources": [],
                })

            def invoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return type("Resp", (), {"content": "garbage"})()
                return type("Resp", (), {"content": self._canned})()

        agent = MetadataAgent(llm_factory=lambda r: BadThenGoodLLM(), max_retries=2)
        result = agent.run(sample_agent_context)
        assert isinstance(result, MetadataResult)
        assert result.title == "Good Paper"

    def test_total_failure_returns_safe_defaults(self, sample_agent_context):
        from agents.metadata_agent import MetadataAgent

        class AlwaysBadLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, messages):
                self.calls += 1
                return type("Resp", (), {"content": "still not json"})()

        agent = MetadataAgent(llm_factory=lambda r: AlwaysBadLLM(), max_retries=2)
        result = agent.run(sample_agent_context)
        assert isinstance(result, MetadataResult)
        assert result.title is None
        assert result.paper_id == sample_agent_context.paper_id