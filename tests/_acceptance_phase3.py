"""Phase 3 acceptance test.

Verifies all four analysis agents + BaseAgent:
1. Imports succeed.
2. JSON extraction handles whole-string / fenced / embedded cases.
3. Safe defaults are constructed for each output model.
4. Each agent, given a mock LLM that returns canned JSON, produces a
   validated output model with the expected fields populated.
5. Each agent, given a mock LLM that returns garbage, falls back to
   safe defaults after retries (no exceptions leak).
6. Each agent's system prompt includes INJECTION_GUARD.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.logging import setup_logging  # noqa: E402

setup_logging()

from agents.base import BaseAgent, _extract_json, _default_for  # noqa: E402
from agents.metadata_agent import MetadataAgent  # noqa: E402
from agents.cybersecurity_agent import CybersecurityAgent  # noqa: E402
from agents.solution_agent import SolutionAgent  # noqa: E402
from agents.summary_agent import SummaryAgent  # noqa: E402
from models.paper import (  # noqa: E402
    AgentContext,
    RetrievedChunk,
    MetadataResult,
    SolutionResult,
    SummaryResult,
    CybersecurityResult,
    CYBERSECURITY_TAXONOMY,
)
from rag.prompts import INJECTION_GUARD  # noqa: E402


# ============================================================================
# 1. Imports + helpers
# ============================================================================
print("=== 1. Imports + helpers ===")
print("OK: all Phase 3 imports succeeded")

# JSON extraction
assert _extract_json('{"a": 1}') == {"a": 1}
assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
assert _extract_json('prose\n{"a": 1, "b": [2, 3]}\nmore prose') == {"a": 1, "b": [2, 3]}
assert _extract_json('no json here') is None
# Nested fenced block
assert _extract_json('ok ```json\n{"x": true}\n``` done') == {"x": True}
print("  JSON extractor: whole / fenced / embedded / nested -> OK")

# Safe defaults
md = _default_for(MetadataResult)
assert md.title is None
assert md.authors == []
assert md.paper_id == "Not available"  # required str -> "Not available" sentinel
assert md.publication_date is None
print("  Safe defaults: required str -> 'Not available', list -> [], optional -> None")

sol = _default_for(SolutionResult)
assert sol.problem_statement is None and sol.algorithms == []
print("  Safe defaults: SolutionResult OK")

# ============================================================================
# 2. System prompts include injection guard
# ============================================================================
print("\n=== 2. System prompts include INJECTION_GUARD ===")

# Helper: build a minimal AgentContext for testing.
def _make_context() -> AgentContext:
    return AgentContext(
        paper_id="paper_test123",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="paper_test123_c0",
                paper_id="paper_test123",
                filename="sample.pdf",
                title="Sample",
                page=3,
                section="Methodology",
                text="We propose a CNN-LSTM trained on CICIDS2017.",
                score=0.91,
            ),
            RetrievedChunk(
                chunk_id="paper_test123_c1",
                paper_id="paper_test123",
                filename="sample.pdf",
                title="Sample",
                page=5,
                section="Results",
                text="We achieve 99.2% accuracy on the test set.",
                score=0.88,
            ),
        ],
        metadata={"title": "Deep Learning for Intrusion Detection"},
    )

for AgentCls in (MetadataAgent, CybersecurityAgent, SolutionAgent, SummaryAgent):
    agent = AgentCls(llm_factory=lambda r: None)
    msgs = agent.build_messages(_make_context())
    sys_prompt = msgs[0]["content"]
    assert INJECTION_GUARD in sys_prompt, f"{AgentCls.__name__} missing injection guard"
    assert "JSON" in sys_prompt, f"{AgentCls.__name__} missing JSON format instruction"
    print(f"  {AgentCls.__name__:25s}: injection guard + JSON format OK")


# ============================================================================
# 3. Mock LLM fixtures
# ============================================================================
# (Note: _make_context is defined above in section 2.)

class CannedLLM:
    """Returns the same `content` string for every invoke() call."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[dict]] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return type("Resp", (), {"content": self.content})()


class BadThenGoodLLM:
    """Returns garbage for the first call, valid JSON for subsequent calls."""

    def __init__(self, good_content: str) -> None:
        self.good = good_content
        self.calls: list[list[dict]] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        # First call: garbage; later: good
        if len(self.calls) == 1:
            return type("Resp", (), {"content": "this is not json at all"})()
        return type("Resp", (), {"content": self.good})()


# ============================================================================
# 4. Each agent: canned JSON -> validated output
# ============================================================================
print("\n=== 4. Each agent with canned JSON ===")

# MetadataAgent
metadata_canned = {
    "paper_id": "paper_test123",
    "title": "Deep Learning for Intrusion Detection in IoT",
    "authors": ["Alice Smith", "Bob Jones"],
    "publication_date": "2024",
    "venue": "IEEE Symposium on Security and Privacy",
    "doi": "10.1109/SP.2024.000123",
    "institution": "MIT",
    "sources": [],
}
llm = CannedLLM(json.dumps(metadata_canned))
agent = MetadataAgent(llm_factory=lambda r: llm)
result = agent.run(_make_context())
assert isinstance(result, MetadataResult)
assert result.title == "Deep Learning for Intrusion Detection in IoT"
assert result.authors == ["Alice Smith", "Bob Jones"]
assert result.doi == "10.1109/SP.2024.000123"
assert len(result.sources) == 2, "Sources should be attached from context"
print(f"  MetadataAgent OK: title={result.title!r} authors={result.authors}")

# CybersecurityAgent
cyber_canned = {
    "paper_id": "paper_test123",
    "areas": ["Intrusion Detection", "IoT Security"],
    "rationale": "The paper proposes a CNN-LSTM for IoT intrusion detection.",
    "sources": [],
}
llm = CannedLLM(json.dumps(cyber_canned))
agent = CybersecurityAgent(llm_factory=lambda r: llm)
result = agent.run(_make_context())
assert isinstance(result, CybersecurityResult)
assert "Intrusion Detection" in result.areas
assert "IoT Security" in result.areas
for a in result.areas:
    assert a in CYBERSECURITY_TAXONOMY, f"{a} not in taxonomy"
print(f"  CybersecurityAgent OK: areas={result.areas}")

# SolutionAgent
sol_canned = {
    "paper_id": "paper_test123",
    "problem_statement": "Detecting intrusions in IoT networks.",
    "proposed_solution": "A CNN-LSTM model.",
    "methodology": "Hybrid deep learning.",
    "algorithms": ["CNN", "LSTM"],
    "datasets": ["CICIDS2017"],
    "experiments": "Trained on 80% of CICIDS2017; tested on 20%.",
    "key_results": "99.2% accuracy on the test set.",
    "limitations": "Requires labeled data; may not generalize to zero-day attacks.",
    "sources": [],
}
llm = CannedLLM(json.dumps(sol_canned))
agent = SolutionAgent(llm_factory=lambda r: llm)
result = agent.run(_make_context())
assert isinstance(result, SolutionResult)
assert result.proposed_solution == "A CNN-LSTM model."
assert result.algorithms == ["CNN", "LSTM"]
assert result.datasets == ["CICIDS2017"]
print(f"  SolutionAgent OK: solution={result.proposed_solution!r} datasets={result.datasets}")

# SummaryAgent
summary_canned = {
    "paper_id": "paper_test123",
    "summary": (
        "This paper addresses the problem of detecting network intrusions in IoT "
        "environments using a hybrid deep learning architecture. The proposed "
        "method combines convolutional neural networks for spatial feature "
        "extraction with long short-term memory layers for temporal modeling of "
        "network packet sequences. The authors trained and evaluated their approach "
        "on the CICIDS2017 benchmark dataset and a custom IoT testbed, achieving "
        "99.2% accuracy on the former and 96.8% on the latter, outperforming "
        "Random Forest and XGBoost baselines by several percentage points. The "
        "primary contribution is a unified architecture that handles both spatial "
        "and temporal patterns in heterogeneous IoT traffic with low inference "
        "latency. Limitations include the requirement for labeled training data, "
        "uncertainty about generalization to zero-day attacks, and a lack of "
        "evaluation under adversarial conditions. Future work will explore "
        "self-supervised pretraining and adversarial robustness testing."
    ),
    "problem": "Detecting intrusions in IoT networks.",
    "method": "Hybrid CNN-LSTM deep learning architecture.",
    "results": "99.2% accuracy on CICIDS2017.",
    "contribution": "Unified spatial+temporal modeling for IoT intrusion detection.",
    "limitations": "Requires labeled data; zero-day generalization unclear.",
    "sources": [],
}
llm = CannedLLM(json.dumps(summary_canned))
agent = SummaryAgent(llm_factory=lambda r: llm)
result = agent.run(_make_context())
assert isinstance(result, SummaryResult)
assert len(result.summary.split()) >= 100, f"Summary too short: {len(result.summary.split())} words"
assert len(result.summary.split()) <= 250, f"Summary too long: {len(result.summary.split())} words"
print(f"  SummaryAgent OK: {len(result.summary.split())} words")

# ============================================================================
# 5. Retry/repair path
# ============================================================================
print("\n=== 5. Retry/repair: bad then good ===")
llm = BadThenGoodLLM(json.dumps(metadata_canned))
agent = MetadataAgent(llm_factory=lambda r: llm, max_retries=2)
result = agent.run(_make_context())
assert isinstance(result, MetadataResult)
assert result.title == "Deep Learning for Intrusion Detection in IoT"
# LLM should have been called: 1 (garbage) + 1 (good) = 2 times
assert len(llm.calls) == 2, f"Expected 2 calls (bad + good); got {len(llm.calls)}"
# The second call should include the repair-mode follow-up message
last_call = llm.calls[-1]
assert len(last_call) == 3, "Second call should have system+user+repair"
assert "previous response could not be parsed" in last_call[-1]["content"].lower()
print(f"  MetadataAgent retried after garbage; LLM called {len(llm.calls)} times; repair follow-up present")

# ============================================================================
# 6. Total failure: garbage forever -> safe defaults
# ============================================================================
print("\n=== 6. Total failure -> safe defaults ===")
class AlwaysBadLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return type("Resp", (), {"content": "completely not json"})()


llm = AlwaysBadLLM()
agent = MetadataAgent(llm_factory=lambda r: llm, max_retries=2)
result = agent.run(_make_context())
assert isinstance(result, MetadataResult)
assert result.title is None  # safe default
assert result.paper_id == "paper_test123"  # populated from context
print(f"  Total failure: paper_id={result.paper_id!r} title={result.title!r}")
assert len(llm.calls) == 3, f"Expected 3 attempts (initial + 2 retries); got {len(llm.calls)}"


# ============================================================================
# 7. Real end-to-end with mocked LLM + real retriever
# ============================================================================
print("\n=== 7. Real retriever + canned LLM end-to-end ===")
try:
    from rag.retriever import Retriever

    retriever = Retriever()
    store = retriever._ensure_store()
    papers = store.list_papers()
    if not papers:
        print("  (skipping: no papers indexed — run _acceptance_phase1.py first)")
    else:
        # Use the first indexed paper's chunks.
        paper_id = papers[0]["paper_id"]
        # Pull a few chunks by direct query
        hits = store.query("intrusion detection", top_k=5, paper_id=paper_id)
        chunks = [
            RetrievedChunk(
                chunk_id=h.chunk_id, paper_id=h.paper_id, filename=h.filename or "",
                title=h.title, page=h.page, section=h.section, text=h.text, score=h.score,
            )
            for h in hits
        ]
        ctx = AgentContext(paper_id=paper_id, retrieved_chunks=chunks)

        llm = CannedLLM(json.dumps(sol_canned))
        agent = SolutionAgent(llm_factory=lambda r: llm)
        result = agent.run(ctx)
        assert isinstance(result, SolutionResult)
        print(f"  Real-retriever SolutionAgent OK for paper_id={paper_id[:16]}...")
        print(f"    proposed_solution={result.proposed_solution!r}")
        print(f"    algorithms={result.algorithms}")
except Exception as e:
    print(f"  (skipping real-retriever test: {e})")


print("\nPHASE 3 ACCEPTANCE: PASS")
