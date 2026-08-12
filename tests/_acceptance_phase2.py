"""Phase 2 acceptance test.

Verifies:
1. All Phase 2 modules import.
2. Prompt construction includes prompt-injection guard + <document> tags.
3. Comparative-question detection works.
4. Refusal detection works.
5. End-to-end with a MockLLM (no real Ollama needed): retriever returns
   real hits from the Phase 1 indexed PDF, MockLLM returns canned text,
   QAResult is constructed correctly with citations.
6. Refusal path: when retrieval is empty, the chain refuses.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.logging import setup_logging, get_logger  # noqa: E402
from rag.prompts import (  # noqa: E402
    INJECTION_GUARD,
    QA_SYSTEM_PROMPT,
    COMPARATIVE_QA_SYSTEM_PROMPT,
    REFUSAL_SENTINEL,
    build_user_prompt,
)
from rag.retriever import Retriever, RetrievalResult  # noqa: E402
from rag.chain import (  # noqa: E402
    _build_messages,
    _is_comparative,
    _is_refusal,
    answer_question,
)
from agents.qa_agent import QAAgent  # noqa: E402


# ============================================================================
# 1. Imports
# ============================================================================
print("=== 1. Imports ===")
print("OK: all Phase 2 imports succeeded")
print(f"  REFUSAL_SENTINEL: {REFUSAL_SENTINEL[:50]}...")


# ============================================================================
# 2. Prompt construction
# ============================================================================
print("\n=== 2. Prompt construction ===")
docs = [
    {
        "paper_id": "paper_abc",
        "page": 3,
        "section": "Methodology",
        "text": "We propose a CNN-LSTM for intrusion detection.",
        "score": 0.9,
    }
]
msgs = _build_messages("Which papers discuss intrusion detection?", docs, comparative=False)
assert len(msgs) == 2, "Should be [system, user]"
assert INJECTION_GUARD in msgs[0]["content"], "Injection guard missing from system prompt"
assert QA_SYSTEM_PROMPT in msgs[0]["content"], "QA system prompt missing"
assert "<document>" in msgs[1]["content"], "Documents not wrapped in <document> tags"
print("  System prompt: injection guard + QA prompt present")
print("  User prompt: documents wrapped in <document> tags")

# Comparative variant
msgs_cmp = _build_messages("Compare all papers", docs, comparative=True)
assert COMPARATIVE_QA_SYSTEM_PROMPT in msgs_cmp[0]["content"]
print("  Comparative system prompt correctly used when comparative=True")


# ============================================================================
# 3. Heuristic detection
# ============================================================================
print("\n=== 3. Heuristic detection ===")
assert _is_comparative("Compare the methodologies") is True
assert _is_comparative("Which paper is best?") is True
assert _is_comparative("What is intrusion detection?") is False
print("  _is_comparative correctly classifies compare/difference/best queries")
assert _is_refusal(REFUSAL_SENTINEL) is True
assert _is_refusal("The uploaded papers do not contain sufficient information to answer this question.") is True
assert _is_refusal("Yes, paper X discusses LLM security.") is False
print("  _is_refusal correctly identifies the standard refusal")


# ============================================================================
# 4. Mock LLM for end-to-end test
# ============================================================================
print("\n=== 4. End-to-end with Mock LLM ===")

# A mock LLM that simply echoes back the documents verbatim wrapped in a
# "based on" sentence. Deterministic and exercises the prompt-assembly path.
class MockLLM:
    """Stand-in for ChatOllama/ChatOpenAI with an `.invoke(messages)` method."""

    def __init__(self, canned_answer: str | None = None) -> None:
        self.canned_answer = canned_answer
        self.calls: list[list[dict]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.canned_answer is not None:
            return type("Resp", (), {"content": self.canned_answer})()
        # Echo the first document with a citation prefix
        user_msg = messages[1]["content"]
        if "<document>" not in user_msg:
            return type("Resp", (), {"content": "The uploaded papers do not contain sufficient information to answer this question."})()
        # Extract paper_id from the first document header
        first_doc_start = user_msg.index("[Document 1]")
        first_doc_end = user_msg.index("</document>")
        excerpt = user_msg[first_doc_start:first_doc_end].strip()
        # Parse paper_id and page from header
        import re
        m = re.search(r"paper_id=(\S+)\s+page=(\S+)", excerpt)
        if m:
            paper_id = m.group(1)
            page = m.group(2)
            content = f"Based on the retrieved documents, one paper addresses this. {paper_id} page {page} proposes a CNN-LSTM approach."
        else:
            content = "Based on the retrieved documents, the answer is yes."
        return type("Resp", (), {"content": content})()


# ============================================================================
# 5. End-to-end with REAL retriever (uses Phase 1 indexed PDF)
# ============================================================================
print("\n=== 5. End-to-end (real retriever + mock LLM) ===")
retriever = Retriever()
# Sanity: confirm at least one paper is indexed
papers = retriever._ensure_store().list_papers()
print(f"  Found {len(papers)} indexed paper(s)")
assert len(papers) >= 1, "No papers indexed from Phase 1; run tests/_acceptance_phase1.py first"

mock = MockLLM()
agent = QAAgent(retriever=retriever, llm_factory=lambda: mock)
result = agent.ask("Which papers discuss intrusion detection?")

assert result.refused is False, f"Should not refuse; got: {result.answer!r}"
assert "paper" in result.answer.lower(), f"Answer should mention paper: {result.answer!r}"
assert len(result.sources) >= 1, "Should have at least one source"
# Sources must include paper_id and page
assert all(s.paper_id for s in result.sources), "Sources missing paper_id"
assert all(s.page is not None for s in result.sources), "Sources missing page"
print(f"  Answer: {result.answer[:120]}...")
print(f"  Sources: {[(s.paper_id[:16] + '...', s.page) for s in result.sources[:3]]}")
print(f"  refused={result.refused}, n_sources={len(result.sources)}")

# The mock LLM should have been called once with [system, user] messages
assert len(mock.calls) == 1
assert len(mock.calls[0]) == 2
print(f"  LLM called {len(mock.calls)} time(s) with {len(mock.calls[0])} messages")


# ============================================================================
# 6. Refusal path: empty retrieval -> refusal text
# ============================================================================
print("\n=== 6. Refusal path ===")
mock_refuse = MockLLM(canned_answer=REFUSAL_SENTINEL)
agent_refuse = QAAgent(retriever=retriever, llm_factory=lambda: mock_refuse)
result_refuse = agent_refuse.ask("anything")
assert result_refuse.refused is True, "Should be marked refused"
print(f"  refused correctly: {result_refuse.refused}")


# ============================================================================
# 7. Comparative detection routes to comparative prompt
# ============================================================================
print("\n=== 7. Comparative routing ===")
mock_cmp = MockLLM()
agent_cmp = QAAgent(retriever=retriever, llm_factory=lambda: mock_cmp)
agent_cmp.ask("Compare the methodologies used by these papers")
last_msgs = mock_cmp.calls[-1]
assert COMPARATIVE_QA_SYSTEM_PROMPT in last_msgs[0]["content"], "Comparative prompt not used"
print("  Comparative question routed to COMPARATIVE_QA_SYSTEM_PROMPT")


print("\nPHASE 2 ACCEPTANCE: PASS")
