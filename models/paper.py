"""Pydantic models for paper analysis, citations, and agent communication.

These models are the canonical contract between:
- Ingestion → vector store (chunks, metadata)
- Vector store → agents (RetrievedChunk, AgentContext)
- Agents → coordinator / UI (PaperAnalysis, SourceReference)

All models are strict: missing fields are explicit `None` or empty lists,
never silently omitted. This is the "no hallucination" guarantee
(AGENTS.md §11.2, §24).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============================================================================
# Source / citation primitives
# ============================================================================
class SourceReference(BaseModel):
    """A single citation pointing to a chunk in a paper."""

    model_config = ConfigDict(frozen=False, extra="ignore", populate_by_name=True)

    paper_id: str = Field(default="", description="Stable identifier for the paper.")
    page: int | None = Field(default=None, description="1-based page number, if known.")
    section: str | None = Field(default=None, description="Section name (e.g. 'Methodology').")
    chunk_id: str | None = Field(default=None, description="Stable chunk identifier.")
    text: str | None = Field(default=None, description="Short excerpt supporting the claim.")
    # Common LLM-emitted aliases — accepted but always read into `text`
    # in the validator below, so the rest of the code only ever sees
    # the canonical field.
    content: str | None = Field(default=None, exclude=True)
    excerpt: str | None = Field(default=None, exclude=True)
    quote: str | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _coerce_aliases(cls, data: Any) -> Any:
        """Accept the field aliases LLMs commonly invent (content,
        excerpt, quote) and copy them into `text`. Also accept `page_number`
        as an alias for `page`. Any other unexpected keys are dropped
        silently (extra='ignore') so a single bad source doesn't fail
        the whole agent output."""
        if not isinstance(data, dict):
            return data
        # text <- content / excerpt / quote (first non-empty wins)
        if not data.get("text"):
            for alias in ("content", "excerpt", "quote"):
                v = data.get(alias)
                if v:
                    data["text"] = v
                    break
        # page <- page_number
        if data.get("page") is None and data.get("page_number") is not None:
            data["page"] = data["page_number"]
        return data


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store, plus its similarity score."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    paper_id: str
    filename: str
    title: str | None = None
    page: int | None = None
    section: str | None = None
    text: str
    score: float | None = Field(default=None, description="Similarity score (0..1).")
    # Pre-extracted paper-level metadata (optional). When present,
    # Coordinator._build_context uses these so analysis agents see
    # bibliographic fields without re-deriving them from chunk text.
    authors: list[str] | None = None
    publication_date: str | None = None
    venue: str | None = None
    doi: str | None = None
    institution: str | None = None


# ============================================================================
# Agent I/O
# ============================================================================
class AgentContext(BaseModel):
    """Shared context handed to every agent.

    Built by the coordinator from retrieved chunks and any pre-computed
    metadata. Agents MUST treat `retrieved_chunks` as untrusted data
    (prompt-injection shield; AGENTS.md §32).
    """

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Pre-extracted PDF metadata (title, authors, etc.).",
    )
    previous_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Outputs of agents that ran earlier (for cross-agent use).",
    )


# ============================================================================
# Per-agent output models
# ============================================================================
class MetadataResult(BaseModel):
    """Metadata Agent output (AGENTS.md §11.2)."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_date: str | None = None
    venue: str | None = Field(default=None, description="Conference or journal name.")
    doi: str | None = None
    institution: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)


class SolutionResult(BaseModel):
    """Solution / Contribution Agent output (AGENTS.md §13)."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    problem_statement: str | None = None
    proposed_solution: str | None = None
    methodology: str | None = None
    algorithms: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    experiments: str | None = None
    key_results: str | None = None
    limitations: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)


class SummaryResult(BaseModel):
    """Summary Agent output — 100–250 word technical summary (AGENTS.md §14)."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    summary: str
    problem: str | None = None
    method: str | None = None
    results: str | None = None
    contribution: str | None = None
    limitations: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)


# AGENTS.md §12 — taxonomy used by the Cybersecurity Agent.
CYBERSECURITY_TAXONOMY: list[str] = [
    "Network Security",
    "Application Security",
    "Cloud Security",
    "Container Security",
    "IoT Security",
    "Mobile Security",
    "Endpoint Security",
    "Malware",
    "Ransomware",
    "Intrusion Detection",
    "Threat Detection",
    "Threat Intelligence",
    "Vulnerability Management",
    "Penetration Testing",
    "Digital Forensics",
    "Privacy",
    "Cryptography",
    "Authentication",
    "Authorization",
    "Identity Management",
    "Zero Trust",
    "Security Operations",
    "SIEM",
    "SOC",
    "Adversarial Machine Learning",
    "AI Security",
    "LLM Security",
    "Agent Security",
    "Multi-Agent Security",
    "Cyber-Physical Systems",
    "OT/ICS Security",
    "Blockchain Security",
    "Quantum Security",
    "Supply Chain Security",
    "Other",
]


class CybersecurityResult(BaseModel):
    """Cybersecurity Classification Agent output (AGENTS.md §12)."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    areas: list[str] = Field(
        default_factory=list,
        description="One or more labels from CYBERSECURITY_TAXONOMY.",
    )
    rationale: str | None = Field(
        default=None,
        description="Brief, evidence-grounded justification.",
    )
    sources: list[SourceReference] = Field(default_factory=list)


# ============================================================================
# Final structured analysis (AGENTS.md §18)
# ============================================================================
class PaperAnalysis(BaseModel):
    """The final, validated output for one paper.

    This is what the UI table renders and what /api/papers returns.
    """

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    filename: str | None = Field(
        default=None,
        description="Original PDF filename. Used by the UI to correlate "
        "freshly-uploaded papers with their analysis result.",
    )
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_date: str | None = None
    summary: str | None = None
    cybersecurity_area: list[str] = Field(default_factory=list)
    problem_statement: str | None = None
    proposed_solution: str | None = None
    methodology: str | None = None
    datasets: list[str] = Field(default_factory=list)
    key_results: str | None = None
    limitations: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)
    analysis_status: Literal["pending", "ok", "failed"] = "pending"
    analysis_error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# QA agent output (AGENTS.md §15, §34)
# ============================================================================
class QAResult(BaseModel):
    """Question Answering Agent output."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    refused: bool = Field(
        default=False,
        description="True when evidence was insufficient (AGENTS.md §15).",
    )
