"""Application configuration loaded from environment variables.

Uses pydantic-settings for typed, validated configuration. Every external
dependency (LLM, embeddings, vector store, PDF loader) is configurable via
.env — no values are hardcoded in business logic.

LLM / Embedding swap-points: this file contains commented alternative
provider blocks right next to the active code so users can switch by
uncommenting (or via env vars). See PLAN.md §6.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed application settings loaded from .env (or env vars).

    All fields have sensible defaults so the app can start without a .env
    file present. Critical secrets (OPENAI_API_KEY) default to empty string.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== LLM (general) =====
    # We use the **native `ollama` Python package** (not langchain-ollama)
    # for both chat and embeddings. See `llm/ollama_client.py` and
    # `embeddings/embedding_service.py`.
    llm_provider: Literal["ollama", "openai"] = Field(
        default="ollama",
        description="Active LLM provider. Switch to 'openai' for OpenAI-compatible APIs.",
    )
    llm_model: str = Field(
        default="qwen2.5-coder:7b-instruct",
        description=(
            "Default chat model. Default: qwen2.5-coder:7b-instruct "
            "(Ollama) — verified to respond quickly to JSON-formatted "
            "agent prompts. Reasoning models like qwen3.x and deepseek-r1 "
            "are auto-detected and get think=False passed automatically."
        ),
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the local Ollama daemon.",
    )
    openai_api_key: str = Field(default="", description="OpenAI-compatible API key.")
    openai_base_url: str = Field(
        default="",
        description="OpenAI-compatible base URL (leave empty for official OpenAI).",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # ===== LLM resilience =====
    # Hard timeout (seconds) for a single LLM chat call. If Ollama
    # doesn't respond in this time, the call raises TimeoutError and
    # the affected agent fails fast (without retry). Prevents the
    # entire background-processing thread from hanging indefinitely.
    #
    # Default 600s (10 min): the analysis agents each send a ~3-5K-token
    # prompt (system prompt with taxonomy + retrieved chunks) and
    # expect ~300-500 tokens of JSON output. On typical CPU/consumer-GPU
    # hardware, that can be 2-8 minutes per agent. 4 agents run in
    # parallel, so wall-clock per paper is bounded by max(agent_time).
    #
    # If your model needs more, raise via LLM_TIMEOUT_SECONDS in .env.
    llm_timeout_seconds: float = Field(
        default=600.0,
        ge=30.0,
        le=3600.0,
        description="Per-request timeout for LLM chat calls (seconds). Default 600s.",
    )

    # Hard cap on output tokens per call (forwarded as Ollama's
    # `num_predict`). Without this, a model that ignores the prompt's
    # length instruction can run for minutes. Default 512 is enough
    # for the JSON outputs our agents need (~200-400 tokens each).
    llm_num_predict: int = Field(
        default=512,
        ge=64,
        le=32768,
        description="Per-call output token cap (Ollama num_predict). Default 512.",
    )

    # ===== Specialized LLM roles (per-agent overrides) =====
    # Each falls back to llm_model if empty.
    llm_model_summarization: str = Field(
        default="", description="Model used by SummaryAgent. Empty -> use llm_model."
    )
    llm_model_cybersecurity: str = Field(
        default="", description="Model used by CybersecurityAgent. Empty -> use llm_model."
    )
    llm_model_solution: str = Field(
        default="", description="Model used by SolutionAgent. Empty -> use llm_model."
    )
    llm_model_metadata: str = Field(
        default="", description="Model used by MetadataAgent. Empty -> use llm_model."
    )
    llm_model_qa: str = Field(
        default="", description="Model used by QAAgent. Empty -> use llm_model."
    )

    # ===== Embeddings =====
    # Native `ollama` package is the active provider. Switching to
    # sentence_transformers (or another provider) only requires changing
    # `embedding_provider` and `embedding_model` here or in .env.
    embedding_provider: Literal["sentence_transformers", "openai", "ollama"] = Field(
        default="ollama",
        description="Active embedding provider. Default: ollama (native).",
    )
    embedding_model: str = Field(
        default="embeddinggemma:latest",
        description=(
            "Embedding model identifier for the active provider. "
            "Default: embeddinggemma (Google's embedding model via Ollama)."
        ),
    )

    # ===== RAG =====
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    # ===== Chunking =====
    # chunk_overlap is intentionally allowed up to chunk_size (with a
    # generous absolute ceiling) so larger sliding-window configurations
    # (e.g. for long, section-aware chunking) are supported. A model
    # validator below auto-clamps overlap to < chunk_size if the user
    # misconfigures it (the chunker cursor would otherwise stall).
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=4000)

    @model_validator(mode="after")
    def _clamp_chunk_overlap(self) -> "Settings":
        # Overlap must be strictly less than chunk_size, otherwise the
        # chunker's cursor `len(piece) - chunk_overlap` never advances.
        if self.chunk_overlap >= self.chunk_size:
            import warnings

            original = self.chunk_overlap
            self.chunk_overlap = max(0, self.chunk_size - 50)
            warnings.warn(
                f"CHUNK_OVERLAP={original} is >= CHUNK_SIZE={self.chunk_size}; "
                f"auto-clamped to {self.chunk_overlap}.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    # ===== Storage =====
    chroma_persist_directory: Path = Field(
        default=PROJECT_ROOT / "data" / "chroma",
        description="Persistent ChromaDB directory.",
    )
    upload_directory: Path = Field(
        default=PROJECT_ROOT / "data" / "uploads",
        description="Directory where uploaded PDFs are saved.",
    )
    sqlite_path: Path = Field(
        default=PROJECT_ROOT / "data" / "app.sqlite",
        description="SQLite path for application metadata.",
    )

    # ===== OCR (opt-in) =====
    ocr_enabled: bool = Field(default=False)
    ocr_min_chars_per_page: int = Field(default=50, ge=1)
    tesseract_cmd: str = Field(default="", description="Path to tesseract binary.")

    # ===== Flask =====
    max_upload_mb: int = Field(default=50, ge=1, le=2048)
    secret_key: str = Field(default="change-me")

    # ===== Logging =====
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )

    # ===== Helpers =====
    def model_for_role(self, role: str) -> str:
        """Return the model name to use for a given agent role.

        Role names match the suffix of LLM_MODEL_<ROLE> env vars:
        "summarization", "cybersecurity", "solution", "metadata", "qa".
        Falls back to the global llm_model when no per-role override is set.
        """
        attr = f"llm_model_{role.lower()}"
        override = getattr(self, attr, "") or ""
        return override.strip() or self.llm_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Read this instead of instantiating Settings directly."""
    return Settings()


# Convenience handle for static import sites (e.g. log formatters).
settings = get_settings()


# ============================================================================
# LLM factory — single switch point between providers.
# ============================================================================
# Per PLAN.md §6.2: each call site may use make_llm(role) to get a client
# tuned for that agent. The active provider is Ollama; OpenAI-compatible is
# documented inline as a commented alternative. To switch, either set
# LLM_PROVIDER=openai in .env OR uncomment the OpenAI block below.
# ============================================================================
def make_llm(
    role: str = "default",
    temperature: float | None = None,
):
    """Construct a chat model for the given role.

    Args:
        role: One of "default", "summarization", "cybersecurity", "solution",
              "metadata", "qa". Determines which model name is used.
        temperature: Override temperature; defaults to settings.temperature.

    Returns:
        An object with `.invoke(messages)` returning an object with `.content`.
        Compatible with the existing agents / QA chain.

    Notes:
        Models are loaded lazily (only when make_llm is called) so the
        config module can be imported in environments without Ollama.
        Per-role model names come from `LLM_MODEL_<ROLE>` env vars
        (defaults to `LLM_MODEL`).
    """
    s = get_settings()
    model_name = s.model_for_role(role)
    temp = settings.temperature if temperature is None else temperature
    options = {"temperature": temp} if temp is not None else {}

    if s.llm_provider == "ollama":
        # --- Active: native Ollama client ---
        from llm.ollama_client import OllamaChatModel

        return OllamaChatModel(
            model=model_name,
            host=s.ollama_base_url,
            options=options,
            timeout=s.llm_timeout_seconds,
            num_predict=s.llm_num_predict,
        )

        # --- Alternative: OpenAI-compatible API (uncomment to switch) ---
        # from openai import OpenAI
        # class _OpenAIChatModel:
        #     def __init__(self):
        #         self.client = OpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url or None)
        #         self.model = model_name
        #     def invoke(self, messages):
        #         resp = self.client.chat.completions.create(
        #             model=self.model,
        #             messages=messages,
        #             temperature=temp,
        #         )
        #         return _OllamaResponse(resp.choices[0].message.content or "")
        # return _OpenAIChatModel()

        # --- Alternative: Anthropic (uncomment + `pip install anthropic`) ---
        # import anthropic
        # class _AnthropicChatModel:
        #     def __init__(self):
        #         self.client = anthropic.Anthropic()
        #         self.model = model_name
        #     def invoke(self, messages):
        #         # Anthropic uses a separate "system" parameter.
        #         system = next((m["content"] for m in messages if m["role"] == "system"), "")
        #         rest = [m for m in messages if m["role"] != "system"]
        #         resp = self.client.messages.create(
        #             model=self.model, system=system, max_tokens=2048,
        #             temperature=temp, messages=rest,
        #         )
        #         return _OllamaResponse(resp.content[0].text)
        # return _AnthropicChatModel()

    raise ValueError(
        f"Unsupported LLM_PROVIDER={s.llm_provider!r}. "
        "Use 'ollama' (active) or 'openai' (requires manual wiring)."
    )


# ============================================================================
# Ensure runtime data directories exist (idempotent).
# ============================================================================
def ensure_directories() -> None:
    """Create data directories if they don't already exist."""
    s = get_settings()
    for d in (s.upload_directory, s.chroma_persist_directory):
        d.mkdir(parents=True, exist_ok=True)
    # SQLite parent dir
    s.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
