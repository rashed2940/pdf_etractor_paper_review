# PLAN.md — Cybersecurity Paper RAG Implementation Plan

> Companion to `AGENTS.md`. This is the **execution roadmap**. Update it when decisions change.

**Date:** 2026-08-10
**Status:** Approved — awaiting Phase 0 kickoff

---

## 1. Locked-in Decisions

| Area | Choice |
|---|---|
| Default LLM | Ollama `qwen2.5-coder:14b-instruct` |
| OCR | Implement fallback now (Tesseract); opt-in via `OCR_ENABLED` env var |
| Processing | Background threads per upload; UI polls status |
| UI | Jinja + Bootstrap 5 + vanilla JS |
| Embeddings (default) | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persistent at `./data/chroma`) |
| PDF parser (text) | PyMuPDF |
| RAG framework | LangChain |
| Config | `pydantic-settings` |
| App metadata store | SQLite at `./data/app.sqlite` |

---

## 2. Project Structure

```text
pdf_etractor_paper_review/
├── AGENTS.md                          (spec)
├── PLAN.md                            (this file)
├── README.md                          (install, run, usage)
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py                             (Flask entrypoint)
├── config/
│   ├── __init__.py
│   └── settings.py                    (env loader, LLM/embedding factories)
├── agents/
│   ├── __init__.py
│   ├── base.py                        (BaseAgent, AgentContext, retry/repair)
│   ├── coordinator.py
│   ├── metadata_agent.py
│   ├── cybersecurity_agent.py
│   ├── solution_agent.py
│   ├── summary_agent.py
│   └── qa_agent.py
├── ingestion/
│   ├── __init__.py
│   ├── pdf_loader.py                  (PyMuPDF text extraction)
│   ├── ocr_loader.py                  (Tesseract fallback)
│   ├── metadata_extractor.py          (PDF metadata + regex heuristics)
│   ├── text_cleaner.py
│   └── chunker.py                     (section-aware → recursive fallback)
├── embeddings/
│   ├── __init__.py
│   └── embedding_service.py           (abstract + SentenceTransformers impl)
├── vectorstore/
│   ├── __init__.py
│   └── chroma_store.py                (persistent ChromaDB)
├── rag/
│   ├── __init__.py
│   ├── retriever.py                   (similarity + filters + top_k)
│   ├── chain.py                       (QA chain with citations)
│   └── prompts.py                     (centralized prompts + injection guards)
├── models/
│   ├── __init__.py
│   └── paper.py                       (PaperAnalysis, SourceReference, AgentContext)
├── services/
│   ├── __init__.py
│   ├── paper_service.py               (ingest → index)
│   ├── analysis_service.py            (agents → PaperAnalysis)
│   └── processing_status.py           (thread-safe status registry)
├── templates/
│   ├── base.html
│   ├── index.html                     (upload + progress chips)
│   ├── results.html                   (sortable/searchable table)
│   ├── question.html                  (Q&A interface)
│   └── partials/
│       ├── paper_row.html
│       └── status_chip.html
├── static/
│   ├── css/style.css
│   └── js/app.js
├── data/
│   ├── uploads/                       (gitignored)
│   ├── chroma/                        (gitignored, persistent)
│   └── app.sqlite                     (gitignored)
└── tests/
    ├── conftest.py                    (mocked LLM, temp dirs)
    ├── test_pdf_loader.py
    ├── test_ocr_loader.py
    ├── test_text_cleaner.py
    ├── test_chunker.py
    ├── test_metadata_extractor.py
    ├── test_embedding_service.py
    ├── test_vectorstore.py
    ├── test_retriever.py
    ├── test_prompts.py
    ├── test_models.py
    ├── test_agents.py                 (mocked LLM)
    ├── test_coordinator.py
    ├── test_rag_chain.py
    └── test_flask_routes.py
```

---

## 3. Phased Execution

### Phase 0 — Foundations
**Goal:** Scaffold the project; nothing runs yet but everything imports.

- [ ] Create directory tree
- [ ] `requirements.txt` with pinned versions
- [ ] `.env.example`, `.gitignore`
- [ ] `config/settings.py` using `pydantic-settings`
- [ ] `models/paper.py` — `PaperAnalysis`, `SourceReference`, `AgentContext`, `RetrievedChunk`
- [ ] Structured logging (`logging.config.dictConfig`)
- [ ] `processing_status.py` thread-safe registry

**Acceptance:** `python -c "import config.settings; import models.paper"` succeeds.

---

### Phase 1 — Ingestion Pipeline
**Goal:** PDFs in → embeddings in ChromaDB out.

- [ ] `ingestion/pdf_loader.py` — PyMuPDF, page-tracked extraction, scanned detection
- [ ] `ingestion/ocr_loader.py` — Tesseract fallback (only when text < threshold)
- [ ] `ingestion/metadata_extractor.py` — PDF metadata + DOI/author regex + first-page heuristics
- [ ] `ingestion/text_cleaner.py` — hyphenation, whitespace, headers/footers, ligatures
- [ ] `ingestion/chunker.py` — section-aware (Abstract/Intro/Methodology/…) → recursive fallback
- [ ] `embeddings/embedding_service.py` — abstract `EmbeddingService` + SentenceTransformers impl
- [ ] `vectorstore/chroma_store.py` — persistent client, upsert with full metadata
- [ ] SHA-256 duplicate detection
- [ ] `services/paper_service.py` — orchestrator

**Acceptance:** CLI `python -m services.paper_service ./sample_pdfs` indexes N PDFs and `vectorstore.query("intrusion detection", k=3)` returns relevant chunks with page numbers.

---

### Phase 2 — RAG + QA Agent
**Goal:** Ask questions; get grounded answers with citations.

- [ ] `rag/prompts.py` — centralized prompts with prompt-injection guards (AGENTS.md §32)
- [ ] `rag/retriever.py` — similarity + metadata filters + top_k + score threshold
- [ ] `rag/chain.py` — LangChain retrieval chain with citations
- [ ] `agents/qa_agent.py` — multi-paper comparative QA, grounded refusal

**Acceptance:** Asking "Which papers discuss LLM security?" returns answer with `Paper X — Page Y` references.

---

### Phase 3 — Specialized Analysis Agents
**Goal:** Each agent produces validated structured output.

- [ ] `agents/base.py` — `BaseAgent.run(context) -> BaseModel` with JSON parse + Pydantic validation + retry/repair
- [ ] `agents/metadata_agent.py` — title, authors, date, venue, DOI, institution
- [ ] `agents/cybersecurity_agent.py` — multi-label from taxonomy
- [ ] `agents/solution_agent.py` — problem, solution, method, datasets, results, limitations
- [ ] `agents/summary_agent.py` — 100–250 word technical summary

**Acceptance:** Each agent unit-tested with mocked LLM returning canned JSON.

---

### Phase 4 — Coordinator + Parallel Execution
**Goal:** End-to-end `PaperAnalysis` for a paper; multi-paper comparison for QA.

- [ ] `agents/coordinator.py` — builds shared `AgentContext`, fans out via `asyncio.gather` (or threads for sync LLM clients), merges results, produces `PaperAnalysis`
- [ ] Multi-paper comparative path in `agents/qa_agent.py`
- [ ] `services/analysis_service.py` — `analyze_paper(paper_id)`, `compare_papers(question)`

**Acceptance:** Coordinator runs all 4 analysis agents; final Pydantic model validates.

---

### Phase 5 — Flask UI + Background Processing
**Goal:** Full user experience.

- [ ] Background processing: `threading.Thread` per PDF; status in `processing_status.py`
- [ ] `app.py` endpoints:
  - `GET /` — upload form + status panel
  - `POST /upload` — save PDFs, spawn threads, return `upload_id`
  - `GET /status/<upload_id>` — JSON status for polling
  - `POST /process` — manual reprocess trigger
  - `GET /results` — render table
  - `GET /paper/<paper_id>` — detail view
  - `POST /ask` — Q&A
  - `GET /api/papers`, `POST /api/query` — REST
- [ ] Templates with Bootstrap 5:
  - `base.html` — nav, layout
  - `index.html` — multi-file picker, status chips
  - `results.html` — searchable/sortable/filterable table
  - `question.html` — Q&A form + sources panel
- [ ] `static/js/app.js` — async polling, table sort/search
- [ ] Security: `secure_filename`, extension whitelist, max size 50 MB, Jinja autoescape

**Acceptance:** Upload 5 PDFs → see progress → table populates → ask question → get cited answer.

---

### Phase 6 — Quality-of-Life Hooks
**Goal:** Extensibility without over-engineering.

- [ ] Hybrid search interface (BM25 hook)
- [ ] Cross-encoder rerank hook
- [ ] OCR fully wired (auto-fallback in `pdf_loader.py`)
- [ ] Pagination + expandable row details
- [ ] SQLite metadata store for upload history

**Acceptance:** Hooks exist as empty/working classes; nothing breaks if disabled.

---

### Phase 7 — Testing
**Goal:** Reliable test suite; no real LLM needed in CI.

- [ ] `tests/conftest.py` — temp dirs, mocked embeddings, mocked LLM (canned JSON), Flask test client
- [ ] Unit tests for every module
- [ ] `pytest tests/` passes
- [ ] Coverage target: ≥ 70%

**Acceptance:** `pytest -q` all green.

---

## 4. Configuration (`.env.example`)

```dotenv
# ===== LLM (general: chat / analysis / summarization) =====
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5-coder:14b-instruct
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=
OPENAI_BASE_URL=
TEMPERATURE=0

# ===== Specialized LLM roles (override per-task models) =====
# Set any of these to override LLM_MODEL for that role.
# Leave empty to fall back to LLM_MODEL.
LLM_MODEL_SUMMARIZATION=        # e.g. qwen2.5-coder:7b-instruct (faster summary)
LLM_MODEL_CYBERSECURITY=        # e.g. llama3.1:8b (domain classification)
LLM_MODEL_SOLUTION=             # e.g. qwen2.5-coder:14b-instruct (deep reasoning)
LLM_MODEL_METADATA=             # e.g. qwen2.5-coder:7b-instruct (extraction)
LLM_MODEL_QA=                   # e.g. qwen2.5-coder:14b-instruct (RAG answer)

# ===== Embeddings =====
# Default: local SentenceTransformers
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# Optional override (kept commented — uncomment to switch):
# EMBEDDING_PROVIDER=openai
# EMBEDDING_MODEL=text-embedding-3-small
# OPENAI_EMBEDDING_API_KEY=
# EMBEDDING_PROVIDER=ollama
# EMBEDDING_MODEL=nomic-embed-text

# ===== RAG =====
TOP_K=5
SCORE_THRESHOLD=0.0

# ===== Chunking =====
CHUNK_SIZE=1000
CHUNK_OVERLAP=150

# ===== Storage =====
CHROMA_PERSIST_DIRECTORY=./data/chroma
UPLOAD_DIRECTORY=./data/uploads
SQLITE_PATH=./data/app.sqlite

# ===== OCR (opt-in) =====
OCR_ENABLED=false
OCR_MIN_CHARS_PER_PAGE=50
TESSERACT_CMD=

# ===== Flask =====
MAX_UPLOAD_MB=50
SECRET_KEY=change-me

# ===== Logging =====
LOG_LEVEL=INFO
```

---

## 5. Key Architectural Principles

1. **Pluggable everywhere.** `EmbeddingService`, `VectorStore`, `PDFLoader`, `LLMClient` are abstractions. Chroma→Qdrant, Ollama→OpenAI, PyMuPDF→OCR are local swaps (AGENTS.md §40).
2. **Prompt injection shield.** System prompt explicitly states retrieved text is untrusted data; PDF content wrapped in `<document>` delimiters (AGENTS.md §32).
3. **Citations retained.** Every chunk carries `paper_id`, `page`, `section`, `chunk_id`; agents reference pages; UI shows them (AGENTS.md §24).
4. **No hallucinated metadata.** Cross-check PDF metadata + first-page regex; missing → `null` (AGENTS.md §11.2).
5. **Parallel where independent.** Metadata, Cybersecurity, Solution, Summary agents run concurrently after context retrieval (AGENTS.md §23).
6. **Validated output.** All LLM output parsed → Pydantic model → retry/repair on failure (AGENTS.md §18).
7. **Don't over-engineer.** Ship Phases 0–5 first; Phase 6 are hooks, not blocking.
8. **LLM swap-points with commented fallbacks.** Every place an LLM or embedding is instantiated must include commented alternative models/providers so users can switch by uncommenting. See §6 for the standard pattern.

---

## 6. LLM & Embedding Swap-Points (Commented Switch Pattern)

**Rule:** Wherever code creates an LLM client or embedding client, include **commented alternative configurations** right next to the active one. Users switch by uncommenting — no need to read docs.

### 6.1 In `config/settings.py` (general LLM factory)

```python
# --- Active: Ollama (default) ---
llm = ChatOllama(model=settings.LLM_MODEL,
                 base_url=settings.OLLAMA_BASE_URL,
                 temperature=0)

# --- Alternative: OpenAI-compatible API (uncomment to switch) ---
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model=settings.LLM_MODEL,
#                  api_key=settings.OPENAI_API_KEY,
#                  base_url=settings.OPENAI_BASE_URL or None,
#                  temperature=0)

# --- Alternative: Anthropic (uncomment to switch) ---
# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model=settings.LLM_MODEL, temperature=0)
```

### 6.2 Per-role LLM factory (different model per agent task)

```python
# --- Per-role LLM selection (each falls back to LLM_MODEL if env unset) ---
def make_llm(role: str) -> ChatOllama:
    role_model = getattr(settings, f"LLM_MODEL_{role.upper()}", None)
    model = role_model or settings.LLM_MODEL

    return ChatOllama(model=model, base_url=settings.OLLAMA_BASE_URL, temperature=0)

    # --- Alternative: OpenAI-compatible (uncomment to switch) ---
    # from langchain_openai import ChatOpenAI
    # return ChatOpenAI(model=model, api_key=settings.OPENAI_API_KEY,
    #                   base_url=settings.OPENAI_BASE_URL or None, temperature=0)

# Usage in agents (each can use a different model):
#   llm = make_llm("summarization")     # uses LLM_MODEL_SUMMARIZATION or LLM_MODEL
#   llm = make_llm("cybersecurity")    # uses LLM_MODEL_CYBERSECURITY or LLM_MODEL
#   llm = make_llm("solution")         # uses LLM_MODEL_SOLUTION or LLM_MODEL
#   llm = make_llm("metadata")         # uses LLM_MODEL_METADATA or LLM_MODEL
#   llm = make_llm("qa")               # uses LLM_MODEL_QA or LLM_MODEL
```

### 6.3 In `embeddings/embedding_service.py`

```python
# --- Active: local SentenceTransformers ---
self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

# --- Alternative: Ollama embeddings (uncomment to switch) ---
# from langchain_ollama import OllamaEmbeddings
# self.model = OllamaEmbeddings(model=settings.EMBEDDING_MODEL,
#                               base_url=settings.OLLAMA_BASE_URL)

# --- Alternative: OpenAI embeddings (uncomment to switch) ---
# from langchain_openai import OpenAIEmbeddings
# self.model = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL,
#                               api_key=settings.OPENAI_API_KEY)

# --- Alternative: stronger local model for higher quality (uncomment to switch) ---
# EMBEDDING_MODEL=BAAI/bge-large-en-v1.5                  # 1024-dim, better retrieval
# EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2 # 768-dim, balanced
```

### 6.4 Switch guide (cheat sheet)

| Want to... | Edit `.env` |
|---|---|
| Use OpenAI instead of Ollama | `LLM_PROVIDER=openai`, set `OPENAI_API_KEY`, `LLM_MODEL=gpt-4o-mini` |
| Use a stronger local LLM | `LLM_MODEL=qwen2.5-coder:32b-instruct` |
| Use a faster local LLM for summaries | `LLM_MODEL_SUMMARIZATION=qwen2.5-coder:7b-instruct` |
| Use a different model per agent | Set `LLM_MODEL_CYBERSECURITY`, `LLM_MODEL_QA`, etc. |
| Use OpenAI embeddings | `EMBEDDING_PROVIDER=openai`, `EMBEDDING_MODEL=text-embedding-3-small` |
| Use better local embeddings | `EMBEDDING_MODEL=BAAI/bge-large-en-v1.5` |
| Use Ollama embeddings | `EMBEDDING_PROVIDER=ollama`, `EMBEDDING_MODEL=nomic-embed-text` |

### 6.5 Why commented-out code (not docs)

- **Discoverable inline:** the alternative is right where you'd change it.
- **One-line swap:** users uncomment, no docs to read.
- **Safe to keep:** commented code never runs; it documents intent.
- **Easy to maintain:** when new providers are added, add another commented block in the same file.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Ollama model not installed | Startup check + clear error (AGENTS.md §30) |
| Malformed LLM JSON | `BaseAgent` retry + Pydantic validation; failed → `processing_status="failed"` |
| Large PDFs slow ingestion | Background threads; UI polls; per-paper try/except so one bad PDF doesn't break batch |
| Prompt injection in PDFs | Hardened system prompt + document-data delimiters; no tool-calling from PDF content |
| Hallucinated metadata | Cross-check PDF metadata + first-page regex; missing → `null` |
| ChromaDB schema drift | Versioned collection name; recreate-on-mismatch with logged warning |
| Long LLM latency | Async fan-out for independent agents (AGENTS.md §23) |
| OCR slows text PDFs | `OCR_ENABLED=false` default; only triggered when text/page < threshold |
| Mixing embedding models silently breaks retrieval | Embedding model name stored in ChromaDB collection metadata; warn on mismatch at load |

---

## 8. Definition of Done (mirrors AGENTS.md §39)

- [ ] `python app.py` starts the server
- [ ] User can upload ≥ 5 PDFs and see per-file status
- [ ] Chunks + embeddings created; ChromaDB persists across restarts
- [ ] `/results` shows table with Title, Authors, Published, Summary, Cyber Area, Solution
- [ ] `/ask` returns grounded answer + source paper + page
- [ ] Multi-paper comparative questions work
- [ ] No stack traces in UI; meaningful error messages
- [ ] All `pytest` tests pass with mocked LLM
- [ ] `README.md` documents install, env vars, Ollama setup, OCR setup, usage
- [ ] Each LLM/embedding call site has commented alternative configurations (PLAN.md §6)

---

## 9. Execution Order (Next Up)

When approved, execute in this order:

1. **Phase 0** — scaffold (directories, requirements, config, models, logging, status registry)
2. **Phase 1** — ingestion → embeddings → ChromaDB (with OCR adapter)
3. **Phase 2** — RAG + QA agent
4. **Phase 3** — specialized analysis agents
5. **Phase 4** — coordinator with parallel execution
6. **Phase 5** — Flask UI (Bootstrap 5) + background processing
7. **Phase 6** — quality-of-life hooks
8. **Phase 7** — pytest suite with mocked LLM

Each phase ends with a working checkpoint; user can stop and resume.
