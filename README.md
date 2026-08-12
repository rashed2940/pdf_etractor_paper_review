# Cybersecurity Paper Analyzer

A multi-agent RAG (Retrieval-Augmented Generation) system for cybersecurity research papers. Upload PDFs of scientific papers; the system extracts text, chunks, embeds, and runs four specialized agents in parallel to produce a structured analysis. Ask questions and get grounded answers with paper + page citations.

Built per [AGENTS.md](AGENTS.md) (spec) and [PLAN.md](PLAN.md) (execution roadmap). See [PHASE_LOG.md](PHASE_LOG.md) for the complete build history.

---

## Features

- 📄 **PDF ingestion** (PyMuPDF) with section-aware chunking
- 🤖 **4 parallel agents** (Metadata / Cybersecurity / Solution / Summary) producing validated structured output
- 🔍 **Semantic search** with ChromaDB + optional BM25 hybrid + cross-encoder rerank hooks
- 🌐 **Flask web UI** (Bootstrap 5, vanilla JS) with upload, status polling, results table, Q&A
- 💾 **SQLite persistence** for analyzed papers (survives restart)
- 🛡️ **Prompt-injection shield** on every agent (PDF text is untrusted data)
- 📑 **Citation-aware answers** with paper + page references
- 🔄 **Pluggable LLM**: Ollama via the native `ollama` Python package (default) or OpenAI-compatible API
- 🔄 **Pluggable embeddings**: Ollama `embeddinggemma` (default), SentenceTransformers, or OpenAI
- 🎯 **100–250 word summaries** + per-aspect breakdown (problem/method/results/contribution/limitations)
- 📚 **Cybersecurity taxonomy** (35 labels) for classification

---

## Quick Start

### 1. Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **Ollama** for the default LLM ([install](https://ollama.com))
  - `ollama serve` (in one terminal)
  - `ollama pull llama3.1:8b` (~4.9 GB download)
  - `ollama pull embeddinggemma` (~621 MB download)
- **Tesseract** (optional, for OCR fallback)
  - Windows: download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
  - macOS: `brew install tesseract`
  - Linux: `apt install tesseract-ocr`

### 2. Install

```powershell
git clone <repo-url>
cd pdf_etractor_paper_review
pip install -r requirements.txt
cp .env.example .env
```

### 3. Run

```powershell
python app.py
```

Open http://127.0.0.1:5000 in your browser.

### 4. Use

1. **Upload** PDF files via the upload form
2. **Watch** status chips update in real time (queued → extracting → analyzing → done)
3. **Browse** results in the table (sort, search, filter by cybersecurity area)
4. **Click** any row to expand inline details, or click the title for the full paper page
5. **Ask** questions in the Q&A page — answers cite paper + page

---

## Configuration

All configuration lives in `.env` (copy from `.env.example`). Key settings:

```dotenv
# === LLM ===
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434

# === Embeddings ===
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=embeddinggemma:latest

# === Storage ===
CHROMA_PERSIST_DIRECTORY=./data/chroma
SQLITE_PATH=./data/app.sqlite

# === OCR (opt-in) ===
OCR_ENABLED=false
```

### Switching Models

The application supports multiple LLM providers and models with NO code changes:

#### Use OpenAI instead of Ollama
```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

#### Use different models per agent
```dotenv
LLM_MODEL=llama3.1:8b                     # default
LLM_MODEL_SUMMARIZATION=qwen2.5-coder:7b  # faster for summaries
LLM_MODEL_CYBERSECURITY=llama3.1:8b       # domain classification
LLM_MODEL_SOLUTION=qwen2.5-coder:14b      # deeper reasoning
LLM_MODEL_QA=llama3.1:8b                  # grounded answers
```

#### Switch embeddings to SentenceTransformers
```dotenv
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2   # 384-dim, CPU-friendly
# EMBEDDING_MODEL=BAAI/bge-large-en-v1.5                   # 1024-dim, better retrieval
```

#### Switch embeddings to a different Ollama model
```dotenv
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
# EMBEDDING_MODEL=mxbai-embed-large
# EMBEDDING_MODEL=all-minilm
```

All alternative providers/models are documented inline in [config/settings.py](config/settings.py) and [embeddings/embedding_service.py](embeddings/embedding_service.py) as commented-out code blocks.

---

## Architecture

```text
Flask App
   ↓
Background thread per PDF
   ↓
PDF Ingestion → Cleaner → Chunker → Embedder → ChromaDB
                                                       ↓
                          AgentContext (chunks + metadata)
                                                       ↓
                Coordinator (parallel via ThreadPoolExecutor)
                  ├── MetadataAgent      → MetadataResult
                  ├── CybersecurityAgent → CybersecurityResult
                  ├── SolutionAgent      → SolutionResult
                  └── SummaryAgent       → SummaryResult
                                                       ↓
                            PaperAnalysis (merged)
                                                       ↓
                      ResultsStore (SQLite + memory cache)
                                                       ↓
                              Flask UI table
```

For RAG questions, `Retriever` queries ChromaDB → builds prompt with prompt-injection guard → `QAAgent` returns `QAResult` with answer + sources.

See [PLAN.md §2](PLAN.md) for the full architecture diagram.

---

## Project Structure

```text
pdf_etractor_paper_review/
├── app.py                       # Flask entrypoint
├── config/
│   ├── settings.py              # Env config + LLM/embedding factories
│   └── logging.py               # Structured logging with secret scrubbing
├── agents/
│   ├── base.py                  # BaseAgent (JSON parse + retry/repair)
│   ├── coordinator.py           # Parallel orchestration
│   ├── metadata_agent.py
│   ├── cybersecurity_agent.py
│   ├── solution_agent.py
│   ├── summary_agent.py
│   └── qa_agent.py
├── ingestion/
│   ├── pdf_loader.py
│   ├── ocr_loader.py            # Tesseract fallback
│   ├── metadata_extractor.py
│   ├── text_cleaner.py
│   └── chunker.py
├── llm/
│   ├── __init__.py
│   └── ollama_client.py            # Native `ollama` package wrapper
├── embeddings/
│   └── embedding_service.py        # Ollama + SentenceTransformers + alternatives
├── vectorstore/
│   └── chroma_store.py
├── rag/
│   ├── prompts.py               # Centralized prompts + injection guard
│   ├── retriever.py
│   ├── chain.py
│   ├── bm25.py                  # Optional hybrid search
│   ├── hybrid.py                # RRF merge
│   └── reranker.py              # Optional cross-encoder hook
├── models/
│   └── paper.py                 # Pydantic models (PaperAnalysis, etc.)
├── services/
│   ├── paper_service.py         # PDF → indexed chunks
│   ├── analysis_service.py      # Coordinator + QAAgent facade
│   ├── processing_runner.py     # Background threading
│   ├── processing_status.py     # Thread-safe status registry
│   └── results_store.py         # SQLite + memory cache
├── templates/                   # Jinja + Bootstrap 5
├── static/                      # CSS + JS
├── data/
│   ├── uploads/                 # User PDFs
│   ├── chroma/                  # Persistent vector DB
│   └── app.sqlite               # Analyzed papers
├── tests/                       # 76 pytest tests (no real Ollama needed)
├── AGENTS.md                    # Spec
├── PLAN.md                      # Execution roadmap
├── PHASE_LOG.md                 # Build history
└── README.md                    # This file
```

---

## Testing

```powershell
# Run all 76 tests
python -m pytest --tb=short tests/

# Run just one phase
python -m pytest tests/test_phase3_agents.py

# Run a specific test
python -m pytest tests/test_phase4_coordinator.py::TestCoordinator::test_parallel_execution_speedup
```

All tests use mocked LLMs (no real Ollama needed). Tests run in ~17 seconds.

For per-phase acceptance scripts (more verbose end-to-end checks), see `tests/_acceptance_phaseN.py`.

---

## Development Phases

| Phase | Goal | Status |
|---|---|---|
| 0 | Foundations (scaffold, config, models, logging) | ✅ |
| 1 | Ingestion (PDF → chunks → ChromaDB) | ✅ |
| 2 | RAG + QA Agent (grounded answers with citations) | ✅ |
| 3 | Specialized Agents (Metadata/Cyber/Solution/Summary) | ✅ |
| 4 | Coordinator (parallel execution + merging) | ✅ |
| 5 | Flask UI + Background Processing | ✅ |
| 6 | Quality-of-Life Hooks (BM25 hybrid, rerank, OCR, SQLite) | ✅ |
| 7 | pytest Suite (76 tests, CI-friendly) | ✅ |

See [PHASE_LOG.md](PHASE_LOG.md) for per-phase details, bug fixes, and architectural decisions.

---

## Security

- ✅ Filename sanitization via `werkzeug.secure_filename`
- ✅ Upload extension whitelist (`.pdf` only)
- ✅ Upload size limit (default 50 MB)
- ✅ Jinja autoescape on (no raw HTML rendering)
- ✅ Secret scrubbing in logs (API keys, tokens never logged)
- ✅ Prompt-injection shield on every agent (PDF text is untrusted data)
- ✅ Per-paper exception isolation (one bad PDF doesn't break batch)
- ✅ Citation-backed answers (no fabrication)

---

## License

Internal project. See spec for details.