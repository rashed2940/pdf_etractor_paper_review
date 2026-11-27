# Cybersecurity Paper Analyzer

A multi-agent RAG (Retrieval-Augmented Generation) system for cybersecurity research papers. Upload PDFs of scientific papers; the system extracts text, chunks, embeds, and runs four specialized agents in parallel to produce a structured analysis. Ask questions and get grounded answers with paper + page citations, or export the full analysis as a single Excel workbook.

Built per [AGENTS.md](AGENTS.md) (spec) and [PLAN.md](PLAN.md) (execution roadmap). See [PHASE_LOG.md](PHASE_LOG.md) for the complete build history.

---

## Features

- 📄 **PDF ingestion** (PyMuPDF) with section-aware chunking
- 🤖 **4 parallel agents** (Metadata / Cybersecurity / Solution / Summary) producing validated structured output
- 🔍 **Semantic search** with ChromaDB + optional BM25 hybrid + cross-encoder rerank hooks
- 🌐 **Flask web UI** (Bootstrap 5, vanilla JS) with upload, status polling, results table, Q&A, and Excel export
- 💾 **SQLite persistence** for analyzed papers (survives restart)
- 🛡️ **Prompt-injection shield** on every agent (PDF text is untrusted data)
- 📑 **Citation-aware answers** with paper + page references
- 📊 **One-click Excel export** — download every analyzed paper as a single `.xlsx` with full untruncated content
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
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
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
4. **Click** any row title to open the full paper detail page
5. **Export** the full table to Excel with the green **⬇ Export to Excel** button above the table
6. **Ask** questions at `/ask` — answers cite paper + page

---

## Configuration

All configuration lives in `.env` (copy from `.env.example`). Key settings:

```dotenv
# === LLM ===
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
LLM_NUM_PREDICT=4096          # output token cap (raise for long analyses)
LLM_MAX_RETRIES=5             # retries on transient errors (429/5xx/timeouts)
LLM_RETRY_BACKOFF=2.0         # backoff base in seconds (2^n, capped at 60s)
TEMPERATURE=0

# === Embeddings ===
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=embeddinggemma:latest

# === Storage ===
CHROMA_PERSIST_DIRECTORY=./data/chroma
SQLITE_PATH=./data/app.sqlite

# === Chunking / Retrieval ===
CHUNK_SIZE=2000
CHUNK_OVERLAP=400
TOP_K=6
SCORE_THRESHOLD=0.0

# === Throughput ===
PAPER_CONCURRENCY=3            # papers analysed in parallel (see below)

# === Uploads ===
MAX_UPLOAD_MB=2048             # per-request upload limit (1–2048 MB)

# === OCR (opt-in) ===
OCR_ENABLED=false
```

> **Environment precedence.** OS environment variables **override** `.env`
> (this is `pydantic-settings` behaviour). If a setting seems "stuck" after
> editing `.env`, check for a stale shell variable and clear it:
>
> ```powershell
> Get-ChildItem Env: | Where-Object { $_.Name -match '^(LLM_|EMBEDDING_|OLLAMA_|CHROMA_|CHUNK_|UPLOAD_|MAX_|PAPER_|OCR_|SECRET_|LOG_|SQLITE_|TOP_K|SCORE_|TEMPERATURE)' } |
>   ForEach-Object { Remove-Item "Env:\$($_.Name)" }
> ```
>
> Then restart the app. (VS Code's `python.terminal.useEnvFile: true`
> snapshots `.env` into the terminal when the terminal is created, so a
> terminal opened *before* your edit will keep the old values.)

### Throughput tuning — `PAPER_CONCURRENCY`

Each paper is analysed by 4 parallel agents, so an unbounded number of
uploaded PDFs produces an unbounded number of concurrent LLM requests.
That is the usual cause of `429 too many concurrent requests` errors and
of slow, throttled analysis.

`PAPER_CONCURRENCY` puts a semaphore around the per-paper analysis so
at most N papers are in flight at once. The default of **3** keeps the
Ollama instance healthy while still using the hardware well.

| Symptom | Fix |
| --- | --- |
| `429 too many concurrent requests` in the log | Lower `PAPER_CONCURRENCY` to 2 |
| Analysis feels slow, GPU/CPU idle | Raise `PAPER_CONCURRENCY` to 4–6 |
| `Analysis Error: ... output token cap` | Raise `LLM_NUM_PREDICT` |

> **Note:** raising `PAPER_CONCURRENCY` also raises total LLM token use,
> so watch your provider's rate limits.

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

The **Excel export** endpoint serializes every paper in `ResultsStore` into a single `.xlsx` workbook (16 columns of full untruncated content, frozen header row, sized column widths, timestamped filename) and streams it as an attachment download.

See [PLAN.md §2](PLAN.md) for the full architecture diagram.

---

## Metadata Extraction

Getting the title, authors and publication date right is the part of
the pipeline that varies the most between publishers, so it is handled
in two independent places.

### 1. Front matter chunk (structural)

The section-aware chunker starts at the first detected section heading
(`Abstract`, `Introduction`, …). Everything *before* that — the title,
the author block and the affiliations — would therefore be dropped, and
the `MetadataAgent` would never see the authors at all.

`ingestion/chunker.py` now emits that block as its own chunk with the
section name `Front Matter`. It is embedded and stored like any other
chunk, so the agent can cite it (page 1) and the retriever can find it.

### 2. Layout-aware author heuristics

`ingestion/metadata_extractor.py` recognises the author layouts used by
the major publishers. Each strategy is tried in turn and the first one
that yields a confident list wins:

| Strategy | Layout | Example |
| --- | --- | --- |
| 2 | Name on its own line, affiliation below | `Ricky Diaz` / `Faculty of Engineering, Udayana University` |
| 2b | Bare run of name lines | `Brian Singer` / `Lakshmi Adiga` |
| 2c | Comma-separated list on one line | `Tri Cao1 *†, Chengyu Huang1*, …` |
| 2d | `NAME, Affiliation` on one line (ALL CAPS) | `ZEHANG DENG, Department of Computer Science, …` |
| 2e | Name + affiliation + ORCID iD stack | `Lokanatha Reddy` / `Verizon` / `0009-0006-6058-3326` |
| 2f | One name per line + `·` separated names | `Andreas Happe1` / `· Aaron Kaplan2 · Jürgen Cito1` |

A final cleanup pass removes institution names and metric-like names
(`State Management`, `Tool Usage Accuracy`) that would otherwise be
mistaken for people.

### 3. Never leave a cell blank

`datasets` and `limitations` are the two columns most often empty. The
`SolutionAgent` prompt now requires them to be non-empty, and the
`Coordinator` applies a last-resort fallback so the table always shows
something meaningful rather than a blank cell:

- `datasets` → `No named dataset — the evaluation was performed ad hoc by the authors`
- `limitations` → `The paper does not explicitly discuss limitations.`

These notes record an *observation*, they do not invent content
(`AGENTS.md` §24 — hallucination prevention).

---

## API Endpoints

The Flask app exposes both HTML pages and JSON endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Upload page + LLM health indicator |
| POST | `/upload` | Save PDFs, spawn background processing, return `upload_id` |
| GET | `/status/<upload_id>` | JSON status for polling |
| GET | `/results` | Analyzed papers table (sortable, searchable, filterable) |
| GET | `/paper/<paper_id>` | Detail view for a single paper |
| GET, POST | `/ask` | Q&A form + grounded answer with sources |
| GET | `/export/excel` | Download the full table as a single `.xlsx` |
| GET | `/api/papers` | JSON list of all analyzed papers |
| DELETE | `/api/papers/<paper_id>` | Delete a paper (ChromaDB + SQLite + PDF) |
| GET | `/api/uploads` | List in-memory upload sessions |
| DELETE | `/api/uploads/<upload_id>` | Clear a stuck upload session |
| GET | `/api/llm_status` | Quick LLM health probe (latency, model, speed) |
| GET | `/api/export` | Export metadata: `{count, download_url}` |
| POST | `/api/query` | JSON Q&A endpoint |

---

## Project Structure

```text
pdf_etractor_paper_review/
├── app.py                       # Flask entrypoint + all routes
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
│   └── ollama_client.py         # Native `ollama` package wrapper
├── embeddings/
│   └── embedding_service.py     # Ollama + SentenceTransformers + alternatives
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
│   ├── processing_runner.py     # Background threading per upload
│   ├── processing_status.py     # Thread-safe in-memory status registry
│   └── results_store.py         # SQLite + memory cache
├── templates/                   # Jinja + Bootstrap 5
│   ├── base.html
│   ├── index.html               # Upload + status
│   ├── results.html             # Analyzed papers table
│   ├── paper_detail.html
│   ├── question.html
│   └── partials/                # Shared row / status chip fragments
├── static/                      # CSS + JS
│   ├── css/style.css
│   └── js/app.js
├── data/
│   ├── uploads/                 # User PDFs (gitignored)
│   ├── chroma/                  # Persistent vector DB (gitignored)
│   └── app.sqlite               # Analyzed papers (gitignored)
├── tests/                       # pytest suite (mocked LLMs, no Ollama needed)
├── AGENTS.md                    # Spec
├── PLAN.md                      # Execution roadmap
├── PHASE_LOG.md                 # Build history
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md                    # This file
```

---

## Testing

```powershell
# Run the full pytest suite
python -m pytest --tb=short tests/

# Run just one phase
python -m pytest tests/test_phase3_agents.py

# Run a specific test
python -m pytest tests/test_phase4_coordinator.py::TestCoordinator::test_parallel_execution_speedup
```

All tests use mocked LLMs (no real Ollama needed). For per-phase acceptance scripts (more verbose end-to-end checks), see `tests/_acceptance_phaseN.py`.

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
| 7 | pytest Suite (CI-friendly) | ✅ |
| 8 | Excel export + UI polish (results table, navbar, footer) | ✅ |

See [PHASE_LOG.md](PHASE_LOG.md) for per-phase details, bug fixes, and architectural decisions.

---

## UI Workflow

1. **Upload page** (`/`) — pick PDFs, watch per-paper status chips, see LLM health indicator
2. **Results table** (`/results`) — sortable, searchable, filterable by cybersecurity area; click a row title for the full paper page; click **⬇ Export to Excel** in the toolbar to download all papers as `.xlsx`
3. **Paper detail** (`/paper/<id>`) — full analysis with citations

The navbar contains **Upload** and **Results**. The Q&A page and Excel export are accessible from their dedicated URLs (or via the toolbar button on the Results page).

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
- ✅ SHA-256 duplicate detection on uploaded PDFs (no redundant embedding work)

---

## Author

Created by **Sk Rashed Bin Mohammad**.

---

## License

Internal project. See spec for details.
