# Analysis — Step-by-Step Map of App Actions to Source Files

This document walks through every user-facing action of the **Cybersecurity Paper Analyzer** and shows which file(s) implement each step. The goal is to give any contributor a "follow-the-code" guide: take any action, read the linked files, and you'll find exactly where the work happens.

For the architecture overview, see [README.md](README.md). For the spec, see [AGENTS.md](AGENTS.md). For the execution plan, see [PLAN.md](PLAN.md).

---

## Table of Contents

1. [App startup & wiring](#1-app-startup--wiring)
2. [User opens the upload page (GET /)](#2-user-opens-the-upload-page-get-)
3. [User uploads one or more PDFs (POST /upload)](#3-user-uploads-one-or-more-pdfs-post-upload)
4. [Background processing per PDF (ingest → analyze → store)](#4-background-processing-per-pdf-ingest--analyze--store)
   - 4.1 [Ingest: read PDF, clean, chunk, embed, store in ChromaDB](#41-ingest-read-pdf-clean-chunk-embed-store-in-chromadb)
   - 4.2 [Analyze: Coordinator runs 4 agents in parallel](#42-analyze-coordinator-runs-4-agents-in-parallel)
   - 4.3 [Persist: write PaperAnalysis to SQLite + memory](#43-persist-write-paperanalysis-to-sqlite--memory)
5. [Status polling (GET /status/&lt;upload_id&gt;)](#5-status-polling-get-statusupload_id)
6. [Results page (GET /results)](#6-results-page-get-results)
7. [Paper detail page (GET /paper/&lt;paper_id&gt;)](#7-paper-detail-page-get-paperpaper_id)
8. [Q&A page (GET/POST /ask)](#8-qa-page-getpost-ask)
9. [Excel export (GET /export/excel)](#9-excel-export-get-exportexcel)
10. [REST API endpoints](#10-rest-api-endpoints)
11. [Delete a paper (DELETE /api/papers/&lt;paper_id&gt;)](#11-delete-a-paper-delete-apipaperspaper_id)
12. [LLM health probe (GET /api/llm_status)](#12-llm-health-probe-get-api_llm_status)
13. [Cross-cutting concerns](#13-cross-cutting-concerns)
14. [Reverse map: Python file → step](#14-reverse-map-python-file--step)
15. [Quick file index](#15-quick-file-index)

---

## 1. App startup & wiring

When you run `python app.py`:

| Step | Where | What happens |
| --- | --- | --- |
| Entry point | [app.py](app.py) (bottom: `if __name__ == "__main__"`) | Calls `create_app()` then `app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)`. `threaded=True` lets the dev server run background PDF processing alongside request handling. |
| App factory | [app.py](app.py) → `create_app()` | Calls `setup_logging()`, `get_settings()`, `ensure_directories()`; configures Flask secret + max upload size; constructs the singletons: `registry` (status), `results` (paper store), `analysis_service` (Coordinator facade), `qa_agent` (Q&A). |
| Logging | [config/logging.py](config/logging.py) | `setup_logging()` + `get_logger(name)`. Strips secrets from log records. |
| Settings | [config/settings.py](config/settings.py) | `get_settings()` reads `.env` via pydantic-settings, returns a typed `Settings` object. `make_llm(role)` builds the right LLM per agent role. |
| Directory bootstrap | [config/settings.py](config/settings.py) → `ensure_directories()` | Creates `data/uploads/`, `data/chroma/`, etc., if missing. |
| Suppress telemetry | [app.py](app.py) → `os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")` | Keeps chromadb quiet. |
| Singletons | [services/processing_status.py](services/processing_status.py) → `get_registry()`; [services/results_store.py](services/results_store.py) → `get_results_store()`; [vectorstore/chroma_store.py](vectorstore/chroma_store.py) → `get_vector_store()` (lazy, in app.py) | Module-level singletons so all routes see the same state. |

---

## 2. User opens the upload page (GET /)

| Step | Where | What happens |
| --- | --- | --- |
| Route handler | [app.py](app.py) → `@app.route("/")` → `def index()` | Renders the upload page. |
| Active-model badge | [app.py](app.py) → `s.model_for_role("default")`, `s.embedding_model` | Reads from settings; shows which LLM and embedding model the server is using. |
| Template | [templates/index.html](templates/index.html) | Bootstrap 5 form with file input, status panel, pipeline sidebar. |
| LLM health check (client-side) | [static/js/app.js](static/js/app.js) | Polls `/api/llm_status` and updates the small ⏳/✅/⚠ indicator. |
| Styles | [static/css/style.css](static/css/style.css) | Custom styles on top of Bootstrap. |

---

## 3. User uploads one or more PDFs (POST /upload)

| Step | Where | What happens |
| --- | --- | --- |
| Route handler | [app.py](app.py) → `@app.route("/upload", methods=["POST"])` → `def upload()` | Receives `request.files.getlist("pdfs")`. |
| Security checks | [app.py](app.py) → `_allowed_file()`, `werkzeug.utils.secure_filename` | Whitelist (`.pdf` only), sanitize filename, collision-suffix on duplicates. |
| Save to disk | [app.py](app.py) → `f.save(dest)` | Writes to `settings.upload_directory`. |
| Create upload session | [services/processing_status.py](services/processing_status.py) → `registry.create_upload(filenames)` | Returns `upload_id`; per-paper `ProcessingStatus` rows initialized to `queued`. |
| Spawn background threads | [services/processing_runner.py](services/processing_runner.py) → `start_background_processing(upload_id, paths_by_paper_id, analysis_service=...)` | Starts one daemon thread per PDF. |
| Return JSON | [app.py](app.py) → `jsonify({upload_id, filenames, rejected, status_url})` | Client polls `status_url`. |

---

## 4. Background processing per PDF (ingest → analyze → store)

This is the heart of the pipeline. For each PDF, a daemon thread runs the flow below. The thread body is in [services/processing_runner.py](services/processing_runner.py) → `_run_for_paper()`.

```
PDF on disk
   ↓
ingest_pdf()              [services/paper_service.py]
   ↓
analyze_paper()           [services/analysis_service.py → agents/coordinator.py]
   ↓
results_store.put()       [services/results_store.py]
   ↓
registry.mark_done()      [services/processing_status.py]
```

### 4.1 Ingest: read PDF, clean, chunk, embed, store in ChromaDB

| Step | Where | What happens |
| --- | --- | --- |
| Thread entry | [services/processing_runner.py](services/processing_runner.py) → `_run_for_paper()` | Updates status to `extracting` (10%), then `chunking` (30%). |
| Concurrency gate | [services/processing_runner.py](services/processing_runner.py) → `_get_paper_semaphore()` | A global semaphore (size = `settings.paper_concurrency`, default 3) limits how many papers are analysed at once. Without it, N PDFs × 4 agents = up to 4N concurrent LLM requests, which triggers `429` rate limits. The slot is released in a `finally` block, so a failure never leaks it. |
| SHA-256 dedupe | [services/paper_service.py](services/paper_service.py) → `ingest_pdf()` | Hashes the file; if already indexed, marks `is_duplicate=True` and returns the existing `paper_id` so the analysis still runs. |
| Open + extract text | [ingestion/pdf_loader.py](ingestion/pdf_loader.py) | PyMuPDF (`fitz`) page-by-page text extraction; per-page metadata (page number, source). |
| Section detection (optional) | [ingestion/text_cleaner.py](ingestion/text_cleaner.py) → `clean_text()` | Strips headers/footers, normalizes whitespace. |
| Metadata extraction | [ingestion/metadata_extractor.py](ingestion/metadata_extractor.py) → `extract_metadata()` | Title, authors, date, DOI, venue, institution. Runs 6 layout-aware author strategies (see §4.1.1). |
| Chunk | [ingestion/chunker.py](ingestion/chunker.py) | Token-aware chunking (`chunk_size`, `chunk_overlap` from settings). Section-aware when the structure is detectable. |
| Front matter chunk | [ingestion/chunker.py](ingestion/chunker.py) → `_chunk_by_sections()` | The text *before* the first section heading (title + author block) is emitted as its own chunk with section `Front Matter`. Without it the author names never reach the vector store, so `MetadataAgent` cannot recover them. |
| Embed | [embeddings/embedding_service.py](embeddings/embedding_service.py) | Calls the configured provider — `ollama` (default, native package), `sentence_transformers`, or `openai`. |
| Store in ChromaDB | [vectorstore/chroma_store.py](vectorstore/chroma_store.py) → `add_chunks()` | Persists vectors + per-chunk metadata (`paper_id`, `filename`, `title`, `authors`, `publication_date`, `page`, `section`, `chunk_id`). |
| Return | [services/paper_service.py](services/paper_service.py) → `IngestionResult(paper_id, sha256, is_duplicate, num_chunks)` | Used by the runner. |

#### 4.1.1 Author extraction strategies

`ingestion/metadata_extractor.py` → `_guess_authors()` tries these in order and returns the first confident list:

| # | Strategy | Function | Layout |
| --- | --- | --- | --- |
| 2 | Name line + affiliation below | inline in `_guess_authors()` | `Ricky Diaz` / `Faculty of Engineering, Udayana University` |
| 2b | Bare run of name lines | `_guess_authors_from_block()` | `Brian Singer` / `Lakshmi Adiga` |
| 2c | Comma-separated line | `_guess_authors_from_csv_line()` | `Tri Cao1 *†, Chengyu Huang1*, …` |
| 2d | `NAME, Affiliation` on one line | `_guess_authors_from_inline_affiliation()` | `ZEHANG DENG, Department of Computer Science, …` |
| 2e | Name + affiliation + ORCID stack | `_guess_authors_from_orcid_block()` | `Lokanatha Reddy` / `Verizon` / `0009-0006-6058-3326` |
| 2f | One name per line + `·` list | `_guess_authors_stacked()` | `Andreas Happe1` / `· Aaron Kaplan2 · Jürgen Cito1` |

Supporting helpers: `_author_region()` (bounds the front matter), `_looks_like_person_name()`, `_looks_like_allcaps_person_name()`, `_looks_like_affiliation()`, `_looks_like_location_line()`, `_looks_like_orcid()`, `_looks_like_metric_name()`, `_strip_author_marks()`, `_split_author_line()`, and the final `_clean_author_list()` pass that removes institution names and duplicates.

### 4.2 Analyze: Coordinator runs 4 agents in parallel

| Step | Where | What happens |
| --- | --- | --- |
| Status update | [services/processing_runner.py](services/processing_runner.py) → `state="analyzing", progress=0.6` | UI chip flips to "Analyzing". |
| Coordinator entry | [agents/coordinator.py](agents/coordinator.py) → `Coordinator.analyze_paper(paper_id, top_k=...)` | Builds an `AgentContext` and fans out to 4 agents. |
| Build context | [agents/coordinator.py](agents/coordinator.py) → `_build_context()` | Calls `Retriever.retrieve(paper_id)` to fetch top-k chunks, plus any pre-extracted PDF metadata. |
| Retrieve chunks | [rag/retriever.py](rag/retriever.py) | Queries ChromaDB filtered by `paper_id`; returns `list[RetrievedChunk]`. |
| Parallel execution | [agents/coordinator.py](agents/coordinator.py) → `ThreadPoolExecutor(max_workers=4)` | Each agent runs on its own thread; each uses its own per-role LLM (`make_llm(role)`). |
| Metadata agent | [agents/metadata_agent.py](agents/metadata_agent.py) | Extracts title, authors, publication date, venue, DOI, institution. |
| Cybersecurity agent | [agents/cybersecurity_agent.py](agents/cybersecurity_agent.py) | Classifies into one or more of the 35 cybersecurity taxonomy labels from [models/paper.py](models/paper.py) → `CYBERSECURITY_TAXONOMY`. |
| Solution agent | [agents/solution_agent.py](agents/solution_agent.py) | Identifies problem statement, proposed solution, methodology, datasets, key results, limitations. |
| Summary agent | [agents/summary_agent.py](agents/summary_agent.py) | Produces a 100–250 word technical summary plus per-aspect breakdown. |
| Shared base | [agents/base.py](agents/base.py) → `BaseAgent.run()` | Prompt assembly, JSON parse + retry/repair, prompt-injection shield (treats retrieved chunks as untrusted data). |
| Prompts | [rag/prompts.py](rag/prompts.py) | All agent prompts centralized here. |
| Merge | [agents/coordinator.py](agents/coordinator.py) → `_merge()` | Combines 4 agent outputs into one `PaperAnalysis`; aggregates `SourceReference`s; sets `analysis_status`. |
| Never-blank columns | [agents/coordinator.py](agents/coordinator.py) → `_datasets_or_note()` / `_limitations_or_note()` | If the solution agent returns an empty `datasets` list or a null `limitations`, a descriptive note is substituted so no table cell is blank. The notes record an observation — they never invent content. |
| Failure handling | [agents/coordinator.py](agents/coordinator.py) | Per-agent exceptions become safe defaults; if everything fails, `analysis_status="failed"` and `analysis_error` is set. The coordinator **never** raises. |

### 4.3 Persist: write PaperAnalysis to SQLite + memory

| Step | Where | What happens |
| --- | --- | --- |
| Save result | [services/processing_runner.py](services/processing_runner.py) → `store.put(real_paper_id, analysis)` | Persists the merged `PaperAnalysis`. |
| Storage layer | [services/results_store.py](services/results_store.py) → `ResultsStore.put()` | Writes to `data/app.sqlite` (via `sqlite3`) **and** updates the in-memory cache. |
| Schema/model | [models/paper.py](models/paper.py) → `PaperAnalysis` (Pydantic) | Validated structured output: title, authors, summary, cybersecurity_area, problem/solution/methodology/datasets/results/limitations, sources, status, timestamps. |
| Status update | [services/processing_runner.py](services/processing_runner.py) → `registry.mark_done(upload_id, paper_id)` | UI chip flips to "Done". |
| Exception isolation | [services/processing_runner.py](services/processing_runner.py) | Any uncaught exception → status flips to `failed` with the message; other papers in the batch keep processing. |

---

## 5. Status polling (GET /status/&lt;upload_id&gt;)

| Step | Where | What happens |
| --- | --- | --- |
| Route handler | [app.py](app.py) → `@app.route("/status/<upload_id>")` → `def status(upload_id)` | Looks up the upload session, returns JSON. |
| Registry lookup | [services/processing_status.py](services/processing_status.py) → `registry.get_upload(upload_id)` | Returns `dict[paper_id, ProcessingStatus]`. |
| Serialize | [app.py](app.py) → `s.to_dict()` per status | JSON-friendly state, progress, message. |
| Client polling | [static/js/app.js](static/js/app.js) | `setInterval` polls every ~2s while there are `pending`/`processing` chips; stops when all are `done` or `failed`. |

---

## 6. Results page (GET /results)

| Step | Where | What happens |
| --- | --- | --- |
| Route handler | [app.py](app.py) → `@app.route("/results")` → `def results_page()` | Renders the analyzed-papers table; passes `results_url`, `question_url`, `export_url`. |
| Template | [templates/results.html](templates/results.html) | Bootstrap 5 table with search, area filter, sort, pagination, delete buttons, **⬇ Export to Excel** button. |
| Row partial | [templates/partials/paper_row.html](templates/partials/paper_row.html) | One `<tr>` per paper, with `data-*` attributes for client-side filtering. |
| Initial fetch | [static/js/app.js](static/js/app.js) | `GET /api/papers` → renders rows. |
| Search/filter | [static/js/app.js](static/js/app.js) | Filters by `data-title`, `data-summary`, `data-area` against the search box + dropdown. |
| Sort | [static/js/app.js](static/js/app.js) + [static/css/style.css](static/css/style.css) (`▲/▼` indicators) | Click a column header to sort. |
| Pagination | [static/js/app.js](static/js/app.js) | Client-side pagination; `«/»` buttons. |
| Delete | [static/js/app.js](static/js/app.js) | Opens a Bootstrap modal (`templates/results.html` inline) → `DELETE /api/papers/<paper_id>`. |
| Styles | [static/css/style.css](static/css/style.css) | Per-column widths, wrap-friendly cells for Summary/Solution/Cyber Area. |

---

## 7. Paper detail page (GET /paper/&lt;paper_id&gt;)

| Step | Where | What happens |
| --- | --- | --- |
| Route handler | [app.py](app.py) → `@app.route("/paper/<paper_id>")` → `def paper_detail(paper_id)` | Looks up the `PaperAnalysis` from `results_store`. |
| Lookup | [services/results_store.py](services/results_store.py) → `results.get(paper_id)` | Memory-first; falls back to SQLite. |
| Template | [templates/paper_detail.html](templates/paper_detail.html) | Renders full analysis: title, authors, summary, problem/solution/methodology/datasets/results/limitations, source citations. |
| 404 path | [app.py](app.py) | If `analysis is None` (still analyzing or unknown), renders the template with `analysis=None` and returns 404 so the UI can show "Not found / still processing". |

---

## 8. Q&A page (GET/POST /ask)

| Step | Where | What happens |
| --- | --- | --- |
| Route handler | [app.py](app.py) → `@app.route("/ask", methods=["GET", "POST"])` → `def question()` | GET renders the empty form; POST runs the question. |
| Template | [templates/question.html](templates/question.html) | Bootstrap form (input + button), then answer panel with sources. |
| QAAgent invocation | [app.py](app.py) → `qa_agent.ask(question_text)` (where `qa_agent` is built in `create_app`) | High-level Q&A entry. |
| QAAgent | [agents/qa_agent.py](agents/qa_agent.py) | Wraps the RAG chain; returns `QAResult(question, answer, sources, refused)`. |
| RAG chain | [rag/chain.py](rag/chain.py) | 1) Retrieve chunks. 2) Detect comparative vs. single-paper question. 3) Build prompt with prompt-injection guard. 4) Call LLM. 5) Parse + validate answer. |
| Retrieval | [rag/retriever.py](rag/retriever.py) | Queries ChromaDB with optional paper/area filter; returns top-k chunks with scores. |
| Prompts | [rag/prompts.py](rag/prompts.py) | Single-paper prompt + comparative prompt + system-level "PDF text is untrusted" guard. |
| Refusal handling | [agents/qa_agent.py](agents/qa_agent.py) | If evidence is insufficient, `QAResult.refused=True`; the UI surfaces that as "The uploaded papers do not contain sufficient information to answer this question." |
| Error humanization | [app.py](app.py) → `_humanize_ask_error()` | Converts raw exceptions (Ollama down, model missing, embedding dim mismatch, JSON parse error, etc.) into user-friendly messages; never exposes stack traces. |
| Display | [templates/question.html](templates/question.html) | Renders the answer + per-source citation (paper title — page). |

---

## 9. Excel export (GET /export/excel)

| Step | Where | What happens |
| --- | --- | --- |
| Route handler | [app.py](app.py) → `@app.route("/export/excel")` → `def export_excel()` | Builds an `.xlsx` in memory and streams it as an attachment. |
| Data source | [services/results_store.py](services/results_store.py) → `results.all()` | All `PaperAnalysis` records. |
| Workbook build | [app.py](app.py) (`openpyxl.Workbook`) | One sheet `Analyzed Papers`. |
| Columns | [app.py](app.py) (16 columns) | Paper ID, Title, Authors, Published, Filename, Cybersecurity Area, Summary, Problem Statement, Proposed Solution, Methodology, Datasets, Key Results, Limitations, Analysis Status, Analysis Error, Created At. |
| Formatting | [app.py](app.py) (openpyxl styles) | Frozen header (`freeze_panes = "A2"`), wrapped text (`wrap_text=True, vertical="top"`), sized column widths, dark header fill + white bold font, timestamped filename `analyzed_papers_YYYYMMDD_HHMMSS.xlsx`. |
| Stream | [app.py](app.py) → `send_file(buf, as_attachment=True, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")` | Browser downloads the file. |
| REST metadata | [app.py](app.py) → `@app.route("/api/export")` → `api_export()` | Returns `{count, download_url}` so clients can check before downloading. |
| Toolbar button | [templates/results.html](templates/results.html) | Green **⬇ Export to Excel** button next to Refresh. |
| Dependency | [requirements.txt](requirements.txt) → `openpyxl>=3.1.0,<4.0.0` | Installed in the venv. |

---

## 10. REST API endpoints

| Method + path | File | Purpose |
| --- | --- | --- |
| `GET /api/papers` | [app.py](app.py) | JSON list of all `PaperAnalysis` records (`results.all()`). |
| `GET /api/uploads` | [app.py](app.py) | List all in-memory upload sessions (incl. stuck ones). |
| `DELETE /api/uploads/<upload_id>` | [app.py](app.py) | Forget an in-memory session (does not delete the paper or its chunks). |
| `GET /api/llm_status` | [app.py](app.py) | LLM health probe — see [§12](#12-llm-health-probe-get-api_llm_status). |
| `GET /api/export` | [app.py](app.py) | `{count, download_url}` metadata for the Excel export. |
| `POST /api/query` | [app.py](app.py) | JSON Q&A; same `qa_agent.ask()` as `/ask`. |
| `DELETE /api/papers/<paper_id>` | [app.py](app.py) | See [§11](#11-delete-a-paper-delete-apipaperspaper_id). |
| Error handlers | [app.py](app.py) → `@app.errorhandler(404/413)` | JSON for `/api/*`, HTML for everything else. |

---

## 11. Delete a paper (DELETE /api/papers/&lt;paper_id&gt;)

Cascade cleanup. Lives entirely in [app.py](app.py) (`api_delete_paper`).

| Step | Where | What happens |
| --- | --- | --- |
| Lookup | [services/results_store.py](services/results_store.py) → `results.get(paper_id)` | 404 if unknown. |
| Delete chunks | [vectorstore/chroma_store.py](vectorstore/chroma_store.py) → `store.delete_paper(paper_id)` | Removes all vector chunks where `metadata.paper_id == paper_id`. |
| Delete row | [services/results_store.py](services/results_store.py) → `results.delete(paper_id)` | Removes the SQLite row + memory cache. |
| Delete PDF | [app.py](app.py) | `settings.upload_directory.glob(f"*{paper_id}*")` → `unlink(missing_ok=True)`. Best-effort. |
| Scrub status | [app.py](app.py) | Removes any `ProcessingStatus` rows that reference this `paper_id` across all upload sessions. |
| Audit log | [app.py](app.py) → `logger.info("Deleted paper %s: %s", ...)` | One INFO line summarising what was removed. |

---

## 12. LLM health probe (GET /api/llm_status)

| Step | Where | What happens |
| --- | --- | --- |
| Route handler | [app.py](app.py) → `api_llm_status()` | Sends a trivial "Reply with OK" prompt. |
| LLM invocation | [app.py](app.py) → `default_llm_factory("default").invoke(...)` | Honors any test/mock injected via `create_app(llm_factory=...)`. |
| Speed buckets | [app.py](app.py) | `fast` < 20s, `ok` 20–60s, `slow` > 60s. |
| UI use | [static/js/app.js](static/js/app.js) | Updates the ⏳/✅/⚠ badge in [templates/index.html](templates/index.html); shows the slow-LLM warning banner. |
| Failure surfaces | [app.py](app.py) | Returns `{ok: false, error, type}` — used by the UI to warn before the user uploads. |

---

## 13. Cross-cutting concerns

These run alongside every flow and are worth knowing about before changing anything.

| Concern | Files | Notes |
| --- | --- | --- |
| **Settings** | [config/settings.py](config/settings.py), [.env.example](.env.example), [.gitignore](.gitignore) | All env-driven; never hardcode model names or paths. |
| **Logging** | [config/logging.py](config/logging.py) | Structured; scrubs secrets. Called via `from config.logging import get_logger`. |
| **Models** | [models/paper.py](models/paper.py) | Pydantic contracts: `PaperAnalysis`, `RetrievedChunk`, `SourceReference`, `AgentContext`, plus 4 per-agent result models and the 35-label `CYBERSECURITY_TAXONOMY`. |
| **LLM client** | [llm/ollama_client.py](llm/ollama_client.py) | Thin wrapper around the native `ollama` package. |
| **Embeddings** | [embeddings/embedding_service.py](embeddings/embedding_service.py) | Provider-agnostic; switches on `EMBEDDING_PROVIDER`. |
| **Vector store** | [vectorstore/chroma_store.py](vectorstore/chroma_store.py) | Singleton, persistent at `CHROMA_PERSIST_DIRECTORY`. |
| **Prompts** | [rag/prompts.py](rag/prompts.py) | One file for all system + agent prompts; easy to audit. |
| **Prompt-injection shield** | [agents/base.py](agents/base.py) (`BaseAgent`), [rag/prompts.py](rag/prompts.py) | System prompt tells every agent that retrieved PDF text is untrusted data; the agent's system message establishes the trust boundary. |
| **Tests** | [tests/](tests/) | pytest suite with mocked LLMs — no real Ollama needed. |
| **Error pages** | [app.py](app.py) → `@app.errorhandler(404/413)` | JSON for `/api/*`, HTML otherwise. |

---

## 14. Reverse map: Python file → step

The previous sections go **action → file**. This section is the inverse: **file → action**. Open any Python file, find it in the table below, and you'll see exactly which user-facing steps it implements and which functions to read.

### 14.1 `app.py` (Flask entrypoint + all routes)

| Function / route | Step | What it does |
| --- | --- | --- |
| `create_app(llm_factory=None)` | [§1](#1-app-startup--wiring) | App factory: logging, settings, singletons, blueprints, error handlers. |
| `index()` → `GET /` | [§2](#2-user-opens-the-upload-page-get-) | Renders the upload page; passes active models. |
| `upload()` → `POST /upload` | [§3](#3-user-uploads-one-or-more-pdfs-post-upload) | Validates, sanitises, saves PDFs, spawns background threads. |
| `status(upload_id)` → `GET /status/<upload_id>` | [§5](#5-status-polling-get-statusupload_id) | Returns per-paper status JSON for polling. |
| `api_list_uploads()` → `GET /api/uploads` | [§10](#10-rest-api-endpoints) | Lists in-memory upload sessions. |
| `api_clear_upload(upload_id)` → `DELETE /api/uploads/<upload_id>` | [§10](#10-rest-api-endpoints) | Forgets a stuck session (does not delete paper). |
| `api_llm_status()` → `GET /api/llm_status` | [§12](#12-llm-health-probe-get-api_llm_status) | Trivial LLM health probe. |
| `results_page()` → `GET /results` | [§6](#6-results-page-get-results) | Renders the analysed-papers table. |
| `api_papers()` → `GET /api/papers` | [§10](#10-rest-api-endpoints) | JSON list of all `PaperAnalysis`. |
| `api_delete_paper(paper_id)` → `DELETE /api/papers/<paper_id>` | [§11](#11-delete-a-paper-delete-apipaperspaper_id) | Cascade delete: ChromaDB + SQLite + PDF + status. |
| `export_excel()` → `GET /export/excel` | [§9](#9-excel-export-get-exportexcel) | Builds and streams the `.xlsx` workbook. |
| `api_export()` → `GET /api/export` | [§9](#9-excel-export-get-exportexcel) | `{count, download_url}` metadata. |
| `paper_detail(paper_id)` → `GET /paper/<paper_id>` | [§7](#7-paper-detail-page-get-paperpaper_id) | Renders the full paper page. |
| `question()` → `GET/POST /ask` | [§8](#8-qa-page-getpost-ask) | Q&A form + handler. |
| `api_query()` → `POST /api/query` | [§10](#10-rest-api-endpoints) | JSON Q&A endpoint. |
| `_humanize_ask_error(exc)` | [§8](#8-qa-page-getpost-ask) | Maps raw exceptions to user-friendly strings. |
| `get_vector_store()` | [§1](#1-app-startup--wiring), [§11](#11-delete-a-paper-delete-apipaperspaper_id) | Lazy singleton for `ChromaVectorStore`. |
| `errorhandler(404/413)` | [§10](#10-rest-api-endpoints) | JSON for `/api/*`, HTML otherwise. |

### 14.2 `config/settings.py`

| Symbol | Step | What it does |
| --- | --- | --- |
| `Settings` (pydantic-settings) | [§1](#1-app-startup--wiring) | Reads `.env` into a typed object. **OS env vars take priority over `.env`.** |
| `get_settings()` (lru_cache) | [§1](#1-app-startup--wiring) | Process-wide settings singleton. |
| `make_llm(role)` | [§1](#1-app-startup--wiring), [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Builds the per-role LLM (Ollama or OpenAI-compatible). Passes `max_retries` / `backoff_base` from settings. |
| `model_for_role(role)` | [§2](#2-user-opens-the-upload-page-get-) | Per-agent model override (e.g. `LLM_MODEL_SUMMARIZATION`). |
| `ensure_directories()` | [§1](#1-app-startup--wiring) | Creates `data/uploads/`, `data/chroma/`, etc. |
| `paper_concurrency` | [§4.1](#41-ingest-read-pdf-clean-chunk-embed-store-in-chromadb) | Max papers analysed in parallel (default 3). |
| `llm_num_predict` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Output token cap per agent call (default 4096). |
| `llm_max_retries` / `llm_retry_backoff` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Retry count and exponential-backoff base for transient LLM errors. |
| `max_upload_mb` | [§3](#3-user-uploads-one-or-more-pdfs-post-upload) | Per-request upload limit in MB (1–2048). |

### 14.3 `config/logging.py`

| Symbol | Step | What it does |
| --- | --- | --- |
| `setup_logging()` | [§1](#1-app-startup--wiring) | Configures structured logging. |
| `get_logger(name)` | All steps | Returns a logger with secret scrubbing. |

### 14.4 `agents/base.py`

| Symbol | Step | What it does |
| --- | --- | --- |
| `BaseAgent.run(ctx)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel), [§8](#8-qa-page-getpost-ask) | Assembles prompt, calls LLM, parses JSON, retries/repairs on failure. Enforces the prompt-injection shield. |
| `BaseAgent._coerce_aliases()` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Accepts common LLM field aliases (`excerpt`, `quote` → `text`). |
| `BaseAgent._looks_truncated(text)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Detects a response cut off mid-JSON (usually the output token cap) so the agent can report a clear error instead of a parse failure. |

### 14.5 `agents/coordinator.py`

| Symbol | Step | What it does |
| --- | --- | --- |
| `Coordinator.analyze_paper(paper_id, top_k=None)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Main entry: builds context, fans out 4 agents, merges results. |
| `Coordinator._build_context(paper_id)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Calls `Retriever` to fetch top-k chunks; assembles `AgentContext`. |
| `Coordinator._run_one(agent, ctx)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Wraps one agent call with exception capture (returns safe default). |
| `Coordinator._datasets_or_note()` / `_limitations_or_note()` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Substitute a descriptive note when `datasets` / `limitations` come back empty, so no table cell is blank. |
| `Coordinator._merge(per_agent_results)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Combines outputs into a single `PaperAnalysis`; aggregates sources. |
| `DEFAULT_MAX_WORKERS = 4` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | One worker per analysis agent. |

### 14.6 Per-agent files

Each agent inherits from `BaseAgent` and overrides `_system_prompt` + `_user_prompt` + `_output_model`.

| File | Step | What it does |
| --- | --- | --- |
| `agents/metadata_agent.py` → `MetadataAgent` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Extracts title, authors, publication date, venue, DOI, institution. |
| `agents/cybersecurity_agent.py` → `CybersecurityAgent` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Classifies into 1+ of 35 cybersecurity taxonomy labels. |
| `agents/solution_agent.py` → `SolutionAgent` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Identifies problem, solution, methodology, datasets, key results, limitations. |
| `agents/summary_agent.py` → `SummaryAgent` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Produces 100–250 word technical summary + per-aspect breakdown. |
| `agents/qa_agent.py` → `QAAgent.ask(question)` | [§8](#8-qa-page-getpost-ask) | Wraps the RAG chain; returns `QAResult(answer, sources, refused)`. |

### 14.7 `ingestion/`

| File | Symbol | Step | What it does |
| --- | --- | --- | --- |
| `pdf_loader.py` | `extract_pages(path)` | [§4.1](#41-ingest-read-pdf-clean-chunk-embed-store-in-chromadb) | PyMuPDF page-by-page text extraction. |
| `text_cleaner.py` | `clean_text(text)` | [§4.1](#41-ingest-read-pdf-clean-chunk-embed-store-in-chromadb) | Strips headers/footers, normalises whitespace. |
| `chunker.py` | `chunk_pages(pages, chunk_size, chunk_overlap)` | [§4.1](#41-ingest-read-pdf-clean-chunk-embed-store-in-chromadb) | Token-aware, section-aware chunking. |
| `chunker.py` | `FRONT_MATTER_SECTION` / `_chunk_by_sections()` | [§4.1](#41-ingest-read-pdf-clean-chunk-embed-store-in-chromadb) | Emits the title + author block as its own chunk so it reaches the vector store and the `MetadataAgent`. |
| `metadata_extractor.py` | `extract_metadata(pdf_meta, first_page_text, filename)` | [§4.1](#41-ingest-read-pdf-clean-chunk-embed-store-in-chromadb) | Title, authors, date, DOI, venue, institution from the PDF info dict + first-page heuristics. |
| `metadata_extractor.py` | `_guess_authors()` + 6 strategy helpers | [§4.1.1](#411-author-extraction-strategies) | Layout-aware author extraction (see §4.1.1). |
| `metadata_extractor.py` | `_coerce_year()` | [§4.1](#41-ingest-read-pdf-clean-chunk-embed-store-in-chromadb) | Parses the PDF `D:YYYYMMDD…` date format and free-text years. |
| `ocr_loader.py` | `ocr_pages(path)` | [§4.1](#41-ingest-read-pdf-clean-chunk-embed-store-in-chromadb) | Tesseract fallback (opt-in via `OCR_ENABLED`). |

### 14.8 `llm/ollama_client.py`

| Symbol | Step | What it does |
| --- | --- | --- |
| `OllamaChatModel.invoke(messages)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel), [§8](#8-qa-page-getpost-ask), [§12](#12-llm-health-probe-get-api_llm_status) | Wraps the native `ollama` SDK chat call with retries + timeout. |
| `OllamaChatModel._invoke_once(messages)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | The single underlying chat call. |
| `OllamaChatModel._sleep_before_retry(attempt, exc)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Exponential backoff (`llm_retry_backoff ** attempt`, capped at 60 s, with jitter). |
| `_is_retryable(exc)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Classifies `429` / `5xx` / timeouts / connection resets as transient; everything else fails fast. |
| `OllamaChatModel.model` | [§2](#2-user-opens-the-upload-page-get-), [§12](#12-llm-health-probe-get-api_llm_status) | Active model name (for display in UI). |

### 14.9 `embeddings/embedding_service.py`

| Symbol | Step | What it does |
| --- | --- | --- |
| `EmbeddingService.embed_documents(texts)` | [§4.1](#41-ingest-read-pdf-clean-chunk-embed-store-in-chromadb) | Batch-embed chunks. |
| `EmbeddingService.embed_query(query)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel), [§8](#8-qa-page-getpost-ask) | Embed a single query. |
| `get_embedding_service()` | [§1](#1-app-startup--wiring) | Lazy singleton; switches on `EMBEDDING_PROVIDER`. |

### 14.10 `vectorstore/chroma_store.py`

| Symbol | Step | What it does |
| --- | --- | --- |
| `ChromaVectorStore.add_chunks(paper_id, chunks, embeddings, metadata)` | [§4.1](#41-ingest-read-pdf-clean-chunk-embed-store-in-chromadb) | Persist vectors + metadata. |
| `ChromaVectorStore.query(embedding, top_k, where=None)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel), [§8](#8-qa-page-getpost-ask) | Similarity search with optional metadata filter. |
| `ChromaVectorStore.delete_paper(paper_id)` | [§11](#11-delete-a-paper-delete-apipaperspaper_id) | Remove all chunks where `paper_id == X`. |

### 14.11 `rag/`

| File | Symbol | Step | What it does |
| --- | --- | --- | --- |
| `retriever.py` | `Retriever.retrieve(paper_id, top_k)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Get top-k chunks for one paper. |
| `retriever.py` | `Retriever.retrieve_for_query(query, top_k, filter)` | [§8](#8-qa-page-getpost-ask) | Get top-k chunks for a question (optionally filtered). |
| `chain.py` | `RAGChain.ask(question)` | [§8](#8-qa-page-getpost-ask) | Detect comparative vs single-paper, build prompt, call LLM, parse answer. |
| `chain.py` | `RAGChain._is_comparative(question)` | [§8](#8-qa-page-getpost-ask) | Heuristic: routes to comparative prompt. |
| `prompts.py` | `SYSTEM_PROMPT`, `METADATA_PROMPT`, `CYBERSECURITY_PROMPT`, `SOLUTION_PROMPT`, `SUMMARY_PROMPT`, `QA_PROMPT`, `COMPARATIVE_QA_PROMPT` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel), [§8](#8-qa-page-getpost-ask) | All prompts centralised. Includes prompt-injection shield. |
| `bm25.py` | `BM25Index` | [§8](#8-qa-page-getpost-ask) (optional) | Lexical search for hybrid retrieval. |
| `hybrid.py` | `rrf_merge(ranked_lists, k=...)` | [§8](#8-qa-page-getpost-ask) (optional) | Reciprocal-rank-fusion merge. |
| `reranker.py` | `CrossEncoderReranker.rerank(query, chunks, top_k)` | [§8](#8-qa-page-getpost-ask) (optional) | Re-score top-k with a cross-encoder. |

### 14.12 `models/paper.py`

| Symbol | Step | What it does |
| --- | --- | --- |
| `PaperAnalysis` | [§4.3](#43-persist-write-paperanalysis-to-sqlite--memory), [§6](#6-results-page-get-results), [§9](#9-excel-export-get-exportexcel) | The final, validated analysis object. |
| `MetadataResult`, `CybersecurityResult`, `SolutionResult`, `SummaryResult` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Per-agent output models. |
| `QAResult` | [§8](#8-qa-page-getpost-ask) | Q&A answer + sources + `refused` flag. |
| `RetrievedChunk`, `SourceReference`, `AgentContext` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | RAG ↔ agent contract. |
| `CYBERSECURITY_TAXONOMY` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | 35 classification labels. |

### 14.13 `services/`

| File | Symbol | Step | What it does |
| --- | --- | --- | --- |
| `paper_service.py` | `ingest_pdf(path, vector_store=None)` | [§4.1](#41-ingest-read-pdf-clean-chunk-embed-store-in-chromadb) | End-to-end ingest: SHA-256 dedupe → extract → clean → chunk → embed → store. Returns `IngestionResult(paper_id, sha256, is_duplicate, num_chunks)`. |
| `paper_service.py` | `IngestionResult` | [§4.1](#41-ingest-read-pdf-clean-chunk-embed-store-in-chromadb) | Typed result of `ingest_pdf`. |
| `analysis_service.py` | `AnalysisService.analyze_paper(paper_id, top_k=None)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Thin facade over `Coordinator.analyze_paper`. |
| `analysis_service.py` | `AnalysisService.compare_papers(question, top_k=None)` | [§8](#8-qa-page-getpost-ask) | Multi-paper comparative Q&A (delegates to `QAAgent`). |
| `analysis_service.py` | `AnalysisService.analyze_all_papers(paper_ids)` | [§4.2](#42-analyze-coordinator-runs-4-agents-in-parallel) | Bulk analyse (failures are isolated per paper). |
| `processing_runner.py` | `start_background_processing(upload_id, paths_by_paper_id, analysis_service=...)` | [§3](#3-user-uploads-one-or-more-pdfs-post-upload) | Spawns one daemon thread per PDF. |
| `processing_runner.py` | `_run_for_paper(upload_id, paper_id, path, ...)` | [§4](#4-background-processing-per-pdf-ingest--analyze--store) | Per-thread entry: ingest → analyze → store, with status updates and exception isolation. |
| `processing_runner.py` | `_get_paper_semaphore()` | [§4.1](#41-ingest-read-pdf-clean-chunk-embed-store-in-chromadb) | Lazily-built global semaphore sized by `settings.paper_concurrency`; caps concurrent paper analyses. |
| `processing_runner.py` | `_safe_name(path)` | [§4](#4-background-processing-per-pdf-ingest--analyze--store) | Tolerant filename helper (accepts `str` or `Path`) used in log calls. |
| `processing_status.py` | `get_registry()` | [§1](#1-app-startup--wiring), [§4](#4-background-processing-per-pdf-ingest--analyze--store) | Process-wide `ProcessingStatusRegistry` singleton. |
| `processing_status.py` | `ProcessingStatusRegistry.create_upload(filenames)` | [§3](#3-user-uploads-one-or-more-pdfs-post-upload) | Allocate a new `upload_id` and per-paper status rows. |
| `processing_status.py` | `ProcessingStatusRegistry.update(upload_id, paper_id, state, progress, message)` | [§4](#4-background-processing-per-pdf-ingest--analyze--store) | Update a single paper's chip. |
| `processing_status.py` | `ProcessingStatusRegistry.mark_done(upload_id, paper_id)` | [§4.3](#43-persist-write-paperanalysis-to-sqlite--memory) | Flip the chip to `done`. |
| `processing_status.py` | `ProcessingStatusRegistry.get_upload(upload_id)` | [§5](#5-status-polling-get-statusupload_id) | Read current state for polling. |
| `results_store.py` | `get_results_store()` | [§1](#1-app-startup--wiring), [§4.3](#43-persist-write-paperanalysis-to-sqlite--memory) | Process-wide `ResultsStore` singleton. |
| `results_store.py` | `ResultsStore.put(paper_id, analysis)` | [§4.3](#43-persist-write-paperanalysis-to-sqlite--memory) | Persist `PaperAnalysis` to SQLite + memory cache. |
| `results_store.py` | `ResultsStore.get(paper_id)` | [§7](#7-paper-detail-page-get-paperpaper_id), [§11](#11-delete-a-paper-delete-apipaperspaper_id) | Memory-first, SQLite-fallback lookup. |
| `results_store.py` | `ResultsStore.all()` | [§6](#6-results-page-get-results), [§9](#9-excel-export-get-exportexcel) | All papers (for the table + Excel export). |
| `results_store.py` | `ResultsStore.delete(paper_id)` | [§11](#11-delete-a-paper-delete-apipaperspaper_id) | Remove row from SQLite + evict from memory cache. |

### 14.14 "What runs when" cheat sheet

A quick lookup if you only know the function name or the symptom:

| If you see… | Look in… |
| --- | --- |
| `werkzeug.secure_filename`, `_allowed_file` | `app.py` (`upload` route) — file sanitisation. |
| `ingest_pdf` not dedup-ing | `services/paper_service.py` — SHA-256 logic. |
| Chunks missing section info | `ingestion/chunker.py` + `ingestion/text_cleaner.py`. |
| Agent output fails Pydantic validation | `agents/base.py` (`BaseAgent.run`) + the agent's `_output_model` in `models/paper.py`. |
| Coordinator hangs / one agent blocks the rest | `agents/coordinator.py` — `ThreadPoolExecutor` + `_run_one`. |
| Status chip stuck on `analyzing` | `services/processing_runner.py` → `_run_for_paper` (check the LLM/embedder call). |
| Excel export missing columns | `app.py` → `export_excel()` (the `headers` list). |
| `/ask` returns generic error | `app.py` → `_humanize_ask_error` (maps common failure modes). |
| `429 too many concurrent requests` in the log | `config/settings.py` → `paper_concurrency` (lower it). `services/processing_runner.py` → `_get_paper_semaphore()`. |
| Analysis feels slow | Same two places — raise `paper_concurrency` if the LLM is not rate-limiting. |
| `authors` column empty for a paper | `ingestion/metadata_extractor.py` → `_guess_authors()` (see §4.1.1); also check that the `Front Matter` chunk exists in `ingestion/chunker.py`. |
| `datasets` / `limitations` column empty | `agents/solution_agent.py` (prompt rules) + `agents/coordinator.py` → `_datasets_or_note()` / `_limitations_or_note()`. |
| A `.env` change has no effect | OS env vars override `.env`. Clear the stale shell variable and restart the app (see README → Configuration). |
| `RuntimeError: does not match collection dimensionality` | `embeddings/embedding_service.py` — embedding model switched without re-indexing. |
| Prompt-injection attempts in PDF text not blocked | `agents/base.py` + `rag/prompts.py` — re-check the system prompt's untrusted-data clause. |
| Status registry grows without bound | `services/processing_status.py` — `api_clear_upload` removes one; restart clears all. |
| `app.sqlite` getting large | `services/results_store.py` — no built-in vacuum; manual `VACUUM` if needed. |

---

## 15. Quick file index

```
app.py                          # All routes; thin glue
config/
  settings.py                   # Env → typed Settings
  logging.py                    # setup_logging + get_logger
agents/
  base.py                       # BaseAgent (retry/repair, injection shield)
  coordinator.py                # Parallel fan-out for the 4 agents
  metadata_agent.py
  cybersecurity_agent.py
  solution_agent.py
  summary_agent.py
  qa_agent.py                   # Q&A facade over the RAG chain
ingestion/
  pdf_loader.py                 # PyMuPDF text extraction
  text_cleaner.py
  chunker.py                    # Token-aware section-aware chunking
  metadata_extractor.py
  ocr_loader.py                 # Tesseract fallback (opt-in)
llm/
  ollama_client.py              # Native ollama SDK wrapper
embeddings/
  embedding_service.py          # Provider switch: ollama / sentence_transformers / openai
vectorstore/
  chroma_store.py               # ChromaDB wrapper (add, query, delete)
rag/
  retriever.py                  # Top-k retrieval
  chain.py                      # RAG chain + comparative detection
  prompts.py                    # All prompts in one place
  bm25.py                       # Optional hybrid
  hybrid.py                     # RRF merge
  reranker.py                   # Optional cross-encoder
models/
  paper.py                      # Pydantic models + CYBERSECURITY_TAXONOMY
services/
  paper_service.py              # ingest_pdf (PDF → chunks + vectors)
  analysis_service.py           # Coordinator + QAAgent facade
  processing_runner.py          # Background threading per PDF
  processing_status.py          # In-memory status registry
  results_store.py              # SQLite + memory cache
templates/                      # Jinja + Bootstrap 5
  base.html, index.html, results.html, paper_detail.html, question.html
  partials/paper_row.html, partials/status_chip.html
static/
  css/style.css
  js/app.js                     # Polling, sort, filter, delete
data/                           # Runtime data (gitignored)
tests/                          # pytest suite
AGENTS.md                       # Spec
PLAN.md                         # Execution roadmap
PHASE_LOG.md                    # Build history
README.md                       # Project overview
ANALYSIS.md                     # This file
requirements.txt
pytest.ini
.env.example
```
