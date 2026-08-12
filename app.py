"""Cybersecurity Paper RAG — Flask web application.

Routes (PLAN.md §3 Phase 5):
- GET  /                       — upload page + status panel
- POST /upload                 — save PDFs, spawn background threads, return upload_id
- GET  /status/<upload_id>     — JSON status for polling
- GET  /results                — render Bootstrap table of all papers
- GET  /paper/<paper_id>       — detail view
- POST /ask                    — Q&A -> JSON answer + sources
- GET  /api/papers             — REST: list all papers
- POST /api/query              — REST: ask a question

Security (AGENTS.md §31):
- secure_filename on upload
- extension whitelist (.pdf only)
- max upload size enforced (settings.max_upload_mb)
- Jinja autoescape on (default)
- All user input rendered through Jinja (no raw HTML)
"""
from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from agents.qa_agent import QAAgent
from config.logging import get_logger, setup_logging
from config.settings import ensure_directories, get_settings, make_llm
from models.paper import QAResult
from services.analysis_service import AnalysisService
from services.processing_runner import start_background_processing
from services.processing_status import get_registry
from services.results_store import get_results_store
from vectorstore.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


# ----- Lazy vector-store singleton ----------------------------------------
# ChromaVectorStore + EmbeddingService are expensive to construct (the
# embedding model has to be loaded/probed). We share one instance across
# all routes so deletes and ingests see the same collection.
_vector_store: ChromaVectorStore | None = None
_vector_store_lock = threading.Lock()


def get_vector_store() -> ChromaVectorStore:
    """Process-wide singleton vector store."""
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                from embeddings.embedding_service import get_embedding_service

                _vector_store = ChromaVectorStore(get_embedding_service())
                logger.info(
                    "Vector store ready: collection=%s",
                    _vector_store._collection.name,
                )
    return _vector_store


def _humanize_ask_error(exc: Exception) -> str:
    """Convert a raw exception from the QA pipeline into a user-friendly message.

    Common failure modes get specific, actionable messages; everything
    else falls back to a generic but honest description. We never expose
    Python stack traces to the user (logged separately via logger.exception).
    """
    msg = str(exc) or type(exc).__name__
    lower = msg.lower()
    # ChromaDB dimension mismatch — happens when the user switches the
    # embedding model after ingesting papers (the legacy collection is
    # auto-migrated, but only for the FIRST switch).
    if "does not match collection dimensionality" in lower or "invaliddimension" in lower:
        return (
            "Embedding model mismatch: the papers in your library were "
            "indexed with a different embedding model than the one "
            "currently selected. Switch `EMBEDDING_MODEL` in `.env` back "
            "to the original, or re-upload your PDFs."
        )
    if "embedding model mismatch" in lower:
        return (
            "Embedding model mismatch detected. Retrieval results may be "
            "unreliable. Either set `EMBEDDING_MODEL` in `.env` to the "
            "model used at upload time, or re-upload the PDFs."
        )
    # Ollama connection issues
    if "connection refused" in lower or "failed to establish" in lower:
        return (
            "Could not reach the Ollama daemon. Make sure `ollama serve` "
            "is running and that the configured model is pulled "
            "(`ollama pull <model>`)."
        )
    if "configured ollama model is not available" in lower or "model '" in lower and "not found" in lower:
        return (
            f"Configured Ollama model is not available: {msg}. "
            "Run `ollama pull <model>` to install it."
        )
    # Ollama timeout — the model hung or took too long
    if "did not respond within" in lower or "timeout" in type_name.lower():
        return (
            "The Ollama model did not respond within the configured "
            "timeout. It may be loading, out of memory, or hung. "
            "Try a smaller model (e.g. llama3.1:8b or gemma3:4b), "
            "free GPU/RAM, or increase LLM_TIMEOUT_SECONDS in .env."
        )
    # No documents ingested yet
    if "no documents" in lower or "empty" in lower and "collection" in lower:
        return (
            "No papers have been ingested yet. Upload a PDF on the home "
            "page and wait for processing to complete, then try again."
        )
    # JSON parse / malformed LLM output
    if "json" in lower and ("parse" in lower or "decode" in lower):
        return (
            "The language model returned a malformed response. Please "
            "rephrase your question or try a different model."
        )
    # Truncate long raw exceptions for the UI
    if len(msg) > 400:
        msg = msg[:400] + "..."
    return f"An error occurred while answering your question: {msg}"


# ----- App factory --------------------------------------------------------
def create_app(*, llm_factory=None) -> Flask:
    """Application factory. Keeps module-level globals out of the way
    and makes the app easy to instantiate in tests.

    Args:
        llm_factory: Optional callable that overrides `make_llm`. Useful
            in tests to inject a mock LLM. Defaults to settings.make_llm.
    """
    setup_logging()
    settings = get_settings()
    ensure_directories()

    # Suppress chromadb telemetry noise
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

    app = Flask(__name__)
    app.config["SECRET_KEY"] = settings.secret_key
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024

    # Singletons — `registry` and `results` are accessed lazily via
    # get_registry()/get_results_store() inside each route so that
    # tests can swap the singleton after create_app() returns.
    registry = get_registry()
    results = get_results_store()  # captured for convenience; routes can override

    # AnalysisService expects a role-aware factory (role: str) -> LLM.
    # QAAgent expects a zero-arg factory. Wrap the user-provided factory
    # so both consumers get what they expect.
    if llm_factory is None:
        analysis_factory = make_llm  # role-aware
        qa_factory = lambda: make_llm("default")  # zero-arg
    else:
        # If the user gave us a zero-arg factory, use it for both; if
        # they gave us a role-aware factory, also use it directly for
        # analysis but wrap for QA.
        analysis_factory = llm_factory
        qa_factory = (lambda f=llm_factory: f("default")) if callable(llm_factory) else llm_factory

    # Store the default-role factory so /api/llm_status (and other
    # ad-hoc LLM probes) can build an LLM the same way the rest of the
    # app does. Tests inject their mock via `llm_factory` and this
    # picks it up automatically.
    default_llm_factory = analysis_factory

    analysis_service = AnalysisService(llm_factory=analysis_factory)
    qa_agent = QAAgent(llm_factory=qa_factory)

    ALLOWED_EXTENSIONS = {".pdf"}

    def _allowed_file(filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in ALLOWED_EXTENSIONS

    # ----------------------------------------------------------------- routes
    @app.route("/")
    def index():
        # Surface the active model identity so users always know which
        # models are running. Helps debug "why is X model loaded in Ollama?"
        # and "did my .env change take effect?" questions.
        s = get_settings()
        active_models = {
            "llm": s.model_for_role("default"),
            "embeddings": s.embedding_model,
            "llm_provider": s.llm_provider,
            "embedding_provider": s.embedding_provider,
        }
        return render_template(
            "index.html",
            upload_action_url=url_for("upload"),
            results_url=url_for("results_page"),
            question_url=url_for("question"),
            max_upload_mb=settings.max_upload_mb,
            active_models=active_models,
        )

    @app.route("/upload", methods=["POST"])
    def upload():
        files = request.files.getlist("pdfs")
        if not files or all(f.filename == "" for f in files):
            return jsonify({"error": "No files uploaded."}), 400

        # Filter + secure_filename
        sanitized: list[tuple[str, Path]] = []
        rejected: list[str] = []
        for f in files:
            if not f or not f.filename:
                continue
            if not _allowed_file(f.filename):
                rejected.append(f.filename)
                continue
            safe_name = secure_filename(f.filename) or f"upload_{uuid.uuid4().hex[:8]}.pdf"
            dest = settings.upload_directory / safe_name
            # If name collides, append a short uuid suffix.
            if dest.exists():
                stem = dest.stem
                dest = settings.upload_directory / f"{stem}_{uuid.uuid4().hex[:6]}.pdf"
            f.save(dest)
            sanitized.append((safe_name, dest))

        if not sanitized:
            return (
                jsonify(
                    {
                        "error": "No valid PDF files uploaded.",
                        "rejected": rejected,
                    }
                ),
                400,
            )

        filenames = [name for name, _ in sanitized]
        upload_id = registry.create_upload(filenames)
        paths_by_paper_id = {name: path for name, path in sanitized}
        start_background_processing(
            upload_id,
            paths_by_paper_id,
            analysis_service=analysis_service,
        )

        logger.info("Upload: upload_id=%s files=%d", upload_id, len(sanitized))
        return jsonify(
            {
                "upload_id": upload_id,
                "filenames": filenames,
                "rejected": rejected,
                "status_url": url_for("status", upload_id=upload_id),
            }
        )

    @app.route("/status/<upload_id>")
    def status(upload_id: str):
        papers = registry.get_upload(upload_id)
        if not papers:
            return jsonify({"error": "upload_id not found"}), 404
        payload: dict[str, Any] = {
            "upload_id": upload_id,
            "papers": {pid: s.to_dict() for pid, s in papers.items()},
        }
        return jsonify(payload)

    @app.route("/api/uploads", methods=["GET"])
    def api_list_uploads():
        """REST: list all in-memory upload sessions (incl. stuck ones)."""
        out: list[dict[str, Any]] = []
        for uid in registry.list_uploads():
            papers = registry.get_upload(uid)
            out.append(
                {
                    "upload_id": uid,
                    "papers": [
                        {"paper_id": p.paper_id, "state": p.state, "filename": p.filename}
                        for p in papers.values()
                    ],
                }
            )
        return jsonify(out)

    @app.route("/api/uploads/<upload_id>", methods=["DELETE"])
    def api_clear_upload(upload_id: str):
        """REST: forget an in-memory upload session.

        Use this when a chip is stuck (e.g. the background thread
        crashed or the LLM hung) and you don't want to restart Flask.
        Does NOT delete the paper or its chunks — only clears the
        status chip.
        """
        papers = registry.get_upload(upload_id)
        if not papers:
            return jsonify({"error": "upload_id not found"}), 404
        # Mutate the registry's internal dict so the snapshot disappears.
        registry._uploads.pop(upload_id, None)  # noqa: SLF001
        logger.info("Cleared upload %s (papers: %d)", upload_id, len(papers))
        return jsonify({"cleared": upload_id, "papers": len(papers)})

    @app.route("/api/llm_status", methods=["GET"])
    def api_llm_status():
        """Quick health check: ping the configured LLM with a trivial prompt.

        Returns `{ok: true, latency_seconds: N}` on success, or
        `{ok: false, error: "..."}` on any failure (timeout, connection
        refused, model not found, etc.). Use this from the UI to warn
        users BEFORE they upload if their LLM is broken.
        """
        import time
        try:
            # Use the same factory as the rest of the app — this honors
            # any test/mock injection via create_app(llm_factory=...).
            llm = default_llm_factory("default")
            model_name = getattr(llm, "model", "?")
            start = time.time()
            # Tiny prompt; Ollama can answer in <1s with a real model.
            llm.invoke(
                [{"role": "user", "content": "Reply with the single word OK."}]
            )
            elapsed = round(time.time() - start, 2)
            return jsonify(
                {"ok": True, "latency_seconds": elapsed, "model": model_name}
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify(
                {"ok": False, "error": str(exc), "type": type(exc).__name__}
            )

    @app.route("/results")
    def results_page():
        return render_template(
            "results.html",
            results_url=url_for("results_page"),
            question_url=url_for("question"),
        )

    @app.route("/api/papers")
    def api_papers():
        """REST: list all analyzed papers."""
        return jsonify([p.model_dump(mode="json") for p in results.all()])

    @app.route("/api/papers/<paper_id>", methods=["DELETE"])
    def api_delete_paper(paper_id: str):
        """REST: delete a paper from the analyzed library.

        Cleans up:
          - ChromaDB vector chunks for the paper (via delete_paper)
          - SQLite row + in-memory cache (ResultsStore.delete)
          - The original PDF file in data/uploads/, if present
          - All ProcessingStatus entries referencing the paper_id

        Returns 200 with a summary on success, 404 if the paper is
        unknown, 500 on unexpected error.
        """
        existing = results.get(paper_id)
        if existing is None:
            return jsonify({"error": f"Unknown paper_id: {paper_id}"}), 404

        summary: dict[str, Any] = {"paper_id": paper_id, "deleted": {}}

        # 1. ChromaDB chunks
        try:
            store = get_vector_store()
            removed = store.delete_paper(paper_id)
            summary["deleted"]["chunks"] = removed
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to delete ChromaDB chunks for %s", paper_id)
            summary["deleted"]["chunks_error"] = str(exc)

        # 2. SQLite + memory
        try:
            results.delete(paper_id)
            summary["deleted"]["analysis"] = True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to delete results row for %s", paper_id)
            summary["deleted"]["analysis_error"] = str(exc)

        # 3. Original PDF (best-effort — failure is non-fatal)
        try:
            for p in settings.upload_directory.glob(f"*{paper_id}*"):
                if p.is_file() and p.suffix.lower() == ".pdf":
                    p.unlink(missing_ok=True)
                    summary["deleted"].setdefault("files", []).append(p.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to remove uploaded PDF for %s: %s", paper_id, exc
            )

        # 4. ProcessingStatus registry entries that mention this paper
        try:
            n = 0
            for upload_id in list(registry.list_uploads()):
                papers = registry.get_upload(upload_id)
                if paper_id in papers:
                    papers.pop(paper_id, None)
                    n += 1
            summary["deleted"]["status_entries"] = n
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to scrub status entries for %s: %s", paper_id, exc
            )

        logger.info("Deleted paper %s: %s", paper_id, summary["deleted"])
        return jsonify(summary)

    @app.route("/paper/<paper_id>")
    def paper_detail(paper_id: str):
        analysis = results.get(paper_id)
        if analysis is None:
            # Maybe still analyzing or not yet started.
            return render_template("paper_detail.html", analysis=None, paper_id=paper_id), 404
        return render_template("paper_detail.html", analysis=analysis, paper_id=paper_id)

    # ------------------------------------------------------------------ Q&A
    @app.route("/ask", methods=["GET", "POST"])
    def question():
        if request.method == "GET":
            return render_template("question.html", qa=None, question_text="")

        question_text = (request.form.get("question") or "").strip()
        if not question_text:
            return render_template(
                "question.html",
                qa=None,
                question_text="",
                error="Please enter a question.",
            ), 400

        try:
            result: QAResult = qa_agent.ask(question_text)
        except Exception as exc:  # noqa: BLE001 — surface real error to the user
            logger.exception("QAAgent.ask failed")
            return render_template(
                "question.html",
                qa=None,
                question_text=question_text,
                error=_humanize_ask_error(exc),
            ), 500
        return render_template("question.html", qa=result, question_text=question_text)

    @app.route("/api/query", methods=["POST"])
    def api_query():
        """REST: ask a question, return JSON."""
        payload = request.get_json(silent=True) or {}
        question_text = (payload.get("question") or "").strip()
        if not question_text:
            return jsonify({"error": "Missing 'question' field."}), 400
        try:
            result: QAResult = qa_agent.ask(question_text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("QAAgent.ask failed")
            return jsonify({"error": _humanize_ask_error(exc)}), 500
        return jsonify(result.model_dump(mode="json"))

    # -------------------------------------------------------- error handlers
    @app.errorhandler(413)
    def _too_large(_exc):
        return (
            jsonify(
                {
                    "error": (
                        f"Upload too large. Max size is "
                        f"{settings.max_upload_mb} MB per request."
                    )
                }
            ),
            413,
        )

    @app.errorhandler(404)
    def _not_found(_exc):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found"}), 404
        return render_template("base.html", content="<h1>404</h1>"), 404

    return app


# ----- Entrypoint ---------------------------------------------------------
app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    logger.info("Starting Flask app on http://127.0.0.1:5000")
    # debug=False to keep output predictable; threaded=True so the
    # background processing threads can report status.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)