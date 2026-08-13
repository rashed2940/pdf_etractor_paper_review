"""Phase 0 tests: settings, models, status registry, logging.

These tests verify the foundation modules work correctly. No real LLM
or external services required.
"""
from __future__ import annotations


class TestSettings:
    def test_settings_defaults_load(self):
        """Verify the *field-level* defaults declared in Settings.

        We instantiate Settings with `_env_file=None` so the test is
        independent of any .env the developer happens to have on disk.
        """
        from config.settings import Settings

        s = Settings(_env_file=None)
        assert s.llm_provider == "ollama"
        # Default chat model: qwen2.5-coder:7b-instruct. Verified to
        # respond quickly to the JSON-formatted prompts our agents send
        # (see ollama_test/test_qwen.py for smoke testing).
        assert s.llm_model == "qwen2.5-coder:7b-instruct"
        # Default embeddings: Ollama embeddinggemma (768-dim).
        assert s.embedding_provider == "ollama"
        assert s.embedding_model == "embeddinggemma:latest"
        # Default per-call output cap.
        assert s.llm_num_predict == 512
        # Default timeout: 600s (10 min) — long enough for 3-5K-token
        # prompts on typical CPU/consumer-GPU hardware.
        assert s.llm_timeout_seconds == 600

    def test_settings_per_role_model_fallback(self):
        from config.settings import get_settings

        s = get_settings()
        # No override -> falls back to llm_model
        assert s.model_for_role("summarization") == s.llm_model
        assert s.model_for_role("nonexistent") == s.llm_model

    def test_ensure_directories_idempotent(self, tmp_path, monkeypatch):
        from config.settings import ensure_directories

        # Ensure cached settings use the tmp path
        monkeypatch.setenv("UPLOAD_DIRECTORY", str(tmp_path / "uploads"))
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "app.sqlite"))
        from config import settings as settings_mod
        settings_mod.get_settings.cache_clear()

        # Call twice — should not raise
        ensure_directories()
        ensure_directories()
        assert (tmp_path / "uploads").exists()
        assert (tmp_path / "chroma").exists()
        assert (tmp_path / "app.sqlite").parent.exists()

    def test_make_llm_returns_callable(self):
        from config.settings import make_llm

        llm = make_llm("summarization")
        # LangChain chat models have .invoke method
        assert hasattr(llm, "invoke") or callable(llm)


class TestModels:
    def test_paper_analysis_round_trip(self):
        from models.paper import PaperAnalysis, SourceReference

        p = PaperAnalysis(
            paper_id="p1",
            title="Sample",
            authors=["Alice", "Bob"],
            publication_date="2025",
            summary="A study.",
            cybersecurity_area=["LLM Security"],
            proposed_solution="A framework.",
            datasets=["CICD-2017"],
            sources=[SourceReference(paper_id="p1", page=3, chunk_id="p1_c3")],
            analysis_status="ok",
        )
        assert p.paper_id == "p1"
        assert p.analysis_status == "ok"
        assert len(p.model_dump()) >= 15

    def test_paper_analysis_literal_validation(self):
        from models.paper import PaperAnalysis
        from pydantic import ValidationError

        # Invalid analysis_status should be rejected
        try:
            PaperAnalysis(paper_id="p2", analysis_status="bogus")
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass

    def test_cybersecurity_taxonomy_count(self):
        from models.paper import CYBERSECURITY_TAXONOMY

        assert len(CYBERSECURITY_TAXONOMY) == 35
        assert "LLM Security" in CYBERSECURITY_TAXONOMY
        assert "Intrusion Detection" in CYBERSECURITY_TAXONOMY

    def test_agent_context_with_chunks(self):
        from models.paper import AgentContext, RetrievedChunk

        ctx = AgentContext(
            paper_id="p1",
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="c1", paper_id="p1", filename="x.pdf",
                    page=1, text="hello",
                )
            ],
        )
        assert len(ctx.retrieved_chunks) == 1
        assert ctx.retrieved_chunks[0].text == "hello"


class TestProcessingStatusRegistry:
    def test_create_and_update_upload(self):
        from services.processing_status import (
            ProcessingStatusRegistry,
            get_registry,
        )

        reg = ProcessingStatusRegistry()
        uid = reg.create_upload(["a.pdf", "b.pdf"])
        assert len(reg.get_upload(uid)) == 2

        reg.update(uid, "a.pdf", state="extracting", progress=0.5)
        snap = reg.get_upload(uid)
        assert snap["a.pdf"].state == "extracting"
        assert snap["a.pdf"].progress == 0.5

    def test_mark_done_and_failed(self):
        from services.processing_status import ProcessingStatusRegistry

        reg = ProcessingStatusRegistry()
        uid = reg.create_upload(["a.pdf"])
        reg.mark_done(uid, "a.pdf")
        assert reg.get_upload(uid)["a.pdf"].state == "done"
        reg.mark_failed(uid, "a.pdf", "broken")
        assert reg.get_upload(uid)["a.pdf"].state == "failed"
        assert reg.get_upload(uid)["a.pdf"].error == "broken"

    def test_get_registry_singleton(self):
        from services.processing_status import get_registry

        a = get_registry()
        b = get_registry()
        assert a is b


class TestLogging:
    def test_secret_scrubber(self):
        from config.logging import _SecretScrubber
        import logging

        scrubber = _SecretScrubber()
        rec = logging.LogRecord(
            "name", logging.INFO, "path", 1, "api_key=abc123 token=xyz", (), None
        )
        assert scrubber.filter(rec) is True
        # Scrubber preserves the key name and masks the value
        assert "abc123" not in rec.msg
        assert "xyz" not in rec.msg
        assert "api_key" in rec.msg  # key name preserved

    def test_setup_logging_idempotent(self):
        from config.logging import setup_logging

        setup_logging()
        setup_logging()  # Second call should not raise

    def test_get_logger_returns_logger(self):
        from config.logging import get_logger

        log = get_logger("test")
        assert log.name == "test"