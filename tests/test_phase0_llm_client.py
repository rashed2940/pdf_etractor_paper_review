"""Phase 0 (LLM client) tests — settings, model loading, timeouts.

These tests verify the LLM client surface (native ollama package wrapper)
behaves correctly without needing a real Ollama daemon running.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


class TestOllamaClient:
    def test_invoke_returns_content(self):
        """Happy path: a non-hung client returns a _OllamaResponse."""
        from llm.ollama_client import OllamaChatModel, _OllamaResponse

        fake_response = {"message": {"content": "OK"}}
        with patch("ollama.Client") as MockClient:
            instance = MockClient.return_value
            instance.chat.return_value = fake_response

            llm = OllamaChatModel("llama3.1:8b", timeout=5.0)
            result = llm.invoke([{"role": "user", "content": "hi"}])
            assert isinstance(result, _OllamaResponse)
            assert result.content == "OK"

    def test_timeout_raises_timeouterror(self):
        """Hung daemon: the client raises TimeoutError within `timeout`s.

        Regression test for the 'Analyzing 60% forever' bug — without
        this, a hung Ollama would block the background thread
        indefinitely and leave the user staring at a stuck chip.
        """
        from llm.ollama_client import OllamaChatModel

        # Simulate an httpx-style timeout that propagates out of
        # ollama.Client.chat.
        class FakeTimeout(Exception):
            pass

        with patch("ollama.Client") as MockClient:
            instance = MockClient.return_value
            instance.chat.side_effect = FakeTimeout("simulated hang")

            llm = OllamaChatModel("qwen3.5:4b", timeout=1.0)
            start = time.time()
            with pytest.raises(TimeoutError) as exc_info:
                llm.invoke([{"role": "user", "content": "hi"}])
            elapsed = time.time() - start
            # Must raise quickly (well under the 1s timeout).
            assert elapsed < 2.0
            assert "qwen3.5:4b" in str(exc_info.value)

    def test_timeout_is_passed_to_underlying_client(self):
        """The configured timeout must reach the ollama.Client ctor."""
        from llm.ollama_client import OllamaChatModel

        with patch("ollama.Client") as MockClient:
            OllamaChatModel("llama3.1:8b", timeout=42.5)
            kwargs = MockClient.call_args.kwargs
            assert kwargs.get("timeout") == 42.5, (
                f"Expected timeout=42.5 to be passed to ollama.Client, "
                f"got kwargs={kwargs}"
            )

    def test_default_timeout_is_generous(self):
        """Default timeout is 600s — long enough for big models on
        first-call load, short enough that a true hang doesn't block
        forever."""
        from llm.ollama_client import DEFAULT_TIMEOUT_SECONDS, OllamaChatModel

        assert DEFAULT_TIMEOUT_SECONDS >= 60
        assert DEFAULT_TIMEOUT_SECONDS <= 3600
        # And the default is applied when no timeout arg is given.
        with patch("ollama.Client") as MockClient:
            OllamaChatModel("llama3.1:8b")
            kwargs = MockClient.call_args.kwargs
            assert kwargs.get("timeout") == DEFAULT_TIMEOUT_SECONDS

    def test_default_num_predict_is_set(self):
        """Default num_predict must reach the options dict to prevent
        runaway generations."""
        from llm.ollama_client import DEFAULT_NUM_PREDICT, OllamaChatModel

        with patch("ollama.Client") as MockClient:
            llm = OllamaChatModel("llama3.1:8b")
            assert llm.options.get("num_predict") == DEFAULT_NUM_PREDICT
            assert DEFAULT_NUM_PREDICT >= 64

    def test_reasoning_model_detection(self):
        """is_reasoning_model() must identify known reasoning families."""
        from llm.ollama_client import is_reasoning_model

        # Detected (must auto-get think=False)
        assert is_reasoning_model("qwen3:4b")
        assert is_reasoning_model("qwen3.5:2b")
        assert is_reasoning_model("deepseek-r1:7b")
        assert is_reasoning_model("gpt-oss:20b")
        # Not detected (must NOT get think=False — would 400)
        assert not is_reasoning_model("llama3.1:8b")
        assert not is_reasoning_model("qwen2.5-coder:7b-instruct")
        assert not is_reasoning_model("gemma3:4b")
        assert not is_reasoning_model("mistral:7b")
        assert is_reasoning_model("") is False

    def test_reasoning_model_passes_think_false(self):
        """For a reasoning model, invoke() must pass think=False so the
        model emits a visible answer instead of hidden reasoning.
        Regression: qwen3.5:2b hung for 120s without this.
        """
        from llm.ollama_client import OllamaChatModel

        # Object-style response (current ollama SDK)
        class FakeMsg:
            content = "OK"
        class FakeResp:
            message = FakeMsg()

        with patch("ollama.Client") as MockClient:
            instance = MockClient.return_value
            instance.chat.return_value = FakeResp()
            llm = OllamaChatModel("qwen3.5:2b")
            llm.invoke([{"role": "user", "content": "hi"}])
            kwargs = instance.chat.call_args.kwargs
            assert kwargs.get("think") is False, (
                f"Expected think=False for reasoning model, got {kwargs}"
            )

    def test_non_reasoning_model_omits_think(self):
        """For non-reasoning models, the think kwarg must be omitted
        entirely — Ollama returns 400 if you send think to a model
        that doesn't accept it."""
        from llm.ollama_client import OllamaChatModel

        class FakeMsg:
            content = "OK"
        class FakeResp:
            message = FakeMsg()

        with patch("ollama.Client") as MockClient:
            instance = MockClient.return_value
            instance.chat.return_value = FakeResp()
            llm = OllamaChatModel("llama3.1:8b")
            llm.invoke([{"role": "user", "content": "hi"}])
            kwargs = instance.chat.call_args.kwargs
            assert "think" not in kwargs, (
                f"Expected no `think` kwarg for non-reasoning model, "
                f"got {kwargs}"
            )