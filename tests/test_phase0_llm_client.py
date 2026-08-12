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