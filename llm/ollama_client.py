"""Native Ollama client wrapper.

Thin wrapper around the official `ollama` Python package that exposes a
LangChain-compatible `.invoke(messages)` interface so the existing
agents (BaseAgent, QA chain, Coordinator) work without changes.

This is the **single switch point** between LLM providers. To add a new
provider (e.g. llama.cpp, vLLM, OpenAI), wrap it here and the rest of
the app picks it up automatically.

Key behaviors:
- `.invoke(messages)` returns an object with a `.content` attribute,
  matching what BaseAgent and the QA chain expect.
- A configurable HTTP timeout prevents the app from hanging forever
  if Ollama stalls (model load, OOM, broken weights, etc.).
- Streaming is supported via `stream=True` in the underlying call.
- Errors are surfaced as exceptions so callers can retry/repair.
"""
from __future__ import annotations

from typing import Any

from config.logging import get_logger

logger = get_logger(__name__)

# Default per-request HTTP timeout (seconds). Ollama itself doesn't
# surface a request-level timeout, so we set it on the underlying httpx
# client. 120 seconds covers the slowest expected inference for a 4-8B
# model with a long context. If your model genuinely needs more, raise
# `LLM_TIMEOUT_SECONDS` in .env — but a hung LLM is a broken LLM; fix
# the model instead of raising the timeout.
DEFAULT_TIMEOUT_SECONDS = 120


class _OllamaResponse:
    """Mimics LangChain's BaseMessage.content interface."""

    def __init__(self, content: str) -> None:
        self.content = content


class OllamaChatModel:
    """Thin chat-completion wrapper around `ollama.chat`.

    Args:
        model: Ollama model tag, e.g. "llama3.1:8b", "qwen2.5-coder:14b-instruct".
        host: Ollama daemon URL. Default: http://localhost:11434.
        options: Optional dict passed to `ollama.chat` (temperature, num_ctx, etc.).

    Usage:
        llm = OllamaChatModel("llama3.1:8b")
        response = llm.invoke([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "Hello!"},
        ])
        print(response.content)
    """

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://localhost:11434",
        options: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.options = options or {}
        self.timeout = (
            timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
        )
        # Lazy-import ollama so test environments without ollama installed
        # can still import this module (the import itself is fine; the
        # chat() call is what fails).
        try:
            import ollama  # noqa: F401 — checked lazily in .invoke()
            # Pass the timeout to the underlying httpx client so a stalled
            # Ollama daemon can't hang the application indefinitely.
            self._client = ollama.Client(
                host=self.host, timeout=self.timeout
            )
        except ImportError as exc:
            raise ImportError(
                "The `ollama` Python package is required. "
                "Install with `pip install ollama`."
            ) from exc

    def invoke(self, messages: list[dict[str, str]]) -> _OllamaResponse:
        """Send a chat completion request and return a response with `.content`.

        Args:
            messages: List of {"role", "content"} dicts (system/user/assistant).

        Raises:
            TimeoutError: If Ollama doesn't respond within `self.timeout` s.
            Exception:     Any underlying Ollama/HTTP error.
        """
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                stream=False,
                options=self.options or None,
            )
        except Exception as exc:
            type_name = type(exc).__name__
            msg = str(exc) or type_name
            # The `ollama` package wraps httpx.TimeoutException as a
            # generic ResponseError in some versions; check the type name.
            if "Timeout" in type_name or "timeout" in msg.lower():
                logger.error(
                    "Ollama chat TIMEOUT after %.0fs: model=%s",
                    self.timeout, self.model,
                )
                raise TimeoutError(
                    f"Ollama did not respond within {self.timeout:.0f}s "
                    f"for model {self.model!r}. The model may be loading, "
                    f"out of memory, or hung. Try a smaller model, free "
                    f"GPU/RAM, or restart Ollama."
                ) from exc
            logger.error(
                "Ollama chat call failed: model=%s err=%s", self.model, exc
            )
            raise

        # `ollama.Client.chat(stream=False)` returns a dict like:
        #   {"message": {"role": "assistant", "content": "..."},
        #    "model": "...", "done": True, ...}
        content = ""
        if isinstance(response, dict):
            msg = response.get("message") or {}
            content = msg.get("content") or ""
        else:
            # Some client versions return an object; fall back to attr access.
            msg = getattr(response, "message", None)
            if msg is not None:
                content = getattr(msg, "content", "") or str(msg)
            else:
                content = str(response)

        return _OllamaResponse(content)

    def stream(self, messages: list[dict[str, str]]):
        """Generator yielding response chunks. Not used by current code."""
        return self._client.chat(
            model=self.model, messages=messages, stream=True, options=self.options or None
        )


__all__ = ["OllamaChatModel", "_OllamaResponse"]