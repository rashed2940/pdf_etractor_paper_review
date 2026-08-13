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
- **Reasoning-model aware**: detects `qwen3`, `deepseek-r1`, `gpt-oss`,
  and similar "thinking" model families and explicitly passes
  `think=False` so the model emits a visible answer instead of burning
  its output budget on internal chain-of-thought.
- **Hard cap on output tokens** via `num_predict` to prevent runaway
  generations from running for minutes.
- Streaming is supported via `stream=True` in the underlying call.
- Errors are surfaced as exceptions so callers can retry/repair.
"""
from __future__ import annotations

import re
from typing import Any

from config.logging import get_logger

logger = get_logger(__name__)

# Default per-request HTTP timeout (seconds). Ollama itself doesn't
# surface a request-level timeout, so we set it on the underlying httpx
# client. 600 seconds (10 min) covers the slowest expected inference
# for an 8B model with a 3-5K-token prompt on CPU. If your model
# genuinely needs more, raise `LLM_TIMEOUT_SECONDS` in .env — but a
# hung LLM is a broken LLM; fix the model instead of raising the
# timeout.
DEFAULT_TIMEOUT_SECONDS = 600

# Hard cap on output tokens per call. Without this, a model that
# ignores the prompt's length instruction can run for minutes. 512
# is enough for the JSON outputs our agents need (~200-400 tokens).
DEFAULT_NUM_PREDICT = 512

# Model-name patterns that indicate a "reasoning" model which emits
# internal chain-of-thought in a separate `thinking` field. Without
# `think=False`, these models spend their entire output budget on
# hidden reasoning and never produce a visible answer (or hang).
_REASONING_MODEL_RE = re.compile(
    r"(?:^|[:/])(?:"
    r"qwen3(?:\.\d+)?|qwen3\.5|"          # qwen3, qwen3.5
    r"deepseek[-_]?r1|deepseek[-_]?reasoner|"
    r"gpt-oss|"                            # OpenAI's open-source reasoning
    r"magistral|mistral[-_]?(?:small|medium)-3\.\d|"
    r"qwq|sky-t1|qwen-qwq|"
    r"minimax[-_]?m3|"                     # minimax M3 (cloud)
    r"\S+-cloud"                           # any ollama cloud-hosted reasoning model
    r")",
    re.IGNORECASE,
)


def is_reasoning_model(model_name: str) -> bool:
    """Return True if the named model is a reasoning / thinking model.

    Reasoning models (qwen3.x, deepseek-r1, gpt-oss, etc.) emit
    internal chain-of-thought before any visible answer. They need
    `think=False` to be useful for short-form tasks like our agent
    JSON outputs.
    """
    if not model_name:
        return False
    # Strip the optional `:tag` so e.g. "qwen3:4b" still matches.
    base = model_name.split(":")[0]
    return bool(_REASONING_MODEL_RE.search(base))


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
        timeout: Per-request HTTP timeout in seconds. Default: 120.
        num_predict: Hard cap on output tokens (forwarded as
            `options.num_predict`). Default: 2048.

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
        num_predict: int | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.options = dict(options or {})
        # Inject num_predict unless the caller already supplied one.
        if num_predict is not None and "num_predict" not in self.options:
            self.options["num_predict"] = num_predict
        elif "num_predict" not in self.options:
            self.options["num_predict"] = DEFAULT_NUM_PREDICT
        self.timeout = (
            timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
        )
        # Detect reasoning-model families so we can pass think=False.
        self._is_reasoning = is_reasoning_model(model)
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
        chat_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self.options or None,
        }
        # Reasoning models emit visible answers only when asked NOT to
        # think. For non-reasoning models, sending `think=False` is a
        # 400 from Ollama, so we omit it.
        if self._is_reasoning:
            chat_kwargs["think"] = False

        try:
            response = self._client.chat(**chat_kwargs)
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

        # `ollama.Client.chat(stream=False)` returns a ChatResponse object
        # with `.message.content` (and `.message.thinking` for reasoning
        # models, which we explicitly disable above). Older or test-
        # mocked clients may return plain dicts — handle both.
        content = ""
        msg = None
        if isinstance(response, dict):
            msg = response.get("message")
        else:
            msg = getattr(response, "message", None)

        if msg is not None:
            if isinstance(msg, dict):
                content = msg.get("content") or ""
            else:
                content = getattr(msg, "content", "") or ""
                # Some ChatResponse.message objects expose thinking via
                # attribute (the standard). The dict form is rare; we
                # only fall back to attr access here.
        if not content:
            content = str(response)

        return _OllamaResponse(content)

    def stream(self, messages: list[dict[str, str]]):
        """Generator yielding response chunks. Not used by current code."""
        chat_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self.options or None,
        }
        if self._is_reasoning:
            chat_kwargs["think"] = False
        return self._client.chat(**chat_kwargs)


__all__ = [
    "OllamaChatModel",
    "_OllamaResponse",
    "is_reasoning_model",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_NUM_PREDICT",
]