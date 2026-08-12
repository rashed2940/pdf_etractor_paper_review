"""LLM provider adapters (single switch point for chat model providers).

The `make_llm(role)` factory in this package returns an object with a
LangChain-compatible `.invoke(messages)` method, which the agents use.

To add a new provider (OpenAI, Anthropic, llama.cpp, vLLM, …), implement
an adapter class here and add it to `make_llm()`'s dispatch table.

Currently supported:
- **ollama** (default) — uses the native `ollama` Python package
  (`llm.ollama_client.OllamaChatModel`). Recommended for local/private
  use. Models like `llama3.1:8b`, `qwen2.5-coder:14b-instruct`,
  `embeddinggemma`, etc.

Providers left as commented-out alternatives in `make_llm()` for easy
swap-in when needed.
"""