"""Embedding service abstraction + implementations (PLAN.md §6.3).

Active default uses the native **Ollama** Python package via the
`embeddinggemma` model (or any other Ollama embedding model). Other
providers (SentenceTransformers, OpenAI) are available as commented
alternatives right next to the active code.

To switch, either:
1. Set `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL` in `.env`, OR
2. Uncomment the desired provider block in `get_embedding_service()`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from config.logging import get_logger
from config.settings import get_settings

logger = get_logger(__name__)


class EmbeddingService(ABC):
    """Pluggable embedding backend interface."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the active model (used in ChromaDB collection metadata)."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output dimensionality (must match the stored collection's expected dim)."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents (no query prefix)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query (may apply a query prefix)."""


class OllamaEmbeddingError(RuntimeError):
    """Raised when the configured Ollama embedding model is missing."""


class OllamaEmbeddingService(EmbeddingService):
    """Embedding backend using Ollama's native `ollama.embeddings()` API.

    Recommended default. Works with any Ollama embedding model, e.g.:
      - `embeddinggemma` (Google's embedding model via Ollama)
      - `nomic-embed-text`
      - `mxbai-embed-large`
      - `all-minilm`

    The first call to `embed_documents()` lazily verifies the model is
    available locally; if missing, raises an `OllamaEmbeddingError` with
    an actionable message ("ollama pull <model>").
    """

    def __init__(self, model_name: str | None = None) -> None:
        s = get_settings()
        self._model_name = model_name or s.embedding_model
        try:
            import ollama

            self._client = ollama.Client(host=s.ollama_base_url)
        except ImportError as exc:
            raise ImportError(
                "The `ollama` Python package is required. "
                "Install with `pip install ollama`."
            ) from exc

        # Probe the model to fail fast with a clear error if missing.
        try:
            probe = self._client.embeddings(model=self._model_name, prompt="probe")
            self._dim = len(probe["embedding"])
        except Exception as exc:
            raise OllamaEmbeddingError(
                f"Ollama embedding model '{self._model_name}' is unavailable. "
                f"Run `ollama pull {self._model_name}` and ensure Ollama is "
                f"running at {s.ollama_base_url}. Underlying error: {exc}"
            ) from exc

        logger.info(
            "Ollama embedding model loaded: %s (dim=%d)",
            self._model_name, self._dim,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Ollama 0.6.x doesn't expose a native batch endpoint for embeddings;
        # call once per text. For very large corpora, switch to a model
        # that supports batch (e.g. via an OpenAI-compatible server).
        vectors: list[list[float]] = []
        for text in texts:
            resp = self._client.embeddings(model=self._model_name, prompt=text)
            vectors.append([float(x) for x in resp["embedding"]])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embeddings(model=self._model_name, prompt=text)
        return [float(x) for x in resp["embedding"]]


class SentenceTransformerEmbeddingService(EmbeddingService):
    """Local SentenceTransformers (CPU-friendly alternative).

    Uncomment the call in `get_embedding_service()` to enable. Useful
    for environments where Ollama's embedding model is too heavy.
    """

    def __init__(self, model_name: str | None = None) -> None:
        s = get_settings()
        self._model_name = model_name or s.embedding_model
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SentenceTransformer model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.astype(float).tolist() for v in np.asarray(vectors)]

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vec)[0].astype(float).tolist()


def get_embedding_service() -> EmbeddingService:
    """Factory: returns the configured EmbeddingService.

    Active default: **Ollama** via the native `ollama` package.
    SentenceTransformers is available as a commented alternative.
    """
    s = get_settings()

    if s.embedding_provider == "ollama":
        # --- Active: Ollama native embeddings (recommended) ---
        return OllamaEmbeddingService()

    if s.embedding_provider == "sentence_transformers":
        # --- Alternative: SentenceTransformers (CPU-friendly) ---
        return SentenceTransformerEmbeddingService()

    # --- Alternative: OpenAI embeddings (uncomment + `pip install openai`) ---
    # if s.embedding_provider == "openai":
    #     from openai import OpenAI
    #     class _OpenAIEmbeddingService(EmbeddingService):
    #         def __init__(self):
    #             self._model_name = s.embedding_model
    #             self._client = OpenAI(api_key=s.openai_api_key)
    #             self._dim = None
    #         @property
    #         def model_name(self): return self._model_name
    #         @property
    #         def dimension(self):
    #             if self._dim is None:
    #                 resp = self._client.embeddings.create(
    #                     model=self._model_name, input="probe"
    #                 )
    #                 self._dim = len(resp.data[0].embedding)
    #             return self._dim
    #         def embed_documents(self, texts):
    #             resp = self._client.embeddings.create(
    #                 model=self._model_name, input=texts
    #             )
    #             return [d.embedding for d in resp.data]
    #         def embed_query(self, text):
    #             return self.embed_documents([text])[0]
    #     return _OpenAIEmbeddingService()

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER={s.embedding_provider!r}. "
        "Use 'ollama' (active) or 'sentence_transformers'."
    )


__all__ = [
    "EmbeddingService",
    "OllamaEmbeddingService",
    "OllamaEmbeddingError",
    "SentenceTransformerEmbeddingService",
    "get_embedding_service",
]