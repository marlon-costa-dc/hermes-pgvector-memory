"""Ollama embedding client.

Deliberately built on ``urllib`` from the standard library rather than httpx or
requests: this module is imported inside the Hermes agent process, and a memory
plugin has no business pinning an HTTP library version for its host.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "nomic-embed-text"

# Dimensions of embedders commonly served by Ollama. Used only to fail fast on
# a dimension mismatch against an existing table; an unknown model is fine, we
# just learn its size from the first response.
KNOWN_DIMS = {
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "mxbai-embed-large": 1024,
    "bge-m3": 1024,
    "snowflake-arctic-embed": 1024,
}


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced.

    Never swallowed into a zero vector: a zero vector is silently equidistant
    from everything and would poison recall while looking healthy.
    """


class OllamaEmbedder:
    """Minimal client for Ollama's ``/api/embed`` endpoint."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
    ) -> None:
        # Defensive: a caller passing None (e.g. config that resolved to a
        # missing key) must land on the default, not crash on .rstrip() with
        # a traceback that says nothing about the real cause.
        self.host = (host or DEFAULT_HOST).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self._dims: int | None = KNOWN_DIMS.get(self.model)

    @property
    def dims(self) -> int | None:
        """Embedding width, once known (from KNOWN_DIMS or the first call)."""
        return self._dims

    def is_available(self) -> bool:
        """True when the Ollama daemon answers and serves the configured model.

        Called from ``MemoryProvider.is_available()``, which the host documents
        as cheap — this is a single localhost GET with a short timeout.
        """
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                payload = json.load(resp)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.debug("ollama unreachable at %s: %s", self.host, exc)
            return False

        served = {m.get("name", "") for m in payload.get("models", [])}
        # Ollama reports "nomic-embed-text:latest" for a bare "nomic-embed-text".
        return any(name == self.model or name.startswith(f"{self.model}:") for name in served)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed one or more texts. Order of the result matches the input."""
        if not texts:
            return []

        body = json.dumps({"model": self.model, "input": list(texts)}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise EmbeddingError(
                f"Ollama returned HTTP {exc.code} for model {self.model!r}: {detail}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise EmbeddingError(f"Cannot reach Ollama at {self.host}: {exc}") from exc
        except ValueError as exc:
            raise EmbeddingError(f"Ollama returned malformed JSON: {exc}") from exc

        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError(
                f"Ollama returned {len(vectors) if isinstance(vectors, list) else 'no'} "
                f"embeddings for {len(texts)} inputs"
            )

        width = len(vectors[0])
        if self._dims is None:
            self._dims = width
        elif width != self._dims:
            raise EmbeddingError(
                f"Model {self.model!r} returned {width}-dim vectors, expected {self._dims}. "
                "Changing embedder requires re-embedding the whole table."
            )
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def to_pgvector(vector: Sequence[float]) -> str:
    """Render a vector in pgvector's literal syntax: ``[1.0,2.0,3.0]``.

    Passed as a bound parameter, never string-interpolated into SQL.
    """
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"
