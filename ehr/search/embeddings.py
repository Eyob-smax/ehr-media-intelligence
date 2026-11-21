"""Text embedders.

``SentenceTransformerEmbedder`` (all-MiniLM-L6-v2, 384-d) is the real model.
``HashEmbedder`` is a dependency-free bag-of-words hashing embedder with the
same interface; it lets the test suite and CI run without downloading model
weights, and is selected with ``EHR_EMBEDDER=hash``.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from functools import lru_cache
from typing import Protocol, Sequence

from ..config import Settings


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.name = model_name
        self._model = None
        self._lock = threading.Lock()
        self.dim = 384

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer  # heavy import, deferred

                self._model = SentenceTransformer(self.name, device="cpu")
                get_dim = (getattr(self._model, "get_embedding_dimension", None)
                           or self._model.get_sentence_embedding_dimension)
                self.dim = get_dim() or self.dim
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        vecs = model.encode(list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()


_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Feature-hashed unigrams + bigrams, L2-normalized. Lexical, not semantic, but deterministic and instant."""

    def __init__(self, dim: int = 384):
        self.name = f"hash-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        toks = _TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


@lru_cache(maxsize=4)
def _cached(kind: str, model_name: str) -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    if kind in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"unknown EHR_EMBEDDER={kind!r} (use sentence-transformers or hash)")


def get_embedder(settings: Settings) -> Embedder:
    return _cached(settings.embedder, settings.embedding_model)
