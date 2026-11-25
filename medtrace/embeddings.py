"""Dense embeddings.

`HashingEmbedder` is an offline fallback (feature hashing over words and word bigrams) so the
project runs and tests pass without downloading a model. For real use, set
MEDTRACE_EMBEDDER to a sentence-transformers model such as BAAI/bge-small-en-v1.5.
"""

from __future__ import annotations

import hashlib
from itertools import pairwise
from typing import Protocol

import numpy as np

from .text import content_words


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
        return h % self.dim, (1.0 if (h >> 64) & 1 else -1.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = content_words(text)
            feats = words + [f"{a}_{b}" for a, b in pairwise(words)]
            for f in feats:
                j, sign = self._bucket(f)
                out[i, j] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(name: str) -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(name)
