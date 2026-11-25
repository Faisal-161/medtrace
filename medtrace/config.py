"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Chunking (in words)
    chunk_size: int = 120
    chunk_overlap: int = 30

    # Retrieval
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60

    # Abstain when the best chunk's relevance (0..1) is below this
    min_relevance: float = 0.15
    # A sentence is "supported" if this share of its content words appears in its cited chunks
    support_threshold: float = 0.5

    # Models. "hashing" and "extractive" run fully offline (used in tests and CI).
    embedder: str = "hashing"  # or a sentence-transformers model, e.g. BAAI/bge-small-en-v1.5
    reranker: str = "none"  # or a cross-encoder, e.g. cross-encoder/ms-marco-MiniLM-L-6-v2
    generator: str = "extractive"  # or "claude"
    claude_model: str = "claude-sonnet-5-5"

    # Paths
    index_dir: str = "index"
    kg_path: str = "data/kg/triples.json"

    @classmethod
    def from_env(cls) -> Settings:
        s = cls()
        for name, value in vars(cls()).items():
            raw = os.environ.get(f"MEDTRACE_{name.upper()}")
            if raw is not None:
                setattr(s, name, type(value)(raw))
        return s


def get_settings() -> Settings:
    return Settings.from_env()
