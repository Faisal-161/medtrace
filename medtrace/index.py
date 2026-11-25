"""Hybrid index: BM25 keyword search + dense vector search, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .embeddings import Embedder
from .schemas import Chunk, RetrievedChunk


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, vectors: np.ndarray | None = None):
        self.chunks = chunks
        self.embedder = embedder
        self.bm25 = BM25([c.text for c in chunks])
        self.vectors = vectors if vectors is not None else embedder.embed([c.text for c in chunks])

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int = 5, candidate_k: int = 20, rrf_k: int = 60) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        bm = np.asarray(self.bm25.scores(query))
        qv = self.embedder.embed([query])[0]
        dense = self.vectors @ qv  # cosine similarity: vectors are normalized

        bm_order = [i for i in np.argsort(-bm)[:candidate_k] if bm[i] > 0]
        dense_order = list(np.argsort(-dense)[:candidate_k])
        bm_rank = {int(i): r for r, i in enumerate(bm_order)}
        dense_rank = {int(i): r for r, i in enumerate(dense_order)}

        fused: dict[int, float] = {}
        for ranks in (bm_rank, dense_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + r + 1)

        results = []
        for i, score in sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]:
            # Absolute (not per-query) relevance in 0..1, so an off-topic question scores low
            # even for its best match. BM25 is squashed with s / (s + 5).
            keyword = float(bm[i]) / (float(bm[i]) + 5.0)
            relevance = 0.5 * keyword + 0.5 * max(0.0, float(dense[i]))
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[i],
                    score=score,
                    relevance=round(relevance, 4),
                    bm25_rank=bm_rank.get(i),
                    dense_rank=dense_rank.get(i),
                )
            )
        return results

    # Persistence -------------------------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "chunks.jsonl").write_text("\n".join(c.model_dump_json() for c in self.chunks), encoding="utf-8")
        np.save(d / "vectors.npy", self.vectors)
        (d / "meta.json").write_text(json.dumps({"embedder": self.embedder.name, "count": len(self)}))

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> HybridIndex:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        if meta["embedder"] != embedder.name:
            raise ValueError(f"Index was built with {meta['embedder']!r}, not {embedder.name!r}. Re-run ingest.")
        lines = (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        chunks = [Chunk.model_validate_json(line) for line in lines if line.strip()]
        return cls(chunks, embedder, np.load(d / "vectors.npy"))
