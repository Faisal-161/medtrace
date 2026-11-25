"""Optional cross-encoder reranking of retrieved chunks."""

from __future__ import annotations

import math

from .schemas import RetrievedChunk


class NoReranker:
    def rerank(self, query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return results


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not results:
            return results
        scores = self.model.predict([(query, r.chunk.text) for r in results])
        for r, s in zip(results, scores, strict=True):
            r.relevance = round(1 / (1 + math.exp(-float(s))), 4)  # logit -> 0..1
        return sorted(results, key=lambda r: -r.relevance)


def get_reranker(name: str):
    return NoReranker() if name == "none" else CrossEncoderReranker(name)
