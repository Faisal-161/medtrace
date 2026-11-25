"""Data models shared across the pipeline and the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    page: int | None = None
    position: int = 0  # chunk order within the source


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float  # fused rank score
    relevance: float  # 0..1, used for abstention
    bm25_rank: int | None = None
    dense_rank: int | None = None


class Citation(BaseModel):
    marker: int  # the [n] number shown in the answer
    chunk_id: str
    source: str
    page: int | None = None
    snippet: str


class SentenceCheck(BaseModel):
    sentence: str
    markers: list[int] = Field(default_factory=list)
    supported: bool
    support_score: float
    kg_status: Literal["confirmed", "contradicted", "unknown"] = "unknown"


class Answer(BaseModel):
    question: str
    answer: str
    abstained: bool
    reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    checks: list[SentenceCheck] = Field(default_factory=list)
    unsupported_ratio: float = 0.0
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
