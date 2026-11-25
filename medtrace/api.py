"""FastAPI service.

Run:  uvicorn medtrace.api:app --reload
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .pipeline import MedTrace
from .schemas import Answer

app = FastAPI(
    title="MedTrace",
    version=__version__,
    description="Citation-grounded medical RAG that abstains when evidence is weak.",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    strict: bool = True


class IngestRequest(BaseModel):
    path: str = "data/sample"


@lru_cache
def get_engine() -> MedTrace:
    try:
        return MedTrace.from_settings()
    except FileNotFoundError as exc:
        raise HTTPException(503, "No index found. POST /ingest first.") from exc


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "version": __version__, "embedder": s.embedder, "generator": s.generator}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    engine = MedTrace.build(req.path)
    get_engine.cache_clear()
    return {"chunks": len(engine.index)}


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    return get_engine().ask(req.question, strict=req.strict)
