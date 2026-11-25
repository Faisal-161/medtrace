"""Evaluate MedTrace on a JSONL question set.

Metrics
- retrieval_hit_rate: answerable questions whose expected source is among the retrieved chunks
- answer_coverage: answerable questions whose answer contains every `must_include` phrase
- citation_precision: share of cited sources that are expected sources
- correct_abstention_rate: unanswerable questions the system refused
- false_abstention_rate: answerable questions the system wrongly refused
- mean_unsupported_ratio: share of draft sentences without valid supporting citations
- latency_ms_p50 / p95
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _pct(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def run(engine, dataset: str) -> dict:
    rows, latencies = [], []
    for item in load(dataset):
        t0 = time.perf_counter()
        ans = engine.ask(item["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved_sources = {r.chunk.source for r in ans.retrieved}
        cited_sources = [c.source for c in ans.citations]
        expected = set(item.get("expected_sources", []))
        row = {
            "id": item["id"],
            "question": item["question"],
            "answerable": item["answerable"],
            "abstained": ans.abstained,
            "reason": ans.reason,
            "answer": ans.answer,
            "cited_sources": cited_sources,
            "unsupported_ratio": ans.unsupported_ratio,
        }
        if item["answerable"]:
            row["retrieval_hit"] = bool(expected & retrieved_sources)
            row["covered"] = (not ans.abstained) and all(
                p.lower() in ans.answer.lower() for p in item.get("must_include", [])
            )
            row["citation_precision"] = (
                sum(s in expected for s in cited_sources) / len(cited_sources) if cited_sources else None
            )
        rows.append(row)

    ans_rows = [r for r in rows if r["answerable"]]
    neg_rows = [r for r in rows if not r["answerable"]]
    precisions = [r["citation_precision"] for r in ans_rows if r.get("citation_precision") is not None]
    drafted = [r["unsupported_ratio"] for r in ans_rows if not r["abstained"]]
    lat = sorted(latencies)
    summary = {
        "questions": len(rows),
        "retrieval_hit_rate": _pct([r["retrieval_hit"] for r in ans_rows]),
        "answer_coverage": _pct([r["covered"] for r in ans_rows]),
        "citation_precision": round(statistics.mean(precisions), 3) if precisions else None,
        "correct_abstention_rate": _pct([r["abstained"] for r in neg_rows]),
        "false_abstention_rate": _pct([r["abstained"] for r in ans_rows]),
        "mean_unsupported_ratio": round(statistics.mean(drafted), 3) if drafted else None,
        "latency_ms_p50": round(lat[len(lat) // 2], 1) if lat else None,
        "latency_ms_p95": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 1) if lat else None,
    }
    return {"summary": summary, "rows": rows}
