from pathlib import Path

import pytest

from eval.run_eval import run
from medtrace.pipeline import REFUSAL, MedTrace

ROOT = Path(__file__).resolve().parents[1]


class ScriptedGenerator:
    """Returns a fixed draft, to simulate an LLM that hallucinates or mis-cites."""

    def __init__(self, draft: str):
        self.draft = draft

    def generate(self, question, results):
        return self.draft


def with_generator(engine: MedTrace, draft: str) -> MedTrace:
    return MedTrace(engine.index, ScriptedGenerator(draft), engine.settings, engine.kg)


def test_answers_with_citations(engine):
    ans = engine.ask("What drug is a monoclonal antibody that targets HER2?")
    assert not ans.abstained
    assert "Trastuzumab" in ans.answer
    assert ans.citations and ans.citations[0].source == "her2_and_receptors.md"


def test_abstains_on_off_topic_question(engine):
    ans = engine.ask("Who won the 2022 FIFA World Cup?")
    assert ans.abstained and ans.answer == REFUSAL
    assert ans.reason == "low_retrieval_relevance"


def test_drops_hallucinated_sentence_in_strict_mode(engine):
    draft = (
        "Trastuzumab is a monoclonal antibody that targets HER2. [1] "
        "Trastuzumab also cures diabetes in most patients within weeks. [1]"
    )
    ans = with_generator(engine, draft).ask("What does trastuzumab target?")
    assert not ans.abstained
    assert "diabetes" not in ans.answer
    assert ans.unsupported_ratio == 0.5


def test_uncited_and_out_of_range_markers_are_unsupported(engine):
    draft = "Trastuzumab targets HER2. Tamoxifen blocks estrogen receptors on cancer cells. [99]"
    ans = with_generator(engine, draft).ask("What does trastuzumab target?")
    assert ans.abstained and ans.reason == "no_supported_claims"
    assert all(not c.supported for c in ans.checks)


def test_kg_contradiction_is_dropped(engine):
    draft = (
        "Trastuzumab is a monoclonal antibody that targets HER2. [1] Trastuzumab is not a drug that targets HER2. [1]"
    )
    ans = with_generator(engine, draft).ask("What does trastuzumab target?")
    assert "not a drug" not in ans.answer
    assert any(c.kg_status == "contradicted" for c in ans.checks)


def test_generator_refusal_is_respected(engine):
    ans = with_generator(engine, "INSUFFICIENT_EVIDENCE").ask("What does trastuzumab target?")
    assert ans.abstained and ans.reason == "generator_insufficient_evidence"


def test_eval_baseline_quality_gate(engine):
    summary = run(engine, str(ROOT / "eval/questions.jsonl"))["summary"]
    assert summary["retrieval_hit_rate"] >= 0.9
    assert summary["correct_abstention_rate"] >= 0.75
    assert summary["false_abstention_rate"] <= 0.25


def test_api_health_and_ask(engine, settings, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from medtrace import api

    monkeypatch.setattr(api, "get_engine", lambda: engine)
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    body = client.post("/ask", json={"question": "How is HER2 status tested?"}).json()
    assert body["abstained"] is False and body["citations"]


def test_load_or_build_creates_index_when_missing(settings):
    from medtrace.pipeline import MedTrace as MT

    engine = MT.load_or_build(str(ROOT / "data/sample"), settings)
    assert len(engine.index) > 0
    assert (Path(settings.index_dir) / "meta.json").exists()
    assert len(MT.load_or_build(str(ROOT / "data/sample"), settings).index) == len(engine.index)
