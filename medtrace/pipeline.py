"""The MedTrace pipeline: retrieve -> rerank -> abstain? -> generate -> verify citations -> KG check."""

from __future__ import annotations

from .config import Settings, get_settings
from .embeddings import get_embedder
from .generator import INSUFFICIENT, Generator, get_generator
from .index import HybridIndex
from .ingest import ingest_path
from .kg import KnowledgeGraph
from .rerank import get_reranker
from .schemas import Answer, Citation, RetrievedChunk, SentenceCheck
from .text import content_words, find_markers, split_sentences

REFUSAL = "I don't have enough evidence in the indexed sources to answer that."


def support_score(sentence: str, cited: list[RetrievedChunk]) -> float:
    """Share of the sentence's content words that appear in the chunks it cites."""
    words = content_words(sentence)
    if not words or not cited:
        return 0.0
    vocab = set()
    for r in cited:
        vocab.update(content_words(r.chunk.text))
    return sum(w in vocab for w in words) / len(words)


class MedTrace:
    def __init__(
        self,
        index: HybridIndex,
        generator: Generator,
        settings: Settings | None = None,
        kg: KnowledgeGraph | None = None,
        reranker=None,
    ):
        self.index = index
        self.generator = generator
        self.settings = settings or get_settings()
        self.kg = kg or KnowledgeGraph([])
        self.reranker = reranker or get_reranker("none")

    # Construction helpers ---------------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> MedTrace:
        s = settings or get_settings()
        index = HybridIndex.load(s.index_dir, get_embedder(s.embedder))
        return cls(
            index,
            get_generator(s.generator, s.claude_model),
            s,
            KnowledgeGraph.load(s.kg_path),
            get_reranker(s.reranker),
        )

    @classmethod
    def build(cls, docs_path: str, settings: Settings | None = None, save: bool = True) -> MedTrace:
        s = settings or get_settings()
        chunks = ingest_path(docs_path, s.chunk_size, s.chunk_overlap)
        index = HybridIndex(chunks, get_embedder(s.embedder))
        if save:
            index.save(s.index_dir)
        return cls(
            index,
            get_generator(s.generator, s.claude_model),
            s,
            KnowledgeGraph.load(s.kg_path),
            get_reranker(s.reranker),
        )

    # Main entry point -------------------------------------------------------------------------
    def retrieve(self, question: str) -> list[RetrievedChunk]:
        s = self.settings
        hits = self.index.search(question, top_k=s.candidate_k, candidate_k=s.candidate_k, rrf_k=s.rrf_k)
        return self.reranker.rerank(question, hits)[: s.top_k]

    def ask(self, question: str, strict: bool = True) -> Answer:
        s = self.settings
        retrieved = self.retrieve(question)

        def refuse(reason: str, **extra) -> Answer:
            return Answer(
                question=question, answer=REFUSAL, abstained=True, reason=reason, retrieved=retrieved, **extra
            )

        if not retrieved or max(r.relevance for r in retrieved) < s.min_relevance:
            return refuse("low_retrieval_relevance")

        draft = self.generator.generate(question, retrieved)
        if INSUFFICIENT in draft:
            return refuse("generator_insufficient_evidence")

        checks: list[SentenceCheck] = []
        for sent in split_sentences(draft):
            markers = [m for m in find_markers(sent) if 1 <= m <= len(retrieved)]
            cited = [retrieved[m - 1] for m in markers]
            score = support_score(sent, cited)
            checks.append(
                SentenceCheck(
                    sentence=sent,
                    markers=markers,
                    supported=bool(markers) and score >= s.support_threshold,
                    support_score=round(score, 3),
                    kg_status=self.kg.check(sent),
                )
            )

        unsupported_ratio = sum(not c.supported for c in checks) / len(checks) if checks else 1.0
        kept = [c for c in checks if c.supported and c.kg_status != "contradicted"] if strict else checks
        if not kept:
            return refuse("no_supported_claims", checks=checks, unsupported_ratio=unsupported_ratio)

        used = sorted({m for c in kept for m in c.markers})
        citations = [
            Citation(
                marker=m,
                chunk_id=retrieved[m - 1].chunk.id,
                source=retrieved[m - 1].chunk.source,
                page=retrieved[m - 1].chunk.page,
                snippet=retrieved[m - 1].chunk.text[:240],
            )
            for m in used
        ]
        return Answer(
            question=question,
            answer=" ".join(c.sentence for c in kept),
            abstained=False,
            citations=citations,
            checks=checks,
            unsupported_ratio=round(unsupported_ratio, 3),
            retrieved=retrieved,
        )
