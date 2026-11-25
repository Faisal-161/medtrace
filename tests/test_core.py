from medtrace.bm25 import BM25
from medtrace.embeddings import HashingEmbedder
from medtrace.index import HybridIndex
from medtrace.ingest import chunk_text
from medtrace.kg import KnowledgeGraph, Triple
from medtrace.text import find_markers, split_sentences


def test_sentence_split_keeps_citation_with_its_sentence():
    text = "Trastuzumab targets HER2. [1] Tamoxifen blocks estrogen receptors. [2][3]"
    assert split_sentences(text) == ["Trastuzumab targets HER2. [1]", "Tamoxifen blocks estrogen receptors. [2][3]"]
    assert find_markers(split_sentences(text)[1]) == [2, 3]


def test_chunks_never_cut_sentences_and_overlap():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(20))
    chunks = chunk_text(text, size=30, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert c.endswith(".")
    # last sentence of one chunk reappears at the start of the next
    assert split_sentences(chunks[0])[-1] == split_sentences(chunks[1])[0]


def test_bm25_ranks_keyword_match_first():
    bm = BM25(["the cat sat", "trastuzumab targets her2", "dogs bark loudly"])
    scores = bm.scores("trastuzumab")
    assert scores.index(max(scores)) == 1


def test_hybrid_index_roundtrip(engine, tmp_path):
    engine.index.save(tmp_path / "ix")
    loaded = HybridIndex.load(tmp_path / "ix", HashingEmbedder())
    assert len(loaded) == len(engine.index)
    assert loaded.search("trastuzumab HER2")[0].chunk.source == "her2_and_receptors.md"


def test_kg_confirms_and_detects_contradiction():
    kg = KnowledgeGraph([Triple(subject="trastuzumab", relation="targets", object="HER2")])
    assert kg.check("Trastuzumab is an antibody that targets HER2.") == "confirmed"
    assert kg.check("Trastuzumab does not act on HER2.") == "contradicted"
    assert kg.check("Chemotherapy causes nausea.") == "unknown"
