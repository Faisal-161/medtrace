"""Okapi BM25 for keyword retrieval. Dependency-free so it runs anywhere."""

from __future__ import annotations

import math
from collections import Counter

from .text import STOPWORDS, tokenize


def _terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(_terms(d)) for d in docs]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = _terms(query)
        out = []
        for tf, dl in zip(self.tfs, self.lengths, strict=True):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf[t] * f * (self.k1 + 1) / denom
            out.append(s)
        return out
