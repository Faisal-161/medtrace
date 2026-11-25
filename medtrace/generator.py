"""Answer generation from numbered sources.

- `ClaudeGenerator` calls the Anthropic API and must cite every sentence as [n].
- `ExtractiveGenerator` needs no API key: it selects the source sentences that best match the
  question. It is the offline baseline used in tests, CI, and evals.
"""

from __future__ import annotations

import os
from typing import Protocol

from .schemas import RetrievedChunk
from .text import content_words, split_sentences

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = f"""You are MedTrace, a careful medical literature assistant.
Answer ONLY from the numbered sources provided. Rules:
1. End every sentence with the source number(s) that support it, like [1] or [2][3].
2. Never use outside knowledge. If a fact is not in the sources, leave it out.
3. If the sources do not answer the question, reply with exactly: {INSUFFICIENT}
4. Be concise: at most 5 sentences. No preamble.
This is for education and research, not medical advice."""


def format_sources(results: list[RetrievedChunk]) -> str:
    lines = []
    for n, r in enumerate(results, start=1):
        loc = f"{r.chunk.source}" + (f", p.{r.chunk.page}" if r.chunk.page else "")
        lines.append(f"[{n}] ({loc})\n{r.chunk.text}")
    return "\n\n".join(lines)


class Generator(Protocol):
    def generate(self, question: str, results: list[RetrievedChunk]) -> str: ...


class ExtractiveGenerator:
    def __init__(self, max_sentences: int = 3, min_overlap: int = 2):
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def generate(self, question: str, results: list[RetrievedChunk]) -> str:
        q = set(content_words(question))
        scored = []
        for n, r in enumerate(results, start=1):
            for sent in split_sentences(r.chunk.text):
                overlap = len(q & set(content_words(sent)))
                if overlap >= self.min_overlap:
                    scored.append((overlap, -n, sent, n))
        if not scored:
            return INSUFFICIENT
        picked, seen = [], set()
        for _, _, sent, n in sorted(scored, reverse=True):
            if sent in seen:
                continue
            seen.add(sent)
            picked.append(f"{sent.rstrip()} [{n}]")
            if len(picked) == self.max_sentences:
                break
        return " ".join(picked)


class ClaudeGenerator:
    def __init__(self, model: str):
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY to use the Claude generator.")
        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, question: str, results: list[RetrievedChunk]) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Sources:\n\n{format_sources(results)}\n\nQuestion: {question}"}],
        )
        return "".join(block.text for block in msg.content if block.type == "text").strip()


def get_generator(name: str, claude_model: str) -> Generator:
    if name == "claude":
        return ClaudeGenerator(claude_model)
    return ExtractiveGenerator()
