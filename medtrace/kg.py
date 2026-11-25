"""Knowledge-graph claim checking.

A deliberately simple baseline: facts are (subject, relation, object) triples in a JSON file.
A sentence that mentions a triple's subject and object is "confirmed"; if it also negates the
link ("does not", "no", "not"), it is "contradicted". Swap `KnowledgeGraph` for a Neo4j-backed
version (e.g. OncoKG) behind the same `check` method.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

Status = Literal["confirmed", "contradicted", "unknown"]
_NEGATION = re.compile(r"\b(not|no|never|cannot|doesn't|don't|isn't|aren't|without)\b", re.IGNORECASE)


class Triple(BaseModel):
    subject: str
    relation: str
    object: str
    aliases: dict[str, list[str]] = {}


def _mentions(text: str, term: str, aliases: list[str]) -> bool:
    low = text.lower()
    return any(re.search(rf"\b{re.escape(t.lower())}\b", low) for t in [term, *aliases])


class KnowledgeGraph:
    def __init__(self, triples: list[Triple]):
        self.triples = triples

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeGraph:
        p = Path(path)
        if not p.exists():
            return cls([])
        return cls([Triple(**t) for t in json.loads(p.read_text(encoding="utf-8"))])

    def check(self, sentence: str) -> Status:
        status: Status = "unknown"
        for t in self.triples:
            if _mentions(sentence, t.subject, t.aliases.get("subject", [])) and _mentions(
                sentence, t.object, t.aliases.get("object", [])
            ):
                if _NEGATION.search(sentence):
                    return "contradicted"
                status = "confirmed"
        return status
