"""Small text utilities: tokenizing, stopwords, sentence splitting."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
# Split after . ! ? or after a trailing citation like "...growth. [2]" -- never before a "[n]" marker.
_SENT = re.compile(r"(?:(?<=[.!?])|(?<=\]))\s+(?=[A-Z0-9(])")
_MARKER = re.compile(r"\[(\d+)\]")

STOPWORDS = frozenset(
    """a an the and or but if then of to in on for with by at from as is are was were be been being
    it its this that these those which who whom what when where why how than so such can could may might
    must should would will shall do does did not no nor only also very more most less least into over
    under about after before between during through up down out off again further once there here all
    any both each few other some own same too just i you he she we they them their our your his her""".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_words(text: str) -> list[str]:
    return [t for t in tokenize(_MARKER.sub(" ", text)) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    parts = _SENT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def find_markers(text: str) -> list[int]:
    return [int(m) for m in _MARKER.findall(text)]


def strip_markers(text: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", text).strip()
