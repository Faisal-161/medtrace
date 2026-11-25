"""Load documents (.txt, .md, .pdf) and split them into overlapping, citable chunks."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .schemas import Chunk
from .text import split_sentences

SUPPORTED = {".txt", ".md", ".pdf"}


def load_pages(path: Path) -> list[tuple[int | None, str]]:
    """Return (page_number, text) pairs. Text files are one page with no number."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return [(None, path.read_text(encoding="utf-8", errors="ignore"))]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Pack whole sentences into chunks of about `size` words.

    Neighbouring chunks share trailing sentences worth about `overlap` words, so a fact that
    straddles a boundary is still retrievable. A chunk never cuts a sentence in half, which keeps
    citations readable.
    """
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    sentences = [s for para in text.split("\n\n") for s in split_sentences(" ".join(para.split()))]
    chunks: list[str] = []
    current: list[str] = []
    for sent in sentences:
        if current and sum(len(s.split()) for s in current) + len(sent.split()) > size:
            chunks.append(" ".join(current))
            carry: list[str] = []
            for s in reversed(current):
                if sum(len(c.split()) for c in carry) + len(s.split()) > overlap:
                    break
                carry.insert(0, s)
            current = carry
        current.append(sent)
    if current and (not chunks or " ".join(current) not in chunks[-1]):
        chunks.append(" ".join(current))
    return chunks


def _chunk_id(source: str, page: int | None, position: int) -> str:
    raw = f"{source}|{page}|{position}".encode()
    return hashlib.sha1(raw).hexdigest()[:12]


def ingest_path(path: str | Path, size: int = 120, overlap: int = 30) -> list[Chunk]:
    """Ingest a file or every supported file under a folder."""
    root = Path(path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)
    chunks: list[Chunk] = []
    for file in files:
        position = 0
        for page, text in load_pages(file):
            for piece in chunk_text(text, size, overlap):
                chunks.append(
                    Chunk(
                        id=_chunk_id(file.name, page, position),
                        text=piece,
                        source=file.name,
                        page=page,
                        position=position,
                    )
                )
                position += 1
    return chunks
