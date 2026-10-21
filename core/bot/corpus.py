"""BM25 retrieval over data/corpus/*.md."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by can do for from have i if in is it its me my of on or "
    "so that the then this to was we when which with you your any keep need tips".split()
)


def _stem(tok: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 3:
            return tok[: -len(suffix)]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Doc:
    name: str
    text: str


class Retriever:
    def __init__(self, docs: list[Doc]):
        if not docs:
            raise ValueError("corpus is empty")
        self.docs = docs
        self._bm25 = BM25Okapi([tokenize(d.text) for d in docs])

    @classmethod
    def from_dir(cls, path: Path = CORPUS_DIR) -> Retriever:
        docs = [Doc(p.stem, p.read_text()) for p in sorted(path.glob("*.md"))]
        return cls(docs)

    def search(self, query: str, k: int = 2) -> list[Doc]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in ranked[:k] if scores[i] > 0]
