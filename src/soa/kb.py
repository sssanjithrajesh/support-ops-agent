"""Knowledge base: twelve Markdown articles with front matter, searched with TF-IDF cosine.

Hand-rolled on purpose. Twelve articles do not need a vector database, and a reviewer should be
able to see exactly why an article was chosen: the overlapping terms are returned with the hit.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from soa.triage import tokens

_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass
class Article:
    id: str
    title: str
    queues: list[str]
    keywords: str
    body: str
    path: Path

    @property
    def summary(self) -> str:
        return self.body.strip().split("\n\n")[0].strip()

    @property
    def steps(self) -> list[str]:
        out = []
        for line in self.body.splitlines():
            m = re.match(r"^\s*\d+\.\s+(.*)$", line)
            if m:
                out.append(m.group(1).strip())
        return out

    @property
    def closing(self) -> str:
        parts = [p.strip() for p in self.body.strip().split("\n\n") if p.strip()]
        last = parts[-1] if parts else ""
        return "" if last.startswith("Steps:") or re.match(r"^\s*\d+\.", last) else last


def parse_article(path: Path) -> Article:
    text = path.read_text(encoding="utf-8")
    m = _FRONT.match(text)
    if not m:
        raise ValueError(f"{path} has no front matter")
    meta = yaml.safe_load(m.group(1))
    return Article(
        id=meta["id"],
        title=meta["title"],
        queues=list(meta.get("queues", [])),
        keywords=str(meta.get("keywords", "")),
        body=m.group(2),
        path=path,
    )


@dataclass
class Hit:
    article: Article
    score: float
    overlap: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.article.id,
            "title": self.article.title,
            "score": round(self.score, 3),
            "overlap": self.overlap,
        }


class KnowledgeBase:
    def __init__(self, articles: list[Article]) -> None:
        self.articles = sorted(articles, key=lambda a: a.id)
        docs = [tokens(f"{a.title} {a.title} {a.keywords} {a.keywords} {a.body}") for a in self.articles]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs) or 1
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self.vectors = [self._vec(d) for d in docs]

    @classmethod
    def load(cls, directory: Path) -> KnowledgeBase:
        return cls([parse_article(p) for p in sorted(directory.glob("*.md"))])

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        v = {t: (1 + math.log(c)) * self.idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, query: str, top_k: int = 3, queue: str | None = None) -> list[Hit]:
        q = self._vec(tokens(query))
        hits = []
        for a, v in zip(self.articles, self.vectors, strict=True):
            common = [t for t in q if t in v]
            score = sum(q[t] * v[t] for t in common)
            if queue and queue in a.queues:
                score *= 1.15  # small nudge toward articles written for the predicted queue
            if score > 0:
                common.sort(key=lambda t: -(q[t] * v[t]))
                hits.append(Hit(a, score, common[:6]))
        hits.sort(key=lambda h: (-h.score, h.article.id))
        return hits[:top_k]

    def get(self, article_id: str) -> Article | None:
        return next((a for a in self.articles if a.id == article_id), None)
