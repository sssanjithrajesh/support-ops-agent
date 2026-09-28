"""Ticket record and the committed public sample."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from soa.config import ROOT

SAMPLE_PATH = ROOT / "data" / "raw" / "tickets_sample.json.gz"


@dataclass
class Ticket:
    id: str
    subject: str
    body: str
    requester: str = "customer@example.com"
    queue: str = ""  # label when known (dataset or human); empty for live tickets
    priority: str = ""
    type: str = ""
    tags: list[str] = field(default_factory=list)
    split: str = ""

    @property
    def text(self) -> str:
        return f"{self.subject}\n{self.body}"

    @classmethod
    def from_record(cls, r: dict) -> Ticket:
        return cls(
            id=str(r["id"]),
            subject=r.get("subject", ""),
            body=r.get("body", ""),
            requester=r.get("requester", "customer@example.com"),
            queue=r.get("queue", ""),
            priority=r.get("priority", ""),
            type=r.get("type", ""),
            tags=list(r.get("tags", [])),
            split=r.get("split", ""),
        )


@lru_cache(maxsize=1)
def load_sample(path: Path = SAMPLE_PATH) -> list[Ticket]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [Ticket.from_record(r) for r in json.load(fh)]


def training_tickets() -> list[Ticket]:
    return [t for t in load_sample() if t.split == "train"]


def heldout_tickets() -> list[Ticket]:
    return [t for t in load_sample() if t.split == "heldout"]
