"""Eval case schema and loader. Cases reference dataset tickets by id or carry their own text."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from soa.config import ROOT
from soa.decision import ACTIONS
from soa.tickets import Ticket, load_sample

CASES_DIR = ROOT / "evals" / "cases"


@dataclass
class Expected:
    queue: str
    priority: str
    action: str
    must_not_auto_send: bool = False
    signals: list[str] | None = None  # None = not asserted; [] = must be none of the escalating kind
    kb_article: str | None = None


@dataclass
class EvalCase:
    id: str
    source: str
    ticket: Ticket
    expected: Expected
    dataset_label: dict = field(default_factory=dict)
    note: str = ""


def load_cases(path: Path = CASES_DIR / "hand_labelled.yaml") -> list[EvalCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    by_id = {t.id: t for t in load_sample()}
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for r in raw:
        if r["id"] in seen:
            raise ValueError(f"duplicate case id {r['id']}")
        seen.add(r["id"])
        e = r["expected"]
        if e["action"] not in ACTIONS:
            raise ValueError(f"{r['id']}: unknown action {e['action']!r}")
        if r["source"] == "dataset":
            src = by_id[r["ticket"]]
            if src.split != "heldout":
                raise ValueError(f"{r['id']}: ticket {src.id} is in the training split")
            ticket = Ticket(id=src.id, subject=src.subject, body=src.body, queue=src.queue, priority=src.priority)
        else:
            ticket = Ticket(
                id=r["id"], subject=r["subject"], body=r["body"], requester=r.get("requester", "customer@example.com")
            )
        cases.append(
            EvalCase(
                id=r["id"],
                source=r["source"],
                ticket=ticket,
                expected=Expected(
                    queue=e["queue"],
                    priority=e["priority"],
                    action=e["action"],
                    must_not_auto_send=bool(e.get("must_not_auto_send", False)),
                    signals=list(e["signals"]) if "signals" in e else None,
                    kb_article=e.get("kb_article"),
                ),
                dataset_label=dict(r.get("dataset_label", {})),
                note=r.get("note", ""),
            )
        )
    return cases
