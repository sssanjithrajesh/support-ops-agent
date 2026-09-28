"""Load config/policy.yaml into typed objects. The YAML is the business logic; this file only reads it."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "config" / "policy.yaml"

PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class Queue:
    name: str
    tag: str
    owner: str


@dataclass(frozen=True)
class Signal:
    name: str
    description: str
    patterns: tuple[re.Pattern, ...]

    def matches(self, text: str) -> list[str]:
        """Return the matched snippets (deduplicated, in order) so the trace can quote them."""
        found: list[str] = []
        for p in self.patterns:
            for m in p.finditer(text):
                snippet = m.group(0)
                if snippet not in found:
                    found.append(snippet)
        return found


@dataclass(frozen=True)
class EscalationRule:
    name: str
    any_signal: tuple[str, ...] = ()
    all_signals: tuple[str, ...] = ()
    min_priority: str | None = None

    def fires(self, signals: set[str], priority: str) -> bool:
        if self.any_signal and not (signals & set(self.any_signal)):
            return False
        if self.all_signals and not set(self.all_signals) <= signals:
            return False
        if self.min_priority and PRIORITY_ORDER[priority] < PRIORITY_ORDER[self.min_priority]:
            return False
        return bool(self.any_signal or self.all_signals or self.min_priority)


@dataclass(frozen=True)
class AutoSendRule:
    queues: tuple[str, ...]
    max_priority: str
    min_queue_confidence: float
    min_kb_score: float
    forbidden_signals: tuple[str, ...]


@dataclass(frozen=True)
class QueueHint:
    queue: str
    patterns: tuple[re.Pattern, ...]

    def matches(self, text: str) -> list[str]:
        found: list[str] = []
        for p in self.patterns:
            m = p.search(text)
            if m and m.group(0).lower() not in found:
                found.append(m.group(0).lower())
        return found


@dataclass(frozen=True)
class Policy:
    queues: tuple[Queue, ...]
    priorities: tuple[str, ...]
    queue_hints: tuple[QueueHint, ...]
    queue_hint_weight: float
    classifier_temperature_tokens: float
    signals: tuple[Signal, ...]
    escalate_when: tuple[EscalationRule, ...]
    auto_send: AutoSendRule
    priority_boosts: dict[str, int]
    kb_dir: Path
    kb_top_k: int
    slack_channel_hint: str
    source: Path = field(default=POLICY_PATH)

    def queue_by_name(self, name: str) -> Queue:
        for q in self.queues:
            if q.name == name:
                return q
        raise KeyError(name)

    @property
    def queue_names(self) -> list[str]:
        return [q.name for q in self.queues]


def load_policy(path: Path = POLICY_PATH) -> Policy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    signals = tuple(
        Signal(
            name=name,
            description=spec.get("description", ""),
            patterns=tuple(re.compile(p, re.IGNORECASE | re.DOTALL) for p in spec["patterns"]),
        )
        for name, spec in raw["signals"].items()
    )
    known = {s.name for s in signals}
    rules = []
    for r in raw["escalate_when"]:
        for s in list(r.get("any_signal", [])) + list(r.get("all_signals", [])):
            if s not in known:
                raise ValueError(f"escalation rule {r['name']!r} references unknown signal {s!r}")
        rules.append(
            EscalationRule(
                name=r["name"],
                any_signal=tuple(r.get("any_signal", [])),
                all_signals=tuple(r.get("all_signals", [])),
                min_priority=r.get("min_priority"),
            )
        )
    a = raw["auto_send"]
    queue_names = {q["name"] for q in raw["queues"]}
    hints = []
    for qname, pats in raw.get("queue_hints", {}).items():
        if qname not in queue_names:
            raise ValueError(f"queue_hints references unknown queue {qname!r}")
        hints.append(QueueHint(qname, tuple(re.compile(p, re.IGNORECASE) for p in pats)))
    return Policy(
        queues=tuple(Queue(q["name"], q["tag"], q["owner"]) for q in raw["queues"]),
        priorities=tuple(raw["priorities"]),
        queue_hints=tuple(hints),
        queue_hint_weight=float(raw.get("queue_hint_weight", 1.0)),
        classifier_temperature_tokens=float(raw.get("classifier_temperature_tokens", 6)),
        signals=signals,
        escalate_when=tuple(rules),
        auto_send=AutoSendRule(
            queues=tuple(a["queues"]),
            max_priority=a["max_priority"],
            min_queue_confidence=float(a["min_queue_confidence"]),
            min_kb_score=float(a["min_kb_score"]),
            forbidden_signals=tuple(a["forbidden_signals"]),
        ),
        priority_boosts={k: int(v) for k, v in raw.get("priority_boosts", {}).items()},
        kb_dir=(path.parent.parent / raw["knowledge_base"]["dir"]).resolve(),
        kb_top_k=int(raw["knowledge_base"].get("top_k", 3)),
        slack_channel_hint=raw.get("slack", {}).get("channel_hint", "#support"),
        source=path,
    )


@lru_cache(maxsize=1)
def policy() -> Policy:
    return load_policy()
