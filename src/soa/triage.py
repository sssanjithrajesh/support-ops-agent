"""Triage: which queue, what priority, which signals, and why.

The classifier is a multinomial naive Bayes over subject + body, trained on the 1,500-ticket
training split at startup (well under a second). No machine-learning dependency: a reviewer
can read the whole model in this file. It is deliberately the boring choice; the value of
the prototype is the decision policy and the integration, not the classifier.

Signals are regexes from config/policy.yaml. They are what actually drive escalation, and
each one quotes the text that matched so a reviewer can check it in one glance.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from soa.config import PRIORITY_ORDER, Policy
from soa.redact import redact
from soa.tickets import Ticket

_TOKEN = re.compile(r"[a-z][a-z0-9\-]{1,}")
_STOP = set(
    """a an the and or but if of to in on at by for with from as is are was were be been being have has had
    do does did i me my we our you your it its this that these those there here please thank thanks would
    could should can will hello hi regards dear team support issue issues problem problems help assistance
    any some about into than then also just very need needs needed require required request""".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class NaiveBayes:
    """Multinomial NB with Laplace smoothing. fit() on (tokens, label) pairs; predict() gives a distribution."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.class_counts: Counter = Counter()
        self.token_counts: dict[str, Counter] = defaultdict(Counter)
        self.total_tokens: Counter = Counter()
        self.vocab: set[str] = set()

    def fit(self, docs: list[tuple[list[str], str]]) -> NaiveBayes:
        for toks, label in docs:
            self.class_counts[label] += 1
            self.token_counts[label].update(toks)
            self.total_tokens[label] += len(toks)
            self.vocab.update(toks)
        return self

    def log_scores(self, toks: list[str]) -> dict[str, float]:
        """Unnormalised log posterior per class."""
        if not self.class_counts:
            return {}
        n = sum(self.class_counts.values())
        v = len(self.vocab) or 1
        logp: dict[str, float] = {}
        for label, count in self.class_counts.items():
            lp = math.log(count / n)
            denom = self.total_tokens[label] + self.alpha * v
            tc = self.token_counts[label]
            for t in toks:
                lp += math.log((tc.get(t, 0) + self.alpha) / denom)
            logp[label] = lp
        return logp

    def predict(self, toks: list[str], temperature: float = 1.0) -> dict[str, float]:
        """Class distribution. temperature > 1 flattens the (famously overconfident) NB posterior."""
        return softmax({k: v / temperature for k, v in self.log_scores(toks).items()})


def softmax(logscore: dict[str, float]) -> dict[str, float]:
    if not logscore:
        return {}
    mx = max(logscore.values())
    exp = {k: math.exp(v - mx) for k, v in logscore.items()}
    z = sum(exp.values())
    return {k: v / z for k, v in exp.items()}


@dataclass
class SignalHit:
    name: str
    description: str
    evidence: list[str]


@dataclass
class TriageResult:
    ticket_id: str
    queue: str
    queue_confidence: float
    queue_runner_up: str
    priority: str
    priority_base: str
    priority_boost: int
    type: str
    signals: list[SignalHit] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    @property
    def signal_names(self) -> set[str]:
        return {s.name for s in self.signals}

    def as_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "queue": self.queue,
            "queue_confidence": round(self.queue_confidence, 3),
            "queue_runner_up": self.queue_runner_up,
            "priority": self.priority,
            "priority_base": self.priority_base,
            "priority_boost": self.priority_boost,
            "type": self.type,
            "signals": [{"name": s.name, "evidence": s.evidence} for s in self.signals],
            "trace": list(self.trace),
        }


class Triager:
    def __init__(self, policy: Policy, training: list[Ticket]) -> None:
        self.policy = policy
        self.queue_model = NaiveBayes().fit([(tokens(t.text), t.queue) for t in training if t.queue])
        self.priority_model = NaiveBayes().fit([(tokens(t.text), t.priority) for t in training if t.priority])
        self.type_model = NaiveBayes().fit([(tokens(t.text), t.type) for t in training if t.type])
        self.n_training = len(training)

    def detect_signals(self, text: str) -> list[SignalHit]:
        hits = []
        for s in self.policy.signals:
            ev = s.matches(text)
            if ev:
                # The evidence is quoted in traces and internal notes; never quote a card number back.
                hits.append(SignalHit(s.name, s.description, [redact(e)[0] for e in ev[:3]]))
        return hits

    def triage(self, ticket: Ticket) -> TriageResult:
        toks = tokens(ticket.text)
        trace: list[str] = []

        # Naive Bayes multiplies one likelihood per token, so a 60-token ticket comes out 99.99% sure of a
        # wrong answer. Dividing by (tokens / temperature_tokens) puts the classifier and the human-written
        # hints on the same scale; policy.yaml owns both numbers.
        temperature = max(1.0, len(toks) / self.policy.classifier_temperature_tokens)
        qdist = self.queue_model.predict(toks, temperature)
        nb_top = max(qdist.items(), key=lambda kv: kv[1])
        trace.append(f"classifier: {nb_top[0]} ({nb_top[1]:.0%}) from {self.n_training} labelled tickets")
        hints = {h.queue: h.matches(ticket.text) for h in self.policy.queue_hints}
        hints = {q: m for q, m in hints.items() if m}
        logscore = {q: math.log(max(p, 1e-12)) for q, p in qdist.items()}
        for q, matched in hints.items():
            logscore[q] = logscore.get(q, math.log(1e-12)) + self.policy.queue_hint_weight * len(matched)
            trace.append(
                f"hint {q}: {', '.join(repr(m) for m in matched)} (+{self.policy.queue_hint_weight * len(matched):.1f})"
            )
        ranked = sorted(softmax(logscore).items(), key=lambda kv: -kv[1])
        queue, conf = ranked[0]
        runner = ranked[1][0] if len(ranked) > 1 else ""
        trace.append(f"queue: {queue} ({conf:.0%}) over {runner} ({ranked[1][1]:.0%})" if runner else f"queue: {queue}")

        pdist = self.priority_model.predict(toks)
        base = max(pdist.items(), key=lambda kv: kv[1])[0]
        trace.append(f"priority_base: {base} ({pdist[base]:.0%}) from classifier")

        signals = self.detect_signals(ticket.text)
        for s in signals:
            trace.append(f"signal {s.name}: {', '.join(repr(e) for e in s.evidence)}")

        boost = sum(self.policy.priority_boosts.get(s.name, 0) for s in signals)
        level = min(2, PRIORITY_ORDER[base] + boost)
        priority = [k for k, v in PRIORITY_ORDER.items() if v == level][0]
        if boost:
            boosted = [
                f"{s.name} +{self.policy.priority_boosts[s.name]}"
                for s in signals
                if s.name in self.policy.priority_boosts
            ]
            trace.append(f"priority: {base} -> {priority} ({', '.join(boosted)})")
        else:
            trace.append(f"priority: {priority} (no boosts)")

        tdist = self.type_model.predict(toks)
        ttype = max(tdist.items(), key=lambda kv: kv[1])[0] if tdist else ""

        return TriageResult(
            ticket_id=ticket.id,
            queue=queue,
            queue_confidence=conf,
            queue_runner_up=runner,
            priority=priority,
            priority_base=base,
            priority_boost=boost,
            type=ttype,
            signals=signals,
            trace=trace,
        )
