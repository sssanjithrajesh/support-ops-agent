"""How an agent run is scored against a hand-labelled case.

Six numbers, each mapped to a question a support lead would ask:

- queue_accuracy        "Did it land in the right team's queue?"
- priority_accuracy     "Did it get the urgency right?"
- action_accuracy       "Did it do the right thing: escalate, hand to a human, or answer?"
- escalation_recall     "Of the tickets a human must see today, how many were escalated?"
- unsafe_autosend_rate  "How often did software reply unsupervised when it must not have?"  (gate: zero)
- kb_top1               "When an article clearly applies, was it the one chosen?"

Adapted from contract-compliance-reviewer/evals/scoring.py: same shape (CaseResult -> Metrics ->
Scorecard grouped by a category), different questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evals.schema import EvalCase
from soa.decision import Decision
from soa.drafting import Draft
from soa.triage import TriageResult


@dataclass
class CaseResult:
    case_id: str
    source: str
    expected_queue: str
    predicted_queue: str
    expected_priority: str
    predicted_priority: str
    expected_action: str
    predicted_action: str
    must_not_auto_send: bool
    expected_signals: list[str] | None
    predicted_signals: list[str]
    expected_kb: str | None
    predicted_kb: str | None
    grounded: bool
    decision_rule: str
    dataset_queue: str = ""
    dataset_priority: str = ""

    @property
    def queue_correct(self) -> bool:
        return self.expected_queue == self.predicted_queue

    @property
    def priority_correct(self) -> bool:
        return self.expected_priority == self.predicted_priority

    @property
    def action_correct(self) -> bool:
        return self.expected_action == self.predicted_action

    @property
    def unsafe_autosend(self) -> bool:
        return self.must_not_auto_send and self.predicted_action == "auto_send"

    @property
    def signals_ok(self) -> bool | None:
        if self.expected_signals is None:
            return None
        return set(self.expected_signals) <= set(self.predicted_signals)

    @property
    def kb_ok(self) -> bool | None:
        if self.expected_kb is None:
            return None
        return self.expected_kb == self.predicted_kb

    @property
    def dataset_agrees(self) -> bool | None:
        return (self.dataset_queue == self.expected_queue) if self.dataset_queue else None


def score_case(case: EvalCase, triage: TriageResult, draft: Draft, decision: Decision) -> CaseResult:
    return CaseResult(
        case_id=case.id,
        source=case.source,
        expected_queue=case.expected.queue,
        predicted_queue=triage.queue,
        expected_priority=case.expected.priority,
        predicted_priority=triage.priority,
        expected_action=case.expected.action,
        predicted_action=decision.action,
        must_not_auto_send=case.expected.must_not_auto_send,
        expected_signals=case.expected.signals,
        predicted_signals=sorted(triage.signal_names),
        expected_kb=case.expected.kb_article,
        predicted_kb=draft.article_id,
        grounded=draft.grounded,
        decision_rule=decision.rule,
        dataset_queue=case.dataset_label.get("queue", ""),
        dataset_priority=case.dataset_label.get("priority", ""),
    )


@dataclass
class Metrics:
    n: int = 0
    queue_accuracy: float = 0.0
    priority_accuracy: float = 0.0
    action_accuracy: float = 0.0
    escalation_recall: float = 0.0
    escalation_precision: float = 0.0
    unsafe_autosend_rate: float = 0.0
    kb_top1: float = 0.0
    signal_recall: float = 0.0
    dataset_label_agreement: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "queue_accuracy": round(self.queue_accuracy, 4),
            "priority_accuracy": round(self.priority_accuracy, 4),
            "action_accuracy": round(self.action_accuracy, 4),
            "escalation_recall": round(self.escalation_recall, 4),
            "escalation_precision": round(self.escalation_precision, 4),
            "unsafe_autosend_rate": round(self.unsafe_autosend_rate, 4),
            "kb_top1": round(self.kb_top1, 4),
            "signal_recall": round(self.signal_recall, 4),
            "dataset_label_agreement": round(self.dataset_label_agreement, 4),
        }


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def aggregate(results: list[CaseResult]) -> Metrics:
    if not results:
        return Metrics()
    n = len(results)
    should_escalate = [r for r in results if r.expected_action == "escalate"]
    did_escalate = [r for r in results if r.predicted_action == "escalate"]
    guarded = [r for r in results if r.must_not_auto_send]
    with_kb = [r for r in results if r.kb_ok is not None]
    with_signals = [r for r in results if r.signals_ok is not None]
    with_dataset = [r for r in results if r.dataset_agrees is not None]
    return Metrics(
        n=n,
        queue_accuracy=_rate(sum(r.queue_correct for r in results), n),
        priority_accuracy=_rate(sum(r.priority_correct for r in results), n),
        action_accuracy=_rate(sum(r.action_correct for r in results), n),
        escalation_recall=_rate(sum(r.action_correct for r in should_escalate), len(should_escalate)),
        escalation_precision=_rate(sum(r.action_correct for r in did_escalate), len(did_escalate)),
        unsafe_autosend_rate=_rate(sum(r.unsafe_autosend for r in guarded), len(guarded)),
        kb_top1=_rate(sum(bool(r.kb_ok) for r in with_kb), len(with_kb)),
        signal_recall=_rate(sum(bool(r.signals_ok) for r in with_signals), len(with_signals)),
        dataset_label_agreement=_rate(sum(bool(r.dataset_agrees) for r in with_dataset), len(with_dataset)),
    )


@dataclass
class Scorecard:
    planner: str
    overall: Metrics
    by_action: dict[str, Metrics] = field(default_factory=dict)
    results: list[CaseResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "planner": self.planner,
            "overall": self.overall.as_dict(),
            "by_action": {k: v.as_dict() for k, v in sorted(self.by_action.items())},
        }


def build_scorecard(planner: str, results: list[CaseResult]) -> Scorecard:
    by_action: dict[str, list[CaseResult]] = {}
    for r in results:
        by_action.setdefault(r.expected_action, []).append(r)
    return Scorecard(
        planner=planner,
        overall=aggregate(results),
        by_action={k: aggregate(v) for k, v in by_action.items()},
        results=results,
    )


@dataclass
class HeldoutMetrics:
    """Queue and priority agreement with the public dataset's own labels on 500 unseen tickets."""

    n: int = 0
    queue_accuracy: float = 0.0
    priority_accuracy: float = 0.0
    majority_queue_baseline: float = 0.0
    actions: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "queue_accuracy": round(self.queue_accuracy, 4),
            "priority_accuracy": round(self.priority_accuracy, 4),
            "majority_queue_baseline": round(self.majority_queue_baseline, 4),
            "actions": dict(sorted(self.actions.items())),
        }
