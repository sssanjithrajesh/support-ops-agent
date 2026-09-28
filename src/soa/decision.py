"""Turn triage + draft into one of three actions, with the rule that decided it written down.

    escalate   -> Slack, internal note, no customer reply until a human speaks
    review     -> draft attached to the ticket as an internal note; a human approves, edits or rejects
    auto_send  -> the draft goes to the customer; only for a narrow, configurable set of cases

The order matters: escalation wins over everything, and auto-send has to pass every check. A
ticket that passes none of the auto-send checks is not an error; review is the default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from soa.config import PRIORITY_ORDER, Policy
from soa.drafting import Draft
from soa.triage import TriageResult

ACTIONS = ("escalate", "review", "auto_send")


@dataclass
class Decision:
    action: str
    rule: str
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"action": self.action, "rule": self.rule, "reasons": list(self.reasons)}


def decide(policy: Policy, triage: TriageResult, draft: Draft) -> Decision:
    signals = triage.signal_names
    for rule in policy.escalate_when:
        if rule.fires(signals, triage.priority):
            fired = sorted(signals & (set(rule.any_signal) | set(rule.all_signals)))
            reasons = [f"rule {rule.name} fired on signals {fired}"]
            if rule.min_priority:
                reasons.append(f"priority {triage.priority} >= {rule.min_priority}")
            return Decision("escalate", rule.name, reasons)

    a = policy.auto_send
    checks: list[tuple[bool, str]] = [
        (triage.queue in a.queues, f"queue {triage.queue} in auto-send list"),
        (
            PRIORITY_ORDER[triage.priority] <= PRIORITY_ORDER[a.max_priority],
            f"priority {triage.priority} <= {a.max_priority}",
        ),
        (
            triage.queue_confidence >= a.min_queue_confidence,
            f"queue confidence {triage.queue_confidence:.0%} >= {a.min_queue_confidence:.0%}",
        ),
        (
            draft.grounded and draft.kb_score >= a.min_kb_score,
            f"grounded draft with kb score {draft.kb_score:.2f} >= {a.min_kb_score}",
        ),
        (not (signals & set(a.forbidden_signals)), f"no forbidden signals (found {sorted(signals)})"),
        (draft.redactions == 0, "nothing had to be redacted"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    if not failed:
        return Decision("auto_send", "auto_send", [msg for _, msg in checks])
    return Decision("review", "auto_send_failed", [f"not: {m}" for m in failed])
