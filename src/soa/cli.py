"""Triage the committed sample offline and report what the agent would do.

    uv run soa-triage             # all 2,000 tickets; prints a summary, writes docs/latest-run.json, docs/img/*.svg
    uv run soa-triage --heldout   # only the 500 held-out tickets
    uv run soa-triage --no-charts

Nothing is written to any ticketing system; writes go to an in-memory recording sink.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from rich.console import Console
from rich.table import Table

from soa.agent import Agent, RecordingSink
from soa.config import ROOT, policy
from soa.kb import KnowledgeBase
from soa.report import actions_by_queue, signal_bars
from soa.tickets import heldout_tickets, load_sample, training_tickets
from soa.triage import Triager

OUT_JSON = ROOT / "docs" / "latest-run.json"
IMG = ROOT / "docs" / "img"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args(argv)

    p = policy()
    agent = Agent(p, Triager(p, training_tickets()), KnowledgeBase.load(p.kb_dir), RecordingSink())
    tickets = heldout_tickets() if args.heldout else load_sample()

    by_queue: dict[str, Counter] = defaultdict(Counter)
    actions: Counter = Counter()
    rules: Counter = Counter()
    signals: Counter = Counter()
    signal_escalated: Counter = Counter()
    grounded = redactions = 0
    for t in tickets:
        run = agent.run(t)
        a = run.decision.action
        actions[a] += 1
        rules[run.decision.rule] += 1
        by_queue[run.triage.queue][a] += 1
        for s in run.triage.signal_names:
            signals[s] += 1
            if a == "escalate":
                signal_escalated[s] += 1
        grounded += run.draft.grounded
        redactions += run.draft.redactions

    console = Console()
    table = Table(title=f"What the agent would do with {len(tickets)} tickets")
    table.add_column("queue")
    for a in ("escalate", "review", "auto_send", "total"):
        table.add_column(a, justify="right")
    rows = sorted(by_queue.items(), key=lambda kv: -sum(kv[1].values()))
    for q, c in rows:
        table.add_row(q, str(c["escalate"]), str(c["review"]), str(c["auto_send"]), str(sum(c.values())))
    table.add_section()
    table.add_row(
        "ALL",
        str(actions["escalate"]),
        str(actions["review"]),
        str(actions["auto_send"]),
        str(len(tickets)),
        style="bold",
    )
    console.print(table)
    console.print(
        "escalation rules: "
        + ", ".join(f"{k} {v}" for k, v in rules.most_common() if k not in ("auto_send", "auto_send_failed"))
    )
    console.print(
        "signals (escalated/fired): " + ", ".join(f"{k} {signal_escalated[k]}/{v}" for k, v in signals.most_common())
    )
    console.print(f"grounded drafts: {grounded}/{len(tickets)}; numbers redacted: {redactions}")

    summary = {
        "tickets": len(tickets),
        "actions": dict(actions),
        "rules": dict(rules.most_common()),
        "signals": {k: {"fired": v, "escalated": signal_escalated[k]} for k, v in signals.most_common()},
        "by_queue": {q: dict(c) for q, c in rows},
        "grounded_drafts": grounded,
        "redactions": redactions,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.no_charts:
        IMG.mkdir(parents=True, exist_ok=True)
        (IMG / "actions-by-queue.svg").write_text(actions_by_queue([(q, dict(c)) for q, c in rows]), encoding="utf-8")
        (IMG / "signals.svg").write_text(
            signal_bars([(k, v, signal_escalated[k]) for k, v in signals.most_common()]), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
