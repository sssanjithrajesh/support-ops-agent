"""Run the agent over the eval set, score it, write reports, and optionally enforce the gate.

    uv run soa-eval                      # 30 hand-labelled cases + 500 held-out, write docs/scorecard.md and svg
    uv run soa-eval --gate               # exit 1 if any metric fell below evals/gate.json
    uv run soa-eval --planner llm        # needs ANTHROPIC_API_KEY and `uv sync --extra llm`
    uv run soa-eval --show-misses        # print every case where action or queue was wrong

The gate is a floor recorded from a real run, never edited upward to make CI pass. If the number
gets worse, CI fails and the person changing the policy explains why in the PR.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from evals.report import print_terminal, write_reports
from evals.schema import EvalCase, load_cases
from evals.scoring import HeldoutMetrics, Scorecard, build_scorecard, score_case
from soa.agent import Agent, RecordingSink, choose_planner
from soa.config import ROOT, policy
from soa.kb import KnowledgeBase
from soa.tickets import heldout_tickets, training_tickets
from soa.triage import Triager

GATE_PATH = ROOT / "evals" / "gate.json"
RESULTS_PATH = ROOT / "evals" / "results" / "latest.json"
SVG_PATH = ROOT / "docs" / "img" / "scorecard.svg"
MD_PATH = ROOT / "docs" / "scorecard.md"

# Higher is better for these; lower is better for the rest.
HIGHER_IS_BETTER = {"queue_accuracy", "priority_accuracy", "action_accuracy", "escalation_recall", "kb_top1"}
LOWER_IS_BETTER = {"unsafe_autosend_rate"}


def build_agent(planner_name: str | None = None) -> Agent:
    p = policy()
    return Agent(
        p, Triager(p, training_tickets()), KnowledgeBase.load(p.kb_dir), RecordingSink(), choose_planner(planner_name)
    )


def run_cases(agent: Agent, cases: list[EvalCase]) -> Scorecard:
    results = []
    for case in cases:
        run = agent.run(case.ticket)
        results.append(score_case(case, run.triage, run.draft, run.decision))
    return build_scorecard(agent.planner.name, results)


def run_heldout(agent: Agent) -> HeldoutMetrics:
    tickets = heldout_tickets()
    queue_hits = prio_hits = 0
    actions: Counter = Counter()
    for t in tickets:
        run = agent.run(t)
        queue_hits += run.triage.queue == t.queue
        prio_hits += run.triage.priority == t.priority
        actions[run.decision.action] += 1
    majority = Counter(t.queue for t in tickets).most_common(1)[0][1] / len(tickets)
    return HeldoutMetrics(
        n=len(tickets),
        queue_accuracy=queue_hits / len(tickets),
        priority_accuracy=prio_hits / len(tickets),
        majority_queue_baseline=majority,
        actions=dict(actions),
    )


def check_gate(card: Scorecard, heldout: HeldoutMetrics, gate_path: Path = GATE_PATH) -> list[str]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    failures = []
    overall = card.overall.as_dict()
    for key, floor in gate["hand_labelled"].items():
        if key == "n":
            if overall["n"] < floor:
                failures.append(f"hand_labelled n {overall['n']} < {floor}")
            continue
        value = overall[key]
        if key in LOWER_IS_BETTER and value > floor:
            failures.append(f"{key} {value:.4f} > gate {floor:.4f}")
        elif key in HIGHER_IS_BETTER and value < floor:
            failures.append(f"{key} {value:.4f} < gate {floor:.4f}")
    h = heldout.as_dict()
    for key, floor in gate["heldout"].items():
        if key == "n" and h["n"] != floor:
            failures.append(f"heldout n {h['n']} != {floor}")
        elif key != "n" and h[key] < floor:
            failures.append(f"heldout {key} {h[key]:.4f} < gate {floor:.4f}")
    return failures


def show_misses(card: Scorecard) -> None:
    for r in card.results:
        if not (r.action_correct and r.queue_correct):
            print(
                f"{r.case_id}: action {r.expected_action}->{r.predicted_action} ({r.decision_rule}); "
                f"queue {r.expected_queue}->{r.predicted_queue}; "
                f"priority {r.expected_priority}->{r.predicted_priority}; "
                f"signals {r.predicted_signals}; kb {r.expected_kb}->{r.predicted_kb}"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="deterministic")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args(argv)

    agent = build_agent(args.planner)
    card = run_cases(agent, load_cases())
    heldout = run_heldout(agent)
    print_terminal(card, heldout)
    if args.show_misses:
        show_misses(card)

    if not args.no_report:
        write_reports(card, heldout, SVG_PATH, MD_PATH)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **card.as_dict(),
            "heldout": heldout.as_dict(),
            "cases": [
                {
                    "id": r.case_id,
                    "source": r.source,
                    "expected": {
                        "queue": r.expected_queue,
                        "priority": r.expected_priority,
                        "action": r.expected_action,
                    },
                    "predicted": {
                        "queue": r.predicted_queue,
                        "priority": r.predicted_priority,
                        "action": r.predicted_action,
                        "rule": r.decision_rule,
                        "signals": r.predicted_signals,
                        "kb": r.predicted_kb,
                    },
                }
                for r in card.results
            ],
        }
        RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {MD_PATH.relative_to(ROOT)}, {SVG_PATH.relative_to(ROOT)}, {RESULTS_PATH.relative_to(ROOT)}")

    if args.gate:
        failures = check_gate(card, heldout)
        if failures:
            print("EVAL GATE FAILED:\n  " + "\n  ".join(failures))
            return 1
        print("eval gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
