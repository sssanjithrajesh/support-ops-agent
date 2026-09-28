"""Eval-set integrity, scoring math, the gate file, and the MCP surface."""

from __future__ import annotations

import asyncio
import json

from evals.run import GATE_PATH, build_agent, run_cases
from evals.schema import load_cases
from evals.scoring import CaseResult, aggregate
from soa import mcp_server
from soa.db import make_session_factory
from soa.service import Service
from soa.zendesk import Zendesk, ZendeskConfig


def test_eval_set_integrity(kb):
    cases = load_cases()
    assert len(cases) == 30
    assert len({c.id for c in cases}) == 30
    assert sum(c.source == "dataset" for c in cases) == 23
    article_ids = {a.id for a in kb.articles}
    for c in cases:
        if c.expected.kb_article:
            assert c.expected.kb_article in article_ids, c.id
        if c.expected.action == "escalate":
            assert c.expected.must_not_auto_send, f"{c.id}: escalations must be guarded"


def test_gate_keys_match_metrics_and_current_run_passes():
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    card = run_cases(build_agent(), load_cases())
    overall = card.overall.as_dict()
    for key in gate["hand_labelled"]:
        assert key in overall, key
    assert overall["unsafe_autosend_rate"] == 0.0
    assert overall["escalation_recall"] >= gate["hand_labelled"]["escalation_recall"]


def test_scoring_math():
    def r(exp, pred, guard=False, kb=("KB-001", "KB-001")):
        return CaseResult(
            case_id="c", source="x", expected_queue="A", predicted_queue="A", expected_priority="low",
            predicted_priority="high", expected_action=exp, predicted_action=pred, must_not_auto_send=guard,
            expected_signals=None, predicted_signals=[], expected_kb=kb[0], predicted_kb=kb[1], grounded=True,
            decision_rule="r",
        )  # fmt: skip

    m = aggregate(
        [
            r("escalate", "escalate", True),
            r("escalate", "review", True),
            r("review", "auto_send", True),
            r("auto_send", "auto_send"),
        ]
    )
    assert m.n == 4
    assert m.queue_accuracy == 1.0 and m.priority_accuracy == 0.0
    assert m.action_accuracy == 0.5
    assert m.escalation_recall == 0.5 and m.escalation_precision == 1.0
    assert m.unsafe_autosend_rate == 1 / 3
    assert m.kb_top1 == 1.0


def test_mcp_tools_share_the_policy(tmp_path):
    factory = make_session_factory(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    cfg = ZendeskConfig("http://nowhere", "a@example.com", "t", "s")
    service = Service(session_factory=factory, zd=Zendesk(cfg), slack_client=None)
    server = mcp_server.build(service)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"get_ticket", "triage_ticket", "search_kb", "send_public_reply", "escalate", "review_queue"} <= names

    def call(tool, **args):
        result = asyncio.run(server.call_tool(tool, args))
        assert not result.is_error, result
        return json.loads(result.content[0].text)

    out = call("triage_ticket", ticket_id="m1", subject="Hacked", body="Someone hacked our account and stole data")
    assert "security" in {s["name"] for s in out["signals"]}
    held = call(
        "send_public_reply",
        ticket_id="m1",
        subject="Hacked",
        body_of_ticket="Someone hacked our account and stole data",
        reply="No problem, all fine.",
    )
    assert held["status"] == "held_for_review"
    q = call("review_queue")
    assert q["pending"] and q["pending"][0]["ticket_id"] == "m1"
