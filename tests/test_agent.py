"""Planner behaviour and, more importantly, what the tools refuse to do."""

from __future__ import annotations

from types import SimpleNamespace

from soa.agent import Agent, DeterministicPlanner, RecordingSink, Tools
from soa.llm import LLMPlanner
from soa.tickets import Ticket

OUTAGE = Ticket("t1", "Everything is down", "Our whole company gets 503 errors since 9am. Is there an outage?")
PRESALES = Ticket("t2", "Demo and pricing", "Could we get a demo and a quote for about 80 seats?", "j.t@example.com")


def test_deterministic_planner_records_every_step(pol, triager, kb):
    sink = RecordingSink()
    run = Agent(pol, triager, kb, sink).run(OUTAGE)
    names = [c.name for c in run.tool_calls]
    assert names[:4] == ["triage_ticket", "search_kb", "draft_reply", "decide_action"]
    assert run.decision.action == "escalate"
    assert names[-1] == "escalate"
    kinds = [w["kind"] for w in sink.writes]
    assert kinds == ["set_fields", "internal_note", "escalation"]
    assert "public_reply" not in kinds


def test_auto_send_path_writes_a_public_reply_once(pol, triager, kb):
    sink = RecordingSink()
    agent = Agent(pol, triager, kb, sink)
    run = agent.run(PRESALES)
    assert run.decision.action == "auto_send"
    assert [w["kind"] for w in sink.writes] == ["set_fields", "internal_note", "public_reply"]
    agent.run(PRESALES)  # second run: same keys, nothing new lands
    assert len(sink.writes) == 3


def test_tools_hold_a_reply_the_policy_did_not_approve(pol, triager, kb):
    sink = RecordingSink()
    tools = Tools(pol, triager, kb, sink)
    # A planner that skips decide_action and tries to send anyway.
    tools.triage_ticket(OUTAGE)
    out = tools.send_public_reply(OUTAGE.id, "All good, ignore it.", None)
    assert out["status"] == "held_for_review"
    assert sink.writes[-1]["kind"] == "review_item"
    # And one that did decide, but policy said escalate.
    tools.search_kb(OUTAGE.text, ticket_id=OUTAGE.id)
    tools.draft_reply(OUTAGE, "KB-008")
    tools.decide_action(OUTAGE.id)
    out = tools.send_public_reply(OUTAGE.id, "All good, ignore it.", "KB-008")
    assert out["status"] == "held_for_review" and "escalate" in out["reason"]


class _FakeAnthropic:
    """Scripted tool-use responses: each item is a list of (tool, input) calls for one turn."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        if not self.turns:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="done")], stop_reason="end_turn")
        calls = self.turns.pop(0)
        blocks = [SimpleNamespace(type="tool_use", id=f"u{i}", name=n, input=a) for i, (n, a) in enumerate(calls)]
        return SimpleNamespace(content=blocks, stop_reason="tool_use")


def test_llm_planner_cannot_bypass_policy(pol, triager, kb):
    sink = RecordingSink()
    fake = _FakeAnthropic([[("send_public_reply", {"body": "Sure, refund granted.", "article_id": None})]])
    run = Agent(pol, triager, kb, sink, LLMPlanner(client=fake)).run(OUTAGE)
    assert run.planner == "llm"
    assert run.decision.rule == "planner_did_not_decide"
    assert all(w["kind"] != "public_reply" for w in sink.writes)
    assert any(w["kind"] == "review_item" for w in sink.writes)


def test_llm_planner_following_the_protocol_matches_deterministic(pol, triager, kb):
    script = [
        [("triage_ticket", {})],
        [("search_kb", {"query": PRESALES.text, "queue": "Sales and Pre-Sales"})],
        [("draft_reply", {"article_id": "KB-009"})],
        [("decide_action", {})],
        [("set_ticket_fields", {"tags": ["q_sales"], "priority": "low"}), ("add_internal_note", {"text": "triaged"})],
        [("send_public_reply", {"body": "x", "article_id": "KB-009"})],
    ]
    sink = RecordingSink()
    run = Agent(pol, triager, kb, sink, LLMPlanner(client=_FakeAnthropic(script))).run(PRESALES)
    det = Agent(pol, triager, kb, RecordingSink(), DeterministicPlanner()).run(PRESALES)
    assert run.decision.action == det.decision.action == "auto_send"
    assert [w["kind"] for w in sink.writes][-1] == "public_reply"
