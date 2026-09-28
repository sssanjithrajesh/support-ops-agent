"""The agent: a small set of tools and a planner that calls them.

Two planners share one tool surface:

- DeterministicPlanner: triage -> search_kb -> draft_reply -> decide -> write. Runs offline, in the
  evals, in CI, and in production when no LLM key is present. Every step is recorded as a ToolCall.
- LLMPlanner (soa/llm.py): Claude picks tools with tool-use, behind ANTHROPIC_API_KEY.

The important property is that policy is enforced in the tools, not in the planner. A planner that
calls send_public_reply on a ticket the policy says must be reviewed gets a review item, not a send.
So the human review path holds even if the planner is wrong, prompted badly, or malicious.

Writes go through a WriteSink so the same agent code runs against the outbox (production), a
recording list (evals, tests), or nothing at all.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from soa.config import Policy
from soa.decision import Decision, decide
from soa.drafting import Draft, draft_from_article, redact
from soa.kb import Hit, KnowledgeBase
from soa.tickets import Ticket
from soa.triage import Triager, TriageResult


class WriteSink(Protocol):
    def write(self, kind: str, ticket_id: str, key: str, payload: dict) -> str:
        """Record an intended write. Returns a status string: queued | duplicate | recorded."""


class RecordingSink:
    def __init__(self) -> None:
        self.writes: list[dict] = []
        self._keys: set[str] = set()

    def write(self, kind: str, ticket_id: str, key: str, payload: dict) -> str:
        if key in self._keys:
            return "duplicate"
        self._keys.add(key)
        self.writes.append({"kind": kind, "ticket_id": ticket_id, "key": key, "payload": payload})
        return "recorded"


@dataclass
class ToolCall:
    name: str
    args: dict
    result: dict | list

    def as_dict(self) -> dict:
        return {"name": self.name, "args": self.args, "result": self.result}


@dataclass
class AgentRun:
    ticket: Ticket
    planner: str
    triage: TriageResult
    hits: list[Hit]
    draft: Draft
    decision: Decision
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ticket_id": self.ticket.id,
            "planner": self.planner,
            "triage": self.triage.as_dict(),
            "kb_hits": [h.as_dict() for h in self.hits],
            "draft": self.draft.as_dict(),
            "decision": self.decision.as_dict(),
            "tool_calls": [c.as_dict() for c in self.tool_calls],
        }


class Tools:
    """The tool surface. Each public method is one tool; the MCP server exposes the same ones."""

    def __init__(self, policy: Policy, triager: Triager, kb: KnowledgeBase, sink: WriteSink) -> None:
        self.policy = policy
        self.triager = triager
        self.kb = kb
        self.sink = sink
        self.calls: list[ToolCall] = []
        self._triage: dict[str, TriageResult] = {}
        self._draft: dict[str, Draft] = {}
        self._decision: dict[str, Decision] = {}

    def _record(self, name: str, args: dict, result: dict | list) -> dict | list:
        self.calls.append(ToolCall(name, args, result))
        return result

    # -- read tools -------------------------------------------------------------------------------

    def triage_ticket(self, ticket: Ticket) -> dict:
        t = self.triager.triage(ticket)
        self._triage[ticket.id] = t
        return self._record("triage_ticket", {"ticket_id": ticket.id}, t.as_dict())

    def search_kb(self, query: str, queue: str | None = None, ticket_id: str = "") -> list[dict]:
        hits = self.kb.search(query, top_k=self.policy.kb_top_k, queue=queue)
        self._last_hits = hits
        return self._record("search_kb", {"ticket_id": ticket_id, "queue": queue}, [h.as_dict() for h in hits])

    def draft_reply(self, ticket: Ticket, article_id: str | None) -> dict:
        hit = None
        if article_id:
            article = self.kb.get(article_id)
            if article is not None:
                hit = next((h for h in getattr(self, "_last_hits", []) if h.article.id == article_id), None)
                if hit is None:
                    hit = Hit(article, score=0.0)
        d = draft_from_article(ticket, hit)
        self._draft[ticket.id] = d
        return self._record("draft_reply", {"ticket_id": ticket.id, "article_id": article_id}, d.as_dict())

    def decide_action(self, ticket_id: str) -> dict:
        t, d = self._triage.get(ticket_id), self._draft.get(ticket_id)
        if t is None or d is None:
            result = {"error": "triage_ticket and draft_reply must run before decide_action"}
            return self._record("decide_action", {"ticket_id": ticket_id}, result)
        decision = decide(self.policy, t, d)
        self._decision[ticket_id] = decision
        return self._record("decide_action", {"ticket_id": ticket_id}, decision.as_dict())

    # -- write tools (idempotent by key; policy-enforced) -------------------------------------------

    def set_ticket_fields(self, ticket_id: str, tags: list[str], priority: str) -> dict:
        key = f"{ticket_id}:fields:{','.join(sorted(tags))}:{priority}"
        status = self.sink.write("set_fields", ticket_id, key, {"tags": tags, "priority": priority})
        return self._record(
            "set_ticket_fields", {"ticket_id": ticket_id, "tags": tags, "priority": priority}, {"status": status}
        )

    def add_internal_note(self, ticket_id: str, text: str, note_kind: str = "triage") -> dict:
        text, n = redact(text)
        key = f"{ticket_id}:note:{note_kind}"
        status = self.sink.write("internal_note", ticket_id, key, {"text": text})
        return self._record(
            "add_internal_note", {"ticket_id": ticket_id, "kind": note_kind}, {"status": status, "redactions": n}
        )

    def send_public_reply(self, ticket_id: str, body: str, article_id: str | None) -> dict:
        """Send only if the policy decided auto_send for this ticket; otherwise create a review item."""
        decision = self._decision.get(ticket_id)
        body, n = redact(body)
        if decision is None or decision.action != "auto_send":
            why = "no decision recorded" if decision is None else f"policy decided {decision.action} ({decision.rule})"
            status = self.sink.write(
                "review_item", ticket_id, f"{ticket_id}:review", {"body": body, "article_id": article_id, "reason": why}
            )
            return self._record(
                "send_public_reply",
                {"ticket_id": ticket_id, "article_id": article_id},
                {"status": "held_for_review", "reason": why, "sink": status},
            )
        status = self.sink.write(
            "public_reply", ticket_id, f"{ticket_id}:reply", {"body": body, "article_id": article_id}
        )
        return self._record(
            "send_public_reply", {"ticket_id": ticket_id, "article_id": article_id}, {"status": status, "redactions": n}
        )

    def escalate(self, ticket_id: str, summary: str) -> dict:
        summary, _ = redact(summary)
        status = self.sink.write("escalation", ticket_id, f"{ticket_id}:escalate", {"summary": summary})
        return self._record(
            "escalate", {"ticket_id": ticket_id}, {"status": status, "channel": self.policy.slack_channel_hint}
        )


def _note_text(ticket: Ticket, t: TriageResult, hits: list[Hit], d: Draft, decision: Decision) -> str:
    lines = [f"[support-ops-agent] {decision.action.upper()} by rule {decision.rule}", ""]
    lines += ["Triage:"] + [f"  - {x}" for x in t.trace]
    if hits:
        lines += ["Knowledge base:"] + [f"  - {h.article.id} {h.article.title} (score {h.score:.2f})" for h in hits]
    lines += ["Decision:"] + [f"  - {r}" for r in decision.reasons]
    if d.redactions:
        lines.append(f"Redacted {d.redactions} number(s) from the subject before drafting.")
    return "\n".join(lines)


class DeterministicPlanner:
    name = "deterministic"

    def run(self, tools: Tools, ticket: Ticket) -> AgentRun:
        tools.calls.clear()
        t_dict = tools.triage_ticket(ticket)
        triage = tools._triage[ticket.id]
        hit_dicts = tools.search_kb(ticket.text, queue=t_dict["queue"], ticket_id=ticket.id)
        hits = tools._last_hits
        top = hit_dicts[0]["id"] if hit_dicts else None
        tools.draft_reply(ticket, top)
        draft = tools._draft[ticket.id]
        tools.decide_action(ticket.id)
        decision = tools._decision[ticket.id]

        queue = tools.policy.queue_by_name(triage.queue)
        tags = [queue.tag, "soa_triaged"] + [f"sig_{s}" for s in sorted(triage.signal_names)]
        tools.set_ticket_fields(ticket.id, tags, triage.priority)
        tools.add_internal_note(ticket.id, _note_text(ticket, triage, hits, draft, decision))
        if decision.action == "escalate":
            summary = (
                f"{ticket.id} {ticket.subject} | {triage.queue}, {triage.priority} | rule {decision.rule} | "
                f"signals {sorted(triage.signal_names)}"
            )
            tools.escalate(ticket.id, summary)
        else:
            tools.send_public_reply(ticket.id, draft.body, draft.article_id)
        return AgentRun(ticket, self.name, triage, hits, draft, decision, list(tools.calls))


class Agent:
    def __init__(self, policy: Policy, triager: Triager, kb: KnowledgeBase, sink: WriteSink, planner=None) -> None:
        self.tools = Tools(policy, triager, kb, sink)
        self.planner = planner or DeterministicPlanner()

    def run(self, ticket: Ticket) -> AgentRun:
        return self.planner.run(self.tools, ticket)


PlannerFactory = Callable[[], object]


def choose_planner(name: str | None = None):
    """'deterministic' (default) or 'llm' (requires ANTHROPIC_API_KEY and the llm extra)."""
    if name in (None, "", "deterministic"):
        return DeterministicPlanner()
    if name == "llm":
        from soa.llm import LLMPlanner

        return LLMPlanner()
    raise ValueError(f"unknown planner {name!r}")


def dumps(run: AgentRun) -> str:
    return json.dumps(run.as_dict(), indent=2)
