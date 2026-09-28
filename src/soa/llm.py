"""LLM planner: Claude chooses which tools to call, with tool-use, against the same Tools surface.

Behind ANTHROPIC_API_KEY and `uv sync --extra llm`. Not exercised in CI (no key there). The unit test
drives the loop with a fake client to prove that a planner that skips decide_action, or asks to send a
reply the policy would hold, still cannot reach the customer: the tools enforce policy, not the prompt.
"""

from __future__ import annotations

import json
import os

from soa.agent import AgentRun, Tools
from soa.decision import Decision
from soa.drafting import Draft
from soa.tickets import Ticket

MODEL = os.environ.get("SOA_LLM_MODEL", "claude-sonnet-4-5")
MAX_STEPS = 10

SYSTEM = """You are a support operations agent. For the ticket you are given:
1. call triage_ticket, 2. call search_kb with the ticket text and the predicted queue,
3. call draft_reply with the best article id (or null if none applies), 4. call decide_action,
5. call set_ticket_fields and add_internal_note, 6. if the decision is escalate call escalate,
otherwise call send_public_reply (it will be held for review if policy requires).
Never invent facts that are not in a knowledge base article. Stop when the writes are done."""

TOOL_SCHEMAS = [
    {
        "name": "triage_ticket",
        "description": "Classify the ticket.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_kb",
        "description": "Search the knowledge base.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "queue": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "draft_reply",
        "description": "Draft a grounded reply from an article.",
        "input_schema": {"type": "object", "properties": {"article_id": {"type": ["string", "null"]}}},
    },
    {
        "name": "decide_action",
        "description": "Apply policy: escalate, review or auto_send.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_ticket_fields",
        "description": "Set tags and priority.",
        "input_schema": {
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}, "priority": {"type": "string"}},
            "required": ["tags", "priority"],
        },
    },
    {
        "name": "add_internal_note",
        "description": "Add an internal note.",
        "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
    {
        "name": "send_public_reply",
        "description": "Send or hold the customer reply.",
        "input_schema": {
            "type": "object",
            "properties": {"body": {"type": "string"}, "article_id": {"type": ["string", "null"]}},
            "required": ["body"],
        },
    },
    {
        "name": "escalate",
        "description": "Escalate to Slack.",
        "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
    },
]


def dispatch(tools: Tools, ticket: Ticket, name: str, args: dict) -> dict | list:
    if name == "triage_ticket":
        return tools.triage_ticket(ticket)
    if name == "search_kb":
        return tools.search_kb(args.get("query") or ticket.text, args.get("queue"), ticket_id=ticket.id)
    if name == "draft_reply":
        return tools.draft_reply(ticket, args.get("article_id"))
    if name == "decide_action":
        return tools.decide_action(ticket.id)
    if name == "set_ticket_fields":
        return tools.set_ticket_fields(ticket.id, list(args.get("tags", [])), args.get("priority", "medium"))
    if name == "add_internal_note":
        return tools.add_internal_note(ticket.id, args.get("text", ""), "llm")
    if name == "send_public_reply":
        return tools.send_public_reply(ticket.id, args.get("body", ""), args.get("article_id"))
    if name == "escalate":
        return tools.escalate(ticket.id, args.get("summary", ""))
    return {"error": f"unknown tool {name}"}


class LLMPlanner:
    name = "llm"

    def __init__(self, client=None) -> None:
        if client is None:
            import anthropic  # optional dependency

            client = anthropic.Anthropic()
        self.client = client

    def run(self, tools: Tools, ticket: Ticket) -> AgentRun:
        tools.calls.clear()
        messages = [{"role": "user", "content": f"Ticket {ticket.id}\nSubject: {ticket.subject}\n\n{ticket.body}"}]
        for _ in range(MAX_STEPS):
            resp = self.client.messages.create(
                model=MODEL, max_tokens=1500, system=SYSTEM, tools=TOOL_SCHEMAS, messages=messages
            )
            uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if not uses:
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for u in uses:
                out = dispatch(tools, ticket, u.name, dict(u.input or {}))
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": json.dumps(out)[:6000]})
            messages.append({"role": "user", "content": results})
            if resp.stop_reason == "end_turn":
                break
        return self._finish(tools, ticket)

    def _finish(self, tools: Tools, ticket: Ticket) -> AgentRun:
        """Whatever the model did, the record is built from the tools' state, never from the model's prose."""
        triage = tools._triage.get(ticket.id) or tools.triager.triage(ticket)
        hits = getattr(tools, "_last_hits", [])
        draft = tools._draft.get(ticket.id) or Draft(ticket.id, None, "", grounded=False, kb_score=0.0)
        decision = tools._decision.get(ticket.id) or Decision(
            "review", "planner_did_not_decide", ["LLM planner ended without decide_action"]
        )
        if ticket.id not in tools._decision:
            tools.sink.write(
                "review_item",
                ticket.id,
                f"{ticket.id}:review",
                {"body": draft.body, "article_id": draft.article_id, "reason": decision.rule},
            )
        return AgentRun(ticket, self.name, triage, hits, draft, decision, list(tools.calls))
