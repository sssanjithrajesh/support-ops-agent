"""MCP server for the ticketing system, so any MCP client (Claude Desktop, Cursor, another agent) can work tickets.

    uv run soa-mcp        # stdio transport

Read tools return facts; write tools go through the same policy and outbox as the built-in planner.
An MCP client cannot send a customer reply the policy would have held for review: send_public_reply
turns into a review item exactly as it does for the deterministic planner.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from soa.agent import Tools
from soa.http import OfflineError, PermanentError, RetryableError
from soa.service import OutboxSink, Service
from soa.tickets import Ticket


def _safe(fn):
    def inner(*a, **kw):
        try:
            return fn(*a, **kw)
        except OfflineError as exc:
            return {"status": "unavailable", "reason": "offline mode", "detail": str(exc), "retry": False}
        except RetryableError as exc:
            return {
                "status": "unavailable",
                "reason": "upstream rate limited or down",
                "detail": str(exc),
                "retry": True,
            }
        except PermanentError as exc:
            return {"status": "error", "reason": "upstream rejected the request", "detail": str(exc), "retry": False}

    inner.__name__ = fn.__name__
    inner.__doc__ = fn.__doc__
    return inner


def build(service: Service | None = None) -> MCPServer:
    service = service or Service()
    server = MCPServer(
        "support-ops-tickets",
        instructions=(
            "Ticketing tools for support operations. Triage first, search the knowledge base, draft, then decide. "
            "Customer-facing replies are sent only when policy allows; otherwise they wait for a human."
        ),
    )

    def tools_for(session) -> Tools:
        return Tools(service.policy, service.triager, service.kb, OutboxSink(session))

    @server.tool(description="Fetch a ticket from the ticketing system by id.")
    @_safe
    def get_ticket(ticket_id: str) -> dict:
        t = service.zd.get_ticket(ticket_id)
        return {
            "id": str(t["id"]),
            "subject": t.get("subject"),
            "body": t.get("description"),
            "priority": t.get("priority"),
            "tags": t.get("tags", []),
            "status": t.get("status"),
        }

    @server.tool(description="List open tickets in the ticketing system (id, subject, priority, tags).")
    @_safe
    def list_open_tickets() -> list[dict]:
        return [
            {"id": str(t["id"]), "subject": t.get("subject"), "priority": t.get("priority"), "tags": t.get("tags", [])}
            for t in service.zd.list_tickets("open")
        ]

    @server.tool(description="Classify a ticket: queue, priority, signals, and the trace explaining each.")
    def triage_ticket(ticket_id: str, subject: str, body: str) -> dict:
        with service.sessions() as s:
            return tools_for(s).triage_ticket(Ticket(id=ticket_id, subject=subject, body=body))

    @server.tool(description="Search the knowledge base. Returns article ids, titles, scores and overlapping terms.")
    def search_kb(query: str, queue: str | None = None) -> list[dict]:
        with service.sessions() as s:
            return tools_for(s).search_kb(query, queue=queue)

    @server.tool(description="Read one knowledge base article in full.")
    def get_kb_article(article_id: str) -> dict:
        a = service.kb.get(article_id)
        if a is None:
            return {"status": "error", "reason": f"no article {article_id}"}
        return {"id": a.id, "title": a.title, "queues": a.queues, "body": a.body}

    @server.tool(
        description="Run the whole agent on a ticket: triage, draft, decide, queue writes. Returns decision and trace."
    )
    def process_ticket(ticket_id: str, subject: str, body: str, requester: str = "customer@example.com") -> dict:
        run = service.process_ticket(Ticket(id=ticket_id, subject=subject, body=body, requester=requester))
        return {
            "action": run.decision.action,
            "rule": run.decision.rule,
            "queue": run.triage.queue,
            "priority": run.triage.priority,
            "trace": run.triage.trace,
            "draft": run.draft.body,
        }

    @server.tool(description="Add an internal (agent-only) note to a ticket. Idempotent per ticket and note kind.")
    def add_internal_note(ticket_id: str, text: str, note_kind: str = "manual") -> dict:
        with service.sessions() as s:
            out = tools_for(s).add_internal_note(ticket_id, text, note_kind)
            s.commit()
        return out

    @server.tool(description="Set queue tags and priority on a ticket. Idempotent.")
    def set_ticket_fields(ticket_id: str, tags: list[str], priority: str) -> dict:
        with service.sessions() as s:
            out = tools_for(s).set_ticket_fields(ticket_id, tags, priority)
            s.commit()
        return out

    @server.tool(
        description=(
            "Propose a customer reply. Sent only if policy decides auto_send for this ticket; "
            "otherwise held for human review."
        )
    )
    def send_public_reply(
        ticket_id: str, subject: str, body_of_ticket: str, reply: str, article_id: str | None = None
    ) -> dict:
        with service.sessions() as s:
            tools = tools_for(s)
            ticket = Ticket(id=ticket_id, subject=subject, body=body_of_ticket)
            tools.triage_ticket(ticket)
            tools.search_kb(ticket.text, ticket_id=ticket_id)
            tools.draft_reply(ticket, article_id)
            tools.decide_action(ticket_id)
            out = tools.send_public_reply(ticket_id, reply, article_id)
            s.commit()
        return out

    @server.tool(
        description="Escalate a ticket to the on-call Slack channel with a one-line summary. Idempotent per ticket."
    )
    def escalate(ticket_id: str, summary: str) -> dict:
        with service.sessions() as s:
            out = tools_for(s).escalate(ticket_id, summary)
            s.commit()
        return out

    @server.tool(description="Show drafts waiting for a human, and outbox delivery status.")
    def review_queue() -> dict:
        q = service.queue()
        return {
            "pending": [
                {"item_id": i.id, "ticket_id": i.ticket_id, "reason": i.reason, "article_id": i.article_id}
                for i, _ in q["pending"]
            ],
            "actions": q["actions"],
            "outbox": q["outbox"],
        }

    @server.tool(
        description="Deliver queued writes to the ticketing system and Slack now (normally a worker does this)."
    )
    def drain_outbox() -> dict:
        return service.drain_outbox()

    return server


def main() -> None:
    build().run(transport="stdio")
