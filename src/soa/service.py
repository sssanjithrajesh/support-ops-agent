"""Orchestration: webhook in -> agent -> rows out; human review decisions; outbox drain.

Everything the web app and the MCP server do goes through here, so the same behaviour is testable
without HTTP. The Service owns one Agent (deterministic unless ANTHROPIC_API_KEY is set and the
planner is requested), one Zendesk client, an optional Slack client, and a session factory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from soa import outbox, slack, zendesk
from soa.agent import Agent, AgentRun, WriteSink, choose_planner
from soa.config import Policy, policy
from soa.db import AgentRunRow, ReviewItem, TicketRow, WebhookEvent, make_session_factory, now
from soa.kb import KnowledgeBase
from soa.tickets import Ticket, training_tickets
from soa.triage import Triager
from soa.zendesk import SIG_HEADER, TS_HEADER, Zendesk, ZendeskConfig, verify_webhook


class OutboxSink(WriteSink):
    """Turns agent writes into outbox rows (remote) or review items (local), in the caller's session."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def write(self, kind: str, ticket_id: str, key: str, payload: dict) -> str:
        if kind == "review_item":
            if self.session.scalar(select(ReviewItem).where(ReviewItem.key == key)) is not None:
                return "duplicate"
            self.session.add(
                ReviewItem(
                    ticket_id=ticket_id,
                    key=key,
                    draft=payload["body"],
                    article_id=payload.get("article_id") or "",
                    reason=payload.get("reason", ""),
                )
            )
            return "queued"
        target = "slack" if kind == "escalation" else "zendesk"
        item = outbox.enqueue(self.session, target, kind, ticket_id, key, {"ticket_id": ticket_id, **payload})
        return "queued" if item else "duplicate"


@dataclass
class WebhookResult:
    outcome: str  # processed | duplicate | rejected
    detail: str
    ticket_id: str = ""
    action: str = ""


class Service:
    def __init__(
        self,
        session_factory: sessionmaker | None = None,
        zd: Zendesk | None = None,
        slack_client: slack.Slack | None = None,
        planner: str | None = None,
        pol: Policy | None = None,
    ) -> None:
        self.policy = pol or policy()
        self.sessions = session_factory or make_session_factory()
        self.zd = zd or Zendesk(ZendeskConfig.from_env())
        self.slack = slack_client if slack_client is not None else slack.from_env()
        self.base_url = os.environ.get("SOA_BASE_URL", "http://127.0.0.1:8020")
        self.triager = Triager(self.policy, training_tickets())
        self.kb = KnowledgeBase.load(self.policy.kb_dir)
        self.planner_name = planner or (
            "llm" if os.environ.get("ANTHROPIC_API_KEY") and planner == "llm" else "deterministic"
        )
        self.planner = choose_planner(self.planner_name)
        self.dispatchers = {
            "zendesk": zendesk.make_dispatcher(self.zd),
            "slack": slack.make_dispatcher(self.slack, self.base_url),
        }

    # -- inbound ----------------------------------------------------------------------------------

    def handle_webhook(self, headers: dict[str, str], body: bytes) -> WebhookResult:
        hdr = {k.lower(): v for k, v in headers.items()}
        ok = verify_webhook(
            self.zd.cfg.webhook_secret, hdr.get(TS_HEADER.lower(), ""), body, hdr.get(SIG_HEADER.lower(), "")
        )
        with self.sessions() as s:
            if not ok:
                s.add(
                    WebhookEvent(
                        event_id=f"rejected:{now().timestamp()}",
                        signature_ok=False,
                        outcome="rejected",
                        detail="bad signature",
                    )
                )
                s.commit()
                return WebhookResult("rejected", "bad signature")
            event = json.loads(body)
            event_id = str(event.get("id") or "")
            if not event_id:
                return WebhookResult("rejected", "no event id")
            if s.scalar(select(WebhookEvent).where(WebhookEvent.event_id == event_id)) is not None:
                return WebhookResult("duplicate", "event already processed", str(event.get("ticket", {}).get("id", "")))
            t = event["ticket"]
            ticket = Ticket(
                id=str(t["id"]),
                subject=t.get("subject", ""),
                body=t.get("description", ""),
                requester=(t.get("requester") or {}).get("email", "customer@example.com"),
            )
            run = self._process(s, ticket)
            s.add(
                WebhookEvent(
                    event_id=event_id,
                    ticket_id=ticket.id,
                    signature_ok=True,
                    outcome="processed",
                    detail=run.decision.action,
                )
            )
            s.commit()
            return WebhookResult("processed", run.decision.rule, ticket.id, run.decision.action)

    def process_ticket(self, ticket: Ticket) -> AgentRun:
        """Run the agent on a ticket that arrived some other way (poller, CLI, MCP)."""
        with self.sessions() as s:
            run = self._process(s, ticket)
            s.commit()
            return run

    def _process(self, s: Session, ticket: Ticket) -> AgentRun:
        row = s.get(TicketRow, ticket.id)
        if row is None:
            row = TicketRow(id=ticket.id, subject=ticket.subject, body=ticket.body, requester=ticket.requester)
            s.add(row)
        agent = Agent(self.policy, self.triager, self.kb, OutboxSink(s), self.planner)
        run = agent.run(ticket)
        row.queue, row.priority = run.triage.queue, run.triage.priority
        row.action, row.rule = run.decision.action, run.decision.rule
        s.add(
            AgentRunRow(
                ticket_id=ticket.id,
                planner=run.planner,
                action=run.decision.action,
                rule=run.decision.rule,
                payload=json.dumps(run.as_dict()),
            )
        )
        return run

    # -- human review -----------------------------------------------------------------------------

    def review(self, item_id: int, verdict: str, reviewer: str, final_body: str | None = None) -> ReviewItem:
        if verdict not in ("approve", "edit", "reject"):
            raise ValueError(f"unknown verdict {verdict!r}")
        if not reviewer.strip():
            raise ValueError("reviewer name is required; approvals are attributed")
        with self.sessions() as s:
            item = s.get(ReviewItem, item_id)
            if item is None:
                raise KeyError(item_id)
            if item.status != "pending":
                return item
            item.reviewer, item.decided_at = reviewer.strip(), now()
            if verdict == "reject":
                item.status = "rejected"
                outbox.enqueue(
                    s, "zendesk", "internal_note", item.ticket_id, f"{item.ticket_id}:note:rejected",
                    {
                        "ticket_id": item.ticket_id,
                        "text": f"[support-ops-agent] Draft rejected by {item.reviewer}. Human reply needed.",
                    },
                )  # fmt: skip
            else:
                body = (final_body if verdict == "edit" and final_body is not None else item.draft).strip()
                item.status = "approved" if verdict == "approve" else "edited"
                item.final_body = body
                outbox.enqueue(
                    s, "zendesk", "public_reply", item.ticket_id, f"{item.ticket_id}:reply",
                    {
                        "ticket_id": item.ticket_id,
                        "body": body,
                        "article_id": item.article_id,
                        "approved_by": item.reviewer,
                    },
                )  # fmt: skip
            s.commit()
            return item

    # -- outbound ---------------------------------------------------------------------------------

    def drain_outbox(self, limit: int = 50) -> dict:
        with self.sessions() as s:
            return outbox.process_once(s, self.dispatchers, limit=limit)

    # -- read models for the UI / MCP -------------------------------------------------------------

    def queue(self) -> dict:
        with self.sessions() as s:
            pending = s.scalars(
                select(ReviewItem).where(ReviewItem.status == "pending").order_by(ReviewItem.created_at)
            ).all()
            tickets = {t.id: t for t in s.scalars(select(TicketRow)).all()}
            actions = dict(s.execute(select(TicketRow.action, func.count()).group_by(TicketRow.action)).all())
            recent = s.scalars(select(TicketRow).order_by(TicketRow.received_at.desc()).limit(50)).all()
            decided = s.scalars(
                select(ReviewItem)
                .where(ReviewItem.status != "pending")
                .order_by(ReviewItem.decided_at.desc())
                .limit(20)
            ).all()
            return {
                "pending": [(p, tickets.get(p.ticket_id)) for p in pending],
                "actions": actions,
                "recent": recent,
                "decided": decided,
                "outbox": outbox.summary(s),
                "n_tickets": len(tickets),
            }

    def ticket_detail(self, ticket_id: str) -> dict | None:
        with self.sessions() as s:
            row = s.get(TicketRow, ticket_id)
            if row is None:
                return None
            run = s.scalars(
                select(AgentRunRow).where(AgentRunRow.ticket_id == ticket_id).order_by(AgentRunRow.id.desc())
            ).first()
            items = s.scalars(select(ReviewItem).where(ReviewItem.ticket_id == ticket_id)).all()
            ob = s.scalars(
                select(outbox.OutboxItem).where(outbox.OutboxItem.ticket_id == ticket_id).order_by(outbox.OutboxItem.id)
            ).all()
            return {"ticket": row, "run": json.loads(run.payload) if run else None, "review_items": items, "outbox": ob}
