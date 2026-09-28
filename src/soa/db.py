"""SQLAlchemy tables: the local mirror of tickets, agent runs, review items, webhook events, outbox.

SQLite by default (./data/soa.db); set SOA_DATABASE_URL for Postgres.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from soa.config import ROOT


def now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class TicketRow(Base):
    __tablename__ = "tickets"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # ticketing system id
    subject: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    requester: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(20), default="open")
    queue: Mapped[str] = mapped_column(String(60), default="")
    priority: Mapped[str] = mapped_column(String(10), default="")
    action: Mapped[str] = mapped_column(String(20), default="")
    rule: Mapped[str] = mapped_column(String(60), default="")
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AgentRunRow(Base):
    __tablename__ = "agent_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[str] = mapped_column(String(40), index=True)
    planner: Mapped[str] = mapped_column(String(30))
    action: Mapped[str] = mapped_column(String(20))
    rule: Mapped[str] = mapped_column(String(60))
    payload: Mapped[str] = mapped_column(Text)  # AgentRun.as_dict() as JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ReviewItem(Base):
    __tablename__ = "review_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[str] = mapped_column(String(40), index=True)
    key: Mapped[str] = mapped_column(String(120), unique=True)
    draft: Mapped[str] = mapped_column(Text)
    article_id: Mapped[str] = mapped_column(String(20), default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|approved|edited|rejected
    reviewer: Mapped[str] = mapped_column(String(120), default="")
    final_body: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WebhookEvent(Base):
    __tablename__ = "webhook_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[str] = mapped_column(String(120), unique=True)
    ticket_id: Mapped[str] = mapped_column(String(40), default="")
    signature_ok: Mapped[bool] = mapped_column(Boolean, default=False)
    outcome: Mapped[str] = mapped_column(String(20))  # processed|duplicate|rejected
    detail: Mapped[str] = mapped_column(Text, default="")
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class OutboxItem(Base):
    __tablename__ = "integration_outbox"
    id: Mapped[int] = mapped_column(primary_key=True)
    target: Mapped[str] = mapped_column(String(30))  # zendesk | slack
    kind: Mapped[str] = mapped_column(String(50))  # set_fields | internal_note | public_reply | escalation
    ticket_id: Mapped[str] = mapped_column(String(40), default="", index=True)
    idempotency_key: Mapped[str] = mapped_column(String(200), unique=True)
    payload: Mapped[str] = mapped_column(Text)  # JSON
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|sent|failed|skipped
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    last_error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    remote_id: Mapped[str] = mapped_column(String(200), default="")


def database_url() -> str:
    url = os.environ.get("SOA_DATABASE_URL")
    if url:
        return url
    (ROOT / "data").mkdir(exist_ok=True)
    return f"sqlite:///{(ROOT / 'data' / 'soa.db').as_posix()}"


def make_session_factory(url: str | None = None) -> sessionmaker:
    url = url or database_url()
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    engine = create_engine(url, future=True, **kwargs)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
