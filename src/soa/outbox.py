"""Durable outbox for writes to the ticketing system and Slack.

The agent never calls Zendesk directly. It writes an outbox row in the same transaction as its
decision, and a worker drains the outbox. A rate-limited Zendesk cannot make the agent fail; the
write lands seconds or minutes later, once, because every row carries an idempotency key that
the dispatcher also embeds in the remote write.

Lifted from contract-compliance-reviewer; the only change is the ticket_id column for the UI.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from soa.db import OutboxItem, now
from soa.http import OfflineError, PermanentError, RetryableError

MAX_ATTEMPTS = 8


def enqueue(session: Session, target: str, kind: str, ticket_id: str, key: str, payload: dict) -> OutboxItem | None:
    """Add a row unless one with the same key exists. Caller commits."""
    if session.scalar(select(OutboxItem).where(OutboxItem.idempotency_key == key)) is not None:
        return None
    item = OutboxItem(target=target, kind=kind, ticket_id=ticket_id, idempotency_key=key, payload=json.dumps(payload))
    session.add(item)
    return item


Dispatcher = Callable[[str, str, dict], str]
"""(kind, idempotency_key, payload) -> remote id. Raise RetryableError / PermanentError / OfflineError."""


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def process_once(
    session: Session, dispatchers: dict[str, Dispatcher], at: datetime | None = None, limit: int = 50
) -> dict:
    """Drain due items once. Returns counts. Safe to run from cron, a loop, a request, or a test."""
    at = at or now()
    due = session.scalars(
        select(OutboxItem).where(OutboxItem.status == "pending").order_by(OutboxItem.next_attempt_at).limit(limit)
    ).all()
    due = [item for item in due if _aware(item.next_attempt_at) <= at]
    counts = {"sent": 0, "retried": 0, "failed": 0, "skipped": 0}
    for item in due:
        dispatcher = dispatchers.get(item.target)
        if dispatcher is None:
            item.status, item.last_error = "skipped", f"no dispatcher for target {item.target!r}"
            counts["skipped"] += 1
            continue
        item.attempts += 1
        try:
            item.remote_id = dispatcher(item.kind, item.idempotency_key, json.loads(item.payload)) or ""
        except OfflineError as exc:
            item.last_error = str(exc)
            item.attempts -= 1
            item.next_attempt_at = at + timedelta(minutes=10)
            counts["skipped"] += 1
        except RetryableError as exc:
            item.last_error = str(exc)
            if item.attempts >= MAX_ATTEMPTS:
                item.status = "failed"
                counts["failed"] += 1
            else:
                delay = exc.retry_after or min(3600, 30 * (2 ** (item.attempts - 1)))
                delay *= 1 + random.uniform(-0.2, 0.2)  # noqa: S311
                item.next_attempt_at = at + timedelta(seconds=delay)
                counts["retried"] += 1
        except PermanentError as exc:
            item.status, item.last_error = "failed", str(exc)
            counts["failed"] += 1
        else:
            item.status, item.sent_at, item.last_error = "sent", at, ""
            counts["sent"] += 1
    session.commit()
    return counts


def summary(session: Session) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in session.scalars(select(OutboxItem)).all():
        out[r.status] = out.get(r.status, 0) + 1
    return out
