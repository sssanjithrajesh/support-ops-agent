"""Zendesk Support API client (the subset the agent needs) and webhook signature verification.

Works against a real Zendesk instance (ZENDESK_SUBDOMAIN + email/API token) or the local sandbox
(soa.sandbox), which implements the same endpoints, the same signature scheme and the same failures.

Idempotency, the part people get wrong:
- Ticket creation: Zendesk honours an `Idempotency-Key` header. We always send one.
- Ticket updates (comments, tags, priority) have no such header. A retried PUT would post the
  comment twice. So every update carries a marker tag `soa_k_<hash of the outbox key>` in the same
  PUT as the comment, and before writing we GET the ticket and skip if the marker is already there.
  A crash between "Zendesk accepted" and "outbox marked sent" is therefore recoverable: the retry
  sees the marker and returns without writing.

Webhook verification (Zendesk scheme): signature = base64(HMAC-SHA256(secret, timestamp + body)),
sent as X-Zendesk-Webhook-Signature with X-Zendesk-Webhook-Signature-Timestamp. Constant-time
compare; timestamp must be within five minutes to defeat replay.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass

import httpx

from soa.http import Client, PermanentError, RetryableError

SIG_HEADER = "X-Zendesk-Webhook-Signature"
TS_HEADER = "X-Zendesk-Webhook-Signature-Timestamp"
MARKER_PREFIX = "soa_k_"


def marker_tag(idempotency_key: str) -> str:
    return MARKER_PREFIX + hashlib.sha256(idempotency_key.encode()).hexdigest()[:16]


def sign_webhook(secret: str, timestamp: str, body: bytes) -> str:
    mac = hmac.new(secret.encode(), timestamp.encode() + body, hashlib.sha256).digest()
    return base64.b64encode(mac).decode()


def verify_webhook(secret: str, timestamp: str, body: bytes, signature: str, at: float | None = None) -> bool:
    """Testing note: build negative-test signatures by flipping a known character, not by guessing."""
    if not secret or not timestamp or not signature:
        return False
    try:
        ts = _parse_timestamp(timestamp)
    except ValueError:
        return False
    if abs((at or time.time()) - ts) > 300:
        return False
    return hmac.compare_digest(sign_webhook(secret, timestamp, body), signature)


def _parse_timestamp(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        # Zendesk sends RFC 3339, e.g. 2026-09-27T10:00:00Z
        from datetime import datetime

        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


@dataclass
class ZendeskConfig:
    base_url: str
    email: str
    token: str
    webhook_secret: str

    @classmethod
    def from_env(cls) -> ZendeskConfig:
        sub = os.environ.get("ZENDESK_SUBDOMAIN", "")
        base = f"https://{sub}.zendesk.com" if sub else os.environ.get("SOA_SANDBOX_URL", "http://127.0.0.1:8021")
        return cls(
            base_url=base,
            email=os.environ.get("ZENDESK_EMAIL", "agent@example.com"),
            token=os.environ.get("ZENDESK_API_TOKEN", "change-me"),
            webhook_secret=os.environ.get("ZENDESK_WEBHOOK_SECRET", "change-me-too"),
        )

    @property
    def is_sandbox(self) -> bool:
        return "zendesk.com" not in self.base_url


class Zendesk:
    def __init__(self, cfg: ZendeskConfig, transport: httpx.BaseTransport | None = None, http=None, sleep=time.sleep):
        self.cfg = cfg
        self.client = Client(
            cfg.base_url, auth=(f"{cfg.email}/token", cfg.token), transport=transport, http=http, sleep=sleep
        )

    # -- reads ------------------------------------------------------------------------------------

    def get_ticket(self, ticket_id: str) -> dict:
        return self.client.get(f"/api/v2/tickets/{ticket_id}.json").json()["ticket"]

    def list_tickets(self, status: str = "open") -> list[dict]:
        return self.client.get("/api/v2/tickets.json", params={"status": status}).json()["tickets"]

    def list_comments(self, ticket_id: str) -> list[dict]:
        return self.client.get(f"/api/v2/tickets/{ticket_id}/comments.json").json()["comments"]

    # -- writes -----------------------------------------------------------------------------------

    def create_ticket(self, subject: str, body: str, requester: str, idempotency_key: str, **fields) -> dict:
        payload = {
            "ticket": {"subject": subject, "comment": {"body": body}, "requester": {"email": requester}, **fields}
        }
        r = self.client.post("/api/v2/tickets.json", json=payload, headers={"Idempotency-Key": idempotency_key})
        return r.json()["ticket"]

    def update_ticket(
        self,
        ticket_id: str,
        idempotency_key: str,
        comment: str | None = None,
        public: bool = False,
        tags: list[str] | None = None,
        priority: str | None = None,
    ) -> str:
        """Idempotent update. Returns 'applied', 'applied_despite_error' or 'already_applied'.

        The PUT is sent exactly once per call (attempts=1). A blind transport-level retry of a PUT that
        carries a comment is how duplicate replies happen: Zendesk can apply the update and still return
        a 500 or time out. So on any retryable failure we read the ticket back; if our marker tag is
        there the write landed and we return; if not, we raise and the outbox retries later, at which
        point this method starts again with the marker check.
        """
        marker = marker_tag(idempotency_key)
        if self._has_marker(ticket_id, marker):
            return "already_applied"
        ticket: dict = {"additional_tags": sorted(set(tags or []) | {marker})}
        if comment is not None:
            ticket["comment"] = {"body": comment, "public": public}
        if priority is not None:
            ticket["priority"] = priority
        try:
            self.client.put(f"/api/v2/tickets/{ticket_id}.json", json={"ticket": ticket}, attempts=1)
        except RetryableError:
            if self._has_marker(ticket_id, marker):
                return "applied_despite_error"
            raise
        return "applied"

    def _has_marker(self, ticket_id: str, marker: str) -> bool:
        return marker in (self.get_ticket(ticket_id).get("tags") or [])


def make_dispatcher(zd: Zendesk):
    """Outbox dispatcher for target 'zendesk'."""

    def dispatch(kind: str, key: str, payload: dict) -> str:
        ticket_id = payload["ticket_id"]
        if kind == "set_fields":
            return zd.update_ticket(ticket_id, key, tags=payload["tags"], priority=payload["priority"])
        if kind == "internal_note":
            return zd.update_ticket(ticket_id, key, comment=payload["text"], public=False)
        if kind == "public_reply":
            return zd.update_ticket(ticket_id, key, comment=payload["body"], public=True, tags=["soa_replied"])
        raise PermanentError(f"zendesk dispatcher does not handle kind {kind!r}")

    return dispatch


def webhook_body(ticket: dict, event_id: str, event_type: str = "ticket.created") -> bytes:
    """The payload the sandbox (and a real Zendesk webhook configured this way) sends."""
    return json.dumps({"id": event_id, "type": event_type, "ticket": ticket}, separators=(",", ":")).encode()
