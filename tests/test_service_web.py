"""End to end: sandbox fires a signed webhook -> agent -> review queue -> human approves -> reply lands once."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from soa import slack
from soa.db import make_session_factory
from soa.sandbox import SandboxState
from soa.sandbox import build as build_sandbox
from soa.service import Service
from soa.web import build as build_web
from soa.zendesk import SIG_HEADER, TS_HEADER, Zendesk, ZendeskConfig, sign_webhook, webhook_body

CFG = ZendeskConfig(base_url="http://sandbox", email="agent@example.com", token="change-me", webhook_secret="s3cret")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Sandbox + service + web app wired in-process. The sandbox delivers webhooks straight into the web app."""
    monkeypatch.delenv("SOA_OFFLINE", raising=False)
    holder: dict = {}

    def deliver(url, headers, body):
        holder["web"].post("/webhooks/zendesk", headers=headers, content=body)

    state = SandboxState(
        email=CFG.email, token=CFG.token, webhook_url="http://web/webhooks/zendesk", webhook_secret=CFG.webhook_secret,
        faults=["429", "apply_then_500", "503"], fault_every=3, deliver=deliver,
    )  # fmt: skip
    sb_http = TestClient(build_sandbox(state), base_url="http://sandbox")
    sb_http.auth = (f"{CFG.email}/token", CFG.token)
    zd = Zendesk(CFG, http=sb_http, sleep=lambda s: None)
    factory = make_session_factory(f"sqlite:///{(tmp_path / 'soa.db').as_posix()}")
    slack_calls: list[dict] = []

    class FakeSlack(slack.Slack):
        def __init__(self):
            pass

        def notify(self, text, blocks=None):
            slack_calls.append({"text": text, "blocks": blocks})
            return "ok"

    service = Service(session_factory=factory, zd=zd, slack_client=FakeSlack())
    web = TestClient(build_web(service, drain_interval=0), base_url="http://web")
    holder["web"] = web
    return {"state": state, "zd": zd, "service": service, "web": web, "slack": slack_calls}


def test_webhook_to_review_queue_to_approved_reply(world):
    state, zd, web, service = world["state"], world["zd"], world["web"], world["service"]
    t = zd.create_ticket(
        "Locked out after changing phones",
        "I replaced my phone and the authenticator did not carry over. Please reset two-factor.",
        "dana@example.com",
        idempotency_key="e2e:1",
    )
    tid = str(t["id"])
    detail = service.ticket_detail(tid)
    assert detail is not None and detail["ticket"].action == "review"
    assert len(detail["review_items"]) == 1 and detail["review_items"][0].status == "pending"

    # The triage note and fields reached the sandbox despite injected faults, and exactly once.
    service.drain_outbox()
    for _ in range(5):  # retries are scheduled in the future; pull them forward
        _force_due(service)
        service.drain_outbox()
    internal = [c for c in state.comments[t["id"]] if not c["public"]]
    assert len(internal) == 1 and "REVIEW by rule auto_send_failed" in internal[0]["body"]
    assert "q_technical" in state.tickets[t["id"]]["tags"]

    # Queue page shows it; approving without a name is refused; with a name it sends once.
    page = web.get("/").text
    assert "Locked out after changing phones" in page and "awaiting a human" in page
    item_id = detail["review_items"][0].id
    assert web.post(f"/review/{item_id}", data={"verdict": "approve", "reviewer": ""}).status_code == 400
    r = web.post(f"/review/{item_id}", data={"verdict": "approve", "reviewer": "Priya N."}, follow_redirects=False)
    assert r.status_code == 303
    for _ in range(5):
        _force_due(service)
        service.drain_outbox()
    public = [c for c in state.comments[t["id"]] if c["public"]]
    assert len(public) == 2  # the customer's description + exactly one agent reply
    assert "[KB-001: Regaining access" in public[1]["body"]
    assert "soa_replied" in state.tickets[t["id"]]["tags"]
    # Approving again is a no-op.
    web.post(f"/review/{item_id}", data={"verdict": "approve", "reviewer": "Someone Else"})
    assert len([c for c in state.comments[t["id"]] if c["public"]]) == 2
    assert "Priya N." in web.get(f"/tickets/{tid}").text


def test_escalation_goes_to_slack_and_not_to_the_customer(world):
    state, zd, service = world["state"], world["zd"], world["service"]
    t = zd.create_ticket(
        "Entire company cannot reach the dashboard",
        "Every one of our users gets a 503 error since 9am. Is there an outage?",
        "ops@example.com",
        idempotency_key="e2e:2",
    )
    for _ in range(5):
        _force_due(service)
        service.drain_outbox()
    assert len(world["slack"]) == 1 and "outage_high" in world["slack"][0]["text"]
    assert not [c for c in state.comments[t["id"]][1:] if c["public"]]
    assert service.ticket_detail(str(t["id"]))["review_items"] == []


def test_bad_signature_and_replayed_events_are_rejected(world):
    web, service = world["web"], world["service"]
    body = webhook_body({"id": 999, "subject": "x", "description": "y"}, "evt_dup")
    ts = str(int(time.time()))
    good = {"Content-Type": "application/json", TS_HEADER: ts, SIG_HEADER: sign_webhook(CFG.webhook_secret, ts, body)}
    bad = dict(good, **{SIG_HEADER: "v0=nope"})
    assert web.post("/webhooks/zendesk", headers=bad, content=body).status_code == 401
    assert web.post("/webhooks/zendesk", headers=good, content=body).json()["outcome"] == "processed"
    assert web.post("/webhooks/zendesk", headers=good, content=body).json()["outcome"] == "duplicate"
    assert service.ticket_detail("999") is not None


def test_reject_leaves_a_note_for_the_human(world):
    state, zd, service, web = world["state"], world["zd"], world["service"], world["web"]
    t = zd.create_ticket(
        "Question about my invoice", "Why is my invoice higher this month?", "b@example.com", idempotency_key="e2e:3"
    )
    item = service.ticket_detail(str(t["id"]))["review_items"][0]
    web.post(f"/review/{item.id}", data={"verdict": "reject", "reviewer": "Sam"})
    for _ in range(5):
        _force_due(service)
        service.drain_outbox()
    notes = [c["body"] for c in state.comments[t["id"]] if not c["public"]]
    assert any("Draft rejected by Sam" in n for n in notes)
    assert not [c for c in state.comments[t["id"]][1:] if c["public"]]


def _force_due(service: Service) -> None:
    from datetime import timedelta

    from soa.db import OutboxItem, now

    with service.sessions() as s:
        for item in s.query(OutboxItem).filter_by(status="pending"):
            item.next_attempt_at = now() - timedelta(seconds=1)
        s.commit()
