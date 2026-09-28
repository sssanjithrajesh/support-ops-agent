"""HTTP retry, webhook signatures, Zendesk idempotency against the sandbox, outbox scheduling."""

from __future__ import annotations

import time
from datetime import timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from soa import outbox
from soa.db import OutboxItem, make_session_factory, now
from soa.http import Client, OfflineError, PermanentError, RetryableError, RetryPolicy
from soa.sandbox import SandboxState, build
from soa.zendesk import Zendesk, ZendeskConfig, marker_tag, sign_webhook, verify_webhook

CFG = ZendeskConfig(base_url="http://sandbox", email="agent@example.com", token="change-me", webhook_secret="s3cret")


def _client_with(responses, sleeps):
    it = iter(responses)

    def handler(request):
        return next(it)

    return Client(
        "http://x", transport=httpx.MockTransport(handler), sleep=sleeps.append, policy=RetryPolicy(attempts=4)
    )


def test_retry_honours_retry_after_then_backs_off_with_jitter():
    sleeps: list[float] = []
    c = _client_with(
        [httpx.Response(429, headers={"Retry-After": "2"}), httpx.Response(503), httpx.Response(200, json={"ok": 1})],
        sleeps,
    )
    assert c.get("/a").json() == {"ok": 1}
    assert c.attempts_made == 3
    assert sleeps[0] == 2.0  # Retry-After wins
    assert 0.75 <= sleeps[1] <= 1.25  # 0.5 * 2**1 with +-25% jitter


def test_retry_gives_up_and_permanent_errors_do_not_retry():
    sleeps: list[float] = []
    c = _client_with([httpx.Response(503)] * 4, sleeps)
    with pytest.raises(RetryableError):
        c.get("/a")
    assert c.attempts_made == 4 and len(sleeps) == 3
    c2 = _client_with([httpx.Response(400, text="bad")], sleeps)
    with pytest.raises(PermanentError):
        c2.get("/a")
    assert c2.attempts_made == 1


def test_offline_guard(monkeypatch):
    monkeypatch.setenv("SOA_OFFLINE", "1")
    c = _client_with([httpx.Response(200)], [])
    with pytest.raises(OfflineError):
        c.get("/a")
    assert c.attempts_made == 0


def test_webhook_signature_positive_negative_and_replay():
    body, ts = b'{"id":"evt_1","ticket":{"id":7}}', str(int(time.time()))
    good = sign_webhook("s3cret", ts, body)
    assert verify_webhook("s3cret", ts, body, good)
    flipped = ("A" if good[0] != "A" else "B") + good[1:]  # provably different, not probably different
    assert not verify_webhook("s3cret", ts, body, flipped)
    assert not verify_webhook("s3cret", ts, body + b" ", good)
    assert not verify_webhook("other", ts, body, good)
    old = str(int(time.time()) - 600)
    assert not verify_webhook("s3cret", old, body, sign_webhook("s3cret", old, body))
    assert not verify_webhook("s3cret", "", body, good)


def _sandbox(faults, every=1):
    state = SandboxState(email=CFG.email, token=CFG.token, faults=faults, fault_every=every, deliver=lambda *a: None)
    http = TestClient(build(state), base_url="http://sandbox")
    http.auth = (f"{CFG.email}/token", CFG.token)
    return state, Zendesk(CFG, http=http, sleep=lambda s: None)


def test_ticket_creation_is_idempotent_across_faults():
    # Every write faults: 503 first, then created-but-500. The Idempotency-Key makes blind retries safe.
    state, zd = _sandbox(["503", "apply_then_500"], every=1)
    t1 = zd.create_ticket("Hello", "body", "c@example.com", idempotency_key="seed:1")
    assert zd.client.attempts_made == 3  # 503, apply_then_500, idempotent replay
    t2 = zd.create_ticket("Hello", "body", "c@example.com", idempotency_key="seed:1")
    assert t1["id"] == t2["id"] and len(state.tickets) == 1


def test_update_applies_exactly_once_even_when_server_applies_then_fails():
    state, zd = _sandbox(["apply_then_500"], every=1)  # every write: applied, then 500
    t = zd.create_ticket("Hello", "body", "c@example.com", idempotency_key="k0")
    tid = str(t["id"])
    # The sandbox applies the update and returns 500; the client reads the ticket back, sees its marker, stops.
    status = zd.update_ticket(tid, "T:reply", comment="Thanks!", public=True)
    assert status == "applied_despite_error"
    public = [c for c in state.comments[t["id"]] if c["public"] and c["body"] == "Thanks!"]
    assert len(public) == 1
    assert marker_tag("T:reply") in state.tickets[t["id"]]["tags"]
    # A second call with the same key is a no-op.
    assert zd.update_ticket(tid, "T:reply", comment="Thanks!", public=True) == "already_applied"
    assert len([c for c in state.comments[t["id"]] if c["public"]]) == 2  # the original description + one reply


def test_naive_retry_would_have_duplicated_the_comment():
    """Documents the failure the marker tag prevents: PUT twice without checking = two comments."""
    state, zd = _sandbox([], every=0)
    t = zd.create_ticket("Hello", "body", "c@example.com", idempotency_key="k1")
    for _ in range(2):
        zd.client.put(f"/api/v2/tickets/{t['id']}.json", json={"ticket": {"comment": {"body": "dup", "public": True}}})
    assert len([c for c in state.comments[t["id"]] if c["body"] == "dup"]) == 2


def test_wrong_credentials_are_permanent():
    state, _ = _sandbox([], every=0)
    bad = Zendesk(
        ZendeskConfig("http://sandbox", "agent@example.com", "wrong", "x"),
        http=TestClient(build(state), base_url="http://sandbox"),
        sleep=lambda s: None,
    )
    bad.client._http.auth = ("agent@example.com/token", "wrong")
    with pytest.raises(PermanentError):
        bad.list_tickets()


def test_outbox_schedules_retries_and_fails_after_max(tmp_path):
    factory = make_session_factory(f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    with factory() as s:
        outbox.enqueue(s, "zendesk", "public_reply", "1", "1:reply", {"ticket_id": "1", "body": "hi"})
        assert outbox.enqueue(s, "zendesk", "public_reply", "1", "1:reply", {}) is None  # same key, no second row
        outbox.enqueue(s, "slack", "escalation", "2", "2:escalate", {"ticket_id": "2", "summary": "x"})
        s.commit()

    calls = {"n": 0}

    def flaky(kind, key, payload):
        calls["n"] += 1
        raise RetryableError("503", retry_after=5)

    def permanent(kind, key, payload):
        raise PermanentError("400")

    with factory() as s:
        counts = outbox.process_once(s, {"zendesk": flaky, "slack": permanent})
        assert counts == {"sent": 0, "retried": 1, "failed": 1, "skipped": 0}
        item = s.query(OutboxItem).filter_by(idempotency_key="1:reply").one()
        assert item.status == "pending" and item.attempts == 1
        assert item.next_attempt_at.replace(tzinfo=None) > now().replace(tzinfo=None) + timedelta(seconds=3)
        # Not due yet: nothing happens.
        assert outbox.process_once(s, {"zendesk": flaky})["retried"] == 0
        # Force through to the max.
        for _ in range(outbox.MAX_ATTEMPTS):
            item.next_attempt_at = now() - timedelta(seconds=1)
            s.commit()
            outbox.process_once(s, {"zendesk": flaky})
        assert s.query(OutboxItem).filter_by(idempotency_key="1:reply").one().status == "failed"


def test_outbox_offline_keeps_rows_pending_without_burning_attempts(tmp_path, monkeypatch):
    factory = make_session_factory(f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    with factory() as s:
        outbox.enqueue(s, "zendesk", "internal_note", "1", "1:note", {"ticket_id": "1", "text": "x"})
        s.commit()

    def offline_dispatch(kind, key, payload):
        raise OfflineError("SOA_OFFLINE")

    with factory() as s:
        counts = outbox.process_once(s, {"zendesk": offline_dispatch})
        assert counts["skipped"] == 1
        item = s.query(OutboxItem).one()
        assert item.status == "pending" and item.attempts == 0
