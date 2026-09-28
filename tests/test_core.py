"""Policy, triage, knowledge base, drafting, decision."""

from __future__ import annotations

from soa.decision import decide
from soa.drafting import draft_from_article, is_grounded, redact
from soa.tickets import Ticket, heldout_tickets, load_sample


def test_policy_is_consistent(pol):
    assert len(pol.queues) == 10
    names = {s.name for s in pol.signals}
    for rule in pol.escalate_when:
        assert set(rule.any_signal) | set(rule.all_signals) <= names, rule.name
    assert set(pol.auto_send.forbidden_signals) <= names
    assert set(pol.priority_boosts) <= names
    assert {h.queue for h in pol.queue_hints} <= set(pol.queue_names)


def test_sample_split_is_fixed():
    s = load_sample()
    assert len(s) == 2000
    assert sum(t.split == "train" for t in s) == 1500
    assert sum(t.split == "heldout" for t in s) == 500
    assert len({t.id for t in s}) == 2000


def test_signals_quote_their_evidence(triager):
    t = Ticket("x", "Charged twice", "I will file a chargeback. My card is 4111 1111 1111 1111.")
    r = triager.triage(t)
    names = {s.name: s.evidence for s in r.signals}
    assert "legal_threat" in names and names["legal_threat"] == ["chargeback"]
    assert "pii_in_ticket" in names and names["pii_in_ticket"] == ["[redacted]"]
    assert any("chargeback" in line for line in r.trace)
    assert "4111" not in " ".join(r.trace)


def test_triage_is_deterministic_and_priority_caps_at_high(triager):
    t = Ticket("x", "Outage", "The service is down for all our users after a breach and we lost all data.")
    a, b = triager.triage(t), triager.triage(t)
    assert a.as_dict() == b.as_dict()
    assert a.priority == "high"
    assert a.priority_boost >= 4  # outage + security + data_loss, capped when applied


def test_queue_hints_change_the_route_and_say_so(triager):
    t = Ticket("x", "Return a headset", "I want to return or exchange the defective headset from order 55821.")
    r = triager.triage(t)
    assert r.queue == "Returns and Exchanges"
    assert any(line.startswith("hint Returns and Exchanges") for line in r.trace)


def test_kb_articles_are_well_formed(kb):
    ids = [a.id for a in kb.articles]
    assert len(ids) == 12 and len(set(ids)) == 12
    for a in kb.articles:
        assert a.summary and len(a.steps) == 3, a.id
    hits = kb.search("I forgot my password and I am locked out")
    assert hits[0].article.id == "KB-001"
    assert "password" in hits[0].overlap


def test_draft_is_grounded_and_cites(kb):
    t = Ticket("x", "Locked out", "Forgot my password", requester="dana.w@example.com")
    hit = kb.search(t.text)[0]
    d = draft_from_article(t, hit)
    assert d.grounded and d.citations == [hit.article.id]
    assert d.body.startswith("Hi Dana,")
    assert is_grounded(d.body, hit.article)
    assert not is_grounded(d.body + "\nWe will refund you in full today.", hit.article)


def test_redaction():
    text, n = redact("card 4111 1111 1111 1111 and ssn 123-45-6789 but order 55821 stays")
    assert n == 2 and "55821" in text and "4111" not in text


def test_decision_order(pol, triager, kb):
    # escalation beats everything
    t = Ticket("x", "Breach", "Our account was hacked", requester="a@example.com")
    r = triager.triage(t)
    d = draft_from_article(t, kb.search(t.text)[0])
    assert decide(pol, r, d).action == "escalate"
    # a clean pre-sales question with a grounded article may auto-send
    t2 = Ticket("y", "Demo and pricing for 80 seats", "Could we get a demo and a quote for about 80 seats?")
    r2 = triager.triage(t2)
    d2 = draft_from_article(t2, kb.search(t2.text, queue=r2.queue)[0])
    dec = decide(pol, r2, d2)
    assert dec.action == "auto_send", dec.reasons
    # the same question with a pasted card number is held for review
    t3 = Ticket("z", t2.subject, t2.body + " Card 4111 1111 1111 1111.")
    r3 = triager.triage(t3)
    d3 = draft_from_article(t3, kb.search(t3.text, queue=r3.queue)[0])
    assert decide(pol, r3, d3).action == "review"


def test_heldout_never_auto_sends_a_guarded_signal(pol, triager, kb):
    forbidden = set(pol.auto_send.forbidden_signals)
    for t in heldout_tickets():
        r = triager.triage(t)
        hits = kb.search(t.text, queue=r.queue)
        d = draft_from_article(t, hits[0] if hits else None)
        if decide(pol, r, d).action == "auto_send":
            assert not (r.signal_names & forbidden), t.id
            assert d.grounded
