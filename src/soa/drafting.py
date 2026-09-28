"""Draft a customer reply grounded in one knowledge-base article; redact anything that looks like a card or id number.

The deterministic draft only ever says what the article says. That is the property the eval checks
("grounded" = every sentence of the draft appears in the cited article or the fixed template), and it is
why the auto-send path can exist at all. The LLM planner, when a key is present, must cite an article id
too; drafts without a citation are routed to review regardless of what the model thinks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from soa.kb import Article, Hit
from soa.redact import redact
from soa.tickets import Ticket

__all__ = ["Draft", "draft_from_article", "is_grounded", "redact"]


@dataclass
class Draft:
    ticket_id: str
    article_id: str | None
    body: str
    grounded: bool
    kb_score: float
    citations: list[str] = field(default_factory=list)
    redactions: int = 0

    def as_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "article_id": self.article_id,
            "grounded": self.grounded,
            "kb_score": round(self.kb_score, 3),
            "citations": self.citations,
            "redactions": self.redactions,
            "body": self.body,
        }


def _first_name(requester: str) -> str:
    local = requester.split("@")[0]
    name = re.split(r"[._\-+]", local)[0]
    return name.capitalize() if name and name.isalpha() else "there"


def draft_from_article(ticket: Ticket, hit: Hit | None) -> Draft:
    subject, redactions = redact(ticket.subject)
    if hit is None:
        body = (
            f"Hi {_first_name(ticket.requester)},\n\n"
            f'Thanks for contacting us about "{subject}". I have passed this to the right team and a '
            f"specialist will reply on this ticket. If there is anything you can add in the meantime "
            f"(screenshots, the time it happened, an order or invoice number), please reply here.\n\n"
            f"Best regards,\nSupport"
        )
        return Draft(ticket.id, None, body, grounded=False, kb_score=0.0, redactions=redactions)
    a: Article = hit.article
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(a.steps, start=1))
    closing = f"\n\n{a.closing}" if a.closing else ""
    body = (
        f"Hi {_first_name(ticket.requester)},\n\n"
        f'Thanks for contacting us about "{subject}".\n\n'
        f"{a.summary}\n\n"
        f"{steps}{closing}\n\n"
        f"If this does not resolve it, reply on this ticket and a specialist will pick it up.\n\n"
        f"Best regards,\nSupport\n[{a.id}: {a.title}]"
    )
    return Draft(ticket.id, a.id, body, grounded=True, kb_score=hit.score, citations=[a.id], redactions=redactions)


def is_grounded(body: str, article: Article | None) -> bool:
    """Every sentence of the draft must come from the article or the fixed template."""
    if article is None:
        return False
    allowed = article.body + " " + " ".join(
        [
            "Hi", "Thanks for contacting us about",
            "If this does not resolve it, reply on this ticket and a specialist will pick it up.",
            "Best regards, Support", f"{article.id}: {article.title}",
        ]
    )  # fmt: skip
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()  # noqa: E731
    haystack = norm(allowed)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
        s = norm(re.sub(r"^\d+\.\s*", "", sentence))
        s = re.sub(r"^hi \w+$", "hi", s)
        s = re.sub(r"^thanks for contacting us about .*$", "thanks for contacting us about", s)
        if len(s) < 4:
            continue
        if s not in haystack:
            return False
    return True
