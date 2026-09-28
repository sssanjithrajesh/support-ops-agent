"""Slack escalations via incoming webhook. Goes through the outbox like every other write."""

from __future__ import annotations

import os
import time

import httpx

from soa.http import Client, PermanentError


class Slack:
    def __init__(self, webhook_url: str, transport: httpx.BaseTransport | None = None, http=None, sleep=time.sleep):
        self.webhook_url = webhook_url
        self.client = Client("https://hooks.slack.com", transport=transport, http=http, sleep=sleep)

    def notify(self, text: str, blocks: list[dict] | None = None) -> str:
        payload: dict = {"text": text}
        if blocks:
            payload["blocks"] = blocks
        r = self.client.post(self.webhook_url, json=payload)
        if r.text.strip() != "ok":
            raise PermanentError(f"slack webhook returned {r.text[:100]!r}")
        return "ok"


def escalation_message(ticket_id: str, summary: str, url: str) -> tuple[str, list[dict]]:
    text = f"Escalation: {summary}"
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Escalation* <{url}|{ticket_id}>\n{summary}"}},
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "Agent paused. No customer reply until a human speaks."}],
        },
    ]
    return text, blocks


def make_dispatcher(slack: Slack | None, base_url: str = ""):
    def dispatch(kind: str, key: str, payload: dict) -> str:
        if kind != "escalation":
            raise PermanentError(f"slack dispatcher does not handle kind {kind!r}")
        if slack is None:
            # No SLACK_WEBHOOK_URL: the escalation is recorded in the outbox and the internal note; nothing is lost.
            return "recorded-no-slack"
        text, blocks = escalation_message(
            payload["ticket_id"], payload["summary"], f"{base_url}/tickets/{payload['ticket_id']}"
        )
        return slack.notify(text, blocks)

    return dispatch


def from_env(transport: httpx.BaseTransport | None = None) -> Slack | None:
    url = os.environ.get("SLACK_WEBHOOK_URL", "")
    return Slack(url, transport=transport) if url else None
