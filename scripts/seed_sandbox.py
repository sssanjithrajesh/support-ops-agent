"""Create tickets in the sandbox so the webhook -> agent -> review-queue loop runs end to end.

    uv run soa-sandbox          # terminal 1
    uv run soa-serve            # terminal 2
    uv run soa-seed --n 40      # terminal 3: 40 held-out dataset tickets + the 7 handwritten eval cases

Ticket creation uses an Idempotency-Key, so running the seed twice does not create duplicates, even
though the sandbox injects 429s, 503s and apply-then-500s while we do it.
"""

from __future__ import annotations

import argparse

import yaml

from soa.config import ROOT
from soa.tickets import heldout_tickets
from soa.zendesk import Zendesk, ZendeskConfig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--run", default="seed1", help="changes the idempotency keys; use a new value to seed again")
    args = ap.parse_args(argv)

    zd = Zendesk(ZendeskConfig.from_env())
    created = 0
    for t in heldout_tickets()[: args.n]:
        zd.create_ticket(t.subject, t.body, f"{t.id.lower()}@example.com", idempotency_key=f"{args.run}:{t.id}")
        created += 1
    cases = yaml.safe_load((ROOT / "evals" / "cases" / "hand_labelled.yaml").read_text(encoding="utf-8"))
    for c in cases:
        if c["source"] == "handwritten":
            zd.create_ticket(
                c["subject"],
                c["body"],
                c.get("requester", "customer@example.com"),
                idempotency_key=f"{args.run}:{c['id']}",
            )
            created += 1
    print(f"created {created} tickets in {zd.cfg.base_url} with {zd.client.attempts_made} HTTP attempts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
