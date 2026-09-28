"""A local stand-in for a Zendesk developer sandbox.

    uv run soa-sandbox            # http://127.0.0.1:8021

Implements the endpoints the agent uses (tickets list/get/create/update, comments), Basic auth with
email/token, Idempotency-Key on ticket creation, and signed webhooks on ticket creation using
Zendesk's signature scheme. It also injects the failures a real integration meets, on a schedule:

    429 with Retry-After     rate limit; the client must wait the stated time, then retry
    503                      transient; exponential backoff with jitter
    apply_then_500           the write IS applied, then the response is a 500. This is the one that
                             creates duplicate comments in naive clients and is why every update
                             carries a marker tag.

SOA_SANDBOX_FAULTS=429,503,apply_then_500 (default) and SOA_SANDBOX_FAULT_EVERY=3 control it;
set SOA_SANDBOX_FAULTS= (empty) for a well-behaved server. Everything is in memory.
"""

from __future__ import annotations

import base64
import itertools
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from html import escape

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from soa.zendesk import SIG_HEADER, TS_HEADER, sign_webhook, webhook_body


class SandboxState:
    def __init__(
        self,
        email: str = "agent@example.com",
        token: str = "change-me",
        webhook_url: str = "",
        webhook_secret: str = "change-me-too",
        faults: list[str] | None = None,
        fault_every: int = 3,
        deliver=None,
    ) -> None:
        self.email, self.token = email, token
        self.webhook_url, self.webhook_secret = webhook_url, webhook_secret
        self.faults = faults if faults is not None else ["429", "503", "apply_then_500"]
        self._fault_cycle = itertools.cycle(self.faults) if self.faults else None
        self.fault_every = fault_every
        self.deliver = deliver or _deliver_http
        self.tickets: dict[int, dict] = {}
        self.comments: dict[int, list[dict]] = {}
        self.idempotency: dict[str, int] = {}
        self.log: list[dict] = []
        self.write_count = 0
        self._next_id = 1001
        self.lock = threading.Lock()

    def next_fault(self) -> str | None:
        self.write_count += 1
        if self._fault_cycle and self.fault_every and self.write_count % self.fault_every == 0:
            return next(self._fault_cycle)
        return None

    def record(self, method: str, path: str, status: int, fault: str | None = None) -> None:
        self.log.append({"t": time.time(), "method": method, "path": path, "status": status, "fault": fault})

    def create(
        self, subject: str, body: str, requester: str, priority: str = "", tags: list[str] | None = None
    ) -> dict:
        tid = self._next_id
        self._next_id += 1
        ts = datetime.now(UTC).isoformat()
        ticket = {
            "id": tid,
            "subject": subject,
            "description": body,
            "requester": {"email": requester},
            "status": "open",
            "priority": priority or None,
            "tags": list(tags or []),
            "created_at": ts,
            "updated_at": ts,
        }
        self.tickets[tid] = ticket
        self.comments[tid] = [{"id": 1, "body": body, "public": True, "author": requester, "created_at": ts}]
        return ticket

    def fire_webhook(self, ticket: dict) -> None:
        if not self.webhook_url:
            return
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        body = webhook_body(ticket, event_id)
        ts = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            SIG_HEADER: sign_webhook(self.webhook_secret, ts, body),
            TS_HEADER: ts,
            "X-Sandbox-Event-Id": event_id,
        }
        self.deliver(self.webhook_url, headers, body)


def _deliver_http(url: str, headers: dict, body: bytes) -> None:
    def go() -> None:
        try:
            httpx.post(url, headers=headers, content=body, timeout=10)
        except httpx.HTTPError:
            pass  # a real Zendesk retries webhooks; the sandbox drops them and the poller catches up

    threading.Thread(target=go, daemon=True).start()


def build(state: SandboxState | None = None) -> FastAPI:
    state = state or SandboxState(
        email=os.environ.get("ZENDESK_EMAIL", "agent@example.com"),
        token=os.environ.get("ZENDESK_API_TOKEN", "change-me"),
        webhook_url=os.environ.get(
            "SOA_WEBHOOK_URL", os.environ.get("SOA_BASE_URL", "http://127.0.0.1:8020") + "/webhooks/zendesk"
        ),
        webhook_secret=os.environ.get("ZENDESK_WEBHOOK_SECRET", "change-me-too"),
        faults=[f for f in os.environ.get("SOA_SANDBOX_FAULTS", "429,503,apply_then_500").split(",") if f],
        fault_every=int(os.environ.get("SOA_SANDBOX_FAULT_EVERY", "3")),
    )
    app = FastAPI(title="Zendesk sandbox (local)")
    app.state.sandbox = state

    def auth(authorization: str | None) -> None:
        if not authorization or not authorization.startswith("Basic "):
            raise HTTPException(401, "missing basic auth")
        try:
            user, pw = base64.b64decode(authorization[6:]).decode().split(":", 1)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(401, "bad basic auth") from exc
        if user != f"{state.email}/token" or pw != state.token:
            raise HTTPException(401, "wrong credentials")

    def fault_response(fault: str, method: str, path: str) -> Response | None:
        if fault == "429":
            state.record(method, path, 429, fault)
            return JSONResponse({"error": "RateLimited"}, status_code=429, headers={"Retry-After": "1"})
        if fault == "503":
            state.record(method, path, 503, fault)
            return JSONResponse({"error": "ServiceUnavailable"}, status_code=503)
        return None  # apply_then_500 is handled after the write

    @app.get("/api/v2/tickets.json")
    def list_tickets(status: str | None = None, authorization: str | None = Header(default=None)):
        auth(authorization)
        with state.lock:
            items = [t for t in state.tickets.values() if not status or t["status"] == status]
        state.record("GET", "/api/v2/tickets.json", 200)
        return {"tickets": items, "count": len(items)}

    @app.get("/api/v2/tickets/{ticket_id}.json")
    def get_ticket(ticket_id: int, authorization: str | None = Header(default=None)):
        auth(authorization)
        t = state.tickets.get(ticket_id)
        if t is None:
            raise HTTPException(404, "ticket not found")
        state.record("GET", f"/api/v2/tickets/{ticket_id}.json", 200)
        return {"ticket": t}

    @app.get("/api/v2/tickets/{ticket_id}/comments.json")
    def get_comments(ticket_id: int, authorization: str | None = Header(default=None)):
        auth(authorization)
        if ticket_id not in state.tickets:
            raise HTTPException(404, "ticket not found")
        return {"comments": state.comments[ticket_id]}

    @app.post("/api/v2/tickets.json", status_code=201)
    async def create_ticket(
        request: Request,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        auth(authorization)
        data = (await request.json())["ticket"]
        with state.lock:
            if idempotency_key and idempotency_key in state.idempotency:
                state.record("POST", "/api/v2/tickets.json", 200, "idempotent_replay")
                return JSONResponse({"ticket": state.tickets[state.idempotency[idempotency_key]]}, status_code=200)
            fault = state.next_fault()
            if fault in ("429", "503"):
                return fault_response(fault, "POST", "/api/v2/tickets.json")
            ticket = state.create(
                data["subject"],
                data["comment"]["body"],
                data.get("requester", {}).get("email", "customer@example.com"),
                data.get("priority") or "",
                data.get("tags"),
            )
            if idempotency_key:
                state.idempotency[idempotency_key] = ticket["id"]
        state.fire_webhook(ticket)
        if fault == "apply_then_500":
            state.record("POST", "/api/v2/tickets.json", 500, fault)
            return JSONResponse({"error": "InternalError (ticket was created)"}, status_code=500)
        state.record("POST", "/api/v2/tickets.json", 201)
        return {"ticket": ticket}

    @app.put("/api/v2/tickets/{ticket_id}.json")
    async def update_ticket(ticket_id: int, request: Request, authorization: str | None = Header(default=None)):
        auth(authorization)
        path = f"/api/v2/tickets/{ticket_id}.json"
        with state.lock:
            t = state.tickets.get(ticket_id)
            if t is None:
                raise HTTPException(404, "ticket not found")
            fault = state.next_fault()
            if fault in ("429", "503"):
                return fault_response(fault, "PUT", path)
            data = (await request.json())["ticket"]
            for tag in data.get("additional_tags", []):
                if tag not in t["tags"]:
                    t["tags"].append(tag)
            if data.get("priority"):
                t["priority"] = data["priority"]
            if data.get("status"):
                t["status"] = data["status"]
            if "comment" in data:
                c = data["comment"]
                state.comments[ticket_id].append(
                    {
                        "id": len(state.comments[ticket_id]) + 1,
                        "body": c["body"],
                        "public": bool(c.get("public", False)),
                        "author": state.email,
                        "created_at": datetime.now(UTC).isoformat(),
                    }
                )
            t["updated_at"] = datetime.now(UTC).isoformat()
        if fault == "apply_then_500":
            state.record("PUT", path, 500, fault)
            return JSONResponse({"error": "InternalError (update was applied)"}, status_code=500)
        state.record("PUT", path, 200)
        return {"ticket": t}

    @app.get("/_sandbox/log")
    def sandbox_log():
        faults = [x for x in state.log if x["fault"]]
        return {"requests": len(state.log), "faults": len(faults), "log": state.log[-200:]}

    @app.post("/_sandbox/reset")
    def sandbox_reset():
        with state.lock:
            state.tickets.clear()
            state.comments.clear()
            state.idempotency.clear()
            state.log.clear()
            state.write_count = 0
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def index():
        rows = []
        for t in sorted(state.tickets.values(), key=lambda x: -x["id"]):
            comments = state.comments[t["id"]]
            public = [c for c in comments[1:] if c["public"]]
            internal = [c for c in comments[1:] if not c["public"]]
            tags = " ".join(f"<code>{escape(x)}</code>" for x in t["tags"] if not x.startswith("soa_k_"))
            markers = sum(1 for x in t["tags"] if x.startswith("soa_k_"))
            rows.append(
                f"<tr><td>{t['id']}</td><td>{escape(t['subject'])}</td><td>{escape(t.get('priority') or '-')}</td>"
                f"<td>{tags}<br><small>{markers} idempotency marker(s)</small></td>"
                f"<td>{len(public)} public / {len(internal)} internal</td></tr>"
            )
        faults = [x for x in state.log if x["fault"]]
        by_kind: dict[str, int] = {}
        for f in faults:
            by_kind[f["fault"]] = by_kind.get(f["fault"], 0) + 1
        fault_txt = ", ".join(f"{k}: {v}" for k, v in sorted(by_kind.items())) or "none"
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>Zendesk sandbox</title>
<style>body{{font-family:Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#1a1a1a}}
table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #e6e6e6;padding:6px 8px;text-align:left;font-size:13px;vertical-align:top}}
code{{background:#f3f3f3;padding:1px 4px;border-radius:3px;font-size:12px}}
.stat{{display:inline-block;margin-right:24px}}</style></head>
<body><h2>Zendesk sandbox (local)</h2>
<p><span class="stat"><b>{len(state.tickets)}</b> tickets</span>
<span class="stat"><b>{len(state.log)}</b> API requests</span>
<span class="stat"><b>{len(faults)}</b> injected faults ({escape(fault_txt)})</span>
<span class="stat">webhooks to <code>{escape(state.webhook_url or "disabled")}</code></span></p>
<table><tr><th>id</th><th>subject</th><th>priority</th><th>tags</th><th>comments</th></tr>{"".join(rows)}</table>
</body></html>"""

    return app


def app_factory() -> FastAPI:
    return build()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "soa.sandbox:app_factory",
        factory=True,
        host="127.0.0.1",
        port=int(os.environ.get("SOA_SANDBOX_PORT", "8021")),
        log_level="warning",
    )
