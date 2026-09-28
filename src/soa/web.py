"""The review queue: what a support agent sees.

    uv run soa-serve     # http://127.0.0.1:8020

Routes: / (queue), /tickets/{id} (full decision trace), POST /review/{id} (approve / edit / reject,
attributed to a named reviewer), POST /webhooks/zendesk (signed inbound), POST /outbox/drain.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from soa.service import Service

TEMPLATES = Path(__file__).parent / "templates"


def build(service: Service | None = None, drain_interval: float | None = None) -> FastAPI:
    """drain_interval: seconds between outbox drains by a background thread (0 disables; env SOA_OUTBOX_INTERVAL)."""
    service = service or Service()
    app = FastAPI(title="support-ops-agent")
    app.state.service = service
    templates = Jinja2Templates(directory=str(TEMPLATES))
    interval = float(os.environ.get("SOA_OUTBOX_INTERVAL", "5")) if drain_interval is None else drain_interval
    if interval > 0:
        _start_drainer(service, interval)

    @app.get("/health")
    def health():
        return {"ok": True, "planner": service.planner_name, "ticketing": service.zd.cfg.base_url}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        q = service.queue()
        return templates.TemplateResponse(request, "index.html", {"q": q, "planner": service.planner_name})

    @app.get("/tickets/{ticket_id}", response_class=HTMLResponse)
    def ticket(request: Request, ticket_id: str):
        d = service.ticket_detail(ticket_id)
        if d is None:
            raise HTTPException(404, "unknown ticket")
        return templates.TemplateResponse(request, "ticket.html", {"d": d})

    @app.post("/review/{item_id}")
    def review(item_id: int, verdict: str = Form(...), reviewer: str = Form(""), final_body: str = Form("")):
        try:
            item = service.review(item_id, verdict, reviewer, final_body or None)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "unknown review item") from exc
        service.drain_outbox()
        return RedirectResponse(f"/tickets/{item.ticket_id}", status_code=303)

    @app.post("/webhooks/zendesk")
    async def webhook(request: Request):
        body = await request.body()
        result = service.handle_webhook(dict(request.headers), body)
        if result.outcome == "rejected":
            return JSONResponse({"outcome": result.outcome, "detail": result.detail}, status_code=401)
        # Best effort immediate delivery; anything that fails stays in the outbox for the next drain.
        service.drain_outbox()
        return {
            "outcome": result.outcome,
            "ticket_id": result.ticket_id,
            "action": result.action,
            "rule": result.detail,
        }

    @app.post("/outbox/drain")
    def drain():
        return service.drain_outbox()

    return app


def _start_drainer(service: Service, interval: float) -> None:
    """The outbox worker. In a real deployment this is its own process; here a daemon thread is enough."""

    def loop() -> None:
        while True:
            time.sleep(interval)
            try:
                service.drain_outbox()
            except Exception:  # noqa: BLE001  a broken drain must not kill the worker
                pass

    threading.Thread(target=loop, daemon=True, name="outbox-drainer").start()


def app_factory() -> FastAPI:
    return build()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "soa.web:app_factory",
        factory=True,
        host="127.0.0.1",
        port=int(os.environ.get("SOA_PORT", "8020")),
        log_level="warning",
    )
