"""
Fin Coach web server.

Serves the single-page UI and exposes two kinds of endpoints:

  * /api/chat        — talk to the agent (short-term + long-term memory active)
  * /api/facts       — read the long-term facts the agent has stored in Lakebase
  * /api/db          — a raw peek at the Lakebase Postgres tables + row counts
                       (the "it's literally just Postgres rows" reveal)

The memory endpoints are what make this demoable: you can watch long-term facts
appear in Lakebase as you chat, then start a brand-new session (new thread_id)
and see the agent recall them without you restating anything.
"""

import logging
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agent import run_agent, list_facts, clear_facts, open_lakebase, set_pools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the Lakebase checkpointer + store ONCE and reuse for the process life.
    Also runs setup() which creates the memory tables if they don't exist."""
    async with open_lakebase() as (checkpointer, store):
        # setup() is idempotent DDL. If the tables already exist and this
        # identity lacks CREATE (e.g. they were created by someone else), that's
        # fine — we can still read/write. Only re-raise if the pool is unusable.
        for name, resource in (("store", store), ("checkpointer", checkpointer)):
            try:
                await resource.setup()
            except Exception as e:
                logger.warning(f"{name}.setup() skipped ({type(e).__name__}: {e}) — "
                               "assuming memory tables already exist.")
        set_pools(checkpointer, store)
        logger.info("Lakebase memory ready — pools opened for process lifetime.")
        yield


app = FastAPI(title="Fin Coach — Lakebase Memory Demo", lifespan=lifespan)


def resolve_user(request: Request) -> str:
    """Identify the person. In a deployed Databricks App the authenticated user
    arrives in a header; locally we fall back to a query param or a default.

    When the request carries ?demo=1, we append a "::autodemo" suffix so the
    auto-demo reads/writes an ISOLATED long-term-memory namespace. This keeps the
    scripted demo completely separate from the user's real saved memory — the
    demo never wipes or pollutes the real profile, and can be re-run cleanly."""
    raw = (
        request.headers.get("X-Forwarded-Preferred-Username")
        or request.query_params.get("user")
        or "demo_user"
    )
    uid = raw.replace(".", "_").replace("@", "_at_")
    if request.query_params.get("demo") in ("1", "true", "yes"):
        uid += "::autodemo"
    return uid


class ChatRequest(BaseModel):
    input: list            # [{role, content}, ...] — the new turn(s)
    thread_id: Optional[str] = None


@app.get("/")
async def root():
    return HTMLResponse((HERE / "static" / "index.html").read_text())


@app.get("/api/whoami")
async def whoami(request: Request):
    return {"user_id": resolve_user(request)}


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    user_id = resolve_user(request)
    messages = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in req.input]
    result = await run_agent(messages, thread_id=req.thread_id, user_id=user_id)
    return {"output": result["output"], "thread_id": result["thread_id"], "user_id": user_id}


@app.get("/api/facts")
async def facts(request: Request):
    """The long-term memory panel data — every fact stored for this user."""
    user_id = resolve_user(request)
    return {"user_id": user_id, "facts": await list_facts(user_id)}


@app.post("/api/reset")
async def reset(request: Request):
    """Clear this user's long-term facts so a demo can start from empty memory."""
    user_id = resolve_user(request)
    removed = await clear_facts(user_id)
    return {"user_id": user_id, "removed": removed}


@app.get("/api/db")
async def db_view(request: Request):
    """Best-effort raw view of the Lakebase memory tables + row counts.
    Degrades gracefully if a raw connection isn't reachable in this environment."""
    try:
        from db import table_stats
        return await table_stats()
    except Exception as e:  # never break the demo over the bonus tab
        logger.warning(f"raw db view unavailable: {e}")
        return JSONResponse({"available": False, "reason": str(e)}, status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(__import__("os").environ.get("PORT", "8000")))
