"""
Fin Coach — a stateful Personal Finance Coach agent.

This is the "brain". It is a LangGraph agent wired to TWO memory systems, both
backed by the SAME Lakebase Autoscaling Postgres instance:

  1. Short-term memory  ->  AsyncCheckpointSaver
       Automatically records every message + tool call, keyed by `thread_id`.
       This is what lets the agent follow a multi-turn conversation.
       Lakebase tables: checkpoints, checkpoint_writes, checkpoint_blobs

  2. Long-term  memory  ->  AsyncDatabricksStore
       Curated facts the agent explicitly decides to save, keyed by `user_id`.
       These survive across threads, sessions and days, and are retrieved by
       semantic search. This is what lets a returning user skip re-introducing
       themselves.
       Lakebase tables: store, store_vectors

The agent exposes memory to the LLM as two tools (`save_memory`,
`recall_memories`) plus a couple of deterministic finance calculators so the
coach can give grounded, numeric answers.

The pattern here mirrors the Databricks Academy "Lakebase for AI Agents" course
(bakehouse-agent-app), re-themed for a relatable daily-life scenario.
"""

import os
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Optional, Tuple

import mlflow
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, InjectedStore
from langgraph.store.base import BaseStore
from databricks_langchain import ChatDatabricks, AsyncCheckpointSaver, AsyncDatabricksStore

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Configuration (env-driven so the same code runs locally and as a Databricks App)
# --------------------------------------------------------------------------
LLM_ENDPOINT = os.environ.get("SERVING_ENDPOINT_NAME", "databricks-claude-sonnet-5")
LB_PROJECT = os.environ.get("LAKEBASE_PROJECT", "")            # e.g. "fincoach"
LB_BRANCH = os.environ.get("LAKEBASE_BRANCH", "production")
MEMORY_SCHEMA = os.environ.get("MEMORY_SCHEMA", "fincoach")     # Postgres schema for all memory tables
EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
EMBEDDING_DIMS = int(os.environ.get("EMBEDDING_DIMS", "1024"))
EXPERIMENT_ID = os.environ.get("MLFLOW_EXPERIMENT_ID", "")

# Namespace used for a user's long-term facts. Keeping it a named constant makes
# the inspector endpoints (server.py) and the agent agree on where facts live.
FACT_NS = "finance_facts"

# Categories the coach is asked to tag facts with. Purely for a tidy demo UI —
# the store itself is schemaless key/value.
FACT_CATEGORIES = [
    "identity", "income", "goal", "risk_tolerance", "recurring_bill",
    "asset", "debt", "household", "preference",
]

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", f"""\
You are **Fin Coach**, a warm, practical personal finance coach. You help one
person understand their money, plan toward their goals, and make everyday
decisions ("can I afford this?", "how long until I hit my goal?").

CRITICAL — you have a long-term memory that persists across every conversation:
1. At the START of every conversation, call `recall_memories` to load what you
   already know about this person (their income, goals, risk tolerance, bills,
   debts, preferences). Greet a returning person by referencing what you know.
2. Whenever the person shares a durable fact, call `save_memory` to persist it.
   ALWAYS save their name the moment they give it (key "name", category
   "identity"). Also save: take-home pay, a savings goal and its timeframe, risk
   tolerance, a recurring bill, an asset, a debt, household details, or a stated
   preference. Choose a short lowercase snake_case `key`, a concise `value`, and
   a `category` from this list: {", ".join(FACT_CATEGORIES)}.
   Save one fact per call. Re-saving the same key updates it.
3. Do NOT ask the person to repeat things you have already saved. Rely on memory.

When a question needs math (affordability, savings timelines, debt payoff), use
the calculator tools rather than guessing. Always show the numbers you used and
give one clear, actionable recommendation. Be concise and encouraging. You are a
coach, not a licensed financial advisor — add a brief reminder of that only when
giving significant investment guidance.""")

# --------------------------------------------------------------------------
# MLflow tracing (optional; only when an experiment id is provided)
# --------------------------------------------------------------------------
try:
    if EXPERIMENT_ID:  # only trace when an experiment is configured (avoids noise locally)
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(experiment_id=EXPERIMENT_ID)
        mlflow.langchain.autolog()
except Exception as e:  # tracing is best-effort, never fatal
    logger.warning(f"MLflow autolog not configured: {e}")

# --------------------------------------------------------------------------
# Long-lived Lakebase connection pools
#
# The checkpointer and store each hold a psycopg AsyncConnectionPool. Opening a
# new pool per request exhausts Lakebase connections under load (PoolTimeout).
# server.py opens ONE pair at startup (lifespan) and hands them here.
# --------------------------------------------------------------------------
_pools: Optional[Tuple[AsyncCheckpointSaver, AsyncDatabricksStore]] = None


def set_pools(checkpointer: AsyncCheckpointSaver, store: AsyncDatabricksStore) -> None:
    global _pools
    _pools = (checkpointer, store)


@asynccontextmanager
async def open_lakebase():
    """Open a fresh (checkpointer, store) pair against Lakebase.

    Called once by server.py's lifespan, and as a fallback when the agent is run
    outside the server (e.g. a notebook or the smoke-test script).
    """
    lk = {"project": LB_PROJECT, "branch": LB_BRANCH}
    async with AsyncDatabricksStore(
        **lk, schema=MEMORY_SCHEMA,
        embedding_endpoint=EMBEDDING_ENDPOINT, embedding_dims=EMBEDDING_DIMS,
    ) as store, AsyncCheckpointSaver(**lk, schema=MEMORY_SCHEMA) as cp:
        yield cp, store


@asynccontextmanager
async def acquire():
    if _pools is not None:
        yield _pools
    else:
        async with open_lakebase() as pair:
            yield pair


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@tool
def save_memory(
    key: str,
    value: str,
    category: str,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore],
) -> str:
    """Persist ONE durable fact about this person's finances so it is remembered
    in every future conversation. `key` is a short snake_case label
    (e.g. "monthly_take_home", "house_deposit_goal"). `value` is the fact.
    `category` is one of: income, goal, risk_tolerance, recurring_bill, asset,
    debt, household, preference."""
    user_id = config["configurable"].get("user_id", "anonymous")
    store.put(
        (FACT_NS, user_id),
        key,
        {"content": value, "category": category},
    )
    return f"Saved [{category}] {key} = {value!r}"


@tool
def recall_memories(
    query: str,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore],
) -> str:
    """Look up what is already known about this person by semantic search over
    their saved facts. Call this at the start of a conversation with a broad
    query like "financial profile goals income" to load context."""
    user_id = config["configurable"].get("user_id", "anonymous")
    results = store.search((FACT_NS, user_id), query=query, limit=10)
    if not results:
        return "No saved facts for this person yet — this looks like a new user."
    lines = [f"  [{r.value.get('category','?')}] {r.key}: {r.value.get('content','')}" for r in results]
    return "Known facts:\n" + "\n".join(lines)


@tool
def savings_projection(monthly_contribution: float, months: int, annual_rate_pct: float = 0.0) -> str:
    """Project the future value of saving `monthly_contribution` every month for
    `months` months at an optional `annual_rate_pct` (e.g. 4.5 for a 4.5% HYSA).
    Returns total contributed, interest earned, and ending balance."""
    r = (annual_rate_pct / 100.0) / 12.0
    if r == 0:
        balance = monthly_contribution * months
    else:
        balance = monthly_contribution * (((1 + r) ** months - 1) / r)
    contributed = monthly_contribution * months
    interest = balance - contributed
    return (f"Over {months} months at {annual_rate_pct:.2f}% APR: "
            f"contributed ${contributed:,.0f}, interest ${interest:,.0f}, "
            f"ending balance ${balance:,.0f}.")


@tool
def affordability_check(monthly_take_home: float, monthly_committed: float, new_monthly_cost: float) -> str:
    """Check whether a new recurring cost fits. Given monthly take-home pay,
    existing monthly committed spend, and the new_monthly_cost, returns remaining
    free cash flow before and after, and the new commitment as a % of income."""
    free_before = monthly_take_home - monthly_committed
    free_after = free_before - new_monthly_cost
    pct = (new_monthly_cost / monthly_take_home * 100.0) if monthly_take_home else 0.0
    verdict = "comfortable" if free_after > 0.2 * monthly_take_home else (
        "tight" if free_after >= 0 else "not affordable — you'd be cash-flow negative")
    return (f"Free cash flow before: ${free_before:,.0f}/mo. After the new "
            f"${new_monthly_cost:,.0f}/mo cost: ${free_after:,.0f}/mo. "
            f"That's {pct:.1f}% of take-home. Verdict: {verdict}.")


@tool
def months_to_goal(target_amount: float, current_saved: float, monthly_contribution: float) -> str:
    """How many months to reach a savings goal, given the target, what's already
    saved, and the monthly contribution. Ignores interest for a conservative
    estimate."""
    remaining = max(0.0, target_amount - current_saved)
    if monthly_contribution <= 0:
        return "With no monthly contribution the goal is never reached at this rate."
    months = remaining / monthly_contribution
    return (f"Remaining to save: ${remaining:,.0f}. At ${monthly_contribution:,.0f}/mo "
            f"that's about {months:.1f} months (~{months/12:.1f} years).")


FINANCE_TOOLS = [savings_projection, affordability_check, months_to_goal]
MEMORY_TOOLS = [save_memory, recall_memories]
ALL_TOOLS = MEMORY_TOOLS + FINANCE_TOOLS


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
def _build_graph(checkpointer, store, model_with_tools, tools):
    def agent_node(state: MessagesState):
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        return {"messages": [model_with_tools.invoke(msgs)]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    wf = StateGraph(MessagesState)
    wf.add_node("agent", agent_node)
    wf.add_node("tools", ToolNode(tools))
    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", should_continue, ["tools", END])
    wf.add_edge("tools", "agent")
    # store= activates long-term memory; checkpointer= activates short-term.
    return wf.compile(checkpointer=checkpointer, store=store)


def _blocks_to_text(blocks) -> str:
    parts = []
    for block in blocks:
        if isinstance(block, dict):
            if block.get("type") in ("text", None) and block.get("text"):
                parts.append(block["text"])
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts).strip()


def _content_to_text(content) -> str:
    """Coalesce a message's content to plain text. Reasoning-capable models
    (e.g. Claude Sonnet 5) return the visible answer alongside thinking blocks —
    sometimes as a Python list of typed blocks, sometimes as a JSON string of
    those same blocks. We keep only the visible text and drop the reasoning."""
    if isinstance(content, list):
        return _blocks_to_text(content)
    if isinstance(content, str):
        s = content.lstrip()
        # Some responses arrive as a JSON-serialized list of content blocks.
        if s.startswith("[") and "\"type\"" in s:
            import json
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    text = _blocks_to_text(parsed)
                    if text:
                        return text
            except Exception:
                pass
        return content
    return str(content)


async def _run(messages, thread_id, user_id, model_with_tools, tools):
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    async with acquire() as (cp, store):
        config["configurable"]["store"] = store
        graph = _build_graph(cp, store, model_with_tools, tools)
        result = await graph.ainvoke({"messages": messages}, config)
    return _content_to_text(result["messages"][-1].content)


@mlflow.trace(name="fincoach_run", span_type="AGENT")
async def run_agent(messages, thread_id=None, user_id="anonymous"):
    """Public entry point. `messages` is a list of {role, content} for the new turn(s).
    The full history is reconstructed from Lakebase via the checkpointer + thread_id."""
    thread_id = thread_id or str(uuid.uuid4())
    model = ChatDatabricks(endpoint=LLM_ENDPOINT)
    model_with_tools = model.bind_tools(ALL_TOOLS)
    output = await _run(messages, thread_id, user_id, model_with_tools, ALL_TOOLS)
    return {"output": output, "thread_id": thread_id}


# --------------------------------------------------------------------------
# Read helpers used by the memory-inspector endpoints in server.py
# --------------------------------------------------------------------------
async def clear_facts(user_id: str) -> int:
    """Delete all long-term facts for a user (used by the demo reset). Returns
    the number of facts removed. Does not touch conversation checkpoints."""
    async with acquire() as (_cp, store):
        items = await store.asearch((FACT_NS, user_id), query=None, limit=500)
        for it in items:
            await store.adelete((FACT_NS, user_id), it.key)
    return len(items)


async def list_facts(user_id: str):
    """Return every long-term fact stored for a user (via the store API, so it
    does not depend on the raw Postgres table layout)."""
    async with acquire() as (_cp, store):
        # query=None -> return items by recency rather than semantic ranking
        items = await store.asearch((FACT_NS, user_id), query=None, limit=200)
    facts = []
    for it in items:
        v = it.value or {}
        facts.append({
            "key": it.key,
            "content": v.get("content", ""),
            "category": v.get("category", "uncategorized"),
            "created_at": str(getattr(it, "created_at", "") or ""),
            "updated_at": str(getattr(it, "updated_at", "") or ""),
        })
    return facts
