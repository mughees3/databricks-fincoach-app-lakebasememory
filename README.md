# Fin Coach — Stateful Agent Memory on Lakebase

A reference app that shows **how to give an AI agent durable memory using
Databricks Lakebase** (managed Postgres), and makes that memory *visible* so you
can see it working live. The scenario is a **personal finance coach** — a
relatable, daily-life assistant where memory is genuinely load-bearing: it
remembers your income, goals, debts and risk tolerance across sessions, so you
never repeat yourself and its advice is always grounded in *your* situation.

> Built on the same pattern as the Databricks Academy course *"Lakebase for AI
> Agents"* (LangGraph dual memory + Lakebase + a Databricks App), re-themed and
> extended with a **live memory-inspector UI** and a **one-click auto-demo**.

---

## Why memory matters

A stateless agent forgets everything between API calls — every conversation starts
from zero and the user has to re-explain themselves. That's a bad product. This app
demonstrates the fix with **two complementary memory layers, both backed by one
Lakebase Autoscaling Postgres instance**:

| | **Short-term** — `AsyncCheckpointSaver` | **Long-term** — `AsyncDatabricksStore` |
|---|---|---|
| Stores | Every message & tool call (full state) | Curated facts the agent chooses to save |
| Keyed by | `thread_id` (one conversation) | `user_id` (the person, forever) |
| Written | Automatically, every graph step | By the agent, via a `save_memory` tool |
| Survives a new session? | ❌ No | ✅ Yes |
| Lakebase tables | `checkpoints`, `checkpoint_writes`, `checkpoint_blobs` | `store`, `store_vectors` |

The key moment: tell Fin Coach your finances, start a **brand-new session**
(the chat clears), then ask *"what's my savings goal?"* — it answers correctly,
because the facts live in Lakebase, not in the conversation.

---

## Architecture

```
  Browser (static/index.html)
    │   chat  +  live memory inspector (facts / session / raw Lakebase tables)
    ▼
  FastAPI  (server.py)
    │
    ▼
  LangGraph agent (agent.py)
    ├─ ChatDatabricks  ───────────►  Foundation Model API (Claude Sonnet)
    ├─ tools: save_memory / recall_memories + finance calculators
    ├─ AsyncCheckpointSaver  ─┐
    └─ AsyncDatabricksStore  ─┴──►  Lakebase Autoscaling Postgres  (schema: fincoach)
                                     store / store_vectors + checkpoint* tables
                                     embeddings via databricks-gte-large-en
```

Everything the agent "remembers" is ordinary Postgres rows in Lakebase — governable,
queryable, backup-able, and visible in the app's **🗄️ Lakebase** tab.

## Files

| File | Purpose |
|---|---|
| `agent.py` | LangGraph agent: dual memory, memory tools, finance calculators |
| `server.py` | FastAPI: `/api/chat`, `/api/facts`, `/api/reset`, `/api/db`, serves the UI |
| `db.py` | Opens a plain Postgres connection to show raw table row-counts (the "it's just Postgres" reveal) |
| `static/index.html` | Single-page UI: chat + memory inspector + **auto-demo** |
| `app.yaml` / `databricks.yml` | Databricks App config + Asset Bundle for deployment |
| `grant_app_access.py` | Grants the deployed app's service principal access to the memory schema |
| `requirements.txt` | Python dependencies |
| `inspect_memory.py` | CLI to dump long-term facts + short-term threads straight from Lakebase |
| `run_local.sh` / `.env.example` | Local development |

---

## Prerequisites

- A Databricks workspace with **Lakebase** and **Foundation Model APIs** (Claude +
  an embedding endpoint) available.
- Databricks CLI ≥ 1.0 authenticated: `databricks auth login --host <url> --profile <name>`
- A **Lakebase Autoscaling project**. Create your
  own with:
  ```bash
  databricks postgres create-project <your-lakebase-project> \
    --json '{"spec": {"display_name": "Fin Coach - Lakebase Memory Demo"}}' -p <your-profile>
  ```
- Python 3.12 + [uv](https://docs.astral.sh/uv/) for local runs.

---

## Run locally

```bash
cp .env.example .env          # adjust profile / project / endpoints if needed
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
./run_local.sh                # -> http://localhost:8000
```

The app authenticates to Lakebase and the model endpoints using your CLI profile
(`DATABRICKS_CONFIG_PROFILE` in `.env`). On first run it creates the memory tables
in the `fincoach` schema automatically.

**Demo multiple people locally:** append `?user=<id>` to the URL
(e.g. `http://localhost:8000/?user=alex`). Each user has isolated long-term memory.

---

## The demo

### Option A — one click
Press **▶ Auto-demo** in the header. It runs a narrated 6-step walkthrough:
clears memory → user shares their finances (watch facts appear in Lakebase) →
adds a debt & risk preference → **starts a fresh session** → asks a recall question
that Fin Coach answers from long-term memory → ends on the raw Lakebase tables.

The auto-demo is **non-destructive**: it runs in an isolated memory namespace
(`<user>::autodemo`, via `?demo=1` on the API calls), so it never touches your
real saved profile. Each run resets only that sandbox, so "watch facts appear
from empty" always works, and when it finishes the app returns to your real
profile. (Your real memory is only ever cleared by the explicit **🧹 Reset**
button, which asks for confirmation.)

### Option B — drive it yourself
1. **🧹 Reset** to start from empty memory.
2. Type: *"My take-home is $6,000/month, I'm saving $60k for a house in 2 years, and I'm risk-averse."*
   → Watch **🧠 Long-term memory** fill with income / goal / risk-tolerance facts.
3. Ask a grounded question: *"Can I afford a $400/month car payment?"* → it uses your saved income.
4. Click **↻ New session** (fresh `thread_id`, chat clears).
5. Ask: *"Remind me — what's my savings goal?"* → **it recalls without you restating.** 🎯
6. Open the **🗄️ Lakebase** tab → the facts are just rows in the `store` table.

---

## Deploy as a Databricks App

```bash
databricks bundle deploy -p <your-profile> --var="lakebase_project=<your-lakebase-project>"   # upload + create app
databricks bundle run fincoach -p <your-profile>          # start the app
```

The bundle grants the app's **service principal** query access to the Claude +
embedding endpoints and `CAN_CONNECT_AND_CREATE` on the Lakebase branch.

### ⚠️ One extra step: grant the app SP access to the memory schema

A Databricks App runs as its own service principal. If the memory schema/tables
were **first created by a human** (e.g. during local dev), that human owns them and
the app SP can't read/write — the app will crash on startup with
`permission denied for schema public`. Fix it once:

```bash
# get the app's service principal client id
APP_SP=$(databricks apps get fincoach -p <your-profile> -o json | jq -r .service_principal_client_id)

# grant it access to the memory schema
APP_SP=$APP_SP LAKEBASE_PROJECT=<your-lakebase-project> MEMORY_SCHEMA=fincoach \
  DATABRICKS_CONFIG_PROFILE=<your-profile> python grant_app_access.py

# restart the app
databricks bundle run fincoach -p <your-profile>
```

(If instead you let the app SP create the schema itself on first run — i.e. no human
touched it first — the SP owns everything and this step is unnecessary.)

Once deployed, your app URL is shown by `databricks bundle run` (e.g. `https://fincoach-<id>.<region>.databricksapps.com`).

---

## Adapting it for your own use case

This is a template — swap the *domain*, keep the *mechanics*:

1. **Change the persona & fact categories** in `agent.py` (`SYSTEM_PROMPT`,
   `FACT_CATEGORIES`) — e.g. a healthcare intake assistant (allergies, meds,
   conditions), a B2B support agent (account, entitlements, past tickets), a
   shopping concierge (sizes, brands, budget).
2. **Add real tools.** Fin Coach ships with deterministic finance calculators;
   point tools at Unity Catalog Functions or your own APIs to query live data.
3. **Keep the two memory layers** — checkpointer for the conversation, store for
   durable per-user facts. That division is the reusable idea.
4. **Point at your own Lakebase project & model endpoints** via
   `app.yaml` / `databricks.yml`.

## Cleanup

```bash
databricks bundle destroy -p <your-profile>                 # remove the app
databricks postgres delete-project projects/<your-lakebase-project> -p <your-profile>  # remove Lakebase (deletes all memory)
```
