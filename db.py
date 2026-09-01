"""
Raw Lakebase peek for the "🗄️ Lakebase" tab.

The long-term-facts panel reads memory through the store's high-level API. This
module instead opens a plain Postgres connection to the SAME Lakebase instance
and runs ordinary SQL, so the demo can show that agent memory is nothing more
than governable Postgres rows: `store`, `store_vectors`, and the three
`checkpoint*` tables.

Uses `LakebasePool` from databricks-ai-bridge, so authentication (OAuth token
minting against the endpoint) is identical whether running locally with a CLI
profile or deployed as a Databricks App service principal.
"""

import os
import asyncio
from databricks_ai_bridge.lakebase import LakebasePool

LB_PROJECT = os.environ.get("LAKEBASE_PROJECT", "")
LB_BRANCH = os.environ.get("LAKEBASE_BRANCH", "production")
MEMORY_SCHEMA = os.environ.get("MEMORY_SCHEMA", "fincoach")

# Friendly description of each memory table for the UI.
TABLE_ROLE = {
    "store": "long-term facts (key/value)",
    "store_vectors": "fact embeddings (semantic search)",
    "checkpoints": "conversation state per thread",
    "checkpoint_writes": "pending writes per thread",
    "checkpoint_blobs": "large message payloads",
    "store_migrations": "store schema version",
    "checkpoint_migrations": "checkpointer schema version",
}

_pool: LakebasePool | None = None


def _get_pool() -> LakebasePool:
    global _pool
    if _pool is None:
        _pool = LakebasePool(project=LB_PROJECT, branch=LB_BRANCH, schema=MEMORY_SCHEMA)
    return _pool


def _query_stats() -> dict:
    pool = _get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = %s ORDER BY tablename",
                (MEMORY_SCHEMA,),
            )
            # The pool may use a dict row_factory; normalize either shape to values.
            names = [(r["tablename"] if isinstance(r, dict) else r[0]) for r in cur.fetchall()]
            tables = []
            for name in names:
                cur.execute(f'SELECT count(*) AS n FROM "{MEMORY_SCHEMA}"."{name}"')
                row = cur.fetchone()
                rows = row["n"] if isinstance(row, dict) else row[0]
                tables.append({
                    "schema": MEMORY_SCHEMA,
                    "name": name,
                    "rows": rows,
                    "role": TABLE_ROLE.get(name, ""),
                })
    return {
        "available": True,
        "project": LB_PROJECT,
        "branch": LB_BRANCH,
        "schema": MEMORY_SCHEMA,
        "tables": tables,
    }


async def table_stats() -> dict:
    # LakebasePool.connection() is synchronous; run it off the event loop.
    return await asyncio.to_thread(_query_stats)
