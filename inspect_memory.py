"""
Peek at what the agent has stored in Lakebase — both memory types.

Usage:
    python inspect_memory.py                 # summarise all users
    python inspect_memory.py "Ada Lovelace"  # focus on one user_id

Long-term memory  -> fincoach.store          (facts, keyed by namespace = FACT_NS.user_id)
Short-term memory -> fincoach.checkpoints    (conversation state, keyed by thread_id)
"""
import os, sys
from databricks_ai_bridge.lakebase import LakebasePool

PROJECT = os.environ.get("LAKEBASE_PROJECT", "fincoach")
BRANCH = os.environ.get("LAKEBASE_BRANCH", "production")
SCHEMA = os.environ.get("MEMORY_SCHEMA", "fincoach")
user_filter = sys.argv[1] if len(sys.argv) > 1 else None

pool = LakebasePool(project=PROJECT, branch=BRANCH, schema=SCHEMA)
with pool.connection() as conn:
    conn.autocommit = True
    cur = conn.cursor()

    print("=" * 70)
    print("LONG-TERM MEMORY  (fincoach.store)")
    print("=" * 70)
    sql = "SELECT prefix, key, value, updated_at FROM store"
    if user_filter:
        sql += " WHERE prefix LIKE %s"
        cur.execute(sql + " ORDER BY prefix, updated_at", (f"%{user_filter}%",))
    else:
        cur.execute(sql + " ORDER BY prefix, updated_at")
    rows = cur.fetchall()
    if not rows:
        print("  (no facts stored)")
    last_ns = None
    for r in rows:
        ns = r["prefix"]                       # e.g. "finance_facts.Mughees Ahmed"
        if ns != last_ns:
            print(f"\n  namespace: {ns}")
            last_ns = ns
        v = r["value"] or {}
        print(f"    • [{v.get('category','?'):14}] {r['key']:22} = {v.get('content','')}")

    print("\n" + "=" * 70)
    print("SHORT-TERM MEMORY  (fincoach.checkpoints — conversation state per thread)")
    print("=" * 70)
    cur.execute("""
        SELECT thread_id, count(*) AS checkpoints, max(checkpoint_id) AS latest
        FROM checkpoints GROUP BY thread_id ORDER BY 2 DESC LIMIT 15
    """)
    threads = cur.fetchall()
    if not threads:
        print("  (no conversation threads)")
    for r in threads:
        print(f"    thread {r['thread_id'][:12]}…  {r['checkpoints']:>3} checkpoints saved")

pool.close()
print("\nTip: in the app UI, the same data is the 🧠 Long-term / 💬 This session / 🗄️ Lakebase tabs.")
