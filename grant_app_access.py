"""
Grant the deployed app's service principal access to the Lakebase memory schema.

Why this is needed
------------------
A Databricks App runs as its own service principal (SP). Lakebase maps that SP
to a Postgres role named after the SP's *application id*. When the memory schema
and tables were first created by a human (e.g. during local development), that
human owns them and the app SP cannot read or write until it is granted access.

Run this ONCE after the first deploy (and re-run if you recreate the schema):

    # find the app's service principal client id:
    databricks apps get fincoach -p PROFILE -o json | jq -r .service_principal_client_id

    # then grant it:
    APP_SP=<that-client-id> python grant_app_access.py

Alternatively, if the app SP creates the schema itself on first run (because it
holds CAN_CONNECT_AND_CREATE and no human pre-created it), this step is not
required — the SP owns everything it creates.
"""

import os
import sys
from databricks_ai_bridge.lakebase import LakebasePool

PROJECT = os.environ.get("LAKEBASE_PROJECT", "fincoach")
BRANCH = os.environ.get("LAKEBASE_BRANCH", "production")
SCHEMA = os.environ.get("MEMORY_SCHEMA", "fincoach")
APP_SP = os.environ.get("APP_SP") or (sys.argv[1] if len(sys.argv) > 1 else None)

if not APP_SP:
    sys.exit("Set APP_SP=<service principal application id> (see docstring).")

STATEMENTS = [
    f'GRANT USAGE, CREATE ON SCHEMA "{SCHEMA}" TO "{APP_SP}"',
    f'GRANT ALL ON ALL TABLES IN SCHEMA "{SCHEMA}" TO "{APP_SP}"',
    f'GRANT ALL ON ALL SEQUENCES IN SCHEMA "{SCHEMA}" TO "{APP_SP}"',
    # future objects created by the schema owner become usable by the app too
    f'ALTER DEFAULT PRIVILEGES IN SCHEMA "{SCHEMA}" GRANT ALL ON TABLES TO "{APP_SP}"',
    f'ALTER DEFAULT PRIVILEGES IN SCHEMA "{SCHEMA}" GRANT ALL ON SEQUENCES TO "{APP_SP}"',
]

pool = LakebasePool(project=PROJECT, branch=BRANCH)
with pool.connection() as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        for stmt in STATEMENTS:
            cur.execute(stmt)
            print("✓", stmt)
pool.close()
print(f"\nGranted app SP '{APP_SP}' full access to schema '{SCHEMA}' on {PROJECT}/{BRANCH}.")
