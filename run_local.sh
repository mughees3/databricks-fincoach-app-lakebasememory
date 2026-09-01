#!/usr/bin/env bash
# Run Fin Coach locally against a live Lakebase instance.
# Prereqs: `databricks auth login` for the profile in .env, and the venv created
# via:  uv venv --python 3.12 .venv && uv pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a
source .venv/bin/activate
echo "Fin Coach -> http://localhost:${PORT:-8000}  (profile: $DATABRICKS_CONFIG_PROFILE, lakebase: $LAKEBASE_PROJECT)"
exec python server.py
