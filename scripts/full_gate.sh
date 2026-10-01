#!/usr/bin/env bash
# Full regression gate: legacy routes (nodes/server/manager smoke tests)
# + the new backend suite + frontend syntax check.
#
#     scripts/full_gate.sh
#
# Lives in the repo (not /tmp -- scripts here survive reboots; the
# previous copy was lost to a /tmp clear and recreated 2026-10-01).
# The visual smoke (backend/tests/visual_smoke.py) is NOT part of this
# gate: it needs a live server on 8766 plus the Playwright venv.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV_PYTHON="${VENV_PYTHON:-/home/okolenmi/comfy/venv/bin/python}"

echo "== legacy suites: nodes + server + manager =="
"$VENV_PYTHON" run_tests.py

echo "== backend suite =="
"$VENV_PYTHON" backend/tests/run_all.py

echo "== frontend module syntax =="
for f in $(find frontend/js -name '*.js'); do
  node --check "$f"
done
echo "node --check: all modules OK"

echo "FULL GATE: ALL GREEN"
