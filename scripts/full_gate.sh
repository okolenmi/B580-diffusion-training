#!/usr/bin/env bash
# Full regression gate: legacy routes (nodes/manager smoke tests;
# server/'s six retired with archive/ in M9) + the new backend suite +
# frontend syntax check.
#
#     scripts/full_gate.sh
#
# Lives in the repo (not /tmp -- scripts here survive reboots; the
# previous copy was lost to a /tmp clear and recreated 2026-10-01).
# The visual smoke (backend/tests/visual_smoke.py) is NOT part of this
# gate: it needs a live server on 8766 plus the Playwright venv.
#
# The interpreter is discovered, not hardcoded (docs 07 F-17): a path
# baked in made the gate unrunnable on any other machine, and silently
# wrong on this one after a venv move. run_tests.py already owns that
# discovery (this interpreter -> $VENV_PYTHON -> .env -> fail loudly), so
# the gate asks it instead of growing a second opinion.
set -euo pipefail
cd "$(dirname "$0")/.."

GATE_PYTHON="$("${PYTHON:-python3}" -c 'import run_tests; print(run_tests.resolve_interpreter())')"
echo "gate interpreter: $GATE_PYTHON"

echo "== legacy suites: nodes + manager =="
"$GATE_PYTHON" run_tests.py

echo "== backend suite =="
"$GATE_PYTHON" backend/tests/run_all.py

echo "== frontend module syntax =="
for f in $(find frontend/js -name '*.js'); do
  node --check "$f"
done
echo "node --check: all modules OK"

echo "FULL GATE: ALL GREEN"
