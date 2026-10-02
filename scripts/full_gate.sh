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

# Frontend unit tests (pure logic, no browser). Skipped with a notice
# when node is unavailable rather than failing the gate, matching how
# ruff is treated above -- but a *failing* test is never skipped.
if command -v node >/dev/null 2>&1; then
  echo "== frontend unit tests =="
  node --test frontend/tests
else
  echo "== frontend unit tests skipped (node not installed) =="
fi

# Unused imports, undefined names, syntax errors. The structure audit
# (docs 08 S-25) removed a pile of dead imports by hand; this keeps them
# from coming back, and it is the same check that found them. F + E9
# only -- the project does not adopt a style linter. Skipped with a
# notice when the interpreter has no ruff, so the gate still runs on a
# bare venv.
echo "== documentation links and citations =="
# Every markdown link and every doc path cited from source must
# resolve. Two dangling citations had already accumulated before this
# existed (a deleted tracking doc, an abbreviated path), and nothing
# renders these docs, so nothing else would notice.
"$GATE_PYTHON" scripts/check_doc_links.py --quiet

if "$GATE_PYTHON" -c 'import ruff' 2>/dev/null; then
  echo "== backend lint (F, E9) =="
  "$GATE_PYTHON" -m ruff check --select F,E9 backend/
else
  echo "== backend lint skipped (ruff not installed in the gate interpreter) =="
fi

# The wider ruff + mypy run, measured against scripts/quality_baseline.json
# (docs 08 N-11). The F,E9 check above stays because it is the subset that
# must be zero outright -- undefined names are always a bug, whereas a
# complexity score or a variance warning is a judgement call. This one
# fails on the count going *up* for any rule in any file, and says where
# it went down. A tool nobody runs is not a gate, so it runs here.
#
# Skipped with a notice when the gate interpreter has neither tool, for the
# same reason as above. mypy alone is optional: it is a dev dependency,
# not something the app needs to run.
echo "== quality baseline (ruff, mypy) =="
"$GATE_PYTHON" scripts/check_quality.py

echo "FULL GATE: ALL GREEN"
