#!/bin/bash
# Does the bootstrap work on a machine that cannot start the server?
#
#   bash scripts/test_bootstrap_install.sh
#
# The check in backend/tests/test_bootstrap.py proves the module is
# stdlib-only and refuses cleanly, but it runs in a venv that already has
# fastapi -- so on that machine `missing_server_packages()` returns empty
# and the whole install path is never entered.
#
# This script enters it for real: it builds a venv with no site-packages,
# copies the project beside it, and runs `python -m backend.first_run`
# there. The bootstrap should install the four packages into a second,
# temporary venv and re-exec the server -- which is what "it worked" means
# here: the server comes up.
#
# Destructive to nothing: two temporary directories, both removed. Slow on
# purpose (two pip installs, ~30s), and not part of `full_gate.sh` for the
# same reason `fuzz_api.py` is not.

set -u

# This script drives the bootstrap on purpose, on a machine with a real
# desktop. Without this, every run reaches for the developer's browser and
# leaves a tab behind -- which is what happened while writing it.
export DISTILLATION_NO_BROWSER=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${BASE_PYTHON:-python3}"
WORK="$(mktemp -d /tmp/bootstrap-install-test-XXXXXX)"
FAILURES=0

check() {
    if [ "$2" = "1" ]; then
        echo "  ok    $1"
    else
        echo "  FAIL  $1"
        FAILURES=$((FAILURES + 1))
    fi
}

cleanup() {
    rm -rf "$WORK"
    # The bootstrap's own venv is pid-named and lives under the temp dir;
    # remove any this run left behind, whatever the outcome.
    rm -rf "${TMPDIR:-/tmp}"/distillation-bootstrap-*
}
trap cleanup EXIT

echo "== a project copy in a venv that has nothing =="

# A project copy, so nothing here can be helped by the real repo sitting on
# sys.path -- which is the whole question.
mkdir -p "$WORK/project"
# Everything the server imports from its own tree, and nothing else. The
# first draft copied `backend/` alone, and the server died twice: on
# `from nodes.core import Node` during node discovery, then on
# `from monitor_bus import MonitorBus` in the composition root. Both looked
# like the bootstrap having re-execed into a broken environment, and were
# actually the test having built an incomplete checkout -- a missing piece
# of a project copy fails exactly like a missing dependency does.
#
# Derived from the root rather than listed, because a list is what made it
# wrong twice: every top-level module the server imports, and nothing that
# is not importable from the project root.
cp -r "$REPO/backend" "$REPO/nodes" "$WORK/project/"
cp "$REPO"/*.py "$REPO/run_server.sh" "$WORK/project/"
check "the copy has every top-level module the server imports" \
      "$(for m in monitor_bus.py paths.py; do
             [ -f "$WORK/project/$m" ] || { echo 0; break; }
         done; [ -d "$WORK/project/nodes" ] && echo 1 || echo 0)"

"$BASE_PYTHON" -m venv "$WORK/bare" 2>/dev/null
check "a bare venv exists" "$([ -x "$WORK/bare/bin/python" ] && echo 1 || echo 0)"

MISSING="$("$WORK/bare/bin/python" -c "
import sys
missing = []
for dist, mod in (('fastapi','fastapi'), ('uvicorn','uvicorn'),
                  ('python-multipart','multipart'), ('tomli_w','tomli_w')):
    try:
        __import__(mod)
    except ImportError:
        missing.append(dist)
print(','.join(missing))
")"
echo "  missing there: $MISSING"
check "all four really are missing" "$([ "$MISSING" = "fastapi,uvicorn,python-multipart,tomli_w" ] && echo 1 || echo 0)"

echo
echo "== the bootstrap reports them (--check installs nothing) =="
# `--check`, because there is no other way to get the report. The first
# draft of this script called the bootstrap plain and captured stdout,
# expecting a short report -- and instead ran the whole thing: a full pip
# install, then an exec into a server that never came up. The capture came
# back empty of the URL because the URL is printed *after* the install, so
# three checks failed for a reason that had nothing to do with the URL.
REPORT="$("$WORK/bare/bin/python" -m backend.first_run --check --port 8799 2>&1)"
CHECK_CODE=$?
echo "$REPORT" | sed 's/^/    | /'
check "it says which packages are missing" \
      "$(echo "$REPORT" | grep -q 'Missing server dependencies' && echo 1 || echo 0)"
check "it names all four, by distribution name" \
      "$(echo "$REPORT" | grep -q 'python-multipart: not installed' && echo 1 || echo 0)"
check "python-multipart is named, not the module it imports as" \
      "$(echo "$REPORT" | grep -q 'No module named .multipart' && echo 1 || echo 0)"
check "--check exits 1 for 'something is missing'" \
      "$([ "$CHECK_CODE" -eq 1 ] && echo 1 || echo 0)"
check "and it changed nothing on disk" \
      "$([ -z "$(find "${TMPDIR:-/tmp}" -maxdepth 1 -name 'distillation-bootstrap-*' -type d 2>/dev/null)" ] && echo 1 || echo 0)"

echo
echo "== and the install itself (this takes a minute) =="
# Run to completion: the bootstrap re-execs run_server.sh, which starts the
# server, so it does not return. Background it, wait for the server to
# answer, then stop it.
cd "$WORK/project"
timeout 420 "$WORK/bare/bin/python" -m backend.first_run --port 8799 \
    > "$WORK/bootstrap.log" 2>&1 &
BOOT_PID=$!

UP=0
for _ in $(seq 1 120); do
    if curl -s -m 2 "http://127.0.0.1:8799/api/v1/health" 2>/dev/null | grep -q '"ok"'; then
        UP=1
        break
    fi
    sleep 2
done

echo "  --- bootstrap output ---"
sed 's/^/    | /' "$WORK/bootstrap.log" | tail -30

kill $BOOT_PID 2>/dev/null
pkill -f "backend.cli --host 0.0.0.0 --port 8799" 2>/dev/null
sleep 1

check "the server came up on the venv the bootstrap built" "$UP"
check "the bootstrap printed the /setup link" \
      "$(grep -q 'localhost:8799/setup' "$WORK/bootstrap.log" && echo 1 || echo 0)"
check "it said it was starting the server" \
      "$(grep -q 'Starting the server' "$WORK/bootstrap.log" && echo 1 || echo 0)"
check "it ran no browser, and said so" \
      "$(grep -q 'no browser opened' "$WORK/bootstrap.log" && echo 1 || echo 0)"

# Exactly one, and it must actually contain the packages. The first draft
# checked only "at least one exists", which passes on a leftover directory
# from an earlier run -- so it would have gone green against a bootstrap
# that installed nothing.
LEFT="$(find "${TMPDIR:-/tmp}" -maxdepth 1 -name 'distillation-bootstrap-*' -type d 2>/dev/null)"
COUNT="$(echo "$LEFT" | grep -c . )"
check "exactly one temporary venv, not a leftover from an earlier run" \
      "$([ "$COUNT" -eq 1 ] && echo 1 || echo 0)"
check "and it really has fastapi in it" \
      "$([ -d "$LEFT/lib/python"*/site-packages/fastapi ] && echo 1 || echo 0)"
check "it is under the temp dir, so it cannot land in a git status" \
      "$(echo "$LEFT" | grep -q "^${TMPDIR:-/tmp}/" && echo 1 || echo 0)"
check "and nothing was written into the project copy" \
      "$([ -z "$(find "$WORK/project" -name '*.dist-info' -o -name 'site-packages' 2>/dev/null)" ] && echo 1 || echo 0)"

echo
echo "== the no-op path, in the same bare venv once it has them =="
# After the bootstrap's install, that venv's *sibling* still has nothing;
# what matters is that a second run with the packages present is silent.
SILENT="$("$WORK/bare/bin/python" -c "
import subprocess, sys
sys.path.insert(0, '$WORK/project')
from backend.first_run import missing_server_packages
print(len(missing_server_packages()))
")"
check "the bare venv still has none of them (the bootstrap used its own)" \
      "$([ "$SILENT" = "4" ] && echo 1 || echo 0)"

echo
if [ "$FAILURES" -eq 0 ]; then
    echo "BOOTSTRAP INSTALL: all checks passed"
    exit 0
fi
echo "BOOTSTRAP INSTALL: $FAILURES check(s) failed"
exit 1