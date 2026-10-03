#!/bin/bash
# Does the install executor really install? Nothing here is simulated.
#
#   bash scripts/test_install_executor.sh
#
# `backend/tests/test_install.py` proves the executor's *rules* -- the
# refusals, the job states, the concurrency guard, the command line -- with
# a scripted fake pip. This proves the thing a fake cannot: that a venv is
# really created and pip really runs and the packages really land.
#
# Two packages, both pure-Python and a few kB, so the whole thing is about
# 15 MB and a handful of seconds. What is being tested is the plumbing, not
# the download -- a 2.5 GB torch wheel exercises the same code with a
# longer wait.
#
# Destructive to nothing: one temporary directory, removed on exit.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${GATE_PYTHON:-/home/okolenmi/comfy/venv/bin/python}"
WORK="$(mktemp -d /tmp/install-executor-test-XXXXXX)"
PORT="${PORT:-8791}"
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
    # The server first, or its venv vanishes under it.
    [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null
    sleep 1
    rm -rf "$WORK"
}
trap cleanup EXIT

# A project copy with no ComfyUI anywhere near it, so the wizard state is
# genuinely unconfigured -- an in-process container cannot be, because
# `paths` resolves the developer's own ComfyUI.
echo "== an unconfigured project, in a venv with only the server's packages =="
mkdir -p "$WORK/project"
cp -r "$REPO/backend" "$REPO/nodes" "$WORK/project/"
cp "$REPO"/*.py "$REPO/run_server.sh" "$WORK/project/"

"$PYTHON" -m venv "$WORK/server"
"$WORK/server/bin/pip" install --quiet --disable-pip-version-check \
    fastapi uvicorn python-multipart tomli_w packaging 2>/dev/null

check "a bare server venv exists" \
      "$([ -x "$WORK/server/bin/python" ] && echo 1 || echo 0)"
MISSING="$("$WORK/server/bin/python" -c "
import sys
for m in ('torch','numpy','safetensors','PIL'):
    try:
        __import__(m)
    except ImportError:
        print(m)
" | tr '\n' ',')"
echo "  training packages missing there: $MISSING"
check "the training stack really is absent, so there is something to install" \
      "$([ -n "$MISSING" ] && echo 1 || echo 0)"

echo
echo "== the server, on that interpreter =="
cd "$WORK/project"
BACKEND_DB_PATH="$WORK/backend.db" "$WORK/server/bin/python" -m backend.cli \
    --host 127.0.0.1 --port "$PORT" > "$WORK/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 40); do
    curl -s -m 2 "http://127.0.0.1:$PORT/api/v1/health" 2>/dev/null | grep -q '"ok"' && break
    sleep 1
done
check "it starts" \
      "$(curl -s -m 5 "http://127.0.0.1:$PORT/api/v1/health" | grep -q '"ok"' && echo 1 || echo 0)"
check "and is unconfigured, which is the only state the install gate allows" \
      "$(curl -s -m 10 "http://127.0.0.1:$PORT/api/v1/installer/state" | grep -q '"configured": false' && echo 1 || echo 0)"

echo
echo "== a refusal that must not write =="
curl -s -m 30 -X POST -H 'Content-Type: application/json' \
    -d '{"target":"comfy","packages":["torch"]}' \
    "http://127.0.0.1:$PORT/api/v1/installer/install" > "$WORK/refused.json"
check "installing torch into ComfyUI's venv is refused" \
      "$(grep -q 'never-install' "$WORK/refused.json" && echo 1 || echo 0)"
check "with a code the client can read" \
      "$(grep -q '"code": *"install_refused"' "$WORK/refused.json" && echo 1 || echo 0)"
check "and no venv was created for it" \
      "$([ ! -d "$WORK/project/venv" ] && echo 1 || echo 0)"

echo
echo "== a real install (this is the part a fake cannot check) =="
JOB="$(curl -s -m 60 -X POST -H 'Content-Type: application/json' \
    -d '{"target":"new","packages":["tomli_w","packaging"]}' \
    "http://127.0.0.1:$PORT/api/v1/installer/install" \
    | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["id"])' 2>/dev/null)"
check "a job id comes back immediately" "$([ -n "$JOB" ] && echo 1 || echo 0)"

for _ in $(seq 1 90); do
    STATE="$(curl -s -m 10 "http://127.0.0.1:$PORT/api/v1/installer/install/$JOB" \
        | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["state"])' 2>/dev/null)"
    case "$STATE" in succeeded|failed) break ;; esac
    sleep 2
done
echo "  final state: $STATE"
check "the job reaches succeeded" "$([ "$STATE" = "succeeded" ] && echo 1 || echo 0)"

echo
echo "== and the disk actually changed =="
check "a virtualenv was created at project/venv" \
      "$([ -x "$WORK/project/venv/bin/python" ] && echo 1 || echo 0)"
if [ -x "$WORK/project/venv/bin/python" ]; then
    "$WORK/project/venv/bin/python" -c "import tomli_w, packaging" 2>/dev/null
    check "and the installed packages are importable in it" \
          "$([ $? -eq 0 ] && echo 1 || echo 0)"
    check "the interpreter it recorded is the one it created" \
          "$(curl -s -m 10 "http://127.0.0.1:$PORT/api/v1/installer/install/$JOB" \
             | grep -q "$WORK/project/venv/bin/python" && echo 1 || echo 0)"
fi
check "no constraints file was left in the project" \
      "$([ ! -f "$WORK/project/constraints.txt" ] && echo 1 || echo 0)"
check "no stale constraint directories in the temp dir" \
      "$([ -z "$(find /tmp -maxdepth 1 -name 'distillation-install-*' -type d 2>/dev/null)" ] && echo 1 || echo 0)"

echo
echo "== a second install that DOES carry pins, into a target we own =="
# The successful install above carried no constraints, so it never made a
# constraints file and never exercised the cleanup of one. This one does:
# the pins are sent, so the port writes a file and must remove it after.
JOB2="$(curl -s -m 60 -X POST -H 'Content-Type: application/json' \
    -d '{"target":"new","packages":["tomli_w"],
         "constraints":["packaging==26.3"]}' \
    "http://127.0.0.1:$PORT/api/v1/installer/install" \
    | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["id"])' 2>/dev/null)"
for _ in $(seq 1 60); do
    STATE2="$(curl -s -m 10 "http://127.0.0.1:$PORT/api/v1/installer/install/$JOB2" \
        | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["state"])' 2>/dev/null)"
    case "$STATE2" in succeeded|failed) break ;; esac
    sleep 2
done
check "an install carrying pins also completes (got $STATE2)" \
      "$([ "$STATE2" = "succeeded" ] && echo 1 || echo 0)"
check "and the constraints file it wrote is gone afterwards" \
      "$([ -z "$(find /tmp -maxdepth 1 -name 'distillation-install-*' -type d 2>/dev/null)" ] && echo 1 || echo 0)"

echo
echo "== an unknown job is a 404, not a fabricated success =="
STATUS="$(curl -s -o /dev/null -w '%{http_code}' -m 10 \
    "http://127.0.0.1:$PORT/api/v1/installer/install/never-existed")"
check "404 for an id this process never issued (got $STATUS)" \
      "$([ "$STATUS" = "404" ] && echo 1 || echo 0)"

echo
if [ "$FAILURES" -eq 0 ]; then
    echo "INSTALL EXECUTOR: all checks passed"
    exit 0
fi
echo "INSTALL EXECUTOR: $FAILURES check(s) failed"
exit 1