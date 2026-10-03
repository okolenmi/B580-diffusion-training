#!/bin/bash
# Does the suite pass on a machine that has never heard of this developer?
#
#   bash scripts/check_bare_checkout.sh
#
# Round-4 R4-03. The suite used to depend on the machine it ran on: three
# files passed here and failed on a fresh clone, because they resolved a
# ComfyUI directory and this checkout's `.env` names one. Reproduced on a
# `git archive` of HEAD before the fix:
#
#     test_config      RuntimeError: Cannot find ComfyUI directory
#     test_settings    RuntimeError: Cannot find ComfyUI directory
#     test_installer   5 of 64 checks failed
#
# with the same three files and no COMFY_DIR anywhere.
#
# `git archive HEAD | tar -x` is the important part: it is what a fresh
# clone looks like, so a `.env` that happens to be untracked -- and every
# local path in it -- cannot be what makes the tests pass. An
# `env -u COMFY_DIR` on this checkout would not be the same test, because
# `.env` would still be read.
#
# Slow (two pip-free venv-less copies and three suites) and therefore NOT
# part of full_gate.sh, for the same reason fuzz_api.py and
# test_install_executor.sh are not. Run it when touching support.py,
# paths.py, or anything that resolves a ComfyUI path.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${GATE_PYTHON:-/home/okolenmi/comfy/venv/bin/python}"
WORK="$(mktemp -d /tmp/bare-checkout-XXXXXX)"
FILES="${BARE_FILES:-test_config.py test_settings.py test_installer.py}"
FAILURES=0

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

check() {
    if [ "$2" = "1" ]; then
        echo "  ok    $1"
    else
        echo "  FAIL  $1"
        FAILURES=$((FAILURES + 1))
    fi
}

echo "== a bare checkout, from HEAD, with nothing of yours in it =="
git -C "$REPO" archive HEAD | tar -x -C "$WORK"
check "the archive extracted" "$([ -f "$WORK/run_server.sh" ] && echo 1 || echo 0)"

# Anything untracked is exactly what a fresh clone would not have. If any of
# it is present the test is not testing what it claims.
check "no .env in it" "$([ ! -f "$WORK/.env" ] && echo 1 || echo 0)"
check "no __pycache__ in it" \
      "$([ -z "$(find "$WORK" -name __pycache__ -type d 2>/dev/null | head -1)" ] && echo 1 || echo 0)"

# Working-tree changes are excluded by `git archive HEAD` on purpose. If a
# fix is uncommitted this check would test the old code, so say so rather
# than quietly reporting a pass.
if ! git -C "$REPO" diff --quiet -- backend/tests/support.py; then
    echo "  note  support.py has uncommitted changes; this check ran against"
    echo "        HEAD's version, not the one on disk."
fi

echo
echo "== the three files the review named, with no COMFY_DIR =="
for name in $FILES; do
    printf "  %-22s " "$name"
    output="$(cd "$WORK" && env -u COMFY_DIR "$PYTHON" "backend/tests/$name" 2>&1)"
    verdict="$(echo "$output" | grep -oE "SMOKE TEST: .*" | tail -1)"
    if echo "$verdict" | grep -q "ALL"; then
        echo "ok    $verdict"
    else
        echo "FAIL  ${verdict:-exited without a verdict}"
        echo "$output" | grep -E "Cannot find ComfyUI|Error|error:" | head -2 | sed 's/^/          /'
        FAILURES=$((FAILURES + 1))
    fi
done

echo
if [ "$FAILURES" -eq 0 ]; then
    echo "BARE CHECKOUT: all checks passed"
    exit 0
fi
echo "BARE CHECKOUT: $FAILURES check(s) failed"
exit 1