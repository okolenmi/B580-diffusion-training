#!/usr/bin/env bash
# The batch of real-hardware runs behind docs/known-issues/pending-testing.md's
# five entries (plus the managed-route perf question from
# docs/design/resources-controller/09-trainer-integration-and-vram-safety.md).
# One process per experiment, sequential (single GPU), continues past failures
# -- an OOM (exit 2) or a strict-mode raise is a *result*, not a driver error.
#
#   ./scripts/hw_validation_batch.sh            # everything
#   ./scripts/hw_validation_batch.sh A_after    # only labels given as args
#
# Per-run output: runs/hw_validation/<label>/{summary.json,steps.jsonl,console.log}

set -u
cd "$(dirname "$0")/.."

PYTHON="${VENV_PYTHON:-$(grep -oP '(?<=^VENV_PYTHON=).*' .env 2>/dev/null || true)}"
PYTHON="${PYTHON:-$(command -v python)}"
RUNNER="scripts/hw_validate.py"
OUT_ROOT="runs/hw_validation"

# label            route     dataset       steps  extra-env/args
EXPERIMENTS=(
  "A_after|main|1024|40|--budget 11500"
  "A_before|main|1024|40|--budget 11500|HW_DISABLE_ATTENTION_CKPT=1"
  "B_ratchet|main|non-square|60|--budget 11500"
  "C_pressure|main|1image|30|--budget 2500"
  "C_strict|main|1image|10|--budget 2500 --strict"
  "D_managed|managed|1024|40|--budget 11500"
  "E_managed_nonsq|managed|non-square|60|--budget 8000 --profile"
)

FILTER=("$@")
declare -A RESULTS

for spec in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r label route dataset steps extra env_assign <<<"$spec"
  if [ ${#FILTER[@]} -gt 0 ]; then
    keep=""
    for f in "${FILTER[@]}"; do [ "$label" = "$f" ] && keep=1; done
    [ -z "$keep" ] && continue
  fi

  echo
  echo "################################################################"
  echo "## $label  ($route / $dataset / $steps steps / $extra ${env_assign:+/ $env_assign})"
  echo "################################################################"
  mkdir -p "$OUT_ROOT/$label"
  if [ -n "${env_assign:-}" ]; then
    env "$env_assign" "$PYTHON" "$RUNNER" "$route" \
        --label "$label" --dataset "$dataset" --steps "$steps" $extra \
        2>&1 | tee "$OUT_ROOT/$label/console.log"
  else
    "$PYTHON" "$RUNNER" "$route" \
        --label "$label" --dataset "$dataset" --steps "$steps" $extra \
        2>&1 | tee "$OUT_ROOT/$label/console.log"
  fi
  RESULTS[$label]=${PIPESTATUS[0]}
done

echo
echo "=================== BATCH SUMMARY ==================="
for label in "${!RESULTS[@]}"; do
  rc=${RESULTS[$label]}
  case $rc in
    0) verdict="ok/strict_raise" ;;
    2) verdict="OOM (a result -- read the summary)" ;;
    *) verdict="ERROR (exit $rc)" ;;
  esac
  echo "  $label: $verdict"
  [ -f "$OUT_ROOT/$label/summary.json" ] && \
    grep -o '"outcome": "[^"]*"' "$OUT_ROOT/$label/summary.json" | head -1 | sed 's/^/      /'
done
