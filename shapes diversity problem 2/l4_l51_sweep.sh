#!/usr/bin/env bash
# L4's measurement (no code) plus L5.1's checkpointing sweep.
#
# L4 asks whether step time is flat as batch grows. It has to be flat for the
# per-sample-captioning change to be worth anything: the loader forms batches
# per (prompt, neg_prompt, size) and the trainer encodes ONE prompt per batch,
# so images with unique captions cannot share one, and the fix is only free if
# a bigger batch costs no more per step.
#
# `non-square` has exactly one distinct prompt -- every prewarm line in this
# task's runs says "1 distinct prompt(s)" -- so its batches form at any batch
# size and it is the right dataset for the question. That also means L4's
# measurement does NOT test heterogeneous captions; it establishes the premise
# those captions would rely on.
#
# Serial, one run at a time: two concurrent runs on this card cause DEVICE_LOST.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$ROOT" || exit 1
[ -f scripts/hw_validate.py ] || { echo "FATAL: $ROOT is not the repo root" >&2; exit 1; }

PY=/home/okolenmi/comfy/venv/bin/python
export COMFY_DIR=/home/okolenmi/comfy/ComfyUI
STEPS=${STEPS:-150}
DATASET=${DATASET:-non-square}

run() {  # label extra-args...
  local label=$1; shift
  if [ -f "runs/hw_validation/$label/summary.json" ]; then
    echo "== $label exists, skipping"; return 0
  fi
  echo "== $label"
  timeout 3000 "$PY" -u scripts/hw_validate.py managed \
    --label "$label" --dataset "$DATASET" --steps "$STEPS" --seed 1234 \
    --shape-bucket-multiple 32 --budget 11500 "$@" 2>&1 \
    | grep -Ev "Rusticl|^ +[0-9]+x[0-9]+ +->" | tail -3
  [ -f "runs/hw_validation/$label/summary.json" ] \
    || { echo "!! $label produced no summary.json -- FAILED"; return 1; }
  "$PY" -c "
import json
s = json.load(open('runs/hw_validation/$label/summary.json'))
a = s.get('run_aggregates', {})
print(f\"   outcome={s['outcome']} steps={a.get('steps_recorded')} \"
      f\"steps/s={a.get('steps_per_sec_steady')} \"
      f\"peak_reserved={(a.get('per_step_peak_reserved_mb') or {}).get('max')}\")"
}

fail=0
# L4: step time vs batch size, at a shape count small enough that the
# compile cost is out of the way (3 buckets from bucketing x32).
run L4_b1 --batch 1 || fail=1
run L4_b2 --batch 2 || fail=1
run L4_b4 --batch 4 || fail=1
run L4_b8 --batch 8 || fail=1
# L5.1: what activation checkpointing costs, at a batch that fits.
run L5_1_ckpt_on  --batch 2 || fail=1
run L5_1_ckpt_off --batch 2 --no-checkpoint || fail=1
echo "== done (fail=$fail)"
