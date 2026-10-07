#!/usr/bin/env bash
# L2: does shape bucketing cost quality, and which multiple is best?
#
# One measurement per configuration, serially -- two concurrent runs on this
# card cause DEVICE_LOST, which is not a number anyone wants to learn twice.
#
# Each run does double duty:
#   * (b) first-sighting seconds, steps/s and pad fraction come from the run
#     itself (steps.jsonl + the build-time pad report in console.log);
#   * (c) the fixed unpadded holdout MSE in summary.json answers the quality
#     question a step time cannot.
#
# The holdout is built with bucketing OFF in every arm, so all of them score
# byte-identical inputs -- summary.json's holdout.digest proves it, and
# L2_x0_b is the noise control that says how large a difference is real.
set -u
# One level up from this script's own directory: the repo root. Resolved
# through `pwd` so the script works from any cwd, and checked, because a
# silently-wrong root produced five runs that all failed with "can't open
# file" and still exited 0 -- a measurement script that cannot fail is not
# one.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$ROOT" || exit 1
if [ ! -f scripts/hw_validate.py ]; then
  echo "FATAL: $ROOT does not look like the repo root (no scripts/hw_validate.py)" >&2
  exit 1
fi

PY=/home/okolenmi/comfy/venv/bin/python
export COMFY_DIR=/home/okolenmi/comfy/ComfyUI
STEPS=${STEPS:-300}
BATCH=${BATCH:-2}
HOLDOUT=${HOLDOUT:-16}
DATASET=${DATASET:-non-square}

run() {  # label multiple seed
  local label=$1 multiple=$2 seed=$3
  if [ -f "runs/hw_validation/$label/summary.json" ]; then
    echo "== $label already exists, skipping"
    return 0
  fi
  echo "== $label (bucket x$multiple, seed $seed, $STEPS steps)"
  timeout 3000 "$PY" -u scripts/hw_validate.py managed \
    --label "$label" --dataset "$DATASET" \
    --steps "$STEPS" --batch "$BATCH" --seed "$seed" \
    --shape-bucket-multiple "$multiple" \
    --holdout-batches "$HOLDOUT" --budget 11500 \
    2>&1 | grep -Ev "Rusticl|^ +[0-9]+x[0-9]+ +->" | tail -6
  # The pipe above swallows the exit status, so a run that died of an OOM
  # would look like a run that finished. Check the artefact the harness
  # writes instead of trusting the pipeline.
  if [ ! -f "runs/hw_validation/$label/summary.json" ]; then
    echo "!! $label produced no summary.json -- treating as FAILED"
    return 1
  fi
  "$PY" -c "
import json, sys
s = json.load(open('runs/hw_validation/$label/summary.json'))
h = s.get('holdout') or {}
print(f\"   outcome={s['outcome']} steps={s.get('run_aggregates',{}).get('steps_recorded')} \"
      f\"steps/s={s.get('run_aggregates',{}).get('steps_per_sec_steady')} \"
      f\"holdout_digest={h.get('digest')} holdout_mse={h.get('mse_mean')}\")
"
}

fail=0
run L2_x0    0  1234 || fail=1
run L2_x0_b  0  4321 || fail=1
run L2_x16  16  1234 || fail=1
run L2_x24  24  1234 || fail=1
run L2_x32  32  1234 || fail=1
echo "== sweep done (fail=$fail)"
