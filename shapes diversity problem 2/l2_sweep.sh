#!/usr/bin/env bash
# L2: does shape bucketing cost quality, and which multiple is best?
#
# RE-RUN 2026-10-07 after the managed route's LossPhase was found to ignore the
# bucketing mask. Every run the first sweep produced trained on a plain mean
# over a canvas that is 13-25% padding, with the loss and gradient scaled by
# the valid fraction, so all five are void. They were moved aside rather than
# deleted, under /tmp/opencode/void_runs, because their *throughput* numbers
# are still valid -- the bug was in the loss, not the shapes or the launches.
#
# One measurement per configuration, serially: two concurrent runs on this card
# cause DEVICE_LOST, which is not a number anyone wants to learn twice.
#
# Each run does double duty:
#   * (b) first-sighting seconds, steps/s and pad fraction from the run itself
#     (steps.jsonl + the build-time pad report in console.log);
#   * (c) the fixed unpadded evaluation MSE in summary.json.
#
# L2_x8 and L2_x48 are in here too, not because L2 asked for them but because
# they are what separates "pad fraction" from "latent sides that are multiples
# of 16" -- a question the first sweep could not answer either way. x48 is all
# multiples of 16 (like x32) at 50.6% pad; x8 has the lowest pad at 4.8% but
# produces non-multiple-of-16 sides. See divisibility_sweep.sh for the
# reasoning.
set -u
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
    echo "== $label already exists, skipping"; return 0
  fi
  echo "== $label (bucket x$multiple, seed $seed, $STEPS steps)"
  timeout 3000 "$PY" -u scripts/hw_validate.py managed \
    --label "$label" --dataset "$DATASET" \
    --steps "$STEPS" --batch "$BATCH" --seed "$seed" \
    --shape-bucket-multiple "$multiple" \
    --holdout-batches "$HOLDOUT" --budget 11500 \
    2>&1 | grep -Ev "Rusticl|^ +[0-9]+x[0-9]+ +->" | tail -4
  # The pipe swallows the exit status, so a run that died of an OOM would look
  # like one that finished. Check the artefact instead of trusting the pipeline.
  if [ ! -f "runs/hw_validation/$label/summary.json" ]; then
    echo "!! $label produced no summary.json -- treating as FAILED"; return 1
  fi
  "$PY" -c "
import json
s = json.load(open('runs/hw_validation/$label/summary.json'))
h = s.get('holdout') or {}
a = s.get('run_aggregates', {})
print(f\"   outcome={s['outcome']} steps={a.get('steps_recorded')} \"
      f\"steps/s={a.get('steps_per_sec_steady')} \"
      f\"digest={h.get('digest')} mse={h.get('mse_mean')}\")"
}

fail=0
run L2_x0    0  1234 || fail=1
run L2_x0_b  0  4321 || fail=1
run L2_x16  16  1234 || fail=1
run L2_x24  24  1234 || fail=1
run L2_x32  32  1234 || fail=1
run L2_x8    8  1234 || fail=1
run L2_x48  48  1234 || fail=1
echo "== sweep done (fail=$fail)"
