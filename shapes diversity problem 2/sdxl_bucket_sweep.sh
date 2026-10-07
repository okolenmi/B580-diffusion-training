#!/usr/bin/env bash
# The run that separates "padding costs quality" from "shapes outside SDXL's
# training buckets cost quality".
#
# Why x48 (queued in l2_sweep.sh) is NOT the answer to that question, and x64
# is. Both hypotheses fit the first five arms perfectly, because among
# {x16, x32, x24} pad fraction and in-distribution fraction happen to rank the
# arms the same way:
#
#   x16  3/7 SDXL buckets, 10.5% pad  -> indistinguishable from unbucketed
#   x32  3/3 SDXL buckets, 13.3% pad  -> indistinguishable from unbucketed
#   x24  0/5 SDXL buckets, 25.1% pad  -> measurably worse
#
# x48 breaks nothing: its shapes (384x768, 768x384, 768x768) are 0/3 SDXL
# buckets AND its pad is 50.6%, the highest of any multiple. It is bad on both
# counts, so a bad result identifies neither and a good result would be a
# surprise worth knowing.
#
# x64 separates them. Its shapes are 512x512, 512x1024, 1024x512 -- the same
# bucket family as x32's, and 3/3 if the 1024-px-long-side buckets count
# (SDXL's published set includes 1024x512; the strict 1024x1024 / 512x512 /
# 768x512 / 512x768 set gives 1/3). Its pad is 23.4%, within a point and a
# half of x24's 25.1%.
#
#   shapes matter -> x64 behaves like x32 (fine)
#   padding matters -> x64 behaves like x24 (worse)
#
# One caveat stated up front: 512x1024 and 1024x512 are 2x the area of x32's
# largest bucket, so peak memory rises. At batch 2 with 3 buckets this should
# still fit, but if it OOMs that is a result too -- it would mean the in-
# distribution shapes that help are also the expensive ones, which is itself
# the tradeoff.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$ROOT" || exit 1
[ -f scripts/hw_validate.py ] || { echo "FATAL: not the repo root" >&2; exit 1; }

PY=/home/okolenmi/comfy/venv/bin/python
export COMFY_DIR=/home/okolenmi/comfy/ComfyUI

run() {  # label multiple
  local label=$1 multiple=$2
  if [ -f "runs/hw_validation/$label/summary.json" ]; then
    echo "== $label exists, skipping"; return 0
  fi
  echo "== $label (bucket x$multiple)"
  timeout 3000 "$PY" -u scripts/hw_validate.py managed \
    --label "$label" --dataset "non-square" --steps 300 --batch 2 --seed 1234 \
    --shape-bucket-multiple "$multiple" --holdout-batches 16 --budget 11500 \
    2>&1 | grep -Ev "Rusticl|^ +[0-9]+x[0-9]+ +->" | tail -4
  [ -f "runs/hw_validation/$label/summary.json" ] \
    || { echo "!! $label produced no summary.json -- FAILED"; return 1; }
  "$PY" -c "
import json
s = json.load(open('runs/hw_validation/$label/summary.json'))
h = s.get('holdout') or {}; a = s.get('run_aggregates', {})
print(f\"   outcome={s['outcome']} steps/s={a.get('steps_per_sec_steady')} \"
      f\"peak={(a.get('per_step_peak_reserved_mb') or {}).get('max')} \"
      f\"digest={h.get('digest')} mse={h.get('mse_mean')}\")"
}

fail=0
run L2_x64 64 || fail=1
echo "== done (fail=$fail)"
