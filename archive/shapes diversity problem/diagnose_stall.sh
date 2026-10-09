#!/usr/bin/env bash
# Run a training while sampling which thread burns the CPU, and capture oneDNN's
# per-primitive JIT timings at the same time.
#
# Two questions, one run:
#   1. does the first sighting's cost show up as oneDNN `jit` time?  If the
#      total JIT seconds match the ~3.85 s per new shape, the stall is oneDNN
#      JIT and is parallelisable with threads. If JIT time is small, the cost
#      is somewhere else and optimising JIT would be optimising nothing.
#   2. which thread is busy -- ours (Python/main) or the runtime's (a UR/SYCL
#      worker)? That decides whether the fix is "do less" or "do it in parallel".
#
# ONEDNN_VERBOSE goes to stderr; hw_validate's own output goes to stdout, so the
# two are captured to different files and cannot interleave into nonsense.
set -u
cd /home/okolenmi/Desktop/B580-diffusion-training

OUT=/tmp/opencode/shapes
mkdir -p "$OUT"
LABEL=${1:-STALL_diag}
STEPS=${2:-40}

ONEDNN_VERBOSE=2 ONEDNN_PRIMITIVE_CACHE_CAPACITY=2048 \
  /home/okolenmi/comfy/venv/bin/python -u scripts/hw_validate.py main \
    --label "$LABEL" --dataset "non-square" --steps "$STEPS" --batch 2 \
    --checkpoint div_4.safetensors \
    > "$OUT/${LABEL}.stdout" 2> "$OUT/${LABEL}.onednn.log" &
TRAIN_PID=$!

# The python process is a child of the shell; find the one that is running the
# trainer by matching its command line, not by assuming $!.
sleep 20
SAMPLER_PID=""
for pid in $(pgrep -f "hw_validate.py main --label $LABEL"); do
  /home/okolenmi/comfy/venv/bin/python "archive/shapes diversity problem/thread_watch.py" \
      "$pid" --interval 0.2 --out "$OUT/${LABEL}.threads.csv" \
      > "$OUT/${LABEL}.threads.txt" 2>&1 &
  SAMPLER_PID=$!
  echo "sampling pid $pid (sampler $SAMPLER_PID)"
  break
done

wait "$TRAIN_PID"
TRAIN_RC=$?
if [ -n "$SAMPLER_PID" ]; then
  # thread_watch exits when the pid it watches disappears.
  wait "$SAMPLER_PID" 2>/dev/null
fi
echo "train exit=$TRAIN_RC"
echo "--- oneDNN verbose lines: $(wc -l < "$OUT/${LABEL}.onednn.log") ---"
