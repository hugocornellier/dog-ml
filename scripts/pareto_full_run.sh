#!/bin/bash
# Full two-phase run for the cheap-head candidate, queued behind the screens.
#
# Phase-1 screen put taper_128_96_64_48 at 10.7152 against the baseline's 10.5740,
# a 0.141 deficit. The baseline then gained 2.01 in phase 2 (10.574 -> 8.564); if
# the taper gains only the same amount it lands at ~8.71, which is a regression and
# not shippable. But the taper converges slower -- it was still improving when
# phase 1 hit its 100-epoch cap, where the baseline had already early-stopped at
# 87 -- so the deficit is plausibly a schedule artifact rather than a capacity
# ceiling.
#
# Training cost is not one of the three Pareto axes (size, accuracy, latency), so
# the cheaper head is given a longer phase 2 (600 epochs vs 400) to see whether it
# can reach the baseline's accuracy. If it does, the result is a genuine win: same
# size, same accuracy, ~2.4x faster. If it plateaus above 8.566, it is a trade and
# gets reported as one.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# Wait for any in-flight screen run to finish so the two do not contend for the GPU.
while pgrep -f "pareto_screen.sh" > /dev/null; do sleep 30; done

TAG=taper_full
echo "=== $TAG starting $(date) ==="
TF_CPP_MIN_LOG_LEVEL=2 $PY scripts/train_dog_face_landmarks.py \
  --experiment small_v3large_384_long \
  --deconv-channels 128,96,64,48 \
  --finetune-epochs 600 \
  --out-dir "artifacts/pareto/$TAG" \
  > "artifacts/pareto/${TAG}.log" 2>&1
echo "$TAG exit=$? $(date)"
tail -3 "artifacts/pareto/${TAG}.log"
echo "FULL RUN DONE"
