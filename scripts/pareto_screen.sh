#!/bin/bash
# Phase-1-only accuracy screens for the cheap-head candidates.
#
# Phase 1 freezes the backbone and trains the head alone, so it is exactly the
# right test for "is this head still expressive enough" -- and at ~1 hour per
# candidate it is 6x cheaper than the full two-phase run. The baseline
# small_v3large_384_long reached val NME_IOD 10.574 in phase 1 (epoch 80 of 87
# before early stopping); a candidate has to match that to be worth the full run.
#
# Both candidates below land at ~21 ms in the static-shape export, versus 56.8 ms
# for the uniform-128 baseline head, so this is a matched-latency comparison of
# where to spend the budget: thin channels at 192^2 heatmaps, or full channels at
# 96^2.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run () {
  local tag="$1"; shift
  echo "=== $tag ==="
  TF_CPP_MIN_LOG_LEVEL=2 $PY scripts/train_dog_face_landmarks.py \
    --experiment small_v3large_384_long \
    --skip-finetune \
    --out-dir "artifacts/pareto/$tag" \
    "$@" > "artifacts/pareto/${tag}.log" 2>&1
  echo "$tag exit=$?"
  $PY - "$tag" <<'EOF'
import csv, sys
from pathlib import Path
tag = sys.argv[1]
p = Path("artifacts/pareto")/tag/"train_log.csv"
if not p.exists():
    print(f"SCREEN {tag}: no train_log.csv"); raise SystemExit
rows=[r for r in csv.reader(open(p)) if r and r[0]!="epoch"]
best=min(float(r[5]) if len(r)==7 else float(r[6]) for r in rows)
print(f"SCREEN {tag}: phase1_best_val_nme_iod={best:.4f} epochs={len(rows)} (baseline 10.5740)")
EOF
}

run taper_128_96_64_48 --deconv-channels 128,96,64,48
run d3_128             --num-deconv-layers 3
echo "ALL SCREENS DONE"
