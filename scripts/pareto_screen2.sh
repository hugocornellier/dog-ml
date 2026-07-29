#!/bin/bash
# Second round of phase-1 head screens, queued behind the full taper run.
#
# The aggressive taper (128/96/64/48) is tracking ~0.2 behind the baseline
# through phase 2, so the likely outcome is that it is a trade rather than a win.
# That makes the interesting question no longer "does this one head work" but
# "is there ANY cheaper head that holds accuracy, or is the deconv head simply
# not over-parameterized?"
#
# Answering that needs points along the cost axis, not one point. Combined with
# the runs already done this gives five configurations spanning 15-57 ms of head
# cost, which is enough to state the shape of the frontier instead of a single
# anecdote.
#
# Phase 1 only (backbone frozen), which is the regime that isolates head
# capacity, ~1 hour each. Baseline reference: 10.5740.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# Do not contend with the full run.
while pgrep -f "pareto_full_run.sh" > /dev/null; do sleep 60; done

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
    print(f"SCREEN2 {tag}: no train_log.csv"); raise SystemExit
rows=[r for r in csv.reader(open(p)) if r and r[0]!="epoch"]
best=min(float(r[5]) if len(r)==7 else float(r[6]) for r in rows)
print(f"SCREEN2 {tag}: phase1_best_val_nme_iod={best:.4f} epochs={len(rows)} (baseline 10.5740)")
EOF
}

# Milder taper: keeps 128 through the two cheap low-res deconvs, thins only the
# two expensive high-res ones. ~32 ms in tf.lite Python vs 21 for the aggressive
# taper and 57 for the baseline head.
run taper_mild_128_128_96_64 --deconv-channels 128,128,96,64

# Widen the cheapest layer instead of thinning it. deconv_1 runs at 12x12, where
# capacity is nearly free (2.2 ms), and holds most of the head's parameters. If
# the deficit is about total head capacity rather than about width at high
# resolution, this recovers it at almost no latency cost.
run taper_wide1_192_96_64_48 --deconv-channels 192,96,64,48

echo "ALL SCREEN2 DONE"
