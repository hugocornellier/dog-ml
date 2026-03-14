#!/bin/bash
# Overnight experiment marathon — run until stopped
# Started: $(date)
set -e

LOG="artifacts/overnight_log.txt"
echo "=== OVERNIGHT MARATHON STARTED: $(date) ===" | tee -a "$LOG"

# Step 1: Wait for 384px training to finish (it's already running)
echo "" | tee -a "$LOG"
echo ">>> Step 1: 384px training already running, will evaluate when done" | tee -a "$LOG"

# Check if 384px model exists (training already running in another process)
while [ ! -f "artifacts/tight_margin_384/best.keras" ]; do
    sleep 60
done
echo ">>> 384px training COMPLETE: $(date)" | tee -a "$LOG"

# Step 2: Run comprehensive evaluation
echo "" | tee -a "$LOG"
echo ">>> Step 2: Comprehensive evaluation of all models" | tee -a "$LOG"
echo ">>> Started: $(date)" | tee -a "$LOG"
python scripts/eval_all_models.py 2>&1 | tee -a "$LOG"
echo ">>> Eval COMPLETE: $(date)" | tee -a "$LOG"

# Step 3: Train 320px with cosine annealing + 300 epoch Phase 2
echo "" | tee -a "$LOG"
echo ">>> Step 3: 320px cosine annealing + 300ep Phase 2" | tee -a "$LOG"
echo ">>> Started: $(date)" | tee -a "$LOG"
python scripts/train_dog_face_landmarks.py \
    --experiment tight_margin_320_cosine \
    --out artifacts/tight_margin_320_cosine 2>&1 | tee -a "$LOG"
echo ">>> 320px cosine training COMPLETE: $(date)" | tee -a "$LOG"

# Step 4: Eval the new model
echo "" | tee -a "$LOG"
echo ">>> Step 4: Re-eval all models including 320_cosine" | tee -a "$LOG"
python scripts/eval_all_models.py 2>&1 | tee -a "$LOG"

# Step 5: Train DenseNet121 + per-landmark heatmap supervision
echo "" | tee -a "$LOG"
echo ">>> Step 5: DenseNet121 + per-landmark heatmap at 56x56" | tee -a "$LOG"
echo ">>> Started: $(date)" | tee -a "$LOG"
python scripts/train_dog_face_landmarks.py \
    --experiment densenet121_heatmap_56 \
    --out artifacts/densenet121_heatmap_56 2>&1 | tee -a "$LOG"
echo ">>> DenseNet121 heatmap COMPLETE: $(date)" | tee -a "$LOG"

# Step 6: Final comprehensive eval
echo "" | tee -a "$LOG"
echo ">>> Step 6: Final comprehensive evaluation" | tee -a "$LOG"
python scripts/eval_all_models.py 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== OVERNIGHT MARATHON FINISHED: $(date) ===" | tee -a "$LOG"
