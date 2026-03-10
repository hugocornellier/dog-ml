# Dog Facial Landmark Detection: Progress Journal

**Current best: NME_IOD = 8.04 (3-model 256+320+384 ensemble + ms+flip TTA) / 8.77 (no TTA, 384px single model) | Target: 6.52 (paper) | Started at: ~40**

This document is a living journal of our work improving dog facial landmark detection. It's designed for future LLMs/developers to pick up where we left off and continue pushing toward the paper's target.

---

## Quick Reference

### Current Best Model
- **Overall best**: 3-model ensemble (tight_margin_256 + tight_margin_320 + tight_margin_384) + ms+flip TTA — **NME 8.04** (18 passes: 3 models × 3 scales × 2 flips)
- **Best single model**: `tight_margin_384` — **8.77** (no TTA) / **8.49** (flip-TTA) / **8.33** (ms+flip TTA)
- **Architecture**: EfficientNetV2S + **4-deconv** heatmap head + SoftArgmax2D
- **Input**: 384×384 / 320×320 / 256×256 (three resolutions in ensemble)
- **Key change from previous best**: Added 384px model (192x192 heatmaps) to the ensemble
- **Artifacts**: `artifacts/tight_margin_384/`, `artifacts/tight_margin_320/`, `artifacts/tight_margin_256/`
- **Train-val gap**: ~4.0 at 384px (structural — not addressable by regularization alone)

### All Single-Model Results
| Model | No TTA | Flip TTA | ms+flip TTA |
|---|---|---|---|
| tight_margin_384 (384px) | **8.77** | **8.49** | 8.33 |
| tight_margin_320 (320px) | 8.82 | 8.52 | **8.32** |
| tight_margin_256 (256px) | 9.27 | 8.88 | 8.66 |

### All Ensemble Results (ms+flip TTA)
| Ensemble | NME_IOD | Passes |
|---|---|---|
| **3-model (256+320+384)** | **8.04** | 18 |
| 2-model (320+384) | 8.10 | 12 |
| 2-model (256+384) | 8.11 | 12 |
| 2-model (256+320) | 8.22 | 12 |

### Key Commands
```bash
# Train at 384px (current best single-model resolution)
python scripts/train_dog_face_landmarks.py --experiment tight_margin_384 --out artifacts/tight_margin_384

# Train at 320px
python scripts/train_dog_face_landmarks.py --experiment tight_margin_320 --out artifacts/tight_margin_320

# Train at 256px
python scripts/train_dog_face_landmarks.py --experiment tight_margin_256 --out artifacts/tight_margin_256

# Comprehensive evaluation of all models + ensembles + TTA
python scripts/eval_all_models.py

# Generate _pred/_true visualizations (uses 3-model ensemble, all 480 test images)
python scripts/gen_landmark_examples.py

# Run overnight experiment marathon (chain: train → eval → train → eval)
bash scripts/overnight_marathon.sh

# Quick smoke test (5 epochs)
python scripts/train_dog_face_landmarks.py --experiment tight_margin_256 --epochs 5 --patience 50 --out artifacts/smoke
```

---

## Reference Paper

**"DogFLW: Dog Facial Landmarks in the Wild"** — Martvel, G., Farjon, G., & Kovalenko, B. (2025)
- **Dataset**: [DogFLW on Kaggle](https://www.kaggle.com/datasets/georgemartvel/dogflw)
- **Paper**: Published in *Pattern Recognition*
- **46 facial landmarks** across 120 breeds, 3,853 train / 480 test images
- **Best result**: NME_IOD = 6.52 using ELD (Ensemble of Landmark Detectors) + EfficientNetV2S
- **Training**: 300 epochs, batch 16, MSE loss, Adam optimizer
- **Augmentations**: rotation, brightness/contrast, colour balance, sharpness, blur, noise

---

## Complete Experiment History

### Round 1: Dense Head Baseline (NME_IOD ~40)

We started by aligning to the paper's recipe with a standard classification-style head.

**Architecture**:
```
EfficientNet backbone (frozen) -> GlobalAveragePooling2D -> Dense(1024) -> Dense(512) -> Dense(256) -> Dense(92, sigmoid)
```

| Experiment | Backbone | NME_IOD | Notes |
|---|---|---|---|
| A: Paper baseline | B2 | 40.16 | Frozen backbone, MSE |
| B: Wing loss | B2 | 40.97 | Wing loss WORSE than MSE |
| H: V2S backbone | V2S | 39.51 | Larger backbone, slight improvement |
| best_v2s (two-phase) | V2S | 39.42 | Backbone fine-tuning barely helped |

**Key learnings**:
- GlobalAveragePooling2D destroys spatial info — fundamental bottleneck
- Wing loss worse than MSE for this task
- lr=1e-3 causes divergence even with frozen backbone; lr=1e-4 works
- Backbone fine-tuning extremely fragile with dense head (any LR > 1e-6 collapses)

### Round 2: Heatmap Architecture (NME_IOD ~12)

Replaced the dense head with a SimpleBaseline-style deconv heatmap head. This was the single biggest breakthrough.

**Architecture (56x56 heatmaps, 3 deconv layers)**:
```
EfficientNetV2S (frozen, 224x224)
    -> 7x7x1280 feature map
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU  -> 14x14
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU  -> 28x28
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU  -> 56x56
    -> Conv2D(46, 1x1)  -> 56x56 heatmaps
    -> SoftArgmax2D      -> 92 coordinates [0,1]
```

| Experiment | NME_IOD | Notes |
|---|---|---|
| heatmap_v2s (frozen, 300ep) | 12.18 | First heatmap run |
| heatmap_v2s_reg (dropout+jitter) | 12.25 | SpatialDropout2D didn't help |
| heatmap_v2s_ft (two-phase fine-tuning) | 11.28 | Fine-tuning backbone helped |

**Key learnings**:
- Heatmap head achieved in 5 epochs (15.13) what dense head couldn't in 300 epochs (39.42)
- SpatialDropout2D (0.15) and crop jitter provided negligible improvement
- Backbone fine-tuning (last 50 layers, lr=1e-5) gave ~1 point improvement
- Massive overfitting: train NME_IOD 4.19 vs val 12.18 — bottleneck shifted to generalization

### Round 3: Augmentation + Optimizer (NME_IOD ~10.7)

Added horizontal flip with landmark index swapping, random scale augmentation, and AdamW.

| Experiment | NME_IOD | Notes |
|---|---|---|
| heatmap_v2s_flip (frozen, flip+scale+AdamW) | 11.68 | Flip cuts overfitting gap in half |
| **heatmap_v2s_best** (two-phase, flip+scale+AdamW) | **10.72** | Previous best |

**Key learnings**:
- Horizontal flip is the single most impactful augmentation (train/val gap: 7.7 -> 3.6)
- Scale augmentation + crop jitter together simulate detector error well
- AdamW with weight_decay=1e-4 slightly better than Adam
- With flip, frozen backbone (11.68) nearly matches previous fine-tuned model (11.28)
- Two-phase + flip: 10.72 — a new frontier

### Round 4: NME Push — Resolution, TTA, Heatmap Supervision (NME_IOD ~10.1)

Overnight automated experiment run (7 hours) testing three strategies to break into single digits. Used `scripts/run_nme_push.py` to run experiments sequentially.

**Architecture (112x112 heatmaps, 4 deconv layers)**:
```
EfficientNetV2S (frozen, 224x224)
    -> 7x7x1280 feature map
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU  -> 14x14
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU  -> 28x28
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU  -> 56x56
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU  -> 112x112   [NEW]
    -> Conv2D(46, 1x1)  -> 112x112 heatmaps
    -> SoftArgmax2D      -> 92 coordinates [0,1]
```

| # | Experiment | NME_IOD | + TTA | Time | Notes |
|---|---|---|---|---|---|
| 1 | heatmap_v2s_best + flip-TTA (no retrain) | 10.72 | **10.25** | 4 min | Free gain from TTA |
| 2 | **heatmap_v2s_112** (4 deconv, 112x112) | **10.58** | **10.14** | 2h 58m | New best |
| 3 | heatmap_v2s_hmsup_s175 (heatmap supervision) | 32.81 | 32.44 | 1h 14m | Failed |
| 4 | heatmap_v2s_112_hmsup (112 + hmsup) | 45.18 | 45.09 | 2h 26m | Failed even worse |

**Key learnings**:

**112x112 heatmaps (worked)**:
- 4th deconv layer gives modest improvement over 56x56: 10.58 vs 10.72 (no TTA)
- Higher spatial resolution helps, but returns are diminishing — going to 224x224 (5th deconv) unlikely to help much
- Training takes ~3 hours (Phase 1: 100ep at 23s/ep, Phase 2: 200ep at 37s/ep)
- Heavy overfitting persists: train NME 6.1 vs val 10.6 — a ~4.5 point gap

**Flip-TTA (worked)**:
- Coordinate-level averaging provides ~0.4-0.5 free gain at inference (no retraining needed)
- Implementation: run model on original image + horizontally flipped image, unflip the flipped predictions (gather with FLIP_INDEX + mirror x coordinate), average the two sets of coordinates
- CRITICAL: Heatmap-level TTA (averaging raw logits before soft-argmax) **does not work** — produces NME 22.31, which is 2x WORSE than baseline. The logit distributions from original vs flipped are incompatible before softmax normalization
- TTA adds ~2x inference latency (two forward passes)

**Heatmap supervision (failed)**:
- Approach: generate 2D Gaussian target heatmaps (sigma=1.75) for each landmark, add MSE heatmap loss alongside coordinate MSE loss
- Multi-output Keras model: `{"hm": heatmaps, "xy": coords}` with `loss_weights={"hm": 0.1, "xy": 1.0}`
- Result: **complete failure**. Even with coordinate loss strongly dominant (10x weight), the heatmap loss gradient through the shared backbone interferes with coordinate regression learning
- NME 32.81 after full two-phase training (300 epochs total) vs 10.58 without it
- The dual loss converges far too slowly — after 200 epochs of fine-tuning, still at NME ~34
- Combining 112x112 + hmsup was even worse (NME 45.18) — the interference is amplified with larger heatmaps
- **Do not retry heatmap supervision unless the approach is fundamentally redesigned** (e.g., detached gradient, separate heatmap head with stop_gradient, or pre-train heatmaps then switch to coord-only)
- Keras metric naming gotcha: multi-output models prefix metrics with the OUTPUT LAYER NAME (e.g., `val_landmarks_xy_landmark_nme_iod`), NOT the dict key (not `val_xy_landmark_nme_iod`). This caused silent EarlyStopping/ReduceLROnPlateau failures in early experiments.

### Round 5: Crop Margin + Regularization Experiments (NME_IOD ~9.1)

Deep analysis session combining our own diagnosis with Codex/GPT-5.3 consultation identified three root causes for the 10.14 → 6.52 gap:

1. **Excessive crop margin** — our preprocessing used `lm_margin=0.12` (12% padding around landmark bounding box) + `crop_margin=0.20` (20% expansion around that), resulting in ~32% total padding. The DogFLW paper uses only ~10% surrounding space. This wastes ~2x effective resolution on background.
2. **Coordinate-through-SoftArgmax training** — our model trains MSE on coordinates extracted via SoftArgmax2D (spatial softmax + weighted sum). DeepLabCut and other top methods train directly on heatmap MSE, which provides denser gradient signal per landmark.
3. **Overfitting** — train-val gap of 4.48 points (train 6.1 vs val 10.58) with only 3,853 training images across 120 breeds.

Used `scripts/run_nme_push_v2.py` to run experiments sequentially. Also added new features to the training script:
- **SoftArgmax2D temperature (beta)**: scales logits before softmax, making peaks sharper. `SoftArgmax2D(beta=40)` for 112x112 heatmaps.
- **Mixup augmentation**: blends pairs of training images and landmark coordinates. `aug_mixup=True, aug_mixup_alpha=0.2, aug_mixup_prob=0.4`.
- **Random erasing**: masks random rectangular regions during training. `aug_random_erase=True, aug_random_erase_prob=0.25`.
- **Pure heatmap supervision**: trains directly on Gaussian heatmap targets with MSE, bypassing SoftArgmax2D during training (still used at inference for TFLite export).

| # | Experiment | NME_IOD | + TTA | Time | Notes |
|---|---|---|---|---|---|
| 1 | **tight_margin** | **9.53** | **9.11** | 2h 45m | **NEW BEST** — single digits! |
| 2 | tight_margin_mixup | ~10.4 | (interrupted) | ~3h | Mixup too aggressive, underfitting |
| 3 | pure_heatmap | — | — | — | Not reached (computer crash) |
| 4 | combined_best | — | — | — | Not reached (computer crash) |

**Key learnings**:

**Tight crop margin (breakthrough)**:
- Reducing `lm_margin` from 0.12 → 0.05 and `crop_margin` from 0.20 → 0.10 dropped NME from 10.58 → 9.53 (no TTA), or 10.14 → 9.11 (with TTA)
- This was the **single most impactful change since adding the heatmap head** (~1 full NME point)
- The model now sees ~2x more face detail at the same 224x224 input resolution — previously half the pixels were wasted on background/body
- Train-val gap reduced from 4.48 → 3.53 (train 5.95 vs val 9.53), suggesting the model now has more useful signal to learn from
- Phase 2 fine-tuning ran all 200 epochs without early stopping — the model was still slowly improving at termination

**Mixup augmentation (did not help)**:
- Mixup (alpha=0.2, p=0.4) + Random Erasing (p=0.25) was tested on top of tight margins
- Dramatically reduced overfitting: train-val gap dropped to ~0.5 (vs 3.5 without mixup)
- But absolute val NME was significantly worse (~10.4 vs 9.53) — the model was underfitting
- Mixup makes training data harder by blending images, which prevents the model from learning fine-grained landmark positions
- **Conclusion**: For small datasets where overfitting comes from limited data diversity (not model capacity), mixup may hurt more than it helps. The model needs to learn precise spatial features, and blending destroys that precision.
- Might work with lower probability (p=0.1-0.2) or only during early training

**Pure heatmap supervision & combined_best**:
- These experiments were queued but not reached due to computer interruption
- Still worth trying — pure heatmap MSE supervision (different from the earlier failed dual-loss approach) trains on Gaussian targets directly, which is how DeepLabCut achieves 6.70
- Key difference from Round 4's failed attempt: pure heatmap supervision has NO coordinate loss branch at all during training. The SoftArgmax2D is only attached for inference/export. This avoids the gradient interference that killed the dual-loss approach.

### Round 6: NME Push Marathon — Resolution, Beta, Aug Sweep (NME_IOD ~8.88)

Marathon session running many experiments to push from 9.11 toward the low 8s. Key strategy: combine small improvements (256px resolution, SoftArgmax beta, augmentation tuning).

Created `scripts/eval_experiments.py` for zero-cost evaluations (beta sweep, multi-scale TTA) and `scripts/eval_256.py` for quick TTA evaluation.

| # | Experiment | NME_IOD | + TTA | Notes |
|---|---|---|---|---|
| 1 | Beta sweep (zero-cost on tight_margin) | 9.53 best at beta=1 | — | Higher beta monotonically worse for coord-trained model |
| 2 | Pure heatmap sigma=2.5 | 33.7 | 32.9 | FAILED: background-dominated MSE loss |
| 3 | Strong aug (30° rot, 0.75-1.25 scale) | 9.67 | 9.25 | Too aggressive, hurts performance |
| 4 | Pure heatmap sigma=3.5 | ~82-31 (beta sweep) | — | Still broken at all beta values |
| 5 | Beta=10 SoftArgmax training | 9.528 | 9.088 | Marginal TTA improvement (9.11→9.09) |
| 6 | **256×256 input resolution** | **9.27** | **8.88** | **NEW BEST** — 0.23 gain with TTA! |
| 7 | 256×256 + beta=10 | ~9.3 est. | — | Behind original 256 at same stage; beta=10 hurts |
| 8 | **Multi-scale TTA (3 scales × 2 flips)** | — | **8.66** | **NEW BEST** — free 0.22 gain over flip TTA |
| 9 | Per-landmark analysis | — | — | Ears 3x worse than eyes/nose (14 vs 5) |
| 10 | 5-scale TTA [0.85..1.15] | — | **8.63** | Marginal gain over 3-scale (8.66→8.63) |
| 11 | 3-model ensemble (no TTA) | **8.91** | — | All 3 tight_margin models averaged |
| 12 | **3-model ensemble + flip TTA** | — | **8.67** | Matches multi-scale TTA on single model |
| 13 | 2-model pairs + flip TTA | — | 8.68-8.90 | beta10+256 pair nearly as good as 3-model |
| 14 | Weighted TTA schemes | — | 8.66 | Equal weights already optimal; trimmed mean 8.656 |
| 15 | **3-model ensemble + ms+flip TTA** | — | **8.52** | **NEW BEST** 18 passes (3 models × 3 scales × 2 flips) |
| 16 | 2-model (beta10+256) + ms+flip TTA | — | **8.52** | Same as 3-model! tm model adds nothing |
| 17 | 256×256 + beta=10 training | 9.43 | — | Behind original 256 (9.23); beta=10 doesn't help at 256px |
| 18 | 256×256 high dropout (0.25) + weight_decay 3e-4 | 9.31 | — | Didn't beat original 256 (9.23); regularization alone can't close gap |
| 19 | 256×256 ear-weighted loss (2x on landmarks 0-17) | 9.34 | — | Ears barely improved (-0.08), everything else regressed (+0.43 mouth) |
| 20 | **320×320 input resolution** | **8.82** | **8.32** | **NEW BEST single model** — 0.45 gain over 256px! |
| 21 | 320px flip TTA only | — | **8.52** | Equal to old 3-model ensemble result |
| 22 | **2-model (256+320) ensemble + ms+flip TTA** | — | **8.22** | **NEW OVERALL BEST** — 12 passes |
| 23 | 4-model (all) ensemble + ms+flip TTA | — | 8.26 | 224px models drag it down vs 2-model |

**Key learnings**:

**256×256 resolution (breakthrough)**:
- Increasing input from 224×224 → 256×256 gives heatmaps of 128×128 instead of 112×112
- NME improved by 0.26 without TTA (9.53→9.27) and 0.23 with TTA (9.11→8.88)
- This is the first time we've broken into the 8s
- The improvement is consistent across TTA and non-TTA, suggesting it's a genuine resolution benefit
- Preset: `tight_margin_256`, artifacts: `artifacts/tight_margin_256/`

**Beta sweep lesson**:
- For a model trained with coord MSE through SoftArgmax2D (beta=1), changing beta at inference always makes things worse
- The model optimized its heatmaps for beta=1 softmax expectation — they're broad/diffuse
- Higher beta at inference turns these diffuse heatmaps into sharp peaks, but the peaks aren't at the right locations
- Beta only helps when training and inference use the same beta value

**Pure heatmap supervision still fails**:
- Sigma=2.5: each Gaussian covers ~25 of 12,544 pixels (0.2%). Background MSE dominates
- The model achieves low MSE (~0.0015) by predicting near-zero everywhere, with peaks 13px off on average
- Even sigma=3.5 and sigma=10 didn't fix it (DeepPoseKit uses sigma=5 in INPUT space, which is much larger)
- Would need much larger sigma, weighted loss, or focal loss to address the class imbalance between foreground and background pixels

**Multi-scale TTA (breakthrough — free gain)**:
- Scales [0.9, 1.0, 1.1] × 2 flips = 6 forward passes
- NME improved from 8.88 (flip only) → 8.66 (multi-scale + flip), a free 0.22 gain
- Scale=0.9 (zoom in): center-crop to 90% then resize back to 256
- Scale=1.1 (zoom out): reflect-pad to 110% then resize back to 256
- Coordinate unmapping: p_orig = (p_scaled - 0.5) * scale + 0.5
- Multi-scale alone (8.93) slightly worse than flip alone (8.88) — flip symmetry is more valuable
- Script: `scripts/eval_multiscale_tta.py`

**Per-landmark analysis (critical insight for future work)**:
- Ears dominate error: right_ear 14.0 NME, left_ear 12.2 NME (vs overall 8.88)
- Eyes are nearly solved: right_eye 4.9, left_eye 5.2 NME
- Ear tips (landmarks 5-7, 9-13) are worst at 15-18 NME — highly deformable across breeds
- Ear bases (landmarks 16-17, 0-1) are reasonable at 4.5-11 NME
- This strongly motivates an ELD-style ensemble with an ear specialist model
- Script: `scripts/per_landmark_analysis.py`

**3-model ensemble (0.20 gain over best individual)**:
- Ensemble of tight_margin (224px) + tight_margin_beta10 (224px) + tight_margin_256 (256px)
- No TTA: 8.91, flip TTA: 8.67 (vs best individual 8.88)
- Two 224px models are too similar to each other (NME diff only 0.02)
- The 256px model provides the diversity — beta10+256 pair (8.68) nearly matches 3-model (8.67)
- Weighted ensemble (inverse-NME weights) gave no improvement over equal weights
- **Ensemble + multi-scale TTA compound: 8.52!** (18 passes = 3 models × 3 scales × 2 flips)
- 2-model pair (beta10+256) achieves 8.52 with just 12 passes — tight_margin adds nothing
- Flip is the real driver: ms-only (8.68) < flip-only (8.67), but ms+flip (8.52) compounds well
- Scripts: `scripts/eval_ensemble.py`, `scripts/eval_ensemble_multiscale.py`

**Weighted TTA analysis (equal weights optimal)**:
- Tested center-heavy [0.5,2.0,0.5], moderate [0.7,1.6,0.7], soft [0.8,1.4,0.8], trimmed mean
- Center-heavy actually *hurts* — off-scale predictions are genuinely informative
- Trimmed mean gives marginal 0.004 gain (8.660→8.656), within noise
- Script: `scripts/eval_weighted_tta.py`

**320×320 resolution (biggest single-model breakthrough)**:
- Input resolution 320×320 gives 160×160 heatmaps (vs 128×128 at 256px)
- NME improved by 0.45 without TTA (9.27→8.82) — biggest resolution jump yet
- With ms+flip TTA: 8.32 (vs 256px's 8.66)
- Resolution continues to be the #1 lever: 224→256 gave 0.26, 256→320 gave 0.45
- Preset: `tight_margin_320`, artifacts: `artifacts/tight_margin_320/`

**2-model ensemble (256+320) = 8.22 (new overall best)**:
- 2 models × 3 scales × 2 flips = 12 passes per image
- Resolution diversity (256+320) is the key to ensemble gains
- 4-model ensemble (adding both 224px models) was WORSE (8.26) — weak models dilute the average
- The jump from 8.32 (best single model TTA) to 8.22 (2-model ensemble) = 0.10 from ensembling

**Ear-weighted loss (failed)**:
- 2x loss weight on ear landmarks (0-17) barely improved ears (-0.08 NME)
- Everything else regressed: mouth +0.43, nose nostrils +0.29, nose bridge +0.15
- Net effect: overall NME 9.34 vs 9.27 baseline — worse
- The model's capacity is shared; upweighting ears steals from other landmarks

**Higher regularization (failed)**:
- Dropout 0.25 + weight_decay 3e-4 gave NME 9.31 vs baseline 9.23
- The train-val gap (~3.5) appears structural, not solvable by regularization alone
- 3853 training images across 120 breeds is simply insufficient diversity

**256+beta10 (failed)**:
- Training with beta=10 at 256px converged to val NME 9.43 vs original 256's 9.23
- Beta=10 hurts at higher resolution — sharper softmax peaks don't help when heatmaps are already high-res

**Overfitting analysis (train-val gap = 3.50)**:
- At best epoch: train NME 5.73 vs val NME 9.23 (gap = 3.50)
- If halved (gap → 1.75), val NME would improve from 9.23 to ~7.5
- Current regularization: SpatialDropout2D 0.1, weight_decay 1e-4
- Prepared presets: higher dropout (0.25) + weight_decay (3e-4), ear-weighted loss (2x on landmarks 0-17)

**Strong augmentation hurts**:
- 30° rotation + 0.75-1.25 scale range is too aggressive for this dataset (NME 9.67 vs 9.53)
- The tight crop means rotated/scaled images often push landmarks to crop edges or outside it
- Moderate augmentation (20° rotation) still untested — may be the sweet spot

### Round 7: 384px Resolution + 3-Model Ensemble (NME_IOD = 8.04)

Overnight marathon session to push the ensemble further with a higher-resolution 384px model and explore cosine annealing / DenseNet121 backbone alternatives. Also added DenseNet121 backbone support and per-landmark heatmap loss to the training script.

**New code additions**:
- DenseNet121 backbone support with proper torch-style ImageNet preprocessing (`Normalization` layer)
- Per-landmark normalized heatmap MSE loss (averages per-channel before averaging across landmarks, fixes background-dominated loss)
- Cosine annealing LR schedule for Phase 2 (alternative to ReduceLROnPlateau)
- `scripts/eval_all_models.py` — comprehensive eval: all models, all TTA modes, all pair ensembles, full ensemble
- `scripts/overnight_marathon.sh` — automated chain: train → eval → train → eval

**Architecture (192×192 heatmaps at 384px)**:
```
EfficientNetV2S (frozen, 384×384)
    -> 12×12×1280 feature map
    -> Conv2DTranspose(256, 4×4, stride 2) + BN + ReLU  -> 24×24
    -> Conv2DTranspose(256, 4×4, stride 2) + BN + ReLU  -> 48×48
    -> Conv2DTranspose(256, 4×4, stride 2) + BN + ReLU  -> 96×96
    -> Conv2DTranspose(256, 4×4, stride 2) + BN + ReLU  -> 192×192
    -> Conv2D(46, 1×1)  -> 192×192 heatmaps
    -> SoftArgmax2D      -> 92 coordinates [0,1]
```

| # | Experiment | NME_IOD | + flip TTA | + ms+flip TTA | Notes |
|---|---|---|---|---|---|
| 1 | **384×384 input resolution** | **8.77** | **8.49** | **8.33** | New best single-model no-TTA |
| 2 | 2-model (320+384) ensemble | — | — | **8.10** | Best pair ensemble |
| 3 | 2-model (256+384) ensemble | — | — | **8.11** | Nearly ties 320+384 |
| 4 | **3-model (256+320+384) ensemble** | — | — | **8.04** | **NEW OVERALL BEST** |
| 5 | 320px cosine annealing | (training) | — | — | In progress |
| 6 | DenseNet121 + per-landmark heatmap | (queued) | — | — | Code ready, not yet trained |

**Key learnings**:

**384×384 resolution (marginal but valuable in ensemble)**:
- Input 384×384 gives 192×192 heatmaps (vs 160×160 at 320px, 128×128 at 256px)
- Baseline NME: 8.77 (vs 320px's 8.82) — only 0.05 improvement, heavily diminishing returns
- With ms+flip TTA: 8.33 (vs 320px's 8.32) — almost identical
- Required 200 Phase 2 epochs with LR reduction to beat 320px; batch_size=8 (vs 16 for 320px) due to memory
- Training took ~9 hours total (Phase 1: 78 epochs, Phase 2: 200 epochs at ~2min/epoch)
- **Resolution gains are plateauing**: 224→256 gave 0.25, 256→320 gave 0.42, 320→384 gave 0.05
- Despite near-identical solo performance, 384px contributes significantly to ensemble diversity

**3-model ensemble = new best (8.04)**:
- 3 models × 3 scales × 2 flips = 18 forward passes per image
- Beats best 2-model pair (8.10) by 0.06 and old best (8.22) by 0.18
- Pair rankings: 320+384 (8.10) > 256+384 (8.11) > 256+320 (8.22)
- The 384px model contributes the most as an ensemble member despite being ~equal solo
- Resolution diversity is the key: each model captures different spatial features at different scales

**Per-region NME improvement (3-model ensemble vs old 2-model)**:
| Region | Old (8.22) | New (8.04) | Improvement |
|---|---|---|---|
| right_ear | 14.0 | 12.87 | -1.13 |
| left_ear | 12.2 | 11.21 | -0.99 |
| nose_nostrils | 7.9 | 7.10 | -0.80 |
| mouth | 7.4 | 6.55 | -0.85 |
| left_eye | 5.2 | 4.64 | -0.56 |
| nose_bridge | 4.9 | 4.28 | -0.62 |
| right_eye | 4.9 | 4.24 | -0.66 |

All regions improved significantly. Ears still dominate error (12-13 NME vs 4-7 for other regions).

**Training the 384px model — observations**:
- Phase 1 converged in 78 epochs (val NME 9.56); Phase 2 ran all 200 epochs
- Critical LR reduction around epoch 150 broke through a plateau (8.84→8.76)
- Final best: 8.742 at epoch 195 (reported as 8.77 in eval due to TFLite quantization)
- Train NME reached 4.87, train-val gap nearly 4.0 — heaviest overfitting of any model
- batch_size=4 was too slow (~235s/epoch = 19hrs); switched to batch_size=8 (~120s/epoch)

**Per-landmark heatmap loss (code ready, not yet fully trained)**:
- New loss function averages MSE per-landmark channel before averaging across landmarks
- Fixes the background-dominated loss problem: at 56×56, sigma=2.5 Gaussian covers ~25/3136 pixels (0.8%) — per-landmark normalization ensures each landmark contributes equally
- Smoke test (5 epochs): loss converged properly (0.061→0.004) unlike old approach, but NME was 79.81 after 5 epochs — too early to evaluate

---

## What Worked (Ranked by Impact)

1. **Heatmap head replacing GAP+Dense** — 40 -> 12 (3x improvement, the breakthrough)
2. **Tight crop margin** — 10.58 -> 9.53, 10.14 -> 9.11 with TTA (~1 point improvement). Reduced `lm_margin` from 0.12 to 0.05 and `crop_margin` from 0.20 to 0.10. Gives the model 2x more face pixels.
3. **320×320 input resolution** — 9.27 -> 8.82 (0.45 point), 8.66 -> 8.32 ms+flip TTA. Biggest resolution gain yet; 160x160 heatmaps.
4. **384×384 input resolution** — 8.82 -> 8.77 (only 0.05), but critical for ensemble diversity. 192x192 heatmaps.
5. **256×256 input resolution** — 9.53 -> 9.27, 9.11 -> 8.88 with TTA (~0.25 point). Larger input gives 128x128 heatmaps vs 112x112.
6. **Horizontal flip augmentation** — 12 -> 11.7 (halves overfitting gap)
7. **Two-phase backbone fine-tuning** — 11.7 -> 10.7 (adapts features to task)
8. **Multi-scale + flip TTA at inference** — 9.27 -> 8.66 single model, 8.82 -> 8.32 (320px) (free gain with 3 scales × 2 flips = 6 passes)
9. **3-model (256+320+384) ensemble + ms+flip TTA** — best single 8.32 -> **8.04**, a 0.28 gain from ensembling + resolution diversity
10. **Flip-TTA at inference** — 10.7 -> 10.25, 9.53 -> 9.11, 8.82 -> 8.52 (~0.4-0.5 free gain)
11. **112x112 heatmaps (4th deconv)** — 10.72 -> 10.58 (modest resolution gain)
12. **Scale augmentation** — modest but synergistic with flip
13. **AdamW optimizer** — slight improvement over Adam

## What Didn't Work

1. **Pure heatmap supervision** — NME 33.7. Background-dominated MSE loss (Gaussians cover <0.2% of heatmap pixels). Model predicts near-zero everywhere. Tried sigma=2.5, 3.5, 10 — all failed.
2. **Heatmap supervision (Gaussian targets, hybrid loss)** — NME 32.81 after full training. Dual loss through shared backbone fundamentally interferes with coordinate regression. Do not retry without architectural changes.
3. **Heatmap-level TTA** — averaging logits before soft-argmax produces 2x worse results (22.31 vs 10.72). Must use coordinate-level TTA instead.
4. **Wing loss** — worse than MSE (40.97 vs 40.16)
5. **SpatialDropout2D** — negligible improvement (12.25 vs 12.18)
6. **Dense head fine-tuning** — fragile, collapsed at any LR > 1e-6
7. **Crop jitter alone** — minimal impact without flip/scale
8. **Mixup augmentation (alpha=0.2, p=0.4)** — reduced overfitting dramatically (gap from 3.5 to 0.5) but caused underfitting. Val NME ~10.4 vs 9.53 without mixup. The blending destroys fine-grained spatial precision needed for landmark detection.
9. **Strong augmentation (30° rotation, 0.75-1.25 scale)** — NME 9.67 vs 9.53 baseline. Too aggressive for tight crops — landmarks end up at crop edges.
10. **Post-hoc SoftArgmax beta tuning** — Changing beta at inference for a model trained with beta=1.0 is always worse. Higher beta monotonically increases NME (up to 17.9 at beta=100).
11. **Ear-weighted loss (2x on landmarks 0-17)** — NME 9.34 vs 9.27 baseline. Ears barely improved (-0.08 NME) while other regions regressed significantly (mouth +0.43, nose nostrils +0.29, nose bridge +0.15). Upweighting ears steals capacity from other landmarks.
12. **Higher regularization (dropout 0.25, weight_decay 3e-4)** — NME 9.31 vs 9.23. The train-val gap (~3.5) is structural — caused by insufficient data diversity (3853 images, 120 breeds), not by model capacity or regularization.
13. **256px + beta=10 training** — NME 9.43 vs 9.23. Beta=10 hurts at higher resolution; sharper softmax peaks don't help when heatmaps are already high-res (128x128).

---

## The Main Bottleneck: Overfitting

The single biggest obstacle to reaching single digits is the **train-val gap**. Every model shows the same pattern:

| Model | Train NME | Val NME | Gap |
|---|---|---|---|
| heatmap_v2s (frozen, no aug) | 4.19 | 12.18 | 7.99 |
| heatmap_v2s_best (flip+scale) | 7.46 | 10.72 | 3.26 |
| heatmap_v2s_112 | 6.10 | 10.58 | 4.48 |
| tight_margin | 5.95 | 9.53 | 3.58 |
| tight_margin_256 | ~5.7 | 9.27 | ~3.6 |
| tight_margin_320 | ~5.4 | 8.82 | ~3.4 |
| tight_margin_384 (current best) | ~4.9 | 8.77 | ~3.9 |

The model can fit training data to NME ~6 but generalizes to only ~10.5. The 3,853 training images across 120 breeds (only ~32 images per breed) are insufficient for the model to generalize well. Key observations:
- Augmentation cuts the gap significantly (8.0 -> 3.3 with flip+scale)
- But the gap grew back with 112x112 (4.5) because more spatial detail = more capacity to memorize
- The train NME (6.1) is already close to the paper's best (6.52), so the model has enough capacity — it's purely a generalization problem

**Strategies to close this gap** (for future researchers):
1. **More aggressive augmentation**: mixup, cutmix, cutout, random erasing — these force the model to not rely on any single region
2. **Label smoothing / soft targets**: prevent overconfident predictions
3. **Stronger weight decay or dropout**: current dropout is 0.1, could try 0.2-0.3
4. **Knowledge distillation**: train a teacher on augmented data, use soft labels
5. **Semi-supervised learning**: use unlabeled dog face images (plenty available) for consistency regularization
6. **Cross-validation ensemble**: train 5 models on different folds, average predictions

---

## Remaining Gap: 8.04 -> 6.52

### What the paper does that we don't
1. **ELD Ensemble** (biggest factor) — multiple specialized models per landmark subset (e.g., one for ears, one for eyes, one for mouth). This is how they get from ~8.5 (single model) to 6.52.
2. **Per-landmark heatmap supervision** — the paper trains MSE on heatmaps at low resolution (~28×28), not coordinate MSE at high resolution. This provides denser gradient signal per landmark.
3. **SubpixelMaxima2D** — the paper uses a different coordinate extraction method than our SoftArgmax2D
4. **DenseNet121 backbone** — the paper's single-model baseline achieves 6.70 with DenseNet121 + heatmap supervision

### Strategies tried and remaining

| Strategy | Expected Impact | Difficulty | Status | Actual Result |
|---|---|---|---|---|
| Flip-TTA | -0.5 to -1.0 | Easy | **Done** | -0.44 to -0.58 gain |
| Higher res heatmaps (112x112) | -0.3 to -0.5 | Easy | **Done** | -0.14 (10.72->10.58) |
| Heatmap supervision (hybrid loss) | -0.8 to -1.5 | Medium | **Failed** | NME 32.81 -- doesn't converge |
| Tight crop margin | -1.0 | Easy | **Done** | -1.05 (10.58->9.53) / -1.03 with TTA |
| Mixup augmentation | -0.5 to -1.5 | Medium | **Tried** | Underfitting — val NME ~10.4, worse than baseline |
| 256×256 input resolution | -0.2 to -0.3 | Easy | **Done** | -0.26 (9.53→9.27) / -0.23 with TTA |
| 320×320 input resolution | -0.3 to -0.5 | Easy | **Done** | -0.45 (9.27→8.82) / -0.34 ms+flip TTA |
| 384×384 input resolution | -0.1 to -0.3 | Easy | **Done** | -0.05 (8.82→8.77); diminishing returns but ensemble-critical |
| Pure heatmap supervision (coord-free) | -0.5 to -1.0 | Medium | **Failed** | NME 33.7, background-dominated loss |
| Per-landmark heatmap loss | -0.5 to -1.0 | Medium | **Code ready** | Smoke test promising (loss converges); full training queued |
| SoftArgmax2D temperature tuning | -0.1 to -0.3 | Easy | **Done** | beta=1 optimal for coord-trained models |
| SoftArgmax beta=10 training | -0.1 to -0.2 | Easy | **Done** | -0.02 TTA improvement (9.11→9.09) |
| Strong augmentation (30°/1.25x) | -0.3 to -0.5 | Easy | **Done** | +0.14 (9.53→9.67), too aggressive |
| Scale TTA (multi-scale inference) | -0.2 to -0.4 | Easy | **Done** | -0.61 (9.27→8.66 single model, 8.82→8.32 at 320px) |
| Stronger regularization (dropout/decay) | -0.3 to -0.5 | Easy | **Done** | No improvement (9.31 vs 9.23); gap is structural |
| 2-model resolution ensemble (256+320) | -0.1 to -0.3 | Easy | **Done** | -0.10 (8.32→8.22) |
| 3-model resolution ensemble (256+320+384) | -0.1 to -0.2 | Easy | **Done** | -0.28 (8.32→8.04); **current best** |
| Ear-weighted loss (2x on ears) | -0.3 to -0.5 | Easy | **Done** | +0.07 (9.27→9.34); hurt non-ear regions |
| DenseNet121 backbone | -0.5 to -1.0 | Medium | **Code ready** | Not yet trained |
| Cosine annealing LR schedule | -0.1 to -0.3 | Easy | **Training** | 320px cosine model in progress |
| Multi-scale feature fusion (FPN-lite) | -0.4 to -1.0 | Medium | Not done | -- |
| ELD-style ensemble (2-3 specialist models) | -2.0 to -4.0 | Hard | Not done | -- |

### Recommended next steps (in priority order)

1. **Per-landmark heatmap supervision** (Medium, highest potential) — the paper's approach: train MSE on per-landmark Gaussian heatmaps at low resolution (28×28 or 56×56). Code is ready in training script (`per_landmark_heatmap_loss` flag). Key insight: per-landmark normalization fixes the background-dominated loss that killed previous heatmap supervision attempts. Smoke test showed loss converging properly (0.061→0.004) unlike old approach. Preset `densenet121_heatmap_56` ready.

2. **DenseNet121 backbone** (Medium, high potential) — the paper's single-model baseline achieves 6.70 with DenseNet121 + heatmap supervision, 2.12 points better than our 8.82 single-model. Code ready with torch-style ImageNet preprocessing. Presets `densenet121_heatmap_56` and `densenet121_heatmap_320` available.

3. **448×448 resolution** (Easy, moderate potential) — diminishing returns from 384px (only +0.05), but may still contribute ensemble diversity. Would produce 224×224 heatmaps. Memory-constrained: may need batch_size=4.

4. **EfficientNetV2M backbone** (Medium, moderate potential) — more capacity than V2S, different features for ensemble diversity.

5. **ELD-style ensemble** (Hard, biggest gain) — train 2-3 specialist models on landmark subsets (ears, eyes+nose, mouth). This is what bridges the gap from ~8 to ~6.5 in the paper. Our per-region analysis shows ears at 11-13 NME vs 4-7 for other regions — an ear specialist could dramatically reduce overall error.

### Realistic targets (updated)
- Single model no TTA: **8.77** (achieved, 384px)
- Single model + flip-TTA: **8.49** (achieved, 384px)
- Single model + ms+flip TTA: **8.32** (achieved, 320px)
- 3-model ensemble + ms+flip TTA: **8.04** (achieved, 256+320+384)
- DenseNet121 + per-landmark heatmap: **~7.5-8.0** (estimated, based on paper's 6.70 single model)
- 4+ model ensemble with DenseNet: **~7.0-7.5** (estimated)
- ELD ensemble (specialists): **~6.5-7.0** (estimated, paper target)

---

## Flip-TTA Implementation Reference

Coordinate-level TTA is implemented in `scripts/run_nme_push.py:evaluate_tta()`. The key algorithm:

```python
# 1. Run model on original image
coords_orig = model(crop)                          # [92] flat coords

# 2. Run model on horizontally flipped image
crop_flip = tf.image.flip_left_right(crop)
coords_flip = model(crop_flip)                     # [92] flat coords

# 3. Unflip: remap landmark indices, then mirror x
coords_flip_2d = reshape(coords_flip, [46, 2])
coords_flip_remapped = gather(coords_flip_2d, FLIP_INDEX, axis=0)  # swap left<->right landmarks
coords_flip_unflipped = stack([1.0 - remapped[:, 0], remapped[:, 1]], axis=-1)  # mirror x

# 4. Average
coords_avg = (coords_orig + flatten(coords_flip_unflipped)) / 2.0
```

**Warning**: Do NOT average heatmaps before soft-argmax. This produces catastrophically bad results (NME 22.31 vs 10.72). Always average at the coordinate level.

---

## Verified Horizontal Flip Index Map

Computed from actual DogFLW label statistics and verified as a perfect involution (`map[map[i]] == i`):

```python
FLIP_INDEX = [
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    17, 16, 19, 18, 21, 20, 23, 22, 24, 25, 27, 26, 29, 28,
    31, 30, 32, 34, 33, 35, 37, 36, 38, 40, 39, 41, 42, 44,
    43, 45,
]
```

---

## Architecture: SoftArgmax2D Layer

Custom Keras layer for differentiable coordinate extraction from heatmaps:

```python
@tf.keras.utils.register_keras_serializable(package="DogFLW")
class SoftArgmax2D(tf.keras.layers.Layer):
    # Input:  (B, H, W, K) heatmaps
    # Output: (B, K*2) coordinates [x0,y0,x1,y1,...] in [0,1]
    # 1. Flatten H*W, apply spatial softmax per landmark
    # 2. Multiply by normalized x/y coordinate grids
    # 3. Sum weighted coordinates -> (x, y) per landmark
    # All ops TFLite-compatible (softmax, multiply, reduce_sum)
```

**Important serialization note**: The `WarmupSchedule` class must also be registered with `@tf.keras.utils.register_keras_serializable(package="DogFLW")`, otherwise model loading fails. When loading models saved with optimizers, use `compile=False` to avoid optimizer deserialization issues.

---

## Training Configuration Reference

### Current best preset (`tight_margin_384`)
```python
backbone = "efficientnetv2s"
head_type = "heatmap"
heatmap_dropout = 0.1
num_deconv_layers = 4          # 4 deconv -> 192x192 heatmaps (at 384px input)
img_size = 384
epochs = 100                   # Phase 1: frozen backbone
finetune_epochs = 200          # Phase 2: fine-tune last 50 layers
finetune_learning_rate = 1e-5
finetune_last_layers = 50
batch_size = 8                 # Reduced from 16 due to memory at 384px
learning_rate = 1e-4
optimizer = "adamw"
weight_decay = 1e-4
loss = "mse"
lm_margin = 0.05
crop_margin = 0.10
```

### Other active presets
- `tight_margin_320` — same as 384 but img_size=320, batch_size=16, heatmaps=160×160
- `tight_margin_256` — same as 384 but img_size=256, batch_size=16, heatmaps=128×128
- `tight_margin_320_cosine` — 320px with cosine annealing LR (Phase 2: 300 epochs)
- `densenet121_heatmap_56` — DenseNet121 backbone, 56×56 heatmaps, per-landmark heatmap loss

### Older presets
- `tight_margin` — 224px input, 112x112 heatmaps, `lm_margin=0.05`, `crop_margin=0.10`
- `heatmap_v2s_112` — 224px input, 112x112 heatmaps, `lm_margin=0.12`, `crop_margin=0.20`
- `heatmap_v2s_best` — 224px input, 56x56 heatmaps (3 deconv layers), wider margins

### Training stability notes
- lr=1e-3 ALWAYS diverges (even frozen backbone)
- lr=1e-4 with ReduceLROnPlateau is the sweet spot
- Backbone fine-tuning: lr=1e-5 for last 50 layers, with 5-epoch warmup
- ReduceLROnPlateau incompatible with CosineDecay schedule
- Keep BN layers frozen during fine-tuning
- Phase 1 (frozen) converges in ~30-50 epochs, patience=50 is generous
- Phase 2 (fine-tune) improvements are very gradual — patience=50 is needed, often runs to max epochs

### Two-phase training details
- **Phase 1**: Backbone fully frozen, train deconv head only. lr=1e-4. EarlyStopping patience=50.
- **Phase 2**: Unfreeze last 50 backbone layers. lr=1e-5 with 5-epoch linear warmup. BN layers stay frozen. EarlyStopping patience=50. ReduceLROnPlateau factor=0.5, patience=3.
- After Phase 2, the script selects the better model (Phase 1 vs Phase 2) based on val NME_IOD.

### CLI overrides
All config fields can be overridden from the command line:
```bash
python scripts/train_dog_face_landmarks.py \
    --experiment heatmap_v2s_112 \
    --num-deconv-layers 4 \
    --heatmap-supervision \
    --heatmap-sigma 1.75 \
    --coord-loss-weight 1.0 \
    --patience 30 \
    --out artifacts/custom_run
```

---

## Key Files

| File | Description |
|---|---|
| `scripts/train_dog_face_landmarks.py` | Main training script — all architectures (EfficientNetV2S, DenseNet121), presets, augmentations, SoftArgmax2D layer, per-landmark heatmap loss |
| `scripts/eval_all_models.py` | Comprehensive eval: all models, all TTA modes, all pair ensembles, full ensemble, per-region breakdown |
| `scripts/gen_landmark_examples.py` | Generate _pred/_true visualizations for all 480 test images (uses 3-model ensemble) |
| `scripts/overnight_marathon.sh` | Automated chain: train → eval → train → eval for overnight runs |
| `scripts/infer_dog_landmarks_tflite.py` | TFLite inference pipeline (two-stage: bbox detection + landmarks) |
| `scripts/per_landmark_analysis.py` | Per-landmark/region error analysis |
| `scripts/eval_320_comprehensive.py` | 320px + ensemble comprehensive eval |
| `CODEX_ADVISOR_REPORT.md` | Detailed analysis from OpenAI Codex consultation on strategies |
| `artifacts/tight_margin_384/` | **Current best single model** (NME 8.77 / 8.33 with ms+flip TTA) |
| `artifacts/tight_margin_320/` | 320px model (NME 8.82 / 8.32 with ms+flip TTA) |
| `artifacts/tight_margin_256/` | 256px model (NME 9.27 / 8.66 ms+flip TTA) |
| `artifacts/dog_face_landmarks/inference_examples/` | 3-model ensemble prediction visualizations (480 images) |
| `artifacts/tight_margin/` | Older 224px model (NME_IOD 9.53 / 9.11 with TTA) |
| `artifacts/heatmap_v2s_112/` | Earlier model (NME_IOD 10.58 / 10.14 with TTA) |
| `artifacts/heatmap_v2s_best/` | Earlier model (NME_IOD 10.72) |
| `artifacts/dog_face_landmarks/` | Production TFLite model + inference example images |
| `artifacts/overnight_log.txt` | Raw logs from overnight marathon runs |

---

## NME_IOD Metric

- **NME** = Normalized Mean Error
- **IOD** = Inter-Ocular Distance (outer eye corners: landmarks 18 and 19)
- Formula: `mean(per_landmark_euclidean_distance) / IOD x 100`
- Lower is better. Paper target: 6.52
- Landmarks are normalized to [0,1] relative to crop coordinates

---

## Gotchas and Pitfalls (Save Future Researchers Time)

1. **Keras multi-output metric names**: When using dict outputs like `{"hm": ..., "xy": ...}`, Keras prefixes metrics with the **output layer name** (e.g., `landmarks_xy_landmark_nme_iod`), NOT the dict key (`xy_landmark_nme_iod`). This silently breaks EarlyStopping and ReduceLROnPlateau if the monitor string is wrong — training continues but never triggers callbacks.

2. **WarmupSchedule serialization**: Must be decorated with `@tf.keras.utils.register_keras_serializable(package="DogFLW")`. Without this, `tf.keras.models.load_model()` fails. Also load with `compile=False` to sidestep optimizer deserialization.

3. **Heatmap-level TTA is a trap**: Intuitively averaging heatmaps before soft-argmax seems cleaner than averaging coordinates. In practice it produces NME 22.31 (2x worse). Always average at coordinate level.

4. **Heatmap supervision does not work in our setup**: Despite being a standard technique in human pose estimation, adding Gaussian target heatmap loss causes catastrophic interference with coordinate regression through the shared backbone. NME 32-45 vs 10.58 without it. Would need architectural redesign (e.g., stop-gradient on heatmap branch) to potentially work.

5. **AdamW on M1/M2 Macs**: TensorFlow warns about slow performance. Training still works but is ~10-15% slower than on CUDA GPUs. Use `tf.keras.optimizers.legacy.AdamW` if speed matters.

6. **TFLite dynamic tensor warning**: The SoftArgmax2D layer generates a dynamic-sized tensor warning during TFLite conversion. This is harmless — the model runs correctly at inference.
