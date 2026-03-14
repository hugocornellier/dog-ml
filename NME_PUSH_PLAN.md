# NME Push Plan: 9.11 → Sub-8.0

**Date**: 2026-02-28
**Current best**: NME_IOD = 9.11 (with flip-TTA) / 9.53 (without)
**Target**: 6.52 (paper ELD ensemble) / 6.70 (DeepLabCut DenseNet121 single model)
**Realistic single-model target**: 7.5–8.5

---

## 1. Diagnosis: Why Are We at 9.11 and Not 6.70?

The DogFLW paper's single-model results (Scientific Reports 2025, Table 2) show that DeepLabCut via DeepPoseKit with DenseNet121 achieves **6.70** — beating our EfficientNetV2S (a stronger backbone) by **2.83 NME points**. This is not a backbone problem. It's a training paradigm problem. Here's why:

### A. Training Objective: Coordinate MSE vs Heatmap MSE (estimated ~1.0–1.5 points)

**Our approach**: We train with MSE on coordinates extracted via SoftArgmax2D. The gradients flow through the soft-argmax layer, which means the model learns to produce heatmaps that, when softmax-weighted-averaged, produce the right coordinates. This is fundamentally an *expectation* — it optimizes for the center-of-mass of the probability distribution, not the peak.

**DeepLabCut/DeepPoseKit approach**: They train directly on Gaussian heatmap targets with MSE loss. Each landmark gets a 2D Gaussian target (sigma=5 in input-space pixels), and the model learns to reproduce that Gaussian. At inference, they use **SubpixelMaxima2D** — a Fourier-domain registration method that finds the actual *peak* of each heatmap with subpixel precision.

**Why this matters**:
- Coordinate-through-SoftArgmax training produces broad, diffuse heatmaps because the gradient pushes all spatial positions to contribute to the weighted average. A broad heatmap with the right mean is equivalent to a sharp peak under SoftArgmax.
- Heatmap MSE training produces sharp, well-localized Gaussians because the pixel-wise loss penalizes any activation away from the target center.
- At inference, our SoftArgmax computes `E[x]` (expectation), while their SubpixelMaxima finds `argmax(x)` (mode). For a perfectly symmetric unimodal distribution these are identical, but real heatmaps have noise, secondary modes, and asymmetry — the argmax is more robust.

**This is the single biggest gap in our pipeline.** The `pure_heatmap` preset was built to test this but never ran due to a computer crash.

### B. Coordinate Extraction: SoftArgmax2D vs SubpixelMaxima2D (estimated ~0.3–0.7 points)

Even if our heatmaps were perfect, SoftArgmax2D (weighted spatial average) is suboptimal:

- **SoftArgmax2D** with beta=1.0 computes `sum(coords * softmax(heatmap))`. With 112×112 = 12,544 positions, softmax makes the distribution very flat — every pixel contributes, pulling the prediction toward the center.
- **Higher beta** (20–60) makes the distribution peakier, approximating argmax. But the model was trained with beta=1.0, so the heatmap magnitudes were learned for that temperature.
- **Hard argmax + Taylor/parabolic refinement** (already implemented in `eval_experiments.py:argmax_with_refinement`) finds the max pixel then fits a parabola to the 3×3 neighborhood for subpixel precision. This is closer to what DeepPoseKit does.

The `eval_experiments.py` script already has both beta sweep and argmax+refinement implemented but was run against the *old* model (heatmap_v2s_112, not tight_margin). We need to run it against the current best model.

### C. Heatmap Sigma (estimated ~0.3–0.5 points)

Our Gaussian heatmap targets use sigma=1.75–2.5 pixels (in heatmap space). DeepPoseKit uses **sigma=5 in input space** — for 224×224 input downsampled to output, this is considerably larger. A larger sigma:
- Creates smoother, more overlapping gradients — easier to optimize
- Provides more gradient signal per pixel — each landmark's loss affects a larger spatial region
- Allows the model to start learning even when predictions are rough, then refine

With our 112×112 heatmaps, sigma=5 in input space corresponds to sigma ≈ 2.5 in heatmap space (since 112/224 × 5 = 2.5). We should try sigma values from 2.5 to 5.0 in heatmap space.

### D. Augmentation Aggressiveness (estimated ~0.2–0.5 points)

DeepPoseKit uses much more aggressive augmentation:
- **Full 360° rotation** (vs our ±15°)
- **75–125% scaling** (vs our 85–115%)
- Both horizontal AND vertical flip
- Dropout (image-level)

Our rotation is extremely conservative. Dog faces can appear at many angles (tilted heads, lying down, etc.). Expanding to ±30° or even ±45° would better represent real-world variation without the blending artifacts of mixup.

### E. Overfitting Gap (3.58 points: train 5.95 vs val 9.53)

The train NME of 5.95 proves the model has sufficient capacity. The remaining gap is generalization. Key insight: the paper's DeepLabCut likely has a smaller gap because:
1. Heatmap MSE provides per-pixel regularization (every pixel in the 112×112 heatmap contributes to loss, not just 92 coordinate values)
2. Larger sigma spreads the loss signal, acting as implicit regularization
3. More aggressive augmentation fills the data distribution better

---

## 2. Ranked Experiments

### Experiment 1: Pure Heatmap Supervision + Tight Margins (HIGH PRIORITY)
**Expected impact**: -0.8 to -1.5 NME points (8.0–8.5 target)
**Difficulty**: Medium (code already exists in `pure_heatmap` preset)
**Time**: ~3 hours training

This is the highest-priority experiment because it addresses the #1 diagnosed issue (training objective). The `pure_heatmap` preset is already defined but needs modification:

**Changes from existing `pure_heatmap` preset**:
- Remove mixup and random erasing (these caused underfitting — don't compound unknowns)
- Sweep heatmap sigma: try 2.0, 2.5, 3.5, 5.0 (in heatmap pixels)
- Keep tight margins (lm_margin=0.05, crop_margin=0.10)

**Inference**: After training, sweep decode methods:
1. SoftArgmax with beta in [1, 10, 20, 40, 60, 100]
2. Hard argmax + parabolic refinement
3. Best decode + flip-TTA

**New preset**: `pure_heatmap_clean`
```python
ExperimentConfig(
    name="pure_heatmap_clean",
    backbone="efficientnetv2s",
    head_type="heatmap",
    heatmap_dropout=0.1,
    num_deconv_layers=4,
    epochs=100,
    finetune_epochs=200,
    finetune_learning_rate=1e-5,
    finetune_last_layers=50,
    batch_size=16,
    learning_rate=1e-4,
    loss="mse",
    optimizer="adamw",
    weight_decay=1e-4,
    lm_margin=0.05,
    crop_margin=0.10,
    pure_heatmap_supervision=True,
    heatmap_sigma=2.5,         # Start here, sweep later
    aug_rotation=True,
    aug_rotation_deg=15.0,
    aug_flip=True,
    aug_crop_jitter=True,
    aug_crop_jitter_frac=0.08,
    aug_scale=True,
    aug_brightness=True,
    aug_contrast=True,
    aug_saturation=True,
    aug_color_balance=True,
    aug_sharpness=True,
    aug_blur=True,
    aug_noise=True,
    # NO mixup, NO random erasing
    patience=50,
)
```

**Success criteria**: NME_IOD < 9.0 without TTA validates heatmap supervision helps. NME_IOD < 8.5 with TTA would be a major breakthrough.

**Risk**: Pure heatmap supervision might produce poor coordinate-space metrics during training since we monitor `val_loss` (heatmap MSE) not NME_IOD. Need to add a custom callback or post-training evaluation to track actual NME_IOD.

### Experiment 2: Zero-Cost Eval on Current Best Model (EASY, DO FIRST)
**Expected impact**: -0.1 to -0.5 NME points
**Difficulty**: Easy (no training needed)
**Time**: ~30 minutes

Run `eval_experiments.py` against the **current best model** (`artifacts/tight_margin/best.keras`) instead of the old `heatmap_v2s_112` model. The script needs a one-line path change.

Tests:
1. Temperature sweep (beta 1–100) — find the optimal beta for the tight-margin model
2. Argmax + parabolic refinement — might be better than any beta
3. Multi-scale TTA (0.9×, 1.0×, 1.1×) combined with best decode method

**Success criteria**: Any improvement over 9.11 is free and compounds with future training improvements.

### Experiment 3: Stronger Augmentation (MEDIUM PRIORITY)
**Expected impact**: -0.3 to -0.8 NME points
**Difficulty**: Easy (config changes only)
**Time**: ~3 hours training

Increase augmentation aggressiveness to match DeepPoseKit:

**Changes**:
- `aug_rotation_deg`: 15 → 30 (or even 45)
- `aug_scale_range`: (0.85, 1.15) → (0.75, 1.25)
- `aug_crop_jitter_frac`: 0.08 → 0.12
- Consider adding vertical flip for some breeds (some dogs are photographed upside down)

**New preset**: `strong_aug`
```python
# Same as tight_margin but with stronger geometric augmentation
aug_rotation_deg=30.0,       # was 15
aug_scale_range=(0.75, 1.25),  # was (0.85, 1.15)
aug_crop_jitter_frac=0.12,    # was 0.08
```

Test this both with coordinate loss (current approach) and with pure heatmap supervision (stacks with Experiment 1).

**Success criteria**: Train-val gap reduced below 3.0 while val NME improves or stays flat. If val NME increases, augmentation is too strong.

**Risk**: Too much rotation could hurt because dog faces have strong bilateral symmetry assumptions. Monitor per-landmark errors — if ear landmarks degrade, dial back rotation.

### Experiment 4: Larger Heatmap Sigma Sweep (MEDIUM PRIORITY)
**Expected impact**: -0.2 to -0.5 NME points (compounds with Experiment 1)
**Difficulty**: Easy (config change)
**Time**: ~9 hours (3 sigma values × 3 hours each)

Run pure heatmap supervision with three different sigma values:
- sigma=2.0 (tighter, higher precision if model can learn it)
- sigma=3.5 (DeepPoseKit-esque)
- sigma=5.0 (exactly matching DeepPoseKit's default)

The sigma affects both the ease of optimization AND the precision ceiling. Larger sigma = easier to optimize but lower maximum precision. There's a sweet spot.

**Success criteria**: Identify the sigma that minimizes val NME_IOD.

### Experiment 5: Argmax + Offset Refinement Head (MEDIUM-HIGH PRIORITY)
**Expected impact**: -0.5 to -1.0 NME points
**Difficulty**: Medium (new code required)
**Time**: ~4 hours (1 hour code + 3 hours training)

Instead of SoftArgmax2D at inference, add a **location refinement field** inspired by DeepLabCut's original architecture:

**Architecture change**:
- Current: `deconv_head → Conv2D(46) → SoftArgmax2D → coords`
- New: `deconv_head → Conv2D(46) [heatmap] + Conv2D(92) [offset_xy] → argmax(heatmap) + offset → coords`

The offset head predicts subpixel (dx, dy) displacements at each heatmap position. At inference:
1. Find the argmax of each heatmap channel
2. Read the (dx, dy) offset at that position from the offset head
3. Final coord = argmax_position + offset

This is how DeepLabCut achieves subpixel precision without SoftArgmax. The offset head is trained with a Huber loss only at the GT landmark positions (or nearby positions within the Gaussian radius).

**Code changes needed**:
- `_build_deconv_head()`: Add a parallel `Conv2D(NUM_LANDMARKS * 2)` offset head
- New loss function for the offset head (L1 or Huber at GT positions)
- New inference function that combines argmax + offset
- Modify `build_model()` to support the new head type

**Success criteria**: NME_IOD < 8.5 would validate the approach.

**Risk**: Adding a second head increases complexity. The offset prediction quality depends on accurate heatmap peaks, so this only helps if the heatmaps are already reasonable.

### Experiment 6: Higher Input Resolution (MEDIUM PRIORITY)
**Expected impact**: -0.3 to -0.8 NME points
**Difficulty**: Easy (config change)
**Time**: ~4 hours (slower training at higher res)

DeepLabCut's default `crop_size` is 400×400 (not 224×224). Try 256×256 or 320×320 input:

**Changes**:
- `img_size`: 224 → 256 (or 320)
- This automatically increases heatmap resolution: 256 → 128×128 heatmaps (4 deconv), 320 → 160×160

EfficientNetV2S accepts variable input sizes. The backbone's feature map scales proportionally.

**Success criteria**: Any improvement over 9.11 with 256×256. The 320×320 may be too slow on M1/M2 but worth trying if 256 helps.

**Risk**: Memory — larger inputs mean larger activations. batch_size may need to drop from 16 to 8. Also, the pretrained ImageNet weights were learned at 224×224, so larger inputs might need more fine-tuning.

### Experiment 7: Mild Mixup + CutOut (LOW-MEDIUM PRIORITY)
**Expected impact**: -0.2 to -0.5 NME points
**Difficulty**: Easy
**Time**: ~3 hours

The previous mixup experiment (alpha=0.2, p=0.4) was too aggressive. Try:
- **CutOut** (not CutMix): mask 1–3 small random patches per image. This is less destructive than mixup because it doesn't blend spatial information across images. Similar to the existing random erasing but more targeted.
- **Very mild mixup**: p=0.1, alpha=0.1 — only 10% of batches, with very conservative blending
- **Phase-1-only mixup**: Apply mixup only during Phase 1 (frozen backbone), then train Phase 2 clean

**Success criteria**: Val NME < 9.53 (must beat baseline without mixup). If it still causes underfitting, abandon mixup entirely.

### Experiment 8: ELD-Style Coarse-to-Fine (HARD, HIGHEST CEILING)
**Expected impact**: -1.5 to -3.0 NME points (target: 7.0–7.5)
**Difficulty**: Hard (significant new code)
**Time**: ~12 hours (4 specialist models × 3 hours each)

Implement the paper's ELD "magnifying" cascade:

1. **Stage 1**: Use current best model as the coarse all-landmark predictor
2. **Stage 2**: Train 3–4 specialist models, each on a region:
   - **Ears**: Landmarks 0–17 (ear tips and bases — the hardest region, NME ~13)
   - **Eyes+nose**: Landmarks 18–31
   - **Mouth+chin**: Landmarks 32–45
3. Each specialist:
   - Crop the region from the original image using coarse predictions
   - Resize to 224×224 (giving 4–8× effective resolution boost for small regions)
   - Predict only its subset of landmarks
4. Assembly: Replace coarse predictions with specialist refinements

**Key insight from the paper**: The ELD uses coordinate regression (GAP + Dense), NOT heatmaps, for both coarse and specialist models. But since our heatmap approach is already better for single models, we should use heatmap specialists.

**Code changes needed**:
- New training script (or mode) for specialist models with landmark subset
- Region-specific crop computation from coarse predictions
- Assembly/inference pipeline that chains coarse → region crop → specialist → combine
- Training data generation pipeline (crop regions from GT landmarks for training)

**Success criteria**: Combined NME_IOD < 8.0 would be a strong result. < 7.5 would be excellent.

**Risk**: Training 4 separate models is time-intensive. The region cropping at inference adds complexity and latency. Start with just ears (highest error) to validate the approach before doing all regions.

---

## 3. Implementation Order

```
Day 1 (Quick Wins):
  ├── [30min] Experiment 2: Zero-cost eval on tight_margin model
  │     → Run beta sweep + argmax + multi-scale TTA
  │     → Get exact baseline numbers with best decode method
  │
  └── [3hr]  Experiment 1a: Pure heatmap supervision (sigma=2.5)
              → Train, evaluate with beta sweep + argmax + TTA

Day 2 (Augmentation + Sigma):
  ├── [3hr]  Experiment 3: Strong augmentation (rotation 30°, scale 0.75-1.25)
  │     → With coordinate loss first (to isolate augmentation effect)
  │
  └── [3hr]  Experiment 1b: Pure heatmap supervision (sigma=3.5)
              → Compare with 1a to find optimal sigma

Day 3 (Resolution + Refinement):
  ├── [3hr]  Best of {1a, 1b} + strong augmentation combined
  │
  └── [4hr]  Experiment 5: Offset refinement head
              → 1hr code, 3hr train

Day 4 (Input Resolution):
  ├── [4hr]  Experiment 6: 256×256 input resolution
  │     → With best training recipe from Days 1-3
  │
  └── [3hr]  Experiment 7: Mild CutOut regularization

Day 5+ (Ensemble):
  └── [12hr] Experiment 8: ELD-style specialists
              → Start with ear specialist only to validate
```

---

## 4. Code Changes Needed

### For Experiment 1 (Pure Heatmap Supervision):
- **`scripts/train_dog_face_landmarks.py`**: Add `pure_heatmap_clean` preset (remove mixup/erasing from `pure_heatmap`)
- **`scripts/eval_experiments.py`**: Change `MODEL_PATH` to `artifacts/tight_margin/best.keras` and `cfg` to use `tight_margin` preset for Experiment 2

### For Experiment 3 (Strong Augmentation):
- **`scripts/train_dog_face_landmarks.py`**: Add `strong_aug` preset

### For Experiment 5 (Offset Refinement Head):
- **`scripts/train_dog_face_landmarks.py`**:
  - Add `head_type="heatmap_offset"` option to `ExperimentConfig`
  - New function `_build_deconv_head_with_offset()` that adds parallel Conv2D(92) offset head
  - New loss function for offset regression (Huber loss at GT positions)
  - Modify `build_model()` to route to new head builder
  - New inference function `argmax_plus_offset()` for coordinate extraction

### For Experiment 6 (Higher Resolution):
- **`scripts/train_dog_face_landmarks.py`**: Add preset with `img_size=256`
- No architectural changes needed — EfficientNetV2S handles variable input sizes

### For Experiment 8 (ELD Specialists):
- New script `scripts/train_specialist.py` or new mode in existing script
- New `ExperimentConfig` fields: `landmark_subset`, `region_crop_source`
- New inference pipeline script for the full cascade

### For the New Experiment Runner (all experiments):
- **`scripts/run_nme_push_v3.py`**: New runner that executes the above experiments in order, with full logging and comparison tables

---

## 5. Success Criteria Summary

| Experiment | Expected NME_IOD | Validates |
|---|---|---|
| 2: Zero-cost eval | 8.8–9.1 | Best decode method for current model |
| 1: Pure heatmap | 8.0–9.0 | Heatmap supervision > coord supervision |
| 3: Strong aug | 8.5–9.3 | More augmentation helps generalization |
| 4: Sigma sweep | 8.0–8.8 | Optimal Gaussian target width |
| 5: Offset head | 7.8–8.5 | Subpixel precision matters |
| 6: 256×256 input | 8.0–8.8 | Resolution helps |
| 1+3+5 combined | 7.5–8.3 | All single-model improvements stack |
| 8: ELD ensemble | 7.0–7.5 | Specialist cascade is key |

**Milestone targets**:
- Break 9.0 → Validates pure heatmap supervision (should be achievable quickly)
- Break 8.5 → Validates combined improvements
- Break 8.0 → Excellent single-model result, approaching paper baselines
- Break 7.5 → Would require ensemble or major breakthrough
- Break 7.0 → Would match paper's best single models (DenseNet121 at 6.70)

---

## 6. Risk Assessment

### What Could Go Wrong

1. **Pure heatmap supervision doesn't converge**: The earlier dual-loss attempt (Round 4) failed catastrophically. The key difference is that pure supervision has NO coordinate loss branch — but if the heatmap output at 112×112 has gradient issues, this could also fail.
   - **Mitigation**: Monitor training loss carefully. If heatmap loss doesn't decrease within first 5 epochs, check for numerical issues.
   - **Fallback**: If pure heatmap fails, try heatmap supervision with stop_gradient on the heatmap branch (so it only regularizes, doesn't interfere with coord gradients).

2. **Argmax at inference is worse than SoftArgmax**: If the model was trained with SoftArgmax (coord loss), switching to argmax at inference could be worse because the heatmaps were optimized for expectation, not mode.
   - **Mitigation**: Only switch to argmax for models trained with heatmap supervision. For coord-trained models, use beta sweep instead.

3. **Higher resolution doesn't help because of padding/cropping**: With tight margins, the face already fills most of the 224×224 crop. Going to 256×256 just adds 14% more pixels.
   - **Mitigation**: Try 320×320 for a more dramatic difference. Also consider going tighter on margins (lm_margin=0.02, crop_margin=0.05).

4. **Strong augmentation hurts like mixup did**: Aggressive rotation or scaling might destroy fine-grained landmark precision.
   - **Mitigation**: Monitor train NME alongside val NME. If both go up, augmentation is too strong. Use intermediate values (rotation=20° before jumping to 30°).

5. **Sigma too large → heatmaps too broad → coord precision drops**: With sigma=5.0, the Gaussian peaks overlap significantly for closely-spaced landmarks (e.g., eye corner landmarks).
   - **Mitigation**: Start with sigma=2.5, only increase if optimization converges too slowly.

### What to Watch For During Training

- **Heatmap loss not decreasing**: Sign of numerical issues or sigma mismatch
- **Train-val gap increasing**: Model is overfitting more, need more regularization
- **Train NME stuck above 7.0 during Phase 2**: Model not converging properly
- **Phase 2 making things worse**: Fine-tuning learning rate too high, or Phase 1 already converged fully
- **NaN/Inf in loss**: Usually means learning rate too high or sigma too small

---

## 7. Key Insight from Paper Analysis

The most surprising finding from studying the paper: **the ELD ensemble uses coordinate regression (GAP + Dense), NOT heatmaps.** The paper's best single-model results (DeepLabCut at 6.70) DO use heatmaps, but the ELD compensates for coordinate regression's lower spatial precision by using the "magnifying" cascade. This means:

1. **Heatmap training is the superior single-model paradigm** — validated by DenseNet121 beating the ELD's coarse model
2. **But the ELD's cascade principle (coarse → magnify → refine) is orthogonal and stacks with any base approach**
3. **The optimal strategy is: heatmap-trained single model + ELD cascade + TTA**

This reinforces that Experiment 1 (pure heatmap supervision) should be the highest priority.
