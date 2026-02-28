# Dog Facial Landmark Detection: Progress Journal

**Current best: NME_IOD = 10.14 (with TTA) / 10.58 (without TTA) | Target: 6.52 (paper) | Started at: ~40**

This document is a living journal of our work improving dog facial landmark detection. It's designed for future LLMs/developers to pick up where we left off and continue pushing toward the paper's target.

---

## Quick Reference

### Current Best Model
- **Preset**: `heatmap_v2s_112`
- **Val NME_IOD**: 10.58 (no TTA) / **10.14** (with flip-TTA)
- **Architecture**: EfficientNetV2S + **4-deconv** heatmap head (112x112) + SoftArgmax2D
- **Key improvements over Round 1**: heatmap head, 4th deconv layer (112x112), horizontal flip augmentation, scale augmentation, AdamW, two-phase backbone fine-tuning, flip-TTA
- **Artifacts**: `artifacts/heatmap_v2s_112/`
- **TFLite**: `artifacts/dog_face_landmarks/dog_face_landmarks_224_float16.tflite` (55 MB)
- **Previous best**: `heatmap_v2s_best` at NME_IOD 10.72 (artifacts in `artifacts/heatmap_v2s_best/`)

### Key Commands
```bash
# Train with the current best preset (112x112 heatmaps)
python scripts/train_dog_face_landmarks.py --experiment heatmap_v2s_112 --out artifacts/my_run

# Train with the previous best preset (56x56 heatmaps)
python scripts/train_dog_face_landmarks.py --experiment heatmap_v2s_best --out artifacts/my_run

# Generate _pred/_true visualizations (uses ALL 480 test images)
python scripts/gen_landmark_examples.py

# Run automated NME push experiments
python scripts/run_nme_push.py

# Quick smoke test (5 epochs)
python scripts/train_dog_face_landmarks.py --experiment heatmap_v2s_112 --epochs 5 --patience 50 --out artifacts/smoke
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

---

## What Worked (Ranked by Impact)

1. **Heatmap head replacing GAP+Dense** — 40 -> 12 (3x improvement, the breakthrough)
2. **Horizontal flip augmentation** — 12 -> 11.7 (halves overfitting gap)
3. **Two-phase backbone fine-tuning** — 11.7 -> 10.7 (adapts features to task)
4. **Flip-TTA at inference** — 10.7 -> 10.25, 10.58 -> 10.14 (~0.4-0.5 free gain)
5. **112x112 heatmaps (4th deconv)** — 10.72 -> 10.58 (modest resolution gain)
6. **Scale augmentation** — modest but synergistic with flip
7. **AdamW optimizer** — slight improvement over Adam

## What Didn't Work

1. **Heatmap supervision (Gaussian targets, hybrid loss)** — NME 32.81 after full training. Dual loss through shared backbone fundamentally interferes with coordinate regression. Do not retry without architectural changes.
2. **Heatmap-level TTA** — averaging logits before soft-argmax produces 2x worse results (22.31 vs 10.72). Must use coordinate-level TTA instead.
3. **Wing loss** — worse than MSE (40.97 vs 40.16)
4. **SpatialDropout2D** — negligible improvement (12.25 vs 12.18)
5. **Dense head fine-tuning** — fragile, collapsed at any LR > 1e-6
6. **Crop jitter alone** — minimal impact without flip/scale

---

## The Main Bottleneck: Overfitting

The single biggest obstacle to reaching single digits is the **train-val gap**. Every model shows the same pattern:

| Model | Train NME | Val NME | Gap |
|---|---|---|---|
| heatmap_v2s (frozen, no aug) | 4.19 | 12.18 | 7.99 |
| heatmap_v2s_best (flip+scale) | 7.46 | 10.72 | 3.26 |
| heatmap_v2s_112 (current best) | 6.10 | 10.58 | 4.48 |

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

## Remaining Gap: 10.14 -> 6.52

### What the paper does that we don't
1. **ELD Ensemble** (biggest factor) — multiple specialized models per landmark subset (e.g., one for ears, one for eyes, one for mouth). This is how they get from ~8.5 (single model) to 6.52.
2. **Multi-scale TTA** — we only do flip-TTA; the paper likely uses scale TTA and possibly rotation TTA
3. **Better generalization** — the paper may use more aggressive augmentation or curriculum learning

### Strategies tried and remaining

| Strategy | Expected Impact | Difficulty | Status | Actual Result |
|---|---|---|---|---|
| Flip-TTA | -0.5 to -1.0 | Easy | **Done** | -0.44 to -0.58 gain |
| Higher res heatmaps (112x112) | -0.3 to -0.5 | Easy | **Done** | -0.14 (10.72->10.58) |
| Heatmap supervision (hybrid loss) | -0.8 to -1.5 | Medium | **Failed** | NME 32.81 -- doesn't converge |
| Scale TTA (multi-scale inference) | -0.2 to -0.4 | Easy | Not done | -- |
| Aggressive augmentation (mixup/cutout) | -0.5 to -1.5 | Medium | Not done | -- |
| Multi-scale feature fusion (FPN-lite) | -0.4 to -1.0 | Medium | Not done | -- |
| Stronger regularization (dropout/decay) | -0.3 to -0.5 | Easy | Not done | -- |
| ELD-style ensemble (2-3 specialist models) | -2.0 to -4.0 | Hard | Not done | -- |

### Recommended next steps (in priority order)

1. **Mixup / CutMix augmentation** (Easy, high potential) — directly attacks the overfitting gap. Mix pairs of training images and their landmark coordinates. This is the lowest-hanging fruit since overfitting is the primary bottleneck.

2. **Scale TTA** (Easy, free gain) — run inference at 3 scales (e.g., 0.9x, 1.0x, 1.1x), average predictions. Stacks with existing flip-TTA for ~6 forward passes total. Expected additional gain: 0.2-0.4.

3. **Stronger dropout** (Easy, quick test) — increase heatmap_dropout from 0.1 to 0.2 or 0.3. Quick experiment, 1-2 training runs.

4. **ELD-style ensemble** (Hard, biggest gain) — this is what bridges the gap from ~10 to ~6.5 in the paper. Train 2-3 specialist models on landmark subsets (ears, eyes+nose, mouth). Each model sees all landmarks but is optimized for its subset. Average specialist predictions. Expected gain: 2-4 points.

### Realistic targets (updated)
- Single model + flip-TTA: **10.14** (achieved)
- Single model + better regularization + multi-scale TTA: **~9.0-9.5**
- ELD ensemble (2-3 specialists) + TTA: **~7.0-8.0**
- Full ELD ensemble (5+ specialists) + aggressive TTA: **~6.5-7.0**

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

### Current best preset (`heatmap_v2s_112`)
```python
backbone = "efficientnetv2s"
head_type = "heatmap"
heatmap_dropout = 0.1
num_deconv_layers = 4          # 4 deconv -> 112x112 heatmaps (vs 3 -> 56x56)
epochs = 100                   # Phase 1: frozen backbone
finetune_epochs = 200          # Phase 2: fine-tune last 50 layers
finetune_learning_rate = 1e-5
finetune_last_layers = 50
batch_size = 16
learning_rate = 1e-4
optimizer = "adamw"
weight_decay = 1e-4
loss = "mse"
# Augmentations: rotation, flip, crop_jitter (0.08), scale (0.85-1.15),
#                brightness, contrast, saturation, color_balance, sharpness, blur, noise
```

### Previous best preset (`heatmap_v2s_best`)
Same as above but with `num_deconv_layers = 3` (56x56 heatmaps).

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
| `scripts/train_dog_face_landmarks.py` | Main training script — all architectures, presets, augmentations, SoftArgmax2D layer |
| `scripts/gen_landmark_examples.py` | Generate _pred/_true visualizations for all 480 test images |
| `scripts/infer_dog_landmarks_tflite.py` | TFLite inference pipeline (two-stage: bbox detection + landmarks) |
| `scripts/run_nme_push.py` | Automated experiment runner with TTA evaluation |
| `scripts/run_experiments.py` | Earlier experiment runner for ablation studies |
| `CODEX_ADVISOR_REPORT.md` | Detailed analysis from OpenAI Codex consultation on strategies |
| `artifacts/heatmap_v2s_112/` | **Current best model** (NME_IOD 10.58 / 10.14 with TTA) |
| `artifacts/heatmap_v2s_best/` | Previous best model (NME_IOD 10.72) |
| `artifacts/dog_face_landmarks/` | Production TFLite model + inference example images |
| `artifacts/nme_push_results.md` | Raw experiment logs from overnight NME push run |

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
