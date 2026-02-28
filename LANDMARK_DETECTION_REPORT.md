# Dog Facial Landmark Detection: Progress Journal

**Current best: NME_IOD = 10.72 | Target: 6.52 (paper) | Started at: ~40**

This document is a living journal of our work improving dog facial landmark detection. It's designed for future LLMs/developers to pick up where we left off and continue pushing toward the paper's target.

---

## Quick Reference

### Current Best Model
- **Preset**: `heatmap_v2s_best`
- **Val NME_IOD**: 10.72 (best epoch 79 of Phase 2)
- **Architecture**: EfficientNetV2S + 3-deconv heatmap head + SoftArgmax2D
- **Key improvements over baseline**: heatmap head, horizontal flip augmentation, scale augmentation, AdamW optimizer, two-phase backbone fine-tuning
- **Artifacts**: `artifacts/heatmap_v2s_best/`
- **TFLite**: `artifacts/dog_face_landmarks/dog_face_landmarks_224_float16.tflite` (53 MB)

### Key Commands
```bash
# Train with the best preset
python scripts/train_dog_face_landmarks.py --experiment heatmap_v2s_best --out artifacts/my_run

# Generate _pred/_true visualizations (uses ALL 480 test images)
python scripts/gen_landmark_examples.py

# Quick smoke test (5 epochs)
python scripts/train_dog_face_landmarks.py --experiment heatmap_v2s_best --epochs 5 --patience 50 --out artifacts/smoke
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

Replaced the dense head with a SimpleBaseline-style deconv heatmap head.

**Architecture**:
```
EfficientNetV2S (frozen, 224×224)
    -> 7×7×1280 feature map
    -> Conv2DTranspose(256, 4×4, stride 2) + BN + ReLU  -> 14×14
    -> Conv2DTranspose(256, 4×4, stride 2) + BN + ReLU  -> 28×28
    -> Conv2DTranspose(256, 4×4, stride 2) + BN + ReLU  -> 56×56
    -> Conv2D(46, 1×1)  -> 56×56 heatmaps
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
| **heatmap_v2s_best** (two-phase, flip+scale+AdamW) | **10.72** | **Current best** |

**Key learnings**:
- Horizontal flip is the single most impactful augmentation (train/val gap: 7.7 -> 3.6)
- Scale augmentation + crop jitter together simulate detector error well
- AdamW with weight_decay=1e-4 slightly better than Adam
- With flip, frozen backbone (11.68) nearly matches previous fine-tuned model (11.28)
- Two-phase + flip: 10.72 — a new frontier

---

## What Worked (Ranked by Impact)

1. **Heatmap head replacing GAP+Dense** — 40 → 12 (3x improvement, the breakthrough)
2. **Horizontal flip augmentation** — 12 → 11.7 (halves overfitting gap)
3. **Two-phase backbone fine-tuning** — 11.7 → 10.7 (adapts features to task)
4. **Scale augmentation** — modest but synergistic with flip
5. **AdamW optimizer** — slight improvement over Adam

## What Didn't Work

1. **Wing loss** — worse than MSE (40.97 vs 40.16)
2. **SpatialDropout2D** — negligible improvement (12.25 vs 12.18)
3. **Dense head fine-tuning** — fragile, collapsed at any LR > 1e-6
4. **Crop jitter alone** — minimal impact without flip/scale

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

## Remaining Gap: 10.72 → 6.52

### What the paper does that we don't
1. **ELD Ensemble** (biggest factor) — multiple specialized models per landmark subset
2. **Test-time augmentation (TTA)** — average predictions from flipped/scaled versions
3. **Possibly heatmap-level supervision** — Gaussian targets on heatmaps vs our coord-only MSE

### Strategies to try next (from Codex advisor analysis in CODEX_ADVISOR_REPORT.md)

| Strategy | Expected Impact | Difficulty | Status |
|---|---|---|---|
| Test-time augmentation (flip-TTA) | -0.5 to -1.0 | Easy | Not done |
| Heatmap target supervision (hybrid loss) | -0.8 to -1.5 | Medium | Not done |
| Multi-scale feature fusion (FPN-lite) | -0.4 to -1.0 | Medium | Not done |
| ELD-style ensemble (2-3 specialist models) | -2.0 to -4.0 | Hard | Not done |
| Eye-specific cascade refinement | Eye accuracy boost | Medium | Not done |
| Higher res heatmaps (4th deconv → 112×112) | -0.3 to -0.5 | Easy | Not done |

**Realistic targets**:
- Single model + TTA: ~9.5-10.0
- Single model + heatmap loss + TTA: ~8.5-9.5
- ELD ensemble: ~7.0-8.0

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

---

## Training Configuration Reference

### Best preset (`heatmap_v2s_best`)
```python
backbone = "efficientnetv2s"
head_type = "heatmap"
heatmap_dropout = 0.1
epochs = 100 (Phase 1) + 200 (Phase 2)
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

### Training stability notes
- lr=1e-3 ALWAYS diverges (even frozen backbone)
- lr=1e-4 with ReduceLROnPlateau is the sweet spot
- Backbone fine-tuning: lr=1e-5 for last 50 layers, with 5-epoch warmup
- ReduceLROnPlateau incompatible with CosineDecay schedule
- Keep BN layers frozen during fine-tuning

---

## Key Files

| File | Description |
|---|---|
| `scripts/train_dog_face_landmarks.py` | Main training script — all architectures, presets, augmentations |
| `scripts/gen_landmark_examples.py` | Generate _pred/_true visualizations for all test images |
| `scripts/infer_dog_landmarks_tflite.py` | TFLite inference pipeline |
| `scripts/run_experiments.py` | Experiment runner for ablation studies |
| `CODEX_ADVISOR_REPORT.md` | Detailed analysis from OpenAI Codex consultation |
| `artifacts/heatmap_v2s_best/` | Current best model (NME_IOD 10.72) |
| `artifacts/dog_face_landmarks/` | Production TFLite model + inference examples |

---

## NME_IOD Metric

- **NME** = Normalized Mean Error
- **IOD** = Inter-Ocular Distance (outer eye corners: landmarks 18 and 19)
- Formula: `mean(per_landmark_euclidean_distance) / IOD × 100`
- Lower is better. Paper target: 6.52
- Landmarks are normalized to [0,1] relative to crop coordinates
