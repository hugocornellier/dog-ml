# Deep Analysis & Optimization Prompt for Dog Facial Landmark Detection

You are tasked with pushing a dog facial landmark detection model's NME_IOD from **8.22** toward the paper's target of **6.52**.

## Current state (as of March 2026)

- **Overall best**: NME_IOD = **8.22** — 2-model ensemble (256px + 320px) + multi-scale + flip TTA (12 forward passes)
- **Best single model**: NME_IOD = **8.82** no TTA / **8.52** flip TTA / **8.32** ms+flip TTA (tight_margin_320, 320×320 input, 160×160 heatmaps)
- **Architecture**: EfficientNetV2S backbone + 4× Conv2DTranspose(256) + Conv2D(46) + SoftArgmax2D
- **Train-val gap**: ~3.4 (train ~5.4 vs val 8.82) — structural, not fixable by regularization
- **Biggest remaining error**: Ears (NME 12-14) are 3× worse than eyes (NME ~5). Ear tips (landmarks 5-7, 9-13) at 15-18 NME.
- **Paper's best**: 6.52 using ELD (Ensemble of Landmark Detectors) — specialized coarse-to-fine cascade with region-specific models

## Step 1: Read the full progress journal

Read `LANDMARK_DETECTION_REPORT.md` in the project root. This is a comprehensive living journal documenting every experiment across 6 rounds, what worked, what failed, and why. Pay close attention to:
- The "What Worked" and "What Didn't Work" ranked lists
- The Round 6 experiment table (23 experiments covering resolution, beta, augmentation, ensemble, regularization)
- The per-landmark error analysis (ears dominate the remaining error)
- The overfitting analysis table (every model shows ~3.4-3.6 gap)
- The "Remaining Gap: 8.22 → 6.52" section

**NOTE**: The "Recommended next steps" section in the journal is STALE — it still lists pure heatmap supervision and beta sweep as priorities, but both have been thoroughly tried and failed. Ignore that section and focus on the analysis below.

## Step 2: Study the research paper

Search for "DogFLW: Dog Facial Landmarks in the Wild" by Martvel, Farjon, & Kovalenko (2025), published in Pattern Recognition. Also check the Kaggle dataset page: https://www.kaggle.com/datasets/georgemartvel/dogflw

Focus specifically on:
- **How ELD achieves 6.52** — what are the specialist landmark subsets? How many models? How are regions cropped and predictions assembled?
- **Their single-model baseline** — DeepLabCut via DeepPoseKit with DenseNet121 achieves **6.70** as a single model. Understand how.
- **Their training recipe** — augmentation details, LR schedule, optimizer, epochs, batch size
- **SubpixelMaxima2D vs SoftArgmax2D** — DeepPoseKit uses Fourier-domain peak finding, not spatial expectation

## Step 3: Read the codebase

Read these files:
- `scripts/train_dog_face_landmarks.py` — main training script (all presets, SoftArgmax2D, data pipeline, two-phase training)
- `scripts/eval_320_comprehensive.py` — 320px eval + 2/4-model ensemble comparison
- `scripts/eval_multiscale_tta.py` — multi-scale TTA implementation
- `scripts/per_landmark_analysis.py` — per-landmark/region error breakdown

Also skim `scripts/eval_ensemble_multiscale.py` and `scripts/gen_landmark_examples.py` for the inference and visualization pipeline.

## Step 4: Formulate a plan

The gap from 8.22 to 6.52 is **1.70 points**. Based on the trajectory of experiments, here are the most promising remaining levers. Create an actionable, prioritized plan from these and any other ideas from the paper:

### High-confidence levers (resolution keeps working)
1. **384×384 or 448×448 input resolution** — Resolution has been the #1 lever throughout: 224→256 gave 0.26, 256→320 gave 0.45. The returns haven't diminished. 384px should give another 0.2-0.4. Training will be slower (~5-6 hrs) and may need batch_size=8.
2. **EfficientNetV2M backbone** — More capacity than V2S. Also creates a genuinely different model for ensembling (unlike our 224px models which were too similar). The backbone diversity should compound with resolution diversity.
3. **3+ model ensemble with resolution diversity** — Current 2-model (256+320) gives 8.22. Adding a 384px model (and potentially a V2M model) should push further. Resolution diversity > model count.

### Medium-confidence levers
4. **Cosine annealing LR schedule** — Current ReduceLROnPlateau may plateau too early. Cosine annealing with warm restarts could find better minima. Note: our code has a compatibility note that ReduceLROnPlateau is incompatible with CosineDecay — need to choose one.
5. **Longer Phase 2 training at 320px** (300 epochs) — Phase 2 often runs to max epochs without early stopping, suggesting the model is still improving. More epochs = more fine-tuning.
6. **ELD-style specialist models** — Train separate models for high-error regions (especially ears, landmarks 0-17). The specialist crops the ear region from coarse predictions, resizes to full input resolution (giving 4-8× effective resolution boost), and predicts only ear landmarks. This is how the paper bridges from ~8.5 to 6.52.

### Speculative levers
7. **Multi-scale feature fusion (FPN-lite)** — Add lateral connections from earlier backbone stages to the deconv head for better multi-scale feature representation.
8. **Moderate augmentation tuning** — 30° rotation was too aggressive, but 20° is untested. May be the sweet spot.
9. **Knowledge distillation** — Use the ensemble's predictions as soft labels to train a single stronger model.

## Important context
- Hardware: M-series Mac with Metal GPU (single GPU training)
- Framework: TensorFlow/Keras with TFLite export requirement
- Dataset: DogFLW — 3,853 train / 480 test, 46 landmarks, 120 dog breeds
- Training time: ~3-4 hours at 320px, ~5-6 hours estimated at 384px
- The model MUST export to TFLite for mobile inference

## What NOT to try (all thoroughly tested and failed)
1. **Pure heatmap supervision** — NME 33.7. Background-dominated MSE loss. Tried sigma=2.5, 3.5, 10 — all failed. Gaussians cover <0.2% of heatmap pixels.
2. **Heatmap supervision / Gaussian target hybrid loss** — NME 32.81. Dual loss through shared backbone fundamentally interferes with coordinate regression.
3. **Heatmap-level TTA** — Produces 2× worse results (NME 22.31). Must average at coordinate level, never at heatmap level.
4. **Wing loss** — Worse than MSE (40.97 vs 40.16)
5. **SpatialDropout2D** — Negligible impact
6. **Dense regression head (GAP+Dense)** — 3× worse than heatmap head
7. **Mixup augmentation** — Causes underfitting. Blending destroys fine-grained spatial precision.
8. **Strong augmentation (30° rotation, 0.75-1.25 scale)** — NME 9.67 vs 9.53 baseline. Too aggressive for tight crops.
9. **Post-hoc SoftArgmax beta tuning** — Higher beta at inference always worse for coord-trained models. Beta monotonically increases NME.
10. **Training with beta=10** — NME 9.43 vs 9.23 at 256px. No benefit at higher resolutions.
11. **Ear-weighted loss (2× on ear landmarks)** — Ears barely improved (-0.08), everything else regressed (mouth +0.43). Net effect: worse.
12. **Higher regularization (dropout 0.25, weight_decay 3e-4)** — NME 9.31 vs 9.23. The train-val gap is structural (insufficient data diversity), not solvable by regularization.
13. **Weighted TTA / weighted ensemble** — Equal weights already optimal. Center-heavy weighting hurts. Trimmed mean gives <0.01 improvement.

## Key constants and landmarks
- **46 landmarks** total
- **FLIP_INDEX**: verified left-right swap (perfect involution)
- **Eye corners for IOD**: landmarks 18 (left) and 19 (right)
- **Landmark regions**: right_ear 0-8, left_ear 9-17, right_eye 18-23, left_eye 24-29, nose_bridge 30-33, nose_nostrils 34-41, mouth 42-45
- **Per-region NME** (with flip TTA): right_ear 14.0, left_ear 12.2, mouth 7.4, nose_nostrils 7.9, left_eye 5.2, right_eye 4.9, nose_bridge 4.9
