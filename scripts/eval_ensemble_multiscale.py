#!/usr/bin/env python3
"""3-model ensemble with multi-scale TTA evaluation.

Combines two independent improvements that should compound:
  1. 3-model ensemble (diversity across training runs / architectures)
  2. Multi-scale TTA  (3 scales x 2 flip orientations = 6 passes per model)

Models:
  1. artifacts/tight_margin/best.keras       -- 224x224, preset tight_margin
  2. artifacts/tight_margin_beta10/best.keras -- 224x224, preset tight_margin_beta10
  3. artifacts/tight_margin_256/best.keras   -- 256x256, preset tight_margin_256

All share crop_margin=0.10, lm_margin=0.05 so normalized [0,1] landmarks are
in the same coordinate system regardless of img_size.

Configurations tested:
  A) 3-model x flip TTA only           =  6 predictions  (baseline, ~8.67 expected)
  B) 3-model x multi-scale (no flip)   =  9 predictions
  C) 3-model x multi-scale + flip TTA  = 18 predictions  (main target, ~8.3-8.5)
  D) Best-2 (beta10+256) x ms+flip     = 12 predictions

Scales: [0.9, 1.0, 1.1]

Coordinate mapping for scale s:
  p_original = (p_scaled - 0.5) * s + 0.5

Usage:
    PYTHONUNBUFFERED=1 python scripts/eval_ensemble_multiscale.py
"""

from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dog_face_landmarks import (
    NUM_LANDMARKS,
    FLIP_INDEX,
    LEFT_OUTER_EYE_IDX,
    RIGHT_OUTER_EYE_IDX,
    EXPERIMENT_PRESETS,
    SoftArgmax2D,
    WarmupSchedule,
    set_seed,
    configure_ca_bundle,
    load_split_records,
    crop_and_normalize,
)

DATA_ROOT = (
    Path.home() / ".cache" / "kagglehub" / "datasets"
    / "georgemartvel" / "dogflw" / "versions" / "1" / "DogFLW"
)

ARTIFACTS_ROOT = Path("artifacts")

MODELS_CONFIG = [
    {
        "name": "tight_margin",
        "path": ARTIFACTS_ROOT / "tight_margin" / "best.keras",
        "preset": "tight_margin",
        "individual_nme_no_tta": 9.53,
        "individual_nme_flip_tta": 9.11,
    },
    {
        "name": "tight_margin_beta10",
        "path": ARTIFACTS_ROOT / "tight_margin_beta10" / "best.keras",
        "preset": "tight_margin_beta10",
        "individual_nme_no_tta": 9.53,
        "individual_nme_flip_tta": 9.09,
    },
    {
        "name": "tight_margin_256",
        "path": ARTIFACTS_ROOT / "tight_margin_256" / "best.keras",
        "preset": "tight_margin_256",
        "individual_nme_no_tta": 9.27,
        "individual_nme_flip_tta": 8.88,
    },
]

SCALES = [0.9, 1.0, 1.1]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: Path):
    custom_objects = {"SoftArgmax2D": SoftArgmax2D, "WarmupSchedule": WarmupSchedule}
    return tf.keras.models.load_model(
        str(model_path), custom_objects=custom_objects, compile=False
    )


# ---------------------------------------------------------------------------
# Scale / flip helpers  (ported from eval_multiscale_tta.py)
# ---------------------------------------------------------------------------

def scale_crop(crop: tf.Tensor, scale: float) -> tf.Tensor:
    """Transform a HxW crop tensor to simulate a different viewing scale.

    scale < 1.0 (e.g. 0.9): zoom in  -- center-crop to scale*H, resize back.
    scale = 1.0             : identity.
    scale > 1.0 (e.g. 1.1): zoom out -- pad borders to scale*H, resize back.
    """
    if abs(scale - 1.0) < 1e-6:
        return crop

    h = tf.shape(crop)[0]
    w = tf.shape(crop)[1]
    h_f = tf.cast(h, tf.float32)
    w_f = tf.cast(w, tf.float32)

    if scale < 1.0:
        new_h = tf.cast(tf.math.round(h_f * scale), tf.int32)
        new_w = tf.cast(tf.math.round(w_f * scale), tf.int32)
        offset_y = (h - new_h) // 2
        offset_x = (w - new_w) // 2
        cropped = tf.image.crop_to_bounding_box(crop, offset_y, offset_x, new_h, new_w)
        return tf.image.resize(cropped, [h, w], antialias=True)
    else:
        pad_h = tf.cast(tf.math.round(h_f * (scale - 1.0) / 2.0), tf.int32)
        pad_w = tf.cast(tf.math.round(w_f * (scale - 1.0) / 2.0), tf.int32)
        padded = tf.pad(
            crop,
            [[pad_h, pad_h], [pad_w, pad_w], [0, 0]],
            mode="REFLECT",
        )
        return tf.image.resize(padded, [h, w], antialias=True)


def unscale_coords(coords_2d: tf.Tensor, scale: float) -> tf.Tensor:
    """Map coords predicted in the scaled frame back to the original frame.

    Inverse of: p_scaled = (p_original - 0.5) / s + 0.5
    =>           p_original = (p_scaled - 0.5) * s + 0.5
    """
    if abs(scale - 1.0) < 1e-6:
        return coords_2d
    return (coords_2d - 0.5) * scale + 0.5


def predict_scaled(model, crop: tf.Tensor, scale: float) -> tf.Tensor:
    """Run inference on a scaled crop; return [NUM_LANDMARKS, 2] in original frame."""
    scaled = scale_crop(crop, scale)
    raw = model(tf.expand_dims(scaled, 0), training=False)[0]
    coords_2d = tf.reshape(raw, [NUM_LANDMARKS, 2])
    return unscale_coords(coords_2d, scale)


def predict_scaled_flipped(
    model, crop: tf.Tensor, scale: float, flip_idx: tf.Tensor
) -> tf.Tensor:
    """Run inference on a flipped+scaled crop; return [NUM_LANDMARKS, 2] in original frame."""
    scaled = scale_crop(crop, scale)
    flipped = tf.image.flip_left_right(scaled)
    raw = model(tf.expand_dims(flipped, 0), training=False)[0]
    coords_2d = tf.reshape(raw, [NUM_LANDMARKS, 2])
    remapped = tf.gather(coords_2d, flip_idx, axis=0)
    unflipped = tf.stack([1.0 - remapped[:, 0], remapped[:, 1]], axis=-1)
    return unscale_coords(unflipped, scale)


# ---------------------------------------------------------------------------
# NME helper
# ---------------------------------------------------------------------------

def compute_nme(pred_2d: tf.Tensor, true_2d: tf.Tensor) -> float:
    """Compute NME_IOD (%) for a single sample."""
    iod = tf.sqrt(
        tf.reduce_sum(
            tf.square(true_2d[LEFT_OUTER_EYE_IDX] - true_2d[RIGHT_OUTER_EYE_IDX])
        ) + 1e-8
    )
    dists = tf.sqrt(tf.reduce_sum(tf.square(pred_2d - true_2d), axis=-1) + 1e-8)
    return float(tf.reduce_mean(dists) / tf.maximum(iod, 1e-8) * 100.0)


# ---------------------------------------------------------------------------
# Core evaluator: collect all per-model per-scale predictions, then average
# ---------------------------------------------------------------------------

def evaluate_config(
    models_info: list[dict],
    val_records,
    use_flip: bool,
    scales: list[float],
    tag: str,
) -> tuple[float, float]:
    """Evaluate an ensemble configuration.

    For each sample:
      - For each model, run inference at each scale (and optionally its flip).
      - Average all predictions.

    Returns (mean_nme, elapsed_seconds).
    """
    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)
    num_passes = len(models_info) * len(scales) * (2 if use_flip else 1)

    print(
        f"\n  [{tag}]  models={len(models_info)}  "
        f"scales={scales}  flip={use_flip}  "
        f"total_passes_per_sample={num_passes}"
    )

    nmes = []
    t0 = time.time()

    for i, rec in enumerate(val_records):
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        bbox = tf.constant(rec.bbox_xyxy_abs, tf.float32)
        lm_flat = tf.constant(
            [coord for pt in rec.landmarks_abs for coord in pt], dtype=tf.float32
        )

        # Ground-truth landmarks: use first model's crop geometry (all share same
        # crop_margin / lm_margin so normalized coords are identical).
        ref_cfg = models_info[0]["cfg"]
        _, lm_norm = crop_and_normalize(
            image, bbox, lm_flat, ref_cfg.img_size, ref_cfg.crop_margin,
        )
        true_2d = tf.reshape(lm_norm, [NUM_LANDMARKS, 2])

        # Collect all prediction tensors: [NUM_LANDMARKS, 2] each.
        all_preds = []
        for info in models_info:
            cfg = info["cfg"]
            lm_flat_dummy = tf.zeros([NUM_LANDMARKS * 2], dtype=tf.float32)
            crop, _ = crop_and_normalize(
                image, bbox, lm_flat_dummy, cfg.img_size, cfg.crop_margin,
            )
            for s in scales:
                all_preds.append(predict_scaled(info["model"], crop, s))
                if use_flip:
                    all_preds.append(predict_scaled_flipped(info["model"], crop, s, flip_idx))

        avg_pred = tf.reduce_mean(tf.stack(all_preds, axis=0), axis=0)
        nmes.append(compute_nme(avg_pred, true_2d))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"    [{i+1}/{len(val_records)}] "
                f"NME={np.mean(nmes):.4f}  "
                f"elapsed={elapsed:.0f}s"
            )

    elapsed = time.time() - t0
    return float(np.mean(nmes)), elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    configure_ca_bundle()
    set_seed(42)

    print("=" * 72)
    print("DOG LANDMARK: 3-MODEL ENSEMBLE + MULTI-SCALE TTA")
    print("=" * 72)
    print(f"Scales:  {SCALES}")
    print(f"Models:  {[mc['name'] for mc in MODELS_CONFIG]}")

    # Load val records (all presets share same lm_margin).
    ref_preset = copy.deepcopy(EXPERIMENT_PRESETS["tight_margin"])
    print(f"\nLoading val records from {DATA_ROOT} ...")
    val_records = load_split_records(DATA_ROOT, "test", ref_preset.lm_margin)
    print(f"Val records: {len(val_records)}")

    # Load all models.
    models_info = []
    for mc in MODELS_CONFIG:
        print(f"\nLoading {mc['name']} from {mc['path']} ...")
        t0 = time.time()
        model = load_model(mc["path"])
        cfg = copy.deepcopy(EXPERIMENT_PRESETS[mc["preset"]])
        models_info.append({"name": mc["name"], "model": model, "cfg": cfg, **mc})
        print(f"  Loaded in {time.time() - t0:.1f}s  (img_size={cfg.img_size})")

    all_results = []

    # ------------------------------------------------------------------
    # Config A: 3-model x flip TTA only (baseline reproduce ~8.67)
    # ------------------------------------------------------------------
    nme_a, t_a = evaluate_config(
        models_info, val_records,
        use_flip=True, scales=[1.0],
        tag="A: 3-model x flip TTA only  (6 preds)",
    )
    all_results.append({"label": "A) 3-model x flip TTA only         (  6 preds)", "nme": nme_a, "time": t_a})
    print(f"    => NME_IOD={nme_a:.4f}  [{t_a:.0f}s]")

    # ------------------------------------------------------------------
    # Config B: 3-model x multi-scale (no flip) = 9 predictions
    # ------------------------------------------------------------------
    nme_b, t_b = evaluate_config(
        models_info, val_records,
        use_flip=False, scales=SCALES,
        tag="B: 3-model x multi-scale, no flip  (9 preds)",
    )
    all_results.append({"label": "B) 3-model x multi-scale (no flip) (  9 preds)", "nme": nme_b, "time": t_b})
    print(f"    => NME_IOD={nme_b:.4f}  [{t_b:.0f}s]")

    # ------------------------------------------------------------------
    # Config C: 3-model x multi-scale + flip TTA = 18 predictions (MAIN)
    # ------------------------------------------------------------------
    nme_c, t_c = evaluate_config(
        models_info, val_records,
        use_flip=True, scales=SCALES,
        tag="C: 3-model x multi-scale + flip TTA  (18 preds)",
    )
    all_results.append({"label": "C) 3-model x ms + flip TTA         ( 18 preds)", "nme": nme_c, "time": t_c})
    print(f"    => NME_IOD={nme_c:.4f}  [{t_c:.0f}s]")

    # ------------------------------------------------------------------
    # Config D: best-2 (beta10 + 256) x multi-scale + flip = 12 predictions
    # ------------------------------------------------------------------
    best2 = [mi for mi in models_info if mi["name"] in ("tight_margin_beta10", "tight_margin_256")]
    nme_d, t_d = evaluate_config(
        best2, val_records,
        use_flip=True, scales=SCALES,
        tag="D: best-2 (beta10+256) x ms + flip  (12 preds)",
    )
    all_results.append({"label": "D) best-2 (beta10+256) x ms + flip ( 12 preds)", "nme": nme_d, "time": t_d})
    print(f"    => NME_IOD={nme_d:.4f}  [{t_d:.0f}s]")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print(f"{'Configuration':<52} {'NME_IOD':>8}  {'Time':>8}")
    print("-" * 72)

    # Reference lines
    print(f"  {'[REF] tight_margin_256 single-model flip TTA':<50} {'8.8800':>8}  {'':>8}")
    print(f"  {'[REF] 3-model flip TTA (eval_ensemble.py)':<50} {'~8.67':>8}  {'':>8}")
    print("-" * 72)

    for r in all_results:
        print(f"  {r['label']:<50} {r['nme']:>8.4f}  {r['time']:>6.0f}s")

    best = min(all_results, key=lambda r: r["nme"])
    print()
    print("=" * 72)
    print(f"BEST CONFIG: {best['label'].strip()}")
    print(f"  NME_IOD        = {best['nme']:.4f}")
    print(f"  vs best single = 8.88  (tight_margin_256 flip TTA)")
    print(f"  Improvement    = {8.88 - best['nme']:+.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
