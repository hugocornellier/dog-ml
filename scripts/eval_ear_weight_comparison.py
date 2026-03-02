#!/usr/bin/env python3
"""Evaluation and per-region comparison for tight_margin_256 vs tight_margin_256_ear_weight.

Part 1 — TTA evaluation of tight_margin_256_ear_weight:
  1. Baseline (no TTA)
  2. Flip TTA only
  3. Multi-scale + flip TTA (scales [0.9, 1.0, 1.1], 6 passes)

Part 2 — Per-region NME_IOD comparison table:
  Both models evaluated with flip-TTA, results shown side-by-side.
  Regions: right_ear 0-8, left_ear 9-17, right_eye 18-23, left_eye 24-29,
           nose_bridge 30-33, nose_nostrils 34-41, mouth 42-45

Usage:
    PYTHONUNBUFFERED=1 python scripts/eval_ear_weight_comparison.py
"""

from __future__ import annotations

import copy
import os
import sys
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

MODEL_BASE = Path(__file__).resolve().parent.parent / "artifacts" / "tight_margin_256" / "best.keras"
MODEL_EAR  = Path(__file__).resolve().parent.parent / "artifacts" / "tight_margin_256_ear_weight" / "best.keras"

SCALES = [0.9, 1.0, 1.1]

REGIONS = {
    "right_ear":     list(range(0, 9)),
    "left_ear":      list(range(9, 18)),
    "right_eye":     list(range(18, 24)),
    "left_eye":      list(range(24, 30)),
    "nose_bridge":   list(range(30, 34)),
    "nose_nostrils": list(range(34, 42)),
    "mouth":         list(range(42, 46)),
}

CUSTOM_OBJECTS = {"SoftArgmax2D": SoftArgmax2D, "WarmupSchedule": WarmupSchedule}


# ---------------------------------------------------------------------------
# Scale / flip helpers (same as eval_multiscale_tta.py)
# ---------------------------------------------------------------------------

def scale_crop(crop: tf.Tensor, scale: float) -> tf.Tensor:
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
    if abs(scale - 1.0) < 1e-6:
        return coords_2d
    return (coords_2d - 0.5) * scale + 0.5


def predict_scaled(model, crop: tf.Tensor, scale: float) -> tf.Tensor:
    scaled = scale_crop(crop, scale)
    raw = model(tf.expand_dims(scaled, 0), training=False)[0]
    coords_2d = tf.reshape(raw, [NUM_LANDMARKS, 2])
    return unscale_coords(coords_2d, scale)


def predict_scaled_flipped(model, crop: tf.Tensor, scale: float, flip_idx: tf.Tensor) -> tf.Tensor:
    scaled = scale_crop(crop, scale)
    flipped = tf.image.flip_left_right(scaled)
    raw = model(tf.expand_dims(flipped, 0), training=False)[0]
    coords_2d = tf.reshape(raw, [NUM_LANDMARKS, 2])
    remapped = tf.gather(coords_2d, flip_idx, axis=0)
    unflipped = tf.stack([1.0 - remapped[:, 0], remapped[:, 1]], axis=-1)
    return unscale_coords(unflipped, scale)


def compute_nme(pred_2d: tf.Tensor, true_2d: tf.Tensor) -> float:
    iod = tf.sqrt(
        tf.reduce_sum(
            tf.square(true_2d[LEFT_OUTER_EYE_IDX] - true_2d[RIGHT_OUTER_EYE_IDX])
        ) + 1e-8
    )
    dists = tf.sqrt(tf.reduce_sum(tf.square(pred_2d - true_2d), axis=-1) + 1e-8)
    return float(tf.reduce_mean(dists) / tf.maximum(iod, 1e-8) * 100.0)


# ---------------------------------------------------------------------------
# Part 1: multi-mode TTA evaluation
# ---------------------------------------------------------------------------

def evaluate_tta(model, val_records, cfg, label: str):
    """Run baseline, flip-TTA, and multi-scale+flip-TTA over val_records."""
    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)

    nmes_baseline = []
    nmes_flip = []
    nmes_ms_flip = []

    n = len(val_records)
    for i, rec in enumerate(val_records):
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        lm_flat = tf.constant(
            [c for pt in rec.landmarks_abs for c in pt], dtype=tf.float32
        )
        crop, lm_norm = crop_and_normalize(
            image,
            tf.constant(rec.bbox_xyxy_abs, tf.float32),
            lm_flat,
            cfg.img_size,
            cfg.crop_margin,
        )
        true_2d = tf.reshape(lm_norm, [NUM_LANDMARKS, 2])

        # Baseline
        coords_base = predict_scaled(model, crop, 1.0)
        nmes_baseline.append(compute_nme(coords_base, true_2d))

        # Flip TTA
        coords_flip = predict_scaled_flipped(model, crop, 1.0, flip_idx)
        coords_flip_avg = (coords_base + coords_flip) / 2.0
        nmes_flip.append(compute_nme(coords_flip_avg, true_2d))

        # Multi-scale + flip TTA (6 passes)
        preds_ms      = [predict_scaled(model, crop, s) for s in SCALES]
        preds_ms_flip = [predict_scaled_flipped(model, crop, s, flip_idx) for s in SCALES]
        all_preds = preds_ms + preds_ms_flip
        coords_msf_avg = tf.reduce_mean(tf.stack(all_preds, axis=0), axis=0)
        nmes_ms_flip.append(compute_nme(coords_msf_avg, true_2d))

        if (i + 1) % 100 == 0:
            print(
                f"  [{i+1}/{n}]  "
                f"baseline={np.mean(nmes_baseline):.4f}  "
                f"flip={np.mean(nmes_flip):.4f}  "
                f"ms+flip={np.mean(nmes_ms_flip):.4f}"
            )

    return {
        "baseline":       float(np.mean(nmes_baseline)),
        "flip_tta":       float(np.mean(nmes_flip)),
        "multiscale_flip": float(np.mean(nmes_ms_flip)),
    }


# ---------------------------------------------------------------------------
# Part 2: per-landmark / per-region analysis (flip-TTA)
# ---------------------------------------------------------------------------

def compute_per_landmark_errors(model, val_records, cfg):
    """Return array [N, 46] of per-landmark NME_IOD using flip-TTA."""
    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)
    all_errors = []

    for i, rec in enumerate(val_records):
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        lm_flat = tf.constant(
            [c for pt in rec.landmarks_abs for c in pt], dtype=tf.float32
        )
        crop, lm_norm = crop_and_normalize(
            image,
            tf.constant(rec.bbox_xyxy_abs, tf.float32),
            lm_flat,
            cfg.img_size,
            cfg.crop_margin,
        )

        coords_orig = model(tf.expand_dims(crop, 0), training=False)[0]

        crop_flip = tf.image.flip_left_right(crop)
        coords_flip = model(tf.expand_dims(crop_flip, 0), training=False)[0]
        cf2d = tf.reshape(coords_flip, [NUM_LANDMARKS, 2])
        cf_remap = tf.gather(cf2d, flip_idx, axis=0)
        cf_unflip = tf.stack([1.0 - cf_remap[:, 0], cf_remap[:, 1]], axis=-1)

        coords_avg = (coords_orig + tf.reshape(cf_unflip, [NUM_LANDMARKS * 2])) / 2.0

        pred_2d = tf.reshape(coords_avg, [NUM_LANDMARKS, 2])
        true_2d = tf.reshape(lm_norm, [NUM_LANDMARKS, 2])

        iod = float(tf.sqrt(
            tf.reduce_sum(tf.square(
                true_2d[LEFT_OUTER_EYE_IDX] - true_2d[RIGHT_OUTER_EYE_IDX]
            )) + 1e-8
        ))
        iod = max(iod, 1e-8)

        dists = np.sqrt(np.sum((pred_2d.numpy() - true_2d.numpy()) ** 2, axis=-1))
        all_errors.append(dists / iod * 100.0)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(val_records)}]  overall NME_IOD (flip-TTA) = {np.mean(all_errors):.4f}")

    return np.array(all_errors)  # [N, 46]


def region_nmes_from_per_lm(per_lm_mean):
    result = {}
    for name, indices in REGIONS.items():
        result[name] = float(np.mean(per_lm_mean[indices]))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    configure_ca_bundle()
    set_seed(42)

    # ------------------------------------------------------------------
    # Part 1: TTA evaluation of tight_margin_256_ear_weight
    # ------------------------------------------------------------------
    print("=" * 68)
    print("PART 1 — TTA Evaluation: tight_margin_256_ear_weight")
    print("=" * 68)

    cfg_ear = copy.deepcopy(EXPERIMENT_PRESETS["tight_margin_256_ear_weight"])
    print(f"Preset:      tight_margin_256_ear_weight")
    print(f"img_size:    {cfg_ear.img_size}")
    print(f"crop_margin: {cfg_ear.crop_margin}")
    print(f"lm_margin:   {cfg_ear.lm_margin}")
    print(f"Scales:      {SCALES}")
    print(f"Model:       {MODEL_EAR}")

    print(f"\nLoading val set...")
    val_records_ear = load_split_records(DATA_ROOT, "test", cfg_ear.lm_margin)
    print(f"Val records: {len(val_records_ear)}")

    print(f"\nLoading ear_weight model...")
    model_ear = tf.keras.models.load_model(
        str(MODEL_EAR), custom_objects=CUSTOM_OBJECTS, compile=False
    )
    print(f"Model output shape: {model_ear.output_shape}")

    print(f"\nRunning TTA evaluation ({len(val_records_ear)} samples)...")
    results_ear_tta = evaluate_tta(model_ear, val_records_ear, cfg_ear, "ear_weight")

    print()
    print("=" * 68)
    print("TTA RESULTS — tight_margin_256_ear_weight")
    print("=" * 68)
    print(f"  1. Baseline (no TTA):          {results_ear_tta['baseline']:.4f}")
    print(f"  2. Flip TTA only:              {results_ear_tta['flip_tta']:.4f}")
    print(f"  3. Multi-scale + flip (6p):    {results_ear_tta['multiscale_flip']:.4f}")
    print("=" * 68)
    print(f"\nReference tight_margin_256 (baseline):   9.27")
    print(f"Reference tight_margin_256 (flip TTA):   8.88")
    print(f"Reference tight_margin_256 (ms+flip):    8.66")

    # ------------------------------------------------------------------
    # Part 2: Per-region comparison table
    # ------------------------------------------------------------------
    print()
    print("=" * 68)
    print("PART 2 — Per-Region NME_IOD Comparison (flip-TTA)")
    print("=" * 68)

    # --- tight_margin_256 ---
    cfg_base = copy.deepcopy(EXPERIMENT_PRESETS["tight_margin_256"])
    print(f"\nLoading val set for tight_margin_256 (lm_margin={cfg_base.lm_margin})...")
    val_records_base = load_split_records(DATA_ROOT, "test", cfg_base.lm_margin)
    print(f"Val records: {len(val_records_base)}")

    print(f"\nLoading tight_margin_256 model...")
    model_base = tf.keras.models.load_model(
        str(MODEL_BASE), custom_objects=CUSTOM_OBJECTS, compile=False
    )
    print(f"Model output shape: {model_base.output_shape}")

    print(f"\nComputing per-landmark errors for tight_margin_256 (flip-TTA)...")
    errors_base = compute_per_landmark_errors(model_base, val_records_base, cfg_base)
    per_lm_base = errors_base.mean(axis=0)

    # --- tight_margin_256_ear_weight (reuse already-loaded model) ---
    print(f"\nComputing per-landmark errors for tight_margin_256_ear_weight (flip-TTA)...")
    errors_ear = compute_per_landmark_errors(model_ear, val_records_ear, cfg_ear)
    per_lm_ear = errors_ear.mean(axis=0)

    # Build comparison table
    rnmes_base = region_nmes_from_per_lm(per_lm_base)
    rnmes_ear  = region_nmes_from_per_lm(per_lm_ear)
    overall_base = float(np.mean(per_lm_base))
    overall_ear  = float(np.mean(per_lm_ear))

    print()
    print("=" * 80)
    print("PER-REGION NME_IOD COMPARISON (flip-TTA)")
    print("=" * 80)
    print(
        f"  {'Region':<20}  {'LMs':<6}  {'base_256':>10}  {'ear_weight':>10}  {'Delta':>10}  {'Winner':>8}"
    )
    print(f"  {'-'*74}")
    for region_name, indices in REGIONS.items():
        nme_b = rnmes_base[region_name]
        nme_e = rnmes_ear[region_name]
        delta = nme_e - nme_b
        winner = "ear_wt" if nme_e < nme_b else ("base" if nme_b < nme_e else "tie")
        lm_range = f"{indices[0]}-{indices[-1]}"
        print(
            f"  {region_name:<20}  {lm_range:<6}  {nme_b:>10.4f}  {nme_e:>10.4f}  "
            f"{delta:>+10.4f}  {winner:>8}"
        )
    print(f"  {'-'*74}")
    print(
        f"  {'OVERALL':<20}  {'0-45':<6}  {overall_base:>10.4f}  {overall_ear:>10.4f}  "
        f"{overall_ear - overall_base:>+10.4f}  "
        f"{'ear_wt' if overall_ear < overall_base else 'base':>8}"
    )
    print("=" * 80)

    # Per-landmark detail for ears
    print()
    print("Per-landmark NME_IOD for ear landmarks (0-17):")
    lm_to_region = {}
    for rname, idxs in REGIONS.items():
        for idx in idxs:
            lm_to_region[idx] = rname
    print(f"  {'LM':<6}  {'Region':<14}  {'base_256':>10}  {'ear_weight':>10}  {'Delta':>10}")
    print(f"  {'-'*56}")
    for lm_idx in range(18):
        b = per_lm_base[lm_idx]
        e = per_lm_ear[lm_idx]
        print(
            f"  {lm_idx:<6}  {lm_to_region[lm_idx]:<14}  {b:>10.4f}  {e:>10.4f}  {e-b:>+10.4f}"
        )

    print()
    print("Per-landmark NME_IOD for non-ear landmarks (18-45):")
    print(f"  {'LM':<6}  {'Region':<16}  {'base_256':>10}  {'ear_weight':>10}  {'Delta':>10}")
    print(f"  {'-'*58}")
    for lm_idx in range(18, 46):
        b = per_lm_base[lm_idx]
        e = per_lm_ear[lm_idx]
        print(
            f"  {lm_idx:<6}  {lm_to_region[lm_idx]:<16}  {b:>10.4f}  {e:>10.4f}  {e-b:>+10.4f}"
        )


if __name__ == "__main__":
    main()
