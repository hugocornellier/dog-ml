#!/usr/bin/env python3
"""Weighted multi-scale TTA evaluation for the tight_margin_256 model.

Tests whether weighted averaging of the 6-pass multi-scale+flip TTA predictions
improves over equal-weight averaging (baseline: NME_IOD ~8.66).

The 6 passes are ordered as:
  [scale=0.9 normal, scale=1.0 normal, scale=1.1 normal,
   scale=0.9 flipped, scale=1.0 flipped, scale=1.1 flipped]

Weighting schemes tested:
  1. Equal weights:    [1,   1,   1,   1,   1,   1  ]  (baseline ~8.66)
  2. Center-heavy:     [0.5, 2.0, 0.5, 0.5, 2.0, 0.5]
  3. Moderate center:  [0.7, 1.6, 0.7, 0.7, 1.6, 0.7]
  4. Soft center:      [0.8, 1.4, 0.8, 0.8, 1.4, 0.8]
  5. Trimmed mean:     drop per-landmark min/max across the 6 predictions, average remaining 4

Usage:
    PYTHONUNBUFFERED=1 python scripts/eval_weighted_tta.py
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
MODEL_PATH = Path("artifacts/tight_margin_256/best.keras")

SCALES = [0.9, 1.0, 1.1]

# Weighting schemes: weights for the 3 scales (applied equally to normal and flipped passes).
WEIGHT_SCHEMES = [
    ("Equal weights [1,1,1]",       [1.0, 1.0, 1.0]),
    ("Center-heavy [0.5,2.0,0.5]",  [0.5, 2.0, 0.5]),
    ("Moderate center [0.7,1.6,0.7]", [0.7, 1.6, 0.7]),
    ("Soft center [0.8,1.4,0.8]",   [0.8, 1.4, 0.8]),
]


def scale_crop(crop: tf.Tensor, scale: float) -> tf.Tensor:
    """Transform a 256x256 crop tensor to simulate a different scale."""
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
    """Map predicted coordinates from the scaled crop frame back to original frame."""
    if abs(scale - 1.0) < 1e-6:
        return coords_2d
    return (coords_2d - 0.5) * scale + 0.5


def predict_scaled(model, crop: tf.Tensor, scale: float) -> tf.Tensor:
    """Run inference on a scaled crop, return coords in original frame [NUM_LANDMARKS, 2]."""
    scaled = scale_crop(crop, scale)
    raw = model(tf.expand_dims(scaled, 0), training=False)[0]
    coords_2d = tf.reshape(raw, [NUM_LANDMARKS, 2])
    return unscale_coords(coords_2d, scale)


def predict_scaled_flipped(model, crop: tf.Tensor, scale: float, flip_idx: tf.Tensor) -> tf.Tensor:
    """Run inference on a flipped + scaled crop, return coords in original frame."""
    scaled = scale_crop(crop, scale)
    flipped = tf.image.flip_left_right(scaled)
    raw = model(tf.expand_dims(flipped, 0), training=False)[0]
    coords_2d = tf.reshape(raw, [NUM_LANDMARKS, 2])
    remapped = tf.gather(coords_2d, flip_idx, axis=0)
    unflipped = tf.stack([1.0 - remapped[:, 0], remapped[:, 1]], axis=-1)
    return unscale_coords(unflipped, scale)


def compute_nme(pred_2d: tf.Tensor, true_2d: tf.Tensor) -> float:
    """Compute NME_IOD (%) for a single sample."""
    iod = tf.sqrt(
        tf.reduce_sum(
            tf.square(true_2d[LEFT_OUTER_EYE_IDX] - true_2d[RIGHT_OUTER_EYE_IDX])
        ) + 1e-8
    )
    dists = tf.sqrt(tf.reduce_sum(tf.square(pred_2d - true_2d), axis=-1) + 1e-8)
    return float(tf.reduce_mean(dists) / tf.maximum(iod, 1e-8) * 100.0)


def weighted_average(preds: list[tf.Tensor], weights: list[float]) -> tf.Tensor:
    """Weighted average of a list of [NUM_LANDMARKS, 2] tensors.

    Args:
        preds: list of N tensors each [NUM_LANDMARKS, 2].
        weights: list of N float weights (need not sum to 1).

    Returns:
        Weighted average tensor [NUM_LANDMARKS, 2].
    """
    total_w = sum(weights)
    stacked = tf.stack(preds, axis=0)  # [N, NUM_LANDMARKS, 2]
    w_tensor = tf.constant(weights, dtype=tf.float32)  # [N]
    # Broadcast weights: [N, 1, 1]
    w_broadcast = tf.reshape(w_tensor, [-1, 1, 1])
    return tf.reduce_sum(stacked * w_broadcast, axis=0) / total_w


def trimmed_mean(preds: list[tf.Tensor]) -> tf.Tensor:
    """Trimmed mean: for each (landmark, coord), drop min and max, average the rest.

    Args:
        preds: list of N tensors each [NUM_LANDMARKS, 2].

    Returns:
        Trimmed-mean tensor [NUM_LANDMARKS, 2].
    """
    stacked = tf.stack(preds, axis=0)  # [N, NUM_LANDMARKS, 2]
    n = stacked.shape[0]
    sorted_preds = tf.sort(stacked, axis=0)  # sort along N dimension
    # Drop first and last (min and max), average middle N-2
    trimmed = sorted_preds[1:-1, :, :]  # [N-2, NUM_LANDMARKS, 2]
    return tf.reduce_mean(trimmed, axis=0)


def evaluate(model, val_records, cfg):
    """Run all weighting schemes + trimmed mean over val_records.

    Returns dict: scheme_label -> list of per-sample NME values.
    """
    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)

    # Initialize accumulators
    scheme_labels = [label for label, _ in WEIGHT_SCHEMES] + ["Trimmed mean (drop min/max)"]
    nmes = {label: [] for label in scheme_labels}

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

        # Collect all 6 predictions (3 normal + 3 flipped) in scale order
        preds_normal = [predict_scaled(model, crop, s) for s in SCALES]
        preds_flipped = [predict_scaled_flipped(model, crop, s, flip_idx) for s in SCALES]
        # Order: [0.9n, 1.0n, 1.1n, 0.9f, 1.0f, 1.1f]
        all_preds = preds_normal + preds_flipped

        # Weighted averaging schemes
        for label, scale_weights in WEIGHT_SCHEMES:
            # Each scale weight applies to both normal and flipped pass at that scale
            full_weights = scale_weights + scale_weights  # len=6
            avg = weighted_average(all_preds, full_weights)
            nmes[label].append(compute_nme(avg, true_2d))

        # Trimmed mean
        trimmed = trimmed_mean(all_preds)
        nmes["Trimmed mean (drop min/max)"].append(compute_nme(trimmed, true_2d))

        if (i + 1) % 100 == 0:
            parts = "  ".join(
                f"{lbl.split()[0]}={np.mean(vals):.4f}"
                for lbl, vals in nmes.items()
            )
            print(f"  [{i+1}/{n}]  {parts}")

    return {label: float(np.mean(vals)) for label, vals in nmes.items()}


def main():
    configure_ca_bundle()
    set_seed(42)

    cfg = copy.deepcopy(EXPERIMENT_PRESETS["tight_margin_256"])
    print(f"Preset:      tight_margin_256")
    print(f"img_size:    {cfg.img_size}")
    print(f"crop_margin: {cfg.crop_margin}")
    print(f"lm_margin:   {cfg.lm_margin}")
    print(f"Scales:      {SCALES}  (x2 with flip = 6 passes)")

    print(f"\nLoading val set (lm_margin={cfg.lm_margin})...")
    val_records = load_split_records(DATA_ROOT, "test", cfg.lm_margin)
    print(f"Val records: {len(val_records)}")

    print(f"\nLoading model from {MODEL_PATH} ...")
    custom_objects = {"SoftArgmax2D": SoftArgmax2D, "WarmupSchedule": WarmupSchedule}
    model = tf.keras.models.load_model(
        str(MODEL_PATH), custom_objects=custom_objects, compile=False
    )
    print(f"Model output shape: {model.output_shape}")

    num_schemes = len(WEIGHT_SCHEMES) + 1  # +1 for trimmed mean
    print(
        f"\nRunning weighted TTA evaluation "
        f"({len(val_records)} samples, 6 passes each, {num_schemes} schemes)..."
    )
    results = evaluate(model, val_records, cfg)

    col_w = 38
    print()
    print("=" * (col_w + 16))
    print("RESULTS — tight_margin_256 weighted multi-scale TTA")
    print("=" * (col_w + 16))
    print(f"  {'Scheme':<{col_w}} NME_IOD (%)")
    print(f"  {'-' * col_w} -----------")
    best_label = min(results, key=results.__getitem__)
    for idx, (label, nme) in enumerate(results.items(), start=1):
        marker = " <-- best" if label == best_label else ""
        print(f"  {idx}. {label:<{col_w - 3}} {nme:.4f}{marker}")
    print("=" * (col_w + 16))
    print(f"\nReference equal-weight ms+flip (8.66) and flip-only TTA (9.11)")
    print(f"Best: {best_label}  NME={results[best_label]:.4f}")


if __name__ == "__main__":
    main()
