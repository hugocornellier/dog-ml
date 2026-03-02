#!/usr/bin/env python3
"""Per-landmark NME_IOD analysis for the tight_margin_256 model.

Computes per-landmark error to identify easiest/hardest landmarks and regions,
informing whether an ELD-style ensemble (specialist per region) would help.

Usage:
    python scripts/per_landmark_analysis.py
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
MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "tight_margin_256" / "best.keras"

# DogFLW 46-landmark region groupings
REGIONS = {
    "right_ear":     list(range(0, 9)),    # 9 points
    "left_ear":      list(range(9, 18)),   # 9 points
    "right_eye":     list(range(18, 24)),  # 6 points
    "left_eye":      list(range(24, 30)),  # 6 points
    "nose_bridge":   list(range(30, 34)),  # 4 points
    "nose_nostrils": list(range(34, 42)),  # 8 points
    "mouth":         list(range(42, 46)),  # 4 points
}


def compute_per_landmark_errors(model, val_records, cfg):
    """For each val sample, compute per-landmark NME_IOD (46 values).

    Returns array of shape [N, 46] where each entry is the NME_IOD for that
    landmark on that sample.
    """
    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)
    all_errors = []  # list of [46] arrays

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

        # Original prediction
        coords_orig = model(tf.expand_dims(crop, 0), training=False)[0]

        # Flipped prediction (flip-TTA)
        crop_flip = tf.image.flip_left_right(crop)
        coords_flip = model(tf.expand_dims(crop_flip, 0), training=False)[0]

        # Unflip: remap landmark indices, then mirror x
        cf2d = tf.reshape(coords_flip, [NUM_LANDMARKS, 2])
        cf_remap = tf.gather(cf2d, flip_idx, axis=0)
        cf_unflip = tf.stack([1.0 - cf_remap[:, 0], cf_remap[:, 1]], axis=-1)

        # Average original + unflipped (TTA)
        coords_avg = (coords_orig + tf.reshape(cf_unflip, [NUM_LANDMARKS * 2])) / 2.0

        pred_2d = tf.reshape(coords_avg, [NUM_LANDMARKS, 2])
        true_2d = tf.reshape(lm_norm, [NUM_LANDMARKS, 2])

        # IOD from true landmarks
        iod = float(tf.sqrt(
            tf.reduce_sum(tf.square(
                true_2d[LEFT_OUTER_EYE_IDX] - true_2d[RIGHT_OUTER_EYE_IDX]
            )) + 1e-8
        ))
        iod = max(iod, 1e-8)

        # Per-landmark euclidean distance / IOD * 100
        dists = np.sqrt(
            np.sum(
                (pred_2d.numpy() - true_2d.numpy()) ** 2,
                axis=-1
            )
        )
        per_lm_nme = dists / iod * 100.0  # shape [46]
        all_errors.append(per_lm_nme)

        if (i + 1) % 100 == 0:
            arr = np.array(all_errors)
            overall_nme = float(np.mean(arr))
            print(f"  [{i+1}/{len(val_records)}]  overall NME_IOD (TTA) = {overall_nme:.4f}")

    return np.array(all_errors)  # [N, 46]


def print_region_summary(per_lm_mean):
    """Print per-region mean NME_IOD sorted worst to best."""
    region_nmes = {}
    for region_name, indices in REGIONS.items():
        region_nmes[region_name] = float(np.mean(per_lm_mean[indices]))

    sorted_regions = sorted(region_nmes.items(), key=lambda x: x[1], reverse=True)

    print("\nPer-region NME_IOD (worst to best):")
    print(f"  {'Region':<20}  {'Landmarks':<16}  {'Mean NME_IOD':>12}  {'Delta vs overall':>16}")
    overall = float(np.mean(per_lm_mean))
    print(f"  {'-'*70}")
    for region_name, nme in sorted_regions:
        indices = REGIONS[region_name]
        delta = nme - overall
        lm_range = f"{indices[0]}-{indices[-1]}"
        print(f"  {region_name:<20}  {lm_range:<16}  {nme:>12.4f}  {delta:>+15.4f}")


def print_hardest_easiest(per_lm_mean, n=10):
    """Print top-N hardest and easiest landmarks."""
    # Map landmark index to region name
    lm_to_region = {}
    for region_name, indices in REGIONS.items():
        for idx in indices:
            lm_to_region[idx] = region_name

    ranked = np.argsort(per_lm_mean)  # ascending

    print(f"\nTop {n} hardest landmarks:")
    print(f"  {'Rank':<6}  {'LM idx':<8}  {'Region':<20}  {'NME_IOD':>10}")
    print(f"  {'-'*50}")
    for rank, lm_idx in enumerate(ranked[::-1][:n], start=1):
        print(
            f"  {rank:<6}  {lm_idx:<8}  {lm_to_region[lm_idx]:<20}  "
            f"{per_lm_mean[lm_idx]:>10.4f}"
        )

    print(f"\nTop {n} easiest landmarks:")
    print(f"  {'Rank':<6}  {'LM idx':<8}  {'Region':<20}  {'NME_IOD':>10}")
    print(f"  {'-'*50}")
    for rank, lm_idx in enumerate(ranked[:n], start=1):
        print(
            f"  {rank:<6}  {lm_idx:<8}  {lm_to_region[lm_idx]:<20}  "
            f"{per_lm_mean[lm_idx]:>10.4f}"
        )


def print_overall_stats(per_lm_mean):
    """Print overall statistics across 46 landmark means."""
    print("\nOverall stats across 46 per-landmark mean NME_IOD values:")
    print(f"  Mean:   {np.mean(per_lm_mean):.4f}")
    print(f"  Median: {np.median(per_lm_mean):.4f}")
    print(f"  Std:    {np.std(per_lm_mean):.4f}")
    print(f"  Min:    {np.min(per_lm_mean):.4f}  (LM {int(np.argmin(per_lm_mean))})")
    print(f"  Max:    {np.max(per_lm_mean):.4f}  (LM {int(np.argmax(per_lm_mean))})")


def print_all_landmarks(per_lm_mean):
    """Print all 46 per-landmark NME_IOD values."""
    lm_to_region = {}
    for region_name, indices in REGIONS.items():
        for idx in indices:
            lm_to_region[idx] = region_name

    print("\nAll 46 per-landmark mean NME_IOD (sorted by index):")
    print(f"  {'LM idx':<8}  {'Region':<20}  {'NME_IOD':>10}")
    print(f"  {'-'*44}")
    for i, nme in enumerate(per_lm_mean):
        print(f"  {i:<8}  {lm_to_region[i]:<20}  {nme:>10.4f}")


def main():
    configure_ca_bundle()
    set_seed(42)

    cfg = copy.deepcopy(EXPERIMENT_PRESETS["tight_margin_256"])
    print("=" * 60)
    print("Per-Landmark NME_IOD Analysis — tight_margin_256")
    print("=" * 60)
    print(f"Preset:      tight_margin_256")
    print(f"img_size:    {cfg.img_size}")
    print(f"crop_margin: {cfg.crop_margin}")
    print(f"lm_margin:   {cfg.lm_margin}")
    print(f"Model:       {MODEL_PATH}")

    print(f"\nLoading val set (lm_margin={cfg.lm_margin})...")
    val_records = load_split_records(DATA_ROOT, "test", cfg.lm_margin)
    print(f"Val records: {len(val_records)}")

    print(f"\nLoading model from {MODEL_PATH} ...")
    custom_objects = {"SoftArgmax2D": SoftArgmax2D, "WarmupSchedule": WarmupSchedule}
    model = tf.keras.models.load_model(
        str(MODEL_PATH), custom_objects=custom_objects, compile=False
    )
    print(f"Model output shape: {model.output_shape}")

    print("\nRunning per-landmark evaluation (with flip-TTA) on val set...")
    errors = compute_per_landmark_errors(model, val_records, cfg)
    # errors: [N, 46]

    per_lm_mean = errors.mean(axis=0)  # [46]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print_overall_stats(per_lm_mean)
    print_region_summary(per_lm_mean)
    print_hardest_easiest(per_lm_mean, n=10)
    print_all_landmarks(per_lm_mean)

    print("\n" + "=" * 60)
    print("ELD ENSEMBLE GUIDANCE")
    print("=" * 60)
    region_nmes = {
        r: float(np.mean(per_lm_mean[idxs])) for r, idxs in REGIONS.items()
    }
    overall = float(np.mean(per_lm_mean))
    worst_region, worst_nme = max(region_nmes.items(), key=lambda x: x[1])
    best_region, best_nme = min(region_nmes.items(), key=lambda x: x[1])
    spread = worst_nme - best_nme
    print(f"  Worst region: {worst_region} ({worst_nme:.4f})")
    print(f"  Best region:  {best_region} ({best_nme:.4f})")
    print(f"  Spread (worst-best): {spread:.4f}")
    if spread > 3.0:
        print("  -> Large spread: ELD-style ensemble would likely help significantly.")
    elif spread > 1.5:
        print("  -> Moderate spread: ELD ensemble may offer modest improvement.")
    else:
        print("  -> Small spread: ELD ensemble unlikely to help much.")


if __name__ == "__main__":
    main()
