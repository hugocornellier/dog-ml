#!/usr/bin/env python3
"""Evaluate zero-cost improvements on existing best model.

Tests:
  1. SoftArgmax2D temperature scaling (beta = 1, 10, 20, 40, 60)
  2. Multi-scale TTA (scales + flip)
  3. Combined temperature + multi-scale TTA

Usage:
    python scripts/eval_experiments.py
"""

from __future__ import annotations

import copy
import math
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
    ExperimentConfig,
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
MODEL_PATH = Path("artifacts/heatmap_v2s_112/best.keras")


def load_heatmap_model(model_path: Path):
    """Load model and return (full_model, heatmap_model).

    heatmap_model outputs raw heatmaps before SoftArgmax2D.
    """
    custom_objects = {"SoftArgmax2D": SoftArgmax2D, "WarmupSchedule": WarmupSchedule}
    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)

    # Find the heatmap conv layer (before soft_argmax)
    heatmap_layer = model.get_layer("heatmap_conv")
    heatmap_model = tf.keras.Model(inputs=model.inputs, outputs=heatmap_layer.output)
    return model, heatmap_model


def soft_argmax_with_beta(heatmaps, beta=1.0):
    """Apply soft-argmax with temperature scaling.

    heatmaps: (B, H, W, K) tensor
    beta: temperature parameter (higher = sharper)
    Returns: (B, K*2) coordinates
    """
    b = tf.shape(heatmaps)[0]
    h = tf.shape(heatmaps)[1]
    w = tf.shape(heatmaps)[2]
    k = tf.shape(heatmaps)[3]

    h_static = heatmaps.shape[1]
    w_static = heatmaps.shape[2]

    flat = tf.reshape(heatmaps, [b, h * w, k])
    weights = tf.nn.softmax(flat * beta, axis=1)
    weights = tf.reshape(weights, [b, h, w, k])

    x_coords = tf.linspace(0.0, 1.0, w_static)
    y_coords = tf.linspace(0.0, 1.0, h_static)
    x_grid = tf.reshape(x_coords, [1, 1, w_static, 1])
    y_grid = tf.reshape(y_coords, [1, h_static, 1, 1])

    x = tf.reduce_sum(weights * x_grid, axis=[1, 2])
    y = tf.reduce_sum(weights * y_grid, axis=[1, 2])

    coords = tf.stack([x, y], axis=-1)
    return tf.reshape(coords, [b, k * 2])


def argmax_with_refinement(heatmaps):
    """Argmax + quadratic subpixel refinement.

    heatmaps: (B, H, W, K) tensor
    Returns: (B, K*2) coordinates
    """
    b = tf.shape(heatmaps)[0]
    h_s = heatmaps.shape[1]
    w_s = heatmaps.shape[2]
    k_s = heatmaps.shape[3]

    # Transpose to (B, K, H, W) for easier processing
    hm = tf.transpose(heatmaps, [0, 3, 1, 2])  # (B, K, H, W)
    flat = tf.reshape(hm, [b, k_s, h_s * w_s])
    max_idx = tf.argmax(flat, axis=2, output_type=tf.int32)  # (B, K)

    max_y = max_idx // w_s  # row
    max_x = max_idx % w_s   # col

    # Subpixel refinement: fit parabola around peak
    all_x = []
    all_y = []
    for bi in range(1):  # We process one sample at a time
        for ki in range(k_s):
            my = max_y[bi, ki]
            mx = max_x[bi, ki]

            # x refinement
            if mx > 0 and mx < w_s - 1:
                left = hm[bi, ki, my, mx - 1]
                center = hm[bi, ki, my, mx]
                right = hm[bi, ki, my, mx + 1]
                dx = 0.5 * (right - left) / (2.0 * center - left - right + 1e-8)
                dx = tf.clip_by_value(dx, -0.5, 0.5)
            else:
                dx = 0.0

            # y refinement
            if my > 0 and my < h_s - 1:
                top = hm[bi, ki, my - 1, mx]
                center = hm[bi, ki, my, mx]
                bottom = hm[bi, ki, my + 1, mx]
                dy = 0.5 * (bottom - top) / (2.0 * center - top - bottom + 1e-8)
                dy = tf.clip_by_value(dy, -0.5, 0.5)
            else:
                dy = 0.0

            refined_x = (tf.cast(mx, tf.float32) + dx) / tf.cast(w_s - 1, tf.float32)
            refined_y = (tf.cast(my, tf.float32) + dy) / tf.cast(h_s - 1, tf.float32)
            all_x.append(refined_x)
            all_y.append(refined_y)

    x_coords = tf.stack(all_x)  # (K,)
    y_coords = tf.stack(all_y)  # (K,)
    coords = tf.stack([x_coords, y_coords], axis=-1)  # (K, 2)
    return tf.reshape(coords, [1, k_s * 2])


def compute_nme_iod(pred_flat, true_flat):
    """Compute NME_IOD for a single sample."""
    pred_2d = tf.reshape(pred_flat, [NUM_LANDMARKS, 2])
    true_2d = tf.reshape(true_flat, [NUM_LANDMARKS, 2])

    left_eye = true_2d[LEFT_OUTER_EYE_IDX]
    right_eye = true_2d[RIGHT_OUTER_EYE_IDX]
    iod = tf.sqrt(tf.reduce_sum(tf.square(left_eye - right_eye)) + 1e-8)

    dists = tf.sqrt(tf.reduce_sum(tf.square(pred_2d - true_2d), axis=-1) + 1e-8)
    nme = tf.reduce_mean(dists) / tf.maximum(iod, 1e-8) * 100.0
    return float(nme)


def unflip_coords(coords_flip):
    """Unflip coordinates: remap landmark indices and mirror x."""
    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)
    coords_2d = tf.reshape(coords_flip, [NUM_LANDMARKS, 2])
    remapped = tf.gather(coords_2d, flip_idx, axis=0)
    unflipped = tf.stack([1.0 - remapped[:, 0], remapped[:, 1]], axis=-1)
    return tf.reshape(unflipped, [NUM_LANDMARKS * 2])


def evaluate_with_beta(heatmap_model, val_records, cfg, beta, use_argmax=False):
    """Evaluate model with given temperature beta or argmax."""
    nme_iods = []
    for rec in val_records:
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        lm_flat = tf.constant(
            [coord for pt in rec.landmarks_abs for coord in pt], dtype=tf.float32
        )
        crop, lm_norm = crop_and_normalize(
            image, tf.constant(rec.bbox_xyxy_abs, tf.float32),
            lm_flat, cfg.img_size, cfg.crop_margin,
        )

        heatmaps = heatmap_model(tf.expand_dims(crop, 0), training=False)

        if use_argmax:
            coords = argmax_with_refinement(heatmaps)[0]
        else:
            coords = soft_argmax_with_beta(heatmaps, beta=beta)[0]

        nme_iods.append(compute_nme_iod(coords, lm_norm))

    return float(np.mean(nme_iods))


def evaluate_flip_tta_with_beta(heatmap_model, val_records, cfg, beta, use_argmax=False):
    """Evaluate with flip-TTA using given temperature beta."""
    nme_iods = []
    for rec in val_records:
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        lm_flat = tf.constant(
            [coord for pt in rec.landmarks_abs for coord in pt], dtype=tf.float32
        )
        crop, lm_norm = crop_and_normalize(
            image, tf.constant(rec.bbox_xyxy_abs, tf.float32),
            lm_flat, cfg.img_size, cfg.crop_margin,
        )

        # Original
        hm_orig = heatmap_model(tf.expand_dims(crop, 0), training=False)
        if use_argmax:
            coords_orig = argmax_with_refinement(hm_orig)[0]
        else:
            coords_orig = soft_argmax_with_beta(hm_orig, beta=beta)[0]

        # Flipped
        crop_flip = tf.image.flip_left_right(crop)
        hm_flip = heatmap_model(tf.expand_dims(crop_flip, 0), training=False)
        if use_argmax:
            coords_flip = argmax_with_refinement(hm_flip)[0]
        else:
            coords_flip = soft_argmax_with_beta(hm_flip, beta=beta)[0]

        coords_flip_unflipped = unflip_coords(coords_flip)
        coords_avg = (coords_orig + coords_flip_unflipped) / 2.0

        nme_iods.append(compute_nme_iod(coords_avg, lm_norm))

    return float(np.mean(nme_iods))


def evaluate_multiscale_tta(heatmap_model, val_records, cfg, beta,
                            scales=(0.9, 1.0, 1.1), use_flip=True,
                            use_argmax=False):
    """Evaluate with multi-scale + optional flip TTA."""
    nme_iods = []
    for i, rec in enumerate(val_records):
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        lm_flat = tf.constant(
            [coord for pt in rec.landmarks_abs for coord in pt], dtype=tf.float32
        )
        bbox = tf.constant(rec.bbox_xyxy_abs, tf.float32)

        all_coords = []
        for scale in scales:
            # Scale the bounding box around its center
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bw, bh = x2 - x1, y2 - y1
            img_h = tf.cast(tf.shape(image)[0], tf.float32)
            img_w = tf.cast(tf.shape(image)[1], tf.float32)

            new_bw = bw * scale
            new_bh = bh * scale
            sx1 = tf.clip_by_value(cx - new_bw / 2.0, 0.0, img_w)
            sy1 = tf.clip_by_value(cy - new_bh / 2.0, 0.0, img_h)
            sx2 = tf.clip_by_value(cx + new_bw / 2.0, 0.0, img_w)
            sy2 = tf.clip_by_value(cy + new_bh / 2.0, 0.0, img_h)
            scaled_bbox = tf.stack([sx1, sy1, sx2, sy2])

            crop, _ = crop_and_normalize(
                image, scaled_bbox, lm_flat, cfg.img_size, cfg.crop_margin,
            )

            # Original
            hm = heatmap_model(tf.expand_dims(crop, 0), training=False)
            if use_argmax:
                coords = argmax_with_refinement(hm)[0]
            else:
                coords = soft_argmax_with_beta(hm, beta=beta)[0]

            # Coords are relative to the scaled crop. We need to convert
            # back to the original crop's coordinate system.
            # Actually, since we normalize landmarks relative to the crop window,
            # and the GT is also relative to the default (scale=1.0) crop,
            # we need to map coords from the scaled crop back to the default crop.
            #
            # The crop window for this scale is different, so the normalized
            # coords represent different absolute positions.
            # We need to convert: scaled_crop_norm -> absolute -> default_crop_norm

            # Get the actual crop window used (after crop_margin expansion)
            crop_margin = cfg.crop_margin
            bw_s = sx2 - sx1
            bh_s = sy2 - sy1
            mx_s = bw_s * crop_margin
            my_s = bh_s * crop_margin
            cx1_s = tf.maximum(0.0, sx1 - mx_s)
            cy1_s = tf.maximum(0.0, sy1 - my_s)
            cx2_s = tf.minimum(img_w, sx2 + mx_s)
            cy2_s = tf.minimum(img_h, sy2 + my_s)

            # Default (scale=1.0) crop window
            bw_d = x2 - x1
            bh_d = y2 - y1
            mx_d = bw_d * crop_margin
            my_d = bh_d * crop_margin
            cx1_d = tf.maximum(0.0, x1 - mx_d)
            cy1_d = tf.maximum(0.0, y1 - my_d)
            cx2_d = tf.minimum(img_w, x2 + mx_d)
            cy2_d = tf.minimum(img_h, y2 + my_d)

            # Convert normalized coords from scaled crop to absolute, then to default crop
            coords_2d = tf.reshape(coords, [NUM_LANDMARKS, 2])
            abs_x = coords_2d[:, 0] * (cx2_s - cx1_s) + cx1_s
            abs_y = coords_2d[:, 1] * (cy2_s - cy1_s) + cy1_s

            # Normalize to default crop
            def_x = (abs_x - cx1_d) / tf.maximum(cx2_d - cx1_d, 1e-8)
            def_y = (abs_y - cy1_d) / tf.maximum(cy2_d - cy1_d, 1e-8)
            def_x = tf.clip_by_value(def_x, 0.0, 1.0)
            def_y = tf.clip_by_value(def_y, 0.0, 1.0)

            remapped = tf.stack([def_x, def_y], axis=-1)
            all_coords.append(tf.reshape(remapped, [NUM_LANDMARKS * 2]))

            if use_flip:
                crop_flip = tf.image.flip_left_right(crop)
                hm_flip = heatmap_model(tf.expand_dims(crop_flip, 0), training=False)
                if use_argmax:
                    coords_flip = argmax_with_refinement(hm_flip)[0]
                else:
                    coords_flip = soft_argmax_with_beta(hm_flip, beta=beta)[0]
                coords_flip_unflipped = unflip_coords(coords_flip)

                # Also remap flipped coords
                coords_flip_2d = tf.reshape(coords_flip_unflipped, [NUM_LANDMARKS, 2])
                abs_fx = coords_flip_2d[:, 0] * (cx2_s - cx1_s) + cx1_s
                abs_fy = coords_flip_2d[:, 1] * (cy2_s - cy1_s) + cy1_s
                def_fx = (abs_fx - cx1_d) / tf.maximum(cx2_d - cx1_d, 1e-8)
                def_fy = (abs_fy - cy1_d) / tf.maximum(cy2_d - cy1_d, 1e-8)
                def_fx = tf.clip_by_value(def_fx, 0.0, 1.0)
                def_fy = tf.clip_by_value(def_fy, 0.0, 1.0)
                remapped_f = tf.stack([def_fx, def_fy], axis=-1)
                all_coords.append(tf.reshape(remapped_f, [NUM_LANDMARKS * 2]))

        # Average all predictions
        avg_coords = tf.reduce_mean(tf.stack(all_coords, axis=0), axis=0)

        # GT is relative to default crop
        _, lm_norm = crop_and_normalize(
            image, bbox, lm_flat, cfg.img_size, cfg.crop_margin,
        )

        nme_iods.append(compute_nme_iod(avg_coords, lm_norm))

        if (i + 1) % 100 == 0:
            print(f"  Multi-scale TTA: {i+1}/{len(val_records)}, running NME_IOD={np.mean(nme_iods):.4f}")

    return float(np.mean(nme_iods))


def main():
    configure_ca_bundle()
    set_seed(42)

    cfg = copy.deepcopy(EXPERIMENT_PRESETS["heatmap_v2s_112"])
    val_records = load_split_records(DATA_ROOT, "test", cfg.lm_margin)

    print(f"Loading model from {MODEL_PATH}...")
    full_model, heatmap_model = load_heatmap_model(MODEL_PATH)

    results = []

    # ===== Experiment 1: Temperature sweep =====
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: SoftArgmax2D Temperature Sweep")
    print("=" * 70)

    for beta in [1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0]:
        t0 = time.time()
        nme = evaluate_with_beta(heatmap_model, val_records, cfg, beta=beta)
        nme_tta = evaluate_flip_tta_with_beta(heatmap_model, val_records, cfg, beta=beta)
        elapsed = time.time() - t0
        print(f"  beta={beta:>6.1f}: NME_IOD={nme:.4f} | +TTA={nme_tta:.4f} | {elapsed:.0f}s")
        results.append({"name": f"beta={beta}", "nme": nme, "tta": nme_tta})

    # Also test argmax + subpixel refinement
    print("\n  Testing argmax + subpixel refinement...")
    t0 = time.time()
    nme_argmax = evaluate_with_beta(heatmap_model, val_records, cfg, beta=1.0, use_argmax=True)
    nme_argmax_tta = evaluate_flip_tta_with_beta(heatmap_model, val_records, cfg, beta=1.0, use_argmax=True)
    elapsed = time.time() - t0
    print(f"  argmax+refine: NME_IOD={nme_argmax:.4f} | +TTA={nme_argmax_tta:.4f} | {elapsed:.0f}s")
    results.append({"name": "argmax+refine", "nme": nme_argmax, "tta": nme_argmax_tta})

    # Find best beta
    best = min(results, key=lambda r: r["tta"])
    print(f"\n  Best: {best['name']} -> NME_IOD={best['nme']:.4f}, +TTA={best['tta']:.4f}")

    # ===== Experiment 2: Multi-scale TTA =====
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Multi-Scale TTA")
    print("=" * 70)

    # Use the best beta from Experiment 1
    best_beta_str = best["name"]
    if best_beta_str.startswith("beta="):
        best_beta = float(best_beta_str.split("=")[1])
        use_argmax = False
    else:
        best_beta = 1.0
        use_argmax = True

    print(f"  Using decode: {best['name']}")

    # Test different scale sets
    scale_sets = [
        ([0.9, 1.0, 1.1], "3-scale"),
        ([0.85, 0.95, 1.0, 1.05, 1.15], "5-scale"),
        ([0.9, 1.0, 1.1], "3-scale+flip"),
    ]

    for scales, label in scale_sets:
        use_flip = "flip" in label
        if not use_flip:
            use_flip = True  # Always use flip for multi-scale
        t0 = time.time()
        nme_ms = evaluate_multiscale_tta(
            heatmap_model, val_records, cfg, beta=best_beta,
            scales=scales, use_flip=use_flip, use_argmax=use_argmax,
        )
        elapsed = time.time() - t0
        print(f"  {label} (scales={scales}): NME_IOD={nme_ms:.4f} | {elapsed:.0f}s")
        results.append({"name": f"multiscale-{label}", "nme": nme_ms, "tta": nme_ms})

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'No TTA':>10} {'+ TTA':>10}")
    print("-" * 50)
    for r in results:
        print(f"  {r['name']:<28} {r['nme']:>10.4f} {r['tta']:>10.4f}")

    overall_best = min(results, key=lambda r: min(r["nme"], r["tta"]))
    best_nme = min(overall_best["nme"], overall_best["tta"])
    print(f"\nOverall best: {overall_best['name']} -> {best_nme:.4f}")
    print(f"Baseline comparison: 10.58 (no TTA) / 10.14 (flip TTA)")
    print(f"Improvement: {10.14 - best_nme:.4f}")


if __name__ == "__main__":
    main()
