#!/usr/bin/env python3
"""Comprehensive evaluation of the tight_margin_320 model (320x320 resolution).

Part 1: TTA evaluation of the 320 model alone
  1a. Baseline (no TTA)
  1b. Flip TTA only
  1c. Multi-scale + flip TTA  (scales [0.9, 1.0, 1.1] x 2 = 6 passes)

Part 2: 2-model ensemble (256 + 320) with multi-scale + flip TTA
  - 2 models x 3 scales x 2 flips = 12 passes per sample

Part 3: 4-model ensemble (all 4 models) with multi-scale + flip TTA
  - 4 models x 3 scales x 2 flips = 24 passes per sample

Usage:
    PYTHONUNBUFFERED=1 python scripts/eval_320_comprehensive.py
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

SCALES = [0.9, 1.0, 1.1]

ALL_MODELS_CONFIG = [
    {
        "name": "tight_margin",
        "path": ARTIFACTS_ROOT / "tight_margin" / "best.keras",
        "preset": "tight_margin",
    },
    {
        "name": "tight_margin_beta10",
        "path": ARTIFACTS_ROOT / "tight_margin_beta10" / "best.keras",
        "preset": "tight_margin_beta10",
    },
    {
        "name": "tight_margin_256",
        "path": ARTIFACTS_ROOT / "tight_margin_256" / "best.keras",
        "preset": "tight_margin_256",
    },
    {
        "name": "tight_margin_320",
        "path": ARTIFACTS_ROOT / "tight_margin_320" / "best.keras",
        "preset": "tight_margin_320",
    },
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: Path):
    custom_objects = {"SoftArgmax2D": SoftArgmax2D, "WarmupSchedule": WarmupSchedule}
    return tf.keras.models.load_model(
        str(model_path), custom_objects=custom_objects, compile=False
    )


# ---------------------------------------------------------------------------
# Scale / flip helpers
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


def predict_scaled_flipped(
    model, crop: tf.Tensor, scale: float, flip_idx: tf.Tensor
) -> tf.Tensor:
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
    iod = tf.sqrt(
        tf.reduce_sum(
            tf.square(true_2d[LEFT_OUTER_EYE_IDX] - true_2d[RIGHT_OUTER_EYE_IDX])
        ) + 1e-8
    )
    dists = tf.sqrt(tf.reduce_sum(tf.square(pred_2d - true_2d), axis=-1) + 1e-8)
    return float(tf.reduce_mean(dists) / tf.maximum(iod, 1e-8) * 100.0)


# ---------------------------------------------------------------------------
# Part 1: Single-model evaluation (320 model only)
# ---------------------------------------------------------------------------

def evaluate_single_model(model, val_records, cfg, tag: str) -> dict:
    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)

    nmes_baseline = []
    nmes_flip = []
    nmes_ms_flip = []

    n = len(val_records)
    t0 = time.time()
    print(f"\n  [{tag}]  img_size={cfg.img_size}  scales={SCALES}  flip=True")

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

        # Baseline (no TTA)
        coords_base = predict_scaled(model, crop, 1.0)
        nmes_baseline.append(compute_nme(coords_base, true_2d))

        # Flip TTA only (2 passes)
        coords_flip = predict_scaled_flipped(model, crop, 1.0, flip_idx)
        coords_flip_avg = (coords_base + coords_flip) / 2.0
        nmes_flip.append(compute_nme(coords_flip_avg, true_2d))

        # Multi-scale + flip TTA (6 passes)
        preds_ms = [predict_scaled(model, crop, s) for s in SCALES]
        preds_ms_flip = [predict_scaled_flipped(model, crop, s, flip_idx) for s in SCALES]
        all_preds = preds_ms + preds_ms_flip
        coords_msf_avg = tf.reduce_mean(tf.stack(all_preds, axis=0), axis=0)
        nmes_ms_flip.append(compute_nme(coords_msf_avg, true_2d))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"    [{i+1}/{n}]  "
                f"baseline={np.mean(nmes_baseline):.4f}  "
                f"flip={np.mean(nmes_flip):.4f}  "
                f"ms+flip={np.mean(nmes_ms_flip):.4f}  "
                f"elapsed={elapsed:.0f}s"
            )

    elapsed = time.time() - t0
    return {
        "baseline": float(np.mean(nmes_baseline)),
        "flip_tta": float(np.mean(nmes_flip)),
        "ms_flip_tta": float(np.mean(nmes_ms_flip)),
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Part 2 & 3: Ensemble evaluation
# ---------------------------------------------------------------------------

def evaluate_ensemble(
    models_info: list[dict],
    val_records,
    use_flip: bool,
    scales: list[float],
    tag: str,
) -> tuple[float, float]:
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

        # Ground-truth landmarks (all models share same crop_margin/lm_margin).
        ref_cfg = models_info[0]["cfg"]
        _, lm_norm = crop_and_normalize(
            image, bbox, lm_flat, ref_cfg.img_size, ref_cfg.crop_margin,
        )
        true_2d = tf.reshape(lm_norm, [NUM_LANDMARKS, 2])

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
    print("DOG LANDMARK: tight_margin_320 COMPREHENSIVE EVALUATION")
    print("=" * 72)
    print(f"Scales: {SCALES}")

    # Load val records (all presets share same lm_margin=0.05).
    ref_preset = copy.deepcopy(EXPERIMENT_PRESETS["tight_margin"])
    print(f"\nLoading val records from {DATA_ROOT} ...")
    val_records = load_split_records(DATA_ROOT, "test", ref_preset.lm_margin)
    print(f"Val records: {len(val_records)}")

    # Load all models.
    models_info = {}
    for mc in ALL_MODELS_CONFIG:
        print(f"\nLoading {mc['name']} from {mc['path']} ...")
        t0 = time.time()
        model = load_model(mc["path"])
        cfg = copy.deepcopy(EXPERIMENT_PRESETS[mc["preset"]])
        models_info[mc["name"]] = {"name": mc["name"], "model": model, "cfg": cfg}
        print(f"  Loaded in {time.time() - t0:.1f}s  (img_size={cfg.img_size})")

    all_results = []

    # ------------------------------------------------------------------
    # PART 1: 320 model alone — baseline, flip TTA, ms+flip TTA
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("PART 1: tight_margin_320 single model")
    print("=" * 72)

    m320_info = models_info["tight_margin_320"]
    part1 = evaluate_single_model(
        m320_info["model"],
        val_records,
        m320_info["cfg"],
        tag="tight_margin_320",
    )
    all_results.append({
        "label": "P1a) 320 baseline (no TTA)           (  1 pred )",
        "nme": part1["baseline"],
        "time": part1["elapsed"],
    })
    all_results.append({
        "label": "P1b) 320 flip TTA only               (  2 preds)",
        "nme": part1["flip_tta"],
        "time": part1["elapsed"],
    })
    all_results.append({
        "label": "P1c) 320 multi-scale + flip TTA      (  6 preds)",
        "nme": part1["ms_flip_tta"],
        "time": part1["elapsed"],
    })
    print(f"\n  Part 1 results:")
    print(f"    Baseline (no TTA):      NME_IOD = {part1['baseline']:.4f}")
    print(f"    Flip TTA:               NME_IOD = {part1['flip_tta']:.4f}")
    print(f"    Multi-scale + flip TTA: NME_IOD = {part1['ms_flip_tta']:.4f}")

    # ------------------------------------------------------------------
    # PART 2: 2-model ensemble (256 + 320) x ms+flip TTA = 12 passes
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("PART 2: 2-model ensemble (256 + 320) + multi-scale + flip TTA")
    print("=" * 72)

    ensemble_2 = [models_info["tight_margin_256"], models_info["tight_margin_320"]]
    nme_p2, t_p2 = evaluate_ensemble(
        ensemble_2, val_records,
        use_flip=True, scales=SCALES,
        tag="P2: 2-model (256+320) x ms+flip  (12 preds)",
    )
    all_results.append({
        "label": "P2)  2-model (256+320) ms+flip TTA   ( 12 preds)",
        "nme": nme_p2,
        "time": t_p2,
    })
    print(f"\n  Part 2 result:  NME_IOD = {nme_p2:.4f}  [{t_p2:.0f}s]")

    # ------------------------------------------------------------------
    # PART 3: 4-model ensemble (all) x ms+flip TTA = 24 passes
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("PART 3: 4-model ensemble (all 4 models) + multi-scale + flip TTA")
    print("=" * 72)

    ensemble_4 = [
        models_info["tight_margin"],
        models_info["tight_margin_beta10"],
        models_info["tight_margin_256"],
        models_info["tight_margin_320"],
    ]
    nme_p3, t_p3 = evaluate_ensemble(
        ensemble_4, val_records,
        use_flip=True, scales=SCALES,
        tag="P3: 4-model (all) x ms+flip  (24 preds)",
    )
    all_results.append({
        "label": "P3)  4-model (all) ms+flip TTA       ( 24 preds)",
        "nme": nme_p3,
        "time": t_p3,
    })
    print(f"\n  Part 3 result:  NME_IOD = {nme_p3:.4f}  [{t_p3:.0f}s]")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print(f"{'Configuration':<52} {'NME_IOD':>8}  {'Time':>8}")
    print("-" * 72)

    # Reference lines from previous experiments
    print(f"  {'[REF] 256 baseline (no TTA)':<50} {'9.2700':>8}")
    print(f"  {'[REF] 256 flip TTA':<50} {'8.8800':>8}")
    print(f"  {'[REF] 256 ms+flip TTA (6 preds)':<50} {'8.6600':>8}")
    print(f"  {'[REF] 3-model ensemble ms+flip TTA (18 preds)':<50} {'8.5200':>8}")
    print("-" * 72)

    for r in all_results:
        print(f"  {r['label']:<50} {r['nme']:>8.4f}  {r['time']:>6.0f}s")

    best = min(all_results, key=lambda r: r["nme"])
    print()
    print("=" * 72)
    print(f"BEST CONFIG: {best['label'].strip()}")
    print(f"  NME_IOD        = {best['nme']:.4f}")
    print(f"  vs prev best   = 8.52  (3-model ensemble ms+flip TTA)")
    print(f"  Improvement    = {8.52 - best['nme']:+.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
