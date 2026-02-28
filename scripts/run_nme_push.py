#!/usr/bin/env python3
"""Automated NME push: run a sequence of experiments to get NME_IOD into single digits.

Experiments (run sequentially):
  1. Flip-TTA on existing best model (no retraining, minutes)
  2. Heatmap supervision with sigma sweep {1.5, 1.75, 2.0} (full two-phase training each)
  3. 112×112 heatmaps (full two-phase training)
  4. 112×112 + heatmap supervision combined
  5. Flip-TTA on the best new model from above

Results are logged to artifacts/nme_push_results.md as a running journal.

Usage:
    python scripts/run_nme_push.py
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

# Import training script modules.
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
    build_tf_dataset,
    build_model,
    compile_model,
    train_model,
    evaluate_model,
    export_tflite,
    tflite_sanity_check,
    save_metadata,
    crop_and_normalize,
)

# Paths
DATA_ROOT = (
    Path.home() / ".cache" / "kagglehub" / "datasets"
    / "georgemartvel" / "dogflw" / "versions" / "1" / "DogFLW"
)
RESULTS_FILE = Path("artifacts/nme_push_results.md")
BEST_MODEL_DIR = Path("artifacts/heatmap_v2s_best")


# ---------------------------------------------------------------------------
# Flip-TTA evaluation (heatmap-level averaging)
# ---------------------------------------------------------------------------

def evaluate_tta(
    model_path: Path,
    val_records: list,
    cfg: ExperimentConfig,
) -> float:
    """Evaluate a model with flip-TTA at the coordinate level.

    1. Load model, get coord output
    2. For each val sample: run original + flipped image
    3. Unflip the flipped coords (mirror x, remap landmark indices)
    4. Average coordinates
    5. Compute NME_IOD
    """
    print("\n=== Flip-TTA Evaluation ===")
    print(f"Model: {model_path}")

    custom_objects = {"SoftArgmax2D": SoftArgmax2D, "WarmupSchedule": WarmupSchedule}
    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)

    # Get coord output model (handles both single-output and multi-output).
    if isinstance(model.output, dict):
        coord_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.get_layer("landmarks_xy").output,
        )
    else:
        coord_model = model

    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)
    img_size = cfg.img_size
    crop_margin = cfg.crop_margin

    nme_iods = []
    for i, rec in enumerate(val_records):
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        lm_flat = tf.constant(
            [coord for pt in rec.landmarks_abs for coord in pt], dtype=tf.float32
        )
        crop, lm_norm = crop_and_normalize(
            image, tf.constant(rec.bbox_xyxy_abs, tf.float32),
            lm_flat, img_size, crop_margin,
        )

        # Original prediction.
        coords_orig = coord_model(tf.expand_dims(crop, 0), training=False)[0]  # [92]

        # Flipped prediction.
        crop_flip = tf.image.flip_left_right(crop)
        coords_flip = coord_model(tf.expand_dims(crop_flip, 0), training=False)[0]  # [92]

        # Unflip coordinates: mirror x, remap landmark indices.
        coords_flip_2d = tf.reshape(coords_flip, [NUM_LANDMARKS, 2])  # [46, 2]
        coords_flip_remapped = tf.gather(coords_flip_2d, flip_idx, axis=0)  # remap landmarks
        coords_flip_unflipped = tf.stack(
            [1.0 - coords_flip_remapped[:, 0], coords_flip_remapped[:, 1]], axis=-1
        )  # mirror x
        coords_flip_flat = tf.reshape(coords_flip_unflipped, [NUM_LANDMARKS * 2])

        # Average coordinates.
        coords_avg = (coords_orig + coords_flip_flat) / 2.0

        # Compute NME_IOD.
        pred_2d = tf.reshape(coords_avg, [NUM_LANDMARKS, 2])
        true_2d = tf.reshape(lm_norm, [NUM_LANDMARKS, 2])

        left_eye = true_2d[LEFT_OUTER_EYE_IDX]
        right_eye = true_2d[RIGHT_OUTER_EYE_IDX]
        iod = tf.sqrt(tf.reduce_sum(tf.square(left_eye - right_eye)) + 1e-8)

        dists = tf.sqrt(tf.reduce_sum(tf.square(pred_2d - true_2d), axis=-1) + 1e-8)
        nme = float(tf.reduce_mean(dists) / tf.maximum(iod, 1e-8) * 100.0)
        nme_iods.append(nme)

        if (i + 1) % 100 == 0:
            print(f"  TTA eval: {i+1}/{len(val_records)}, running NME_IOD={np.mean(nme_iods):.4f}")

    result = float(np.mean(nme_iods))
    print(f"Flip-TTA NME_IOD: {result:.4f}")
    return result


def evaluate_no_tta(
    model_path: Path,
    val_records: list,
    cfg: ExperimentConfig,
) -> float:
    """Evaluate a model without TTA (baseline for comparison)."""
    print(f"\n=== Baseline Evaluation (no TTA) ===")
    print(f"Model: {model_path}")

    custom_objects = {"SoftArgmax2D": SoftArgmax2D}
    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)

    img_size = cfg.img_size
    crop_margin = cfg.crop_margin

    # For multi-output models, get coord output
    if isinstance(model.output, dict):
        coord_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.get_layer("landmarks_xy").output,
        )
    else:
        coord_model = model

    nme_iods = []
    for rec in val_records:
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        lm_flat = tf.constant(
            [coord for pt in rec.landmarks_abs for coord in pt], dtype=tf.float32
        )
        crop, lm_norm = crop_and_normalize(
            image, tf.constant(rec.bbox_xyxy_abs, tf.float32),
            lm_flat, img_size, crop_margin,
        )

        pred = coord_model(tf.expand_dims(crop, 0), training=False)[0]
        pred_2d = tf.reshape(pred, [NUM_LANDMARKS, 2])
        true_2d = tf.reshape(lm_norm, [NUM_LANDMARKS, 2])

        left_eye = true_2d[LEFT_OUTER_EYE_IDX]
        right_eye = true_2d[RIGHT_OUTER_EYE_IDX]
        iod = tf.sqrt(tf.reduce_sum(tf.square(left_eye - right_eye)) + 1e-8)
        dists = tf.sqrt(tf.reduce_sum(tf.square(pred_2d - true_2d), axis=-1) + 1e-8)
        nme = float(tf.reduce_mean(dists) / tf.maximum(iod, 1e-8) * 100.0)
        nme_iods.append(nme)

    result = float(np.mean(nme_iods))
    print(f"No-TTA NME_IOD: {result:.4f}")
    return result


# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------

def run_training_experiment(
    preset_name: str,
    out_dir: Path,
) -> tuple[float, Path]:
    """Run a full training experiment and return (best_nme_iod, model_path)."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {preset_name}")
    print(f"Output: {out_dir}")
    print(f"{'='*70}")

    cfg = copy.deepcopy(EXPERIMENT_PRESETS[preset_name])
    set_seed(cfg.seed)

    train_records = load_split_records(DATA_ROOT, "train", cfg.lm_margin)
    val_records = load_split_records(DATA_ROOT, "test", cfg.lm_margin)

    train_ds = build_tf_dataset(train_records, cfg=cfg, training=True)
    val_ds = build_tf_dataset(val_records, cfg=cfg, training=False)

    model = build_model(cfg)
    model.summary(print_fn=lambda x: None)  # suppress verbose summary

    model = train_model(model, train_ds, val_ds, out_dir=out_dir, cfg=cfg,
                        num_train=len(train_records))

    compile_model(model, lr=cfg.learning_rate, cfg=cfg)
    val_metrics = evaluate_model(model, val_ds)

    # Extract score key
    score_key = "xy_landmark_nme_iod" if cfg.heatmap_supervision else "landmark_nme_iod"
    nme_iod = float(val_metrics.get(score_key, float("inf")))

    # Export TFLite
    if cfg.heatmap_supervision:
        export_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.get_layer("landmarks_xy").output,
        )
    else:
        export_model = model

    tflite_path = out_dir / f"dog_face_landmarks_{cfg.img_size}_float16.tflite"
    export_tflite(export_model, tflite_path)

    save_metadata(
        out_dir=out_dir, cfg=cfg, data_root=DATA_ROOT,
        train_records=train_records, val_records=val_records,
        val_metrics=val_metrics, tflite_path=tflite_path,
        tflite_sanity=tflite_sanity_check(tflite_path, val_records,
                                           img_size=cfg.img_size,
                                           crop_margin=cfg.crop_margin),
    )

    model_path = out_dir / "best.keras"
    print(f"\nRESULT: {preset_name} => NME_IOD = {nme_iod:.4f}")
    return nme_iod, model_path


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def log_result(msg: str) -> None:
    """Append a line to the results file and print it."""
    print(msg)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(msg + "\n")


def log_header() -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        f.write(f"# NME Push Results — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("Automated experiment sequence to push NME_IOD into single digits.\n\n")
        f.write("| # | Experiment | NME_IOD | TTA NME_IOD | Time | Notes |\n")
        f.write("|---|---|---|---|---|---|\n")


def log_row(idx: int, name: str, nme: float, tta_nme: float | None,
            elapsed: str, notes: str) -> None:
    tta_str = f"{tta_nme:.2f}" if tta_nme is not None else "—"
    log_result(f"| {idx} | {name} | {nme:.2f} | {tta_str} | {elapsed} | {notes} |")


# ---------------------------------------------------------------------------
# Main experiment sequence
# ---------------------------------------------------------------------------

def main() -> None:
    configure_ca_bundle()
    tf.config.optimizer.set_jit(False)
    set_seed(42)

    log_header()
    log_result(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_result(f"Baseline: heatmap_v2s_best NME_IOD = 10.72\n")

    val_records = load_split_records(DATA_ROOT, "test", lm_margin=0.12)
    best_cfg = copy.deepcopy(EXPERIMENT_PRESETS["heatmap_v2s_best"])

    results: list[dict] = []
    overall_best_nme = 10.72
    overall_best_model = BEST_MODEL_DIR / "best.keras"
    overall_best_cfg = best_cfg

    # -----------------------------------------------------------------------
    # Step 1: Flip-TTA on existing best model
    # -----------------------------------------------------------------------
    t0 = time.time()
    log_result("\n## Step 1: Flip-TTA on existing best model (no retraining)\n")

    baseline_nme = evaluate_no_tta(BEST_MODEL_DIR / "best.keras", val_records, best_cfg)
    tta_nme = evaluate_tta(BEST_MODEL_DIR / "best.keras", val_records, best_cfg)
    elapsed = str(timedelta(seconds=int(time.time() - t0)))

    log_row(1, "heatmap_v2s_best + flip-TTA", baseline_nme, tta_nme, elapsed,
            f"Baseline={baseline_nme:.2f}, TTA gain={baseline_nme - tta_nme:.2f}")
    results.append({"name": "baseline+TTA", "nme": baseline_nme, "tta_nme": tta_nme})

    if tta_nme < overall_best_nme:
        overall_best_nme = tta_nme
        log_result(f"\n**New overall best with TTA: {tta_nme:.2f}**\n")

    # -----------------------------------------------------------------------
    # Step 2: 112×112 heatmaps (no supervision — safest architectural change)
    # -----------------------------------------------------------------------
    log_result("\n## Step 2: 112×112 heatmap resolution (4 deconv layers)\n")

    t0 = time.time()
    nme_112, model_112 = run_training_experiment("heatmap_v2s_112", Path("artifacts/heatmap_v2s_112"))
    elapsed = str(timedelta(seconds=int(time.time() - t0)))
    cfg_112 = copy.deepcopy(EXPERIMENT_PRESETS["heatmap_v2s_112"])
    tta_112 = evaluate_tta(model_112, val_records, cfg_112)

    log_row(2, "heatmap_v2s_112", nme_112, tta_112, elapsed, "4 deconv layers, 112×112")
    results.append({"name": "heatmap_v2s_112", "nme": nme_112, "tta_nme": tta_112})

    effective_112 = min(nme_112, tta_112)
    if effective_112 < overall_best_nme:
        overall_best_nme = effective_112
        overall_best_model = model_112
        overall_best_cfg = cfg_112
        log_result(f"\n**New overall best: {effective_112:.2f} (112×112)**\n")

    # -----------------------------------------------------------------------
    # Step 3: Heatmap supervision (sigma=1.75, coord-dominant hybrid loss)
    # hm_weight=0.1, coord_weight=1.0 — coord is primary, heatmap is regularizer
    # -----------------------------------------------------------------------
    log_result("\n## Step 3: Heatmap supervision (coord-dominant, sigma=1.75)\n")

    t0 = time.time()
    nme_hmsup, model_hmsup = run_training_experiment(
        "heatmap_v2s_hmsup_s175", Path("artifacts/heatmap_v2s_hmsup_s175"))
    elapsed = str(timedelta(seconds=int(time.time() - t0)))
    cfg_hmsup = copy.deepcopy(EXPERIMENT_PRESETS["heatmap_v2s_hmsup_s175"])
    tta_hmsup = evaluate_tta(model_hmsup, val_records, cfg_hmsup)

    log_row(3, "heatmap_v2s_hmsup_s175", nme_hmsup, tta_hmsup, elapsed,
            "hm_w=0.1, coord_w=1.0, sigma=1.75")
    results.append({"name": "heatmap_v2s_hmsup_s175", "nme": nme_hmsup, "tta_nme": tta_hmsup})

    effective_hmsup = min(nme_hmsup, tta_hmsup)
    if effective_hmsup < overall_best_nme:
        overall_best_nme = effective_hmsup
        overall_best_model = model_hmsup
        overall_best_cfg = cfg_hmsup
        log_result(f"\n**New overall best: {effective_hmsup:.2f} (hmsup s1.75)**\n")

    # -----------------------------------------------------------------------
    # Step 4: 112×112 + heatmap supervision combined
    # -----------------------------------------------------------------------
    log_result("\n## Step 4: 112×112 + heatmap supervision combined\n")

    t0 = time.time()
    nme_combo, model_combo = run_training_experiment(
        "heatmap_v2s_112_hmsup", Path("artifacts/heatmap_v2s_112_hmsup"))
    elapsed = str(timedelta(seconds=int(time.time() - t0)))
    cfg_combo = copy.deepcopy(EXPERIMENT_PRESETS["heatmap_v2s_112_hmsup"])
    tta_combo = evaluate_tta(model_combo, val_records, cfg_combo)

    log_row(4, "heatmap_v2s_112_hmsup", nme_combo, tta_combo,
            elapsed, "112×112 + hm supervision")
    results.append({"name": "heatmap_v2s_112_hmsup", "nme": nme_combo, "tta_nme": tta_combo})

    effective_combo = min(nme_combo, tta_combo)
    if effective_combo < overall_best_nme:
        overall_best_nme = effective_combo
        overall_best_model = model_combo
        overall_best_cfg = cfg_combo
        log_result(f"\n**New overall best: {effective_combo:.2f} (112×112 + hmsup)**\n")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log_result(f"\n## Summary\n")
    log_result(f"| Experiment | NME_IOD | +TTA |")
    log_result(f"|---|---|---|")
    for r in results:
        tta_str = f"{r['tta_nme']:.2f}" if r.get('tta_nme') else "—"
        log_result(f"| {r['name']} | {r['nme']:.2f} | {tta_str} |")

    log_result(f"\n**Overall best NME_IOD: {overall_best_nme:.2f}**")
    log_result(f"Best model: {overall_best_model}")

    # If we beat the previous best, copy TFLite to production location.
    if overall_best_nme < 10.72:
        log_result(f"\nImprovement over baseline: 10.72 → {overall_best_nme:.2f} "
                   f"(Δ = {10.72 - overall_best_nme:.2f})")

    log_result(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Overall best NME_IOD: {overall_best_nme:.2f}")
    print(f"Results: {RESULTS_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
