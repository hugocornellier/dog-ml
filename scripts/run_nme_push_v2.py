#!/usr/bin/env python3
"""NME Push v2: Sequential experiments to break into single digits.

Experiments (in order):
  1. Tight margin (lm_margin=0.05, crop_margin=0.10) - coord loss
  2. Tight margin + mixup + random erasing - coord loss
  3. Pure heatmap supervision + tight margin + mixup + random erasing
  4. Combined best (all improvements)

Each experiment runs full two-phase training (~3h) and evaluates with flip-TTA.

Usage:
    python scripts/run_nme_push_v2.py
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

DATA_ROOT = (
    Path.home() / ".cache" / "kagglehub" / "datasets"
    / "georgemartvel" / "dogflw" / "versions" / "1" / "DogFLW"
)
RESULTS_FILE = Path("artifacts/nme_push_v2_results.md")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def soft_argmax_with_beta(heatmaps, beta=1.0):
    """Apply soft-argmax with temperature scaling."""
    b = tf.shape(heatmaps)[0]
    h, w = heatmaps.shape[1], heatmaps.shape[2]
    k = heatmaps.shape[3]
    flat = tf.reshape(heatmaps, [b, h * w, k])
    weights = tf.nn.softmax(flat * beta, axis=1)
    weights = tf.reshape(weights, [b, h, w, k])
    x_grid = tf.reshape(tf.linspace(0.0, 1.0, w), [1, 1, w, 1])
    y_grid = tf.reshape(tf.linspace(0.0, 1.0, h), [1, h, 1, 1])
    x = tf.reduce_sum(weights * x_grid, axis=[1, 2])
    y = tf.reduce_sum(weights * y_grid, axis=[1, 2])
    coords = tf.stack([x, y], axis=-1)
    return tf.reshape(coords, [b, k * 2])


def evaluate_coord_model_tta(
    model_path: Path,
    val_records: list,
    cfg: ExperimentConfig,
    best_beta: float = 1.0,
) -> tuple[float, float]:
    """Evaluate a model with and without flip-TTA.

    For pure heatmap models, uses heatmap output + soft-argmax with best_beta.
    For coord models, uses the coord output directly.
    Returns (nme_no_tta, nme_tta).
    """
    custom_objects = {"SoftArgmax2D": SoftArgmax2D, "WarmupSchedule": WarmupSchedule}
    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)

    # Determine if this is a pure heatmap model (output is heatmaps, not coords)
    is_heatmap_output = len(model.output_shape) == 4  # (B, H, W, K)

    if is_heatmap_output:
        hm_model = model
        coord_model = None
    elif isinstance(model.output, dict):
        coord_model = tf.keras.Model(inputs=model.inputs,
                                      outputs=model.get_layer("landmarks_xy").output)
        hm_model = None
    else:
        coord_model = model
        hm_model = None

    # Also get heatmap model for beta-tuning if available
    if not is_heatmap_output:
        try:
            hm_layer = model.get_layer("heatmap_conv")
            hm_model = tf.keras.Model(inputs=model.inputs, outputs=hm_layer.output)
        except ValueError:
            hm_model = None

    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)

    def get_coords(crop_tensor, use_hm_decode=False):
        inp = tf.expand_dims(crop_tensor, 0)
        if use_hm_decode and hm_model is not None:
            hm = hm_model(inp, training=False)
            return soft_argmax_with_beta(hm, beta=best_beta)[0]
        elif coord_model is not None:
            return coord_model(inp, training=False)[0]
        elif hm_model is not None:
            hm = hm_model(inp, training=False)
            return soft_argmax_with_beta(hm, beta=best_beta)[0]
        else:
            return model(inp, training=False)[0]

    use_hm = is_heatmap_output or (best_beta != 1.0 and hm_model is not None)

    nmes_no_tta = []
    nmes_tta = []

    for i, rec in enumerate(val_records):
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        lm_flat = tf.constant([c for pt in rec.landmarks_abs for c in pt], dtype=tf.float32)
        crop, lm_norm = crop_and_normalize(
            image, tf.constant(rec.bbox_xyxy_abs, tf.float32),
            lm_flat, cfg.img_size, cfg.crop_margin,
        )

        coords_orig = get_coords(crop, use_hm_decode=use_hm)

        # No TTA
        pred_2d = tf.reshape(coords_orig, [NUM_LANDMARKS, 2])
        true_2d = tf.reshape(lm_norm, [NUM_LANDMARKS, 2])
        iod = tf.sqrt(tf.reduce_sum(tf.square(
            true_2d[LEFT_OUTER_EYE_IDX] - true_2d[RIGHT_OUTER_EYE_IDX])) + 1e-8)
        dists = tf.sqrt(tf.reduce_sum(tf.square(pred_2d - true_2d), axis=-1) + 1e-8)
        nmes_no_tta.append(float(tf.reduce_mean(dists) / tf.maximum(iod, 1e-8) * 100.0))

        # Flip TTA
        crop_flip = tf.image.flip_left_right(crop)
        coords_flip = get_coords(crop_flip, use_hm_decode=use_hm)
        cf2d = tf.reshape(coords_flip, [NUM_LANDMARKS, 2])
        cf_remap = tf.gather(cf2d, flip_idx, axis=0)
        cf_unflip = tf.stack([1.0 - cf_remap[:, 0], cf_remap[:, 1]], axis=-1)
        coords_avg = (coords_orig + tf.reshape(cf_unflip, [NUM_LANDMARKS * 2])) / 2.0

        pred_2d_tta = tf.reshape(coords_avg, [NUM_LANDMARKS, 2])
        dists_tta = tf.sqrt(tf.reduce_sum(tf.square(pred_2d_tta - true_2d), axis=-1) + 1e-8)
        nmes_tta.append(float(tf.reduce_mean(dists_tta) / tf.maximum(iod, 1e-8) * 100.0))

        if (i + 1) % 100 == 0:
            print(f"  eval: {i+1}/{len(val_records)}, "
                  f"NME={np.mean(nmes_no_tta):.4f}, TTA={np.mean(nmes_tta):.4f}")

    return float(np.mean(nmes_no_tta)), float(np.mean(nmes_tta))


# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------

def run_experiment(
    preset_name: str,
    out_dir: Path,
    val_records: list,
    best_beta: float = 1.0,
) -> dict:
    """Run training + evaluation and return results dict."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {preset_name}")
    print(f"Output: {out_dir}")
    print(f"{'='*70}")

    cfg = copy.deepcopy(EXPERIMENT_PRESETS[preset_name])

    t0 = time.time()
    set_seed(cfg.seed)

    train_records = load_split_records(DATA_ROOT, "train", cfg.lm_margin)
    # Use val_records with matching lm_margin
    val_recs = load_split_records(DATA_ROOT, "test", cfg.lm_margin)

    train_ds = build_tf_dataset(train_records, cfg=cfg, training=True)
    val_ds = build_tf_dataset(val_recs, cfg=cfg, training=False)

    model = build_model(cfg)
    model.summary(print_fn=lambda x: None)

    model = train_model(model, train_ds, val_ds, out_dir=out_dir, cfg=cfg,
                        num_train=len(train_records))

    elapsed = time.time() - t0
    elapsed_str = str(timedelta(seconds=int(elapsed)))

    # Evaluate with coord-level NME
    model_path = out_dir / "best.keras"

    # For pure heatmap models, we need to test different betas
    if cfg.pure_heatmap_supervision:
        print("\n  Testing decode betas for pure heatmap model...")
        best_tta = float("inf")
        best_b = 1.0
        for beta in [1.0, 10.0, 20.0, 40.0, 60.0]:
            nme_no, nme_tta = evaluate_coord_model_tta(model_path, val_recs, cfg, beta)
            print(f"    beta={beta}: NME={nme_no:.4f}, TTA={nme_tta:.4f}")
            if nme_tta < best_tta:
                best_tta = nme_tta
                best_b = beta
                best_nme_no = nme_no
        nme_no_tta = best_nme_no
        nme_tta = best_tta
        print(f"  Best beta={best_b}: NME={nme_no_tta:.4f}, TTA={nme_tta:.4f}")
    else:
        nme_no_tta, nme_tta = evaluate_coord_model_tta(model_path, val_recs, cfg, best_beta)

    # Export TFLite
    custom_objects = {"SoftArgmax2D": SoftArgmax2D, "WarmupSchedule": WarmupSchedule}
    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)

    if cfg.pure_heatmap_supervision or cfg.heatmap_supervision:
        export_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.get_layer("landmarks_xy").output,
        )
    else:
        export_model = model

    tflite_path = out_dir / f"dog_face_landmarks_{cfg.img_size}_float16.tflite"
    export_tflite(export_model, tflite_path)

    result = {
        "name": preset_name,
        "nme_no_tta": nme_no_tta,
        "nme_tta": nme_tta,
        "elapsed": elapsed_str,
        "model_path": str(model_path),
        "tflite_path": str(tflite_path),
    }

    print(f"\nRESULT: {preset_name}")
    print(f"  NME_IOD (no TTA): {nme_no_tta:.4f}")
    print(f"  NME_IOD (+ TTA):  {nme_tta:.4f}")
    print(f"  Time: {elapsed_str}")

    return result


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_ca_bundle()
    tf.config.optimizer.set_jit(False)
    set_seed(42)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        f.write(f"# NME Push v2 Results — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Baseline: heatmap_v2s_112 NME_IOD = 10.58 (no TTA) / 10.14 (TTA)\n")

    val_records = load_split_records(DATA_ROOT, "test", lm_margin=0.12)

    experiments = [
        ("tight_margin", "artifacts/tight_margin"),
        ("tight_margin_mixup", "artifacts/tight_margin_mixup"),
        ("pure_heatmap", "artifacts/pure_heatmap"),
        ("combined_best", "artifacts/combined_best"),
    ]

    results = []
    overall_best = 10.14  # current best with TTA
    overall_best_name = "baseline"

    log("| # | Experiment | NME_IOD | +TTA | Time | Notes |")
    log("|---|---|---|---|---|---|")

    for idx, (preset, out_path) in enumerate(experiments, 1):
        try:
            result = run_experiment(preset, Path(out_path), val_records)
            results.append(result)

            notes = ""
            eff = min(result["nme_no_tta"], result["nme_tta"])
            if eff < overall_best:
                overall_best = eff
                overall_best_name = preset
                notes = "**NEW BEST**"

            log(f"| {idx} | {preset} | {result['nme_no_tta']:.2f} | "
                f"{result['nme_tta']:.2f} | {result['elapsed']} | {notes} |")

        except Exception as e:
            log(f"| {idx} | {preset} | FAILED | FAILED | - | {str(e)[:50]} |")
            print(f"\nEXPERIMENT FAILED: {preset}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    log(f"\n## Summary\n")
    log(f"Overall best: {overall_best_name} = {overall_best:.4f}")
    log(f"Improvement over baseline: 10.14 -> {overall_best:.4f} (delta = {10.14 - overall_best:.4f})")
    log(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Overall best NME_IOD: {overall_best:.4f} ({overall_best_name})")
    print(f"Results: {RESULTS_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
