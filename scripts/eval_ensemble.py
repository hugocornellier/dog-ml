#!/usr/bin/env python3
"""Ensemble evaluation: average predictions from multiple trained dog landmark models.

Models:
  1. artifacts/tight_margin/best.keras       -- 224x224, preset tight_margin
  2. artifacts/tight_margin_beta10/best.keras -- 224x224, preset tight_margin_beta10
  3. artifacts/tight_margin_256/best.keras   -- 256x256, preset tight_margin_256

All three share crop_margin=0.10 and lm_margin=0.05, so normalized [0,1] landmarks
from each model are in the same coordinate system regardless of img_size.

Ensemble configs tested:
  A) 3-model average (no TTA)
  B) 3-model average + flip TTA
  C) Best 2-model pairs (all three pairs)
  D) Weighted ensemble (inverse-NME weights, both no-TTA and flip-TTA)

Known individual baselines (for reference):
  tight_margin:       9.53 (no TTA), 9.11 (flip TTA)
  tight_margin_beta10: 9.53 (no TTA), 9.09 (flip TTA)
  tight_margin_256:   9.27 (no TTA), 8.88 (flip TTA)

Usage:
    PYTHONUNBUFFERED=1 python scripts/eval_ensemble.py
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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: Path):
    custom_objects = {"SoftArgmax2D": SoftArgmax2D, "WarmupSchedule": WarmupSchedule}
    model = tf.keras.models.load_model(
        str(model_path), custom_objects=custom_objects, compile=False
    )
    return model


# ---------------------------------------------------------------------------
# NME helpers
# ---------------------------------------------------------------------------

def compute_nme_iod(pred_flat, true_flat):
    pred_2d = tf.reshape(pred_flat, [NUM_LANDMARKS, 2])
    true_2d = tf.reshape(true_flat, [NUM_LANDMARKS, 2])

    left_eye = true_2d[LEFT_OUTER_EYE_IDX]
    right_eye = true_2d[RIGHT_OUTER_EYE_IDX]
    iod = tf.sqrt(tf.reduce_sum(tf.square(left_eye - right_eye)) + 1e-8)

    dists = tf.sqrt(tf.reduce_sum(tf.square(pred_2d - true_2d), axis=-1) + 1e-8)
    nme = tf.reduce_mean(dists) / tf.maximum(iod, 1e-8) * 100.0
    return float(nme)


def unflip_coords(coords_flip):
    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)
    coords_2d = tf.reshape(coords_flip, [NUM_LANDMARKS, 2])
    remapped = tf.gather(coords_2d, flip_idx, axis=0)
    unflipped = tf.stack([1.0 - remapped[:, 0], remapped[:, 1]], axis=-1)
    return tf.reshape(unflipped, [NUM_LANDMARKS * 2])


# ---------------------------------------------------------------------------
# Per-image prediction helpers
# ---------------------------------------------------------------------------

def predict_single(model, image, bbox, img_size, crop_margin):
    """Return normalized coords [92] for a single image, no TTA."""
    lm_flat_dummy = tf.zeros([NUM_LANDMARKS * 2], dtype=tf.float32)
    crop, _ = crop_and_normalize(image, bbox, lm_flat_dummy, img_size, crop_margin)
    coords = model(tf.expand_dims(crop, 0), training=False)[0]
    return coords


def predict_single_flip_tta(model, image, bbox, img_size, crop_margin):
    """Return flip-TTA coords [92]: average of original and unflipped-flipped."""
    lm_flat_dummy = tf.zeros([NUM_LANDMARKS * 2], dtype=tf.float32)
    crop, _ = crop_and_normalize(image, bbox, lm_flat_dummy, img_size, crop_margin)

    coords_orig = model(tf.expand_dims(crop, 0), training=False)[0]

    crop_flip = tf.image.flip_left_right(crop)
    coords_flip = model(tf.expand_dims(crop_flip, 0), training=False)[0]
    coords_flip_unflipped = unflip_coords(coords_flip)

    return (coords_orig + coords_flip_unflipped) / 2.0


# ---------------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------------

def evaluate_ensemble(models_info, val_records, use_flip_tta: bool, weights=None):
    """Evaluate a weighted ensemble over all val records.

    models_info: list of dicts with keys 'model', 'cfg'
    weights: None (equal) or list of floats summing to 1.0
    use_flip_tta: whether each model uses flip-TTA

    Returns mean NME_IOD.
    """
    n = len(models_info)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    predict_fn = predict_single_flip_tta if use_flip_tta else predict_single

    nme_iods = []
    for i, rec in enumerate(val_records):
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        bbox = tf.constant(rec.bbox_xyxy_abs, tf.float32)
        lm_flat = tf.constant(
            [coord for pt in rec.landmarks_abs for coord in pt], dtype=tf.float32
        )

        # Get GT landmarks normalized to each model's crop (should be identical
        # since all presets share the same crop geometry -- only img_size differs,
        # but normalized coords are resolution-independent).
        # We use the first model's cfg to get GT.
        ref_cfg = models_info[0]["cfg"]
        _, lm_norm = crop_and_normalize(
            image, bbox, lm_flat, ref_cfg.img_size, ref_cfg.crop_margin,
        )

        # Ensemble: weighted average of each model's predictions.
        avg_coords = None
        for info, w in zip(models_info, weights):
            coords = predict_fn(
                info["model"], image, bbox,
                info["cfg"].img_size, info["cfg"].crop_margin,
            )
            coords = tf.cast(coords, tf.float32)
            if avg_coords is None:
                avg_coords = w * coords
            else:
                avg_coords = avg_coords + w * coords

        nme_iods.append(compute_nme_iod(avg_coords, lm_norm))

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(val_records)}] running NME_IOD={np.mean(nme_iods):.4f}")

    return float(np.mean(nme_iods))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    configure_ca_bundle()
    set_seed(42)

    print("=" * 70)
    print("DOG LANDMARK ENSEMBLE EVALUATION")
    print("=" * 70)

    # Load val records (use lm_margin from any tight_margin preset -- all same)
    ref_preset = copy.deepcopy(EXPERIMENT_PRESETS["tight_margin"])
    print(f"\nLoading val records from {DATA_ROOT}...")
    val_records = load_split_records(DATA_ROOT, "test", ref_preset.lm_margin)
    print(f"Val records: {len(val_records)}")

    # Load models
    models_info = []
    for mc in MODELS_CONFIG:
        print(f"\nLoading {mc['name']} from {mc['path']}...")
        t0 = time.time()
        model = load_model(mc["path"])
        cfg = copy.deepcopy(EXPERIMENT_PRESETS[mc["preset"]])
        models_info.append({"name": mc["name"], "model": model, "cfg": cfg, **mc})
        print(f"  Loaded in {time.time() - t0:.1f}s  (img_size={cfg.img_size})")

    all_results = []

    # ------------------------------------------------------------------
    # SECTION A: Individual model baselines (verify against known scores)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION A: Individual Model Baselines (sanity check)")
    print("=" * 70)

    for info in models_info:
        mc_ref = next(m for m in MODELS_CONFIG if m["name"] == info["name"])

        print(f"\n  {info['name']} (img_size={info['cfg'].img_size})")

        t0 = time.time()
        nme_no_tta = evaluate_ensemble(
            [info], val_records, use_flip_tta=False
        )
        print(f"    No TTA:   NME_IOD={nme_no_tta:.4f}  "
              f"(expected {mc_ref['individual_nme_no_tta']:.2f})  "
              f"[{time.time()-t0:.0f}s]")

        t0 = time.time()
        nme_flip = evaluate_ensemble(
            [info], val_records, use_flip_tta=True
        )
        print(f"    Flip TTA: NME_IOD={nme_flip:.4f}  "
              f"(expected {mc_ref['individual_nme_flip_tta']:.2f})  "
              f"[{time.time()-t0:.0f}s]")

        all_results.append({
            "config": f"individual_{info['name']}_no_tta",
            "nme": nme_no_tta,
        })
        all_results.append({
            "config": f"individual_{info['name']}_flip_tta",
            "nme": nme_flip,
        })

    # ------------------------------------------------------------------
    # SECTION B: 3-model ensemble (equal weights)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION B: 3-Model Ensemble (equal weights)")
    print("=" * 70)

    print("\n  3-model, no TTA...")
    t0 = time.time()
    nme_3m = evaluate_ensemble(models_info, val_records, use_flip_tta=False)
    print(f"  NME_IOD={nme_3m:.4f}  [{time.time()-t0:.0f}s]")
    all_results.append({"config": "3-model_equal_no_tta", "nme": nme_3m})

    print("\n  3-model + flip TTA...")
    t0 = time.time()
    nme_3m_tta = evaluate_ensemble(models_info, val_records, use_flip_tta=True)
    print(f"  NME_IOD={nme_3m_tta:.4f}  [{time.time()-t0:.0f}s]")
    all_results.append({"config": "3-model_equal_flip_tta", "nme": nme_3m_tta})

    # ------------------------------------------------------------------
    # SECTION C: 2-model pairs (all combinations)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION C: 2-Model Pairs (equal weights)")
    print("=" * 70)

    pairs = [
        (0, 1),  # tight_margin + tight_margin_beta10
        (0, 2),  # tight_margin + tight_margin_256
        (1, 2),  # tight_margin_beta10 + tight_margin_256
    ]

    for i, j in pairs:
        pair_info = [models_info[i], models_info[j]]
        pair_label = f"{models_info[i]['name']} + {models_info[j]['name']}"

        print(f"\n  [{pair_label}]")

        t0 = time.time()
        nme_pair = evaluate_ensemble(pair_info, val_records, use_flip_tta=False)
        print(f"    No TTA:   NME_IOD={nme_pair:.4f}  [{time.time()-t0:.0f}s]")
        all_results.append({"config": f"pair_{models_info[i]['name']}+{models_info[j]['name']}_no_tta", "nme": nme_pair})

        t0 = time.time()
        nme_pair_tta = evaluate_ensemble(pair_info, val_records, use_flip_tta=True)
        print(f"    Flip TTA: NME_IOD={nme_pair_tta:.4f}  [{time.time()-t0:.0f}s]")
        all_results.append({"config": f"pair_{models_info[i]['name']}+{models_info[j]['name']}_flip_tta", "nme": nme_pair_tta})

    # ------------------------------------------------------------------
    # SECTION D: Weighted ensemble (inverse-NME weights)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION D: Weighted Ensemble (inverse-NME weights)")
    print("=" * 70)

    # Weights from known individual NME scores (lower NME -> higher weight)
    individual_nmes_no_tta = [mc["individual_nme_no_tta"] for mc in MODELS_CONFIG]
    individual_nmes_flip = [mc["individual_nme_flip_tta"] for mc in MODELS_CONFIG]

    w_no_tta = [1.0 / nme for nme in individual_nmes_no_tta]
    w_flip = [1.0 / nme for nme in individual_nmes_flip]

    w_sum_no_tta = sum(w_no_tta)
    w_sum_flip = sum(w_flip)

    print(f"\n  Weights (no TTA):   " +
          "  ".join(f"{mi['name']}={w/w_sum_no_tta:.3f}" for mi, w in zip(models_info, w_no_tta)))
    print(f"  Weights (flip TTA): " +
          "  ".join(f"{mi['name']}={w/w_sum_flip:.3f}" for mi, w in zip(models_info, w_flip)))

    print("\n  Weighted 3-model, no TTA...")
    t0 = time.time()
    nme_w = evaluate_ensemble(models_info, val_records, use_flip_tta=False, weights=w_no_tta)
    print(f"  NME_IOD={nme_w:.4f}  [{time.time()-t0:.0f}s]")
    all_results.append({"config": "3-model_weighted_no_tta", "nme": nme_w})

    print("\n  Weighted 3-model + flip TTA...")
    t0 = time.time()
    nme_w_tta = evaluate_ensemble(models_info, val_records, use_flip_tta=True, weights=w_flip)
    print(f"  NME_IOD={nme_w_tta:.4f}  [{time.time()-t0:.0f}s]")
    all_results.append({"config": "3-model_weighted_flip_tta", "nme": nme_w_tta})

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Config':<55} {'NME_IOD':>8}")
    print("-" * 65)

    # Print individual baselines first
    print("\n  -- Individual Baselines --")
    for r in all_results:
        if r["config"].startswith("individual_"):
            label = r["config"].replace("individual_", "").replace("_no_tta", " (no TTA)").replace("_flip_tta", " (flip TTA)")
            print(f"  {label:<53} {r['nme']:>8.4f}")

    print("\n  -- 3-Model Ensemble --")
    for r in all_results:
        if r["config"].startswith("3-model"):
            label = r["config"].replace("_no_tta", " (no TTA)").replace("_flip_tta", " (flip TTA)")
            print(f"  {label:<53} {r['nme']:>8.4f}")

    print("\n  -- 2-Model Pairs --")
    for r in all_results:
        if r["config"].startswith("pair_"):
            label = r["config"].replace("pair_", "").replace("_no_tta", " (no TTA)").replace("_flip_tta", " (flip TTA)")
            print(f"  {label:<53} {r['nme']:>8.4f}")

    best = min(all_results, key=lambda r: r["nme"])
    print("\n" + "=" * 70)
    print(f"BEST ENSEMBLE: {best['config']}")
    print(f"  NME_IOD = {best['nme']:.4f}")
    print(f"  vs best individual (flip TTA): 8.88  (tight_margin_256)")
    print(f"  Improvement: {8.88 - best['nme']:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
