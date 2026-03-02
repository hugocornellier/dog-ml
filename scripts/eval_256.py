#!/usr/bin/env python3
"""Quick flip-TTA evaluation for the tight_margin_256 model.

Usage:
    python scripts/eval_256.py
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


def evaluate_flip_tta(model, val_records, cfg):
    """Evaluate model with and without coordinate-level flip-TTA.

    Returns (nme_no_tta, nme_tta).
    """
    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)

    nmes_no_tta = []
    nmes_tta = []

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

        # NME without TTA
        pred_2d = tf.reshape(coords_orig, [NUM_LANDMARKS, 2])
        true_2d = tf.reshape(lm_norm, [NUM_LANDMARKS, 2])
        iod = tf.sqrt(
            tf.reduce_sum(tf.square(
                true_2d[LEFT_OUTER_EYE_IDX] - true_2d[RIGHT_OUTER_EYE_IDX]
            )) + 1e-8
        )
        dists = tf.sqrt(tf.reduce_sum(tf.square(pred_2d - true_2d), axis=-1) + 1e-8)
        nmes_no_tta.append(float(tf.reduce_mean(dists) / tf.maximum(iod, 1e-8) * 100.0))

        # Flipped prediction
        crop_flip = tf.image.flip_left_right(crop)
        coords_flip = model(tf.expand_dims(crop_flip, 0), training=False)[0]

        # Unflip: remap landmark indices, then mirror x
        cf2d = tf.reshape(coords_flip, [NUM_LANDMARKS, 2])
        cf_remap = tf.gather(cf2d, flip_idx, axis=0)
        cf_unflip = tf.stack([1.0 - cf_remap[:, 0], cf_remap[:, 1]], axis=-1)

        # Average original + unflipped
        coords_avg = (coords_orig + tf.reshape(cf_unflip, [NUM_LANDMARKS * 2])) / 2.0

        pred_2d_tta = tf.reshape(coords_avg, [NUM_LANDMARKS, 2])
        dists_tta = tf.sqrt(
            tf.reduce_sum(tf.square(pred_2d_tta - true_2d), axis=-1) + 1e-8
        )
        nmes_tta.append(
            float(tf.reduce_mean(dists_tta) / tf.maximum(iod, 1e-8) * 100.0)
        )

        if (i + 1) % 100 == 0:
            print(
                f"  [{i+1}/{len(val_records)}]  "
                f"NME={np.mean(nmes_no_tta):.4f}  "
                f"TTA={np.mean(nmes_tta):.4f}"
            )

    return float(np.mean(nmes_no_tta)), float(np.mean(nmes_tta))


def main():
    configure_ca_bundle()
    set_seed(42)

    cfg = copy.deepcopy(EXPERIMENT_PRESETS["tight_margin_256"])
    print(f"Preset:      tight_margin_256")
    print(f"img_size:    {cfg.img_size}")
    print(f"crop_margin: {cfg.crop_margin}")
    print(f"lm_margin:   {cfg.lm_margin}")

    print(f"\nLoading val set (lm_margin={cfg.lm_margin})...")
    val_records = load_split_records(DATA_ROOT, "test", cfg.lm_margin)
    print(f"Val records: {len(val_records)}")

    print(f"\nLoading model from {MODEL_PATH} ...")
    custom_objects = {"SoftArgmax2D": SoftArgmax2D, "WarmupSchedule": WarmupSchedule}
    model = tf.keras.models.load_model(
        str(MODEL_PATH), custom_objects=custom_objects, compile=False
    )
    print(f"Model output shape: {model.output_shape}")

    print("\nRunning flip-TTA evaluation on 480 val samples...")
    nme_no_tta, nme_tta = evaluate_flip_tta(model, val_records, cfg)

    print()
    print("=" * 50)
    print("RESULTS — tight_margin_256")
    print("=" * 50)
    print(f"  NME_IOD (no TTA): {nme_no_tta:.4f}")
    print(f"  NME_IOD (+ TTA):  {nme_tta:.4f}")
    print(f"  TTA improvement:  {nme_no_tta - nme_tta:+.4f}")
    print("=" * 50)
    print(f"\nBaseline (tight_margin 224):  9.53 (no TTA) / 9.11 (flip TTA)")
    delta_no_tta = 9.53 - nme_no_tta
    delta_tta = 9.11 - nme_tta
    print(f"Delta vs baseline (no TTA):  {delta_no_tta:+.4f}")
    print(f"Delta vs baseline (+TTA):    {delta_tta:+.4f}")


if __name__ == "__main__":
    main()
