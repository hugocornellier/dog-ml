#!/usr/bin/env python3
"""Re-export a trained landmark model as int8 for mobile deployment.

The float16 export produced by `train_dog_face_landmarks.py --export` is ~54.6 MiB
for the 384px model, which is heavy to bundle in an app. This re-exports the same
weights with dynamic-range int8 quantization: int8 weights, float32 activations.

Why dynamic range rather than full integer quantization:
  - Weights dominate the file (28.7M params), so int8 weights capture nearly all
    the available saving; quantizing activations too buys almost nothing.
  - Full integer conversion measured LARGER (30.3 MiB vs 29.0 MiB) because the
    SoftArgmax2D tail falls back to float and the converter inserts
    quantize/dequantize nodes around it.
  - SoftArgmax2D (spatial softmax over HxW then an expectation against a
    coordinate grid) is numerically delicate. Keeping activations in float
    leaves that math untouched.

Input/output signature is unchanged: float32 [1, S, S, 3] -> float32 [1, 92],
so this is a drop-in replacement for the float16 file on device.

Usage:
  python scripts/quantize_landmarks_int8.py \
      --keras artifacts/tight_margin_384/best.keras \
      --out artifacts/tight_margin_384/dog_face_landmarks_384_int8.tflite
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

# Registers SoftArgmax2D / WarmupSchedule for deserialization.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dog_face_landmarks import (  # noqa: E402,F401
    SoftArgmax2D, WarmupSchedule, NUM_LANDMARKS,
    load_split_records, crop_and_normalize,
)


def quantize(keras_path: Path, out_path: Path) -> int:
    model = tf.keras.models.load_model(str(keras_path), compile=False)
    params = model.count_params()
    print(f"Loaded {keras_path} ({params:,} params)")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # DEFAULT with no supported_types override == dynamic range int8 weights.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_bytes)
    n = len(tflite_bytes)
    print(f"Saved TFLite: {out_path} ({n/1024/1024:.2f} MiB)")
    return n


def compare_nme(
    tflite_paths: list[Path],
    data_root: Path,
    img_size: int,
    lm_margin: float,
    crop_margin: float,
    limit: int,
) -> None:
    """Evaluate each model on the test split and print NME_IOD side by side."""
    records = load_split_records(data_root, "test", lm_margin)
    if limit:
        records = records[:limit]

    for path in tflite_paths:
        interp = tf.lite.Interpreter(model_path=str(path), num_threads=os.cpu_count())
        interp.allocate_tensors()
        in_det, out_det = interp.get_input_details()[0], interp.get_output_details()[0]

        nmes = []
        for rec in records:
            image = tf.image.convert_image_dtype(
                tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3), tf.float32
            )
            lm_flat = tf.constant(
                [c for pt in rec.landmarks_abs for c in pt], dtype=tf.float32
            )
            crop, lm_norm = crop_and_normalize(
                image, tf.constant(rec.bbox_xyxy_abs, tf.float32),
                lm_flat, img_size, crop_margin,
            )
            interp.set_tensor(
                in_det["index"],
                tf.expand_dims(crop, 0).numpy().astype(in_det["dtype"]),
            )
            interp.invoke()
            pred = np.clip(interp.get_tensor(out_det["index"])[0], 0.0, 1.0)

            true_2d = lm_norm.numpy().reshape(NUM_LANDMARKS, 2)
            pred_2d = pred.reshape(NUM_LANDMARKS, 2)
            iod = float(np.linalg.norm(true_2d[18] - true_2d[19]))
            if iod < 1e-6:
                continue
            dist = np.linalg.norm(pred_2d - true_2d, axis=-1)
            nmes.append(float(dist.mean() / iod * 100.0))

        size_mib = path.stat().st_size / 1024 / 1024
        print(f"{path.name:48s} {size_mib:6.1f} MiB   "
              f"NME_IOD {np.mean(nmes):6.3f}   median {np.median(nmes):6.3f}   n={len(nmes)}")


def main() -> None:
    default_root = (
        Path.home() / ".cache" / "kagglehub" / "datasets"
        / "georgemartvel" / "dogflw" / "versions" / "1" / "DogFLW"
    )
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keras", type=Path,
                   default=Path("artifacts/tight_margin_384/best.keras"))
    p.add_argument("--out", type=Path,
                   default=Path("artifacts/tight_margin_384/dog_face_landmarks_384_int8.tflite"))
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--lm-margin", type=float, default=0.05,
                   help="Must match the value the model was trained with.")
    p.add_argument("--crop-margin", type=float, default=0.10,
                   help="Must match the value the model was trained with.")
    p.add_argument("--data-root", type=Path, default=default_root)
    p.add_argument("--compare-against", type=Path, default=None,
                   help="Existing float16 TFLite to benchmark the int8 file against.")
    p.add_argument("--eval-limit", type=int, default=0,
                   help="Limit test images during comparison (0 = all 480).")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    quantize(args.keras, args.out)

    if args.skip_eval:
        return

    paths = [args.out]
    if args.compare_against and args.compare_against.exists():
        paths.insert(0, args.compare_against)

    print("\nAccuracy comparison (crop_margin matches training):")
    compare_nme(paths, args.data_root, args.img_size,
                args.lm_margin, args.crop_margin, args.eval_limit)


if __name__ == "__main__":
    main()
