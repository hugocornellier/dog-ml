"""Split the landmark model into stages and time each one under TFLite.

The deconv head upsamples 12x12 -> 192x192 at 128 channels, which on paper is
several times the backbone's cost. Before spending a 6-hour training run on a
cheaper head it is worth confirming that with a measurement rather than a FLOP
estimate, so this exports truncated static-shape models and times them in the
same harness the candidates use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import train_dog_face_landmarks as T  # noqa: E402  (registers custom objects)
from pareto_harness import bench_tflite, load_val_cache  # noqa: E402
from reexport_static import op_histogram  # noqa: E402

OUT = REPO / "artifacts" / "pareto" / "profile"


def export_upto(model: tf.keras.Model, layer_name: str, out_path: Path,
                img_size: int = 384) -> None:
    sub = tf.keras.Model(model.input, model.get_layer(layer_name).output)

    @tf.function(input_signature=[
        tf.TensorSpec([1, img_size, img_size, 3], tf.float32, name="crop")
    ])
    def serve(x):
        return sub(x, training=False)

    conv = tf.lite.TFLiteConverter.from_concrete_functions(
        [serve.get_concrete_function()], sub
    )
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    out_path.write_bytes(conv.convert())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(
        REPO / "artifacts" / "small_v3large_384_long" / "best.keras", compile=False
    )
    print([l.name for l in model.layers])

    crops, _ = load_val_cache()

    stages = [
        ("backbone",      "MobilenetV3large"),
        ("deconv1",       "deconv_relu_1"),
        ("deconv2",       "deconv_relu_2"),
        ("deconv3",       "deconv_relu_3"),
        ("deconv4",       "deconv_relu_4"),
        ("heatmap_conv",  "heatmap_conv"),
        ("full",          "landmarks_xy"),
    ]

    prev = None
    for tag, layer in stages:
        path = OUT / f"{tag}.tflite"
        try:
            export_upto(model, layer, path)
        except Exception as exc:  # layer name mismatch across Keras versions
            print(f"  skip {tag} ({layer}): {exc}")
            continue
        lat = bench_tflite(path, crops, threads=4, warmup=5, runs=30)
        cum = lat["median_ms"]
        delta = "" if prev is None else f"  (+{cum - prev:.1f} ms)"
        print(f"{tag:14s} cumulative {cum:7.2f} ms{delta}   "
              f"size {path.stat().st_size/1024/1024:5.2f} MB")
        prev = cum


if __name__ == "__main__":
    main()
