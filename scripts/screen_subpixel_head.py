"""Measure a sub-pixel (conv + depth-to-space) head against the deconv head.

A transpose-conv with kernel 4 / stride 2 is algebraically identical to a regular
conv with kernel 3 producing 4x the channels (verified by solve_subpixel_mapping.py;
kernel 2 is NOT equivalent, because the two sub-positions draw on 2x2 windows offset by
one, so the kernel must be 3x3 to cover both, with 20 of its 36 taps structurally zero), followed by depth-to-space with block
size 2. Each output pixel of the stride-2 deconv draws on a 2x2 input neighbourhood,
and the four sub-positions interleave exactly as depth-to-space does. MAC count is NOT identical: the deconv does 16 taps per output group, the k=3
sub-pixel form does 36, so 2.25x the arithmetic with 56% of it multiplying by zero.

Why it might be the best available option:

  * `TRANSPOSE_CONV` disappears from the graph, so the v4-versus-v3 GPU gate is moot
    and there is nothing to unfuse.
  * XNNPACK's slow transpose-conv kernel is likewise moot; `CONV_2D` is XNNPACK's most
    optimised path.
  * A fused ReLU on `CONV_2D` is fine for both delegates, unlike on `TRANSPOSE_CONV`.
  * Converting trained weights is an exact rearrangement, so no retraining and no
    accuracy change.

This script only answers the latency-and-size half, which needs no trained weights.
If the numbers justify it, the weight rearrangement is the next piece of work.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import train_dog_face_landmarks as T  # noqa: E402
from bench_litert_macos import run, LITERT_VERSION, dylib_for  # noqa: E402
from reexport_static import export_static_fp16, op_histogram  # noqa: E402

OUT = REPO / "artifacts" / "pareto" / ("subpixel_%s" % __import__("os").environ.get("NLM","46"))
NUM_LANDMARKS = int(__import__("os").environ.get("NLM","46"))


def build_subpixel(cfg, widths):
    """Same backbone and head widths, but each 2x upsample is conv + depth-to-space."""
    inputs = tf.keras.Input(shape=(cfg.img_size, cfg.img_size, 3), name="crop")
    x = tf.keras.layers.Rescaling(scale=255.0, offset=0.0, name="to_0_255")(inputs)
    backbone = tf.keras.applications.MobileNetV3Large(
        input_shape=(cfg.img_size, cfg.img_size, 3), include_top=False,
        minimalistic=False, weights=None,
    )
    backbone.trainable = False
    x = backbone(x, training=False)

    for i, c in enumerate(widths):
        # 4*c channels at the pre-upsample resolution, then interleave to 2x.
        x = tf.keras.layers.Conv2D(
            4 * c, kernel_size=int(__import__("os").environ.get("KK","3")), padding="same", use_bias=False,
            name=f"subpix_conv_{i+1}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"subpix_bn_{i+1}")(x)
        x = tf.keras.layers.ReLU(name=f"subpix_relu_{i+1}")(x)
        x = tf.keras.layers.Lambda(
            lambda t: tf.nn.depth_to_space(t, 2), name=f"subpix_d2s_{i+1}",
        )(x)

    heatmaps = tf.keras.layers.Conv2D(
        NUM_LANDMARKS, kernel_size=1, padding="same", name="heatmap_conv")(x)
    coords = T.SoftArgmax2D(beta=1.0, name="soft_argmax")(heatmaps)
    coords = tf.keras.layers.Identity(name="landmarks_xy")(coords)
    return tf.keras.Model(inputs, coords, name="dog_face_landmark_regressor")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = Path("/private/tmp/claude-501/-Users-hugocornellier-IdeaProjects-dog-detection"
                 "/ee61cbfd-dd84-4c4a-82ea-cb1141e124d6/scratchpad/valcache")
    X = np.asarray(np.load(cache / "crops_384_0.05_0.1.npy", mmap_mode="r")[:8])

    cfg = copy.deepcopy(T.EXPERIMENT_PRESETS["small_v3large_384_long"])
    tf.keras.backend.clear_session()
    model = build_subpixel(cfg, [128, 128, 128, 128])
    print("heatmap resolution:", model.get_layer("heatmap_conv").output_shape)

    static = OUT / "subpixel_static.tflite"
    export_static_fp16(model, static, cfg.img_size)
    dyn = OUT / "subpixel_dynamic.tflite"
    T.export_tflite(model, dyn)

    for tag, p in [("static", static), ("dynamic", dyn)]:
        h = op_histogram(p)
        print(f"{tag}: {sum(h.values())} ops | TRANSPOSE_CONV {h.get('TRANSPOSE_CONV',0)}"
              f" | DEPTH_TO_SPACE {h.get('DEPTH_TO_SPACE',0)}"
              f" | SHAPE {h.get('SHAPE',0)} | size {p.stat().st_size/1024/1024:.2f} MB")

    print()
    print(f"flutter_litert {LITERT_VERSION}, 4 threads, median ms")
    print(f"{'graph':10s} {'backend':10s} {'ms':>9s}")
    for tag, p in [("static", static), ("dynamic", dyn)]:
        for backend in ("xnnpack", "none", "metal"):
            try:
                if backend == "metal":
                    from bench_metal import bench_metal
                    r = bench_metal(p, X, warmup=8, runs=30)["median_ms"]
                else:
                    r = run(p, X, threads=4, warmup=8, runs=40,
                            use_xnnpack=(backend == "xnnpack"))[0]["median_ms"]
                print(f"{tag:10s} {backend:10s} {r:9.2f}")
            except Exception as exc:
                print(f"{tag:10s} {backend:10s} {str(exc)[:40]:>9s}")


if __name__ == "__main__":
    main()
