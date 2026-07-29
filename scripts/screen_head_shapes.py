"""Measure size and latency of candidate head shapes before training any of them.

Latency and file size do not depend on the trained weights, so the whole
speed/size half of the Pareto question can be answered in minutes with randomly
initialised models. Only the shapes that actually buy something get a training
run afterwards.

All exports go through the same static-shape concrete-function path as
reexport_static.py, so the numbers are directly comparable to the re-exported
baseline rather than to the shipped dynamic-shape file.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import tensorflow as tf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import train_dog_face_landmarks as T  # noqa: E402
from pareto_harness import bench_tflite, load_val_cache  # noqa: E402
from reexport_static import export_static_fp16  # noqa: E402

OUT = REPO / "artifacts" / "pareto" / "shapes"

# (tag, num_deconv, deconv_channels)  -- None means uniform heatmap_channels=128
CANDIDATES = [
    ("d4_128_baseline",   4, None),
    ("d4_taper_128_96_64_48", 4, (128, 96, 64, 48)),
    ("d4_taper_128_128_64_32", 4, (128, 128, 64, 32)),
    ("d4_taper_128_128_96_64", 4, (128, 128, 96, 64)),
    ("d3_128",            3, None),
    ("d3_taper_128_96_64", 3, (128, 96, 64)),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    crops, _ = load_val_cache()

    base = copy.deepcopy(T.EXPERIMENT_PRESETS["small_v3large_384_long"])
    rows = []
    for tag, nd, chans in CANDIDATES:
        cfg = copy.deepcopy(base)
        cfg.num_deconv_layers = nd
        cfg.deconv_channels = chans
        tf.keras.backend.clear_session()
        model = T.build_model(cfg)
        params = model.count_params()
        path = OUT / f"{tag}.tflite"
        export_static_fp16(model, path, cfg.img_size)
        lat = bench_tflite(path, crops, threads=4, warmup=5, runs=30)
        hm = model.get_layer("heatmap_conv").output_shape
        rows.append((tag, hm[1], params, path.stat().st_size / 1024 / 1024,
                     lat["median_ms"]))
        print(f"  {tag}: {lat['median_ms']:.1f} ms")

    print()
    print(f"{'candidate':26s} {'heatmap':>8s} {'params':>10s} {'size MB':>8s} {'ms':>7s}")
    for tag, hmres, params, mb, ms in rows:
        print(f"{tag:26s} {hmres:>4d}^2   {params:>10,d} {mb:>8.2f} {ms:>7.1f}")


if __name__ == "__main__":
    main()
