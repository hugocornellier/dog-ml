"""Re-measure candidate head shapes as DYNAMIC exports on the shipping runtime.

The first pass (screen_head_shapes.py) exported every candidate through the
static concrete-function path and timed it on flutter_litert 3.6.0, where static
was the faster of the two graph shapes. On 3.7.0 that reversed: the baseline
dynamic graph runs at 29.0 ms against 59.8 ms static.

That invalidates the earlier head-shape latency table for deciding anything. Those
numbers said the baseline head cost 56.8 ms and a tapered head 21.2 ms, implying a
large win was available. If 3.7.0's speedup comes from handling the deconv head
better, the headroom a cheaper head can recover may be much smaller or gone.

So: same candidate shapes, exported the way the models actually ship (dynamic),
timed on the version the packages actually resolve. Weights are random, which does
not affect latency or file size.
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
from bench_litert_macos import run, LITERT_VERSION  # noqa: E402
from pareto_harness import load_val_cache  # noqa: E402

OUT = REPO / "artifacts" / "pareto" / "shapes_dyn"

CANDIDATES = [
    ("baseline_uniform_128",   4, None),
    ("taper_128_96_64_48",     4, (128, 96, 64, 48)),
    ("taper_128_128_96_64",    4, (128, 128, 96, 64)),
    ("d3_uniform_128",         3, None),
    ("taper_wide1_192_96_64_48", 4, (192, 96, 64, 48)),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    crops, _ = load_val_cache()
    X = np.asarray(crops[:8])

    base = copy.deepcopy(T.EXPERIMENT_PRESETS["small_v3large_384_long"])
    rows = []
    for tag, nd, chans in CANDIDATES:
        cfg = copy.deepcopy(base)
        cfg.num_deconv_layers = nd
        cfg.deconv_channels = chans
        tf.keras.backend.clear_session()
        model = T.build_model(cfg)
        path = OUT / f"{tag}.tflite"
        T.export_tflite(model, path)          # dynamic, the shipping path
        lat, _ = run(path, X, threads=4, warmup=10, runs=60)
        rows.append((tag, model.get_layer("heatmap_conv").output_shape[1],
                     path.stat().st_size / 1024 / 1024, lat["median_ms"]))
        print(f"  {tag}: {lat['median_ms']:.1f} ms")

    print()
    print(f"flutter_litert {LITERT_VERSION}, dynamic export, XNNPACK 4 threads")
    print(f"{'candidate':28s} {'heatmap':>8s} {'size MB':>8s} {'median ms':>10s}")
    for tag, hm, mb, ms in rows:
        print(f"{tag:28s} {hm:>4d}^2   {mb:>8.2f} {ms:>10.1f}")


if __name__ == "__main__":
    main()
