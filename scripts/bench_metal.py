"""Time the Metal GPU delegate, for graphs where it actually engages.

Only meaningful on the static-shape export: the dynamic-shape landmark graph fails
`TfLiteInterpreterCreate` under the Metal delegate and then blocks in
`mutex.cc RAW: Lock blocking`, so it cannot be timed at all.

IMPORTANT CAVEAT: this shares the GPU with anything else using Metal. If a
tensorflow-metal training job is running, treat the result as an upper bound on
latency (a lower bound on achievable speed), not as a clean measurement.

Engagement must be confirmed separately with the dev != 0.0 test in
test_delegate_engagement.py. A no-op delegate would time fast and mean nothing.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_litert_macos import DYLIB, LITERT_VERSION, _bind  # noqa: E402

GPU_DYLIB = DYLIB.parent / "libtensorflowlite_gpu-mac.dylib"


def bench_metal(model_path: Path, x: np.ndarray, warmup=5, runs=30):
    lib = ctypes.CDLL(str(DYLIB))
    _bind(lib)
    g = ctypes.CDLL(str(GPU_DYLIB))
    g.TFLGpuDelegateCreate.restype = ctypes.c_void_p
    g.TFLGpuDelegateCreate.argtypes = [ctypes.c_void_p]

    model = lib.TfLiteModelCreateFromFile(str(model_path).encode())
    opts = lib.TfLiteInterpreterOptionsCreate()
    lib.TfLiteInterpreterOptionsSetNumThreads(opts, 4)
    delegate = g.TFLGpuDelegateCreate(None)
    if not delegate:
        raise RuntimeError("Metal delegate creation failed")
    lib.TfLiteInterpreterOptionsAddDelegate(opts, delegate)

    interp = lib.TfLiteInterpreterCreate(model, opts)
    if not interp:
        raise RuntimeError("interpreter creation failed (expected on dynamic graphs)")
    if lib.TfLiteInterpreterAllocateTensors(interp) != 0:
        raise RuntimeError("AllocateTensors failed")

    in_t = lib.TfLiteInterpreterGetInputTensor(interp, 0)
    out_t = lib.TfLiteInterpreterGetOutputTensor(interp, 0)
    n_out = lib.TfLiteTensorByteSize(out_t) // 4
    out = np.zeros(n_out, np.float32)

    def one(i):
        buf = np.ascontiguousarray(x[i % x.shape[0]], dtype=np.float32)
        lib.TfLiteTensorCopyFromBuffer(in_t, buf.ctypes.data_as(ctypes.c_void_p),
                                       buf.nbytes)
        rc = lib.TfLiteInterpreterInvoke(interp)
        lib.TfLiteTensorCopyToBuffer(out_t, out.ctypes.data_as(ctypes.c_void_p),
                                     out.nbytes)
        return rc

    for i in range(warmup):
        one(i)
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        one(i)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median_ms": statistics.median(times),
        "p10_ms": float(np.percentile(times, 10)),
        "p90_ms": float(np.percentile(times, 90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", type=Path)
    args = ap.parse_args()
    from pareto_harness import load_val_cache
    crops, _ = load_val_cache()
    x = np.asarray(crops[:8])
    print(f"flutter_litert {LITERT_VERSION}, Metal GPU delegate")
    for m in args.models:
        try:
            r = bench_metal(m, x)
            print(f"{m.name:44s} median {r['median_ms']:7.2f} ms "
                  f"(p10 {r['p10_ms']:.2f}, p90 {r['p90_ms']:.2f})")
        except Exception as exc:
            print(f"{m.name:44s} FAILED: {exc}")


if __name__ == "__main__":
    main()
