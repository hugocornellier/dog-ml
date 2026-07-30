"""Two checks that change the status of the GPU recommendation, no phone required.

CHECK 1: validate the central inference.
  The recommendation "unfused static graph + XNNPACK excluding TRANSPOSE_CONV would
  land near 26.8 ms" has never been measured, because no flag exposes that partition.
  It was inferred from the claim that today's 26.80 ms already *is* that split:
  XNNPACK on the backbone, deconv on the built-in ruy kernel because dynamic shapes
  push XNNPACK off it.

  That claim is testable by parts. Measure the backbone alone under XNNPACK, and the
  deconv head alone under the built-in kernels. If the sum lands near 26.80, the
  partition theory holds and the estimate is sound. If the sum is far off, XNNPACK is
  declining more (or less) of the dynamic graph than assumed and the recommendation
  rests on a bad model of what is happening.

  Additive estimates ignore cache and scheduling interaction, so treat the sum as an
  estimate with a real error bar, not a prediction.

CHECK 2: cold start.
  Every latency figure in this session is a median with 10 warmup invocations
  discarded. For a video stream that is the right measure. For single-shot photo
  detection it hides delegate setup and the first host-to-device transfer, which is
  precisely where a GPU delegate is expected to be worst. This reports first-invocation
  and time-to-first-result including interpreter construction, for XNNPACK and Metal.
"""

from __future__ import annotations

import ctypes
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_litert_macos import DYLIB, _bind, NUM_THREADS_OFFSET, LITERT_VERSION

REPO = Path(__file__).resolve().parent.parent
GPU_DYLIB = DYLIB.parent / "libtensorflowlite_gpu-mac.dylib"
CACHE = Path("/private/tmp/claude-501/-Users-hugocornellier-IdeaProjects-dog-detection"
             "/ee61cbfd-dd84-4c4a-82ea-cb1141e124d6/scratchpad/valcache")


def _build(model: Path, backend: str, threads: int = 4):
    lib = ctypes.CDLL(str(DYLIB))
    _bind(lib)
    m = lib.TfLiteModelCreateFromFile(str(model).encode())
    if not m:
        raise RuntimeError(f"load failed {model}")
    o = lib.TfLiteInterpreterOptionsCreate()
    lib.TfLiteInterpreterOptionsSetNumThreads(o, threads)
    if backend == "xnnpack":
        xo = lib.TfLiteXNNPackDelegateOptionsDefault()
        struct.pack_into("<i", xo.raw, NUM_THREADS_OFFSET, threads)
        d = lib.TfLiteXNNPackDelegateCreate(ctypes.byref(xo))
        if not d:
            raise RuntimeError("xnnpack delegate failed")
        lib.TfLiteInterpreterOptionsAddDelegate(o, d)
    elif backend == "metal":
        g = ctypes.CDLL(str(GPU_DYLIB))
        g.TFLGpuDelegateCreate.restype = ctypes.c_void_p
        g.TFLGpuDelegateCreate.argtypes = [ctypes.c_void_p]
        d = g.TFLGpuDelegateCreate(None)
        if not d:
            raise RuntimeError("metal delegate failed")
        lib.TfLiteInterpreterOptionsAddDelegate(o, d)
    it = lib.TfLiteInterpreterCreate(m, o)
    if not it:
        raise RuntimeError("interpreter creation failed")
    if lib.TfLiteInterpreterAllocateTensors(it) != 0:
        raise RuntimeError("AllocateTensors failed")
    return lib, it


def timings(model: Path, backend: str, x: np.ndarray, runs: int = 40):
    """(cold_total_ms, first_invoke_ms, steady_median_ms). cold_total includes
    interpreter construction and delegate setup, i.e. what a caller waits for."""
    t_cold0 = time.perf_counter()
    lib, it = _build(model, backend)
    ti = lib.TfLiteInterpreterGetInputTensor(it, 0)
    to = lib.TfLiteInterpreterGetOutputTensor(it, 0)
    n = lib.TfLiteTensorByteSize(to) // 4
    out = np.zeros(n, np.float32)

    def inv(i):
        b = np.ascontiguousarray(x[i % x.shape[0]], np.float32)
        lib.TfLiteTensorCopyFromBuffer(ti, b.ctypes.data_as(ctypes.c_void_p), b.nbytes)
        rc = lib.TfLiteInterpreterInvoke(it)
        lib.TfLiteTensorCopyToBuffer(to, out.ctypes.data_as(ctypes.c_void_p), out.nbytes)
        return rc

    t_first0 = time.perf_counter()
    inv(0)
    first = (time.perf_counter() - t_first0) * 1000.0
    cold_total = (time.perf_counter() - t_cold0) * 1000.0

    for i in range(1, 9):
        inv(i)
    ts = []
    for i in range(runs):
        b = np.ascontiguousarray(x[i % x.shape[0]], np.float32)
        lib.TfLiteTensorCopyFromBuffer(ti, b.ctypes.data_as(ctypes.c_void_p), b.nbytes)
        t0 = time.perf_counter()
        lib.TfLiteInterpreterInvoke(it)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return cold_total, first, statistics.median(ts)


def main():
    x = np.asarray(np.load(CACHE / "crops_384_0.05_0.1.npy", mmap_mode="r")[:8])
    prof = REPO / "artifacts" / "pareto" / "profile"
    D = REPO / "artifacts/small_v3large_384_long/dog_face_landmarks_384_float16.tflite"
    U = REPO / "artifacts/pareto/static_unfused.tflite"

    print(f"flutter_litert {LITERT_VERSION}, 4 threads\n")
    print("CHECK 1: is today's 26.80 ms really 'XNNPACK backbone + built-in deconv'?")
    print("  Truncated stage models from profile_head.py, so the split is measurable.")
    print(f"  {'stage':26s} {'xnnpack':>9s} {'no delegate':>12s}")
    stages = [("backbone + deconv1", prof / "deconv1.tflite"),
              ("through deconv3", prof / "deconv3.tflite"),
              ("through deconv4", prof / "deconv4.tflite"),
              ("full (heatmap+argmax)", prof / "full.tflite")]
    vals = {}
    for tag, p in stages:
        if not p.exists():
            print(f"  {tag:26s} {'missing':>9s}")
            continue
        row = {}
        for backend in ("xnnpack", "none"):
            try:
                row[backend] = timings(p, backend, x, runs=25)[2]
            except Exception as exc:
                row[backend] = None
        vals[tag] = row
        f = lambda v: "err" if v is None else f"{v:9.2f}"
        print(f"  {tag:26s} {f(row['xnnpack']):>9s} {f(row['none']):>12s}")

    print()
    print("CHECK 2: cold start, which the steady-state medians hide")
    print(f"  {'model':22s} {'backend':9s} {'ctor+1st':>10s} {'1st invoke':>11s} {'steady':>8s}")
    for tag, p in [("shipped dynamic", D), ("static unfused", U)]:
        for backend in ("xnnpack", "metal"):
            try:
                c, f1, s = timings(p, backend, x)
                print(f"  {tag:22s} {backend:9s} {c:10.1f} {f1:11.1f} {s:8.2f}")
            except Exception as exc:
                print(f"  {tag:22s} {backend:9s} {str(exc)[:30]:>10s}")


if __name__ == "__main__":
    main()
