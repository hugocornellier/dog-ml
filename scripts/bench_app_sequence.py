"""Benchmark using the exact call sequence animal_detection uses, resize included.

`bench_litert_macos.py` creates the interpreter and invokes. The real package does
something subtly different in `landmark_model_runner.dart`:

    final interpreter = Interpreter.fromBuffer(bytes, options: options);  // delegate already in options
    interpreter.resizeInputTensor(0, [1, inputSize, inputSize, 3]);
    interpreter.allocateTensors();

The delegate is attached at construction and the resize happens *afterwards*. In
TFLite, ResizeInputTensor invalidates the prepared plan and the following
AllocateTensors re-prepares it, which can change whether a delegate ends up
covering the graph. A latency number taken without the resize is therefore not
automatically the number the app sees.

This matters specifically for the static-shape fix: pinning the batch to 1 removes
the dynamic dimension from the input signature, and it is worth confirming that a
resize to the identical shape is still accepted and still leaves the graph fully
delegated, rather than assuming it.

Note the face localizer does NOT resize (face_localizer_model.dart has no such
call), so only landmark models need this path.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_litert_macos import (  # noqa: E402
    DYLIB, LITERT_VERSION, NUM_THREADS_OFFSET, _bind,
)


def run_app_sequence(model_path: Path, inputs: np.ndarray, img_size: int,
                     threads: int = 4, warmup: int = 10, runs: int = 60,
                     do_resize: bool = True, collect: bool = False):
    lib = ctypes.CDLL(str(DYLIB))
    _bind(lib)
    lib.TfLiteInterpreterResizeInputTensor.restype = ctypes.c_int
    lib.TfLiteInterpreterResizeInputTensor.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int), ctypes.c_int32,
    ]

    model = lib.TfLiteModelCreateFromFile(str(model_path).encode())
    if not model:
        raise RuntimeError(f"could not load {model_path}")

    opts = lib.TfLiteInterpreterOptionsCreate()
    lib.TfLiteInterpreterOptionsSetNumThreads(opts, threads)
    xnn = lib.TfLiteXNNPackDelegateOptionsDefault()
    struct.pack_into("<i", xnn.raw, NUM_THREADS_OFFSET, threads)
    delegate = lib.TfLiteXNNPackDelegateCreate(ctypes.byref(xnn))
    lib.TfLiteInterpreterOptionsAddDelegate(opts, delegate)

    interp = lib.TfLiteInterpreterCreate(model, opts)
    if not interp:
        raise RuntimeError("interpreter creation failed")

    resize_rc = None
    if do_resize:
        dims = (ctypes.c_int * 4)(1, img_size, img_size, 3)
        resize_rc = lib.TfLiteInterpreterResizeInputTensor(interp, 0, dims, 4)

    if lib.TfLiteInterpreterAllocateTensors(interp) != 0:
        raise RuntimeError("AllocateTensors failed")

    in_t = lib.TfLiteInterpreterGetInputTensor(interp, 0)
    out_t = lib.TfLiteInterpreterGetOutputTensor(interp, 0)
    n_out = lib.TfLiteTensorByteSize(out_t) // 4
    out = np.zeros(n_out, dtype=np.float32)
    preds = np.zeros((inputs.shape[0], n_out), np.float32) if collect else None

    def invoke(i):
        buf = np.ascontiguousarray(inputs[i], dtype=np.float32)
        lib.TfLiteTensorCopyFromBuffer(in_t, buf.ctypes.data_as(ctypes.c_void_p),
                                       buf.nbytes)
        if lib.TfLiteInterpreterInvoke(interp) != 0:
            raise RuntimeError("Invoke failed")
        lib.TfLiteTensorCopyToBuffer(out_t, out.ctypes.data_as(ctypes.c_void_p),
                                     out.nbytes)

    n = inputs.shape[0]
    for i in range(warmup):
        invoke(i % n)
    times = []
    for i in range(runs):
        idx = i % n
        buf = np.ascontiguousarray(inputs[idx], dtype=np.float32)
        lib.TfLiteTensorCopyFromBuffer(in_t, buf.ctypes.data_as(ctypes.c_void_p),
                                       buf.nbytes)
        t0 = time.perf_counter()
        lib.TfLiteInterpreterInvoke(interp)
        times.append((time.perf_counter() - t0) * 1000.0)
    if collect:
        for i in range(n):
            invoke(i)
            preds[i] = out

    lib.TfLiteXNNPackDelegateDelete(delegate)
    return {
        "median_ms": statistics.median(times),
        "p10_ms": float(np.percentile(times, 10)),
        "p90_ms": float(np.percentile(times, 90)),
        "resize_rc": resize_rc,
    }, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--runs", type=int, default=60)
    args = ap.parse_args()

    from pareto_harness import load_val_cache
    crops, _ = load_val_cache()
    X = np.asarray(crops[:8])

    print(f"flutter_litert {LITERT_VERSION}, XNNPACK, 4 threads")
    print("resize_rc 0 = ResizeInputTensor accepted\n")
    print(f"{'model':44s} {'no resize':>10s} {'app seq':>10s} {'rc':>4s} {'delta':>8s}")
    for m in args.models:
        a, _ = run_app_sequence(m, X, args.img_size, runs=args.runs, do_resize=False)
        b, _ = run_app_sequence(m, X, args.img_size, runs=args.runs, do_resize=True)
        print(f"{m.name:44s} {a['median_ms']:10.2f} {b['median_ms']:10.2f} "
              f"{str(b['resize_rc']):>4s} {b['median_ms'] - a['median_ms']:+8.2f}")


if __name__ == "__main__":
    main()
