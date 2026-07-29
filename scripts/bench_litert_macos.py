"""Benchmark a .tflite against the exact runtime the dog_detection package uses.

The Python `tf.lite` numbers are only good for ranking candidates inside this
repo. The package ships flutter_litert (see LITERT_VERSION below), whose macOS
bundle contains its own bazel-built libtensorflowlite_c-mac.dylib, and its
InterpreterFactory auto-mode on macOS builds an XNNPACK delegate with
numThreads = min(4, nproc) alongside options.threads = 4.

Keep LITERT_VERSION in step with the packages' pubspec constraint. The dylib is
rebuilt between releases even when InterpreterFactory is unchanged (3.6.0 and
3.7.0 differ in binary but are identical in that file), so a latency number is
only meaningful alongside the version it was measured on.

This driver loads that dylib directly through ctypes and reproduces that setup
field for field (including the QS8|QU8 flag pair flutter_litert sets by hand in
XNNPackDelegateOptions), so the reported latency is what the package would see on
this machine rather than an extrapolation from the TF Python runtime.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import struct
import time
from pathlib import Path

import numpy as np

def dylib_for(version: str) -> Path:
    return (Path.home() / f".pub-cache/hosted/pub.dev/flutter_litert-{version}/macos"
            / "flutter_litert/Sources/flutter_litert/Resources"
            / "libtensorflowlite_c-mac.dylib")


# Keep this pinned to whatever dog_detection/cat_detection actually resolve to.
# The binary differs between releases even when InterpreterFactory does not, so a
# latency figure is only meaningful next to the version it was taken on.
LITERT_VERSION = "3.7.0"
DYLIB = dylib_for(LITERT_VERSION)

TFLITE_XNNPACK_DELEGATE_FLAG_QS8 = 1
TFLITE_XNNPACK_DELEGATE_FLAG_QU8 = 2


# The options struct is filled by the dylib's own TfLiteXNNPackDelegateOptionsDefault
# rather than being reconstructed field by field. Hand-building it is what
# flutter_litert does in Dart, but this dylib defaults weight_cache_file_descriptor
# to -1 where a zeroed struct leaves it 0, and its struct is wider than the
# 48 bytes the Dart bindings declare -- handing Default() a 48-byte buffer lets it
# write past the end and corrupt the caller's stack, which segfaults on return.
# Over-allocating is safe: the callee only touches the fields it knows about.
# Only num_threads is overridden, at offset 0, which is stable across versions.
OPTIONS_SIZE = 256
NUM_THREADS_OFFSET = 0
FLAGS_OFFSET = 8


class XNNPackOptions(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_ubyte * OPTIONS_SIZE)]


def _bind(lib):
    lib.TfLiteModelCreateFromFile.restype = ctypes.c_void_p
    lib.TfLiteModelCreateFromFile.argtypes = [ctypes.c_char_p]
    lib.TfLiteInterpreterOptionsCreate.restype = ctypes.c_void_p
    lib.TfLiteInterpreterOptionsSetNumThreads.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.TfLiteInterpreterOptionsAddDelegate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.TfLiteInterpreterCreate.restype = ctypes.c_void_p
    lib.TfLiteInterpreterCreate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.TfLiteInterpreterAllocateTensors.argtypes = [ctypes.c_void_p]
    lib.TfLiteInterpreterAllocateTensors.restype = ctypes.c_int
    lib.TfLiteInterpreterGetInputTensor.restype = ctypes.c_void_p
    lib.TfLiteInterpreterGetInputTensor.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.TfLiteInterpreterGetOutputTensor.restype = ctypes.c_void_p
    lib.TfLiteInterpreterGetOutputTensor.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.TfLiteTensorByteSize.restype = ctypes.c_size_t
    lib.TfLiteTensorByteSize.argtypes = [ctypes.c_void_p]
    lib.TfLiteTensorCopyFromBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.TfLiteTensorCopyFromBuffer.restype = ctypes.c_int
    lib.TfLiteTensorCopyToBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.TfLiteTensorCopyToBuffer.restype = ctypes.c_int
    lib.TfLiteInterpreterInvoke.argtypes = [ctypes.c_void_p]
    lib.TfLiteInterpreterInvoke.restype = ctypes.c_int
    lib.TfLiteXNNPackDelegateCreate.restype = ctypes.c_void_p
    lib.TfLiteXNNPackDelegateCreate.argtypes = [ctypes.c_void_p]
    lib.TfLiteXNNPackDelegateDelete.argtypes = [ctypes.c_void_p]
    lib.TfLiteXNNPackDelegateOptionsDefault.restype = XNNPackOptions


def run(model_path: Path, inputs: np.ndarray, threads: int = 4,
        warmup: int = 10, runs: int = 60, collect: bool = False,
        dylib: Path | None = None, use_xnnpack: bool = True):
    """use_xnnpack=False reproduces PerformanceMode.disabled: interpreter threads
    only, no delegate, so the built-in TFLite kernels run everything. Comparing
    the two isolates what the delegate is actually contributing per op."""
    lib = ctypes.CDLL(str(dylib or DYLIB))
    _bind(lib)

    model = lib.TfLiteModelCreateFromFile(str(model_path).encode())
    if not model:
        raise RuntimeError(f"could not load {model_path}")

    opts = lib.TfLiteInterpreterOptionsCreate()
    lib.TfLiteInterpreterOptionsSetNumThreads(opts, threads)

    delegate = None
    if use_xnnpack:
        xnn = lib.TfLiteXNNPackDelegateOptionsDefault()
        struct.pack_into("<i", xnn.raw, NUM_THREADS_OFFSET, threads)
        assert struct.unpack_from("<I", xnn.raw, FLAGS_OFFSET)[0] == (
            TFLITE_XNNPACK_DELEGATE_FLAG_QS8 | TFLITE_XNNPACK_DELEGATE_FLAG_QU8
        ), "default flags differ from the QS8|QU8 pair flutter_litert sets"
        delegate = lib.TfLiteXNNPackDelegateCreate(ctypes.byref(xnn))
        if not delegate:
            raise RuntimeError("XNNPACK delegate creation failed")
        lib.TfLiteInterpreterOptionsAddDelegate(opts, delegate)

    interp = lib.TfLiteInterpreterCreate(model, opts)
    if not interp:
        raise RuntimeError("interpreter creation failed")
    if lib.TfLiteInterpreterAllocateTensors(interp) != 0:
        raise RuntimeError("AllocateTensors failed")

    in_t = lib.TfLiteInterpreterGetInputTensor(interp, 0)
    out_t = lib.TfLiteInterpreterGetOutputTensor(interp, 0)

    # Read the output width from the graph rather than hardcoding it: dogs emit
    # 92 values (46 landmarks) and cats 96 (48).
    n_out = lib.TfLiteTensorByteSize(out_t) // 4
    out = np.zeros(n_out, dtype=np.float32)
    preds = np.zeros((inputs.shape[0], n_out), dtype=np.float32) if collect else None

    def invoke(i):
        buf = np.ascontiguousarray(inputs[i], dtype=np.float32)
        lib.TfLiteTensorCopyFromBuffer(in_t, buf.ctypes.data_as(ctypes.c_void_p),
                                       buf.nbytes)
        rc = lib.TfLiteInterpreterInvoke(interp)
        if rc != 0:
            raise RuntimeError(f"Invoke failed rc={rc}")
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
        rc = lib.TfLiteInterpreterInvoke(interp)
        times.append((time.perf_counter() - t0) * 1000.0)
        if rc != 0:
            raise RuntimeError(f"Invoke failed rc={rc}")

    if collect:
        for i in range(n):
            invoke(i)
            preds[i] = out

    if delegate:
        lib.TfLiteXNNPackDelegateDelete(delegate)
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.fmean(times),
        "p10_ms": float(np.percentile(times, 10)),
        "p90_ms": float(np.percentile(times, 90)),
        "threads": threads,
        "runs": runs,
    }, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=60)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pareto_harness import load_val_cache

    crops, _ = load_val_cache()
    sample = np.asarray(crops[:8])

    print(f"dylib: {DYLIB.name} (flutter_litert {LITERT_VERSION})")
    print(f"{'model':44s} {'median ms':>10s} {'p10':>8s} {'p90':>8s}")
    for m in args.models:
        r, _ = run(m, sample, args.threads, runs=args.runs)
        print(f"{m.name:44s} {r['median_ms']:10.2f} {r['p10_ms']:8.2f} {r['p90_ms']:8.2f}")


if __name__ == "__main__":
    main()
